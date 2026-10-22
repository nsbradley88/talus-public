// GLM-5.3-Flash (glm5-next) kernels for the Talus Rust host: mHC, KDA conv/gate, MLA/DSA bookkeeping, MoE routing,
// plus C wrappers over the tuned GLM HIP kernels (talus::glm_spike). C ABI in include/talus_glm_kernels.h.
//
// Semantics follow llama.cpp's glm5-next graph (src/models/glm5-next.cpp) and ggml-cpu reference ops
// (dsv4_hc_pre_mix, dsv4_hc_post, ssm_conv, swiglu_clamp, k-pool indexer). Block formats are ggml's (MIT).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "talus_glm_kernels.h"
#include "strata/kernels/glm_spike_kernels.hpp"

namespace {

constexpr int WS = 32;  // wave32 (gfx11)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, WS);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, WS));
    return v;
}

// block-wide sum for blockDim.x <= 1024 (multiple of 32); sh must hold 32 floats
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float r = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (w == 0) r = warp_sum(r);
    if (threadIdx.x == 0) sh[0] = r;
    __syncthreads();
    r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

struct block_q8_0 { __half d; int8_t qs[32]; };
static_assert(sizeof(block_q8_0) == 34, "q8_0");
struct block_q4_K { __half d; __half dmin; uint8_t scales[12]; uint8_t qs[128]; };
static_assert(sizeof(block_q4_K) == 144, "q4_K");
struct block_q3_K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; __half d; };
static_assert(sizeof(block_q3_K) == 110, "q3_K");
struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; __half d; __half dmin; };
static_assert(sizeof(block_q2_K) == 84, "q2_K");

inline int err(hipError_t e) { return static_cast<int>(e); }
inline int last() { return static_cast<int>(hipGetLastError()); }
inline hipStream_t S(void* s) { return reinterpret_cast<hipStream_t>(s); }

// ------------------------------------------------------------------------------------------------------------------
// mHC

// mHC stream storage (GLM_HC_BF16, prompt path): the 4 x n_embd residual streams in fp32 (default) or 16 bits. Math stays
// fp32; a value is rounded once, where it is stored, and every consumer in the same kernel (Gram, next mixes) uses the
// rounded value, so the mixes / norms always describe exactly the stored streams. fp16 saturates at +-65504.
struct hc_f16 { uint16_t b; };
struct hc_bf16 { uint16_t b; };
template <typename ST> struct HcS;
template <> struct HcS<float> {
    static __device__ __forceinline__ float ld(const float* p, int i) { return p[i]; }
    static __device__ __forceinline__ float rnd(float v) { return v; }
    static __device__ __forceinline__ void st2(float* p, int i, float a, float b) {
        *reinterpret_cast<float2*>(p + i) = make_float2(a, b);
    }
    static __device__ __forceinline__ void st1(float* p, int i, float a) { p[i] = a; }
};
template <> struct HcS<hc_f16> {
    static __device__ __forceinline__ uint16_t enc(float v) {
        return __half_as_ushort(__float2half_rn(fminf(fmaxf(v, -65504.0f), 65504.0f)));
    }
    static __device__ __forceinline__ float dec(uint16_t b) { return __half2float(__ushort_as_half(b)); }
    static __device__ __forceinline__ float ld(const hc_f16* p, int i) { return dec(p[i].b); }
    static __device__ __forceinline__ float rnd(float v) { return dec(enc(v)); }
    static __device__ __forceinline__ void st2(hc_f16* p, int i, float a, float b) {
        *reinterpret_cast<uint32_t*>(p + i) = (uint32_t) enc(a) | ((uint32_t) enc(b) << 16);
    }
    static __device__ __forceinline__ void st1(hc_f16* p, int i, float a) { p[i].b = enc(a); }
};
template <> struct HcS<hc_bf16> {
    static __device__ __forceinline__ uint16_t enc(float v) {   // round to nearest even (finite inputs)
        const uint32_t u = __float_as_uint(v);
        return (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
    static __device__ __forceinline__ float dec(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }
    static __device__ __forceinline__ float ld(const hc_bf16* p, int i) { return dec(p[i].b); }
    static __device__ __forceinline__ float rnd(float v) { return dec(enc(v)); }
    static __device__ __forceinline__ void st2(hc_bf16* p, int i, float a, float b) {
        *reinterpret_cast<uint32_t*>(p + i) = (uint32_t) enc(a) | ((uint32_t) enc(b) << 16);
    }
    static __device__ __forceinline__ void st1(hc_bf16* p, int i, float a) { p[i].b = enc(a); }
};


__global__ void k_hc_repeat(const float* __restrict__ x, float* __restrict__ xhc, int n_embd) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x) {
        const float v = x[(size_t) t * n_embd + i];
        float* o = xhc + (size_t) t * 4 * n_embd;
        o[i] = v; o[n_embd + i] = v; o[2 * n_embd + i] = v; o[3 * n_embd + i] = v;
    }
}

// grid (nsplit, T), block 256: each thread owns 4 consecutive elements of the slice
template <int M>
__global__ __launch_bounds__(256) void k_hc_mix(const block_q8_0* __restrict__ w, const float* __restrict__ xhc,
                                                float* __restrict__ parts, int K, int nsplit) {
    __shared__ float sh[M + 1][8];
    const int p = blockIdx.x, t = blockIdx.y;
    const int slice = K / nsplit;
    const float* xs = xhc + (size_t) t * K + (size_t) p * slice;
    float acc[M + 1];
#pragma unroll
    for (int r = 0; r <= M; ++r) acc[r] = 0.0f;
    for (int e = threadIdx.x * 4; e < slice; e += 256 * 4) {
        const float4 xv = *reinterpret_cast<const float4*>(xs + e);
        acc[M] += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        const int gi = p * slice + e;           // global element index
        const int b = gi >> 5, o = gi & 31;
#pragma unroll
        for (int r = 0; r < M; ++r) {
            const block_q8_0& blk = w[(size_t) r * (K >> 5) + b];
            const float d = __half2float(blk.d);
            const int8_t* q = blk.qs + o;   // 2-byte aligned only: byte loads
            acc[r] += d * ((float) q[0] * xv.x + (float) q[1] * xv.y + (float) q[2] * xv.z + (float) q[3] * xv.w);
        }
    }
    const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r <= M; ++r) {
        const float v = warp_sum(acc[r]);
        if (lane == 0) sh[r][wv] = v;
    }
    __syncthreads();
    if (threadIdx.x <= M) {
        float v = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) v += sh[threadIdx.x][i];
        parts[((size_t) t * nsplit + p) * 32 + threadIdx.x] = v;
    }
}

// q8_1 block from 32 lane values (lane = element): ds = (d, sum x), q = round(x / d) (native_quantize_q8_1 format)
struct q8_1_blk { __half2 ds; int8_t qs[32]; };
__device__ __forceinline__ void q8_1_store_lane(float x, q8_1_blk* blk, int lane) {
    const float amax = warp_max(fabsf(x));
    const float sum = warp_sum(x);
    const float d = amax / 127.0f;
    const int8_t q = (int8_t) (amax == 0.0f ? 0 : __float2int_rn(x / d));
    blk->qs[lane] = q;
    if (lane == 0) blk->ds = __halves2half2(__float2half(d), __float2half(sum));
}

// one block (256) per token; warp 0 computes the mixes / Sinkhorn with lane-parallel reductions
__global__ __launch_bounds__(256) void k_hc_pre(const float* __restrict__ xhc, const float* __restrict__ parts,
                                                int nsplit, const float* __restrict__ scale,
                                                const float* __restrict__ base, const float* __restrict__ norm_w,
                                                float* __restrict__ y, float* __restrict__ pc, int n_embd,
                                                float rms_eps, float norm_eps, float hc_eps, int iters,
                                                q8_1_blk* __restrict__ yq) {
    constexpr int HC = 4, MIX = 24;
    __shared__ float pre_s[HC];
    __shared__ float red[32];
    __shared__ float ybuf[4096];
    const int t = blockIdx.x;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float m = 0.0f;
        if (lane <= MIX) {
            for (int p = 0; p < nsplit; ++p) m += parts[((size_t) t * nsplit + p) * 32 + lane];
        }
        const float ss = __shfl(m, MIX, WS);
        const float r = 1.0f / sqrtf(ss / (float) (HC * n_embd) + rms_eps);
        m *= r;
        float* o = pc + (size_t) t * 32;
        if (lane < HC) {
            pre_s[lane] = sigmoidf_(m * scale[0] + base[lane]) + hc_eps;
        } else if (lane < 2 * HC) {
            o[lane - HC] = 2.0f * sigmoidf_(m * scale[1] + base[lane]);
        }
        // comb: lane i < 16 owns c[i], i = dst + 4 src; mixes 8.. live in lanes 8..23
        const float mc = __shfl(m, 2 * HC + (lane & 15), WS);
        float c = mc * scale[2] + base[2 * HC + (lane & 15)];
        // softmax over dst (lanes 4s..4s+3)
        float mx = fmaxf(c, __shfl_xor(c, 1, WS));
        mx = fmaxf(mx, __shfl_xor(mx, 2, WS));
        c = expf(c - mx);
        float sm = c + __shfl_xor(c, 1, WS);
        sm += __shfl_xor(sm, 2, WS);
        c = c / sm + hc_eps;
        auto col = [&](float v) {   // sum over src for fixed dst: lanes dst, dst+4, dst+8, dst+12
            float s2 = v + __shfl_xor(v, 4, WS);
            s2 += __shfl_xor(s2, 8, WS);
            return s2;
        };
        auto row = [&](float v) {   // sum over dst for fixed src
            float s2 = v + __shfl_xor(v, 1, WS);
            s2 += __shfl_xor(s2, 2, WS);
            return s2;
        };
        c = c / (col(c) + hc_eps);
        for (int i = 1; i < iters; ++i) {
            c = c / (row(c) + hc_eps);
            c = c / (col(c) + hc_eps);
        }
        if (lane < 16) o[HC + lane] = c;
    }
    __syncthreads();
    const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
    const float* xt = xhc + (size_t) t * HC * n_embd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float v = xt[i] * p0 + xt[n_embd + i] * p1 + xt[2 * n_embd + i] * p2 + xt[3 * n_embd + i] * p3;
        ybuf[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float s = 1.0f / sqrtf(ss / (float) n_embd + norm_eps);
    float* yt = y + (size_t) t * n_embd;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float v = ybuf[i] * s * norm_w[i];
        yt[i] = v;
        if (yq) q8_1_store_lane(v, yq + ((size_t) t * n_embd + i) / 32, threadIdx.x & 31);
    }
}

__global__ void k_hc_post(const float* __restrict__ xo, const float* res, const float* __restrict__ pc, float* out,
                          int n_embd) {
    const int t = blockIdx.y;
    const float* c = pc + (size_t) t * 32;
    const float* r = res + (size_t) t * 4 * n_embd;
    float* o = out + (size_t) t * 4 * n_embd;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x) {
        const float x = xo[(size_t) t * n_embd + i];
        const float r0 = r[i], r1 = r[n_embd + i], r2 = r[2 * n_embd + i], r3 = r[3 * n_embd + i];
        float v[4];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            float a = x * c[d];
            a += r0 * c[4 + d + 0];
            a += r1 * c[4 + d + 4];
            a += r2 * c[4 + d + 8];
            a += r3 * c[4 + d + 12];
            v[d] = a;
        }
        o[i] = v[0]; o[n_embd + i] = v[1]; o[2 * n_embd + i] = v[2]; o[3 * n_embd + i] = v[3];
    }
}

template <typename ST = float>
__global__ __launch_bounds__(256) void k_hc_head(const ST* __restrict__ xhc, const float* __restrict__ w,
                                                 float* __restrict__ h, int n_embd, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const ST* x = xhc + (size_t) t * 4 * n_embd;
    using H = HcS<ST>;
    float* o = h + (size_t) t * n_embd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float v = H::ld(x, i);
        v += H::ld(x, n_embd + i);
        v += H::ld(x, 2 * n_embd + i);
        v += H::ld(x, 3 * n_embd + i);
        v *= 0.25f;
        o[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float s = 1.0f / sqrtf(ss / (float) n_embd + eps);
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) o[i] = o[i] * s * w[i];
}

// ------------------------------------------------------------------------------------------------------------------
// generic

__global__ __launch_bounds__(256) void k_rms_norm(const float* __restrict__ x, const float* __restrict__ w,
                                                  float* __restrict__ y, int cols, int xs, int ys, float eps) {
    __shared__ float red[32];
    const int r = blockIdx.x;
    const float* xr = x + (size_t) r * xs;
    float* yr = y + (size_t) r * ys;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { const float v = xr[i]; ss += v * v; }
    ss = block_sum(ss, red);
    const float s = 1.0f / sqrtf(ss / (float) cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) yr[i] = xr[i] * s * (w ? w[i] : 1.0f);
}

// rows of <= 128 cols: one wave per row (cols <= 128)
__global__ __launch_bounds__(256) void k_rms_norm_small(const float* __restrict__ x, const float* __restrict__ w,
                                                        float* __restrict__ y, int rows, int cols, int xs, int ys,
                                                        float eps) {
    const int r = blockIdx.x * (blockDim.x / WS) + (threadIdx.x / WS);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* xr = x + (size_t) r * xs;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = lane + j * WS;
        v[j] = i < cols ? xr[i] : 0.0f;
        ss += v[j] * v[j];
    }
    ss = warp_sum(ss);
    const float s = 1.0f / sqrtf(ss / (float) cols + eps);
    float* yr = y + (size_t) r * ys;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = lane + j * WS;
        if (i < cols) yr[i] = v[j] * s * (w ? w[i] : 1.0f);
    }
}

__global__ void k_add(float* __restrict__ y, const float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] += x[i];
}

__global__ void k_swiglu_clamp(const float* g, const float* __restrict__ u, float* o,   // o may alias g (in place)
                               int n, float limit) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float gv = g[i], uv = u[i];
        if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
        o[i] = gv / (1.0f + expf(-gv)) * uv;
    }
}

__device__ __forceinline__ void q4k_scale_min(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

// one block (256 threads) per token: thread handles one element per 256-block
__global__ void k_embed_q4k(const block_q4_K* __restrict__ tab, const int32_t* __restrict__ tok,
                            float* __restrict__ out, int n_embd) {
    const int t = blockIdx.x;
    const int row = tok[t];
    const int nb = n_embd / 256;
    const block_q4_K* b0 = tab + (size_t) row * nb;
    for (int e = threadIdx.x; e < n_embd; e += blockDim.x) {
        const block_q4_K& b = b0[e / 256];
        const int i = e % 256;
        const int j64 = i / 64, w = i % 64, hi = w / 32, l = w % 32;
        uint8_t sc, mn;
        q4k_scale_min(2 * j64 + hi, b.scales, sc, mn);
        const uint8_t q = b.qs[j64 * 32 + l];
        const int qv = hi ? (q >> 4) : (q & 0xF);
        out[(size_t) t * n_embd + e] = __half2float(b.d) * sc * qv - __half2float(b.dmin) * mn;
    }
}

__global__ void k_embed_f32(const float* __restrict__ tab, const int32_t* __restrict__ tok, float* __restrict__ out,
                            int n_embd) {
    const int t = blockIdx.x;
    const float* r = tab + (size_t) tok[t] * n_embd;
    for (int e = threadIdx.x; e < n_embd; e += blockDim.x) out[(size_t) t * n_embd + e] = r[e];
}

// ------------------------------------------------------------------------------------------------------------------
// KDA

// thread per channel c of group g (q/k/v): grid ((C + 255)/256, 3)
__global__ void k_kda_conv(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                           float* st, const int32_t* __restrict__ ids, int K, const float* __restrict__ wq,
                           const float* __restrict__ wk, const float* __restrict__ wv, float* __restrict__ oq,
                           float* __restrict__ ok, float* __restrict__ ov, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (c >= C) return;
    const float* x = g == 0 ? q : (g == 1 ? k : v);
    const float* w = g == 0 ? wq : (g == 1 ? wk : wv);
    float* o = g == 0 ? oq : (g == 1 ? ok : ov);
    const size_t slot_stride = (size_t) 3 * C * 3;
    const size_t ch = ((size_t) g * C + c) * 3;
    const float* sin = st + (size_t) ids[0] * slot_stride + ch;
    float s0 = sin[0], s1 = sin[1], s2 = sin[2];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float xv = x[(size_t) t * C + c];
        float a = s0 * w0;
        a += s1 * w1;
        a += s2 * w2;
        a += xv * w3;
        o[(size_t) t * C + c] = a / (1.0f + expf(-a));
        s0 = s1; s1 = s2; s2 = xv;
        const int kk = t - (T - K);
        if (kk >= 0) {
            float* so = st + (size_t) ids[kk] * slot_stride + ch;
            so[0] = s0; so[1] = s1; so[2] = s2;
        }
    }
}

// one wave per (t, h), D = 128
__global__ __launch_bounds__(256) void k_kda_gate(const float* __restrict__ o, const float* __restrict__ g2,
                                                  const float* __restrict__ w, float* __restrict__ out, int rows,
                                                  int D, float eps) {
    const int r = blockIdx.x * (blockDim.x / WS) + (threadIdx.x / WS);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* x = o + (size_t) r * D;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = x[lane + j * WS]; ss += v[j] * v[j]; }
    ss = warp_sum(ss);
    const float s = 1.0f / sqrtf(ss / (float) D + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = lane + j * WS;
        out[(size_t) r * D + i] = v[j] * s * w[i] * sigmoidf_(g2[(size_t) r * D + i]);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// MLA / DSA

__global__ __launch_bounds__(256) void k_mla_store(const float* __restrict__ kv, const float* __restrict__ w,
                                                   float eps, __half* __restrict__ cache,
                                                   const int32_t* __restrict__ ctl, int D) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const int p = ctl[0] + t;
    const float* x = kv + (size_t) t * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red);
    const float s = 1.0f / sqrtf(ss / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) cache[(size_t) p * D + i] = __float2half(x[i] * s * w[i]);
}

// one wave per token: LayerNorm over 128 (mean, var), store key | gate f16
__global__ __launch_bounds__(32) void k_idx_store(const float* __restrict__ ik, const float* __restrict__ lw,
                                                  const float* __restrict__ lb, float eps,
                                                  const float* __restrict__ ig, __half* __restrict__ kg,
                                                  const int32_t* __restrict__ ctl) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const int p = ctl[0] + t;
    float v[4];
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = ik[(size_t) t * 128 + lane + j * WS]; s += v[j]; }
    const float mean = warp_sum(s) / 128.0f;
    float var = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] -= mean; var += v[j] * v[j]; }
    var = warp_sum(var) / 128.0f;
    const float inv = 1.0f / sqrtf(var + eps);
    __half* row = kg + (size_t) (p & (TKG_KG_RING - 1)) * 256;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = lane + j * WS;
        row[i] = __float2half(v[j] * inv * lw[i] + lb[i]);
        row[128 + i] = __float2half(ig[(size_t) t * 128 + i]);
    }
}

// block of 128 threads (one per dim) per window token
__global__ __launch_bounds__(128) void k_idx_pool(const __half* __restrict__ kg, const float* __restrict__ ape,
                                                  __half* __restrict__ pooled, const int32_t* __restrict__ ctl,
                                                  int kpool) {
    const int t = blockIdx.x, d = threadIdx.x;
    const int p = ctl[0] + t;
    if (((p + 1) % kpool) != 0) return;
    const int p0 = p - kpool + 1;
    float lg[8], kv[8];
    float mx = -INFINITY;
    for (int k = 0; k < kpool; ++k) {
        const __half* row = kg + (size_t) ((p0 + k) & (TKG_KG_RING - 1)) * 256;
        kv[k] = __half2float(row[d]);
        lg[k] = __half2float(row[128 + d]) + ape[d + 128 * k];
        mx = fmaxf(mx, lg[k]);
    }
    float sum = 0.0f;
    for (int k = 0; k < kpool; ++k) { lg[k] = expf(lg[k] - mx); sum += lg[k]; }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
    for (int k = 0; k < kpool; ++k) acc += lg[k] * inv * kv[k];
    pooled[(size_t) (p / kpool) * 128 + d] = __float2half(acc);
}

// order-preserving float -> uint key (larger float -> larger key)
__device__ __forceinline__ uint32_t fkey(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block (1024) per token: radix-select the threshold of the top-k pool scores, then emit rows ascending
__global__ __launch_bounds__(1024) void k_dsa_select(const float* __restrict__ scores, int stride,
                                                     const int32_t* __restrict__ ctl, int kpool, int top,
                                                     int n_kv_max, int32_t* __restrict__ idx) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_mask, s_need, s_eq_need;
    __shared__ uint32_t wsum[32];
    __shared__ uint32_t s_base;
    const int t = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const int p = ctl[0] + t;
    const int n_vis = (p + 1) / kpool;
    const int n_tail = (p + 1) % kpool;
    const float* sc = scores + (size_t) t * stride;
    int32_t* out = idx + (size_t) t * n_kv_max;
    const bool all = n_vis <= top;
    // radix select (4 passes of 8 bits) for the top-th largest key
    if (tid == 0) { s_prefix = 0; s_mask = 0; s_need = (uint32_t) top; }
    __syncthreads();
    if (!all) {
        for (int pass = 3; pass >= 0; --pass) {
            for (int i = tid; i < 256; i += nt) hist[i] = 0;
            __syncthreads();
            const uint32_t prefix = s_prefix, mask = s_mask;
            for (int j = tid; j < n_vis; j += nt) {
                const uint32_t kk = fkey(sc[j]);
                if ((kk & mask) == prefix) atomicAdd(&hist[(kk >> (pass * 8)) & 0xFF], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                uint32_t need = s_need;
                int b = 255;
                for (; b > 0; --b) {
                    if (hist[b] >= need) break;
                    need -= hist[b];
                }
                s_prefix = prefix | ((uint32_t) b << (pass * 8));
                s_mask = mask | (0xFFu << (pass * 8));
                s_need = need;  // how many keys equal to the final threshold are still needed
            }
            __syncthreads();
        }
    }
    const uint32_t thr = s_prefix;
    if (tid == 0) { s_eq_need = all ? 0u : s_need; s_base = 0; }
    __syncthreads();
    // emit selected pools ascending: chunked block scan over pool index
    const int lane = tid & 31, w = tid >> 5;
    for (int j0 = 0; j0 < n_vis; j0 += nt) {
        const int j = j0 + tid;
        bool sel = false, eq = false;
        if (j < n_vis) {
            if (all) sel = true;
            else {
                const uint32_t kk = fkey(sc[j]);
                sel = kk > thr;
                eq = kk == thr;
            }
        }
        // ties at the threshold: take the lowest pool indices first (deterministic)
        if (!all) {
            const uint32_t be = (uint32_t) __ballot(eq);
            const uint32_t below = __popc(be & ((1u << lane) - 1u));
            if (lane == 0) wsum[w] = __popc(be);
            __syncthreads();
            uint32_t before = 0;
            for (int i = 0; i < w; ++i) before += wsum[i];
            const uint32_t need = s_eq_need;
            if (eq && before + below < need) sel = true;
            __syncthreads();
            if (tid == 0) {
                uint32_t tot = 0;
                for (int i = 0; i < nt / 32; ++i) tot += wsum[i];
                s_eq_need = need > tot ? need - tot : 0u;
            }
            __syncthreads();
        }
        const uint32_t bs = (uint32_t) __ballot(sel);
        const uint32_t below = __popc(bs & ((1u << lane) - 1u));
        if (lane == 0) wsum[w] = __popc(bs);
        __syncthreads();
        uint32_t before = 0;
        for (int i = 0; i < w; ++i) before += wsum[i];
        const uint32_t base = s_base;
        if (sel) {
            const uint32_t slot = (base + before + below) * kpool;
            for (int k = 0; k < kpool; ++k) {
                if ((int) (slot + k) < n_kv_max) out[slot + k] = j * kpool + k;
            }
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t tot = 0;
            for (int i = 0; i < nt / 32; ++i) tot += wsum[i];
            s_base = base + tot;
        }
        __syncthreads();
    }
    const int n_pool_sel = (int) s_base;
    const int n_rows = n_pool_sel * kpool;
    for (int i = tid; i < n_kv_max - n_rows; i += nt) {
        const int slot = n_rows + i;
        out[slot] = i < n_tail ? (p - n_tail + 1 + i) : -1;
    }
}

// ------------------------------------------------------------------------------------------------------------------
// MoE

// one wave per token; n_exp <= 512, k <= 16
__global__ __launch_bounds__(32) void k_router(const float* __restrict__ logits, const float* __restrict__ bias,
                                               int32_t* __restrict__ ids, float* __restrict__ wout, int n_exp, int k,
                                               float scale) {
    constexpr int MAXE = 16;  // per-lane experts (512 / 32)
    const int t = blockIdx.x, lane = threadIdx.x;
    float prob[MAXE], sel[MAXE];
    const int per = (n_exp + WS - 1) / WS;
    for (int j = 0; j < MAXE; ++j) {
        const int e = lane + j * WS;
        if (j < per && e < n_exp) {
            prob[j] = sigmoidf_(logits[(size_t) t * n_exp + e]);
            sel[j] = prob[j] + (bias ? bias[e] : 0.0f);
        } else {
            prob[j] = 0.0f;
            sel[j] = -INFINITY;
        }
    }
    float wsel[16];
    int isel[16];
    float wsum = 0.0f;
    for (int r = 0; r < k; ++r) {
        // lane-local best (lowest expert index on ties)
        float bv = -INFINITY;
        int bj = -1;
        for (int j = 0; j < per; ++j) {
            if (sel[j] > bv) { bv = sel[j]; bj = j; }
        }
        int be = bj >= 0 ? lane + bj * WS : 0x7fffffff;
        // wave argmax: max value, then smallest expert index
        float mv = warp_max(bv);
        int cand = (bv == mv) ? be : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) cand = min(cand, __shfl_xor(cand, o, WS));
        const int owner = cand % WS, oj = cand / WS;
        float pv = 0.0f;
        if (lane == owner) { pv = prob[oj]; sel[oj] = -INFINITY; }
        pv = __shfl(pv, owner, WS);
        isel[r] = cand;
        wsel[r] = pv;
        wsum += pv;
    }
    if (lane == 0) {
        const float inv = scale / fmaxf(wsum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            ids[(size_t) t * k + r] = isel[r];
            wout[(size_t) t * k + r] = wsel[r] * inv;
        }
    }
}

__global__ void k_moe_combine(const float* __restrict__ y, const float* __restrict__ w,
                              const float* __restrict__ sh, float* __restrict__ out, int k, int D) {
    const int t = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < D; i += gridDim.x * blockDim.x) {
        float a = 0.0f;
        for (int j = 0; j < k; ++j) a += w[(size_t) t * k + j] * y[((size_t) t * k + j) * D + i];
        if (sh) a += sh[(size_t) t * D + i];
        out[(size_t) t * D + i] = a;
    }
}

// dequant helpers (one 256-block -> per-lane 8 values at offset l8 = lane * 8)
__device__ void deq_q3k(const block_q3_K& b, int base, float* yv /* 8 */) {
    // generic: element index e in [0, 256): base..base+7 inside one 16-element run
    uint32_t aux[4];
    memcpy(aux, b.scales, 12);
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    const float d = __half2float(b.d);
    // layout: n in {0,128}; j in 0..3 (shift 2j, mask bit m = 1 << (4*(n/128) + j)); two 16-runs (l, l+16)
    const int n = base / 128, r = base % 128;
    const int j = r / 32, half = (r % 32) / 16, l0 = r % 16;
    const int is = n * 8 + j * 2 + half;
    const float dl = d * (scales[is] - 32);
    const uint8_t* q = b.qs + n * 32;
    const uint8_t m = (uint8_t) (1u << (n * 4 + j));
    for (int i = 0; i < 8; ++i) {
        const int l = l0 + i + half * 16;
        yv[i] = dl * ((int8_t) ((q[l] >> (2 * j)) & 3) - ((b.hmask[l] & m) ? 0 : 4));
    }
}

__device__ void deq_q2k(const block_q2_K& b, int base, float* yv) {
    const float d = __half2float(b.d), dmin = __half2float(b.dmin);
    // n in {0,128}; j in 0..3 shift 2j; two 16-runs with scales is, is+1
    const int n = base / 128, r = base % 128;
    const int j = r / 32, half = (r % 32) / 16, l0 = r % 16;
    const int is = n * 8 + j * 2 + half;
    const uint8_t sc = b.scales[is];
    const float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
    const uint8_t* q = b.qs + n * 32;
    for (int i = 0; i < 8; ++i) {
        const int l = l0 + i + half * 16;
        yv[i] = dl * ((int8_t) ((q[l] >> (2 * j)) & 3)) - ml;
    }
}

// grid (n_out / 8, n_ids), block 256 = 8 waves; one wave per row
template <int TYPE>
__global__ __launch_bounds__(256) void k_mmvq_ids_f32(const uint8_t* __restrict__ w, size_t expert_bytes,
                                                      const int32_t* __restrict__ ids, int k,
                                                      const float* __restrict__ x, int x_per_id,
                                                      float* __restrict__ y, int n_in, int n_out) {
    const int i = blockIdx.y;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_out) return;
    const int e = ids[i];
    const float* xv = x + (size_t) (x_per_id ? i : i / k) * n_in;
    const int nb = n_in / 256;
    const size_t bsz = TYPE == 11 ? sizeof(block_q3_K) : sizeof(block_q2_K);
    const uint8_t* wr = w + (size_t) e * expert_bytes + (size_t) row * nb * bsz;
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        float yv[8];
        if (TYPE == 11) deq_q3k(*reinterpret_cast<const block_q3_K*>(wr + b * bsz), lane * 8, yv);
        else deq_q2k(*reinterpret_cast<const block_q2_K*>(wr + b * bsz), lane * 8, yv);
        const float* xb = xv + b * 256 + lane * 8;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += yv[j] * xb[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t) i * n_out + row] = acc;
}

__global__ void k_gather_rows(const uint8_t* __restrict__ src, size_t row_bytes, const int32_t* __restrict__ ids,
                              uint8_t* __restrict__ dst) {
    const int r = blockIdx.x;
    const uint8_t* s = src + (size_t) ids[r] * row_bytes;
    uint8_t* d = dst + (size_t) r * row_bytes;
    for (size_t i = threadIdx.x * 4; i + 4 <= row_bytes; i += blockDim.x * 4)
        *reinterpret_cast<uint32_t*>(d + i) = *reinterpret_cast<const uint32_t*>(s + i);
}


// ------------------------------------------------------------------------------------------------------------------
// decode helpers (v2)

// y[t][r] = W[r] . x[t] for T <= 8 tokens: one wave per row, float4 loads (4 in flight), weights read once
template <int T, bool PF = false>
__global__ __launch_bounds__(256) void k_f32mv(const float* __restrict__ W, const float* __restrict__ x,
                                               float* __restrict__ y, int n_in, int n_out) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= n_out) return;
    const float4* w4 = reinterpret_cast<const float4*>(W + (size_t) r * n_in);
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    int i0 = lane;
    if constexpr (PF) {   // 8 steps' loads in flight, the same per-step expression in order (bitwise)
        for (; i0 + 7 * 32 < n_in / 4; i0 += 8 * 32) {
            float4 wb[8], xb[8][T];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                wb[u] = w4[i0 + 32 * u];
#pragma unroll
                for (int t = 0; t < T; ++t) xb[u][t] = reinterpret_cast<const float4*>(x + (size_t) t * n_in)[i0 + 32 * u];
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int u = 0; u < 8; ++u)
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 w = wb[u], v = xb[u][t];
                    acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
                }
        }
    }
#pragma unroll 4
    for (int i = i0; i < n_in / 4; i += 32) {
        const float4 w = w4[i];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 v = reinterpret_cast<const float4*>(x + (size_t) t * n_in)[i];
            acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) y[(size_t) t * n_out + r] = v;
    }
}

// per-head Q8_0 mat-vec, f32 activations: y[t][h][r] = W[h][r] . x[t][h]  (W [H][n_out][n_in], n_in <= 512)
// one wave per (h, row) for all T tokens; lane l covers elements [l*K/32 ...]
template <int T>
__global__ __launch_bounds__(256) void k_heads_q8_0(const block_q8_0* __restrict__ W, const float* __restrict__ x,
                                                    float* __restrict__ y, int H, int n_in, int n_out) {
    const int h = blockIdx.y;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= n_out) return;
    const int nb = n_in / 32;   // blocks per row
    const block_q8_0* wr = W + ((size_t) h * n_out + r) * nb;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    // each lane: 8 consecutive elements; lanes 4b..4b+3 share block b; loop over block groups of 8
    for (int e0 = lane * 8; e0 < n_in; e0 += 256) {
        const block_q8_0& b = wr[e0 / 32];
        const float d = __half2float(b.d);
        const int o = e0 % 32;
        float wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) wv[j] = (float) b.qs[o + j] * d;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float* xv = x + ((size_t) t * H + h) * n_in + e0;
            const float4 a0 = *reinterpret_cast<const float4*>(xv);
            const float4 a1 = *reinterpret_cast<const float4*>(xv + 4);
            acc[t] += wv[0] * a0.x + wv[1] * a0.y + wv[2] * a0.z + wv[3] * a0.w + wv[4] * a1.x + wv[5] * a1.y +
                      wv[6] * a1.z + wv[7] * a1.w;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) y[((size_t) t * H + h) * n_out + r] = v;
    }
}

// clamp-SwiGLU with q8_1 output (n % 32 == 0)
__global__ void k_swiglu_clamp_q8(const float* __restrict__ g, const float* __restrict__ u, q8_1_blk* __restrict__ oq,
                                  int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
    q8_1_store_lane(gv / (1.0f + expf(-gv)) * uv, oq + i / 32, threadIdx.x & 31);
}

// KDA output gate with q8_1 output
__global__ __launch_bounds__(256) void k_kda_gate_q8(const float* __restrict__ o, const float* __restrict__ g2,
                                                     const float* __restrict__ w, q8_1_blk* __restrict__ out,
                                                     int rows, int D, float eps) {
    const int r = blockIdx.x * (blockDim.x / WS) + (threadIdx.x / WS);
    const int lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float* x = o + (size_t) r * D;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) { v[j] = x[lane + j * WS]; ss += v[j] * v[j]; }
    ss = warp_sum(ss);
    const float s = 1.0f / sqrtf(ss / (float) D + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int i = lane + j * WS;
        const float val = v[j] * s * w[i] * sigmoidf_(g2[(size_t) r * D + i]);
        q8_1_store_lane(val, out + ((size_t) r * D + j * WS) / 32, lane);
    }
}

// K-quant expert mat-vec v2 (Q3_K / Q2_K), f32 activations: one wave per (row, id); lane = 256-block (strided),
// scales decoded once per block. y[i][r] = W[ids[i]][r] . x[x_per_id ? i : i / k]
template <int TYPE>
__global__ __launch_bounds__(256) void k_kq_ids_v2(const uint8_t* __restrict__ w, size_t expert_bytes,
                                                   const int32_t* __restrict__ ids, int k,
                                                   const float* __restrict__ x, int x_per_id,
                                                   float* __restrict__ y, int n_in, int n_out) {
    const int i = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int nb = n_in / 256;
    // lanes split as (row sub-group, block): rows per wave = 32 / nb (nb = 8 or 16)
    const int rpw = 32 / nb;
    const int row = (blockIdx.x * 8 + (threadIdx.x >> 5)) * rpw + lane / nb;
    const int b = lane % nb;
    const bool ok = row < n_out;
    const int e = ids[i];
    const float* xb = x + (size_t) (x_per_id ? i : i / k) * n_in + b * 256;
    const size_t bsz = TYPE == 11 ? sizeof(block_q3_K) : sizeof(block_q2_K);
    float acc = 0.0f;
    if (ok) {
        const uint8_t* wb = w + (size_t) e * expert_bytes + ((size_t) row * nb + b) * bsz;
        if (TYPE == 11) {
            const block_q3_K& q = *reinterpret_cast<const block_q3_K*>(wb);
            uint32_t aux[4];
            memcpy(aux, q.scales, 12);
            const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
            const float d = __half2float(q.d);
            for (int n = 0; n < 2; ++n) {
                for (int j = 0; j < 4; ++j) {
                    const uint8_t m = (uint8_t) (1u << (n * 4 + j));
                    for (int hf = 0; hf < 2; ++hf) {
                        const float dl = d * (sc[n * 8 + j * 2 + hf] - 32);
                        float part = 0.0f;
                        for (int l = 0; l < 16; ++l) {
                            const int li = l + hf * 16;
                            const int qv = (int8_t) ((q.qs[n * 32 + li] >> (2 * j)) & 3) - ((q.hmask[li] & m) ? 0 : 4);
                            part += qv * xb[n * 128 + j * 32 + hf * 16 + l];
                        }
                        acc += dl * part;
                    }
                }
            }
        } else {
            const block_q2_K& q = *reinterpret_cast<const block_q2_K*>(wb);
            const float d = __half2float(q.d), dmin = __half2float(q.dmin);
            for (int n = 0; n < 2; ++n) {
                for (int j = 0; j < 4; ++j) {
                    for (int hf = 0; hf < 2; ++hf) {
                        const uint8_t s8 = q.scales[n * 8 + j * 2 + hf];
                        float part = 0.0f, xs = 0.0f;
                        for (int l = 0; l < 16; ++l) {
                            const int li = l + hf * 16;
                            const float xv = xb[n * 128 + j * 32 + hf * 16 + l];
                            part += (float) ((q.qs[n * 32 + li] >> (2 * j)) & 3) * xv;
                            xs += xv;
                        }
                        acc += d * (s8 & 0xF) * part - dmin * (s8 >> 4) * xs;
                    }
                }
            }
        }
    }
    // reduce the nb lanes of a row
    for (int o = nb / 2; o > 0; o >>= 1) acc += __shfl_xor(acc, o, WS);
    if (ok && b == 0) y[(size_t) i * n_out + row] = acc;
}


// K-quant expert mat-vec v3 (Q3_K / Q2_K), f32 activations, bitwise the v2 result per (id, row). v2 is
// limited by its activation reads, not the weights: every lane reads its 1 KiB slice of x per row from the caches
// (exceeding the weight bytes). v3 stages the id's x row in LDS once per block (rows of 256 padded by 4 floats against
// bank conflicts) and runs v2's per-lane arithmetic on it unchanged. Optionally gate + up in one launch
// (blockIdx.z = matrix; w1 / y1 null = one matrix).
template <int TYPE>
__global__ __launch_bounds__(256) void k_kq_ids_v3(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                                   size_t expert_bytes, const int32_t* __restrict__ ids, int k,
                                                   const float* __restrict__ x, int x_per_id, float* __restrict__ y0,
                                                   float* __restrict__ y1, int n_in, int n_out) {
    constexpr int XP = 256 + 4;
    __shared__ float xs[16 * XP];
    const int i = blockIdx.y;
    const uint8_t* __restrict__ w = blockIdx.z ? w1 : w0;
    float* __restrict__ y = blockIdx.z ? y1 : y0;
    const int lane = threadIdx.x & 31;
    const int nb = n_in / 256;
    {
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t) (x_per_id ? i : i / k) * n_in);
        for (int q = threadIdx.x; q < n_in / 4; q += 256) {
            const int bb = q / 64, o = (q % 64) * 4;
            *reinterpret_cast<float4*>(&xs[bb * XP + o]) = xr[q];
        }
    }
    __syncthreads();
    const int rpw = 32 / nb;
    const int row = (blockIdx.x * 8 + (threadIdx.x >> 5)) * rpw + lane / nb;
    const int b = lane % nb;
    const bool ok = row < n_out;
    const int e = ids[i];
    const float* xb = xs + b * XP;
    const size_t bsz = TYPE == 11 ? sizeof(block_q3_K) : sizeof(block_q2_K);
    float acc = 0.0f;
    if (ok) {
        const uint8_t* wb = w + (size_t) e * expert_bytes + ((size_t) row * nb + b) * bsz;
        if (TYPE == 11) {
            const block_q3_K& q = *reinterpret_cast<const block_q3_K*>(wb);
            uint32_t aux[4];
            memcpy(aux, q.scales, 12);
            const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu, tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            const int8_t* sc = reinterpret_cast<const int8_t*>(aux);
            const float d = __half2float(q.d);
            for (int n = 0; n < 2; ++n) {
                for (int j = 0; j < 4; ++j) {
                    const uint8_t m = (uint8_t) (1u << (n * 4 + j));
                    for (int hf = 0; hf < 2; ++hf) {
                        const float dl = d * (sc[n * 8 + j * 2 + hf] - 32);
                        float part = 0.0f;
                        for (int l = 0; l < 16; ++l) {
                            const int li = l + hf * 16;
                            const int qv = (int8_t) ((q.qs[n * 32 + li] >> (2 * j)) & 3) - ((q.hmask[li] & m) ? 0 : 4);
                            part += qv * xb[n * 128 + j * 32 + hf * 16 + l];
                        }
                        acc += dl * part;
                    }
                }
            }
        } else {
            const block_q2_K& q = *reinterpret_cast<const block_q2_K*>(wb);
            const float d = __half2float(q.d), dmin = __half2float(q.dmin);
            for (int n = 0; n < 2; ++n) {
                for (int j = 0; j < 4; ++j) {
                    for (int hf = 0; hf < 2; ++hf) {
                        const uint8_t s8 = q.scales[n * 8 + j * 2 + hf];
                        float part = 0.0f, xs_ = 0.0f;
                        for (int l = 0; l < 16; ++l) {
                            const int li = l + hf * 16;
                            const float xv = xb[n * 128 + j * 32 + hf * 16 + l];
                            part += (float) ((q.qs[n * 32 + li] >> (2 * j)) & 3) * xv;
                            xs_ += xv;
                        }
                        acc += d * (s8 & 0xF) * part - dmin * (s8 >> 4) * xs_;
                    }
                }
            }
        }
    }
    // reduce the nb lanes of a row
    for (int o = nb / 2; o > 0; o >>= 1) acc += __shfl_xor(acc, o, WS);
    if (ok && b == 0) y[(size_t) i * n_out + row] = acc;
}

// greedy argmax of `rows` rows of n floats (row stride ld) on the device, the host sample::argmax rule: the
// first index of the maximum, NaN and -inf entries never win (all of them: index 0). out_tok[r] = map ? map[i] : i,
// out_idx[r] = i (either may be null).
__global__ __launch_bounds__(1024) void k_argmax_rows(const float* __restrict__ x, int n, long ld,
                                                      const int32_t* __restrict__ map, int32_t* __restrict__ out_tok,
                                                      int32_t* __restrict__ out_idx) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* xr = x + (size_t) blockIdx.x * ld;
    float bv = -INFINITY;
    int bi = n;  // sentinel: nothing found
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        if (v > bv) { bv = v; bi = i; }
    }
    // (value, index) max with the smaller index on equal values; sentinel entries carry -inf / n
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(bv, o, WS);
        const int oi = __shfl_xor(bi, o, WS);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    if (lane == 0) { sv[w] = bv; si[w] = bi; }
    __syncthreads();
    if (w == 0) {
        bv = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : n;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, WS);
            const int oi = __shfl_xor(bi, o, WS);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            const int i = bi >= n ? 0 : bi;
            if (out_tok) out_tok[blockIdx.x] = map ? map[i] : i;
            if (out_idx) out_idx[blockIdx.x] = i;
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// KDA decode block in one launch: ssm_conv (k_kda_conv) + the KDA step (glm_kda_step_v2_kernel, its
// prologue on) + the output gate to q8_1 (k_kda_gate_q8). Each piece is the original code; per value the arithmetic
// is unchanged (bitwise the three launches). Block (head h, column group y) of the v2 step layout:
//  - every block convolves its head's 128 q and 128 k channels (into LDS; cheap) and its own v columns (v conv state
//    snapshots written by their owner block); the q / k conv state snapshots and the gate are written by the head's
//    last block to finish (atomic counter per head, reset by that block), when no block of the head still reads the
//    old q / k state and every column of o is in memory.
template <int CTRL>
__device__ __forceinline__ float kf_dpp(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, false));
}
// = talus::glm_spike glm_wave32_sum_dpp
__device__ __forceinline__ float kf_wave32_sum_dpp(float v) {
    v += kf_dpp<0xB1>(v);
    v += kf_dpp<0x4E>(v);
    v += kf_dpp<0x141>(v);
    v += kf_dpp<0x140>(v);
    v += __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    return v;
}
// one conv channel over T tokens (k_kda_conv's loop); writes the outputs to o[t * ld], the state snapshots if `snap`
__device__ __forceinline__ void kf_conv_channel(const float* __restrict__ x, float* st, const int32_t* __restrict__ ids,
                                                int K, const float* __restrict__ w, float* o, int ld, int T, int C,
                                                int g, int c, bool snap) {
    const size_t slot_stride = (size_t) 3 * C * 3;
    const size_t ch = ((size_t) g * C + c) * 3;
    const float* sin = st + (size_t) ids[0] * slot_stride + ch;
    float s0 = sin[0], s1 = sin[1], s2 = sin[2];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float xv = x[(size_t) t * C + c];
        float a = s0 * w0;
        a += s1 * w1;
        a += s2 * w2;
        a += xv * w3;
        if (o) o[t * ld] = a / (1.0f + expf(-a));
        s0 = s1; s1 = s2; s2 = xv;
        const int kk = t - (T - K);
        if (snap && kk >= 0) {
            float* so = st + (size_t) ids[kk] * slot_stride + ch;
            so[0] = s0; so[1] = s1; so[2] = s2;
        }
    }
}

template <int CPW, int NWV>
__global__ __launch_bounds__(32 * NWV) void k_kda_fused(
    const float* __restrict__ qr, const float* __restrict__ kr, const float* __restrict__ vr, float* conv_st,
    const float* __restrict__ cwq, const float* __restrict__ cwk, const float* __restrict__ cwv,
    const float* __restrict__ data_g, const float* __restrict__ data_beta, float* __restrict__ data_state,
    const int32_t* __restrict__ data_ids, const float* __restrict__ data_dtb, const float* __restrict__ data_a,
    float* __restrict__ data_dst, const float* __restrict__ g2, const float* __restrict__ onw,
    q8_1_blk* __restrict__ out, int* __restrict__ cnt, uint32_t H, uint32_t n_tokens, uint32_t K_snap, float scale,
    float pro_lb, float pro_l2_eps, float gate_eps) {
    constexpr uint32_t S_v = 128, NC = CPW * NWV;
    __shared__ float cq[8][S_v], ck[8][S_v], cv[8][NC];
    __shared__ float4 sq[2][32], sk[2][32], sgx[2][32];
    __shared__ float sbeta[2];
    __shared__ int last_blk;

    const uint32_t head_id = blockIdx.x;
    const uint32_t w = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col0 = blockIdx.y * NC;
    const uint32_t col_base = (blockIdx.y * NWV + w) * CPW;
    const uint32_t sv2 = S_v * S_v;
    const int C = (int) (H * S_v);
    const int T = (int) n_tokens;

    // conv: q / k channels of the head (outputs only), this block's v columns (outputs + state snapshots)
    for (int i = threadIdx.x; i < 2 * (int) S_v + (int) NC; i += 32 * NWV) {
        if (i < (int) S_v)
            kf_conv_channel(qr, conv_st, data_ids, (int) K_snap, cwq, &cq[0][i], S_v, T, C, 0, head_id * S_v + i, false);
        else if (i < 2 * (int) S_v)
            kf_conv_channel(kr, conv_st, data_ids, (int) K_snap, cwk, &ck[0][i - S_v], S_v, T, C, 1,
                            head_id * S_v + i - S_v, false);
        else
            kf_conv_channel(vr, conv_st, data_ids, (int) K_snap, cwv, &cv[0][i - 2 * S_v], NC, T, C, 2,
                            head_id * S_v + col0 + i - 2 * S_v, true);
    }
    __syncthreads();

    // the KDA step (glm_kda_step_v2_kernel, n_B = 1, prologue on)
    const int32_t slot_in = (K_snap > 0) ? data_ids[0] : 0;
    const float4* st_in = reinterpret_cast<const float4*>(data_state + static_cast<size_t>(slot_in) * sv2 * H + head_id * sv2);
    float4 s4[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) s4[c] = st_in[(col_base + c) * 32u + lane];
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const uint32_t token_idx = t;
        const uint32_t vb = token_idx * S_v * H + head_id * S_v;
        const uint32_t buf = t & 1u;
        if (w == 0) {
            float4 q = reinterpret_cast<const float4*>(&cq[t][0])[lane];
            float4 k = reinterpret_cast<const float4*>(&ck[t][0])[lane];
            float4 g4;
            float beta_val = data_beta[token_idx * H + head_id];
            float qq = 0.0f, kk = 0.0f;
            qq = fmaf(q.x, q.x, qq); qq = fmaf(q.y, q.y, qq); qq = fmaf(q.z, q.z, qq); qq = fmaf(q.w, q.w, qq);
            kk = fmaf(k.x, k.x, kk); kk = fmaf(k.y, k.y, kk); kk = fmaf(k.z, k.z, kk); kk = fmaf(k.w, k.w, kk);
            qq = kf_wave32_sum_dpp(qq);
            kk = kf_wave32_sum_dpp(kk);
            const float n = static_cast<float>(S_v);
            const float eps = pro_l2_eps / n;
            const float fq = rsqrtf(qq / n + eps), fk = rsqrtf(kk / n + eps), isn = rsqrtf(n);
            q = make_float4(fq * q.x * isn, fq * q.y * isn, fq * q.z * isn, fq * q.w * isn);
            k = make_float4(fk * k.x * isn, fk * k.y * isn, fk * k.z * isn, fk * k.w * isn);
            beta_val = 1.0f / (1.0f + expf(-beta_val));
            const float a_h = data_a[head_id];
            const float4 gr = reinterpret_cast<const float4*>(data_g + vb)[lane];
            const float4 db = reinterpret_cast<const float4*>(data_dtb + head_id * S_v)[lane];
            auto gx = [&](float gv, float dv) { return expf((1.0f / (1.0f + expf((gv + dv) * a_h))) * pro_lb); };
            g4 = make_float4(gx(gr.x, db.x), gx(gr.y, db.y), gx(gr.z, db.z), gx(gr.w, db.w));
            sq[buf][lane] = q; sk[buf][lane] = k; sgx[buf][lane] = g4;
            if (lane == 0) sbeta[buf] = beta_val;
        }
        __syncthreads();
        const float4 q = sq[buf][lane], k = sk[buf][lane], g = sgx[buf][lane];
        const float beta_val = sbeta[buf];
        const uint32_t attn_off = (token_idx * H + head_id) * S_v;
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const uint32_t col = col_base + c;
            const float v_val = cv[t][col - col0];
            float4& sc = s4[c];
            sc.x *= g.x; sc.y *= g.y; sc.z *= g.z; sc.w *= g.w;
            float kv = 0.0f;
            kv = fmaf(sc.x, k.x, kv); kv = fmaf(sc.y, k.y, kv); kv = fmaf(sc.z, k.z, kv); kv = fmaf(sc.w, k.w, kv);
            const float kv_col = kf_wave32_sum_dpp(kv);
            const float delta = (v_val - kv_col) * beta_val;
            sc.x = fmaf(k.x, delta, sc.x); sc.y = fmaf(k.y, delta, sc.y); sc.z = fmaf(k.z, delta, sc.z); sc.w = fmaf(k.w, delta, sc.w);
            float at = 0.0f;
            at = fmaf(sc.x, q.x, at); at = fmaf(sc.y, q.y, at); at = fmaf(sc.z, q.z, at); at = fmaf(sc.w, q.w, at);
            const float attn_col = kf_wave32_sum_dpp(at);
            if (lane == 0) data_dst[attn_off + col] = attn_col * scale;
        }
        if (K_snap > 0 && t >= n_tokens - K_snap) {
            const uint32_t k_idx = t - (n_tokens - K_snap);
            const int32_t slot_out = data_ids[k_idx];
            if (slot_out >= 0) {
                float4* st_out = reinterpret_cast<float4*>(data_state + static_cast<size_t>(slot_out) * sv2 * H + head_id * sv2);
#pragma unroll
                for (int c = 0; c < CPW; ++c) st_out[(col_base + c) * 32u + lane] = s4[c];
            }
        }
    }

    // the head's last block: q / k conv state snapshots, then the gate of every token's row of the head
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last_blk = atomicAdd(&cnt[head_id], 1) == (int) gridDim.y - 1;
    __syncthreads();
    if (!last_blk) return;
    __threadfence();
    for (int i = threadIdx.x; i < 2 * (int) S_v; i += 32 * NWV) {
        if (i < (int) S_v)
            kf_conv_channel(qr, conv_st, data_ids, (int) K_snap, cwq, nullptr, 0, T, C, 0, head_id * S_v + i, true);
        else
            kf_conv_channel(kr, conv_st, data_ids, (int) K_snap, cwk, nullptr, 0, T, C, 1, head_id * S_v + i - S_v, true);
    }
    for (int t = (int) w; t < T; t += NWV) {   // k_kda_gate_q8, row r = (t, h)
        const int r = t * (int) H + (int) head_id;
        const int D = (int) S_v;
        const float* x = data_dst + (size_t) r * D;
        float v[4];
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) { v[j] = x[lane + j * WS]; ss += v[j] * v[j]; }
        ss = warp_sum(ss);
        const float sc = 1.0f / sqrtf(ss / (float) D + gate_eps);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = lane + j * WS;
            const float val = v[j] * sc * onw[i] * sigmoidf_(g2[(size_t) r * D + i]);
            q8_1_store_lane(val, out + ((size_t) r * D + j * WS) / 32, (int) lane);
        }
    }
    if (threadIdx.x == 0) cnt[head_id] = 0;
}

// prompt-window KDA conv + step in one launch (bitwise tkg_kda_conv then tkg_glm_kda_step: the conv's
// expressions per channel in token order, the v2/v3 step's per token), the raw q / k / v read one token ahead.
// grid (H, 128 / (CPW NWV)); wave 0 convolves the head's q / k channels (lane: 4 + 4), every lane its wave's CPW v
// columns. Conv state snapshots: v by lane 0 of the owning wave; q / k (computed by every block of the head) by the
// head's last block to finish, from LDS (the other blocks may still read the input slot until then).
template <int CPW, int NWV>
__global__ __launch_bounds__(32 * NWV) void k_kda_step_conv_pf(
    const float* __restrict__ qr, const float* __restrict__ kr, const float* __restrict__ vr, float* conv_st,
    const float* __restrict__ cwq, const float* __restrict__ cwk, const float* __restrict__ cwv,
    const float* __restrict__ data_g, const float* __restrict__ data_beta, float* __restrict__ data_state,
    const int32_t* __restrict__ data_ids, const float* __restrict__ data_dtb, const float* __restrict__ data_a,
    float* __restrict__ data_dst, int* __restrict__ cnt, uint32_t H, uint32_t n_tokens, uint32_t K_snap, float scale,
    float pro_lb, float pro_l2_eps) {
    constexpr uint32_t S_v = 128;
    __shared__ float4 sq[2][32], sk[2][32], sgx[2][32];
    __shared__ float sbeta[2];
    __shared__ float csnap[3][2 * S_v][3];   // q / k conv state at the snapshot tokens (wave 0)
    __shared__ int last_blk;
    const uint32_t head_id = blockIdx.x;
    const uint32_t w = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t col_base = (blockIdx.y * NWV + w) * CPW;
    const uint32_t sv2 = S_v * S_v;
    const int C = (int) (H * S_v);
    const size_t slot_stride = (size_t) 3 * C * 3;
    const float* cin = conv_st + (size_t) data_ids[0] * slot_stride;

    // conv channels of this thread: wave 0 q / k channels head * 128 + 4 lane + i; every lane v columns col_base + c
    float qs[4][3], ks[4][3], qw[4][4], kw[4][4];
    if (w == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int cq = (int) (head_id * S_v + 4 * lane) + i;
#pragma unroll
            for (int j = 0; j < 3; ++j) { qs[i][j] = cin[((size_t) 0 * C + cq) * 3 + j]; ks[i][j] = cin[((size_t) 1 * C + cq) * 3 + j]; }
#pragma unroll
            for (int j = 0; j < 4; ++j) { qw[i][j] = cwq[cq * 4 + j]; kw[i][j] = cwk[cq * 4 + j]; }
        }
    }
    float vs[CPW][3], vw[CPW][4];
#pragma unroll
    for (int c = 0; c < CPW; ++c) {
        const int cv = (int) (head_id * S_v + col_base) + c;
#pragma unroll
        for (int j = 0; j < 3; ++j) vs[c][j] = cin[((size_t) 2 * C + cv) * 3 + j];
#pragma unroll
        for (int j = 0; j < 4; ++j) vw[c][j] = cwv[cv * 4 + j];
    }
    auto conv = [](float (&st)[3], const float (&wt)[4], float xv) {   // k_kda_conv's step
        float a = st[0] * wt[0];
        a += st[1] * wt[1];
        a += st[2] * wt[2];
        a += xv * wt[3];
        st[0] = st[1]; st[1] = st[2]; st[2] = xv;
        return a / (1.0f + expf(-a));
    };

    const int32_t slot_in = (K_snap > 0) ? data_ids[0] : 0;
    const float4* st_in = reinterpret_cast<const float4*>(data_state + static_cast<size_t>(slot_in) * sv2 * H + head_id * sv2);
    float4 s4[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) s4[c] = st_in[(col_base + c) * 32u + lane];

    float4 nq = make_float4(0.f, 0.f, 0.f, 0.f), nk = nq, ng = nq, db = nq;
    float nbeta = 0.0f, a_h = 0.0f;
    float nv[CPW];
    auto load_tok = [&](uint32_t t) {
        const uint32_t vb = t * S_v * H + head_id * S_v;
        if (w == 0) {
            nq = reinterpret_cast<const float4*>(qr + vb)[lane];
            nk = reinterpret_cast<const float4*>(kr + vb)[lane];
            ng = reinterpret_cast<const float4*>(data_g + vb)[lane];
            nbeta = data_beta[t * H + head_id];
        }
#pragma unroll
        for (int c = 0; c < CPW; ++c) nv[c] = vr[vb + col_base + c];
    };
    if (w == 0) {
        a_h = data_a[head_id];
        db = reinterpret_cast<const float4*>(data_dtb + head_id * S_v)[lane];
    }
    if (n_tokens > 0) load_tok(0);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const uint32_t token_idx = t;
        const uint32_t buf = t & 1u;
        const int kk = (int) t - (int) (n_tokens - K_snap);   // conv snapshot index (k_kda_conv: K = K_snap)
        float vcur[CPW];
#pragma unroll
        for (int c = 0; c < CPW; ++c) vcur[c] = conv(vs[c], vw[c], nv[c]);
        if (w == 0) {
            const float rq[4] = {nq.x, nq.y, nq.z, nq.w}, rk[4] = {nk.x, nk.y, nk.z, nk.w};
            float cq4[4], ck4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { cq4[i] = conv(qs[i], qw[i], rq[i]); ck4[i] = conv(ks[i], kw[i], rk[i]); }
            if (kk >= 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 3; ++j) { csnap[kk][4 * lane + i][j] = qs[i][j]; csnap[kk][S_v + 4 * lane + i][j] = ks[i][j]; }
            }
            float4 q = make_float4(cq4[0], cq4[1], cq4[2], cq4[3]), k = make_float4(ck4[0], ck4[1], ck4[2], ck4[3]);
            float4 g4;
            float beta_val = nbeta;
            const float4 gr = ng;
            float qq = 0.0f, kq = 0.0f;
            qq = fmaf(q.x, q.x, qq); qq = fmaf(q.y, q.y, qq); qq = fmaf(q.z, q.z, qq); qq = fmaf(q.w, q.w, qq);
            kq = fmaf(k.x, k.x, kq); kq = fmaf(k.y, k.y, kq); kq = fmaf(k.z, k.z, kq); kq = fmaf(k.w, k.w, kq);
            qq = kf_wave32_sum_dpp(qq);
            kq = kf_wave32_sum_dpp(kq);
            const float n = static_cast<float>(S_v);
            const float eps = pro_l2_eps / n;
            const float fq = rsqrtf(qq / n + eps), fk = rsqrtf(kq / n + eps), isn = rsqrtf(n);
            q = make_float4(fq * q.x * isn, fq * q.y * isn, fq * q.z * isn, fq * q.w * isn);
            k = make_float4(fk * k.x * isn, fk * k.y * isn, fk * k.z * isn, fk * k.w * isn);
            beta_val = 1.0f / (1.0f + expf(-beta_val));
            auto gx = [&](float gv, float dv) { return expf((1.0f / (1.0f + expf((gv + dv) * a_h))) * pro_lb); };
            g4 = make_float4(gx(gr.x, db.x), gx(gr.y, db.y), gx(gr.z, db.z), gx(gr.w, db.w));
            sq[buf][lane] = q; sk[buf][lane] = k; sgx[buf][lane] = g4;
            if (lane == 0) sbeta[buf] = beta_val;
        }
        if (kk >= 0 && lane == 0) {   // v conv state snapshots (this wave's columns)
#pragma unroll
            for (int c = 0; c < CPW; ++c) {
                float* so = conv_st + (size_t) data_ids[kk] * slot_stride + ((size_t) 2 * C + head_id * S_v + col_base + c) * 3;
                so[0] = vs[c][0]; so[1] = vs[c][1]; so[2] = vs[c][2];
            }
        }
        if (t + 1 < n_tokens) load_tok(t + 1);
        __syncthreads();
        const float4 q = sq[buf][lane], k = sk[buf][lane], g = sgx[buf][lane];
        const float beta_val = sbeta[buf];
        const uint32_t attn_off = (token_idx * H + head_id) * S_v;
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const uint32_t col = col_base + c;
            const float v_val = vcur[c];
            float4& sc = s4[c];
            sc.x *= g.x; sc.y *= g.y; sc.z *= g.z; sc.w *= g.w;
            float kv = 0.0f;
            kv = fmaf(sc.x, k.x, kv); kv = fmaf(sc.y, k.y, kv); kv = fmaf(sc.z, k.z, kv); kv = fmaf(sc.w, k.w, kv);
            const float kv_col = kf_wave32_sum_dpp(kv);
            const float delta = (v_val - kv_col) * beta_val;
            sc.x = fmaf(k.x, delta, sc.x); sc.y = fmaf(k.y, delta, sc.y); sc.z = fmaf(k.z, delta, sc.z); sc.w = fmaf(k.w, delta, sc.w);
            float at = 0.0f;
            at = fmaf(sc.x, q.x, at); at = fmaf(sc.y, q.y, at); at = fmaf(sc.z, q.z, at); at = fmaf(sc.w, q.w, at);
            const float attn_col = kf_wave32_sum_dpp(at);
            if (lane == 0) data_dst[attn_off + col] = attn_col * scale;
        }
        if (K_snap > 0 && t >= n_tokens - K_snap) {
            const uint32_t k_idx = t - (n_tokens - K_snap);
            const int32_t slot_out = data_ids[k_idx];
            if (slot_out >= 0) {
                float4* st_out = reinterpret_cast<float4*>(data_state + static_cast<size_t>(slot_out) * sv2 * H + head_id * sv2);
#pragma unroll
                for (int c = 0; c < CPW; ++c) st_out[(col_base + c) * 32u + lane] = s4[c];
            }
        }
    }
    // q / k conv snapshots: the head's last block (every block has read the input slot by then)
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last_blk = atomicAdd(&cnt[head_id], 1) == (int) gridDim.y - 1;
    __syncthreads();
    if (!last_blk) return;
    __threadfence();
    for (int i = threadIdx.x; i < (int) (K_snap * 2 * S_v); i += 32 * NWV) {
        const int kk = i / (2 * S_v), ch = i % (2 * S_v);
        const int g = ch < (int) S_v ? 0 : 1, c = (int) head_id * S_v + (ch % S_v);
        float* so = conv_st + (size_t) data_ids[kk] * slot_stride + ((size_t) g * C + c) * 3;
        so[0] = csnap[kk][ch][0]; so[1] = csnap[kk][ch][1]; so[2] = csnap[kk][ch][2];
    }
    if (threadIdx.x == 0) cnt[head_id] = 0;
}

// router logits (k_f32mv) and routing (k_router) in one launch: the f32 mat-vec blocks as k_f32mv, then the
// last block to finish (atomic counter, reset by it) runs k_router's code for every token (wave t = token t).
// Bitwise the two launches.
__device__ __forceinline__ void router_token(const float* __restrict__ logits, const float* __restrict__ bias,
                                             int32_t* __restrict__ ids, float* __restrict__ wout, int n_exp, int k,
                                             float scale, int t, int lane) {
    constexpr int MAXE = 16;  // per-lane experts (512 / 32)
    float prob[MAXE], sel[MAXE];
    const int per = (n_exp + WS - 1) / WS;
    for (int j = 0; j < MAXE; ++j) {
        const int e = lane + j * WS;
        if (j < per && e < n_exp) {
            prob[j] = sigmoidf_(logits[(size_t) t * n_exp + e]);
            sel[j] = prob[j] + (bias ? bias[e] : 0.0f);
        } else {
            prob[j] = 0.0f;
            sel[j] = -INFINITY;
        }
    }
    float wsel[16];
    int isel[16];
    float wsum = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bj = -1;
        for (int j = 0; j < per; ++j) {
            if (sel[j] > bv) { bv = sel[j]; bj = j; }
        }
        int be = bj >= 0 ? lane + bj * WS : 0x7fffffff;
        float mv = warp_max(bv);
        int cand = (bv == mv) ? be : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) cand = min(cand, __shfl_xor(cand, o, WS));
        const int owner = cand % WS, oj = cand / WS;
        float pv = 0.0f;
        if (lane == owner) { pv = prob[oj]; sel[oj] = -INFINITY; }
        pv = __shfl(pv, owner, WS);
        isel[r] = cand;
        wsel[r] = pv;
        wsum += pv;
    }
    if (lane == 0) {
        const float inv = scale / fmaxf(wsum, 6.103515625e-5f);
        for (int r = 0; r < k; ++r) {
            ids[(size_t) t * k + r] = isel[r];
            wout[(size_t) t * k + r] = wsel[r] * inv;
        }
    }
}

template <int T>
__global__ __launch_bounds__(256) void k_f32mv_router(const float* __restrict__ W, const float* __restrict__ x,
                                                      float* __restrict__ y, int n_in, int n_out, int nt,
                                                      const float* __restrict__ bias, int32_t* __restrict__ ids,
                                                      float* __restrict__ wout, int k, float scale,
                                                      int* __restrict__ cnt) {
    __shared__ int last_blk;
    {
        const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
        const int lane = threadIdx.x & 31;
        if (r < n_out) {
            const float4* w4 = reinterpret_cast<const float4*>(W + (size_t) r * n_in);
            float acc[T];
#pragma unroll
            for (int t = 0; t < T; ++t) acc[t] = 0.0f;
#pragma unroll 4
            for (int i = lane; i < n_in / 4; i += 32) {
                const float4 w = w4[i];
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 v = reinterpret_cast<const float4*>(x + (size_t) t * n_in)[i];
                    acc[t] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
                }
            }
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float v = warp_sum(acc[t]);
                if (lane == 0) y[(size_t) t * n_out + r] = v;
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last_blk = atomicAdd(cnt, 1) == (int) gridDim.x - 1;
    __syncthreads();
    if (!last_blk) return;
    __threadfence();
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int t = w; t < nt; t += 8) router_token(y, bias, ids, wout, n_out, k, scale, t, lane);
    if (threadIdx.x == 0) *cnt = 0;
}

template <int TYPE>
int kq_ids_v3_launch(const void* w0, const void* w1, size_t expert_bytes, const int32_t* ids, int n_ids, int k,
                     const float* x, int x_per_id, float* y0, float* y1, int n_in, int n_out, void* s) {
    const int nb = n_in / 256;
    const int rows_per_block = 8 * (32 / nb);
    const dim3 g((n_out + rows_per_block - 1) / rows_per_block, n_ids, w1 ? 2 : 1);
    hipLaunchKernelGGL((k_kq_ids_v3<TYPE>), g, dim3(256), 0, S(s), static_cast<const uint8_t*>(w0),
                       static_cast<const uint8_t*>(w1), expert_bytes, ids, k, x, x_per_id, y0, y1, n_in, n_out);
    return last();
}

// ------------------------------------------------------------------------------------------------------------------
// fused mHC (v2): post + next mixes in one pass, pre (Sinkhorn + collapse + norm) parallel over slices.
// parts2 row layout per (token, slice): [0..23] mixes, [24..33] Gram of the 4 streams (00,01,02,03,11,12,13,22,23,33);
// row stride 64 floats; HC_NS slices of n_embd / HC_NS elements of every stream.
constexpr int HC_NS = 8;
constexpr int HC_PS = 64;

__device__ __forceinline__ int gram_idx(int a, int b) {   // a <= b
    constexpr int base[4] = {0, 4, 7, 9};
    return base[a] + (b - a);
}

// grid (HC_NS, T), block 256. first: streams = x (embedding) repeated; else post from xo / pc. w_next (Q8_0 [24][4 n_embd])
// may be null (no mixes, Gram still written).
template <int MODE, typename ST = float>
__global__ __launch_bounds__(256) void k_hc_post_mix(const float* __restrict__ xo, ST* xhc,
                                                     const float* __restrict__ pc, const block_q8_0* __restrict__ w,
                                                     float* __restrict__ parts, int n_embd) {
    constexpr int M = 24, NV = M + 10;
    __shared__ float sh[NV][8];
    const int p = blockIdx.x, t = blockIdx.y;
    const int slice = n_embd / HC_NS;
    ST* xs = xhc + (size_t) t * 4 * n_embd;
    using H = HcS<ST>;
    float acc[NV];
#pragma unroll
    for (int r = 0; r < NV; ++r) acc[r] = 0.0f;
    float c[20];
    if (MODE == 0) {
#pragma unroll
        for (int j = 0; j < 20; ++j) c[j] = pc[(size_t) t * 32 + j];
    }
    const int K = 4 * n_embd;
    for (int ii = threadIdx.x * 2; ii < slice; ii += 512) {
        const int i = p * slice + ii;
        float v[4][2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            if (MODE == 1) {
                const float x = H::rnd(xo[(size_t) t * n_embd + i + u]);
#pragma unroll
                for (int d = 0; d < 4; ++d) v[d][u] = x;
            } else if (MODE == 2) {
#pragma unroll
                for (int d = 0; d < 4; ++d) v[d][u] = H::ld(xs, d * n_embd + i + u);
            } else {
                const float x = xo[(size_t) t * n_embd + i + u];
                const float r0 = H::ld(xs, i + u), r1 = H::ld(xs, n_embd + i + u), r2 = H::ld(xs, 2 * n_embd + i + u),
                            r3 = H::ld(xs, 3 * n_embd + i + u);
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    float a = x * c[d];
                    a += r0 * c[4 + d];
                    a += r1 * c[8 + d];
                    a += r2 * c[12 + d];
                    a += r3 * c[16 + d];
                    v[d][u] = H::rnd(a);
                }
            }
        }
        if (MODE != 2) {
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                H::st2(xs, d * n_embd + i, v[d][0], v[d][1]);
            }
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = a; b < 4; ++b) acc[M + gram_idx(a, b)] += v[a][0] * v[b][0] + v[a][1] * v[b][1];
        if (w) {
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const int gi = d * n_embd + i;   // flat column
                const int bi = gi >> 5, o = gi & 31;
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const block_q8_0& blk = w[(size_t) r * (K >> 5) + bi];
                    const float dq = __half2float(blk.d);
                    acc[r] += dq * ((float) blk.qs[o] * v[d][0] + (float) blk.qs[o + 1] * v[d][1]);
                }
            }
        }
    }
    const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < NV; ++r) {
        const float val = warp_sum(acc[r]);
        if (lane == 0) sh[r][wv] = val;
    }
    __syncthreads();
    if (threadIdx.x < NV) {
        float val = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) val += sh[threadIdx.x][i];
        parts[((size_t) t * HC_NS + p) * HC_PS + threadIdx.x] = val;
    }
}

template <typename ST = float, typename EY = float>
__global__ __launch_bounds__(256) void k_hc_post_mix_moe(const EY* __restrict__ ey, const float* __restrict__ ew,
                                                         const float* __restrict__ sh_out, ST* xhc,
                                                         const float* __restrict__ pc, const block_q8_0* __restrict__ w,
                                                         float* __restrict__ parts, int k, int n_embd) {
    constexpr int M = 24, NV = M + 10;
    __shared__ float sh[NV][8];
    const int p = blockIdx.x, t = blockIdx.y;
    const int slice = n_embd / HC_NS;
    ST* xs = xhc + (size_t) t * 4 * n_embd;
    using H = HcS<ST>;
    float acc[NV];
#pragma unroll
    for (int r = 0; r < NV; ++r) acc[r] = 0.0f;
    float c[20];
#pragma unroll
    for (int j = 0; j < 20; ++j) c[j] = pc[(size_t) t * 32 + j];
    const int K = 4 * n_embd;
    for (int ii = threadIdx.x * 2; ii < slice; ii += 512) {
        const int i = p * slice + ii;
        float v[4][2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            float x = 0.0f;
            for (int j = 0; j < k; ++j) x += ew[(size_t) t * k + j] * (float) ey[((size_t) t * k + j) * n_embd + i + u];
            if (sh_out) x += sh_out[(size_t) t * n_embd + i + u];
            const float r0 = H::ld(xs, i + u), r1 = H::ld(xs, n_embd + i + u), r2 = H::ld(xs, 2 * n_embd + i + u),
                            r3 = H::ld(xs, 3 * n_embd + i + u);
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                float a = x * c[d];
                a += r0 * c[4 + d];
                a += r1 * c[8 + d];
                a += r2 * c[12 + d];
                a += r3 * c[16 + d];
                v[d][u] = H::rnd(a);
            }
        }
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            H::st2(xs, d * n_embd + i, v[d][0], v[d][1]);
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = a; b < 4; ++b) acc[M + gram_idx(a, b)] += v[a][0] * v[b][0] + v[a][1] * v[b][1];
        if (w) {
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const int gi = d * n_embd + i;
                const int bi = gi >> 5, o = gi & 31;
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const block_q8_0& blk = w[(size_t) r * (K >> 5) + bi];
                    const float dq = __half2float(blk.d);
                    acc[r] += dq * ((float) blk.qs[o] * v[d][0] + (float) blk.qs[o + 1] * v[d][1]);
                }
            }
        }
    }
    const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < NV; ++r) {
        const float val = warp_sum(acc[r]);
        if (lane == 0) sh[r][wv] = val;
    }
    __syncthreads();
    if (threadIdx.x < NV) {
        float val = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) val += sh[threadIdx.x][i];
        parts[((size_t) t * HC_NS + p) * HC_PS + threadIdx.x] = val;
    }
}

// grid (HC_NS, T), block 256: every block derives pre / post / comb (warp 0, redundantly), block 0 stores pc; then its
// slice of y = rmsnorm(sum_h pre_h x_h) * w (sum of squares from the Gram partials) and the q8_1 copy.
template <typename ST = float>
__global__ __launch_bounds__(256) void k_hc_pre2(const ST* __restrict__ xhc, const float* __restrict__ parts,
                                                 const float* __restrict__ scale, const float* __restrict__ base,
                                                 const float* __restrict__ norm_w, float* __restrict__ y,
                                                 float* __restrict__ pc, int n_embd, float rms_eps, float norm_eps,
                                                 float hc_eps, int iters, q8_1_blk* __restrict__ yq) {
    constexpr int HC = 4;
    __shared__ float pre_s[HC];
    __shared__ float s_norm;
    const int p = blockIdx.x, t = blockIdx.y;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float m = 0.0f;
        if (lane < 24 + 10) {
            for (int q = 0; q < HC_NS; ++q) m += parts[((size_t) t * HC_NS + q) * HC_PS + lane];
        }
        // Gram in lanes 24..33: G00, G01, G02, G03, G11, G12, G13, G22, G23, G33
        const float g00 = __shfl(m, 24, WS), g01 = __shfl(m, 25, WS), g02 = __shfl(m, 26, WS), g03 = __shfl(m, 27, WS);
        const float g11 = __shfl(m, 28, WS), g12 = __shfl(m, 29, WS), g13 = __shfl(m, 30, WS), g22 = __shfl(m, 31, WS);
        // G23 and G33 (indices 32, 33: past the wave) are summed by lane 0
        float e23 = 0.0f, e33 = 0.0f;
        if (lane == 0) {
            for (int q = 0; q < HC_NS; ++q) {
                e23 += parts[((size_t) t * HC_NS + q) * HC_PS + 32];
                e33 += parts[((size_t) t * HC_NS + q) * HC_PS + 33];
            }
        }
        const float g23 = __shfl(e23, 0, WS);
        const float g33 = __shfl(e33, 0, WS);
        const float ss_flat = g00 + g11 + g22 + g33;
        const float r = 1.0f / sqrtf(ss_flat / (float) (HC * n_embd) + rms_eps);
        const float mr = m * r;
        float pre_l = 0.0f;
        if (lane < HC) pre_l = sigmoidf_(mr * scale[0] + base[lane]) + hc_eps;
        if (lane >= HC && lane < 2 * HC && p == 0) pc[(size_t) t * 32 + lane - HC] = 2.0f * sigmoidf_(mr * scale[1] + base[lane]);
        const float mc = __shfl(mr, 2 * HC + (lane & 15), WS);
        float c = mc * scale[2] + base[2 * HC + (lane & 15)];
        float mx = fmaxf(c, __shfl_xor(c, 1, WS));
        mx = fmaxf(mx, __shfl_xor(mx, 2, WS));
        c = expf(c - mx);
        float sm = c + __shfl_xor(c, 1, WS);
        sm += __shfl_xor(sm, 2, WS);
        c = c / sm + hc_eps;
        auto col = [&](float v) { float a = v + __shfl_xor(v, 4, WS); return a + __shfl_xor(a, 8, WS); };
        auto row = [&](float v) { float a = v + __shfl_xor(v, 1, WS); return a + __shfl_xor(a, 2, WS); };
        c = c / (col(c) + hc_eps);
        for (int i = 1; i < iters; ++i) {
            c = c / (row(c) + hc_eps);
            c = c / (col(c) + hc_eps);
        }
        if (lane < 16 && p == 0) pc[(size_t) t * 32 + HC + lane] = c;
        const float q0 = __shfl(pre_l, 0, WS), q1 = __shfl(pre_l, 1, WS), q2 = __shfl(pre_l, 2, WS), q3 = __shfl(pre_l, 3, WS);
        if (lane == 0) {
            const float ssy = q0 * q0 * g00 + q1 * q1 * g11 + q2 * q2 * g22 + q3 * q3 * g33 +
                              2.0f * (q0 * q1 * g01 + q0 * q2 * g02 + q0 * q3 * g03 + q1 * q2 * g12 + q1 * q3 * g13 + q2 * q3 * g23);
            s_norm = 1.0f / sqrtf(fmaxf(ssy, 0.0f) / (float) n_embd + norm_eps);
            pre_s[0] = q0; pre_s[1] = q1; pre_s[2] = q2; pre_s[3] = q3;
        }
    }
    __syncthreads();
    const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3], sn = s_norm;
    const ST* xt = xhc + (size_t) t * HC * n_embd;
    using H = HcS<ST>;
    const int slice = n_embd / HC_NS;
    float* yt = y + (size_t) t * n_embd;
    for (int ii = threadIdx.x; ii < slice; ii += blockDim.x) {
        const int i = p * slice + ii;
        const float v = (H::ld(xt, i) * p0 + H::ld(xt, n_embd + i) * p1 + H::ld(xt, 2 * n_embd + i) * p2 +
                         H::ld(xt, 3 * n_embd + i) * p3) * sn * norm_w[i];
        yt[i] = v;
        if (yq) q8_1_store_lane(v, yq + ((size_t) t * n_embd + i) / 32, threadIdx.x & 31);
    }
}

// k_hc_pre2's work for token t in one block (the head's last mix block): the mixes / Sinkhorn once (k_hc_pre2
// runs them identically in each of its HC_NS blocks; block 0 stores pc), then every slice in k_hc_pre2's order.
__device__ __forceinline__ void hc_pre2_block(const float* __restrict__ xhc, const float* __restrict__ parts,
                                              const float* __restrict__ scale, const float* __restrict__ base,
                                              const float* __restrict__ norm_w, float* __restrict__ y,
                                              float* __restrict__ pc, int n_embd, float rms_eps, float norm_eps,
                                              float hc_eps, int iters, q8_1_blk* __restrict__ yq, const int t,
                                              float* pre_s, float& s_norm) {
    constexpr int HC = 4;

    const int p = 0;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float m = 0.0f;
        if (lane < 24 + 10) {
            for (int q = 0; q < HC_NS; ++q) m += parts[((size_t) t * HC_NS + q) * HC_PS + lane];
        }
        // Gram in lanes 24..33: G00, G01, G02, G03, G11, G12, G13, G22, G23, G33
        const float g00 = __shfl(m, 24, WS), g01 = __shfl(m, 25, WS), g02 = __shfl(m, 26, WS), g03 = __shfl(m, 27, WS);
        const float g11 = __shfl(m, 28, WS), g12 = __shfl(m, 29, WS), g13 = __shfl(m, 30, WS), g22 = __shfl(m, 31, WS);
        // G23 and G33 (indices 32, 33: past the wave) are summed by lane 0
        float e23 = 0.0f, e33 = 0.0f;
        if (lane == 0) {
            for (int q = 0; q < HC_NS; ++q) {
                e23 += parts[((size_t) t * HC_NS + q) * HC_PS + 32];
                e33 += parts[((size_t) t * HC_NS + q) * HC_PS + 33];
            }
        }
        const float g23 = __shfl(e23, 0, WS);
        const float g33 = __shfl(e33, 0, WS);
        const float ss_flat = g00 + g11 + g22 + g33;
        const float r = 1.0f / sqrtf(ss_flat / (float) (HC * n_embd) + rms_eps);
        const float mr = m * r;
        float pre_l = 0.0f;
        if (lane < HC) pre_l = sigmoidf_(mr * scale[0] + base[lane]) + hc_eps;
        if (lane >= HC && lane < 2 * HC && p == 0) pc[(size_t) t * 32 + lane - HC] = 2.0f * sigmoidf_(mr * scale[1] + base[lane]);
        const float mc = __shfl(mr, 2 * HC + (lane & 15), WS);
        float c = mc * scale[2] + base[2 * HC + (lane & 15)];
        float mx = fmaxf(c, __shfl_xor(c, 1, WS));
        mx = fmaxf(mx, __shfl_xor(mx, 2, WS));
        c = expf(c - mx);
        float sm = c + __shfl_xor(c, 1, WS);
        sm += __shfl_xor(sm, 2, WS);
        c = c / sm + hc_eps;
        auto col = [&](float v) { float a = v + __shfl_xor(v, 4, WS); return a + __shfl_xor(a, 8, WS); };
        auto row = [&](float v) { float a = v + __shfl_xor(v, 1, WS); return a + __shfl_xor(a, 2, WS); };
        c = c / (col(c) + hc_eps);
        for (int i = 1; i < iters; ++i) {
            c = c / (row(c) + hc_eps);
            c = c / (col(c) + hc_eps);
        }
        if (lane < 16 && p == 0) pc[(size_t) t * 32 + HC + lane] = c;
        const float q0 = __shfl(pre_l, 0, WS), q1 = __shfl(pre_l, 1, WS), q2 = __shfl(pre_l, 2, WS), q3 = __shfl(pre_l, 3, WS);
        if (lane == 0) {
            const float ssy = q0 * q0 * g00 + q1 * q1 * g11 + q2 * q2 * g22 + q3 * q3 * g33 +
                              2.0f * (q0 * q1 * g01 + q0 * q2 * g02 + q0 * q3 * g03 + q1 * q2 * g12 + q1 * q3 * g13 + q2 * q3 * g23);
            s_norm = 1.0f / sqrtf(fmaxf(ssy, 0.0f) / (float) n_embd + norm_eps);
            pre_s[0] = q0; pre_s[1] = q1; pre_s[2] = q2; pre_s[3] = q3;
        }
    }
    __syncthreads();
    const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3], sn = s_norm;
    const float* xt = xhc + (size_t) t * HC * n_embd;
    const int slice = n_embd / HC_NS;
    float* yt = y + (size_t) t * n_embd;
    if (slice % (int) blockDim.x == 0 && n_embd % (4 * (int) blockDim.x) == 0) {
        // every slice in order = element threadIdx.x + k * blockDim.x for k = 0, 1, ...: four elements' loads issued
        // together (one block does all HC_NS slices here), each value's expression as above
        for (int i0 = threadIdx.x; i0 < n_embd; i0 += 4 * blockDim.x) {
            float a[4][4], wn[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = i0 + u * blockDim.x;
                a[u][0] = xt[i]; a[u][1] = xt[n_embd + i]; a[u][2] = xt[2 * n_embd + i]; a[u][3] = xt[3 * n_embd + i];
                wn[u] = norm_w[i];
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int i = i0 + u * blockDim.x;
                const float v = (a[u][0] * p0 + a[u][1] * p1 + a[u][2] * p2 + a[u][3] * p3) * sn * wn[u];
                yt[i] = v;
                if (yq) q8_1_store_lane(v, yq + ((size_t) t * n_embd + i) / 32, threadIdx.x & 31);
            }
        }
        return;
    }
    for (int ps = 0; ps < HC_NS; ++ps)
    for (int ii = threadIdx.x; ii < slice; ii += blockDim.x) {
        const int i = ps * slice + ii;
        const float v = (xt[i] * p0 + xt[n_embd + i] * p1 + xt[2 * n_embd + i] * p2 + xt[3 * n_embd + i] * p3) * sn * norm_w[i];
        yt[i] = v;
        if (yq) q8_1_store_lane(v, yq + ((size_t) t * n_embd + i) / 32, threadIdx.x & 31);
    }
}

template <int MODE>
__global__ __launch_bounds__(256) void k_hc_post_mix_pre(const float* __restrict__ xo, float* xhc,
                                                     const float* __restrict__ pc, const block_q8_0* __restrict__ w,
                                                     float* __restrict__ parts, int n_embd, const float* __restrict__ scale, const float* __restrict__ base, const float* __restrict__ norm_w,
    float* __restrict__ y, float rms_eps, float norm_eps, float hc_eps, int iters, q8_1_blk* __restrict__ yq,
    int* __restrict__ cnt) {
    constexpr int M = 24, NV = M + 10;
    __shared__ float sh[NV][8];
    const int p = blockIdx.x, t = blockIdx.y;
    const int slice = n_embd / HC_NS;
    float* xs = xhc + (size_t) t * 4 * n_embd;
    float acc[NV];
#pragma unroll
    for (int r = 0; r < NV; ++r) acc[r] = 0.0f;
    float c[20];
    if (MODE == 0) {
#pragma unroll
        for (int j = 0; j < 20; ++j) c[j] = pc[(size_t) t * 32 + j];
    }
    const int K = 4 * n_embd;
    for (int ii = threadIdx.x * 2; ii < slice; ii += 512) {
        const int i = p * slice + ii;
        float v[4][2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            if (MODE == 1) {
                const float x = xo[(size_t) t * n_embd + i + u];
#pragma unroll
                for (int d = 0; d < 4; ++d) v[d][u] = x;
            } else if (MODE == 2) {
#pragma unroll
                for (int d = 0; d < 4; ++d) v[d][u] = xs[d * n_embd + i + u];
            } else {
                const float x = xo[(size_t) t * n_embd + i + u];
                const float r0 = xs[i + u], r1 = xs[n_embd + i + u], r2 = xs[2 * n_embd + i + u], r3 = xs[3 * n_embd + i + u];
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    float a = x * c[d];
                    a += r0 * c[4 + d];
                    a += r1 * c[8 + d];
                    a += r2 * c[12 + d];
                    a += r3 * c[16 + d];
                    v[d][u] = a;
                }
            }
        }
        if (MODE != 2) {
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                *reinterpret_cast<float2*>(xs + d * n_embd + i) = make_float2(v[d][0], v[d][1]);
            }
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = a; b < 4; ++b) acc[M + gram_idx(a, b)] += v[a][0] * v[b][0] + v[a][1] * v[b][1];
        if (w) {
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const int gi = d * n_embd + i;   // flat column
                const int bi = gi >> 5, o = gi & 31;
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const block_q8_0& blk = w[(size_t) r * (K >> 5) + bi];
                    const float dq = __half2float(blk.d);
                    acc[r] += dq * ((float) blk.qs[o] * v[d][0] + (float) blk.qs[o + 1] * v[d][1]);
                }
            }
        }
    }
    const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < NV; ++r) {
        const float val = warp_sum(acc[r]);
        if (lane == 0) sh[r][wv] = val;
    }
    __syncthreads();
    if (threadIdx.x < NV) {
        float val = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) val += sh[threadIdx.x][i];
        parts[((size_t) t * HC_NS + p) * HC_PS + threadIdx.x] = val;
    }
    // the token's last block runs k_hc_pre2 for the token (atomic counter per token, reset by that block)
    __threadfence();
    __syncthreads();
    __shared__ int last_blk;
    if (threadIdx.x == 0) last_blk = atomicAdd(&cnt[t], 1) == HC_NS - 1;
    __syncthreads();
    if (!last_blk) return;
    __threadfence();
    __shared__ float pre_s2[4];
    __shared__ float s_norm2;
    hc_pre2_block(xhc, parts, scale, base, norm_w, y, const_cast<float*>(pc), n_embd, rms_eps, norm_eps, hc_eps, iters, yq, t, pre_s2, s_norm2);
    if (threadIdx.x == 0) cnt[t] = 0;
}

template <typename EY = float>
__global__ __launch_bounds__(256) void k_hc_post_mix_moe_pre(const EY* __restrict__ ey, const float* __restrict__ ew,
                                                         const float* __restrict__ sh_out, float* xhc,
                                                         const float* __restrict__ pc, const block_q8_0* __restrict__ w,
                                                         float* __restrict__ parts, int k, int n_embd, const float* __restrict__ scale, const float* __restrict__ base, const float* __restrict__ norm_w,
    float* __restrict__ y, float rms_eps, float norm_eps, float hc_eps, int iters, q8_1_blk* __restrict__ yq,
    int* __restrict__ cnt) {
    constexpr int M = 24, NV = M + 10;
    __shared__ float sh[NV][8];
    const int p = blockIdx.x, t = blockIdx.y;
    const int slice = n_embd / HC_NS;
    float* xs = xhc + (size_t) t * 4 * n_embd;
    float acc[NV];
#pragma unroll
    for (int r = 0; r < NV; ++r) acc[r] = 0.0f;
    float c[20];
#pragma unroll
    for (int j = 0; j < 20; ++j) c[j] = pc[(size_t) t * 32 + j];
    const int K = 4 * n_embd;
    for (int ii = threadIdx.x * 2; ii < slice; ii += 512) {
        const int i = p * slice + ii;
        float v[4][2];
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            float x = 0.0f;
            for (int j = 0; j < k; ++j) x += ew[(size_t) t * k + j] * (float) ey[((size_t) t * k + j) * n_embd + i + u];
            if (sh_out) x += sh_out[(size_t) t * n_embd + i + u];
            const float r0 = xs[i + u], r1 = xs[n_embd + i + u], r2 = xs[2 * n_embd + i + u], r3 = xs[3 * n_embd + i + u];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                float a = x * c[d];
                a += r0 * c[4 + d];
                a += r1 * c[8 + d];
                a += r2 * c[12 + d];
                a += r3 * c[16 + d];
                v[d][u] = a;
            }
        }
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            *reinterpret_cast<float2*>(xs + d * n_embd + i) = make_float2(v[d][0], v[d][1]);
        }
#pragma unroll
        for (int a = 0; a < 4; ++a)
#pragma unroll
            for (int b = a; b < 4; ++b) acc[M + gram_idx(a, b)] += v[a][0] * v[b][0] + v[a][1] * v[b][1];
        if (w) {
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                const int gi = d * n_embd + i;
                const int bi = gi >> 5, o = gi & 31;
#pragma unroll
                for (int r = 0; r < M; ++r) {
                    const block_q8_0& blk = w[(size_t) r * (K >> 5) + bi];
                    const float dq = __half2float(blk.d);
                    acc[r] += dq * ((float) blk.qs[o] * v[d][0] + (float) blk.qs[o + 1] * v[d][1]);
                }
            }
        }
    }
    const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
#pragma unroll
    for (int r = 0; r < NV; ++r) {
        const float val = warp_sum(acc[r]);
        if (lane == 0) sh[r][wv] = val;
    }
    __syncthreads();
    if (threadIdx.x < NV) {
        float val = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) val += sh[threadIdx.x][i];
        parts[((size_t) t * HC_NS + p) * HC_PS + threadIdx.x] = val;
    }
    // the token's last block runs k_hc_pre2 for the token (atomic counter per token, reset by that block)
    __threadfence();
    __syncthreads();
    __shared__ int last_blk;
    if (threadIdx.x == 0) last_blk = atomicAdd(&cnt[t], 1) == HC_NS - 1;
    __syncthreads();
    if (!last_blk) return;
    __threadfence();
    __shared__ float pre_s2[4];
    __shared__ float s_norm2;
    hc_pre2_block(xhc, parts, scale, base, norm_w, y, const_cast<float*>(pc), n_embd, rms_eps, norm_eps, hc_eps, iters, yq, t, pre_s2, s_norm2);
    if (threadIdx.x == 0) cnt[t] = 0;
}



// up to 3 Q8_0 mat-vecs with f32 activations in one launch: y_m[t][r] = W_m[r] . x_m[t]  (n_in % 32 == 0, T <= 8)
struct MultiQ8 {
    const block_q8_0* w[3];
    const float* x[3];
    float* y[3];
    int n_out[3];
};
template <int T, bool PF = false>
__global__ __launch_bounds__(256) void k_multi_q8_0(MultiQ8 m, int n_in) {
    constexpr bool glm_mq8_pf_dev = PF;
    const int mi = blockIdx.y;
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= m.n_out[mi]) return;
    const int nb = n_in / 32;
    const block_q8_0* wr = m.w[mi] + (size_t) r * nb;
    const float* xm = m.x[mi];
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    int e0 = lane * 8;
    // 4 K steps' loads issued together (the same per-step expression in ascending order: bitwise the
    // one-step loop below, which waited for each step's loads in turn); GLM_MQ8_PF=0 = one step at a time
    if (T <= 4 && glm_mq8_pf_dev) {
        for (; e0 + 3 * 256 < n_in; e0 += 4 * 256) {
            uint2 qw[4];
            __half dh[4];
            float4 xa[4][T][2];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int e = e0 + 256 * u;
                const block_q8_0& b = wr[e / 32];
                dh[u] = b.d;
                qw[u] = *reinterpret_cast<const uint2*>(&b.qs[e % 32]);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float* xv = xm + (size_t) t * n_in + e;
                    xa[u][t][0] = *reinterpret_cast<const float4*>(xv);
                    xa[u][t][1] = *reinterpret_cast<const float4*>(xv + 4);
                }
            }
            __builtin_amdgcn_sched_barrier(0);   // keep the loads together
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float d = __half2float(dh[u]);
                const int8_t* q = reinterpret_cast<const int8_t*>(&qw[u]);
                float wv[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) wv[j] = (float) q[j] * d;
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 a0 = xa[u][t][0], a1 = xa[u][t][1];
                    acc[t] += wv[0] * a0.x + wv[1] * a0.y + wv[2] * a0.z + wv[3] * a0.w + wv[4] * a1.x + wv[5] * a1.y +
                              wv[6] * a1.z + wv[7] * a1.w;
                }
            }
        }
    }
    for (; e0 < n_in; e0 += 256) {
        const block_q8_0& b = wr[e0 / 32];
        const float d = __half2float(b.d);
        const int o = e0 % 32;
        float wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) wv[j] = (float) b.qs[o + j] * d;
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float* xv = xm + (size_t) t * n_in + e0;
            const float4 a0 = *reinterpret_cast<const float4*>(xv);
            const float4 a1 = *reinterpret_cast<const float4*>(xv + 4);
            acc[t] += wv[0] * a0.x + wv[1] * a0.y + wv[2] * a0.z + wv[3] * a0.w + wv[4] * a1.x + wv[5] * a1.y +
                      wv[6] * a1.z + wv[7] * a1.w;
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_sum(acc[t]);
        if (lane == 0) m.y[mi][(size_t) t * m.n_out[mi] + r] = v;
    }
}


// thread-per-row Q8_0 mat-vec for short rows (n_in <= 256, e.g. KDA f_b / g_b 128 -> 8192), up to 2 matrices
template <int T>
__global__ void k_rows_q8_short(MultiQ8 m, int n_in) {
    const int mi = blockIdx.y;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m.n_out[mi]) return;
    const int nb = n_in / 32;
    const block_q8_0* wr = m.w[mi] + (size_t) r * nb;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t] = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const float d = __half2float(wr[b].d);
        float part[T];
#pragma unroll
        for (int t = 0; t < T; ++t) part[t] = 0.0f;
#pragma unroll 8
        for (int j = 0; j < 32; ++j) {
            const float q = (float) wr[b].qs[j];
#pragma unroll
            for (int t = 0; t < T; ++t) part[t] += q * m.x[mi][(size_t) t * n_in + b * 32 + j];
        }
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] += d * part[t];
    }
#pragma unroll
    for (int t = 0; t < T; ++t) m.y[mi][(size_t) t * m.n_out[mi] + r] = acc[t];
}


// MoE routing sort for the grouped prefill MMQ: rows (t, slot) grouped by expert (stable in t). One block.
// rowsrc[r] = token t, rownat[r] = t * k + slot, bounds[e] = first sorted row of expert e (bounds[n_exp] = T * k)
__global__ __launch_bounds__(1024) void k_moe_route_sort(const int32_t* __restrict__ ids, int n, int k, int n_exp,
                                                         int32_t* __restrict__ rowsrc, int32_t* __restrict__ rownat,
                                                         int32_t* __restrict__ bounds) {
    __shared__ int cnt[1024];
    __shared__ int off[1025];
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        for (int e = 0; e < n_exp; ++e) { off[e] = a; bounds[e] = a; a += cnt[e]; }
        off[n_exp] = a;
        bounds[n_exp] = a;
    }
    __syncthreads();
    // stable placement: one thread per expert scans the entries in order (n <= 8 * 1024)
    for (int e = threadIdx.x; e < n_exp; e += blockDim.x) {
        int pos = off[e];
        for (int i = 0; i < n; ++i) {
            if (ids[i] == e) { rowsrc[pos] = i / k; rownat[pos] = i; ++pos; }
        }
    }
}


// the same stable sort (rowsrc / rownat / bounds identical to k_moe_route_sort) with every wave busy: wave w owns
// the w-th of 32 contiguous segments of the entries; per-segment expert counts (LDS), then per expert a running offset
// over the segments, then each wave places its segment in entry order, 32 entries a step (a lane's rank among the
// lanes with the same expert below it). n_exp <= RS2_MAXE.
constexpr int RS2_MAXE = 320;
__global__ __launch_bounds__(1024) void k_moe_route_sort2(const int32_t* __restrict__ ids, int n, int k, int n_exp,
                                                          int32_t* __restrict__ rowsrc, int32_t* __restrict__ rownat,
                                                          int32_t* __restrict__ bounds) {
    __shared__ int cs[32][RS2_MAXE];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < 32 * RS2_MAXE; i += 1024) (&cs[0][0])[i] = 0;
    __syncthreads();
    const int seg = (n + 31) / 32, s0 = w * seg, s1 = min(n, s0 + seg);
    for (int i = s0 + lane; i < s1; i += 32) atomicAdd(&cs[w][ids[i]], 1);
    __syncthreads();
    __shared__ int eoff[RS2_MAXE + 1];
    for (int e = threadIdx.x; e < n_exp; e += 1024) {   // expert totals
        int c = 0;
        for (int q = 0; q < 32; ++q) c += cs[q][e];
        eoff[e] = c;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        for (int e = 0; e < n_exp; ++e) { const int c = eoff[e]; eoff[e] = a; bounds[e] = a; a += c; }
        eoff[n_exp] = a;
        bounds[n_exp] = a;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < n_exp; e += 1024) {   // running offsets per (segment, expert)
        int a = eoff[e];
        for (int q = 0; q < 32; ++q) { const int c = cs[q][e]; cs[q][e] = a; a += c; }
    }
    __syncthreads();
    for (int b = s0; b < s1; b += 32) {
        const int i = b + lane;
        const int id = i < s1 ? ids[i] : -1 - lane;   // (distinct negatives: no match)
        int rank = 0, cnt = 0;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int o = __shfl(id, j, 32);
            const int m = o == id;
            rank += (j < lane) & m;
            cnt += m;
        }
        if (i < s1) {
            const int pos = cs[w][id] + rank;
            rowsrc[pos] = i / k;
            rownat[pos] = i;
            if (rank == cnt - 1) cs[w][id] += cnt;   // the last lane of each expert moves the segment's offset
        }
    }
}

// q8_1 rows gathered through ids: xq row r = q8_1(x[ids[r]]) (n_in % 32 == 0); one wave per 32-block
__global__ void k_gather_q8_1(const float* __restrict__ x, const int32_t* __restrict__ ids, q8_1_blk* __restrict__ xq,
                              int n_in, int rows) {
    const int nb = n_in / 32;
    const int blk = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (blk >= rows * nb) return;
    const int r = blk / nb, b = blk % nb;
    const int src = ids ? ids[r] : r;
    q8_1_store_lane(x[(size_t) src * n_in + b * 32 + (threadIdx.x & 31)], xq + blk, threadIdx.x & 31);
}

inline unsigned blocks_for(int n, int bs, int cap = 4096) {
    const int b = (n + bs - 1) / bs;
    return (unsigned) (b < 1 ? 1 : (b > cap ? cap : b));
}

}  // namespace

// ------------------------------------------------------------------------------------------------------------------
// Wide (prompt-chunk) forms of k_heads_q8_0 / k_multi_q8_0 / k_f32mv. Each output (token, row) is computed with
// the same per-lane partials as the <=8-token kernels (lane l: elements l*8 + 256 s, the same 8-term expression,
// ascending s) and the same xor-16/8/4/2/1 pairing tree as warp_sum, so every value is bitwise the decode kernel's.
// What changes: the weights of RW rows stay in registers for a 64-token tile (not re-read per 8 tokens), and the
// cross-lane reduction of 8 tokens is a recursive halving (9 shuffles instead of 40). GLM_MV_WIDE=0 = 8-token tiles.
static bool glm_mv_wide_on() {
    static const bool on = [] { const char* e = std::getenv("GLM_MV_WIDE"); return !e || std::atoi(e) != 0; }();
    return on;
}
constexpr int MVW_TT = 64;   // tokens per block

// 8 per-token partials of one row -> lane l holds the full warp_sum of token ((l>>4)&1)*4 + ((l>>3)&1)*2 + ((l>>2)&1)
__device__ __forceinline__ float mvw_reduce8(float (&v)[8], int lane) {
    {   const bool up = lane & 16;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float snd = up ? v[i] : v[i + 4];
            const float rcv = __shfl_xor(snd, 16, WS);
            v[i] = (up ? v[i + 4] : v[i]) + rcv;
        } }
    {   const bool up = lane & 8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float snd = up ? v[i] : v[i + 2];
            const float rcv = __shfl_xor(snd, 8, WS);
            v[i] = (up ? v[i + 2] : v[i]) + rcv;
        } }
    {   const bool up = lane & 4;
        const float snd = up ? v[0] : v[1];
        const float rcv = __shfl_xor(snd, 4, WS);
        v[0] = (up ? v[1] : v[0]) + rcv; }
    v[0] += __shfl_xor(v[0], 2, WS);
    v[0] += __shfl_xor(v[0], 1, WS);
    return v[0];
}
__device__ __forceinline__ int mvw_tok(int lane) { return ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1); }

// groups (blockIdx.z): heads mode (n_mats == 0) = head g of W [H][n_out][n_in] with x [t][H][n_in], y [t][H][n_out];
// multi mode = matrix g with x_g [t][n_in], y_g [t][n_out_g]
struct WideQ8 {
    const block_q8_0* w[3];
    const float* x[3];
    float* y[3];
    int n_out[3];
    int n_mats;
    int H;
};
// PF (rc-glm, GLM_MVW_PF=0 = off): the 8 tokens' x loads of a K step issued together before their FMAs (the same
// expressions in the same order per accumulator: bitwise; the PF = false form waits for each token's loads in turn)
template <int KS, int RW, bool PF = false>
__global__ __launch_bounds__(256) void k_wide_q8_0(WideQ8 m, int n_in, int T) {
    const int g = blockIdx.z;
    const int mi = m.n_mats ? g : 0;
    const int n_out = m.n_out[mi];
    const int nb = n_in / 32;
    const block_q8_0* W = m.w[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_out * nb);
    const float* X = m.x[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_in);
    float* Y = m.y[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_out);
    const size_t xts = m.n_mats ? (size_t) n_in : (size_t) m.H * n_in;
    const size_t yts = m.n_mats ? (size_t) n_out : (size_t) m.H * n_out;
    const int lane = threadIdx.x & 31;
    const int r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RW;
    if (r0 >= n_out) return;
    float wv[RW][KS][8];
#pragma unroll
    for (int j = 0; j < RW; ++j)
#pragma unroll
        for (int s = 0; s < KS; ++s) {
            const int e0 = lane * 8 + 256 * s;
            const int r = r0 + j < n_out ? r0 + j : n_out - 1;
            if (e0 < n_in) {
                const block_q8_0& b = W[(size_t) r * nb + e0 / 32];
                const float d = __half2float(b.d);
                const int o = e0 % 32;
#pragma unroll
                for (int q = 0; q < 8; ++q) wv[j][s][q] = (float) b.qs[o + q] * d;
            } else {
#pragma unroll
                for (int q = 0; q < 8; ++q) wv[j][s][q] = 0.0f;
            }
        }
    const int t_end = min(T, (int) (blockIdx.y + 1) * MVW_TT);
    for (int t0 = blockIdx.y * MVW_TT; t0 < t_end; t0 += 8) {
        float acc[RW][8];
#pragma unroll
        for (int j = 0; j < RW; ++j)
#pragma unroll
            for (int tk = 0; tk < 8; ++tk) acc[j][tk] = 0.0f;
        if constexpr (PF) {
            // all KS x 8 tokens' x first (addresses clamped inside the row: the values of lanes past n_in are unused)
            float4 xa0[KS][8], xa1[KS][8];
#pragma unroll
            for (int s = 0; s < KS; ++s)
#pragma unroll
                for (int tk = 0; tk < 8; ++tk) {
                    const float* xv = X + (size_t) min(t0 + tk, T - 1) * xts + min(lane * 8 + 256 * s, n_in - 8);
                    xa0[s][tk] = *reinterpret_cast<const float4*>(xv);
                    xa1[s][tk] = *reinterpret_cast<const float4*>(xv + 4);
                }
            __builtin_amdgcn_sched_barrier(0);   // keep the loads together (no sinking to their uses)
#pragma unroll
            for (int s = 0; s < KS; ++s) {
                const int e0 = lane * 8 + 256 * s;
                if (e0 < n_in) {
#pragma unroll
                    for (int tk = 0; tk < 8; ++tk) {
                        const float4 a0 = xa0[s][tk], a1 = xa1[s][tk];
#pragma unroll
                        for (int j = 0; j < RW; ++j) {
                            const float* wq = wv[j][s];
                            acc[j][tk] += wq[0] * a0.x + wq[1] * a0.y + wq[2] * a0.z + wq[3] * a0.w + wq[4] * a1.x +
                                          wq[5] * a1.y + wq[6] * a1.z + wq[7] * a1.w;
                        }
                    }
                }
            }
        } else {
#pragma unroll
        for (int s = 0; s < KS; ++s) {
            const int e0 = lane * 8 + 256 * s;
            if (e0 < n_in) {
#pragma unroll
                for (int tk = 0; tk < 8; ++tk) {
                    const int t = min(t0 + tk, T - 1);
                    const float* xv = X + (size_t) t * xts + e0;
                    const float4 a0 = *reinterpret_cast<const float4*>(xv);
                    const float4 a1 = *reinterpret_cast<const float4*>(xv + 4);
#pragma unroll
                    for (int j = 0; j < RW; ++j) {
                        const float* wq = wv[j][s];
                        acc[j][tk] += wq[0] * a0.x + wq[1] * a0.y + wq[2] * a0.z + wq[3] * a0.w + wq[4] * a1.x +
                                      wq[5] * a1.y + wq[6] * a1.z + wq[7] * a1.w;
                    }
                }
            }
        }
        }
        const int tk = mvw_tok(lane);
#pragma unroll
        for (int j = 0; j < RW; ++j) {
            const float v = mvw_reduce8(acc[j], lane);
            if ((lane & 3) == 0 && t0 + tk < T && r0 + j < n_out) Y[(size_t) (t0 + tk) * yts + r0 + j] = v;
        }
    }
}

// n_in <= 128 (KDA f_b / g_b, 128 -> 8192): in the decode kernels lanes 16-31 hold no elements (their partial
// stays +0.0) and the xor-16 level of warp_sum adds that +0.0. Here lanes 16-31 take tokens t0 + 8 .. t0 + 15 with the
// same element map (lane & 15); the xor-16 level becomes an explicit + 0.0f (bitwise the same: x + 0 = x, -0 -> +0),
// then levels 8 / 4 / 2 / 1 inside each half as a recursive halving (warp_sum's pairing). 16 tokens per pass, no idle lanes.
template <int RW, bool PF = false>
__global__ __launch_bounds__(256) void k_wide_q8_0_h(WideQ8 m, int n_in, int T) {
#pragma clang fp contract(off)
    const int g = blockIdx.z;
    const int mi = m.n_mats ? g : 0;
    const int n_out = m.n_out[mi];
    const int nb = n_in / 32;
    const block_q8_0* W = m.w[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_out * nb);
    const float* X = m.x[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_in);
    float* Y = m.y[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_out);
    const size_t xts = m.n_mats ? (size_t) n_in : (size_t) m.H * n_in;
    const size_t yts = m.n_mats ? (size_t) n_out : (size_t) m.H * n_out;
    const int lane = threadIdx.x & 31, l16 = lane & 15, half = lane >> 4;
    const int r0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * RW;
    if (r0 >= n_out) return;
    const int e0 = l16 * 8;
    const bool act = e0 < n_in;
    float wv[RW][8];
#pragma unroll
    for (int j = 0; j < RW; ++j) {
        const int r = r0 + j < n_out ? r0 + j : n_out - 1;
        if (act) {
            const block_q8_0& b = W[(size_t) r * nb + e0 / 32];
            const float d = __half2float(b.d);
            const int o = e0 % 32;
#pragma unroll
            for (int q = 0; q < 8; ++q) wv[j][q] = (float) b.qs[o + q] * d;
        } else {
#pragma unroll
            for (int q = 0; q < 8; ++q) wv[j][q] = 0.0f;
        }
    }
    const int t_end = min(T, (int) (blockIdx.y + 1) * MVW_TT);
    for (int t0 = blockIdx.y * MVW_TT; t0 < t_end; t0 += 16) {
        float acc[RW][8];
#pragma unroll
        for (int j = 0; j < RW; ++j)
#pragma unroll
            for (int tk = 0; tk < 8; ++tk) acc[j][tk] = 0.0f;
        if (act) {
            float4 xa0[PF ? 8 : 1], xa1[PF ? 8 : 1];
            if constexpr (PF) {
#pragma unroll
                for (int tk = 0; tk < 8; ++tk) {
                    const float* xv = X + (size_t) min(t0 + half * 8 + tk, T - 1) * xts + e0;
                    xa0[tk] = *reinterpret_cast<const float4*>(xv);
                    xa1[tk] = *reinterpret_cast<const float4*>(xv + 4);
                }
                __builtin_amdgcn_sched_barrier(0);   // keep the loads together (no sinking to their uses)
            }
#pragma unroll
            for (int tk = 0; tk < 8; ++tk) {
                const int t = min(t0 + half * 8 + tk, T - 1);
                const float* xv = X + (size_t) t * xts + e0;
                const float4 a0 = PF ? xa0[PF ? tk : 0] : *reinterpret_cast<const float4*>(xv);
                const float4 a1 = PF ? xa1[PF ? tk : 0] : *reinterpret_cast<const float4*>(xv + 4);
#pragma unroll
                for (int j = 0; j < RW; ++j) {
                    const float* w = wv[j];
                    float p = w[1] * a0.y;
                    p = __builtin_fmaf(w[0], a0.x, p);
                    p = __builtin_fmaf(w[2], a0.z, p);
                    p = __builtin_fmaf(w[3], a0.w, p);
                    p = __builtin_fmaf(w[4], a1.x, p);
                    p = __builtin_fmaf(w[5], a1.y, p);
                    p = __builtin_fmaf(w[6], a1.z, p);
                    p = __builtin_fmaf(w[7], a1.w, p);
                    acc[j][tk] = acc[j][tk] + p;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < RW; ++j) {
            float* v = acc[j];
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = v[i] + 0.0f;   // warp_sum's xor-16 level: + the empty lane's +0.0
            {   const bool up = lane & 8;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float snd = up ? v[i] : v[i + 4];
                    const float rcv = __shfl_xor(snd, 8, WS);
                    v[i] = (up ? v[i + 4] : v[i]) + rcv;
                } }
            {   const bool up = lane & 4;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const float snd = up ? v[i] : v[i + 2];
                    const float rcv = __shfl_xor(snd, 4, WS);
                    v[i] = (up ? v[i + 2] : v[i]) + rcv;
                } }
            {   const bool up = lane & 2;
                const float snd = up ? v[0] : v[1];
                const float rcv = __shfl_xor(snd, 2, WS);
                v[0] = (up ? v[1] : v[0]) + rcv; }
            v[0] += __shfl_xor(v[0], 1, WS);
            const int tk = half * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
            if ((lane & 1) == 0 && t0 + tk < T && r0 + j < n_out) Y[(size_t) (t0 + tk) * yts + r0 + j] = v[0];
        }
    }
}

// f32 weights: lane l covers float4 i = l + 32 s (as k_f32mv)
template <int KS>
__global__ __launch_bounds__(256) void k_wide_f32mv(const float* __restrict__ Wt, const float* __restrict__ x,
                                                    float* __restrict__ y, int n_in, int n_out, int T) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= n_out) return;
    const int n4 = n_in / 4;
    const float4* w4 = reinterpret_cast<const float4*>(Wt + (size_t) r * n_in);
    float4 wv[KS];
#pragma unroll
    for (int s = 0; s < KS; ++s) {
        const int i = lane + 32 * s;
        wv[s] = i < n4 ? w4[i] : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const int t_end = min(T, (int) (blockIdx.y + 1) * MVW_TT);
    for (int t0 = blockIdx.y * MVW_TT; t0 < t_end; t0 += 8) {
        float acc[8];
#pragma unroll
        for (int tk = 0; tk < 8; ++tk) acc[tk] = 0.0f;
#pragma unroll
        for (int s = 0; s < KS; ++s) {
            const int i = lane + 32 * s;
            if (i < n4) {
                const float4 w = wv[s];
#pragma unroll
                for (int tk = 0; tk < 8; ++tk) {
                    const int t = min(t0 + tk, T - 1);
                    const float4 v = reinterpret_cast<const float4*>(x + (size_t) t * n_in)[i];
                    acc[tk] += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
                }
            }
        }
        const int tk = mvw_tok(lane);
        const float v = mvw_reduce8(acc, lane);
        if ((lane & 3) == 0 && t0 + tk < T) y[(size_t) (t0 + tk) * n_out + r] = v;
    }
}

// rb-glm E1b: LDS-tiled forms of the 4096-input mat-vecs of a prompt chunk (KDA f_a / g_a / beta: k_multi_q8_0;
// router / indexer projection: k_f32mv). Every output (token, row) is bitwise the decode kernel's:
//  * lane l of a wave keeps the decode kernel's lane-l partial: the same elements (Q8_0: e0 = 8 l + 256 s; f32: float4
//    i = l + 32 s), ascending s, starting from 0.0f, with the per-step expression exactly as the compiler contracts the
//    decode kernels' source (gfx1151 ISA of k_multi_q8_0 / k_f32mv: p = w1*x1; p = fma(w0, x0, p); p = fma(w2, x2, p);
//    ... ; acc = acc + p), written out with contraction off so it can't drift;
//  * the Q8_0 weights are dequantized as (float) q * d (one rounding, as the decode kernels' v_fma_mix with -0);
//  * the 32 lane partials are combined with warp_sum's xor 16/8/4/2/1 pairing tree, as a recursive halving.
// What changes: a wave holds RW rows x TT tokens of accumulators (weights reused TT times, x RW times), and the
// block's 8 waves share a TT-token x tile staged through LDS (SC s-steps per stage, the next stage's global loads in
// flight during the current stage's math). x values pass through LDS unchanged. GLM_MV_TILE=0 = the previous paths.
static bool glm_mv_tile_on() {
    static const bool on = [] { const char* e = std::getenv("GLM_MV_TILE"); return !e || std::atoi(e) != 0; }();
    return on;
}

// N per-lane values (N >= 32, a power of two) -> lane l holds v[0 .. N/32) = the warp_sum totals of the original
// indices mvt_base(l, N) + i (the pairing at every level is warp_sum's: own + lane l ^ o, for the index kept)
template <int N, int O>
__device__ __forceinline__ void mvt_level(float (&v)[N], int lane) {
    constexpr int H = N * O / 32;   // values kept after this level (level o = 16: N / 2, ..., o = 1: N / 32)
    const bool up = (lane & O) != 0;
#pragma unroll
    for (int i = 0; i < H; ++i) {
        const float snd = up ? v[i] : v[i + H];
        const float rcv = __shfl_xor(snd, O, WS);
        v[i] = (up ? v[i + H] : v[i]) + rcv;
    }
}
template <int N>
__device__ __forceinline__ void mvt_reduce(float (&v)[N], int lane) {
    static_assert(N >= 32 && (N & (N - 1)) == 0, "mvt_reduce: N must be a power of two >= 32");
    mvt_level<N, 16>(v, lane);
    mvt_level<N, 8>(v, lane);
    mvt_level<N, 4>(v, lane);
    mvt_level<N, 2>(v, lane);
    mvt_level<N, 1>(v, lane);
}
__device__ __forceinline__ int mvt_base(int lane, int n) {
    return ((lane >> 4) & 1) * (n / 2) + ((lane >> 3) & 1) * (n / 4) + ((lane >> 2) & 1) * (n / 8) +
           ((lane >> 1) & 1) * (n / 16) + (lane & 1) * (n / 32);
}

struct TileMV {
    const void* w[3];
    const float* x[3];
    float* y[3];
    int n_out[3];
    int H;   // 0: matrix mi = blockIdx.z; > 0: heads mode (Q8_0 only) - head g = blockIdx.z of W [H][n_out][n_in], x [t][H][n_in]
};

// grid (ceil(max n_out / (8 RW)), ceil(T / TT), n_mats), block 256. Q8: W block_q8_0 [n_out][n_in / 32]; else f32
// [n_out][n_in]. x_m [T][n_in], y_m [T][n_out_m]. n_in % (SC * STEP) == 0.
template <bool Q8, int RW, int TT, int SC>
__global__ __launch_bounds__(256) void k_tile_mv(TileMV m, int n_in, int T) {
#pragma clang fp contract(off)
    constexpr int STEP = Q8 ? 256 : 128;   // elements per lane step s (a wave covers STEP elements)
    constexpr int CH4 = SC * STEP / 4;     // float4 per token per stage
    constexpr int TILE4 = TT * CH4;        // float4 per stage
    constexpr int PER = TILE4 / 256;       // float4 staged per thread
    static_assert(TILE4 % 256 == 0, "k_tile_mv: stage tile must split over 256 threads");
    constexpr int NV = RW * TT;
    __shared__ float4 xs[TILE4];
    const int mi = m.H ? 0 : blockIdx.z;
    const int hg = m.H ? blockIdx.z : 0;
    const int n_out = m.n_out[mi];
    const size_t xts = m.H ? (size_t) m.H * n_in : (size_t) n_in;   // token strides of x / y
    const size_t yts = m.H ? (size_t) m.H * n_out : (size_t) n_out;
    const int rb = blockIdx.x * 8 * RW;
    if (rb >= n_out) return;   // block-uniform, before any barrier
    const int lane = threadIdx.x & 31;
    const int r0 = rb + (threadIdx.x >> 5) * RW;
    const int t0 = blockIdx.y * TT;
    const float* X = m.x[mi] + (size_t) hg * n_in;
    float* Y = m.y[mi] + (size_t) hg * n_out;

    // staging map: thread th stages float4 f = th + 256 p (p < 4) of the stage tile: token f / CH4, float4 f % CH4
    static_assert(PER == 2 || PER == 4, "k_tile_mv: 2 or 4 staged float4 per thread");
    float4 st0, st1, st2, st3;
#define MVT_LD1(v_, p_, c_)                                                                                           \
    {                                                                                                                 \
        const int f = threadIdx.x + 256 * (p_);                                                                       \
        const int t = min(t0 + f / CH4, T - 1);                                                                       \
        v_ = *reinterpret_cast<const float4*>(X + (size_t) t * xts + (size_t) (c_) * (CH4 * 4) + (f % CH4) * 4);    \
    }
#define MVT_GLOAD(c_) { MVT_LD1(st0, 0, c_) MVT_LD1(st1, 1, c_) if constexpr (PER == 4) { MVT_LD1(st2, 2, c_) MVT_LD1(st3, 3, c_) } }
    // Q8: element e = 4 f4 -> step e / 256, lane (e % 256) / 8, half (e % 8) / 4; f32: lane l of step s reads l + 32 s
#define MVT_ST1(v_, p_)                                                                                               \
    {                                                                                                                 \
        const int f = threadIdx.x + 256 * (p_);                                                                       \
        const int f4 = f % CH4;                                                                                       \
        const int slot = Q8 ? ((f4 * 4 / 256) * 2 + ((f4 * 4 % 8) >> 2)) * 32 + ((f4 * 4 % 256) >> 3) : f4;          \
        xs[(f / CH4) * CH4 + slot] = v_;                                                                              \
    }
    float acc[NV];
#pragma unroll
    for (int i = 0; i < NV; ++i) acc[i] = 0.0f;
    const int nst = n_in / (CH4 * 4);
    MVT_GLOAD(0)
    for (int c = 0; c < nst; ++c) {
        __syncthreads();   // the previous stage's readers are done
        MVT_ST1(st0, 0) MVT_ST1(st1, 1) if constexpr (PER == 4) { MVT_ST1(st2, 2) MVT_ST1(st3, 3) }
        __syncthreads();
        MVT_GLOAD(c + 1 < nst ? c + 1 : c)   // (the last stage re-reads itself: no branch around the loads)
#pragma unroll
        for (int sl = 0; sl < SC; ++sl) {
            const int s = c * SC + sl;
            if constexpr (Q8) {
                const int nb = n_in / 32;
                const block_q8_0* W = static_cast<const block_q8_0*>(m.w[mi]) + (size_t) hg * n_out * nb;
                const int e0 = lane * 8 + 256 * s;
                const int o = e0 % 32;
                float wv[RW][8];
#pragma unroll
                for (int j = 0; j < RW; ++j) {
                    const int r = min(r0 + j, n_out - 1);
                    const block_q8_0& b = W[(size_t) r * nb + e0 / 32];
                    const float d = __half2float(b.d);
#pragma unroll
                    for (int q = 0; q < 8; ++q) wv[j][q] = (float) b.qs[o + q] * d;
                }
#pragma unroll
                for (int tk = 0; tk < TT; ++tk) {
                    const float4 a0 = xs[tk * CH4 + (sl * 2 + 0) * 32 + lane];
                    const float4 a1 = xs[tk * CH4 + (sl * 2 + 1) * 32 + lane];
#pragma unroll
                    for (int j = 0; j < RW; ++j) {
                        const float* w = wv[j];
                        float p = w[1] * a0.y;
                        p = __builtin_fmaf(w[0], a0.x, p);
                        p = __builtin_fmaf(w[2], a0.z, p);
                        p = __builtin_fmaf(w[3], a0.w, p);
                        p = __builtin_fmaf(w[4], a1.x, p);
                        p = __builtin_fmaf(w[5], a1.y, p);
                        p = __builtin_fmaf(w[6], a1.z, p);
                        p = __builtin_fmaf(w[7], a1.w, p);
                        acc[j * TT + tk] = acc[j * TT + tk] + p;
                    }
                }
            } else {
                const float4* W4 = static_cast<const float4*>(m.w[mi]);
                const int n4 = n_in / 4;
                float4 wv[RW];
#pragma unroll
                for (int j = 0; j < RW; ++j) wv[j] = W4[(size_t) min(r0 + j, n_out - 1) * n4 + lane + 32 * s];
#pragma unroll
                for (int tk = 0; tk < TT; ++tk) {
                    const float4 v = xs[tk * CH4 + sl * 32 + lane];
#pragma unroll
                    for (int j = 0; j < RW; ++j) {
                        const float4 w = wv[j];
                        float p = w.y * v.y;
                        p = __builtin_fmaf(w.x, v.x, p);
                        p = __builtin_fmaf(w.z, v.z, p);
                        p = __builtin_fmaf(w.w, v.w, p);
                        acc[j * TT + tk] = acc[j * TT + tk] + p;
                    }
                }
            }
        }
    }
    mvt_reduce<NV>(acc, lane);
    const int base = mvt_base(lane, NV);
#pragma unroll
    for (int i = 0; i < NV / 32; ++i) {
        const int idx = base + i, j = idx / TT, tk = idx % TT;
        const int r = r0 + j, t = t0 + tk;
        if (r < n_out && t < T) Y[(size_t) t * yts + r] = acc[i];
    }
#undef MVT_GLOAD
#undef MVT_LD1
#undef MVT_ST1
}

// Q8_0 (n_in > 512) / f32 prompt-chunk launch; false = not covered (GLM_MV_TILE=0, or n_in not a multiple of the stage)
// GLM_MV_TILE = 1 (default) / 2 / 3: tile configurations (RW rows x TT tokens per wave, SC steps per stage), every one
// bitwise the decode kernels (A/B timing only):  Q8_0 1: 8 x 8 x 2   2: 4 x 16 x 1   3: 8 x 8 x 1
//                                              f32  1: 4 x 16 x 2   2: 8 x 8 x 2    3: 2 x 32 x 1
static int glm_mv_tile_cfg() {
    static const int c = [] { const char* e = std::getenv("GLM_MV_TILE"); return e ? std::atoi(e) : 1; }();
    return c;
}
template <bool Q8, int RW, int TT, int SC>
static bool tile_mv_go(const TileMV& m, int n_mats, int n_out_max, int n_in, int T, hipStream_t st) {
    if (n_in % (SC * (Q8 ? 256 : 128))) return false;
    const dim3 g((unsigned) ((n_out_max + 8 * RW - 1) / (8 * RW)), (unsigned) ((T + TT - 1) / TT), (unsigned) n_mats);
    hipLaunchKernelGGL((k_tile_mv<Q8, RW, TT, SC>), g, dim3(256), 0, st, m, n_in, T);
    return true;
}
// per-head Q8_0 mat-vecs (MLA wk_b 256 -> 512, wv_b 512 -> 256 per head) on k_tile_mv, opt-in GLM_MV_HEADS =
// 1: 4 x 16 x 1, 2: 2 x 16 x 1, 3: 8 x 8 x 1 (bitwise the decode kernel k_heads_q8_0; measured slower than the wide
// kernel at 512 rows (wk_b 2.2 -> 2.9-3.6 ms, wv_b ~equal), so off by default)
static bool tile_heads_launch(const TileMV& m, int H, int n_out, int n_in, int T, hipStream_t st) {
    static const int c = [] { const char* e = std::getenv("GLM_MV_HEADS"); return e ? std::atoi(e) : 0; }();
    if (c <= 0 || !glm_mv_tile_on() || T <= 8 || n_in % 256) return false;
    if (c == 2) return tile_mv_go<true, 2, 16, 1>(m, H, n_out, n_in, T, st);
    if (c == 3) return tile_mv_go<true, 8, 8, 1>(m, H, n_out, n_in, T, st);
    return tile_mv_go<true, 4, 16, 1>(m, H, n_out, n_in, T, st);
}
static bool tile_mv_launch(bool q8, const TileMV& m, int n_mats, int n_out_max, int n_in, int T, hipStream_t st) {
    if (!glm_mv_tile_on() || T <= 8 || n_mats < 1 || n_mats > 3 || n_out_max < 1) return false;
    const int c = glm_mv_tile_cfg();
    if (q8) {
        if (n_in <= 512) return false;
        if (c == 2) return tile_mv_go<true, 4, 16, 1>(m, n_mats, n_out_max, n_in, T, st);
        if (c == 3) return tile_mv_go<true, 8, 8, 1>(m, n_mats, n_out_max, n_in, T, st);
        return tile_mv_go<true, 8, 8, 2>(m, n_mats, n_out_max, n_in, T, st);
    }
    if (c == 2) return tile_mv_go<false, 8, 8, 2>(m, n_mats, n_out_max, n_in, T, st);
    if (c == 3) return tile_mv_go<false, 2, 32, 1>(m, n_mats, n_out_max, n_in, T, st);
    return tile_mv_go<false, 4, 16, 2>(m, n_mats, n_out_max, n_in, T, st);
}

namespace talus::prefill { extern int g_w16_ed16; }
#include <utility>
// rc-glm N2 (default ON, GLM_N2=0 = the fp32 wide kernels; output-changing: summation order only): the wide Q8_0 mat-vecs (MLA wk_b / wv_b per
// head, KDA f_b / g_b) as fp16 WMMA GEMMs. A = the Q8_0 codes (exact in fp16), B = x split into fp16 hi + lo
// (x = hi + lo to ~22 bits; both products accumulate in fp32), each 32-value block's partial scaled by its d in fp32:
// numerically an fp32 dot product with a different association. Tile 64 rows x 64 tokens per block, K by 32.
typedef _Float16 n2h16 __attribute__((ext_vector_type(16)));
typedef float n2f8 __attribute__((ext_vector_type(8)));
constexpr int N2_KC = 128, N2_RS = N2_KC + 8;   // K chunk (4 Q8_0 blocks) and the staged row stride (halves)
__global__ __launch_bounds__(256) void k_wide_q8_wmma(WideQ8 m, int n_in, int T) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    __shared__ __attribute__((aligned(16))) _Float16 sa[64][N2_RS];
    __shared__ __attribute__((aligned(16))) _Float16 sh[64][N2_RS], sl[64][N2_RS];
    __shared__ float sd[64][N2_KC / 32];
    const int g = blockIdx.z;
    const int mi = m.n_mats ? g : 0;
    const int n_out = m.n_out[mi];
    const int r0 = blockIdx.x * 64;
    if (r0 >= n_out) return;
    const int nb = n_in / 32;
    const block_q8_0* W = m.w[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_out * nb);
    const float* X = m.x[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_in);
    float* Y = m.y[mi] + (m.n_mats ? (size_t) 0 : (size_t) g * n_out);
    const size_t xts = m.n_mats ? (size_t) n_in : (size_t) m.H * n_in;
    const size_t yts = m.n_mats ? (size_t) n_out : (size_t) m.H * n_out;
    const int t0 = blockIdx.y * 64;
    const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wr = (wv & 3) * 16, wt = (wv >> 2) * 32;   // this wave: rows wr..+15, tokens wt..+31 (2 tiles)
    n2f8 acc[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) acc[j] = (n2f8){0, 0, 0, 0, 0, 0, 0, 0};
    for (int c0 = 0; c0 < n_in; c0 += N2_KC) {
        const int nqb = min(N2_KC, n_in - c0) / 32;   // Q8_0 blocks in this chunk (n_in % 32 == 0)
        __syncthreads();
        {   // weights: thread = (row tid / 4, block tid % 4): its 34-byte block as 17 halfwords
            const int r = tid >> 2, qb = tid & 3;
            if (qb < nqb) {
                const uint16_t* bp = reinterpret_cast<const uint16_t*>(&W[(size_t) min(r0 + r, n_out - 1) * nb + c0 / 32 + qb]);
                uint16_t u[17];
#pragma unroll
                for (int e = 0; e < 17; ++e) u[e] = bp[e];
                sd[r][qb] = __half2float(__ushort_as_half(u[0]));
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    sa[r][32 * qb + 2 * e] = (_Float16) (float) (int8_t) (u[1 + e] & 0xff);
                    sa[r][32 * qb + 2 * e + 1] = (_Float16) (float) (int8_t) (u[1 + e] >> 8);
                }
            }
        }
        {   // x: thread = (token tid / 4, quarter tid % 4): 32 values split hi / lo
            const int r = tid >> 2, c = (tid & 3) * 32;
            if (c < 32 * nqb) {
                const float* xv = X + (size_t) min(t0 + r, T - 1) * xts + c0 + c;
#pragma unroll
                for (int q4 = 0; q4 < 8; ++q4) {
                    const float4 a = *reinterpret_cast<const float4*>(xv + 4 * q4);
                    const float xs[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const _Float16 h = (_Float16) xs[q];
                        sh[r][c + 4 * q4 + q] = h;
                        sl[r][c + 4 * q4 + q] = (_Float16) (xs[q] - (float) h);
                    }
                }
            }
        }
        __syncthreads();
        for (int qb = 0; qb < nqb; ++qb) {
            n2f8 tmp[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) tmp[j] = (n2f8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int kk = 32 * qb + 16 * ks;
                const n2h16 A = *reinterpret_cast<const n2h16*>(&sa[wr + l16][kk]);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int tt = wt + 16 * j + l16;
                    const n2h16 Bh = *reinterpret_cast<const n2h16*>(&sh[tt][kk]);
                    const n2h16 Bl = *reinterpret_cast<const n2h16*>(&sl[tt][kk]);
                    tmp[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, Bh, tmp[j]);
                    tmp[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, Bl, tmp[j]);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float d = sd[wr + 2 * i + hi][qb];
#pragma unroll
                for (int j = 0; j < 2; ++j) acc[j][i] = fmaf(d, tmp[j][i], acc[j][i]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int t = t0 + wt + 16 * j + l16;
        if (t >= T) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = r0 + wr + 2 * i + hi;
            if (r < n_out) Y[(size_t) t * yts + r] = acc[j][i];
        }
    }
#else
    (void) m; (void) n_in; (void) T;
#endif
}

// launch the Q8_0 wide kernel for n_in in {<=256, <=512, <=4096} (multiples of 32); false = not covered
static bool wide_q8_launch(const WideQ8& m, int n_groups, int n_out_max, int n_in, int T, hipStream_t st) {
    // For n_in > 512 (KDA f_a / g_a / beta, 4096 inputs), the register-resident form uses 128 weight VGPRs
    // with low occupancy and the same x traffic, so keep 8-token tiles.
    if (!glm_mv_wide_on() || n_in % 32 || n_in > 512) return false;
    static const bool n2 = [] { const char* e = std::getenv("GLM_N2"); return !e || e[0] != '0'; }();   // default ON (rc-glm card; GLM_N2=0 = fp32 wide kernels)
    if (n2 && n_out_max % 64 == 0) {   // fp16 WMMA (default on, output-changing)
        const dim3 g((unsigned) (n_out_max / 64), (unsigned) ((T + 63) / 64), (unsigned) n_groups);
        hipLaunchKernelGGL(k_wide_q8_wmma, g, dim3(256), 0, st, m, n_in, T);
        return true;
    }
    const unsigned ty = (unsigned) ((T + MVW_TT - 1) / MVW_TT);
    static const bool pf = [] { const char* e = std::getenv("GLM_MVW_PF"); return !e || e[0] != '0'; }();
#define MVW_L(KS_, RW_)                                                                                               \
    {                                                                                                                 \
        const dim3 g((unsigned) ((n_out_max + 8 * RW_ - 1) / (8 * RW_)), ty, (unsigned) n_groups);                   \
        if (pf) hipLaunchKernelGGL((k_wide_q8_0<KS_, RW_, true>), g, dim3(256), 0, st, m, n_in, T);                   \
        else hipLaunchKernelGGL((k_wide_q8_0<KS_, RW_>), g, dim3(256), 0, st, m, n_in, T);                            \
        return true;                                                                                                  \
    }
    if (n_in <= 128 && glm_mv_tile_on()) {   // two tokens per wave (GLM_MV_TILE=0 = k_wide_q8_0<1, 8>)
        const dim3 g((unsigned) ((n_out_max + 63) / 64), ty, (unsigned) n_groups);
        if (pf) hipLaunchKernelGGL((k_wide_q8_0_h<8, true>), g, dim3(256), 0, st, m, n_in, T);
        else hipLaunchKernelGGL((k_wide_q8_0_h<8>), g, dim3(256), 0, st, m, n_in, T);
        return true;
    }
    if (n_in <= 256) MVW_L(1, 8)
    MVW_L(2, 4)
#undef MVW_L
    return false;
}


extern "C" {

int tkg_abi_version(void) { return TKG_ABI_VERSION; }

int tkg_hc_repeat(const float* x, float* xhc, int T, int n_embd, void* s) {
    hipLaunchKernelGGL(k_hc_repeat, dim3(blocks_for(n_embd, 256, 64), T), dim3(256), 0, S(s), x, xhc, n_embd);
    return last();
}

int tkg_hc_mix(const void* w, const float* xhc, float* parts, int T, int K, int M, int nsplit, void* s) {
    if (M != 24 || K % nsplit != 0 || (K / nsplit) % 32 != 0) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL((k_hc_mix<24>), dim3(nsplit, T), dim3(256), 0, S(s),
                       static_cast<const block_q8_0*>(w), xhc, parts, K, nsplit);
    return last();
}

int tkg_hc_pre(const float* xhc, const float* parts, int nsplit, const float* scale, const float* base,
               const float* norm_w, float* y, float* pc, int T, int n_embd, float rms_eps, float norm_eps,
               float hc_eps, int iters, void* yq, void* s) {
    if (n_embd > 4096 || n_embd % 256) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_hc_pre, dim3(T), dim3(256), 0, S(s), xhc, parts, nsplit, scale, base, norm_w, y, pc, n_embd,
                       rms_eps, norm_eps, hc_eps, iters, static_cast<q8_1_blk*>(yq));
    return last();
}

int tkg_hc_post(const float* xo, const float* res, const float* pc, float* out, int T, int n_embd, void* s) {
    hipLaunchKernelGGL(k_hc_post, dim3(blocks_for(n_embd, 256, 64), T), dim3(256), 0, S(s), xo, res, pc, out, n_embd);
    return last();
}

int tkg_hc_head(const float* xhc, const float* w, float* h, int T, int n_embd, float eps, void* s) {
    hipLaunchKernelGGL(k_hc_head<float>, dim3(T), dim3(256), 0, S(s), xhc, w, h, n_embd, eps);
    return last();
}

int tkg_rms_norm(const float* x, const float* w, float* y, int rows, int cols, int xs, int ys, float eps, void* s) {
    if (cols <= 128) {
        hipLaunchKernelGGL(k_rms_norm_small, dim3((rows + 7) / 8), dim3(256), 0, S(s), x, w, y, rows, cols, xs, ys, eps);
    } else {
        hipLaunchKernelGGL(k_rms_norm, dim3(rows), dim3(256), 0, S(s), x, w, y, cols, xs, ys, eps);
    }
    return last();
}

int tkg_add(float* y, const float* x, int n, void* s) {
    hipLaunchKernelGGL(k_add, dim3(blocks_for(n, 256)), dim3(256), 0, S(s), y, x, n);
    return last();
}

int tkg_swiglu_clamp(const float* g, const float* u, float* o, int n, float limit, void* s) {
    hipLaunchKernelGGL(k_swiglu_clamp, dim3(blocks_for(n, 256)), dim3(256), 0, S(s), g, u, o, n, limit);
    return last();
}

int tkg_embed(int type, const void* table, const int32_t* tok, float* out, int T, int n_embd, void* s) {
    if (type == 12) {
        if (n_embd % 256) return err(hipErrorInvalidValue);
        hipLaunchKernelGGL(k_embed_q4k, dim3(T), dim3(256), 0, S(s), static_cast<const block_q4_K*>(table), tok, out,
                           n_embd);
    } else if (type == 0) {
        hipLaunchKernelGGL(k_embed_f32, dim3(T), dim3(256), 0, S(s), static_cast<const float*>(table), tok, out, n_embd);
    } else {
        return err(hipErrorInvalidValue);
    }
    return last();
}

int tkg_kda_conv(const float* q, const float* k, const float* v, float* st, const int32_t* ids, int K,
                 const float* wq, const float* wk, const float* wv, float* oq, float* ok, float* ov, int T, int C,
                 void* s) {
    if (K < 1 || K > T) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_kda_conv, dim3((C + 255) / 256, 3), dim3(256), 0, S(s), q, k, v, st, ids, K, wq, wk, wv,
                       oq, ok, ov, T, C);
    return last();
}

int tkg_kda_gate(const float* o, const float* g2, const float* w, float* out, int T, int H, int D, float eps,
                 void* s) {
    if (D != 128) return err(hipErrorInvalidValue);
    const int rows = T * H;
    hipLaunchKernelGGL(k_kda_gate, dim3((rows + 7) / 8), dim3(256), 0, S(s), o, g2, w, out, rows, D, eps);
    return last();
}

int tkg_mla_store(const float* kv, const float* w, float eps, void* cache, const int32_t* ctl, int T, int D,
                  void* s) {
    hipLaunchKernelGGL(k_mla_store, dim3(T), dim3(256), 0, S(s), kv, w, eps, static_cast<__half*>(cache), ctl, D);
    return last();
}

int tkg_idx_store(const float* ik, const float* lw, const float* lb, float eps, const float* ig, void* kg,
                  const int32_t* ctl, int T, void* s) {
    hipLaunchKernelGGL(k_idx_store, dim3(T), dim3(32), 0, S(s), ik, lw, lb, eps, ig, static_cast<__half*>(kg), ctl);
    return last();
}

int tkg_idx_pool(const void* kg, const float* ape, void* pooled, const int32_t* ctl, int T, int kpool, void* s) {
    if (kpool < 1 || kpool > 8) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_idx_pool, dim3(T), dim3(128), 0, S(s), static_cast<const __half*>(kg), ape,
                       static_cast<__half*>(pooled), ctl, kpool);
    return last();
}

int tkg_dsa_select(const float* scores, int stride, const int32_t* ctl, int T, int kpool, int top, int n_kv_max,
                   int32_t* idx, void* s) {
    hipLaunchKernelGGL(k_dsa_select, dim3(T), dim3(1024), 0, S(s), scores, stride, ctl, kpool, top, n_kv_max, idx);
    return last();
}

int tkg_router(const float* logits, const float* bias, int32_t* ids, float* w, int T, int n_exp, int k, float scale,
               void* s) {
    if (n_exp > 512 || k > 16) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_router, dim3(T), dim3(32), 0, S(s), logits, bias, ids, w, n_exp, k, scale);
    return last();
}

int tkg_moe_combine(const float* y, const float* w, const float* sh, float* out, int T, int k, int D, void* s) {
    hipLaunchKernelGGL(k_moe_combine, dim3(blocks_for(D, 256, 64), T), dim3(256), 0, S(s), y, w, sh, out, k, D);
    return last();
}

int tkg_mmvq_ids_f32(int type, const void* w, size_t expert_bytes, const int32_t* ids, int n_ids, int k,
                     const float* x, int x_per_id, float* y, int n_in, int n_out, void* s) {
    if (n_in % 256) return err(hipErrorInvalidValue);
    const dim3 grid((n_out + 7) / 8, n_ids);
    if (type == 11) {
        hipLaunchKernelGGL((k_mmvq_ids_f32<11>), grid, dim3(256), 0, S(s), static_cast<const uint8_t*>(w),
                           expert_bytes, ids, k, x, x_per_id, y, n_in, n_out);
    } else if (type == 10) {
        hipLaunchKernelGGL((k_mmvq_ids_f32<10>), grid, dim3(256), 0, S(s), static_cast<const uint8_t*>(w),
                           expert_bytes, ids, k, x, x_per_id, y, n_in, n_out);
    } else {
        return err(hipErrorInvalidValue);
    }
    return last();
}

int tkg_gather_rows(const void* src, size_t row_bytes, const int32_t* ids, int n, void* dst, void* s) {
    if (row_bytes % 4) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_gather_rows, dim3(n), dim3(256), 0, S(s), static_cast<const uint8_t*>(src), row_bytes, ids,
                       static_cast<uint8_t*>(dst));
    return last();
}

// device greedy argmax over rows (see k_argmax_rows)
int tkg_argmax_rows(const float* x, int n, int rows, long ld, const int32_t* map, int32_t* out_tok, int32_t* out_idx,
                    void* s) {
    if (n <= 0 || rows <= 0) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_argmax_rows, dim3(rows), dim3(1024), 0, S(s), x, n, ld, map, out_tok, out_idx);
    return last();
}



int tkg_f32_matvec(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* s) {
    // The wide f32 form (k_wide_f32mv) is opt-in (GLM_MV_WIDE=2) because 8-token tiles have lower register pressure
    // on the 4096-input router / indexer projections.
    static const bool f32_wide = [] { const char* e = std::getenv("GLM_MV_WIDE"); return e && std::atoi(e) == 2; }();
    if (T > 8) {   // rb-glm E1b: LDS-tiled prompt-chunk form (bitwise the 8-token tiles)
        TileMV m{};
        m.w[0] = W;
        m.x[0] = x;
        m.y[0] = y;
        m.n_out[0] = n_out;
        if (tile_mv_launch(false, m, 1, n_out, n_in, T, S(s))) return last();
    }
    if (T > 8 && n_in % 4 == 0 && n_in <= 4096 && f32_wide) {
        const dim3 g((unsigned) ((n_out + 7) / 8), (unsigned) ((T + MVW_TT - 1) / MVW_TT));
        hipLaunchKernelGGL((k_wide_f32mv<32>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out, T);
        return last();
    }
    if (T > 8) {   // 8-token tiles (the decode kernel)
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int r = tkg_f32_matvec(W, x + (size_t) t0 * n_in, y + (size_t) t0 * n_out, n_in, n_out,
                                         T - t0 < 8 ? T - t0 : 8, s);
            if (r) return r;
        }
        return 0;
    }
    if (n_in % 4 || T < 1 || T > 8) return err(hipErrorInvalidValue);
    const dim3 g((n_out + 7) / 8);
    static const bool f32pf = [] { const char* e = std::getenv("GLM_F32MV_PF"); return !e || e[0] != '0'; }();
    if (f32pf && T <= 4 && n_in >= 1024) {   // batched loads (bitwise)
        switch (T) {
            case 1: hipLaunchKernelGGL((k_f32mv<1, true>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
            case 2: hipLaunchKernelGGL((k_f32mv<2, true>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
            case 3: hipLaunchKernelGGL((k_f32mv<3, true>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
            default: hipLaunchKernelGGL((k_f32mv<4, true>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
        }
        return last();
    }
    switch (T) {
        case 1: hipLaunchKernelGGL((k_f32mv<1>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
        case 2: hipLaunchKernelGGL((k_f32mv<2>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
        case 3: hipLaunchKernelGGL((k_f32mv<3>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
        case 4: hipLaunchKernelGGL((k_f32mv<4>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;
        default: hipLaunchKernelGGL((k_f32mv<8>), g, dim3(256), 0, S(s), W, x, y, n_in, n_out); break;  // T 5..8
    }
    return last();
}

// tkg_f32_matvec (n_out experts) + tkg_router in one launch, bitwise; cnt = 1 zeroed int (left zero).
// -1 = shape not supported (use the two).
int tkg_f32mv_router(const float* W, const float* x, float* logits, int n_in, int n_exp, int T, const float* bias,
                     int32_t* ids, float* w, int k, float scale, int* cnt, void* s) {
    // T <= 4: above 4 tokens, register pressure in the fused f32 mat-vec + router is slower than the two separate launches
    if (n_in % 4 || T < 1 || T > 4 || n_exp > 512 || k > 16) return -1;
    const dim3 g((n_exp + 7) / 8);
    switch (T) {
        case 1: hipLaunchKernelGGL((k_f32mv_router<1>), g, dim3(256), 0, S(s), W, x, logits, n_in, n_exp, T, bias, ids, w, k, scale, cnt); break;
        case 2: hipLaunchKernelGGL((k_f32mv_router<2>), g, dim3(256), 0, S(s), W, x, logits, n_in, n_exp, T, bias, ids, w, k, scale, cnt); break;
        case 3: hipLaunchKernelGGL((k_f32mv_router<3>), g, dim3(256), 0, S(s), W, x, logits, n_in, n_exp, T, bias, ids, w, k, scale, cnt); break;
        case 4: hipLaunchKernelGGL((k_f32mv_router<4>), g, dim3(256), 0, S(s), W, x, logits, n_in, n_exp, T, bias, ids, w, k, scale, cnt); break;
        default: return -1;
    }
    return last();
}

int tkg_heads_q8_0(const void* W, const float* x, float* y, int H, int n_in, int n_out, int T, void* s) {
    if (T > 8 && n_in % 256 == 0) {   // prompt chunks in one launch
        WideQ8 m{};
        m.w[0] = static_cast<const block_q8_0*>(W);
        m.x[0] = x;
        m.y[0] = y;
        m.n_out[0] = n_out;
        m.n_mats = 0;
        m.H = H;
        {   // opt-in LDS-tiled form (GLM_MV_HEADS)
            TileMV tm{};
            tm.w[0] = W;
            tm.x[0] = x;
            tm.y[0] = y;
            tm.n_out[0] = n_out;
            tm.H = H;
            if (tile_heads_launch(tm, H, n_out, n_in, T, S(s))) return last();
        }
        if (wide_q8_launch(m, H, n_out, n_in, T, S(s))) return last();
        for (int t0 = 0; t0 < T; t0 += 8) {
            const int r = tkg_heads_q8_0(W, x + (size_t) t0 * H * n_in, y + (size_t) t0 * H * n_out, H, n_in, n_out,
                                         T - t0 < 8 ? T - t0 : 8, s);
            if (r) return r;
        }
        return 0;
    }
    if (n_in % 256 || T < 1 || T > 8) return err(hipErrorInvalidValue);
    const dim3 g((n_out + 7) / 8, H);
    const block_q8_0* w = static_cast<const block_q8_0*>(W);
    switch (T) {
        case 1: hipLaunchKernelGGL((k_heads_q8_0<1>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
        case 2: hipLaunchKernelGGL((k_heads_q8_0<2>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
        case 3: hipLaunchKernelGGL((k_heads_q8_0<3>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
        case 4: hipLaunchKernelGGL((k_heads_q8_0<4>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
        case 5: hipLaunchKernelGGL((k_heads_q8_0<5>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
        case 6: hipLaunchKernelGGL((k_heads_q8_0<6>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
        case 7: hipLaunchKernelGGL((k_heads_q8_0<7>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
        default: hipLaunchKernelGGL((k_heads_q8_0<8>), g, dim3(256), 0, S(s), w, x, y, H, n_in, n_out); break;
    }
    return last();
}

int tkg_swiglu_clamp_q8(const float* g, const float* u, void* oq, int n, float limit, void* s) {
    if (n % 32) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_swiglu_clamp_q8, dim3((n + 255) / 256), dim3(256), 0, S(s), g, u, static_cast<q8_1_blk*>(oq), n,
                       limit);
    return last();
}

int tkg_kda_gate_q8(const float* o, const float* g2, const float* w, void* oq, int T, int H, int D, float eps, void* s) {
    if (D != 128) return err(hipErrorInvalidValue);
    const int rows = T * H;
    hipLaunchKernelGGL(k_kda_gate_q8, dim3((rows + 7) / 8), dim3(256), 0, S(s), o, g2, w, static_cast<q8_1_blk*>(oq),
                       rows, D, eps);
    return last();
}

int tkg_kq_ids(int type, const void* w, size_t expert_bytes, const int32_t* ids, int n_ids, int k, const float* x,
               int x_per_id, float* y, int n_in, int n_out, void* s) {
    const int nb = n_in / 256;
    if (n_in % 256 || (nb != 8 && nb != 16)) return err(hipErrorInvalidValue);
    const int rows_per_block = 8 * (32 / nb);
    const dim3 g((n_out + rows_per_block - 1) / rows_per_block, n_ids);
    if (type == 11) {
        hipLaunchKernelGGL((k_kq_ids_v2<11>), g, dim3(256), 0, S(s), static_cast<const uint8_t*>(w), expert_bytes, ids, k,
                           x, x_per_id, y, n_in, n_out);
    } else if (type == 10) {
        hipLaunchKernelGGL((k_kq_ids_v2<10>), g, dim3(256), 0, S(s), static_cast<const uint8_t*>(w), expert_bytes, ids, k,
                           x, x_per_id, y, n_in, n_out);
    } else {
        return err(hipErrorInvalidValue);
    }
    return last();
}

// tkg_kq_ids for one or two matrices (w1/y1 null = one), distinct experts read once per window; bitwise
// tkg_kq_ids per (id, row). -1 = shape not supported (caller falls back to tkg_kq_ids).
int tkg_kq_ids2(int type, const void* w0, const void* w1, size_t expert_bytes, const int32_t* ids, int n_ids, int k,
                const float* x, int x_per_id, float* y0, float* y1, int n_in, int n_out, void* s) {
    const int nb = n_in / 256;
    if (n_in % 256 || (nb != 8 && nb != 16) || n_ids <= 0) return -1;
    if (type == 11) return kq_ids_v3_launch<11>(w0, w1, expert_bytes, ids, n_ids, k, x, x_per_id, y0, y1, n_in, n_out, s);
    if (type == 10) return kq_ids_v3_launch<10>(w0, w1, expert_bytes, ids, n_ids, k, x, x_per_id, y0, y1, n_in, n_out, s);
    return -1;
}


int tkg_hc_post_mix(const float* xo, float* xhc, const float* pc, const void* w_next, float* parts, int T, int n_embd,
                    int first, void* s) {
    if (n_embd % (HC_NS * 2)) return err(hipErrorInvalidValue);
    const dim3 g(HC_NS, T);
    const block_q8_0* w = static_cast<const block_q8_0*>(w_next);
    if (first == 1) {
        hipLaunchKernelGGL((k_hc_post_mix<1>), g, dim3(256), 0, S(s), xo, xhc, pc, w, parts, n_embd);
    } else if (first == 2) {
        hipLaunchKernelGGL((k_hc_post_mix<2>), g, dim3(256), 0, S(s), xo, xhc, pc, w, parts, n_embd);
    } else {
        hipLaunchKernelGGL((k_hc_post_mix<0>), g, dim3(256), 0, S(s), xo, xhc, pc, w, parts, n_embd);
    }
    return last();
}

int tkg_hc_post_mix_moe(const float* y, const float* w, const float* shexp, float* xhc, const float* pc,
                        const void* w_next, float* parts, int T, int k, int n_embd, void* s) {
    if (n_embd % (HC_NS * 2)) return err(hipErrorInvalidValue);
    const dim3 g(HC_NS, T);
    if (std::exchange(talus::prefill::g_w16_ed16, 0)) {   // the expert rows are fp16
        hipLaunchKernelGGL((k_hc_post_mix_moe<float, _Float16>), g, dim3(256), 0, S(s), reinterpret_cast<const _Float16*>(y), w, shexp,
                           xhc, pc, static_cast<const block_q8_0*>(w_next), parts, k, n_embd);
        return last();
    }
    hipLaunchKernelGGL(k_hc_post_mix_moe<float>, g, dim3(256), 0, S(s), y, w, shexp, xhc, pc,
                       static_cast<const block_q8_0*>(w_next), parts, k, n_embd);
    return last();
}

int tkg_hc_pre2(const float* xhc, const float* parts, const float* scale, const float* base, const float* norm_w,
                float* y, float* pc, int T, int n_embd, float rms_eps, float norm_eps, float hc_eps, int iters, void* yq,
                void* s) {
    if ((n_embd / HC_NS) % 32) return err(hipErrorInvalidValue);
    hipLaunchKernelGGL(k_hc_pre2<float>, dim3(HC_NS, T), dim3(256), 0, S(s), xhc, parts, scale, base, norm_w, y, pc, n_embd,
                       rms_eps, norm_eps, hc_eps, iters, static_cast<q8_1_blk*>(yq));
    return last();
}

// GLM_HC_BF16 (prompt path): the same four launches with 16-bit streams. fmt 1 = fp16 (saturating), 2 = bf16.
// Default entries above are unchanged (their kernels are the ST = float instantiations of the same templates).
int tkg_hc_post_mix16(const float* xo, void* xhc, const float* pc, const void* w_next, float* parts, int T, int n_embd,
                      int first, int fmt, void* s) {
    if (n_embd % (HC_NS * 2) || (fmt != 1 && fmt != 2)) return err(hipErrorInvalidValue);
    const dim3 g(HC_NS, T);
    const block_q8_0* w = static_cast<const block_q8_0*>(w_next);
#define HC16_PM(M, ST) hipLaunchKernelGGL((k_hc_post_mix<M, ST>), g, dim3(256), 0, S(s), xo, static_cast<ST*>(xhc), pc, w, parts, n_embd)
    if (fmt == 1) {
        if (first == 1) HC16_PM(1, hc_f16); else if (first == 2) HC16_PM(2, hc_f16); else HC16_PM(0, hc_f16);
    } else {
        if (first == 1) HC16_PM(1, hc_bf16); else if (first == 2) HC16_PM(2, hc_bf16); else HC16_PM(0, hc_bf16);
    }
#undef HC16_PM
    return last();
}

int tkg_hc_post_mix_moe16(const float* y, const float* w, const float* shexp, void* xhc, const float* pc,
                          const void* w_next, float* parts, int T, int k, int n_embd, int fmt, void* s) {
    if (talus::prefill::g_w16_ed16) return err(hipErrorInvalidValue);   // rc-glm N4 is not combined with GLM_HC_BF16
    if (n_embd % (HC_NS * 2) || (fmt != 1 && fmt != 2)) return err(hipErrorInvalidValue);
    const dim3 g(HC_NS, T);
    if (fmt == 1)
        hipLaunchKernelGGL(k_hc_post_mix_moe<hc_f16>, g, dim3(256), 0, S(s), y, w, shexp, static_cast<hc_f16*>(xhc), pc,
                           static_cast<const block_q8_0*>(w_next), parts, k, n_embd);
    else
        hipLaunchKernelGGL(k_hc_post_mix_moe<hc_bf16>, g, dim3(256), 0, S(s), y, w, shexp, static_cast<hc_bf16*>(xhc), pc,
                           static_cast<const block_q8_0*>(w_next), parts, k, n_embd);
    return last();
}

int tkg_hc_pre2_16(const void* xhc, const float* parts, const float* scale, const float* base, const float* norm_w,
                   float* y, float* pc, int T, int n_embd, float rms_eps, float norm_eps, float hc_eps, int iters, void* yq,
                   int fmt, void* s) {
    if ((n_embd / HC_NS) % 32 || (fmt != 1 && fmt != 2)) return err(hipErrorInvalidValue);
    if (fmt == 1)
        hipLaunchKernelGGL(k_hc_pre2<hc_f16>, dim3(HC_NS, T), dim3(256), 0, S(s), static_cast<const hc_f16*>(xhc), parts,
                           scale, base, norm_w, y, pc, n_embd, rms_eps, norm_eps, hc_eps, iters, static_cast<q8_1_blk*>(yq));
    else
        hipLaunchKernelGGL(k_hc_pre2<hc_bf16>, dim3(HC_NS, T), dim3(256), 0, S(s), static_cast<const hc_bf16*>(xhc), parts,
                           scale, base, norm_w, y, pc, n_embd, rms_eps, norm_eps, hc_eps, iters, static_cast<q8_1_blk*>(yq));
    return last();
}

int tkg_hc_head16(const void* xhc, const float* w, float* h, int T, int n_embd, float eps, int fmt, void* s) {
    if (fmt == 1)
        hipLaunchKernelGGL(k_hc_head<hc_f16>, dim3(T), dim3(256), 0, S(s), static_cast<const hc_f16*>(xhc), w, h, n_embd, eps);
    else if (fmt == 2)
        hipLaunchKernelGGL(k_hc_head<hc_bf16>, dim3(T), dim3(256), 0, S(s), static_cast<const hc_bf16*>(xhc), w, h, n_embd, eps);
    else
        return err(hipErrorInvalidValue);
    return last();
}

// tkg_hc_post_mix (mode 0 / 1) / tkg_hc_post_mix_moe + tkg_hc_pre2 in one launch: the token's last mix block
// runs k_hc_pre2's work (bitwise the two launches). cnt = T zeroed ints (left zero). w_next must be the next mix.
int tkg_hc_post_mix_pre(const float* xo, float* xhc, float* pc, const void* w_next, float* parts, int T, int n_embd,
                        int first, const float* scale, const float* base, const float* norm_w, float* y, float rms_eps,
                        float norm_eps, float hc_eps, int iters, void* yq, int* cnt, void* s) {
    if (n_embd % (HC_NS * 2) || (n_embd / HC_NS) % 32 || !w_next || first == 2) return -1;
    const dim3 g(HC_NS, T);
    const block_q8_0* w = static_cast<const block_q8_0*>(w_next);
    auto* q = static_cast<q8_1_blk*>(yq);
    if (first == 1)
        hipLaunchKernelGGL((k_hc_post_mix_pre<1>), g, dim3(256), 0, S(s), xo, xhc, pc, w, parts, n_embd, scale, base,
                           norm_w, y, rms_eps, norm_eps, hc_eps, iters, q, cnt);
    else
        hipLaunchKernelGGL((k_hc_post_mix_pre<0>), g, dim3(256), 0, S(s), xo, xhc, pc, w, parts, n_embd, scale, base,
                           norm_w, y, rms_eps, norm_eps, hc_eps, iters, q, cnt);
    return last();
}

int tkg_hc_post_mix_moe_pre(const float* ey, const float* ew, const float* shexp, float* xhc, float* pc,
                            const void* w_next, float* parts, int T, int k, int n_embd, const float* scale,
                            const float* base, const float* norm_w, float* y, float rms_eps, float norm_eps,
                            float hc_eps, int iters, void* yq, int* cnt, void* s) {
    if (n_embd % (HC_NS * 2) || (n_embd / HC_NS) % 32 || !w_next) return -1;
    const dim3 g(HC_NS, T);
    if (std::exchange(talus::prefill::g_w16_ed16, 0)) {   // the expert rows are fp16
        hipLaunchKernelGGL(k_hc_post_mix_moe_pre<_Float16>, g, dim3(256), 0, S(s), reinterpret_cast<const _Float16*>(ey), ew, shexp,
                           xhc, pc, static_cast<const block_q8_0*>(w_next), parts, k, n_embd, scale, base, norm_w, y, rms_eps,
                           norm_eps, hc_eps, iters, static_cast<q8_1_blk*>(yq), cnt);
        return last();
    }
    hipLaunchKernelGGL(k_hc_post_mix_moe_pre<float>, g, dim3(256), 0, S(s), ey, ew, shexp, xhc, pc,
                       static_cast<const block_q8_0*>(w_next), parts, k, n_embd, scale, base, norm_w, y, rms_eps,
                       norm_eps, hc_eps, iters, static_cast<q8_1_blk*>(yq), cnt);
    return last();
}


int tkg_multi_q8_0(int n_mats, const void* const* w, const float* const* x, float* const* y, const int* n_out, int n_in,
                   int T, void* s) {
    if (T > 8 && n_mats >= 1 && n_mats <= 3 && n_in % 32 == 0) {   // prompt chunks in one launch
        WideQ8 m{};
        int mx = 0;
        for (int i = 0; i < n_mats; ++i) {
            m.w[i] = static_cast<const block_q8_0*>(w[i]);
            m.x[i] = x[i];
            m.y[i] = y[i];
            m.n_out[i] = n_out[i];
            mx = n_out[i] > mx ? n_out[i] : mx;
        }
        m.n_mats = n_mats;
        if (wide_q8_launch(m, n_mats, mx, n_in, T, S(s))) return last();
        {   // rb-glm E1b: 4096-input matrices (KDA f_a / g_a / beta) on the LDS-tiled form (bitwise the 8-token tiles)
            TileMV tm{};
            for (int i = 0; i < n_mats; ++i) {
                tm.w[i] = w[i];
                tm.x[i] = x[i];
                tm.y[i] = y[i];
                tm.n_out[i] = n_out[i];
            }
            if (tile_mv_launch(true, tm, n_mats, mx, n_in, T, S(s))) return last();
        }
        for (int t0 = 0; t0 < T; t0 += 8) {
            const float* xt[3];
            float* yt[3];
            for (int i = 0; i < n_mats; ++i) {
                xt[i] = x[i] + (size_t) t0 * n_in;
                yt[i] = y[i] + (size_t) t0 * n_out[i];
            }
            const int r = tkg_multi_q8_0(n_mats, w, xt, yt, n_out, n_in, T - t0 < 8 ? T - t0 : 8, s);
            if (r) return r;
        }
        return 0;
    }
    if (n_mats < 1 || n_mats > 3 || n_in % 32 || T < 1 || T > 8) return err(hipErrorInvalidValue);
    MultiQ8 m{};
    int mx = 0;
    for (int i = 0; i < n_mats; ++i) {
        m.w[i] = static_cast<const block_q8_0*>(w[i]);
        m.x[i] = x[i];
        m.y[i] = y[i];
        m.n_out[i] = n_out[i];
        mx = n_out[i] > mx ? n_out[i] : mx;
    }
    const dim3 g((mx + 7) / 8, n_mats);
    static const bool pf = [] { const char* e = std::getenv("GLM_MQ8_PF"); return !e || e[0] != '0'; }();
    if (pf && T <= 4 && n_in >= 1024) {   // 4 K steps' loads in flight (bitwise)
        switch (T) {
            case 1: hipLaunchKernelGGL((k_multi_q8_0<1, true>), g, dim3(256), 0, S(s), m, n_in); break;
            case 2: hipLaunchKernelGGL((k_multi_q8_0<2, true>), g, dim3(256), 0, S(s), m, n_in); break;
            case 3: hipLaunchKernelGGL((k_multi_q8_0<3, true>), g, dim3(256), 0, S(s), m, n_in); break;
            default: hipLaunchKernelGGL((k_multi_q8_0<4, true>), g, dim3(256), 0, S(s), m, n_in); break;
        }
        return last();
    }
    switch (T) {
        case 1: hipLaunchKernelGGL((k_multi_q8_0<1>), g, dim3(256), 0, S(s), m, n_in); break;
        case 2: hipLaunchKernelGGL((k_multi_q8_0<2>), g, dim3(256), 0, S(s), m, n_in); break;
        case 3: hipLaunchKernelGGL((k_multi_q8_0<3>), g, dim3(256), 0, S(s), m, n_in); break;
        case 4: hipLaunchKernelGGL((k_multi_q8_0<4>), g, dim3(256), 0, S(s), m, n_in); break;
        case 5: hipLaunchKernelGGL((k_multi_q8_0<5>), g, dim3(256), 0, S(s), m, n_in); break;
        case 6: hipLaunchKernelGGL((k_multi_q8_0<6>), g, dim3(256), 0, S(s), m, n_in); break;
        case 7: hipLaunchKernelGGL((k_multi_q8_0<7>), g, dim3(256), 0, S(s), m, n_in); break;
        default: hipLaunchKernelGGL((k_multi_q8_0<8>), g, dim3(256), 0, S(s), m, n_in); break;
    }
    return last();
}


int tkg_moe_route_sort(const int32_t* ids, int T, int k, int n_exp, int32_t* rowsrc, int32_t* rownat, int32_t* bounds,
                       void* s) {
    if (n_exp > 1024) return err(hipErrorInvalidValue);
    static const bool v2 = [] { const char* e = std::getenv("GLM_SORT2"); return !e || e[0] != '0'; }();
    if (v2 && n_exp <= RS2_MAXE) {   // the parallel form (identical outputs); GLM_SORT2=0 = one thread per expert
        hipLaunchKernelGGL(k_moe_route_sort2, dim3(1), dim3(1024), 0, S(s), ids, T * k, k, n_exp, rowsrc, rownat, bounds);
        return last();
    }
    hipLaunchKernelGGL(k_moe_route_sort, dim3(1), dim3(1024), 0, S(s), ids, T * k, k, n_exp, rowsrc, rownat, bounds);
    return last();
}


int tkg_gather_q8_1(const float* x, const int32_t* ids, void* xq, int n_in, int rows, void* s) {
    if (n_in % 32) return err(hipErrorInvalidValue);
    const int nblk = rows * (n_in / 32);
    hipLaunchKernelGGL(k_gather_q8_1, dim3((nblk + 7) / 8), dim3(256), 0, S(s), x, ids, static_cast<q8_1_blk*>(xq), n_in,
                       rows);
    return last();
}

void tkg_glm_set_grouped_ver(int v) { talus::glm_spike::glm_set_grouped_ver(v); }

int tkg_glm_mmvq_grouped(int qtype, const void* w, size_t expert_bytes, int n_experts, const int32_t* bounds,
                         const void* xq, const int32_t* ids_dst, float* y, int N, int K, int ld_y, void* s) {
    return err(talus::glm_spike::glm_mmvq_grouped(qtype, w, expert_bytes, n_experts, bounds, xq, ids_dst, y, N, K, ld_y,
                                                  S(s)));
}

int tkg_quantize_q8_1(const float* x, void* xq, int n, void* s) {
    return err(talus::glm_spike::glm_quantize_q8_1(x, xq, n, S(s)));
}

int tkg_glm_mmvq_ids(int qtype, const void* w, const void* xq, const int32_t* ids, float* y, int N, int K,
                     int n_expert_used, int n_tokens, int x_group, int ids_s1, void* s) {
    return err(talus::glm_spike::glm_mmvq_ids(qtype, w, xq, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1,
                                              S(s)));
}

int tkg_glm_mmvq_ids_pair(int qtype, const void* w, const void* w2, const void* xq, const int32_t* ids, float* y,
                          float* y2, int N, int K, int n_expert_used, int n_tokens, int x_group, int ids_s1, void* s) {
    const hipError_t e = talus::glm_spike::glm_mmvq_ids_pair(qtype, w, w2, xq, ids, y, y2, N, K, n_expert_used, n_tokens,
                                                            x_group, ids_s1, S(s));
    if (e == hipErrorNotSupported) return -1;   // not launched: the caller runs two glm_mmvq_ids
    return err(e);
}

int tkg_glm_kda_step(const float* q, const float* k, const float* v, const float* g, const float* beta, float* state,
                     const int32_t* ids, const float* dtb, const float* a, float* dst, int H, int S_v, int T,
                     int K_snap, float scale, float pro_lb, float pro_l2_eps, void* s) {
    return err(talus::glm_spike::glm_kda_step(q, k, v, g, beta, state, ids, dtb, a, dst, H, S_v, T, K_snap, 1, scale,
                                              true, pro_lb, pro_l2_eps, 0, S(s)));
}

// KDA conv + step + gate (q8_1 out) in one launch, bitwise the three launches (tkg_kda_conv, tkg_glm_kda_step,
// tkg_kda_gate_q8); `cnt` = H ints of zeros (left zero). -1 = configuration without a fused kernel (use the three).
int tkg_kda_fused(const float* qr, const float* kr, const float* vr, float* conv_state, const float* wq, const float* wk,
                  const float* wv, const float* g, const float* beta, float* state, const int32_t* ids, const float* dtb,
                  const float* a, float* o, const float* g2, const float* onorm_w, void* oq, int* cnt, int H, int D,
                  int T, int K_snap, float scale, float pro_lb, float pro_l2_eps, float gate_eps, void* s) {
    if (D != 128 || T < 1 || T > 8 || K_snap < 1 || K_snap > T) return -1;
    // the step layout glm_kda_step picks on this arch (GLM HIP v2: gfx1151 auto, or GLM_KDA_CPW = 10000 + 100 NWV + CPW)
    static const int cfg_env = [] { const char* e = std::getenv("GLM_KDA_CPW"); return e ? std::atoi(e) : 0; }();
    static const bool gfx1151 = [] {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        return std::strncmp(p.gcnArchName, "gfx1151", 7) == 0;
    }();
    const int cfg = cfg_env > 0 ? cfg_env : (gfx1151 ? (T <= 1 ? 11601 : 10808) : 0);
#define KF_L(CPW_, NWV_)                                                                                               \
    if (cfg == 10000 + 100 * NWV_ + CPW_) {                                                                          \
        hipLaunchKernelGGL((k_kda_fused<CPW_, NWV_>), dim3(H, 128 / (CPW_ * NWV_)), dim3(32 * NWV_), 0, S(s), qr, kr, \
                           vr, conv_state, wq, wk, wv, g, beta, state, ids, dtb, a, o, g2, onorm_w,                  \
                           static_cast<q8_1_blk*>(oq), cnt, (uint32_t) H, (uint32_t) T, (uint32_t) K_snap, scale,     \
                           pro_lb, pro_l2_eps, gate_eps);                                                            \
        return last();                                                                                               \
    }
    KF_L(1, 16) KF_L(8, 8) KF_L(2, 8) KF_L(4, 8) KF_L(1, 8) KF_L(2, 16) KF_L(4, 16)
#undef KF_L
    return -1;
}

// prompt-window conv + step (k_kda_step_conv_pf), bitwise tkg_kda_conv then tkg_glm_kda_step; -1 = not covered
// (GLM_KDA_CONV_PF=0, T <= 8, K_snap outside 1..3, D != 128). `cnt` = H ints of zeros (left zero).
int tkg_kda_step_conv_pf(const float* qr, const float* kr, const float* vr, float* conv_state, const float* wq,
                         const float* wk, const float* wv, const float* g, const float* beta, float* state,
                         const int32_t* ids, const float* dtb, const float* a, float* o, int* cnt, int H, int D, int T,
                         int K_snap, float scale, float pro_lb, float pro_l2_eps, void* s) {
    // opt-in (GLM_KDA_CONV_PF=1): measured slower than conv + the v3 step (the per-token conv work sits on the
    // serial token chain: 32K 9.7 s vs 6.1 s per prompt)
    static const bool on = [] { const char* e = std::getenv("GLM_KDA_CONV_PF"); return e && e[0] == '1'; }();
    if (!on || D != 128 || T <= 8 || K_snap < 1 || K_snap > 3) return -1;
    hipLaunchKernelGGL((k_kda_step_conv_pf<8, 8>), dim3(H, 2), dim3(256), 0, S(s), qr, kr, vr, conv_state, wq, wk, wv, g,
                       beta, state, ids, dtb, a, o, cnt, (uint32_t) H, (uint32_t) T, (uint32_t) K_snap, scale, pro_lb,
                       pro_l2_eps);
    return last();
}

int tkg_glm_indexer(const float* q, const void* k, const float* w, float* dst, int D, int H, int n_kv, int n_tokens,
                    void* s) {
    return err(talus::glm_spike::glm_dsa_lightning_indexer(q, k, w, dst, D, H, n_kv, n_tokens, 1, true, S(s)));
}

int tkg_glm_sparse_fa_idx(const float* q, const void* kv, const void* zero_mask, const int32_t* idx, float* split,
                          float* dst, int D, int H, int N, int n_kv_max, uint32_t q_nb1, uint32_t q_nb2,
                          uint32_t k_stride, float scale, void* s) {
    return err(talus::glm_spike::glm_dsa_sparse_fa_idx(q, static_cast<const __half*>(kv),
                                                       static_cast<const __half*>(zero_mask), idx, split, dst, D, H, N,
                                                       n_kv_max, q_nb1, q_nb2, k_stride, scale, S(s)));
}

size_t tkg_glm_sparse_fa_split_bytes(int D, int H, int N, int n_kv_max) {
    return talus::glm_spike::glm_dsa_sparse_fa_split_bytes(D, H, N, n_kv_max);
}

}  // extern "C"
