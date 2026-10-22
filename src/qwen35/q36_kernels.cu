// src/qwen35/q36_kernels.cu - Extracted __global__ kernels and host launchers from q36_engine.cu,
// plus C ABI entry points (tk_q36_*) for libtalus_kernels.so.
#include "strata/qwen35/q36_kernels.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/tk_internal.hpp"
#include "strata/prefill/fattn_prefill.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "q36_adraft.hpp"
#include "q36_dflash.hpp"
#include "talus_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

using namespace talus::q36;

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// Element i of a q8_1 vector (blocks of 32), all 32 lanes of a wave holding the 32 values of one block in lane order:
/// native_quantize_q8_1's scale, sum and value, bit for bit.
__device__ __forceinline__ void tq_q8_put(float xi, void* q8, size_t i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o, 32);
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    TqQ81* b = (TqQ81*) q8 + i / 32;
    b->qs[i % 32] = strata::kernels::q8_1_quant(xi, d, amax);
    if (i % 32 == 0) b->ds = strata::kernels::q8_1_ds(d, sum);
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps, void* q8 = nullptr) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i] * r * w[i];
        yr[i] = xi;
        if (q8) tq_q8_put(xi, q8, (size_t) blockIdx.x * n + i);
    }
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps,
                         void* q8 = nullptr) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i] * r * w[i];
        yr[i] = xi;
        if (q8) tq_q8_put(xi, q8, (size_t) blockIdx.x * ld_out + i);
    }
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps, void* q8 = nullptr) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i] * r * w[i];
        hr[i] = xi;
        if (q8) tq_q8_put(xi, q8, (size_t) blockIdx.x * n + i);
    }
}
/// Talus: k_add_rms and native_quantize_q8_1 of h in one launch (decode / verify rows): h as k_add_rms writes it and
/// its q8_1 image (blocks of 32: d = amax / 127, the fp16 sum) as native_quantize_q8_1 writes it, bit for bit - the 32
/// values of a block are one wave's (i = tid + 256 k), reduced by the same XOR butterflies.  q8: T rows of n / 32 blocks.
/// (1024 threads: a 5120-row in five strides; the sum of squares by wave butterflies, then the waves' sums in order)
__global__ void __launch_bounds__(1024) k_add_rms_q8(float* x, const float* y, const float* w, float* h, void* q8, int n,
                                                     float eps) {
    __shared__ float wsum[32];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    TqQ81* qr = (TqQ81*) q8 + (size_t) blockIdx.x * (n / 32);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, 32);
    if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = s;
    __syncthreads();
    s = 0.f;
    for (int k = 0; k < (int) (blockDim.x >> 5); ++k) s += wsum[k];
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i] * r * w[i];
        hr[i] = xi;
        float amax = fabsf(xi), sum = xi;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o, 32);
        const float d = strata::kernels::q8_1_finite(amax / 127.0f);
        qr[i / 32].qs[i % 32] = strata::kernels::q8_1_quant(xi, d, amax);
        if (i % 32 == 0) qr[i / 32].ds = strata::kernels::q8_1_ds(d, sum);
    }
}

/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps, void* q8 = nullptr) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i] * r * wn[i];
        h[(size_t) t * n + i] = xi;
        if (q8) tq_q8_put(xi, q8, (size_t) t * n + i);
    }
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
/// Talus: k_gdn_ab with one block per output row (8 waves split n_in, the waves' partial sums added in order) and every
/// one of the T <= 8 rows of x against each weight load.  grid (2 HV), block 256; n_in % 1024 == 0.
__device__ __forceinline__ void gdn_ab_body(int o, const float* Wa, const float* Wb, const float* x, const float* dt,
                                            const float* ssm_a, float* alpha, float* beta, int n_in, int HV, int T) {
    __shared__ float part[8][16];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const int q4 = n_in / 4, per = q4 / 8;   // float4s per wave
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in) + warp * per;
    float acc[16] = {};
    for (int i = lane; i < per; i += 32) {
        const float4 a = wr[i];
#pragma unroll
        for (int t = 0; t < 16; ++t) {
            if (t >= T) break;
            const float4 b = ((const float4*) (x + (size_t) t * n_in))[warp * per + i];
            acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < 16; ++t) {
        if (t >= T) break;
        float v = acc[t];
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
        if (lane == 0) part[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x >= T) return;
    const int t = threadIdx.x;
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) v += part[k][t];
    if (isb) beta[(size_t) t * HV + r] = sigm(v);
    else alpha[(size_t) t * HV + r] = softplus(v + dt[r]) * ssm_a[r];
}
__global__ void __launch_bounds__(256) k_gdn_ab2(const float* Wa, const float* Wb, const float* x, const float* dt,
                                                 const float* ssm_a, float* alpha, float* beta, int n_in, int HV, int T) {
    gdn_ab_body(blockIdx.x, Wa, Wb, x, dt, ssm_a, alpha, beta, n_in, HV, T);
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// k_gdn_gb on a prompt chunk's fused alpha|beta product ab [T][2 HV] : alpha / beta [T][HV] out.
__global__ void k_gdn_gb2(const float* ab, const float* dt, const float* ssm_a, float* alpha, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV, t = i / HV;
    alpha[i] = softplus(ab[(size_t) t * 2 * HV + h] + dt[h]) * ssm_a[h];
    beta[i] = sigm(ab[(size_t) t * 2 * HV + HV + h]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// The cell-blocked V copy of k_attn_wmma: element (kv head, cell, d) of a cache with vt_ctx cells.

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// grid (H + HKV, T), block 256 (HD <= 256).
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, __half* kc, __half* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, __half* vt = nullptr,
                            int vt_ctx = 0) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = *pos0 + t;
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const float theta = (float) pos * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        kc[((size_t) kh * ctx + pos) * HD + i] = __float2half(v);
        const __half hv = __float2half(vx[((size_t) t * HKV + kh) * HD + i]);
        vc[((size_t) kh * ctx + pos) * HD + i] = hv;
        if (vt) vt[vt_index(kh, pos, i, vt_ctx)] = hv;   // k_attn_wmma's cell-blocked copy
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.
__global__ void k_attn_part(const float* q, const __half* kc, const __half* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = *pos0 + t + 1;
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        const __half2* kp = (const __half2*) (kc + ((size_t) kvh * ctx + p) * HD);
        float a = 0.f;
        for (int i = 0; i < HD / 2; ++i) {
            const float2 k2 = __half22float2(kp[i]);
            a = fmaf(qs[2 * i], k2.x, a);
            a = fmaf(qs[2 * i + 1], k2.y, a);
        }
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    const __half* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * __half2float(vp[(size_t) k * HD]);
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.  `chunk`: the cells per chunk.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, int chunk = kChunkCells,
                             void* q8 = nullptr) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = *pos0 + t + 1;
    const int nc = att_chunk_count(npos, chunk);
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    const float o = num / den * sigm(gate[gi]);
    out[gi] = o;
    if (q8) tq_q8_put(o, q8, gi);   // (Talus: the output projection's q8_1 input)
}

// ---- decode / verify attention on the gfx11 matrix cores (WMMA f16 -> f32)
// One block per (chunk of `chunk` cells, KV head) takes every query row of that KV head at once - the G query heads of
// the group times the T tokens of the batch (R = T G <= 48 rows, RT tiles of 16) - so each K / V cell is read once per
// KV head, not once per query head and token.  Per tile of 32 cells: S = Q K^T (4 waves: 2 cell tiles x 2 halves of
// the head dim, the halves summed in a fixed order), the masked online softmax over the tile, then O += P V (each wave
// 64 of the 256 dims).  Q and P are FP16 (the K / V cache is), the accumulation FP32.  V is read from `vt`, a copy of
// the V cache in blocks of 16 cells, [kv head][cell / 16][dim][cell % 16], so each WMMA operand is one contiguous 32-byte
// load.  The chunk boundaries do not depend on T and every row's arithmetic is its own (a row's masked cells add exact
// zeros), so a verify window's row is bitwise the decode step's row at that position.  Writes the chunk's partials as
// k_attn_part does (k_attn_merge merges them).  grid (n_chunk, HKV), block 128; head dim 256, chunk % 32 == 0.
typedef _Float16 tq_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 tq_h8 __attribute__((ext_vector_type(8)));
typedef float tq_f8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ tq_h16 tq_lds16(const _Float16* p) {   // two 16-byte loads (LDS rows are 16-byte aligned)
    const tq_h8 lo = *(const tq_h8*) p, hi = *(const tq_h8*) (p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
__device__ __forceinline__ tq_f8 tq_wmma(tq_h16 a, tq_h16 b, tq_f8 c) {
#if defined(__gfx11__) || defined(__GFX11__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b;
    return c;   // (never launched: wmma_attn_ok() is false off gfx11)
#endif
}
template <int RT>
__global__ void __launch_bounds__(128) k_attn_wmma(const float* __restrict__ q, const __half* __restrict__ kc,
                                                   const __half* __restrict__ vt, const int32_t* pos0, int T, int H,
                                                   int HKV, int ctx, int vt_ctx, float scale, int chunk, int n_chunk,
                                                   float* __restrict__ pacc, float* __restrict__ pm,
                                                   float* __restrict__ pl, int r0 = 0, int ngrp = 1) {
    constexpr int HD = 256, QP = HD + 8, PP = kAttTile + 8, MR = 16 * RT;
    __shared__ _Float16 Q16[MR * QP];
    __shared__ _Float16 P16[MR * PP];
    __shared__ float Sp[MR][kAttTile + 1];
    __shared__ float mxw[2][MR], smw[2][MR], m_run[MR], l_run[MR], alf[MR];
    // Talus: ngrp row groups of MR rows per chunk in one launch, as adjacent blocks (x = chunk * ngrp +
    // group), so the blocks of one chunk read its K / V cells at about the same time (the second read from cache) -
    // the separate launches per row group would otherwise re-read the cache from memory
    const int kvh = blockIdx.y, c = (int) blockIdx.x / ngrp, tid = threadIdx.x;
    r0 += ((int) blockIdx.x % ngrp) * MR;
    const int w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    // Talus: this block's rows r0 .. r0 + R - 1 of the T G (more than 32 rows run as row groups of 2 tiles, so no
    // instance needs 3 tiles' registers: RT = 3 spilled); a row's arithmetic does not depend on its tile mates
    const int G = H / HKV, R = min(T * G - r0, MR);
    if (R <= 0) return;
    const int p0 = *pos0, nmax = p0 + T;
    int cbeg, clen;
    att_chunk_range(c, chunk, cbeg, clen);
    if (cbeg >= nmax) return;
    const int cend = min(cbeg + clen, nmax);
    for (int i = tid; i < MR * HD; i += 128) {
        const int r = i / HD, d = i % HD;
        float v = 0.f;
        if (r < R) v = q[((size_t) ((r + r0) / G) * H + kvh * G + (r + r0) % G) * HD + d];
        Q16[r * QP + d] = (_Float16) v;
    }
    if (tid < MR) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.f;
    }
    tq_f8 oacc[RT][4];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int dt = 0; dt < 4; ++dt)
#pragma unroll
            for (int i = 0; i < 8; ++i) oacc[rt][dt][i] = 0.f;
    __syncthreads();
    const _Float16* kbase = (const _Float16*) kc + (size_t) kvh * ctx * HD;
    const _Float16* vbase = (const _Float16*) vt + (size_t) kvh * vt_ctx * HD;
    const int ct = w & 1, kh = w >> 1;
    // the tiles below every row's position (all their cells are every row's), then the tail (some row's position is
    // in or below the tile: the per-row sums), as two loops so the main one carries no tail code
    auto tile = [&](int base, auto tailc) {
        // S = Q K^T: this wave's 16 cells x half the head dim
        tq_f8 sacc[RT];
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int i = 0; i < 8; ++i) sacc[rt][i] = 0.f;
        const int cell = base + ct * 16 + l16;
        const bool cv = cell < nmax;
        const _Float16* kp = kbase + (size_t) (cv ? cell : 0) * HD;
#pragma unroll 2
        for (int k8 = 0; k8 < 8; ++k8) {
            const int ks = kh * 8 + k8;
            tq_h16 b = *(const tq_h16*) (kp + ks * 16);
            if (!cv) b = (tq_h16) {};
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) sacc[rt] = tq_wmma(tq_lds16(&Q16[(rt * 16 + l16) * QP + ks * 16]), b, sacc[rt]);
        }
        if (kh == 1) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) Sp[rt * 16 + 2 * i + hi][ct * 16 + l16] = sacc[rt][i];
        }
        __syncthreads();
        if (kh == 0) {   // the scores (masked per row: token t sees the cells < p0 + t + 1), the tile's row maxima
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    const bool ok = r < R && cell < p0 + (r + r0) / G + 1;
                    const float s = ok ? (sacc[rt][i] + Sp[r][ct * 16 + l16]) * scale : -INFINITY;
                    sacc[rt][i] = s;
                    float m = s;
#pragma unroll
                    for (int o = 1; o < 16; o <<= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
                    if (l16 == 0) mxw[ct][r] = m;
                }
        }
        __syncthreads();
        if (kh == 0) {   // P = exp(S - the new running max) as FP16, the tile's row sums
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    const float mn = fmaxf(m_run[r], fmaxf(mxw[0][r], mxw[1][r]));
                    const float p = sacc[rt][i] == -INFINITY ? 0.f : expf(sacc[rt][i] - mn);
                    P16[r * PP + ct * 16 + l16] = (_Float16) p;
                    float s = p;
#pragma unroll
                    for (int o = 1; o < 16; o <<= 1) s += __shfl_xor(s, o, 32);
                    if (l16 == 0) smw[ct][r] = s;
                }
        }
        __syncthreads();
        if (tid < MR) {
            const float mo = m_run[tid], mn = fmaxf(mo, fmaxf(mxw[0][tid], mxw[1][tid]));
            const float a = mn == -INFINITY ? 1.f : expf(mo - mn);
            l_run[tid] = l_run[tid] * a + (smw[0][tid] + smw[1][tid]);
            m_run[tid] = mn;
            alf[tid] = a;
        }
        __syncthreads();
        // O = O * alpha + P V over this wave's 64 dims.  The matrix cores' sum of a row is not independent of the V of
        // the cells its P zeroes (0 x V moves it), and a decode step and a verify window hold different values past a
        // row's position.  So: a tile whose 32 cells are all the row's (p >= base + 31) adds the WMMA product P V (from
        // zero, then to O); the tile holding the row's position adds the scalar sum over the cells <= p only, in cell
        // order; tiles past it add nothing.  Every row's arithmetic is then the same in a decode step and a window.
        constexpr bool tail = decltype(tailc)::value;   // some row has its position in or below this tile
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
            const int d = w * 64 + dt * 16 + l16;
            const _Float16* vd = vbase + (size_t) d * 16;   // + (cell / 16) * HD * 16 + cell % 16
            tq_h16 b[2];
#pragma unroll
            for (int cs = 0; cs < 2; ++cs) b[cs] = *(const tq_h16*) (vd + (size_t) ((base >> 4) + cs) * HD * 16);
            tq_f8 dacc[RT];
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) dacc[rt][i] = 0.f;
#pragma unroll
                for (int cs = 0; cs < 2; ++cs) dacc[rt] = tq_wmma(tq_lds16(&P16[(rt * 16 + l16) * PP + cs * 16]), b[cs], dacc[rt]);
            }
            if constexpr (tail) {   // the rows whose position is in this tile: the scalar sum over their cells, in cell order
                float own[RT][8];
#pragma unroll
                for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) own[rt][i] = 0.f;
#pragma unroll 1
                for (int c = 0; c < kAttTile; ++c) {
                    const float v = (float) vd[(size_t) ((base + c) >> 4) * HD * 16 + ((base + c) & 15)];
#pragma unroll
                    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int r = rt * 16 + 2 * i + hi;
                            const int pr = p0 + (r + r0) / G;
                            if (r < R && pr >= base + c && pr < base + kAttTile - 1)
                                own[rt][i] = fmaf((float) P16[r * PP + c], v, own[rt][i]);
                        }
                }
#pragma unroll
                for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int r = rt * 16 + 2 * i + hi;
                        const int pr = p0 + (r + r0) / G;
                        if (r < R && pr < base + kAttTile - 1) dacc[rt][i] = own[rt][i];
                    }
            }
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    if (r < R && p0 + (r + r0) / G >= base) oacc[rt][dt][i] = fmaf(oacc[rt][dt][i], alf[r], dacc[rt][i]);
                }
        }
        __syncthreads();   // P16 / Sp / the row stats are rewritten by the next tile
    };
    const int tail_start = max(cbeg, ((p0 + 1) / kAttTile) * kAttTile);
    for (int base = cbeg; base < min(cend, tail_start); base += kAttTile) tile(base, std::false_type());
    for (int base = tail_start; base < cend; base += kAttTile) tile(base, std::true_type());
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = rt * 16 + 2 * i + hi;
            if (r >= R) continue;
            const size_t row = (size_t) ((r + r0) / G) * H + kvh * G + (r + r0) % G;
#pragma unroll
            for (int dt = 0; dt < 4; ++dt) pacc[(row * n_chunk + c) * HD + w * 64 + dt * 16 + l16] = oacc[rt][dt][i];
        }
    if (tid < R) {
        const size_t row = (size_t) ((tid + r0) / G) * H + kvh * G + (tid + r0) % G;
        pm[row * n_chunk + c] = m_run[tid];
        pl[row * n_chunk + c] = l_run[tid];
    }
}
/// k_attn_wmma with 8 waves and the next tile's K / V operands loaded while the current tile computes: S over 2 cell
/// tiles x 4 quarters of the head dim (the quarters summed in order), O over 8 x 32 dims.  The same per-row
/// independence (a verify row is the decode row bitwise); its sums differ from k_attn_wmma's (another split of the
/// head dim).  grid (n_chunk, HKV), block 256.
template <int RT>
__global__ void __launch_bounds__(256) k_attn_wmma8(const float* __restrict__ q, const __half* __restrict__ kc,
                                                    const __half* __restrict__ vt, const int32_t* pos0, int T, int H,
                                                    int HKV, int ctx, int vt_ctx, float scale, int chunk, int n_chunk,
                                                    float* __restrict__ pacc, float* __restrict__ pm,
                                                    float* __restrict__ pl) {
    constexpr int HD = 256, QP = HD + 8, PP = kAttTile + 8, MR = 16 * RT;
    __shared__ _Float16 Q16[MR * QP];
    __shared__ _Float16 P16[MR * PP];
    __shared__ float Sp[3][MR][kAttTile + 1];
    __shared__ float mxw[2][MR], smw[2][MR], m_run[MR], l_run[MR], alf[MR];
    const int kvh = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const int w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int G = H / HKV, R = T * G;
    const int p0 = *pos0, nmax = p0 + T;
    int cbeg, clen;
    att_chunk_range(c, chunk, cbeg, clen);
    if (cbeg >= nmax) return;
    const int cend = min(cbeg + clen, nmax);
    for (int i = tid; i < MR * HD; i += 256) {
        const int r = i / HD, d = i % HD;
        float v = 0.f;
        if (r < R) v = q[((size_t) (r / G) * H + kvh * G + r % G) * HD + d];
        Q16[r * QP + d] = (_Float16) v;
    }
    if (tid < MR) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.f;
    }
    tq_f8 oacc[RT][2];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int dt = 0; dt < 2; ++dt)
#pragma unroll
            for (int i = 0; i < 8; ++i) oacc[rt][dt][i] = 0.f;
    const _Float16* kbase = (const _Float16*) kc + (size_t) kvh * ctx * HD;
    const _Float16* vbase = (const _Float16*) vt + (size_t) kvh * vt_ctx * HD;
    const int ct = w & 1, kq = w >> 1;   // QK: this wave's cell tile and quarter of the head dim
    // this tile's operands: K (4 k-steps of this wave's 16 cells), V (2 x 16 cells x this wave's 2 dim tiles)
    tq_h16 kf[4], kn[4], vf[2][2];
    auto load_k = [&](int base, tq_h16 (&kk)[4]) {
        const int cell = base + ct * 16 + l16;
        const bool cv = cell < nmax;
        const _Float16* kp = kbase + (size_t) (cv ? cell : 0) * HD + kq * 64;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            kk[j] = *(const tq_h16*) (kp + j * 16);
            if (!cv) kk[j] = (tq_h16) {};
        }
    };
    load_k(cbeg, kf);
    __syncthreads();
    for (int base = cbeg; base < cend; base += kAttTile) {
        // this tile's V (read in the PV step, after the scores and the softmax) and the next tile's K
#pragma unroll
        for (int cs = 0; cs < 2; ++cs)
#pragma unroll
            for (int dt = 0; dt < 2; ++dt)
                vf[cs][dt] = *(const tq_h16*) (vbase + ((size_t) ((base >> 4) + cs) * HD + w * 32 + dt * 16 + l16) * 16);
        if (base + kAttTile < cend) load_k(base + kAttTile, kn);
        tq_f8 sacc[RT];
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int i = 0; i < 8; ++i) sacc[rt][i] = 0.f;
        const int cell = base + ct * 16 + l16;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int ks = kq * 4 + j;
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) sacc[rt] = tq_wmma(tq_lds16(&Q16[(rt * 16 + l16) * QP + ks * 16]), kf[j], sacc[rt]);
        }
        if (kq > 0) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) Sp[kq - 1][rt * 16 + 2 * i + hi][ct * 16 + l16] = sacc[rt][i];
        }
        __syncthreads();
        if (kq == 0) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    const bool ok = r < R && cell < p0 + r / G + 1;
                    const int cc = ct * 16 + l16;
                    const float s = ok ? (((sacc[rt][i] + Sp[0][r][cc]) + Sp[1][r][cc]) + Sp[2][r][cc]) * scale
                                       : -INFINITY;
                    sacc[rt][i] = s;
                    float m = s;
#pragma unroll
                    for (int o = 1; o < 16; o <<= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
                    if (l16 == 0) mxw[ct][r] = m;
                }
        }
        __syncthreads();
        if (kq == 0) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    const float mn = fmaxf(m_run[r], fmaxf(mxw[0][r], mxw[1][r]));
                    const float p = sacc[rt][i] == -INFINITY ? 0.f : expf(sacc[rt][i] - mn);
                    P16[r * PP + ct * 16 + l16] = (_Float16) p;
                    float sm = p;
#pragma unroll
                    for (int o = 1; o < 16; o <<= 1) sm += __shfl_xor(sm, o, 32);
                    if (l16 == 0) smw[ct][r] = sm;
                }
        }
        __syncthreads();
        if (tid < MR) {
            const float mo = m_run[tid], mn = fmaxf(mo, fmaxf(mxw[0][tid], mxw[1][tid]));
            const float a = mn == -INFINITY ? 1.f : expf(mo - mn);
            l_run[tid] = l_run[tid] * a + (smw[0][tid] + smw[1][tid]);
            m_run[tid] = mn;
            alf[tid] = a;
        }
        __syncthreads();
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float a = alf[rt * 16 + 2 * i + hi];
#pragma unroll
                for (int dt = 0; dt < 2; ++dt) oacc[rt][dt][i] *= a;
            }
#pragma unroll
        for (int cs = 0; cs < 2; ++cs) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                const tq_h16 a = tq_lds16(&P16[(rt * 16 + l16) * PP + cs * 16]);
#pragma unroll
                for (int dt = 0; dt < 2; ++dt) oacc[rt][dt] = tq_wmma(a, vf[cs][dt], oacc[rt][dt]);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) kf[j] = kn[j];
        __syncthreads();
    }
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = rt * 16 + 2 * i + hi;
            if (r >= R) continue;
            const size_t row = (size_t) (r / G) * H + kvh * G + r % G;
#pragma unroll
            for (int dt = 0; dt < 2; ++dt) pacc[(row * n_chunk + c) * HD + w * 32 + dt * 16 + l16] = oacc[rt][dt][i];
        }
    if (tid < R) {
        const size_t row = (size_t) (tid / G) * H + kvh * G + tid % G;
        pm[row * n_chunk + c] = m_run[tid];
        pl[row * n_chunk + c] = l_run[tid];
    }
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2,
                         const int32_t* map = nullptr) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = map ? map[bi[0]] : bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}

/// Talus: the draft of a sampled request, drawn from q = softmax(l / tau) of the draft head's row (Gumbel-max with a
/// counter-based hash of (seed, round, step, id)); tau <= 0: the greedy pick.  The host reads q back (the row and its
/// log-sum-exp) to accept the draft with probability min(1, p / q).
__device__ __forceinline__ float tq_u01(uint64_t k) {
    k += 0x9E3779B97F4A7C15ull;
    k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9ull;
    k = (k ^ (k >> 27)) * 0x94D049BB133111EBull;
    k ^= k >> 31;
    return ((float) (uint32_t) (k >> 40) + 0.5f) * (1.0f / 16777216.0f);
}
/// k_argmax's stage 1 over l / tau + Gumbel noise (grid kArgBlocks, block kThreads).
__global__ void k_argmax_samp(const float* x, int n, float* pv, int32_t* pi, const DraftSamp* ds, int step,
                              const float* thr = nullptr) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    const float tau = ds->tau;
    const float th = thr && tau > 0.f ? *thr : -INFINITY;   // (x / tau below th: outside the draft's support)
    const uint64_t base = ((uint64_t) ds->seed << 32) ^ ((uint64_t) ds->ctr * 64 + (uint64_t) step) * 0xD6E8FEB86659FD93ull;
    float v = -INFINITY;
    int id = 0x7fffffff;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        if (tau > 0.f && !(x[i] / tau >= th)) continue;
        const float xv = tau > 0.f ? x[i] / tau - logf(-logf(tq_u01(base + (uint64_t) i * 0x9E3779B97F4A7C15ull))) : x[i];
        if (xv > v) { v = xv; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
}
/// out = log sum exp(l / tau) of a row of n (tau <= 0: 0) over the tokens with l / tau >= thr = max(l / tau) + lmin
/// (Talus: the draft's min-p support; *thr_out gets thr).  One block of 1024.
__global__ void __launch_bounds__(1024) k_lse(const float* x, int n, const DraftSamp* ds, float* out, float* thr_out = nullptr) {
    __shared__ float sh[32];
    const float tau = ds->tau;
    if (tau <= 0.f) { if (threadIdx.x == 0) *out = 0.f; return; }
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i] / tau);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    m = sh[0];
    for (int k = 1; k < 32; ++k) m = fmaxf(m, sh[k]);
    __syncthreads();
    const float th = ds->lmin > -INFINITY ? m + ds->lmin : -INFINITY;   // (min-p on the tempered probabilities, as sample())
    if (thr_out && threadIdx.x == 0) *thr_out = th;
    float e = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = x[i] / tau;
        if (xi >= th) e += expf(xi - m);
    }
    for (int o = 16; o > 0; o >>= 1) e += __shfl_xor(e, o, 32);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = e;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < 32; ++k) t += sh[k];
        *out = m + logf(t);
    }
}

// ---- Talus: speculative rounds on the device (see Engine::spec_*)
/// The end of a forward of <= kVerify rows with the fused commit: a verify window's stored rows become the replay set.
__global__ void k_ctl_end(int32_t* ctl, bool flip) {
    if (flip) ctl[1] ^= 1;
    ctl[0] = 0;
}
/// Graph B's start, rctl = [a, x'] from the host: the MTP rows at positions P + 1 + i (P = *pos0, the window's first
/// position) read d_{i+1} for i < a and x' at i = a (later rows: any token), the next window starts at x' (tok[0], at
/// P + a + 1), the next forward replays the a + 1 kept rows.  One block of >= n threads.
__global__ void k_round_prep(const int32_t* rctl, int32_t* tok, int32_t* tok_m, int32_t* pos0, int32_t* pos0_m,
                             int32_t* gctl, int n) {
    const int a = rctl[0], xn = rctl[1], i = threadIdx.x;
    int v = 0;
    if (i < n) v = i < a ? tok[i + 1] : xn;
    __syncthreads();
    if (i < n) tok_m[i] = v;
    if (i == 0) {
        const int P = *pos0;
        *pos0_m = P + 1;
        *pos0 = P + a + 1;
        if (gctl) gctl[0] = a + 1;
        tok[0] = xn;
    }
}
/// dst = row *idx of src (rows of n floats).
__global__ void k_row_copy(float* dst, const float* src, const int32_t* idx, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) dst[j] = src[(size_t) *idx * n + j];
}
/// Chained draft k (2 .. D): the MTP row at position *pos0 + k - 1 reads draft k - 1.
__global__ void k_chain_prep(const int32_t* tok, int32_t* tok_m, const int32_t* pos0, int32_t* pos0_m, int k) {
    tok_m[0] = tok[k - 1];
    *pos0_m = *pos0 + k - 1;
}


/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked small 8 x 8 / 16 x 16 tiles for these shapes.  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
/// Talus: the commit of a verify window fused into the next forward.  A window stores its rows' GDN inputs (the conv
/// inputs and outputs, gates, betas) into set 1 - ctl[1] of a layer's pending buffers; once its kept count is known,
/// ctl[0] = that count, and the next forward of <= kVerify rows first replays those rows from set ctl[1] - the same
/// loop iterations as the window's own, so the state is bitwise commit()'s - then runs its own.  A window writes the
/// state right after the replay (its own rows are the next window's to replay), a decode step at its end.
/// k_conv_seq's body for channel c, the hb-th 128-channel head of its block (part: 4 floats per head of the block).
__device__ __forceinline__ void conv_body(float* hist, const float* qkv, const float* w, float* h, int C, bool qk,
                                          float eps, int T, bool write, const GdnPend& pd, int c, float* part) {
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const int nr = pd.ctl ? pd.ctl[0] : 0, ps = pd.ctl ? pd.ctl[1] : 0;
    for (int tt = 0; tt < nr + T; ++tt) {
        const bool rp = tt < nr;
        const int t = rp ? tt : tt - nr;
        const float x = rp ? pd.q[ps][(size_t) tt * C + c] : qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[(threadIdx.x >> 5) & 3] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        if (rp) {
            if (tt == nr - 1 && !write) {   // a verify window: the replayed rows committed
                hist[c * 3] = v0;
                hist[c * 3 + 1] = v1;
                hist[c * 3 + 2] = v2;
            }
            continue;
        }
        h[(size_t) t * C + c] = y;
        if (pd.store) {
            pd.q[1 - ps][(size_t) t * C + c] = x;
            pd.h[1 - ps][(size_t) t * C + c] = y;
        }
    }
    if (!write) return;   // a verify window: the history is commit()'s to move
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write, GdnPend pd = GdnPend()) {
    __shared__ float part[4];
    conv_body(hist, qkv, w, h, C, (int) blockIdx.x < qk_heads, eps, T, write, pd, blockIdx.x * 128 + threadIdx.x, part);
}
/// Talus: k_conv_seq (two 128-channel heads per block, the same per-channel arithmetic) and k_gdn_ab2 in one launch:
/// blocks [0, C / 256) the conv, then 2 HV blocks of alpha / beta.  block 256.
__global__ void __launch_bounds__(256) k_conv_ab(float* hist, const float* qkv, const float* w, float* hc, int C,
                                                 int qk_heads, float eps, int T, bool write, GdnPend pd, const float* Wa,
                                                 const float* Wb, const float* x, const float* dt, const float* ssm_a,
                                                 float* alpha, float* beta, int n_in, int HV) {
    __shared__ float part[8];
    const int nconv = C / 256;
    if ((int) blockIdx.x < nconv) {
        const int hb = threadIdx.x >> 7;
        conv_body(hist, qkv, w, hc, C, (int) blockIdx.x * 2 + hb < qk_heads, eps, T, write, pd,
                  blockIdx.x * 256 + threadIdx.x, part + hb * 4);
        return;
    }
    gdn_ab_body(blockIdx.x - nconv, Wa, Wb, x, dt, ssm_a, alpha, beta, n_in, HV, T);
}

/// Talus prefill: k_conv_seq for a prompt chunk with every token in parallel (the 4-tap conv is not recurrent): grid
/// (C / 128, T), block 128; the same arithmetic per output.  The history is then moved by k_conv_hist.
__global__ void __launch_bounds__(128) k_conv_par(const float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T) {
    __shared__ float part[4];
    const int c = blockIdx.x * 128 + threadIdx.x, t = blockIdx.y;
    auto in = [&](int j) { return j >= 0 ? qkv[(size_t) j * C + c] : hist[c * 3 + 3 + j]; };
    const float v0 = in(t - 3), v1 = in(t - 2), v2 = in(t - 1), x = qkv[(size_t) t * C + c];
    const float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        y *= rsqrtf(part[0] + part[1] + part[2] + part[3] + eps);
    }
    h[(size_t) t * C + c] = y;
}
/// The conv history after a chunk of T tokens: its last 3 inputs (with the old history when T < 3).
__global__ void k_conv_hist(float* hist, const float* qkv, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int i = 0; i < 3; ++i) {
        const int j = T - 3 + i;
        v[i] = j >= 0 ? qkv[(size_t) j * C + c] : hist[c * 3 + 3 + j];
    }
    for (int i = 0; i < 3; ++i) hist[c * 3 + i] = v[i];
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 GdnPend pd = GdnPend(), void* yq = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    // Talus: every shared array twice, by row parity, and the next row's k / q written during this row:
    // 3 barriers per output row (were 5) and 2 per replayed row (were 3).  A buffer is rewritten only after every
    // thread has passed a barrier that follows its last read; the arithmetic and its order are the same, bit for bit.
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[2][RG][S];
    __shared__ float red_o[2][RG][S];
    __shared__ float wsum[2][S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    const int nr = pd.ctl ? pd.ctl[0] : 0, ps = pd.ctl ? pd.ctl[1] : 0;
    // a row's inputs are loaded one row ahead (registers), so their latency hides behind the row before
    float c_k = 0.f, c_q = 0.f, c_v = 0.f, c_g = 0.f, c_b = 0.f;
    auto fetch = [&](int tt2, float& fk, float& fq, float& fv, float& fg, float& fb) {
        if (tt2 >= nr + T) return;
        const bool rp2 = tt2 < nr;
        const int t2 = rp2 ? tt2 : tt2 - nr;
        const float* h2 = rp2 ? pd.h[ps] + (size_t) tt2 * C : h + (size_t) t2 * C;
        fg = rp2 ? pd.a[ps][(size_t) tt2 * h_v + head] : gate[(size_t) t2 * h_v + head];
        fb = rp2 ? pd.b[ps][(size_t) tt2 * h_v + head] : beta[(size_t) t2 * h_v + head];
        fv = h2[2 * h_k * S + head * S + col];
        if (tid < S) { fk = h2[h_k * S + qh * S + tid]; fq = h2[qh * S + tid]; }
    };
    fetch(0, c_k, c_q, c_v, c_g, c_b);
    if (tid < S) { sk[0][tid] = c_k; sq[0][tid] = c_q; }
    __syncthreads();
    for (int tt = 0; tt < nr + T; ++tt) {
        const int pb = tt & 1;
        const bool rp = tt < nr;
        const int t = rp ? tt : tt - nr;
        const float gv = c_g, bv = c_b, vv = c_v;
        if (!rp && pd.store && tid == 0) {
            pd.a[1 - ps][(size_t) t * h_v + head] = gv;
            pd.b[1 - ps][(size_t) t * h_v + head] = bv;
        }
        fetch(tt + 1, c_k, c_q, c_v, c_g, c_b);   // (this row's sk / sq are in shared memory)
        const float g = __expf(gv);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[pb][rg * RPG + r], kv);
        red_kv[pb][rg][col] = kv;
        __syncthreads();
        if (tid < S) { sk[1 - pb][tid] = c_k; sq[1 - pb][tid] = c_q; }   // the next row's (read after a barrier below)
        const float kv_col = red_kv[pb][0][col] + red_kv[pb][1][col] + red_kv[pb][2][col] + red_kv[pb][3][col];
        const float delta = (vv - g * kv_col) * bv;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[pb][rg * RPG + r] * delta);
            o = fmaf(s[r], sq[pb][rg * RPG + r], o);
        }
        if (rp) {   // a replayed row: the state only
            __syncthreads();
            if (tt == nr - 1 && !write) {   // a verify window: the replayed rows committed
#pragma unroll
                for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
            }
            continue;
        }
        red_o[pb][rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[pb][0][col] + red_o[pb][1][col] + red_o[pb][2][col] + red_o[pb][3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[pb][tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[pb][0] + wsum[pb][1] + wsum[pb][2] + wsum[pb][3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            const float yv = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
            y[(size_t) t * ZV + head * S + col] = yv;
            if (yq) tq_q8_put(yv, yq, (size_t) t * ZV + head * S + col);   // (Talus: ssm_out's q8_1 input)
        }
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
        __syncthreads();
        if (rg == 0) {
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) t * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// Talus prefill: the prompt's gated delta rule with each wave owning NCOL state columns (llama.cpp's
/// gated_delta_net_cuda layout, widened): a column's 128 rows over the 32 lanes (4 each, in registers), kv and o reduced
/// with shuffles - no block barrier per token; NCOL independent columns per wave hide the shuffle / load latency, and
/// few enough waves (HV * 128 / NCOL) stay resident at once.  The next token's k / q / v / gate / beta are loaded while
/// this one is computed.  The state stays [row][head][col] (the decode kernels' layout).  Writes o_raw like
/// k_gdn_seq_cols.  grid (HV, 128 / (4 NCOL)), block (32, 4).
template <int NCOL>
__global__ void __launch_bounds__(128) k_gdn_seq_warp(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, RPL = S / 32;
    const int head = blockIdx.x, lane = threadIdx.x;
    const int col0 = (blockIdx.y * 4 + threadIdx.y) * NCOL;
    const int qh = head % h_k;
    const size_t row_stride = (size_t) h_v * S;
    float* base = state + (size_t) head * S + col0;
    float sv[NCOL][RPL];
#pragma unroll
    for (int c = 0; c < NCOL; ++c)
#pragma unroll
        for (int r = 0; r < RPL; ++r) sv[c][r] = base[(size_t) (r * 32 + lane) * row_stride + c];
    const float* qp = h + qh * S + lane;
    const float* kp = h + h_k * S + qh * S + lane;
    const float* vp = h + 2 * h_k * S + head * S + col0;
    float kn[RPL], qn[RPL], vn[NCOL], gn = 0.0f, bn = 0.0f;
    if (T > 0) {
#pragma unroll
        for (int r = 0; r < RPL; ++r) { kn[r] = kp[r * 32]; qn[r] = qp[r * 32]; }
#pragma unroll
        for (int c = 0; c < NCOL; ++c) vn[c] = vp[c];
        gn = gate[head]; bn = beta[head];
    }
    const float osc = rsqrtf((float) S);
    for (int t = 0; t < T; ++t) {
        float kr[RPL], qr[RPL], vt[NCOL];
#pragma unroll
        for (int r = 0; r < RPL; ++r) { kr[r] = kn[r]; qr[r] = qn[r]; }
#pragma unroll
        for (int c = 0; c < NCOL; ++c) vt[c] = vn[c];
        const float g = __expf(gn), bt = bn;
        if (t + 1 < T) {
            const size_t o = (size_t) (t + 1) * C;
#pragma unroll
            for (int r = 0; r < RPL; ++r) { kn[r] = kp[o + r * 32]; qn[r] = qp[o + r * 32]; }
#pragma unroll
            for (int c = 0; c < NCOL; ++c) vn[c] = vp[o + c];
            gn = gate[(size_t) (t + 1) * h_v + head];
            bn = beta[(size_t) (t + 1) * h_v + head];
        }
        float kv[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) {
            kv[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < RPL; ++r) kv[c] = fmaf(sv[c][r], kr[r], kv[c]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
#pragma unroll
            for (int c = 0; c < NCOL; ++c) kv[c] += __shfl_xor_sync(0xffffffffu, kv[c], off);
        float o[NCOL];
#pragma unroll
        for (int c = 0; c < NCOL; ++c) {
            const float delta = (vt[c] - g * kv[c]) * bt;
            o[c] = 0.0f;
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                sv[c][r] = fmaf(g, sv[c][r], kr[r] * delta);
                o[c] = fmaf(sv[c][r], qr[r], o[c]);
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
#pragma unroll
            for (int c = 0; c < NCOL; ++c) o[c] += __shfl_xor_sync(0xffffffffu, o[c], off);
        if (lane < NCOL) {   // lane c writes column col0 + c
            float oc = o[0];
#pragma unroll
            for (int c = 1; c < NCOL; ++c) if (lane == c) oc = o[c];
            o_raw[(size_t) t * h_v * S + head * S + col0 + lane] = oc * osc;
        }
    }
#pragma unroll
    for (int c = 0; c < NCOL; ++c)
#pragma unroll
        for (int r = 0; r < RPL; ++r) base[(size_t) (r * 32 + lane) * row_stride + c] = sv[c][r];
}

/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}


/// Reads n float4 (a cache flush that leaves no dirty lines behind).
__global__ void k_flush_read(const float4* p, size_t n, float* out) {
    float a = 0.f;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) a += p[i].x;
    if (a == 12345.f) *out = a;
}

}   // namespace

namespace talus::q36 {

/// k_attn_wmma (4 waves) or k_attn_wmma8 (8 waves, w8) for T rows, RT = ceil(T G / 16) tiles of 16 query rows.
void launch_attn_wmma(bool w8, dim3 grid, cudaStream_t st, const float* q, const __half* kc, const __half* vt,
                      const int32_t* pos0, int T, int H, int HKV, int ctx, int vt_ctx, float sc, int chunk, int n_chunk,
                      float* pacc, float* pm, float* pl) {
    const int rt = (T * (H / HKV) + 15) / 16;
#define TQ_ATT(K, R, NT) K<R><<<grid, NT, 0, st>>>(q, kc, vt, pos0, T, H, HKV, ctx, vt_ctx, sc, chunk, n_chunk, pacc, pm, pl)
    if (w8) {
        if (rt == 1) TQ_ATT(k_attn_wmma8, 1, 256);
        else if (rt == 2) TQ_ATT(k_attn_wmma8, 2, 256);
        else TQ_ATT(k_attn_wmma8, 3, 256);
    } else {
        static const bool rt3 = std::getenv("STRATA_ATTN_RT3") != nullptr;   // the 3-tile instance (spills; A/B)
        // windows above 16 rows as groups of 16 rows (the 1-tile instance at 181 VGPRs / 8 waves per SIMD,
        // adjacent blocks per chunk so the second group reads K / V from cache) instead of 2-tile groups (256 VGPRs,
        // 5 waves); a row's arithmetic does not depend on its tile mates, so the result is bitwise the same.
        // TALUS_ATTN_RT1G=0: revert to 2-tile groups.
        static const int rt1g = [] { const char* e = std::getenv("TALUS_ATTN_RT1G"); return e ? std::atoi(e) : 1; }();
        if (rt > 1 && rt1g) {
            const int ng = (T * (H / HKV) + 15) / 16;
            k_attn_wmma<1><<<dim3(grid.x * ng, grid.y), 128, 0, st>>>(q, kc, vt, pos0, T, H, HKV, ctx, vt_ctx, sc, chunk,
                                                                      n_chunk, pacc, pm, pl, 0, ng);
            return;
        }
        if (rt == 1) TQ_ATT(k_attn_wmma, 1, 128);
        else if (rt == 2) TQ_ATT(k_attn_wmma, 2, 128);
        else if (rt3 && rt == 3) TQ_ATT(k_attn_wmma, 3, 128);
        else {   // rows in groups of 32 (2 tiles): one launch, a chunk's groups adjacent (STRATA_ATTN_SPLIT=0: a launch per group)
            static const bool split = [] { const char* e = std::getenv("STRATA_ATTN_SPLIT"); return !(e && e[0] == '0'); }();
            const int R = T * (H / HKV);
            if (split) {
                const int ng = (R + 31) / 32;
                k_attn_wmma<2><<<dim3(grid.x * ng, grid.y), 128, 0, st>>>(q, kc, vt, pos0, T, H, HKV, ctx, vt_ctx, sc, chunk,
                                                                          n_chunk, pacc, pm, pl, 0, ng);
            } else
                for (int r0 = 0; r0 < R; r0 += 32) {
                    if (R - r0 > 16)
                        k_attn_wmma<2><<<grid, 128, 0, st>>>(q, kc, vt, pos0, T, H, HKV, ctx, vt_ctx, sc, chunk, n_chunk, pacc, pm, pl, r0);
                    else
                        k_attn_wmma<1><<<grid, 128, 0, st>>>(q, kc, vt, pos0, T, H, HKV, ctx, vt_ctx, sc, chunk, n_chunk, pacc, pm, pl, r0);
                }
        }
    }
#undef TQ_ATT
}
bool g_attn_w8 = false;   // k_attn_wmma8: no faster (--attn-bench) and not exact (its P V is the shared WMMA); kept for reference


void launch_rms(int rows, cudaStream_t st, const float* x, const float* w, float* y, int n, float eps, void* q8) {
    k_rms<<<rows, kThreads, 0, st>>>(x, w, y, n, eps, q8);
}

void launch_rms_ld(int rows, cudaStream_t st, const float* x, int ldx, const float* w, float* out, int ld_out, int n,
                   float eps, void* q8) {
    k_rms_ld<<<rows, kThreads, 0, st>>>(x, ldx, w, out, ld_out, n, eps, q8);
}

void launch_add_rms(int rows, cudaStream_t st, float* x, const float* y, const float* w, float* h, int n, float eps,
                    void* q8) {
    k_add_rms<<<rows, kThreads, 0, st>>>(x, y, w, h, n, eps, q8);
}

void launch_add_rms_q8(int rows, cudaStream_t st, float* x, const float* y, const float* w, float* h, void* q8, int n,
                       float eps) {
    k_add_rms_q8<<<rows, 1024, 0, st>>>(x, y, w, h, q8, n, eps);
}

void launch_moe_out(int rows, cudaStream_t st, float* x, const float* parts, const int32_t* prow, const float* wts,
                    int K_, const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                    float eps, void* q8) {
    k_moe_out<<<rows, kThreads, 0, st>>>(x, parts, prow, wts, K_, shared, sgate, sg_stride, wn, h, n, eps, q8);
}

void launch_gdn_ab(int T, cudaStream_t st, const float* Wa, const float* Wb, const float* x, const float* dt,
                   const float* ssm_a, float* alpha, float* beta, int n_in, int HV) {
    k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(Wa, Wb, x, dt, ssm_a, alpha, beta, n_in, HV);
}

void launch_gdn_ab2(cudaStream_t st, const float* Wa, const float* Wb, const float* x, const float* dt,
                    const float* ssm_a, float* alpha, float* beta, int n_in, int HV, int T) {
    k_gdn_ab2<<<2 * HV, 256, 0, st>>>(Wa, Wb, x, dt, ssm_a, alpha, beta, n_in, HV, T);
}

void launch_add(cudaStream_t st, float* x, const float* y, int64_t n) {
    k_add<<<grid1(n), kThreads, 0, st>>>(x, y, n);
}

void launch_swiglu(cudaStream_t st, const float* g, const float* u, float* h, int64_t n) {
    k_swiglu<<<grid1(n), kThreads, 0, st>>>(g, u, h, n);
}

void launch_gemv_f32(int T, cudaStream_t st, const float* W, const float* x, float* y, int n_in, int n_out) {
    k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, x, y, n_in, n_out);
}

void launch_gdn_gb(cudaStream_t st, float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(alpha, dt, ssm_a, beta, HV, T);
}

void launch_gdn_gb2(cudaStream_t st, const float* ab, const float* dt, const float* ssm_a, float* alpha,
                    float* beta, int HV, int T) {
    k_gdn_gb2<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(ab, dt, ssm_a, alpha, beta, HV, T);
}

void launch_gate_rows(cudaStream_t st, float* sh, const float* gate, int n, int T) {
    k_gate_rows<<<grid1((int64_t) n * T), kThreads, 0, st>>>(sh, gate, n, T);
}

void launch_attn_prep(int T, cudaStream_t st, const float* qg, const float* kx, const float* vx, const float* qw,
                      const float* kw, float* q, float* gate, __half* kc, __half* vc, int H, int HKV, int HD,
                      int n_rot, const int32_t* pos0, int ctx, float base, float eps, __half* vt, int vt_ctx) {
    k_attn_prep<<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, qw, kw, q, gate, kc, vc, H, HKV, HD, n_rot, pos0,
                                                       ctx, base, eps, vt, vt_ctx);
}

void launch_attn_part(dim3 grid, cudaStream_t st, const float* q, const __half* kc, const __half* vc,
                      const int32_t* pos0, int t0, int H, int HKV, int ctx, float scale, int n_chunk, float* pacc,
                      float* pm, float* pl) {
    k_attn_part<<<grid, kChunkCells, 0, st>>>(q, kc, vc, pos0, t0, H, HKV, ctx, scale, n_chunk, pacc, pm, pl);
}

void launch_attn_merge(int rows, int HD, cudaStream_t st, const float* pacc, const float* pm, const float* pl,
                       const float* gate, const int32_t* pos0, int t0, int H, int n_chunk, float* out, int chunk,
                       void* q8) {
    k_attn_merge<<<rows, HD, 0, st>>>(pacc, pm, pl, gate, pos0, t0, H, n_chunk, out, chunk, q8);
}

void launch_group(cudaStream_t st, const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs,
                  unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                  int32_t* ent_dst, int32_t* ent_tok) {
    k_group<<<1, kThreads, 0, st>>>(ids, n_ent, K_, n_expert, blobs, blob, grp_ptr, grp_start, n_groups, ent_dst,
                                    ent_tok);
}

void launch_group_hy(cudaStream_t st, const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                     const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                     int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, K_, n_expert, slot_of, blobs, blob, grp_ptr, grp_start, n_groups,
                                       ent_dst, ent_tok);
}

void launch_merge_cpu(int n_ent, cudaStream_t st, float* parts, const float* cpu, const int32_t* miss, int n) {
    k_merge_cpu<<<n_ent, kThreads, 0, st>>>(parts, cpu, miss, n);
}

void launch_attn_softmax16(int H, cudaStream_t st, const float* S, __half* S16, int ld, int T, int P, int pos0,
                           float scale) {
    k_attn_softmax16<<<dim3(T, H), kThreads, 0, st>>>(S, S16, ld, T, P, pos0, scale);
}

void launch_gate_mul(cudaStream_t st, float* att, const float* gate, int64_t n) {
    k_gate_mul<<<grid1(n), kThreads, 0, st>>>(att, gate, n);
}

void launch_route_count(cudaStream_t st, const int32_t* ids, int n_ent, int32_t* cnt) {
    k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, cnt);
}

void launch_route_scan(cudaStream_t st, const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur,
                       int32_t* max_rows) {
    k_route_scan<<<1, 32, 0, st>>>(cnt, n_expert, bounds, cur, max_rows);
}

void launch_route_scatter(cudaStream_t st, const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst,
                          int32_t* src, int32_t* inv) {
    k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, n_ent, K_, cur, dst, src, inv);
}

void launch_argmax(dim3 grid, cudaStream_t st, const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2,
                   const int32_t* map) {
    k_argmax<<<grid, kThreads, 0, st>>>(x, n, out, pv, pi, stage2, map);
}

void launch_argmax_samp(cudaStream_t st, const float* x, int n, float* pv, int32_t* pi, const DraftSamp* ds, int step,
                        const float* thr) {
    k_argmax_samp<<<kArgBlocks, kThreads, 0, st>>>(x, n, pv, pi, ds, step, thr);
}

void launch_lse(cudaStream_t st, const float* x, int n, const DraftSamp* ds, float* out, float* thr_out) {
    k_lse<<<1, 1024, 0, st>>>(x, n, ds, out, thr_out);
}

void launch_ctl_end(cudaStream_t st, int32_t* ctl, bool flip) {
    k_ctl_end<<<1, 1, 0, st>>>(ctl, flip);
}

void launch_round_prep(cudaStream_t st, const int32_t* rctl, int32_t* tok, int32_t* tok_m, int32_t* pos0,
                       int32_t* pos0_m, int32_t* gctl, int n) {
    k_round_prep<<<1, 32, 0, st>>>(rctl, tok, tok_m, pos0, pos0_m, gctl, n);
}

void launch_row_copy(cudaStream_t st, float* dst, const float* src, const int32_t* idx, int n) {
    k_row_copy<<<grid1(n), kThreads, 0, st>>>(dst, src, idx, n);
}

void launch_chain_prep(cudaStream_t st, const int32_t* tok, int32_t* tok_m, const int32_t* pos0, int32_t* pos0_m,
                       int k) {
    k_chain_prep<<<1, 1, 0, st>>>(tok, tok_m, pos0, pos0_m, k);
}

void launch_sgemm_nt(int tm, dim3 grid, cudaStream_t st, const float* W, const float* x, float* out, int K, int n_out,
                     int T, int ldo, int kz) {
    if (tm == 4)
        k_sgemm_nt<4><<<grid, 256, 0, st>>>(W, x, out, K, n_out, T, ldo, kz);
    else
        k_sgemm_nt<1><<<grid, 256, 0, st>>>(W, x, out, K, n_out, T, ldo, kz);
}

void launch_sum_splits(cudaStream_t st, const float* part, float* out, int64_t n, int splits) {
    k_sum_splits<<<grid1(n), kThreads, 0, st>>>(part, out, n, splits);
}

void launch_to_f16(cudaStream_t st, const float* x, __half* y, int64_t n) {
    k_to_f16<<<grid1(n), kThreads, 0, st>>>(x, y, n);
}

void launch_conv_seq(cudaStream_t st, float* hist, const float* qkv, const float* w, float* h, int C, int qk_heads,
                     float eps, int T, bool write, GdnPend pd) {
    k_conv_seq<<<C / 128, 128, 0, st>>>(hist, qkv, w, h, C, qk_heads, eps, T, write, pd);
}

void launch_conv_ab(cudaStream_t st, float* hist, const float* qkv, const float* w, float* hc, int C, int qk_heads,
                    float eps, int T, bool write, GdnPend pd, const float* Wa, const float* Wb, const float* x,
                    const float* dt, const float* ssm_a, float* alpha, float* beta, int n_in, int HV) {
    k_conv_ab<<<C / 256 + 2 * HV, 256, 0, st>>>(hist, qkv, w, hc, C, qk_heads, eps, T, write, pd, Wa, Wb, x, dt,
                                                ssm_a, alpha, beta, n_in, HV);
}

void launch_conv_par(cudaStream_t st, const float* hist, const float* qkv, const float* w, float* h, int C,
                     int qk_heads, float eps, int T) {
    k_conv_par<<<dim3(C / 128, T), 128, 0, st>>>(hist, qkv, w, h, C, qk_heads, eps, T);
}

void launch_conv_hist(cudaStream_t st, float* hist, const float* qkv, int C, int T) {
    k_conv_hist<<<(C + 127) / 128, 128, 0, st>>>(hist, qkv, C, T);
}

void launch_gdn_seq(cudaStream_t st, float* state, const float* h, int C, const float* gate, const float* beta,
                    const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int T, bool write,
                    GdnPend pd, void* yq) {
    k_gdn_seq<<<h_v, dim3(128, 4), 0, st>>>(state, h, C, gate, beta, z, gamma, eps, y, h_k, h_v, T, write, pd, yq);
}

void launch_gdn_seq_cols(cudaStream_t st, float* state, const float* h, int C, const float* gate, const float* beta,
                         float* o_raw, int h_k, int h_v, int T) {
    k_gdn_seq_cols<<<dim3(h_v, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(state, h, C, gate, beta, o_raw,
                                                                                       h_k, h_v, T);
}

void launch_gdn_seq_warp(int ncol, cudaStream_t st, float* state, const float* h, int C, const float* gate,
                         const float* beta, float* o_raw, int h_k, int h_v, int T) {
    if (ncol == 1)
        k_gdn_seq_warp<1><<<dim3(h_v, 128 / 4), dim3(32, 4), 0, st>>>(state, h, C, gate, beta, o_raw, h_k, h_v, T);
    else if (ncol == 2)
        k_gdn_seq_warp<2><<<dim3(h_v, 128 / 8), dim3(32, 4), 0, st>>>(state, h, C, gate, beta, o_raw, h_k, h_v, T);
    else if (ncol == 4)
        k_gdn_seq_warp<4><<<dim3(h_v, 128 / 16), dim3(32, 4), 0, st>>>(state, h, C, gate, beta, o_raw, h_k, h_v, T);
    else if (ncol == 16)
        k_gdn_seq_warp<16><<<dim3(h_v, 128 / 64), dim3(32, 4), 0, st>>>(state, h, C, gate, beta, o_raw, h_k, h_v, T);
    else
        k_gdn_seq_warp<8><<<dim3(h_v, 128 / 32), dim3(32, 4), 0, st>>>(state, h, C, gate, beta, o_raw, h_k, h_v, T);
}

void launch_gdn_norm(int T, cudaStream_t st, const float* o_raw, const float* z, const float* gamma, float eps,
                     float* y, int h_v) {
    k_gdn_norm<<<dim3(h_v, T), 128, 0, st>>>(o_raw, z, gamma, eps, y, h_v);
}

void launch_flush_read(cudaStream_t st, const float4* p, size_t n, float* out) {
    k_flush_read<<<1024, 256, 0, st>>>(p, n, out);
}

}   // namespace talus::q36

namespace {
inline talus::q36::GdnPend to_gdn_pend(const tk_q36_gdn_pend* pd) {
    talus::q36::GdnPend out{};
    if (pd) {
        out.q[0] = pd->q[0];
        out.q[1] = pd->q[1];
        out.h[0] = pd->h[0];
        out.h[1] = pd->h[1];
        out.a[0] = pd->a[0];
        out.a[1] = pd->a[1];
        out.b[0] = pd->b[0];
        out.b[1] = pd->b[1];
        out.ctl = pd->ctl;
        out.store = pd->store;
    }
    return out;
}
inline cudaStream_t as_cuda_stream(tk_stream s) noexcept {
    return reinterpret_cast<cudaStream_t>(s);
}

inline tk_status check_last_cuda_error(tk_ctx* ctx, const char* op) noexcept {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string msg = std::string(op) + ": " + cudaGetErrorString(err);
        tk_set_error(ctx, msg.c_str());
        return TK_ERR_HIP;
    }
    return TK_OK;
}

inline void ensure_q36_env() noexcept {
    static const bool init = [] {
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
        setenv("STRATA_Q8_PACKED", "1", 0);
        setenv("STRATA_MMQ_KQ", "1", 0);
        return true;
    }();
    (void) init;
}
}   // namespace

extern "C" {

tk_status tk_q36_rms(tk_ctx* ctx, int32_t rows, const float* x, const float* w, float* y, int32_t n, float eps,
                     tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_rms(rows, as_cuda_stream(stream), x, w, y, n, eps);
    return check_last_cuda_error(ctx, "tk_q36_rms");
    TK_TRY_END(ctx);
}

tk_status tk_q36_rms_ld(tk_ctx* ctx, int32_t rows, const float* x, int32_t ldx, const float* w, float* out,
                        int32_t ld_out, int32_t n, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_rms_ld(rows, as_cuda_stream(stream), x, ldx, w, out, ld_out, n, eps);
    return check_last_cuda_error(ctx, "tk_q36_rms_ld");
    TK_TRY_END(ctx);
}

tk_status tk_q36_add_rms(tk_ctx* ctx, int32_t rows, float* x, const float* y, const float* w, float* h, int32_t n,
                         float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_add_rms(rows, as_cuda_stream(stream), x, y, w, h, n, eps);
    return check_last_cuda_error(ctx, "tk_q36_add_rms");
    TK_TRY_END(ctx);
}

tk_status tk_q36_add_rms_q8(tk_ctx* ctx, int32_t rows, float* x, const float* y, const float* w, float* h, void* q8,
                            int32_t n, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_add_rms_q8(rows, as_cuda_stream(stream), x, y, w, h, q8, n, eps);
    return check_last_cuda_error(ctx, "tk_q36_add_rms_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_moe_out(tk_ctx* ctx, int32_t rows, float* x, const float* parts, const int32_t* prow,
                         const float* wts, int32_t k, const float* shared, const float* sgate, int32_t sg_stride,
                         const float* wn, float* h, int32_t n, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_moe_out(rows, as_cuda_stream(stream), x, parts, prow, wts, k, shared, sgate, sg_stride, wn, h, n,
                               eps);
    return check_last_cuda_error(ctx, "tk_q36_moe_out");
    TK_TRY_END(ctx);
}

tk_status tk_q36_rms_q8(tk_ctx* ctx, int32_t rows, const float* x, const float* w, float* y, void* q8, int32_t n,
                        float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_rms(rows, as_cuda_stream(stream), x, w, y, n, eps, q8);
    return check_last_cuda_error(ctx, "tk_q36_rms_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_rms_ld_q8(tk_ctx* ctx, int32_t rows, const float* x, int32_t ldx, const float* w, float* out,
                           int32_t ld_out, void* q8, int32_t n, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_rms_ld(rows, as_cuda_stream(stream), x, ldx, w, out, ld_out, n, eps, q8);
    return check_last_cuda_error(ctx, "tk_q36_rms_ld_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_add_rms_exact_q8(tk_ctx* ctx, int32_t rows, float* x, const float* y, const float* w, float* h,
                                  void* q8, int32_t n, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_add_rms(rows, as_cuda_stream(stream), x, y, w, h, n, eps, q8);
    return check_last_cuda_error(ctx, "tk_q36_add_rms_exact_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_moe_out_q8(tk_ctx* ctx, int32_t rows, float* x, const float* parts, const int32_t* prow,
                            const float* wts, int32_t k, const float* shared, const float* sgate, int32_t sg_stride,
                            const float* wn, float* h, void* q8, int32_t n, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_moe_out(rows, as_cuda_stream(stream), x, parts, prow, wts, k, shared, sgate, sg_stride, wn, h, n,
                               eps, q8);
    return check_last_cuda_error(ctx, "tk_q36_moe_out_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_ab(tk_ctx* ctx, int32_t n_tok, const float* wa, const float* wb, const float* x, const float* dt,
                        const float* ssm_a, float* alpha, float* beta, int32_t n_in, int32_t hv,
                        tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_ab(n_tok, as_cuda_stream(stream), wa, wb, x, dt, ssm_a, alpha, beta, n_in, hv);
    return check_last_cuda_error(ctx, "tk_q36_gdn_ab");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_ab2(tk_ctx* ctx, const float* wa, const float* wb, const float* x, const float* dt,
                         const float* ssm_a, float* alpha, float* beta, int32_t n_in, int32_t hv, int32_t n_tok,
                         tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_ab2(as_cuda_stream(stream), wa, wb, x, dt, ssm_a, alpha, beta, n_in, hv, n_tok);
    return check_last_cuda_error(ctx, "tk_q36_gdn_ab2");
    TK_TRY_END(ctx);
}

tk_status tk_q36_add(tk_ctx* ctx, float* x, const float* y, int64_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_add(as_cuda_stream(stream), x, y, n);
    return check_last_cuda_error(ctx, "tk_q36_add");
    TK_TRY_END(ctx);
}

tk_status tk_q36_swiglu(tk_ctx* ctx, const float* g, const float* u, float* h, int64_t n,
                        tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_swiglu(as_cuda_stream(stream), g, u, h, n);
    return check_last_cuda_error(ctx, "tk_q36_swiglu");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gemv_f32(tk_ctx* ctx, int32_t n_tok, const float* w, const float* x, float* y, int32_t n_in,
                          int32_t n_out, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gemv_f32(n_tok, as_cuda_stream(stream), w, x, y, n_in, n_out);
    return check_last_cuda_error(ctx, "tk_q36_gemv_f32");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_gb(tk_ctx* ctx, float* alpha, const float* dt, const float* ssm_a, float* beta, int32_t hv,
                        int32_t n_tok, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_gb(as_cuda_stream(stream), alpha, dt, ssm_a, beta, hv, n_tok);
    return check_last_cuda_error(ctx, "tk_q36_gdn_gb");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_gb2(tk_ctx* ctx, const float* ab, const float* dt, const float* ssm_a, float* alpha, float* beta,
                         int32_t hv, int32_t n_tok, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_gb2(as_cuda_stream(stream), ab, dt, ssm_a, alpha, beta, hv, n_tok);
    return check_last_cuda_error(ctx, "tk_q36_gdn_gb2");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gate_rows(tk_ctx* ctx, float* sh, const float* gate, int32_t n, int32_t n_tok,
                           tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gate_rows(as_cuda_stream(stream), sh, gate, n, n_tok);
    return check_last_cuda_error(ctx, "tk_q36_gate_rows");
    TK_TRY_END(ctx);
}

tk_status tk_q36_attn_prep(tk_ctx* ctx, int32_t n_tok, const float* qg, const float* kx, const float* vx,
                           const float* qw, const float* kw, float* q, float* gate, void* kc, void* vc, int32_t h,
                           int32_t hkv, int32_t hd, int32_t n_rot, const int32_t* pos0, int32_t ctx_len, float base,
                           float eps, void* vt, int32_t vt_ctx, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_attn_prep(n_tok, as_cuda_stream(stream), qg, kx, vx, qw, kw, q, gate, static_cast<__half*>(kc),
                                 static_cast<__half*>(vc), h, hkv, hd, n_rot, pos0, ctx_len, base, eps,
                                 static_cast<__half*>(vt), vt_ctx);
    return check_last_cuda_error(ctx, "tk_q36_attn_prep");
    TK_TRY_END(ctx);
}

tk_status tk_q36_attn_part(tk_ctx* ctx, int32_t grid_x, int32_t grid_y, const float* q, const void* kc, const void* vc,
                           const int32_t* pos0, int32_t t0, int32_t h, int32_t hkv, int32_t ctx_len, float scale,
                           int32_t n_chunk, float* pacc, float* pm, float* pl, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_attn_part(dim3((unsigned) grid_x, (unsigned) grid_y), as_cuda_stream(stream), q,
                                 static_cast<const __half*>(kc), static_cast<const __half*>(vc), pos0, t0, h, hkv,
                                 ctx_len, scale, n_chunk, pacc, pm, pl);
    return check_last_cuda_error(ctx, "tk_q36_attn_part");
    TK_TRY_END(ctx);
}

tk_status tk_q36_attn_merge(tk_ctx* ctx, int32_t rows, int32_t hd, const float* pacc, const float* pm, const float* pl,
                            const float* gate, const int32_t* pos0, int32_t t0, int32_t h, int32_t n_chunk, float* out,
                            int32_t chunk, void* q8, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_attn_merge(rows, hd, as_cuda_stream(stream), pacc, pm, pl, gate, pos0, t0, h, n_chunk, out,
                                  chunk, q8);
    return check_last_cuda_error(ctx, "tk_q36_attn_merge");
    TK_TRY_END(ctx);
}

tk_status tk_q36_attn_wmma(tk_ctx* ctx, bool w8, int32_t grid_x, int32_t grid_y, const float* q, const void* kc,
                           const void* vt, const int32_t* pos0, int32_t n_tok, int32_t h, int32_t hkv, int32_t ctx_len,
                           int32_t vt_ctx, float sc, int32_t chunk, int32_t n_chunk, float* pacc, float* pm, float* pl,
                           tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_attn_wmma(w8, dim3((unsigned) grid_x, (unsigned) grid_y), as_cuda_stream(stream), q,
                                 static_cast<const __half*>(kc), static_cast<const __half*>(vt), pos0, n_tok, h, hkv,
                                 ctx_len, vt_ctx, sc, chunk, n_chunk, pacc, pm, pl);
    return check_last_cuda_error(ctx, "tk_q36_attn_wmma");
    TK_TRY_END(ctx);
}

tk_status tk_q36_group(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t k, int32_t n_expert,
                       const uint8_t* blobs, uint64_t blob, uint64_t* grp_ptr, int32_t* grp_start, int32_t* n_groups,
                       int32_t* ent_dst, int32_t* ent_tok, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    static_assert(sizeof(unsigned long long) == sizeof(uint64_t), "uint64_t must match unsigned long long");
    talus::q36::launch_group(as_cuda_stream(stream), ids, n_ent, k, n_expert, blobs, (unsigned long long) blob,
                             reinterpret_cast<unsigned long long*>(grp_ptr), grp_start, n_groups, ent_dst, ent_tok);
    return check_last_cuda_error(ctx, "tk_q36_group");
    TK_TRY_END(ctx);
}

tk_status tk_q36_group_hy(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t k, int32_t n_expert,
                          const int32_t* slot_of, const uint8_t* blobs, uint64_t blob, uint64_t* grp_ptr,
                          int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok,
                          tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_group_hy(as_cuda_stream(stream), ids, n_ent, k, n_expert, slot_of, blobs,
                                (unsigned long long) blob, reinterpret_cast<unsigned long long*>(grp_ptr), grp_start,
                                n_groups, ent_dst, ent_tok);
    return check_last_cuda_error(ctx, "tk_q36_group_hy");
    TK_TRY_END(ctx);
}

tk_status tk_q36_merge_cpu(tk_ctx* ctx, int32_t n_ent, float* parts, const float* cpu, const int32_t* miss, int32_t n,
                           tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_merge_cpu(n_ent, as_cuda_stream(stream), parts, cpu, miss, n);
    return check_last_cuda_error(ctx, "tk_q36_merge_cpu");
    TK_TRY_END(ctx);
}

tk_status tk_q36_attn_softmax16(tk_ctx* ctx, int32_t h, const float* s_in, void* s16_out, int32_t ld, int32_t n_tok,
                                int32_t p_cols, int32_t pos0, float scale, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_attn_softmax16(h, as_cuda_stream(stream), s_in, static_cast<__half*>(s16_out), ld, n_tok, p_cols,
                                      pos0, scale);
    return check_last_cuda_error(ctx, "tk_q36_attn_softmax16");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gate_mul(tk_ctx* ctx, float* att, const float* gate, int64_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gate_mul(as_cuda_stream(stream), att, gate, n);
    return check_last_cuda_error(ctx, "tk_q36_gate_mul");
    TK_TRY_END(ctx);
}

tk_status tk_q36_route_count(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t* cnt,
                             tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_route_count(as_cuda_stream(stream), ids, n_ent, cnt);
    return check_last_cuda_error(ctx, "tk_q36_route_count");
    TK_TRY_END(ctx);
}

tk_status tk_q36_route_scan(tk_ctx* ctx, const int32_t* cnt, int32_t n_expert, int32_t* bounds, int32_t* cur,
                            int32_t* max_rows, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_route_scan(as_cuda_stream(stream), cnt, n_expert, bounds, cur, max_rows);
    return check_last_cuda_error(ctx, "tk_q36_route_scan");
    TK_TRY_END(ctx);
}

tk_status tk_q36_route_scatter(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t k, int32_t* cur, int32_t* dst,
                               int32_t* src, int32_t* inv, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_route_scatter(as_cuda_stream(stream), ids, n_ent, k, cur, dst, src, inv);
    return check_last_cuda_error(ctx, "tk_q36_route_scatter");
    TK_TRY_END(ctx);
}

tk_status tk_q36_argmax(tk_ctx* ctx, int32_t grid_x, int32_t grid_y, const float* x, int32_t n, int32_t* out, float* pv,
                        int32_t* pi, bool stage2, const int32_t* map, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_argmax(dim3((unsigned) grid_x, (unsigned) grid_y), as_cuda_stream(stream), x, n, out, pv, pi,
                              stage2, map);
    return check_last_cuda_error(ctx, "tk_q36_argmax");
    TK_TRY_END(ctx);
}

tk_status tk_q36_argmax_samp(tk_ctx* ctx, const float* x, int32_t n, float* pv, int32_t* pi,
                             const tk_q36_draft_samp* ds_dev, int32_t step, const float* thr,
                             tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    static_assert(sizeof(tk_q36_draft_samp) == sizeof(talus::q36::DraftSamp), "DraftSamp ABI size mismatch");
    talus::q36::launch_argmax_samp(as_cuda_stream(stream), x, n, pv, pi,
                                   reinterpret_cast<const talus::q36::DraftSamp*>(ds_dev), step, thr);
    return check_last_cuda_error(ctx, "tk_q36_argmax_samp");
    TK_TRY_END(ctx);
}

tk_status tk_q36_lse(tk_ctx* ctx, const float* x, int32_t n, const tk_q36_draft_samp* ds_dev, float* out,
                     float* thr_out, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_lse(as_cuda_stream(stream), x, n, reinterpret_cast<const talus::q36::DraftSamp*>(ds_dev), out,
                           thr_out);
    return check_last_cuda_error(ctx, "tk_q36_lse");
    TK_TRY_END(ctx);
}

tk_status tk_q36_ctl_end(tk_ctx* ctx, int32_t* ctl, bool flip, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_ctl_end(as_cuda_stream(stream), ctl, flip);
    return check_last_cuda_error(ctx, "tk_q36_ctl_end");
    TK_TRY_END(ctx);
}

tk_status tk_q36_round_prep(tk_ctx* ctx, const int32_t* rctl, int32_t* tok, int32_t* tok_m, int32_t* pos0,
                            int32_t* pos0_m, int32_t* gctl, int32_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_round_prep(as_cuda_stream(stream), rctl, tok, tok_m, pos0, pos0_m, gctl, n);
    return check_last_cuda_error(ctx, "tk_q36_round_prep");
    TK_TRY_END(ctx);
}

tk_status tk_q36_row_copy(tk_ctx* ctx, float* dst, const float* src, const int32_t* idx, int32_t n,
                          tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_row_copy(as_cuda_stream(stream), dst, src, idx, n);
    return check_last_cuda_error(ctx, "tk_q36_row_copy");
    TK_TRY_END(ctx);
}

tk_status tk_q36_chain_prep(tk_ctx* ctx, const int32_t* tok, int32_t* tok_m, const int32_t* pos0, int32_t* pos0_m,
                            int32_t k, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_chain_prep(as_cuda_stream(stream), tok, tok_m, pos0, pos0_m, k);
    return check_last_cuda_error(ctx, "tk_q36_chain_prep");
    TK_TRY_END(ctx);
}

tk_status tk_q36_sgemm_nt(tk_ctx* ctx, int32_t tm, int32_t grid_x, int32_t grid_y, int32_t grid_z, const float* w,
                          const float* x, float* out, int32_t k_dim, int32_t n_out, int32_t n_tok, int32_t ldo,
                          int32_t kz, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_sgemm_nt(tm, dim3((unsigned) grid_x, (unsigned) grid_y, (unsigned) grid_z),
                                as_cuda_stream(stream), w, x, out, k_dim, n_out, n_tok, ldo, kz);
    return check_last_cuda_error(ctx, "tk_q36_sgemm_nt");
    TK_TRY_END(ctx);
}

tk_status tk_q36_sum_splits(tk_ctx* ctx, const float* part, float* out, int64_t n, int32_t splits,
                            tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_sum_splits(as_cuda_stream(stream), part, out, n, splits);
    return check_last_cuda_error(ctx, "tk_q36_sum_splits");
    TK_TRY_END(ctx);
}

tk_status tk_q36_to_f16(tk_ctx* ctx, const float* x, void* y_f16, int64_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_to_f16(as_cuda_stream(stream), x, static_cast<__half*>(y_f16), n);
    return check_last_cuda_error(ctx, "tk_q36_to_f16");
    TK_TRY_END(ctx);
}

tk_status tk_q36_conv_seq(tk_ctx* ctx, float* hist, const float* qkv, const float* w, float* h, int32_t c_channels,
                          int32_t qk_heads, float eps, int32_t n_tok, bool write_state, const tk_q36_gdn_pend* pd,
                          tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_conv_seq(as_cuda_stream(stream), hist, qkv, w, h, c_channels, qk_heads, eps, n_tok, write_state,
                                to_gdn_pend(pd));
    return check_last_cuda_error(ctx, "tk_q36_conv_seq");
    TK_TRY_END(ctx);
}

tk_status tk_q36_conv_ab(tk_ctx* ctx, float* hist, const float* qkv, const float* w, float* hc, int32_t c_channels,
                         int32_t qk_heads, float eps, int32_t n_tok, bool write_state, const tk_q36_gdn_pend* pd,
                         const float* wa, const float* wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int32_t n_in, int32_t hv, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_conv_ab(as_cuda_stream(stream), hist, qkv, w, hc, c_channels, qk_heads, eps, n_tok, write_state,
                               to_gdn_pend(pd), wa, wb, x, dt, ssm_a, alpha, beta, n_in, hv);
    return check_last_cuda_error(ctx, "tk_q36_conv_ab");
    TK_TRY_END(ctx);
}

tk_status tk_q36_conv_par(tk_ctx* ctx, const float* hist, const float* qkv, const float* w, float* h,
                          int32_t c_channels, int32_t qk_heads, float eps, int32_t n_tok,
                          tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_conv_par(as_cuda_stream(stream), hist, qkv, w, h, c_channels, qk_heads, eps, n_tok);
    return check_last_cuda_error(ctx, "tk_q36_conv_par");
    TK_TRY_END(ctx);
}

tk_status tk_q36_conv_hist(tk_ctx* ctx, float* hist, const float* qkv, int32_t c_channels, int32_t n_tok,
                           tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_conv_hist(as_cuda_stream(stream), hist, qkv, c_channels, n_tok);
    return check_last_cuda_error(ctx, "tk_q36_conv_hist");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_seq(tk_ctx* ctx, float* state, const float* h, int32_t c_channels, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int32_t h_k,
                         int32_t h_v, int32_t n_tok, bool write_state, const tk_q36_gdn_pend* pd, void* yq,
                         tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_seq(as_cuda_stream(stream), state, h, c_channels, gate, beta, z, gamma, eps, y, h_k, h_v,
                               n_tok, write_state, to_gdn_pend(pd), yq);
    return check_last_cuda_error(ctx, "tk_q36_gdn_seq");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_seq_cols(tk_ctx* ctx, float* state, const float* h, int32_t c_channels, const float* gate,
                              const float* beta, float* o_raw, int32_t h_k, int32_t h_v, int32_t n_tok,
                              tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_seq_cols(as_cuda_stream(stream), state, h, c_channels, gate, beta, o_raw, h_k, h_v, n_tok);
    return check_last_cuda_error(ctx, "tk_q36_gdn_seq_cols");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_seq_warp(tk_ctx* ctx, int32_t ncol, float* state, const float* h, int32_t c_channels,
                              const float* gate, const float* beta, float* o_raw, int32_t h_k, int32_t h_v,
                              int32_t n_tok, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_seq_warp(ncol, as_cuda_stream(stream), state, h, c_channels, gate, beta, o_raw, h_k, h_v,
                                    n_tok);
    return check_last_cuda_error(ctx, "tk_q36_gdn_seq_warp");
    TK_TRY_END(ctx);
}

tk_status tk_q36_gdn_norm(tk_ctx* ctx, int32_t n_tok, const float* o_raw, const float* z, const float* gamma, float eps,
                          float* y, int32_t h_v, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_gdn_norm(n_tok, as_cuda_stream(stream), o_raw, z, gamma, eps, y, h_v);
    return check_last_cuda_error(ctx, "tk_q36_gdn_norm");
    TK_TRY_END(ctx);
}

tk_status tk_q36_flush_read(tk_ctx* ctx, const void* p, size_t n, float* out, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::launch_flush_read(as_cuda_stream(stream), static_cast<const float4*>(p), n, out);
    return check_last_cuda_error(ctx, "tk_q36_flush_read");
    TK_TRY_END(ctx);
}

tk_status tk_q36_draft_conf(tk_ctx* ctx, const float* x, int32_t n, const float* tau, const float* lmin,
                            const int32_t* pick, const int32_t* map, float* conf, float* top, int32_t step,
                            const float* lse, const float* thr, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::draft_conf(x, n, tau, lmin, pick, map, conf, top, step, stream, lse, thr);
    return check_last_cuda_error(ctx, "tk_q36_draft_conf");
    TK_TRY_END(ctx);
}

tk_status tk_q36_draft_support(tk_ctx* ctx, const float* x, int32_t n, const float* tau, const float* lmin,
                               const float* top_p, const int32_t* top_k, float* lse, float* thr,
                               tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::draft_support(x, n, tau, lmin, top_p, top_k, lse, thr, stream);
    return check_last_cuda_error(ctx, "tk_q36_draft_support");
    TK_TRY_END(ctx);
}

int32_t tk_q36_dflash_attn_chunks(int32_t window) TK_NOEXCEPT {
    return talus::q36::dflash::attn_chunks(window);
}

size_t tk_q36_dflash_attn_scratch_floats(int32_t heads, int32_t hd, int32_t window) TK_NOEXCEPT {
    return talus::q36::dflash::attn_scratch_floats(heads, hd, window);
}

tk_status tk_q36_dflash_rms_rows(tk_ctx* ctx, const float* x, const float* w, float* out, int32_t n, int32_t rows,
                                 float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::rms_rows(x, w, out, n, rows, eps, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_rms_rows");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_rms_rows_q8(tk_ctx* ctx, const float* x, const float* w, float* out, void* q8, int32_t n,
                                    int32_t rows, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::rms_rows(x, w, out, n, rows, eps, stream, q8);
    return check_last_cuda_error(ctx, "tk_q36_dflash_rms_rows_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_sum5_rms(tk_ctx* ctx, const float* const* ys, const float* w, float* out, int32_t n,
                                 int32_t rows, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::sum5_rms(ys, w, out, n, rows, eps, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_sum5_rms");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_sum5_rms_q8(tk_ctx* ctx, const float* const* ys, const float* w, float* out, void* q8,
                                    int32_t n, int32_t rows, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::sum5_rms(ys, w, out, n, rows, eps, stream, q8);
    return check_last_cuda_error(ctx, "tk_q36_dflash_sum5_rms_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_dyn_conv(tk_ctx* ctx, const float* x_in, const float* dyn, const float* base, float* out,
                                 int32_t n, int32_t rows, int32_t group, int32_t side,
                                 tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::dyn_conv(x_in, dyn, base, out, n, rows, group, side, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_dyn_conv");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_dyn_conv_q8(tk_ctx* ctx, const float* x_in, const float* dyn, const float* base, float* out,
                                    void* q8, int32_t n, int32_t rows, int32_t group, int32_t side,
                                    tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::dyn_conv(x_in, dyn, base, out, n, rows, group, side, stream, q8);
    return check_last_cuda_error(ctx, "tk_q36_dflash_dyn_conv_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_qk_norm_rope(tk_ctx* ctx, float* x, const float* w, int32_t heads, int32_t hd, int32_t rows,
                                     const int32_t* pos, float theta, float eps, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::qk_norm_rope(x, w, heads, hd, rows, pos, theta, eps, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_qk_norm_rope");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_kv_store(tk_ctx* ctx, const float* k, const float* v, float* kring, float* vring, int32_t width,
                                 int32_t rows, const int32_t* pos, int32_t ring, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::kv_store(k, v, kring, vring, width, rows, pos, ring, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_kv_store");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_kv_store_f16(tk_ctx* ctx, const float* k, const float* v, void* kring_f16, void* vring_f16,
                                     int32_t k_width, int32_t v_width, int32_t rows, const int32_t* pos, int32_t ring,
                                     tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::kv_store_f16(k, v, kring_f16, vring_f16, k_width, v_width, rows, pos, ring, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_kv_store_f16");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_block_attn(tk_ctx* ctx, const float* q, const float* kring, const float* vring,
                                   const int32_t* pos, int32_t heads, int32_t heads_kv, int32_t hd, int32_t ring,
                                   int32_t window, float scale, float* part, float* out,
                                   tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::block_attn(q, kring, vring, pos, heads, heads_kv, hd, ring, window, scale, part, out,
                                   stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_block_attn");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_swiglu(tk_ctx* ctx, const float* g, const float* u, float* out, int64_t n,
                               tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::swiglu(g, u, out, n, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_swiglu");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_swiglu_q8(tk_ctx* ctx, const float* g, const float* u, float* out, void* q8, int64_t n,
                                  tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::swiglu(g, u, out, n, stream, q8);
    return check_last_cuda_error(ctx, "tk_q36_dflash_swiglu_q8");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_add_inplace(tk_ctx* ctx, float* x, const float* y, int64_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::add_inplace(x, y, n, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_add_inplace");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_block_tokens(tk_ctx* ctx, const int32_t* anchor, int32_t* out, int32_t n, int32_t mask,
                                     tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::block_tokens(anchor, out, n, mask, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_block_tokens");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_set_i32(tk_ctx* ctx, int32_t* p, int32_t v, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::set_i32(p, v, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_set_i32");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_top16(tk_ctx* ctx, const float* logits, int32_t nv, int32_t rows, const int32_t* map,
                              int32_t* cand, float* unary, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::top16(logits, nv, rows, map, cand, unary, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_top16");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_select_chain(tk_ctx* ctx, const int32_t* cand, const float* unary, const float* gate,
                                     const void* pred, const void* succ, const int32_t* anchor, int32_t* out,
                                     float* score, int32_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::select_chain(cand, unary, gate, pred, succ, anchor, out, score, n, stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_select_chain");
    TK_TRY_END(ctx);
}

tk_status tk_q36_dflash_select_chain_q(tk_ctx* ctx, const int32_t* cand, const float* unary, const float* gate,
                                       const void* pred, const void* succ, const int32_t* anchor, int32_t* out,
                                       float* score, float* q, const float* tau, const uint32_t* seed,
                                       const uint32_t* ctr, int32_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    talus::q36::dflash::select_chain_q(cand, unary, gate, pred, succ, anchor, out, score, q, tau, seed, ctr, n,
                                       stream);
    return check_last_cuda_error(ctx, "tk_q36_dflash_select_chain_q");
    TK_TRY_END(ctx);
}

bool tk_q36_mmvq_supported(int32_t ggml_type) TK_NOEXCEPT {
    return strata::kernels::native_mmvq_supported(ggml_type);
}

bool tk_q36_q8_0_pack_should(int32_t ggml_type, int32_t n_in, int32_t n_out) TK_NOEXCEPT {
    ensure_q36_env();
    return ggml_type == 8 &&
           strata::kernels::native_q8_0_packed_enabled() &&
           strata::kernels::native_q8_0_packed_eligible(n_in, n_out) &&
           !strata::kernels::native_mmvq_w32_active();
}

void tk_q36_q8_0_pack_host(const void* gguf_blocks, void* out, int32_t n_in, int32_t n_out) TK_NOEXCEPT {
    strata::kernels::native_q8_0_pack_host(gguf_blocks, out, n_in, n_out);
}

void tk_q36_q8_0_packed_register(const void* gguf_weights, const void* packed, int32_t n_in, int32_t n_out) TK_NOEXCEPT {
    strata::kernels::native_q8_0_packed_register(gguf_weights, packed, n_in, n_out);
}

void tk_q36_q8_0_packed_unregister(const void* gguf_weights) TK_NOEXCEPT {
    strata::kernels::native_q8_0_packed_unregister(gguf_weights);
}

tk_status tk_q36_embed_rows(tk_ctx* ctx, int32_t ggml_type, const void* table, size_t row_bytes,
                            const int32_t* tok_dev, int64_t n_tok, int64_t n_embd, float* out,
                            tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    strata::kernels::iq_embed_rows(ggml_type, table, row_bytes, tok_dev, n_tok, n_embd, out, stream);
    return check_last_cuda_error(ctx, "tk_q36_embed_rows");
    TK_TRY_END(ctx);
}

bool tk_q36_mmq_built(void) TK_NOEXCEPT {
    ensure_q36_env();
    return strata::prefill::mmq::built();
}

bool tk_q36_mmq_supported_fits(int32_t ggml_type, int64_t w_rows) TK_NOEXCEPT {
    ensure_q36_env();
    return strata::prefill::mmq::supported(ggml_type) && strata::prefill::mmq::fits(ggml_type, w_rows);
}

int32_t tk_q36_mmq_q8_layout(int32_t ggml_type) TK_NOEXCEPT {
    return strata::prefill::mmq::q8_layout(ggml_type);
}

size_t tk_q36_mmq_q8_bytes(int64_t rows, int64_t cols) TK_NOEXCEPT {
    return strata::prefill::mmq::q8_bytes(rows, cols);
}

size_t tk_q36_mmq_matrix_bytes(int32_t ggml_type, int64_t rows, int64_t cols) TK_NOEXCEPT {
    return strata::prefill::mmq::matrix_bytes(ggml_type, rows, cols);
}

tk_status tk_q36_mmq_iota(tk_ctx* ctx, int32_t* dst, int64_t n, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    strata::prefill::mmq::iota(dst, n, stream);
    return check_last_cuda_error(ctx, "tk_q36_mmq_iota");
    TK_TRY_END(ctx);
}

tk_status tk_q36_mmq_set_bounds(tk_ctx* ctx, int32_t* bounds, int32_t rows, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    strata::prefill::mmq::set_bounds(bounds, rows, stream);
    return check_last_cuda_error(ctx, "tk_q36_mmq_set_bounds");
    TK_TRY_END(ctx);
}

extern "C" __attribute__((weak)) void tk_register_mmq_q8_buf(const void* xq) noexcept;

tk_status tk_q36_mmq_quantize(tk_ctx* ctx, const float* x, const int32_t* ids, void* xq, int32_t ggml_type,
                              int64_t cols, int64_t ld, int64_t rows, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    if (tk_register_mmq_q8_buf != nullptr) {
        tk_register_mmq_q8_buf(xq);
    }
    strata::prefill::mmq::quantize(x, ids, xq, ggml_type, cols, ld, rows, stream);
    return check_last_cuda_error(ctx, "tk_q36_mmq_quantize");
    TK_TRY_END(ctx);
}

tk_status tk_q36_mmq_quantize_scatter(tk_ctx* ctx, const float* x, const int32_t* smap, void* xq, int32_t ggml_type,
                                      int64_t cols, int64_t n_tok, int64_t n_rows, int32_t k, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    if (tk_register_mmq_q8_buf != nullptr) {
        tk_register_mmq_q8_buf(xq);
    }
    strata::prefill::mmq::quantize_scatter(x, smap, xq, ggml_type, cols, n_tok, n_rows, k, stream);
    return check_last_cuda_error(ctx, "tk_q36_mmq_quantize_scatter");
    TK_TRY_END(ctx);
}

tk_status tk_q36_mmq_swiglu(tk_ctx* ctx, const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved,
                            tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    strata::prefill::mmq::swiglu(gu, h, rows, n_ff, interleaved, stream);
    return check_last_cuda_error(ctx, "tk_q36_mmq_swiglu");
    TK_TRY_END(ctx);
}

bool tk_q36_fattn_supported(int32_t head_dim) TK_NOEXCEPT {
    return strata::prefill::fattn::supported(head_dim);
}

size_t tk_q36_fattn_mask_bytes(int64_t n_tok, int64_t n_kv) TK_NOEXCEPT {
    return strata::prefill::fattn::mask_bytes(n_tok, n_kv);
}

tk_status tk_q36_router_topk(tk_ctx* ctx, const float* logits, int32_t* ids, float* weights, int32_t n_expert,
                             int32_t k, int32_t n_tok, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    strata::kernels::native_router_topk(logits, ids, weights, n_expert, k, n_tok, stream);
    return check_last_cuda_error(ctx, "tk_q36_router_topk");
    TK_TRY_END(ctx);
}

bool tk_q36_expert_supported(int32_t gu_type, int32_t d_type, int64_t n_embd, int64_t n_ff) TK_NOEXCEPT {
    return strata::kernels::native_expert_supported(gu_type, d_type, n_embd, n_ff);
}

void tk_q36_expert_layout(int32_t gu_type, int32_t d_type, int64_t n_embd, int64_t n_ff, size_t* out_gu_row,
                          size_t* out_d_row, size_t* out_up_off, size_t* out_down_off, size_t* out_bytes,
                          size_t* out_tail_off) TK_NOEXCEPT {
    const auto L = strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    if (out_gu_row) *out_gu_row = L.gu_row;
    if (out_d_row) *out_d_row = L.d_row;
    if (out_up_off) *out_up_off = L.up_off;
    if (out_down_off) *out_down_off = L.down_off;
    if (out_bytes) *out_bytes = L.bytes;
    if (out_tail_off) *out_tail_off = L.tail_off;
}

size_t tk_q36_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff) TK_NOEXCEPT {
    return strata::kernels::native_expert_scratch_bytes(cap_entries, n_ff);
}

tk_status tk_q36_expert_grouped(tk_ctx* ctx, int32_t gu_type, int32_t d_type, int64_t n_embd, int64_t n_ff,
                                const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                                const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                                int64_t cap_entries, const void* x_q8_1, void* scratch, float* out,
                                int64_t grid_groups, tk_stream stream) TK_NOEXCEPT {
    TK_TRY_BEGIN
    const auto L = strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    strata::kernels::native_expert_grouped(
        L, reinterpret_cast<const unsigned long long*>(grp_ptr), grp_start, n_groups, ent_dst, ent_tok,
        cap_groups, cap_entries, x_q8_1, scratch, out, stream, grid_groups);
    return check_last_cuda_error(ctx, "tk_q36_expert_grouped");
    TK_TRY_END(ctx);
}

}   // extern "C"
