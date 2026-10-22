// src/qwen35/q36_dflash.cu - Talus (talus-next, lever 6): the DFlash2 drafter's kernels (see q36_dflash.hpp).
#include "q36_dflash.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/qwen35/q36_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>

namespace talus::q36::dflash {
namespace {

constexpr int kRows = 8;      // the block (anchor + 7 masks)
constexpr int kChunk = 128;   // keys per attention block
constexpr int kTile = 32;     // keys per LDS tile

__device__ __forceinline__ void dfa_q8_put(float xi, void* q8, size_t i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    TqQ81* b = (TqQ81*) q8 + i / 32;
    b->qs[i % 32] = strata::kernels::q8_1_quant(xi, d, amax);
    if ((i & 31) == 0) b->ds = strata::kernels::q8_1_ds(d, sum);
}

__device__ __forceinline__ float block_sum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    const int w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[w] = v;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < nw; ++k) t += sh[k];
    return t;
}

__global__ void k_rms_rows(const float* x, const float* w, float* out, int n, float eps, void* q8 = nullptr) {
    __shared__ float sh[32];
    const size_t o0 = (size_t) blockIdx.x * n;
    const float* xr = x + o0;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / n + eps);
    float* o = out + o0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = xr[i] * r * w[i];
        o[i] = xi;
        if (q8) dfa_q8_put(xi, q8, o0 + i);
    }
}

struct Ptr5 { const float* p[5]; };
__global__ void k_sum5_rms(Ptr5 ys, const float* w, float* out, int n, float eps, void* q8 = nullptr) {
    __shared__ float sh[32];
    const size_t o0 = (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = ys.p[0][o0 + i] + ys.p[1][o0 + i] + ys.p[2][o0 + i] + ys.p[3][o0 + i] + ys.p[4][o0 + i];
        out[o0 + i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = out[o0 + i] * r * w[i];
        out[o0 + i] = xi;
        if (q8) dfa_q8_put(xi, q8, o0 + i);
    }
}

__global__ void k_dyn_conv(const float* in, const float* dyn, const float* base, float* out, int n, int group, int side,
                           void* q8 = nullptr) {
    const int i = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const int ng = n / group, g = c / group;
    const float* d = dyn + (size_t) i * (4 * ng) + side * 2 * ng;
    const float* b = base + (size_t) side * 2 * n;
    float v = (b[c] + d[g]) * in[(size_t) i * n + c];
    if (i > 0) v += (b[n + c] + d[ng + g]) * in[(size_t) (i - 1) * n + c];
    const size_t idx = (size_t) i * n + c;
    out[idx] = v;
    if (q8) dfa_q8_put(v, q8, idx);
}

/// one block per (head, row), hd threads
__global__ void k_qk_norm_rope(float* x, const float* w, int heads, int hd, const int32_t* pos, float theta, float eps) {
    __shared__ float sh[32];
    __shared__ float v[256];
    const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    float* xr = x + ((size_t) r * heads + h) * hd;
    const float a = xr[d];
    const float s = block_sum(a * a, sh);
    v[d] = a * rsqrtf(s / hd + eps) * w[d];
    __syncthreads();
    const int half = hd / 2, j = d < half ? d : d - half;
    const float p = (float) (*pos + r);
    const float th = p * powf(theta, -2.0f * (float) j / (float) hd);
    const float cs = cosf(th), sn = sinf(th);
    const float x0 = v[j], x1 = v[j + half];
    xr[d] = d < half ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
}

__global__ void k_kv_store(const float* k, const float* v, float* kring, float* vring, int width, const int32_t* pos, int ring) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const size_t cell = (size_t) ((*pos + r) % ring);
    kring[cell * width + i] = k[(size_t) r * width + i];
    vring[cell * width + i] = v[(size_t) r * width + i];
}

__global__ void k_kv_store_f16(const float* k, const float* v, __half* kring, __half* vring,
                               int k_width, int v_width, const int32_t* pos, int ring) {
    const int r = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t cell = (size_t) ((*pos + r) % ring);
    if (i < k_width) kring[cell * k_width + i] = __float2half(k[(size_t) r * k_width + i]);
    if (i < v_width) vring[cell * v_width + i] = __float2half(v[(size_t) r * v_width + i]);
}

/// grid (heads_kv, chunks), 256 threads; 32 query vectors per KV head (8 rows x 4 heads: gq must be 4, hd 128)
__global__ void __launch_bounds__(256) k_dfa_part(const float* q, const float* kring, const float* vring, const int32_t* pos,
                                                  int heads, int heads_kv, int ring, int window, float scale, float* part) {
    constexpr int HD = 128, QV = 32;
    __shared__ float sq[QV][HD];
    __shared__ float sk[kTile][HD + 1];
    __shared__ float ss[QV][kChunk];
    const int g = blockIdx.x, c = blockIdx.y, tid = threadIdx.x, gq = heads / heads_kv, nch = gridDim.y;
    const int P = *pos, lo = max(0, P - window + 1), hi = P + kRows - 1, k0 = lo + c * kChunk;
    float* acc = part + ((size_t) c * heads_kv + g) * QV * HD;
    float* pm = part + (size_t) nch * heads_kv * QV * HD + ((size_t) c * heads_kv + g) * QV;
    float* pl = pm + (size_t) nch * heads_kv * QV;
    if (k0 > hi) {
        if (tid < QV) { pm[tid] = -INFINITY; pl[tid] = 0.f; }
        return;
    }
    for (int idx = tid; idx < QV * HD; idx += 256) {
        const int qi = idx / HD, d = idx % HD, i = qi / gq, j = qi % gq;
        sq[qi][d] = q[((size_t) i * heads + g * gq + j) * HD + d] * scale;
    }
    for (int t = 0; t < kChunk / kTile; ++t) {
        __syncthreads();
        for (int idx = tid; idx < kTile * HD; idx += 256) {
            const int kk = idx / HD, d = idx % HD, kp = k0 + t * kTile + kk;
            sk[kk][d] = kp <= hi ? kring[((size_t) (kp % ring) * heads_kv + g) * HD + d] : 0.f;
        }
        __syncthreads();
        const int qi = tid >> 3, i = qi / gq;
        for (int m = 0; m < kTile / 8; ++m) {
            const int kk = (tid & 7) + 8 * m, kp = k0 + t * kTile + kk;
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < HD; ++d) dot += sq[qi][d] * sk[kk][d];
            const bool ok = kp <= hi && kp >= P + i - window + 1;
            ss[qi][t * kTile + kk] = ok ? dot : -INFINITY;
        }
    }
    __syncthreads();
    {   // per query vector: max and sum over the chunk (8 lanes per vector)
        const int qi = tid >> 3, l8 = tid & 7;
        float mx = -INFINITY;
        for (int k = l8; k < kChunk; k += 8) mx = fmaxf(mx, ss[qi][k]);
        for (int o = 4; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
        float sum = 0.f;
        for (int k = l8; k < kChunk; k += 8) {
            const float p = mx == -INFINITY ? 0.f : expf(ss[qi][k] - mx);
            ss[qi][k] = p;
            sum += p;
        }
        for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
        if (l8 == 0) { pm[qi] = mx; pl[qi] = sum; }
    }
    const int d = tid % HD, qh = tid / HD;
    float a[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) a[r] = 0.f;
    for (int t = 0; t < kChunk / kTile; ++t) {
        __syncthreads();
        for (int idx = tid; idx < kTile * HD; idx += 256) {
            const int kk = idx / HD, dd = idx % HD, kp = k0 + t * kTile + kk;
            sk[kk][dd] = kp <= hi ? vring[((size_t) (kp % ring) * heads_kv + g) * HD + dd] : 0.f;
        }
        __syncthreads();
        for (int kk = 0; kk < kTile; ++kk) {
            const float v = sk[kk][d];
#pragma unroll
            for (int r = 0; r < 16; ++r) a[r] += ss[qh * 16 + r][t * kTile + kk] * v;
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[(size_t) (qh * 16 + r) * HD + d] = a[r];
}

/// grid (heads, 8 rows), hd threads: the chunks' partials merged
__global__ void k_dfa_merge(const float* part, int heads, int heads_kv, int nch, float* out) {
    constexpr int HD = 128, QV = 32;
    const int h = blockIdx.x, i = blockIdx.y, d = threadIdx.x, gq = heads / heads_kv, g = h / gq, qi = i * gq + h % gq;
    const float* pm = part + (size_t) nch * heads_kv * QV * HD;
    const float* pl = pm + (size_t) nch * heads_kv * QV;
    float M = -INFINITY;
    for (int c = 0; c < nch; ++c) M = fmaxf(M, pm[((size_t) c * heads_kv + g) * QV + qi]);
    float L = 0.f, o = 0.f;
    for (int c = 0; c < nch; ++c) {
        const size_t mi = ((size_t) c * heads_kv + g) * QV + qi;
        if (pm[mi] == -INFINITY) continue;
        const float e = expf(pm[mi] - M);
        L += pl[mi] * e;
        o += part[(((size_t) c * heads_kv + g) * QV + qi) * HD + d] * e;
    }
    out[((size_t) i * heads + h) * HD + d] = L > 0.f ? o / L : 0.f;
}

__global__ void k_swiglu(const float* g, const float* u, float* out, int64_t n, void* q8 = nullptr) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = g[i];
        const float xi = x / (1.f + expf(-x)) * u[i];
        out[i] = xi;
        if (q8) dfa_q8_put(xi, q8, (size_t) i);
    }
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_block_tokens(const int32_t* anchor, int32_t* out, int n, int32_t mask) {
    const int i = threadIdx.x;
    if (i < n) out[i] = i == 0 ? *anchor : mask;
}
__global__ void k_set(int32_t* p, int32_t v) { *p = v; }

/// grid rows - 1 (row = blockIdx.x + 1), 1024 threads: 16 rounds of a block argmax under the order (value desc,
/// index asc), each strictly after the last pick
__global__ void __launch_bounds__(1024) k_top16(const float* logits, int nv, const int32_t* map, int32_t* cand, float* unary) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ float pv;
    __shared__ int pi;
    const int r = blockIdx.x + 1;
    const float* l = logits + (size_t) r * nv;
    if (threadIdx.x == 0) { pv = INFINITY; pi = -1; }
    for (int k = 0; k < 16; ++k) {
        __syncthreads();
        const float lv = pv;
        const int li = pi;
        float v = -INFINITY;
        int id = 0x7fffffff;
        for (int i = threadIdx.x; i < nv; i += blockDim.x) {
            const float x = l[i];
            if (!(x < lv || (x == lv && i > li))) continue;
            if (x > v || (x == v && i < id)) { v = x; id = i; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o, 32);
            const int oi = __shfl_xor(id, o, 32);
            if (ov > v || (ov == v && oi < id)) { v = ov; id = oi; }
        }
        if ((threadIdx.x & 31) == 0) { bv[threadIdx.x >> 5] = v; bi[threadIdx.x >> 5] = id; }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int w = 1; w < (int) (blockDim.x >> 5); ++w)
                if (bv[w] > bvv || (bv[w] == bvv && bi[w] < bii)) { bvv = bv[w]; bii = bi[w]; }
            pv = bvv;
            pi = bii;
            cand[r * 16 + k] = bii < nv ? (map ? map[bii] : bii) : 0;
            unary[r * 16 + k] = bvv;
        }
    }
}

/// value r (0 .. 255) of a 256-wide Q4_K row (one block: d, dmin, scales[12], qs[128])
__device__ __forceinline__ float q4k_at(const uint8_t* b, int r) {
    __half hd, hm;
    memcpy(&hd, b, 2);
    memcpy(&hm, b + 2, 2);
    const float d = __half2float(hd), dmin = __half2float(hm);
    const uint8_t* sc = b + 4;
    const uint8_t* qs = b + 16;
    const int g64 = r >> 6, w = r & 63, hi = w >= 32, l = w & 31, j = 2 * g64 + hi;
    int s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    const int q = hi ? qs[g64 * 32 + l] >> 4 : qs[g64 * 32 + l] & 0xF;
    return d * (float) s * (float) q - dmin * (float) m;
}

__device__ __forceinline__ float u01_hash(uint64_t k) {
    k += 0x9E3779B97F4A7C15ull;
    k = (k ^ (k >> 30)) * 0xBF58476D1CE4E5B9ull;
    k = (k ^ (k >> 27)) * 0x94D049BB133111EBull;
    k ^= k >> 31;
    return ((float) (uint32_t) (k >> 40) + 0.5f) * (1.0f / 16777216.0f);
}

/// one block of 256 (8 waves of 32)
__global__ void __launch_bounds__(256) k_select_chain(const int32_t* cand, const float* unary, const float* gate,
                                                      const uint8_t* pred, const uint8_t* succ, const int32_t* anchor,
                                                      int32_t* out, float* score, int n, float* qout = nullptr,
                                                      const float* tau_p = nullptr, const uint32_t* seed_p = nullptr,
                                                      const uint32_t* ctr_p = nullptr) {
    constexpr size_t kRowB = 144;
    __shared__ float cond[256];
    __shared__ float sc[16];
    __shared__ int prev;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    if (tid == 0) prev = *anchor;
    for (int i = 1; i <= n; ++i) {
        __syncthreads();
        cond[tid] = q4k_at(pred + (size_t) prev * kRowB, tid) * gate[(size_t) i * 256 + tid];
        __syncthreads();
        for (int cc = 2 * w; cc < 2 * w + 2; ++cc) {
            const uint8_t* srow = succ + (size_t) cand[i * 16 + cc] * kRowB;
            float s = 0.f;
            for (int r = lane * 8; r < lane * 8 + 8; ++r) s += q4k_at(srow, r) * cond[r];
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
            if (lane == 0) sc[cc] = s + unary[i * 16 + cc];
        }
        __syncthreads();
        if (tid == 0) {
            const float tau = tau_p ? *tau_p : 0.f;
            const float te = tau > 0.f ? tau : 1.f;
            int b = 0;
            for (int j = 1; j < 16; ++j)
                if (sc[j] > sc[b]) b = j;
            float z = 0.f;
            for (int j = 0; j < 16; ++j) z += expf((sc[j] - sc[b]) / te);
            if (score) score[i - 1] = 1.f / z;   // max_j q_i(j)
            if (qout)
                for (int j = 0; j < 16; ++j) qout[(i - 1) * 16 + j] = expf((sc[j] - sc[b]) / te) / z;
            if (tau > 0.f) {   // a draw from q_i: Gumbel-max over score / tau
                const uint64_t base = ((uint64_t) *seed_p << 32) ^ ((uint64_t) *ctr_p * 64 + 32 + (uint64_t) i) * 0xD6E8FEB86659FD93ull;
                float bv = -INFINITY;
                for (int j = 0; j < 16; ++j) {
                    const float v = sc[j] / tau - logf(-logf(u01_hash(base + (uint64_t) j * 0x9E3779B97F4A7C15ull)));
                    if (v > bv) { bv = v; b = j; }
                }
            }
            out[i - 1] = cand[i * 16 + b];
            prev = cand[i * 16 + b];
        }
    }
}

}  // namespace

void rms_rows(const float* x, const float* w, float* out, int n, int rows, float eps, void* stream, void* q8) {
    k_rms_rows<<<rows, 256, 0, (cudaStream_t) stream>>>(x, w, out, n, eps, q8);
}
void sum5_rms(const float* const* ys, const float* w, float* out, int n, int rows, float eps, void* stream, void* q8) {
    Ptr5 p;
    for (int k = 0; k < 5; ++k) p.p[k] = ys[k];
    k_sum5_rms<<<rows, 256, 0, (cudaStream_t) stream>>>(p, w, out, n, eps, q8);
}
void dyn_conv(const float* in, const float* dyn, const float* base, float* out, int n, int rows, int group, int side,
              void* stream, void* q8) {
    k_dyn_conv<<<dim3((n + 255) / 256, rows), 256, 0, (cudaStream_t) stream>>>(in, dyn, base, out, n, group, side, q8);
}
void qk_norm_rope(float* x, const float* w, int heads, int hd, int rows, const int32_t* pos, float theta, float eps,
                  void* stream) {
    k_qk_norm_rope<<<dim3(heads, rows), hd, 0, (cudaStream_t) stream>>>(x, w, heads, hd, pos, theta, eps);
}
void kv_store(const float* k, const float* v, float* kring, float* vring, int width, int rows, const int32_t* pos,
              int ring, void* stream) {
    k_kv_store<<<dim3((width + 255) / 256, rows), 256, 0, (cudaStream_t) stream>>>(k, v, kring, vring, width, pos, ring);
}
void kv_store_f16(const float* k, const float* v, void* kring_f16, void* vring_f16, int k_width, int v_width,
                  int rows, const int32_t* pos, int ring, void* stream) {
    const int max_w = k_width > v_width ? k_width : v_width;
    k_kv_store_f16<<<dim3((max_w + 255) / 256, rows), 256, 0, (cudaStream_t) stream>>>(
        k, v, static_cast<__half*>(kring_f16), static_cast<__half*>(vring_f16), k_width, v_width, pos, ring);
}
int attn_chunks(int window) { return (window + kRows - 1 + kChunk - 1) / kChunk; }
size_t attn_scratch_floats(int heads, int hd, int window) {
    return (size_t) attn_chunks(window) * heads * kRows / 4 * 4 * (hd + 2) + 64;
}
void block_attn(const float* q, const float* kring, const float* vring, const int32_t* pos, int heads, int heads_kv,
                int hd, int ring, int window, float scale, float* part, float* out, void* stream) {
    const int nch = attn_chunks(window);
    k_dfa_part<<<dim3(heads_kv, nch), 256, 0, (cudaStream_t) stream>>>(q, kring, vring, pos, heads, heads_kv, ring, window,
                                                                      scale, part);
    k_dfa_merge<<<dim3(heads, kRows), hd, 0, (cudaStream_t) stream>>>(part, heads, heads_kv, nch, out);
}
void swiglu(const float* g, const float* u, float* out, int64_t n, void* stream, void* q8) {
    k_swiglu<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(g, u, out, n, q8);
}
void add_inplace(float* x, const float* y, int64_t n, void* stream) {
    k_add<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, y, n);
}
void block_tokens(const int32_t* anchor, int32_t* out, int n, int32_t mask, void* stream) {
    k_block_tokens<<<1, 32, 0, (cudaStream_t) stream>>>(anchor, out, n, mask);
}
void set_i32(int32_t* p, int32_t v, void* stream) { k_set<<<1, 1, 0, (cudaStream_t) stream>>>(p, v); }
void top16(const float* logits, int nv, int rows, const int32_t* map, int32_t* cand, float* unary, void* stream) {
    k_top16<<<rows - 1, 1024, 0, (cudaStream_t) stream>>>(logits, nv, map, cand, unary);
}
void select_chain(const int32_t* cand, const float* unary, const float* gate, const void* pred, const void* succ,
                  const int32_t* anchor, int32_t* out, float* score, int n, void* stream) {
    k_select_chain<<<1, 256, 0, (cudaStream_t) stream>>>(cand, unary, gate, (const uint8_t*) pred, (const uint8_t*) succ,
                                                        anchor, out, score, n);
}
void select_chain_q(const int32_t* cand, const float* unary, const float* gate, const void* pred, const void* succ,
                    const int32_t* anchor, int32_t* out, float* score, float* q, const float* tau, const uint32_t* seed,
                    const uint32_t* ctr, int n, void* stream) {
    k_select_chain<<<1, 256, 0, (cudaStream_t) stream>>>(cand, unary, gate, (const uint8_t*) pred, (const uint8_t*) succ,
                                                        anchor, out, score, n, q, tau, seed, ctr);
}

}  // namespace talus::q36::dflash
