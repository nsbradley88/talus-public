// src/qwen35/q36_adraft.cu - Talus (talus-next, lever 5): the draft-confidence kernel (see q36_adraft.hpp).
#include "q36_adraft.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace talus::q36 {
namespace {

/// One block of 1024: the row's max and log-sum-exp at T (tau, or 1 for greedy) over the min-p support (the same
/// float tests as k_lse: x / tau >= max + lmin), the drafted token's row (a scan of the map), conf and top.
__global__ void __launch_bounds__(1024) k_draft_conf(const float* x, int n, const float* tau_p, const float* lmin_p,
                                                     const int32_t* pick, const int32_t* map, float* conf, float* top,
                                                     int step, const float* lse_p, const float* thr_p) {
    __shared__ float sh[32];
    __shared__ int row;
    const float tau = *tau_p;
    const float te = tau > 0.f ? tau : 1.f;
    const int want = *pick;
    if (threadIdx.x == 0) row = map ? -1 : want;
    __syncthreads();
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        m = fmaxf(m, x[i] / te);
        if (map && map[i] == want) row = i;   // (the map's ids are distinct: one writer)
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    m = sh[0];
    for (int k = 1; k < 32; ++k) m = fmaxf(m, sh[k]);
    __syncthreads();
    const float lmin = *lmin_p;
    if (tau > 0.f && lse_p && thr_p) {   // the support draft_support chose
        if (threadIdx.x == 0) {
            const float th = *thr_p, lse = *lse_p;
            const int r = row;
            conf[step] = r >= 0 && r < n && x[r] / te >= th ? expf(x[r] / te - lse) : 0.f;
            top[step] = expf(m - lse);
        }
        return;
    }
    const float th = tau > 0.f && lmin > -INFINITY ? m + lmin : -INFINITY;
    float e = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = x[i] / te;
        if (xi >= th) e += expf(xi - m);
    }
    for (int o = 16; o > 0; o >>= 1) e += __shfl_xor(e, o, 32);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = e;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int k = 0; k < 32; ++k) t += sh[k];
        const float lse = m + logf(t);
        const int r = row;
        conf[step] = r >= 0 && r < n && x[r] / te >= th ? expf(x[r] / te - lse) : 0.f;
        top[step] = expf(m - lse);
    }
}

constexpr int kCand = 4096;

__device__ float bsum(float v, float* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.f;
    for (int k = 0; k < 32; ++k) t += sh[k];
    return t;
}

__global__ void __launch_bounds__(1024) k_draft_support(const float* x, int n, const float* tau_p, const float* lmin_p,
                                                        const float* topp_p, const int* topk_p, float* lse_out,
                                                        float* thr_out) {
    __shared__ float sh[32];
    __shared__ float cv[kCand];
    __shared__ int nc;
    const float tau = *tau_p;
    if (tau <= 0.f) { if (threadIdx.x == 0) *lse_out = 0.f; return; }
    float m = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i] / tau);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    if (threadIdx.x == 0) nc = 0;
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    m = sh[0];
    for (int k = 1; k < 32; ++k) m = fmaxf(m, sh[k]);
    __syncthreads();
    const float lmin = *lmin_p;
    const float tm = lmin > -INFINITY ? m + lmin : -INFINITY;
    const float cut = fmaxf(tm, m - 30.f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i] / tau;
        if (v >= cut) {
            const int j = atomicAdd(&nc, 1);
            if (j < kCand) cv[j] = v;
        }
    }
    __syncthreads();
    const int N = nc;
    float th = tm;
    if (N <= kCand) {
        const int K = *topk_p > 0 ? *topk_p : 64;
        const float P = *topp_p;
        float base = cut;
        if (N > K) {   // top-k: the largest t with count(v >= t) >= K
            float lo = cut, hi = m;
            for (int it = 0; it < 24; ++it) {
                const float mid = 0.5f * (lo + hi);
                float c = 0.f;
                for (int j = threadIdx.x; j < N; j += blockDim.x) c += cv[j] >= mid ? 1.f : 0.f;
                c = bsum(c, sh);
                if (c >= (float) K) lo = mid; else hi = mid;
            }
            base = lo;
        }
        th = fmaxf(th, base);
        if (P > 0.f && P < 1.f) {   // top-p over the top-k set: the largest t with mass(v >= t) >= P mass(v >= base)
            float z = 0.f;
            for (int j = threadIdx.x; j < N; j += blockDim.x) z += cv[j] >= base ? expf(cv[j] - m) : 0.f;
            z = bsum(z, sh);
            float lo = base, hi = m;
            for (int it = 0; it < 24; ++it) {
                const float mid = 0.5f * (lo + hi);
                float s = 0.f;
                for (int j = threadIdx.x; j < N; j += blockDim.x) s += cv[j] >= mid ? expf(cv[j] - m) : 0.f;
                s = bsum(s, sh);
                if (s >= P * z) lo = mid; else hi = mid;
            }
            th = fmaxf(th, lo);
        }
        th = fminf(th, m);   // (the max always stays in)
    }
    float e = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float xi = x[i] / tau;
        if (xi >= th) e += expf(xi - m);
    }
    e = bsum(e, sh);
    if (threadIdx.x == 0) {
        *lse_out = m + logf(e);
        *thr_out = th;
    }
}

}  // namespace

void draft_conf(const float* x, int n, const float* tau, const float* lmin, const int32_t* pick, const int32_t* map,
                float* conf, float* top, int step, void* stream, const float* lse, const float* thr) {
    k_draft_conf<<<1, 1024, 0, (cudaStream_t) stream>>>(x, n, tau, lmin, pick, map, conf, top, step, lse, thr);
}

void draft_support(const float* x, int n, const float* tau, const float* lmin, const float* top_p, const int* top_k,
                   float* lse, float* thr, void* stream) {
    k_draft_support<<<1, 1024, 0, (cudaStream_t) stream>>>(x, n, tau, lmin, top_p, top_k, lse, thr);
}

}  // namespace talus::q36
