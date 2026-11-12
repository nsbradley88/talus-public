// src/prefill/tp_w4i.cu - W4A4 prompt GEMM on v_wmma_i32_16x16x16_iu4 with 32-value activation groups and
// INTEGER group scales (opt-in, quality-gated; not used by any default path).
//
// Activations (per token t, after a seeded-sign 256-point Hadamard per 256-value block, as tp_w4a4):
//   S_t = max|v| / (7 * 127), k_g = clamp(ceil(amax_g / (7 S_t)), 1, 127) per 32-value group g,
//   q = clamp(rint(v / (k_g S_t)), -7, 7)                                      (int4, two per byte, low nibble first)
// so v ~= q * k_g * S_t.  The GEMM needs sum_g k_g p_g (p_g = the 32-value partial of two iu4 WMMAs).  With the
// running sum P_g = p_1 + .. + p_g (the WMMA accumulator, never reset inside a weight block) that is
// sum_g P_g (k_g - k_{g+1}) (Abel summation, k past the block's end = 0), so the pack stores d_g = k_g - k_{g+1}
// (int8) and the GEMM adds P * d_g into an int32 accumulator: ONE v_mad_i32_i24 per output element per 32 K
// (|P| <= 49 K < 2^23 for K < 171000, |d| <= 127), no float convert, no accumulator reset.  The integer sum is
// exact (int32 wraps cancel: the true total is < 2^31); the float epilogue (weight scale per output row, or per
// 256-value weight group, times S_t) runs once per row / weight group.
// Offline (synthetic Gaussian data): integer scales with kmax 127 closely match float group scales.
//
// Layouts:
//   wq [N/16][K/32][16 rows][16 B]  (int4 weights in the rotated basis, signed, low nibble = even k)
//   ws [N] (WG = 0: per output row) or [N][K/256] (WG = 256)
//   xq [T][K/2] int4 activations, xk [T][K/32] int8 d_g (Abel form; blocks of WB = the weight group or K), xS [T]
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "strata/kernels/variant_trace.hpp"   // S1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "tp_w4i_quant.hpp"

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define TP_W4I_WMMA 1
#endif

extern "C" __device__ int __ockl_mul24_i32(int, int);

namespace talus::prefill::w4i {
namespace {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8i wmma_i4(v2i a, v2i b, v8i c) {
#if defined(TP_W4I_WMMA)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false);
#else
    (void) a; (void) b;
    return c;
#endif
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const void* p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short) 0, (int) bytes, 0x31004000);
}
__device__ __forceinline__ v4i ld16(__amdgpu_buffer_rsrc_t r, uint32_t voff, uint32_t soff) {
    return __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0);
}
__device__ __forceinline__ uint32_t ld4(__amdgpu_buffer_rsrc_t r, uint32_t voff, uint32_t soff) {
    return __builtin_amdgcn_raw_buffer_load_b32(r, voff, soff, 0);
}

template <bool SWI>
__device__ __forceinline__ void load8(float v[8], const float* __restrict__ x, const float* __restrict__ g, size_t o) {
    const float4 a0 = *(const float4*) (x + o), a1 = *(const float4*) (x + o + 4);
    v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w; v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    if constexpr (SWI) {   // x = up, g = gate: silu(g) * u with q36_engine's expression
        const float4 b0 = *(const float4*) (g + o), b1 = *(const float4*) (g + o + 4);
        const float gv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = w4i_swiglu(gv[e], v[e]);
    }
}

/// one block (256 threads) per token: pass 1 rotates every 256-value block and keeps the 32-value group maxima in
/// LDS, pass 2 rotates again and packs with the integer group scales.  K <= 32768.  (rc-a4's kernel, with
/// rd-q4quant's options: SM = per-channel smoothing inv_s before the rotation, KM = the group-scale rule; the
/// fallback of k_w4i_pack1 for rows that do not fit its registers, and TALUS_PF_W4I_PACK1=0)
template <bool SWI, bool SM = false, int KM = 0>
__global__ void __launch_bounds__(256) k_w4i_pack(const float* __restrict__ x, const float* __restrict__ g, int K,
                                                  int64_t ldx, const int8_t* __restrict__ signs,
                                                  const float* __restrict__ inv_s, uint8_t* __restrict__ xq,
                                                  uint8_t* __restrict__ xk, float* __restrict__ xS, int wb) {
    __shared__ float gmax[1024];
    __shared__ float wmax[8];
    __shared__ uint8_t kch[KM ? 1024 : 1];
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NB = K / kHB;
    const size_t row = (size_t) t * ldx;
    float m = 0.f;
    for (int b = w; b < NB; b += 8) {
        const int c0 = b * kHB + lane * 8;
        float v[8];
        load8<SWI>(v, x, g, row + c0);
        w4i_rot256(v, signs + c0, SM ? inv_s + c0 : nullptr, lane);
        const float a = w4i_group_amax(v);
        if ((lane & 3) == 0) gmax[c0 / kAG] = a;
        m = fmaxf(m, a);
    }
#pragma unroll
    for (int s = 4; s < 32; s <<= 1) m = fmaxf(m, __shfl_xor(m, s, 32));
    if (lane == 0) wmax[w] = m;
    __syncthreads();
    m = wmax[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) m = fmaxf(m, wmax[i]);
    const W4iRow r = w4i_row(m);
    if (threadIdx.x == 0) xS[t] = r.S;
    uint8_t* xk_t = xk + (size_t) t * (K / kAG);
    for (int b = w; b < NB; b += 8) {
        const int c0 = b * kHB + lane * 8;
        float v[8];
        load8<SWI>(v, x, g, row + c0);
        w4i_rot256(v, signs + c0, SM ? inv_s + c0 : nullptr, lane);
        w4i_store_block<KM>(v, c0, gmax, wb, r, xq + (size_t) t * (K / 2), xk_t, kch, lane);
    }
    if constexpr (KM != 0) {
        __syncthreads();
        w4i_store_abel(kch, K, wb, xk_t);
    }
}

/// rd-q4quant-B: the row's rotated values stay in registers, so every input is read (and SwiGLU'd and rotated) once.
/// Block = NW waves (blockDim 32 NW), wave w owns the 256-value blocks w, w + NW, .. (at most NBW of them).  The same
/// per-block functions as k_w4i_pack; the row maximum is order-free, so the bytes equal k_w4i_pack's.
/// SM: rs = the smoothing vector inv_s; KM (k choice) as tp_w4i_quant.hpp; kch: LDS [K / 32] for KM != 0.
template <int NBW, bool SM, int KM, class Load>
__device__ __forceinline__ void w4i_pack_row(const Load& load, int K, const int8_t* __restrict__ signs,
                                             const float* __restrict__ rs, uint8_t* __restrict__ xq_t,
                                             uint8_t* __restrict__ xk_t, float* __restrict__ xS_t, int wb,
                                             float* gmax, float* wmax, uint8_t* kch) {
    const int NW = (int) (blockDim.x >> 5), w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NB = K / kHB;
    float v[NBW][8];
    float m = 0.f;
#pragma unroll
    for (int j = 0; j < NBW; ++j) {
        const int b = w + j * NW;
        if (b < NB) {
            const int c0 = b * kHB + lane * 8;
            load(v[j], c0);
            w4i_rot256(v[j], signs + c0, SM ? rs + c0 : nullptr, lane);
            const float a = w4i_group_amax(v[j]);
            if ((lane & 3) == 0) gmax[c0 / kAG] = a;
            m = fmaxf(m, a);
        }
    }
#pragma unroll
    for (int s = 4; s < 32; s <<= 1) m = fmaxf(m, __shfl_xor(m, s, 32));
    if (lane == 0) wmax[w] = m;
    __syncthreads();
    m = wmax[0];
    for (int i = 1; i < NW; ++i) m = fmaxf(m, wmax[i]);
    const W4iRow r = w4i_row(m);
    if (threadIdx.x == 0) *xS_t = r.S;
#pragma unroll
    for (int j = 0; j < NBW; ++j) {
        const int b = w + j * NW;
        if (b < NB) w4i_store_block<KM>(v[j], b * kHB + lane * 8, gmax, wb, r, xq_t, xk_t, kch, lane);
    }
    if constexpr (KM != 0) {
        __syncthreads();
        w4i_store_abel(kch, K, wb, xk_t);
    }
}

template <int NBW, bool SWI, bool SM, int KM>
__global__ void __launch_bounds__(512) k_w4i_pack1(const float* __restrict__ x, const float* __restrict__ g, int K,
                                                   int64_t ldx, const int8_t* __restrict__ signs,
                                                   const float* __restrict__ inv_s, uint8_t* __restrict__ xq,
                                                   uint8_t* __restrict__ xk, float* __restrict__ xS, int wb) {
    __shared__ float gmax[1024];
    __shared__ float wmax[16];
    __shared__ uint8_t kch[KM ? 1024 : 1];
    const int t = blockIdx.x;
    const size_t row = (size_t) t * ldx;
    auto load = [&](float v[8], int c0) { load8<SWI>(v, x, g, row + c0); };
    w4i_pack_row<NBW, SM, KM>(load, K, signs, inv_s, xq + (size_t) t * (K / 2), xk + (size_t) t * (K / kAG), xS + t,
                              wb, gmax, wmax, kch);
}

/// rd-q4quant-B: k_add_rms (q36_kernels.cu: x += y, h = rms(x) * w, 256 threads, the same block_sum tree) with the
/// w4i pack of h in its epilogue.  h is written only when h != nullptr (a plane of the layer still reads it).
__device__ float w4i_block_sum256(float v, float* sh) {   // q36_kernels.cu block_sum (blockDim 256)
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

/// SM / KM apply to both packed sets (the caller only pairs sets with the same options; inv_s / inv_s2 may differ)
template <int NBW, bool SM, int KM>
__global__ void __launch_bounds__(256) k_w4i_add_rms_pack(float* __restrict__ x, const float* __restrict__ y,
                                                         const float* __restrict__ w, float* __restrict__ h, int n,
                                                         float eps, const int8_t* __restrict__ signs,
                                                         const float* __restrict__ inv_s, int wb,
                                                         uint8_t* __restrict__ xq, uint8_t* __restrict__ xk,
                                                         float* __restrict__ xS, const int8_t* __restrict__ signs2,
                                                         const float* __restrict__ inv_s2, int wb2,
                                                         uint8_t* __restrict__ xq2, uint8_t* __restrict__ xk2,
                                                         float* __restrict__ xS2) {
    __shared__ float sh[256];
    __shared__ float gmax[1024];
    __shared__ float wmax[8];
    __shared__ uint8_t kch[KM ? 1024 : 1];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    const float* yr = y + (size_t) t * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = w4i_block_sum256(s, sh);   // (ends with a barrier: the row's new x is visible to the whole block)
    const float r = rsqrtf(s / (float) n + eps);
    float* hr = h ? h + (size_t) t * n : nullptr;
    auto hload = [&](float v[8], int c0, bool store) {
        const float4 a0 = *(const float4*) (xr + c0), a1 = *(const float4*) (xr + c0 + 4);
        const float4 w0 = *(const float4*) (w + c0), w1 = *(const float4*) (w + c0 + 4);
        const float xv[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = xv[e] * r * wv[e];   // k_add_rms: xr[i] * r * w[i]
        if (store) {
            *(float4*) (hr + c0) = make_float4(v[0], v[1], v[2], v[3]);
            *(float4*) (hr + c0 + 4) = make_float4(v[4], v[5], v[6], v[7]);
        }
    };
    auto load1 = [&](float v[8], int c0) { hload(v, c0, hr != nullptr); };
    w4i_pack_row<NBW, SM, KM>(load1, n, signs, inv_s, xq + (size_t) t * (n / 2), xk + (size_t) t * (n / kAG), xS + t,
                              wb, gmax, wmax, kch);
    if (signs2) {   // a second matrix of the layer with another (signs, wg, inv_s): the same h, rotated again
        __syncthreads();   // (gmax / wmax / kch are reused)
        auto load2 = [&](float v[8], int c0) { hload(v, c0, false); };
        w4i_pack_row<NBW, SM, KM>(load2, n, signs2, inv_s2, xq2 + (size_t) t * (n / 2), xk2 + (size_t) t * (n / kAG),
                                  xS2 + t, wb2, gmax, wmax, kch);
    }
}

/// GEMM: block = WGN (row) x WGT (token) waves, a wave WM x WT 16 x 16 tiles; B (activations + k_g) staged through
/// LDS KS 32-value blocks per stage (double-buffered, one barrier per stage); A (weights) from global straight into
/// registers one block ahead.  The accumulator lane holds token l16, rows 2 i + hi (as tp_i8gemm v10).
/// GU (rd-q4quant-B): wq / ws hold gate and up interleaved by 16-row tiles (tile 2 j = gate tile j, 2 j + 1 = up tile
/// j; N = 2 F), a wave's tile pair (a, a + 1) is one feature tile, and the epilogue writes silu(g) * u [T][ldd = F]
/// with g and u exactly as the plain kernel writes them (the down projection's input, without the f32 g / u round trip).
template <int WGN, int WGT, int WM, int WT, int KS, int WG, bool GU = false>
__global__ void __launch_bounds__(32 * WGN * WGT) k_w4i_gemm(const uint8_t* __restrict__ wq, const float* __restrict__ ws,
                                                            const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xk,
                                                            const float* __restrict__ xS, float* __restrict__ dst,
                                                            int K, int T, int64_t ldd, int N) {
    constexpr int NT = 32 * WGN * WGT, MT = WGN * WM, BT = 16 * WGT * WT;
    static_assert(KS % 2 == 0 && KS <= 4, "KS");
    static_assert(WG == 0 || WG % (32 * KS) == 0, "WG");
    struct Stage {
        uint8_t b[KS][BT][16];
        uint8_t k[BT][4];
    };
    __shared__ __attribute__((aligned(16))) Stage st[2];
    const int KB = K / 32, NS = KB / KS;
    const int tid = threadIdx.x, wv = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = wv % WGN, wt = wv / WGN;
    const int mb0 = blockIdx.y * MT, tb0 = blockIdx.x * BT;
    const __amdgpu_buffer_rsrc_t rq = rsrc(wq, (uint32_t) N * (uint32_t) (K / 2));
    const __amdgpu_buffer_rsrc_t rx = rsrc(xq, (uint32_t) T * (uint32_t) (K / 2));
    const __amdgpu_buffer_rsrc_t rk = rsrc(xk, (uint32_t) T * (uint32_t) KB);
    // B staging: 16-B units u = tid + j NT -> (t = u / KS, kk = u % KS); k_g: thread t < BT loads KS bytes
    constexpr int NB = KS * BT;
    static_assert(NB % NT == 0, "B staging");
    constexpr int RB = NB / NT;
    uint32_t vb[RB];
#pragma unroll
    for (int j = 0; j < RB; ++j) {
        const int u = tid + j * NT, t = u / KS, kk = u % KS;
        vb[j] = (uint32_t) min(tb0 + t, T - 1) * (uint32_t) (K / 2) + kk * 16;
    }
    const uint32_t vk = (uint32_t) min(tb0 + (tid % BT), T - 1) * (uint32_t) KB;
    v4i rb[RB];
    uint32_t rkv = 0;
    auto gload = [&](int s) {
        const uint32_t so = (uint32_t) s * KS * 16;
#pragma unroll
        for (int j = 0; j < RB; ++j) rb[j] = ld16(rx, vb[j], so);
        // (every thread: no branch inside the K loop; threads tid and tid + BT load / store the same bytes)
        if constexpr (KS == 4) rkv = ld4(rk, vk, (uint32_t) s * KS);
        else rkv = (uint32_t) __builtin_amdgcn_raw_buffer_load_b16(rk, vk, (uint32_t) s * KS, 0);   // (KS 2: 2-B aligned)
    };
    auto lstore = [&](Stage& S) {
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int u = tid + j * NT, t = u / KS, kk = u % KS;
            *(v4i*) &S.b[kk][t][0] = rb[j];
        }
        *(uint32_t*) &S.k[tid % BT][0] = rkv;
    };
    uint32_t va[WM];
#pragma unroll
    for (int a = 0; a < WM; ++a) va[a] = (uint32_t) (mb0 + wn * WM + a) * (uint32_t) KB * 256u + l16 * 16;
    auto aload = [&](int kb, v4i* A) {
#pragma unroll
        for (int a = 0; a < WM; ++a) A[a] = ld16(rq, va[a], (uint32_t) kb * 256u);
    };
    int acc[WM][WT][8];
    v8i P[WM][WT];
    float sum[WM][WT][8];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WT; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[a][b][i] = 0;
                P[a][b][i] = 0;
                sum[a][b][i] = 0.f;
            }
    auto flush = [&](int gw) {   // WG > 0: the weight group's float scale, then a fresh integer accumulator
#pragma unroll
        for (int a = 0; a < WM; ++a) {
            float dw[8];
#pragma unroll
            for (int i = 0; i < 8; ++i)
                dw[i] = ws[(size_t) ((mb0 + wn * WM + a) * 16 + 2 * i + hi) * (K / (WG ? WG : 1)) + gw];
#pragma unroll
            for (int b = 0; b < WT; ++b)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    sum[a][b][i] = __builtin_fmaf((float) acc[a][b][i], dw[i], sum[a][b][i]);
                    acc[a][b][i] = 0;
                    P[a][b][i] = 0;
                }
        }
    };
    auto block = [&](const Stage& S, int kk, const v4i* A) {
        v4i B[WT];
        int kv[WT];
#pragma unroll
        for (int b = 0; b < WT; ++b) {
            const int t = wt * 16 * WT + b * 16 + l16;
            B[b] = *(const v4i*) &S.b[kk][t][0];
            kv[b] = (int) (int8_t) S.k[t][kk];
        }
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int b = 0; b < WT; ++b) {
                P[a][b] = wmma_i4((v2i){A[a][0], A[a][1]}, (v2i){B[b][0], B[b][1]}, P[a][b]);
                P[a][b] = wmma_i4((v2i){A[a][2], A[a][3]}, (v2i){B[b][2], B[b][3]}, P[a][b]);
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[a][b][i] += __ockl_mul24_i32(P[a][b][i], kv[b]);
            }
    };
    v4i ring0[WM], ring1[WM];
    aload(0, ring0);
    gload(0);
    lstore(st[0]);
    __syncthreads();
    // SPG stages per weight group (WG 0: 1), unrolled so the stage buffer index and the flush are static (no branch
    // inside the K loop: a branch let the compiler sink the mads below every WMMA and keep 4 copies of P alive)
    constexpr int SPG = WG ? WG / (32 * KS) : 1;
    static_assert(SPG == 1 || SPG % 2 == 0, "SPG");
    auto stage = [&](int s, const Stage& S, Stage& Snext) {
#pragma unroll
        for (int kk = 0; kk < KS; kk += 2) {
            const int kb = s * KS + kk;
            aload(min(kb + 1, KB - 1), ring1);
            if (kk == 0) gload(min(s + 1, NS - 1));
            block(S, kk, ring0);
            __builtin_amdgcn_sched_barrier(0);
            aload(min(kb + 2, KB - 1), ring0);
            block(S, kk + 1, ring1);
            __builtin_amdgcn_sched_barrier(0);
        }
        lstore(Snext);
        __syncthreads();
    };
    if constexpr (SPG == 1 && WG == 0) {
#pragma unroll 1
        for (int s = 0; s < NS; ++s) stage(s, st[s & 1], st[(s + 1) & 1]);
    } else if constexpr (SPG == 1) {   // one stage per weight group (wg 128 at KS 4): two stages per trip
#pragma unroll 1
        for (int s0 = 0; s0 < NS; s0 += 2) {
            stage(s0, st[0], st[1]);
            flush(s0);
            stage(s0 + 1, st[1], st[0]);
            flush(s0 + 1);
        }
    } else {
#pragma unroll 1
        for (int s0 = 0; s0 < NS; s0 += SPG) {
#pragma unroll
            for (int j = 0; j < SPG; ++j) stage(s0 + j, st[j & 1], st[(j + 1) & 1]);
            flush(s0 / SPG);
        }
    }
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = tb0 + wt * 16 * WT + b * 16 + l16;
        if (t >= T) continue;
        const float sx = xS[t];
        if constexpr (GU) {
            static_assert(WM % 2 == 0, "GU: tile pairs");
#pragma unroll
            for (int a = 0; a < WM; a += 2)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int tg = mb0 + wn * WM + a;   // even: the gate tile of feature tile tg / 2
                    const int rg = tg * 16 + 2 * i + hi, ru = rg + 16;
                    const float vg = WG ? sum[a][b][i] : (float) acc[a][b][i] * ws[rg];
                    const float vu = WG ? sum[a + 1][b][i] : (float) acc[a + 1][b][i] * ws[ru];
                    const float g = vg * sx, u = vu * sx;   // (= the plain kernel's dst values)
                    dst[(size_t) t * ldd + (tg / 2) * 16 + 2 * i + hi] = w4i_swiglu(g, u);
                }
        } else {
#pragma unroll
            for (int a = 0; a < WM; ++a)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = (mb0 + wn * WM + a) * 16 + 2 * i + hi;
                    const float v = WG ? sum[a][b][i] : (float) acc[a][b][i] * ws[r];
                    dst[(size_t) t * ldd + r] = v * sx;
                }
        }
    }
}

template <int WGN, int WGT, int WM, int WT, int KS, int WG, bool GU = false>
bool launch(const uint8_t* wq, const float* ws, int N, int K, const uint8_t* xq, const uint8_t* xk, const float* xS,
            int T, float* dst, int64_t ld, hipStream_t s) {
    constexpr int BN = 16 * WGN * WM, BT = 16 * WGT * WT;
    if (N % BN != 0 || (K / 32) % KS != 0 || (WG && K % WG != 0)) return false;
    if (WG && (K / (32 * KS)) % 2 != 0) return false;
    const dim3 grid((unsigned) ((T + BT - 1) / BT), (unsigned) (N / BN));
    k_w4i_gemm<WGN, WGT, WM, WT, KS, WG, GU><<<grid, 32 * WGN * WGT, 0, s>>>(wq, ws, xq, xk, xS, dst, K, T, ld, N);
    return true;
}

template <int WG, bool GU = false>
bool launch_v(int v, const uint8_t* wq, const float* ws, int N, int K, const uint8_t* xq, const uint8_t* xk,
              const float* xS, int T, float* dst, int64_t ld, hipStream_t s) {
    switch (v) {
        case 1: return launch<4, 2, 2, 4, 4, WG, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // 128 x 128 (v106's)
        case 2: return launch<4, 2, 2, 4, 2, WG, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // KS 2
        case 3: return launch<2, 4, 4, 2, 4, WG, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // waves 2 x 4, 4 x 2
        case 4:   // 256 x 128 (spills; not built for GU: v1)
            if constexpr (!GU) return launch<4, 2, 4, 4, 4, WG>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);
            else return launch<4, 2, 2, 4, 4, WG, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);
        case 5: return launch<4, 2, 2, 2, 4, WG, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // 128 x 64
        default: return launch<4, 2, 2, 4, 4, WG, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);
    }
}

/// rd-q4quant-B: gate / up weight tiles interleaved for k_w4i_gemm<.., GU>: block = one output tile (16 rows)
__global__ void __launch_bounds__(256) k_w4i_interleave_gu(const uint4* __restrict__ gq, const uint4* __restrict__ uq,
                                                           const float* __restrict__ gs, const float* __restrict__ us,
                                                           int K, int G, uint4* __restrict__ oq,
                                                           float* __restrict__ os) {
    const int j = blockIdx.x >> 1, up = blockIdx.x & 1;
    const size_t n16 = (size_t) K / 2;   // uint4 per 16-row tile: 16 rows x K / 2 bytes / 16
    const uint4* src = (up ? uq : gq) + (size_t) j * n16;
    uint4* dst = oq + (size_t) blockIdx.x * n16;
    for (size_t i = threadIdx.x; i < n16; i += blockDim.x) dst[i] = src[i];
    const float* ss = (up ? us : gs) + (size_t) j * 16 * G;   // G = scales per row (1, or K / wg)
    for (int i = threadIdx.x; i < 16 * G; i += blockDim.x) os[(size_t) blockIdx.x * 16 * G + i] = ss[i];
}

/// k_w4i_pack1's registers per lane: NBW blocks of 8 values; waves = ceil(NB / NBW) <= TALUS_PF_W4I_PACK_NW (default
/// 16, at most 16).  0 = the two-pass k_w4i_pack (TALUS_PF_W4I_PACK1=0, or rows too long for the registers).
inline int pack1_nbw(int NB) {
    static const int on = [] { const char* e = std::getenv("TALUS_PF_W4I_PACK1"); return e ? std::atoi(e) : 1; }();
    static const int nw = [] {
        const char* e = std::getenv("TALUS_PF_W4I_PACK_NW");
        const int v = e ? std::atoi(e) : 16;
        return v < 1 ? 1 : (v > 16 ? 16 : v);
    }();
    if (!on) return 0;
    const int nbw = (NB + nw - 1) / nw;
    return nbw <= 8 ? nbw : 0;
}
template <int V> struct IntC { static constexpr int value = V; };
template <class F> inline void with_nbw(int nbw, F&& f) {
    switch (nbw) {
        case 1: f(IntC<1>{}); break;
        case 2: f(IntC<2>{}); break;
        case 3: f(IntC<3>{}); break;
        case 4: f(IntC<4>{}); break;
        case 5: f(IntC<5>{}); break;
        case 6: f(IntC<6>{}); break;
        case 7: f(IntC<7>{}); break;
        case 8: f(IntC<8>{}); break;
        default: break;
    }
}

}  // namespace
}  // namespace talus::prefill::w4i

/// rc-a4 / rd-q4quant / rd-q4quant-B: launch the pack of T rows (gate != nullptr: silu(gate) * x) with smoothing inv_s
/// (nullptr: none) and group-scale rule kmode (0 / 1) for weight groups wg (0 / 128 / 256)
static int w4i_pack_launch(const float* x, const float* gate, int T, int K, int64_t ldx, const int8_t* signs, int wg,
                           const float* inv_s, int kmode, uint8_t* xq, uint8_t* xk, float* xS, void* stream) {
    using namespace talus::prefill::w4i;
    if (T <= 0 || K % kHB != 0 || K > 32768 || ldx % 4 != 0 || (wg != 0 && wg != 128 && wg != 256) ||
        (kmode != 0 && kmode != 1))
        return -1;
    const hipStream_t s = (hipStream_t) stream;
    const int wb = wg ? wg : K;   // the Abel blocks = the GEMM's weight groups (it resets its running sums there)
    const int nbw = pack1_nbw(K / kHB);
    auto go = [&](auto swi, auto sm, auto km) {
        constexpr bool SWI = decltype(swi)::value != 0, SM = decltype(sm)::value != 0;
        constexpr int KM = decltype(km)::value;
        bool ok = false;
        if (nbw > 0) {
            const unsigned nt = 32u * (unsigned) ((K / kHB + nbw - 1) / nbw);
            with_nbw(nbw, [&](auto c) {
                constexpr int NBW = decltype(c)::value;
                k_w4i_pack1<NBW, SWI, SM, KM><<<T, nt, 0, s>>>(x, gate, K, ldx, signs, inv_s, xq, xk, xS, wb);
                ok = true;
            });
        }
        if (!ok) k_w4i_pack<SWI, SM, KM><<<T, 256, 0, s>>>(x, gate, K, ldx, signs, inv_s, xq, xk, xS, wb);
    };
    auto km = [&](auto swi, auto sm) {
        if (kmode) go(swi, sm, IntC<1>{});
        else go(swi, sm, IntC<0>{});
    };
    auto sm = [&](auto swi) {
        if (inv_s) km(swi, IntC<1>{});
        else km(swi, IntC<0>{});
    };
    if (gate) sm(IntC<1>{});
    else sm(IntC<0>{});
    return 0;
}

/// pack T rows of FP32 activations x [T][ldx] (gate == nullptr) or silu(gate) * x (gate [T][ldx], the SwiGLU
/// input of the down projection) into the w4i layout for a GEMM with weight groups wg (0 / 128 / 256).  0 ok, -1 not
/// covered.
extern "C" int talus_pf_w4i_pack(const float* x, const float* gate, int T, int K, int64_t ldx, const int8_t* signs,
                                 int wg, uint8_t* xq, uint8_t* xk, float* xS, void* stream) {
    return w4i_pack_launch(x, gate, T, K, ldx, signs, wg, nullptr, 0, xq, xk, xS, stream);
}

/// talus_pf_w4i_pack + optional smoothing inv_s [K] (nullptr: none) and kmode (0 absmax k, 1 best of 4
/// clips); wg 0 / 128 / 256.  0 ok, -1 not covered.
extern "C" int talus_pf_w4i_pack2(const float* x, const float* gate, int T, int K, int64_t ldx, const int8_t* signs,
                                  int wg, const float* inv_s, int kmode, uint8_t* xq, uint8_t* xk, float* xS,
                                  void* stream) {
    return w4i_pack_launch(x, gate, T, K, ldx, signs, wg, inv_s, kmode, xq, xk, xS, stream);
}

/// rd-q4quant-B: x [T][K] += y; h = rms(x) * w (as tk_q36_add_rms, bit for bit; h == nullptr: not written) and the
/// w4i pack of h (as talus_pf_w4i_pack2(h, signs, wg, inv_s, kmode), bit for bit) in one launch; signs2 != nullptr:
/// also the pack with signs2 / wg2 / inv_s2 (same kmode; inv_s2 present iff inv_s) into xq2 / xk2 / xS2 (the
/// layer's other matrix).  0 ok, -1 not covered (nothing launched).
extern "C" int talus_pf_w4i_add_rms_pack(float* x, const float* y, const float* w, float* h, int T, int K, float eps,
                                         const int8_t* signs, int wg, const float* inv_s, uint8_t* xq, uint8_t* xk,
                                         float* xS, const int8_t* signs2, int wg2, const float* inv_s2, uint8_t* xq2,
                                         uint8_t* xk2, float* xS2, int kmode, void* stream) {
    using namespace talus::prefill::w4i;
    auto wg_ok = [](int g) { return g == 0 || g == 128 || g == 256; };
    if (T <= 0 || K % kHB != 0 || K > 32768 || !wg_ok(wg) || (signs2 && !wg_ok(wg2)) || (kmode != 0 && kmode != 1) ||
        (signs2 && (inv_s2 != nullptr) != (inv_s != nullptr)))
        return -1;
    const int NB = K / kHB;
    if (NB > 8 * 8) return -1;   // 8 waves x at most 8 blocks of registers
    const hipStream_t s = (hipStream_t) stream;
    bool ok = false;
    auto go = [&](auto sm, auto km) {
        constexpr bool SM = decltype(sm)::value != 0;
        constexpr int KM = decltype(km)::value;
        with_nbw((NB + 7) / 8, [&](auto c) {
            constexpr int NBW = decltype(c)::value;
            k_w4i_add_rms_pack<NBW, SM, KM><<<T, 256, 0, s>>>(x, y, w, h, K, eps, signs, inv_s, wg ? wg : K, xq, xk,
                                                               xS, signs2, inv_s2, wg2 ? wg2 : K, xq2, xk2, xS2);
            ok = true;
        });
    };
    if (inv_s) {
        if (kmode) go(IntC<1>{}, IntC<1>{});
        else go(IntC<1>{}, IntC<0>{});
    } else {
        if (kmode) go(IntC<0>{}, IntC<1>{});
        else go(IntC<0>{}, IntC<0>{});
    }
    return ok ? 0 : -1;
}

/// the GEMM's tile choice (shared by the plain and the GU GEMM, so a pair runs the shape its matrices would):
/// wg 0: TALUS_PF_W4I_VARIANT (1..5, default 1); wg 128 (one K stage per weight group at KS 4):
/// TALUS_PF_W4I_V128 (1, 2, 5; default 5); wg 256: TALUS_PF_W4I_V256 (1, 2, 3, 5; default 5: v1 spills at wg 256).
/// The shape does not change the result (exact integer partials, per-group float flush in K order).
template <bool GU>
static bool w4i_gemm_dispatch(const uint8_t* wq, const float* ws, int wg, int N, int K, const uint8_t* xq,
                              const uint8_t* xk, const float* xS, int T, float* dst, int64_t ld, hipStream_t s) {
    using namespace talus::prefill::w4i;
    static const int v = [] { const char* e = std::getenv("TALUS_PF_W4I_VARIANT"); return e ? std::atoi(e) : 1; }();
    static const int v128 = [] { const char* e = std::getenv("TALUS_PF_W4I_V128"); return e ? std::atoi(e) : 5; }();
    static const int v256 = [] { const char* e = std::getenv("TALUS_PF_W4I_V256"); return e ? std::atoi(e) : 5; }();
    TALUS_VTRACE("pf_w4i", GU ? "gu" : "one", wg, wg == 128 ? v128 : wg == 256 ? (v256 == 4 ? 5 : v256) : v, N, K);
    if (wg == 128)
        return v128 == 1 ? launch<4, 2, 2, 4, 4, 128, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s)
             : v128 == 2 ? launch<4, 2, 2, 4, 2, 128, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s)
                         : launch<4, 2, 2, 2, 4, 128, GU>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // (v5 shape)
    if (wg == 256) return launch_v<256, GU>(v256 == 4 ? 5 : v256, wq, ws, N, K, xq, xk, xS, T, dst, ld, s);
    return launch_v<0, GU>(v, wq, ws, N, K, xq, xk, xS, T, dst, ld, s);
}

/// dst [T][ld] = W4A4 product of the packed activations with wq / ws (wg 0: per-row scales, 128 / 256: per
/// weight group).  0 ok, -1 shape not covered.
extern "C" int talus_pf_w4i_gemm(const uint8_t* wq, const float* ws, int wg, int N, int K, const uint8_t* xq,
                                 const uint8_t* xk, const float* xS, int T, float* dst, int64_t ld, void* stream) {
    using namespace talus::prefill::w4i;
    if (T <= 0 || K % kHB != 0 || (wg != 0 && wg != 128 && wg != 256)) return -1;
    return w4i_gemm_dispatch<false>(wq, ws, wg, N, K, xq, xk, xS, T, dst, ld, (hipStream_t) stream) ? 0 : -1;
}

/// rd-q4quant-B: dst [T][ld] = silu(gate x) * (up x) for a gate / up pair interleaved by 16-row tiles
/// (talus_pf_w4i_interleave_gu; 2 F rows, weight groups wg 0 / 128 / 256, both in the same rotated basis = the same
/// packed activations).  Bit for bit talus_pf_w4i_gemm(gate) + talus_pf_w4i_gemm(up) + the SwiGLU of
/// talus_pf_w4i_pack(up, gate); the same tile shape as those.  0 ok, -1 shape not covered.
extern "C" int talus_pf_w4i_gemm_gu(const uint8_t* wq_gu, const float* ws_gu, int wg, int F, int K, const uint8_t* xq,
                                    const uint8_t* xk, const float* xS, int T, float* dst, int64_t ld, void* stream) {
    using namespace talus::prefill::w4i;
    if (T <= 0 || K % kHB != 0 || F <= 0 || (wg != 0 && wg != 128 && wg != 256)) return -1;
    return w4i_gemm_dispatch<true>(wq_gu, ws_gu, wg, 2 * F, K, xq, xk, xS, T, dst, ld, (hipStream_t) stream) ? 0 : -1;
}

/// rd-q4quant-B: build the GU operand of talus_pf_w4i_gemm_gu from a gate / up pair (F x K each, weight groups wg):
/// oq [2 F x K / 2] bytes, os [2 F x (wg ? K / wg : 1)].  0 ok, -1 not covered.
extern "C" int talus_pf_w4i_interleave_gu(const uint8_t* gq, const uint8_t* uq, const float* gs, const float* us,
                                          int wg, int F, int K, uint8_t* oq, float* os, void* stream) {
    using namespace talus::prefill::w4i;
    if (F <= 0 || F % 16 != 0 || K % kHB != 0 || (wg != 0 && wg != 128 && wg != 256)) return -1;
    k_w4i_interleave_gu<<<2 * (F / 16), 256, 0, (hipStream_t) stream>>>((const uint4*) gq, (const uint4*) uq, gs, us,
                                                                       K, wg ? K / wg : 1, (uint4*) oq, os);
    return 0;
}
