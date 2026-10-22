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
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

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

constexpr int kHB = 256;   // Hadamard block
constexpr int kAG = 32;    // activation group

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

/// lane L of a wave holds values 8 L .. 8 L + 7 of a 256-value block: v <- H (D v) / 16 (Sylvester order, tp_w4a4's)
__device__ __forceinline__ void rot256(float v[8], const int8_t* __restrict__ sg, int lane) {
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] *= (float) sg[e];
#pragma unroll
    for (int h = 1; h < 8; h <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (!(e & h)) {
                const float a = v[e], b = v[e + h];
                v[e] = a + b;
                v[e + h] = a - b;
            }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float p = __shfl_xor(v[e], m, 32);
            v[e] = (lane & m) ? p - v[e] : v[e] + p;
        }
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] *= 0.0625f;
}

template <bool SWI>
__device__ __forceinline__ void load8(float v[8], const float* __restrict__ x, const float* __restrict__ g, size_t o) {
    const float4 a0 = *(const float4*) (x + o), a1 = *(const float4*) (x + o + 4);
    v[0] = a0.x; v[1] = a0.y; v[2] = a0.z; v[3] = a0.w; v[4] = a1.x; v[5] = a1.y; v[6] = a1.z; v[7] = a1.w;
    if constexpr (SWI) {   // x = up, g = gate: silu(g) * u with q36_engine's expression
        const float4 b0 = *(const float4*) (g + o), b1 = *(const float4*) (g + o + 4);
        const float gv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] = gv[e] / (1.0f + expf(-gv[e])) * v[e];
    }
}

/// one block (256 threads) per token: pass 1 rotates every 256-value block and keeps the 32-value group maxima in
/// LDS, pass 2 rotates again and packs with the integer group scales.  K <= 32768.
template <bool SWI>
__global__ void __launch_bounds__(256) k_w4i_pack(const float* __restrict__ x, const float* __restrict__ g, int K,
                                                  int64_t ldx, const int8_t* __restrict__ signs,
                                                  uint8_t* __restrict__ xq, uint8_t* __restrict__ xk,
                                                  float* __restrict__ xS, int wb) {
    __shared__ float gmax[1024];
    __shared__ float wmax[8];
    const int t = blockIdx.x, w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NB = K / kHB;
    const size_t row = (size_t) t * ldx;
    float m = 0.f;
    for (int b = w; b < NB; b += 8) {
        const int c0 = b * kHB + lane * 8;
        float v[8];
        load8<SWI>(v, x, g, row + c0);
        rot256(v, signs + c0, lane);
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(v[e]));
        a = fmaxf(a, __shfl_xor(a, 1, 32));
        a = fmaxf(a, __shfl_xor(a, 2, 32));
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
    const float S = m / (7.0f * 127.0f);
    const float inv7S = m > 0.f ? 127.0f / m : 0.f;   // = 1 / (7 S)
    auto kof = [&](float a) { return m > 0.f ? min(127, max(1, (int) ceilf(a * inv7S))) : 1; };
    if (threadIdx.x == 0) xS[t] = S;
    for (int b = w; b < NB; b += 8) {
        const int c0 = b * kHB + lane * 8;
        float v[8];
        load8<SWI>(v, x, g, row + c0);
        rot256(v, signs + c0, lane);
        const int gi = c0 / kAG;
        const int k = kof(gmax[gi]);
        const int kn = ((gi + 1) * kAG) % wb != 0 ? kof(gmax[gi + 1]) : 0;   // (next group, inside the weight block)
        const float inv = m > 0.f ? 1.0f / ((float) k * S) : 0.f;
        uint32_t word = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int q = (int) fminf(fmaxf(rintf(v[e] * inv), -7.f), 7.f);
            word |= (uint32_t) (q & 0xF) << (4 * e);
        }
        *(uint32_t*) (xq + (size_t) t * (K / 2) + c0 / 2) = word;
        if ((lane & 3) == 0) xk[(size_t) t * (K / kAG) + gi] = (uint8_t) (int8_t) (k - kn);
    }
}

/// GEMM: block = WGN (row) x WGT (token) waves, a wave WM x WT 16 x 16 tiles; B (activations + k_g) staged through
/// LDS KS 32-value blocks per stage (double-buffered, one barrier per stage); A (weights) from global straight into
/// registers one block ahead.  The accumulator lane holds token l16, rows 2 i + hi (as tp_i8gemm v10).
template <int WGN, int WGT, int WM, int WT, int KS, int WG>
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
    if constexpr (SPG == 1) {
#pragma unroll 1
        for (int s = 0; s < NS; ++s) stage(s, st[s & 1], st[(s + 1) & 1]);
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

template <int WGN, int WGT, int WM, int WT, int KS, int WG>
bool launch(const uint8_t* wq, const float* ws, int N, int K, const uint8_t* xq, const uint8_t* xk, const float* xS,
            int T, float* dst, int64_t ld, hipStream_t s) {
    constexpr int BN = 16 * WGN * WM, BT = 16 * WGT * WT;
    if (N % BN != 0 || (K / 32) % KS != 0 || (WG && K % WG != 0)) return false;
    if (WG && (K / (32 * KS)) % 2 != 0) return false;
    const dim3 grid((unsigned) ((T + BT - 1) / BT), (unsigned) (N / BN));
    k_w4i_gemm<WGN, WGT, WM, WT, KS, WG><<<grid, 32 * WGN * WGT, 0, s>>>(wq, ws, xq, xk, xS, dst, K, T, ld, N);
    return true;
}

template <int WG>
bool launch_v(int v, const uint8_t* wq, const float* ws, int N, int K, const uint8_t* xq, const uint8_t* xk,
              const float* xS, int T, float* dst, int64_t ld, hipStream_t s) {
    switch (v) {
        case 1: return launch<4, 2, 2, 4, 4, WG>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // 128 x 128 (v106's shape)
        case 2: return launch<4, 2, 2, 4, 2, WG>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // KS 2
        case 3: return launch<2, 4, 4, 2, 4, WG>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // waves 2 x 4, 4 x 2 tiles
        case 4: return launch<4, 2, 4, 4, 4, WG>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // 256 x 128
        case 5: return launch<4, 2, 2, 2, 4, WG>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);   // 128 x 64
        default: return launch<4, 2, 2, 4, 4, WG>(wq, ws, N, K, xq, xk, xS, T, dst, ld, s);
    }
}

}  // namespace
}  // namespace talus::prefill::w4i

/// pack T rows of FP32 activations x [T][ldx] (gate == nullptr) or silu(gate) * x (gate [T][ldx], the SwiGLU
/// input of the down projection) into the w4i layout for a GEMM with weight groups wg (0 / 256).  0 ok, -1 not covered.
extern "C" int talus_pf_w4i_pack(const float* x, const float* gate, int T, int K, int64_t ldx, const int8_t* signs,
                                 int wg, uint8_t* xq, uint8_t* xk, float* xS, void* stream) {
    using namespace talus::prefill::w4i;
    if (T <= 0 || K % kHB != 0 || K > 32768 || ldx % 4 != 0 || (wg != 0 && wg != 256)) return -1;
    const hipStream_t s = (hipStream_t) stream;
    const int wb = wg ? wg : K;   // the Abel blocks = the GEMM's weight groups (it resets its running sums there)
    if (gate) k_w4i_pack<true><<<T, 256, 0, s>>>(x, gate, K, ldx, signs, xq, xk, xS, wb);
    else k_w4i_pack<false><<<T, 256, 0, s>>>(x, nullptr, K, ldx, signs, xq, xk, xS, wb);
    return 0;
}

/// dst [T][ld] = W4A4 product of the packed activations with wq / ws (wg 0: per-row scales, 256: per 256-value
/// weight group).  TALUS_PF_W4I_VARIANT picks the tile shape (bench).  0 ok, -1 shape not covered.
extern "C" int talus_pf_w4i_gemm(const uint8_t* wq, const float* ws, int wg, int N, int K, const uint8_t* xq,
                                 const uint8_t* xk, const float* xS, int T, float* dst, int64_t ld, void* stream) {
    using namespace talus::prefill::w4i;
    if (T <= 0 || K % kHB != 0 || (wg != 0 && wg != 256)) return -1;
    static const int v = [] { const char* e = std::getenv("TALUS_PF_W4I_VARIANT"); return e ? std::atoi(e) : 1; }();
    const hipStream_t s = (hipStream_t) stream;
    const bool ok = wg ? launch_v<256>(v, wq, ws, N, K, xq, xk, xS, T, dst, ld, s)
                       : launch_v<0>(v, wq, ws, N, K, xq, xk, xS, T, dst, ld, s);
    return ok ? 0 : -1;
}
