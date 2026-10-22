// src/prefill/tp_i8gemm.cu - see include/strata/prefill/tp_i8gemm.hpp: the GEMM on expanded weights.
//
// Per output element (weight row n, token t) and 32-value block b, in increasing b: c = the int32 dot of the block
// (two v_wmma_i32_16x16x16_iu8, as MMQ's 16x8-int tile on RDNA3), sum += c * w_scale * x_scale - the expression and
// order of llama.cpp's vec_dot_q8_0_q8_1_mma (D4), so the result is bitwise MMQ's.
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "strata/prefill/tp_i8gemm.hpp"
#include "strata/prefill/tp_tune.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define TP_I8_WMMA 1
#endif

namespace talus::prefill {
namespace {

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8i wmma_i8(v4i a, v4i b, v8i c) {
#if defined(TP_I8_WMMA)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
#else
    (void) a; (void) b;
    return c;
#endif
}

constexpr int kXBlock = 144;   // block_q8_1_mmq: 4 FP32 scales + 128 int8

/// grid (ceil(T / 128), n / 128), block 256: 8 waves = (8 / WGT) (weight rows) x WGT (tokens); a wave owns WM x WT
/// 16 x 16 tiles (128 = 16 WM (8 / WGT) rows, 128 = 16 WT WGT tokens per block).  A = weights (rows n), B = activations
/// (columns t): the C tile's lane holds token l16 and rows 2 i + hi.  kPre: the next 32-value block's operands are
/// loaded while this block's WMMAs run (registers, double-buffered).
template <int WM, int WT, bool kPre>
__global__ void __launch_bounds__(256) k_i8gemm(const int8_t* __restrict__ qs, const float* __restrict__ ws,
                                                const char* __restrict__ xq, float* __restrict__ dst, int K, int T,
                                                int64_t ldd) {
    constexpr int WGT = 128 / (16 * WT), WGN = 8 / WGT;
    static_assert(WGN * WM * 16 == 128, "block rows");
    const int KB = K / 32;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = w % WGN, wt = w / WGN;
    const int m0 = blockIdx.y * 8 + wn * WM;               // first 16-row tile of this wave
    const int t0 = blockIdx.x * 128 + wt * (16 * WT);      // first token of this wave
    float sum[WM][WT][8];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WT; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
    const char* xr[WT];
#pragma unroll
    for (int b = 0; b < WT; ++b) xr[b] = xq + (size_t) min(t0 + b * 16 + l16, T - 1) * kXBlock;
    const int8_t* qa[WM];
    const float* wa[WM];
#pragma unroll
    for (int a = 0; a < WM; ++a) {
        qa[a] = qs + (size_t) (m0 + a) * KB * 512 + l16 * 32;
        wa[a] = ws + (size_t) (m0 + a) * KB * 16 + hi * 8;
    }
    const size_t xstride = (size_t) T * kXBlock;   // one 128-value block row of all tokens
    struct Ops {
        v4i A[WM][2];
        float4 dw[WM][2];
        v4i B[WT][2];
        float dx[WT];
    };
    auto load = [&](int kb, Ops& o) {
#pragma unroll
        for (int a = 0; a < WM; ++a) {
            const int8_t* p = qa[a] + (size_t) kb * 512;
            o.A[a][0] = *(const v4i*) p;
            o.A[a][1] = *(const v4i*) (p + 16);
            o.dw[a][0] = *(const float4*) (wa[a] + (size_t) kb * 16);
            o.dw[a][1] = *(const float4*) (wa[a] + (size_t) kb * 16 + 4);
        }
        const size_t xo = (size_t) (kb >> 2) * xstride;
        const int sub = kb & 3;
#pragma unroll
        for (int b = 0; b < WT; ++b) {
            const char* p = xr[b] + xo;
            o.B[b][0] = *(const v4i*) (p + 16 + sub * 32);
            o.B[b][1] = *(const v4i*) (p + 32 + sub * 32);
            o.dx[b] = ((const float*) p)[sub];
        }
    };
    auto compute = [&](const Ops& o) {
#pragma unroll
        for (int a = 0; a < WM; ++a) {
            const float dw[8] = {o.dw[a][0].x, o.dw[a][0].y, o.dw[a][0].z, o.dw[a][0].w,
                                 o.dw[a][1].x, o.dw[a][1].y, o.dw[a][1].z, o.dw[a][1].w};
#pragma unroll
            for (int b = 0; b < WT; ++b) {
                v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
                c = wmma_i8(o.A[a][0], o.B[b][0], c);
                c = wmma_i8(o.A[a][1], o.B[b][1], c);
#pragma unroll
                for (int i = 0; i < 8; ++i) sum[a][b][i] += c[i] * dw[i] * o.dx[b];
            }
        }
    };
    if constexpr (kPre) {
        Ops cur, nxt;
        load(0, cur);
        for (int kb = 0; kb < KB; ++kb) {
            if (kb + 1 < KB) load(kb + 1, nxt);
            compute(cur);
            cur = nxt;
        }
    } else {
        for (int kb = 0; kb < KB; ++kb) {
            Ops cur;
            load(kb, cur);
            compute(cur);
        }
    }
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = t0 + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[(size_t) t * ldd + (m0 + a) * 16 + 2 * i + hi] = sum[a][b][i];
    }
}

/// Variant 6: the block's operands staged through LDS, 64 columns (two 32-value blocks) per stage, double-buffered; the
/// next stage's global loads issued into registers while this stage's WMMAs run.  8 waves = 4 (rows) x 2 (tokens), a
/// wave 2 x 4 tiles.  MODE (the MMQ dot it reproduces, per output element and block in increasing order):
///   0  vec_dot_q8_0_q8_1 (D4):     sum += c32 * w_scale * x_d
///   1  vec_dot_q8_0_16_q8_1 (D4):  sum += c16a * w_scale_a * x_d;  sum += c16b * w_scale_b * x_d
///   2  vec_dot_q8_1_q8_1 (DS4):    sum += w_d * x_d * c32;  sum += w_m * x_s
template <int MODE, int kVar = 6>
__global__ void __launch_bounds__(256) k_i8gemm_lds(const int8_t* __restrict__ qs, const float* __restrict__ ws,
                                                    const char* __restrict__ xq, float* __restrict__ dst, int K, int T,
                                                    int64_t ldd) {
    constexpr int WM = 2, WT = 4, SC = MODE == 0 ? 1 : 2;
    struct Stage {
        int8_t a[8][2][16][32];   // 8 KB
        float as[8][2][SC][16];   // [row tile][block][scale][rows 0,2,..,14 then 1,3,..,15]
        int8_t b[2][128][32];     // 8 KB
        float bs[2][SC][128];     // x_d (and x_s for MODE 2)
    };
    __shared__ __attribute__((aligned(16))) Stage st[2];
    const int KB = K / 32, NS = KB / 2;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = w & 3, wt = w >> 2;
    const int mb0 = blockIdx.y * 8;          // the block's first 16-row tile
    const int tb0 = blockIdx.x * 128;        // the block's first token
    // global -> register staging: A: thread = (row tile, block, row) 32 bytes; B: thread = (token, block) 32 bytes + scale
    const int a_mt = tid >> 5, a_r = (tid & 31) >> 1, a_kb = tid & 1;   // 8 tiles x 16 rows x 2 blocks
    const int b_t = tid >> 1, b_kb = tid & 1;                            // 128 tokens x 2 blocks
    const int a_so = (a_r & 1) * 8 + (a_r >> 1);                         // (the expanded scales are stored in that order)
    const int8_t* ga = qs + ((size_t) (mb0 + a_mt) * KB) * 512 + a_r * 32;
    const float* gas = ws + ((size_t) (mb0 + a_mt) * KB) * 16 * SC;
    const char* gb = xq + (size_t) min(tb0 + b_t, T - 1) * kXBlock;
    const size_t xstride = (size_t) T * kXBlock;
    v4i ra[2], rb[2];
    float ras[SC], rbs[SC];
    auto gload = [&](int s) {
        const int kb = 2 * s + a_kb;
        const int8_t* p = ga + (size_t) kb * 512;
        ra[0] = *(const v4i*) p;
        ra[1] = *(const v4i*) (p + 16);
#pragma unroll
        for (int j = 0; j < SC; ++j) ras[j] = gas[((size_t) kb * SC + j) * 16 + a_so];
        const int kbb = 2 * s + b_kb;
        const char* q = gb + (size_t) (kbb >> 2) * xstride;
        rb[0] = *(const v4i*) (q + 16 + (kbb & 3) * 32);
        rb[1] = *(const v4i*) (q + 32 + (kbb & 3) * 32);
        if constexpr (MODE == 2) {
            const float2 ds = __half22float2(((const __half2*) q)[kbb & 3]);
            rbs[0] = ds.x;
            rbs[SC - 1] = ds.y;
        } else {
            rbs[0] = ((const float*) q)[kbb & 3];
        }
    };
    auto lstore = [&](Stage& S) {
        *(v4i*) &S.a[a_mt][a_kb][a_r][0] = ra[0];
        *(v4i*) &S.a[a_mt][a_kb][a_r][16] = ra[1];
#pragma unroll
        for (int j = 0; j < SC; ++j) S.as[a_mt][a_kb][j][a_so] = ras[j];
        *(v4i*) &S.b[b_kb][b_t][0] = rb[0];
        *(v4i*) &S.b[b_kb][b_t][16] = rb[1];
        S.bs[b_kb][0][b_t] = rbs[0];
        if constexpr (MODE == 2) S.bs[b_kb][SC - 1][b_t] = rbs[SC - 1];
    };
    float sum[WM][WT][8];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WT; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
    gload(0);
    lstore(st[0]);
    __syncthreads();
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const bool next = s + 1 < NS;
        if (kVar != 8 && next) gload(s + 1);
        const Stage& S = st[s & 1];
#pragma unroll 1
        for (int kk = 0; kk < 2; ++kk) {
            v4i B[WT][2];
            float dx[WT], sx[WT];
#pragma unroll
            for (int b = 0; b < WT; ++b) {
                const int t = wt * 64 + b * 16 + l16;
                B[b][0] = *(const v4i*) &S.b[kk][t][0];
                B[b][1] = *(const v4i*) &S.b[kk][t][16];
                dx[b] = S.bs[kk][0][t];
                sx[b] = MODE == 2 ? S.bs[kk][SC - 1][t] : 0.f;
            }
#pragma unroll
            for (int a = 0; a < WM; ++a) {
                const int mt = wn * WM + a;
                const v4i A0 = *(const v4i*) &S.a[mt][kk][l16][0], A1 = *(const v4i*) &S.a[mt][kk][l16][16];
                float dw[SC][8];
#pragma unroll
                for (int j = 0; j < SC; ++j) {
                    const float4 s0 = *(const float4*) &S.as[mt][kk][j][hi * 8], s1 = *(const float4*) &S.as[mt][kk][j][hi * 8 + 4];
                    dw[j][0] = s0.x; dw[j][1] = s0.y; dw[j][2] = s0.z; dw[j][3] = s0.w;
                    dw[j][4] = s1.x; dw[j][5] = s1.y; dw[j][6] = s1.z; dw[j][7] = s1.w;
                }
                if constexpr (kVar >= 7) {
                    // v7: the int32 dot is accumulated onto 0x4B400000 (1.5 * 2^23) so that its bits read as the FP32
                    // value 12582912 + c, exact for |c| < 2^22 (here |c| <= 32 * 128 * 128); subtracting 12582912
                    // gives (float) c exactly - a VOPD-pairable add instead of v_cvt_f32_i32.  All WT tiles' WMMAs are
                    // issued first, then each epilogue stage over all 8 x WT independent values (no dependent chains,
                    // so the compiler pairs them into v_dual_*).  Same operations and rounding per element as below.
                    constexpr int kB = 0x4B400000;
                    constexpr float kM = 12582912.0f;
                    if constexpr (MODE == 1) {
                        v8i c0[WT], c1[WT];
#pragma unroll
                        for (int b = 0; b < WT; ++b) {
                            c0[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                            c1[b] = c0[b];
                            c0[b] = wmma_i8(A0, B[b][0], c0[b]);
                            c1[b] = wmma_i8(A1, B[b][1], c1[b]);
                        }
                        float f0[WT][8], f1[WT][8];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                f0[b][i] = (__int_as_float(c0[b][i]) - kM) * dw[0][i];
                                f1[b][i] = (__int_as_float(c1[b][i]) - kM) * dw[1][i];
                            }
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                sum[a][b][i] = __builtin_fmaf(f0[b][i], dx[b], sum[a][b][i]);
                                sum[a][b][i] = __builtin_fmaf(f1[b][i], dx[b], sum[a][b][i]);
                            }
                    } else {
                        v8i c[WT];
#pragma unroll
                        for (int b = 0; b < WT; ++b) {
                            c[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                            c[b] = wmma_i8(A0, B[b][0], c[b]);
                            c[b] = wmma_i8(A1, B[b][1], c[b]);
                        }
                        float f[WT][8];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) f[b][i] = __int_as_float(c[b][i]) - kM;
                        if constexpr (MODE == 2) {
#pragma unroll
                            for (int b = 0; b < WT; ++b)
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    sum[a][b][i] = __builtin_fmaf(dw[0][i] * dx[b], f[b][i], sum[a][b][i]);
                                    sum[a][b][i] = __builtin_fmaf(dw[SC - 1][i], sx[b], sum[a][b][i]);
                                }
                        } else {
#pragma unroll
                            for (int b = 0; b < WT; ++b)
#pragma unroll
                                for (int i = 0; i < 8; ++i) f[b][i] *= dw[0][i];
#pragma unroll
                            for (int b = 0; b < WT; ++b)
#pragma unroll
                                for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                        }
                    }
                } else
#pragma unroll
                for (int b = 0; b < WT; ++b) {
                    if constexpr (MODE == 1) {   // two 16-value blocks, each its own scale
                        v8i c0 = {0, 0, 0, 0, 0, 0, 0, 0}, c1 = {0, 0, 0, 0, 0, 0, 0, 0};
                        c0 = wmma_i8(A0, B[b][0], c0);
                        c1 = wmma_i8(A1, B[b][1], c1);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] += c0[i] * dw[0][i] * dx[b];
                            sum[a][b][i] += c1[i] * dw[1][i] * dx[b];
                        }
                    } else {
                        v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
                        c = wmma_i8(A0, B[b][0], c);
                        c = wmma_i8(A1, B[b][1], c);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            if constexpr (MODE == 2) {
                                sum[a][b][i] += dw[0][i] * dx[b] * c[i];
                                sum[a][b][i] += dw[SC - 1][i] * sx[b];
                            } else {
                                sum[a][b][i] += c[i] * dw[0][i] * dx[b];
                            }
                        }
                    }
                }
            }
#if defined(TP_I8_WMMA)
            if constexpr (kVar < 7) {
            // interleave each tile's scale epilogue (VALU) with the next tiles' WMMAs instead of all 16 WMMAs then all
            // the VALU (a wave's VALU would otherwise wait for the last WMMA)
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
#pragma unroll
            for (int j = 0; j < 14; ++j) {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x002, 10, 0);
            }
            }
#endif
        }
        if (kVar != 9) {
            if (kVar != 96 && next) lstore(st[(s + 1) & 1]);
            if (kVar != 97) __syncthreads();
        }
    }
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = tb0 + wt * 64 + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[(size_t) t * ldd + (mb0 + wn * WM + a) * 16 + 2 * i + hi] = sum[a][b][i];
    }
}


/// v8: the LDS kernel generalized over the block shape, with the v7 epilogue and bank-conflict-free LDS layouts.
/// Block = WGN (rows) x WGT (tokens) waves, a wave WM x WT 16 x 16 tiles: BN = 16 WGN WM rows, BT = 16 WGT WT tokens.
/// KS 32-value blocks per LDS stage (double-buffered, one barrier per stage; the next stage's global loads are issued
/// into registers before this stage's WMMAs).  LDS: A as [tile][block][half][16 rows][16 B], B as [block][half][token]
/// [16 B], so a ds_read_b128 by 16 lanes touches 256 consecutive bytes.  Per output element and block, in increasing
/// block order, the same arithmetic as k_i8gemm_lds (bitwise MMQ).
template <int MODE, int WGN, int WGT, int WM, int WT, int KS>
__global__ void __launch_bounds__(32 * WGN * WGT) k_i8v8(const int8_t* __restrict__ qs, const float* __restrict__ ws,
                                                        const char* __restrict__ xq, float* __restrict__ dst, int K,
                                                        int T, int64_t ldd) {
    constexpr int NT = 32 * WGN * WGT, SC = MODE == 0 ? 1 : 2, BS = MODE == 2 ? 2 : 1;
    constexpr int MT = WGN * WM, BT = 16 * WGT * WT;   // row tiles, tokens per block
    struct Stage {
        int8_t a[MT][KS][2][16][16];
        float as[MT][KS][SC][16];
        int8_t b[KS][2][BT][16];
        float bs[KS][BS][BT];
    };
    __shared__ __attribute__((aligned(16))) Stage st[2];
    const int KB = K / 32, NS = KB / KS;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = w % WGN, wt = w / WGN;
    const int mb0 = blockIdx.y * MT;
    const int tb0 = blockIdx.x * BT;
    // staging units (16 B each): A: MT * KS * 32 (tile, block, row, half); B: KS * BT * 2 (block, token, half);
    // A scales: MT * KS * SC * 4 float4; B scales: KS * BT (one float or half2)
    constexpr int NA = MT * KS * 32, NB = KS * BT * 2, NAS = MT * KS * SC * 4, NBS = KS * BT;
    constexpr int RA = (NA + NT - 1) / NT, RB = (NB + NT - 1) / NT, RAS = (NAS + NT - 1) / NT, RBS = (NBS + NT - 1) / NT;
    v4i ra[RA], rb[RB];
    float4 ras[RAS];
    float rbs[RBS][BS];
    const size_t xstride = (size_t) T * kXBlock;
    auto gload = [&](int s) {
        const int kb0 = s * KS;
#pragma unroll
        for (int j = 0; j < RA; ++j) {
            const int u = tid + j * NT;
            if (NA % NT == 0 || u < NA) {
                const int h = u & 1, r = (u >> 1) & 15, kk = (u >> 5) % KS, mt = (u >> 5) / KS;
                ra[j] = *(const v4i*) (qs + ((size_t) (mb0 + mt) * KB + kb0 + kk) * 512 + r * 32 + h * 16);
            }
        }
#pragma unroll
        for (int j = 0; j < RAS; ++j) {
            const int u = tid + j * NT;
            if (NAS % NT == 0 || u < NAS) {
                const int q = u % (SC * 4), kk = (u / (SC * 4)) % KS, mt = u / (SC * 4 * KS);
                ras[j] = *(const float4*) (ws + ((size_t) (mb0 + mt) * KB + kb0 + kk) * 16 * SC + q * 4);
            }
        }
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int u = tid + j * NT;
            if (NB % NT == 0 || u < NB) {
                const int h = u & 1, t = (u >> 1) % BT, kk = (u >> 1) / BT, kb = kb0 + kk;
                const char* q = xq + (size_t) min(tb0 + t, T - 1) * kXBlock + (size_t) (kb >> 2) * xstride;
                rb[j] = *(const v4i*) (q + 16 + (kb & 3) * 32 + h * 16);
            }
        }
#pragma unroll
        for (int j = 0; j < RBS; ++j) {
            const int u = tid + j * NT;
            if (NBS % NT == 0 || u < NBS) {
                const int t = u % BT, kk = u / BT, kb = kb0 + kk;
                const char* q = xq + (size_t) min(tb0 + t, T - 1) * kXBlock + (size_t) (kb >> 2) * xstride;
                if constexpr (MODE == 2) {
                    const float2 ds = __half22float2(((const __half2*) q)[kb & 3]);
                    rbs[j][0] = ds.x;
                    rbs[j][BS - 1] = ds.y;
                } else {
                    rbs[j][0] = ((const float*) q)[kb & 3];
                }
            }
        }
    };
    auto lstore = [&](Stage& S) {
#pragma unroll
        for (int j = 0; j < RA; ++j) {
            const int u = tid + j * NT;
            if (NA % NT == 0 || u < NA) {
                const int h = u & 1, r = (u >> 1) & 15, kk = (u >> 5) % KS, mt = (u >> 5) / KS;
                *(v4i*) &S.a[mt][kk][h][r][0] = ra[j];
            }
        }
#pragma unroll
        for (int j = 0; j < RAS; ++j) {
            const int u = tid + j * NT;
            if (NAS % NT == 0 || u < NAS) {
                const int q = u % (SC * 4), kk = (u / (SC * 4)) % KS, mt = u / (SC * 4 * KS);
                *(float4*) &S.as[mt][kk][q >> 2][(q & 3) * 4] = ras[j];
            }
        }
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int u = tid + j * NT;
            if (NB % NT == 0 || u < NB) {
                const int h = u & 1, t = (u >> 1) % BT, kk = (u >> 1) / BT;
                *(v4i*) &S.b[kk][h][t][0] = rb[j];
            }
        }
#pragma unroll
        for (int j = 0; j < RBS; ++j) {
            const int u = tid + j * NT;
            if (NBS % NT == 0 || u < NBS) {
                const int t = u % BT, kk = u / BT;
#pragma unroll
                for (int q = 0; q < BS; ++q) S.bs[kk][q][t] = rbs[j][q];
            }
        }
    };
    float sum[WM][WT][8];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WT; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
    constexpr int kB = 0x4B400000;   // see k_i8gemm_lds v7: bits of 12582912 + c, exact for |c| < 2^22
    constexpr float kM = 12582912.0f;
    gload(0);
    lstore(st[0]);
    __syncthreads();
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const bool next = s + 1 < NS;
        if (next) gload(s + 1);
        const Stage& S = st[s & 1];
#pragma unroll 1
        for (int kk = 0; kk < KS; ++kk) {
            v4i B[WT][2];
            float dx[WT], sx[WT];
#pragma unroll
            for (int b = 0; b < WT; ++b) {
                const int t = wt * 16 * WT + b * 16 + l16;
                B[b][0] = *(const v4i*) &S.b[kk][0][t][0];
                B[b][1] = *(const v4i*) &S.b[kk][1][t][0];
                dx[b] = S.bs[kk][0][t];
                sx[b] = MODE == 2 ? S.bs[kk][BS - 1][t] : 0.f;
            }
#pragma unroll
            for (int a = 0; a < WM; ++a) {
                const int mt = wn * WM + a;
                const v4i A0 = *(const v4i*) &S.a[mt][kk][0][l16][0], A1 = *(const v4i*) &S.a[mt][kk][1][l16][0];
                float dw[SC][8];
#pragma unroll
                for (int j = 0; j < SC; ++j) {
                    const float4 s0 = *(const float4*) &S.as[mt][kk][j][hi * 8], s1 = *(const float4*) &S.as[mt][kk][j][hi * 8 + 4];
                    dw[j][0] = s0.x; dw[j][1] = s0.y; dw[j][2] = s0.z; dw[j][3] = s0.w;
                    dw[j][4] = s1.x; dw[j][5] = s1.y; dw[j][6] = s1.z; dw[j][7] = s1.w;
                }
                if constexpr (MODE == 1) {   // the two 16-value halves in turn (same per-element order)
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        v8i c[WT];
#pragma unroll
                        for (int b = 0; b < WT; ++b) {
                            c[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                            c[b] = wmma_i8(hh ? A1 : A0, B[b][hh], c[b]);
                        }
                        float f[WT][8];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) f[b][i] = (__int_as_float(c[b][i]) - kM) * dw[hh][i];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                    }
                } else {
                    v8i c[WT];
#pragma unroll
                    for (int b = 0; b < WT; ++b) {
                        c[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                        c[b] = wmma_i8(A0, B[b][0], c[b]);
                        c[b] = wmma_i8(A1, B[b][1], c[b]);
                    }
                    float f[WT][8];
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[b][i] = __int_as_float(c[b][i]) - kM;
                    if constexpr (MODE == 2) {
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                sum[a][b][i] = __builtin_fmaf(dw[0][i] * dx[b], f[b][i], sum[a][b][i]);
                                sum[a][b][i] = __builtin_fmaf(dw[SC - 1][i], sx[b], sum[a][b][i]);
                            }
                    } else {
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) f[b][i] *= dw[0][i];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                    }
                }
            }
        }
        if (next) lstore(st[(s + 1) & 1]);
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = tb0 + wt * 16 * WT + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[(size_t) t * ldd + (mb0 + wn * WM + a) * 16 + 2 * i + hi] = sum[a][b][i];
    }
}

template <int WGN, int WGT, int WM, int WT, int KS>
bool launch_v8(const I8Mat& w, const char* x, int T, float* dst, int64_t ld, hipStream_t s) {
    constexpr int BN = 16 * WGN * WM, BT = 16 * WGT * WT;
    if (w.n % BN != 0 || (w.k / 32) % KS != 0) return false;
    const dim3 grid((unsigned) ((T + BT - 1) / BT), (unsigned) (w.n / BN));
    constexpr int NT = 32 * WGN * WGT;
    if (w.mode == 1) k_i8v8<1, WGN, WGT, WM, WT, KS><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld);
    else if (w.mode == 2) k_i8v8<2, WGN, WGT, WM, WT, KS><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld);
    else k_i8v8<0, WGN, WGT, WM, WT, KS><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld);
    return true;
}


/// v9: A (weights) straight from global memory into the WMMA operands (each wave its own WM row tiles, the next
/// block's A and scales prefetched into registers while this block's WMMAs run); only B (activations, shared by the
/// WGN row waves) goes through LDS, KS blocks per stage, double-buffered, one barrier per stage.  Halves the LDS stores
/// and (KS 4) the barriers of v7.  Block = WGN x WGT waves, WM x WT tiles per wave.  Same per-element arithmetic.
template <int MODE, int WGN, int WGT, int WM, int WT, int KS>
__global__ void __launch_bounds__(32 * WGN * WGT) k_i8v9(const int8_t* __restrict__ qs, const float* __restrict__ ws,
                                                        const char* __restrict__ xq, float* __restrict__ dst, int K,
                                                        int T, int64_t ldd) {
    constexpr int NT = 32 * WGN * WGT, SC = MODE == 0 ? 1 : 2, BS = MODE == 2 ? 2 : 1;
    constexpr int MT = WGN * WM, BT = 16 * WGT * WT;
    constexpr bool kPS = MODE == 0;   // prefetch the A scales with A (mode 0; modes 1 / 2 would spill)
    struct Stage {
        int8_t b[KS][2][BT][16];
        float bs[KS][BS][BT];
    };
    __shared__ __attribute__((aligned(16))) Stage st[2];
    const int KB = K / 32, NS = KB / KS;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = w % WGN, wt = w / WGN;
    const int mb0 = blockIdx.y * MT;
    const int tb0 = blockIdx.x * BT;
    constexpr int NB = KS * BT * 2, NBS = KS * BT;
    constexpr int RB = (NB + NT - 1) / NT, RBS = (NBS + NT - 1) / NT;
    v4i rb[RB];
    float rbs[RBS][BS];
    const size_t xstride = (size_t) T * kXBlock;
    auto gload = [&](int s) {
        const int kb0 = s * KS;
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int u = tid + j * NT;
            if (NB % NT == 0 || u < NB) {
                const int h = u & 1, t = (u >> 1) % BT, kk = (u >> 1) / BT, kb = kb0 + kk;
                const char* q = xq + (size_t) min(tb0 + t, T - 1) * kXBlock + (size_t) (kb >> 2) * xstride;
                rb[j] = *(const v4i*) (q + 16 + (kb & 3) * 32 + h * 16);
            }
        }
#pragma unroll
        for (int j = 0; j < RBS; ++j) {
            const int u = tid + j * NT;
            if (NBS % NT == 0 || u < NBS) {
                const int t = u % BT, kk = u / BT, kb = kb0 + kk;
                const char* q = xq + (size_t) min(tb0 + t, T - 1) * kXBlock + (size_t) (kb >> 2) * xstride;
                if constexpr (MODE == 2) {
                    const float2 ds = __half22float2(((const __half2*) q)[kb & 3]);
                    rbs[j][0] = ds.x;
                    rbs[j][BS - 1] = ds.y;
                } else {
                    rbs[j][0] = ((const float*) q)[kb & 3];
                }
            }
        }
    };
    auto lstore = [&](Stage& S) {
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int u = tid + j * NT;
            if (NB % NT == 0 || u < NB) {
                const int h = u & 1, t = (u >> 1) % BT, kk = (u >> 1) / BT;
                *(v4i*) &S.b[kk][h][t][0] = rb[j];
            }
        }
#pragma unroll
        for (int j = 0; j < RBS; ++j) {
            const int u = tid + j * NT;
            if (NBS % NT == 0 || u < NBS) {
                const int t = u % BT, kk = u / BT;
#pragma unroll
                for (int q = 0; q < BS; ++q) S.bs[kk][q][t] = rbs[j][q];
            }
        }
    };
    // this wave's A tiles: lane l16 reads row l16 (lanes 16-31 the same rows), scales of rows 2 i + hi
    const int8_t* ga[WM];
    const float* gs[WM];
#pragma unroll
    for (int a = 0; a < WM; ++a) {
        ga[a] = qs + (size_t) (mb0 + wn * WM + a) * KB * 512 + l16 * 32;
        gs[a] = ws + (size_t) (mb0 + wn * WM + a) * KB * 16 * SC + hi * 8;
    }
    struct AOps {
        v4i A[WM][2];
        float4 d[kPS ? WM : 1][SC][2];
    };
    auto aload = [&](int kb, AOps& o) {
#pragma unroll
        for (int a = 0; a < WM; ++a) {
            const int8_t* p = ga[a] + (size_t) kb * 512;
            o.A[a][0] = *(const v4i*) p;
            o.A[a][1] = *(const v4i*) (p + 16);
            if constexpr (kPS) {
#pragma unroll
                for (int j = 0; j < SC; ++j) {
                    const float* q = gs[a] + ((size_t) kb * SC + j) * 16;
                    o.d[a][j][0] = *(const float4*) q;
                    o.d[a][j][1] = *(const float4*) (q + 4);
                }
            }
        }
    };
    float sum[WM][WT][8];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WT; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
    constexpr int kB = 0x4B400000;   // see k_i8gemm_lds v7
    constexpr float kM = 12582912.0f;
    AOps cur, nxt;
    aload(0, cur);
    gload(0);
    lstore(st[0]);
    __syncthreads();
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const bool next = s + 1 < NS;
        if (next) gload(s + 1);
        const Stage& S = st[s & 1];
#pragma unroll 1
        for (int kk = 0; kk < KS; ++kk) {
            const int kb = s * KS + kk;
            if (kb + 1 < KB) aload(kb + 1, nxt);
            v4i B[WT][2];
            float dx[WT], sx[WT];
#pragma unroll
            for (int b = 0; b < WT; ++b) {
                const int t = wt * 16 * WT + b * 16 + l16;
                B[b][0] = *(const v4i*) &S.b[kk][0][t][0];
                B[b][1] = *(const v4i*) &S.b[kk][1][t][0];
                dx[b] = S.bs[kk][0][t];
                sx[b] = MODE == 2 ? S.bs[kk][BS - 1][t] : 0.f;
            }
#pragma unroll
            for (int a = 0; a < WM; ++a) {
                const v4i A0 = cur.A[a][0], A1 = cur.A[a][1];
                float dw[SC][8];
#pragma unroll
                for (int j = 0; j < SC; ++j) {
                    float4 s0, s1;
                    if constexpr (kPS) {
                        s0 = cur.d[a][j][0];
                        s1 = cur.d[a][j][1];
                    } else {   // (modes 1 / 2: the scales at use, to stay in registers)
                        const float* q = gs[a] + ((size_t) kb * SC + j) * 16;
                        s0 = *(const float4*) q;
                        s1 = *(const float4*) (q + 4);
                    }
                    dw[j][0] = s0.x; dw[j][1] = s0.y; dw[j][2] = s0.z; dw[j][3] = s0.w;
                    dw[j][4] = s1.x; dw[j][5] = s1.y; dw[j][6] = s1.z; dw[j][7] = s1.w;
                }
                if constexpr (MODE == 1) {
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        v8i c[WT];
#pragma unroll
                        for (int b = 0; b < WT; ++b) {
                            c[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                            c[b] = wmma_i8(hh ? A1 : A0, B[b][hh], c[b]);
                        }
                        float f[WT][8];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) f[b][i] = (__int_as_float(c[b][i]) - kM) * dw[hh][i];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                    }
                } else {
                    v8i c[WT];
#pragma unroll
                    for (int b = 0; b < WT; ++b) {
                        c[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                        c[b] = wmma_i8(A0, B[b][0], c[b]);
                        c[b] = wmma_i8(A1, B[b][1], c[b]);
                    }
                    float f[WT][8];
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[b][i] = __int_as_float(c[b][i]) - kM;
                    if constexpr (MODE == 2) {
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                sum[a][b][i] = __builtin_fmaf(dw[0][i] * dx[b], f[b][i], sum[a][b][i]);
                                sum[a][b][i] = __builtin_fmaf(dw[SC - 1][i], sx[b], sum[a][b][i]);
                            }
                    } else {
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) f[b][i] *= dw[0][i];
#pragma unroll
                        for (int b = 0; b < WT; ++b)
#pragma unroll
                            for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                    }
                }
            }
            cur = nxt;
        }
        if (next) lstore(st[(s + 1) & 1]);
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = tb0 + wt * 16 * WT + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[(size_t) t * ldd + (mb0 + wn * WM + a) * 16 + 2 * i + hi] = sum[a][b][i];
    }
}

template <int WGN, int WGT, int WM, int WT, int KS>
bool launch_v9(const I8Mat& w, const char* x, int T, float* dst, int64_t ld, hipStream_t s) {
    constexpr int BN = 16 * WGN * WM, BT = 16 * WGT * WT;
    if (w.n % BN != 0 || (w.k / 32) % KS != 0) return false;
    const dim3 grid((unsigned) ((T + BT - 1) / BT), (unsigned) (w.n / BN));
    constexpr int NT = 32 * WGN * WGT;
    if (w.mode == 1) k_i8v9<1, WGN, WGT, WM, WT, KS><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld);
    else if (w.mode == 2) k_i8v9<2, WGN, WGT, WM, WT, KS><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld);
    else k_i8v9<0, WGN, WGT, WM, WT, KS><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld);
    return true;
}


/// Buffer resource over [p, p + bytes) (gfx11 raw buffer, out-of-range reads return 0).
__device__ __forceinline__ __amdgpu_buffer_rsrc_t tp_rsrc(const void* p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short) 0, (int) bytes, 0x31004000);
}
__device__ __forceinline__ v4i tp_ld16(__amdgpu_buffer_rsrc_t r, uint32_t voff, uint32_t soff) {
    return __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0);
}
__device__ __forceinline__ uint32_t tp_ld4(__amdgpu_buffer_rsrc_t r, uint32_t voff, uint32_t soff) {
    return __builtin_amdgcn_raw_buffer_load_b32(r, voff, soff, 0);
}

/// v10: v9's dataflow (A + scales from global straight into registers, one block ahead; B through LDS, KS blocks
/// per stage, double-buffered, one barrier per stage) with the per-iteration overhead removed: every global load is a
/// buffer load whose per-lane offset is fixed for the whole kernel and whose K offset is a scalar (no 64-bit address
/// VALU), the K loop is unrolled two stages at a time (LDS buffer and A ring index are compile-time: no register copies).
/// The VALU work left per 32-value block is the epilogue.  Same per-element arithmetic and order as v9 (bitwise MMQ).
/// rb-q27: the SwiGLU epilogue's operands (EPI 1 / 2): this GEMM is the up projection, g [T][ldg] the gate projection's
/// FP32 output, yq the q8_1 MMQ blocks (layout D4 for EPI 1, DS4 for EPI 2) of silu(g) * u for the down projection.
struct SwiQ8 {
    const float* g = nullptr;
    int64_t ldg = 0;
    char* yq = nullptr;
};

__device__ __forceinline__ float swq8_silu(float v) { return v / (1.0f + expf(-v)); }   // = tp_fuse.cu's silu_f

/// float -> half bits, IEEE round-to-nearest-even (= the hardware conversion of k_swiglu_q8's make_half2; NaN payloads
/// aside): integer ops only - the true16 conversion made the register allocator spill k_i8v10_swq8's main loop.
/// Checked against numpy's float16 cast on all 2^31 non-negative floats (0 mismatches).
__device__ __forceinline__ uint32_t swq8_f2h(float x) {
    const uint32_t f = __float_as_uint(x), sg = (f >> 16) & 0x8000u, a = f & 0x7fffffffu;
    uint32_t h;
    if (a > 0x7f800000u) h = 0x7e00u | ((a >> 13) & 0x3ffu);                       // NaN
    else if (a >= 0x477ff000u) h = 0x7c00u;                                          // >= 65520: inf
    else if (a >= 0x38800000u) h = (a - 0x38000000u + 0xfffu + ((a >> 13) & 1u)) >> 13;   // normal half
    else if (a > 0x33000000u) {                                                      // subnormal half
        const uint32_t sh = 126u - (a >> 23), m = (a & 0x7fffffu) | 0x800000u;
        const uint32_t q = m >> sh, rem = m & ((1u << sh) - 1u), hf = 1u << (sh - 1u);
        h = q + ((rem > hf || (rem == hf && (q & 1u))) ? 1u : 0u);
    } else h = 0u;
    return sg | h;
}

/// The SwiGLU + q8_1 epilogue for one 16-token tile b of a wave whose two row tiles (WM == 2) are 32 consecutive,
/// 32-aligned output columns c0 .. c0 + 31 = one q8_1 scale group.  Bitwise tp_fuse.cu's k_swiglu_q8 on the stored
/// g / u: there, thread k of an 8-thread group holds columns 4k .. 4k + 3 of the group, x = silu(g) * u, amax by
/// xor-shuffles (max: order-free), sum4 = ((x0 + x1) + x2) + x3, the 32-sum by xor 4, 2, 1 shuffles, i.e. (fp32 add
/// commutes) ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7)); d = 1 / (127 / amax), q = roundf(x * (127 / amax)).
/// Here lane (l16, hi) holds rows 2 i + hi of both tiles; one xor-16 exchange gives lane hi the 16 columns
/// c0 + 16 hi .. + 15 in order, and the same expressions are evaluated in the same order.
template <int EPI>
__device__ __forceinline__ void swq8_store(const float (&s0)[8], const float (&s1)[8], int hi, int t, int T, int c0,
                                           const SwiQ8& e) {
    // no FMA contraction: in k_swiglu_q8 the products reach the adds through a phi (no contraction there either), so
    // every mul / add below must stay a separate rounding step (silu(g) * u, sum4, x * d_inv before roundf)
#pragma clang fp contract(off)
    float mine[8], recv[8], u[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        mine[i] = hi ? s1[i] : s0[i];
        recv[i] = __shfl_xor(hi ? s0[i] : s1[i], 16, 32);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        u[2 * i] = hi ? recv[i] : mine[i];
        u[2 * i + 1] = hi ? mine[i] : recv[i];
    }
    const int tl = min(t, T - 1);
    const float4* gp = (const float4*) (e.g + (size_t) tl * e.ldg + c0 + 16 * hi);
    float x[16];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 gv = gp[j];
        x[4 * j + 0] = swq8_silu(gv.x) * u[4 * j + 0];
        x[4 * j + 1] = swq8_silu(gv.y) * u[4 * j + 1];
        x[4 * j + 2] = swq8_silu(gv.z) * u[4 * j + 2];
        x[4 * j + 3] = swq8_silu(gv.w) * u[4 * j + 3];
    }
    float amax = fabsf(x[0]);
#pragma unroll
    for (int j = 1; j < 16; ++j) amax = fmaxf(amax, fabsf(x[j]));
    amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
    float sum = 0.f;
    if constexpr (EPI == 2) {
        float q4[4], o4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            q4[j] = x[4 * j] + x[4 * j + 1];
            q4[j] = q4[j] + x[4 * j + 2];
            q4[j] = q4[j] + x[4 * j + 3];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) o4[j] = __shfl_xor(q4[j], 16, 32);
        // a = s0 + s4 etc. (each lane adds its own and its partner's sum4; fp32 add commutes, so both lanes get the
        // same bits, and those of k_swiglu_q8's xor-4 step)
        const float a = q4[0] + o4[0], b = q4[2] + o4[2], c = q4[1] + o4[1], d = q4[3] + o4[3];
        const float ab = a + b, cd = c + d;
        sum = ab + cd;
    }
    const float d_inv = 127.0f / amax;
    alignas(16) int8_t q[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) q[j] = roundf(x[j] * d_inv);
    const float d = 1.0f / d_inv;
    if (t >= T) return;
    // block_q8_1_mmq [c / 128][T]: 16 B of scales (d4 float[4] / ds4 half2[4]), then qs[128]
    char* blk = e.yq + ((size_t) (c0 / 128) * T + t) * kXBlock;
    const int sub = (c0 % 128) / 32;
    *(v4i*) (blk + 16 + sub * 32 + hi * 16) = *(const v4i*) q;
    if (hi == 0) {
        if constexpr (EPI == 2) ((uint32_t*) blk)[sub] = swq8_f2h(d) | (swq8_f2h(sum) << 16);
        else ((float*) blk)[sub] = d;
    }
}

/// OPT (rc-gemm, schedule only: every output's arithmetic and order unchanged): bit 0 = the A scales of a stage are
/// staged through LDS with B (one 16 B load per thread per stage instead of 2 x 16 B per row tile and block at use;
/// needs !kPS), bit 1 = no scheduling barriers between the two blocks of a K step.
template <int MODE, int WGN, int WGT, int WM, int WT, int KS, bool kPS, int EPI, int OPT = 0>
__device__ __forceinline__ void i8v10_body(const int8_t* __restrict__ qs, const float* __restrict__ ws,
                                           const char* __restrict__ xq, float* __restrict__ dst, int K, int T,
                                           int64_t ldd, int N, const SwiQ8& epi) {
    constexpr int NT = 32 * WGN * WGT, SC = MODE == 0 ? 1 : 2, BS = MODE == 2 ? 2 : 1;
    constexpr int MT = WGN * WM, BT = 16 * WGT * WT;
    static_assert(KS % 2 == 0 && (KS % 4 == 0 || KS == 2), "KS");
    constexpr bool kSL = (OPT & 1) != 0, kNoSB = (OPT & 2) != 0;
    // OPT 2048 (rc-gemm v122): the A tiles too are staged through LDS (with their scales, bit 0): every compute operand
    // comes from LDS, the stage's global loads are issued a whole stage ahead; each A tile is fetched once per block
    constexpr bool kAL = (OPT & 2048) != 0;
    static_assert(!(kSL && kPS), "OPT bit 0 replaces the prefetched scales");
    static_assert(!kAL || (kSL && !kPS), "OPT 2048 needs the scales in LDS (bit 0)");
    constexpr int NAS = kSL ? MT * KS * SC * 16 : 4;   // A scale floats per stage: [mt][kk][j][16]
    struct StageB {
        int8_t b[KS][2][BT][16];
        float bs[KS][BS][BT];
    };
    struct StageSL : StageB {
        float as[NAS];
    };
    // kAL: A tile (mt, kk) = 16 rows x 32 B; row r's 16-byte halves stored swapped when (r >> 3) & 1 (conflict-free reads)
    struct StageAL : StageSL {
        int8_t a[kAL ? KS : 1][kAL ? MT : 1][512];
    };
    using Stage = std::conditional_t<kAL, StageAL, std::conditional_t<kSL, StageSL, StageB>>;
    __shared__ __attribute__((aligned(16))) Stage st[2];
    const int KB = K / 32, NS = KB / KS;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int wn = w % WGN, wt = w / WGN;
    const int mb0 = blockIdx.y * MT;
    const int tb0 = blockIdx.x * BT;
    const uint32_t xstride = (uint32_t) T * kXBlock;
    const __amdgpu_buffer_rsrc_t rq = tp_rsrc(qs, (uint32_t) N * (uint32_t) K);
    const __amdgpu_buffer_rsrc_t rw = tp_rsrc(ws, (uint32_t) (N / 16) * (uint32_t) KB * 64u * SC);
    const __amdgpu_buffer_rsrc_t rx = tp_rsrc(xq, (uint32_t) (K / 128) * xstride);
    // B staging units (16 B): u = tid + j NT -> (h, t, kk); scales: u -> (t, kk)
    constexpr int NB = KS * BT * 2, NBS = KS * BT;
    static_assert(NB % NT == 0 && NBS % NT == 0, "B staging");
    constexpr int RB = NB / NT, RBS = NBS / NT;
    uint32_t vb[RB], vbs[RBS];
#pragma unroll
    for (int j = 0; j < RB; ++j) {
        const int u = tid + j * NT, h = u & 1, t = (u >> 1) % BT, kk = (u >> 1) / BT;
        vb[j] = (uint32_t) min(tb0 + t, T - 1) * kXBlock + (kk >> 2) * xstride + 16 + (kk & 3) * 32 + h * 16;
    }
#pragma unroll
    for (int j = 0; j < RBS; ++j) {
        const int u = tid + j * NT, t = u % BT, kk = u / BT;
        vbs[j] = (uint32_t) min(tb0 + t, T - 1) * kXBlock + (kk >> 2) * xstride + (kk & 3) * 4;
    }
    v4i rb[RB];
    uint32_t rbs[RBS];
    // kSL: the stage's A scales, NAS / 4 16-byte units (contiguous per row tile in ws: [mt][kb][j][16])
    constexpr int NAU = NAS / 4, RAU = kSL ? (NAU + NT - 1) / NT : 0;
    v4i ras[RAU > 0 ? RAU : 1];
    constexpr int NAA = kAL ? KS * MT * 32 : 0, RAA = kAL ? NAA / NT : 0;   // A 16-byte units per stage
    static_assert(!kAL || NAA % NT == 0, "A staging");
    v4i raa[RAA > 0 ? RAA : 1];
    auto gload = [&](int s, int part = -1) {   // part 0 / 1: half the 16-byte units (OPT 8192), -1 all
        const int kb0 = s * KS;
        const uint32_t so = (uint32_t) (kb0 >> 2) * xstride + (kb0 & 3) * 32, sos = (uint32_t) (kb0 >> 2) * xstride + (kb0 & 3) * 4;
#pragma unroll
        for (int j = 0; j < RB; ++j)
            if (part < 0 || (j & 1) == part) rb[j] = tp_ld16(rx, vb[j], so);
#pragma unroll
        for (int j = 0; j < RBS; ++j)
            if (part <= 0) rbs[j] = tp_ld4(rx, vbs[j], sos);
        if constexpr (kSL) {
#pragma unroll
            for (int j = 0; j < RAU; ++j) {
                const int u = tid + j * NT, ml = u / (KS * SC * 4), o = u % (KS * SC * 4);
                if ((NAU % NT == 0 || u < NAU) && part <= 0)
                    ras[j] = tp_ld16(rw, (uint32_t) (mb0 + ml) * (uint32_t) KB * 64u * SC + (uint32_t) o * 16u,
                                     (uint32_t) kb0 * 64u * SC);
            }
        }
        if constexpr (kAL) {
#pragma unroll
            for (int j = 0; j < RAA; ++j) {
                const int u = tid + j * NT, o = u & 31, mt = (u >> 5) % MT, kk = (u >> 5) / MT;
                if (part >= 0 && (j & 1) != part) continue;
                raa[j] = tp_ld16(rq, (uint32_t) (mb0 + mt) * (uint32_t) KB * 512u + (uint32_t) o * 16u,
                                 (uint32_t) (kb0 + kk) * 512u);
            }
        }
    };
    auto lstore = [&](Stage& S, int part = -1) {
#pragma unroll
        for (int j = 0; j < RB; ++j) {
            const int u = tid + j * NT, h = u & 1, t = (u >> 1) % BT, kk = (u >> 1) / BT;
            if (part >= 0 && (j & 1) != part) continue;
            *(v4i*) &S.b[kk][h][t][0] = rb[j];
        }
#pragma unroll
        for (int j = 0; j < RBS; ++j) {
            const int u = tid + j * NT, t = u % BT, kk = u / BT;
            if (part == 1) continue;
            if constexpr (MODE == 2) {
                const float2 ds = __half22float2(__builtin_bit_cast(__half2, rbs[j]));
                S.bs[kk][0][t] = ds.x;
                S.bs[kk][1][t] = ds.y;
            } else {
                S.bs[kk][0][t] = __uint_as_float(rbs[j]);
            }
        }
        if constexpr (kSL) {
#pragma unroll
            for (int j = 0; j < RAU; ++j) {
                const int u = tid + j * NT;
                if ((NAU % NT == 0 || u < NAU) && part <= 0) *(v4i*) &((StageSL&) S).as[u * 4] = ras[j];
            }
        }
        if constexpr (kAL) {
#pragma unroll
            for (int j = 0; j < RAA; ++j) {
                const int u = tid + j * NT, o = u & 31, mt = (u >> 5) % MT, kk = (u >> 5) / MT;
                const int r = o >> 1, h = (o & 1) ^ ((r >> 3) & 1);
                if (part >= 0 && (j & 1) != part) continue;
                *(v4i*) &((StageAL&) S).a[kk][mt][r * 32 + h * 16] = raa[j];
            }
        }
    };
    // this wave's A tiles: lane l16 reads row l16 (lanes 16-31 the same rows), scales of rows 2 i + hi
    uint32_t va[WM], vs[WM];
#pragma unroll
    for (int a = 0; a < WM; ++a) {
        const uint32_t mt = (uint32_t) (mb0 + wn * WM + a);
        va[a] = mt * (uint32_t) KB * 512u + l16 * 32;
        vs[a] = mt * (uint32_t) KB * 64u * SC + hi * 32;
    }
    struct AOps {
        v4i A[WM][2];
        v4i d[kPS ? WM : 1][SC][2];   // kPS: the scales prefetched with A (else loaded at use)
    };
    // OPT 128 / 256 / 512: timing diagnostics only (wrong results): A and scales always from block 0 / no B staging
    // inside the loop / no stage barrier
    constexpr bool kDiagA = (OPT & 128) != 0, kDiagB = (OPT & 256) != 0, kDiagBar = (OPT & 512) != 0;
    auto aload = [&](int kb, AOps& o) {
        if constexpr (kAL) return;   // A comes from LDS
        if constexpr (kDiagA) kb = 0;
        const uint32_t so = (uint32_t) kb * 512u, sos = (uint32_t) kb * 64u * SC;
#pragma unroll
        for (int a = 0; a < WM; ++a) {
            o.A[a][0] = tp_ld16(rq, va[a], so);
            o.A[a][1] = tp_ld16(rq, va[a] + 16, so);
            if constexpr (kPS) {
#pragma unroll
                for (int j = 0; j < SC; ++j) {
                    o.d[a][j][0] = tp_ld16(rw, vs[a] + j * 64, sos);
                    o.d[a][j][1] = tp_ld16(rw, vs[a] + j * 64 + 16, sos);
                }
            }
        }
    };
    float sum[WM][WT][8];
#pragma unroll
    for (int a = 0; a < WM; ++a)
#pragma unroll
        for (int b = 0; b < WT; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
    constexpr int kB = 0x4B400000;   // see k_i8gemm_lds v7
    constexpr float kM = 12582912.0f;
    AOps ring[2];
    // OPT 1024: the block's A scales are loaded before the next block's A prefetch (VMEM returns in order: waiting for
    // the scales then no longer waits for the prefetch too)
    constexpr bool kSE = (OPT & 1024) != 0;
    static_assert(!(kSE && (kPS || kSL)), "OPT 1024: scales at use only");
    struct SOps {
        v4i d[kSE ? WM : 1][SC][2];
    };
    auto sload = [&](int kb, SOps& o) {
        if constexpr (kSE) {
#pragma unroll
            for (int a = 0; a < WM; ++a)
#pragma unroll
                for (int j = 0; j < SC; ++j) {
                    o.d[a][j][0] = tp_ld16(rw, vs[a] + j * 64, (uint32_t) kb * 64u * SC);
                    o.d[a][j][1] = tp_ld16(rw, vs[a] + j * 64 + 16, (uint32_t) kb * 64u * SC);
                }
        }
    };
    auto block = [&](const Stage& S, int kk, int kb, const AOps& cur, const SOps& sc) {
        v4i B[WT][2];
        float dx[WT], sx[WT];
#pragma unroll
        for (int b = 0; b < WT; ++b) {
            const int t = wt * 16 * WT + b * 16 + l16;
            B[b][0] = *(const v4i*) &S.b[kk][0][t][0];
            B[b][1] = *(const v4i*) &S.b[kk][1][t][0];
            dx[b] = S.bs[kk][0][t];
            sx[b] = MODE == 2 ? S.bs[kk][BS - 1][t] : 0.f;
        }
#pragma unroll
        for (int a = 0; a < WM; ++a) {
            v4i A0, A1;
            if constexpr (kAL) {
                const int8_t* ta = &((const StageAL&) S).a[kk][wn * WM + a][l16 * 32];
                const int sw = (l16 >> 3) & 1;
                A0 = *(const v4i*) (ta + 16 * sw);
                A1 = *(const v4i*) (ta + 16 * (sw ^ 1));
            } else {
                A0 = cur.A[a][0];
                A1 = cur.A[a][1];
            }
            float dw[SC][8];
#pragma unroll
            for (int j = 0; j < SC; ++j) {
                v4i d0, d1;
                if constexpr (kPS) {
                    d0 = cur.d[a][j][0];
                    d1 = cur.d[a][j][1];
                } else if constexpr (kSL) {
                    const float* sp = &((const StageSL&) S).as[(((wn * WM + a) * KS + kk) * SC + j) * 16 + hi * 8];
                    d0 = *(const v4i*) sp;
                    d1 = *(const v4i*) (sp + 4);
                } else if constexpr (kSE) {
                    d0 = sc.d[a][j][0];
                    d1 = sc.d[a][j][1];
                } else {
                    if constexpr (kDiagA) kb = 0;
                    d0 = tp_ld16(rw, vs[a] + j * 64, (uint32_t) kb * 64u * SC);
                    d1 = tp_ld16(rw, vs[a] + j * 64 + 16, (uint32_t) kb * 64u * SC);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    dw[j][i] = __int_as_float(d0[i]);
                    dw[j][4 + i] = __int_as_float(d1[i]);
                }
            }
            if constexpr (MODE == 1) {
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    v8i c[WT];
#pragma unroll
                    for (int b = 0; b < WT; ++b) {
                        c[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                        c[b] = wmma_i8(hh ? A1 : A0, B[b][hh], c[b]);
                    }
                    float f[WT][8];
                    if constexpr ((OPT & 8) != 0 && WT % 2 == 0) {
                        // the same per-element expressions, emitted diagonally ((i, b) next to (i ^ 1, b + 1))
#pragma unroll
                        for (int b = 0; b < WT; b += 2) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                f[b][i] = (__int_as_float(c[b][i]) - kM) * dw[hh][i];
                                f[b + 1][i ^ 1] = (__int_as_float(c[b + 1][i ^ 1]) - kM) * dw[hh][i ^ 1];
                            }
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                                sum[a][b + 1][i ^ 1] = __builtin_fmaf(f[b + 1][i ^ 1], dx[b + 1], sum[a][b + 1][i ^ 1]);
                            }
                            if constexpr ((OPT & 32) != 0) __builtin_amdgcn_sched_barrier(0);
                        }
                    } else {
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[b][i] = (__int_as_float(c[b][i]) - kM) * dw[hh][i];
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                    }
                }
            } else {
                v8i c[WT];
#pragma unroll
                for (int b = 0; b < WT; ++b) {
                    c[b] = (v8i){kB, kB, kB, kB, kB, kB, kB, kB};
                    c[b] = wmma_i8(A0, B[b][0], c[b]);
                    c[b] = wmma_i8(A1, B[b][1], c[b]);
                }
                float f[WT][8];
#pragma unroll
                for (int b = 0; b < WT; ++b)
#pragma unroll
                    for (int i = 0; i < 8; ++i) f[b][i] = __int_as_float(c[b][i]) - kM;
                if constexpr (MODE == 2 && (OPT & 16) && WT % 2 == 0) {
                    // as OPT 8, but both runs per pair of token tiles (shorter live ranges)
#pragma unroll
                    for (int b = 0; b < WT; b += 2) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] = __builtin_fmaf(dw[0][i] * dx[b], f[b][i], sum[a][b][i]);
                            sum[a][b + 1][i ^ 1] =
                                __builtin_fmaf(dw[0][i ^ 1] * dx[b + 1], f[b + 1][i ^ 1], sum[a][b + 1][i ^ 1]);
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] = __builtin_fmaf(dw[SC - 1][i], sx[b], sum[a][b][i]);
                            sum[a][b + 1][i ^ 1] = __builtin_fmaf(dw[SC - 1][i ^ 1], sx[b + 1], sum[a][b + 1][i ^ 1]);
                        }
                    }
                } else if constexpr (MODE == 2 && (OPT & 8) && WT % 2 == 0) {
                    // the same two fmas per element in the same order; all first fmas, then all second fmas, each
                    // emitted "diagonally" ((i, b) next to (i ^ 1, b + 1)) so that neighbours differ in both the row
                    // operand (dw / dm) and the token operand (dx / sx): VOPD needs different VGPR banks per source
#pragma unroll
                    for (int b = 0; b < WT; b += 2) {
                        float p0[8], p1[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            p0[i] = dw[0][i] * dx[b];
                            p1[i ^ 1] = dw[0][i ^ 1] * dx[b + 1];
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] = __builtin_fmaf(p0[i], f[b][i], sum[a][b][i]);
                            sum[a][b + 1][i ^ 1] = __builtin_fmaf(p1[i ^ 1], f[b + 1][i ^ 1], sum[a][b + 1][i ^ 1]);
                        }
                        if constexpr ((OPT & 32) != 0) __builtin_amdgcn_sched_barrier(0);
                    }
#pragma unroll
                    for (int b = 0; b < WT; b += 2)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] = __builtin_fmaf(dw[SC - 1][i], sx[b], sum[a][b][i]);
                            sum[a][b + 1][i ^ 1] = __builtin_fmaf(dw[SC - 1][i ^ 1], sx[b + 1], sum[a][b + 1][i ^ 1]);
                        }
                } else if constexpr (MODE == 2 && (OPT & 4)) {
                    // the same two fmas per element in the same order, issued as independent runs of 8 (VOPD pairing)
#pragma unroll
                    for (int b = 0; b < WT; ++b) {
                        float p[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) p[i] = dw[0][i] * dx[b];
#pragma unroll
                        for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(p[i], f[b][i], sum[a][b][i]);
#pragma unroll
                        for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(dw[SC - 1][i], sx[b], sum[a][b][i]);
                    }
                } else if constexpr (MODE == 2) {
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] = __builtin_fmaf(dw[0][i] * dx[b], f[b][i], sum[a][b][i]);
                            sum[a][b][i] = __builtin_fmaf(dw[SC - 1][i], sx[b], sum[a][b][i]);
                        }
                } else if constexpr ((OPT & 8) != 0 && WT % 2 == 0) {
                    // rc-gemm (mode 0): the same expressions, emitted diagonally ((i, b) next to (i ^ 1, b + 1))
#pragma unroll
                    for (int b = 0; b < WT; b += 2) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            f[b][i] *= dw[0][i];
                            f[b + 1][i ^ 1] *= dw[0][i ^ 1];
                        }
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                            sum[a][b + 1][i ^ 1] = __builtin_fmaf(f[b + 1][i ^ 1], dx[b + 1], sum[a][b + 1][i ^ 1]);
                        }
                        if constexpr ((OPT & 32) != 0) __builtin_amdgcn_sched_barrier(0);
                    }
                } else {
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[b][i] *= dw[0][i];
#pragma unroll
                    for (int b = 0; b < WT; ++b)
#pragma unroll
                        for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[b][i], dx[b], sum[a][b][i]);
                }
            }
        }
    };
    // stage loop (KS blocks, KS even), blocks unrolled inside it (A ring index static), no branches: the next stage's B
    // loads are issued after block 0's A prefetch (clamped at the last stage), stored into the other buffer after the
    // last block (the last stage's store goes to a buffer nobody reads), one barrier per stage
    aload(0, ring[0]);
    gload(0);
    lstore(st[0]);
    __syncthreads();
#pragma unroll 1
    for (int s = 0; s < NS; ++s) {
        const Stage& S = st[s & 1];
#pragma unroll 1
        for (int kk = 0; kk < KS; kk += 2) {
            const int kb = s * KS + kk;
            SOps sc;
            sload(kb, sc);
            aload(min(kb + 1, KB - 1), ring[1]);
            constexpr bool kLate = (OPT & 4096) != 0;   // the next stage's global loads after this stage's first block
            constexpr bool kSplit = (OPT & 8192) != 0 && KS == 2;   // half the next stage per block, stored right after it
            if (!kDiagB && !kLate && !kSplit && kk == 0) gload(min(s + 1, NS - 1));
            if constexpr (kSplit) gload(min(s + 1, NS - 1), 0);
            block(S, kk, kb, ring[0], sc);
            if constexpr (!kNoSB) __builtin_amdgcn_sched_barrier(0);
            if constexpr (kSplit) {
                lstore(st[(s + 1) & 1], 0);
                gload(min(s + 1, NS - 1), 1);
            }
            if (!kDiagB && kLate && kk == KS - 2) gload(min(s + 1, NS - 1));
            sload(kb + 1, sc);
            aload(min(kb + 2, KB - 1), ring[0]);
            block(S, kk + 1, kb + 1, ring[1], sc);
            if constexpr (!kNoSB) __builtin_amdgcn_sched_barrier(0);
        }
        if constexpr (!kDiagB) lstore(st[(s + 1) & 1], ((OPT & 8192) != 0 && KS == 2) ? 1 : -1);
        if constexpr (!kDiagBar) __syncthreads();
    }
    if constexpr (EPI == 0) {
#pragma unroll
    for (int b = 0; b < WT; ++b) {
        const int t = tb0 + wt * 16 * WT + b * 16 + l16;
        if (t >= T) continue;
#pragma unroll
        for (int a = 0; a < WM; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[(size_t) t * ldd + (mb0 + wn * WM + a) * 16 + 2 * i + hi] = sum[a][b][i];
    }
    } else {
        static_assert(WM == 2, "the SwiGLU epilogue needs a wave's two row tiles = one 32-column scale group");
        (void) dst;
        (void) ldd;
        // one token tile at a time (sched barriers: the g loads / temporaries of all WT tiles at once would not fit
        // next to the 64 accumulators)
#pragma unroll
        for (int b = 0; b < WT; ++b) {
            __builtin_amdgcn_sched_barrier(0);
            swq8_store<EPI>(sum[0][b], sum[1][b], hi, tb0 + wt * 16 * WT + b * 16 + l16, T, (mb0 + wn * WM) * 16, epi);
        }
    }
}

template <int MODE, int WGN, int WGT, int WM, int WT, int KS, bool kPS, int OPT = 0>
__global__ void __launch_bounds__(32 * WGN * WGT) k_i8v10(const int8_t* __restrict__ qs, const float* __restrict__ ws,
                                                         const char* __restrict__ xq, float* __restrict__ dst, int K,
                                                         int T, int64_t ldd, int N) {
    i8v10_body<MODE, WGN, WGT, WM, WT, KS, kPS, 0, OPT>(qs, ws, xq, dst, K, T, ldd, N, SwiQ8{});
}

/// k_i8v10 with an occupancy-6 register budget (amdgpu_waves_per_eu 6: <= 240 VGPRs) for the OPT variants
/// whose schedule would otherwise cost a wave (the plain kernel keeps its codegen).
template <int MODE, int WGN, int WGT, int WM, int WT, int KS, bool kPS, int OPT>
__global__ void __launch_bounds__(32 * WGN * WGT) __attribute__((amdgpu_waves_per_eu(6)))
k_i8v10w6(const int8_t* __restrict__ qs, const float* __restrict__ ws, const char* __restrict__ xq,
          float* __restrict__ dst, int K, int T, int64_t ldd, int N) {
    i8v10_body<MODE, WGN, WGT, WM, WT, KS, kPS, 0, OPT>(qs, ws, xq, dst, K, T, ldd, N, SwiQ8{});
}

/// rb-q27: k_i8v10 (up projection) with the SwiGLU + q8_1 epilogue (EPI 1: D4, 2: DS4) instead of the FP32 store.
template <int MODE, int WGN, int WGT, int WM, int WT, int KS, bool kPS, int EPI>
__global__ void __launch_bounds__(32 * WGN * WGT) k_i8v10_swq8(const int8_t* __restrict__ qs,
                                                              const float* __restrict__ ws, const char* __restrict__ xq,
                                                              int K, int T, int N, SwiQ8 epi) {
    i8v10_body<MODE, WGN, WGT, WM, WT, KS, kPS, EPI>(qs, ws, xq, nullptr, K, T, 0, N, epi);
}

/// k_i8v10_swq8 with an OPT schedule and the occupancy-6 register budget (see k_i8v10w6).
template <int MODE, int WGN, int WGT, int WM, int WT, int KS, bool kPS, int EPI, int OPT>
__global__ void __launch_bounds__(32 * WGN * WGT) __attribute__((amdgpu_waves_per_eu(6)))
k_i8v10_swq8w6(const int8_t* __restrict__ qs, const float* __restrict__ ws, const char* __restrict__ xq, int K, int T,
               int N, SwiQ8 epi) {
    i8v10_body<MODE, WGN, WGT, WM, WT, KS, kPS, EPI, OPT>(qs, ws, xq, nullptr, K, T, 0, N, epi);
}

template <int WGN, int WGT, int WM, int WT, int KS, bool kPS = true, int OPT = 0>
bool launch_v10(const I8Mat& w, const char* x, int T, float* dst, int64_t ld, hipStream_t s) {
    constexpr int BN = 16 * WGN * WM, BT = 16 * WGT * WT;
    if (w.n % BN != 0 || (w.k / 32) % KS != 0) return false;
    const dim3 grid((unsigned) ((T + BT - 1) / BT), (unsigned) (w.n / BN));
    constexpr int NT = 32 * WGN * WGT;
    constexpr int O = OPT & ~64;
    if constexpr ((OPT & 64) != 0) {   // waves_per_eu 6 entry
        if (w.mode == 1) k_i8v10w6<1, WGN, WGT, WM, WT, KS, kPS, O><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld, w.n);
        else if (w.mode == 2) k_i8v10w6<2, WGN, WGT, WM, WT, KS, kPS, O><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld, w.n);
        else k_i8v10w6<0, WGN, WGT, WM, WT, KS, kPS, O><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld, w.n);
        return true;
    }
    if (w.mode == 1) k_i8v10<1, WGN, WGT, WM, WT, KS, kPS, O><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld, w.n);
    else if (w.mode == 2) k_i8v10<2, WGN, WGT, WM, WT, KS, kPS, O><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld, w.n);
    else k_i8v10<0, WGN, WGT, WM, WT, KS, kPS, O><<<grid, NT, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld, w.n);
    return true;
}

}  // namespace

void i8gemm(const I8Mat& w, const void* xq, int T, float* dst, int64_t ld_dst, void* stream) {
    if (T <= 0) return;
    const dim3 grid((unsigned) ((T + 127) / 128), (unsigned) (w.n / 128));
    static const int env = [] { const char* e = std::getenv("TP_I8_VARIANT"); return e ? std::atoi(e) : 0; }();
    const TpTune& tune = tp_tune_current();
    const int mv = tune.i8_mode_variant[w.mode % 3];
    int v = env ? env : mv ? mv : tune.i8_variant;
    if (w.mode != 0 && v < 6) v = 6;   // (modes 1 / 2: the LDS kernels only)
    const hipStream_t s = (hipStream_t) stream;
    const char* x = (const char*) xq;
    if (v >= 100 && v < 200) {   // v10 shapes (falls back to v7 when the shape does not divide)
        bool ok = false;
        switch (v) {
            case 100: ok = launch_v10<4, 2, 2, 4, 4>(w, x, T, dst, ld_dst, s); break;   // 128 x 128, KS 4 (= v91)
            case 101: ok = launch_v10<4, 2, 2, 4, 2>(w, x, T, dst, ld_dst, s); break;   // KS 2
            case 102: ok = launch_v10<2, 4, 4, 2, 4>(w, x, T, dst, ld_dst, s); break;   // waves 2 x 4, 4 x 2 tiles
            case 103: ok = launch_v10<8, 1, 1, 8, 4>(w, x, T, dst, ld_dst, s); break;   // 8 row waves, 1 x 8
            case 105: ok = launch_v10<2, 2, 2, 4, 4>(w, x, T, dst, ld_dst, s); break;   // 4 waves, 64 x 128
            case 106: ok = launch_v10<4, 2, 2, 4, 4, false>(w, x, T, dst, ld_dst, s); break;   // v100, scales at use
            case 107: ok = launch_v10<4, 2, 2, 4, 2, false>(w, x, T, dst, ld_dst, s); break;   // v101, scales at use
            // v106 schedules (bitwise v106)
            case 108: ok = launch_v10<4, 2, 2, 4, 4, false, 1>(w, x, T, dst, ld_dst, s); break;   // A scales via LDS
            case 109: ok = launch_v10<4, 2, 2, 4, 4, false, 3>(w, x, T, dst, ld_dst, s); break;   // + no sched barriers
            case 110: ok = launch_v10<4, 2, 2, 4, 4, false, 2>(w, x, T, dst, ld_dst, s); break;   // v106, no sched barriers
            case 111: ok = launch_v10<4, 2, 2, 4, 4, false, 4>(w, x, T, dst, ld_dst, s); break;   // v106, mode-2 runs of 8
            case 112: ok = launch_v10<4, 2, 2, 4, 4, false, 6>(w, x, T, dst, ld_dst, s); break;   // 111, no sched barriers
            case 113: ok = launch_v10<4, 2, 2, 4, 4, false, 8>(w, x, T, dst, ld_dst, s); break;   // v106, mode-2 diagonal runs
            case 114: ok = launch_v10<4, 2, 2, 4, 4, false, 16>(w, x, T, dst, ld_dst, s); break;   // diagonal per tile pair
            case 115: ok = launch_v10<4, 2, 2, 4, 4, false, 18>(w, x, T, dst, ld_dst, s); break;   // 114, no sched barriers
            case 116: ok = launch_v10<4, 2, 2, 4, 4, false, 40>(w, x, T, dst, ld_dst, s); break;   // 113 + fences per tile pair
            case 117: ok = launch_v10<4, 2, 2, 4, 4, false, 40 | 64>(w, x, T, dst, ld_dst, s); break;   // 116, <= 240 VGPR
            case 118: ok = launch_v10<4, 2, 2, 4, 4, false, 8 | 64>(w, x, T, dst, ld_dst, s); break;    // 113, <= 240 VGPR
            // timing diagnostics (wrong results): v106 with A / scales from block 0, no B staging, no barrier, all
            case 122: ok = launch_v10<4, 2, 2, 4, 2, false, 2048 | 1>(w, x, T, dst, ld_dst, s); break;   // A + scales via LDS, KS 2
            case 123: ok = launch_v10<4, 2, 2, 4, 2, false, 2048 | 1 | 64>(w, x, T, dst, ld_dst, s); break;   // 122, <= 240 VGPR
            case 124: ok = launch_v10<4, 2, 2, 4, 2, false, 2048 | 1 | 40 | 64>(w, x, T, dst, ld_dst, s); break;   // 123 + v117 epilogue
            case 125: ok = launch_v10<4, 2, 2, 4, 2, false, 4096 | 2048 | 1>(w, x, T, dst, ld_dst, s); break;   // 122, late stage loads
            case 126: ok = launch_v10<4, 2, 2, 4, 2, false, 4096 | 2048 | 1 | 64>(w, x, T, dst, ld_dst, s); break;   // 125, <= 240
            case 128: ok = launch_v10<4, 2, 2, 4, 2, false, 8192 | 2048 | 1>(w, x, T, dst, ld_dst, s); break;   // 122, split staging
            case 129: ok = launch_v10<4, 2, 2, 4, 2, false, 8192 | 2048 | 1 | 64>(w, x, T, dst, ld_dst, s); break;   // 128, <= 240
            case 119: ok = launch_v10<4, 2, 2, 4, 4, false, 1024>(w, x, T, dst, ld_dst, s); break;   // v106, scales before A prefetch
            case 120: ok = launch_v10<4, 2, 2, 4, 4, false, 1024 | 64>(w, x, T, dst, ld_dst, s); break;   // 119, <= 240 VGPR
            case 121: ok = launch_v10<4, 2, 2, 4, 4, false, 1024 | 40 | 64>(w, x, T, dst, ld_dst, s); break;   // 119 + v117
            case 191: ok = launch_v10<4, 2, 2, 4, 4, false, 128>(w, x, T, dst, ld_dst, s); break;
            case 192: ok = launch_v10<4, 2, 2, 4, 4, false, 256>(w, x, T, dst, ld_dst, s); break;
            case 193: ok = launch_v10<4, 2, 2, 4, 4, false, 512 | 256>(w, x, T, dst, ld_dst, s); break;
            case 194: ok = launch_v10<4, 2, 2, 4, 4, false, 512 | 256 | 128>(w, x, T, dst, ld_dst, s); break;
            default: break;
        }
        if (ok) return;
        v = 7;
    }
    if (v >= 90 && v < 96) {   // v9 shapes (falls back to v7 when the shape does not divide)
        bool ok = false;
        switch (v) {
            case 90: ok = launch_v9<4, 2, 2, 4, 2>(w, x, T, dst, ld_dst, s); break;   // 128 x 128, KS 2
            case 91: ok = launch_v9<4, 2, 2, 4, 4>(w, x, T, dst, ld_dst, s); break;   // 128 x 128, KS 4
            case 92: ok = launch_v9<4, 2, 2, 4, 1>(w, x, T, dst, ld_dst, s); break;   // KS 1
            case 93: ok = launch_v9<2, 4, 4, 2, 4>(w, x, T, dst, ld_dst, s); break;   // waves 2 x 4, 4 x 2 tiles
            case 94: ok = launch_v9<4, 2, 1, 4, 4>(w, x, T, dst, ld_dst, s); break;   // 64 x 128, 1 x 4 tiles
            case 95: ok = launch_v9<8, 1, 1, 8, 4>(w, x, T, dst, ld_dst, s); break;   // 128 x 128, 8 row waves, 1 x 8
            default: break;
        }
        if (ok) return;
        v = 7;
    }
    if (v >= 80 && v < 90) {   // v8 block shapes (falls back to v7 when the shape does not divide)
        bool ok = false;
        switch (v) {
            case 80: ok = launch_v8<4, 2, 2, 4, 2>(w, x, T, dst, ld_dst, s); break;   // = v7's shape, 128 x 128
            case 82: ok = launch_v8<2, 2, 2, 4, 2>(w, x, T, dst, ld_dst, s); break;   // 4 waves, 64 x 128
            case 83: ok = launch_v8<2, 4, 4, 2, 2>(w, x, T, dst, ld_dst, s); break;   // 128 x 128, waves 2 x 4, 4 x 2 tiles
            case 84: ok = launch_v8<4, 2, 4, 2, 2>(w, x, T, dst, ld_dst, s); break;   // 256 x 64
            case 85: ok = launch_v8<2, 2, 2, 4, 4>(w, x, T, dst, ld_dst, s); break;   // 4 waves, KS 4
            case 86: ok = launch_v8<4, 2, 2, 4, 1>(w, x, T, dst, ld_dst, s); break;   // KS 1
            case 87: ok = launch_v8<2, 4, 2, 2, 2>(w, x, T, dst, ld_dst, s); break;   // 64 x 128 waves 2 x 4, 2 x 2 tiles
            default: break;
        }
        if (ok) return;
        v = 7;
    }
    switch (v) {
        case 1: k_i8gemm<2, 4, false><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;
        case 3: k_i8gemm<4, 2, true><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;
        case 4: k_i8gemm<1, 8, true><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;
        case 5: k_i8gemm<4, 2, false><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;
        case 7:
            if (w.mode == 1) k_i8gemm_lds<1, 7><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst);
            else if (w.mode == 2) k_i8gemm_lds<2, 7><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst);
            else k_i8gemm_lds<0, 7><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst);
            break;
        case 98: k_i8gemm_lds<0, 8><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;   // timing only
        case 99: k_i8gemm_lds<0, 9><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;   // timing only
        case 96: k_i8gemm_lds<0, 96><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;   // timing only
        case 97: k_i8gemm_lds<0, 97><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;   // timing only
        case 6:
            if (w.mode == 1) k_i8gemm_lds<1><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst);
            else if (w.mode == 2) k_i8gemm_lds<2><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst);
            else k_i8gemm_lds<0><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst);
            break;
        default: k_i8gemm<2, 4, true><<<grid, 256, 0, s>>>(w.qs, w.ws, x, dst, w.k, T, ld_dst); break;
    }
}

I8Cache::~I8Cache() {
    for (auto& kv : m_) hipFree(kv.second.qs);
}

size_t I8Cache::add(const void* w, int ggml_type, int n, int k, void* stream) {
    if (m_.count(w) || !i8x_supported(ggml_type) || n % 128 != 0 || k % 256 != 0) return 0;
    const int mode = i8x_mode(ggml_type);
    const size_t b = i8x_bytes(n, k, mode);
    void* p = nullptr;
    if (hipMalloc(&p, b) != hipSuccess) return 0;
    I8Mat m;
    m.qs = (int8_t*) p;
    m.ws = (float*) ((char*) p + (size_t) n * k);
    m.mode = mode;
    if (!i8x_expand(w, ggml_type, n, k, m, stream)) {
        hipFree(p);
        return 0;
    }
    m_[w] = m;
    bytes_ += b;
    return b;
}

}  // namespace talus::prefill

namespace talus::prefill {
namespace {
template <int EPI>
void launch_swq8(const I8Mat& w, const char* x, int T, const SwiQ8& e, hipStream_t s) {
    // v106's shape (tp_tune gfx1151: 128 x 128, waves 4 x 2, a wave 2 x 4 tiles, KS 4, scales at use)
    constexpr int WGN = 4, WGT = 2, WM = 2, WT = 4, KS = 4, BN = 16 * WGN * WM, BT = 16 * WGT * WT, NT = 32 * WGN * WGT;
    const dim3 grid((unsigned) ((T + BT - 1) / BT), (unsigned) (w.n / BN));
    (void) BN;
    // the up GEMM follows its mode's i8 variant (TP_I8_VARIANT / tp_tune i8_mode_variant) for the v10
    // schedules 117 (OPT 40, occupancy-6 budget), 119 (OPT 1024), 121 (both); the same per-element arithmetic and order
    static const int env = [] { const char* v = std::getenv("TP_I8_VARIANT"); return v ? std::atoi(v) : 0; }();
    const int mv = env ? env : tp_tune_current().i8_mode_variant[w.mode % 3];
    auto go = [&]<int M>() -> bool {
        if (mv == 117) k_i8v10_swq8w6<M, WGN, WGT, WM, WT, KS, false, EPI, 40><<<grid, NT, 0, s>>>(w.qs, w.ws, x, w.k, T, w.n, e);
        else if (mv == 119) k_i8v10_swq8w6<M, WGN, WGT, WM, WT, KS, false, EPI, 1024><<<grid, NT, 0, s>>>(w.qs, w.ws, x, w.k, T, w.n, e);
        else if (mv == 121) k_i8v10_swq8w6<M, WGN, WGT, WM, WT, KS, false, EPI, 1064><<<grid, NT, 0, s>>>(w.qs, w.ws, x, w.k, T, w.n, e);
        else return false;
        return true;
    };
    if ((w.mode == 0 && go.template operator()<0>()) || (w.mode == 1 && go.template operator()<1>()) ||
        (w.mode == 2 && go.template operator()<2>()))
        return;
    if (w.mode == 1) k_i8v10_swq8<1, WGN, WGT, WM, WT, KS, false, EPI><<<grid, NT, 0, s>>>(w.qs, w.ws, x, w.k, T, w.n, e);
    else if (w.mode == 2) k_i8v10_swq8<2, WGN, WGT, WM, WT, KS, false, EPI><<<grid, NT, 0, s>>>(w.qs, w.ws, x, w.k, T, w.n, e);
    else k_i8v10_swq8<0, WGN, WGT, WM, WT, KS, false, EPI><<<grid, NT, 0, s>>>(w.qs, w.ws, x, w.k, T, w.n, e);
}
}  // namespace
}  // namespace talus::prefill

/// rb-q27: the up projection on its expanded matrix with silu(g) * u quantized straight into MMQ q8_1 blocks (the
/// down projection's activations, `layout` = mmq_q8_1_ds_layout of the down weights: 0 D4, 1 DS4) - bitwise
/// talus_pf_i8gemm into u followed by talus_pf_swiglu_q8_mmq(g, u).  g: the gate projection [T][ldg] FP32.
/// -1 (nothing launched): shape / layout / device not covered (n % 512, k % 256, D2S6), or TALUS_PF_SWQ8=0.
extern "C" int talus_pf_i8gemm_swiglu_q8(const int8_t* qs, const float* ws, int n, int k, int mode, const void* xq,
                                         int T, const float* g, int64_t ldg, void* yq, int layout, void* stream) {
    static const bool on = [] { const char* e = std::getenv("TALUS_PF_SWQ8"); return e == nullptr || e[0] != '0'; }();
    if (!on || n % 512 != 0 || k % 256 != 0 || mode < 0 || mode > 2 || (layout != 0 && layout != 1) || T <= 0 ||
        ldg % 4 != 0 || ((uintptr_t) g) % 16 != 0 || !talus::prefill::tp_gfx11_wmma())
        return -1;
    talus::prefill::I8Mat m;
    m.mode = mode;
    m.qs = const_cast<int8_t*>(qs);
    m.ws = const_cast<float*>(ws);
    m.n = n;
    m.k = k;
    talus::prefill::SwiQ8 e;
    e.g = g;
    e.ldg = ldg;
    e.yq = (char*) yq;
    const hipStream_t s = (hipStream_t) stream;
    if (layout == 1) talus::prefill::launch_swq8<2>(m, (const char*) xq, T, e, s);
    else talus::prefill::launch_swq8<1>(m, (const char*) xq, T, e, s);
    return 0;
}

extern "C" int talus_pf_i8gemm(const int8_t* qs, const float* ws, int n, int k, int mode, const void* xq, int T,
                               float* dst, int64_t ld_dst, void* stream) {
    if (n % 128 != 0 || k % 256 != 0 || mode < 0 || mode > 2 || !talus::prefill::tp_gfx11_wmma()) return -1;
    talus::prefill::I8Mat m;
    m.mode = mode;
    m.qs = const_cast<int8_t*>(qs);
    m.ws = const_cast<float*>(ws);
    m.n = n;
    m.k = k;
    talus::prefill::i8gemm(m, xq, T, dst, ld_dst, stream);
    return 0;
}
