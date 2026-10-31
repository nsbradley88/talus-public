// src/prefill/tp_moe.cu - grouped expert GEMM for the MoE prompt path (exact: bitwise llama.cpp MMQ).
//
// The experts stay in their GGUF blocks.  A block (256 threads = 8 waves) owns one expert's 128 weight rows and up to
// 128 of the expert's routed (sorted) activation rows.  Per 256-value K step the weight tile goes through MMQ's own
// tile loader (ggml_cuda_mmq_get_load_tiles, the J = 128 configuration) into LDS - the very int8 values and scales
// mul_mat_q multiplies - and each wave multiplies its 16 weight rows against only as many 16-row activation tiles as
// the expert has (1..8): no 64/128-wide token tiles mostly empty at ~10-80 rows per expert, which is where llama.cpp's
// grouped MMQ loses half its rate.  The activations (MMQ's q8_1 rows) are read straight from global memory into the
// WMMA operands.  Per output element and 32-value block, in increasing block order, the arithmetic of tp_i8gemm (the
// same three MMQ dot families, bitwise MMQ), with the int32 -> FP32 conversion folded into the WMMA accumulator
// (0x4B400000 bias, see tp_i8gemm.cu v7).
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "common.cuh"
#include "mmq.cuh"

#include "strata/prefill/tp_i8gemm.hpp"
#include "strata/prefill/tp_moe.hpp"
#include "strata/prefill/moe_w16.hpp"
#include "tp_tiles.cuh"
#include "strata/prefill/tp_tune.hpp"

#include <cstdio>
#include <cstdlib>

namespace talus::prefill {
namespace {

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v3i __attribute__((ext_vector_type(3)));

constexpr int kJ = 128;          // MMQ's tile configuration whose loader we borrow (256 threads, I = 128 rows)
constexpr int kXBlock = 144;     // block_q8_1_mmq
// the MMQ tile's row stride (ints) by mode: Q8_0 / Q8_1 layouts 76, Q3_K's (the per-16 scale types) 84, Q2_K's 100
// (mode 3: Q2_K, MMQ's vec_dot_q2_K_q8_1_mma arithmetic on the D2S6 q8_1 layout)
// (mode 4, Q6_K, MMQ's vec_dot_q6_K_q8_1_mma arithmetic on the D4 q8_1 layout; one-expert products only)
constexpr int kStride[5] = {ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0),
                            ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K),
                            ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1),
                            ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K),
                            ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K)};

__device__ __forceinline__ v8i wmma_i8(v4i a, v4i b, v8i c) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
#else
    (void) a; (void) b;
    return c;
#endif
}

/// grid (ceil(max_rows / 128), n / 128, n_experts), block (32, 8).  MODE as tp_i8gemm (0 q8_0 dot, 1 q8_0_16, 2 q8_1).
template <ggml_type type, int MODE>
__global__ void __launch_bounds__(256) k_moe(const char* __restrict__ wq, size_t expert_bytes, int k,
                                             const char* __restrict__ xq, const int32_t* __restrict__ bounds,
                                             const int32_t* __restrict__ ids, int total_rows, float* __restrict__ dst,
                                             int64_t ldd) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    constexpr int I = TpTiles<type>::I;
    constexpr int ss = TpTiles<type>::ss;
    constexpr int blocks_per_iter = TpTiles<type>::blocks_per_iter;
    static_assert(I == 128, "the 128-row MMQ tile");
    static_assert(ss <= kStride[MODE], "the host's LDS size (kStride) covers this layout");
    extern __shared__ int tile[];
    const int e = blockIdx.z;
    const int r0 = bounds[e] + blockIdx.x * 128, r1 = min(bounds[e + 1], r0 + 128);
    if (r0 >= r1) return;
    const int nt = (r1 - r0 + 15) >> 4;   // 16-row activation tiles of this block (1..8)
    const int lane = threadIdx.x, w = threadIdx.y, l16 = lane & 15, hi = lane >> 4;
    const int row0 = blockIdx.y * I;
    const int stride_blocks = k / (MMQ_ITER_K / blocks_per_iter);
    const char* we = wq + (size_t) e * expert_bytes;
    const size_t xstride = (size_t) total_rows * kXBlock;
    const char* xr[8];
#pragma unroll
    for (int b = 0; b < 8; ++b) xr[b] = xq + (size_t) min(r0 + b * 16 + l16, total_rows - 1) * kXBlock;
    const int* x_qs = tile;
    const float* x_df = (const float*) (tile + 2 * MMQ_TILE_NE_K);
    const half2* x_dm = (const half2*) (tile + 2 * MMQ_TILE_NE_K);
    const int ra = w * 16 + l16;   // this lane's A row in the tile
    constexpr int kB = 0x4B400000;
    constexpr float kM = 12582912.0f;
    float sum[8][8];
#pragma unroll
    for (int b = 0; b < 8; ++b)
#pragma unroll
        for (int i = 0; i < 8; ++i) sum[b][i] = 0.f;
    const int KI = k / MMQ_ITER_K;
#pragma unroll 1
    for (int ki = 0; ki < KI; ++ki) {
        if (ki) __syncthreads();
        TpTiles<type>::load(we, tile, row0 * stride_blocks + ki * blocks_per_iter, I - 1, stride_blocks);
        __syncthreads();
#pragma unroll 1
        for (int j = 0; j < 8; ++j) {   // 32-value block j of this K step
            const int kb = ki * 8 + j;
            const v4i A0 = *(const v4i*) &x_qs[ra * ss + j * 8], A1 = *(const v4i*) &x_qs[ra * ss + j * 8 + 4];
            float dw[2][8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = w * 16 + 2 * i + hi;
                if constexpr (MODE == 0) {
                    dw[0][i] = x_df[r * ss + j];
                } else if constexpr (MODE == 1) {
                    dw[0][i] = x_df[r * ss + 2 * j];
                    dw[1][i] = x_df[r * ss + 2 * j + 1];
                } else {
                    const float2 dm = __half22float2(x_dm[r * ss + j]);
                    dw[0][i] = dm.x;
                    dw[1][i] = dm.y;
                }
            }
            const size_t xo = (size_t) (kb >> 2) * xstride;
            const int sub = kb & 3;
#pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (b >= nt) break;
                const char* p = xr[b] + xo;
                const v4i B0 = *(const v4i*) (p + 16 + sub * 32), B1 = *(const v4i*) (p + 32 + sub * 32);
                float dx, sx = 0.f;
                if constexpr (MODE == 2) {
                    const float2 ds = __half22float2(((const half2*) p)[sub]);
                    dx = ds.x;
                    sx = ds.y;
                } else {
                    dx = ((const float*) p)[sub];
                }
                if constexpr (MODE == 1) {
                    v8i c0 = {kB, kB, kB, kB, kB, kB, kB, kB}, c1 = c0;
                    c0 = wmma_i8(A0, B0, c0);
                    c1 = wmma_i8(A1, B1, c1);
                    float f0[8], f1[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        f0[i] = (__int_as_float(c0[i]) - kM) * dw[0][i];
                        f1[i] = (__int_as_float(c1[i]) - kM) * dw[1][i];
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        sum[b][i] = __builtin_fmaf(f0[i], dx, sum[b][i]);
                        sum[b][i] = __builtin_fmaf(f1[i], dx, sum[b][i]);
                    }
                } else {
                    v8i c = {kB, kB, kB, kB, kB, kB, kB, kB};
                    c = wmma_i8(A0, B0, c);
                    c = wmma_i8(A1, B1, c);
                    float f[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) f[i] = __int_as_float(c[i]) - kM;
                    if constexpr (MODE == 2) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sum[b][i] = __builtin_fmaf(dw[0][i] * dx, f[i], sum[b][i]);
                            sum[b][i] = __builtin_fmaf(dw[1][i], sx, sum[b][i]);
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[i] *= dw[0][i];
#pragma unroll
                        for (int i = 0; i < 8; ++i) sum[b][i] = __builtin_fmaf(f[i], dx, sum[b][i]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b >= nt) break;
        const int r = r0 + b * 16 + l16;
        if (r >= r1) continue;
        const int64_t o = (int64_t) (ids ? ids[r] : r) * ldd + row0 + w * 16 + hi;
#pragma unroll
        for (int i = 0; i < 8; ++i) dst[o + 2 * i] = sum[b][i];
    }
#else
    (void) wq; (void) expert_bytes; (void) k; (void) xq; (void) bounds; (void) ids; (void) total_rows; (void) dst; (void) ldd;
#endif
}


/// v4: k_moe with each wave on two 16-row weight tiles (rows 32 rp .. 32 rp + 31) and every other 16-row activation tile
/// (tg, tg + 2, ..): each B fragment read from global memory feeds two WMMA tiles instead of one.  Same per-element
/// arithmetic (bitwise MMQ).
template <ggml_type type, int MODE, int SKIP = 0>   // SKIP (timing only): 1 no tile loads, 2 no WMMA work
__global__ void __launch_bounds__(256) k_moe_rp(const char* __restrict__ wq, size_t expert_bytes, int k,
                                                const char* __restrict__ xq, const int32_t* __restrict__ bounds,
                                                const int32_t* __restrict__ ids, int total_rows, float* __restrict__ dst,
                                                int64_t ldd) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    constexpr int I = TpTiles<type>::I;
    constexpr int ss = TpTiles<type>::ss;
    constexpr int blocks_per_iter = TpTiles<type>::blocks_per_iter;
    static_assert(I == 128, "the 128-row MMQ tile");
    static_assert(ss <= kStride[MODE], "the host's LDS size (kStride) covers this layout");
    extern __shared__ int tile[];
    const int e = blockIdx.z;
    const int r0 = bounds[e] + blockIdx.x * 128, r1 = min(bounds[e + 1], r0 + 128);
    if (r0 >= r1) return;
    const int nt = (r1 - r0 + 15) >> 4;
    const int lane = threadIdx.x, w = threadIdx.y, l16 = lane & 15, hi = lane >> 4;
    const int rp = w & 3, tg = w >> 2;
    const int row0 = blockIdx.y * I;
    const int stride_blocks = k / (MMQ_ITER_K / blocks_per_iter);
    const char* we = wq + (size_t) e * expert_bytes;
    const size_t xstride = (size_t) total_rows * kXBlock;
    const char* xr[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) xr[b] = xq + (size_t) min(r0 + (2 * b + tg) * 16 + l16, total_rows - 1) * kXBlock;
    const int* x_qs = tile;
    const float* x_df = (const float*) (tile + 2 * MMQ_TILE_NE_K);
    const half2* x_dm = (const half2*) (tile + 2 * MMQ_TILE_NE_K);
    constexpr int kB = 0x4B400000;
    constexpr float kM = 12582912.0f;
    float sum[2][4][8];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
    const int KI = k / MMQ_ITER_K;
#pragma unroll 1
    for (int ki = 0; ki < KI; ++ki) {
        if (ki) __syncthreads();
        if (SKIP != 1 || ki == 0) TpTiles<type>::load(we, tile, row0 * stride_blocks + ki * blocks_per_iter, I - 1, stride_blocks);
        __syncthreads();
        if (SKIP == 2) continue;
#pragma unroll 1
        for (int j = 0; j < 8; ++j) {
            const int kb = ki * 8 + j;
            v4i A[2][2];
            float dw[2][2][8];
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int ra = rp * 32 + a * 16 + l16;
                A[a][0] = *(const v4i*) &x_qs[ra * ss + j * 8];
                A[a][1] = *(const v4i*) &x_qs[ra * ss + j * 8 + 4];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rp * 32 + a * 16 + 2 * i + hi;
                    if constexpr (MODE == 0) {
                        dw[a][0][i] = x_df[r * ss + j];
                    } else if constexpr (MODE == 1) {
                        dw[a][0][i] = x_df[r * ss + 2 * j];
                        dw[a][1][i] = x_df[r * ss + 2 * j + 1];
                    } else {
                        const float2 dm = __half22float2(x_dm[r * ss + j]);
                        dw[a][0][i] = dm.x;
                        dw[a][1][i] = dm.y;
                    }
                }
            }
            const size_t xo = (size_t) (kb >> 2) * xstride;
            const int sub = kb & 3;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (2 * b + tg >= nt) break;
                const char* p = xr[b] + xo;
                const v4i B0 = *(const v4i*) (p + 16 + sub * 32), B1 = *(const v4i*) (p + 32 + sub * 32);
                float dx, sx = 0.f;
                if constexpr (MODE == 2) {
                    const float2 ds = __half22float2(((const half2*) p)[sub]);
                    dx = ds.x;
                    sx = ds.y;
                } else {
                    dx = ((const float*) p)[sub];
                }
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    if constexpr (MODE == 1) {
#pragma unroll
                        for (int hh = 0; hh < 2; ++hh) {
                            v8i c = {kB, kB, kB, kB, kB, kB, kB, kB};
                            c = wmma_i8(A[a][hh], hh ? B1 : B0, c);
                            float f[8];
#pragma unroll
                            for (int i = 0; i < 8; ++i) f[i] = (__int_as_float(c[i]) - kM) * dw[a][hh][i];
#pragma unroll
                            for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[i], dx, sum[a][b][i]);
                        }
                    } else {
                        v8i c = {kB, kB, kB, kB, kB, kB, kB, kB};
                        c = wmma_i8(A[a][0], B0, c);
                        c = wmma_i8(A[a][1], B1, c);
                        float f[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[i] = __int_as_float(c[i]) - kM;
                        if constexpr (MODE == 2) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                sum[a][b][i] = __builtin_fmaf(dw[a][0][i] * dx, f[i], sum[a][b][i]);
                                sum[a][b][i] = __builtin_fmaf(dw[a][1][i], sx, sum[a][b][i]);
                            }
                        } else {
#pragma unroll
                            for (int i = 0; i < 8; ++i) f[i] *= dw[a][0][i];
#pragma unroll
                            for (int i = 0; i < 8; ++i) sum[a][b][i] = __builtin_fmaf(f[i], dx, sum[a][b][i]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        if (2 * b + tg >= nt) break;
        const int r = r0 + (2 * b + tg) * 16 + l16;
        if (r >= r1) continue;
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            const int64_t o = (int64_t) (ids ? ids[r] : r) * ldd + row0 + rp * 32 + a * 16 + hi;
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[o + 2 * i] = sum[a][b][i];
        }
    }
#else
    (void) wq; (void) expert_bytes; (void) k; (void) xq; (void) bounds; (void) ids; (void) total_rows; (void) dst; (void) ldd;
#endif
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t moe_rsrc(const void* p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short) 0, (int) bytes, 0x31004000);
}

/// v5: k_moe_rp's tiling (wave = 32 weight rows x the token tiles tg, tg + 2, ..) without its per-block global
/// latency: NTW (token tiles per wave) is a template chosen per block (uniform), the B operands of every tile are loaded
/// unconditionally (rows clamped into the expert, masked at the store) with buffer loads (fixed per-lane offsets, scalar
/// K offsets) one 32-value block ahead into a two-deep register ring.  Same per-element arithmetic as k_moe_rp.
/// kAS = the split decoders compute the sign masks (no sign-table lookups); GU = fused gate/up + SwiGLU (below).
/// XA >= 0 (v73, balanced): the block's odd last token tile (2 NTW) rides in every wave, for weight tile a = XA only
/// (the waves tg = 0 / 1 take a = 0 / 1 of it: every wave has the same WMMA work); GU: the up half (XA 1) goes
/// through LDS to the gate half's wave for the SwiGLU.
template <ggml_type type, int MODE, int NTW, bool kFast, bool kMidBarrier = true, int SKIP = 0, bool kAS = false, bool GU = false,
          int XA = -1, bool Q8 = false>
__device__ __forceinline__ void moe5_body(const int* __restrict__ tile, const char* __restrict__ we, int k,
                                          __amdgpu_buffer_rsrc_t rx, uint32_t xstride, int r0, int r1,
                                          const int32_t* __restrict__ ids, float* __restrict__ dst, int64_t ldd,
                                          int row0, const int* __restrict__ tab, const char* __restrict__ we2 = nullptr,
                                          float limit = 0.f) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    constexpr int I = TpTiles<type>::I;
    if constexpr (MODE == 3)
        static_assert(I == 128 && ggml_cuda_mmq_get_nthreads(type, TpTiles<type>::kJ, false) == 256 && TpTiles<type>::ss <= kStride[3],
                      "Q2_K: MMQ's 128-row, 256-thread tile");
    if constexpr (MODE == 4)
        static_assert(I == 128 && ggml_cuda_mmq_get_nthreads(type, TpTiles<type>::kJ, false) == 256 && TpTiles<type>::ss <= kStride[4],
                      "Q6_K: MMQ's 128-row, 256-thread tile");
    constexpr bool kSplit = kFast && TpSplit<type>::ok;
    // GU : the tile's rows 0-63 are gate rows row0.., 64-127 the up rows row0..; wave rp multiplies gate tile rp
    // (a = 0) and up tile rp (a = 1) - the same 16 features - and writes silu(g) u (k_swiglu_clamp's expression)
    static_assert(!GU || kSplit, "fused gate/up: split decoders");
    constexpr int NBK = 8;   // 32-value blocks per K step
    constexpr int ss = kSplit ? kTpSplitSS : TpTiles<type>::ss;
    constexpr int blocks_per_iter = TpTiles<type>::blocks_per_iter;
    const int lane = threadIdx.x, w = threadIdx.y, l16 = lane & 15, hi = lane >> 4;
    const int rp = w & 3, tg = w >> 2;
    const int stride_blocks = k / (MMQ_ITER_K / blocks_per_iter);
    const int* x_qs = tile;
    const float* x_df = (const float*) (tile + 2 * MMQ_TILE_NE_K);
    const half2* x_dm = (const half2*) (tile + 2 * MMQ_TILE_NE_K);
    constexpr int NT2 = NTW + (XA >= 0 ? 1 : 0);   // token tiles this wave loads (the extra one last)
    uint32_t vb[NT2 > 0 ? NT2 : 1];
#pragma unroll
    for (int b = 0; b < NTW; ++b) vb[b] = (uint32_t) min(r0 + (2 * b + tg) * 16 + l16, r1 - 1) * kXBlock;
    if constexpr (XA >= 0) vb[NTW] = (uint32_t) min(r0 + 2 * NTW * 16 + l16, r1 - 1) * kXBlock;
    struct BOps {
        v4i B[NT2 > 0 ? NT2 : 1][2];
        uint32_t d[NT2 > 0 ? NT2 : 1];
        uint32_t s[MODE == 3 && NT2 > 0 ? NT2 : 1];   // mode 3: ds[1 + sub] (the D2S6 layout's 16-value sums of the block)
    };
    const int KB = (k + MMQ_ITER_K - 1) / MMQ_ITER_K * 8;   // (a K tail as MMQ: the whole last K step; the blocks
                                                            // past k read zero activations: the q8_1 padding / out of range)
    auto bload = [&](int kb, BOps& o) {
        const uint32_t so = (uint32_t) (kb >> 2) * xstride + (kb & 3) * 32 + 16, sd = (uint32_t) (kb >> 2) * xstride + (kb & 3) * 4;
#pragma unroll
        for (int b = 0; b < NT2; ++b) {
            o.B[b][0] = __builtin_amdgcn_raw_buffer_load_b128(rx, vb[b], so, 0);
            o.B[b][1] = __builtin_amdgcn_raw_buffer_load_b128(rx, vb[b] + 16, so, 0);
            if constexpr (MODE == 3) {   // D2S6: ds[0] = {d of values 0-63, d of 64-127}, ds[1..3] the sums of groups 0-5
                const uint32_t s0 = (uint32_t) (kb >> 2) * xstride;
                o.d[b] = __builtin_amdgcn_raw_buffer_load_b32(rx, vb[b], s0, 0);
                o.s[b] = __builtin_amdgcn_raw_buffer_load_b32(rx, vb[b], s0 + 4 * (1 + (kb & 3)), 0);
            } else {
                o.d[b] = __builtin_amdgcn_raw_buffer_load_b32(rx, vb[b], sd, 0);
            }
        }
    };
    constexpr int kB = 0x4B400000;
    constexpr float kM = 12582912.0f;
    float sum[2][NTW > 0 ? NTW : 1][8], sumx[8];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < NTW; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) sum[a][b][i] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sumx[i] = 0.f;
    // the accumulators of (weight tile a, token tile b): the extra tile (b == NTW) has only a == XA
#define TP_ACC(a, b) ((b) < NTW ? sum[a][(b) < NTW ? (b) : 0] : sumx)
    // B prefetch distance: two blocks ahead (ring of 4) for up to 2 token tiles per wave, else one (ring of 2)
    constexpr int PD = NT2 <= 2 ? 2 : 1;
    BOps ring[2 * PD];
    bload(0, ring[0]);
    if constexpr (PD == 2) bload(min(1, KB - 1), ring[1]);
    constexpr bool kCS = kFast && TpSplit<type>::ok;   // compact scales (split decoders)
    constexpr int SC = MODE == 0 ? 1 : 2;
    const float* cs = (const float*) (tab + tp_table_words<type>());
    // SUB (mode 3): j & 3, compile-time (the 32-value block's place in its 128-value activation block); -1 elsewhere
    auto block = [&]<int SUB>(int j, const BOps& cur) {
        v4i A[2][2];
        float dw[2][2][8];
        int scq[MODE == 4 ? 2 : 1][2][8];   // mode 4: the int8 scales of the block's two 16-value halves
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            const int rt = GU ? a * 4 + rp : rp * 2 + a;   // 16-row tile of the weight tile
            const int ra = rt * 16 + l16;
            A[a][0] = *(const v4i*) &x_qs[ra * ss + j * 8];
            A[a][1] = *(const v4i*) &x_qs[ra * ss + j * 8 + 4];
            if constexpr (kCS) {
#pragma unroll
                for (int c = 0; c < SC; ++c) {
                    const float* q = cs + ((rt * NBK + j) * SC + c) * 16 + hi * 8;
                    const float4 s0 = *(const float4*) q, s1 = *(const float4*) (q + 4);
                    dw[a][c][0] = s0.x; dw[a][c][1] = s0.y; dw[a][c][2] = s0.z; dw[a][c][3] = s0.w;
                    dw[a][c][4] = s1.x; dw[a][c][5] = s1.y; dw[a][c][6] = s1.z; dw[a][c][7] = s1.w;
                }
                continue;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int r = rt * 16 + 2 * i + hi;
                if constexpr (MODE == 0) {
                    dw[a][0][i] = x_df[r * ss + j];
                } else if constexpr (MODE == 1) {
                    dw[a][0][i] = x_df[r * ss + 2 * j];
                    dw[a][1][i] = x_df[r * ss + 2 * j + 1];
                } else if constexpr (MODE == 3) {
                    // (loaded per 16-value group below)
                } else if constexpr (MODE == 4) {
                    // MMQ's Q6_K tile: row r's d (fp32) at int 2 * MMQ_TILE_NE_K, the 16 int8 scales from int + 1
                    dw[a][0][i] = x_df[r * ss];
                    const int8_t* scb = (const int8_t*) (x_df + r * ss + 1);
                    scq[a][0][i] = scb[2 * j];
                    scq[a][1][i] = scb[2 * j + 1];
                } else {
                    const float2 dm = __half22float2(x_dm[r * ss + j]);
                    dw[a][0][i] = dm.x;
                    dw[a][1][i] = dm.y;
                }
            }
        }
#pragma unroll
        for (int b = 0; b < NT2; ++b) {
            const v4i B0 = cur.B[b][0], B1 = cur.B[b][1];
            if constexpr (MODE == 3) {
                // MMQ's Q2_K mma dot per 16-value group g of the 128-value activation block (sub = j & 3: groups 2 sub,
                // 2 sub + 1): tmp = (d sc) C;  for the last 32 values (sub 3, no stored sums) tmp = fma(C, d sc,
                // -fl((dmin m) Cm)) with Cm the group's code sum;  sum = fma(tmp, dB, sum);  sum = fma(-(dmin m), sB, sum)
                // (sB = 0 for sub 3) - the instruction sequence of mul_mat_q<Q2_K> on gfx1151 (fma_mix of the halves)
                constexpr int sub = SUB;
                static_assert(SUB >= 0 && SUB < 4, "mode 3: the block's place is compile-time");
                const half2 dds = __builtin_bit_cast(half2, cur.d[b]);
                const float dB = sub < 2 ? __low2float(dds) : __high2float(dds);
                const half2 sss = __builtin_bit_cast(half2, cur.s[b]);
                const float sB[2] = {sub < 3 ? __low2float(sss) : 0.0f, sub < 3 ? __high2float(sss) : 0.0f};
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const v4i Bh = hh ? B1 : B0;
                    float cmf = 0.0f;
                    if constexpr (sub == 3) {
                        int cm = 0;
#pragma unroll
                        for (int q = 0; q < 4; ++q) cm = __builtin_amdgcn_sudot4(true, Bh[q], true, 0x01010101, cm, false);
                        cmf = (float) cm;
                    }
#pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        if (b >= NTW && a != XA) continue;
                        const int rt = rp * 2 + a;
                        half2 dmh[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) dmh[i] = x_dm[(rt * 16 + 2 * i + hi) * ss + 2 * j + hh];
                        v8i c = {kB, kB, kB, kB, kB, kB, kB, kB};
                        c = wmma_i8(A[a][hh], Bh, c);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float f = __int_as_float(c[i]) - kM;
                            const float dmx = __low2float(dmh[i]), dmy = __high2float(dmh[i]);
                            float tmp;
                            if constexpr (sub == 3) tmp = __builtin_fmaf(f, dmx, -(dmy * cmf));
                            else tmp = dmx * f;
                            TP_ACC(a, b)[i] = __builtin_fmaf(tmp, dB, TP_ACC(a, b)[i]);
                            TP_ACC(a, b)[i] = __builtin_fmaf(-dmy, sB[hh], TP_ACC(a, b)[i]);
                        }
                    }
                }
                continue;
            }
            float dx, sx = 0.f;
            if constexpr (MODE == 2) {
                const float2 ds = __half22float2(__builtin_bit_cast(half2, cur.d[b]));
                dx = ds.x;
                sx = ds.y;
            } else {
                dx = __uint_as_float(cur.d[b]);
            }
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                if (b >= NTW && a != XA) continue;
                if constexpr (MODE == 4) {
                    // mul_mat_q<Q6_K> on gfx1151 per 16-value half: sum = fma(fl(fl(C sc) d), dB, sum) (v_mul_lo_u32,
                    // v_cvt_f32_i32, v_mul_f32, v_fmac_f32), first half then second
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
                        c = wmma_i8(A[a][hh], hh ? B1 : B0, c);
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const float f = (float) (c[i] * scq[a][hh][i]) * dw[a][0][i];
                            TP_ACC(a, b)[i] = __builtin_fmaf(f, dx, TP_ACC(a, b)[i]);
                        }
                    }
                } else if constexpr (MODE == 1) {
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        v8i c = {kB, kB, kB, kB, kB, kB, kB, kB};
                        c = wmma_i8(A[a][hh], hh ? B1 : B0, c);
                        float f[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[i] = (__int_as_float(c[i]) - kM) * dw[a][hh][i];
#pragma unroll
                        for (int i = 0; i < 8; ++i) TP_ACC(a, b)[i] = __builtin_fmaf(f[i], dx, TP_ACC(a, b)[i]);
                    }
                } else {
                    v8i c = {kB, kB, kB, kB, kB, kB, kB, kB};
                    c = wmma_i8(A[a][0], B0, c);
                    c = wmma_i8(A[a][1], B1, c);
                    float f[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) f[i] = __int_as_float(c[i]) - kM;
                    if constexpr (MODE == 2) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            TP_ACC(a, b)[i] = __builtin_fmaf(dw[a][0][i] * dx, f[i], TP_ACC(a, b)[i]);
                            TP_ACC(a, b)[i] = __builtin_fmaf(dw[a][1][i], sx, TP_ACC(a, b)[i]);
                        }
                    } else {
#pragma unroll
                        for (int i = 0; i < 8; ++i) f[i] *= dw[a][0][i];
#pragma unroll
                        for (int i = 0; i < 8; ++i) TP_ACC(a, b)[i] = __builtin_fmaf(f[i], dx, TP_ACC(a, b)[i]);
                    }
                }
            }
        }
    };
    const int KI = (k + 32 * NBK - 1) / (32 * NBK);
    auto compute = [&](int ki) {
        if constexpr (PD == 2) {
#pragma unroll 1
            for (int j = 0; j < NBK; j += 4) {
                const int kb = ki * NBK + j;
                auto step = [&]<int U>() {
                    bload(min(kb + U + 2, KB - 1), ring[(U + 2) & 3]);
                    block.template operator()<MODE == 3 ? U : -1>(j + U, ring[U]);
                    __builtin_amdgcn_sched_barrier(0);
                };
                step.template operator()<0>();
                step.template operator()<1>();
                step.template operator()<2>();
                step.template operator()<3>();
            }
        } else if constexpr (MODE == 3) {   // 4 blocks a step: the blocks' places (j & 3) compile-time
#pragma unroll 1
            for (int j = 0; j < NBK; j += 4) {
                const int kb = ki * NBK + j;
                bload(min(kb + 1, KB - 1), ring[1]);
                block.template operator()<0>(j, ring[0]);
                __builtin_amdgcn_sched_barrier(0);
                bload(min(kb + 2, KB - 1), ring[0]);
                block.template operator()<1>(j + 1, ring[1]);
                __builtin_amdgcn_sched_barrier(0);
                bload(min(kb + 3, KB - 1), ring[1]);
                block.template operator()<2>(j + 2, ring[0]);
                __builtin_amdgcn_sched_barrier(0);
                bload(min(kb + 4, KB - 1), ring[0]);
                block.template operator()<3>(j + 3, ring[1]);
                __builtin_amdgcn_sched_barrier(0);
            }
        } else {
#pragma unroll 1
            for (int j = 0; j < NBK; j += 2) {
                const int kb = ki * NBK + j;
                bload(min(kb + 1, KB - 1), ring[1]);
                block.template operator()<-1>(j, ring[0]);
                if constexpr (kMidBarrier) __builtin_amdgcn_sched_barrier(0);
                bload(min(kb + 2, KB - 1), ring[0]);
                block.template operator()<-1>(j + 1, ring[1]);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
    };
    if constexpr (kFast && TpSplit<type>::ok) {
        // the next K step's weights are fetched into registers before this step's WMMAs, decoded into the tile after
        TpSplit<type> raw;
        float* csw = (float*) (tab + tp_table_words<type>());
        raw.fetch(we, row0 * stride_blocks, I - 1, stride_blocks, GU ? we2 : nullptr);
        raw.template decode<kAS>((int*) tile, tab, I - 1, csw);
        __syncthreads();
#pragma unroll 1
        for (int ki = 0; ki < KI; ++ki) {
            if (SKIP != 1) raw.fetch(we, row0 * stride_blocks + min(ki + 1, KI - 1) * blocks_per_iter, I - 1, stride_blocks, GU ? we2 : nullptr);
            if (SKIP != 2) compute(ki);
            __syncthreads();
            if (SKIP != 1 && ki + 1 < KI) {
                raw.template decode<kAS>((int*) tile, tab, I - 1, csw);
                __syncthreads();
            }
        }
    } else {
#pragma unroll 1
        for (int ki = 0; ki < KI; ++ki) {
            if (ki) __syncthreads();
            if constexpr (kFast && tp_table_words<type>() > 0)
                tp_load_tiles_fast<type>(we, (int*) tile, row0 * stride_blocks + ki * blocks_per_iter, I - 1, stride_blocks, tab);
            else
                TpTiles<type>::load(we, (int*) tile, row0 * stride_blocks + ki * blocks_per_iter, I - 1, stride_blocks);
            __syncthreads();
            compute(ki);
        }
    }
    if constexpr (GU && Q8) {
        // the SwiGLU values quantized straight into the down projection's MMQ q8_1 rows (D4 layout: one fp32
        // scale per 32 values), bitwise llama.cpp's quantize_mmq_q8_1 of the FP32 rows this kernel would write:
        //   amax over the 32 values (fmaxf: exact, any order), d_inv = 127 / amax, q = (int8) roundf(v d_inv),
        //   d = 1 / d_inv.  Here dst = the q8 buffer (block_q8_1_mmq [ne1 = ldd rows] per 128 features).
        // The 32-value group = features row0 + 32 (rp >> 1) + 0..31 = waves rp, rp ^ 1 (same tg): amax via LDS.
        float val[NTW + 1][8];
#pragma unroll
        for (int b = 0; b < NTW; ++b)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float gv = sum[0][b][i], uv = sum[1][b][i];
                if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
                val[b][i] = gv / (1.0f + expf(-gv)) * uv;
            }
        bool hasx = false;   // this wave finishes the extra token tile (XA 0 waves)
        if constexpr (XA >= 0) {
            float* ex = (float*) tile + (rp * 32 + lane) * 8;
            if constexpr (XA == 1) {
                *(float4*) ex = make_float4(sumx[0], sumx[1], sumx[2], sumx[3]);
                *(float4*) (ex + 4) = make_float4(sumx[4], sumx[5], sumx[6], sumx[7]);
            }
            __syncthreads();
            if constexpr (XA == 0) {
                hasx = true;
                const float4 u0 = *(const float4*) ex, u1 = *(const float4*) (ex + 4);
                const float up[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float gv = sumx[i], uv = up[i];
                    if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
                    val[NTW][i] = gv / (1.0f + expf(-gv)) * uv;
                }
            }
        }
        // per token (column l16) the amax of this wave's 16 features, then the partner wave's via LDS
        float* am = (float*) tile + 4 * 32 * 8;   // [rp][token tile T (<= 7)][16]
        float amw[NTW + 1];
#pragma unroll
        for (int b = 0; b <= NTW; ++b) {
            if (b == NTW && !hasx) { amw[b] = 0.0f; continue; }
            float m = fabsf(val[b][0]);
#pragma unroll
            for (int i = 1; i < 8; ++i) m = fmaxf(m, fabsf(val[b][i]));
            m = fmaxf(m, __shfl_xor(m, 16, 32));
            amw[b] = m;
            const int T = b < NTW ? 2 * b + tg : 2 * NTW;
            if (hi == 0) am[(rp * 8 + T) * 16 + l16] = m;
        }
        __syncthreads();
        char* yq = (char*) dst;
        const int64_t ne1 = ldd;
#pragma unroll
        for (int b = 0; b <= NTW; ++b) {
            if (b == NTW && !hasx) continue;
            const int T = b < NTW ? 2 * b + tg : 2 * NTW;
            const int r = r0 + T * 16 + l16;
            const float amax = fmaxf(amw[b], am[((rp ^ 1) * 8 + T) * 16 + l16]);
            const float d_inv = 127.0f / amax;
            const float d = 1.0f / d_inv;
            char qv[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) qv[i] = roundf(val[b][i] * d_inv);
            uint32_t w0 = 0, w1 = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                w0 |= (uint32_t) (uint8_t) qv[i] << (8 * i);
                w1 |= (uint32_t) (uint8_t) qv[4 + i] << (8 * i);
            }
            // features rp * 16 + hi + 2 i: lane hi 0 writes bytes 0-7 of the wave's 16, hi 1 bytes 8-15
            const uint32_t send = hi ? w0 : w1;
            const uint32_t par = (uint32_t) __shfl_xor((int) send, 16, 32);
            const uint32_t own = hi ? w1 : w0;
            const uint32_t ev = hi ? par : own, od = hi ? own : par;   // even-feature bytes, odd-feature bytes
            const uint32_t o0 = __builtin_amdgcn_perm(od, ev, 0x05010400u), o1 = __builtin_amdgcn_perm(od, ev, 0x07030602u);
            if (r < r1) {
                const int f = row0 + rp * 16;
                const int64_t row = ids ? ids[r] : r;
                char* blk = yq + ((int64_t) (f / 128) * ne1 + row) * 144;
                *(uint2*) (blk + 16 + (f % 128) + hi * 8) = make_uint2(o0, o1);
                if ((rp & 1) == 0 && hi == 0) ((float*) blk)[(f % 128) / 32] = d;
            }
        }
        return;
    }
#pragma unroll
    for (int b = 0; b < NTW; ++b) {
        const int r = r0 + (2 * b + tg) * 16 + l16;
        if (r >= r1) continue;
        if constexpr (GU) {
            const int64_t o = (int64_t) (ids ? ids[r] : r) * ldd + row0 + rp * 16 + hi;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float gv = sum[0][b][i], uv = sum[1][b][i];
                if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
                dst[o + 2 * i] = gv / (1.0f + expf(-gv)) * uv;
            }
            continue;
        }
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            const int64_t o = (int64_t) (ids ? ids[r] : r) * ldd + row0 + rp * 32 + a * 16 + hi;
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[o + 2 * i] = sum[a][b][i];
        }
    }
    if constexpr (XA >= 0) {   // the extra token tile 2 NTW, weight tile XA
        const int r = r0 + 2 * NTW * 16 + l16;
        if constexpr (GU) {
            // the up half (XA 1) to LDS (the tile is free: the K loop ended with a barrier), the gate half finishes
            float* ex = (float*) tile + (rp * 32 + lane) * 8;
            if constexpr (XA == 1) {
                *(float4*) ex = make_float4(sumx[0], sumx[1], sumx[2], sumx[3]);
                *(float4*) (ex + 4) = make_float4(sumx[4], sumx[5], sumx[6], sumx[7]);
            }
            __syncthreads();
            if constexpr (XA == 0) {
                if (r < r1) {
                    const float4 u0 = *(const float4*) ex, u1 = *(const float4*) (ex + 4);
                    const float up[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
                    const int64_t o = (int64_t) (ids ? ids[r] : r) * ldd + row0 + rp * 16 + hi;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        float gv = sumx[i], uv = up[i];
                        if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
                        dst[o + 2 * i] = gv / (1.0f + expf(-gv)) * uv;
                    }
                }
            }
        } else if (r < r1) {
            const int64_t o = (int64_t) (ids ? ids[r] : r) * ldd + row0 + rp * 32 + XA * 16 + hi;
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[o + 2 * i] = sumx[i];
        }
    }
#undef TP_ACC
#else
    (void) tile; (void) we; (void) k; (void) rx; (void) xstride; (void) r0; (void) r1; (void) ids; (void) dst; (void) ldd; (void) row0; (void) tab;
#endif
}

/// grid (ceil(max_rows / 32 NTWM), n / 128 (GU: n / 64), n_experts), block (32, 8).  NTWM : token tiles per wave
/// at most - 3 = 96-row passes (fewer VGPRs than v6's 4); the expert's 32 NTWM-row chunks are the grid's x.
template <ggml_type type, int MODE, bool kFast, bool kMid = true, int SKIP = 0, int NTWM = 4, bool kAS = false, bool GU = false,
          bool BAL = false, bool Q8 = false>
__global__ void __launch_bounds__(256) k_moe5(const char* __restrict__ wq, size_t expert_bytes, int k,
                                              const char* __restrict__ xq, const int32_t* __restrict__ bounds,
                                              const int32_t* __restrict__ ids, int total_rows, float* __restrict__ dst,
                                              int64_t ldd, const char* __restrict__ wq2 = nullptr, float limit = 0.f) {
    extern __shared__ int tile[];
    const int e = blockIdx.z;
    const int eb = bounds[e], ee = bounds[e + 1];
    constexpr int CH = 32 * NTWM;   // activation rows per pass (NTWM token tiles per wave)
    const int r0 = eb + blockIdx.x * CH;
    if (r0 >= ee) return;
    const uint32_t xstride = (uint32_t) total_rows * kXBlock;
    const __amdgpu_buffer_rsrc_t rx = moe_rsrc(xq, (uint32_t) (k / 128) * xstride);
    const char* we = wq + (size_t) e * expert_bytes;
    const int row0 = blockIdx.y * (GU ? 64 : 128);
    const char* we2 = GU ? wq2 + (size_t) e * expert_bytes : nullptr;
    int* tab = nullptr;
    if constexpr (kFast && (tp_table_words<type>() > 0 || TpSplit<type>::ok)) {   // tables (+ compact scales) after the tile
        tab = tile + 128 * (TpSplit<type>::ok ? kTpSplitSS : kStride[MODE]);
        tp_tables_init<type>(tab);
        __syncthreads();
    }
    {
        const int r1 = min(ee, r0 + CH);
        const int nt = (r1 - r0 + 15) >> 4;   // 16-row activation tiles (1..2 NTWM); a wave takes every other one
        // this wave's token tiles: tg, tg + 2, .. < nt (wave-uniform; the barriers inside are the same for every count)
        if constexpr (BAL) {
            // v73: an odd last tile shared by the two wave halves (weight tile tg of it): the same work in every wave
            static_assert(NTWM == 3 || NTWM == 4, "v73 / v74: 96- / 128-row passes");
            const int ntw = nt >> 1;
            const bool g1 = (threadIdx.y >> 2) != 0;
#define TP_MOE5_B(N, X) moe5_body<type, MODE, N, kFast, kMid, SKIP, kAS, GU, X, Q8>(tile, we, k, rx, xstride, r0, r1, ids, dst, ldd, row0, tab, we2, limit)
            if (nt & 1) {
                if (ntw == 0) { if (g1) TP_MOE5_B(0, 1); else TP_MOE5_B(0, 0); }
                else if (ntw == 1) { if (g1) TP_MOE5_B(1, 1); else TP_MOE5_B(1, 0); }
                else if (NTWM == 3 || ntw == 2) { if (g1) TP_MOE5_B(2, 1); else TP_MOE5_B(2, 0); }
                else if constexpr (NTWM >= 4) { if (g1) TP_MOE5_B(3, 1); else TP_MOE5_B(3, 0); }
            } else {
                if (ntw == 1) TP_MOE5_B(1, -1);
                else if (ntw == 2) TP_MOE5_B(2, -1);
                else if (NTWM == 3 || ntw == 3) TP_MOE5_B(3, -1);
                else if constexpr (NTWM >= 4) TP_MOE5_B(4, -1);
            }
#undef TP_MOE5_B
            return;
        }
        const int ntw = (nt + 1 - (int) (threadIdx.y >> 2)) >> 1;
        if (ntw == 0) moe5_body<type, MODE, 0, kFast, kMid, SKIP, kAS, GU>(tile, we, k, rx, xstride, r0, r1, ids, dst, ldd, row0, tab, we2, limit);
        else if (ntw == 1) moe5_body<type, MODE, 1, kFast, kMid, SKIP, kAS, GU>(tile, we, k, rx, xstride, r0, r1, ids, dst, ldd, row0, tab, we2, limit);
        else if (ntw == 2) moe5_body<type, MODE, 2, kFast, kMid, SKIP, kAS, GU>(tile, we, k, rx, xstride, r0, r1, ids, dst, ldd, row0, tab, we2, limit);
        else if constexpr (NTWM == 3) moe5_body<type, MODE, 3, kFast, kMid, SKIP, kAS, GU>(tile, we, k, rx, xstride, r0, r1, ids, dst, ldd, row0, tab, we2, limit);
        else if constexpr (NTWM >= 4) {
            if (ntw == 3) moe5_body<type, MODE, 3, kFast, kMid, SKIP, kAS, GU>(tile, we, k, rx, xstride, r0, r1, ids, dst, ldd, row0, tab, we2, limit);
            else moe5_body<type, MODE, 4, kFast, kMid, SKIP, kAS, GU>(tile, we, k, rx, xstride, r0, r1, ids, dst, ldd, row0, tab, we2, limit);
        }
    }
}

// ---- IQ1_M on plain q8_1 rows, bitwise the dp4a grouped kernel (strata::kernels::mimo_grouped_mmq_kernel<29>) ------
//
// What the dp4a kernel computes per output (weight row n, routed row r), read from its gfx1151 code:
//   per 32-value block b:  X_b = (sum_16 (g + delta) u) sc0 + (sum_16 (g + delta) u) sc1   (exact in fp32: multiples of
//                          1/8 below 2^18), c_b = fl(dw_sb * d8_b) (dw the superblock's fp16 scale, d8 the q8_1 row's
//                          fp16 d), and the lane's partial  p = fma(c_b, X_b, p)  (v_fmac_f32, from p = +0);
//   lane L (0..31) of the wave owns the blocks b = 32 i + L (i = 0.. K / 1024 - 1, in that order), and the 32 partials
//   are summed by the xor-1, 2, 4, 8, 16 butterfly (mimo_reduce_rows): a balanced pairwise tree in lane order.
// Here: X_b from two int8 WMMAs (the IQ1_M weights exactly as int8 8 (g + delta), the activations the q8_1 codes), with
// the fp32 X = fma(f0, sc0 / 8, f1 * sc1 / 8) (exact), c = dw * d8, p = fma(c, X, p) per leaf L in the same block
// order, and the tree folded leaf by leaf (a stack of 5 partial sums; float addition commutes, so the butterfly's
// operand order does not matter).  The K loop runs leaf-major: leaf L = 8 g + s takes sub-block s of the superblocks
// 4 i + g.  Output bits = the dp4a kernel's for every row (talus-pf-bench moe1m checks it).
//
// Tiles: a block = 64 weight rows x 64 routed rows (one pass; longer experts take more blocks along grid x).  Wave w:
// token tile w >> 1, weight tiles 2 (w & 1), 2 (w & 1) + 1 (the weights are the WMMA's B operand, so a C lane holds one
// weight row and 8 routed rows).  Per leaf the 64 x (K / 1024 x 32) weight values are decoded into an LDS tile
// (double-buffered, the next leaf's raw bytes fetched into registers before this leaf's WMMAs).
namespace iq1m {
constexpr int kRows = 64;     // weight rows per block
constexpr int kTok = 64;      // routed rows per block
constexpr int kWS = 36;       // ints per weight row of a leaf tile (NI = 4: 32 + 4 padding)
constexpr int kQB = 36;       // bytes per q8_1 block
}

// GU: fused gate / up + SwiGLU - the block's 64 tile rows are 32 gate rows (wq) and the same 32 up rows (wq2); wave wp
// multiplies gate tile wp (a = 0) and up tile wp (a = 1) and writes g / (1 + exp(-g)) u (tk_silu_mul's expression).
// TOKT: routed rows per block (64: 8 waves; 128: 16 waves, one pass for experts up to 128 rows)
// ONE (TP_MOE_IQ1M_ONE=1): one C tile per wave (TOKT / 4 waves: token tile w >> 2, tile row block w & 3) -
// half the accumulator / tree registers per wave; the leaf's activations staged in LDS once per block (the 4 waves of
// a token tile share them); GU: the up waves hand their results to the gate waves through LDS.  Every C element's
// operand sequence (WMMAs, epilogue, leaf chain, tree) is unchanged: bitwise.
template <int NI, bool GU = false, int TOKT = 64, bool LDSG = false, bool ONE = false>
__global__ void __launch_bounds__(TOKT * (ONE ? 8 : 4)) k_moe1m(const char* __restrict__ wq, size_t expert_bytes, int k, int n,
                                               const char* __restrict__ xq, const int32_t* __restrict__ bounds,
                                               const int32_t* __restrict__ ids, int total_rows, float* __restrict__ dst,
                                               int64_t ldd, const char* __restrict__ wq2 = nullptr) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    using namespace iq1m;
    static_assert(NI * 8 + 4 <= kWS, "leaf tile row");
    // LDS: two leaf tiles {values [64][kWS] ints, scales [64][NI (+1)] float4 {sc0/8, sc1/8, dw, 0}}, d8 [2][4 tt][NI][2][8]
    __shared__ __attribute__((aligned(16))) int wt[2][kRows * kWS];
    constexpr int SCS = NI + 1;   // float4 per weight row of the scale table (+1: 16 lanes' b128 reads hit distinct banks)
    __shared__ __attribute__((aligned(16))) float4 wsc[2][kRows * SCS];
    __shared__ __attribute__((aligned(16))) float d8s[2][(TOKT / 16) * NI * 16];
    // LDSG (TP_MOE_IQ1M_LDSGRID=1): the IQ1_S/M grid (2048 x 32 bits) in LDS, so the decode's lookups do not
    // wait on global memory, and each block's activations loaded one block ahead; the same words and operands (bitwise)
    __shared__ uint32_t gtab[LDSG ? NGRID_IQ1S : 1];
    // ONE: the leaf's activation bytes [2][TOKT rows][NI blocks x 32 B + 16 B pad] (ints)
    constexpr int XS = NI * 8 + 4;
    __shared__ __attribute__((aligned(16))) int xs[ONE ? 2 : 1][ONE ? TOKT * XS : 4];
    const int e = blockIdx.z;
    const int eb = bounds[e], ee = bounds[e + 1];
    const int r0 = eb + (int) blockIdx.x * TOKT;
    if (r0 >= ee) return;
    const int r1 = min(ee, r0 + TOKT);
    const int nt = (r1 - r0 + 15) >> 4;
    const int n0 = (int) blockIdx.y * (GU ? kRows / 2 : kRows);
    const int tid = threadIdx.y * 32 + threadIdx.x, lane = threadIdx.x, w = threadIdx.y, l16 = lane & 15, hi = lane >> 4;
    const int tt = ONE ? w >> 2 : w >> 1, wp = ONE ? w & 3 : w & 1;   // token tile, weight tile pair (ONE: tile)
    constexpr int NA = ONE ? 1 : 2;      // C tiles per wave
    const int bpr = k / 256;             // superblocks per row
    const size_t xrow = (size_t) (k / 32) * kQB;
    const block_iq1_m* we = (const block_iq1_m*) (wq + (size_t) e * expert_bytes);
    // decode unit of this thread: tile row dr, superblock index di (sb = 4 di + g); d8 unit: token dt, block di
    const int dr = tid >> 2, di = tid & 3;   // (weights: the first 256 threads; d8: all TOKT x 4)
    const bool wdec = tid < kRows * 4;
    const block_iq1_m* wrow;
    if constexpr (GU) {
        const block_iq1_m* we2 = (const block_iq1_m*) (wq2 + (size_t) e * expert_bytes);
        wrow = (dr < kRows / 2 ? we + (size_t) (n0 + dr) * bpr : we2 + (size_t) (n0 + min(dr, kRows - 1) - kRows / 2) * bpr);
    } else {
        wrow = we + (size_t) (n0 + min(dr, kRows - 1)) * bpr;
    }
    const char* xd = xq + (size_t) min(r0 + dr, r1 - 1) * xrow;   // (d8 unit: token dr, block 32 di + L)
    // this lane's A row (a routed row of token tile tt)
    const char* xa = xq + (size_t) min(r0 + tt * 16 + l16, r1 - 1) * xrow;
    struct Raw { int qs, qh, s01, s23; uint32_t d8; v4i xa; };
    // ONE: activation unit of this thread: routed row tid >> 3, block (tid >> 1) & 3, 16-byte half tid & 1
    const int au_r = tid >> 3, au_i = (tid >> 1) & 3, au_h = tid & 1;
    const char* xau = xq + (size_t) min(r0 + au_r, r1 - 1) * xrow + 4 + 16 * au_h;
    auto fetch = [&](int L, Raw& o) {
        const int g = L >> 3, sub = L & 7;
        if (di < NI) {
            if (wdec) {
                const block_iq1_m* b = wrow + 4 * di + g;
                o.qs = get_int_b2(b->qs, sub);
                o.qh = ((const uint16_t*) b->qh)[sub];
                o.s01 = get_int_b2(b->scales, 0);
                o.s23 = get_int_b2(b->scales, 1);
            }
            o.d8 = *(const uint16_t*) (xd + (size_t) (32 * di + L) * kQB);
        }
        if constexpr (ONE) {
            if (au_r < TOKT && au_i < NI) {
                const int* pi = (const int*) (xau + (size_t) (32 * au_i + L) * kQB);
                o.xa = (v4i){pi[0], pi[1], pi[2], pi[3]};
            }
        }
    };
    auto decode = [&](int L, const Raw& o, int buf) {
        const int sub = L & 7;
        if (di < NI && wdec) {
            int v[8];
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int qhl = o.qh >> (4 * l);
                const int gi = ((o.qs >> (8 * l)) & 0xff) | ((qhl & 0x07) << 8);
                const int grid = LDSG ? (int) gtab[gi] : (int) iq1s_grid_gpu[gi];
                const int off = (qhl & 0x08) ? 0x09090909 : 0x07070707;   // 8 (nibble - 1) + 8 delta = 8 nibble - 9 / - 7
                // per byte 8 nibble - off without borrows: (8 nibble + 128 - off) ^ 128 (8 nibble <= 16, off 7 / 9)
                v[2 * l + 0] = ((((grid >> 0) & 0x0F0F0F0F) << 3 | (int) 0x80808080) - off) ^ (int) 0x80808080;
                v[2 * l + 1] = ((((grid >> 4) & 0x0F0F0F0F) << 3 | (int) 0x80808080) - off) ^ (int) 0x80808080;
            }
            int* q = &wt[buf][dr * kWS + di * 8];
            *(int4*) q = make_int4(v[0], v[1], v[2], v[3]);
            *(int4*) (q + 4) = make_int4(v[4], v[5], v[6], v[7]);
            const uint16_t sc[4] = {(uint16_t) o.s01, (uint16_t) ((uint32_t) o.s01 >> 16), (uint16_t) o.s23, (uint16_t) ((uint32_t) o.s23 >> 16)};
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
            const int tmp = sc[sub / 2] >> (6 * (sub % 2));
            wsc[buf][dr * SCS + di] = make_float4((float) (2 * ((tmp >> 0) & 0x07) + 1) * 0.125f,
                                                 (float) (2 * ((tmp >> 3) & 0x07) + 1) * 0.125f, __half2float(scale.f16), 0.f);
        }
        if constexpr (ONE) {
            if (au_r < TOKT && au_i < NI) *(v4i*) &xs[buf][au_r * XS + au_i * 8 + au_h * 4] = o.xa;
        }
        if (di < NI && (!ONE || dr < TOKT)) {
            // d8 of routed row dr (token tile dr >> 4, m = dr & 15 -> [hi = m & 1][m >> 1]), block 32 di + L
            const int m = dr & 15;
            d8s[buf][(((dr >> 4) * NI + di) * 2 + (m & 1)) * 8 + (m >> 1)] = __half2float(__ushort_as_half((unsigned short) o.d8));
        }
    };
    constexpr int kB = 0x4B400000;
    constexpr float kM = 12582912.0f;
    // per C tile a (weight tile 2 wp + a) and element j (routed row 2 j + hi of tile tt): the leaf chain and the tree
    float ch[NA][8], s0[NA][8], s1[NA][8], s2[NA][8], s3[NA][8], s4[NA][8];
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
        for (int j = 0; j < 8; ++j) s0[a][j] = s1[a][j] = s2[a][j] = s3[a][j] = s4[a][j] = 0.f;
    const bool active = tt < nt;
    // the leaf: chain p over i = 0 .. NI - 1 (blocks 32 i + L), from +0
    auto leaf = [&](int L, int buf) {
#pragma unroll
        for (int a = 0; a < NA; ++a)
#pragma unroll
            for (int j = 0; j < 8; ++j) ch[a][j] = 0.f;
        if (!active) return;
        auto aload = [&](int i, v4i& A0, v4i& A1) {
            if constexpr (ONE) {
                const int* q = &xs[buf][(tt * 16 + l16) * XS + i * 8];
                A0 = *(const v4i*) q;
                A1 = *(const v4i*) (q + 4);
                return;
            }
            const int* pi = (const int*) (xa + (size_t) (32 * i + L) * kQB + 4);   // 4-byte aligned
            A0 = (v4i){pi[0], pi[1], pi[2], pi[3]};
            A1 = (v4i){pi[4], pi[5], pi[6], pi[7]};
        };
        v4i An0, An1;   // LDSG: block i + 1's activations loaded before block i's WMMAs
        if constexpr (LDSG) aload(0, An0, An1);
#pragma unroll
        for (int i = 0; i < NI; ++i) {
            v4i A0, A1;
            if constexpr (LDSG) {
                A0 = An0;
                A1 = An1;
                if (i + 1 < NI) aload(i + 1, An0, An1);
            } else {
                aload(i, A0, A1);
            }
            const float4 dA = *(const float4*) &d8s[buf][((tt * NI + i) * 2 + hi) * 8];
            const float4 dB = *(const float4*) &d8s[buf][((tt * NI + i) * 2 + hi) * 8 + 4];
            const float d8[8] = {dA.x, dA.y, dA.z, dA.w, dB.x, dB.y, dB.z, dB.w};
#pragma unroll
            for (int a = 0; a < NA; ++a) {
                const int wr = ONE ? wp * 16 + l16 : (GU ? a * 2 + wp : 2 * wp + a) * 16 + l16;   // this lane's tile row in the block
                const int* q = &wt[buf][wr * kWS + i * 8];
                const v4i B0 = *(const v4i*) q, B1 = *(const v4i*) (q + 4);
                const float4 sc = wsc[buf][wr * SCS + i];
                v8i c0 = {kB, kB, kB, kB, kB, kB, kB, kB}, c1 = c0;
                c0 = wmma_i8(A0, B0, c0);
                c1 = wmma_i8(A1, B1, c1);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float f0 = __int_as_float(c0[j]) - kM, f1 = __int_as_float(c1[j]) - kM;
                    const float x = __builtin_fmaf(f0, sc.x, f1 * sc.y);   // exact
                    const float c = sc.z * d8[j];
                    ch[a][j] = __builtin_fmaf(c, x, ch[a][j]);
                }
            }
            __builtin_amdgcn_sched_barrier(0);   // keep the blocks' operand loads from piling up in registers
        }
    };
    if constexpr (LDSG) {
        for (int i = tid; i < NGRID_IQ1S; i += TOKT * 4) gtab[i] = iq1s_grid_gpu[i];
        __syncthreads();
    }
    Raw raw;
    fetch(0, raw);
    decode(0, raw, 0);
    __syncthreads();
#pragma unroll 1
    for (int g = 0; g < 4; ++g) {
#pragma unroll
        for (int sub = 0; sub < 8; ++sub) {
            const int L = 8 * g + sub, buf = sub & 1;   // (8 leaves per g: the buffer parity is the sub parity)
            if (L + 1 < 32) fetch(L + 1, raw);
            leaf(L, buf);
            // fold leaf L into the tree (lane-order butterfly): levels 0..2 inside g, 3..4 across g
#pragma unroll
            for (int a = 0; a < NA; ++a)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float x = ch[a][j];
                    if (sub & 1) {
                        x = s0[a][j] + x;
                        if (sub & 2) {
                            x = s1[a][j] + x;
                            if (sub & 4) {
                                x = s2[a][j] + x;   // G_g
                                if (g & 1) {
                                    x = s3[a][j] + x;
                                    if (g & 2) s4[a][j] = s4[a][j] + x;   // the root: G01 + G23
                                    else s4[a][j] = x;
                                } else {
                                    s3[a][j] = x;
                                }
                            } else {
                                s2[a][j] = x;
                            }
                        } else {
                            s1[a][j] = x;
                        }
                    } else {
                        s0[a][j] = x;
                    }
                }
            if (L + 1 < 32) decode(L + 1, raw, buf ^ 1);
            __syncthreads();
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    if constexpr (ONE) {
        // (the K loop ended on a barrier: wt is free) GU: up waves (tiles 2, 3) -> LDS -> gate waves (tiles 0, 1)
        float* xu = (float*) &wt[0][0];
        if constexpr (GU) {
            if (active && wp >= 2) {
#pragma unroll
                for (int j = 0; j < 8; ++j) xu[((tt * 2 + (wp - 2)) * 8 + j) * 32 + lane] = s4[0][j];
            }
            __syncthreads();
            if (!active || wp >= 2) return;
        } else {
            if (!active) return;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int r = r0 + tt * 16 + 2 * j + hi;
            if (r >= r1) continue;
            if constexpr (GU) {
                const float gv = s4[0][j], uv = xu[((tt * 2 + wp) * 8 + j) * 32 + lane];
                dst[(int64_t) (ids ? ids[r] : r) * ldd + n0 + wp * 16 + l16] = gv / (1.0f + expf(-gv)) * uv;
            } else {
                dst[(int64_t) (ids ? ids[r] : r) * ldd + n0 + wp * 16 + l16] = s4[0][j];
            }
        }
        (void) n; (void) total_rows;
        return;
    } else {
    if (!active) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = r0 + tt * 16 + 2 * j + hi;
        if (r >= r1) continue;
        if constexpr (GU) {
            const float gv = s4[0][j], uv = s4[1][j];
            dst[(int64_t) (ids ? ids[r] : r) * ldd + n0 + wp * 16 + l16] = gv / (1.0f + expf(-gv)) * uv;
        } else {
            float* o = dst + (int64_t) (ids ? ids[r] : r) * ldd + n0 + 2 * wp * 16 + l16;
            o[0] = s4[0][j];
            o[16] = s4[NA - 1][j];
        }
    }
    }
    (void) n; (void) total_rows;
#else
    (void) wq; (void) expert_bytes; (void) k; (void) n; (void) xq; (void) bounds; (void) ids; (void) total_rows; (void) dst; (void) ldd;
#endif
}

int g_moe_ver = -1;   // talus-pf-bench: the version under test (moe_set_version); -1 = TP_MOE_VER / default

/// the kernel version of a product: talus-pf-bench's, TP_MOE_VER, or the tune row's for grouped products (one expert -
/// the dense tk_prefill_mmq route - keeps v6 until measured); a K tail always takes v6
int moe_version_for(const MoeProduct& p) {
    static const int ver_env = [] { const char* e = std::getenv("TP_MOE_VER"); return e ? std::atoi(e) : -1; }();
    static const int ver_arch = tp_tune_current().moe_ver;
    int ver0 = g_moe_ver >= 0 ? g_moe_ver : ver_env >= 0 ? ver_env : (p.n_experts > 1 ? ver_arch : 0);
    // TP_MOE_V74_MIN=R: v74 (128-row balanced passes, bitwise) from R routed rows per expert on average.  Off by default:
    // faster in the kernel bench at 128 rows per expert, but slower than v73 in-model (MiMo, --ubatch 4096,
    // 8K prompt; logits identical)
    static const int v74_min = [] { const char* e = std::getenv("TP_MOE_V74_MIN"); return e ? std::atoi(e) : 0; }();
    if (g_moe_ver < 0 && ver_env < 0 && ver0 == 73 && v74_min > 0 && p.n_experts > 1 && p.total_rows >= (int64_t) v74_min * p.n_experts)
        ver0 = 74;
    return p.k % MMQ_ITER_K != 0 ? 6 : ver0;
}

template <ggml_type type, int MODE>
constexpr size_t lds_v6() {
    return (TpSplit<type>::ok ? (size_t) 128 * kTpSplitSS * sizeof(int) : (size_t) 128 * kStride[MODE] * sizeof(int)) +
           (size_t) tp_table_words<type>() * sizeof(int) + (TpSplit<type>::ok ? (size_t) 128 * 8 * (MODE == 0 ? 1 : 2) * sizeof(float) : 0);
}

int moe_version_for(const MoeProduct& p);

template <ggml_type type, int MODE>
bool launch_gu(const MoeProduct& p, cudaStream_t s) {
    if constexpr (!TpSplit<type>::ok) {
        return false;
    } else {
        const dim3 grid((unsigned) ((p.max_rows + 95) / 96), (unsigned) (p.n / 64), (unsigned) p.n_experts);
#define TP_GU_ARGS (const char*) p.w, p.expert_bytes, p.k, (const char*) p.xq, p.bounds, p.ids, (int) p.total_rows, p.dst, p.ld_dst, \
                   (const char*) p.w_up, p.swiglu_limit
        if (p.q8_out != nullptr) {   // SwiGLU -> the down projection's q8_1 rows (v73 body only)
            if (moe_version_for(p) < 73) return false;
            k_moe5<type, MODE, true, true, 0, 3, true, true, true, true><<<grid, dim3(32, 8), lds_v6<type, MODE>(), s>>>(
                (const char*) p.w, p.expert_bytes, p.k, (const char*) p.xq, p.bounds, p.ids, (int) p.total_rows,
                (float*) p.q8_out, p.q8_rows, (const char*) p.w_up, p.swiglu_limit);
        } else if (moe_version_for(p) >= 73)   // v73 / v74: balanced odd tiles (the fused kernel keeps 96-row passes)
            k_moe5<type, MODE, true, true, 0, 3, true, true, true><<<grid, dim3(32, 8), lds_v6<type, MODE>(), s>>>(TP_GU_ARGS);
        else
            k_moe5<type, MODE, true, true, 0, 3, true, true><<<grid, dim3(32, 8), lds_v6<type, MODE>(), s>>>(TP_GU_ARGS);
#undef TP_GU_ARGS
        return true;
    }
}

/// Versions (TP_MOE_VER; default per GPU: tp_tune.hpp moe_ver for grouped products, 0 for one expert): 0 / 6 v6
/// (k_moe5, up to 4 token tiles per wave), 7 v6 without the mid-pair scheduling barrier, 63 v6 with at most 3 token tiles
/// per wave (96-row passes), 72 = 63 + the split decoders' sign masks computed (gfx1151's default), 5 v5 (MMQ loaders),
/// 4 v4 (k_moe_rp), 1 v1 (k_moe); timing only (wrong output): 41 / 42 (v4 without tile loads / WMMAs), 61 / 62 (v6
/// without weight fetch+decode / WMMAs).  A K tail (k % 256 != 0) always takes v6.
template <ggml_type type, int MODE>
void launch(const MoeProduct& p, cudaStream_t s) {
    constexpr size_t lds = (size_t) 128 * kStride[MODE] * sizeof(int);
    constexpr size_t lds6 = lds_v6<type, MODE>();
    const dim3 grid((unsigned) ((p.max_rows + 127) / 128), (unsigned) (p.n / 128), (unsigned) p.n_experts);
    const dim3 grid96((unsigned) ((p.max_rows + 95) / 96), (unsigned) (p.n / 128), (unsigned) p.n_experts);
    // Q4_K split loader (TpSplit<Q4_K>, every kFast body). TP_Q4K_SPLIT = 2 (default): every Q4_K product,
    // 1: one-expert (dense) products only (grouped Q4_K experts - Laguna - on MMQ's synchronous loader), 0: nowhere.
    // Without the split every kFast body runs as kFast = false: for Q4_K (no tables) that is MMQ's loader with the same
    // tiling, the pre-split kernels exactly. (Before this, TP_Q4K_SPLIT=0 only reached v0 / v6: grouped v72 / v73 / v74
    // kept the split.) Improves Laguna Q4_K prompt throughput.
    static const int q4k_split = [] { const char* e = std::getenv("TP_Q4K_SPLIT"); return e ? std::atoi(e) : 2; }();
    const int ver = moe_version_for(p);
    const bool small = p.total_rows < 32 * (int64_t) p.n_experts;
#define TP_MOE5_ARGS (const char*) p.w, p.expert_bytes, p.k, (const char*) p.xq, p.bounds, p.ids, (int) p.total_rows, p.dst, p.ld_dst
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (q4k_split == 0 || (q4k_split == 1 && p.n_experts > 1)) {
            switch (ver) {
                case 7: k_moe5<type, MODE, false, false><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
                case 63: case 72: k_moe5<type, MODE, false, true, 0, 3><<<grid96, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
                case 73: k_moe5<type, MODE, false, true, 0, 3, false, false, true><<<grid96, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
                case 74: k_moe5<type, MODE, false, true, 0, 4, false, false, true><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
                case 0: case 6: case 61: case 62: case 5: k_moe5<type, MODE, false><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
                default: break;   // the k_moe_rp / k_moe versions below have no split path
            }
        }
    }
    if constexpr (MODE == 4) {   // Q6_K (rb-glm): the v6 body, 96-row passes (the per-row int scales cost VGPRs)
        k_moe5<type, MODE, true, true, 0, 3><<<grid96, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS);
        return;
    } else if constexpr (MODE == 3) {   // Q2_K: the v6 body only (96-row passes; 128 for TP_MOE_VER 0 / 6)
        if (ver == 0 || ver == 6) k_moe5<type, MODE, true><<<grid, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS);
        // (v73's balanced tiles do not help Q2_K: Q2_K keeps the v72 body; v74 = balanced 128-row passes)
        else if (ver == 74) k_moe5<type, MODE, true, true, 0, 4, false, false, true><<<grid, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS);
        else k_moe5<type, MODE, true, true, 0, 3><<<grid96, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS);
        return;
    } else {
    switch (ver) {
        case 0: case 6: k_moe5<type, MODE, true><<<grid, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 7: k_moe5<type, MODE, true, false><<<grid, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 61: k_moe5<type, MODE, true, true, 1><<<grid, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 62: k_moe5<type, MODE, true, true, 2><<<grid, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 63: k_moe5<type, MODE, true, true, 0, 3><<<grid96, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 72: k_moe5<type, MODE, true, true, 0, 3, TpSplit<type>::ok><<<grid96, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 73: k_moe5<type, MODE, true, true, 0, 3, TpSplit<type>::ok, false, true><<<grid96, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 74: k_moe5<type, MODE, true, true, 0, 4, TpSplit<type>::ok, false, true><<<grid, dim3(32, 8), lds6, s>>>(TP_MOE5_ARGS); return;
        case 5: k_moe5<type, MODE, false><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
        case 41: k_moe_rp<type, MODE, 1><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
        case 42: k_moe_rp<type, MODE, 2><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
        case 4: k_moe_rp<type, MODE><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
        case 1: k_moe<type, MODE><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS); return;
        default:   // unknown: grouped-expert rule (v4, v1 for the per-16 scale types at < 32 rows per expert)
            if (!(MODE == 1 && small)) k_moe_rp<type, MODE><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS);
            else k_moe<type, MODE><<<grid, dim3(32, 8), lds, s>>>(TP_MOE5_ARGS);
            return;
    }
    }
#undef TP_MOE5_ARGS
}

}  // namespace

/// the 32-value block types (IQ4_NL, Q8_0, Q5_0) also take k % 256 != 0 (Flash-Next's down projection, k 640) on
/// the v6 kernel: MMQ's own K steps (the last one reads the next row's blocks against the zero q8_1 padding).
static bool moe_ktail_type(int t) { return t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q5_0; }

/// Q2_K on tp_moe (mode 3, MMQ's arithmetic): TP_MOE_Q2K=0 keeps MMQ.  Grouped products only (one expert = the dense
/// tk_prefill_mmq route keeps MMQ until measured).
static bool moe_q2k_on() {
#if defined(STRATA_MMQ_KQUANTS)
    static const bool on = [] { const char* v = std::getenv("TP_MOE_Q2K"); return v == nullptr || v[0] != '0'; }();
    return on;
#else
    return false;
#endif
}
static int moe_mode(int t) { return t == GGML_TYPE_Q2_K ? (moe_q2k_on() ? 3 : -1) : i8x_mode(t); }

bool moe_supported(int t, int n, int k) {
    if (moe_mode(t) < 0 || n % 128 != 0 || k <= 0) return false;
    if (k % MMQ_ITER_K == 0) return true;
    static const bool tail = [] { const char* e = std::getenv("TP_MOE_KTAIL"); return e == nullptr || e[0] != '0'; }();
    return tail && moe_ktail_type(t) && k % 32 == 0;
}

void moe_set_version(int v) { g_moe_ver = v; }

/// one-expert (dense) Q6_K product on tp_moe mode 4 (MMQ's Q6_K tile + mma arithmetic, bitwise MMQ); false =
/// not covered (another type / shape, grouped products, no K-quant MMQ, TP_Q6K unset) - the caller keeps MMQ.
bool moe_gemm_q6k_dense(const MoeProduct& p, void* stream) {
#if defined(STRATA_MMQ_KQUANTS)
    // opt-in TP_Q6K=1: bitwise MMQ but measured slower (1536 -> 16384 at 512 rows 1.94 -> 2.11 ms: the int scale multiply
    // + convert per element costs more VALU than MMQ's tile saves)
    static const bool on = [] { const char* e = std::getenv("TP_Q6K"); return e != nullptr && e[0] == '1'; }();
    if (!on || p.type != GGML_TYPE_Q6_K || p.n_experts != 1 || p.w_up != nullptr || p.n % 128 != 0 || p.k <= 0 ||
        p.k % MMQ_ITER_K != 0 || p.max_rows <= 0)
        return false;
    launch<GGML_TYPE_Q6_K, 4>(p, (cudaStream_t) stream);
    return cudaGetLastError() == cudaSuccess;
#else
    (void) p; (void) stream;
    return false;
#endif
}

int moe_dot_mode(int t) { return moe_mode(t); }

bool moe_gu_supported(int t, int n, int k) {
    switch (t) {
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M: break;
        default: return false;
    }
    return n % 64 == 0 && k % MMQ_ITER_K == 0 && k > 0;
}

bool moe_gu_q8_supported(int gu_type, int down_type, int n, int k) {
    if (!moe_gu_supported(gu_type, n, k) || gu_type == GGML_TYPE_IQ1_M || moe_mode(gu_type) != 0) return false;
    if (down_type < 0 || down_type >= GGML_TYPE_COUNT) return false;
    return mmq_get_q8_1_ds_layout((ggml_type) down_type) == MMQ_Q8_1_DS_LAYOUT_D4 && n % 128 == 0;
}

int g_iq1m_tok = 0;   // talus-pf-bench: k_moe1m rows per block under test (0 = TP_MOE_IQ1M_TOK / default)
void moe_iq1m_set_tok(int t) { g_iq1m_tok = t; }

bool moe_iq1m_q8_1_supported(int t, int n, int k) {
    return t == GGML_TYPE_IQ1_M && n % iq1m::kRows == 0 && k == 4096;   // NI = 4 (MiMo's gate / up)
}

bool moe_gemm_iq1m_q8_1(const MoeProduct& p, void* stream) {
    // TP_MOE_IQ1M_X=0: off (the caller keeps the dp4a kernel)
    static const bool on = [] { const char* v = std::getenv("TP_MOE_IQ1M_X"); return v == nullptr || v[0] != '0'; }();
    // TP_MOE_IQ1M_GU=0: no fused gate/up (the caller runs gate, up and tk_silu_mul)
    static const bool gu_on = [] { const char* v = std::getenv("TP_MOE_IQ1M_GU"); return v == nullptr || v[0] != '0'; }();
    if (!on || (p.w_up != nullptr && (!gu_on || p.swiglu_limit != 0.0f)) || !moe_iq1m_q8_1_supported(p.type, p.n, p.k) ||
        p.max_rows <= 0 || p.n_experts <= 0 || !tp_gfx11_wmma())
        return false;
    // TP_MOE_IQ1M_TOK=128: 128 routed rows per block (16 waves) instead of 64
    static const int tok = [] { const char* v = std::getenv("TP_MOE_IQ1M_TOK"); return v && std::atoi(v) == 128 ? 128 : 64; }();
    const int t = g_iq1m_tok > 0 ? g_iq1m_tok : tok;
    const unsigned gx = (unsigned) ((p.max_rows + t - 1) / t);
    const cudaStream_t s = (cudaStream_t) stream;
#define TP_1M_ARGS (const char*) p.w, p.expert_bytes, p.k, p.n, (const char*) p.xq, p.bounds, p.ids, (int) p.total_rows, p.dst, p.ld_dst, (const char*) p.w_up
    const dim3 gg(gx, (unsigned) (p.n / (iq1m::kRows / 2)), (unsigned) p.n_experts), gs(gx, (unsigned) (p.n / iq1m::kRows), (unsigned) p.n_experts);
    // TP_MOE_IQ1M_LDSGRID=1: the grid table in LDS (bitwise: the same table words)
    static const bool ldsg = [] { const char* v = std::getenv("TP_MOE_IQ1M_LDSGRID"); return v != nullptr && v[0] == '1'; }();
    // phase 2, TP_MOE_IQ1M_ONE (default 1; 0 = the 2-tile kernel): one C tile per wave, 16 waves per 64 routed
    // rows, the leaf's activations staged in LDS (169 VGPRs vs 249; bench GU 2048 tok: 46.6 -> 42.8 ms; bitwise)
    static const bool one = [] { const char* v = std::getenv("TP_MOE_IQ1M_ONE"); return v == nullptr || v[0] != '0'; }();
    if (one && t == 64 && !ldsg) {
        if (p.w_up != nullptr) k_moe1m<4, true, 64, false, true><<<gg, dim3(32, 16), 0, s>>>(TP_1M_ARGS);
        else k_moe1m<4, false, 64, false, true><<<gs, dim3(32, 16), 0, s>>>(TP_1M_ARGS);
    } else if (t == 128) {
        if (p.w_up != nullptr) k_moe1m<4, true, 128><<<gg, dim3(32, 16), 0, s>>>(TP_1M_ARGS);
        else k_moe1m<4, false, 128><<<gs, dim3(32, 16), 0, s>>>(TP_1M_ARGS);
    } else if (ldsg) {
        if (p.w_up != nullptr) k_moe1m<4, true, 64, true><<<gg, dim3(32, 8), 0, s>>>(TP_1M_ARGS);
        else k_moe1m<4, false, 64, true><<<gs, dim3(32, 8), 0, s>>>(TP_1M_ARGS);
    } else {
        if (p.w_up != nullptr) k_moe1m<4, true><<<gg, dim3(32, 8), 0, s>>>(TP_1M_ARGS);
        else k_moe1m<4><<<gs, dim3(32, 8), 0, s>>>(TP_1M_ARGS);
    }
#undef TP_1M_ARGS
    return cudaGetLastError() == cudaSuccess;
}

bool moe_gemm(const MoeProduct& p, void* stream) {
    // STRATA_PF_W16_MOE=1: the fp16 WMMA kernels where covered (moe_w16.hpp)
    if (p.q8_out == nullptr && moe_gemm_w16(p, stream)) return true;   // (the q8-out GU form is int8 tp_moe only)
    const cudaStream_t s = (cudaStream_t) stream;
    if (p.w_up != nullptr) {
        if (!moe_gu_supported(p.type, p.n, p.k) || p.max_rows <= 0 || p.n_experts <= 0) return false;
        bool ok = false;
        switch (p.type) {
            case GGML_TYPE_IQ3_S: ok = launch_gu<GGML_TYPE_IQ3_S, 0>(p, s); break;
            case GGML_TYPE_IQ3_XXS: ok = launch_gu<GGML_TYPE_IQ3_XXS, 0>(p, s); break;
            case GGML_TYPE_IQ2_XXS: ok = launch_gu<GGML_TYPE_IQ2_XXS, 0>(p, s); break;
            case GGML_TYPE_IQ2_XS: ok = launch_gu<GGML_TYPE_IQ2_XS, 1>(p, s); break;
            case GGML_TYPE_IQ2_S: ok = launch_gu<GGML_TYPE_IQ2_S, 1>(p, s); break;
            case GGML_TYPE_IQ1_M: ok = launch_gu<GGML_TYPE_IQ1_M, 1>(p, s); break;
            default: break;
        }
        return ok && cudaGetLastError() == cudaSuccess;
    }
    if (!moe_supported(p.type, p.n, p.k) || p.max_rows <= 0 || p.n_experts <= 0) return false;
    switch (p.type) {
        case GGML_TYPE_IQ4_XS: launch<GGML_TYPE_IQ4_XS, 0>(p, s); break;
        case GGML_TYPE_IQ4_NL: launch<GGML_TYPE_IQ4_NL, 0>(p, s); break;
        case GGML_TYPE_IQ3_S: launch<GGML_TYPE_IQ3_S, 0>(p, s); break;
        case GGML_TYPE_IQ3_XXS: launch<GGML_TYPE_IQ3_XXS, 0>(p, s); break;
        case GGML_TYPE_IQ2_XXS: launch<GGML_TYPE_IQ2_XXS, 0>(p, s); break;
        case GGML_TYPE_Q8_0: launch<GGML_TYPE_Q8_0, 0>(p, s); break;
        case GGML_TYPE_Q5_0: launch<GGML_TYPE_Q5_0, 0>(p, s); break;
        case GGML_TYPE_IQ2_XS: launch<GGML_TYPE_IQ2_XS, 1>(p, s); break;
        case GGML_TYPE_IQ2_S: launch<GGML_TYPE_IQ2_S, 1>(p, s); break;
        case GGML_TYPE_IQ1_M: launch<GGML_TYPE_IQ1_M, 1>(p, s); break;
#if defined(STRATA_MMQ_KQUANTS)
        case GGML_TYPE_Q2_K:
            if (p.n_experts <= 1) return false;
            launch<GGML_TYPE_Q2_K, 3>(p, s);
            break;
        case GGML_TYPE_Q3_K: launch<GGML_TYPE_Q3_K, 1>(p, s); break;
        case GGML_TYPE_Q4_K: launch<GGML_TYPE_Q4_K, 2>(p, s); break;
        case GGML_TYPE_Q5_K: launch<GGML_TYPE_Q5_K, 2>(p, s); break;
#endif
        default: return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace talus::prefill

extern "C" int talus_pf_moe_gemm(const void* w, int ggml_type, size_t expert_bytes, int n, int k, int n_experts,
                                 const void* xq, const int32_t* bounds, const int32_t* ids, int64_t total_rows,
                                 int64_t max_rows, float* dst, int64_t ld_dst, void* stream) noexcept {
    talus::prefill::MoeProduct p;
    p.w = w; p.type = ggml_type; p.expert_bytes = expert_bytes; p.n = n; p.k = k; p.n_experts = n_experts;
    p.xq = xq; p.bounds = bounds; p.ids = ids; p.total_rows = total_rows; p.max_rows = max_rows;
    p.dst = dst; p.ld_dst = ld_dst;
    return talus::prefill::moe_gemm(p, stream) ? 0 : -1;
}
