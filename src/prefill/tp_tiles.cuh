// src/prefill/tp_tiles.cuh - the weight-tile loaders of the Talus prefill int8 kernels (tp_i8x expansion, tp_moe).
//
// For every type MMQ covers this is MMQ's own tile loader (ggml_cuda_mmq_get_load_tiles, J = 128: 256 threads, 128 rows,
// one 256-value K step), so the int8 values and scales are bit for bit the ones mul_mat_q multiplies.  IQ1_M has no MMQ
// path; its loader here writes the same 128-row tile in the per-16-scale layout (MMQ's Q3_K SRAM layout, the "q8_0_16"
// dot, I8Mat mode 1), exactly representing the dequantized weights:
//   w = dl (g + delta), g in {-1, 0, 1}, delta = +-1/8, dl = d (2 s + 1)   ->   int8 q = 8 g + 8 delta in {-9,-7,-1,1,7,9},
//   scale dl / 8 per 16 values (exact: a power-of-two division).
// The activations are MMQ's q8_1 rows of the D4 layout (as for IQ2_XS / IQ2_S).
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#pragma once

#include "common.cuh"
#include "mmq.cuh"

namespace talus::prefill {

/// The IQ1_M tile loader (signature of ggml_cuda_mmq_load_tiles_t; launch block (32, 8)).
static __device__ __forceinline__ void tp_load_tiles_iq1_m(const char* __restrict__ x, int* __restrict__ x_tile,
                                                           const int kbx0, const int i_max, const int stride) {
    constexpr int ss = ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K);
    int* x_qs = x_tile;
    float* x_df = (float*) (x_qs + 2 * MMQ_TILE_NE_K);
    constexpr int threads_per_row = 8;   // one 32-value sub-block each
    constexpr int nrows = 32 / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;
#pragma unroll
    for (int i0 = 0; i0 < 128; i0 += 8 * nrows) {
        const int i = min(i0 + (int) threadIdx.y * nrows + (int) threadIdx.x / threads_per_row, i_max);
        const block_iq1_m* bxi = (const block_iq1_m*) x + kbx0 + i * stride;
        const int qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const uint16_t* sc = (const uint16_t*) bxi->scales;
        const int qh = bxi->qh[2 * kqsx] | (bxi->qh[2 * kqsx + 1] << 8);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int qhl = qh >> (4 * l);
            const int grid = iq1s_grid_gpu[qs[l] | ((qhl & 0x07) << 8)];
            const int off = (qhl & 0x08) ? 0x09090909 : 0x07070707;   // 8 (nibble - 1) + 8 delta = 8 nibble - 7 / - 9
            x_qs[i * ss + 8 * kqsx + 2 * l + 0] = __vsub4(((grid >> 0) & 0x0F0F0F0F) << 3, off);
            x_qs[i * ss + 8 * kqsx + 2 * l + 1] = __vsub4(((grid >> 4) & 0x0F0F0F0F) << 3, off);
        }
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d = __half2float(scale.f16);
        const int tmp = sc[kqsx / 2] >> (6 * (kqsx % 2));
        x_df[i * ss + 2 * kqsx + 0] = d * (float) (2 * ((tmp >> 0) & 0x07) + 1) * 0.125f;
        x_df[i * ss + 2 * kqsx + 1] = d * (float) (2 * ((tmp >> 3) & 0x07) + 1) * 0.125f;
    }
}

/// LDS lookup tables of the fast loaders (after the tile in shared memory): sign entries {m0, m1, l0, l1} (byte masks
/// 0x00 / 0xFF of sign bits 0-3 and 4-7 and their low bits), then the type's grid.  The signs are applied as
/// (g ^ m) + l: per byte -g or g with no carry between bytes (every grid byte is >= 1), the bytes of MMQ's
/// __vsub4(g ^ m, m).
template <ggml_type type> struct TpGrid { static constexpr int words = 0; static constexpr bool ksigns = false; };
template <> struct TpGrid<GGML_TYPE_IQ2_S> { static constexpr int words = 2048; static constexpr bool ksigns = false; };
template <> struct TpGrid<GGML_TYPE_IQ3_S> { static constexpr int words = 512; static constexpr bool ksigns = false; };
template <> struct TpGrid<GGML_TYPE_IQ2_XXS> { static constexpr int words = 512; static constexpr bool ksigns = true; };
template <> struct TpGrid<GGML_TYPE_IQ2_XS> { static constexpr int words = 1024; static constexpr bool ksigns = true; };
template <> struct TpGrid<GGML_TYPE_IQ3_XXS> { static constexpr int words = 256; static constexpr bool ksigns = true; };
/// sign entries {m0, m1} (2 words); the low bits l = m & 0x01010101 are recomputed at use
constexpr int kTpSignWords = 2 * 256;
template <ggml_type type> static constexpr int tp_table_words() { return TpGrid<type>::words ? kTpSignWords + TpGrid<type>::words : 0; }
/// the row stride (ints) of the split decoders' tile: 64 ints of values + 4 of padding (rows 4 banks apart: a 16-lane
/// 16-byte read of 16 rows touches all 64 banks once); their scales live in the compact array, not in the tile
constexpr int kTpSplitSS = 68;

template <ggml_type type> static __device__ __forceinline__ const uint32_t* tp_grid_src() {
    if constexpr (type == GGML_TYPE_IQ2_S) return (const uint32_t*) iq2s_grid;
    else if constexpr (type == GGML_TYPE_IQ3_S) return (const uint32_t*) iq3s_grid;
    else if constexpr (type == GGML_TYPE_IQ2_XXS) return (const uint32_t*) iq2xxs_grid;
    else if constexpr (type == GGML_TYPE_IQ2_XS) return (const uint32_t*) iq2xs_grid;
    else return (const uint32_t*) iq3xxs_grid;
}

/// Fill the tables (all threads of the block; a barrier must follow before the first load).
template <ggml_type type> static __device__ __forceinline__ void tp_tables_init(int* tab) {
    if constexpr (TpGrid<type>::words > 0) {
        const int tid = threadIdx.y * blockDim.x + threadIdx.x, nt = blockDim.x * blockDim.y;
        for (int e = tid; e < 256; e += nt) {
            int s8 = e;
            if constexpr (TpGrid<type>::ksigns) s8 = (e & 0x7f) | ((__popc(e & 0x7f) & 1) << 7);   // MMQ's unpack_ksigns
            uint32_t m0 = 0, m1 = 0;
            for (int b = 0; b < 4; ++b) {
                m0 |= ((s8 >> b) & 1) ? 0xFFu << (8 * b) : 0u;
                m1 |= ((s8 >> (b + 4)) & 1) ? 0xFFu << (8 * b) : 0u;
            }
            ((int2*) tab)[e] = make_int2((int) m0, (int) m1);
        }
        const uint32_t* g = tp_grid_src<type>();
        for (int e = tid; e < TpGrid<type>::words; e += nt) tab[kTpSignWords + e] = (int) g[e];
    }
}

/// Fast loaders: MMQ's thread mapping, tile layout and scale expressions; the grid and the sign masks from the LDS
/// tables (output bitwise MMQ's: talus-pf-bench moe).
template <ggml_type type> static __device__ __forceinline__ void tp_load_tiles_fast(const char* __restrict__ x, int* __restrict__ x_tile,
                                                                                   const int kbx0, const int i_max, const int stride,
                                                                                   const int* __restrict__ tab) {
    constexpr int ss = ggml_cuda_mmq_get_sram_stride(type, 128, false);
    int* x_qs = x_tile;
    float* x_df = (float*) (x_qs + 2 * MMQ_TILE_NE_K);
    const int2* tsg = (const int2*) tab;
    const int* tg = tab + kTpSignWords;
    constexpr int threads_per_row = 8, nrows = 4;
    const int kqsx = threadIdx.x % threads_per_row;
    auto apply = [&](int2 g, int s8, int* out) {
        const int2 t = tsg[s8];
        out[0] = (g.x ^ t.x) + (t.x & 0x01010101);
        out[1] = (g.y ^ t.y) + (t.y & 0x01010101);
    };
#pragma unroll
    for (int i0 = 0; i0 < 128; i0 += 8 * nrows) {
        const int i = min(i0 + (int) threadIdx.y * nrows + (int) threadIdx.x / threads_per_row, i_max);
        int* q = x_qs + i * ss + 8 * kqsx;
        if constexpr (type == GGML_TYPE_IQ2_S) {
            const block_iq2_s* bxi = (const block_iq2_s*) x + kbx0 + i * stride;
            const int qs_packed = get_int_b2(bxi->qs, kqsx);
            const uint8_t* qs = (const uint8_t*) &qs_packed;
            const int qh = bxi->qh[kqsx];
            const int signs_packed_32 = get_int_b2(bxi->qs, QK_K / 32 + kqsx);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = qs[l] | ((qh << (8 - 2 * l)) & 0x300);
                apply(*(const int2*) &tg[2 * idx], (signs_packed_32 >> (8 * l)) & 0xff, q + 2 * l);
            }
            const int ls = bxi->scales[kqsx];
            const float d = bxi->d;
            x_df[i * ss + 2 * kqsx + 0] = ((ls & 0x0F) * d + d / 2) / 4;
            x_df[i * ss + 2 * kqsx + 1] = ((ls >> 4) * d + d / 2) / 4;
        } else if constexpr (type == GGML_TYPE_IQ2_XS) {
            const block_iq2_xs* bxi = (const block_iq2_xs*) x + kbx0 + i * stride;
            const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2 * kqsx + 0), get_int_b2(bxi->qs, 2 * kqsx + 1));
            const uint16_t* q2 = (const uint16_t*) &q2_packed;
#pragma unroll
            for (int l = 0; l < 4; ++l) apply(*(const int2*) &tg[2 * (q2[l] & 0x1FF)], q2[l] >> 9, q + 2 * l);
            const int ls = bxi->scales[kqsx];
            const float d = bxi->d;
            x_df[i * ss + 2 * kqsx + 0] = ((ls & 0x0F) * d + d / 2) / 4;
            x_df[i * ss + 2 * kqsx + 1] = ((ls >> 4) * d + d / 2) / 4;
        } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
            const block_iq2_xxs* bxi = (const block_iq2_xxs*) x + kbx0 + i * stride;
            const int q2 = get_int_b2(bxi->qs, 2 * kqsx + 0);
            const uint8_t* aux8 = (const uint8_t*) &q2;
            const uint32_t aux32 = get_int_b2(bxi->qs, 2 * kqsx + 1);
#pragma unroll
            for (int l = 0; l < 4; ++l) apply(*(const int2*) &tg[2 * aux8[l]], (aux32 >> (7 * l)) & 0x7f, q + 2 * l);
            const int ls = aux32 >> 27 | 1;
            const float d = bxi->d;
            x_df[i * ss + kqsx] = d * ls / 8;
        } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
            const block_iq3_xxs* bxi = (const block_iq3_xxs*) x + kbx0 + i * stride;
            const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2 * kqsx + 0), get_int_b2(bxi->qs, 2 * kqsx + 1));
            const uint8_t* q3 = (const uint8_t*) &q3_packed;
            const uint32_t aux32 = get_int_b2(bxi->qs, QK_K / 16 + kqsx);
#pragma unroll
            for (int l = 0; l < 4; ++l) apply(make_int2(tg[q3[2 * l + 0]], tg[q3[2 * l + 1]]), (aux32 >> (7 * l)) & 0x7f, q + 2 * l);
            const int ls = aux32 >> 28;
            const float d = bxi->d;
            x_df[i * ss + kqsx] = (ls * d + d / 2) / 2;
        } else {   // IQ3_S
            const block_iq3_s* bxi = (const block_iq3_s*) x + kbx0 + i * stride;
            const int2 qs_packed = make_int2(get_int_b2(bxi->qs, 2 * kqsx + 0), get_int_b2(bxi->qs, 2 * kqsx + 1));
            const uint8_t* qs = (const uint8_t*) &qs_packed;
            const int qh = bxi->qh[kqsx];
            const int signs_packed_32 = get_int_b2(bxi->signs, kqsx);
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int2 g = make_int2(tg[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 0x100)],
                                         tg[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100)]);
                apply(g, (signs_packed_32 >> (8 * l)) & 0xff, q + 2 * l);
            }
            const int ls = 1 + 2 * ((bxi->scales[kqsx / 2] >> (((2 * kqsx) << 1) & 0x04)) & 0x0F);
            const float d = bxi->d;
            x_df[i * ss + kqsx] = ls * d;
        }
    }
}

/// Split loaders : fetch() issues the global loads of one 128-row x 256-value tile into registers (a thread's
/// 4 rows of one 32-value sub-block, MMQ's thread mapping), decode() writes the tile (same values, layout and scale
/// expressions as the loaders above) - so a kernel can fetch the next K step's weights before computing this one.
/// Types: IQ2_S, IQ3_S, IQ2_XXS, IQ2_XS, IQ3_XXS (LDS tables) and IQ1_M.  fetch(.., x2): rows 64-127 are rows 0-63
/// of the matrix at x2 (fused gate/up); decode<true>: the sign masks computed (tp_sign_alu) instead of looked up.
/// The split decoders write the scales compactly (not into the MMQ tile): cs[TP_CS(row, block, SC, c)], per 16-row
/// tile and 32-value block SC x 16 floats in the order of the i8x expansion (rows 0, 2, .., 14, then 1, 3, .., 15), so
/// the WMMA C-tile lane (rows 2 i + hi) reads its 8 scales with two 16-byte LDS loads.
#define TP_CS(i, j, SC, c) (((((i) >> 4) * 8 + (j)) * (SC) + (c)) * 16 + ((i) & 1) * 8 + (((i) & 15) >> 1))
/// the sign masks {m0, m1} of sign byte s computed (no table): KS = a ksign type's 7-bit index (MMQ's
/// unpack_ksigns parity bit 7).  The same values as the table entries.
template <bool KS> static __device__ __forceinline__ int2 tp_sign_alu(uint32_t s) {
    if constexpr (KS) s = (s & 0x7f) | ((__popc(s & 0x7f) & 1) << 7);
    const uint32_t t0 = ((s & 0xF) * 0x204081u) & 0x01010101u, t1 = (((s >> 4) & 0xF) * 0x204081u) & 0x01010101u;
    return make_int2((int) ((t0 << 8) - t0), (int) ((t1 << 8) - t1));
}

template <ggml_type type> struct TpSplit { static constexpr bool ok = false; };

template <> struct TpSplit<GGML_TYPE_IQ2_S> {
    static constexpr ggml_type type = GGML_TYPE_IQ2_S;
    static constexpr bool ok = true;
    int qs[4], sg[4], qh[4], ls[4];
    float d[4];
    __device__ __forceinline__ void fetch(const char* x, int kbx0, int i_max, int stride, const char* x2 = nullptr) {
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            const block_iq2_s* bxi = x2 ? (const block_iq2_s*) (i >= 64 ? x2 : x) + kbx0 + (i & 63) * stride : (const block_iq2_s*) x + kbx0 + i * stride;
            qs[r] = get_int_b2(bxi->qs, kqsx);
            sg[r] = get_int_b2(bxi->qs, QK_K / 32 + kqsx);
            qh[r] = bxi->qh[kqsx];
            ls[r] = bxi->scales[kqsx];
            d[r] = bxi->d;
        }
    }
    template <bool AS = false>
    __device__ __forceinline__ void decode(int* x_tile, const int* tab, int i_max, float* cs) const {
        constexpr int ss = kTpSplitSS;
        const int2* tsg = (const int2*) tab;
        const int* tg = tab + kTpSignWords;
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            int* q = x_tile + i * ss + 8 * kqsx;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int idx = ((qs[r] >> (8 * l)) & 0xff) | ((qh[r] << (8 - 2 * l)) & 0x300);
                const int2 g = *(const int2*) &tg[2 * idx];
                const int2 t = AS ? tp_sign_alu<false>((sg[r] >> (8 * l)) & 0xff) : tsg[(sg[r] >> (8 * l)) & 0xff];
                q[2 * l + 0] = (g.x ^ t.x) + (t.x & 0x01010101);
                q[2 * l + 1] = (g.y ^ t.y) + (t.y & 0x01010101);
            }
            cs[TP_CS(i, kqsx, 2, 0)] = ((ls[r] & 0x0F) * d[r] + d[r] / 2) / 4;
            cs[TP_CS(i, kqsx, 2, 1)] = ((ls[r] >> 4) * d[r] + d[r] / 2) / 4;
        }
    }
};

template <> struct TpSplit<GGML_TYPE_IQ3_S> {
    static constexpr ggml_type type = GGML_TYPE_IQ3_S;
    static constexpr bool ok = true;
    int qa[4], qb[4], sg[4], qh[4], ls[4];
    float d[4];
    __device__ __forceinline__ void fetch(const char* x, int kbx0, int i_max, int stride, const char* x2 = nullptr) {
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            const block_iq3_s* bxi = x2 ? (const block_iq3_s*) (i >= 64 ? x2 : x) + kbx0 + (i & 63) * stride : (const block_iq3_s*) x + kbx0 + i * stride;
            qa[r] = get_int_b2(bxi->qs, 2 * kqsx + 0);
            qb[r] = get_int_b2(bxi->qs, 2 * kqsx + 1);
            qh[r] = bxi->qh[kqsx];
            sg[r] = get_int_b2(bxi->signs, kqsx);
            ls[r] = bxi->scales[kqsx / 2];
            d[r] = bxi->d;
        }
    }
    template <bool AS = false>
    __device__ __forceinline__ void decode(int* x_tile, const int* tab, int i_max, float* cs) const {
        constexpr int ss = kTpSplitSS;
        const int2* tsg = (const int2*) tab;
        const int* tg = tab + kTpSignWords;
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            int* q = x_tile + i * ss + 8 * kqsx;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int w = l < 2 ? qa[r] : qb[r], sh = 16 * (l & 1);
                const int2 g = make_int2(tg[((w >> sh) & 0xff) | ((qh[r] << (8 - 2 * l)) & 0x100)],
                                         tg[((w >> (sh + 8)) & 0xff) | ((qh[r] << (7 - 2 * l)) & 0x100)]);
                const int2 t = AS ? tp_sign_alu<false>((sg[r] >> (8 * l)) & 0xff) : tsg[(sg[r] >> (8 * l)) & 0xff];
                q[2 * l + 0] = (g.x ^ t.x) + (t.x & 0x01010101);
                q[2 * l + 1] = (g.y ^ t.y) + (t.y & 0x01010101);
            }
            const int lsv = 1 + 2 * ((ls[r] >> (((2 * kqsx) << 1) & 0x04)) & 0x0F);
            cs[TP_CS(i, kqsx, 1, 0)] = lsv * d[r];
        }
    }
};

// IQ2_XXS / IQ3_XXS / IQ2_XS: ksign types
template <> struct TpSplit<GGML_TYPE_IQ2_XXS> {
    static constexpr ggml_type type = GGML_TYPE_IQ2_XXS;
    static constexpr bool ok = true;
    int q2[4], aux[4];
    float d[4];
    __device__ __forceinline__ void fetch(const char* x, int kbx0, int i_max, int stride, const char* x2 = nullptr) {
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            const block_iq2_xxs* bxi = x2 ? (const block_iq2_xxs*) (i >= 64 ? x2 : x) + kbx0 + (i & 63) * stride : (const block_iq2_xxs*) x + kbx0 + i * stride;
            q2[r] = get_int_b2(bxi->qs, 2 * kqsx + 0);
            aux[r] = get_int_b2(bxi->qs, 2 * kqsx + 1);
            d[r] = bxi->d;
        }
    }
    template <bool AS = false>
    __device__ __forceinline__ void decode(int* x_tile, const int* tab, int i_max, float* cs) const {
        constexpr int ss = kTpSplitSS;
        const int2* tsg = (const int2*) tab;
        const int* tg = tab + kTpSignWords;
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            int* q = x_tile + i * ss + 8 * kqsx;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int2 g = *(const int2*) &tg[2 * ((q2[r] >> (8 * l)) & 0xff)];
                const int2 t = AS ? tp_sign_alu<true>(((uint32_t) aux[r] >> (7 * l)) & 0x7f) : tsg[((uint32_t) aux[r] >> (7 * l)) & 0x7f];
                q[2 * l + 0] = (g.x ^ t.x) + (t.x & 0x01010101);
                q[2 * l + 1] = (g.y ^ t.y) + (t.y & 0x01010101);
            }
            const int ls = (int) ((uint32_t) aux[r] >> 27) | 1;
            cs[TP_CS(i, kqsx, 1, 0)] = d[r] * ls / 8;
        }
    }
};

template <> struct TpSplit<GGML_TYPE_IQ3_XXS> {
    static constexpr ggml_type type = GGML_TYPE_IQ3_XXS;
    static constexpr bool ok = true;
    int qa[4], qb[4], aux[4];
    float d[4];
    __device__ __forceinline__ void fetch(const char* x, int kbx0, int i_max, int stride, const char* x2 = nullptr) {
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            const block_iq3_xxs* bxi = x2 ? (const block_iq3_xxs*) (i >= 64 ? x2 : x) + kbx0 + (i & 63) * stride : (const block_iq3_xxs*) x + kbx0 + i * stride;
            qa[r] = get_int_b2(bxi->qs, 2 * kqsx + 0);
            qb[r] = get_int_b2(bxi->qs, 2 * kqsx + 1);
            aux[r] = get_int_b2(bxi->qs, QK_K / 16 + kqsx);
            d[r] = bxi->d;
        }
    }
    template <bool AS = false>
    __device__ __forceinline__ void decode(int* x_tile, const int* tab, int i_max, float* cs) const {
        constexpr int ss = kTpSplitSS;
        const int2* tsg = (const int2*) tab;
        const int* tg = tab + kTpSignWords;
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            int* q = x_tile + i * ss + 8 * kqsx;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int w = l < 2 ? qa[r] : qb[r], sh = 16 * (l & 1);
                const int2 g = make_int2(tg[(w >> sh) & 0xff], tg[(w >> (sh + 8)) & 0xff]);
                const int2 t = AS ? tp_sign_alu<true>(((uint32_t) aux[r] >> (7 * l)) & 0x7f) : tsg[((uint32_t) aux[r] >> (7 * l)) & 0x7f];
                q[2 * l + 0] = (g.x ^ t.x) + (t.x & 0x01010101);
                q[2 * l + 1] = (g.y ^ t.y) + (t.y & 0x01010101);
            }
            const int ls = (int) ((uint32_t) aux[r] >> 28);
            cs[TP_CS(i, kqsx, 1, 0)] = (ls * d[r] + d[r] / 2) / 2;
        }
    }
};

template <> struct TpSplit<GGML_TYPE_IQ2_XS> {
    static constexpr ggml_type type = GGML_TYPE_IQ2_XS;
    static constexpr bool ok = true;
    int qa[4], qb[4], ls[4];
    float d[4];
    __device__ __forceinline__ void fetch(const char* x, int kbx0, int i_max, int stride, const char* x2 = nullptr) {
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            const block_iq2_xs* bxi = x2 ? (const block_iq2_xs*) (i >= 64 ? x2 : x) + kbx0 + (i & 63) * stride : (const block_iq2_xs*) x + kbx0 + i * stride;
            qa[r] = get_int_b2(bxi->qs, 2 * kqsx + 0);
            qb[r] = get_int_b2(bxi->qs, 2 * kqsx + 1);
            ls[r] = bxi->scales[kqsx];
            d[r] = bxi->d;
        }
    }
    template <bool AS = false>
    __device__ __forceinline__ void decode(int* x_tile, const int* tab, int i_max, float* cs) const {
        constexpr int ss = kTpSplitSS;
        const int2* tsg = (const int2*) tab;
        const int* tg = tab + kTpSignWords;
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            int* q = x_tile + i * ss + 8 * kqsx;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int v = ((l < 2 ? qa[r] : qb[r]) >> (16 * (l & 1))) & 0xffff;
                const int2 g = *(const int2*) &tg[2 * (v & 0x1FF)];
                const int2 t = AS ? tp_sign_alu<true>(v >> 9) : tsg[v >> 9];
                q[2 * l + 0] = (g.x ^ t.x) + (t.x & 0x01010101);
                q[2 * l + 1] = (g.y ^ t.y) + (t.y & 0x01010101);
            }
            cs[TP_CS(i, kqsx, 2, 0)] = ((ls[r] & 0x0F) * d[r] + d[r] / 2) / 4;
            cs[TP_CS(i, kqsx, 2, 1)] = ((ls[r] >> 4) * d[r] + d[r] / 2) / 4;
        }
    }
};

template <> struct TpSplit<GGML_TYPE_IQ1_M> {
    static constexpr ggml_type type = GGML_TYPE_IQ1_M;
    static constexpr bool ok = true;
    int qs[4], qh[4], sc01[4], sc23[4];
    __device__ __forceinline__ void fetch(const char* x, int kbx0, int i_max, int stride, const char* x2 = nullptr) {
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            const block_iq1_m* bxi = x2 ? (const block_iq1_m*) (i >= 64 ? x2 : x) + kbx0 + (i & 63) * stride : (const block_iq1_m*) x + kbx0 + i * stride;
            qs[r] = get_int_b2(bxi->qs, kqsx);
            qh[r] = ((const uint16_t*) bxi->qh)[kqsx];
            sc01[r] = get_int_b2(bxi->scales, 0);
            sc23[r] = get_int_b2(bxi->scales, 1);
        }
    }
    template <bool AS = false>
    __device__ __forceinline__ void decode(int* x_tile, const int* tab, int i_max, float* cs) const {
        (void) tab;
        constexpr int ss = kTpSplitSS;
        const int kqsx = threadIdx.x % 8;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = min(r * 32 + (int) threadIdx.y * 4 + (int) threadIdx.x / 8, i_max);
            int* q = x_tile + i * ss + 8 * kqsx;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int qhl = qh[r] >> (4 * l);
                const int grid = iq1s_grid_gpu[((qs[r] >> (8 * l)) & 0xff) | ((qhl & 0x07) << 8)];
                const int off = (qhl & 0x08) ? 0x09090909 : 0x07070707;
                q[2 * l + 0] = __vsub4(((grid >> 0) & 0x0F0F0F0F) << 3, off);
                q[2 * l + 1] = __vsub4(((grid >> 4) & 0x0F0F0F0F) << 3, off);
            }
            const uint16_t sc[4] = {(uint16_t) sc01[r], (uint16_t) (sc01[r] >> 16), (uint16_t) sc23[r], (uint16_t) (sc23[r] >> 16)};
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
            const float d = __half2float(scale.f16);
            const int tmp = sc[kqsx / 2] >> (6 * (kqsx % 2));
            cs[TP_CS(i, kqsx, 2, 0)] = d * (float) (2 * ((tmp >> 0) & 0x07) + 1) * 0.125f;
            cs[TP_CS(i, kqsx, 2, 1)] = d * (float) (2 * ((tmp >> 3) & 0x07) + 1) * 0.125f;
        }
    }
};

/// Q4_K split loader (the dense GLM / shared-expert projections). The next K step's raw blocks are fetched into
/// registers during this step's WMMAs and decoded into the tile afterwards, as the IQ split decoders do (MMQ's Q4_K
/// loader in k_moe5 is a synchronous load + barrier per K step). Bit for bit MMQ's tile (ggml_cuda_mmq_load_tiles_q4_K):
/// int g of a row's qs -> low nibbles at 16 (g / 8) + g % 8, high nibbles at + 8 (the 32-value block layout, row
/// stride kTpSplitSS), scales half2(d, -dmin) * half2(sc, m) in half precision (MMQ's expression; read as floats in
/// the compact-scale table, which is what the kernel's __half22float2 of MMQ's x_dm gives).
/// Thread (x, y): 32-byte qs group g = x & 3 (blocks 2g, 2g + 1) of rows (x >> 2) + 8 y + 64 r, r = 0, 1.
template <> struct TpSplit<GGML_TYPE_Q4_K> {
    static constexpr ggml_type type = GGML_TYPE_Q4_K;
    static constexpr bool ok = true;
    int4 q[2][2];   // qs bytes 32 g .. 32 g + 31 of row r
    int4 hd[2];     // row r's dm (half2) + scales[12]
    static __device__ __forceinline__ int row(int r, int i_max) {
        return min((int) (threadIdx.x >> 2) + 8 * (int) threadIdx.y + 64 * r, i_max);
    }
    __device__ __forceinline__ void fetch(const char* x, int kbx0, int i_max, int stride, const char* x2 = nullptr) {
        (void) x2;
        const int g = threadIdx.x & 3;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const char* b = (const char*) ((const block_q4_K*) x + kbx0 + row(r, i_max) * stride);
            hd[r] = *(const int4*) b;
            q[r][0] = *(const int4*) (b + 16 + 32 * g);
            q[r][1] = *(const int4*) (b + 32 + 32 * g);
        }
    }
    template <bool AS = false>
    __device__ __forceinline__ void decode(int* x_tile, const int* tab, int i_max, float* cs) const {
        (void) tab;
        constexpr int ss = kTpSplitSS;
        const int g = threadIdx.x & 3;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int i = row(r, i_max);
            int4* qd = (int4*) (x_tile + i * ss + 16 * g);
            const int4 a = q[r][0], b = q[r][1];
            qd[0] = make_int4(a.x & 0x0F0F0F0F, a.y & 0x0F0F0F0F, a.z & 0x0F0F0F0F, a.w & 0x0F0F0F0F);
            qd[1] = make_int4(b.x & 0x0F0F0F0F, b.y & 0x0F0F0F0F, b.z & 0x0F0F0F0F, b.w & 0x0F0F0F0F);
            qd[2] = make_int4((a.x >> 4) & 0x0F0F0F0F, (a.y >> 4) & 0x0F0F0F0F, (a.z >> 4) & 0x0F0F0F0F, (a.w >> 4) & 0x0F0F0F0F);
            qd[3] = make_int4((b.x >> 4) & 0x0F0F0F0F, (b.y >> 4) & 0x0F0F0F0F, (b.z >> 4) & 0x0F0F0F0F, (b.w >> 4) & 0x0F0F0F0F);
            const int scales[3] = {hd[r].y, hd[r].z, hd[r].w};
            // blocks 2g, 2g + 1 are sub-blocks (2g) % 4, + 1 of MMQ's ksc = g / 2 group (constant-index calls, selected:
            // a runtime index would put scales[] in scratch)
            const bool k1 = (g >> 1) != 0;
            const int sc32 = k1 ? unpack_scales_q45_K(scales, 1) : unpack_scales_q45_K(scales, 0);
            const int m32 = k1 ? unpack_scales_q45_K(scales, 3) : unpack_scales_q45_K(scales, 2);
            const half2 dm = __builtin_bit_cast(half2, hd[r].x) * make_half2(1.0f, -1.0f);
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int j = 2 * g + u, l = j & 3;
                // MMQ's sc8[l] / m8[l] (bytes of sc32 / m32), as shifts (a runtime byte index would go through scratch)
                const uint8_t sc8l = (uint8_t) ((uint32_t) sc32 >> (8 * l)), m8l = (uint8_t) ((uint32_t) m32 >> (8 * l));
                const half2 v = dm * make_half2(sc8l, m8l);
                cs[TP_CS(i, j, 2, 0)] = __low2float(v);
                cs[TP_CS(i, j, 2, 1)] = __high2float(v);
            }
        }
    }
};

/// The 128-row tile of `type`: loader, SRAM row stride (ints), weight blocks per 256-value K step.
template <ggml_type type> struct TpTiles {
    static constexpr int kJ = 128;
    static constexpr int I = ggml_cuda_mmq_get_I(type, kJ, false);
    static constexpr int ss = ggml_cuda_mmq_get_sram_stride(type, kJ, false);
    static constexpr int blocks_per_iter = MMQ_ITER_K / ggml_cuda_type_traits<type>::qk;
    static __device__ __forceinline__ void load(const char* x, int* tile, int kbx0, int i_max, int stride,
                                                const int* tab = nullptr) {
        if constexpr (TpGrid<type>::words > 0) {
            if (tab != nullptr) {
                tp_load_tiles_fast<type>(x, tile, kbx0, i_max, stride, tab);
                return;
            }
        }
        constexpr ggml_cuda_mmq_load_tiles_t lt = ggml_cuda_mmq_get_load_tiles<type, kJ, false>();
        lt(x, tile, kbx0, i_max, stride);
    }
};
/// Q2_K: MMQ has no J = 128 configuration for Q2_K on RDNA 3.5 (the lookup falls back to a Q8_0-layout default whose
/// 76-int rows cannot hold Q2_K's 16 half2 scales); its J = 80 configuration (the one mul_mat_q<Q2_K> runs) has the same
/// 128-row, 256-thread tile in the Q2_K layout.
template <ggml_type type, int J> struct TpTilesJ {
    static constexpr int kJ = J;
    static constexpr int I = ggml_cuda_mmq_get_I(type, J, false);
    static constexpr int ss = ggml_cuda_mmq_get_sram_stride(type, J, false);
    static constexpr int blocks_per_iter = MMQ_ITER_K / ggml_cuda_type_traits<type>::qk;
    static __device__ __forceinline__ void load(const char* x, int* tile, int kbx0, int i_max, int stride, const int* tab = nullptr) {
        (void) tab;
        static_assert(ggml_cuda_mmq_get_sram_layout(type, J, false) == GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, "Q2_K layout");
        constexpr ggml_cuda_mmq_load_tiles_t lt = ggml_cuda_mmq_get_load_tiles<type, J, false>();
        lt(x, tile, kbx0, i_max, stride);
    }
};
template <> struct TpTiles<GGML_TYPE_Q2_K> : TpTilesJ<GGML_TYPE_Q2_K, 80> {};
template <> struct TpTiles<GGML_TYPE_IQ1_M> {
    static constexpr int I = 128;
    static constexpr int ss = ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K);
    static constexpr int blocks_per_iter = 1;
    static __device__ __forceinline__ void load(const char* x, int* tile, int kbx0, int i_max, int stride,
                                                const int* tab = nullptr) {
        (void) tab;
        tp_load_tiles_iq1_m(x, tile, kbx0, i_max, stride);
    }
};

}  // namespace talus::prefill
