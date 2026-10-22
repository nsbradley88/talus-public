// src/kernels/cuda/talus_iq_split.cuh - Talus: the i-quant weight formats MiMo-V2.6's experts use (IQ2_XXS, IQ2_XS,
// IQ3_XXS, IQ3_S, IQ1_M) and Q2_K as native_mmvq_batch traits.  The load / apply splits are copied unchanged from
// Strata's iq_kernels.cu (Split<TY>, "decode once, apply to every column": bitwise llama.cpp's vec_dot_*_q8_1), the
// Q2_K one is llama.cpp's vec_dot_q2_K_q8_1 (vecdotq.cuh, MIT) split the same way.  A copy rather than a shared
// header so that upstream edits to iq_kernels.cu merge without conflicts.
#pragma once

#include "strata/kernels/dp4a.hpp"

#include <cuda_fp16.h>

#ifndef GGML_COMMON_DECL_CUDA
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"
#endif

namespace talus_iq {

#define ggml_cuda_dp4a(a, b, c) STRATA_DP4A((a), (b), (c))

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}

// ---------------------------------------------------------------- decode once, apply to every column
// The kernels above call Fmt<TY>::dot once per (call, column): each column re-reads the weight words and redoes
// the grid lookups and sign unpacking.  Here each dot is split, as native_mmvq.cu's multi-column traits are, into
// `load` (everything that depends only on the weight: the signed grid words, the integer scales, the fp16 block
// scale as a float) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale
// step and the same float expression).  `apply(load(...))` does the dot's integer and float operations in the same
// order on the same values, so a column of the kernels below is BITWISE equal to the same column of mmvq_kernel /
// native_gu_kernel / native_down_kernel (iq_multi_parity checks it; STRATA_OLD_IQ_MMVQ=1 keeps the old kernels).
// Talus: sign application for the i-quant grids without CUDA's packed-byte intrinsics (emulated on AMD, ~6-7 ops
// each): `sbits` holds at most one set bit per byte (the byte's sign); every grid byte is in 1..127 (checked against
// ggml-common.h: no zero byte), so (g ^ m) + 1 per negated byte never carries into the next byte.  Same bits as
// __vsub4(g ^ __vcmpne4(sbits, 0), __vcmpne4(sbits, 0)).
__device__ __forceinline__ int talus_signed(uint32_t g, uint32_t sbits) {
    const uint32_t one = ((sbits + 0x7f7f7f7fu) & 0x80808080u) >> 7;   // 0x01 per negated byte
    return (int) ((g ^ (one * 0xffu)) + one);
}

// Talus: the same signed bytes from a 4-bit sign nibble (bit i negates byte i): the bit-spread
// multiply puts bit i at bit 8i (the four shifted copies of a 4-bit value do not overlap, so no carries), then
// (g ^ 0xff) + 1 per negated byte as above.  5-6 VALU per 4 bytes instead of the shift/or sign gathering plus the
// SWAR compare (IQ3_S) or the emulated __vcmpne4 / __vsub4 pair (IQ2_S, ~10 each).
__device__ __forceinline__ int talus_sign4(uint32_t g, uint32_t nib) {
    const uint32_t one = ((nib & 0x0fu) * 0x00204081u) & 0x01010101u;
#if defined(__HIP_PLATFORM_AMD__)
    // v_perm_b32 selector 12 gives byte 0x00, 13 gives 0xff: the byte mask without a quarter-rate v_mul_lo_u32
    const uint32_t m = __builtin_amdgcn_perm(0u, 0u, one | 0x0c0c0c0cu);
#else
    const uint32_t m = one * 0xffu;
#endif
    return (int) ((g ^ m) + one);
}
#ifndef TALUS_IQ_SIGN4
#define TALUS_IQ_SIGN4 TALUS_FAST_DOTS
#endif

template<int TY> struct Split;
// Formats with a Split below take the decode-once kernels; the others (Q4_K, Q5_K, Q5_1, Q8_0: UD-Q4_K_XL) the
// per-entry ones, which call Fmt<TY>::dot per column exactly as before #242 (the launchers test kSplit at compile
// time, so the multi kernels are never instantiated for a type without a Split).
template<int TY> inline constexpr bool kSplit = false;
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 22 || TY == 29);

template<> inline constexpr bool kSplit<16> = true;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xxs_grid;
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid_lut[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
#if TALUS_IQ_SIGN4
            r.g[k0 + 0] = talus_sign4(grid_pos.x, signs);
            r.g[k0 + 1] = talus_sign4(grid_pos.y, signs >> 4);
#else
            r.g[k0 + 0] = talus_signed(grid_pos.x, signs & 0x08040201);
            r.g[k0 + 1] = talus_signed(grid_pos.y, signs & 0x80402010);
#endif
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = __half2float(bq2->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = STRATA_IMUL24(sumi, r.ls) / 8;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (STRATA_IMUL24(sumi0, r.ls0) + STRATA_IMUL24(sumi1, r.ls1) + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t* q2 = (const uint16_t*) &q2_packed;
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xs_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid_lut[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
#if TALUS_IQ_SIGN4
            r.g[l0 + 0] = talus_sign4(grid_pos.x, signs);
            r.g[l0 + 1] = talus_sign4(grid_pos.y, signs >> 4);
#else
            r.g[l0 + 0] = talus_signed(grid_pos.x, signs & 0x08040201);
            r.g[l0 + 1] = talus_signed(grid_pos.y, signs & 0x80402010);
#endif
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<22> = true;
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint64_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint64_t*) s_grid;
        else grid_lut = iq2s_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (grid_lut + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
#if TALUS_IQ_SIGN4
            r.g[l0 + 0] = talus_sign4(grid_pos[0], uint32_t(signs_packed_32) >> (4 * l0));
            r.g[l0 + 1] = talus_sign4(grid_pos[1], uint32_t(signs_packed_32) >> (4 * l0 + 4));
#else
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
#endif
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<18> = true;
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
#if TALUS_IQ_SIGN4
            r.g[l0 + 0] = talus_sign4(grid_pos.x, signs);
            r.g[l0 + 1] = talus_sign4(grid_pos.y, signs >> 4);
#else
            r.g[l0 + 0] = talus_signed(grid_pos.x, signs & 0x08040201);
            r.g[l0 + 1] = talus_signed(grid_pos.y, signs & 0x80402010);
#endif
        }
        r.ls = aux32 >> 28;
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (STRATA_IMUL24(r.ls, sumi) + sumi / 2) / 2;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<21> = true;
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
#if TALUS_IQ_SIGN4
            r.g[l0 + 0] = talus_sign4(grid_pos.x, uint32_t(signs_packed_32) >> (4 * l0));
            r.g[l0 + 1] = talus_sign4(grid_pos.y, uint32_t(signs_packed_32) >> (4 * l0 + 4));
#else
            r.g[l0 + 0] = talus_signed(grid_pos.x, ((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21));
            r.g[l0 + 1] = talus_signed(grid_pos.y, ((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17));
#endif
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = STRATA_IMUL24(sumi, r.ls);
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const int qs_packed = get_int_b4(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const uint32_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = s_grid;
        else grid_lut = iq1s_grid_gpu;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = grid_lut[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        r.dw = __half2float(scale.f16);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
            sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * sumy;
        }
        const float d = r.dw * __low2float(bq8_1[iqs].ds);
        return d * ((sumi[0] + sumf[0]) * r.sc0 + (sumi[1] + sumf[1]) * r.sc1);
    }
};

template<> struct Split<10> {   // Q2_K: llama.cpp's vec_dot_q2_K_q8_1 / _impl_mmvq, load = the weight words
    struct W { int v; uint8_t sc[QR2_K]; half2 dm; int bq8_offset; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q2_K* b = (const block_q2_K*) vbq + kbx;
        W r;
        r.bq8_offset = QR2_K * (iqs / QI8_1);
        const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) r.sc[i] = b->scales[scale_offset + 2 * i];
        r.v = get_int_b4(b->qs, iqs);
        r.dm = b->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        float sumf_d = 0.0f, sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < QR2_K; ++i) {
            const int u = get_int_b4(bq8_1[r.bq8_offset + i].qs, iqs % QI8_1);
            const float d8 = __low2float(bq8_1[r.bq8_offset + i].ds);
            const int sc = r.sc[i];
            const int vi = (r.v >> (2 * i)) & 0x03030303;
            sumf_d += d8 * (ggml_cuda_dp4a(vi, u, 0) * (sc & 0xF));
            int m = sc >> 4;
            m |= m << 8;
            m |= m << 16;
            sumf_m += d8 * ggml_cuda_dp4a(m, u, 0);
        }
        const float2 dm2f = __half22float2(r.dm);
        return dm2f.x * sumf_d - dm2f.y * sumf_m;
    }
};

#undef ggml_cuda_dp4a

// values per block, dot calls per block, iqs stride between calls (Strata's Fmt<TY>; Q2_K: QI2_K, VDR 1)
template<int TY> struct IqShape;
template<> struct IqShape<16> { static constexpr int qk = 256, ipb = 8, step = 2; };    // IQ2_XXS
template<> struct IqShape<17> { static constexpr int qk = 256, ipb = 8, step = 2; };    // IQ2_XS
template<> struct IqShape<18> { static constexpr int qk = 256, ipb = 8, step = 2; };    // IQ3_XXS
template<> struct IqShape<21> { static constexpr int qk = 256, ipb = 8, step = 2; };    // IQ3_S
template<> struct IqShape<29> { static constexpr int qk = 256, ipb = 8, step = 1; };    // IQ1_M
template<> struct IqShape<10> { static constexpr int qk = 256, ipb = 16, step = 1; };   // Q2_K

}  // namespace talus_iq
