// src/prefill/moe_w16.cu - see include/strata/prefill/moe_w16.hpp (opt-in bf16 prompt experts for the native packs).
//
// The arithmetic.  Per weight row the GGUF block is decoded straight into bf16 values with the scale folded in:
// IQ4_NL w = bf16(d * kvalue[code]), IQ3_S w = +-bf16(d (1 + 2 s) * grid) (ggml's dequantize_row_* products, each
// rounded once to bf16, round-to-nearest-even); an activation value is bf16(d * q) of its int8 row (the native path's
// quant_act_nat_kernel / H epilogue).  v_wmma_f32_16x16x16_bf16 accumulates in fp32 over the whole K: no per-32-value
// float epilogue, which is what bounds the exact int8 kernels.
//
// The decode (per weight row, per 32 values): the block's 16 (IQ4_NL) or 8 (IQ3_S: the odd grid values 1..15) possible
// products are rounded to bf16 once and split into a low-byte and a high-byte plane of registers; four values at a
// time are then picked from the planes with v_perm_b32 (and v_bfi for the 16-entry codebook) and re-interleaved into
// bf16 pairs; IQ3_S's signs flip the bf16 sign bits (masks from an LDS table per sign byte).
//
// The tiles.  A work item = 256 weight rows x BM (64 or 128) routed rows of one expert.  8 waves, 32 weight rows each:
// every lane decodes its own row in registers (no LDS for weights) and swaps halves with its partner lane
// (v_permlanex16) to form the two replicated 16-row A operands; the BM activation rows are staged in LDS as bf16, 64
// values a stage, double-buffered, and every wave multiplies all BM / 16 token tiles.  Gate/up: lanes 0-15 hold gate
// rows and 16-31 the up rows of the same 16 features, so a C element's gate and up sit in one lane (SwiGLU in
// registers); H is rounded to int8 per 32 features (amax through LDS) in the native layout.  Down: FP32 rows of dm.
// Work items walk the batch's 64-row tiles (fused::group) x weight blocks on a persistent grid.
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "ggml.h"
#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "ggml-common.h"

#include "strata/prefill/moe_w16.hpp"
#include "strata/prefill/tp_tune.hpp"
#include "strata/prefill/tp_moe.hpp"

#include <algorithm>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::fused {
namespace {

constexpr int AB = 80;     // the native activation unit: 64 int8 codes, then {d0, d1, s0, s1}
constexpr int TR = 64;     // fused::group's tile rows (kTileRows)
constexpr int WROWS = 256; // weight rows per work item
constexpr int ASTR = 72;   // bf16 per staged activation row (64 + 8: 144-byte rows, conflict-free 16-lane reads)
constexpr int T_IQ3_S = GGML_TYPE_IQ3_S, T_IQ4_NL = GGML_TYPE_IQ4_NL, T_IQ2_S = GGML_TYPE_IQ2_S, T_IQ2_XXS = GGML_TYPE_IQ2_XXS,
              T_IQ3_XXS = GGML_TYPE_IQ3_XXS;

struct Tables {
    int32_t *cnt, *off, *fill, *ts;
    int2* tiles;
};
size_t tiles_at(int n_expert) { return ((size_t) (4 * n_expert + 2) * 4 + 15) / 16 * 16; }
Tables tables(void* scratch, int n_expert) {
    int32_t* p = (int32_t*) scratch;
    return {p, p + n_expert, p + 2 * n_expert + 1, p + 3 * n_expert + 1, (int2*) ((uint8_t*) scratch + tiles_at(n_expert))};
}

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x2 __attribute__((ext_vector_type(2)));

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define W16_DEV 1
#else
#define W16_DEV 0
#endif

// f rounded to bf16 in the high half of the result: nearest, ties to even (W16_TIES_AWAY: ties away from zero, one
// instruction instead of two)
__device__ __forceinline__ uint32_t rne(float f) {
    const uint32_t u = __float_as_uint(f);
#if defined(W16_TIES_AWAY)
    return u + 0x8000u;
#else
    return u + 0x7fffu + ((u >> 16) & 1u);
#endif
}
__device__ __forceinline__ uint32_t perm(uint32_t s0, uint32_t s1, uint32_t sel) { return __builtin_amdgcn_perm(s0, s1, sel); }
// f as a 16-bit operand value: H fp16 (round to nearest even, saturated to +-65504) in the low half, else bf16 (rne)
// in the high half
template <bool H> __device__ __forceinline__ uint32_t cvt16(float f) {
    if constexpr (H) return (uint32_t) __half_as_ushort(__float2half_rn(f));
    else return rne(f);
}
// two values as one operand dword (a low, b high): fp16 v_cvt_f16_f32 x2 + v_pack_b32_f16 (no saturation: the
// activations are range-scaled per row and a weight would need a block scale > 515 to overflow), or bf16 rne + perm
template <bool H> __device__ __forceinline__ uint32_t pk16(float a, float b) {
    if constexpr (H) return __builtin_bit_cast(uint32_t, __halves2half2(__float2half_rn(a), __float2half_rn(b)));
    else return __builtin_amdgcn_perm(rne(b), rne(a), 0x07060302u);
}
// planes of 4 table entries given as two operand dwords p01 = {e0, e1}, p23 = {e2, e3}
template <bool H> __device__ __forceinline__ void planes4p(uint32_t p01, uint32_t p23, uint32_t& lo, uint32_t& hi) {
    lo = perm(p23, p01, 0x06040200u);
    hi = perm(p23, p01, 0x07050301u);
}
// two cvt16 results as one operand dword (a low, b high)
template <bool H> __device__ __forceinline__ uint32_t pack16(uint32_t a, uint32_t b) {
    return __builtin_amdgcn_perm(b, a, H ? 0x05040100u : 0x07060302u);
}

// ---- raw block bytes: a lane's row data, fetched with dword buffer loads and re-aligned (blocks are 2-byte aligned)
__device__ __forceinline__ __amdgpu_buffer_rsrc_t rsrc(const void* p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), (short) 0, (int) bytes, 0x31004000);
}
// NW dwords from the dword holding byte off.  The callers size NW for a 2-byte-aligned start; from a 4-byte-aligned
// start the last dword holds none of the wanted bytes and may lie past the tensor (a page that is not mapped: the
// gate/up resource spans two tensors), so it is then read at an out-of-range offset (0, no access).
template <int NW> __device__ __forceinline__ void fetch(__amdgpu_buffer_rsrc_t r, uint32_t off, uint32_t (&o)[NW]) {
    const uint32_t a = off & ~3u;
    constexpr int NV = NW - 1;   // the dwords always wanted
#pragma unroll
    for (int m = 0; m + 4 <= NV; m += 4) {
        const i32x4 v = __builtin_amdgcn_raw_buffer_load_b128(r, a + 4 * m, 0, 0);
        o[m] = v[0]; o[m + 1] = v[1]; o[m + 2] = v[2]; o[m + 3] = v[3];
    }
    constexpr int R = NV % 4, M = NV - R;
    if constexpr (R >= 2) {
        const i32x2 v = __builtin_amdgcn_raw_buffer_load_b64(r, a + 4 * M, 0, 0);
        o[M] = v[0]; o[M + 1] = v[1];
    }
    if constexpr (R == 1 || R == 3) o[NV - 1] = __builtin_amdgcn_raw_buffer_load_b32(r, a + 4 * (NV - 1), 0, 0);
    o[NW - 1] = __builtin_amdgcn_raw_buffer_load_b32(r, (off & 3u) ? a + 4 * (NW - 1) : 0x80000000u, 0, 0);
}
template <int NW> __device__ __forceinline__ void align(uint32_t (&w)[NW], uint32_t off) {
    const uint32_t sh = (off & 3u) * 8u;
#pragma unroll
    for (int m = 0; m < NW - 1; ++m) w[m] = __builtin_amdgcn_alignbit(w[m + 1], w[m], sh);
    w[NW - 1] = w[NW - 1] >> sh;
}
template <int B, int NW> __device__ __forceinline__ uint32_t dw(const uint32_t (&w)[NW]) {
    if constexpr (B % 4 == 0) return w[B / 4];
    else return __builtin_amdgcn_alignbit(w[B / 4 + 1], w[B / 4], (B % 4) * 8);
}
template <int B, int NW> __device__ __forceinline__ uint32_t u8(const uint32_t (&w)[NW]) { return (w[B / 4] >> ((B % 4) * 8)) & 0xffu; }
template <int B, int NW> __device__ __forceinline__ float half_at(const uint32_t (&w)[NW]) {
    return __half2float(__ushort_as_half((unsigned short) ((w[B / 4] >> ((B % 4) * 8)) & 0xffffu)));
}

// planes of 4 table entries e0..e3 (16-bit values from cvt16): {lo bytes}, {hi bytes}
template <bool H> __device__ __forceinline__ void planes4(uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3, uint32_t& lo, uint32_t& hi) {
    constexpr uint32_t S = H ? 0x05010400u : 0x07030602u;   // {e0 low byte, e1 low byte, e0 high byte, e1 high byte}
    const uint32_t t01 = perm(e1, e0, S), t23 = perm(e3, e2, S);
    lo = perm(t23, t01, 0x05040100u);
    hi = perm(t23, t01, 0x07060302u);
}
// two bf16 pairs (values 0, 1 and 2, 3) from 4 low and 4 high bytes
__device__ __forceinline__ void interleave(uint32_t lo, uint32_t hi, uint32_t& o0, uint32_t& o1) {
    o0 = perm(hi, lo, 0x05010400u);
    o1 = perm(hi, lo, 0x07030602u);
}

// the sign-mask dword d (values 2 d, 2 d + 1 -> bits 15, 31) of sign byte k of w, computed (the LDS table
// ssgn[byte] holds the same words; the random per-lane table reads bank-conflict): byte k in both 16-bit halves,
// shifted by 15 - 2 d / 14 - 2 d, masked
// (measured slower than the LDS table on gfx1151 - more VALU in the decode - so off: W16_CS 0)
#ifndef W16_CS
#define W16_CS 0
#endif
typedef unsigned short w16_us2 __attribute__((ext_vector_type(2)));
template <int K, int D> __device__ __forceinline__ uint32_t sgn_dw(uint32_t w) {
    const uint32_t b2 = __builtin_amdgcn_perm(0u, w, 0x0c000c00u | (uint32_t) K | ((uint32_t) K << 16));
    const w16_us2 sh = {(unsigned short) (15 - 2 * D), (unsigned short) (14 - 2 * D)};
    return __builtin_bit_cast(uint32_t, __builtin_bit_cast(w16_us2, b2) << sh) & 0x80008000u;
}

// ---- the formats
template <int WT> struct Fmt;

// Per stage (64 values of K) a lane fetches the bytes its row needs into a small raw buffer R (raw dword loads), one
// stage ahead; prep() re-aligns them (the GGUF blocks are 2-byte aligned).  A group = GS stages (one GGUF block or
// whole blocks), unrolled so that every byte position is a compile-time constant.

// IQ4_NL (down, k 640): a stage = 2 blocks of 18 bytes (36 contiguous bytes), a group = 2 stages
template <> struct Fmt<T_IQ4_NL> {
    static constexpr int GS = 2;
    static constexpr int GB = 72;                     // group bytes in a row
    static constexpr int TK = 2;                      // a table per TK 16-value sub-stages (one scale per 32 values)
    static constexpr int NGRID = 0;                   // LDS grid words
    static __device__ __forceinline__ void init_lds(uint32_t*, int) {}
    static __device__ __forceinline__ void init_lds_u(uint32_t*, int) {}
    struct R { uint32_t w[10]; };
    struct Tab { uint32_t lo[4], hi[4]; };
    template <int ST> static __device__ __forceinline__ void fetch_stage(__amdgpu_buffer_rsrc_t r, uint32_t go, R& x) {
        fetch<10>(r, go + 36 * ST, x.w);
    }
    template <int ST> static __device__ __forceinline__ void prep(R& x, uint32_t go) { align<10>(x.w, go + 36 * ST); }
    template <int ST, int KK, bool H> static __device__ __forceinline__ void table(const R& x, const void*, Tab& t) {
        constexpr int B0 = 18 * (KK / 2);
        constexpr float kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        if constexpr (H) {
            // fp16: v_pk_mul_f16 of the block's fp16 scale and the (exact) codebook pairs = fp16(d * kv) rounded once
            const uint32_t d1 = (x.w[B0 / 4] >> ((B0 % 4) * 8)) & 0xffffu;
            const __half2 dd = __builtin_bit_cast(__half2, d1 | (d1 << 16));
            uint32_t p[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) p[i] = __builtin_bit_cast(uint32_t, __hmul2(__floats2half2_rn(kv[2 * i], kv[2 * i + 1]), dd));
#pragma unroll
            for (int k = 0; k < 4; ++k) planes4p<H>(p[2 * k], p[2 * k + 1], t.lo[k], t.hi[k]);
        } else {
            const float d = half_at<B0>(x.w);
            uint32_t e[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) e[i] = cvt16<H>(d * kv[i]);
#pragma unroll
            for (int k = 0; k < 4; ++k) planes4<H>(e[4 * k], e[4 * k + 1], e[4 * k + 2], e[4 * k + 3], t.lo[k], t.hi[k]);
        }
    }
    // 16 values (KK & 1: the high nibbles = values 16..31 of the block) as 8 bf16-pair dwords
    template <int ST, int KK, bool CS = false> static __device__ __forceinline__ void values(const R& x, const void*, const Tab& t, uint32_t (&o)[8]) {
        constexpr int B0 = 18 * (KK / 2) + 2;
        const uint32_t qw[4] = {dw<B0>(x.w), dw<B0 + 4>(x.w), dw<B0 + 8>(x.w), dw<B0 + 12>(x.w)};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t q = qw[m];
            const uint32_t n = (KK & 1) ? (q >> 4) & 0x0f0f0f0fu : q & 0x0f0f0f0fu;
            const uint32_t sel = n & 0x07070707u;
            const uint32_t msk = perm(0u, 0u, ((n >> 3) & 0x01010101u) | 0x0c0c0c0cu);   // 0xff where code >= 8
            const uint32_t lo = (msk & perm(t.lo[3], t.lo[2], sel)) | (~msk & perm(t.lo[1], t.lo[0], sel));
            const uint32_t hi = (msk & perm(t.hi[3], t.hi[2], sel)) | (~msk & perm(t.hi[1], t.hi[0], sel));
            interleave(lo, hi, o[2 * m], o[2 * m + 1]);
        }
    }
};

// IQ3_S (gate/up, k 2560): a group = 4 stages = one 256-value block of 110 bytes; a stage's bytes (sub-blocks
// J = 2 ST, 2 ST + 1): d (0-1), qs (2 + 16 ST, 16), qh (66 + 2 ST, 2), signs (74 + 8 ST, 8), scales (106 + ST, 1)
template <> struct Fmt<T_IQ3_S> {
    static constexpr int GS = 4;
    static constexpr int GB = 110;
    static constexpr int TK = 2, NGRID = 512;
    static __device__ __forceinline__ void init_lds(uint32_t* g, int tid) {
        for (int i = tid; i < 512; i += 256) g[i] = (iq3s_grid[i] >> 1) & 0x07070707u;   // odd 1..15 -> 0..7
    }
    static __device__ __forceinline__ void init_lds_u(uint32_t* g, int tid) {   // (256 threads; the loads in flight together)
        uint32_t v[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) v[j] = iq3s_grid[tid + 256 * j];
#pragma unroll
        for (int j = 0; j < 2; ++j) g[tid + 256 * j] = (v[j] >> 1) & 0x07070707u;
    }
    struct R { uint32_t q[5], sg[3], h, c, d; };
    struct Tab { uint32_t lo[2], hi[2]; };
    struct Lds { const uint32_t* grid; const uint4* sgn; };   // iq3s_grid as table indices; per sign byte the bf16 sign masks of its 8 values
    template <int ST> static __device__ __forceinline__ void fetch_stage(__amdgpu_buffer_rsrc_t r, uint32_t go, R& x) {
        fetch<5>(r, go + 2 + 16 * ST, x.q);
        fetch<3>(r, go + 74 + 8 * ST, x.sg);
        x.h = __builtin_amdgcn_raw_buffer_load_b32(r, (go + 66 + 2 * ST) & ~3u, 0, 0);
        x.c = __builtin_amdgcn_raw_buffer_load_b32(r, (go + 106 + ST) & ~3u, 0, 0);
        x.d = __builtin_amdgcn_raw_buffer_load_b32(r, go & ~3u, 0, 0);
    }
    template <int ST> static __device__ __forceinline__ void prep(R& x, uint32_t go) {
        align<5>(x.q, go + 2 + 16 * ST);
        align<3>(x.sg, go + 74 + 8 * ST);
        x.h = (x.h >> (((go + 66 + 2 * ST) & 3u) * 8u)) & 0xffffu;
        x.c = (x.c >> (((go + 106 + ST) & 3u) * 8u)) & 0xffu;
        x.d = (x.d >> ((go & 3u) * 8u)) & 0xffffu;
    }
    template <int ST, int KK, bool H> static __device__ __forceinline__ void table(const R& x, const void*, Tab& t) {
        constexpr int J = 2 * ST + KK / 2;   // 32-value sub-block of the block
        const float d = __half2float(__ushort_as_half((unsigned short) x.d));
        const int ls = (int) ((x.c >> (4 * (J & 1))) & 0xf);
        const float db = d * (float) (1 + 2 * ls);
        if constexpr (H) {
            uint32_t p[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) p[m] = pk16<H>(db * (float) (4 * m + 1), db * (float) (4 * m + 3));
            planes4p<H>(p[0], p[1], t.lo[0], t.hi[0]);
            planes4p<H>(p[2], p[3], t.lo[1], t.hi[1]);
        } else {
            uint32_t e[8];
#pragma unroll
            for (int m = 0; m < 8; ++m) e[m] = cvt16<H>(db * (float) (2 * m + 1));
            planes4<H>(e[0], e[1], e[2], e[3], t.lo[0], t.hi[0]);
            planes4<H>(e[4], e[5], e[6], e[7], t.lo[1], t.hi[1]);
        }
    }
    template <int ST, int KK, bool CS = false> static __device__ __forceinline__ void values(const R& x, const void* lds, const Tab& t, uint32_t (&o)[8]) {
        constexpr int j = KK / 2, E0 = 4 * (KK & 1);   // sub-block j of the stage, grid entries E0 .. E0 + 3
        const Lds& L = *(const Lds*) lds;
        const uint32_t qh = (x.h >> (8 * j)) & 0xffu, qd = x.q[2 * j + (E0 >> 2)], sw = x.sg[j];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int e = E0 + m;
            const uint32_t idx = ((qd >> (8 * m)) & 0xffu) | (((qh >> e) & 1u) << 8);
            const uint32_t sel = L.grid[idx];   // (the LDS copy holds the table indices (g - 1) / 2 of the grid bytes)
            const uint32_t lo = perm(t.lo[1], t.lo[0], sel), hi = perm(t.hi[1], t.hi[0], sel);
            uint32_t o0, o1;
            interleave(lo, hi, o0, o1);
            if constexpr (CS) {
                // grid entry e: sign byte e / 2, dwords 2 (e & 1), 2 (e & 1) + 1 (e is a compile-time constant here)
                uint32_t s0, s1;
                switch (e) {
                    case 0: s0 = sgn_dw<0, 0>(sw); s1 = sgn_dw<0, 1>(sw); break;
                    case 1: s0 = sgn_dw<0, 2>(sw); s1 = sgn_dw<0, 3>(sw); break;
                    case 2: s0 = sgn_dw<1, 0>(sw); s1 = sgn_dw<1, 1>(sw); break;
                    case 3: s0 = sgn_dw<1, 2>(sw); s1 = sgn_dw<1, 3>(sw); break;
                    case 4: s0 = sgn_dw<2, 0>(sw); s1 = sgn_dw<2, 1>(sw); break;
                    case 5: s0 = sgn_dw<2, 2>(sw); s1 = sgn_dw<2, 3>(sw); break;
                    case 6: s0 = sgn_dw<3, 0>(sw); s1 = sgn_dw<3, 1>(sw); break;
                    default: s0 = sgn_dw<3, 2>(sw); s1 = sgn_dw<3, 3>(sw); break;
                }
                o[2 * m] = o0 ^ s0;
                o[2 * m + 1] = o1 ^ s1;
            } else {
            const uint4 sm = L.sgn[(sw >> (8 * (e >> 1))) & 0xffu];
            o[2 * m] = o0 ^ ((e & 1) ? sm.z : sm.x);
            o[2 * m + 1] = o1 ^ ((e & 1) ? sm.w : sm.y);
            }
        }
    }
};

// IQ2_S (GLM's gate/up): a group = 4 stages = one 256-value block of 82 bytes; a stage's bytes (32-value sub-blocks
// 2 ST, 2 ST + 1): d (0-1), grid index low bytes (2 + 8 ST, 8), signs (34 + 8 ST, 8), qh (66 + 2 ST, 2), scales
// (74 + 2 ST, 2: a 4-bit scale per 16 values).  w = d (0.5 + s) / 4 x {8, 25, 43} x sign: a 3-entry table per 16
// values, looked up with one v_perm per two values (selectors i, i + 4 into the {low bytes, high bytes} planes).
template <> struct Fmt<T_IQ2_S> {
    static constexpr int GS = 4, GB = 82, TK = 1, NGRID = 2048;
    struct R { uint32_t q[3], sg[3], h, c, d; };
    struct Tab { uint32_t lo, hi; };
    struct Lds { const uint32_t* grid; const uint4* sgn; };   // iq2s_grid as table indices (2 words an entry); sign masks
    static __device__ __forceinline__ void init_lds(uint32_t* g, int tid) {
        for (int i = tid; i < 1024; i += 256) {   // grid bytes 0x08, 0x19, 0x2b -> 0, 1, 2
            const uint64_t v = iq2s_grid[i];
            g[2 * i] = ((uint32_t) v >> 4) & 0x0f0f0f0fu;
            g[2 * i + 1] = ((uint32_t) (v >> 32) >> 4) & 0x0f0f0f0fu;
        }
    }
    static __device__ __forceinline__ void init_lds_u(uint32_t* g, int tid) {   // (256 threads; the loads in flight together)
        uint64_t v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = iq2s_grid[tid + 256 * j];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int i = tid + 256 * j;
            g[2 * i] = ((uint32_t) v[j] >> 4) & 0x0f0f0f0fu;
            g[2 * i + 1] = ((uint32_t) (v[j] >> 32) >> 4) & 0x0f0f0f0fu;
        }
    }
    template <int ST> static __device__ __forceinline__ void fetch_stage(__amdgpu_buffer_rsrc_t r, uint32_t go, R& x) {
        fetch<3>(r, go + 2 + 8 * ST, x.q);
        fetch<3>(r, go + 34 + 8 * ST, x.sg);
        x.h = __builtin_amdgcn_raw_buffer_load_b32(r, (go + 66 + 2 * ST) & ~3u, 0, 0);
        x.c = __builtin_amdgcn_raw_buffer_load_b32(r, (go + 74 + 2 * ST) & ~3u, 0, 0);
        x.d = __builtin_amdgcn_raw_buffer_load_b32(r, go & ~3u, 0, 0);
    }
    template <int ST> static __device__ __forceinline__ void prep(R& x, uint32_t go) {
        align<3>(x.q, go + 2 + 8 * ST);
        align<3>(x.sg, go + 34 + 8 * ST);
        x.h = (x.h >> (((go + 66 + 2 * ST) & 3u) * 8u)) & 0xffffu;   // (2-byte fields at even offsets: one dword)
        x.c = (x.c >> (((go + 74 + 2 * ST) & 3u) * 8u)) & 0xffffu;
        x.d = (x.d >> ((go & 3u) * 8u)) & 0xffffu;
    }
    template <int ST, int KK, bool H> static __device__ __forceinline__ void table(const R& x, const void*, Tab& t) {
        constexpr int j = KK / 2;
        const float d = __half2float(__ushort_as_half((unsigned short) x.d));
        const int ls = (int) ((x.c >> (8 * j + 4 * (KK & 1))) & 0xf);
        const float db = d * (float) (2 * ls + 1) * 0.125f;   // (exact: the products below are rounded once)
        planes4p<H>(pk16<H>(db * 8.0f, db * 25.0f), pk16<H>(db * 43.0f, 0.0f), t.lo, t.hi);
    }
    template <int ST, int KK, bool CS = false> static __device__ __forceinline__ void values(const R& x, const void* lds, const Tab& t, uint32_t (&o)[8]) {
        constexpr int j = KK / 2;
        const Lds& L = *(const Lds*) lds;
        const uint32_t qh = (x.h >> (8 * j)) & 0xffu;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            constexpr int l0 = 2 * (KK & 1);
            const int l = l0 + m;
            const uint32_t idx = ((x.q[j] >> (8 * l)) & 0xffu) | ((qh << (8 - 2 * l)) & 0x300u);
            const uint2 g = *(const uint2*) &L.grid[2 * idx];
            uint4 sm;
            if constexpr (CS) {
                const uint32_t w = x.sg[j];
                if (l == 0) sm = make_uint4(sgn_dw<0, 0>(w), sgn_dw<0, 1>(w), sgn_dw<0, 2>(w), sgn_dw<0, 3>(w));
                else if (l == 1) sm = make_uint4(sgn_dw<1, 0>(w), sgn_dw<1, 1>(w), sgn_dw<1, 2>(w), sgn_dw<1, 3>(w));
                else if (l == 2) sm = make_uint4(sgn_dw<2, 0>(w), sgn_dw<2, 1>(w), sgn_dw<2, 2>(w), sgn_dw<2, 3>(w));
                else sm = make_uint4(sgn_dw<3, 0>(w), sgn_dw<3, 1>(w), sgn_dw<3, 2>(w), sgn_dw<3, 3>(w));
            } else {
                sm = L.sgn[(x.sg[j] >> (8 * l)) & 0xffu];
            }
            const uint32_t a0 = g.x | 0x04040404u, a1 = g.y | 0x04040404u;
            o[4 * m] = perm(t.hi, t.lo, perm(a0, g.x, 0x05010400u)) ^ sm.x;
            o[4 * m + 1] = perm(t.hi, t.lo, perm(a0, g.x, 0x07030602u)) ^ sm.y;
            o[4 * m + 2] = perm(t.hi, t.lo, perm(a1, g.y, 0x05010400u)) ^ sm.z;
            o[4 * m + 3] = perm(t.hi, t.lo, perm(a1, g.y, 0x07030602u)) ^ sm.w;
        }
    }
};

template <int WT> constexpr bool w16_ksigns() { return WT == T_IQ2_XXS || WT == T_IQ3_XXS; }

// IQ2_XXS (MiMo's gate/up): a group = 4 stages = one 256-value block of 66 bytes; a stage = 32-value sub-blocks 2 ST,
// 2 ST + 1 = 16 bytes at 2 + 16 ST: per sub-block 4 grid index bytes, then a word of 4 x 7-bit sign codes + the 4-bit
// scale (bits 28-31).  w = d (0.5 + s) / 4 x {8, 25, 43} x sign, the IQ2_S table and lookup.
template <> struct Fmt<T_IQ2_XXS> {
    static constexpr int GS = 4, GB = 66, TK = 2, NGRID = 512;
    struct R { uint32_t q[5], d; };
    struct Tab { uint32_t lo, hi; };
    struct Lds { const uint32_t* grid; const uint4* sgn; };
    static __device__ __forceinline__ void init_lds(uint32_t* g, int tid) {
        for (int i = tid; i < 256; i += 256) {
            const uint64_t v = iq2xxs_grid[i];
            g[2 * i] = ((uint32_t) v >> 4) & 0x0f0f0f0fu;
            g[2 * i + 1] = ((uint32_t) (v >> 32) >> 4) & 0x0f0f0f0fu;
        }
    }
    static __device__ __forceinline__ void init_lds_u(uint32_t* g, int tid) { init_lds(g, tid); }
    template <int ST> static __device__ __forceinline__ void fetch_stage(__amdgpu_buffer_rsrc_t r, uint32_t go, R& x) {
        fetch<5>(r, go + 2 + 16 * ST, x.q);
        x.d = __builtin_amdgcn_raw_buffer_load_b32(r, go & ~3u, 0, 0);
    }
    template <int ST> static __device__ __forceinline__ void prep(R& x, uint32_t go) {
        align<5>(x.q, go + 2 + 16 * ST);
        x.d = (x.d >> ((go & 3u) * 8u)) & 0xffffu;
    }
    template <int ST, int KK, bool H> static __device__ __forceinline__ void table(const R& x, const void*, Tab& t) {
        constexpr int j = KK / 2;
        const float d = __half2float(__ushort_as_half((unsigned short) x.d));
        const float db = d * (float) (2 * (int) (x.q[2 * j + 1] >> 28) + 1) * 0.125f;
        planes4p<H>(pk16<H>(db * 8.0f, db * 25.0f), pk16<H>(db * 43.0f, 0.0f), t.lo, t.hi);
    }
    template <int ST, int KK, bool CS = false> static __device__ __forceinline__ void values(const R& x, const void* lds, const Tab& t, uint32_t (&o)[8]) {
        constexpr int j = KK / 2;
        const Lds& L = *(const Lds*) lds;
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int l = 2 * (KK & 1) + m;
            const uint2 g = *(const uint2*) &L.grid[2 * ((x.q[2 * j] >> (8 * l)) & 0xffu)];
            const uint4 sm = L.sgn[(x.q[2 * j + 1] >> (7 * l)) & 0x7fu];
            const uint32_t a0 = g.x | 0x04040404u, a1 = g.y | 0x04040404u;
            o[4 * m] = perm(t.hi, t.lo, perm(a0, g.x, 0x05010400u)) ^ sm.x;
            o[4 * m + 1] = perm(t.hi, t.lo, perm(a0, g.x, 0x07030602u)) ^ sm.y;
            o[4 * m + 2] = perm(t.hi, t.lo, perm(a1, g.y, 0x05010400u)) ^ sm.z;
            o[4 * m + 3] = perm(t.hi, t.lo, perm(a1, g.y, 0x07030602u)) ^ sm.w;
        }
    }
};

// IQ3_XXS (MiMo's down): a group = 4 stages = one 256-value block of 98 bytes; a stage = sub-blocks 2 ST, 2 ST + 1:
// grid index bytes (2 + 16 ST, 16: 4-value entries) and sign / scale words (66 + 8 ST, 8: 4 x 7-bit sign codes + the
// 4-bit scale).  w = d (0.5 + s) / 2 x {4, 12, .., 52, 62} x sign: an 8-entry table (the IQ3_S planes and lookup).
template <> struct Fmt<T_IQ3_XXS> {
    static constexpr int GS = 4, GB = 98, TK = 2, NGRID = 256;
    struct R { uint32_t q[5], ss[3], d; };
    struct Tab { uint32_t lo[2], hi[2]; };
    struct Lds { const uint32_t* grid; const uint4* sgn; };
    static __device__ __forceinline__ void init_lds(uint32_t* g, int tid) {
        for (int i = tid; i < 256; i += 256) g[i] = (iq3xxs_grid[i] >> 3) & 0x07070707u;   // 4, 12, .., 52, 62 -> 0..7
    }
    static __device__ __forceinline__ void init_lds_u(uint32_t* g, int tid) { init_lds(g, tid); }
    template <int ST> static __device__ __forceinline__ void fetch_stage(__amdgpu_buffer_rsrc_t r, uint32_t go, R& x) {
        fetch<5>(r, go + 2 + 16 * ST, x.q);
        fetch<3>(r, go + 66 + 8 * ST, x.ss);
        x.d = __builtin_amdgcn_raw_buffer_load_b32(r, go & ~3u, 0, 0);
    }
    template <int ST> static __device__ __forceinline__ void prep(R& x, uint32_t go) {
        align<5>(x.q, go + 2 + 16 * ST);
        align<3>(x.ss, go + 66 + 8 * ST);
        x.d = (x.d >> ((go & 3u) * 8u)) & 0xffffu;
    }
    template <int ST, int KK, bool H> static __device__ __forceinline__ void table(const R& x, const void*, Tab& t) {
        constexpr int j = KK / 2;
        const float d = __half2float(__ushort_as_half((unsigned short) x.d));
        const float db = d * (float) (2 * (int) (x.ss[j] >> 28) + 1) * 0.25f;
        uint32_t p[4];
#pragma unroll
        for (int m = 0; m < 3; ++m) p[m] = pk16<H>(db * (float) (16 * m + 4), db * (float) (16 * m + 12));
        p[3] = pk16<H>(db * 52.0f, db * 62.0f);
        planes4p<H>(p[0], p[1], t.lo[0], t.hi[0]);
        planes4p<H>(p[2], p[3], t.lo[1], t.hi[1]);
    }
    template <int ST, int KK, bool CS = false> static __device__ __forceinline__ void values(const R& x, const void* lds, const Tab& t, uint32_t (&o)[8]) {
        constexpr int j = KK / 2, E0 = 4 * (KK & 1);
        const Lds& L = *(const Lds*) lds;
        const uint32_t qd = x.q[2 * j + (E0 >> 2)], sw = x.ss[j];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int e = E0 + m;
            const uint32_t sel = L.grid[(qd >> (8 * m)) & 0xffu];
            const uint32_t lo = perm(t.lo[1], t.lo[0], sel), hi = perm(t.hi[1], t.hi[0], sel);
            uint32_t o0, o1;
            interleave(lo, hi, o0, o1);
            const uint4 sm = L.sgn[(sw >> (7 * (e >> 1))) & 0x7fu];
            o[2 * m] = o0 ^ ((e & 1) ? sm.z : sm.x);
            o[2 * m + 1] = o1 ^ ((e & 1) ? sm.w : sm.y);
        }
    }
};

// ---- the kernel
// MQ: the tp_moe contract (talus::prefill::MoeProduct: stacked GGUF experts, MMQ-layout q8_1 activations (block_q8_1_mmq,
// D4: 4 float scales then 128 int8 a block, blocks of a K step [row]-major), bounds / ids, FP32 dst rows; GU = gate
// and up tensors, dst = the (clamped) SwiGLU in FP32)
struct MqArgs {
    const uint8_t* w = nullptr;
    int64_t up_delta = 0;            // gate/up: w_up - w (bytes)
    size_t expert_bytes = 0;
    uint32_t row_bytes = 0;
    int k = 0;
    const uint8_t* xq = nullptr;
    uint32_t xstride = 0;            // bytes between K steps (total_rows x 144)
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    float* dst = nullptr;
    int64_t ldd = 0;
    float limit = 0.0f;
    int n_experts = 0;               // CMP: the experts behind bounds (the compact item grid scans them)
    // CMP only (GLM entries): src = the activation row of routed row r is xq row src[r] (xq quantized once per
    // token, not per route; xstride then spans the token rows); hq (GU) = the SwiGLU rows quantized in the kernel to
    // MMQ q8_1 D4 blocks at the routed (sorted) row, quantize_mmq_q8_1's arithmetic (127 / amax per 32 values),
    // hq_stride bytes between 128-feature blocks; dst unused then
    const int32_t* src = nullptr;
    uint8_t* hq = nullptr;
    uint32_t hq_stride = 0;
    int ft = 0;                      // CMP: full items take the token-tile loop without exits (TALUS_W16_FT=1)
    int dst16 = 0;                   // CMP down: dst rows written as fp16 (N4, GLM_N4=1; output-changing)
};
constexpr int MQ_XB = 144;   // block_q8_1_mmq

// CMP (MQ only; bitwise the CMP = false kernel): a compact work-item grid (blockIdx.x = weight block, blockIdx.y =
// the item among all experts' BM-row chunks, found by a block-wide scan of the bounds; no blocks per (expert, empty
// chunk)), the item found before the LDS tables are built (blocks past the last item exit at once), the table init
// unrolled (its global loads in flight together), and the rows' range scale read with all threads in flight (the max
// of the same values: order-independent).
template <int WT, bool GU, int BM, bool H = true, int DIAG = 0, bool MQ = false, bool U2 = false, bool CMP = false>
__global__ void __launch_bounds__(256) w16_kernel(const Batch b, const NativeGeom geo, const Tables tb,
                                                  const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
                                                  uint8_t* __restrict__ out, float* __restrict__ dm, const MqArgs mq) {
#if W16_DEV
    using F = Fmt<WT>;
    const int K = MQ ? mq.k : (GU ? 2560 : 640);
    const int NST = K / 64;                     // stages
    const int NG = NST / F::GS;                 // groups
    constexpr int NFB = GU ? 640 / (WROWS / 2) : 2560 / WROWS;   // (native) weight blocks per row tile
    constexpr int BT = BM / 16;                 // token tiles
    const int ACT_LD = (K / 64) * AB;           // (native) bytes per activation row
    constexpr int UPT = BM * 4 / 256;           // 16-value staging units per thread (1 or 2)
    __shared__ __attribute__((aligned(16))) __bf16 abuf[2][BM][ASTR];
    __shared__ uint32_t amax[GU && !MQ ? BM : 1][4];
    __shared__ int csum[GU && !MQ ? BM : 1][4];
    __shared__ float rsc[H ? BM : 1], rinv[H ? BM : 1];   // fp16: per routed row a power of two that keeps it in range
    __shared__ __attribute__((aligned(8))) uint32_t sgrid[F::NGRID > 0 ? F::NGRID : 1];
    __shared__ uint4 ssgn[F::NGRID > 0 ? 256 : 1];
    const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    static_assert(!CMP || MQ, "CMP: tp_moe contract only");
    __shared__ int citem[CMP ? 3 : 1];
    __shared__ int cwsum[CMP ? 8 : 1];
    if constexpr (CMP) {
        // item blockIdx.y of the experts' BM-row chunks (expert order): thread t covers experts 2t, 2t + 1
        const int E = mq.n_experts, ea = 2 * tid, eb = ea + 1;
        int ba = 0, bb = 0, bc = 0;
        if (ea < E) { ba = mq.bounds[ea]; bb = mq.bounds[ea + 1]; }
        if (eb < E) bc = mq.bounds[eb + 1];
        const int ca = ea < E ? (bb - ba + BM - 1) / BM : 0, cb = eb < E ? (bc - bb + BM - 1) / BM : 0;
        int inc = ca + cb;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int o = __shfl_up(inc, d, 32);
            if (lane >= d) inc += o;
        }
        if (lane == 31) cwsum[wv] = inc;
        if (tid == 0) citem[0] = -1;
        __syncthreads();
        for (int w = 0; w < wv; ++w) inc += cwsum[w];
        const int it = (int) blockIdx.y, ex = inc - ca - cb;
        if (it >= ex && it < inc) {
            const bool fa = it < ex + ca;
            citem[0] = fa ? ea : eb;
            citem[1] = fa ? ba + (it - ex) * BM : bb + (it - ex - ca) * BM;
            citem[2] = fa ? bb : bc;
        }
        __syncthreads();
        if (citem[0] < 0) return;   // block-uniform: past the last item
    }
    if constexpr (F::NGRID > 0) {
        if constexpr (CMP) F::init_lds_u(sgrid, tid);
        else F::init_lds(sgrid, tid);
        {   // sign masks: value v of entry (v < 4: low nibble) flips bit 15 / 31 of its bf16 pair dword; the
            // ksigns formats (IQ2_XXS, IQ3_XXS) index them by the 7-bit code (ggml's ksigns_iq2xs)
            const int s = w16_ksigns<WT>() ? (tid < 128 ? (int) ksigns_iq2xs[tid] : 0) : tid;
            uint32_t m[4] = {0, 0, 0, 0};
#pragma unroll
            for (int v = 0; v < 8; ++v)
                if ((s >> v) & 1) m[v >> 1] |= (v & 1) ? 0x80000000u : 0x00008000u;
            ssgn[tid] = make_uint4(m[0], m[1], m[2], m[3]);
        }
    }
    const void* lds = nullptr;
    struct LdsW { const uint32_t* grid; const uint4* sgn; } lw{sgrid, ssgn};
    if constexpr (F::NGRID > 0) lds = &lw;
    // native: a persistent / per-item grid over the batch's 64-row tiles x weight blocks; MQ: one item a block
    // (grid: 128-row chunks of the expert x weight blocks x experts)
    const int nwork = MQ ? 1 : (tb.ts[b.e1] - tb.ts[b.e0]) * NFB;
    for (int wk = MQ ? 0 : (int) blockIdx.x; wk < nwork; wk += MQ ? 1 : (int) gridDim.x) {
        int e, row0, nrows, fb;
        if constexpr (CMP) {
            e = citem[0];
            fb = (int) blockIdx.x;
            row0 = citem[1];
            nrows = min(BM, citem[2] - row0);
        } else if constexpr (MQ) {
            e = (int) blockIdx.z;
            fb = (int) blockIdx.y;
            const int eb = mq.bounds[e], ee = mq.bounds[e + 1];
            row0 = eb + (int) blockIdx.x * BM;
            if (row0 >= ee) break;
            nrows = min(BM, ee - row0);
        } else {
            const int ti = tb.ts[b.e0] + wk / NFB;
            fb = wk % NFB;
            const int2 tl = tb.tiles[ti];
            e = tl.x;
            row0 = tl.y;
            if (BM == 128 && ((ti - tb.ts[e]) & 1)) continue;   // the odd tiles of an expert ride with the even ones
            nrows = min(BM, tb.off[e + 1] - row0);
        }
        const int ntt = (nrows + 15) >> 4;                   // token tiles with rows (the others are skipped)
        __syncthreads();                                     // the previous item is done with the LDS
        // this lane's weight row (the buffer resource is the blob's, wave-uniform; the lane's offset is per lane)
        const int lrow = wv * 32 + lane;
        __amdgpu_buffer_rsrc_t wr, wr2;
        uint32_t roff;
        if constexpr (MQ) {
            // gate/up: one resource over both experts (the launcher checks that they lie within 2 GB of each other):
            // the lanes' loads stay one instruction (two resources require a divergent branch per fetch)
            const uint8_t* bg = mq.w + (size_t) e * mq.expert_bytes;
            const uint8_t* bu = GU ? bg + mq.up_delta : bg;
            if constexpr (U2) {   // (gate and up farther apart than 4 GB: a resource each, a divergent fetch)
                const uint32_t mg = (uint32_t) ((uintptr_t) bg & 3), mu = (uint32_t) ((uintptr_t) bu & 3);
                wr = rsrc((const void*) ((uintptr_t) bg & ~(uintptr_t) 3), (mg + (uint32_t) mq.expert_bytes + 3u) & ~3u);
                wr2 = rsrc((const void*) ((uintptr_t) bu & ~(uintptr_t) 3), (mu + (uint32_t) mq.expert_bytes + 3u) & ~3u);
                roff = (hi ? mu : mg) + (uint32_t) (fb * (WROWS / 2) + wv * 16 + l16) * mq.row_bytes;
            } else {
            const uintptr_t lo = ((uintptr_t) (bu < bg ? bu : bg)) & ~(uintptr_t) 3;
            const uintptr_t end = (uintptr_t) (bu < bg ? bg : bu) + mq.expert_bytes;
            wr = rsrc((const void*) lo, (uint32_t) ((end - lo + 3) & ~(uintptr_t) 3));
            wr2 = wr;
            if constexpr (GU) roff = (uint32_t) ((uintptr_t) (hi ? bu : bg) - lo) + (uint32_t) (fb * (WROWS / 2) + wv * 16 + l16) * mq.row_bytes;
            else roff = (uint32_t) ((uintptr_t) bg - lo) + (uint32_t) (fb * WROWS + lrow) * mq.row_bytes;
            }
        } else {
            const uint8_t* blob = b.blob[e - b.e0];
            const uint32_t mis = (uint32_t) ((uintptr_t) blob & 3);
            wr = rsrc((const void*) ((uintptr_t) blob & ~(uintptr_t) 3),
                      (mis + (uint32_t) (GU ? geo.up_off + 640 * geo.gu_row : geo.down_off + 2560 * geo.d_row) + 3u) & ~3u);
            wr2 = wr;
            if constexpr (GU) roff = mis + (uint32_t) (hi ? geo.up_off : 0) + (uint32_t) (fb * (WROWS / 2) + wv * 16 + l16) * (uint32_t) geo.gu_row;
            else roff = mis + (uint32_t) geo.down_off + (uint32_t) (fb * WROWS + lrow) * (uint32_t) geo.d_row;
        }
        auto wfetch = [&]<int ST>(uint32_t go, typename F::R& x) {
            if constexpr (U2) {
                if (hi) F::template fetch_stage<ST>(wr2, go, x);
                else F::template fetch_stage<ST>(wr, go, x);
            } else {
                F::template fetch_stage<ST>(wr, go, x);
            }
        };
        // staging units: (row, 16-value quarter of a stage)
        const uint8_t* arow[UPT];
        int urow[UPT], uq[UPT];
#pragma unroll
        for (int u = 0; u < UPT; ++u) {
            const int un = tid + 256 * u;
            urow[u] = un >> 2;
            uq[u] = un & 3;
            const int r = row0 + min(urow[u], nrows - 1);
            if constexpr (MQ) arow[u] = mq.xq + (size_t) (CMP && mq.src ? mq.src[r] : r) * MQ_XB;
            else arow[u] = act + (size_t) (GU ? src[r] : r) * ACT_LD;
        }
        if constexpr (GU && !MQ) {
            for (int i = tid; i < BM * 4; i += 256) { ((uint32_t*) amax)[i] = 0u; ((int*) csum)[i] = 0; }
        }
        float rs[UPT];
        if constexpr (H) {
            // fp16 range: a row is staged times 2^-k with its largest |value| (127 x its largest scale) in [2^12, 2^13)
            // and its outputs scaled back by 2^k (exact)
            if constexpr (CMP) {
                // TPR threads a row, up to 16 scale loads each in flight; partial maxima joined across the row's lanes
                constexpr int TPR = 256 / BM;
                const int r = tid / TPR, sub = tid % TPR;
                const int nu = K / 128;
                float mx = 0.0f;
                if (r < 16 * ntt) {
                    const int rr = row0 + min(r, nrows - 1);
                    const uint8_t* p = mq.xq + (size_t) (mq.src ? mq.src[rr] : rr) * MQ_XB;
                    for (int u0 = sub; u0 < nu; u0 += 16 * TPR) {
                        float4 dd[16];
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            const int u = min(u0 + TPR * j, nu - 1);   // (a repeated unit: the same maximum)
                            dd[j] = *(const float4*) (p + (size_t) u * mq.xstride);
                        }
#pragma unroll
                        for (int j = 0; j < 16; ++j) mx = fmaxf(mx, fmaxf(fmaxf(dd[j].x, dd[j].y), fmaxf(dd[j].z, dd[j].w)));
                    }
                }
#pragma unroll
                for (int d = 1; d < TPR; d <<= 1) mx = fmaxf(mx, __shfl_xor(mx, d, 32));
                if (r < 16 * ntt && sub == 0) {
                    const float m = mx * 127.0f;
                    const int k = m > 0.0f ? ilogbf(m) - 12 : 0;
                    rsc[r] = ldexpf(1.0f, -k);
                    rinv[r] = ldexpf(1.0f, k);
                }
            } else
            for (int r = tid; r < 16 * ntt; r += 256) {
                const int rr = row0 + min(r, nrows - 1);
                float mx = 0.0f;
                if constexpr (MQ) {
                    const uint8_t* p = mq.xq + (size_t) rr * MQ_XB;
                    for (int u = 0; u < K / 128; ++u) {
                        const float4 dd = *(const float4*) (p + (size_t) u * mq.xstride);
                        mx = fmaxf(mx, fmaxf(fmaxf(dd.x, dd.y), fmaxf(dd.z, dd.w)));
                    }
                } else {
                    const uint8_t* p = act + (size_t) (GU ? src[rr] : rr) * ACT_LD + 64;
                    for (int u = 0; u < K / 64; ++u) {
                        const float2 dd = *(const float2*) (p + (size_t) u * AB);
                        mx = fmaxf(mx, fmaxf(dd.x, dd.y));
                    }
                }
                const float m = mx * 127.0f;
                // scaled to [2^12, 2^13): in range and well above fp16's subnormals (which would lose small rows)
                const int k = m > 0.0f ? ilogbf(m) - 12 : 0;
                rsc[r] = ldexpf(1.0f, -k);
                rinv[r] = ldexpf(1.0f, k);
            }
            __syncthreads();
#pragma unroll
            for (int u = 0; u < UPT; ++u) rs[u] = urow[u] < 16 * ntt ? rsc[urow[u]] : 1.0f;
        } else {
#pragma unroll
            for (int u = 0; u < UPT; ++u) rs[u] = 1.0f;
        }
        uint4 araw[UPT];
        float ad[UPT];
        auto aload = [&](int s) {
#pragma unroll
            for (int u = 0; u < UPT; ++u) {
                if (urow[u] >= 16 * ntt) continue;
                if constexpr (MQ) {   // K step s / 2, its half s & 1: scales d0..d3 then 128 codes
                    const uint8_t* p = arow[u] + (size_t) (s >> 1) * mq.xstride;
                    araw[u] = *(const uint4*) (p + 16 + 64 * (s & 1) + 16 * uq[u]);
                    ad[u] = *(const float*) (p + 4 * (2 * (s & 1) + (uq[u] >> 1)));
                } else {
                    const uint8_t* p = arow[u] + s * AB;
                    araw[u] = *(const uint4*) (p + 16 * uq[u]);
                    ad[u] = *(const float*) (p + 64 + 4 * (uq[u] >> 1));
                }
            }
        };
        auto astore = [&](int buf) {
#pragma unroll
            for (int u = 0; u < UPT; ++u) {
                if (urow[u] >= 16 * ntt) continue;
                const float d = ad[u] * rs[u], nd = -128.0f * d;
                const uint32_t q[4] = {araw[u].x ^ 0x80808080u, araw[u].y ^ 0x80808080u, araw[u].z ^ 0x80808080u,
                                       araw[u].w ^ 0x80808080u};
                uint32_t o[8];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    o[2 * j] = pk16<H>(fmaf((float) (q[j] & 0xffu), d, nd), fmaf((float) ((q[j] >> 8) & 0xffu), d, nd));
                    o[2 * j + 1] = pk16<H>(fmaf((float) ((q[j] >> 16) & 0xffu), d, nd), fmaf((float) (q[j] >> 24), d, nd));
                }
                uint4* dst = (uint4*) &abuf[buf][urow[u]][16 * uq[u]];
                dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
                dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
            }
        };
        f32x8 acc[2][BT];
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int t = 0; t < BT; ++t) acc[a][t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
        typename F::R rw[2];
        wfetch.template operator()<0>(roff, rw[0]);
        aload(0);
        astore(0);
        __syncthreads();
        auto group = [&]<bool FULL>(int gi) {
            const uint32_t go = roff + (uint32_t) gi * F::GB;
            auto stage = [&]<int ST>() {
                const int s = gi * F::GS + ST;
                const int cb = s & 1;
                typename F::R& cur = rw[ST & 1];
                // the next stage's raw bytes (this group's or the next group's first). CMP: the activation loads
                // first, so the stage-end wait for them (vmcnt is in order) leaves the weight prefetch in flight
                if constexpr (CMP) {
                    // this stage's weights (the previous stage's prefetch: the only loads in flight here) waited for
                    // explicitly before the next loads are issued (vmcnt counts in order; after the divergent aload
                    // branches the compiler would otherwise wait for everything): gfx11 s_waitcnt vmcnt(0)
                    __builtin_amdgcn_s_waitcnt(0x03F7);
                    F::template prep<ST>(cur, go);
                    if (s + 1 < NST) aload(s + 1);
                }
                if constexpr (ST + 1 < F::GS) wfetch.template operator()<ST + 1>(go, rw[(ST + 1) & 1]);
                else if (gi + 1 < NG) wfetch.template operator()<0>(go + F::GB, rw[0]);
                if constexpr (!CMP) {
                    F::template prep<ST>(cur, go);
                    if (s + 1 < NST) aload(s + 1);
                }
                typename F::Tab tab;
                auto sub = [&]<int KK>() {
                    if constexpr (KK % F::TK == 0 && DIAG != 1) F::template table<ST, KK, H>(cur, lds, tab);
                    uint32_t own[8];
                    if constexpr (DIAG == 1) {   // timing only: no decode
#pragma unroll
                        for (int d = 0; d < 8; ++d) own[d] = 0x3c003c00u + (uint32_t) (d + lane);
                    } else {
                        F::template values<ST, KK, CMP && W16_CS>(cur, lds, tab, own);
                    }
                    u32x8 a0, a1;
#pragma unroll
                    for (int d = 0; d < 8; ++d) {
                        const uint32_t pt = (uint32_t) __builtin_amdgcn_permlanex16((int) own[d], (int) own[d], 0x76543210, 0xfedcba98, false, false);
                        a0[d] = hi ? pt : own[d];
                        a1[d] = hi ? own[d] : pt;
                    }
                    using V16 = std::conditional_t<H, f16x16, bf16x16>;
                    const V16 A0 = __builtin_bit_cast(V16, a0), A1 = __builtin_bit_cast(V16, a1);
                    if constexpr (DIAG == 2) { acc[0][0][0] += __uint_as_float(a0[0] ^ a1[7]); acc[1][0][1] += __uint_as_float(a1[3] ^ a0[5]); }
#pragma unroll
                    for (int t = 0; t < BT; ++t) {
                        if ((!FULL && t >= ntt) || DIAG == 2) break;
                        const V16 B = *(const V16*) &abuf[cb][16 * t + l16][16 * KK];
                        if constexpr (H) {
                            acc[0][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A0, B, acc[0][t]);
                            acc[1][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A1, B, acc[1][t]);
                        } else {
                            acc[0][t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(A0, B, acc[0][t]);
                            acc[1][t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(A1, B, acc[1][t]);
                        }
                    }
                };
                sub.template operator()<0>();
                sub.template operator()<1>();
                sub.template operator()<2>();
                sub.template operator()<3>();
                if (s + 1 < NST) astore(cb ^ 1);
                if constexpr (CMP) {
                    // LDS-only barrier: the staged activations are the only memory shared across waves here, so the
                    // in-flight global / buffer loads (the next stage's weights) stay in flight past it (a
                    // __syncthreads() fence at workgroup scope waits for every outstanding load)
                    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
                } else {
                    __syncthreads();
                }
            };
            stage.template operator()<0>();
            stage.template operator()<1>();
            if constexpr (F::GS == 4) {
                stage.template operator()<2>();
                stage.template operator()<3>();
            }
        };
        if (CMP && mq.ft && ntt == BT) {   // (full item: the token-tile loop without per-tile exits)
#pragma unroll 1
            for (int gi = 0; gi < NG; ++gi) group.template operator()<CMP>(gi);
        } else {
#pragma unroll 1
            for (int gi = 0; gi < NG; ++gi) group.template operator()<false>(gi);
        }
        if constexpr (H) {   // undo the rows' range scale (powers of two: exact)
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                if (t >= ntt) break;
                const float iv = rinv[16 * t + l16];
#pragma unroll
                for (int a = 0; a < 2; ++a)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[a][t][i] *= iv;
            }
        }
        if (CMP && GU && mq.hq) {
            // SwiGLU (the f32 path's expression) -> q8_1 D4 blocks: amax per (row, 32 features = waves 2q, 2q + 1)
            // through LDS (the staging buffers are free), d_inv = 127 / amax, codes roundf(h d_inv), d = 1 / d_inv
            const int q = wv >> 1;
            uint32_t* qmax = (uint32_t*) &abuf[0][0][0];                          // [BM][4]
            int8_t (*hqs)[128] = (int8_t (*)[128]) ((uint8_t*) &abuf[0][0][0] + BM * 16);   // [BM][128]
            __syncthreads();
            for (int i = tid; i < BM * 4; i += 256) qmax[i] = 0u;
            __syncthreads();
            auto& h = acc[0];
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                if (t >= ntt) break;
                float mx = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float gv = acc[0][t][i], uv = acc[1][t][i];
                    if (mq.limit > 0.0f) { gv = fminf(gv, mq.limit); uv = fminf(fmaxf(uv, -mq.limit), mq.limit); }
                    h[t][i] = gv / (1.0f + expf(-gv)) * uv;
                    mx = fmaxf(mx, fabsf(h[t][i]));
                }
                const float other = __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(mx), __float_as_int(mx), 0x76543210, 0xfedcba98, false, false));
                mx = fmaxf(mx, other);
                if (hi == 0) atomicMax(&qmax[(16 * t + l16) * 4 + q], __float_as_uint(mx));
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                if (t >= ntt) break;
                const int r = 16 * t + l16;
                const float d_inv = 127.0f / __uint_as_float(qmax[r * 4 + q]);
#pragma unroll
                for (int i = 0; i < 8; ++i) hqs[r][wv * 16 + 2 * i + hi] = (int8_t) roundf(h[t][i] * d_inv);
            }
            __syncthreads();
            for (int r = tid; r < nrows; r += 256) {
                uint8_t* o = mq.hq + (size_t) fb * mq.hq_stride + (size_t) (row0 + r) * MQ_XB;
                float dd[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) dd[j] = 1.0f / (127.0f / __uint_as_float(qmax[r * 4 + j]));
                *(float4*) o = make_float4(dd[0], dd[1], dd[2], dd[3]);
                const uint4* sq = (const uint4*) &hqs[r][0];
#pragma unroll
                for (int j = 0; j < 8; ++j) ((uint4*) (o + 16))[j] = sq[j];
            }
        } else if constexpr (GU && MQ) {
            // tp_moe's GU epilogue: dst[row][f] = g / (1 + exp(-g)) * u (clamped by limit > 0), f = this lane's features
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                if (t >= ntt) break;
                const int r = 16 * t + l16;
                if (r >= nrows) continue;
                const int64_t o = (int64_t) (mq.ids ? mq.ids[row0 + r] : row0 + r) * mq.ldd + fb * (WROWS / 2) + wv * 16 + hi;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float gv = acc[0][t][i], uv = acc[1][t][i];
                    if (mq.limit > 0.0f) { gv = fminf(gv, mq.limit); uv = fminf(fmaxf(uv, -mq.limit), mq.limit); }
                    mq.dst[o + 2 * i] = gv / (1.0f + expf(-gv)) * uv;
                }
            }
        } else if constexpr (GU) {
            // SwiGLU in registers: acc[0] gate, acc[1] up of features wv*16 + 2i + hi (local), token 16 t + l16
            const int q = wv >> 1;   // 32-feature block of the item (0..3)
            auto& h = acc[0];   // h overwrites the gate accumulators
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                if (t >= ntt) break;
                float mx = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float gt = acc[0][t][i], up = acc[1][t][i];
                    h[t][i] = gt / (1.0f + __expf(-gt)) * up;
                    mx = fmaxf(mx, fabsf(h[t][i]));
                }
                const float other = __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(mx), __float_as_int(mx), 0x76543210, 0xfedcba98, false, false));
                mx = fmaxf(mx, other);
                if (hi == 0) atomicMax(&amax[16 * t + l16][q], __float_as_uint(mx));
            }
            __syncthreads();
            uint8_t (*hq)[128] = (uint8_t (*)[128]) &abuf[0][0][0];   // the staging buffers are free now
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                if (t >= ntt) break;
                const int r = 16 * t + l16;
                const float am = __uint_as_float(amax[r][q]);
                const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                int cs = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int c = __float2int_rn(h[t][i] * inv);
                    cs += c;
                    hq[r][wv * 16 + 2 * i + hi] = (uint8_t) (int8_t) c;
                }
                atomicAdd(&csum[r][q], cs);
            }
            __syncthreads();
            // rows x the item's 2 H units (64 features each): codes, {d0, d1, s0, s1}
            for (int u = tid; u < BM * 2; u += 256) {
                const int r = u >> 1, hu = u & 1;
                if (r >= nrows) continue;
                uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (size_t) (2 * fb + hu) * AB;
                const uint4* sq = (const uint4*) &hq[r][64 * hu];
#pragma unroll
                for (int j = 0; j < 4; ++j) ((uint4*) o)[j] = sq[j];
                *(float4*) (o + 64) = make_float4(__uint_as_float(amax[r][2 * hu]) / 127.0f, __uint_as_float(amax[r][2 * hu + 1]) / 127.0f,
                                                  (float) csum[r][2 * hu], (float) csum[r][2 * hu + 1]);
            }
        } else {
#pragma unroll
            for (int t = 0; t < BT; ++t) {
                const int r = 16 * t + l16;
                if (r >= nrows) continue;
#pragma unroll
                for (int a = 0; a < 2; ++a) {
                    if (CMP && mq.dst16) {
                        _Float16* d16 = (_Float16*) mq.dst + (int64_t) (mq.ids ? mq.ids[row0 + r] : row0 + r) * mq.ldd + fb * WROWS + wv * 32 + 16 * a + hi;
#pragma unroll
                        for (int i = 0; i < 8; ++i) d16[2 * i] = (_Float16) acc[a][t][i];
                        continue;
                    }
                    float* d = MQ ? mq.dst + (int64_t) (mq.ids ? mq.ids[row0 + r] : row0 + r) * mq.ldd + fb * WROWS + wv * 32 + 16 * a + hi
                                  : dm + (size_t) (row0 + r) * 2560 + fb * WROWS + wv * 32 + 16 * a + hi;
#pragma unroll
                    for (int i = 0; i < 8; ++i) d[2 * i] = acc[a][t][i];
                }
            }
        }
    }
#else
    (void) b; (void) geo; (void) tb; (void) act; (void) src; (void) out; (void) dm; (void) mq;
#endif
}

int sms() {
    static int n = [] {
        hipDeviceProp_t p{};
        int dev = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return 0;
        return p.multiProcessorCount;
    }();
    return n;
}

}  // namespace

void w16_host_iq3s_grid(uint32_t* out) {
    if (hipMemcpyFromSymbol(out, HIP_SYMBOL(iq3s_grid), 512 * sizeof(uint32_t)) != hipSuccess) std::memset(out, 0, 2048);
}

bool w16_enabled(int gu_type, int d_type) {
    // the tune row's pf_w16: 0 not allowed, 1 opt-in (STRATA_PF_W16=1), 2 on by default (STRATA_PF_W16=0 = kill switch)
    static const int mode = talus::prefill::tp_tune_current().pf_w16;
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_W16");
        return v != nullptr ? v[0] == '1' : talus::prefill::tp_tune_current().pf_w16 >= 2;
    }();
    return on && mode != 0 && gu_type == T_IQ3_S && d_type == T_IQ4_NL;
}

bool experts_native_w16(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch, const void* xa,
                        const int32_t* src, void* ha, float* dm, void* stream, int phase, bool force) {
    if (!(force || w16_enabled(g.gu_type, g.d_type)) || g.gu_type != T_IQ3_S || g.d_type != T_IQ4_NL) return false;
    if (b.e1 <= b.e0 || n <= 0) return true;
    const talus::prefill::TpTune& tu = talus::prefill::tp_tune_current();
    if (tu.pf_w16 == 0 && !force) return false;
    // below pf_w16_min_rows routed rows per expert on average (decode / verify windows, short chunks) the int8 kernels
    // are as fast (memory-bound weight streaming: T 16-1024 x1.00, T 4-8 x0.93-0.97)
    static const int env_min = [] { const char* v = std::getenv("STRATA_PF_W16_MIN"); return v ? std::atoi(v) : -1; }();
    if (!force && n < (int64_t) n_expert * (env_min >= 0 ? env_min : tu.pf_w16_min_rows)) return false;
    const hipStream_t s = (hipStream_t) stream;
    const Tables tb = tables(const_cast<void*>(scratch), n_expert);
    static const bool dbg = std::getenv("STRATA_PF_W16_DEBUG") != nullptr;
    static int ncall = 0;
    if (dbg && ncall < 200) {
        ++ncall;
        std::fprintf(stderr, "w16: experts [%d, %d) n %lld\n", b.e0, b.e1, (long long) n);
    }
    const int64_t tiles = (n + TR - 1) / TR + (b.e1 - b.e0);
    static const int env_per = [] { const char* v = std::getenv("STRATA_PF_W16_BLOCKS"); return v ? std::atoi(v) : 0; }();
    // routed rows per work item: the tune row's, or (0) by the mean rows per expert - 128 from 64 rows per expert
    static const int env_bm = [] { const char* v = std::getenv("STRATA_PF_W16_BM"); return v ? std::atoi(v) : -1; }();
    const int64_t mean = n / std::max(1, n_expert);
    auto pick = [&](int t) { const int v = env_bm >= 0 ? env_bm : t; return v == 64 ? 64 : v == 128 ? 128 : (mean >= 64 ? 128 : 64); };
    static const int env_gu = [] { const char* v = std::getenv("STRATA_PF_W16_BM_GU"); return v ? std::atoi(v) : -1; }();
    const int bm_gu = env_gu > 0 ? (env_gu == 128 ? 128 : 64) : pick(tu.pf_w16_bm_gu), bm_d = pick(tu.pf_w16_bm_d);
    // blocks per CU of a persistent grid (by the item rows), or (0: gfx1151's for 128-row items) a block per work item:
    // the hardware then overlaps an item's prologue / epilogue with the others' main loops on large chunks (while
    // 64-row items keep 2 blocks per CU on the persistent grid)
    auto grid = [&](int bm) {
        const int per = env_per > 0 ? env_per : bm == 128 ? tu.pf_w16_blocks128 : tu.pf_w16_blocks;
        return (unsigned) std::min<int64_t>(tiles * 10, per > 0 ? (int64_t) sms() * per : tiles * 10);
    };
    const unsigned g_gu = grid(bm_gu), g_d = grid(bm_d);
    // STRATA_PF_W16_BF16=1: bf16 operands (8-bit mantissa) instead of fp16 (11-bit, the default for higher mantissa
    // precision; see docs/RESULTS.md); STRATA_PF_W16_DIAG=1 / 2 (timing only, wrong output): no weight
    // decode / no WMMA (BM 128)
    static const bool bf = [] { const char* v = std::getenv("STRATA_PF_W16_BF16"); return v && v[0] == '1'; }();
    static const int diag = [] { const char* v = std::getenv("STRATA_PF_W16_DIAG"); return v ? std::atoi(v) : 0; }();
    const uint8_t* xa8 = (const uint8_t*) xa;
    uint8_t* ha8 = (uint8_t*) ha;
#define W16_GU(BM, H, DG) w16_kernel<T_IQ3_S, true, BM, H, DG><<<g_gu, 256, 0, s>>>(b, g, tb, xa8, src, ha8, nullptr, MqArgs{})
#define W16_D(BM, H, DG) w16_kernel<T_IQ4_NL, false, BM, H, DG><<<g_d, 256, 0, s>>>(b, g, tb, ha8, src, nullptr, dm, MqArgs{})
    if (diag == 1 || diag == 2) {
        if (phase != 2) { if (diag == 1) W16_GU(128, true, 1); else W16_GU(128, true, 2); }
        if (phase != 1) { if (diag == 1) W16_D(128, true, 1); else W16_D(128, true, 2); }
    } else if (bf) {
        if (phase != 2) { if (bm_gu == 128) W16_GU(128, false, 0); else W16_GU(64, false, 0); }
        if (phase != 1) { if (bm_d == 128) W16_D(128, false, 0); else W16_D(64, false, 0); }
    } else {
        if (phase != 2) { if (bm_gu == 128) W16_GU(128, true, 0); else W16_GU(64, true, 0); }
        if (phase != 1) { if (bm_d == 128) W16_D(128, true, 0); else W16_D(64, true, 0); }
    }
#undef W16_GU
#undef W16_D
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        std::fprintf(stderr, "prefill experts (bf16): %s\n", hipGetErrorString(e));
        std::exit(1);
    }
    return true;
}

}  // namespace strata::prefill::fused

namespace talus::prefill {

bool moe_w16_enabled(int type) {
    static const bool on = [] { const char* v = std::getenv("STRATA_PF_W16_MOE"); return v != nullptr && v[0] == '1'; }();
    return on && (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ2_XXS ||
                  type == GGML_TYPE_IQ3_XXS) && tp_tune_current().pf_w16 != 0;
}

static int w16_ft() {
    static const int v = [] { const char* e = std::getenv("TALUS_W16_FT"); return e && e[0] == '1' ? 1 : 0; }();
    return v;
}
bool moe_gemm_w16(const MoeProduct& p, void* stream, bool force) {
    namespace F = strata::prefill::fused;
    if (!(force || moe_w16_enabled(p.type))) return false;
    const bool gu = p.w_up != nullptr;
    const int t = p.type;
    const bool cov = t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_S || t == GGML_TYPE_IQ2_XXS || t == GGML_TYPE_IQ3_XXS ||
                     (!gu && t == GGML_TYPE_IQ4_NL);
    if (!cov) return false;
    if (p.n % (gu ? 128 : 256) != 0 || p.k % (t == GGML_TYPE_IQ4_NL ? 128 : 256) != 0 || p.k <= 0) return false;
    if (p.max_rows <= 0 || p.n_experts <= 0 || p.total_rows <= 0) return false;
    static const int env_min = [] { const char* v = std::getenv("STRATA_PF_W16_MOE_MIN"); return v ? std::atoi(v) : -1; }();
    if (!force && p.total_rows < (int64_t) p.n_experts * (env_min >= 0 ? env_min : tp_tune_current().pf_w16_moe_min_rows)) return false;
    const int bb = t == GGML_TYPE_IQ2_S ? 82 : t == GGML_TYPE_IQ3_S ? 110 : t == GGML_TYPE_IQ2_XXS ? 66 : t == GGML_TYPE_IQ3_XXS ? 98 : 18;
    const int bv = t == GGML_TYPE_IQ4_NL ? 32 : 256;
    F::MqArgs a;
    a.w = (const uint8_t*) p.w;
    a.up_delta = gu ? (int64_t) ((const uint8_t*) p.w_up - (const uint8_t*) p.w) : 0;
    a.expert_bytes = p.expert_bytes;
    a.row_bytes = (uint32_t) (p.k / bv * bb);
    a.k = p.k;
    a.xq = (const uint8_t*) p.xq;
    a.xstride = (uint32_t) (p.total_rows * F::MQ_XB);
    a.bounds = p.bounds;
    a.ids = p.ids;
    a.dst = p.dst;
    a.ldd = p.ld_dst;
    a.limit = p.swiglu_limit;
    a.n_experts = p.n_experts;
    a.ft = w16_ft();
    const uint64_t ad = (uint64_t) (a.up_delta < 0 ? -a.up_delta : a.up_delta);
    // gate and up > 2 GB apart: a resource each (fetch's out-of-range offset is 2^31)
    const bool u2 = gu && p.expert_bytes + ad + 8 > 0x7fffffffull;
    if (p.expert_bytes + 8 > 0x7fffffffull || (uint64_t) p.total_rows * F::MQ_XB * (p.k / 128) > 0xffffffffull) return false;
    // routed rows per item: 128 from a mean of 64 rows per expert (STRATA_PF_W16_BM overrides); a block per item
    static const int env_bm = [] { const char* v = std::getenv("STRATA_PF_W16_BM"); return v ? std::atoi(v) : -1; }();
    static const bool bf = [] { const char* v = std::getenv("STRATA_PF_W16_BF16"); return v && v[0] == '1'; }();
    const int64_t mean = p.total_rows / p.n_experts;
    const int bm = env_bm == 64 ? 64 : env_bm == 128 ? 128 : (mean >= 64 ? 128 : 64);
    const hipStream_t s = (hipStream_t) stream;
    const dim3 grid((unsigned) ((p.max_rows + bm - 1) / bm), (unsigned) (p.n / (gu ? 128 : 256)), (unsigned) p.n_experts);
    // compact item grid (TALUS_W16_CMP=0: the per-(expert, chunk) grid above; bitwise the same products): x = weight
    // block (the blocks sharing an item's activation rows run together), y = item; sum over experts of ceil(rows / bm)
    // <= total_rows / bm + n_experts
    static const bool cmp = [] { const char* v = std::getenv("TALUS_W16_CMP"); return v == nullptr || v[0] != '0'; }();
    const bool use_cmp = cmp && p.n_experts <= 512;
    const dim3 gridc((unsigned) (p.n / (gu ? 128 : 256)), (unsigned) ((p.total_rows + bm - 1) / bm + p.n_experts), 1u);
    const F::Batch nb{};
    const F::NativeGeom ng{};
    const F::Tables nt{};
#define W16_MQ(T, G, BM, HH) do { \
        if (use_cmp) { if (G && u2) F::w16_kernel<T, G, BM, HH, 0, true, G, true><<<gridc, 256, 0, s>>>(nb, ng, nt, nullptr, nullptr, nullptr, nullptr, a); \
                       else F::w16_kernel<T, G, BM, HH, 0, true, false, true><<<gridc, 256, 0, s>>>(nb, ng, nt, nullptr, nullptr, nullptr, nullptr, a); } \
        else if (G && u2) F::w16_kernel<T, G, BM, HH, 0, true, G><<<grid, 256, 0, s>>>(nb, ng, nt, nullptr, nullptr, nullptr, nullptr, a); \
        else F::w16_kernel<T, G, BM, HH, 0, true><<<grid, 256, 0, s>>>(nb, ng, nt, nullptr, nullptr, nullptr, nullptr, a); } while (0)
#define W16_MQ2(T, G) do { if (bf) { if (bm == 128) W16_MQ(T, G, 128, false); else W16_MQ(T, G, 64, false); } \
                           else { if (bm == 128) W16_MQ(T, G, 128, true); else W16_MQ(T, G, 64, true); } } while (0)
    if (gu) {
        if (t == GGML_TYPE_IQ2_S) W16_MQ2(F::T_IQ2_S, true);
        else if (t == GGML_TYPE_IQ2_XXS) W16_MQ2(F::T_IQ2_XXS, true);
        else if (t == GGML_TYPE_IQ3_XXS) W16_MQ2(F::T_IQ3_XXS, true);
        else W16_MQ2(F::T_IQ3_S, true);
    } else {
        if (t == GGML_TYPE_IQ2_S) W16_MQ2(F::T_IQ2_S, false);
        else if (t == GGML_TYPE_IQ3_S) W16_MQ2(F::T_IQ3_S, false);
        else if (t == GGML_TYPE_IQ2_XXS) W16_MQ2(F::T_IQ2_XXS, false);
        else if (t == GGML_TYPE_IQ3_XXS) W16_MQ2(F::T_IQ3_XXS, false);
        else W16_MQ2(F::T_IQ4_NL, false);
    }
#undef W16_MQ2
#undef W16_MQ
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
        std::fprintf(stderr, "prefill experts (bf16, tp_moe contract): %s\n", hipGetErrorString(e));
        std::exit(1);
    }
    return true;
}


int g_w16_ed16 = 0;   // set by tkg_w16_moe_down when its rows are fp16, taken by the combine
namespace {
// the compact (CMP) MQ kernels for the GLM entries below; bm by the mean rows per expert as moe_gemm_w16
bool w16_glm_launch(int t, bool gu, bool u2, int64_t total_rows, int n, int n_experts, const strata::prefill::fused::MqArgs& a,
                    hipStream_t s) {
    namespace F = strata::prefill::fused;
    static const int env_bm = [] { const char* v = std::getenv("STRATA_PF_W16_BM"); return v ? std::atoi(v) : -1; }();
    const int64_t mean = total_rows / n_experts;
    // 128 routed rows per item from a mean of 32 (per-row results do not depend on the item size)
    const int bm = env_bm == 64 ? 64 : env_bm == 128 ? 128 : (mean >= 32 ? 128 : 64);
    const dim3 g((unsigned) (n / (gu ? 128 : 256)), (unsigned) ((total_rows + bm - 1) / bm + n_experts), 1u);
    const F::Batch nb{};
    const F::NativeGeom ng{};
    const F::Tables nt{};
#define W16G(T, G, BM) do { if (G && u2) F::w16_kernel<T, G, BM, true, 0, true, G, true><<<g, 256, 0, s>>>(nb, ng, nt, nullptr, nullptr, nullptr, nullptr, a); \
                            else F::w16_kernel<T, G, BM, true, 0, true, false, true><<<g, 256, 0, s>>>(nb, ng, nt, nullptr, nullptr, nullptr, nullptr, a); } while (0)
#define W16G2(T, G) do { if (bm == 128) W16G(T, G, 128); else W16G(T, G, 64); } while (0)
    if (gu) {
        if (t == GGML_TYPE_IQ2_S) W16G2(F::T_IQ2_S, true);
        else if (t == GGML_TYPE_IQ3_S) W16G2(F::T_IQ3_S, true);
        else return false;
    } else {
        if (t == GGML_TYPE_IQ2_S) W16G2(F::T_IQ2_S, false);
        else if (t == GGML_TYPE_IQ3_S) W16G2(F::T_IQ3_S, false);
        else return false;
    }
#undef W16G2
#undef W16G
    return true;
}
bool w16_glm_on(int type, int64_t total_rows, int n_experts) {
    static const bool on = [] { const char* v = std::getenv("TALUS_W16_GLMQ"); return v == nullptr || v[0] != '0'; }();
    if (!on || !moe_w16_enabled(type) || n_experts <= 0 || n_experts > 512 || total_rows <= 0) return false;
    if (type != GGML_TYPE_IQ2_S && type != GGML_TYPE_IQ3_S) return false;
    static const int env_min = [] { const char* v = std::getenv("STRATA_PF_W16_MOE_MIN"); return v ? std::atoi(v) : -1; }();
    return total_rows >= (int64_t) n_experts * (env_min >= 0 ? env_min : tp_tune_current().pf_w16_moe_min_rows);
}
int w16_bb(int t) { return t == GGML_TYPE_IQ2_S ? 82 : 110; }
}  // namespace

}  // namespace talus::prefill

// GLM entries (tkg_ prefix: looked up as optional symbols by the GLM engine). Return 0 = launched, -1 = not
// covered (nothing launched: the caller runs the tp_moe path), > 0 = HIP error.
// gu_q8: fused gate/up + clamped SwiGLU over the routed rows; activation row r = xq row src[r] (xq: MMQ q8_1 D4 rows of
// the n_src_rows tokens); output = MMQ q8_1 D4 rows (n_ff wide) at the routed rows into hq (bitwise tp_moe's
// quantize of the f32 SwiGLU rows gathered in routed order).
extern "C" int tkg_w16_moe_gu_q8(int type, const void* w_gate, const void* w_up, size_t expert_bytes, int n_ff, int n_embd,
                                 int n_experts, const void* xq, const int32_t* src, int64_t n_src_rows, const int32_t* bounds,
                                 int64_t total_rows, void* hq, float limit, void* stream) {
    using namespace talus::prefill;
    namespace F = strata::prefill::fused;
    if (!w16_glm_on(type, total_rows, n_experts) || n_ff % 128 || n_embd % 256) return -1;
    if (expert_bytes + 8 > 0x7fffffffull || (uint64_t) n_src_rows * F::MQ_XB * (n_embd / 128) > 0xffffffffull ||
        (uint64_t) total_rows * F::MQ_XB * (n_ff / 128) > 0xffffffffull)
        return -1;
    F::MqArgs a;
    a.w = (const uint8_t*) w_gate;
    a.up_delta = (int64_t) ((const uint8_t*) w_up - (const uint8_t*) w_gate);
    a.expert_bytes = expert_bytes;
    a.row_bytes = (uint32_t) (n_embd / 256 * w16_bb(type));
    a.k = n_embd;
    a.xq = (const uint8_t*) xq;
    a.xstride = (uint32_t) (n_src_rows * F::MQ_XB);
    a.bounds = bounds;
    a.limit = limit;
    a.n_experts = n_experts;
    a.ft = w16_ft();
    a.src = src;
    a.hq = (uint8_t*) hq;
    a.hq_stride = (uint32_t) (total_rows * F::MQ_XB);
    const uint64_t ad = (uint64_t) (a.up_delta < 0 ? -a.up_delta : a.up_delta);
    const bool u2 = expert_bytes + ad + 8 > 0x7fffffffull;
    if (!w16_glm_launch(type, true, u2, total_rows, n_ff, n_experts, a, (hipStream_t) stream)) return -1;
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int) e;
}
// down: dst row ids[r] (ld ldd) of routed row r = W_e hq row r (hq: MMQ q8_1 D4 rows, k wide)
extern "C" int tkg_w16_moe_down(int type, const void* w, size_t expert_bytes, int n_out, int k, int n_experts, const void* hq,
                                const int32_t* bounds, const int32_t* ids, int64_t total_rows, float* dst, int64_t ldd,
                                void* stream) {
    using namespace talus::prefill;
    namespace F = strata::prefill::fused;
    if (!w16_glm_on(type, total_rows, n_experts) || n_out % 256 || k % 256) return -1;
    if (expert_bytes + 8 > 0x7fffffffull || (uint64_t) total_rows * F::MQ_XB * (k / 128) > 0xffffffffull) return -1;
    F::MqArgs a;
    a.w = (const uint8_t*) w;
    a.expert_bytes = expert_bytes;
    a.row_bytes = (uint32_t) (k / 256 * w16_bb(type));
    a.k = k;
    a.xq = (const uint8_t*) hq;
    a.xstride = (uint32_t) (total_rows * F::MQ_XB);
    a.bounds = bounds;
    a.ids = ids;
    a.dst = dst;
    a.ldd = ldd;
    a.n_experts = n_experts;
    a.ft = w16_ft();
    static const int n4 = [] { const char* v = std::getenv("GLM_N4"); return v && v[0] == '1' ? 1 : 0; }();
    a.dst16 = n4;
    if (!w16_glm_launch(type, false, false, total_rows, n_out, n_experts, a, (hipStream_t) stream)) return -1;
    talus::prefill::g_w16_ed16 = n4;   // the next combine reads fp16 rows (tkg_hc_post_mix_moe_pre)
    const hipError_t e = hipGetLastError();
    return e == hipSuccess ? 0 : (int) e;
}


