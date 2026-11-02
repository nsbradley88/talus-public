// src/prefill/tp_w4i_quant.hpp - rd-q4quant-B: the W4A4 activation quantizer of tp_w4i as device functions,
// shared by the stand-alone pack kernels (tp_w4i.cu) and the producers that pack in their epilogue (the fused
// residual-add + RMSNorm + pack).  Every caller runs the same per-256-value-block arithmetic, so the packed bytes are
// identical whichever kernel produced them.
//
// Per token t (one row of K values, K % 256 == 0), per 256-value block (lane L of a wave holds values 8 L .. 8 L + 7):
//   v <- H (D v) / 16                                 (seeded signs D, Sylvester Hadamard, tp_w4a4's order)
//   amax_g = max |v| over each 32-value group (4 lanes), m = max_g amax_g over the row
//   S_t = m / (7 * 127), k_g = clamp(ceil(amax_g / (7 S_t)), 1, 127), q = clamp(rint(v / (k_g S_t)), -7, 7)
//   xq: int4 (low nibble first), xk: d_g = k_g - k_{g+1} (Abel form, k past the weight block's end = 0), xS: S_t.
// m is order-free (fmaxf is exact), so the callers may split the row over any number of waves.
//
// Extension points (rd-q4quant phase 1, the parent's talus_pf_w4i_pack2 semantics): `rs` = an optional per-channel
// f32 multiplier applied after the load (and SwiGLU) and before the Hadamard (smoothing inv_s; nullptr = none), KM =
// the per-group k choice (0 = the shipped ceil(amax / 7 S) rule, 1 = best of 4 clips; KM 1 needs a kch[K / 32] LDS
// array and w4i_store_abel after a barrier).  With rs == nullptr and KM 0 the output is the shipped format bit for bit.
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace talus::prefill::w4i {

constexpr int kHB = 256;   // Hadamard block
constexpr int kAG = 32;    // activation group

/// SwiGLU input of the down projection: silu(g) * u with q36_engine's expression (x = u, g = gate)
__device__ __forceinline__ float w4i_swiglu(float g, float u) { return g / (1.0f + expf(-g)) * u; }

/// lane's 8 values of a 256-value block (signs / rs point at the lane's first value): v <- H (D (rs v)) / 16
__device__ __forceinline__ void w4i_rot256(float v[8], const int8_t* __restrict__ sg, const float* __restrict__ rs,
                                           int lane) {
    if (rs) {
#pragma unroll
        for (int e = 0; e < 8; ++e) v[e] *= rs[e];
    }
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

/// max |v| of the lane's 32-value group (4 lanes; every lane of the group gets it)
__device__ __forceinline__ float w4i_group_amax(const float v[8]) {
    float a = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) a = fmaxf(a, fabsf(v[e]));
    a = fmaxf(a, __shfl_xor(a, 1, 32));
    a = fmaxf(a, __shfl_xor(a, 2, 32));
    return a;
}

/// the token's scale terms from the row maximum m
struct W4iRow {
    float m, S, inv7S;
};
__device__ __forceinline__ W4iRow w4i_row(float m) {
    return {m, m / (7.0f * 127.0f), m > 0.f ? 127.0f / m : 0.f};   // inv7S = 1 / (7 S)
}

/// integer group scale k_g of a group with maximum a (the shipped ceil(amax / 7 S) rule)
__device__ __forceinline__ int w4i_k(float a, const W4iRow& r) {
    return r.m > 0.f ? min(127, max(1, (int) ceilf(a * r.inv7S))) : 1;
}

/// the lane's 8 int4 values (low nibble first) for group scale k
__device__ __forceinline__ uint32_t w4i_word(const float v[8], int k, const W4iRow& r) {
    const float inv = r.m > 0.f ? 1.0f / ((float) k * r.S) : 0.f;
    uint32_t word = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int q = (int) fminf(fmaxf(rintf(v[e] * inv), -7.f), 7.f);
        word |= (uint32_t) (q & 0xF) << (4 * e);
    }
    return word;
}

/// k of the group (4 lanes, collective) with maximum gm: KM 0 = kClipCeil; KM 1 = best of 4 clips (rd-q4quant's
/// talus_pf_w4i_pack2 kmode 1: k from amax x {1, 0.9, 0.8, 0.7}, least squared error, first minimum wins, fmaf-pinned)
__device__ __forceinline__ float w4i_kclip(int c) { return c == 0 ? 1.0f : c == 1 ? 0.9f : c == 2 ? 0.8f : 0.7f; }
template <int KM>
__device__ __forceinline__ int w4i_choose_k(const float v[8], float gm, const W4iRow& r) {
    if constexpr (KM == 0) {
        return w4i_k(gm, r);
    } else {
        int k = w4i_k(gm, r);
        float best = 0.f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int kc = w4i_k(gm * w4i_kclip(c), r);
            const float sc = (float) kc * r.S;
            const float iv = r.m > 0.f ? 1.0f / sc : 0.f;
            float err = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float q = fminf(fmaxf(rintf(v[e] * iv), -7.f), 7.f);
                const float d = __builtin_fmaf(-q, sc, v[e]);
                err = __builtin_fmaf(d, d, err);
            }
            err += __shfl_xor(err, 1, 32);
            err += __shfl_xor(err, 2, 32);
            if (c == 0 || err < best) {
                best = err;
                k = kc;
            }
        }
        return k;
    }
}

/// quantize + store one rotated 256-value block of token row (xq_t = xq + t K / 2, xk_t = xk + t K / 32): c0 = the
/// lane's first value, gmax = the row's group maxima (LDS), wb = the Abel block (the GEMM's weight group, or K).
/// KM 0: d_g = k_g - k_{g+1} written here (k_{g+1} from gmax).  KM != 0: k_g goes to kch (LDS) and the caller runs
/// w4i_store_abel after a barrier (the next group's k needs its values).
template <int KM>
__device__ __forceinline__ void w4i_store_block(const float v[8], int c0, const float* gmax, int wb, const W4iRow& r,
                                                uint8_t* __restrict__ xq_t, uint8_t* __restrict__ xk_t,
                                                uint8_t* kch, int lane) {
    const int gi = c0 / kAG;
    const int k = w4i_choose_k<KM>(v, gmax[gi], r);
    *(uint32_t*) (xq_t + c0 / 2) = w4i_word(v, k, r);
    if constexpr (KM == 0) {
        const int kn = ((gi + 1) * kAG) % wb != 0 ? w4i_k(gmax[gi + 1], r) : 0;   // (next group, same weight block)
        if ((lane & 3) == 0) xk_t[gi] = (uint8_t) (int8_t) (k - kn);
    } else {
        if ((lane & 3) == 0) kch[gi] = (uint8_t) k;
    }
}

/// KM != 0, after a barrier: the Abel form d_g = k_g - k_{g+1} of the row from kch (whole block)
__device__ __forceinline__ void w4i_store_abel(const uint8_t* kch, int K, int wb, uint8_t* __restrict__ xk_t) {
    const int NG = K / kAG;
    for (int gi = threadIdx.x; gi < NG; gi += blockDim.x) {
        const int kn = ((gi + 1) * kAG) % wb != 0 ? (int) kch[gi + 1] : 0;
        xk_t[gi] = (uint8_t) (int8_t) ((int) kch[gi] - kn);
    }
}

}  // namespace talus::prefill::w4i
