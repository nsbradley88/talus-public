// src/qwen35/q36_kv8.cu - an int8 copy of the Qwen3.5 / 3.6 / 3.8 (q36) attention KV cache for decode and
// verify attention (TALUS_Q27_KV8=1, default off), and the WMMA reader that uses it.
//
// Why: 27B decode / verify attention reads the f16 K cache and the cell-blocked f16 V copy, 2 x 512 B per cell and KV
// head, about 2.15 GB per round at 32K, and it already runs near the bandwidth ceiling (198 GB/s).  The only lever
// left is bytes.  This copy holds 528 B per cell and KV head (int8 codes + fp16 scales), about 1.11 GB at 32K.
//
// Format (the Flash-Next int8 KV rule, include/strata/kernels/kv_q8.hpp, with an offset-binary code):
//   per cell, KV head and group of 64 values:  scale = fp16(max|x| / 127),  c = clamp(rint(x / scale), -127, 127)
//   stored as the byte u = c + 128 (1..255), so a reader turns 2 bytes into 2 exact f16 values with one byte splice
//   (0x64uu = 1024 + u) and one packed subtract of 1152.  x is the f16 value already in the f16 cache, so the int8 copy
//   is a pure function of the f16 cache: a prompt-cache restore (which copies f16 cells only) rebuilds it bit for bit
//   with talus_q36_kv8_quant over the restored cells.
// Layouts (per attention layer):
//   kq  u8  [kv head][ctx][256]              ks  fp16 [kv head][ctx][4]       (K: one row of 256 codes per cell)
//   vq  u8  [kv head][vt_ctx / 16][256][16]  vs  fp16 [kv head][4][vt_ctx]    (V: the vt cell-blocked layout)
// The prompt path (tp_fa, T > the verify width) keeps reading the f16 cache, so prompt numerics do not change; only the
// decode / verify rows (T <= 16) read this copy.
//
// The reader is k_attn_wmma with three changes; every row's arithmetic still depends on that row only, so a verify
// row stays bitwise the decode row at the same position:
//   S = Q K^T: the K codes enter the f16 WMMA as exact integers, one accumulator per 64-dim group; the group scales
//     are applied per cell (the C column) after the WMMA, in f32, in a fixed order.
//   O += P V: wave w owns dims [64 w, 64 w + 64) = V group w, so the per-cell V scales fold into P (A operand) once per
//     tile: P' = fp16(P * scale), then the WMMA with the V codes.
//   The own-tile scalar sum (the row's position tile) uses P * (code * scale) in f32.
#include "strata/qwen35/q36_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace {

using namespace talus::q36;

constexpr int kHD = 256;
constexpr int kGroup = 64;   // values per scale (kv_q8.hpp KV_Q8_GROUP)
constexpr int kGroups = kHD / kGroup;

typedef _Float16 kv_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 kv_h8 __attribute__((ext_vector_type(8)));
typedef float kv_f8 __attribute__((ext_vector_type(8)));
typedef unsigned kv_u8x __attribute__((ext_vector_type(8)));

__device__ __forceinline__ kv_h16 kv_lds16(const _Float16* p) {
    const kv_h8 lo = *(const kv_h8*) p, hi = *(const kv_h8*) (p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
__device__ __forceinline__ kv_f8 kv_wmma(kv_h16 a, kv_h16 b, kv_f8 c) {
#if defined(__gfx11__) || defined(__GFX11__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b;
    return c;   // (never launched off gfx11: the engine only selects the WMMA attention there)
#endif
}

/// 16 offset-binary code bytes (memory order) -> 16 exact f16 values: c = u - 128, or (OFS) 1024 + u = c + 1152 (the
/// caller subtracts 1152 x the sum of the other operand once per row instead of 8 packed subtracts per operand).
template <bool OFS = false>
__device__ __forceinline__ kv_h16 kv8_dec16(uint4 r) {
    const unsigned w[4] = {r.x, r.y, r.z, r.w};
    kv_u8x h;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        h[2 * j] = (w[j] & 0xffu) | ((w[j] << 8) & 0xff0000u) | 0x64006400u;
        h[2 * j + 1] = ((w[j] >> 16) & 0xffu) | ((w[j] >> 8) & 0xff0000u) | 0x64006400u;
    }
    const kv_h16 x = __builtin_bit_cast(kv_h16, h);   // 1024 + u, exact
    if constexpr (OFS) return x;
    return x - (kv_h16) ((_Float16) 1152.0f);          // u - 128, exact
}

/// The Flash-Next kv_q8 rule for one value against its group's max |x|: the stored scale bits and the offset code.
__device__ __forceinline__ uint8_t kv8_code(float x, float amax, uint16_t* sbits_out) {
    const __half sh = __float2half(amax / 127.0f);
    const float sf = __half2float(sh);
    int q = 0;
    if (sf > 0.0f) {
        q = __float2int_rn(x / sf);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
    }
    *sbits_out = __half_as_ushort(sh);
    return (uint8_t) (q + 128);
}

/// Quantize cells [base, base + n) (base = *pos0 + c0, or c0 without pos0) of one layer's f16 K cache and vt copy into
/// the int8 copy.  grid (16-cell groups + 1, kv heads), block 256 (thread = dim).
__global__ void __launch_bounds__(256) k_kv8_quant(const __half* __restrict__ kc, const __half* __restrict__ vt, int ctx,
                                                   int vt_ctx, uint8_t* __restrict__ kq, uint16_t* __restrict__ ks,
                                                   uint8_t* __restrict__ vq, uint16_t* __restrict__ vs,
                                                   const int32_t* pos0, int c0, int n) {
    __shared__ float red[2][8][16];   // [K / V][32-lane segment][cell]
    const int base = (pos0 ? *pos0 : 0) + c0, end = base + n;
    const int kvh = blockIdx.y, d = threadIdx.x, seg = d >> 5, g = d / kGroup;
    const int cell0 = ((base >> 4) + (int) blockIdx.x) * 16;
    if (n <= 0 || cell0 >= end) return;   // block-uniform
    float kx[16], vx[16];
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        const int cell = cell0 + c;
        const bool in = cell >= base && cell < end && cell < ctx && cell < vt_ctx;
        kx[c] = in ? __half2float(kc[((size_t) kvh * ctx + cell) * kHD + d]) : 0.f;
        vx[c] = in ? __half2float(vt[vt_index(kvh, cell, d, vt_ctx)]) : 0.f;
        float a = fabsf(kx[c]), b = fabsf(vx[c]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a = fmaxf(a, __shfl_xor(a, o, 32));
            b = fmaxf(b, __shfl_xor(b, o, 32));
        }
        if ((d & 31) == 0) {
            red[0][seg][c] = a;
            red[1][seg][c] = b;
        }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        const int cell = cell0 + c;
        if (!(cell >= base && cell < end && cell < ctx && cell < vt_ctx)) continue;
        const float ak = fmaxf(red[0][2 * g][c], red[0][2 * g + 1][c]);
        const float av = fmaxf(red[1][2 * g][c], red[1][2 * g + 1][c]);
        uint16_t sk, sv;
        kq[((size_t) kvh * ctx + cell) * kHD + d] = kv8_code(kx[c], ak, &sk);
        vq[vt_index(kvh, cell, d, vt_ctx)] = kv8_code(vx[c], av, &sv);
        if ((d % kGroup) == 0) {
            ks[((size_t) kvh * ctx + cell) * kGroups + g] = sk;
            vs[((size_t) kvh * kGroups + g) * vt_ctx + cell] = sv;
        }
    }
}

/// k_attn_wmma (src/qwen35/q36_kernels.cu) over the int8 copy: same grid, chunks, tiles, masking, own-tile rule and
/// partial outputs (k_attn_merge merges them).  grid (n_chunk * ngrp, HKV), block 128; head dim 256, chunk % 32 == 0.
#ifndef KV8_WAVES
#define KV8_WAVES 8   // <= 192 VGPRs: 8 waves per SIMD like the f16 kernel (unconstrained it takes 193 -> 7 waves)
#endif
/// OFS (TALUS_Q27_KV8_OFS=1, the A/B variant): the operands keep the +1152 offset (no packed subtracts); S subtracts
/// 1152 x the row's group sum of Q (f16-rounded, summed in f32 in dim order) per group, P'V subtracts 1152 x the row's sum
/// of P' over the tile.  Still per-row arithmetic (decode row == verify row), but not bitwise the OFS=false variant.
template <int RT, bool OFS>
__global__ void __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(KV8_WAVES))) k_attn_wmma_kv8(const float* __restrict__ q, const uint8_t* __restrict__ kq,
                                                       const uint16_t* __restrict__ ks, const uint8_t* __restrict__ vq,
                                                       const uint16_t* __restrict__ vs, const int32_t* pos0, int T,
                                                       int H, int HKV, int ctx, int vt_ctx, float scale, int chunk,
                                                       int n_chunk, float* __restrict__ pacc, float* __restrict__ pm,
                                                       float* __restrict__ pl, int r0 = 0, int ngrp = 1) {
    constexpr int HD = kHD, QP = HD + 8, PP = kAttTile + 8, MR = 16 * RT;
    __shared__ _Float16 Q16[MR * QP];
    __shared__ _Float16 P16[MR * PP];
    __shared__ float Sp[MR][kAttTile + 1];
    __shared__ float mxw[2][MR], smw[2][MR], m_run[MR], l_run[MR], alf[MR];
    __shared__ float qofs[OFS ? MR : 1][kGroups], pofs[OFS ? 4 : 1][OFS ? MR : 1];
    const int kvh = blockIdx.y, c = (int) blockIdx.x / ngrp, tid = threadIdx.x;
    r0 += ((int) blockIdx.x % ngrp) * MR;
    const int w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int G = H / HKV, R = min(T * G - r0, MR);
    if (R <= 0) return;
    const int p0 = *pos0, nmax = p0 + T;
    int cbeg, clen;
    att_chunk_range(c, chunk, cbeg, clen);
    if (cbeg >= nmax) return;
    const int cend = min(cbeg + clen, nmax);
    for (int i = tid; i < MR * HD; i += 128) {
        const int r = i / HD, d = i % HD;
        float v = 0.f;
        if (r < R) v = q[((size_t) ((r + r0) / G) * H + kvh * G + (r + r0) % G) * HD + d];
        Q16[r * QP + d] = (_Float16) v;
    }
    if (tid < MR) {
        m_run[tid] = -INFINITY;
        l_run[tid] = 0.f;
    }
    kv_f8 oacc[RT][4];
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int dt = 0; dt < 4; ++dt)
#pragma unroll
            for (int i = 0; i < 8; ++i) oacc[rt][dt][i] = 0.f;
    __syncthreads();
    if constexpr (OFS) {   // 1152 x each row's per-group sum of its f16 Q, in dim order
        if (tid < MR * kGroups) {
            const int r = tid / kGroups, g = tid % kGroups;
            float a = 0.f;
            for (int d = 0; d < kGroup; ++d) a += (float) Q16[r * QP + g * kGroup + d];
            qofs[r][g] = 1152.f * a;
        }
        __syncthreads();
    }
    const uint8_t* kbase = kq + (size_t) kvh * ctx * HD;
    const _Float16* ksb = (const _Float16*) ks + (size_t) kvh * ctx * kGroups;
    const uint8_t* vbase = vq + (size_t) kvh * vt_ctx * HD;
    const _Float16* vsb = (const _Float16*) vs + ((size_t) kvh * kGroups + (tid >> 5)) * vt_ctx;   // wave w: V group w
    const int ct = w & 1, kh = w >> 1;
    auto tile = [&](int base, auto tailc) {
        // S = Q K^T: this wave's 16 cells x half the head dim (2 scale groups, one accumulator each)
        kv_f8 sa0[RT], sa1[RT];
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int i = 0; i < 8; ++i) sa0[rt][i] = sa1[rt][i] = 0.f;
        const int cell = base + ct * 16 + l16;
        const bool cv = cell < nmax;
        const int cs_ = cv ? cell : 0;
        const uint8_t* kp = kbase + (size_t) cs_ * HD;
        auto kstep = [&](int kst, kv_f8* acc) {
            kv_h16 b = kv8_dec16<OFS>(*(const uint4*) (kp + kst * 16));
            if (!cv) b = (kv_h16) {};
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) acc[rt] = kv_wmma(kv_lds16(&Q16[(rt * 16 + l16) * QP + kst * 16]), b, acc[rt]);
        };
#pragma unroll   // (unroll 2 costs registers: 205 VGPRs)
        for (int k8 = 0; k8 < 4; ++k8) kstep(kh * 8 + k8, sa0);   // scale group 2 kh
#pragma unroll
        for (int k8 = 4; k8 < 8; ++k8) kstep(kh * 8 + k8, sa1);   // scale group 2 kh + 1
        const float ks0 = (float) ksb[(size_t) cs_ * kGroups + kh * 2];
        const float ks1 = (float) ksb[(size_t) cs_ * kGroups + kh * 2 + 1];
        kv_f8 sacc[RT];
#pragma unroll
        for (int rt = 0; rt < RT; ++rt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float s0 = sa0[rt][i], s1 = sa1[rt][i];
                if constexpr (OFS) {   // (a masked cell's operand is zero: no offset in its sum)
                    const int r = rt * 16 + 2 * i + hi;
                    if (cv) {
                        s0 -= qofs[r][kh * 2];
                        s1 -= qofs[r][kh * 2 + 1];
                    }
                }
                sacc[rt][i] = fmaf(s1, ks1, s0 * ks0);
            }
        if (kh == 1) {
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) Sp[rt * 16 + 2 * i + hi][ct * 16 + l16] = sacc[rt][i];
        }
        __syncthreads();
        if (kh == 0) {   // the scores (masked per row: token t sees the cells < p0 + t + 1), the tile's row maxima
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    const bool ok = r < R && cell < p0 + (r + r0) / G + 1;
                    const float s = ok ? (sacc[rt][i] + Sp[r][ct * 16 + l16]) * scale : -INFINITY;
                    sacc[rt][i] = s;
                    float m = s;
#pragma unroll
                    for (int o = 1; o < 16; o <<= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
                    if (l16 == 0) mxw[ct][r] = m;
                }
        }
        __syncthreads();
        if (kh == 0) {   // P = exp(S - the new running max) as FP16, the tile's row sums
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    const float mn = fmaxf(m_run[r], fmaxf(mxw[0][r], mxw[1][r]));
                    const float p = sacc[rt][i] == -INFINITY ? 0.f : expf(sacc[rt][i] - mn);
                    P16[r * PP + ct * 16 + l16] = (_Float16) p;
                    float s = p;
#pragma unroll
                    for (int o = 1; o < 16; o <<= 1) s += __shfl_xor(s, o, 32);
                    if (l16 == 0) smw[ct][r] = s;
                }
        }
        __syncthreads();
        if (tid < MR) {
            const float mo = m_run[tid], mn = fmaxf(mo, fmaxf(mxw[0][tid], mxw[1][tid]));
            const float a = mn == -INFINITY ? 1.f : expf(mo - mn);
            l_run[tid] = l_run[tid] * a + (smw[0][tid] + smw[1][tid]);
            m_run[tid] = mn;
            alf[tid] = a;
        }
        __syncthreads();
        // O = O * alpha + P' V over this wave's 64 dims (= V scale group w): P' = fp16(P * the cell's V scale), once per
        // tile; the codes are exact.  The own-tile rule of k_attn_wmma (see there) is kept: the tile holding a row's
        // position adds that row's scalar sum over its cells only, in cell order, as P * (code * scale) in f32.
        constexpr bool tail = decltype(tailc)::value;
        kv_h16 pa[RT][2];
#pragma unroll
        for (int cs = 0; cs < 2; ++cs) {
            const kv_h16 sv = kv_lds16(vsb + base + cs * 16);   // 32-byte aligned global load (base % 32 == 0)
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) pa[rt][cs] = kv_lds16(&P16[(rt * 16 + l16) * PP + cs * 16]) * sv;
        }
        if constexpr (OFS) {   // 1152 x each row's sum of P' over the tile (this wave's V group), via LDS to the C layout
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
                float a = 0.f;
#pragma unroll
                for (int cs = 0; cs < 2; ++cs)
#pragma unroll
                    for (int j = 0; j < 16; ++j) a += (float) pa[rt][cs][j];
                if (hi == 0) pofs[w][rt * 16 + l16] = 1152.f * a;
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        }
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
            const int d = w * 64 + dt * 16 + l16;
            const uint8_t* vd = vbase + (size_t) d * 16;   // + (cell / 16) * HD * 16 + cell % 16
            kv_h16 b[2];
#pragma unroll
            for (int cs = 0; cs < 2; ++cs) b[cs] = kv8_dec16<OFS>(*(const uint4*) (vd + (size_t) ((base >> 4) + cs) * HD * 16));
            kv_f8 dacc[RT];
#pragma unroll
            for (int rt = 0; rt < RT; ++rt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) dacc[rt][i] = 0.f;
#pragma unroll
                for (int cs = 0; cs < 2; ++cs) dacc[rt] = kv_wmma(pa[rt][cs], b[cs], dacc[rt]);
                if constexpr (OFS) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) dacc[rt][i] -= pofs[w][rt * 16 + 2 * i + hi];
                }
            }
            if constexpr (tail) {
                float own[RT][8];
#pragma unroll
                for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) own[rt][i] = 0.f;
#pragma unroll 1
                for (int cc = 0; cc < kAttTile; ++cc) {
                    const int cl = base + cc;
                    const float v = (float) ((int) vd[(size_t) (cl >> 4) * HD * 16 + (cl & 15)] - 128) * (float) vsb[cl];
#pragma unroll
                    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            const int r = rt * 16 + 2 * i + hi;
                            const int pr = p0 + (r + r0) / G;
                            if (r < R && pr >= cl && pr < base + kAttTile - 1)
                                own[rt][i] = fmaf((float) P16[r * PP + cc], v, own[rt][i]);
                        }
                }
#pragma unroll
                for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const int r = rt * 16 + 2 * i + hi;
                        const int pr = p0 + (r + r0) / G;
                        if (r < R && pr < base + kAttTile - 1) dacc[rt][i] = own[rt][i];
                    }
            }
#pragma unroll
            for (int rt = 0; rt < RT; ++rt)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int r = rt * 16 + 2 * i + hi;
                    if (r < R && p0 + (r + r0) / G >= base) oacc[rt][dt][i] = fmaf(oacc[rt][dt][i], alf[r], dacc[rt][i]);
                }
        }
        __syncthreads();
    };
    const int tail_start = max(cbeg, ((p0 + 1) / kAttTile) * kAttTile);
    for (int base = cbeg; base < min(cend, tail_start); base += kAttTile) tile(base, std::false_type());
    for (int base = tail_start; base < cend; base += kAttTile) tile(base, std::true_type());
#pragma unroll
    for (int rt = 0; rt < RT; ++rt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = rt * 16 + 2 * i + hi;
            if (r >= R) continue;
            const size_t row = (size_t) ((r + r0) / G) * H + kvh * G + (r + r0) % G;
#pragma unroll
            for (int dt = 0; dt < 4; ++dt) pacc[(row * n_chunk + c) * HD + w * 64 + dt * 16 + l16] = oacc[rt][dt][i];
        }
    if (tid < R) {
        const size_t row = (size_t) ((tid + r0) / G) * H + kvh * G + (tid + r0) % G;
        pm[row * n_chunk + c] = m_run[tid];
        pl[row * n_chunk + c] = l_run[tid];
    }
}

int kv8_status() {
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? 0 : (int) e;
}

}   // namespace

/// Format version of the int8 copy (bump on any layout / rounding change).
extern "C" int talus_q36_kv8_version() { return 1; }

/// Quantize cells [base, base + n) of one attention layer into its int8 copy; base = *pos0 + c0 (pos0 device memory,
/// graph-capturable) or c0 (pos0 null).  kc / vt: the layer's f16 K cache and vt copy.  0 = ok, -1 = bad arguments,
/// else the HIP error.
extern "C" int talus_q36_kv8_quant(const void* kc, const void* vt, int ctx, int vt_ctx, int hkv, int hd, void* kq,
                                   void* ks, void* vq, void* vs, const int32_t* pos0, int c0, int n, void* stream) {
    if (hd != kHD || hkv <= 0 || ctx <= 0 || vt_ctx <= 0 || (vt_ctx % 16) != 0 || n < 0 || c0 < 0) return -1;
    if (n == 0) return 0;
    const unsigned gx = pos0 ? (unsigned) ((n + 15) / 16 + 1) : (unsigned) (((c0 + n - 1) >> 4) - (c0 >> 4) + 1);
    k_kv8_quant<<<dim3(gx, (unsigned) hkv), 256, 0, (cudaStream_t) stream>>>(
        (const __half*) kc, (const __half*) vt, ctx, vt_ctx, (uint8_t*) kq, (uint16_t*) ks, (uint8_t*) vq,
        (uint16_t*) vs, pos0, c0, n);
    return kv8_status();
}

/// Decode / verify attention over the int8 copy: k_attn_wmma's partials (merge with tk_q36_attn_merge).  T rows of
/// G = H / HKV query heads per KV head run as row groups of 16 (one 1-tile block per group and chunk, adjacent per
/// chunk), as the f16 kernel's default (TALUS_ATTN_RT1G) does.
/// variant 0: exact codes in the operands; 1: the OFS variant (offset kept, per-row corrections; see the kernel).
extern "C" int talus_q36_attn_wmma_kv8(int n_chunk, int hkv, const float* q, const void* kq, const void* ks,
                                       const void* vq, const void* vs, const int32_t* pos0, int T, int H, int ctx,
                                       int vt_ctx, float sc, int chunk, float* pacc, float* pm, float* pl,
                                       void* stream, int variant) {
    if (hkv <= 0 || H % hkv != 0 || T <= 0 || n_chunk <= 0 || variant < 0 || variant > 1) return -1;
    const int ng = (T * (H / hkv) + 15) / 16;
    const dim3 grid((unsigned) (n_chunk * ng), (unsigned) hkv);
    const uint8_t *kq8 = (const uint8_t*) kq, *vq8 = (const uint8_t*) vq;
    const uint16_t *ks16 = (const uint16_t*) ks, *vs16 = (const uint16_t*) vs;
    if (variant == 1)
        k_attn_wmma_kv8<1, true><<<grid, 128, 0, (cudaStream_t) stream>>>(q, kq8, ks16, vq8, vs16, pos0, T, H, hkv, ctx,
                                                                           vt_ctx, sc, chunk, n_chunk, pacc, pm, pl, 0, ng);
    else
        k_attn_wmma_kv8<1, false><<<grid, 128, 0, (cudaStream_t) stream>>>(q, kq8, ks16, vq8, vs16, pos0, T, H, hkv, ctx,
                                                                            vt_ctx, sc, chunk, n_chunk, pacc, pm, pl, 0, ng);
    return kv8_status();
}
