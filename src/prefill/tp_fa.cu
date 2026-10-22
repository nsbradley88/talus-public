// src/prefill/tp_fa.cu - see include/strata/prefill/tp_fa.hpp.
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "strata/prefill/tp_fa.hpp"
#include "strata/prefill/tp_tune.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define TP_FA_WMMA 1
#endif

namespace talus::prefill {

const TpTune& tp_tune_current() {
    static const TpTune* t = [] {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return &kTpTuneDefault;
        return &tp_tune(p.gcnArchName);
    }();
    return *t;
}

namespace {

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef float f8 __attribute__((ext_vector_type(8)));

constexpr int kD = 256, kBQ = 64, kBK = 32;

__device__ __forceinline__ f8 wmma(h16 a, h16 b, f8 c) {
#if defined(TP_FA_WMMA)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b;
    return c;
#endif
}
__device__ __forceinline__ h16 cat(h8 lo, h8 hi) {
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }   // = q36_engine's

// a row of 256 halfs in LDS as 32 units of 8, unit u stored at u ^ (row & 7) (the 16 rows a fragment reads spread
// over the banks)
__device__ __forceinline__ int swz(int row, int unit) { return row * kD + ((unit ^ (row & 7)) << 3); }

/// grid (ceil(T G / 64), HKV), block 128 (4 waves x 16 query rows).  Column c of KV head kvh = query row
/// (t = c / G, head kvh G + c % G).
template <bool kKvLds>
__global__ void __launch_bounds__(128) k_fa_prefill(const float* __restrict__ q, const _Float16* __restrict__ kc,
                                                    const _Float16* __restrict__ vt, const float* __restrict__ gate,
                                                    float* __restrict__ out, int T, int H, int HKV, int64_t kv_stride,
                                                    int vt_cells, int p0, float scale) {
    __shared__ __attribute__((aligned(16))) _Float16 Qs[kBQ * kD];
    __shared__ __attribute__((aligned(16))) _Float16 Ks[kKvLds ? kBK * kD : 8];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[kKvLds ? kBK * kD : 8];
    const int G = H / HKV, kvh = blockIdx.y, c0 = blockIdx.x * kBQ, ncol = T * G;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int kv_end = p0 + T;   // cells at or past it are read as zeros
    (void) HKV;

    // Q (times the scale) as FP16: thread = (row tid / 2, half of the head dim)
    {
        const int r = tid >> 1, dh = (tid & 1) * 128, c = c0 + r;
        const bool ok = c < ncol;
        const float4* src = (const float4*) (q + ((size_t) (ok ? c / G : 0) * H + kvh * G + (ok ? c % G : 0)) * kD + dh);
#pragma unroll 4
        for (int u = 0; u < 16; ++u) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
            if (ok) {
                a = src[2 * u];
                b = src[2 * u + 1];
            }
            h8 v = {(_Float16) (a.x * scale), (_Float16) (a.y * scale), (_Float16) (a.z * scale), (_Float16) (a.w * scale),
                    (_Float16) (b.x * scale), (_Float16) (b.y * scale), (_Float16) (b.z * scale), (_Float16) (b.w * scale)};
            *(h8*) &Qs[swz(r, (dh >> 3) + u)] = v;
        }
    }

    const int col = c0 + w * 16 + l16;
    const bool colok = col < ncol;
    const int plim = colok ? p0 + col / G : -1;   // this lane's query sees the cells <= plim
    const int t_lo = c0 / G, t_hi = min(T - 1, (c0 + kBQ - 1) / G);
    const int nkeys = p0 + t_hi + 1;
    const int full_end = min(nkeys, ((p0 + t_lo + 1) / kBK) * kBK);   // tiles below every column's position

    f8 o[16];
#pragma unroll
    for (int dt = 0; dt < 16; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[dt][i] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;

    const _Float16* kbase = kc + (size_t) kvh * kv_stride * kD;
    const _Float16* vbase = vt + (size_t) kvh * vt_cells * kD;
    const int qrow = w * 16 + l16;
    const h8 z8 = {};

    auto tile = [&](int base, auto maskc) {
        constexpr bool kMask = decltype(maskc)::value;
        if constexpr (kKvLds) {
            __syncthreads();   // the previous tile's reads (and, first time, Q's writes) are done
#pragma unroll 2
            for (int j = 0; j < 8; ++j) {   // K: 32 cells x 32 units
                const int idx = tid + 128 * j, key = idx >> 5, u = idx & 31, cell = base + key;
                h8 v = z8;
                if (!kMask || cell < kv_end) v = *(const h8*) (kbase + (size_t) cell * kD + u * 8);
                *(h8*) &Ks[swz(key, u)] = v;
            }
#pragma unroll 2
            for (int j = 0; j < 8; ++j) {   // V^T: two blocks of 16 cells, [256][16] each, contiguous
                const int idx = tid + 128 * j, blk = idx >> 9, rem = idx & 511;
                const int cell0 = base + blk * 16 + (rem & 1) * 8;
                h8 v = z8;
                if (!kMask || cell0 < kv_end) {
                    v = *(const h8*) (vbase + ((size_t) (base / 16 + blk) * kD) * 16 + rem * 8);
                    if (kMask)
#pragma unroll
                        for (int e = 0; e < 8; ++e)
                            if (cell0 + e >= kv_end) v[e] = (_Float16) 0.f;
                }
                *(h8*) &Vs[idx * 8] = v;
            }
            __syncthreads();
        }
        // S^T = K Q^T: 2 tiles of 16 cells x this wave's 16 queries
        f8 s[2];
#pragma unroll
        for (int kt = 0; kt < 2; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) s[kt][i] = 0.f;
#pragma unroll 4
        for (int ds = 0; ds < 16; ++ds) {
            const h16 b = cat(*(const h8*) &Qs[swz(qrow, 2 * ds)], *(const h8*) &Qs[swz(qrow, 2 * ds + 1)]);
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                const int key = kt * 16 + l16;
                h16 a;
                if constexpr (kKvLds) {
                    a = cat(*(const h8*) &Ks[swz(key, 2 * ds)], *(const h8*) &Ks[swz(key, 2 * ds + 1)]);
                } else {
                    const int cell = base + key;
                    a = *(const h16*) (kbase + (size_t) (kMask && cell >= kv_end ? 0 : cell) * kD + ds * 16);
                    if (kMask && cell >= kv_end) a = (h16) {};
                }
                s[kt] = wmma(a, b, s[kt]);
            }
        }
        // online softmax: this lane's query column, cells kt * 16 + 2 i + hi (the other half in lane ^ 16)
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < 2; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = s[kt][i];
                if (kMask && base + kt * 16 + 2 * i + hi > plim) v = -INFINITY;
                s[kt][i] = v;
                mx = fmaxf(mx, v);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m_run, mx);
        const float alpha = mn == -INFINITY ? 1.f : expf(m_run - mn);
        float ls = 0.f;
        h16 pb[2];
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            _Float16 ph[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = s[kt][i] == -INFINITY ? 0.f : expf(s[kt][i] - mn);
                ls += p;
                ph[i] = (_Float16) p;
            }
            // P^T as the B operand: lane (query l16) needs its 16 cells; the odd / even half is in lane ^ 16
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                union { _Float16 h[2]; int i; } own, par;
                own.h[0] = ph[2 * m];
                own.h[1] = ph[2 * m + 1];
                par.i = __shfl_xor(own.i, 16, 32);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int i = 2 * m + e;
                    pb[kt][2 * i] = hi ? par.h[e] : own.h[e];
                    pb[kt][2 * i + 1] = hi ? own.h[e] : par.h[e];
                }
            }
        }
        ls += __shfl_xor(ls, 16, 32);
        l_run = l_run * alpha + ls;
        m_run = mn;
        // O^T = O^T alpha + V^T P^T
#pragma unroll
        for (int dt = 0; dt < 16; ++dt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) o[dt][i] *= alpha;
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                h16 a;
                if constexpr (kKvLds) {
                    a = *(const h16*) &Vs[(kt * kD + dt * 16 + l16) * 16];
                } else {
                    a = *(const h16*) (vbase + ((size_t) (base / 16 + kt) * kD + dt * 16 + l16) * 16);
                    if (kMask)
#pragma unroll
                        for (int e = 0; e < 16; ++e)
                            if (base + kt * 16 + e >= kv_end) a[e] = (_Float16) 0.f;
                }
                o[dt] = wmma(a, pb[kt], o[dt]);
            }
        }
    };
    if constexpr (!kKvLds) __syncthreads();   // Q
    int base = 0;
    for (; base < full_end; base += kBK) tile(base, std::false_type());
    for (; base < nkeys; base += kBK) tile(base, std::true_type());

    if (!colok) return;
    const int t = col / G, h = kvh * G + col % G;
    const float inv = 1.0f / l_run;
    float* dst = out + ((size_t) t * H + h) * kD;
    const float* gp = gate ? gate + ((size_t) t * H + h) * kD : nullptr;
#pragma unroll
    for (int dt = 0; dt < 16; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = dt * 16 + 2 * i + hi;
            float v = o[dt][i] * inv;
            if (gp) v *= sigm(gp[d]);
            dst[d] = v;
        }
}

/// Variant 3: grid (ceil(T G / 64), HKV), block 256 = 4 column groups of 16 queries x 2 halves of the head dim.  Wave
/// w = 2 cg + hf keeps Q (its 16 queries x dims hf 128 .. + 128) in registers and O^T for the same dims, so a K / V
/// fragment read from LDS feeds one WMMA of useful work (variant 1: Q re-read every tile).  The scores' two halves are
/// summed through LDS (float addition commutes: both waves hold the same bits), the softmax runs in both waves.  The
/// next tile's K / V are fetched into registers while this tile computes.
__global__ void __launch_bounds__(256) k_fa_prefill_pair(const float* __restrict__ q, const _Float16* __restrict__ kc,
                                                         const _Float16* __restrict__ vt, const float* __restrict__ gate,
                                                         float* __restrict__ out, int T, int H, int64_t kv_stride,
                                                         int vt_cells, int p0, float scale, int HKV) {
    __shared__ __attribute__((aligned(16))) _Float16 Ks[kBK * kD];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[kBK * kD];
    __shared__ float Sx[8][2][8][32];
    const int G = H / HKV, kvh = blockIdx.y, c0 = blockIdx.x * kBQ, ncol = T * G;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int cg = w >> 1, hf = w & 1;
    const unsigned sel_lo = hi ? 0x01000504u : 0x05040100u, sel_hi = hi ? 0x03020706u : 0x07060302u;
    const int kv_end = p0 + T;
    const int col = c0 + cg * 16 + l16;
    const bool colok = col < ncol;
    const int plim = colok ? p0 + col / G : -1;
    const int t_lo = c0 / G, t_hi = min(T - 1, (c0 + kBQ - 1) / G);
    const int nkeys = p0 + t_hi + 1;
    const int full_end = min(nkeys, ((p0 + t_lo + 1) / kBK) * kBK);

    h16 qf[8];
    scale *= 1.44269504088896340736f;   // scores in log2 units: P = 2^(S - max)
    {
        const float* src = q + ((size_t) (colok ? col / G : 0) * H + kvh * G + (colok ? col % G : 0)) * kD + hf * 128;
#pragma unroll
        for (int ds = 0; ds < 8; ++ds) {
#pragma unroll
            for (int e4 = 0; e4 < 4; ++e4) {
                float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
                if (colok) a = *(const float4*) (src + ds * 16 + e4 * 4);
                qf[ds][e4 * 4 + 0] = (_Float16) (a.x * scale);
                qf[ds][e4 * 4 + 1] = (_Float16) (a.y * scale);
                qf[ds][e4 * 4 + 2] = (_Float16) (a.z * scale);
                qf[ds][e4 * 4 + 3] = (_Float16) (a.w * scale);
            }
        }
    }
    f8 o[8];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[dt][i] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;

    const _Float16* kbase = kc + (size_t) kvh * kv_stride * kD;
    const _Float16* vbase = vt + (size_t) kvh * vt_cells * kD;
    const h8 z8 = {};
    h8 kr[4], vr[4];
    auto gk = [&](int base) {   // the tile's K into registers (cells at or past kv_end as zeros)
        if (base + kBK <= kv_end) {   // (uniform) the whole tile is written
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = tid + 256 * j;
                kr[j] = *(const h8*) (kbase + (size_t) (base + (idx >> 5)) * kD + (idx & 31) * 8);
            }
            return;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = tid + 256 * j, key = idx >> 5, u = idx & 31, cell = base + key;
            kr[j] = cell < kv_end ? *(const h8*) (kbase + (size_t) cell * kD + u * 8) : z8;
        }
    };
    auto gv = [&](int base) {   // the tile's V^T blocks into registers
        if (base + kBK <= kv_end) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = tid + 256 * j;
                vr[j] = *(const h8*) (vbase + ((size_t) (base / 16 + (idx >> 9)) * kD) * 16 + (idx & 511) * 8);
            }
            return;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = tid + 256 * j, blk = idx >> 9, rem = idx & 511;
            const int cell0 = base + blk * 16 + (rem & 1) * 8;
            h8 v = z8;
            if (cell0 < kv_end) {
                v = *(const h8*) (vbase + ((size_t) (base / 16 + blk) * kD) * 16 + rem * 8);
                if (cell0 + 8 > kv_end)
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        if (cell0 + e >= kv_end) v[e] = (_Float16) 0.f;
            }
            vr[j] = v;
        }
    };
    auto sk = [&]() {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = tid + 256 * j;
            *(h8*) &Ks[swz(idx >> 5, idx & 31)] = kr[j];
        }
    };
    auto sv = [&]() {
#pragma unroll
        for (int j = 0; j < 4; ++j) *(h8*) &Vs[(tid + 256 * j) * 8] = vr[j];
    };
    auto tile = [&](int base, auto maskc) {
        constexpr bool kMask = decltype(maskc)::value;
        const bool has_next = base + kBK < nkeys;
        if (has_next) gk(base + kBK);
        f8 s[2];
#pragma unroll
        for (int kt = 0; kt < 2; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) s[kt][i] = 0.f;
#pragma unroll
        for (int ds = 0; ds < 8; ++ds) {
            if (ds % 2 == 0) asm volatile("" ::: "memory");   // keep the compiler from hoisting all 16 fragment loads
#pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                const int key = kt * 16 + l16, u = 2 * (hf * 8 + ds);
                const h16 a = cat(*(const h8*) &Ks[swz(key, u)], *(const h8*) &Ks[swz(key, u + 1)]);
                s[kt] = wmma(a, qf[ds], s[kt]);
            }
        }
#pragma unroll
        for (int kt = 0; kt < 2; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) Sx[w][kt][i][lane] = s[kt][i];
        __syncthreads();   // every wave's scores done: Ks is free, Sx complete
        if (has_next) {
            sk();
            gv(base + kBK);
        }
        float mx = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < 2; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = s[kt][i] + Sx[w ^ 1][kt][i][lane];
                if (kMask && base + kt * 16 + 2 * i + hi > plim) v = -INFINITY;
                s[kt][i] = v;
                mx = fmaxf(mx, v);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m_run, mx);
        const float mb = mn == -INFINITY ? 0.f : mn;   // (a column with no cell yet: every p is 2^-inf = 0)
        const float alpha = __builtin_amdgcn_exp2f(m_run - mb);
        const bool rescale = __any(alpha != 1.f);   // wave-uniform: no lane's maximum moved -> O unchanged
        float ls = 0.f;
        union PK { _Float16 h[2]; int i; } pk[2][4];   // this lane's P (its 8 cells per tile), packed in pairs
#pragma unroll
        for (int kt = 0; kt < 2; ++kt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = __builtin_amdgcn_exp2f(s[kt][i] - mb);   // v_exp_f32: 2^-inf = 0
                ls += p;
                pk[kt][i >> 1].h[i & 1] = (_Float16) p;
            }
        ls += __shfl_xor(ls, 16, 32);
        l_run = l_run * alpha + ls;
        m_run = mn;
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            // P^T as the B operand: lane (query l16) needs its 16 cells, the odd / even ones are in lane ^ 16: dword
            // 2m + e = (cell 4m + 2e, cell 4m + 2e + 1) = (own e, partner e) in lanes < 16, (partner e, own e) above
            union { h16 v; unsigned u[8]; } pb;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const unsigned own = (unsigned) pk[kt][m].i;
                const unsigned par = (unsigned) __shfl_xor(pk[kt][m].i, 16, 32);
                // v_perm_b32(a, b, sel): bytes 0-3 = b, 4-7 = a
                pb.u[2 * m] = __builtin_amdgcn_perm(par, own, sel_lo);
                pb.u[2 * m + 1] = __builtin_amdgcn_perm(par, own, sel_hi);
            }
#pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                if (dt % 2 == 0) asm volatile("" ::: "memory");
                if (kt == 0 && rescale)
#pragma unroll
                    for (int i = 0; i < 8; ++i) o[dt][i] *= alpha;
                const h16 a = *(const h16*) &Vs[(kt * kD + hf * 128 + dt * 16 + l16) * 16];
                o[dt] = wmma(a, pb.v, o[dt]);
            }
        }
        __syncthreads();   // Vs and Sx reads done; the next tile's Ks visible
        if (has_next) sv();   // (visible to the next tile's P V after its score barrier)
    };
    gk(0);
    gv(0);
    sk();
    sv();
    __syncthreads();
    int base = 0;
    for (; base < full_end; base += kBK) tile(base, std::false_type());
    for (; base < nkeys; base += kBK) tile(base, std::true_type());

    if (!colok) return;
    const int t = col / G, h = kvh * G + col % G;
    const float inv = 1.0f / l_run;
    float* dst = out + ((size_t) t * H + h) * kD + hf * 128;
    const float* gp = gate ? gate + ((size_t) t * H + h) * kD + hf * 128 : nullptr;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = dt * 16 + 2 * i + hi;
            float v = o[dt][i] * inv;
            if (gp) v *= sigm(gp[d]);
            dst[d] = v;
        }
}

}  // namespace

bool tp_gfx11_wmma() {
    static const bool ok = [] {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return false;
        const char* g = p.gcnArchName;
        for (const char* a : {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"}) {
            const size_t n = std::strlen(a);
            if (std::strncmp(g, a, n) == 0 && (g[n] == '\0' || g[n] == ':')) return true;
        }
        return false;
    }();
    return ok;
}

bool fa_prefill_ok(int head_dim, int H, int HKV) {
    return head_dim == kD && HKV > 0 && H % HKV == 0 && tp_gfx11_wmma();
}

void fa_prefill(const float* q, const void* k, const void* vt, const float* gate, float* out, int T, int H, int HKV,
                int D, int64_t kv_stride, int64_t vt_cells, int64_t p0, float scale, int variant, void* stream) {
    if (!fa_prefill_ok(D, H, HKV)) {
        std::fprintf(stderr, "tp_fa: unsupported shape D %d H %d HKV %d\n", D, H, HKV);
        std::exit(1);
    }
    static const int env = [] { const char* e = std::getenv("TALUS_PF_FA_VARIANT"); return e ? std::atoi(e) : 0; }();
    if (variant == 0) variant = env ? env : tp_tune_current().fa_variant;
    const int G = H / HKV;
    const dim3 grid((unsigned) ((T * G + kBQ - 1) / kBQ), (unsigned) HKV);
    const hipStream_t s = (hipStream_t) stream;
    if (variant == 3)
        k_fa_prefill_pair<<<grid, 256, 0, s>>>(q, (const _Float16*) k, (const _Float16*) vt, gate, out, T, H, kv_stride,
                                               (int) vt_cells, (int) p0, scale, HKV);
    else if (variant == 2)
        k_fa_prefill<false><<<grid, 128, 0, s>>>(q, (const _Float16*) k, (const _Float16*) vt, gate, out, T, H, HKV,
                                                  kv_stride, (int) vt_cells, (int) p0, scale);
    else
        k_fa_prefill<true><<<grid, 128, 0, s>>>(q, (const _Float16*) k, (const _Float16*) vt, gate, out, T, H, HKV,
                                                 kv_stride, (int) vt_cells, (int) p0, scale);
}

}  // namespace talus::prefill

extern "C" int talus_pf_fa_prefill(const float* q, const void* k, const void* vt, const float* gate, float* out, int T,
                                   int H, int HKV, int D, int64_t kv_stride, int64_t vt_cells, int64_t p0, float scale,
                                   void* stream) {
    if (!talus::prefill::fa_prefill_ok(D, H, HKV)) return -1;
    talus::prefill::fa_prefill(q, k, vt, gate, out, T, H, HKV, D, kv_stride, vt_cells, p0, scale, 0, stream);
    return 0;
}
