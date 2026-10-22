// src/kernels/cuda/fnp_qsa_tile_attn.cu - see include/strata/kernels/fnp_qsa_tile_attn.hpp (fnp_qsa_tile_attn).
#include "strata/kernels/fnp_qsa_tile_attn.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int HD = 256, G = 12, TQ = kFnpTileQ, ROWS = TQ * G, MT = ROWS / 16, NGRP = HD / 64;
static_assert(ROWS % 16 == 0, "the tile's rows are whole WMMA tiles");
constexpr int CH = 32;                  // union cells per chunk
constexpr int THREADS = 32 * MT * NGRP; // a wave per (row tile, 64-dim group): 12 waves
constexpr int KLD = HD + 16;            // staged K / V row stride (bytes)
constexpr int UT = 256;                 // union kernel threads

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__)
#define FNP_TA_GFX11 1
#else
#define FNP_TA_GFX11 0
#endif
typedef _Float16 th16 __attribute__((ext_vector_type(16)));
typedef float tf8 __attribute__((ext_vector_type(8)));
__device__ __forceinline__ tf8 twmma(th16 a, th16 b, tf8 c) {
#if FNP_TA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ th16 codes16(const uint8_t* p) {   // 16 int8 codes, exact in FP16
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    th16 h;
#pragma unroll
    for (int i = 0; i < 16; ++i) h[i] = (_Float16) (int) (int8_t) (w[i / 4] >> (8 * (i % 4)));
    return h;
}

// ---- 1. the union of a tile's selections: key = cell * 16 + (1 << query), lists ascending
__device__ __forceinline__ int lkey(const int32_t* ids, int i, int x) { return ids[x] * 16 + (1 << i); }
// merge path: the split of diagonal d between sorted A (la) and B (lb): returns how many of A come first
template <typename FA, typename FB>
__device__ __forceinline__ int merge_split(int d, int la, int lb, FA a, FB b) {
    int lo = d - lb > 0 ? d - lb : 0, hi = d < la ? d : la;
    while (lo < hi) {
        const int i = (lo + hi) >> 1;
        if (a(i) < b(d - 1 - i)) lo = i + 1; else hi = i;
    }
    return lo;
}
__global__ void __launch_bounds__(UT) union_kernel(const int32_t* __restrict__ ids, const int32_t* __restrict__ steps,
                                                   int64_t cap, int64_t n_q, int32_t* __restrict__ uni,
                                                   int32_t* __restrict__ ucount) {
    extern __shared__ int32_t sm[];   // ab [2 cap] | cd [2 cap] | scan [UT]
    const int64_t tile = blockIdx.x;
    const int t = threadIdx.x;
    int len[TQ];
    const int32_t* lst[TQ];
#pragma unroll
    for (int i = 0; i < TQ; ++i) {
        const int64_t qi = tile * TQ + i;
        len[i] = qi < n_q ? steps[qi * kStepCount + kStepWidth] : 0;
        lst[i] = ids + (qi < n_q ? qi : 0) * cap;
    }
    int32_t* ab = sm;
    int32_t* cd = sm + 2 * cap;
    int* scan = (int*) (sm + 4 * cap);
    // level 1: (0, 1) -> ab, (2, 3) -> cd
#pragma unroll
    for (int pr = 0; pr < 2; ++pr) {
        const int ia = 2 * pr, ib = 2 * pr + 1, la = len[ia], lb = len[ib], L = la + lb;
        const int32_t *A = lst[ia], *B = lst[ib];
        int32_t* out = pr ? cd : ab;
        const int seg = (L + UT - 1) / UT, d0 = min(L, t * seg), d1 = min(L, d0 + seg);
        int i = merge_split(d0, la, lb, [&](int x) { return lkey(A, ia, x); }, [&](int x) { return lkey(B, ib, x); });
        int j = d0 - i;
        for (int d = d0; d < d1; ++d) {
            const bool takea = j >= lb || (i < la && lkey(A, ia, i) < lkey(B, ib, j));
            out[d] = takea ? lkey(A, ia, i++) : lkey(B, ib, j++);
        }
    }
    __syncthreads();
    // level 2: ab + cd, merged twice - the heads (first entry of each cell) counted, scanned, then written with the
    // OR of their cell's query bits
    const int la = len[0] + len[1], lb = len[2] + len[3], L = la + lb;
    const int seg = (L + UT - 1) / UT, d0 = min(L, t * seg), d1 = min(L, d0 + seg);
    auto A = [&](int x) { return ab[x]; };
    auto B = [&](int x) { return cd[x]; };
    const int i0 = merge_split(d0, la, lb, A, B), j0 = d0 - i0;
    int prev = -1;   // the cell before d0
    if (d0 > 0) {
        const int pa = i0 > 0 ? ab[i0 - 1] : -1, pb = j0 > 0 ? cd[j0 - 1] : -1;
        prev = (pa > pb ? pa : pb) >> 4;
    }
    int heads = 0;
    {
        int i = i0, j = j0, pc = prev;
        for (int d = d0; d < d1; ++d) {
            const bool takea = j >= lb || (i < la && ab[i] < cd[j]);
            const int k = takea ? ab[i++] : cd[j++];
            heads += (k >> 4) != pc;
            pc = k >> 4;
        }
    }
    scan[t] = heads;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int x = 0; x < UT; ++x) { const int c = scan[x]; scan[x] = a; a += c; }
        ucount[tile] = a;
    }
    __syncthreads();
    int w = scan[t];
    int32_t* u = uni + tile * (int64_t) TQ * cap;
    {
        int i = i0, j = j0, pc = prev, cur = -1, mask = 0;
        bool open = false;
        auto take = [&]() { const bool ta = j >= lb || (i < la && ab[i] < cd[j]); return ta ? ab[i++] : cd[j++]; };
        for (int d = d0; d < d1; ++d) {
            const int k = take();
            if ((k >> 4) != pc) {
                if (open) u[w++] = cur * 16 | mask;
                open = true;
                cur = k >> 4;
                mask = k & 15;
            } else if (open) {
                mask |= k & 15;
            }
            pc = k >> 4;
        }
        if (open) {   // the run may continue past this segment (at most TQ - 1 entries)
            for (int d = d1; d < L && d < d1 + TQ; ++d) {
                const int k = take();
                if ((k >> 4) != cur) break;
                mask |= k & 15;
            }
            u[w++] = cur * 16 | mask;
        }
    }
}

// ---- 2. the attention over the union
struct alignas(16) TileSmem {
    uint8_t k[CH][KLD];
    uint8_t v[CH][KLD];
    float ks[CH][NGRP];
    float vs[CH][NGRP];
    float part[NGRP][ROWS][CH + 1];       // q.k per dim group; p aliases part[0] (each softmax thread reads its cells'
                                          // four partials before it writes their p)
    long long row[CH];
    int mask[CH];
    float alpha[ROWS];
    float lsum[ROWS];
    float mrow[ROWS];
    unsigned qmax[TQ];
};

__global__ void __launch_bounds__(THREADS) tile_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                            const int32_t* __restrict__ uni,
                                                            const int32_t* __restrict__ ucount, int64_t cap,
                                                            int64_t n_q, int n_kv_heads, int page_size,
                                                            float scale_log2, float* __restrict__ attn) {
#if FNP_TA_GFX11
    __shared__ TileSmem S;
    float(*PS)[CH + 1] = S.part[0];
    const int64_t tile = blockIdx.x;
    const int kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5;
    const int mt = wave / NGRP, g = wave % NGRP, col = lane & 15, half = lane >> 4;
    const int64_t q0 = tile * TQ;
    if (t < TQ) S.qmax[t] = 0u;
    if (t < ROWS) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    // per query: the largest |q| over its 12 heads of this KV head (the FP16 image's exponent, as the PA kernel)
    for (int i = t; i < ROWS * HD; i += THREADS) {
        const int r = i / HD, qq = r / G;
        if (q0 + qq >= n_q) continue;
        const float x = fabsf(q[(q0 + qq) * n_head * HD + (size_t) (kvh * G + r % G) * HD + i % HD]);
        atomicMax(&S.qmax[qq], __float_as_uint(x));
    }
    __syncthreads();
    // this wave's q fragments: rows 16 mt .. + 15 (row r = query r / 12, head r % 12), dims 64 g + 16 kk ..
    th16 qh[4];
    {
        const int r = 16 * mt + col, qq = r / G;
        const bool ok = q0 + qq < n_q;
        int qe = 0;
        const float qm = __uint_as_float(S.qmax[qq]);
        if (qm > 0.0f) frexpf(qm, &qe);
        const float qup = ldexpf(1.0f, 14 - qe);
        const float* src = q + (ok ? (q0 + qq) : 0) * n_head * HD + (size_t) (kvh * G + r % G) * HD + 64 * g;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk)
#pragma unroll
            for (int i = 0; i < 16; ++i) qh[kk][i] = (_Float16) (ok ? src[16 * kk + i] * qup : 0.0f);
    }
    float qdown[TQ];
#pragma unroll
    for (int qq = 0; qq < TQ; ++qq) {
        int qe = 0;
        const float qm = __uint_as_float(S.qmax[qq]);
        if (qm > 0.0f) frexpf(qm, &qe);
        qdown[qq] = ldexpf(scale_log2, qe - 14);
    }
    tf8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = tf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    const int nu = ucount[tile];
    const int32_t* u = uni + tile * (int64_t) TQ * cap;
    for (int c0 = 0; c0 < nu; c0 += CH) {
        __syncthreads();   // every wave is done with the previous chunk's K, V, p and scales
        if (t < CH) {
            long long row = -1;
            int mask = 0;
            if (c0 + t < nu) {
                const int e = u[c0 + t], cell = e >> 4;
                mask = e & 15;
                const long long page = (long long) p.page_table[cell / page_size];
                if (page >= 0) row = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
                else mask = 0;   // a block the KV streaming left non-resident: masked, as the PA kernel
            }
            S.row[t] = row;
            S.mask[t] = mask;
        }
        __syncthreads();
        // stage the chunk's K and V rows (256 codes each) and their per-64 scales
        for (int i = t; i < 2 * CH * (HD / 16); i += THREADS) {
            const int which = i / (CH * (HD / 16)), r = (i / (HD / 16)) % CH, q16 = i % (HD / 16);
            const long long row = S.row[r];
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row >= 0) v = *reinterpret_cast<const uint4*>((which ? p.v_q : p.k_q) + row * HD + 16 * q16);
            *reinterpret_cast<uint4*>(which ? &S.v[r][16 * q16] : &S.k[r][16 * q16]) = v;
        }
        if (t < 2 * CH * NGRP) {
            const int which = t / (CH * NGRP), r = (t / NGRP) % CH, gg = t % NGRP;
            const long long row = S.row[r];
            float sc = 0.0f;
            if (row >= 0) sc = __half2float(__ushort_as_half((which ? p.v_scale : p.k_scale)[row * NGRP + gg]));
            (which ? S.vs : S.ks)[r][gg] = sc;
        }
        __syncthreads();
        // q.k over this wave's 64 dims for its 16 rows and the chunk's two 16-cell tiles
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            tf8 s = tf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) s = twmma(qh[kk], codes16(&S.k[nt * 16 + col][64 * g + 16 * kk]), s);
            const float ksc = S.ks[nt * 16 + col][g];
#pragma unroll
            for (int i = 0; i < 8; ++i) S.part[g][16 * mt + 2 * i + half][nt * 16 + col] = s[i] * ksc;
        }
        __syncthreads();
        // online softmax: row t / 8, 4 cells per thread (the four groups' partials in a fixed order)
        {
            const int r = t >> 3, sub = t & 7, qq = r / G;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = (S.mask[c] >> qq) & 1
                           ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown[qq]
                           : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = (x[j] == -INFINITY || m_new == -INFINITY) ? 0.0f : exp2f(x[j] - m_new);
                PS[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? (m_new == -INFINITY ? 1.0f : 0.0f) : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this wave's 64 dims for its 16 rows: 4 tiles of 16 dims, 2 k-steps of 16 cells
        {
            float vmax = S.vs[lane][g];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            tf8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) tmp[j] = tf8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int cb = ks * 16;
                th16 ah;
#pragma unroll
                for (int i = 0; i < 16; ++i) ah[i] = (_Float16) (PS[16 * mt + col][cb + i] * (S.vs[cb + i][g] * vup));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    th16 b;
#pragma unroll
                    for (int i = 0; i < 16; ++i) b[i] = (_Float16) (int) (int8_t) S.v[cb + i][64 * g + 16 * j + col];
                    tmp[j] = twmma(ah, b, tmp[j]);
                }
            }
            float a[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] = S.alpha[16 * mt + 2 * i + half];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[j][i] = fmaf(acc[j][i], a[i], tmp[j][i] * vdown);
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 16 * mt + 2 * i + half, qq = r / G;
        if (q0 + qq >= n_q) continue;
        const float l = S.lsum[r];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = attn + (q0 + qq) * n_head * HD + (size_t) (kvh * G + r % G) * HD + 64 * g + col;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[16 * j] = acc[j][i] * inv;
    }
#else
    __builtin_trap();
#endif
}

bool gfx11_device() {
    static int known[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (known[dev] == 0) {
        cudaDeviceProp prop;
        known[dev] = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && gfx_arch_is_gfx11_wmma(prop.gcnArchName) ? 1 : 2;
        cudaGetLastError();
    }
    return known[dev] == 1;
}

}  // namespace

uint64_t fnp_qsa_tile_scratch_bytes(int64_t n_q, int64_t cap) {
    const int64_t tiles = (n_q + TQ - 1) / TQ;
    return (uint64_t) tiles * (uint64_t) (TQ * cap + 1) * 4 + 256;
}

bool fnp_qsa_tile_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                       const QsaShapes& s, float* attn, int64_t n_q, void* scratch, void* stream) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || pools.k_q == nullptr || pools.v_q == nullptr ||
        pools.k_scale == nullptr || pools.v_scale == nullptr || pools.page_table == nullptr || scratch == nullptr ||
        cap <= 0 || cap > (1 << 20) || !gfx11_device())
        return false;
    const int64_t tiles = (n_q + TQ - 1) / TQ;
    int32_t* uni = (int32_t*) scratch;
    int32_t* ucount = uni + tiles * TQ * cap;
    const cudaStream_t st = (cudaStream_t) stream;
    const size_t usm = (size_t) (4 * cap + UT) * 4;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(union_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) 65536);
        attr = true;
    }
    if (usm > 65536) return false;
    union_kernel<<<(unsigned) tiles, UT, usm, st>>>(ids, steps, cap, n_q, uni, ucount);
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t t0 = 0; t0 < tiles; t0 += 65535) {
        const int64_t nb = tiles - t0 < 65535 ? tiles - t0 : 65535;
        tile_attn_kernel<<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, 0, st>>>(
            q + t0 * TQ * s.n_head * HD, pools, uni + t0 * TQ * cap, ucount + t0, cap, n_q - t0 * TQ, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + t0 * TQ * s.n_head * HD);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "fnp_qsa_tile_attn: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace strata::kernels

extern "C" int fnp_qsa_tile_attn_i8(const float* q, const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale,
                                    const uint16_t* v_scale, const int32_t* page_table, int64_t page_size,
                                    const int32_t* ids, const int32_t* steps, int64_t cap, float* attn, int64_t n_q,
                                    void* scratch, void* stream) {
    strata::kernels::QsaAttnPools p;
    p.k_q = k_q; p.v_q = v_q; p.k_scale = k_scale; p.v_scale = v_scale; p.page_table = page_table;
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.page_size = page_size;
    return strata::kernels::fnp_qsa_tile_attn(q, p, ids, steps, cap, s, attn, n_q, scratch, stream) ? 0 : 1;
}
extern "C" uint64_t fnp_qsa_tile_scratch_bytes_c(int64_t n_q, int64_t cap) {
    return strata::kernels::fnp_qsa_tile_scratch_bytes(n_q, cap);
}
