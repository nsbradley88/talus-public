// src/laguna/laguna_kernels.cu - the few kernels the eager Laguna driver needs.
#include "laguna_kernels.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::laguna {

namespace {

void check_launch(const char* who) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "laguna: %s: %s\n", who, cudaGetErrorString(e));
        std::exit(1);
    }
}

// One block per output row; f64 accumulation so the router's near-ties resolve as an exact product would.
__global__ void f32_matvec_kernel(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                  int n_in) {
    const int o = blockIdx.x;
    const float* row = w + (size_t) o * n_in;
    double acc = 0;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) acc += (double) row[i] * (double) x[i];
    __shared__ double s[256];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st) s[threadIdx.x] += s[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[o] = (float) s[0];
}

// o[h][d] *= softplus(g[h]), ggml's softplus: g > 20 ? g : log1p(exp(g)), in f32.
__global__ void softplus_head_gate_kernel(float* __restrict__ o, const float* __restrict__ g, int n_head,
                                          int head_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_head * head_dim) return;
    const float v = g[i / head_dim];
    const float sp = v > 20.0f ? v : log1pf(expf(v));
    o[i] *= sp;
}

// y += w * x
__global__ void axpy_kernel(float* __restrict__ y, const float* __restrict__ x, float w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += w * x[i];
}

// dst[r] = src[idx[r]] for rows of `row_bytes` bytes (a multiple of 4)
__global__ void gather_rows_kernel(const uint32_t* __restrict__ src, uint32_t* __restrict__ dst,
                                   const int* __restrict__ idx, int row_words) {
    const int r = blockIdx.x;
    const uint32_t* a = src + (size_t) idx[r] * row_words;
    uint32_t* b = dst + (size_t) r * row_words;
    for (int i = threadIdx.x; i < row_words; i += blockDim.x) b[i] = a[i];
}

// x[t][i] += sum_j w[t*k + j] * y[map[t*k + j]][i], j ascending: the eager path's axpy order
__global__ void moe_add_slots_kernel(float* __restrict__ x, const float* __restrict__ shared,
                                     const float* __restrict__ y, const int* __restrict__ map,
                                     const float* __restrict__ w, int n_embd, int k) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = shared[(size_t) t * n_embd + i];
    for (int j = 0; j < k; ++j) acc += w[t * k + j] * y[(size_t) map[t * k + j] * n_embd + i];
    x[(size_t) t * n_embd + i] += acc;
}

// moe_add_slots_kernel with 4 consecutive dims per thread (float4 loads / stores); per element the same
// chain acc = shared; acc += w_j * y_j (j ascending); x += acc - bitwise.  Used when n_embd % 4 == 0 (and aligned).
__global__ void moe_add_slots4_kernel(float* __restrict__ x, const float* __restrict__ shared,
                                      const float* __restrict__ y, const int* __restrict__ map,
                                      const float* __restrict__ w, int n_embd, int k) {
    const int t = blockIdx.y;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n_embd) return;
    float4 acc = *reinterpret_cast<const float4*>(shared + (size_t) t * n_embd + i);
    for (int j = 0; j < k; ++j) {
        const float wj = w[t * k + j];
        const float4 yv = *reinterpret_cast<const float4*>(y + (size_t) map[t * k + j] * n_embd + i);
        acc.x += wj * yv.x;
        acc.y += wj * yv.y;
        acc.z += wj * yv.z;
        acc.w += wj * yv.w;
    }
    float4* xp = reinterpret_cast<float4*>(x + (size_t) t * n_embd + i);
    float4 xv = *xp;
    xv.x += acc.x;
    xv.y += acc.y;
    xv.z += acc.z;
    xv.w += acc.w;
    *xp = xv;
}

// ---- prompt attention: causal (optionally sliding-window) GQA over the f16 KV pool, f32 online softmax ----
// grid (ceil(nq / kQT), n_kv_head), 256 threads.  A block takes kQT query tokens x the G query heads of one KV head
// (R = kQT*G rows) and walks the keys in tiles of kKT, so each K/V tile is read once for R rows.  Q and K meet as
// f16 pairs in v_dot2 (f32 accumulate), like llama.cpp's f16 flash attention; P and V meet in f32.
// (per instantiation: RowsPerWarp rows per warp in QK, lanes x KT/32 keys; PvRows rows per (dim pair, row group) in PV)
constexpr int kD = 128, kD2 = kD / 2, kQT = 4, kKT = 64, kMaxG = 9;

typedef _Float16 h2_t __attribute__((ext_vector_type(2)));

__device__ inline h2_t h2_from_bits(uint32_t u) { h2_t v; __builtin_memcpy(&v, &u, 4); return v; }

// one warp holding 32 consecutive dims of a row: its int8 value and (from lane 0's view) the block's f16 scale
__device__ inline int8_t q8_block(float x, uint16_t& scale_bits) {
    float amax = fabsf(x);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
    scale_bits = strata::kernels::f16_from_f32(amax / 127.0f);
    const float d = strata::kernels::f32_from_f16(scale_bits);
    const float qv = d > 0.0f ? rintf(x / d) : 0.0f;
    return (int8_t) fmaxf(-127.0f, fminf(127.0f, qv));
}
__device__ inline uint16_t h_bits(float f) { return strata::kernels::f16_from_f32(f); }
// a position's slot in a layer pool holding `stride` positions: sliding-window layers keep a ring of stride positions,
// full layers a stride past the context, where this is the identity
__device__ inline int kv_slot(int pos, int stride) { return stride > 0 ? pos % stride : pos; }

template <int QT, int KT>
__global__ void __launch_bounds__(256) prompt_attn_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kp,
                                                          const uint16_t* __restrict__ vp, float* __restrict__ out,
                                                          int nq, int q_pos0, const int* __restrict__ q_pos0_dev,
                                                          int H, int G, int n_kv, int window,
                                                          float* __restrict__ part_acc, float* __restrict__ part_ml,
                                                          int vt_stride, const uint16_t* __restrict__ ksc,
                                                          const uint16_t* __restrict__ vsc) {
    const bool q8 = ksc != nullptr;   // q8 KV: kp/vp hold int8, dequantized into the same f16 LDS tiles
    constexpr int MaxR = QT * kMaxG, RowsPerWarp = (MaxR + 7) / 8, PvRows = (MaxR + 3) / 4, KPL = KT / 32;
    if (q_pos0_dev) q_pos0 = *q_pos0_dev;
    __shared__ h2_t sq[MaxR][kD2];
    __shared__ uint32_t sk[KT][kD2 + 1];          // f16 pairs; +1 so the lanes' key rows fall in distinct banks
    __shared__ uint32_t sv[KT][kD2];
    __shared__ float ss[MaxR][KT + 1];
    __shared__ float sm[MaxR], sl[MaxR], sa[MaxR];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, kvh = blockIdx.y;
    const int t0 = blockIdx.x * QT;
    const int nt = min(QT, nq - t0);
    const int R = QT * G;
    const float scale = 0.08838834764831845f;      // 1/sqrt(128), folded into Q before the f16 round
    for (int i = tid; i < R * kD2; i += blockDim.x) {
        const int r = i / kD2, d2 = i % kD2, tq = r / G, g = r % G;
        h2_t v = {(_Float16) 0.0f, (_Float16) 0.0f};
        if (tq < nt) {
            const float* qr = q + ((size_t) (t0 + tq) * H + kvh * G + g) * kD + 2 * d2;
            v = h2_t{(_Float16) (qr[0] * scale), (_Float16) (qr[1] * scale)};
        }
        sq[r][d2] = v;
    }
    if (tid < R) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }
    const int pd2 = tid % kD2, prg = tid / kD2;     // PV mapping
    float acc0[PvRows], acc1[PvRows];
#pragma unroll
    for (int i = 0; i < PvRows; ++i) acc0[i] = acc1[i] = 0.0f;
    const int qmin = q_pos0 + t0, qmax = q_pos0 + t0 + nt - 1;
    // keys [kb, ke): the whole causal (or window) span, or this block's share of it when split (flash-decoding,
    // gridDim.z > 1: tile-aligned chunks, partial results merged by attn_merge)
    int kb = window > 0 ? max(0, qmin - (window - 1)) : 0;
    int ke = qmax + 1;
    if (gridDim.z > 1) {
        const int n = ke - kb;
        const int chunk = ((n + (int) gridDim.z - 1) / (int) gridDim.z + KT - 1) / KT * KT;
        kb += (int) blockIdx.z * chunk;
        ke = min(ke, kb + chunk);
    }
    const size_t cell = (size_t) n_kv * kD;
    const int r0w = warp * RowsPerWarp;
    __syncthreads();
    for (int k0 = kb; k0 < ke; k0 += KT) {
        if (q8) {   // K: one 16-byte load of 16 dims (and their 32-dim block's scale) per task
            for (int i = tid; i < KT * 8; i += blockDim.x) {
                const int j = i / 8, c16 = i % 8, pos = k0 + j;
                uint32_t pr[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
                if (pos < ke) {
                    const uint4 raw = *reinterpret_cast<const uint4*>(reinterpret_cast<const int8_t*>(kp) +
                                                                       (size_t) kv_slot(pos, vt_stride) * cell + kvh * kD + c16 * 16);
                    const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                    const float sc = strata::kernels::f32_from_f16(ksc[ks_index(kv_slot(pos, vt_stride), kvh, c16 / 2, n_kv)]);
#pragma unroll
                    for (int e = 0; e < 8; ++e)
                        pr[e] = (uint32_t) h_bits(rv[2 * e] * sc) | ((uint32_t) h_bits(rv[2 * e + 1] * sc) << 16);
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) sk[j][c16 * 8 + e] = pr[e];
            }
        } else {
            for (int i = tid; i < KT * kD2; i += blockDim.x) {
                const int j = i / kD2, d2 = i % kD2, pos = k0 + j;
                const bool ok = pos < ke;
                const size_t off = (size_t) kv_slot(pos, vt_stride) * cell + kvh * kD + 2 * d2;
                sk[j][d2] = ok ? *(const uint32_t*) (kp + off) : 0u;
                if (vt_stride == 0) sv[j][d2] = ok ? *(const uint32_t*) (vp + off) : 0u;
            }
        }
        if (vt_stride > 0) {   // V tile-transposed (vt_index): 16-byte runs of 8 keys of one dim
            uint16_t* sv16 = reinterpret_cast<uint16_t*>(&sv[0][0]);   // [KT][128] halves
            for (int i = tid; i < (KT / 8) * kD; i += blockDim.x) {
                const int d = i % kD, j8 = (i / kD) * 8;   // consecutive threads: consecutive dims (LDS banks)
                const int p8 = k0 + j8;
                uint4 w = make_uint4(0, 0, 0, 0);
                uint16_t* wv = reinterpret_cast<uint16_t*>(&w);
                if (q8) {
                    const int8_t* v8 = reinterpret_cast<const int8_t*>(vp);
                    if ((p8 & 7) == 0 && p8 + 8 <= ke) {
                        const uint2 raw = *reinterpret_cast<const uint2*>(v8 + vt_index(kvh, kv_slot(p8, vt_stride), d, vt_stride));
                        const uint4 scl = *reinterpret_cast<const uint4*>(vsc + vs_index(kvh, kv_slot(p8, vt_stride), d / 32, vt_stride));
                        const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                        const uint16_t* sv_ = reinterpret_cast<const uint16_t*>(&scl);
#pragma unroll
                        for (int e = 0; e < 8; ++e) wv[e] = h_bits(rv[e] * strata::kernels::f32_from_f16(sv_[e]));
                    } else {
#pragma unroll
                        for (int e = 0; e < 8; ++e)
                            wv[e] = (p8 + e < ke) ? h_bits(v8[vt_index(kvh, kv_slot(p8 + e, vt_stride), d, vt_stride)] *
                                                           strata::kernels::f32_from_f16(vsc[vs_index(kvh, kv_slot(p8 + e, vt_stride), d / 32, vt_stride)]))
                                                  : (uint16_t) 0;
                    }
                } else if ((p8 & 7) == 0 && p8 + 8 <= ke) {
                    w = *reinterpret_cast<const uint4*>(vp + vt_index(kvh, kv_slot(p8, vt_stride), d, vt_stride));
                } else {   // prompt tiles may start unaligned; read element-wise there
#pragma unroll
                    for (int e = 0; e < 8; ++e) wv[e] = (p8 + e < ke) ? vp[vt_index(kvh, kv_slot(p8 + e, vt_stride), d, vt_stride)] : (uint16_t) 0;
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) sv16[(j8 + e) * kD + d] = wv[e];
            }
        }
        __syncthreads();
        // QK: this warp's rows x keys (lane, lane + 32)
        {
            float dd[RowsPerWarp][KPL];
#pragma unroll
            for (int i = 0; i < RowsPerWarp; ++i)
#pragma unroll
                for (int c = 0; c < KPL; ++c) dd[i][c] = 0.0f;
            for (int d2 = 0; d2 < kD2; ++d2) {
                h2_t kk[KPL];
#pragma unroll
                for (int c = 0; c < KPL; ++c) kk[c] = h2_from_bits(sk[lane + 32 * c][d2]);
#pragma unroll
                for (int i = 0; i < RowsPerWarp; ++i) {
                    if (r0w + i < R) {
                        const h2_t qv = sq[r0w + i][d2];
#pragma unroll
                        for (int c = 0; c < KPL; ++c) dd[i][c] = __builtin_amdgcn_fdot2(qv, kk[c], dd[i][c], false);
                    }
                }
            }
            // online softmax in the warp that holds the row: lanes carry keys lane + 32c
#pragma unroll
            for (int i = 0; i < RowsPerWarp; ++i) {
                const int r = r0w + i;
                if (r < R) {   // uniform across the warp
                    const int qp = qmin + r / G;
                    const bool live = r / G < nt;
                    float sc[KPL];
                    float mx = -INFINITY;
#pragma unroll
                    for (int c = 0; c < KPL; ++c) {
                        const int pk = k0 + lane + 32 * c;
                        const bool v = live && pk < ke && pk <= qp && (window <= 0 || qp - pk < window);
                        sc[c] = v ? dd[i][c] : -INFINITY;
                        mx = fmaxf(mx, sc[c]);
                    }
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
                    const float m_old = sm[r];
                    const float m_new = fmaxf(m_old, mx);
                    float alpha = 1.0f, sum = 0.0f;
#pragma unroll
                    for (int c = 0; c < KPL; ++c) {
                        const float pc = (m_new != -INFINITY && sc[c] != -INFINITY) ? expf(sc[c] - m_new) : 0.0f;
                        ss[r][lane + 32 * c] = pc;
                        sum += pc;
                    }
                    if (m_new != -INFINITY) alpha = expf(m_old - m_new);   // 0 on the first live tile
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
                    if (lane == 0) {
                        sm[r] = m_new;
                        sl[r] = sl[r] * alpha + sum;
                        sa[r] = alpha;
                    }
                }
            }
        }
        __syncthreads();
        // PV: dim pair pd2, rows prg, prg + 4, ...
#pragma unroll
        for (int i = 0; i < PvRows; ++i) {
            const int r = prg + 4 * i;
            if (r < R) { acc0[i] *= sa[r]; acc1[i] *= sa[r]; }
        }
        for (int j = 0; j < KT; ++j) {
            const h2_t v = h2_from_bits(sv[j][pd2]);
            const float v0 = (float) v.x, v1 = (float) v.y;
#pragma unroll
            for (int i = 0; i < PvRows; ++i) {
                const int r = prg + 4 * i;
                if (r < R) {
                    const float pj = ss[r][j];
                    acc0[i] += pj * v0;
                    acc1[i] += pj * v1;
                }
            }
        }
        __syncthreads();
    }
    if (part_acc) {   // split: unnormalised accumulators and (max, sum) per row, merged by attn_merge
#pragma unroll
        for (int i = 0; i < PvRows; ++i) {
            const int r = prg + 4 * i;
            if (r < R && r / G < nt) {
                const size_t row = ((size_t) blockIdx.z * nq + t0 + r / G) * H + kvh * G + r % G;
                part_acc[row * kD + 2 * pd2] = acc0[i];
                part_acc[row * kD + 2 * pd2 + 1] = acc1[i];
                if (pd2 == 0) { part_ml[row * 2] = sm[r]; part_ml[row * 2 + 1] = sl[r]; }
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < PvRows; ++i) {
        const int r = prg + 4 * i;
        if (r < R && r / G < nt) {
            const int tq = r / G, g = r % G;
            const float inv = sl[r] > 0.0f ? 1.0f / sl[r] : 0.0f;
            float* o = out + ((size_t) (t0 + tq) * H + kvh * G + g) * kD + 2 * pd2;
            o[0] = acc0[i] * inv;
            o[1] = acc1[i] * inv;
        }
    }
}

// Y[t][o] = sum_i X[t][i] * W[o][i] in f32 (the batched router; llama.cpp's router product is f32 too).
// 64x64 output tile per block, 256 threads x 4x4 outputs, K in steps of 16.  n_in % 16 == 0.
__global__ void __launch_bounds__(256) f32_matmul_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                         float* __restrict__ y, int n_in, int n_out, int n_tok) {
    __shared__ float sx[16][64 + 4];
    __shared__ float sw[16][64 + 4];
    const int tid = threadIdx.x;
    const int t0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    const int tr = (tid / 16) * 4, oc = (tid % 16) * 4;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < n_in; k0 += 16) {
        for (int i = tid; i < 64 * 16; i += 256) {
            const int r = i / 16, kk = i % 16;
            sx[kk][r] = (t0 + r < n_tok) ? x[(size_t) (t0 + r) * n_in + k0 + kk] : 0.0f;
            sw[kk][r] = (o0 + r < n_out) ? w[(size_t) (o0 + r) * n_in + k0 + kk] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) { a[i] = sx[kk][tr + i]; b[i] = sw[kk][oc + i]; }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (t0 + tr + i < n_tok && o0 + oc + j < n_out) y[(size_t) (t0 + tr + i) * n_out + o0 + oc + j] = acc[i][j];
}

// f32_matmul_kernel re-tiled for LDS bandwidth (the old kernel reads 2 B of LDS per FMA: LDS-bound): a
// block of 64 threads owns a 64 x 64 output tile, 8 x 8 outputs per thread (4 B of LDS reads per 4 FMAs... 1 B/FMA),
// K steps of 16 with the next step's global loads in flight.  Per output the chain is f32_matmul_kernel's:
// acc[i][j] += a[i] * b[j] for k ascending from 0 (the same contraction), so the bits are identical.
// TALUS_F32MM_V2=1 selects it (default off until measured).  n_in % 16 == 0.
__global__ void __launch_bounds__(64) f32_matmul_v2_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                           float* __restrict__ y, int n_in, int n_out, int n_tok) {
    constexpr int KS = 16;
    __shared__ __attribute__((aligned(16))) float sx[KS][64 + 4];
    __shared__ __attribute__((aligned(16))) float sw[KS][64 + 4];
    const int tid = threadIdx.x;
    const int t0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    const int tr = (tid / 8) * 8, oc = (tid % 8) * 8;
    // loader: 64 rows x 16 k = 256 float4 per operand, 4 per thread: row lr + 16 u, k4 lk
    const int lr = tid / 4, lk = (tid % 4) * 4;
    float4 px[4], pw[4];
    auto gload = [&](int k0) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int r = lr + 16 * u;
            px[u] = (t0 + r < n_tok) ? *reinterpret_cast<const float4*>(x + (size_t) (t0 + r) * n_in + k0 + lk)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
            pw[u] = (o0 + r < n_out) ? *reinterpret_cast<const float4*>(w + (size_t) (o0 + r) * n_in + k0 + lk)
                                     : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto lstore = [&]() {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int r = lr + 16 * u;
            sx[lk + 0][r] = px[u].x; sx[lk + 1][r] = px[u].y; sx[lk + 2][r] = px[u].z; sx[lk + 3][r] = px[u].w;
            sw[lk + 0][r] = pw[u].x; sw[lk + 1][r] = pw[u].y; sw[lk + 2][r] = pw[u].z; sw[lk + 3][r] = pw[u].w;
        }
    };
    float acc[8][8] = {};
    gload(0);
    lstore();
    __syncthreads();
    for (int k0 = 0; k0 < n_in; k0 += KS) {
        const bool more = k0 + KS < n_in;
        if (more) gload(k0 + KS);
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&sx[kk][tr]), a1 = *reinterpret_cast<const float4*>(&sx[kk][tr + 4]);
            const float4 b0 = *reinterpret_cast<const float4*>(&sw[kk][oc]), b1 = *reinterpret_cast<const float4*>(&sw[kk][oc + 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
        if (more) {
            lstore();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (t0 + tr + i < n_tok && o0 + oc + j < n_out) y[(size_t) (t0 + tr + i) * n_out + o0 + oc + j] = acc[i][j];
}

// f32_matmul for a few tokens (decode / verify windows: the MiMo router, 4096 x 256 F32).  The tiled kernel
// above runs (n_out / 64) blocks there (4 blocks for MiMo's 256-expert router).  Here a block holds OB whole weight rows in
// LDS (loaded once, every load in flight) and one thread per (token, row) runs the same sequential chain over k as
// f32_matmul_kernel's acc[i][j] += a[i] * b[j] (k ascending from 0, fused multiply-add): bitwise identical.  The
// activations stream from global/L2 in batches loaded one batch ahead of the chain.
template <int OB>
__global__ void __launch_bounds__(256) f32_matmul_few_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                             float* __restrict__ y, int n_in, int n_out, int n_tok) {
    extern __shared__ float4 sw[];                 // OB rows x n_in / 4
    const int tid = threadIdx.x, o0 = blockIdx.x * OB, k4 = n_in / 4;
    for (int i = tid; i < OB * k4; i += 256) {
        const int r = i / k4, c = i % k4;
        sw[i] = (o0 + r < n_out) ? *reinterpret_cast<const float4*>(w + (size_t) (o0 + r) * n_in + 4 * c)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    const int my_o = tid % OB, my_t = tid / OB;    // compute threads: OB x n_tok (<= 64)
    if (my_t >= n_tok || o0 + my_o >= n_out) return;
    const float4* wr = sw + my_o * k4;
    const float4* xv = reinterpret_cast<const float4*>(x + (size_t) my_t * n_in);
    constexpr int NB = 16;                         // n_in % 64 == 0 (checked by the caller)
    float acc = 0.0f;
    float4 a[NB];
#pragma unroll
    for (int j = 0; j < NB; ++j) a[j] = xv[j];
    for (int c = 0; c < k4; c += NB) {
        float4 an[NB];
        const int cn = c + NB < k4 ? c + NB : c;
#pragma unroll
        for (int j = 0; j < NB; ++j) an[j] = xv[cn + j];
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = 0; j < NB; ++j) {
            const float4 b = wr[c + j];
            acc = __builtin_fmaf(a[j].x, b.x, acc);
            acc = __builtin_fmaf(a[j].y, b.y, acc);
            acc = __builtin_fmaf(a[j].z, b.z, acc);
            acc = __builtin_fmaf(a[j].w, b.w, acc);
        }
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int j = 0; j < NB; ++j) a[j] = an[j];
    }
    y[(size_t) my_t * n_out + o0 + my_o] = acc;
}

__global__ void silu_mul_kernel(float* __restrict__ g, const float* __restrict__ u, int64_t n) {
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i < n) { const float v = g[i]; g[i] = v / (1.0f + expf(-v)) * u[i]; }
}

__global__ void fill_pos_kernel(int* __restrict__ pos, int n_tok, int per_tok, int p0) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_tok * per_tok) pos[i] = p0 + i / per_tok;
}

}  // namespace

namespace {
// out[row] = sum_z e^(m_z - m) acc_z / sum_z e^(m_z - m) l_z over the n_split partials; one 128-thread block per row
__global__ void attn_merge_kernel(const float* __restrict__ part_acc, const float* __restrict__ part_ml,
                                  float* __restrict__ out, int n_rows, int n_split) {
    const int row = blockIdx.x, d = threadIdx.x;
    float m = -INFINITY;
    for (int z = 0; z < n_split; ++z) m = fmaxf(m, part_ml[((size_t) z * n_rows + row) * 2]);
    float num = 0.0f, den = 0.0f;
    if (m != -INFINITY) {
        for (int z = 0; z < n_split; ++z) {
            const float mz = part_ml[((size_t) z * n_rows + row) * 2];
            if (mz == -INFINITY) continue;
            const float f = expf(mz - m);
            num += f * part_acc[((size_t) z * n_rows + row) * kD + d];
            den += f * part_ml[((size_t) z * n_rows + row) * 2 + 1];
        }
    }
    out[(size_t) row * kD + d] = den > 0.0f ? num / den : 0.0f;
}
}  // namespace

void prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, float* out, int nq, int q_pos0,
                 int n_head, int n_kv, int window, void* stream, const int* q_pos0_dev, int n_split,
                 float* part_acc, float* part_ml, bool merge, int vt_stride, const uint16_t* ksc,
                 const uint16_t* vsc) {
    const int G = n_head / n_kv;
    if (G > kMaxG || n_head % n_kv != 0) { std::fprintf(stderr, "laguna: prompt_attn: GQA %d unsupported\n", G); std::exit(1); }
    if (n_split < 1) n_split = 1;
    // decode (one query token): QT 1, KT 32 keeps the block's LDS ~21 KB so several fit per WGP; prompts: QT 4, KT 64
    if (nq == 1) {
        const dim3 grid(1, n_kv, n_split);
        prompt_attn_kernel<1, 32><<<grid, 256, 0, (cudaStream_t) stream>>>(q, k_pool, v_pool, out, nq, q_pos0,
                                                                          q_pos0_dev, n_head, G, n_kv, window,
                                                                          n_split > 1 ? part_acc : nullptr, part_ml, vt_stride, ksc, vsc);
    } else {
        static const int shape = [] {
            // tuning: 0 = QT4/KT64, 1 = QT4/KT32 (default on gfx1151: ~30 KB LDS, 2 blocks per WGP),
            // 2 = QT2/KT32, 3 = QT8/KT32
            const char* e = std::getenv("LAGUNA_PA_SHAPE");
            return e ? std::atoi(e) : 1;
        }();
        float* pa = n_split > 1 ? part_acc : nullptr;
        const cudaStream_t st = (cudaStream_t) stream;
        switch (shape) {
        case 1: prompt_attn_kernel<4, 32><<<dim3((nq + 3) / 4, n_kv, n_split), 256, 0, st>>>(q, k_pool, v_pool, out, nq, q_pos0, q_pos0_dev, n_head, G, n_kv, window, pa, part_ml, vt_stride, ksc, vsc); break;
        case 2: prompt_attn_kernel<2, 32><<<dim3((nq + 1) / 2, n_kv, n_split), 256, 0, st>>>(q, k_pool, v_pool, out, nq, q_pos0, q_pos0_dev, n_head, G, n_kv, window, pa, part_ml, vt_stride, ksc, vsc); break;
        case 3: prompt_attn_kernel<8, 32><<<dim3((nq + 7) / 8, n_kv, n_split), 256, 0, st>>>(q, k_pool, v_pool, out, nq, q_pos0, q_pos0_dev, n_head, G, n_kv, window, pa, part_ml, vt_stride, ksc, vsc); break;
        default: prompt_attn_kernel<kQT, kKT><<<dim3((nq + kQT - 1) / kQT, n_kv, n_split), 256, 0, st>>>(q, k_pool, v_pool, out, nq, q_pos0, q_pos0_dev, n_head, G, n_kv, window, pa, part_ml, vt_stride, ksc, vsc); break;
        }
    }
    check_launch("prompt_attn");
    if (n_split > 1 && merge) {
        attn_merge_kernel<<<nq * n_head, kD, 0, (cudaStream_t) stream>>>(part_acc, part_ml, out, nq * n_head, n_split);
        check_launch("attn_merge");
    }
}

void f32_matmul(const float* w, const float* x, float* y, int n_in, int n_out, int n_tok, void* stream) {
    if (n_in % 16 != 0) { std::fprintf(stderr, "laguna: f32_matmul: n_in %d not a multiple of 16\n", n_in); std::exit(1); }
    static const bool old_only = [] { const char* e = std::getenv("TALUS_F32MM_OLD"); return e && e[0] == '1'; }();
    if (!old_only && n_tok <= 16 && n_in % 64 == 0 && n_in <= 8192) {   // decode / verify windows: bitwise, 2 rows per block
        f32_matmul_few_kernel<2><<<(n_out + 1) / 2, 256, (size_t) 2 * n_in * sizeof(float), (cudaStream_t) stream>>>(
            w, x, y, n_in, n_out, n_tok);
        check_launch("f32_matmul_few");
        return;
    }
    static const bool v2 = [] { const char* e = std::getenv("TALUS_F32MM_V2"); return e != nullptr && e[0] == '1'; }();
    if (v2 && !old_only && n_in % 16 == 0) {   // bitwise, 8 x 8 outputs per thread
        f32_matmul_v2_kernel<<<dim3((n_out + 63) / 64, (n_tok + 63) / 64), 64, 0, (cudaStream_t) stream>>>(w, x, y, n_in,
                                                                                                          n_out, n_tok);
        check_launch("f32_matmul_v2");
        return;
    }
    f32_matmul_kernel<<<dim3((n_out + 63) / 64, (n_tok + 63) / 64), 256, 0, (cudaStream_t) stream>>>(w, x, y, n_in,
                                                                                                    n_out, n_tok);
    check_launch("f32_matmul");
}

void silu_mul(float* g, const float* u, int64_t n, void* stream) {
    silu_mul_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(g, u, n);
    check_launch("silu_mul");
}

void fill_pos(int* pos, int n_tok, int per_tok, int p0, void* stream) {
    const int n = n_tok * per_tok;
    fill_pos_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(pos, n_tok, per_tok, p0);
    check_launch("fill_pos");
}

void gather_rows(const void* src, void* dst, const int* idx, int n_rows, size_t row_bytes, void* stream) {
    if (n_rows <= 0) return;
    gather_rows_kernel<<<n_rows, 256, 0, (cudaStream_t) stream>>>((const uint32_t*) src, (uint32_t*) dst, idx,
                                                                  (int) (row_bytes / 4));
    check_launch("gather_rows");
}

void moe_add_slots(float* x, const float* shared, const float* y, const int* map, const float* w, int n_tok,
                   int n_embd, int k, void* stream) {
    static const bool v4 = [] { const char* e = std::getenv("TALUS_MOE_ADD4"); return !(e && e[0] == '0'); }();
    const bool al = ((uintptr_t) x | (uintptr_t) shared | (uintptr_t) y) % 16 == 0;
    if (v4 && n_embd % 4 == 0 && al) {   // float4 per thread (bitwise; TALUS_MOE_ADD4=0: off)
        const dim3 grid((n_embd / 4 + 255) / 256, n_tok);
        moe_add_slots4_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, shared, y, map, w, n_embd, k);
        check_launch("moe_add_slots4");
        return;
    }
    const dim3 grid((n_embd + 255) / 256, n_tok);
    moe_add_slots_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, shared, y, map, w, n_embd, k);
    check_launch("moe_add_slots");
}

void f32_matvec(const float* w, const float* x, float* y, int n_in, int n_out, void* stream) {
    f32_matvec_kernel<<<n_out, 256, 0, (cudaStream_t) stream>>>(w, x, y, n_in);
    check_launch("f32_matvec");
}

void softplus_head_gate(float* o, const float* g, int n_head, int head_dim, void* stream) {
    const int n = n_head * head_dim;
    softplus_head_gate_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(o, g, n_head, head_dim);
    check_launch("softplus_head_gate");
}

void axpy(float* y, const float* x, float w, int n, void* stream) {
    axpy_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(y, x, w, n);
    check_launch("axpy");
}

namespace {

// ---- decode: device-side state, so a whole token step can be captured once and replayed ----

// Laguna's router for one token: p = sigmoid(logit), select top-k of p + bias (ties: lower index), weights
// p / max(sum, 6.1e-5) * scale.  One block of n_expert (<= 256) threads.
__global__ void route_sigmoid_topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                          int32_t* __restrict__ ids, float* __restrict__ w, int n_expert, int k,
                                          float scale) {
    __shared__ float sp[256], ssel[256];
    __shared__ float rv[256];
    __shared__ int ri[256];
    __shared__ int chosen[16];
    const int e = threadIdx.x;
    if (e < n_expert) {
        sp[e] = 1.0f / (1.0f + expf(-logits[e]));
        ssel[e] = sp[e] + bias[e];
    }
    __syncthreads();
    for (int j = 0; j < k; ++j) {
        rv[e] = e < n_expert ? ssel[e] : -INFINITY;
        ri[e] = e;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (e < st) {
                const float a = rv[e], b = rv[e + st];
                if (b > a || (b == a && ri[e + st] < ri[e])) { rv[e] = b; ri[e] = ri[e + st]; }
            }
            __syncthreads();
        }
        if (e == 0) { chosen[j] = ri[0]; ssel[ri[0]] = -INFINITY; }
        __syncthreads();
    }
    if (e == 0) {
        double sum = 0;
        for (int j = 0; j < k; ++j) sum += sp[chosen[j]];
        const double denom = sum > 6.103515625e-5 ? sum : 6.103515625e-5;
        for (int j = 0; j < k; ++j) {
            ids[j] = chosen[j];
            w[j] = (float) (sp[chosen[j]] / denom * scale);
        }
    }
}

// x = dequantized row *tok of a Q8_0 table (34 bytes per 32 values)
__global__ void embed_q8_0_kernel(const uint8_t* __restrict__ table, const int* __restrict__ tok, float* __restrict__ x,
                                  int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const uint8_t* blk = table + (size_t) tok[blockIdx.y] * (n_embd / 32) * 34 + (size_t) (i / 32) * 34;
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    x[(size_t) blockIdx.y * n_embd + i] = strata::kernels::f32_from_f16(dbits) * (float) (int8_t) blk[2 + i % 32];
}

// one token's K/V (n floats each) into the f16 pools at cell *pos
__global__ void kv_store_kernel(uint16_t* __restrict__ kpool, uint16_t* __restrict__ vpool, const float* __restrict__ kc,
                                const float* __restrict__ vc, int n, const int* __restrict__ pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t off = (size_t) *pos * n + i;
    kpool[off] = strata::kernels::f16_from_f32(kc[i]);
    vpool[off] = strata::kernels::f16_from_f32(vc[i]);
}

__global__ void fill_pos_dev_kernel(int* __restrict__ out, int n, const int* __restrict__ pos) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = *pos;
}

// tok = argmax(logits) (ties: lower index); then *pos += 1 (the step's last kernel)
__global__ void argmax_advance_kernel(const float* __restrict__ logits, int n, int* __restrict__ tok,
                                      int* __restrict__ pos) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float v = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (logits[i] > v) { v = logits[i]; idx = i; }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st) {
            const float b = bv[threadIdx.x + st];
            const int j = bi[threadIdx.x + st];
            if (b > bv[threadIdx.x] || (b == bv[threadIdx.x] && j < bi[threadIdx.x])) { bv[threadIdx.x] = b; bi[threadIdx.x] = j; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { *tok = bi[0]; *pos += 1; }
}

}  // namespace

void route_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* w, int n_expert, int k,
                        float scale, void* stream) {
    if (n_expert > 256 || k > 16) { std::fprintf(stderr, "laguna: route_sigmoid_topk: shape\n"); std::exit(1); }
    route_sigmoid_topk_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(logits, bias, ids, w, n_expert, k, scale);
    check_launch("route_sigmoid_topk");
}

void embed_q8_0(const void* table, const int* tok, float* x, int n_embd, void* stream, int n_tok) {
    embed_q8_0_kernel<<<dim3((n_embd + 255) / 256, n_tok), 256, 0, (cudaStream_t) stream>>>((const uint8_t*) table, tok,
                                                                                         x, n_embd);
    check_launch("embed_q8_0");
}

void kv_store(uint16_t* k_pool, uint16_t* v_pool, const float* kc, const float* vc, int n, const int* pos,
              void* stream) {
    kv_store_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(k_pool, v_pool, kc, vc, n, pos);
    check_launch("kv_store");
}

void fill_pos_dev(int* out, int n, const int* pos, void* stream) {
    fill_pos_dev_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t) stream>>>(out, n, pos);
    check_launch("fill_pos_dev");
}

void argmax_advance(const float* logits, int n, int* tok, int* pos, void* stream) {
    argmax_advance_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(logits, n, tok, pos);
    check_launch("argmax_advance");
}

namespace {

// ---- fused small ops (combining norm/rope launches on decode windows) ----

// Per-head RMSNorm (weight w[128]) then NEOX RoPE over the first n_rot dims, in place.  One 128-thread block per
// (token, head) row; pos[row] indexes the (max_pos, n_rot/2) cos/sin tables.
__global__ void __launch_bounds__(128) head_norm_rope_kernel(float* __restrict__ x, const float* __restrict__ w,
                                                             float eps, int n_rot, const float* __restrict__ cos_tab,
                                                             const float* __restrict__ sin_tab,
                                                             const int* __restrict__ pos, int pos_div) {
    __shared__ float sv[128];
    __shared__ float red[4];
    const int r = blockIdx.x, d = threadIdx.x;
    float* row = x + (size_t) r * 128;
    const float v = row[d];
    float ss = v * v;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    if (d % 32 == 0) red[d / 32] = ss;
    __syncthreads();
    const float mean = (red[0] + red[1] + red[2] + red[3]) / 128.0f;
    const float y = v * rsqrtf(mean + eps) * w[d];
    sv[d] = y;
    __syncthreads();
    const int half = n_rot / 2;
    if (d < half) {
        const int p = pos[pos_div > 0 ? r / pos_div : 0];
        const float c = cos_tab[(size_t) p * half + d], sn = sin_tab[(size_t) p * half + d];
        const float a = sv[d], b = sv[d + half];
        row[d] = a * c - b * sn;
        row[d + half] = a * sn + b * c;
    } else if (d >= n_rot) {
        row[d] = y;
    }
}

// dst[t] = RMSNorm(src[t]) * w over cols, one 1024-thread block per row (dst may equal src)
__global__ void __launch_bounds__(1024) rms_norm_to_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                                           const float* __restrict__ w, int cols, float eps) {
    __shared__ float red[32];
    const float* xs = src + (size_t) blockIdx.x * cols;
    float* yd = dst + (size_t) blockIdx.x * cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xs[i] * xs[i];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor(t, o, 32);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float) cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) yd[i] = xs[i] * scale * w[i];
}

// y[o] = w[o] . x in f32, one warp per row, float4 loads (n_in % 128 == 0)
__global__ void __launch_bounds__(256) f32_matvec_warp_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                              float* __restrict__ y, int n_in, int n_out) {
    const int o = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (o >= n_out) return;
    const float4* row = reinterpret_cast<const float4*>(w + (size_t) o * n_in);
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = row[i], b = xv[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[o] = acc;
}

}  // namespace

void head_norm_rope(float* x, const float* w, int rows, float eps, int n_rot, const float* cos_tab,
                    const float* sin_tab, const int* pos, int pos_div, void* stream) {
    head_norm_rope_kernel<<<rows, 128, 0, (cudaStream_t) stream>>>(x, w, eps, n_rot, cos_tab, sin_tab, pos, pos_div);
    check_launch("head_norm_rope");
}

void rms_norm_to(float* dst, const float* src, const float* w, int rows, int cols, float eps, void* stream) {
    rms_norm_to_kernel<<<rows, 1024, 0, (cudaStream_t) stream>>>(dst, src, w, cols, eps);
    check_launch("rms_norm_to");
}

void f32_matvec_fast(const float* w, const float* x, float* y, int n_in, int n_out, void* stream) {
    if (n_in % 128 != 0) { std::fprintf(stderr, "laguna: f32_matvec_fast: n_in %% 128\n"); std::exit(1); }
    f32_matvec_warp_kernel<<<(n_out + 7) / 8, 256, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out);
    check_launch("f32_matvec_fast");
}

namespace {

// ---- M3 decode fusions ----

struct Q81 { half2 ds; int8_t qs[32]; };   // native_mmvq's Q81Block

// x += add + sum_j w[j] * yexp[j] (when given; the moe_add_slots order), then h = RMSNorm(x) * nw, written as f32
// (h_out, when given) and as Q8_1 (q8) exactly as native_quantize_q8_1 would.  One 1024-thread block, n % 1024 == 0.
// AXPY = true first does x = fma(alpha, ax, x) (axpy_kernel's y += w * x), then the same norm + Q8_1.
template <bool AXPY = false>
__global__ void __launch_bounds__(1024) add_norm_quant_kernel(float* __restrict__ x, const float* __restrict__ add,
                                                              const float* __restrict__ yexp,
                                                              const float* __restrict__ w, int k,
                                                              const float* __restrict__ nw, float* __restrict__ h_out,
                                                              Q81* __restrict__ q8, int n, float eps,
                                                              const float* __restrict__ ax = nullptr,
                                                              float alpha = 0.0f) {
    // one block per row (token): x, add, h_out by n; yexp by k rows of n; w by k; q8 by n / 32 blocks
    {
        const size_t r = blockIdx.x;
        x += r * n;
        if constexpr (AXPY) ax += r * n;
        if (add) add += r * n;
        if (yexp) yexp += r * k * n;
        if (w) w += r * k;
        if (h_out) h_out += r * n;
        q8 += r * (n / 32);
    }
    constexpr int kPer = 4;   // n <= 4096
    __shared__ float red[32];
    float v[kPer];
    float ss = 0.0f;
#pragma unroll
    for (int r = 0; r < kPer; ++r) {
        const int i = threadIdx.x + r * 1024;
        v[r] = 0.0f;
        if (i < n) {
            float xi = x[i];
            if constexpr (AXPY) {
                xi = __builtin_fmaf(alpha, ax[i], xi);
                x[i] = xi;
            }
            if (add) {
                float acc = add[i];
                for (int j = 0; j < k; ++j) acc += w[j] * yexp[(size_t) j * n + i];
                xi += acc;
                x[i] = xi;
            }
            v[r] = xi;
            ss += xi * xi;
        }
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float t = red[threadIdx.x];
        for (int o = 16; o > 0; o >>= 1) t += __shfl_xor(t, o, 32);
        if (threadIdx.x == 0) red[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(red[0] / (float) n + eps);
#pragma unroll
    for (int r = 0; r < kPer; ++r) {
        const int i = threadIdx.x + r * 1024;
        if (i < n) {   // whole warps: n % 32 == 0
            const float hi = v[r] * scale * nw[i];
            if (h_out) h_out[i] = hi;
            float amax = fabsf(hi), sum = hi;
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor(amax, o, 32));
                sum += __shfl_xor(sum, o, 32);
            }
            const float d = strata::kernels::q8_1_finite(amax / 127.0f);
            q8[i / 32].qs[i % 32] = strata::kernels::q8_1_quant(hi, d, amax);
            if (i % 32 == 0) q8[i / 32].ds = strata::kernels::q8_1_ds(d, sum);
        }
    }
}

// y[o] = w[o] . x in f32, one 128-thread block (4 warps) per row (decode router: 256 blocks)
__global__ void __launch_bounds__(128) f32_matvec_row_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                             float* __restrict__ y, int n_in) {
    __shared__ float red[4];
    const float4* row = reinterpret_cast<const float4*>(w + (size_t) blockIdx.x * n_in);
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in / 4; i += 128) {
        const float4 a = row[i], b = xv[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 32);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) y[blockIdx.x] = red[0] + red[1] + red[2] + red[3];
}

// the router's top-k in one warp: each lane holds n_expert/32 candidates, k rounds of a shuffle argmax
__global__ void __launch_bounds__(32) route_warp_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                        int32_t* __restrict__ ids, float* __restrict__ w, int k,
                                                        float scale) {
    constexpr int kPer = 8;   // 256 experts
    const int lane = threadIdx.x;
    float p[kPer], sel[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int e = j * 32 + lane;
        p[j] = 1.0f / (1.0f + expf(-logits[e]));
        sel[j] = p[j] + bias[e];
    }
    float wsum = 0.0f;
    int my_id = -1;
    float my_p = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < kPer; ++j) {
            const int e = j * 32 + lane;
            if (sel[j] > bv) { bv = sel[j]; bi = e; }   // ascending e per lane: ties keep the lower index
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        // the owner retires it and publishes its unbiased p
        float pe = 0.0f;
        if ((bi & 31) == lane) {
            const int j = bi >> 5;
#pragma unroll
            for (int jj = 0; jj < kPer; ++jj) if (jj == j) { pe = p[jj]; sel[jj] = -INFINITY; }
        }
        pe = __shfl(pe, bi & 31, 32);
        wsum += pe;
        if (lane == r) { my_id = bi; my_p = pe; }
    }
    const float denom = fmaxf(wsum, 6.103515625e-5f);
    if (lane < k) { ids[lane] = my_id; w[lane] = my_p / denom * scale; }
}

}  // namespace

void add_norm_quant(float* x, const float* add, const float* yexp, const float* w, int k, const float* norm_w,
                    float* h_out, void* q8, int n, float eps, void* stream, int n_rows) {
    if (n % 32 != 0 || n > 4096) { std::fprintf(stderr, "laguna: add_norm_quant: n %d\n", n); std::exit(1); }
    add_norm_quant_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(x, add, yexp, w, k, norm_w, h_out, (Q81*) q8, n,
                                                                      eps);
    check_launch("add_norm_quant");
}

void axpy_norm_quant(float* x, const float* ax, float alpha, const float* norm_w, float* h_out, void* q8, int n,
                     float eps, void* stream, int n_rows) {
    if (n % 32 != 0 || n > 4096) { std::fprintf(stderr, "laguna: axpy_norm_quant: n %d\n", n); std::exit(1); }
    add_norm_quant_kernel<true><<<n_rows, 1024, 0, (cudaStream_t) stream>>>(x, nullptr, nullptr, nullptr, 0, norm_w,
                                                                            h_out, (Q81*) q8, n, eps, ax, alpha);
    check_launch("axpy_norm_quant");
}

void f32_matvec_rows(const float* w, const float* x, float* y, int n_in, int n_out, void* stream) {
    f32_matvec_row_kernel<<<n_out, 128, 0, (cudaStream_t) stream>>>(w, x, y, n_in);
    check_launch("f32_matvec_rows");
}

void route_warp(const float* logits, const float* bias, int32_t* ids, float* w, int k, float scale, void* stream) {
    route_warp_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(logits, bias, ids, w, k, scale);
    check_launch("route_warp");
}

namespace {

// ---- multi-operation decode fusions (reducing per-layer kernel launch count) ----

// rows [0, H): q heads, RMSNorm(qn) + RoPE in place; rows [H, H+n_kv): k heads, RMSNorm(kn) + RoPE, then stored to
// kpool at *pos; rows [H+n_kv, H+2*n_kv): v heads, stored to vpool.  128 threads per row.
__global__ void __launch_bounds__(128) qkv_norm_rope_store_kernel(float* __restrict__ q, const float* __restrict__ kc,
                                                                  const float* __restrict__ vc,
                                                                  const float* __restrict__ qn,
                                                                  const float* __restrict__ kn, int H, int n_kv,
                                                                  float eps, int n_rot,
                                                                  const float* __restrict__ cos_tab,
                                                                  const float* __restrict__ sin_tab,
                                                                  const int* __restrict__ pos,
                                                                  uint16_t* __restrict__ kpool,
                                                                  uint16_t* __restrict__ vpool, int vt_stride,
                                                                  uint16_t* __restrict__ ksc, uint16_t* __restrict__ vsc) {
    __shared__ float sv[128];
    __shared__ float red[4];
    const int r = blockIdx.x, d = threadIdx.x;
    const int p = *pos + (int) blockIdx.y;    // token blockIdx.y of a multi-token step
    const size_t cell = (size_t) n_kv * 128;
    q += (size_t) blockIdx.y * H * 128;
    kc += (size_t) blockIdx.y * cell;
    vc += (size_t) blockIdx.y * cell;
    if (r >= H + n_kv) {   // V: no norm, no rotation
        const int h = r - H - n_kv;
        const int ps = kv_slot(p, vt_stride);
        const size_t off = vt_stride > 0 ? vt_index(h, ps, d, vt_stride) : (size_t) p * cell + h * 128 + d;
        if (vsc) {
            uint16_t sb;
            const int8_t qv = q8_block(vc[h * 128 + d], sb);
            reinterpret_cast<int8_t*>(vpool)[off] = qv;
            if (d % 32 == 0) vsc[vs_index(h, ps, d / 32, vt_stride)] = sb;
        } else {
            vpool[off] = strata::kernels::f16_from_f32(vc[h * 128 + d]);
        }
        return;
    }
    const bool is_q = r < H;
    const float* src = is_q ? q + (size_t) r * 128 : kc + (size_t) (r - H) * 128;
    const float* w = is_q ? qn : kn;
    const float v = src[d];
    float ss = v * v;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor(ss, o, 32);
    if (d % 32 == 0) red[d / 32] = ss;
    __syncthreads();
    const float mean = (red[0] + red[1] + red[2] + red[3]) / 128.0f;
    const float y = v * rsqrtf(mean + eps) * w[d];
    sv[d] = y;
    __syncthreads();
    const int half = n_rot / 2;
    float outv = y;
    if (d < half) {
        const float c = cos_tab[(size_t) p * half + d], sn = sin_tab[(size_t) p * half + d];
        outv = sv[d] * c - sv[d + half] * sn;
    } else if (d < n_rot) {
        const float c = cos_tab[(size_t) p * half + d - half], sn = sin_tab[(size_t) p * half + d - half];
        outv = sv[d - half] * sn + sv[d] * c;
    }
    if (is_q) {
        q[(size_t) r * 128 + d] = outv;
    } else if (ksc) {
        uint16_t sb;
        const int8_t qv = q8_block(outv, sb);
        const int ps = kv_slot(p, vt_stride);
        reinterpret_cast<int8_t*>(kpool)[(size_t) ps * cell + (r - H) * 128 + d] = qv;
        if (d % 32 == 0) ksc[ks_index(ps, r - H, d / 32, n_kv)] = sb;
    } else {
        kpool[(size_t) kv_slot(p, vt_stride) * cell + (r - H) * 128 + d] = strata::kernels::f16_from_f32(outv);
    }
}

// the split partials of one query token merged, times softplus(g[h]), quantized to Q8_1 for attn_output
__global__ void __launch_bounds__(128) attn_merge_gate_quant_kernel(const float* __restrict__ part_acc,
                                                                    const float* __restrict__ part_ml,
                                                                    const float* __restrict__ g, Q81* __restrict__ q8,
                                                                    int n_rows, int n_split) {
    // row = tok * n_rows + head; token tok's split z partial of head h at ((tok * n_split + z) * n_rows + h)
    const int row = blockIdx.x, d = threadIdx.x;
    const int tok = row / n_rows, hh = row % n_rows;
    const size_t base = (size_t) tok * n_split * n_rows + hh;
    float m = -INFINITY;
    for (int z = 0; z < n_split; ++z) m = fmaxf(m, part_ml[(base + (size_t) z * n_rows) * 2]);
    float num = 0.0f, den = 0.0f;
    if (m != -INFINITY) {
        for (int z = 0; z < n_split; ++z) {
            const size_t pr = base + (size_t) z * n_rows;
            const float mz = part_ml[pr * 2];
            if (mz == -INFINITY) continue;
            const float f = expf(mz - m);
            num += f * part_acc[pr * 128 + d];
            den += f * part_ml[pr * 2 + 1];
        }
    }
    const float gv = g[row];
    const float sp = gv > 20.0f ? gv : log1pf(expf(gv));
    const float hi = (den > 0.0f ? num / den : 0.0f) * sp;
    float amax = fabsf(hi), sum = hi;
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o, 32));
        sum += __shfl_xor(sum, o, 32);
    }
    const int i = row * 128 + d;
    const float dq = strata::kernels::q8_1_finite(amax / 127.0f);
    q8[i / 32].qs[i % 32] = strata::kernels::q8_1_quant(hi, dq, amax);
    if (i % 32 == 0) q8[i / 32].ds = strata::kernels::q8_1_ds(dq, sum);
}

// router logits (one 128-thread block per expert) and, in the last block to finish, the top-k
__global__ void __launch_bounds__(128) router_route_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                           float* __restrict__ logits, const float* __restrict__ bias,
                                                           int32_t* __restrict__ ids, float* __restrict__ wout,
                                                           int n_in, int k, float scale,
                                                           unsigned int* __restrict__ counter) {
    __shared__ float red[4];
    __shared__ bool last;
    const float4* row = reinterpret_cast<const float4*>(w + (size_t) blockIdx.x * n_in);
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc = 0.0f;
    if (n_in == 3072) {   // Laguna: 6 float4 per thread, all issued before the first use
        float4 a[6];
#pragma unroll
        for (int u = 0; u < 6; ++u) a[u] = row[threadIdx.x + 128 * u];
#pragma unroll
        for (int u = 0; u < 6; ++u) {
            const float4 b = xv[threadIdx.x + 128 * u];
            acc += a[u].x * b.x + a[u].y * b.y + a[u].z * b.z + a[u].w * b.w;
        }
    } else {
        for (int i = threadIdx.x; i < n_in / 4; i += 128) {
            const float4 a = row[i], b = xv[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 32);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        logits[blockIdx.x] = red[0] + red[1] + red[2] + red[3];
        __threadfence();
        last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x >= 32) return;
    __threadfence();
    // warp 0 of the last block: route_warp's body over the now complete logits
    constexpr int kPer = 8;
    const int lane = threadIdx.x;
    float p[kPer], sel[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int e = j * 32 + lane;
        const float lg = __builtin_nontemporal_load(&logits[e]);
        p[j] = 1.0f / (1.0f + expf(-lg));
        sel[j] = p[j] + bias[e];
    }
    float wsum = 0.0f;
    int my_id = -1;
    float my_p = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            if (sel[j] > bv) { bv = sel[j]; bi = j * 32 + lane; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float pe = 0.0f;
        if ((bi & 31) == lane) {
            const int j = bi >> 5;
#pragma unroll
            for (int jj = 0; jj < kPer; ++jj) if (jj == j) { pe = p[jj]; sel[jj] = -INFINITY; }
        }
        pe = __shfl(pe, bi & 31, 32);
        wsum += pe;
        if (lane == r) { my_id = bi; my_p = pe; }
    }
    const float denom = fmaxf(wsum, 6.103515625e-5f);
    if (lane < k) { ids[lane] = my_id; wout[lane] = my_p / denom * scale; }
    if (lane == 0) *counter = 0u;   // ready for the next layer / replay
}

}  // namespace

void qkv_norm_rope_store(float* q, const float* kc, const float* vc, const float* q_norm, const float* k_norm, int H,
                         int n_kv, float eps, int n_rot, const float* cos_tab, const float* sin_tab, const int* pos,
                         uint16_t* k_pool, uint16_t* v_pool, void* stream, int vt_stride, uint16_t* ksc,
                         uint16_t* vsc, int n_tok) {
    qkv_norm_rope_store_kernel<<<dim3(H + 2 * n_kv, n_tok), 128, 0, (cudaStream_t) stream>>>(q, kc, vc, q_norm, k_norm, H, n_kv, eps,
                                                                                n_rot, cos_tab, sin_tab, pos, k_pool,
                                                                                v_pool, vt_stride, ksc, vsc);
    check_launch("qkv_norm_rope_store");
}

void attn_merge_gate_quant(const float* part_acc, const float* part_ml, const float* g, void* q8, int n_head,
                           int n_split, void* stream, int n_tok) {
    attn_merge_gate_quant_kernel<<<n_tok * n_head, 128, 0, (cudaStream_t) stream>>>(part_acc, part_ml, g, (Q81*) q8,
                                                                                    n_head, n_split);
    check_launch("attn_merge_gate_quant");
}

void router_route(const float* w, const float* x, float* logits, const float* bias, int32_t* ids, float* wout,
                  int n_in, int n_expert, int k, float scale, unsigned int* counter, void* stream) {
    if (n_expert != 256 || n_in % 128 != 0 || k > 32) { std::fprintf(stderr, "laguna: router_route: shape\n"); std::exit(1); }
    router_route_kernel<<<n_expert, 128, 0, (cudaStream_t) stream>>>(w, x, logits, bias, ids, wout, n_in, k, scale,
                                                                     counter);
    check_launch("router_route");
}

namespace {

// ---- prefill attention on WMMA (RDNA3/3.5 v_wmma_f32_16x16x16_f16, wave32) ----
// Each wave takes 16 query rows of one KV head (rows = (token, head-in-group) pairs, G per token) and walks the keys
// in tiles of 16: S = Q K^T (8 WMMAs over the 128 dims), online softmax across the 16-lane halves, P relaid through
// LDS into an A fragment, O += P V (8 WMMAs, V transposed into LDS).  Fragment layout as src/prefill/wmma_gemm.cu:
// A/B lane l holds row/column l & 15 (16 values along K), lanes 16..31 duplicate; C lane l holds column l & 15 at
// rows 2i + (l >> 4).
typedef _Float16 h16_t __attribute__((ext_vector_type(16)));
typedef float f8_t __attribute__((ext_vector_type(8)));

template <int NW>
__global__ void __launch_bounds__(NW * 32) prompt_attn_wmma_kernel(const float* __restrict__ q,
                                                                   const uint16_t* __restrict__ kp,
                                                                   const uint16_t* __restrict__ vp,
                                                                   float* __restrict__ out, int nq, int q_pos0,
                                                                   int H, int G, int n_kv, int window, int vt_stride,
                                                                   const int* __restrict__ q_pos0_dev,
                                                                   float* __restrict__ part_acc,
                                                                   float* __restrict__ part_ml,
                                                                   const uint16_t* __restrict__ ksc,
                                                                   const uint16_t* __restrict__ vsc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    __shared__ __attribute__((aligned(32))) _Float16 sP[NW][16][16];
    const int lane = threadIdx.x % 32, wv = threadIdx.x / 32, kvh = blockIdx.y;
    const int li = lane & 15, hi = lane >> 4;
    const int Rtot = nq * G;
    if (q_pos0_dev) q_pos0 = *q_pos0_dev;
    const int r0 = (blockIdx.x * NW + wv) * 16;
    if (r0 >= Rtot) return;   // whole wave; no block-wide barriers below
    const float scale = 0.08838834764831845f;
    const size_t cell = (size_t) n_kv * 128;

    h16_t qa[8];
    {
        const int r = r0 + li;
        const bool ok = r < Rtot;
        const float* qr = q + ((size_t) (ok ? r / G : 0) * H + kvh * G + (ok ? r % G : 0)) * 128;
#pragma unroll
        for (int kc = 0; kc < 8; ++kc)
#pragma unroll
            for (int e = 0; e < 16; ++e) qa[kc][e] = ok ? (_Float16) (qr[kc * 16 + e] * scale) : (_Float16) 0.0f;
    }
    float m[8], l[8];
    int qp_v[8];
    bool live_v[8];
    f8_t o[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        m[v] = -INFINITY; l[v] = 0.0f;
        const int row = r0 + 2 * v + hi;
        live_v[v] = row < Rtot;
        qp_v[v] = q_pos0 + (live_v[v] ? row / G : 0);
#pragma unroll
        for (int c = 0; c < 8; ++c) o[c][v] = 0.0f;
    }
    const int rlast = min(r0 + 15, Rtot - 1);
    const int qmin = q_pos0 + r0 / G, qmax = q_pos0 + rlast / G;
    // tiles start on multiples of 16 so V^T's 16-key runs are 32-byte aligned; keys before the window are masked
    int kb = (window > 0 ? max(0, qmin - (window - 1)) : 0) & ~15;
    int ke = qmax + 1;
    if (gridDim.z > 1) {   // split-K (decode): this block's 16-aligned share of the keys
        const int chunk = ((ke - kb + (int) gridDim.z - 1) / (int) gridDim.z + 15) & ~15;
        kb += (int) blockIdx.z * chunk;
        ke = min(ke, kb + chunk);
    }
    for (int k0 = kb; k0 < ke; k0 += 16) {
        const int j = k0 + li;
        const bool jok = j < ke;
        // S = Q K^T.  k0 is 16-aligned and pools hold multiples of 16 positions, so a tile's slots are contiguous
        f8_t sacc = {0, 0, 0, 0, 0, 0, 0, 0};
        const int s0 = kv_slot(k0, vt_stride), js = jok ? s0 + li : 0;
        const uint16_t* kr = kp + (size_t) js * cell + kvh * 128;
#pragma unroll
        for (int kc = 0; kc < 8; ++kc) {
            h16_t kb16;
            if (jok && ksc) {   // q8: 16 int8 of key j's dims kc*16.., one scale per 32 dims
                const int8_t* k8 = reinterpret_cast<const int8_t*>(kp) + (size_t) js * cell + kvh * 128 + kc * 16;
                const uint4 raw = *reinterpret_cast<const uint4*>(k8);
                const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                const _Float16 sc = (_Float16) strata::kernels::f32_from_f16(ksc[ks_index(js, kvh, kc / 2, n_kv)]);
#pragma unroll
                for (int e = 0; e < 16; ++e) kb16[e] = (_Float16) rv[e] * sc;
            } else if (jok) {
                const uint4 a = *reinterpret_cast<const uint4*>(kr + kc * 16);
                const uint4 b = *reinterpret_cast<const uint4*>(kr + kc * 16 + 8);
                uint4 tmp[2] = {a, b};
                __builtin_memcpy(&kb16, tmp, 32);
            } else {
#pragma unroll
                for (int e = 0; e < 16; ++e) kb16[e] = (_Float16) 0.0f;
            }
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[kc], kb16, sacc);
        }
        // online softmax per row (rows 2v + hi; the row's 16 columns are the 16 lanes of this half)
        float alpha_v[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int key = k0 + li;
            const bool valid = live_v[v] && jok && key <= qp_v[v] && (window <= 0 || qp_v[v] - key < window);
            const float sv = valid ? sacc[v] : -INFINITY;
            float mx = sv;
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            const float m_new = fmaxf(m[v], mx);
            const float alpha = m_new == -INFINITY ? 1.0f : expf(m[v] - m_new);
            const float p = valid ? expf(sv - m_new) : 0.0f;
            float sum = p;
#pragma unroll
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
            l[v] = l[v] * alpha + sum;
            m[v] = m_new;
            alpha_v[v] = alpha;
            sP[wv][2 * v + hi][li] = (_Float16) p;
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        h16_t pa;
        __builtin_memcpy(&pa, &sP[wv][li][0], 32);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
#pragma unroll
            for (int v = 0; v < 8; ++v) o[c][v] *= alpha_v[v];
            // B = V tile: lane holds dim c*16 + li over the 16 keys, one 32-byte run of V^T (zeros past the written
            // cells; P is 0 there)
            h16_t vb;
            if (vsc) {   // q8: 16 int8 of dim c*16+li over the keys, and the 16 keys' scales for its 32-dim block
                const int8_t* v8 = reinterpret_cast<const int8_t*>(vp) + vt_index(kvh, s0, c * 16 + li, vt_stride);
                const uint4 raw = *reinterpret_cast<const uint4*>(v8);
                const uint16_t* sp_ = vsc + vs_index(kvh, s0, (c * 16 + li) / 32, vt_stride);
                const uint4 s0 = *reinterpret_cast<const uint4*>(sp_);
                const uint4 s1 = *reinterpret_cast<const uint4*>(sp_ + 8);
                const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                uint4 st[2] = {s0, s1};
                h16_t scl;
                __builtin_memcpy(&scl, st, 32);
#pragma unroll
                for (int e = 0; e < 16; ++e) vb[e] = (_Float16) rv[e] * scl[e];
            } else {
                const uint16_t* vc = vp + vt_index(kvh, s0, c * 16 + li, vt_stride);
                const uint4 va = *reinterpret_cast<const uint4*>(vc);
                const uint4 vb2 = *reinterpret_cast<const uint4*>(vc + 8);
                uint4 tmp[2] = {va, vb2};
                __builtin_memcpy(&vb, tmp, 32);
            }
            o[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, o[c]);
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    if (part_acc) {   // split: unnormalised O and (max, sum) per row, merged by attn_merge / attn_merge_gate_quant
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int row = r0 + 2 * v + hi;
            if (row < Rtot) {
                const size_t pr = ((size_t) blockIdx.z * nq + row / G) * H + kvh * G + row % G;
#pragma unroll
                for (int c = 0; c < 8; ++c) part_acc[pr * 128 + c * 16 + li] = o[c][v];
                if (li == 0) { part_ml[pr * 2] = m[v]; part_ml[pr * 2 + 1] = l[v]; }
            }
        }
        return;
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int row = r0 + 2 * v + hi;
        if (row < Rtot) {
            const int t = row / G, g = row % G;
            const float inv = l[v] > 0.0f ? 1.0f / l[v] : 0.0f;
            float* orow = out + ((size_t) t * H + kvh * G + g) * 128;
#pragma unroll
            for (int c = 0; c < 8; ++c) orow[c * 16 + li] = o[c][v] * inv;
        }
    }
#else
    (void) q; (void) kp; (void) vp; (void) out; (void) nq; (void) q_pos0; (void) H; (void) G; (void) n_kv; (void) window;
    (void) vt_stride; (void) q_pos0_dev; (void) part_acc; (void) part_ml; (void) ksc; (void) vsc;
    __builtin_trap();
#endif
}


// prompt_attn_wmma_kernel with the K / V tiles shared by the block.  The NW waves of a block hold
// consecutive 16-row query groups of one KV head, so their key spans overlap almost entirely; prompt_attn_wmma_kernel
// has every wave load and dequantize every 16-key tile itself.  Here the block loads NT consecutive 16-key tiles once
// (K as f16 rows, V^T as f16 dim runs, with the same q8 -> f16 expressions) into LDS and each wave runs exactly its own
// tiles of them, in order, with the same per-tile arithmetic: S, the online softmax, P relaid through LDS, O += P V.
// Keys a wave does not own inside a shared tile (past its last query) only meet masked scores, so every output is
// bitwise prompt_attn_wmma_kernel's (talus-laguna-attn-test).
template <int NW, int NT>
__global__ void __launch_bounds__(NW * 32) prompt_attn_wmma_shared_kernel(const float* __restrict__ q,
                                                                          const uint16_t* __restrict__ kp,
                                                                          const uint16_t* __restrict__ vp,
                                                                          float* __restrict__ out, int nq, int q_pos0,
                                                                          int H, int G, int n_kv, int window,
                                                                          int vt_stride,
                                                                          const int* __restrict__ q_pos0_dev,
                                                                          const uint16_t* __restrict__ ksc,
                                                                          const uint16_t* __restrict__ vsc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int KS = 128 + 16;   // K row stride (halves): 288 B, so 16 rows' 32-byte reads spread over the banks
    __shared__ __attribute__((aligned(32))) _Float16 sK[NT * 16][KS];
    __shared__ __attribute__((aligned(32))) _Float16 sV[NT][128][16];
    __shared__ __attribute__((aligned(32))) _Float16 sP[NW][16][16];
    const int tid = threadIdx.x, lane = tid % 32, wv = tid / 32, kvh = blockIdx.y;
    const int li = lane & 15, hi = lane >> 4;
    const int Rtot = nq * G;
    if (q_pos0_dev) q_pos0 = *q_pos0_dev;
    const int R0 = blockIdx.x * NW * 16;
    if (R0 >= Rtot) return;   // the whole block
    const int r0 = R0 + wv * 16;
    const bool active = r0 < Rtot;
    const float scale = 0.08838834764831845f;
    const size_t cell = (size_t) n_kv * 128;

    h16_t qa[8];
    {
        const int r = r0 + li;
        const bool ok = r < Rtot;
        const float* qr = q + ((size_t) (ok ? r / G : 0) * H + kvh * G + (ok ? r % G : 0)) * 128;
#pragma unroll
        for (int kc = 0; kc < 8; ++kc)
#pragma unroll
            for (int e = 0; e < 16; ++e) qa[kc][e] = ok ? (_Float16) (qr[kc * 16 + e] * scale) : (_Float16) 0.0f;
    }
    float m[8], l[8];
    int qp_v[8];
    bool live_v[8];
    f8_t o[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        m[v] = -INFINITY; l[v] = 0.0f;
        const int row = r0 + 2 * v + hi;
        live_v[v] = row < Rtot;
        qp_v[v] = q_pos0 + (live_v[v] ? row / G : 0);
#pragma unroll
        for (int c = 0; c < 8; ++c) o[c][v] = 0.0f;
    }
    // this wave's key span (as prompt_attn_wmma_kernel) and the block's (its first and last waves')
    const int rlast = min(r0 + 15, Rtot - 1);
    const int qmin = q_pos0 + r0 / G, qmax = q_pos0 + rlast / G;
    const int kb = (window > 0 ? max(0, qmin - (window - 1)) : 0) & ~15;
    const int ke = qmax + 1;
    const int bqmin = q_pos0 + R0 / G, bqmax = q_pos0 + min(R0 + NW * 16 - 1, Rtot - 1) / G;
    const int KB = (window > 0 ? max(0, bqmin - (window - 1)) : 0) & ~15;
    const int KE = bqmax + 1;
    for (int kbase = KB; kbase < KE; kbase += NT * 16) {
        __syncthreads();   // the previous stage's tiles are no longer read
        // K: NT*16 keys x 8 chunks of 16 dims
        for (int u = tid; u < NT * 16 * 8; u += NW * 32) {
            const int jj = u >> 3, kc = u & 7, j = kbase + jj;
            h16_t kb16;
            if (j < KE) {
                const int js = kv_slot(j & ~15, vt_stride) + (j & 15);
                if (ksc) {
                    const int8_t* k8 = reinterpret_cast<const int8_t*>(kp) + (size_t) js * cell + kvh * 128 + kc * 16;
                    const uint4 raw = *reinterpret_cast<const uint4*>(k8);
                    const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                    const _Float16 sc = (_Float16) strata::kernels::f32_from_f16(ksc[ks_index(js, kvh, kc / 2, n_kv)]);
#pragma unroll
                    for (int e = 0; e < 16; ++e) kb16[e] = (_Float16) rv[e] * sc;
                } else {
                    const uint16_t* kr = kp + (size_t) js * cell + kvh * 128;
                    const uint4 a = *reinterpret_cast<const uint4*>(kr + kc * 16);
                    const uint4 b = *reinterpret_cast<const uint4*>(kr + kc * 16 + 8);
                    uint4 tmp[2] = {a, b};
                    __builtin_memcpy(&kb16, tmp, 32);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 16; ++e) kb16[e] = (_Float16) 0.0f;
            }
            *reinterpret_cast<h16_t*>(&sK[jj][kc * 16]) = kb16;
        }
        // V^T: NT tiles x 128 dims, 16 keys each
        for (int u = tid; u < NT * 128; u += NW * 32) {
            const int n = u >> 7, d = u & 127, k0 = kbase + n * 16;
            if (k0 >= KE) continue;
            const int s0 = kv_slot(k0, vt_stride);
            h16_t vb;
            if (vsc) {
                const int8_t* v8 = reinterpret_cast<const int8_t*>(vp) + vt_index(kvh, s0, d, vt_stride);
                const uint4 raw = *reinterpret_cast<const uint4*>(v8);
                const uint16_t* sp_ = vsc + vs_index(kvh, s0, d / 32, vt_stride);
                const uint4 sc0 = *reinterpret_cast<const uint4*>(sp_);
                const uint4 sc1 = *reinterpret_cast<const uint4*>(sp_ + 8);
                const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                uint4 st[2] = {sc0, sc1};
                h16_t scl;
                __builtin_memcpy(&scl, st, 32);
#pragma unroll
                for (int e = 0; e < 16; ++e) vb[e] = (_Float16) rv[e] * scl[e];
            } else {
                const uint16_t* vc = vp + vt_index(kvh, s0, d, vt_stride);
                const uint4 va = *reinterpret_cast<const uint4*>(vc);
                const uint4 vb2 = *reinterpret_cast<const uint4*>(vc + 8);
                uint4 tmp[2] = {va, vb2};
                __builtin_memcpy(&vb, tmp, 32);
            }
            *reinterpret_cast<h16_t*>(&sV[n][d][0]) = vb;
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll 1
        for (int n = 0; n < NT; ++n) {
            const int k0 = kbase + n * 16;
            if (k0 < kb || k0 >= ke) continue;   // not one of this wave's tiles
            const int j = k0 + li;
            const bool jok = j < ke;
            f8_t sacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int kc = 0; kc < 8; ++kc) {
                h16_t kb16 = *reinterpret_cast<const h16_t*>(&sK[n * 16 + li][kc * 16]);
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[kc], kb16, sacc);
            }
            float alpha_v[8];
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int key = k0 + li;
                const bool valid = live_v[v] && jok && key <= qp_v[v] && (window <= 0 || qp_v[v] - key < window);
                const float sv = valid ? sacc[v] : -INFINITY;
                float mx = sv;
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
                const float m_new = fmaxf(m[v], mx);
                const float alpha = m_new == -INFINITY ? 1.0f : expf(m[v] - m_new);
                const float p = valid ? expf(sv - m_new) : 0.0f;
                float sum = p;
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
                l[v] = l[v] * alpha + sum;
                m[v] = m_new;
                alpha_v[v] = alpha;
                sP[wv][2 * v + hi][li] = (_Float16) p;
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            h16_t pa;
            __builtin_memcpy(&pa, &sP[wv][li][0], 32);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
#pragma unroll
                for (int v = 0; v < 8; ++v) o[c][v] *= alpha_v[v];
                const h16_t vb = *reinterpret_cast<const h16_t*>(&sV[n][c * 16 + li][0]);
                o[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, o[c]);
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        }
    }
    if (!active) return;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int row = r0 + 2 * v + hi;
        if (row < Rtot) {
            const int t = row / G, g = row % G;
            const float inv = l[v] > 0.0f ? 1.0f / l[v] : 0.0f;
            float* orow = out + ((size_t) t * H + kvh * G + g) * 128;
#pragma unroll
            for (int c = 0; c < 8; ++c) orow[c * 16 + li] = o[c][v] * inv;
        }
    }
#else
    (void) q; (void) kp; (void) vp; (void) out; (void) nq; (void) q_pos0; (void) H; (void) G; (void) n_kv; (void) window;
    (void) vt_stride; (void) q_pos0_dev; (void) ksc; (void) vsc;
    __builtin_trap();
#endif
}

// longctx: prompt_attn_wmma_shared_kernel computed transposed.  The same registers feed the matrix cores with the
// operands swapped: S^T = K Q^T (A = the K fragment, B = the Q fragment) and O^T = V^T P^T (A = the V^T fragment,
// B = P^T), so a lane holds one query row (lanes l and l ^ 16 hold the even / odd keys of a 16-key tile, or the even /
// odd dims of a 16-dim O tile) instead of one key across 8 rows.  The online softmax then runs mostly inside the lane:
// the row maximum is 7 in-lane fmaxf + one exchange, the row sum is the same butterfly tree as the 16-lane shuffles
// (pairs k ^ 8, k ^ 4, k ^ 2 sit in one lane, k ^ 1 in the partner), one expf for alpha instead of eight, no P relay
// through LDS, and O is rescaled only when some row's maximum moved.  Every value is computed by the same expression
// on the same operands in the same order (a matrix-core output element is the same dot product either way round), so
// the output is bitwise prompt_attn_wmma_shared_kernel's (lagattn / talus-laguna-attn-test).
template <int NW, int NT>
__global__ void __launch_bounds__(NW * 32) prompt_attn_wmma_shared_t_kernel(const float* __restrict__ q,
                                                                            const uint16_t* __restrict__ kp,
                                                                            const uint16_t* __restrict__ vp,
                                                                            float* __restrict__ out, int nq, int q_pos0,
                                                                            int H, int G, int n_kv, int window,
                                                                            int vt_stride,
                                                                            const int* __restrict__ q_pos0_dev,
                                                                            const uint16_t* __restrict__ ksc,
                                                                            const uint16_t* __restrict__ vsc) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)
    constexpr int KS = 128 + 16;
    __shared__ __attribute__((aligned(32))) _Float16 sK[NT * 16][KS];
    __shared__ __attribute__((aligned(32))) _Float16 sV[NT][128][16];
    const int tid = threadIdx.x, lane = tid % 32, wv = tid / 32, kvh = blockIdx.y;
    const int li = lane & 15, hi = lane >> 4;
    const unsigned sel_lo = hi ? 0x01000504u : 0x05040100u, sel_hi = hi ? 0x03020706u : 0x07060302u;
    const int Rtot = nq * G;
    if (q_pos0_dev) q_pos0 = *q_pos0_dev;
    const int R0 = blockIdx.x * NW * 16;
    if (R0 >= Rtot) return;
    const int r0 = R0 + wv * 16;
    const bool active = r0 < Rtot;
    const float scale = 0.08838834764831845f;
    const size_t cell = (size_t) n_kv * 128;

    h16_t qa[8];
    {
        const int r = r0 + li;
        const bool ok = r < Rtot;
        const float* qr = q + ((size_t) (ok ? r / G : 0) * H + kvh * G + (ok ? r % G : 0)) * 128;
#pragma unroll
        for (int kc = 0; kc < 8; ++kc)
#pragma unroll
            for (int e = 0; e < 16; ++e) qa[kc][e] = ok ? (_Float16) (qr[kc * 16 + e] * scale) : (_Float16) 0.0f;
    }
    // this lane's query row r0 + li
    const int row = r0 + li;
    const bool live = row < Rtot;
    const int qp = q_pos0 + (live ? row / G : 0);
    float m = -INFINITY, l = 0.0f;
    f8_t o[8];
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
        for (int v = 0; v < 8; ++v) o[c][v] = 0.0f;
    const int rlast = min(r0 + 15, Rtot - 1);
    const int qmin = q_pos0 + r0 / G, qmax = q_pos0 + rlast / G;
    const int kb = (window > 0 ? max(0, qmin - (window - 1)) : 0) & ~15;
    const int ke = qmax + 1;
    const int bqmin = q_pos0 + R0 / G, bqmax = q_pos0 + min(R0 + NW * 16 - 1, Rtot - 1) / G;
    const int KB = (window > 0 ? max(0, bqmin - (window - 1)) : 0) & ~15;
    const int KE = bqmax + 1;
    for (int kbase = KB; kbase < KE; kbase += NT * 16) {
        __syncthreads();
        for (int u = tid; u < NT * 16 * 8; u += NW * 32) {
            const int jj = u >> 3, kc = u & 7, j = kbase + jj;
            h16_t kb16;
            if (j < KE) {
                const int js = kv_slot(j & ~15, vt_stride) + (j & 15);
                if (ksc) {
                    const int8_t* k8 = reinterpret_cast<const int8_t*>(kp) + (size_t) js * cell + kvh * 128 + kc * 16;
                    const uint4 raw = *reinterpret_cast<const uint4*>(k8);
                    const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                    const _Float16 sc = (_Float16) strata::kernels::f32_from_f16(ksc[ks_index(js, kvh, kc / 2, n_kv)]);
#pragma unroll
                    for (int e = 0; e < 16; ++e) kb16[e] = (_Float16) rv[e] * sc;
                } else {
                    const uint16_t* kr = kp + (size_t) js * cell + kvh * 128;
                    const uint4 a = *reinterpret_cast<const uint4*>(kr + kc * 16);
                    const uint4 b = *reinterpret_cast<const uint4*>(kr + kc * 16 + 8);
                    uint4 tmp[2] = {a, b};
                    __builtin_memcpy(&kb16, tmp, 32);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 16; ++e) kb16[e] = (_Float16) 0.0f;
            }
            *reinterpret_cast<h16_t*>(&sK[jj][kc * 16]) = kb16;
        }
        for (int u = tid; u < NT * 128; u += NW * 32) {
            const int n = u >> 7, d = u & 127, k0 = kbase + n * 16;
            if (k0 >= KE) continue;
            const int s0 = kv_slot(k0, vt_stride);
            h16_t vb;
            if (vsc) {
                const int8_t* v8 = reinterpret_cast<const int8_t*>(vp) + vt_index(kvh, s0, d, vt_stride);
                const uint4 raw = *reinterpret_cast<const uint4*>(v8);
                const uint16_t* sp_ = vsc + vs_index(kvh, s0, d / 32, vt_stride);
                const uint4 sc0 = *reinterpret_cast<const uint4*>(sp_);
                const uint4 sc1 = *reinterpret_cast<const uint4*>(sp_ + 8);
                const int8_t* rv = reinterpret_cast<const int8_t*>(&raw);
                uint4 st[2] = {sc0, sc1};
                h16_t scl;
                __builtin_memcpy(&scl, st, 32);
#pragma unroll
                for (int e = 0; e < 16; ++e) vb[e] = (_Float16) rv[e] * scl[e];
            } else {
                const uint16_t* vc = vp + vt_index(kvh, s0, d, vt_stride);
                const uint4 va = *reinterpret_cast<const uint4*>(vc);
                const uint4 vb2 = *reinterpret_cast<const uint4*>(vc + 8);
                uint4 tmp[2] = {va, vb2};
                __builtin_memcpy(&vb, tmp, 32);
            }
            *reinterpret_cast<h16_t*>(&sV[n][d][0]) = vb;
        }
        __syncthreads();
        if (!active) continue;
#pragma unroll 1
        for (int n = 0; n < NT; ++n) {
            const int k0 = kbase + n * 16;
            if (k0 < kb || k0 >= ke) continue;
            // S^T: element i of this lane = query li, key k0 + 2 i + hi
            f8_t sacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int kc = 0; kc < 8; ++kc) {
                const h16_t kb16 = *reinterpret_cast<const h16_t*>(&sK[n * 16 + li][kc * 16]);
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kb16, qa[kc], sacc);
            }
            float sv[8];
            bool valid[8];
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key = k0 + 2 * i + hi;
                valid[i] = live && key < ke && key <= qp && (window <= 0 || qp - key < window);
                sv[i] = valid[i] ? sacc[i] : -INFINITY;
                mx = fmaxf(mx, sv[i]);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            const float m_new = fmaxf(m, mx);
            const float alpha = m_new == -INFINITY ? 1.0f : expf(m - m_new);
            float p[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) p[i] = valid[i] ? expf(sv[i] - m_new) : 0.0f;
            // the 16-lane butterfly's tree: keys k ^ 8, k ^ 4, k ^ 2 are elements i ^ 4, i ^ 2, i ^ 1 here
            const float b0 = p[0] + p[4], b1 = p[1] + p[5], b2 = p[2] + p[6], b3 = p[3] + p[7];
            const float c0 = b0 + b2, c1 = b1 + b3;
            const float d0 = c0 + c1;
            const float sum = d0 + __shfl_xor(d0, 16, 32);
            l = l * alpha + sum;
            m = m_new;
            // P^T as the B operand: lane (query li) holds its 16 keys in order; the odd / even ones are in lane ^ 16
            union { h16_t v; unsigned u[8]; } pb;
#pragma unroll
            for (int mm = 0; mm < 4; ++mm) {
                union { _Float16 h[2]; unsigned u; } own;
                own.h[0] = (_Float16) p[2 * mm];
                own.h[1] = (_Float16) p[2 * mm + 1];
                const unsigned par = (unsigned) __shfl_xor((int) own.u, 16, 32);
                pb.u[2 * mm] = __builtin_amdgcn_perm(par, own.u, sel_lo);
                pb.u[2 * mm + 1] = __builtin_amdgcn_perm(par, own.u, sel_hi);
            }
            const bool rescale = __any(alpha != 1.0f);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                if (rescale)
#pragma unroll
                    for (int v = 0; v < 8; ++v) o[c][v] *= alpha;
                const h16_t vb = *reinterpret_cast<const h16_t*>(&sV[n][c * 16 + li][0]);
                o[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vb, pb.v, o[c]);
            }
        }
    }
    if (!active || !live) return;
    {
        const int t = row / G, g = row % G;
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* orow = out + ((size_t) t * H + kvh * G + g) * 128;
#pragma unroll
        for (int c = 0; c < 8; ++c)
#pragma unroll
            for (int v = 0; v < 8; ++v) orow[c * 16 + 2 * v + hi] = o[c][v] * inv;
    }
#else
    (void) q; (void) kp; (void) vp; (void) out; (void) nq; (void) q_pos0; (void) H; (void) G; (void) n_kv; (void) window;
    (void) vt_stride; (void) q_pos0_dev; (void) ksc; (void) vsc;
    __builtin_trap();
#endif
}

// -1: TALUS_LAGUNA_ATTN (default 4 = shared tiles, transposed), 2 = shared tiles, 3 = shared tiles at every depth,
// 1 = per-wave tiles.  All bitwise the same.
int g_prompt_attn_version = -1;
int prompt_attn_version() {
    static const int env = [] { const char* e = std::getenv("TALUS_LAGUNA_ATTN"); return e ? std::atoi(e) : 4; }();
    return g_prompt_attn_version < 0 ? env : g_prompt_attn_version;
}

}  // namespace

void prompt_attn_set_version(int v) { g_prompt_attn_version = v; }

void prompt_attn_wmma(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, float* out, int nq, int q_pos0,
                      int n_head, int n_kv, int window, void* stream, int vt_stride, const int* q_pos0_dev, int n_split,
                      float* part_acc, float* part_ml, const uint16_t* ksc, const uint16_t* vsc) {
    if (vt_stride <= 0 || vt_stride % 16 != 0) { std::fprintf(stderr, "laguna: prompt_attn_wmma: V must be transposed, stride %% 16\n"); std::exit(1); }
    const int G = n_head / n_kv;
    const int rows = nq * G;
    if (n_split > 1) {   // decode: one wave per (KV head, key chunk)
        prompt_attn_wmma_kernel<1><<<dim3((rows + 15) / 16, n_kv, n_split), 32, 0, (cudaStream_t) stream>>>(
            q, k_pool, v_pool, out, nq, q_pos0, n_head, G, n_kv, window, vt_stride, q_pos0_dev, part_acc, part_ml, ksc, vsc);
    } else if (prompt_attn_version() >= 2 &&
               (ksc != nullptr || q_pos0_dev != nullptr || q_pos0 + nq > 4096 || prompt_attn_version() == 3 ||
                prompt_attn_version() >= 5)) {
        // (shared tiles: q8 KV x1.18-1.42 at every depth; f16 KV only pays past a few K keys: x0.92-1.0 at depth 0,
        // x1.15-1.93 at 6-30K - talus-laguna-attn-test)
        constexpr int NW = 8, NT = 4;
        const dim3 grid((rows + 16 * NW - 1) / (16 * NW), n_kv);
        if (prompt_attn_version() >= 4)
            prompt_attn_wmma_shared_t_kernel<NW, NT><<<grid, NW * 32, 0, (cudaStream_t) stream>>>(
                q, k_pool, v_pool, out, nq, q_pos0, n_head, G, n_kv, window, vt_stride, q_pos0_dev, ksc, vsc);
        else
            prompt_attn_wmma_shared_kernel<NW, NT><<<grid, NW * 32, 0, (cudaStream_t) stream>>>(
                q, k_pool, v_pool, out, nq, q_pos0, n_head, G, n_kv, window, vt_stride, q_pos0_dev, ksc, vsc);
    } else {
        constexpr int NW = 4;
        const dim3 grid((rows + 16 * NW - 1) / (16 * NW), n_kv);
        prompt_attn_wmma_kernel<NW><<<grid, NW * 32, 0, (cudaStream_t) stream>>>(q, k_pool, v_pool, out, nq, q_pos0,
                                                                                 n_head, G, n_kv, window, vt_stride,
                                                                                 q_pos0_dev, nullptr, nullptr, ksc, vsc);
    }
    check_launch("prompt_attn_wmma");
}

namespace {

// ---- prefill routing on the device (no host round trip per MoE layer) ----

// one warp per token: route_warp's top-k -> sel_e[t*k + j], sel_w[t*k + j]
__global__ void __launch_bounds__(256) route_batch_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                                          int32_t* __restrict__ sel_e, float* __restrict__ sel_w,
                                                          int n_tok, int k, float scale) {
    constexpr int kPer = 8;   // 256 experts
    const int t = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (t >= n_tok) return;
    const float* lg = logits + (size_t) t * 256;
    float p[kPer], sel[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        const int e = j * 32 + lane;
        p[j] = 1.0f / (1.0f + expf(-lg[e]));
        sel[j] = p[j] + bias[e];
    }
    float wsum = 0.0f;
    int my_id = -1;
    float my_p = 0.0f;
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = 1 << 30;
#pragma unroll
        for (int j = 0; j < kPer; ++j)
            if (sel[j] > bv) { bv = sel[j]; bi = j * 32 + lane; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o, 32);
            const int oi = __shfl_xor(bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        float pe = 0.0f;
        if ((bi & 31) == lane) {
            const int j = bi >> 5;
#pragma unroll
            for (int jj = 0; jj < kPer; ++jj) if (jj == j) { pe = p[jj]; sel[jj] = -INFINITY; }
        }
        pe = __shfl(pe, bi & 31, 32);
        wsum += pe;
        if (lane == r) { my_id = bi; my_p = pe; }
    }
    const float denom = fmaxf(wsum, 6.103515625e-5f);
    if (lane < k) { sel_e[t * k + lane] = my_id; sel_w[t * k + lane] = my_p / denom * scale; }
}

// block e: how many entries picked expert e
__global__ void __launch_bounds__(256) expert_count_kernel(const int32_t* __restrict__ sel_e, int n, int* __restrict__ cnt) {
    __shared__ int red[8];
    const int e = blockIdx.x;
    int c = 0;
    for (int i = threadIdx.x; i < n; i += 256) c += sel_e[i] == e;
    for (int o = 16; o > 0; o >>= 1) c += __shfl_xor(c, o, 32);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0;
        for (int w = 0; w < 8; ++w) t += red[w];
        cnt[e] = t;
    }
}

// block e: its rows start at sum(cnt[<e]); entries in ascending (token, slot) order -> rows start, start+1, ...
__global__ void __launch_bounds__(256) expert_scatter_kernel(const int32_t* __restrict__ sel_e, int n, int k,
                                                             const int* __restrict__ cnt, int32_t* __restrict__ bounds,
                                                             int32_t* __restrict__ rtok, int32_t* __restrict__ smap,
                                                             int n_expert) {
    __shared__ int wsum[8];
    __shared__ int base;
    const int e = blockIdx.x;
    if (threadIdx.x == 0) {
        int st = 0;
        for (int i = 0; i < e; ++i) st += cnt[i];
        base = st;
        bounds[e] = st;
        if (e == n_expert - 1) bounds[n_expert] = st + cnt[e];
    }
    __syncthreads();
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    for (int i0 = 0; i0 < n; i0 += 256) {
        const int i = i0 + threadIdx.x;
        const bool hit = i < n && sel_e[i] == e;
        const unsigned long long ball = __ballot(hit);
        const int before = __popcll(ball & ((1ull << lane) - 1ull));
        if (lane == 0) wsum[w] = __popcll(ball);
        __syncthreads();
        int off = 0;
        for (int ww = 0; ww < w; ++ww) off += wsum[ww];
        int tot = 0;
        for (int ww = 0; ww < 8; ++ww) tot += wsum[ww];
        if (hit) {
            const int r = base + off + before;
            rtok[r] = i / k;
            smap[i] = r;
        }
        __syncthreads();
        if (threadIdx.x == 0) base += tot;
        __syncthreads();
    }
}

}  // namespace

void route_group(const float* logits, const float* bias, int n_tok, int k, float scale, int32_t* sel_e, float* sel_w,
                 int* cnt, int32_t* bounds, int32_t* rtok, int32_t* smap, void* stream) {
    const cudaStream_t st = (cudaStream_t) stream;
    route_batch_kernel<<<(n_tok + 7) / 8, 256, 0, st>>>(logits, bias, sel_e, sel_w, n_tok, k, scale);
    check_launch("route_batch");
    expert_count_kernel<<<256, 256, 0, st>>>(sel_e, n_tok * k, cnt);
    check_launch("expert_count");
    expert_scatter_kernel<<<256, 256, 0, st>>>(sel_e, n_tok * k, k, cnt, bounds, rtok, smap, 256);
    check_launch("expert_scatter");
}

namespace {

// V rows (nb tokens x n_kv x 128, f32) into a transposed f16 pool [kv_head][dim][pos] at positions p0..
__global__ void v_store_t_kernel(const float* __restrict__ vc, uint16_t* __restrict__ vpool, int nb, int n_kv, int p0,
                                 int stride) {
    // thread = (dim row, token): consecutive threads take consecutive tokens, so the f16 writes coalesce
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;   // kv_head * 128 + dim
    if (t >= nb) return;
    vpool[vt_index(row / 128, kv_slot(p0 + t, stride), row % 128, stride)] =
        strata::kernels::f16_from_f32(vc[(size_t) t * n_kv * 128 + row]);
}
}  // namespace

void v_store_t(const float* vc, uint16_t* v_pool, int nb, int n_kv, int p0, int stride, void* stream) {
    v_store_t_kernel<<<dim3((nb + 255) / 256, n_kv * 128), 256, 0, (cudaStream_t) stream>>>(vc, v_pool, nb, n_kv, p0, stride);
    check_launch("v_store_t");
}


namespace {

// prefill: nb tokens' K rows (nb x n_kv x 128 f32, normed and rotated) and V rows into the q8 pools at p0..; one
// 128-thread block per (token, kv head)
__global__ void __launch_bounds__(128) kv_store_q8_kernel(const float* __restrict__ kc, const float* __restrict__ vc,
                                                          int8_t* __restrict__ kpool, int8_t* __restrict__ vpool,
                                                          uint16_t* __restrict__ ksc, uint16_t* __restrict__ vsc,
                                                          int n_kv, int p0, int stride) {
    const int t = blockIdx.x / n_kv, h = blockIdx.x % n_kv, d = threadIdx.x, p = kv_slot(p0 + t, stride);
    const size_t cell = (size_t) n_kv * 128;
    uint16_t sb;
    const int8_t qk = q8_block(kc[(size_t) t * cell + h * 128 + d], sb);
    kpool[(size_t) p * cell + h * 128 + d] = qk;
    if (d % 32 == 0) ksc[ks_index(p, h, d / 32, n_kv)] = sb;
    const int8_t qv = q8_block(vc[(size_t) t * cell + h * 128 + d], sb);
    vpool[vt_index(h, p, d, stride)] = qv;
    if (d % 32 == 0) vsc[vs_index(h, p, d / 32, stride)] = sb;
}
}  // namespace

void kv_store_q8(const float* kc, const float* vc, void* k_pool, void* v_pool, uint16_t* k_scale, uint16_t* v_scale,
                 int nb, int n_kv, int p0, int stride, void* stream) {
    kv_store_q8_kernel<<<nb * n_kv, 128, 0, (cudaStream_t) stream>>>(kc, vc, (int8_t*) k_pool, (int8_t*) v_pool, k_scale,
                                                                    v_scale, n_kv, p0, stride);
    check_launch("kv_store_q8");
}

namespace {

// ---- decode attention (one query token) with the next key tile prefetched into registers ----
// grid (n_split, n_kv), 256 threads; V tile-transposed; Q8: int8 K/V + scales.  KT keys per tile (32 or 64: lanes take
// KT/32 keys each).  Tiles start 16-aligned (keys before the window are masked) and are read whole: the pools carry 64
// positions of slack past the context.
template <bool Q8, int KT>
__global__ void __launch_bounds__(256) decode_attn_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kp,
                                                          const uint16_t* __restrict__ vp,
                                                          const uint16_t* __restrict__ ksc,
                                                          const uint16_t* __restrict__ vsc, int H, int G, int n_kv,
                                                          int window, int stride, const int* __restrict__ pos_dev,
                                                          float* __restrict__ part_acc, float* __restrict__ part_ml,
                                                          const int* __restrict__ sel, int sel_k, int sel_per_kv,
                                                          int sel_sh, int n_split, int n_kvb, int tokmajor) {
    constexpr int MaxR = kMaxG, PvRows = 3, KPL = KT / 32;
    constexpr int KF = Q8 ? KT * 8 / 256 : KT * 16 / 256;   // K fetch tasks per thread (16 B each)
    constexpr int VF = KT * 16 / 256;                         // V fetch tasks per thread (8 keys of one dim)
    __shared__ h2_t sq[MaxR][kD2];
    __shared__ uint32_t sk[KT][kD2 + 1];
    __shared__ uint16_t sv16[KT][kD];
    __shared__ float ss[MaxR][KT + 1];
    __shared__ float sm[MaxR], sl[MaxR], sa[MaxR];
    // tokmajor = 1 numbers the blocks token-fastest (1-D grid: token, then KV head, then split), so the
    // blocks of a window's tokens that read the same split of the same KV head run side by side and the later ones
    // find the K / V lines in L2 / MALL instead of re-streaming them from memory.  Only the block -> (split, KV head,
    // token) mapping changes; every block's arithmetic is the same, so the partials are bitwise the same.
    int bsplit, kvh, qt;
    if (tokmajor) {
        const int nt = (int) gridDim.x / (n_split * n_kvb), b = (int) blockIdx.x;
        qt = b % nt;
        kvh = (b / nt) % n_kvb;
        bsplit = b / (nt * n_kvb);
    } else {
        bsplit = (int) blockIdx.x;
        kvh = (int) blockIdx.y;
        qt = (int) blockIdx.z;                    // query token of a multi-token step (its keys: up to itself)
    }
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int R = G;
    const int qp = *pos_dev + qt;
    q += (size_t) qt * H * kD;
    const int wstart = window > 0 ? max(0, qp - (window - 1)) : 0;
    // block-sparse (sel set): the keys are this token's sel_k selected blocks of 2^sel_sh keys, walked as one virtual
    // range (each key mapped to its position: a tile may span blocks); otherwise the dense range
    const int qend = qp + 1;
    const int* selt = sel ? sel + (size_t) (sel_per_kv ? qt * n_kvb + kvh : qt) * sel_k : nullptr;
    int kb = selt ? 0 : wstart & ~15, ke = selt ? sel_k << sel_sh : qend;
    auto phys = [&](int v) { return selt ? (selt[v >> sel_sh] << sel_sh) | (v & ((1 << sel_sh) - 1)) : v; };
    {
        const int chunk = ((ke - kb + n_split - 1) / n_split + KT - 1) / KT * KT;
        kb += bsplit * chunk;
        ke = min(ke, kb + chunk);
    }
    const float scale = 0.08838834764831845f;
    for (int i = tid; i < R * kD2; i += 256) {
        const float* qr = q + (size_t) (kvh * G + i / kD2) * kD + 2 * (i % kD2);
        sq[i / kD2][i % kD2] = h2_t{(_Float16) (qr[0] * scale), (_Float16) (qr[1] * scale)};
    }
    if (tid < R) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }
    const size_t cell = (size_t) n_kv * kD;
    const int8_t* k8 = reinterpret_cast<const int8_t*>(kp);
    const int8_t* v8 = reinterpret_cast<const int8_t*>(vp);
    uint4 kr[KF], vr[VF], vs[VF];
    uint2 v8r[VF];
    uint16_t ksr[KF];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int u = 0; u < KF; ++u) {
            const int i = tid + 256 * u;
            if constexpr (Q8) {
                const int j = i / 8, c16 = i % 8;
                const int ps = kv_slot(phys(k0 + j), stride);
                kr[u] = *reinterpret_cast<const uint4*>(k8 + (size_t) ps * cell + kvh * kD + c16 * 16);
                ksr[u] = ksc[ks_index(ps, kvh, c16 / 2, n_kv)];
            } else {
                const int j = i / 16, c = i % 16;
                kr[u] = *reinterpret_cast<const uint4*>(kp + (size_t) kv_slot(phys(k0 + j), stride) * cell + kvh * kD + c * 8);
            }
        }
#pragma unroll
        for (int u = 0; u < VF; ++u) {
            const int i = tid + 256 * u, d = i % kD, j8 = (i / kD) * 8;
            if constexpr (Q8) {
                const int ps = kv_slot(phys(k0 + j8), stride);   // 8 keys never straddle a block (>= 16 keys)
                v8r[u] = *reinterpret_cast<const uint2*>(v8 + vt_index(kvh, ps, d, stride));
                vs[u] = *reinterpret_cast<const uint4*>(vsc + vs_index(kvh, ps, d / 32, stride));
            } else {
                vr[u] = *reinterpret_cast<const uint4*>(vp + vt_index(kvh, kv_slot(phys(k0 + j8), stride), d, stride));
            }
        }
    };
    auto commit = [&]() {
#pragma unroll
        for (int u = 0; u < KF; ++u) {
            const int i = tid + 256 * u;
            if constexpr (Q8) {
                const int j = i / 8, c16 = i % 8;
                const int8_t* rv = reinterpret_cast<const int8_t*>(&kr[u]);
                _Float16 sch;
                __builtin_memcpy(&sch, &ksr[u], 2);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const h2_t pr = {(_Float16) rv[2 * e] * sch, (_Float16) rv[2 * e + 1] * sch};
                    __builtin_memcpy(&sk[j][c16 * 8 + e], &pr, 4);
                }
            } else {
                const int j = i / 16, c = i % 16;
                const uint32_t* w = reinterpret_cast<const uint32_t*>(&kr[u]);
#pragma unroll
                for (int e = 0; e < 4; ++e) sk[j][c * 4 + e] = w[e];
            }
        }
#pragma unroll
        for (int u = 0; u < VF; ++u) {
            const int i = tid + 256 * u, d = i % kD, j8 = (i / kD) * 8;
            if constexpr (Q8) {
                const int8_t* r8 = reinterpret_cast<const int8_t*>(&v8r[u]);
                const _Float16* sc8 = reinterpret_cast<const _Float16*>(&vs[u]);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const _Float16 hv = (_Float16) r8[e] * sc8[e];
                    __builtin_memcpy(&sv16[j8 + e][d], &hv, 2);
                }
            } else {
                const uint16_t* hv = reinterpret_cast<const uint16_t*>(&vr[u]);
#pragma unroll
                for (int e = 0; e < 8; ++e) sv16[j8 + e][d] = hv[e];
            }
        }
    };
    // PV tasks: (row, dim pair) = (t / 64, t % 64) for t = tid + 256 i, exactly G * 64 of them
    const int n_tasks = R * kD2;
    float acc0[PvRows], acc1[PvRows];
#pragma unroll
    for (int i = 0; i < PvRows; ++i) acc0[i] = acc1[i] = 0.0f;
    if (kb < ke) fetch(kb);
    __syncthreads();
    for (int k0 = kb; k0 < ke; k0 += KT) {
        commit();
        __syncthreads();
        if (k0 + KT < ke) fetch(k0 + KT);   // in flight during this tile's math
        // QK: warp w takes row w (and w + 8 when G > 8); lane takes keys lane + 32c.  No branch inside the loops (a
        // per-row branch there serialized the LDS reads across the tile).
        float dd[2][KPL];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int c = 0; c < KPL; ++c) dd[i][c] = 0.0f;
        if (warp < R) {
            if (warp + 8 < R) {
#pragma unroll 8
                for (int d2 = 0; d2 < kD2; ++d2) {
                    const h2_t qa = sq[warp][d2], qb = sq[warp + 8][d2];
#pragma unroll
                    for (int c = 0; c < KPL; ++c) {
                        const h2_t kk = h2_from_bits(sk[lane + 32 * c][d2]);
                        dd[0][c] = __builtin_amdgcn_fdot2(qa, kk, dd[0][c], false);
                        dd[1][c] = __builtin_amdgcn_fdot2(qb, kk, dd[1][c], false);
                    }
                }
            } else {
#pragma unroll 8
                for (int d2 = 0; d2 < kD2; ++d2) {
                    const h2_t qa = sq[warp][d2];
#pragma unroll
                    for (int c = 0; c < KPL; ++c)
                        dd[0][c] = __builtin_amdgcn_fdot2(qa, h2_from_bits(sk[lane + 32 * c][d2]), dd[0][c], false);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = warp + 8 * i;
            if (r < R) {
                float sc[KPL];
                float mx = -INFINITY;
#pragma unroll
                for (int c = 0; c < KPL; ++c) {
                    const int pk = phys(k0 + lane + 32 * c);
                    sc[c] = (pk >= wstart && pk < qend) ? dd[i][c] : -INFINITY;
                    mx = fmaxf(mx, sc[c]);
                }
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
                const float m_old = sm[r], m_new = fmaxf(m_old, mx);
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < KPL; ++c) {
                    const float pc = (m_new != -INFINITY && sc[c] != -INFINITY) ? expf(sc[c] - m_new) : 0.0f;
                    ss[r][lane + 32 * c] = pc;
                    sum += pc;
                }
                for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
                if (lane == 0) {
                    const float alpha = m_new != -INFINITY ? expf(m_old - m_new) : 1.0f;
                    sm[r] = m_new;
                    sl[r] = sl[r] * alpha + sum;
                    sa[r] = alpha;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < PvRows; ++i) {
            const int t = tid + 256 * i;
            if (t < n_tasks) {
                const int r = t / kD2, pd = t % kD2;
                const float a_ = sa[r];
                float x0 = acc0[i] * a_, x1 = acc1[i] * a_;
#pragma unroll 8
                for (int j = 0; j < KT; ++j) {
                    const h2_t v = h2_from_bits(*reinterpret_cast<const uint32_t*>(&sv16[j][2 * pd]));
                    const float pj = ss[r][j];
                    x0 += pj * (float) v.x; x1 += pj * (float) v.y;
                }
                acc0[i] = x0; acc1[i] = x1;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < PvRows; ++i) {
        const int t = tid + 256 * i;
        if (t < n_tasks) {
            const int r = t / kD2, pd = t % kD2;
            const size_t row = ((size_t) qt * n_split + bsplit) * H + kvh * G + r;
            part_acc[row * kD + 2 * pd] = acc0[i];
            part_acc[row * kD + 2 * pd + 1] = acc1[i];
            if (pd == 0) { part_ml[row * 2] = sm[r]; part_ml[row * 2 + 1] = sl[r]; }
        }
    }
}

// ---- grouped verify attention (several query tokens of a verify window per block) ----
// decode_attn_kernel gives every window token its own block, so each token re-reads and re-dequantizes its keys.  Here a
// block takes NT tokens x the G query heads of one KV head for one split.  Every token keeps exactly its own decode
// geometry: its split range [kb, ke), its tiles k0 = kb + i KT, its keys (block-sparse: its own selection), its mask and
// its per-row arithmetic (the same fdot2 chain per key, the same max / exp / sum order over the tile's lanes, the same
// PV order over the tile's keys), so its partials are bitwise decode_attn_kernel's (= its decode step's).  What is
// shared is the K / V staging: at step i the block gathers the 16-key units (16-aligned in position space) that the
// tokens' i-th tiles need, loads each distinct unit once into LDS (the same dequantization expressions) and every token
// reads its keys through a per-token unit -> slot map.  Units that do not fit (CAP slots) go to a further pass at the
// same step.  (Approach inspired by gufo's verification queries grouped by KV partition, MIT; no gufo code.)
// grid: n_split * n_kv * ceil(n_tok / NT) blocks (token group fastest), 256 threads.
template <bool Q8, int KT, int NT, int GM, int CAP>
__global__ void __launch_bounds__(256) decode_attn_grp_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kp,
                                                              const uint16_t* __restrict__ vp,
                                                              const uint16_t* __restrict__ ksc,
                                                              const uint16_t* __restrict__ vsc, int H, int G, int n_kv,
                                                              int window, int stride, const int* __restrict__ pos_dev,
                                                              float* __restrict__ part_acc, float* __restrict__ part_ml,
                                                              const int* __restrict__ sel, int sel_k, int sel_per_kv,
                                                              int sel_sh, int n_split, int n_tok) {
    constexpr int KPL = KT / 32, UPT = KT / 16, MR = NT * GM, WPT = 8 / NT;   // warps per token
    constexpr int NP = NT * kD2 / 256;                                         // PV (token, dim pair)s per thread
    constexpr int SKS = kD2 + 4, SSS = KT + 4;                                 // LDS row strides (16-B rows)
    static_assert((NT == 4 || NT == 8) && CAP >= UPT && KT % 4 == 0, "shape");
    __shared__ __attribute__((aligned(16))) h2_t sq[MR][kD2];
    __shared__ __attribute__((aligned(16))) uint32_t sk[CAP * 16][SKS];
    __shared__ uint16_t sv16[CAP * 16][kD];
    __shared__ __attribute__((aligned(16))) float ss[MR][SSS];
    __shared__ float sm[MR], sl[MR], sa[MR];
    __shared__ int t_kb[NT], t_ns[NT], t_ws[NT], t_qe[NT], t_sel[NT];   // per token: kb, steps, wstart, qend, sel offset
    __shared__ int s_unit[NT][UPT], s_slot[NT][UPT], s_pass[NT];
    __shared__ int s_res[NT][CAP], s_nres[NT], s_npass, s_maxst, t_cur[NT];   // t_cur: a token's next tile
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int ngrp = (n_tok + NT - 1) / NT, b = (int) blockIdx.x;
    const int grp = b % ngrp, kvh = (b / ngrp) % n_kv, bsplit = b / (ngrp * n_kv);
    const int tok0 = grp * NT;
    const float scale = 0.08838834764831845f;
    if (tid < NT) {
        const int qt = tok0 + tid;
        int kb = 0, ke = 0, ws = 0, qe = 0, so = -1;
        if (qt < n_tok) {
            const int qp = *pos_dev + qt;
            ws = window > 0 ? max(0, qp - (window - 1)) : 0;
            qe = qp + 1;
            if (sel) so = (sel_per_kv ? qt * n_kv + kvh : qt) * sel_k;
            kb = sel ? 0 : ws & ~15;
            ke = sel ? sel_k << sel_sh : qe;
            const int chunk = ((ke - kb + n_split - 1) / n_split + KT - 1) / KT * KT;
            kb += bsplit * chunk;
            ke = min(ke, kb + chunk);
        }
        t_kb[tid] = kb; t_ns[tid] = kb < ke ? (ke - kb + KT - 1) / KT : 0;
        t_ws[tid] = ws; t_qe[tid] = qe; t_sel[tid] = so; t_cur[tid] = 0;
    }
    for (int i = tid; i < MR * kD2; i += 256) {
        const int r = i / kD2, t = r / GM, h = r % GM, qt = tok0 + t;
        h2_t v = {(_Float16) 0.0f, (_Float16) 0.0f};
        if (h < G && qt < n_tok) {
            const float* qr = q + ((size_t) qt * H + kvh * G + h) * kD + 2 * (i % kD2);
            v = h2_t{(_Float16) (qr[0] * scale), (_Float16) (qr[1] * scale)};
        }
        sq[r][i % kD2] = v;
    }
    if (tid < MR) { sm[tid] = -INFINITY; sl[tid] = 0.0f; }
    __syncthreads();
    if (tid == 0) {
        int m = 0;
        for (int t = 0; t < NT; ++t) m = max(m, t_ns[t]);
        s_maxst = m;
    }
    const size_t cell = (size_t) n_kv * kD;
    const int8_t* k8 = reinterpret_cast<const int8_t*>(kp);
    const int8_t* v8 = reinterpret_cast<const int8_t*>(vp);
    // a token's virtual key -> position (its own selection when block-sparse)
    auto phys_t = [&](int t, int v) {
        const int so = t_sel[t];
        return so >= 0 ? (sel[so + (v >> sel_sh)] << sel_sh) | (v & ((1 << sel_sh) - 1)) : v;
    };
    // PV: this thread's (token, dim pair)s p = tid + 256 k (a wave's 32 lanes share the token) x the token's GM rows
    float acc0[NP][GM], acc1[NP][GM];
#pragma unroll
    for (int k = 0; k < NP; ++k)
#pragma unroll
        for (int h = 0; h < GM; ++h) acc0[k][h] = acc1[k][h] = 0.0f;
    __syncthreads();
    (void) s_maxst;
    // Rounds: every token takes its next tile (t_cur) each round, so its tiles stay in order; the tiles of a dense
    // range at the same index coincide, staged in passes of at most CAP distinct units. (Block-sparse selections share
    // too little at the same tile index: the launcher keeps them on the per-token kernel.)
    for (;;) {
        if (tid < NT * UPT) {
            const int t = tid / UPT, jj = tid % UPT, cur = t_cur[t];
            s_unit[t][jj] = cur < t_ns[t] ? phys_t(t, t_kb[t] + cur * KT + 16 * jj) >> 4 : -1;
        }
        __syncthreads();
        if (warp == 0) {   // one warp: lane e = entry (token e / UPT, unit e % UPT); distinct units by a register scan
            constexpr int NE = NT * UPT;
            static_assert(NE <= 32, "plan entries");
            const int u = lane < NE ? s_unit[lane / UPT][lane % UPT] : -1;
            int first = -1;
#pragma unroll
            for (int x = NE - 1; x >= 0; --x) {
                const int ux = __shfl(u, x, 32);
                if (ux == u && u >= 0) first = x;
            }
            const unsigned long long fb = __ballot(first == lane && u >= 0);
            const unsigned fm = (unsigned) fb;
            const int nd = __popc(fm);
            const int rank = __popc(fm & ((1u << lane) - 1u));
            const int frank = __shfl(rank, first < 0 ? 0 : first, 32);
            if (nd <= CAP) {   // one pass (the usual case)
                if (first == lane && u >= 0) s_res[0][rank] = u;
                if (lane < NE) {
                    s_slot[lane / UPT][lane % UPT] = frank;
                    if (lane % UPT == 0) s_pass[lane / UPT] = u >= 0 ? 0 : -1;
                }
                if (lane == 0) { s_nres[0] = nd; s_npass = nd > 0 ? 1 : 0; }   // 0: every token is done
            } else if (lane == 0) {   // more distinct units than slots: greedy passes in token order
                int p = -1, nres = CAP;
                for (int t = 0; t < NT; ++t) {
                    if (s_unit[t][0] < 0) { s_pass[t] = -1; continue; }
                    int need = 0;
                    for (int jj = 0; jj < UPT; ++jj) {
                        const int uu = s_unit[t][jj];
                        bool have = false;
                        for (int x = 0; p >= 0 && x < nres; ++x) have |= s_res[p][x] == uu;
                        for (int y = 0; y < jj; ++y) have |= s_unit[t][y] == uu;
                        need += !have;
                    }
                    if (p < 0 || nres + need > CAP) { ++p; nres = 0; }
                    for (int jj = 0; jj < UPT; ++jj) {
                        const int uu = s_unit[t][jj];
                        int x = 0;
                        while (x < nres && s_res[p][x] != uu) ++x;
                        if (x == nres) s_res[p][nres++] = uu;
                        s_slot[t][jj] = x;
                    }
                    s_pass[t] = p;
                    s_nres[p] = nres;
                }
                s_npass = p + 1;
            }
        }
        __syncthreads();
        const int npass = s_npass;
        if (npass == 0) break;
        for (int p = 0; p < npass; ++p) {
            // stage the pass's units: K as f16 pairs (sk[slot*16 + key][d2]), V as f16 (sv16[slot*16 + key][d])
            const int nres = s_nres[p];
            for (int i = tid; i < nres * 16 * 8; i += 256) {   // K: (unit, key, 16-dim chunk)
                const int x = i / 128, j = (i / 8) % 16, c16 = i % 8;
                const int ps = kv_slot(s_res[p][x] * 16 + j, stride);
                if constexpr (Q8) {
                    const uint4 kr = *reinterpret_cast<const uint4*>(k8 + (size_t) ps * cell + kvh * kD + c16 * 16);
                    const uint16_t ksr = ksc[ks_index(ps, kvh, c16 / 2, n_kv)];
                    const int8_t* rv = reinterpret_cast<const int8_t*>(&kr);
                    _Float16 sch;
                    __builtin_memcpy(&sch, &ksr, 2);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const h2_t pr = {(_Float16) rv[2 * e] * sch, (_Float16) rv[2 * e + 1] * sch};
                        __builtin_memcpy(&sk[x * 16 + j][c16 * 8 + e], &pr, 4);
                    }
                } else {
#pragma unroll
                    for (int hh = 0; hh < 2; ++hh) {
                        const uint4 kr = *reinterpret_cast<const uint4*>(kp + (size_t) ps * cell + kvh * kD + (2 * c16 + hh) * 8);
                        const uint32_t* w = reinterpret_cast<const uint32_t*>(&kr);
#pragma unroll
                        for (int e = 0; e < 4; ++e) sk[x * 16 + j][(2 * c16 + hh) * 4 + e] = w[e];
                    }
                }
            }
            for (int i = tid; i < nres * 2 * kD; i += 256) {   // V: (unit, 8-key half, dim)
                const int x = i / (2 * kD), j8 = ((i / kD) % 2) * 8, d = i % kD;
                const int ps = kv_slot(s_res[p][x] * 16 + j8, stride);
                if constexpr (Q8) {
                    const uint2 v8r = *reinterpret_cast<const uint2*>(v8 + vt_index(kvh, ps, d, stride));
                    const uint4 vs = *reinterpret_cast<const uint4*>(vsc + vs_index(kvh, ps, d / 32, stride));
                    const int8_t* r8 = reinterpret_cast<const int8_t*>(&v8r);
                    const _Float16* sc8 = reinterpret_cast<const _Float16*>(&vs);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const _Float16 hv = (_Float16) r8[e] * sc8[e];
                        __builtin_memcpy(&sv16[x * 16 + j8 + e][d], &hv, 2);
                    }
                } else {
                    const uint4 vr = *reinterpret_cast<const uint4*>(vp + vt_index(kvh, ps, d, stride));
                    const uint16_t* hv = reinterpret_cast<const uint16_t*>(&vr);
#pragma unroll
                    for (int e = 0; e < 8; ++e) sv16[x * 16 + j8 + e][d] = hv[e];
                }
            }
            __syncthreads();
            {   // QK + online softmax: warp w takes token w % NT, its rows w / NT, w / NT + WPT, ...
                const int t = warp % NT;
                if (s_pass[t] == p) {
                    const int k0 = t_kb[t] + t_cur[t] * KT;
                    int krow[KPL];
#pragma unroll
                    for (int c = 0; c < KPL; ++c) krow[c] = s_slot[t][(lane + 32 * c) >> 4] * 16 + ((lane + 32 * c) & 15);
                    constexpr int RPW = (GM + WPT - 1) / WPT;
                    float dd[RPW][KPL];
#pragma unroll
                    for (int i = 0; i < RPW; ++i)
#pragma unroll
                        for (int c = 0; c < KPL; ++c) dd[i][c] = 0.0f;
                    // each row's chain over d2 in order, 4 pairs per LDS load
#pragma unroll 2
                    for (int d4 = 0; d4 < kD2; d4 += 4) {
                        uint4 kq[KPL];
#pragma unroll
                        for (int c = 0; c < KPL; ++c) kq[c] = *reinterpret_cast<const uint4*>(&sk[krow[c]][d4]);
#pragma unroll
                        for (int i = 0; i < RPW; ++i) {
                            const uint4 qq = *reinterpret_cast<const uint4*>(&sq[t * GM + warp / NT + WPT * i][d4]);
                            const uint32_t qw[4] = {qq.x, qq.y, qq.z, qq.w};
#pragma unroll
                            for (int e = 0; e < 4; ++e)
#pragma unroll
                                for (int c = 0; c < KPL; ++c) {
                                    const uint32_t kw[4] = {kq[c].x, kq[c].y, kq[c].z, kq[c].w};
                                    dd[i][c] = __builtin_amdgcn_fdot2(h2_from_bits(qw[e]), h2_from_bits(kw[e]), dd[i][c], false);
                                }
                        }
                    }
                    const int ws = t_ws[t], qe = t_qe[t];
#pragma unroll
                    for (int i = 0; i < RPW; ++i) {
                        const int h = warp / NT + WPT * i;
                        if (h < G) {
                            const int r = t * GM + h;
                            float sc[KPL];
                            float mx = -INFINITY;
#pragma unroll
                            for (int c = 0; c < KPL; ++c) {
                                const int pk = phys_t(t, k0 + lane + 32 * c);
                                sc[c] = (pk >= ws && pk < qe) ? dd[i][c] : -INFINITY;
                                mx = fmaxf(mx, sc[c]);
                            }
                            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
                            const float m_old = sm[r], m_new = fmaxf(m_old, mx);
                            float sum = 0.0f;
#pragma unroll
                            for (int c = 0; c < KPL; ++c) {
                                const float pc = (m_new != -INFINITY && sc[c] != -INFINITY) ? expf(sc[c] - m_new) : 0.0f;
                                ss[r][lane + 32 * c] = pc;
                                sum += pc;
                            }
                            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
                            if (lane == 0) {
                                const float alpha = m_new != -INFINITY ? expf(m_old - m_new) : 1.0f;
                                sm[r] = m_new;
                                sl[r] = sl[r] * alpha + sum;
                                sa[r] = alpha;
                            }
                        }
                    }
                }
            }
            __syncthreads();
#pragma unroll
            for (int k = 0; k < NP; ++k) {   // PV: each row's sum over the tile's keys in key order (j outer: GM chains)
                const int pp = tid + 256 * k, t = pp / kD2, pd = pp % kD2;
                if (s_pass[t] == p) {
                    float x0[GM], x1[GM];
#pragma unroll
                    for (int h = 0; h < GM; ++h) {
                        const float a_ = sa[t * GM + h];
                        x0[h] = acc0[k][h] * a_; x1[h] = acc1[k][h] * a_;
                    }
                    int kslot[UPT];
#pragma unroll
                    for (int u = 0; u < UPT; ++u) kslot[u] = s_slot[t][u] * 16;
#pragma unroll 2
                    for (int j4 = 0; j4 < KT; j4 += 4) {
                        h2_t v[4];
#pragma unroll
                        for (int jj = 0; jj < 4; ++jj)
                            v[jj] = h2_from_bits(*reinterpret_cast<const uint32_t*>(&sv16[kslot[(j4 + jj) >> 4] + ((j4 + jj) & 15)][2 * pd]));
#pragma unroll
                        for (int h = 0; h < GM; ++h) {
                            const float4 p4 = *reinterpret_cast<const float4*>(&ss[t * GM + h][j4]);
                            const float pj[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
                            for (int jj = 0; jj < 4; ++jj) { x0[h] += pj[jj] * (float) v[jj].x; x1[h] += pj[jj] * (float) v[jj].y; }
                        }
                    }
#pragma unroll
                    for (int h = 0; h < GM; ++h) { acc0[k][h] = x0[h]; acc1[k][h] = x1[h]; }
                }
            }
            __syncthreads();
        }
        if (tid < NT && s_pass[tid] >= 0) t_cur[tid] += 1;   // the round's tokens move to their next tile
        __syncthreads();
    }
#pragma unroll
    for (int k = 0; k < NP; ++k) {
        const int pp = tid + 256 * k, t = pp / kD2, pd = pp % kD2, qt = tok0 + t;
        if (qt < n_tok) {
#pragma unroll
            for (int h = 0; h < GM; ++h)
                if (h < G) {
                    const int r = t * GM + h;
                    const size_t row = ((size_t) qt * n_split + bsplit) * H + kvh * G + h;
                    part_acc[row * kD + 2 * pd] = acc0[k][h];
                    part_acc[row * kD + 2 * pd + 1] = acc1[k][h];
                    if (pd == 0) { part_ml[row * 2] = sm[r]; part_ml[row * 2 + 1] = sl[r]; }
                }
        }
    }
}
}  // namespace

// test hook overrides of LAGUNA_DEC_GRP / LAGUNA_DEC_GRP_NT (-1 = the environment); talus_laguna_decode_attn_grp
int g_dec_grp_min = -1, g_dec_grp_nt = -1, g_dec_tokmajor = -1, g_dec_grp_sparse = 0;

void decode_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const uint16_t* k_scale,
                 const uint16_t* v_scale, int n_head, int n_kv, int window, int stride, const int* pos_dev, int n_split,
                 float* part_acc, float* part_ml, void* stream, int n_tok, const int* sel, int sel_k, bool sel_per_kv,
                 int sel_bs) {
    const int sel_sh = sel_bs == 16 ? 4 : sel_bs == 32 ? 5 : 6;
    const int G = n_head / n_kv;
    // rb-vattn kill switch: LAGUNA_DEC_TOKMAJOR=0 restores the (split, KV head, token) grid
    static const int tokmajor_env = [] { const char* e = std::getenv("LAGUNA_DEC_TOKMAJOR"); return e ? std::atoi(e) : 1; }();
    // (default: full layers at any width; sliding-window layers up to 16 tokens, where it measured faster)
    const int tm = n_tok > 1 && (g_dec_tokmajor >= 0 ? g_dec_tokmajor : (tokmajor_env && (window == 0 || n_tok <= 16))) ? 1 : 0;
    const dim3 grid = tm ? dim3(n_split * n_kv * n_tok) : dim3(n_split, n_kv, n_tok);
    static const int kt = [] {   // tuning: LAGUNA_DEC_KT = 32 or 64 keys per tile
        const char* e = std::getenv("LAGUNA_DEC_KT");
        return e && std::atoi(e) == 64 ? 64 : 32;
    }();
    const cudaStream_t st = (cudaStream_t) stream;
    // grouped verify attention (decode_attn_grp_kernel, bitwise the per-token kernel) from LAGUNA_DEC_GRP
    // window tokens (default 2; 0 = off, the kill switch). Dense ranges only (full-attention layers below the sparse
    // threshold, and the sliding-window layers): with block-sparse selections a window's tokens walk different blocks at
    // the same tile index, so they share little and keep the per-token kernel (token-major grid). Shape (talus-vattn-bench,
    // gfx1151): 8 tokens per block for full layers from 8 tokens, else 4; 2 resident units. LAGUNA_DEC_GRP_NT = cap * 10
    // + nt overrides (e.g. 44).
    static const int grp_min_env = [] { const char* e = std::getenv("LAGUNA_DEC_GRP"); return e ? std::atoi(e) : 2; }();
    static const int grp_nt_env = [] { const char* e = std::getenv("LAGUNA_DEC_GRP_NT"); return e ? std::atoi(e) : 0; }();
    const int grp_min = g_dec_grp_min >= 0 ? g_dec_grp_min : grp_min_env;
    const int grp_nt = g_dec_grp_nt >= 0 ? g_dec_grp_nt : grp_nt_env;
    if (grp_min > 0 && n_tok >= grp_min && kt == 32 && G <= 9 && (sel == nullptr || g_dec_grp_sparse)) {
        const int nt = grp_nt > 0 ? ((grp_nt % 10 == 8 && G <= 6) ? 8 : 4) : ((G <= 6 && n_tok >= 8) ? 8 : 4);
        const int cap = grp_nt >= 10 ? grp_nt / 10 : 2;
        const dim3 gg(n_split * n_kv * ((n_tok + nt - 1) / nt));
#define GRP_L(Q, NT_, GM_) if (cap == 2) decode_attn_grp_kernel<Q, 32, NT_, GM_, 2><<<gg, 256, 0, st>>>(q, k_pool, v_pool, k_scale, v_scale, \
            n_head, G, n_kv, window, stride, pos_dev, part_acc, part_ml, sel, sel_k, sel_per_kv ? 1 : 0, sel_sh, n_split, n_tok); \
            else decode_attn_grp_kernel<Q, 32, NT_, GM_, 4><<<gg, 256, 0, st>>>(q, k_pool, v_pool, k_scale, v_scale, \
            n_head, G, n_kv, window, stride, pos_dev, part_acc, part_ml, sel, sel_k, sel_per_kv ? 1 : 0, sel_sh, n_split, n_tok)
#define GRP_Q(Q) \
        if (G <= 6) { if (nt == 8) GRP_L(Q, 8, 6); else GRP_L(Q, 4, 6); } \
        else GRP_L(Q, 4, 9);
        if (k_scale) { GRP_Q(true) } else { GRP_Q(false) }
#undef GRP_Q
#undef GRP_L
        check_launch("decode_attn_grp");
        return;
    }
    if (k_scale && kt == 64)
        decode_attn_kernel<true, 64><<<grid, 256, 0, st>>>(q, k_pool, v_pool, k_scale, v_scale, n_head, G, n_kv, window,
                                                           stride, pos_dev, part_acc, part_ml, sel, sel_k, sel_per_kv ? 1 : 0, sel_sh, n_split, n_kv, tm);
    else if (k_scale)
        decode_attn_kernel<true, 32><<<grid, 256, 0, st>>>(q, k_pool, v_pool, k_scale, v_scale, n_head, G, n_kv, window,
                                                           stride, pos_dev, part_acc, part_ml, sel, sel_k, sel_per_kv ? 1 : 0, sel_sh, n_split, n_kv, tm);
    else if (kt == 64)
        decode_attn_kernel<false, 64><<<grid, 256, 0, st>>>(q, k_pool, v_pool, k_scale, v_scale, n_head, G, n_kv, window,
                                                            stride, pos_dev, part_acc, part_ml, sel, sel_k, sel_per_kv ? 1 : 0, sel_sh, n_split, n_kv, tm);
    else
        decode_attn_kernel<false, 32><<<grid, 256, 0, st>>>(q, k_pool, v_pool, k_scale, v_scale, n_head, G, n_kv, window,
                                                            stride, pos_dev, part_acc, part_ml, sel, sel_k, sel_per_kv ? 1 : 0, sel_sh, n_split, n_kv, tm);
    check_launch("decode_attn");
}

namespace {

__global__ void argmax_rows_kernel(const float* __restrict__ logits, int n, int* __restrict__ out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float* row = logits + (size_t) blockIdx.x * n;
    float v = -INFINITY;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (row[i] > v) { v = row[i]; idx = i; }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if ((int) threadIdx.x < st) {
            const float b = bv[threadIdx.x + st];
            const int j = bi[threadIdx.x + st];
            if (b > bv[threadIdx.x] || (b == bv[threadIdx.x] && j < bi[threadIdx.x])) { bv[threadIdx.x] = b; bi[threadIdx.x] = j; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = bi[0];
}
}  // namespace

void route_topk(const float* logits, const float* bias, int n_tok, int k, float scale, int32_t* ids, float* w,
                void* stream) {
    route_batch_kernel<<<(n_tok + 7) / 8, 256, 0, (cudaStream_t) stream>>>(logits, bias, ids, w, n_tok, k, scale);
    check_launch("route_topk");
}

void argmax_rows(const float* logits, int n_rows, int n, int* out, void* stream) {
    argmax_rows_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n, out);
    check_launch("argmax_rows");
}

namespace {

// y[t][o] = w[o] . x[t] for n_tok <= 8 tokens: one 128-thread block per row, the row read once for every token
__global__ void __launch_bounds__(128) f32_matvec_rows_n_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                                                float* __restrict__ y, int n_in, int n_out, int n_tok) {
    __shared__ float red[4][8];
    const float4* row = reinterpret_cast<const float4*>(w + (size_t) blockIdx.x * n_in);
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = threadIdx.x; i < n_in / 4; i += 128) {
        const float4 a = row[i];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < n_tok) {
                const float4 b = reinterpret_cast<const float4*>(x + (size_t) t * n_in)[i];
                acc[t] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        float v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
        if (threadIdx.x % 32 == 0) red[threadIdx.x / 32][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < n_tok)
        y[(size_t) threadIdx.x * n_out + blockIdx.x] = red[0][threadIdx.x] + red[1][threadIdx.x] + red[2][threadIdx.x] +
                                                        red[3][threadIdx.x];
}
}  // namespace

void f32_matvec_rows_n(const float* w, const float* x, float* y, int n_in, int n_out, int n_tok, void* stream) {
    if (n_tok < 1 || n_tok > 8 || n_in % 4 != 0) { std::fprintf(stderr, "laguna: f32_matvec_rows_n: shape\n"); std::exit(1); }
    f32_matvec_rows_n_kernel<<<n_out, 128, 0, (cudaStream_t) stream>>>(w, x, y, n_in, n_out, n_tok);
    check_launch("f32_matvec_rows_n");
}

// ---- block-sparse decode attention (Quest-style; llama.cpp patch LLAMA_SPARSE_ATTN semantics) ----
// Per full-attention layer, KV head and 64-key block: the elementwise max (dims 0..127) and min (128..255) of the
// dequantized keys, bnd[(kvh * nb_max + b) * 256 + d].
namespace {

// f16 bounds stay bounds: the max rounded up, the min rounded down (one f16 ulp outward when round-to-nearest
// landed inside)
__device__ inline uint16_t half_out(float v, bool up) {
    uint16_t h = strata::kernels::f16_from_f32(v);
    const float r = strata::kernels::f32_from_f16(h);
    if (up ? r < v : r > v) {
        const bool neg = (h & 0x8000u) != 0;
        if ((h & 0x7fffu) == 0) h = up ? 0x0001u : 0x8001u;   // +-0 -> the smallest subnormal outward
        else h = (uint16_t) ((up != neg) ? h + 1 : h - 1);
    }
    return h;
}
template <typename T> __device__ inline void put_bound(T* o, float mx, float mn, int d);
template <> __device__ inline void put_bound<float>(float* o, float mx, float mn, int d) { o[d] = mx; o[128 + d] = mn; }
template <> __device__ inline void put_bound<uint16_t>(uint16_t* o, float mx, float mn, int d) {
    o[d] = half_out(mx, true);
    o[128 + d] = half_out(mn, false);
}

template <typename T>
__global__ void __launch_bounds__(128) kbound_update_kernel(const int8_t* __restrict__ k8,
                                                            const uint16_t* __restrict__ ksc, T* __restrict__ bnd,
                                                            int nb_max, int n_kv, int b_first, int b_last,
                                                            int pos_last, const int* __restrict__ pos_dev, int n_tok,
                                                            int bs) {
    if (pos_dev) { const int p0 = *pos_dev; b_first = p0 / bs; pos_last = p0 + n_tok - 1; b_last = pos_last / bs; }
    const int b = b_first + (int) blockIdx.x, kvh = blockIdx.y, d = threadIdx.x;
    if (b > b_last) return;
    const size_t cell = (size_t) n_kv * 128;
    float mx = -INFINITY, mn = INFINITY;
    const int c1 = min(b * bs + bs - 1, pos_last);
    for (int c = b * bs; c <= c1; ++c) {
        const float v = (float) k8[(size_t) c * cell + kvh * 128 + d] *
                        strata::kernels::f32_from_f16(ksc[ks_index(c, kvh, d / 32, n_kv)]);
        mx = fmaxf(mx, v);
        mn = fminf(mn, v);
    }
    put_bound<T>(bnd + ((size_t) kvh * nb_max + b) * 256, mx, mn, d);
}

template <typename T> __device__ inline float4 load_bound4(const T* p);
template <> __device__ inline float4 load_bound4<float>(const float* p) { return *reinterpret_cast<const float4*>(p); }
template <> __device__ inline float4 load_bound4<uint16_t>(const uint16_t* p) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const _Float16* h = reinterpret_cast<const _Float16*>(&r);
    return make_float4((float) h[0], (float) h[1], (float) h[2], (float) h[3]);
}

// ub[(t * H + h) * nb_max + b] = scale * sum_d max(q_d kmax_d, q_d kmin_d): the bound on q.k over block b.
// 8 warps = the 8 KV heads; a CTA covers 16 blocks; a lane takes 4 dims of the G query heads of its KV head.
template <typename T>
__global__ void __launch_bounds__(256) sparse_scores_kernel(const float* __restrict__ q, const T* __restrict__ bnd,
                                                            float* __restrict__ ub, int nb_max, int H, int G,
                                                            const int* __restrict__ pos_dev, float scale, int bs) {
    constexpr int kMaxG = 8, kPer = 16;
    const int qt = blockIdx.y, lane = threadIdx.x % 32, kvh = threadIdx.x / 32;
    const int nb = (*pos_dev + qt) / bs + 1;
    const int b0 = (int) blockIdx.x * kPer;
    if (b0 >= nb) return;
    float4 qv[kMaxG];
#pragma unroll
    for (int r = 0; r < kMaxG; ++r)
        if (r < G) qv[r] = *reinterpret_cast<const float4*>(q + ((size_t) qt * H + kvh * G + r) * 128 + 4 * lane);
    const int b1 = min(b0 + kPer, nb);
    for (int b = b0; b < b1; ++b) {
        const T* bb = bnd + ((size_t) kvh * nb_max + b) * 256;
        const float4 kx = load_bound4<T>(bb + 4 * lane);
        const float4 kn = load_bound4<T>(bb + 128 + 4 * lane);
#pragma unroll
        for (int r = 0; r < kMaxG; ++r) {
            if (r >= G) break;
            const float4 a = qv[r];
            float v = fmaxf(a.x * kx.x, a.x * kn.x) + fmaxf(a.y * kx.y, a.y * kn.y) + fmaxf(a.z * kx.z, a.z * kn.z) +
                      fmaxf(a.w * kx.w, a.w * kn.w);
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
            if (lane == 0) ub[((size_t) qt * H + kvh * G + r) * nb_max + b] = v * scale;
        }
    }
}

// exclusive prefix sum of v over the block's 1024 threads (thread order), and the total
__device__ int block_scan_1024(int v, int* warp_tot, int& total) {
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    int incl = v;
    for (int o = 1; o < 32; o <<= 1) { const int t = __shfl_up(incl, o, 32); if (lane >= o) incl += t; }
    if (lane == 31) warp_tot[w] = incl;
    __syncthreads();
    if (w == 0) {
        int t = warp_tot[lane], ti = t;
        for (int o = 1; o < 32; o <<= 1) { const int u = __shfl_up(ti, o, 32); if (lane >= o) ti += u; }
        warp_tot[lane] = ti - t;   // exclusive
        if (lane == 31) warp_tot[32] = ti;
    }
    __syncthreads();
    const int r = warp_tot[w] + incl - v;
    total = warp_tot[32];
    __syncthreads();
    return r;
}

// Per token: softmax of the bounds over blocks for each query head, summed over heads, sink / recent blocks forced,
// top-k -> sel[t * k ..] in ascending block order (ties to the lower block: deterministic).
constexpr int kSpMaxNb = 12288;   // 48 KB of LDS: 16-key blocks to ~196K positions
__global__ void __launch_bounds__(1024) sparse_select_kernel(const float* __restrict__ ub, int nb_max, int H,
                                                             const int* __restrict__ pos_dev, int k, int sink,
                                                             int recent, int* __restrict__ sel, int bs) {
    // gridDim.y groups of H / gridDim.y query heads (1: one selection for every head; n_kv: per KV head)
    __shared__ float hm[64], hinv[64];
    __shared__ uint32_t sbits[kSpMaxNb];
    __shared__ int warp_tot[33];
    __shared__ int cnt;
    const int qt = blockIdx.x, tid = threadIdx.x, lane = tid % 32, w = tid / 32;
    const int p = *pos_dev + qt;
    const int nb = p / bs + 1;   // the host keeps nb_max <= kSpMaxNb
    const int hpg = H / (int) gridDim.y;
    const float* u = ub + ((size_t) qt * H + blockIdx.y * hpg) * nb_max;
    H = hpg;
    sel += (size_t) (qt * gridDim.y + blockIdx.y) * k - (size_t) qt * k;   // sel[(qt * groups + g) * k ..] below
    // per head: max and 1 / sum exp over the nb blocks (a warp a head)
    for (int h = w; h < H; h += 32) {
        const float* uh = u + (size_t) h * nb_max;
        float m = -INFINITY;
        for (int b = lane; b < nb; b += 32) m = fmaxf(m, uh[b]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
        float sum = 0.0f;
        for (int b = lane; b < nb; b += 32) sum += expf(uh[b] - m);
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
        if (lane == 0) { hm[h] = m; hinv[h] = 1.0f / sum; }
    }
    __syncthreads();
    // scores (>= 0, so their bits order like the values); thread t owns blocks [t * per, (t + 1) * per)
    const int per = (nb + 1023) / 1024, bl0 = tid * per, bl1 = min(bl0 + per, nb);
    for (int b = bl0; b < bl1; ++b) {
        float sc = 0.0f;
        for (int h = 0; h < H; ++h) sc += expf(u[(size_t) h * nb_max + b] - hm[h]) * hinv[h];
        const int pmax = min(b * bs + bs - 1, p);
        if (b * bs < sink || pmax > p - recent) sc += 1e4f;
        sbits[b] = __float_as_uint(sc);
    }
    __syncthreads();
    const int kk = min(k, nb);
    // the k-th largest value: the largest T with count(bits >= T) >= kk
    uint32_t lo = 0u, hi = 0xffffffffu;
    while (lo < hi) {
        const uint32_t mid = lo + (uint32_t) (((uint64_t) hi - lo + 1) / 2);
        if (tid == 0) cnt = 0;
        __syncthreads();
        int c = 0;
        for (int b = bl0; b < bl1; ++b) c += sbits[b] >= mid;
        if (c) atomicAdd(&cnt, c);
        __syncthreads();
        if (cnt >= kk) lo = mid; else hi = mid - 1;
        __syncthreads();
    }
    const uint32_t T = lo;
    int n_gt = 0, n_eq = 0;
    for (int b = bl0; b < bl1; ++b) { n_gt += sbits[b] > T; n_eq += sbits[b] == T; }
    int tot_gt, tot_eq;
    block_scan_1024(n_gt, warp_tot, tot_gt);
    int eq_rank = block_scan_1024(n_eq, warp_tot, tot_eq);
    const int need_eq = kk - tot_gt;
    int n_sel = 0;
    for (int b = bl0, r = eq_rank; b < bl1; ++b) {
        if (sbits[b] > T) ++n_sel;
        else if (sbits[b] == T) { if (r < need_eq) ++n_sel; ++r; }
    }
    int tot_sel;
    int o = block_scan_1024(n_sel, warp_tot, tot_sel);
    for (int b = bl0, r = eq_rank; b < bl1; ++b) {
        bool take = sbits[b] > T;
        if (sbits[b] == T) { take = r < need_eq; ++r; }
        if (take) sel[qt * k + o++] = b;
    }
}

// The same selection with the scores in global scratch (scr [n_tok][groups][nb_max] bits) and the k-th largest value
// by a 4-pass 8-bit radix select: LDS no longer grows with the context.  Same k-th value, same tie rule, same ascending
// compaction -> the same selections as sparse_select_kernel.
__global__ void __launch_bounds__(1024) sparse_select_radix_kernel(const float* __restrict__ ub, int nb_max, int H,
                                                                   const int* __restrict__ pos_dev, int k, int sink,
                                                                   int recent, int* __restrict__ sel, int bs,
                                                                   uint32_t* __restrict__ scr) {
    __shared__ float hm[64], hinv[64];
    __shared__ int hist[256];
    __shared__ int warp_tot[33];
    __shared__ uint32_t s_prefix;
    __shared__ int s_remain;
    const int qt = blockIdx.x, tid = threadIdx.x, lane = tid % 32, w = tid / 32;
    const int p = *pos_dev + qt;
    const int nb = p / bs + 1;
    const int hpg = H / (int) gridDim.y;
    const float* u = ub + ((size_t) qt * H + blockIdx.y * hpg) * nb_max;
    uint32_t* sb = scr + (size_t) (qt * gridDim.y + blockIdx.y) * nb_max;
    sel += (size_t) (qt * gridDim.y + blockIdx.y) * k;
    for (int h = w; h < hpg; h += 32) {
        const float* uh = u + (size_t) h * nb_max;
        float m = -INFINITY;
        for (int b = lane; b < nb; b += 32) m = fmaxf(m, uh[b]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
        float sum = 0.0f;
        for (int b = lane; b < nb; b += 32) sum += expf(uh[b] - m);
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor(sum, o, 32);
        if (lane == 0) { hm[h] = m; hinv[h] = 1.0f / sum; }
    }
    __syncthreads();
    const int per = (nb + 1023) / 1024, bl0 = tid * per, bl1 = min(bl0 + per, nb);
    for (int b = bl0; b < bl1; ++b) {
        float sc = 0.0f;
        for (int h = 0; h < hpg; ++h) sc += expf(u[(size_t) h * nb_max + b] - hm[h]) * hinv[h];
        const int pmax = min(b * bs + bs - 1, p);
        if (b * bs < sink || pmax > p - recent) sc += 1e4f;
        sb[b] = __float_as_uint(sc);
    }
    __threadfence_block();
    __syncthreads();
    const int kk = min(k, nb);
    if (tid == 0) { s_prefix = 0u; s_remain = kk; }
    // radix select, most significant byte first: the k-th largest bits T
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += 1024) hist[i] = 0;
        __syncthreads();
        const uint32_t pre = s_prefix, pmask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int b = tid; b < nb; b += 1024) {
            const uint32_t v = sb[b];
            if ((v & pmask) == pre) atomicAdd(&hist[(v >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int rem = s_remain, d = 255;
            for (; d > 0; --d) { if (hist[d] >= rem) break; rem -= hist[d]; }
            s_prefix = pre | ((uint32_t) d << shift);
            s_remain = rem;
        }
        __syncthreads();
    }
    const uint32_t T = s_prefix;
    int n_gt = 0, n_eq = 0;
    for (int b = bl0; b < bl1; ++b) { const uint32_t v = sb[b]; n_gt += v > T; n_eq += v == T; }
    int tot_gt, tot_eq;
    block_scan_1024(n_gt, warp_tot, tot_gt);
    int eq_rank = block_scan_1024(n_eq, warp_tot, tot_eq);
    const int need_eq = kk - tot_gt;
    int n_sel = 0;
    for (int b = bl0, r = eq_rank; b < bl1; ++b) {
        const uint32_t v = sb[b];
        if (v > T) ++n_sel;
        else if (v == T) { if (r < need_eq) ++n_sel; ++r; }
    }
    int tot_sel;
    int o = block_scan_1024(n_sel, warp_tot, tot_sel);
    for (int b = bl0, r = eq_rank; b < bl1; ++b) {
        const uint32_t v = sb[b];
        bool take = v > T;
        if (v == T) { take = r < need_eq; ++r; }
        if (take) sel[o++] = b;
    }
}
}  // namespace

void kbound_update(const void* k_pool, const uint16_t* k_scale, void* bnd, int nb_max, int n_kv, int b_first,
                   int b_last, int pos_last, const int* pos_dev, int n_tok, void* stream, int bs, bool f16) {
    const int nblk = pos_dev ? (n_tok + bs - 1) / bs + 1 : b_last - b_first + 1;
    if (nblk <= 0) return;
    const dim3 grid(nblk, n_kv);
    if (f16)
        kbound_update_kernel<uint16_t><<<grid, 128, 0, (cudaStream_t) stream>>>(
            (const int8_t*) k_pool, k_scale, (uint16_t*) bnd, nb_max, n_kv, b_first, b_last, pos_last, pos_dev, n_tok, bs);
    else
        kbound_update_kernel<float><<<grid, 128, 0, (cudaStream_t) stream>>>(
            (const int8_t*) k_pool, k_scale, (float*) bnd, nb_max, n_kv, b_first, b_last, pos_last, pos_dev, n_tok, bs);
    check_launch("kbound_update");
}

void sparse_scores(const float* q, const void* bnd, float* ub, int nb_max, int n_head, int n_kv,
                   const int* pos_dev, int n_tok, void* stream, int bs, bool f16) {
    if (n_kv != 8 || n_head % n_kv != 0 || n_head / n_kv > 8) { std::fprintf(stderr, "laguna: sparse_scores: shape\n"); std::exit(1); }
    const dim3 grid((nb_max + 15) / 16, n_tok);
    if (f16)
        sparse_scores_kernel<uint16_t><<<grid, 256, 0, (cudaStream_t) stream>>>(
            q, (const uint16_t*) bnd, ub, nb_max, n_head, n_head / n_kv, pos_dev, 0.08838834764831845f, bs);
    else
        sparse_scores_kernel<float><<<grid, 256, 0, (cudaStream_t) stream>>>(
            q, (const float*) bnd, ub, nb_max, n_head, n_head / n_kv, pos_dev, 0.08838834764831845f, bs);
    check_launch("sparse_scores");
}

void sparse_select(const float* ub, int nb_max, int n_head, const int* pos_dev, int n_tok, int k, int sink, int recent,
                   int* sel, void* stream, int groups, int bs, uint32_t* scr) {
    if (n_head > 64 || n_head % groups != 0 || (!scr && nb_max > kSpMaxNb)) {
        std::fprintf(stderr, "laguna: sparse_select: shape\n"); std::exit(1);
    }
    if (scr)
        sparse_select_radix_kernel<<<dim3(n_tok, groups), 1024, 0, (cudaStream_t) stream>>>(
            ub, nb_max, n_head, pos_dev, k, sink, recent, sel, bs, scr);
    else
        sparse_select_kernel<<<dim3(n_tok, groups), 1024, 0, (cudaStream_t) stream>>>(ub, nb_max, n_head, pos_dev, k,
                                                                                      sink, recent, sel, bs);
    check_launch("sparse_select");
}

}  // namespace strata::laguna

// test hook (talus-laguna-attn-test): 1 = per-wave tiles, 2 = shared tiles, -1 = TALUS_LAGUNA_ATTN
extern "C" void talus_laguna_prompt_attn_version(int v) { strata::laguna::prompt_attn_set_version(v); }
// test hook (talus-vattn-bench): grouped verify attention from grp_min tokens (0 = off), nt tokens per block (0 = default)
extern "C" void talus_laguna_decode_attn_grp(int grp_min, int nt, int tokmajor, int sparse) {
    strata::laguna::g_dec_grp_min = grp_min; strata::laguna::g_dec_grp_nt = nt; strata::laguna::g_dec_tokmajor = tokmajor;
    strata::laguna::g_dec_grp_sparse = sparse;
}
