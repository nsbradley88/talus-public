// src/prefill/tp_w16.cu - 16-bit weight (BF16 / F16) mat-vec and GEMM for the dense qwen35 engine (opt-in:
// used only when a GGUF carries BF16 / F16 projection matrices, e.g. the Qwen3.8-27B BF16 reference).
//
//   Y[t][n] = sum_k W[n][k] * X[t][k]      W: N x K row-major 16-bit (GGUF ne0 = K contiguous, row stride ldw elements)
//                                           X: T x K fp32 (row stride ldx), Y: T x N fp32 (row stride ldy)
//
// Mat-vec (1..16 columns, decode and verify windows): FP32 activations as given (no rounding of X), each weight
// element widened to fp32 (bf16: a 16-bit shift, f16: v_cvt_f32_f16) and FMA'd into an fp32 accumulator.  Wave32:
// lane l reads 16-byte vectors (8 weights) at vector index l, l+32, l+64, ... of each of its R rows (one coalesced
// 512-byte read per row per step), U vectors per row in flight per lane, then a 5-step butterfly sum.  Every column's
// result is computed with the same per-lane order and the same butterfly whatever the column count, R, U or waves per
// block: a column of a T-column window is BITWISE the 1-column result (exact speculative verify).
//
// GEMM (more than 16 columns, prompt): X rounded once to the weight's 16-bit type (bf16: round to nearest even with
// ggml's NaN rule, as ggml_compute_fp32_to_bf16; the same rounding llama.cpp's BF16 prompt GEMM applies to its
// activations), then the gfx11 WMMA (v_wmma_f32_16x16x16_bf16 / _f16) with fp32 accumulation.  Tile scheme of
// kernel_bk64 in wmma_gemm.cu (S23, pfg): 128 x 256 blocks, BK 64 single LDS buffer with the next tile in named
// staging registers, 8 waves of 64 x 64, grouped tile order.  K a multiple of 64.  On a GPU without the gfx11 WMMA the
// GEMM falls back to the mat-vec in 16-column slices (correct, slow).
//
// C ABI (optional symbols, resolved with dlsym by the Rust host; nothing else links them):
//   int talus_w16_gemv(wtype, w, ldw, n_out, n_in, x, ldx, y, ldy, n_tok, stream)
//   int talus_w16_gemm(wtype, w, ldw, n_out, n_in, x, ldx, y, ldy, n_tok, x16_scratch, scratch_elems, stream)
//   int talus_w16_supported(wtype, n_out, n_in)            1 = both entry points take this matrix
//   int talus_w16_info(what)                                0: ABI version, 1: GEMM uses WMMA on this device
// Return codes: 0 OK, 1 unsupported (type / shape / alignment), 2 HIP error.
// wtype is the GGML type id: 30 = BF16, 1 = F16.
//
// Tuning (per-arch table below, overridable): TALUS_W16_CFG="T:R:U[:W],..." (T = column count 1..16, R rows per wave
// 1|2|4, U vectors per row in flight 1|2|4, W waves per block 1..8); TALUS_W16_GEMM=0 forces the mat-vec slices.
//
// Credits: the bf16 rounding rule is ggml's (ggml-impl.h ggml_compute_fp32_to_bf16, MIT); the GEMM tiling is the S23
// kernel_bk64 of this repository (wmma_gemm.cu).
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define TP_W16_WMMA 1
#endif

namespace talus::prefill::w16 {
namespace {   // internal: only the extern "C" entry points below are exported

constexpr int kF16 = 1;
constexpr int kBF16 = 30;
constexpr int kMaxCols = 16;

// ------------------------------------------------------------------------------------------------ conversions

__host__ __device__ __forceinline__ float u2f(uint32_t u) { return __builtin_bit_cast(float, u); }

__host__ __device__ __forceinline__ void widen8_bf16(const uint4 v, float (&f)[8]) {
    f[0] = u2f(v.x << 16);
    f[1] = u2f(v.x & 0xffff0000u);
    f[2] = u2f(v.y << 16);
    f[3] = u2f(v.y & 0xffff0000u);
    f[4] = u2f(v.z << 16);
    f[5] = u2f(v.z & 0xffff0000u);
    f[6] = u2f(v.w << 16);
    f[7] = u2f(v.w & 0xffff0000u);
}

__host__ __device__ __forceinline__ float h2f(uint32_t bits16) {
    return (float) __builtin_bit_cast(_Float16, (uint16_t) bits16);
}

__host__ __device__ __forceinline__ void widen8_f16(const uint4 v, float (&f)[8]) {
    f[0] = h2f(v.x & 0xffffu);
    f[1] = h2f(v.x >> 16);
    f[2] = h2f(v.y & 0xffffu);
    f[3] = h2f(v.y >> 16);
    f[4] = h2f(v.z & 0xffffu);
    f[5] = h2f(v.z >> 16);
    f[6] = h2f(v.w & 0xffffu);
    f[7] = h2f(v.w >> 16);
}

template <int WT>
__host__ __device__ __forceinline__ void widen8(const uint4 v, float (&f)[8]) {
    if constexpr (WT == kBF16) {
        widen8_bf16(v, f);
    } else {
        widen8_f16(v, f);
    }
}

// ggml_compute_fp32_to_bf16: round to nearest even, NaN kept quiet (ggml, MIT)
__host__ __device__ __forceinline__ uint16_t f32_to_bf16(float x) {
    uint32_t u = __builtin_bit_cast(uint32_t, x);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t) ((u >> 16) | 64u);
    }
    return (uint16_t) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

// ------------------------------------------------------------------------------------------------ mat-vec

// One lane's partial sums for rows row0 .. row0 + R - 1 (clamped to N - 1) and columns 0 .. T - 1: vector index
// c = lane, lane + 32, ... (U per row in flight), 8 weights per vector, k ascending inside it. __host__ too: the CPU
// test (tests/hip/tp_w16_test.cpp) runs this exact code for all 32 lanes and the butterfly of w16_butterfly.
template <int WT, int T, int R, int U>
__host__ __device__ __forceinline__ void gemv_lane(const uint16_t* __restrict__ W, int64_t ldw, int N, int K8,
                                                   const float* __restrict__ X, int64_t ldx, int row0, int lane,
                                                   float (&acc)[R][T]) {
    const uint4* wp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r < N - 1 ? row0 + r : N - 1;
        wp[r] = reinterpret_cast<const uint4*>(W + (size_t) row * (size_t) ldw);
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = 0.0f;

    for (int c0 = lane; c0 < K8; c0 += 32 * U) {
        uint4 wv[U][R];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u;
#pragma unroll
            for (int r = 0; r < R; ++r) wv[u][r] = c < K8 ? wp[r][c] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int c = c0 + 32 * u;
            if (c < K8) {
                float wf[R][8];
#pragma unroll
                for (int r = 0; r < R; ++r) widen8<WT>(wv[u][r], wf[r]);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4* xp = reinterpret_cast<const float4*>(X + (size_t) t * (size_t) ldx) + 2 * c;
                    const float4 a = xp[0], b = xp[1];
                    const float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
                    for (int r = 0; r < R; ++r)
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[r][t] = __builtin_fmaf(wf[r][j], xs[j], acc[r][t]);
                }
            }
        }
    }
}

// The cross-lane sum: v += v[lane ^ 16], ^ 8, ^ 4, ^ 2, ^ 1 (every lane ends with the same value).
__device__ __forceinline__ float w16_butterfly(float v) {
    v += __shfl_xor(v, 16, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 2, 32);
    v += __shfl_xor(v, 1, 32);
    return v;
}

// Columns T (compile time, 1..16), R rows per wave, U 16-byte vectors per row in flight per lane.
template <int WT, int T, int R, int U>
__global__ void __launch_bounds__(256) gemv_kernel(const uint16_t* __restrict__ W, int64_t ldw, int N, int K8,
                                                  const float* __restrict__ X, int64_t ldx, float* __restrict__ Y,
                                                  int64_t ldy) {
    const int lane = threadIdx.x & 31;
    const int nw = blockDim.x >> 5;
    const int row0 = (blockIdx.x * nw + (threadIdx.x >> 5)) * R;
    if (row0 >= N) return;   // wave-uniform
    float acc[R][T];
    gemv_lane<WT, T, R, U>(W, ldw, N, K8, X, ldx, row0, lane, acc);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < T; ++t) acc[r][t] = w16_butterfly(acc[r][t]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (row0 + r < N) {
#pragma unroll
                for (int t = 0; t < T; ++t) Y[(size_t) t * (size_t) ldy + row0 + r] = acc[r][t];
            }
        }
    }
}

struct GemvCfg {
    int r = 2, u = 2, w = 4;
};

// gfx1151 defaults (rd-bf16, before the GPU bench: refine with tools/w16bench + TALUS_W16_CFG). Index = column count.
// Other architectures (gfx1201 seam) take the same table until tuned.
static const GemvCfg kGfx1151Table[kMaxCols + 1] = {
    {1, 4, 4},                                                // (unused)
    {1, 4, 4}, {2, 2, 4}, {2, 2, 4}, {2, 2, 4},               // 1..4
    {2, 2, 4}, {2, 2, 4}, {2, 2, 4}, {2, 2, 4},               // 5..8
    {2, 1, 4}, {2, 1, 4}, {2, 1, 4}, {2, 1, 4},               // 9..12
    {2, 1, 4}, {2, 1, 4}, {2, 1, 4}, {2, 1, 4},               // 13..16
};

struct Tables {
    GemvCfg cfg[kMaxCols + 1];
    bool gemm_on = true;
};

static const Tables& tables() {
    static const Tables t = [] {
        Tables tb;
        for (int i = 0; i <= kMaxCols; ++i) tb.cfg[i] = kGfx1151Table[i];
        if (const char* s = std::getenv("TALUS_W16_CFG")) {
            // "T:R:U[:W],..."
            const char* p = s;
            while (*p) {
                int tt = 0, rr = 0, uu = 0, ww = 0;
                int n = 0;
                if (std::sscanf(p, "%d:%d:%d:%d%n", &tt, &rr, &uu, &ww, &n) == 4 ||
                    (ww = 0, std::sscanf(p, "%d:%d:%d%n", &tt, &rr, &uu, &n) == 3)) {
                    const bool ok_r = rr == 1 || rr == 2 || rr == 4, ok_u = uu == 1 || uu == 2 || uu == 4;
                    for (int c = (tt == 0 ? 1 : tt); c <= (tt == 0 ? kMaxCols : tt) && c <= kMaxCols && c >= 1; ++c) {
                        if (ok_r) tb.cfg[c].r = rr;
                        if (ok_u) tb.cfg[c].u = uu;
                        if (ww >= 1 && ww <= 8) tb.cfg[c].w = ww;
                    }
                    p += n;
                } else {
                    break;
                }
                while (*p == ',' || *p == ' ' || *p == '+') ++p;
            }
        }
        if (const char* g = std::getenv("TALUS_W16_GEMM")) tb.gemm_on = !(g[0] == '0' && g[1] == 0);
        return tb;
    }();
    return t;
}

template <int WT, int T, int R, int U>
static void launch_gemv_tru(const uint16_t* W, int64_t ldw, int N, int K8, const float* X, int64_t ldx, float* Y,
                            int64_t ldy, int waves, hipStream_t s) {
    const int rows_per_block = R * waves;
    const unsigned blocks = (unsigned) ((N + rows_per_block - 1) / rows_per_block);
    gemv_kernel<WT, T, R, U><<<blocks, 32 * waves, 0, s>>>(W, ldw, N, K8, X, ldx, Y, ldy);
}

template <int WT, int T, int R>
static void launch_gemv_tr(int u, const uint16_t* W, int64_t ldw, int N, int K8, const float* X, int64_t ldx, float* Y,
                           int64_t ldy, int waves, hipStream_t s) {
    if constexpr (R == 4 && T >= 12) {   // R 4 x U 4 spills from 12 columns (ISA check): U 4 runs as U 2 there
        if (u == 4) u = 2;
    }
    switch (u) {
        case 1: launch_gemv_tru<WT, T, R, 1>(W, ldw, N, K8, X, ldx, Y, ldy, waves, s); break;
        case 4:
            if constexpr (!(R == 4 && T >= 12)) launch_gemv_tru<WT, T, R, 4>(W, ldw, N, K8, X, ldx, Y, ldy, waves, s);
            break;
        default: launch_gemv_tru<WT, T, R, 2>(W, ldw, N, K8, X, ldx, Y, ldy, waves, s); break;
    }
}

template <int WT, int T>
static void launch_gemv_t(const GemvCfg& c, const uint16_t* W, int64_t ldw, int N, int K8, const float* X,
                          int64_t ldx, float* Y, int64_t ldy, hipStream_t s) {
    if constexpr (WT == kF16) {   // F16 weights: one instance per column count (R 2, U 2; not tuned)
        launch_gemv_tru<WT, T, 2, 2>(W, ldw, N, K8, X, ldx, Y, ldy, c.w, s);
    } else {
        switch (c.r) {
            case 1: launch_gemv_tr<WT, T, 1>(c.u, W, ldw, N, K8, X, ldx, Y, ldy, c.w, s); break;
            case 4: launch_gemv_tr<WT, T, 4>(c.u, W, ldw, N, K8, X, ldx, Y, ldy, c.w, s); break;
            default: launch_gemv_tr<WT, T, 2>(c.u, W, ldw, N, K8, X, ldx, Y, ldy, c.w, s); break;
        }
    }
}

template <int WT>
static void launch_gemv(int T, const GemvCfg& c, const uint16_t* W, int64_t ldw, int N, int K8, const float* X,
                        int64_t ldx, float* Y, int64_t ldy, hipStream_t s) {
    switch (T) {
#define TP_W16_CASE(n) \
    case n: launch_gemv_t<WT, n>(c, W, ldw, N, K8, X, ldx, Y, ldy, s); break;
        TP_W16_CASE(1) TP_W16_CASE(2) TP_W16_CASE(3) TP_W16_CASE(4) TP_W16_CASE(5) TP_W16_CASE(6) TP_W16_CASE(7)
        TP_W16_CASE(8) TP_W16_CASE(9) TP_W16_CASE(10) TP_W16_CASE(11) TP_W16_CASE(12) TP_W16_CASE(13)
        TP_W16_CASE(14) TP_W16_CASE(15) TP_W16_CASE(16)
#undef TP_W16_CASE
        default: break;
    }
}

// ------------------------------------------------------------------------------------------------ GEMM

template <int WT>
__global__ void __launch_bounds__(256) round_x_kernel(const float* __restrict__ X, int64_t ldx, uint16_t* __restrict__ O,
                                                     int64_t K, int64_t T) {
    const int64_t n = T * K;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t t = i / K, k = i - t * K;
        const float v = X[t * ldx + k];
        if constexpr (WT == kBF16) {
            O[i] = f32_to_bf16(v);
        } else {
            O[i] = __builtin_bit_cast(uint16_t, (_Float16) v);
        }
    }
}

namespace g {
typedef __bf16 b16v __attribute__((ext_vector_type(16)));
typedef _Float16 h16v __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
constexpr int BM = 128, BN = 256, WN = 64, BK = 64, LDK = BK + 8, GM = 8;

template <int WT>
struct Frag;
template <>
struct Frag<kBF16> {
    using v = b16v;
};
template <>
struct Frag<kF16> {
    using v = h16v;
};

template <int WT>
__device__ __forceinline__ typename Frag<WT>::v frag(const uint16_t* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(typename Frag<WT>::v, w);
}

template <int WT>
__device__ __forceinline__ f8 mma(typename Frag<WT>::v a, typename Frag<WT>::v b, f8 c) {
#if defined(TP_W16_WMMA)
    if constexpr (WT == kBF16) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
    } else {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    }
#else
    (void) a;
    (void) b;
    __builtin_trap();   // never launched off gfx11 (runtime gate); a stray launch must not return zeros
    return c;
#endif
}

// X: M x K 16-bit (row stride ldx), W: N x K 16-bit (row stride ldw), Y: M x N fp32 (row stride ldy). K % 64 == 0.
template <int WT>
__global__ void __launch_bounds__(256) gemm_kernel(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W,
                                                  float* __restrict__ Y, int M, int N, int K, int64_t ldy, int64_t ldx,
                                                  int64_t ldw) {
    __shared__ __align__(16) uint16_t sA[BM][LDK];
    __shared__ __align__(16) uint16_t sB[BN][LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1) / BM, num_n = (N + BN - 1) / BN, b = blockIdx.x;
    const int group = b / (GM * num_n), first_m = group * GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM * num_n)) % gsize) * BM, n0 = ((b % (GM * num_n)) / gsize) * BN;
    const int sr = tid >> 3, sq = tid & 7;   // 32 rows per pass, 8 x 8 elements per row
    auto ldA = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + r, M - 1) * (size_t) ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int r) -> uint4 {
        return *reinterpret_cast<const uint4*>(W + (size_t) min(n0 + r, N - 1) * (size_t) ldw + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
        rb4 = ldB(k0, sr + 128); rb5 = ldB(k0, sr + 160); rb6 = ldB(k0, sr + 192); rb7 = ldB(k0, sr + 224);
    };
    auto store = [&]() {
        *reinterpret_cast<uint4*>(&sA[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[sr + 96][8 * sq]) = rb3;
        *reinterpret_cast<uint4*>(&sB[sr + 128][8 * sq]) = rb4; *reinterpret_cast<uint4*>(&sB[sr + 160][8 * sq]) = rb5;
        *reinterpret_cast<uint4*>(&sB[sr + 192][8 * sq]) = rb6; *reinterpret_cast<uint4*>(&sB[sr + 224][8 * sq]) = rb7;
    };
    f8 acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = f8{0, 0, 0, 0, 0, 0, 0, 0};
    load(0);
    store();
    __syncthreads();
    const int ar = 64 * wm + l16, br = WN * wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) load(k0 + BK);
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const auto a0 = frag<WT>(&sA[ar][ks]), a1 = frag<WT>(&sA[ar + 16][ks]), a2 = frag<WT>(&sA[ar + 32][ks]),
                       a3 = frag<WT>(&sA[ar + 48][ks]);
            const auto b0 = frag<WT>(&sB[br][ks]), b1 = frag<WT>(&sB[br + 16][ks]), b2 = frag<WT>(&sB[br + 32][ks]),
                       b3 = frag<WT>(&sB[br + 48][ks]);
#define TP_W16_ROW(i, a)                               \
    acc[i][0] = mma<WT>(a, b0, acc[i][0]);             \
    acc[i][1] = mma<WT>(a, b1, acc[i][1]);             \
    acc[i][2] = mma<WT>(a, b2, acc[i][2]);             \
    acc[i][3] = mma<WT>(a, b3, acc[i][3]);
            TP_W16_ROW(0, a0) TP_W16_ROW(1, a1) TP_W16_ROW(2, a2) TP_W16_ROW(3, a3)
#undef TP_W16_ROW
        }
        if (more) {
            __syncthreads();   // every wave is done reading the tile
            store();
            __syncthreads();
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + WN * wn + 16 * j + l16;
            if (n >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = m0 + 64 * wm + 16 * i + 2 * e + hi;
                if (m < M) Y[(size_t) m * (size_t) ldy + n] = acc[i][j][e];
            }
        }
}
}  // namespace g

// ------------------------------------------------------------------------------------------------ host helpers

static bool device_has_wmma() {
    static std::atomic<int> cache[64];   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        (void) hipGetLastError();
        return false;
    }
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        hipDeviceProp_t p{};
        bool ok = false;
        if (hipGetDeviceProperties(&p, dev) == hipSuccess) {
            static const char* const kArch[] = {"gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151"};
            for (const char* a : kArch) {
                const size_t n = std::strlen(a);
                if (std::strncmp(p.gcnArchName, a, n) == 0 && (p.gcnArchName[n] == 0 || p.gcnArchName[n] == ':')) ok = true;
            }
        } else {
            (void) hipGetLastError();
        }
        c = ok ? 1 : 2;
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 1;
}

static bool type_ok(int wt) { return wt == kBF16 || wt == kF16; }

static bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

static int gemv_cols(int wt, const uint16_t* W, int64_t ldw, int N, int K, const float* X, int64_t ldx, float* Y,
                     int64_t ldy, int T, hipStream_t s) {
    const Tables& tb = tables();
    const int K8 = K / 8;
    for (int t0 = 0; t0 < T; t0 += kMaxCols) {
        const int c = T - t0 < kMaxCols ? T - t0 : kMaxCols;
        const float* xc = X + (size_t) t0 * (size_t) ldx;
        float* yc = Y + (size_t) t0 * (size_t) ldy;
        if (wt == kBF16) {
            launch_gemv<kBF16>(c, tb.cfg[c], W, ldw, N, K8, xc, ldx, yc, ldy, s);
        } else {
            launch_gemv<kF16>(c, tb.cfg[c], W, ldw, N, K8, xc, ldx, yc, ldy, s);
        }
    }
    return hipGetLastError() == hipSuccess ? 0 : 2;
}

}  // namespace
}  // namespace talus::prefill::w16

using namespace talus::prefill::w16;

extern "C" int talus_w16_info(int what) {
    switch (what) {
        case 0: return 1;                                    // ABI version
        case 1: return device_has_wmma() && tables().gemm_on ? 1 : 0;
        default: return 0;
    }
}

extern "C" int talus_w16_supported(int wtype, int64_t n_out, int64_t n_in) {
    return type_ok(wtype) && n_out > 0 && n_out < (int64_t(1) << 31) && n_in > 0 && n_in % 8 == 0 &&
                   n_in < (int64_t(1) << 31)
               ? 1
               : 0;
}

extern "C" int talus_w16_gemv(int wtype, const void* w, int64_t ldw, int n_out, int n_in, const float* x, int64_t ldx,
                              float* y, int64_t ldy, int n_tok, void* stream) {
    if (n_tok <= 0) return 0;
    if (ldw <= 0) ldw = n_in;
    if (ldx <= 0) ldx = n_in;
    if (ldy <= 0) ldy = n_out;
    if (!talus_w16_supported(wtype, n_out, n_in) || !w || !x || !y || ldw < n_in || ldx < n_in || ldy < n_out ||
        ldw % 8 != 0 || ldx % 4 != 0 || !aligned16(w) || !aligned16(x)) {
        return 1;
    }
    return gemv_cols(wtype, static_cast<const uint16_t*>(w), ldw, n_out, n_in, x, ldx, y, ldy, n_tok,
                     static_cast<hipStream_t>(stream));
}

extern "C" int talus_w16_gemm(int wtype, const void* w, int64_t ldw, int n_out, int n_in, const float* x, int64_t ldx,
                              float* y, int64_t ldy, int n_tok, void* x16, int64_t scratch_elems, void* stream) {
    if (n_tok <= 0) return 0;
    if (ldw <= 0) ldw = n_in;
    if (ldx <= 0) ldx = n_in;
    if (ldy <= 0) ldy = n_out;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const bool wmma = device_has_wmma() && tables().gemm_on && n_in % 64 == 0 && ldw % 8 == 0 && aligned16(w) &&
                      x16 != nullptr && aligned16(x16) && scratch_elems >= (int64_t) n_tok * n_in;
    if (!wmma) {   // the mat-vec in 16-column slices: correct on any GPU, slow
        return talus_w16_gemv(wtype, w, ldw, n_out, n_in, x, ldx, y, ldy, n_tok, stream);
    }
    if (!talus_w16_supported(wtype, n_out, n_in) || !x || !y || ldw < n_in || ldx < n_in || ldy < n_out) return 1;
    uint16_t* xs = static_cast<uint16_t*>(x16);
    const int64_t n = (int64_t) n_tok * n_in;
    const unsigned rb = (unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096);
    const int num_m = (n_tok + g::BM - 1) / g::BM, num_n = (n_out + g::BN - 1) / g::BN;
    if (wtype == kBF16) {
        round_x_kernel<kBF16><<<rb, 256, 0, s>>>(x, ldx, xs, n_in, n_tok);
        g::gemm_kernel<kBF16><<<(unsigned) (num_m * num_n), 256, 0, s>>>(
            xs, static_cast<const uint16_t*>(w), y, n_tok, n_out, n_in, ldy, (int64_t) n_in, ldw);
    } else {
        round_x_kernel<kF16><<<rb, 256, 0, s>>>(x, ldx, xs, n_in, n_tok);
        g::gemm_kernel<kF16><<<(unsigned) (num_m * num_n), 256, 0, s>>>(
            xs, static_cast<const uint16_t*>(w), y, n_tok, n_out, n_in, ldy, (int64_t) n_in, ldw);
    }
    return hipGetLastError() == hipSuccess ? 0 : 2;
}
