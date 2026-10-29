// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh}
// and ggml/src/ggml-common.h.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "s26_tsum.cuh"
#if !defined(__HIPCC__)
#include "q8_1_il.cuh"
#endif
#include "talus_iq_split.cuh"   // Talus: i-quant / Q2_K batch traits (MiMo)
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <vector>

namespace strata::kernels {
namespace {

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;
constexpr int QUANT_THREADS = 256;

struct Q5KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q20Block {
    half d;
    uint8_t qs[16];
};
struct Q3KBlock {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
struct Q40Block {
    half d;
    uint8_t qs[16];
};
struct Q50Block {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};
struct Q80Block {
    half d;
    int8_t qs[32];
};
struct IQ4NLBlock {
    half d;
    uint8_t qs[16];
};
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 &&
              offsetof(IQ4XSBlock, scales_h) == 2 && offsetof(IQ4XSBlock, scales_l) == 4 &&
              offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 &&
              offsetof(Q5KBlock, qs) == 48 && offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 &&
              offsetof(Q4KBlock, scales) == 4 && offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 &&
              offsetof(Q6KBlock, qh) == 128 && offsetof(Q6KBlock, scales) == 192 &&
              offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 &&
              offsetof(Q50Block, qh) == 2 && offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    }
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP));
    }
    return x;
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_quantize_q8_1_kernel(const float* x_, Q81Block* y_, int n_in) {
    const float* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    Q81Block* STRATA_PDL_RESTRICT y = y_;
    pdl_trigger();   // PDL (pdl.hpp): the projection after this one may start loading its weights
    pdl_wait();
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float xi = x[i];
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

__launch_bounds__(QUANT_THREADS, 1)
__global__ void native_swiglu_quantize_q8_1_kernel(const float* __restrict__ gate,
                                                   const float* __restrict__ up,
                                                   Q81Block* __restrict__ y, int n_in) {
    const int i = int(blockIdx.x) * QUANT_THREADS + int(threadIdx.x);
    if (i >= n_in) return; // n_in is a multiple of 32: only whole warps return.
    const float gi = gate[i];
    const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), up[i]);
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = q8_1_finite(amax / 127.0f);   // #606, as native_quantize_q8_1_kernel: finite blocks bit for bit
    const int8_t q = q8_1_quant(xi, d, amax);
    y[i / Q8K].qs[i % Q8K] = q;
    if (i % Q8K == 0) y[i / Q8K].ds = q8_1_ds(d, sum);
}

// Exact pinned vec_dot_q5_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q5_q8_dot_impl(
    const int* __restrict__ vl, const int* __restrict__ vh, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm5,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * STRATA_IMUL24(dot1, sc[i]);
        sumf_m += d8[i] * STRATA_IMUL24(dot2, m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

__device__ __forceinline__ float q5_q8_dot(const Q5KBlock* __restrict__ bq5,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int vl[2];
    int vh[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q5_q8_dot_impl(vl, vh, u, sc, m, bq5->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q5_k_mmvq_kernel(const Q5KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q5_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q2_0_q8_1: each thread handles one 32-element chunk.
// The weight block is only 2-byte aligned, so qs is intentionally loaded as
// int16_t, unlike the naturally 4-byte aligned activation codes.
__device__ __forceinline__ float q2_q8_dot(const Q20Block* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const float d2 = w->d;
    const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
    const Q81Block* chunk = x + iqs;
    const int* q8 = reinterpret_cast<const int*>(chunk->qs);
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = q8[j * 2];
        const int v = q8[j * 2 + 1];
        const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
        const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = STRATA_DP4A(u, qx, sumi);
        sumi = STRATA_DP4A(v, qy, sumi);
    }
    const float d8 = __low2float(chunk->ds);
    return d2 * d8 * sumi;
}

// Q2_0 generic MMVQ: QK=64, QI=2, VDR=1, 64 blocks per iteration.
// Preserve the same outer accumulation and cross-warp reduction as the oracle.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q2_0_mmvq_kernel(const Q20Block* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 2;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 64;
    float tmp[ROWS] = {};
    for (int kbx = tid / 2; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 2;
        const int kqs = tid % 2;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q2_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Q3_K's 110-byte stride gives alternate blocks only two-byte alignment.
// Preserve the pinned helper's pair of 16-bit loads and little-endian combine.
__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}

__device__ __forceinline__ float q3_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const uint8_t* __restrict__ scales,
                                              int scale_offset, float d3,
                                              const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d3 * sumf;
}

__device__ __forceinline__ float q3_q8_dot(const Q3KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 8);
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    const float d = w->d;
    const int vl = load_int_b2(w->qs, iqs);
    const int vh = ~load_int_b2(w->hmask, iqs % 8) >> bq8_offset;
    int u[4];
    float d8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + i].ds);
    }
    return q3_q8_dot_impl(vl, vh, u, w->scales, scale_offset, d, d8);
}

// Q3_K generic MMVQ: QK=256, QI=16, VDR=1, eight blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q3_k_mmvq_kernel(const Q3KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 16;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 16; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 16;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q3_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The pinned nonlinear IQ4 codebook and its CUDA two-stage byte lookup. The
// explicit alignment satisfies the four 32-bit table loads; values are unchanged.
__device__ __align__(4) int8_t iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ int2 iq4_table_lookup(int q4) {
#if defined(__HIP_PLATFORM_AMD__) && !defined(STRATA_HIP_GFX906) && TALUS_FAST_DOTS
    // Talus (gfx11): the codebook as literal words and v_perm_b32 directly (6 per 8 values).  The
    // generic path below goes through the compat __byte_perm (selector spread, ~4 VALU each, 8 per 8 values) and
    // loads the table from global memory: ~200 VALU per 32 weights in the IQ4_XS matvec.  Same bytes out.
    constexpr uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;   // iq4nl_values
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t se = q_even & 0x07070707u, so = q_odd & 0x07070707u;
    const uint32_t el = __builtin_amdgcn_perm(t1, t0, se);
    const uint32_t ol = __builtin_amdgcn_perm(t1, t0, so);
    const uint32_t eh = __builtin_amdgcn_perm(t3, t2, se);
    const uint32_t oh = __builtin_amdgcn_perm(t3, t2, so);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#elif defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (see iq_kernels.cu get_int_from_table_16): 4 v_perm_b32 per 8 values
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(iq4nl_values);
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}

// Exact pinned vec_dot_iq4_xs_q8_1: a lane consumes one 32-element subblock,
// computes integer dot products, applies signed scale in the integer domain,
// then multiplies the two half scales and integer sum in the original order.
__device__ __forceinline__ float iq4_xs_q8_dot(const IQ4XSBlock* __restrict__ w,
                                              const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = reinterpret_cast<const int*>(w->qs)[iqs + j];
        const int2 v = iq4_table_lookup(aux_q4);
        const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
        const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
        sumi = STRATA_DP4A(v.x, u0, sumi);
        sumi = STRATA_DP4A(v.y, u1, sumi);
    }
    const int ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) |
                   (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi = STRATA_IMUL24(sumi, ls - 32);
    const float d = __half2float(w->d) * __low2float(x[iqs / 4].ds);
    return d * sumi;
}

// IQ4_XS generic MMVQ: QK=256, QI=32, VDR=4,16 blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_iq4_xs_mmvq_kernel(const IQ4XSBlock* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 4 * WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 8; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = 4 * (tid % 8);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += iq4_xs_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * STRATA_IMUL24(dot1, sc[i]);
        sumf_m += d8[i] * STRATA_IMUL24(dot2, m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock* __restrict__ bq4,
                                          const Q81Block* __restrict__ bq8, int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
                     ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
                     ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block* bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

// The generic ncols=1 oracle uses 4 warps and 1 row (or 4 rows for small K),
// eight weight blocks per K iteration, warp-ascending shared sum, then XOR tree.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q4_k_mmvq_kernel(const Q4KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * WARP / QI;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / QK;
    float tmp[ROWS] = {};
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK / Q8K);
        const int kqs = VDR * (tid % (QI / VDR));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            // The source assumes allocator padding for partial row groups. This
            // guard preserves every valid row's math without an out-of-bounds read.
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q4_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

__device__ __forceinline__ float q6_q8_dot(const Q6KBlock* __restrict__ w,
                                          const Q81Block* __restrict__ x, int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int*>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

// Q6_K generic MMVQ: QK=256, QI=32, VDR=1, four blocks per iteration.
template<bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_mmvq_kernel(const Q6KBlock* __restrict__ w,
                                        const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = WARPS * WARP / 32;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 256;
    float tmp[ROWS] = {};
    for (int kbx = tid / 32; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * 8;
        const int kqs = tid % 32;
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += q6_q8_dot(w + block, x + kby, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// The four 32-element formats use native two-byte loads and VDR=2. The affine
// Q4_0/Q5_0 correction consumes the original-input sum stored in Q8_1, exactly
// as the pinned CUDA dot does; a signed-integer code substitution would differ.
__device__ __forceinline__ float small_q8_dot(const Q40Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 4 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q50Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = STRATA_DP4A(vi0, reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = STRATA_DP4A(vi1, reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
    }
    const float2 ds = __half22float2(x->ds);
    const float d = w->d;
    return d * (sumi * ds.x - 8 * ds.y);
}

__device__ __forceinline__ float small_q8_dot(const Q80Block* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
        sumi = STRATA_DP4A(v, u, sumi);
    }
    const float d0 = w->d;
    const float d1 = __low2float(x->ds);
    return d0 * d1 * float(sumi);
}

__device__ __forceinline__ float small_q8_dot(const IQ4NLBlock* __restrict__ w,
                                             const Q81Block* __restrict__ x, int iqs) {
    const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = STRATA_DP4A(v.x, q8[i], sumi);
        sumi = STRATA_DP4A(v.y, q8[i + 4], sumi);
    }
    const float d = __half2float(w->d) * __low2float(x->ds);
    return d * sumi;
}

// QI=4 for Q4_0/Q5_0/IQ4_NL and QI=8 for Q8_0. With VDR=2 this preserves
// the pinned 64/32-block iteration and 2048/1024-element small-K thresholds.
template<typename Weight, int Qi, bool SmallK>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_small_mmvq_kernel(const Weight* __restrict__ w,
                                         const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out) {
    constexpr int ROWS = SmallK ? WARPS : 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int row0 = ROWS * int(blockIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// ============================ plan v0.3 P3: ncols = 2..8 (speculative verify, small batches) ============================
//
// One generic kernel for every format, parameterized by the format's iteration traits below, which are
// transcribed from the ncols = 1 kernels above (same thread-to-block mapping, same blocks per iteration, same
// small-K rule). Column j reads activation blocks x + j * (n_in / 32) and writes y + j * n_out. Each (column,
// row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced with
// the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call on that
// column (checked by bench/micro/native_mmvq_multi.cpp). The ncols = 1 kernels are untouched.
constexpr int MAX_NCOLS = 8;

// Each format splits its dot product into `load` (everything that depends only on the weight block: codes,
// unpacked scales, block scale) and `apply` (the activation loads and the original *_impl expression). `load` runs
// once per (row, block) and `apply` once per column, so adding columns adds only activation work. `apply` calls
// the same impl functions, in the same order, with the same values as the ncols = 1 dot, which is what keeps
// every column bitwise equal to it.
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq5->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq5->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, sc + 2, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    __device__ static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
};
// Talus: Q4_K / Q5_K for the wave32 kernels with one lane per 32-value sub-block (T = 8, as IQ4_XS): the
// lane's nibbles (and Q5_K's fifth bits) and its 6-bit scale / min decoded once per (row, block) in `load`, and per
// column one 32-value q8_1 block - two 16-byte loads and its scale - with 8 + 8 dp4a (the min term's activation sum
// is the same for every row, so the R rows of a wave share it).  The traits above (T = 16, a lane over two sub-blocks'
// 16 values) issue six 4-byte activation loads per column per 16 values.  The value of a sub-block is llama.cpp's
// vec_dot_q4_K / q5_K expression (d * sc * dot - dmin * m * sum, in fp32) over the whole sub-block in one lane, so the
// float sums group differently from the T = 16 traits: a numerics change (every column count uses these, so verify
// rows stay bitwise the decode rows).  STRATA_MMVQ_KQ8=0: the T = 16 traits.
__device__ __forceinline__ void kq_scale_min(const uint8_t* __restrict__ q, int j, int& sc, int& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
template<bool Q5>
struct KQ8Traits {
    using Block = std::conditional_t<Q5, Q5KBlock, Q4KBlock>;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 8; }
    struct W { int v[8]; int sc, m; half2 dm; };
    __device__ static W load(const Block* __restrict__ b, int j) {
        W r;
        const int k = j >> 1, sh = 4 * (j & 1);
        const int4* q = reinterpret_cast<const int4*>(b->qs + 32 * k);
        const int4 q0 = q[0], q1 = q[1];
        const int qi[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) r.v[i] = (qi[i] >> sh) & 0x0f0f0f0f;
        if constexpr (Q5) {
            const int4* h = reinterpret_cast<const int4*>(reinterpret_cast<const Q5KBlock*>(b)->qh);
            const int4 h0 = h[0], h1 = h[1];
            const int hi[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) r.v[i] |= ((hi[i] >> j) & 0x01010101) << 4;
        }
        kq_scale_min(b->scales, j, r.sc, r.m);
        r.dm = b->dm;
        return r;
    }
    template<typename XB>   // Q81Block, or Q81Lds (the LDS-staged copy)
    __device__ static float apply(const W& r, const XB* __restrict__ x, int j) {
        const int4* u4 = reinterpret_cast<const int4*>(x[j].qs);
        const int4 a = u4[0], c = u4[1];
        const int u[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
        int dot1 = 0, dot2 = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) { dot1 = STRATA_DP4A(r.v[i], u[i], dot1); dot2 = STRATA_DP4A(0x01010101, u[i], dot2); }
        const float d8 = __low2float(x[j].ds);
        const float sumf_d = d8 * STRATA_IMUL24(dot1, r.sc);
        const float sumf_m = d8 * STRATA_IMUL24(dot2, r.m);
        const float2 dm = __half22float2(r.dm);
        return dm.x * sumf_d - dm.y * sumf_m;
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    __device__ static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.d2 = w->d;
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = __byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const Q81Block* chunk = x + iqs;
        const int* q8 = reinterpret_cast<const int*>(chunk->qs);
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(q8[j * 2], r.qx[j], sumi);
            sumi = STRATA_DP4A(q8[j * 2 + 1], r.qy[j], sumi);
        }
        const float d8 = __low2float(chunk->ds);
        return r.d2 * d8 * sumi;
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    __device__ static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d;
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + i].ds);
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    template<typename XB>   // Q81Block, or Q81Lds (the LDS-staged copy)
    __device__ static float apply(const W& r, const XB* __restrict__ x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = STRATA_DP4A(r.v[j].x, u0, sumi);
            sumi = STRATA_DP4A(r.v[j].y, u1, sumi);
        }
        sumi = STRATA_IMUL24(sumi, r.ls - 32);
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
};
// The four 32-element formats: `load` decodes the two 32-bit weight chunks and block scale once per (row, block),
// and `apply` runs the exact same STRATA_DP4A and scale arithmetic per activation column as `small_q8_dot`.
template<typename Weight, int Qi>
struct SmallTraits;

template<>
struct SmallTraits<Q40Block, 4> {
    using Block = Q40Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = load_int_b2(w->qs, iqs + i);
            r.vi0[i] = (v >> 0) & 0x0f0f0f0f;
            r.vi1[i] = (v >> 4) & 0x0f0f0f0f;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const float2 ds = __half22float2(x->ds);
        return r.d * (sumi * ds.x - 4 * ds.y);
    }
};

template<>
struct SmallTraits<Q50Block, 4> {
    using Block = Q50Block;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int vi0[2], vi1[2]; float d; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        const int qh = load_int_b2(w->qh, 0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int vl = load_int_b2(w->qs, iqs + i);
            const int vh = qh >> (4 * (iqs + i));
            int vi0 = (vl >> 0) & 0x0f0f0f0f;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            r.vi0[i] = vi0;
            int vi1 = (vl >> 4) & 0x0f0f0f0f;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            r.vi1[i] = vi1;
        }
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.vi0[i], reinterpret_cast<const int*>(x->qs)[iqs + i], sumi);
            sumi = STRATA_DP4A(r.vi1[i], reinterpret_cast<const int*>(x->qs)[iqs + i + 4], sumi);
        }
        const float2 ds = __half22float2(x->ds);
        return r.d * (sumi * ds.x - 8 * ds.y);
    }
};

template<>
struct SmallTraits<Q80Block, 8> {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1, BPI = 2 * WARPS * WARP / 8;
    __device__ static int kqs(int tid) { return 2 * (tid % 4); }
    struct W { int v[2]; float d0; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    template<typename XB>   // Q81Block, or Q81Lds (the LDS-staged copy)
    __device__ static float apply(const W& r, const XB* __restrict__ x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = reinterpret_cast<const int*>(x->qs)[iqs + i];
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = __low2float(x->ds);
        return r.d0 * d1 * float(sumi);
    }
};

template<>
struct SmallTraits<IQ4NLBlock, 4> {
    using Block = IQ4NLBlock;
    static constexpr int DIV = 32, T = 2, KBY = 1, BPI = 2 * WARPS * WARP / 4;
    __device__ static int kqs(int tid) { return 2 * (tid % 2); }
    struct W { int2 v[2]; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        const int* q8 = reinterpret_cast<const int*>(x->qs) + iqs;
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sumi = STRATA_DP4A(r.v[i].x, q8[i], sumi);
            sumi = STRATA_DP4A(r.v[i].y, q8[i + 4], sumi);
        }
        const float d = r.dw * __low2float(x->ds);
        return d * sumi;
    }
};

// NW warps per block and ROWS rows per block. The EXACT layout (NW = 4, ROWS = 1, or 4 for small K) is the
// ncols = 1 layout and keeps every column bitwise equal to a single-column call. The UPSTREAM layout is
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)

// S26 STRATA_TSUM=1 (TS): warp 0's NCOLS x ROWS sums as one transposed butterfly (s26_tsum.cuh), bitwise the same
static bool s26_tsum_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
    return on;
}

// S26 STRATA_LFUSE (PAIR): blocks n_out.. compute the same rows of w2 into y2 (two matrices of one shape on one
// input in one launch; each output's code is the single matrix's)
template<typename F, int NCOLS, int NW, int ROWS, bool TS = false, bool PAIR = false>
__launch_bounds__(NW * WARP, (ROWS <= 2 ? 4 : 1))
__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w, const Q81Block* x_, float* y_,
                                         int n_in, int n_out,
                                         const typename F::Block* __restrict__ w2 = nullptr, float* __restrict__ y2 = nullptr) {
    const Q81Block* STRATA_PDL_RESTRICT x = x_;   // __restrict__ below sm_70 only (pdl.hpp, #1469)
    float* STRATA_PDL_RESTRICT y = y_;
    constexpr int BPI = F::BPI * NW / WARPS;           // blocks per iteration scale with the warp count
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    int bxi = int(blockIdx.x);
    if constexpr (PAIR) {
        if (bxi >= n_out / ROWS) { bxi -= n_out / ROWS; w = w2; y = y2; }
    }
    const int row0 = ROWS * bxi;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;                   // Q8_1 blocks per activation column
    float tmp[NCOLS][ROWS] = {};
    int kbx = tid / F::T;
    if constexpr (kPdlPrefetch) {
        // PDL (pdl.hpp): the first block's weights are loaded before waiting for the activations, then that block
        // is applied exactly as the loop below would apply it (the same rows, columns and order)
        typename F::W w0[ROWS];
        const int kqs = F::kqs(tid);
        if (kbx < blocks_per_row) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (row0 + i < n_out) w0[i] = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
        }
        pdl_trigger();
        pdl_wait();
        if (kbx < blocks_per_row) {
            const int kby = kbx * F::KBY;
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                if (row0 + i < n_out) {
#pragma unroll
                    for (int j = 0; j < NCOLS; ++j)
                        tmp[j][i] += F::apply(w0[i], x + std::size_t(j) * x_stride + kby, kqs);
                }
            }
        }
        kbx += BPI;
    }
    for (; kbx < blocks_per_row; kbx += BPI) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(tid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                const typename F::W wv = F::load(w + block, kqs);      // once per (row, block)
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)                         // then per column
                    tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
    __shared__ float partial[NW - 1 > 0 ? NW - 1 : 1][NCOLS][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    if constexpr (TS) {
        constexpr int V = NCOLS * ROWS, P = s26ts::pow2_ceil(V);
        float v[P];
#pragma unroll
        for (int j = 0; j < NCOLS; ++j)
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
#pragma unroll
                for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
                v[j * ROWS + i] = tmp[j][i];
            }
#pragma unroll
        for (int k = V; k < P; ++k) v[k] = 0.0f;
        const float sum = s26ts::tsum<P>(v, int(threadIdx.x));
        const int k = s26ts::tsum_token<P>(int(threadIdx.x));
        const int j = k / ROWS, i = k % ROWS;
        if (int(threadIdx.x) == s26ts::tsum_lane<P>(k) && k < V && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
        return;
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
#pragma unroll
            for (int l = 0; l < NW - 1; ++l) tmp[j][i] += partial[l][j][i][threadIdx.x];
            tmp[j][i] = warp_sum(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = tmp[j][i];
        }
    }
}

// MODEL-UPDATE: the dense Q8_0 projections of Unsloth's files (all of attention, SSM and the shared expert are Q8_0
// there) take one row per block where that was measured faster - the same per-row sums and reduction as two rows per
// block (each row its own partials), so every output is the same bits.  q8_gemv_bench, 2-4 columns:
//   gfx1201 (R9700): n_in <= 4096 with n_out >= 4096 (qkv 2560->10240 73-79% -> 83-84% of the 644 GB/s, gate 2560->6144
//     51-68% -> 76-78%, q 2560->12288 74-80% -> 85-86%); 6144->2560 is slower there (79% -> 55-59%) and keeps two rows.
//   sm_86 (RTX A4000): every n_out >= 2048 (+1-4% at 2-4 columns, 6144->2560 at 6 columns 70% -> 76%).
// Other cards keep two rows until measured (STRATA_NO_MMVQ_ROWS2=1 forces one row everywhere; =0 keeps two everywhere).
int dev_class() {   // 1 = gfx12, 2 = sm_86, 0 = the rest; per device
    static int cls[16];   // 0 unknown, else class + 1
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
    if (cls[dev] == 0) {
        int c = 0;
#if defined(__HIPCC__)
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess && strata::kernels::gfx_arch_is_gfx12(p.gcnArchName)) c = 1;
#else
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess && major == 8 && minor == 6)
            c = 2;
#endif
        (void) cudaGetLastError();
        cls[dev] = c + 1;
    }
    return cls[dev] - 1;
}
template<typename F>
bool q8_0_rows1(int n_in, int n_out) {
    if constexpr (!std::is_same_v<F, SmallTraits<Q80Block, 8>>) {
        (void) n_in; (void) n_out;
        return false;
    } else {
        static const int env = [] {
            const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
            return v == nullptr || v[0] == '\0' ? -1 : v[0] != '0';
        }();
        if (env == 0) return false;
        const int c = dev_class();
        return (c == 1 && n_in <= 4096 && n_out >= 4096) || (c == 2 && n_out >= 2048);
    }
}

template<typename F, int NCOLS>
void launch_multi_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
        launch_pdl(native_mmvq_multi_kernel<F, NCOLS, NW, 2, false, false>, dim3(blocks), dim3(WARP, NW), 0, s, w, x, y, n_in, n_out,
                   (const typename F::Block*) nullptr, (float*) nullptr);
        return;
    }
    const dim3 threads(WARP, WARPS);
    // the PAIR arguments, unused here (launch_pdl passes every parameter)
    const typename F::Block* const no_w2 = nullptr;
    float* const no_y2 = nullptr;
    // #783 PR-h (stuchapin909): two rows per block for every K, not only small K - each row keeps its own partial sums and
    // reduction, so a row's result does not depend on its neighbour; STRATA_NO_MMVQ_ROWS2=1 keeps one row for large K
    static const bool rows1 = [] {
        const char* v = std::getenv("STRATA_NO_MMVQ_ROWS2");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if ((rows1 || q8_0_rows1<F>(n_in, n_out)) && n_in / F::DIV >= F::BPI) {
        if (s26_tsum_on()) launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, true, false>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
        else launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, false, false>, dim3(unsigned(n_out)), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
        return;
    }
    constexpr int ROWS = 2;
    const unsigned blocks = unsigned((std::size_t(n_out) + ROWS - 1) / ROWS);
    if (s26_tsum_on()) launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS, true, false>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
    else launch_pdl(native_mmvq_multi_kernel<F, NCOLS, WARPS, ROWS, false, false>, dim3(blocks), threads, 0, s, w, x, y, n_in, n_out, no_w2, no_y2);
}

template<typename F>
void launch_multi(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                  void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: launch_multi_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: launch_multi_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: launch_multi_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: launch_multi_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: launch_multi_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: launch_multi_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: launch_multi_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ multi-column launch requires 2 <= ncols <= 8");
    }
}

#if !defined(__HIPCC__)   // CUDA only: the HIP paths keep native_mmvq's kernels
// ============================ ncols = 2..4 from interleaved activations (fork F4, Eddoursul) ============================
//
// `native_quantize_q8_1_il` also writes the columns interleaved (native_mmvq.hpp), so one load reads the same int of
// every column. A warp takes R rows and walks their blocks in the chunks of the exact layout above - its lane l's chunk
// m is what thread 32 (m % 4) + l does in iteration m / 4 - with a sum per virtual warp, added in warp order before the
// xor tree; each weight block is decoded once (the traits' `load`) and every column takes the single-column dot's own
// expression (the same *_q8_dot_impl calls), so each value is bitwise the multi-column kernel's. K-quants read the
// block-major copy; IQ4_XS, whose lanes read the same int of consecutive blocks, the position-major one.
struct IlIQ4XS {
    using F = IQ4XSTraits;
    static constexpr bool PM = true;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        const int b = kby + iqs / 4;
        int u0[4][NC], u1[4][NC];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x.u(b, j, u0[j]);
            x.u(b, j + 4, u1[j]);
        }
        float d8[NC];
        x.scales(b, d8);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            int sumi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sumi = STRATA_DP4A(r.v[j].x, u0[j][c], sumi);
                sumi = STRATA_DP4A(r.v[j].y, u1[j][c], sumi);
            }
            sumi = STRATA_IMUL24(sumi, r.ls - 32);
            const float d = r.dw * d8[c];
            out[c] = d * sumi;
        }
    }
};
struct IlQ4K {
    using F = Q4KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[4][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int b = kby + r.bq8_offset + i;
            x.u(b, (iqs / 2) % 4, u[2 * i]);
            x.u(b, (iqs / 2) % 4 + 4, u[2 * i + 1]);
            x.scales(b, d8[i]);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[4] = {u[0][c], u[1][c], u[2][c], u[3][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q4_q8_dot_impl(r.v, uc, sc, sc + 2, r.dm, dc);
        }
    }
};
struct IlQ5K {
    using F = Q5KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[4][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int b = kby + r.bq8_offset + i;
            x.u(b, (iqs / 2) % 4, u[2 * i]);
            x.u(b, (iqs / 2) % 4 + 4, u[2 * i + 1]);
            x.scales(b, d8[i]);
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[4] = {u[0][c], u[1][c], u[2][c], u[3][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q5_q8_dot_impl(r.vl, r.vh, uc, sc, sc + 2, r.dm, dc);
        }
    }
};
// Q3_K (143-dense): the Q2_0 pack's dense projections are mostly Q3_K, which only had the exact layout's kernel. The
// dot is q3_q8_dot_impl, the multi-column kernel's own call, on the same four blocks' ints and scales.
struct IlQ3K {
    using F = Q3KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[4][NC];
        float d8[4][NC];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int b = kby + r.bq8_offset + i;
            x.u(b, iqs % 8, u[i]);
            x.scales(b, d8[i]);
        }
        // q3_q8_dot_impl's arithmetic with the column-independent part done once: the four scales, and the four sets of
        // signed values (vil - vih, vih = 4 where the high bit is set, is vil | 0xFC in that byte: vil is 0..3, so nothing
        // saturates and it is __vsubss4's result)
        int vi[4], sc[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int isc = r.scale_offset + 2 * i;
            const int isc_low = isc % 8;
            const int sc_shift_low = 4 * (isc / 8);
            const int sc_low = (r.scales[isc_low] >> sc_shift_low) & 0xf;
            const int isc_high = isc % 4;
            const int sc_shift_high = 2 * (isc / 4);
            const int sc_high = ((r.scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
            sc[i] = (sc_low | sc_high) - 32;
            vi[i] = ((r.vl >> (2 * i)) & 0x03030303) | (((r.vh >> i) & 0x01010101) * 252);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            float sumf = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) sumf += d8[i][c] * (STRATA_DP4A(vi[i], u[i][c], 0) * sc[i]);
            out[c] = r.d * sumf;
        }
    }
};
struct IlQ6K {
    using F = Q6KTraits;
    static constexpr bool PM = false;
    template <int NC, class X>
    __device__ static void apply(const F::W& r, const X& x, int kby, int iqs, float (&out)[NC]) {
        int u[2][NC];
        float d8[2][NC];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            x.u(kby + r.bq8_offset + 2 * i, iqs % 8, u[i]);
            x.scales(kby + r.bq8_offset + 2 * i, d8[i]);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const int uc[2] = {u[0][c], u[1][c]};
            const float dc[2] = {d8[0][c], d8[1][c]};
            out[c] = q6_q8_dot_impl(r.vl, r.vh, uc, r.scales, r.d, dc);
        }
    }
};

template <typename I, int NC, int R>
__launch_bounds__(4 * WARP)
__global__ void native_mmvq_il_kernel(const typename I::F::Block* __restrict__ w, const int* __restrict__ xq,
                                      const float* __restrict__ xd, float* __restrict__ y, int n_in, int n_out) {
    using F = typename I::F;
    constexpr int SUB = WARP / F::T;   // weight blocks a chunk covers
    const int lane = int(threadIdx.x) & (WARP - 1);
    const auto x = q8_1_cols<NC, I::PM>(xq, xd, n_in / Q8K);
    const int bpr = n_in / F::DIV;
    const int kqs = F::kqs(lane);
    const int nm = (bpr + SUB - 1) / SUB;
    // A grid-stride loop over the row groups, though the grid gives each warp one: without it ptxas gives several
    // instances fewer registers (the fork measured Q6_K's at 3 columns 10-50% slower).
    for (int grp = int(blockIdx.x) * 4 + (int(threadIdx.x) >> 5); grp * R < n_out; grp += int(gridDim.x) * 4) {
        const int row0 = grp * R;
        float acc[4][R][NC];
#pragma unroll
        for (int v = 0; v < 4; ++v)
#pragma unroll
            for (int i = 0; i < R; ++i)
#pragma unroll
                for (int c = 0; c < NC; ++c) acc[v][i][c] = 0.0f;
        for (int m0 = 0; m0 < nm; m0 += 4) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int kbx = SUB * (m0 + v) + lane / F::T;
                if (kbx < bpr) {
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        const int row = min(row0 + i, n_out - 1);   // a partial last group recomputes its last row
                        const typename F::W wv = F::load(w + std::size_t(row) * bpr + kbx, kqs);
                        float o[NC];
                        I::template apply<NC>(wv, x, kbx * F::KBY, kqs, o);
#pragma unroll
                        for (int c = 0; c < NC; ++c) acc[v][i][c] += o[c];
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i)
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float s = acc[0][i][c];
                s += acc[1][i][c];
                s += acc[2][i][c];
                s += acc[3][i][c];
                s = warp_sum(s);
                if (lane == 0 && row0 + i < n_out) y[std::size_t(c) * n_out + row0 + i] = s;
            }
    }
}

template <typename I, int R>
void launch_il(const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const auto* w = static_cast<const typename I::F::Block*>(weights);
    const unsigned blocks = unsigned(((n_out + R - 1) / R + 3) / 4);
    if (ncols == 1) {   // 143-dense: one column reads the plain q8_1 blocks (x_il is then the caller's x_q8_1)
        native_mmvq_il_kernel<I, 1, R><<<blocks, 4 * WARP, 0, s>>>(w, static_cast<const int*>(x_il), nullptr, y, n_in, n_out);
        return;
    }
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    const int* xq = I::PM ? parts.pm : parts.bm;
    const float* xd = parts.d;
    switch (ncols) {
#define STRATA_IL_CASE(N) case N: native_mmvq_il_kernel<I, N, R><<<blocks, 4 * WARP, 0, s>>>(w, xq, xd, y, n_in, n_out); break;
        STRATA_IL_CASE(2) STRATA_IL_CASE(3) STRATA_IL_CASE(4)
#undef STRATA_IL_CASE
        default: throw std::invalid_argument("native_mmvq_il requires 1 <= ncols <= 4");
    }
}

template <typename I>
void launch_il_rows(int r, const void* weights, const void* x_il, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    switch (r) {
    case 1: launch_il<I, 1>(weights, x_il, y, n_in, n_out, ncols, s); break;
    case 4: launch_il<I, 4>(weights, x_il, y, n_in, n_out, ncols, s); break;
    default: launch_il<I, 2>(weights, x_il, y, n_in, n_out, ncols, s); break;
    }
}

// The interleaved copy (native_mmvq.hpp) of columns already quantized into plain q8_1 blocks, whoever wrote them (the
// quantizer, a fused norm): the same bytes, one int of a block a thread.
__global__ void native_q8_1_interleave_kernel(const Q81Block* __restrict__ y, int* __restrict__ bm, int* __restrict__ pm,
                                              float* __restrict__ dl, int nb, int n_total, int cp) {
    const int i = int(blockIdx.x) * 256 + int(threadIdx.x);
    if (i >= n_total) return;
    const int c = i / (nb * 8), r = i - c * nb * 8, b = r / 8, p = r % 8;
    const Q81Block& blk = y[std::size_t(c) * nb + b];
    const int q = reinterpret_cast<const int*>(blk.qs)[p];
    bm[(std::size_t(b) * 8 + p) * cp + c] = q;
    pm[(std::size_t(p) * nb + b) * cp + c] = q;
    if (p == 0) dl[std::size_t(b) * cp + c] = __low2float(blk.ds);
}

#endif  // !__HIPCC__

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0) {
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    }
    if (ncols < 1 || ncols > 16) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 16");   // (Talus: the wave32 kernels take 16; the others check their own 8)
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0) {
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
    }
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
    }
}


#if defined(STRATA_HIP_GFX906)
// ---- AMD (wave64) layout: one wavefront per R rows, four wavefronts per block, the row's blocks strided over the
// 64 lanes exactly as the CUDA kernels stride them over a block (kbx = lane / T, stride 64 / T), and a 64-lane
// butterfly instead of the LDS partials and __syncthreads of the one-block-per-row layout: on gfx906 a block per
// 2 KB row spent most of its time being launched and joined.  The SAME kernel serves every column count 1..8, so a
// column's sums do not depend on how many columns (verify tokens) ride along.  STRATA_MMVQ_WAVE=0: the CUDA layout.
bool g_wave_off = std::getenv("STRATA_MMVQ_WAVE") && std::string(std::getenv("STRATA_MMVQ_WAVE")) == "0";
template<typename F, int NCOLS, int R>
__launch_bounds__(256)
__global__ void native_mmvq_wave_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                        float* __restrict__ y, int n_in, int n_out) {
    constexpr int BPIW = 64 / F::T;
    static_assert(64 % F::T == 0, "a block's threads must tile the wavefront");
    const int lane = int(threadIdx.x) & 63;
    const int row0 = (int(blockIdx.x) * 4 + (int(threadIdx.x) >> 6)) * R;
    if (row0 >= n_out) return;
    const int blocks_per_row = n_in / F::DIV;
    const int x_stride = n_in / Q8K;
    const int kqs = F::kqs(lane);
    float tmp[NCOLS][R] = {};
    for (int kbx = lane / F::T; kbx < blocks_per_row; kbx += BPIW) {
        const int kby = kbx * F::KBY;
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int j = 0; j < NCOLS; ++j) tmp[j][i] += F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NCOLS; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, 64);
            if (lane == 0 && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
        }
}
template<typename F, int NCOLS>
void wave_launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, cudaStream_t s) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    constexpr int R = 1;
    const unsigned blocks = unsigned((std::size_t(n_out) + 4 * R - 1) / (4 * R));
    native_mmvq_wave_kernel<F, NCOLS, R><<<blocks, 256, 0, s>>>(w, x, y, n_in, n_out);
}
template<typename F>
void wave_launch(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: wave_launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 2: wave_launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 3: wave_launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 4: wave_launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 5: wave_launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 6: wave_launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 7: wave_launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
        case 8: wave_launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
        default: throw std::invalid_argument("native MMVQ (wave) requires 1 <= ncols <= 8");
    }
}
#define STRATA_WAVE_MMVQ(...) \
    if (!g_wave_off) { wave_launch<__VA_ARGS__>(weights, x_q8_1, y, n_in, n_out, ncols, stream); launch_check(); return; }
#else
#define STRATA_WAVE_MMVQ(...)
#endif

template<typename Weight, int Qi>
void small_mmvq(const void* weights, const void* x_q8_1, float* y,
                int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(SmallTraits<Weight, Qi>)
    if (ncols > 1) {
        launch_multi<SmallTraits<Weight, Qi>>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Weight*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 32 < 2 * WARPS * WARP / Qi) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_small_mmvq_kernel<Weight, Qi, true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_small_mmvq_kernel<Weight, Qi, false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

template<typename Weight, int Qi>
void small_f32(const void* weights, const float* x, void* scratch_q8_1,
               float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 32);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    small_mmvq<Weight, Qi>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

// ============================ STRATA_Q8_PACKED=1: the packed Q8_0 decode layout (opt-in) ============================
//
// A lossless load-time repack of a Q8_0 matrix: the SAME int8 values in a qs plane (row-major, n_in bytes per row)
// followed by the SAME fp16 scales in a d plane (row-major, n_in / 32 per row) - no requantization, same bytes.
// The GGUF's 34-byte blocks leave every int 2-byte aligned (load_int_b2 = two 16-bit loads); the planes give each
// thread one aligned 8-byte run, read with streaming (non-temporal) loads. Every thread owns exactly the bytes of
// the ncols = 1 / multi-column EXACT layout (block kbx = tid / 4 + 32 i, ints 2 (tid % 4) and +1), accumulates them
// in the same order with the same expression, and the cross-warp sum and XOR tree are the same: every output is
// BITWISE equal to native_small_mmvq_kernel / native_mmvq_multi_kernel<SmallTraits<Q80Block, 8>, NC, 4, 1>.
// Eligible: 1024 <= n_in <= 4096 and n_out >= 2048, run as 640 persistent workgroups (a grid-stride over rows, two
// partial buffers). Measured on gfx1151 (S26, 8 real UD-IQ4_XS layers per shape, paired, 1-5 columns): 2560 ->
// 6144/10240/12288 +10-12% per call (engine trace: -10.7% on those calls). 6144 -> 2560 (one workgroup per row)
// was +2-4% standalone but -3% in the engine, and rows < 2048 lose: neither is packed.
#if defined(__HIPCC__)
#define STRATA_Q8P_LOAD(p) __builtin_nontemporal_load(p)
#else
#define STRATA_Q8P_LOAD(p) __ldcs(p)
#endif
constexpr int Q8P_PERSIST_GRID = 640;
constexpr int Q8P_PERSIST_MAX_IN = 4096;

template<int NCOLS, bool TS = false>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q8_0_packed_kernel(const int8_t* __restrict__ qs, const half* __restrict__ dpl,
                                          const Q81Block* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int row_step) {
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int nb = n_in / 32;
    const int kqs = 2 * (tid % 4);
    __shared__ float partial[2][WARPS - 1][NCOLS][WARP];
    int buf = 0;
    for (int row = int(blockIdx.x); row < n_out; row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 4; kbx < nb; kbx += 2 * WARPS * WARP / 8) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int* p = reinterpret_cast<const int*>(qs + block * 32) + kqs;
            const int v0 = STRATA_Q8P_LOAD(p);
            const int v1 = STRATA_Q8P_LOAD(p + 1);
            const float d0 = dpl[block];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * nb + kbx;
                const int* u = reinterpret_cast<const int*>(xb->qs);
                int sumi = 0;
                sumi = STRATA_DP4A(v0, u[kqs], sumi);
                sumi = STRATA_DP4A(v1, u[kqs + 1], sumi);
                const float d1 = __low2float(xb->ds);
                tmp[j] += d0 * d1 * float(sumi);
            }
        }
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) partial[buf][threadIdx.y - 1][j][threadIdx.x] = tmp[j];
        }
        __syncthreads();
        if (TS && threadIdx.y == 0) {   // S26 STRATA_TSUM=1: one transposed butterfly, bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(threadIdx.x));
            const int j = s26ts::tsum_token<P>(int(threadIdx.x));
            if (int(threadIdx.x) == s26ts::tsum_lane<P>(j) && j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (threadIdx.y == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                tmp[j] = warp_sum(tmp[j]);
                if (threadIdx.x == 0) y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

struct Q8Packed { const int8_t* qs; const half* d; int n_in, n_out; };
std::unordered_map<const void*, Q8Packed>& q8_packed_registry() {
    static std::unordered_map<const void*, Q8Packed> registry;
    return registry;
}

template<int NCOLS>
void q8_packed_launch(const Q8Packed& w, const Q81Block* x, float* y, cudaStream_t s) {
    const bool persist = w.n_in <= Q8P_PERSIST_MAX_IN;
    const int grid = persist ? (std::min)(w.n_out, Q8P_PERSIST_GRID) : w.n_out;
    if (s26_tsum_on())
        native_q8_0_packed_kernel<NCOLS, true><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.qs, w.d, x, y, w.n_in, w.n_out,
                                                                                          persist ? grid : w.n_out);
    else
        native_q8_0_packed_kernel<NCOLS><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.qs, w.d, x, y, w.n_in, w.n_out,
                                                                                    persist ? grid : w.n_out);
}

bool q8_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q8_packed_registry();
    if (registry.empty() || (ncols > 1 && !g_multi_exact)) return false;   // packed = the EXACT layout only
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 32);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: q8_packed_launch<1>(found->second, x, y, s); break;
        case 2: q8_packed_launch<2>(found->second, x, y, s); break;
        case 3: q8_packed_launch<3>(found->second, x, y, s); break;
        case 4: q8_packed_launch<4>(found->second, x, y, s); break;
        case 5: q8_packed_launch<5>(found->second, x, y, s); break;
        case 6: q8_packed_launch<6>(found->second, x, y, s); break;
        case 7: q8_packed_launch<7>(found->second, x, y, s); break;
        case 8: q8_packed_launch<8>(found->second, x, y, s); break;
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    }
    launch_check();
    return true;
}


// ============================ STRATA_Q6_PACKED=1: the packed Q6_K head layout (opt-in) ============================
//
// The output heads (the main head, 248320 rows, and the MTP draft head's row subset) are Q6_K matrices of 2560-wide
// rows: 10 blocks of 210 bytes each, so every int load is 2-byte aligned and a row is 2100 bytes. The packed copy
// holds the SAME bytes as four planes - ql (128 B per block), qh (64 B), scales (16 B), d (2 B) - so a thread's
// two words are aligned dword loads (non-temporal). Thread ownership (block tid / 32 + 4 i, word tid % 32), the
// q6_q8_dot_impl expression, the per-thread order and the reduction are those of native_q6_k_mmvq_kernel<false> /
// native_mmvq_multi_kernel<Q6KTraits, NC, 4, 1>: every output is bitwise equal. Persistent workgroups.
int g_q6p_grid_override = 0;   // the self-test's grid sweep (STRATA_Q6P_SELFTEST=2)
int q6p_grid() {
    if (g_q6p_grid_override > 0) return g_q6p_grid_override;
    static const int grid = [] {
        const char* v = std::getenv("STRATA_Q6P_GRID");
        const int g = v ? std::atoi(v) : 0;
        return g > 0 ? g : 640;
    }();
    return grid;
}

template<int NCOLS, bool TS = false>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_q6_k_packed_kernel(const int* __restrict__ ql, const int* __restrict__ qh,
                                          const int8_t* __restrict__ sc, const half* __restrict__ dpl,
                                          const Q81Block* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int row_step) {
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int nb = n_in / 256;
    const int x_stride = n_in / Q8K;
    const int iqs = tid % 32;
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int qh_word = 8 * (iqs / 16) + iqs % 8;
    __shared__ float partial[2][WARPS - 1][NCOLS][WARP];
    int buf = 0;
    for (int row = int(blockIdx.x); row < n_out; row += row_step, buf ^= 1) {
        float tmp[NCOLS] = {};
        for (int kbx = tid / 32; kbx < nb; kbx += WARPS * WARP / 32) {
            const std::size_t block = std::size_t(row) * nb + kbx;
            const int vl = STRATA_Q8P_LOAD(ql + block * 32 + iqs);
            const int vh = STRATA_Q8P_LOAD(qh + block * 16 + qh_word) >> vh_shift;
            const int8_t* scales = sc + block * 16 + scale_offset;
            const float d = dpl[block];
            const int kby = kbx * 8;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                const Q81Block* xb = x + std::size_t(j) * x_stride + kby;
                int u[2];
                float d8[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u[i] = reinterpret_cast<const int*>(xb[bq8_offset + 2 * i].qs)[iqs % 8];
                    d8[i] = __low2float(xb[bq8_offset + 2 * i].ds);
                }
                tmp[j] += q6_q8_dot_impl(vl, vh, u, scales, d, d8);
            }
        }
        if (threadIdx.y > 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) partial[buf][threadIdx.y - 1][j][threadIdx.x] = tmp[j];
        }
        __syncthreads();
        if (TS && threadIdx.y == 0) {   // S26 STRATA_TSUM=1: one transposed butterfly, bitwise the same sums
            constexpr int P = s26ts::pow2_ceil(NCOLS);
            float v[P];
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                v[j] = tmp[j];
            }
#pragma unroll
            for (int j = NCOLS; j < P; ++j) v[j] = 0.0f;
            const float sum = s26ts::tsum<P>(v, int(threadIdx.x));
            const int j = s26ts::tsum_token<P>(int(threadIdx.x));
            if (int(threadIdx.x) == s26ts::tsum_lane<P>(j) && j < NCOLS) y[std::size_t(j) * n_out + row] = sum;
        } else if (threadIdx.y == 0) {
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int l = 0; l < WARPS - 1; ++l) tmp[j] += partial[buf][l][j][threadIdx.x];
                tmp[j] = warp_sum(tmp[j]);
                if (threadIdx.x == 0) y[std::size_t(j) * n_out + row] = tmp[j];
            }
        }
    }
}

bool g_q6_bypass = false;   // the load-time self-test times the GGUF-layout kernels with the copy registered

// the packed planes of one Q6_K matrix (owned: allocated by native_q6_k_pack, freed by native_q6_k_unpack)
struct Q6Packed { void* base; const int* ql; const int* qh; const int8_t* sc; const half* d; int n_in, n_out; };
std::unordered_map<const void*, Q6Packed>& q6_packed_registry() {
    static std::unordered_map<const void*, Q6Packed> registry;
    return registry;
}

template<int NCOLS>
void q6_packed_launch(const Q6Packed& w, const Q81Block* x, float* y, cudaStream_t s) {
    const int grid = (std::min)(w.n_out, q6p_grid());
    if (s26_tsum_on())
        native_q6_k_packed_kernel<NCOLS, true><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.ql, w.qh, w.sc, w.d, x, y, w.n_in,
                                                                                          w.n_out, grid);
    else
        native_q6_k_packed_kernel<NCOLS><<<unsigned(grid), dim3(WARP, WARPS), 0, s>>>(w.ql, w.qh, w.sc, w.d, x, y, w.n_in,
                                                                                    w.n_out, grid);
}

bool q6_packed_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    auto& registry = q6_packed_registry();
    if (registry.empty() || g_q6_bypass || (ncols > 1 && !g_multi_exact)) return false;
    const auto found = registry.find(weights);
    if (found == registry.end() || found->second.n_in != n_in || found->second.n_out != n_out) return false;
    validate_shape(n_in, ncols, 256);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 1: q6_packed_launch<1>(found->second, x, y, s); break;
        case 2: q6_packed_launch<2>(found->second, x, y, s); break;
        case 3: q6_packed_launch<3>(found->second, x, y, s); break;
        case 4: q6_packed_launch<4>(found->second, x, y, s); break;
        case 5: q6_packed_launch<5>(found->second, x, y, s); break;
        case 6: q6_packed_launch<6>(found->second, x, y, s); break;
        case 7: q6_packed_launch<7>(found->second, x, y, s); break;
        case 8: q6_packed_launch<8>(found->second, x, y, s); break;
        default: throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
    }
    launch_check();
    return true;
}
} // namespace

bool native_q8_0_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q8_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

bool native_q8_0_packed_eligible(int n_in, int n_out) {
    return n_in > 0 && n_in % 32 == 0 && n_in / 32 >= 2 * WARPS * WARP / 8 && n_in <= Q8P_PERSIST_MAX_IN &&
           n_out >= 2048;
}

void native_q8_0_pack_host(const void* gguf_blocks, void* out, int n_in, int n_out) {
    const auto* blocks = static_cast<const Q80Block*>(gguf_blocks);
    auto* qs = static_cast<int8_t*>(out);
    auto* d = reinterpret_cast<half*>(qs + std::size_t(n_in) * n_out);
    const std::size_t count = std::size_t(n_in / 32) * n_out;
    for (std::size_t b = 0; b < count; ++b) {
        std::memcpy(qs + b * 32, blocks[b].qs, 32);
        std::memcpy(d + b, &blocks[b].d, sizeof(half));
    }
}

void native_q8_0_packed_register(const void* gguf_weights, const void* packed, int n_in, int n_out) {
    validate_pointer(gguf_weights);
    validate_pointer(packed);
    if (!native_q8_0_packed_eligible(n_in, n_out)) throw std::invalid_argument("Q8_0 packed: ineligible shape");
    const auto* qs = static_cast<const int8_t*>(packed);
    q8_packed_registry()[gguf_weights] =
        Q8Packed{qs, reinterpret_cast<const half*>(qs + std::size_t(n_in) * n_out), n_in, n_out};
}

void native_q8_0_packed_unregister(const void* gguf_weights) { q8_packed_registry().erase(gguf_weights); }

bool native_q6_k_packed_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_Q6_PACKED");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    return enabled;
}

bool native_q6_k_pack(const void* weights, int n_in, int n_out, const char* what) {
    if (!weights || n_in <= 0 || n_in % 256 || n_in / 256 < WARPS * WARP / 32 || n_out < 2048) return false;
    if (q6_packed_registry().count(weights)) return true;
    const std::size_t nb = std::size_t(n_in / 256) * std::size_t(n_out);
    const std::size_t bytes = nb * sizeof(Q6KBlock);
    std::vector<Q6KBlock> host(nb);
    if (cudaMemcpy(host.data(), weights, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { cudaGetLastError(); return false; }
    // planes: ql | qh | scales | d, each 256-byte aligned
    const std::size_t o_qh = (nb * 128 + 255) & ~std::size_t(255);
    const std::size_t o_sc = (o_qh + nb * 64 + 255) & ~std::size_t(255);
    const std::size_t o_d = (o_sc + nb * 16 + 255) & ~std::size_t(255);
    const std::size_t total = o_d + nb * 2;
    std::vector<uint8_t> packed(total, 0);
    for (std::size_t b = 0; b < nb; ++b) {
        std::memcpy(packed.data() + b * 128, host[b].ql, 128);
        std::memcpy(packed.data() + o_qh + b * 64, host[b].qh, 64);
        std::memcpy(packed.data() + o_sc + b * 16, host[b].scales, 16);
        std::memcpy(packed.data() + o_d + b * 2, &host[b].d, 2);
    }
    void* dev = nullptr;
    if (cudaMalloc(&dev, total) != cudaSuccess) { cudaGetLastError(); return false; }
    if (cudaMemcpy(dev, packed.data(), total, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev); cudaGetLastError(); return false;
    }
    auto* base = static_cast<uint8_t*>(dev);
    q6_packed_registry()[weights] = Q6Packed{dev, reinterpret_cast<const int*>(base),
                                             reinterpret_cast<const int*>(base + o_qh),
                                             reinterpret_cast<const int8_t*>(base + o_sc),
                                             reinterpret_cast<const half*>(base + o_d), n_in, n_out};
    std::fprintf(stderr, "talus: STRATA_Q6_PACKED=1 packed the %s (%d x %d, +%.1f MiB)\n", what, n_out, n_in,
                 total / 1048576.0);
    // STRATA_Q6P_SELFTEST=1: bitwise check against the GGUF-layout kernels (1..4 columns) and paired timings
    if (const char* v = std::getenv("STRATA_Q6P_SELFTEST"); v && (v[0] == '1' || v[0] == '2')) {
        cudaStream_t s = nullptr;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        const int nq = n_in / Q8K;
        std::vector<Q81Block> hx(std::size_t(4) * nq);
        uint32_t r = 12345u;
        for (auto& blk : hx) {
            for (auto& q : blk.qs) { r = r * 1664525u + 1013904223u; q = int8_t(int((r >> 24) & 255u) - 128); }
            r = r * 1664525u + 1013904223u;
            blk.ds = __floats2half2_rn(0.001f + float(r >> 8) * 1e-9f, 0.0f);
        }
        void *dx = nullptr, *ya = nullptr, *yb = nullptr;
        cudaMalloc(&dx, hx.size() * sizeof(Q81Block));
        cudaMalloc(&ya, std::size_t(4) * n_out * 4);
        cudaMalloc(&yb, std::size_t(4) * n_out * 4);
        cudaMemcpy(dx, hx.data(), hx.size() * sizeof(Q81Block), cudaMemcpyHostToDevice);
        std::vector<float> a(std::size_t(4) * n_out), c(std::size_t(4) * n_out);
        bool same = true;
        for (int nc = 1; nc <= 4; ++nc) {
            g_q6_bypass = true;
            native_q6_k_mmvq(weights, dx, (float*) ya, n_in, n_out, nc, s);
            g_q6_bypass = false;
            native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
            cudaStreamSynchronize(s);
            cudaMemcpy(a.data(), ya, std::size_t(nc) * n_out * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(c.data(), yb, std::size_t(nc) * n_out * 4, cudaMemcpyDeviceToHost);
            same = same && std::memcmp(a.data(), c.data(), std::size_t(nc) * n_out * 4) == 0;
        }
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int nc : {1, 2, 4}) {
            std::vector<float> tb, tc;
            for (int k = 0; k < 11; ++k) {
                for (int pass = 0; pass < 2; ++pass) {
                    const bool cand = (pass == 0) == (k % 2 == 1);
                    g_q6_bypass = !cand;
                    cudaEventRecord(e0, s);
                    for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) (cand ? yb : ya), n_in, n_out, nc, s);
                    cudaEventRecord(e1, s);
                    cudaEventSynchronize(e1);
                    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                    (cand ? tc : tb).push_back(ms * 1000.0f / 5);
                }
            }
            g_q6_bypass = false;
            std::sort(tb.begin(), tb.end()); std::sort(tc.begin(), tc.end());
            std::fprintf(stderr, "talus: Q6_K packed self-test %s ncols %d: GGUF %.1f us (%.1f-%.1f), packed %.1f us "
                         "(%.1f-%.1f), 11 paired samples\n", what, nc, tb[5], tb[0], tb[10], tc[5], tc[0], tc[10]);
        }
        std::fprintf(stderr, "talus: Q6_K packed self-test %s: outputs %s (1-4 columns)\n", what,
                     same ? "BITWISE EQUAL" : "DIFFER");
        if (v[0] == '2') {   // grid sweep of the packed kernel
            for (int grid : {160, 320, 640, 1280, 2560, 5120, 20480}) {
                g_q6p_grid_override = grid;
                for (int nc : {1, 2, 4}) {
                    std::vector<float> tc;
                    for (int k = 0; k < 7; ++k) {
                        cudaEventRecord(e0, s);
                        for (int it = 0; it < 5; ++it) native_q6_k_mmvq(weights, dx, (float*) yb, n_in, n_out, nc, s);
                        cudaEventRecord(e1, s);
                        cudaEventSynchronize(e1);
                        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                        tc.push_back(ms * 1000.0f / 5);
                    }
                    std::sort(tc.begin(), tc.end());
                    std::fprintf(stderr, "talus: Q6_K packed grid sweep %s grid %d ncols %d: %.1f us (%.1f-%.1f)\n",
                                 what, grid, nc, tc[3], tc[0], tc[6]);
                }
            }
            g_q6p_grid_override = 0;
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(dx); cudaFree(ya); cudaFree(yb);
        cudaStreamDestroy(s);
        if (!same) { native_q6_k_unpack(weights); return false; }
    }
    return true;
}

void native_q6_k_unpack(const void* weights) {
    auto& registry = q6_packed_registry();
    const auto found = registry.find(weights);
    if (found == registry.end()) return;
    cudaFree(found->second.base);
    registry.erase(found);
}

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    // Columns are contiguous and n_in is a multiple of 32, so ncols columns quantize as one vector of
    // ncols * n_in elements: every 32-element block stays inside one column.
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    launch_pdl(native_quantize_q8_1_kernel, dim3(blocks), dim3(QUANT_THREADS), 0, static_cast<cudaStream_t>(stream), x,
               static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_swiglu_quantize_q8_1(const float* gate, const float* up, void* x_q8_1,
                                 int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(gate);
    validate_pointer(up);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    const int n_total = n_in * ncols;
    const unsigned blocks = unsigned((std::size_t(n_total) + QUANT_THREADS - 1) / QUANT_THREADS);
    native_swiglu_quantize_q8_1_kernel<<<blocks, QUANT_THREADS, 0,
                                         static_cast<cudaStream_t>(stream)>>>(
        gate, up, static_cast<Q81Block*>(x_q8_1), n_total);
    launch_check();
}

void native_q5_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q5KTraits)
    if (ncols > 1) {
        launch_multi<Q5KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q5KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / QK < VDR * WARPS * WARP / QI) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q5_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q5_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    // Validate all outputs before enqueueing the first operation.
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q5_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q2_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q20Traits)
    if (ncols > 1) {
        launch_multi<Q20Traits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q20Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 64 < WARPS * WARP / 2) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q2_0_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q2_0_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q2_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 64);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q2_0_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q3_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q3KTraits)
    if (ncols > 1) {
        launch_multi<Q3KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q3KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q3_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q3_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q3_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q3_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_xs_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(IQ4XSTraits)
    if (ncols > 1) {
        launch_multi<IQ4XSTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const IQ4XSBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < 4 * WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_iq4_xs_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_iq4_xs_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_iq4_xs_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_iq4_xs_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q4KTraits)
    if (ncols > 1) {
        launch_multi<Q4KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q4KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 16) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q4_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q4_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q4_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q4_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y,
                      int n_in, int n_out, int ncols, void* stream) {
    if (q6_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q6_PACKED=1 heads
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    STRATA_WAVE_MMVQ(Q6KTraits)
    if (ncols > 1) {
        launch_multi<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
        launch_check();
        return;
    }
    const auto* w = static_cast<const Q6KBlock*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    const dim3 threads(WARP, WARPS);
    if (n_in / 256 < WARPS * WARP / 32) {
        const unsigned blocks = unsigned((std::size_t(n_out) + WARPS - 1) / WARPS);
        native_q6_k_mmvq_kernel<true><<<blocks, threads, 0, s>>>(w, x, y, n_in, n_out);
    } else {
        native_q6_k_mmvq_kernel<false><<<unsigned(n_out), threads, 0, s>>>(w, x, y, n_in, n_out);
    }
    launch_check();
}

void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1,
                     float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, 256);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q40Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q4_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q40Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<Q50Block, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q5_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q50Block, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

template<int NCOLS>
void q8_0_pair_launch(const Q80Block* w1, const Q80Block* w2, const Q81Block* x, float* y1, float* y2, int n_in, int n_out,
                      cudaStream_t s) {
    using F = SmallTraits<Q80Block, 8>;
    const dim3 threads(WARP, WARPS);
    if (s26_tsum_on())
        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, true, true><<<unsigned(2 * n_out), threads, 0, s>>>(w1, x, y1, n_in, n_out, w2, y2);
    else
        native_mmvq_multi_kernel<F, NCOLS, WARPS, 1, false, true><<<unsigned(2 * n_out), threads, 0, s>>>(w1, x, y1, n_in, n_out, w2, y2);
}

bool native_mmvq_pair(int ggml_type, const void* w1, const void* w2, const void* x_q8_1, float* y1, float* y2,
                      int n_in, int n_out, int ncols, void* stream) {
    using F = SmallTraits<Q80Block, 8>;
    // only where both calls would run native_mmvq_multi_kernel<F, ncols, WARPS, 1> (exact layout, not packed)
    if (ggml_type != 8 || !g_multi_exact || ncols < 2 || ncols > 8 || n_out <= 0 || n_in <= 0 || n_in % 32 != 0 ||
        n_in / F::DIV < F::BPI || !w1 || !w2 || !x_q8_1 || !y1 || !y2 || !stream)
        return false;
    const auto& reg = q8_packed_registry();
    if (reg.count(w1) || reg.count(w2)) return false;
    const auto* a = static_cast<const Q80Block*>(w1);
    const auto* b = static_cast<const Q80Block*>(w2);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ncols) {
        case 2: q8_0_pair_launch<2>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 3: q8_0_pair_launch<3>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 4: q8_0_pair_launch<4>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 5: q8_0_pair_launch<5>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 6: q8_0_pair_launch<6>(a, b, x, y1, y2, n_in, n_out, s); break;
        case 7: q8_0_pair_launch<7>(a, b, x, y1, y2, n_in, n_out, s); break;
        default: q8_0_pair_launch<8>(a, b, x, y1, y2, n_in, n_out, s); break;
    }
    launch_check();
    return true;
}

void native_q8_0_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    if (q8_packed_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // STRATA_Q8_PACKED=1 tensors
    small_mmvq<Q80Block, 8>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_q8_0_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<Q80Block, 8>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_mmvq(const void* weights, const void* x_q8_1, float* y,
                       int n_in, int n_out, int ncols, void* stream) {
    small_mmvq<IQ4NLBlock, 4>(weights, x_q8_1, y, n_in, n_out, ncols, stream);
}

void native_iq4_nl_f32(const void* weights, const float* x, void* scratch_q8_1,
                      float* y, int n_in, int n_out, int ncols, void* stream) {
    small_f32<IQ4NLBlock, 4>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29 ||
           (ggml_type == 10 && native_mmvq_w32_supported(10));   // Q2_K: the gfx11 wave32 kernels only (Talus)
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 10: block_elems = 256; block_bytes = 84; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out)) {
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    }
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y,
                 int n_in, int n_out, int ncols, void* stream) {
    if (native_mmvq_w32(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream)) return;   // gfx11 (Talus)
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

#if !defined(__HIPCC__)
std::size_t native_q8_1_il_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(n_in / Q8K) * std::size_t(native_q8_1_il_cp(ncols)) * (16 * sizeof(int) + sizeof(float));
}

void native_q8_1_interleave(const void* x_q8_1, void* x_il, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    if (ncols < 2 || ncols > 4) throw std::invalid_argument("native_q8_1_interleave requires 2 <= ncols <= 4");
    validate_pointer(x_q8_1);
    validate_pointer(x_il);
    validate_stream(stream);
    const int nb = n_in / Q8K, cp = native_q8_1_il_cp(ncols);
    const Q81IlParts parts = q8_1_il_parts(x_il, n_in, ncols);
    const int n_total = nb * 8 * ncols;
    native_q8_1_interleave_kernel<<<unsigned((n_total + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const Q81Block*>(x_q8_1), const_cast<int*>(parts.bm), const_cast<int*>(parts.pm),
        const_cast<float*>(parts.d), nb, n_total, cp);
    launch_check();
}

namespace {
// Rows a warp at 2-4 columns ([ncols - 2]) by the matrix's row count, in classes: below 2048 (the keys' 512), 4096 (the
// output projections' 2560), 8192 (the gate projection's 6144), 12288 (qkv's 10240), and more (the queries' 12288, the
// head's 248K); 0: native_mmvq's kernels. The fork tuned its table on an RTX 3090. This one is read off
// mmvq_il_parity --bench on an RTX 3060 (sm_86) and an RTX 5070 (sm_120) together: per cell the rows count (1/2/4) whose
// worse card still takes at least 3% off native_mmvq's time, else 0; classes without a measured shape take their
// neighbour's value. Every choice is bitwise the same output, so the table is only speed.
struct IlRows { int type; uint8_t r[3][5]; };
constexpr IlRows kIlRowsShared[] = {   // sm_86 + sm_120, and the fallback for every other sm_80+ card
    {23, {{0, 0, 0, 0, 0}, {0, 0, 2, 4, 4}, {0, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}}},   // Q4_K
    {13, {{0, 0, 1, 1, 1}, {0, 4, 1, 1, 1}, {0, 1, 1, 1, 2}}},   // Q5_K
    {14, {{0, 0, 1, 1, 0}, {0, 0, 1, 1, 2}, {0, 0, 1, 1, 1}}},   // Q6_K
};
// Per-architecture tables (P10).  A card whose compute capability (major * 10 + minor) is listed here uses its own table;
// every other sm_80+ card uses kIlRowsShared.  sm_89 (Ada) has none yet: nobody measured it (we have no such card), so
// it takes the shared table, which sits between the measured sm_86 and sm_120.  To add one, run
// `mmvq_il_parity --bench --emit-table` on the card (docs/MMVQ_IL_TABLE.md), paste the printed block as a
// `constexpr IlRows kIlRows_89[]` and add {89, kIlRows_89, sizeof(kIlRows_89) / sizeof(IlRows)} below.  Each choice
// is bitwise the same output, so a table is only speed.  Without a rebuild: STRATA_MMVQ_IL_ROWS (see il_env_rows).
// Volta (sm_70): read off mmvq_il_parity --bench on a V100-SXM2-32GB (two runs, mean), the same rule - per cell the
// rows count whose every measured shape takes at least 3% off native_mmvq's time (e.g. the Q5_K head at 3 columns
// 1094 -> 821 us, Q6_K 12288 rows 68.9 -> 47.9 us), else 0; unmeasured classes take a neighbour's value (PR 1401)
constexpr IlRows kIlRows_70[] = {
    {23, {{0, 2, 2, 2, 2}, {0, 2, 2, 2, 2}, {0, 4, 2, 2, 2}}},   // IQ4_XS
    {12, {{0, 2, 2, 4, 4}, {0, 2, 2, 2, 2}, {0, 2, 2, 2, 2}}},   // Q4_K
    {13, {{0, 2, 2, 2, 2}, {0, 2, 2, 2, 2}, {0, 1, 2, 2, 2}}},   // Q5_K
    {14, {{0, 1, 1, 1, 1}, {0, 2, 2, 2, 2}, {0, 2, 2, 4, 2}}},   // Q6_K
};
// Pascal (sm_60), read off mmvq_il_parity --bench --emit-table on a Tesla P100-PCIE-16GB (three runs, identical
// tables), the same 3% rule.  Pascal has no dp4a (strata_dp4a: four vmad), so the kernels are compute-bound and more rows a
// warp (weights and activations reused across rows) win 15-35% on the large shapes.  Bitwise equal to native_mmvq.
constexpr IlRows kIlRows_60[] = {
    {23, {{0, 4, 4, 4, 4}, {2, 4, 2, 4, 4}, {1, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 4, 4, 4, 4}, {0, 2, 2, 2, 2}, {0, 0, 0, 4, 4}}},   // Q4_K
    {13, {{0, 2, 2, 2, 2}, {0, 4, 2, 2, 4}, {0, 2, 2, 2, 2}}},   // Q5_K
    {14, {{0, 2, 2, 2, 4}, {0, 0, 0, 4, 4}, {0, 0, 0, 4, 4}}},   // Q6_K
};
// Pascal with dp4a (sm_61), read off mmvq_il_parity --bench --emit-table on a Tesla P40 (three runs: a cell is non-zero
// only where every run picked a rows count that takes at least 3% off native_mmvq's time; where the runs disagreed on
// the count, the one that won twice).  E.g. IQ4_XS 2560x10240 at 2 columns 112 -> 65 us, Q6_K 12288x2560 at 4
// columns 189 -> 127 us, the Q5_K head at 3 columns 2545 -> 1766 us.  Bitwise equal to native_mmvq.
constexpr IlRows kIlRows_61[] = {
    {23, {{2, 4, 4, 4, 4}, {4, 4, 4, 4, 4}, {2, 2, 2, 2, 2}}},   // IQ4_XS
    {12, {{2, 0, 4, 4, 4}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}},   // Q4_K
    {13, {{2, 0, 4, 4, 4}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}}},   // Q5_K
    {14, {{0, 0, 2, 2, 2}, {0, 4, 4, 4, 4}, {2, 4, 2, 2, 2}}},   // Q6_K
    {11, {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {1, 1, 1, 1, 1}}},   // Q3_K
};
// RTX 30-series (sm_86), 143-dense: read off mmvq_il_parity --bench --emit-table on an RTX A4000 with the engine's real
// shapes (the output projections have 6144 inputs, not 4096), per cell the rows count whose worst measured shape is under
// 0.97 of native_mmvq's time. The four older rows are the shared table's; Q3_K (the Q2_0 pack's main dense type) is new.
constexpr IlRows kIlRows_86[] = {
    {23, {{0, 0, 0, 0, 0}, {0, 0, 2, 4, 4}, {0, 1, 1, 1, 1}}},   // IQ4_XS
    {12, {{0, 0, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}}},   // Q4_K
    {13, {{0, 0, 1, 1, 1}, {0, 4, 1, 1, 1}, {0, 1, 1, 1, 2}}},   // Q5_K
    {14, {{0, 0, 1, 1, 0}, {0, 0, 1, 1, 2}, {0, 0, 1, 1, 1}}},   // Q6_K
    {11, {{1, 2, 2, 2, 2}, {1, 1, 1, 2, 2}, {1, 1, 1, 1, 1}}},   // Q3_K (two runs; 2560x10240 at 4 columns 120 -> 70 us)
};
// One column: rows a warp by n_out class, 0 = native_mmvq's single-column kernels (STRATA_MMVQ_IL_T1=0 turns this off).
struct Il1Rows { int type; uint8_t r[5]; };
constexpr Il1Rows kIl1Rows_86[] = {
    {11, {0, 1, 1, 1, 1}},   // Q3_K: 2560x10240 60 -> 43 us, 6144x2560 30 -> 26 us on the A4000
};
// Tesla P40 (sm_61), the same three runs: IQ4_XS 2560x10240 102 -> 58 us, the Q5_K head 1754 -> 1507 us
constexpr Il1Rows kIl1Rows_61[] = {
    {23, {2, 2, 4, 4, 4}},   // IQ4_XS
    {13, {0, 0, 0, 2, 2}},   // Q5_K
    {11, {0, 0, 2, 2, 2}},   // Q3_K
};
struct Il1Arch { int cc; const Il1Rows* t; size_t n; };
constexpr Il1Arch kIl1Arch[] = {
    {61, kIl1Rows_61, sizeof(kIl1Rows_61) / sizeof(Il1Rows)},
    {86, kIl1Rows_86, sizeof(kIl1Rows_86) / sizeof(Il1Rows)},
};
bool il_env_on(const char* name) {
    const char* v = std::getenv(name);
    return v == nullptr || v[0] != '0';
}
bool il_t1_on() { static const bool on = il_env_on("STRATA_MMVQ_IL_T1"); return on; }
bool il_q3k_on() { static const bool on = il_env_on("STRATA_MMVQ_IL_Q3K"); return on; }
struct IlArch { int cc; const IlRows* t; size_t n; };
constexpr IlArch kIlArch[] = {
    {60, kIlRows_60, sizeof(kIlRows_60) / sizeof(IlRows)},   // Pascal GP100: measured on a Tesla P100
    {61, kIlRows_61, sizeof(kIlRows_61) / sizeof(IlRows)},   // Pascal with dp4a: measured on a Tesla P40
    {70, kIlRows_70, sizeof(kIlRows_70) / sizeof(IlRows)},   // Volta: PR 1401, measured on a V100-SXM2
    {86, kIlRows_86, sizeof(kIlRows_86) / sizeof(IlRows)},   // 143-dense: RTX A4000 with the real shapes (Q3_K added)
};
// STRATA_MMVQ_IL_ROWS="type:ncols:r0,r1,r2,r3,r4;..." overrides single rows of the table for every card (type = ggml
// type 23/12/13/14, ncols 2-4, r* = rows a warp 0/1/2/4 for the five n_out classes).  Parsed once.
struct IlEnvRow { int type, ncols; uint8_t r[5]; };
const std::vector<IlEnvRow>& il_env_rows() {
    static const std::vector<IlEnvRow> v = [] {
        std::vector<IlEnvRow> out;
        const char* e = std::getenv("STRATA_MMVQ_IL_ROWS");
        if (!e) return out;
        std::string str(e);
        size_t pos = 0;
        while (pos < str.size()) {
            size_t end = str.find(';', pos);
            if (end == std::string::npos) end = str.size();
            int t = 0, n = 0, r[5] = {0, 0, 0, 0, 0}, used = 0;
            if (std::sscanf(str.substr(pos, end - pos).c_str(), "%d:%d:%d,%d,%d,%d,%d%n", &t, &n, &r[0], &r[1], &r[2], &r[3],
                            &r[4], &used) >= 7 && n >= 1 && n <= 4) {
                IlEnvRow x{t, n, {}};
                bool ok = true;
                for (int i = 0; i < 5; ++i) { ok = ok && (r[i] == 0 || r[i] == 1 || r[i] == 2 || r[i] == 4); x.r[i] = (uint8_t) r[i]; }
                if (ok) out.push_back(x);
                else std::fprintf(stderr, "talus: STRATA_MMVQ_IL_ROWS: rows must be 0/1/2/4 - entry ignored\n");
            } else {
                std::fprintf(stderr, "talus: STRATA_MMVQ_IL_ROWS: cannot read '%s' (type:ncols:r0,r1,r2,r3,r4) - ignored\n",
                             str.substr(pos, end - pos).c_str());
            }
            pos = end + 1;
        }
        return out;
    }();
    return v;
}
}  // namespace
// The rows a warp for a card of compute capability `cc` (major * 10 + minor); 0: native_mmvq's kernels.  Pure, so a test
// can check every architecture's table without that card.
int native_mmvq_il_rows_for(int cc, int type, int ncols, int n_out) {
    if (ncols < 1 || ncols > 4) return 0;
    const int cls = n_out < 2048 ? 0 : n_out < 4096 ? 1 : n_out < 8192 ? 2 : n_out < 12288 ? 3 : 4;
    for (const IlEnvRow& e : il_env_rows())
        if (e.type == type && e.ncols == ncols) return e.r[cls];
    if (ncols == 1) {   // 143-dense: one column (the plain q8_1 blocks, no interleaved copy); STRATA_MMVQ_IL_T1=0 turns it off
        if (!il_t1_on()) return 0;
        for (const Il1Arch& a : kIl1Arch)
            if (a.cc == cc)
                for (size_t i = 0; i < a.n; ++i)
                    if (a.t[i].type == type) return a.t[i].r[cls];
        return 0;
    }
    if (type == 11 && !il_q3k_on()) return 0;   // STRATA_MMVQ_IL_Q3K=0: Q3_K keeps native_mmvq's kernels
    const IlRows* t = kIlRowsShared;
    size_t n = sizeof(kIlRowsShared) / sizeof(IlRows);
    for (const IlArch& a : kIlArch)
        if (a.cc == cc && a.t != nullptr) { t = a.t; n = a.n; }
    for (size_t i = 0; i < n; ++i)
        if (t[i].type == type) return t[i].r[ncols - 2][cls];
    return 0;
}
namespace {
int il_cc() {
    static int cc[16] = {};   // by device ordinal, 0 unknown
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
    if (cc[dev] == 0) {
        int maj = 0, mnr = 0;
        if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) { cudaGetLastError(); return 0; }
        cc[dev] = maj * 10 + mnr;
    }
    return cc[dev];
}
int il_rows(int type, int ncols, int n_out) { return native_mmvq_il_rows_for(il_cc(), type, ncols, n_out); }
int g_tune_rows = 0;   // native_mmvq_il_tune (tests, benchmarks): rows a warp for every shape (0: the table)
}  // namespace
void native_mmvq_il_tune(int rows) { g_tune_rows = rows; }

namespace {
// sm_80 and newer (measured on sm_86 and sm_120), Volta (sm_70) and Pascal (sm_60, sm_61), each with its own table
// above; Turing keeps native_mmvq's kernels unchanged
// Volta (PR 1401): opt-in with STRATA_SM70_TABLE=1 until the author confirms on a V100 with the final code
bool sm70_opt_in() { const char* v = std::getenv("STRATA_SM70_TABLE"); return v != nullptr && std::atoi(v) != 0; }
// Pascal (sm_60, GP100): the same interleaved kernels (bitwise equal to native_mmvq's, mmvq_il_parity), its own table
// kIlRows_60.  STRATA_SM60_TABLE=0 keeps native_mmvq's kernels on a P100; any other value (or unset) uses the table.
bool sm60_table_on() { const char* v = std::getenv("STRATA_SM60_TABLE"); return v == nullptr || std::atoi(v) != 0; }
// Pascal (sm_61: Tesla P40 / P4, GTX 10 series; native dp4a): the same interleaved kernels (bitwise equal to
// native_mmvq's, mmvq_il_parity), its own table kIlRows_61.  STRATA_SM61_TABLE=0 keeps native_mmvq's kernels.
bool sm61_table_on() { const char* v = std::getenv("STRATA_SM61_TABLE"); return v == nullptr || std::atoi(v) != 0; }
bool il_arch_ok() {
    static int ok[16] = {};   // 0 unknown, 1 yes, -1 no, by device ordinal
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (ok[dev] == 0) {
        int major = 0;
        int minor = 0;
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
        ok[dev] = (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                   (major >= 8 || (major == 7 && minor == 0 && sm70_opt_in()) || (major == 6 && minor == 0 && sm60_table_on()) ||
                    (major == 6 && minor == 1 && sm61_table_on()))) ? 1 : -1;
    }
    return ok[dev] > 0;
}
}  // namespace

bool native_mmvq_il_supported(int ggml_type, int ncols, int n_out) {
    if (ncols < 1 || ncols > 4 || !g_multi_exact || !il_arch_ok()) return false;
    return (g_tune_rows ? g_tune_rows : il_rows(ggml_type, ncols, n_out)) != 0 &&
           (ggml_type == 23 || ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 11);
}

void native_mmvq_il(int ggml_type, const void* weights, const void* x_q8_1, const void* x_il, float* y, int n_in,
                    int n_out, int ncols, void* stream) {
    if (!native_mmvq_il_supported(ggml_type, ncols, n_out)) {
        native_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
        return;
    }
    const int r = g_tune_rows ? g_tune_rows : il_rows(ggml_type, ncols, n_out);
    validate_shape(n_in, ncols, QK);
    validate_pointer(weights);
    if (ncols == 1) x_il = x_q8_1;   // one column: the plain blocks, no interleaved copy
    validate_pointer(x_il);
    validate_pointer(y);
    validate_stream(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
    case 23: launch_il_rows<IlIQ4XS>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 12: launch_il_rows<IlQ4K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 13: launch_il_rows<IlQ5K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 14: launch_il_rows<IlQ6K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    case 11: launch_il_rows<IlQ3K>(r, weights, x_il, y, n_in, n_out, ncols, s); break;
    }
    launch_check();
}

#else  // HIP: no interleaved path
std::size_t native_q8_1_il_bytes(int n_in, int ncols) { return native_q8_1_bytes(n_in, ncols) * 2; }
void native_q8_1_interleave(const void*, void*, int, int, void*) {
    throw std::invalid_argument("native_q8_1_interleave is CUDA only");
}
bool native_mmvq_il_supported(int, int, int) { return false; }
int native_mmvq_il_rows_for(int, int, int, int) { return 0; }
void native_mmvq_il(int ggml_type, const void* weights, const void* x_q8_1, const void*, float* y, int n_in, int n_out,
                    int ncols, void* stream) {
    native_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream);
}
void native_mmvq_il_tune(int) {}
#endif  // !__HIPCC__

// Laguna: mat-vec over experts chosen ON THE DEVICE, llama.cpp's mul_mat_vec_q with ids.  Block (x, y) computes
// ROWS rows of expert ids[y]: its weights at w + ids[y]*expert_bytes, its input column y when x_per_id (else column
// 0, one input shared by every expert), its output at y_out + y*n_out.  The per-(row, block) math is the traits'
// load/apply, the reduction the multi kernel's, so each output equals the one-column call on that expert.
namespace {
template<typename F, int ROWS>
__launch_bounds__(WARPS * WARP, 4)
__global__ void native_mmvq_ids_kernel(const uint8_t* __restrict__ w_base, std::size_t expert_bytes,
                                       const int32_t* __restrict__ ids, const Q81Block* __restrict__ x,
                                       int x_per_id, float* __restrict__ y_out, int n_in, int n_out,
                                       int n_ids, const uint8_t* __restrict__ w_base2, float* __restrict__ y_out2,
                                       int tpr) {
    // PAIR (w_base2 set): blocks y >= n_ids run the same experts of a second matrix into y_out2
    int k = int(blockIdx.y);
    if (w_base2 && k >= n_ids) { k -= n_ids; w_base = w_base2; y_out = y_out2; }
    const int e = ids[k];
    const auto* w = reinterpret_cast<const typename F::Block*>(w_base + std::size_t(e) * expert_bytes);
    const int x_stride = n_in / Q8K;
    const Q81Block* xc = x + (x_per_id ? std::size_t(k) * x_stride : 0);
    float* y = y_out + std::size_t(k) * n_out;
    // tpr threads per row group (64 when a row is too short to keep 128 busy: the down projection's 1024 inputs);
    // the block's 128 threads take 128 / tpr groups of ROWS rows each
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int sub = tid / tpr, ltid = tid % tpr;
    const int groups = (WARPS * WARP) / tpr;
    const int row0 = ROWS * (int(blockIdx.x) * groups + sub);
    const int blocks_per_row = n_in / F::DIV;
    const int bpi = tpr / F::T;
    float tmp[ROWS] = {};
    for (int kbx = ltid / F::T; kbx < blocks_per_row; kbx += bpi) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(ltid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
                tmp[i] += F::apply(wv, xc + kby, kqs);
            }
    }
    __shared__ float partial[WARPS][ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        const float v = warp_sum(tmp[i]);
        if (threadIdx.x == 0) partial[threadIdx.y][i] = v;
    }
    __syncthreads();
    const int wpg = tpr / WARP;   // warps per group
    if (tid < groups * ROWS) {
        const int g = tid / ROWS, i = tid % ROWS, row = ROWS * (int(blockIdx.x) * groups + g) + i;
        float v = 0.0f;
        for (int wv = 0; wv < wpg; ++wv) v += partial[g * wpg + wv][i];
        if (row < n_out) y[row] = v;
    }
}

template<typename F, int ROWS>
void launch_ids_rows(const void* w, std::size_t expert_bytes, const int32_t* ids, int n_ids, const void* x,
                     bool x_per_id, float* y, int n_in, int n_out, cudaStream_t s, const void* w2, float* y2) {
    // threads per row group: the widest of 128/64/32 whose blocks-per-pass divides the row's blocks, so no pass
    // runs part-empty (Q4_K/Q5_K at 3072 inputs: 12 blocks, 64 threads = 4 per pass; at 1024: 4 blocks, 64)
    const int bpr = n_in / F::DIV;
    int tpr = WARP;
    for (int t : {WARPS * WARP, 2 * WARP}) {
        if (t / F::T >= 1 && bpr % (t / F::T) == 0) { tpr = t; break; }
    }
    const int rows_per_block = ROWS * (WARPS * WARP / tpr);
    const dim3 grid(unsigned((n_out + rows_per_block - 1) / rows_per_block), unsigned(w2 ? 2 * n_ids : n_ids));
    native_mmvq_ids_kernel<F, ROWS><<<grid, dim3(WARP, WARPS), 0, s>>>(
        static_cast<const uint8_t*>(w), expert_bytes, ids, static_cast<const Q81Block*>(x), x_per_id ? 1 : 0, y,
        n_in, n_out, n_ids, static_cast<const uint8_t*>(w2), y2, tpr);
}

int ids_rows() {
    static const int r = [] {
        const char* e = std::getenv("STRATA_IDS_ROWS");
        const int v = e ? std::atoi(e) : 2;
        return (v == 1 || v == 4) ? v : 2;
    }();
    return r;
}

template<typename F>
void launch_ids(const void* w, std::size_t expert_bytes, const int32_t* ids, int n_ids, const void* x, bool x_per_id,
                float* y, int n_in, int n_out, cudaStream_t s, const void* w2 = nullptr, float* y2 = nullptr) {
    switch (ids_rows()) {
    case 1: launch_ids_rows<F, 1>(w, expert_bytes, ids, n_ids, x, x_per_id, y, n_in, n_out, s, w2, y2); break;
    case 4: launch_ids_rows<F, 4>(w, expert_bytes, ids, n_ids, x, x_per_id, y, n_in, n_out, s, w2, y2); break;
    default: launch_ids_rows<F, 2>(w, expert_bytes, ids, n_ids, x, x_per_id, y, n_in, n_out, s, w2, y2); break;
    }
}

template<typename F, int ROWS, int NCOLS>
__launch_bounds__(WARPS * WARP, 4)
__global__ void native_mmvq_ids_cols_kernel(const uint8_t* __restrict__ w_base, const Q81Block* __restrict__ x,
                                            float* __restrict__ y_out, int n_in, int n_out,
                                            const uint8_t* __restrict__ w_base2, float* __restrict__ y_out2,
                                            int tpr) {
    if (w_base2 && blockIdx.y > 0) { w_base = w_base2; y_out = y_out2; }
    const auto* w = reinterpret_cast<const typename F::Block*>(w_base);
    const int x_stride = n_in / Q8K;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int sub = tid / tpr, ltid = tid % tpr;
    const int groups = (WARPS * WARP) / tpr;
    const int row0 = ROWS * (int(blockIdx.x) * groups + sub);
    const int blocks_per_row = n_in / F::DIV;
    const int bpi = tpr / F::T;
    float tmp[NCOLS][ROWS] = {};
    for (int kbx = ltid / F::T; kbx < blocks_per_row; kbx += bpi) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(ltid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int c = 0; c < NCOLS; ++c)
                    tmp[c][i] += F::apply(wv, x + std::size_t(c) * x_stride + kby, kqs);
            }
    }
    __shared__ float partial[WARPS][NCOLS][ROWS];
#pragma unroll
    for (int c = 0; c < NCOLS; ++c)
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            const float v = warp_sum(tmp[c][i]);
            if (threadIdx.x == 0) partial[threadIdx.y][c][i] = v;
        }
    __syncthreads();
    const int wpg = tpr / WARP;
    for (int idx = tid; idx < groups * NCOLS * ROWS; idx += WARPS * WARP) {
        const int g = idx / (NCOLS * ROWS), rem = idx % (NCOLS * ROWS);
        const int c = rem / ROWS, i = rem % ROWS;
        const int row = ROWS * (int(blockIdx.x) * groups + g) + i;
        float v = 0.0f;
        for (int wv = 0; wv < wpg; ++wv) v += partial[g * wpg + wv][c][i];
        if (row < n_out) y_out[std::size_t(c) * n_out + row] = v;
    }
}

template<typename F, int ROWS, int NCOLS>
void launch_ids_cols_n(const void* w, const void* x, float* y, int n_in, int n_out, cudaStream_t s,
                       const void* w2, float* y2) {
    const int bpr = n_in / F::DIV;
    int tpr = WARP;
    for (int t : {WARPS * WARP, 2 * WARP}) {
        if (t / F::T >= 1 && bpr % (t / F::T) == 0) { tpr = t; break; }
    }
    const int rows_per_block = ROWS * (WARPS * WARP / tpr);
    const dim3 grid(unsigned((n_out + rows_per_block - 1) / rows_per_block), unsigned(w2 ? 2 : 1));
    native_mmvq_ids_cols_kernel<F, ROWS, NCOLS><<<grid, dim3(WARP, WARPS), 0, s>>>(
        static_cast<const uint8_t*>(w), static_cast<const Q81Block*>(x), y, n_in, n_out,
        static_cast<const uint8_t*>(w2), y2, tpr);
}

template<typename F, int ROWS>
void launch_ids_cols_rows(const void* w, int ncols, const void* x, float* y, int n_in, int n_out, cudaStream_t s,
                          const void* w2, float* y2) {
    switch (ncols) {
    case 1: launch_ids_cols_n<F, ROWS, 1>(w, x, y, n_in, n_out, s, w2, y2); break;
    case 2: launch_ids_cols_n<F, ROWS, 2>(w, x, y, n_in, n_out, s, w2, y2); break;
    case 3: launch_ids_cols_n<F, ROWS, 3>(w, x, y, n_in, n_out, s, w2, y2); break;
    case 4: launch_ids_cols_n<F, ROWS, 4>(w, x, y, n_in, n_out, s, w2, y2); break;
    case 5: launch_ids_cols_n<F, ROWS, 5>(w, x, y, n_in, n_out, s, w2, y2); break;
    case 6: launch_ids_cols_n<F, ROWS, 6>(w, x, y, n_in, n_out, s, w2, y2); break;
    case 7: launch_ids_cols_n<F, ROWS, 7>(w, x, y, n_in, n_out, s, w2, y2); break;
    default: launch_ids_cols_n<F, ROWS, 8>(w, x, y, n_in, n_out, s, w2, y2); break;
    }
}

template<typename F>
void launch_ids_cols(const void* w, int ncols, const void* x, float* y, int n_in, int n_out, cudaStream_t s,
                     const void* w2 = nullptr, float* y2 = nullptr) {
    switch (ids_rows()) {
    case 1: launch_ids_cols_rows<F, 1>(w, ncols, x, y, n_in, n_out, s, w2, y2); break;
    case 4: launch_ids_cols_rows<F, 4>(w, ncols, x, y, n_in, n_out, s, w2, y2); break;
    default: launch_ids_cols_rows<F, 2>(w, ncols, x, y, n_in, n_out, s, w2, y2); break;
    }
}
}  // namespace

void native_mmvq_ids_cols(int ggml_type, const void* weights, int ncols, const void* x_q8_1, float* y, int n_in,
                          int n_out, void* stream, const void* weights2, float* y2) {
    if (ncols <= 0 || ncols > 8 || n_out <= 0)
        throw std::invalid_argument("native_mmvq_ids_cols: ncols in 1..8 and n_out > 0 required");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (weights2) validate_pointer(y2);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
    case 12: validate_shape(n_in, 1, 256); launch_ids_cols<Q4KTraits>(weights, ncols, x_q8_1, y, n_in, n_out, s, weights2, y2); break;
    case 13: validate_shape(n_in, 1, 256); launch_ids_cols<Q5KTraits>(weights, ncols, x_q8_1, y, n_in, n_out, s, weights2, y2); break;
    case 14: validate_shape(n_in, 1, 256); launch_ids_cols<Q6KTraits>(weights, ncols, x_q8_1, y, n_in, n_out, s, weights2, y2); break;
    case 8:  validate_shape(n_in, 1, 32);  launch_ids_cols<SmallTraits<Q80Block, 8>>(weights, ncols, x_q8_1, y, n_in, n_out, s, weights2, y2); break;
    default: throw std::invalid_argument("native_mmvq_ids_cols: ggml type " + std::to_string(ggml_type) + " not covered");
    }
    launch_check();
}

void native_mmvq_ids(int ggml_type, const void* weights, std::size_t expert_bytes, const int32_t* ids, int n_ids,
                     const void* x_q8_1, bool x_per_id, float* y, int n_in, int n_out, void* stream,
                     const void* weights2, float* y2) {
    if (n_ids <= 0 || n_out <= 0) throw std::invalid_argument("native_mmvq_ids: n_ids and n_out must be positive");
    validate_pointer(weights);
    validate_pointer(ids);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    if (weights2) validate_pointer(y2);
    const auto s = static_cast<cudaStream_t>(stream);
    switch (ggml_type) {
    case 12: validate_shape(n_in, 1, 256); launch_ids<Q4KTraits>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, s, weights2, y2); break;
    case 13: validate_shape(n_in, 1, 256); launch_ids<Q5KTraits>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, s, weights2, y2); break;
    case 14: validate_shape(n_in, 1, 256); launch_ids<Q6KTraits>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, s, weights2, y2); break;
    case 8:  validate_shape(n_in, 1, 32);  launch_ids<SmallTraits<Q80Block, 8>>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, s, weights2, y2); break;
    default: throw std::invalid_argument("native_mmvq_ids: ggml type " + std::to_string(ggml_type) + " not covered");
    }
    launch_check();
}

// Laguna: several mat-vecs of different weight types in ONE launch (decode: q Q5_K + k/v Q6_K + gate Q8_0
// on one input; the shared Q6_K and routed Q4_K gate/up; both down projections).  Fewer launches (~4 us of dispatch
// gap each on this GPU) and the small matrices fill the big ones' tail.  Each block finds its item, then runs that
// item's rows exactly as native_mmvq_ids does.
namespace {
template<typename F, int ROWS>
__device__ __forceinline__ void batch_body(const MmvqItem& it, int k, int rblk, int tpr, float (*partial)[4]) {
    const int e = it.ids ? it.ids[k] : 0;
    const auto* w = reinterpret_cast<const typename F::Block*>(static_cast<const uint8_t*>(it.w) +
                                                               std::size_t(e) * it.expert_bytes);
    const int n_in = it.n_in, n_out = it.n_out;
    const Q81Block* xc = static_cast<const Q81Block*>(it.x) +
                         (it.x_per_id ? std::size_t(k / (it.x_group > 0 ? it.x_group : 1)) * (n_in / Q8K) : 0);
    float* y = it.y + std::size_t(k) * n_out;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int sub = tid / tpr, ltid = tid % tpr;
    const int groups = (WARPS * WARP) / tpr;
    const int row0 = ROWS * (rblk * groups + sub);
    const int blocks_per_row = n_in / F::DIV;
    const int bpi = tpr / F::T;
    float tmp[ROWS] = {};
    for (int kbx = ltid / F::T; kbx < blocks_per_row; kbx += bpi) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(ltid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
                tmp[i] += F::apply(wv, xc + kby, kqs);
            }
    }
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
        const float v = warp_sum(tmp[i]);
        if (threadIdx.x == 0) partial[threadIdx.y][i] = v;
    }
    __syncthreads();
    const int wpg = tpr / WARP;
    if (tid < groups * ROWS) {
        const int g = tid / ROWS, i = tid % ROWS, row = ROWS * (rblk * groups + g) + i;
        float v = 0.0f;
        for (int wv = 0; wv < wpg; ++wv) v += partial[g * wpg + wv][i];
        if (row < n_out) y[row] = v;
    }
}

// Talus: the i-quants and Q2_K (MiMo-V2.6's experts) as batch traits over Strata's Split<TY> load / apply
template<int TY> struct IqBlockOf;
template<> struct IqBlockOf<16> { using T = block_iq2_xxs; };
template<> struct IqBlockOf<17> { using T = block_iq2_xs; };
template<> struct IqBlockOf<18> { using T = block_iq3_xxs; };
template<> struct IqBlockOf<21> { using T = block_iq3_s; };
template<> struct IqBlockOf<29> { using T = block_iq1_m; };
template<> struct IqBlockOf<10> { using T = block_q2_K; };
template<int TY> struct IqTraits {
    using Block = typename IqBlockOf<TY>::T;
    using S = talus_iq::IqShape<TY>;
    static constexpr int DIV = S::qk, T = S::ipb, KBY = S::qk / 32, BPI = WARPS * WARP / S::ipb;
    __device__ static int kqs(int tid) { return S::step * (tid % S::ipb); }
    using W = typename talus_iq::Split<TY>::W;
    static constexpr int kTpr = 32;   // one warp per row group: an i-quant row is ~1 KB, a 128-thread group idles
    __device__ static W load(const Block* __restrict__ b, int iqs) { return talus_iq::Split<TY>::load(b, 0, iqs); }
    __device__ static float apply(const W& r, const Q81Block* __restrict__ x, int iqs) {
        return talus_iq::Split<TY>::apply(r, reinterpret_cast<const block_q8_1*>(x), iqs);
    }
};
#define TALUS_IQ_CASES(BODY, ROWS, ...) \
    case 16: BODY<IqTraits<16>, ROWS>(__VA_ARGS__); break; \
    case 17: BODY<IqTraits<17>, ROWS>(__VA_ARGS__); break; \
    case 18: BODY<IqTraits<18>, ROWS>(__VA_ARGS__); break; \
    case 21: BODY<IqTraits<21>, ROWS>(__VA_ARGS__); break; \
    case 29: BODY<IqTraits<29>, ROWS>(__VA_ARGS__); break; \
    case 10: BODY<IqTraits<10>, ROWS>(__VA_ARGS__); break;

// dense item over n_ids <= 8 columns (verify): each weight block loaded once and applied to every column, instead of
// one block per column relying on the cache to share the rows across the verify window
template<typename F, int ROWS>
__device__ __forceinline__ void batch_body_cols(const MmvqItem& it, int rblk, int tpr, float (*partial)[4]) {
    constexpr int NC = 8;
    const int nc = it.n_ids;
    const auto* w = static_cast<const typename F::Block*>(it.w);
    const int n_in = it.n_in, n_out = it.n_out;
    const Q81Block* x = static_cast<const Q81Block*>(it.x);
    const int xs = it.x_per_id ? n_in / Q8K : 0;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int sub = tid / tpr, ltid = tid % tpr;
    const int groups = (WARPS * WARP) / tpr;
    const int row0 = ROWS * (rblk * groups + sub);
    const int blocks_per_row = n_in / F::DIV;
    const int bpi = tpr / F::T;
    float tmp[NC][ROWS] = {};
    for (int kbx = ltid / F::T; kbx < blocks_per_row; kbx += bpi) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(ltid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    if (c < nc) tmp[c][i] += F::apply(wv, x + std::size_t(c) * xs + kby, kqs);
            }
    }
    const int wpg = tpr / WARP;
    for (int c = 0; c < nc; ++c) {   // nc is block-uniform: the barriers are safe
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            const float v = warp_sum(tmp[c][i]);
            if (threadIdx.x == 0) partial[threadIdx.y][i] = v;
        }
        __syncthreads();
        if (tid < groups * ROWS) {
            const int g = tid / ROWS, i = tid % ROWS, row = ROWS * (rblk * groups + g) + i;
            float v = 0.0f;
            for (int wv = 0; wv < wpg; ++wv) v += partial[g * wpg + wv][i];
            if (row < n_out) it.y[std::size_t(c) * n_out + row] = v;
        }
        __syncthreads();
    }
}

constexpr int kBatchMax = 8;
struct BatchArgs {
    MmvqItem it[kBatchMax];
    int start[kBatchMax + 1];   // first block of each item
    int tpr[kBatchMax];
    int rows[kBatchMax];        // rows per thread group: 4 for short rows (<= 4 blocks), else 2
    int rblocks[kBatchMax];     // row blocks per expert
    int cols[kBatchMax];        // dense over 2..8 columns: batch_body_cols, rblocks blocks in all
    int n;
};

__launch_bounds__(WARPS * WARP, 1)
__global__ void native_mmvq_batch_kernel(const BatchArgs a) {
    __shared__ float partial[WARPS][4];
    const int b = int(blockIdx.x);
    int i = 0;
    while (i + 1 < a.n && b >= a.start[i + 1]) ++i;
    // row-block major: the n_ids blocks of one row block are neighbours, so a dense matrix run over several tokens
    // (ids all 0) reads each weight row once from DRAM and the other tokens' blocks hit the cache
    const int local = b - a.start[i];
    const int n_ids = a.it[i].n_ids;
    const bool emaj = a.it[i].expert_major != 0;
    const int rblk = emaj ? local % a.rblocks[i] : local / n_ids, k = emaj ? local / a.rblocks[i] : local % n_ids;
    const MmvqItem& it = a.it[i];
    if (a.cols[i]) {   // dense over several columns: one block per row block
        switch (it.type) {
        case 12: batch_body_cols<Q4KTraits, 2>(it, local, a.tpr[i], partial); break;
        TALUS_IQ_CASES(batch_body_cols, 2, it, local, a.tpr[i], partial)
        case 13: batch_body_cols<Q5KTraits, 2>(it, local, a.tpr[i], partial); break;
        case 14: batch_body_cols<Q6KTraits, 2>(it, local, a.tpr[i], partial); break;
        default: batch_body_cols<SmallTraits<Q80Block, 8>, 2>(it, local, a.tpr[i], partial); break;
        }
        return;
    }
    if (a.rows[i] == 4) {
        switch (it.type) {
        case 12: batch_body<Q4KTraits, 4>(it, k, rblk, a.tpr[i], partial); break;
        TALUS_IQ_CASES(batch_body, 4, it, k, rblk, a.tpr[i], partial)
        case 13: batch_body<Q5KTraits, 4>(it, k, rblk, a.tpr[i], partial); break;
        case 14: batch_body<Q6KTraits, 4>(it, k, rblk, a.tpr[i], partial); break;
        default: batch_body<SmallTraits<Q80Block, 8>, 4>(it, k, rblk, a.tpr[i], partial); break;
        }
        return;
    }
    switch (it.type) {
    case 12: batch_body<Q4KTraits, 2>(it, k, rblk, a.tpr[i], partial); break;
    TALUS_IQ_CASES(batch_body, 2, it, k, rblk, a.tpr[i], partial)
    case 13: batch_body<Q5KTraits, 2>(it, k, rblk, a.tpr[i], partial); break;
    case 14: batch_body<Q6KTraits, 2>(it, k, rblk, a.tpr[i], partial); break;
    default: batch_body<SmallTraits<Q80Block, 8>, 2>(it, k, rblk, a.tpr[i], partial); break;
    }
}

template<typename F>
int batch_tpr(int n_in) {
    if constexpr (requires { F::kTpr; }) return F::kTpr;
    const int bpr = n_in / F::DIV;
    for (int t : {WARPS * WARP, 2 * WARP})
        if (t / F::T >= 1 && bpr % (t / F::T) == 0) return t;
    return WARP;
}

// ---- Laguna wide verify: dense items flagged expert_major == 2 over more than one column, in their own kernel
// (the main batch kernel and its register budget stay as they were) ----
// One block per row block takes every column: each weight block is loaded once per pass of NC (8 / 16) columns
// and applied to all of them.  A column's sum takes batch_body's operations in batch_body's order (same tpr, kbx walk,
// warp_sum, partial order), whatever NC: bitwise its single-column (n = 1) result.
template<typename F, int ROWS, int NC>
__device__ __forceinline__ void wide_cols_pass(const MmvqItem& it, int rblk, int tpr, float* partial, int c0,
                                               int nc) {
    const auto* w = static_cast<const typename F::Block*>(it.w);
    const int n_in = it.n_in, n_out = it.n_out;
    const int xs = n_in / Q8K;
    const Q81Block* x = static_cast<const Q81Block*>(it.x) + std::size_t(c0) * xs;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int sub = tid / tpr, ltid = tid % tpr;
    const int groups = (WARPS * WARP) / tpr;
    const int row0 = ROWS * (rblk * groups + sub);
    const int blocks_per_row = n_in / F::DIV;
    const int bpi = tpr / F::T;
    float tmp[NC][ROWS] = {};
    for (int kbx = ltid / F::T; kbx < blocks_per_row; kbx += bpi) {
        const int kby = kbx * F::KBY;
        const int kqs = F::kqs(ltid);
#pragma unroll
        for (int i = 0; i < ROWS; ++i)
            if (row0 + i < n_out) {
                const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    if (c < nc) tmp[c][i] += F::apply(wv, x + std::size_t(c) * xs + kby, kqs);
            }
    }
    // every (column, row) warp sum first, one barrier, then the partials of the row's warps in batch_body's order
    const int wpg = tpr / WARP;
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            const float v = warp_sum(tmp[c][i]);
            if (threadIdx.x == 0 && c < nc) partial[(threadIdx.y * NC + c) * ROWS + i] = v;
        }
    __syncthreads();
    for (int idx = tid; idx < groups * nc * ROWS; idx += WARPS * WARP) {
        const int g = idx / (nc * ROWS), c = (idx / ROWS) % nc, i = idx % ROWS;
        const int row = ROWS * (rblk * groups + g) + i;
        float v = 0.0f;
        for (int wv = 0; wv < wpg; ++wv) v += partial[((g * wpg + wv) * NC + c) * ROWS + i];
        if (row < n_out) it.y[std::size_t(c0 + c) * n_out + row] = v;
    }
    __syncthreads();
}

template<typename F, int ROWS>
__device__ __forceinline__ void wide_body(const MmvqItem& it, int rblk, int tpr, float* partial) {
    // passes of 16 columns (keeps VGPR pressure bounded)
    for (int c0 = 0; c0 < it.n_ids; c0 += 16) {
        const int nc = min(it.n_ids - c0, 16);
        if (nc > 8) wide_cols_pass<F, ROWS, 16>(it, rblk, tpr, partial, c0, nc);
        else wide_cols_pass<F, ROWS, 8>(it, rblk, tpr, partial, c0, nc);
    }
}

__launch_bounds__(WARPS * WARP, 1)
__global__ void native_mmvq_wide_kernel(const BatchArgs a) {
    __shared__ float partial[WARPS * 16 * 4];   // [warp][column][row] of one pass
    const int b = int(blockIdx.x);
    int i = 0;
    while (i + 1 < a.n && b >= a.start[i + 1]) ++i;
    const int rblk = b - a.start[i];
    const MmvqItem& it = a.it[i];
    if (a.rows[i] == 4) {
        switch (it.type) {
        case 12: wide_body<Q4KTraits, 4>(it, rblk, a.tpr[i], partial); break;
        TALUS_IQ_CASES(wide_body, 4, it, rblk, a.tpr[i], partial)
        case 13: wide_body<Q5KTraits, 4>(it, rblk, a.tpr[i], partial); break;
        case 14: wide_body<Q6KTraits, 4>(it, rblk, a.tpr[i], partial); break;
        default: wide_body<SmallTraits<Q80Block, 8>, 4>(it, rblk, a.tpr[i], partial); break;
        }
        return;
    }
    switch (it.type) {
    case 12: wide_body<Q4KTraits, 2>(it, rblk, a.tpr[i], partial); break;
    TALUS_IQ_CASES(wide_body, 2, it, rblk, a.tpr[i], partial)
    case 13: wide_body<Q5KTraits, 2>(it, rblk, a.tpr[i], partial); break;
    case 14: wide_body<Q6KTraits, 2>(it, rblk, a.tpr[i], partial); break;
    default: wide_body<SmallTraits<Q80Block, 8>, 2>(it, rblk, a.tpr[i], partial); break;
    }
}

// ---- (Laguna wide verify) routed-expert items flagged expert_major == 3, grouped by expert ----
// The row-block-major batch kernel runs one block per (row block, id): a verify window of n tokens x 10 routed
// experts reads a popular expert's rows once per token that picked it (from the cache when the neighbours are in
// flight) and launches n x 10 x row-blocks tiny blocks.  Here block (u, chunk) takes slot u of the id list; it is
// active only when u is the FIRST id equal to e = ids[u], gathers every id k with ids[k] == e (the columns) and runs
// rpb row blocks of expert e over all of them, each weight block loaded once per pass of NC columns.  A column's sum is
// batch_body's operations in batch_body's order (same tpr, kbx walk, warp_sum, partial order): bitwise the
// row-block-major kernel's result, whatever the grouping, NC, ROWS or rpb.
constexpr int kGroupMaxIds = 1024;
struct GroupArgs {
    MmvqItem it[kBatchMax];
    int start[kBatchMax + 1];   // first block of each item
    int tpr[kBatchMax];
    int rows[kBatchMax];
    int rblocks[kBatchMax];     // row blocks per expert
    int rpb[kBatchMax];         // row blocks per block
    int chunks[kBatchMax];      // blocks per id slot: ceil(rblocks / rpb)
    int nexp[kBatchMax];        // experts in the pool (expert_major >> 8), 0 = unknown
    int n;
};

// one column pass over the block's whole row range with ONE barrier: every row block's (column, row) warp sums
// go to LDS first, then the cross-warp partials are added in batch_body's order (v = 0, then the row's warps in order).
// A column's sum is still batch_body's operations in batch_body's order (same tpr, kbx walk, apply, warp_sum).
constexpr int kGroupRpbMax = 32;
template<typename F, int NC>
__device__ __forceinline__ void group_rows_pass(const MmvqItem& it, const typename F::Block* __restrict__ w, int rb0,
                                                int rb1, int tpr, float* partial, const int* cols, int c0, int nc) {
    constexpr int ROWS = 2;
    const int n_in = it.n_in, n_out = it.n_out;
    const int xs = n_in / Q8K;
    const int xg = it.x_group > 0 ? it.x_group : 1;
    const Q81Block* xb = static_cast<const Q81Block*>(it.x);
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int sub = tid / tpr, ltid = tid % tpr;
    const int groups = (WARPS * WARP) / tpr;
    const int blocks_per_row = n_in / F::DIV;
    const int bpi = tpr / F::T;
    const int wpg = tpr / WARP;
    int xo[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) xo[c] = it.x_per_id ? (cols[c0 + min(c, nc - 1)] / xg) * xs : 0;
    const int nrb = rb1 - rb0;
    for (int rb = rb0; rb < rb1; ++rb) {
        const int row0 = ROWS * (rb * groups + sub);
        float tmp[NC][ROWS] = {};
        for (int kbx = ltid / F::T; kbx < blocks_per_row; kbx += bpi) {
            const int kby = kbx * F::KBY;
            const int kqs = F::kqs(ltid);
#pragma unroll
            for (int i = 0; i < ROWS; ++i)
                if (row0 + i < n_out) {
                    const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        if (c < nc) tmp[c][i] += F::apply(wv, xb + xo[c] + kby, kqs);
                }
        }
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (c < nc)
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    const float v = warp_sum(tmp[c][i]);
                    if (threadIdx.x == 0) partial[(((rb - rb0) * WARPS + threadIdx.y) * NC + c) * ROWS + i] = v;
                }
    }
    __syncthreads();
    const int per_rb = groups * nc * ROWS;
    for (int idx = tid; idx < nrb * per_rb; idx += WARPS * WARP) {
        const int rbl = idx / per_rb, r = idx % per_rb;
        const int g = r / (nc * ROWS), c = (r / ROWS) % nc, i = r % ROWS;
        const int row = ROWS * ((rb0 + rbl) * groups + g) + i;
        float v = 0.0f;
        for (int wv = 0; wv < wpg; ++wv) v += partial[((rbl * WARPS + g * wpg + wv) * NC + c) * ROWS + i];
        if (row < n_out) it.y[std::size_t(cols[c0 + c]) * n_out + row] = v;
    }
    __syncthreads();
}

// block (slot u, chunk) takes the u-th DISTINCT expert of the id list (a presence bitmap of the ids, then the
// u-th set bit), so the grid needs only min(n_ids, n_experts) slots per chunk instead of n_ids (n 64: 256 slots, ~8%
// empty, against 640 slots, 63% empty). n_experts = 0 (unknown): the slot is the first-occurrence test of ids[u].
template<typename F, int NC>
__launch_bounds__(WARPS * WARP, 1)
__global__ void native_mmvq_group_kernel(const GroupArgs a) {
    __shared__ float partial[kGroupRpbMax * WARPS * NC * 2];
    __shared__ int cols[kGroupMaxIds];
    __shared__ unsigned bm[32];   // expert presence, ids < 1024
    __shared__ int s_m, s_e;
    const int b = int(blockIdx.x);
    int i = 0;
    while (i + 1 < a.n && b >= a.start[i + 1]) ++i;
    const MmvqItem& it = a.it[i];
    const int local = b - a.start[i];
    const int u = local / a.chunks[i], chunk = local % a.chunks[i];
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    if (a.nexp[i] > 0) {
        if (tid < 32) bm[tid] = 0u;
        if (tid == 0) s_m = 0;
        __syncthreads();
        for (int k = tid; k < it.n_ids; k += WARPS * WARP) {
            const int id = it.ids[k];
            atomicOr(&bm[id >> 5], 1u << (id & 31));
        }
        __syncthreads();
        if (tid == 0) {
            int left = u, e = -1;
            for (int wd = 0; wd < 32 && e < 0; ++wd) {
                const unsigned word = bm[wd];
                const int pc = __popc(word);
                if (left < pc) {
                    unsigned v = word;
                    for (int j = 0; j < left; ++j) v &= v - 1;   // drop the lowest `left` set bits
                    e = wd * 32 + __ffs(v) - 1;
                } else left -= pc;
            }
            s_e = e;
        }
        __syncthreads();
    } else {
        if (tid == 0) { s_m = 0; s_e = it.ids[u]; }
        __syncthreads();
        for (int k = tid; k < u; k += WARPS * WARP)
            if (it.ids[k] == s_e) s_e = -1;   // an earlier slot owns this expert (benign race: all writers write -1)
        __syncthreads();
    }
    const int e = s_e;
    if (e < 0) return;   // block-uniform
    for (int k = tid; k < it.n_ids; k += WARPS * WARP)
        if (it.ids[k] == e) cols[atomicAdd(&s_m, 1)] = k;
    __syncthreads();
    const int m = s_m;
    const int rb0 = chunk * a.rpb[i], rb1 = min(rb0 + a.rpb[i], a.rblocks[i]);
    const auto* w = reinterpret_cast<const typename F::Block*>(static_cast<const uint8_t*>(it.w) +
                                                               std::size_t(e) * it.expert_bytes);
    for (int c0 = 0; c0 < m; c0 += NC) group_rows_pass<F, NC>(it, w, rb0, rb1, a.tpr[i], partial, cols, c0, min(m - c0, NC));
}
}  // namespace

void native_mmvq_batch(const MmvqItem* items, int n_items, void* stream) {
    if (n_items < 1 || n_items > kBatchMax) throw std::invalid_argument("native_mmvq_batch: 1..8 items");
    validate_stream(stream);
    // Dense expert_major == 2 items over several columns (Laguna verify / DFlash) go to native_mmvq_wide_kernel
    {
        int n_wide = 0;
        auto is_wide = [](const MmvqItem& it) {
            return it.expert_major == 2 && it.n_ids > 1 && (!it.ids || it.expert_bytes == 0);
        };
        for (int i = 0; i < n_items; ++i) n_wide += is_wide(items[i]);
        if (n_wide > 0) {
            BatchArgs w{};
            MmvqItem rest[kBatchMax];
            int n_rest = 0, blocks = 0;
            for (int i = 0; i < n_items; ++i) {
                const MmvqItem& it = items[i];
                if (!is_wide(it)) { rest[n_rest++] = it; continue; }
                validate_pointer(it.w);
                validate_pointer(it.x);
                validate_pointer(it.y);
                if (!it.x_per_id) throw std::invalid_argument("native_mmvq_batch: wide item needs x_per_id");
                int tpr = 0;
                switch (it.type) {
                case 12: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q4KTraits>(it.n_in); break;
                case 13: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q5KTraits>(it.n_in); break;
                case 14: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q6KTraits>(it.n_in); break;
                case 8:  validate_shape(it.n_in, 1, 32);  tpr = batch_tpr<SmallTraits<Q80Block, 8>>(it.n_in); break;
                case 16: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<16>>(it.n_in); break;
                case 17: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<17>>(it.n_in); break;
                case 18: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<18>>(it.n_in); break;
                case 21: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<21>>(it.n_in); break;
                case 29: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<29>>(it.n_in); break;
                case 10: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<10>>(it.n_in); break;
                default: throw std::invalid_argument("native_mmvq_batch: ggml type " + std::to_string(it.type) + " not covered");
                }
                // rows per thread group as the main kernel would use for this item (any value is bitwise the same)
                const int div = it.type == 8 ? 32 : 256;
                const int rows = it.n_in / div <= 4 ? 4 : 2;
                const int rows_per_block = rows * (WARPS * WARP / tpr);
                w.it[w.n] = it;
                w.tpr[w.n] = tpr;
                w.rows[w.n] = rows;
                w.rblocks[w.n] = (it.n_out + rows_per_block - 1) / rows_per_block;
                w.start[w.n] = blocks;
                blocks += w.rblocks[w.n];
                ++w.n;
            }
            w.start[w.n] = blocks;
            native_mmvq_wide_kernel<<<unsigned(blocks), dim3(WARP, WARPS), 0, static_cast<cudaStream_t>(stream)>>>(w);
            launch_check();
            if (n_rest > 0) native_mmvq_batch(rest, n_rest, stream);
            return;
        }
    }
    // routed-expert items flagged expert_major == 3 (Laguna verify) go to native_mmvq_group_kernel
    {
        auto is_group = [](const MmvqItem& it) {
            return (it.expert_major & 0xff) == 3 && it.ids && it.expert_bytes > 0 && it.n_ids > 1 && it.n_ids <= kGroupMaxIds &&
                   (it.type == 12 || it.type == 13 || it.type == 14 || it.type == 8);
        };
        int n_group = 0;
        for (int i = 0; i < n_items; ++i) n_group += is_group(items[i]);
        if (n_group > 0) {
            // STRATA_EGROUP_RPB: row blocks per block (default 16); STRATA_EGROUP_NC: columns per pass (2 / 4 / 8, default 4)
            static const int rpb_env = [] { const char* e = std::getenv("STRATA_EGROUP_RPB"); return e ? std::atoi(e) : 0; }();
            static const int nc_env = [] { const char* e = std::getenv("STRATA_EGROUP_NC"); return e ? std::atoi(e) : 4; }();
            MmvqItem rest[kBatchMax];
            int n_rest = 0;
            bool done[kBatchMax] = {};
            for (int i0 = 0; i0 < n_items; ++i0) {
                if (done[i0]) continue;
                if (!is_group(items[i0])) { rest[n_rest++] = items[i0]; done[i0] = true; continue; }
                const int type = items[i0].type;
                GroupArgs g{};
                int blocks = 0;
                for (int i = i0; i < n_items; ++i) {
                    const MmvqItem& it = items[i];
                    if (done[i] || !is_group(it) || it.type != type) continue;
                    done[i] = true;
                    validate_pointer(it.w);
                    validate_pointer(it.x);
                    validate_pointer(it.y);
                    validate_pointer(it.ids);
                    int tpr = 0;
                    switch (type) {
                    case 12: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q4KTraits>(it.n_in); break;
                    case 13: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q5KTraits>(it.n_in); break;
                    case 14: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q6KTraits>(it.n_in); break;
                    default: validate_shape(it.n_in, 1, 32);  tpr = batch_tpr<SmallTraits<Q80Block, 8>>(it.n_in); break;
                    }
                    const int rows_per_rb = 2 * (WARPS * WARP / tpr);   // any rows per group is bitwise the same
                    const int rblocks = (it.n_out + rows_per_rb - 1) / rows_per_rb;
                    const int rpb = std::max(1, std::min({rblocks, kGroupRpbMax, rpb_env > 0 ? rpb_env : 16}));
                    // expert_major = 3 | (n_experts << 8): one slot per distinct expert at most
                    const int nexp = std::min(it.expert_major >> 8, 1024);
                    const int slots = nexp > 0 ? std::min(it.n_ids, nexp) : it.n_ids;
                    g.it[g.n] = it;
                    g.tpr[g.n] = tpr;
                    g.rows[g.n] = 2;
                    g.rblocks[g.n] = rblocks;
                    g.rpb[g.n] = rpb;
                    g.nexp[g.n] = nexp;
                    g.chunks[g.n] = (rblocks + rpb - 1) / rpb;
                    g.start[g.n] = blocks;
                    blocks += g.chunks[g.n] * slots;
                    ++g.n;
                }
                g.start[g.n] = blocks;
                const auto st = static_cast<cudaStream_t>(stream);
#define STRATA_GROUP_LAUNCH(F) \
                if (nc_env == 2) native_mmvq_group_kernel<F, 2><<<unsigned(blocks), dim3(WARP, WARPS), 0, st>>>(g); \
                else if (nc_env == 8) native_mmvq_group_kernel<F, 8><<<unsigned(blocks), dim3(WARP, WARPS), 0, st>>>(g); \
                else native_mmvq_group_kernel<F, 4><<<unsigned(blocks), dim3(WARP, WARPS), 0, st>>>(g);
                switch (type) {
                case 12: STRATA_GROUP_LAUNCH(Q4KTraits) break;
                case 13: STRATA_GROUP_LAUNCH(Q5KTraits) break;
                case 14: STRATA_GROUP_LAUNCH(Q6KTraits) break;
                default: { using Q80T = SmallTraits<Q80Block, 8>; STRATA_GROUP_LAUNCH(Q80T) } break;
                }
#undef STRATA_GROUP_LAUNCH
                launch_check();
            }
            if (n_rest > 0) native_mmvq_batch(rest, n_rest, stream);
            return;
        }
    }
    BatchArgs a{};
    a.n = n_items;
    int blocks = 0;
    for (int i = 0; i < n_items; ++i) {
        const MmvqItem& it = items[i];
        if (it.n_ids <= 0 || it.n_out <= 0) throw std::invalid_argument("native_mmvq_batch: empty item");
        validate_pointer(it.w);
        validate_pointer(it.x);
        validate_pointer(it.y);
        int tpr = 0;
        switch (it.type) {
        case 12: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q4KTraits>(it.n_in); break;
        case 13: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q5KTraits>(it.n_in); break;
        case 14: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<Q6KTraits>(it.n_in); break;
        case 8:  validate_shape(it.n_in, 1, 32);  tpr = batch_tpr<SmallTraits<Q80Block, 8>>(it.n_in); break;
        case 16: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<16>>(it.n_in); break;
        case 17: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<17>>(it.n_in); break;
        case 18: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<18>>(it.n_in); break;
        case 21: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<21>>(it.n_in); break;
        case 29: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<29>>(it.n_in); break;
        case 10: validate_shape(it.n_in, 1, 256); tpr = batch_tpr<IqTraits<10>>(it.n_in); break;
        default: throw std::invalid_argument("native_mmvq_batch: ggml type " + std::to_string(it.type) + " not covered");
        }
        static const int short_rows = [] {
            const char* e = std::getenv("STRATA_BATCH_SHORT_ROWS");
            return e ? std::atoi(e) : 4;
        }();
        const int div = it.type == 8 ? 32 : 256;
        static const int long_rows = [] {
            const char* e = std::getenv("STRATA_BATCH_LONG_ROWS");
            return e ? std::atoi(e) : 2;
        }();
        static const bool cols_on = [] { const char* e = std::getenv("STRATA_BATCH_COLS"); return !(e && e[0] == '0'); }();
        // dense = no ids, or ids into a zero expert stride (every column the same matrix)
        const bool cols = cols_on && (!it.ids || it.expert_bytes == 0) && it.n_ids > 1 && it.n_ids <= 8 &&
                          (it.x_group <= 1);
        const int rows = cols ? 2 : it.n_in / div <= 4 ? (short_rows == 4 ? 4 : 2) : (long_rows == 4 ? 4 : 2);
        const int rows_per_block = rows * (WARPS * WARP / tpr);
        a.it[i] = it;
        a.tpr[i] = tpr;
        a.rows[i] = rows;
        a.rblocks[i] = (it.n_out + rows_per_block - 1) / rows_per_block;
        a.cols[i] = cols ? 1 : 0;
        a.start[i] = blocks;
        blocks += a.rblocks[i] * (cols ? 1 : it.n_ids);
    }
    a.start[n_items] = blocks;
    native_mmvq_batch_kernel<<<unsigned(blocks), dim3(WARP, WARPS), 0, static_cast<cudaStream_t>(stream)>>>(a);
    launch_check();
}


// ============================ gfx11 (wave32): one kernel for 1..8 columns, R rows per wave ============================
// Talus (Qwen3.8-27B decode / MTP verify on gfx1151).  A wave takes R rows and walks their blocks lane-strided
// (kbx = lane / T, stride 32 / T, as native_mmvq_wave_kernel does for wave64), each row's R weight blocks are loaded
// before the columns are applied (the R loads in flight together), and each (row, column) sum is reduced by one XOR
// butterfly.  A sum's operations and their order depend on neither the column count nor R, so every column of a
// verify batch is bitwise the single-column (decode) result, whatever R a shape uses.  4 waves (4 R rows) per block.
// STRATA_MMVQ_W32=0: the layouts above; STRATA_MMVQ_W32_R=1|2|4 forces R.
template<> struct IqBlockOf<22> { using T = block_iq2_s; };
}  // namespace strata::kernels
namespace talus_iq {
template<> struct IqShape<22> { static constexpr int qk = 256, ipb = 8, step = 2; };    // IQ2_S
}  // namespace talus_iq
namespace strata::kernels {
namespace {
// CG < NCOLS: the columns in groups of CG, the wave's weight blocks loaded again per group (from the cache it just
// read them through): fewer live registers at 6..8 columns.  A column's sum is the same operations in the same order.
// Talus: a compile-time loop - f(std::integral_constant<int, I>) for I = 0 .. N-1.  `#pragma unroll` left
// the column loop rolled above 9 columns (tmp[] then indexed through m0: ~150 SALU and serialized loads per iteration).
template<int I, int N, typename Fn>
__device__ __forceinline__ void w32_static_for(Fn&& f) {
    if constexpr (I < N) {
        f(std::integral_constant<int, I>{});
        w32_static_for<I + 1, N>(f);
    }
}
template<typename F, int NCOLS, int R, int U = 1, int CG = NCOLS, int SG = 0>
__device__ __forceinline__ void w32_body(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                         float* __restrict__ y, int n_in, int n_out, int blk) {
    constexpr int BPIW = 32 / F::T;
    static_assert(32 % F::T == 0, "a block's threads must tile the wave");
    const int lane = int(threadIdx.x) & 31;
    const int row0 = (blk * 4 + (int(threadIdx.x) >> 5)) * R;
    if (row0 >= n_out) return;
    const int bpr = n_in / F::DIV;
    const int xs = n_in / Q8K;
    const int kqs = F::kqs(lane);
    const typename F::Block* wr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) wr[i] = w + std::size_t(min(row0 + i, n_out - 1)) * bpr;
#pragma unroll
    for (int c0 = 0; c0 < NCOLS; c0 += CG) {
        float tmp[CG][R] = {};
#pragma unroll U
        for (int kbx = lane / F::T; kbx < bpr; kbx += BPIW) {
            const int kby = kbx * F::KBY;
            typename F::W wv[R];
#pragma unroll
            for (int i = 0; i < R; ++i) wv[i] = F::load(wr[i] + kbx, kqs);
            w32_static_for<0, CG>([&](auto jc) {
                constexpr int j = decltype(jc)::value;
                if (c0 + j < NCOLS)
                    w32_static_for<0, R>([&](auto ic) {
                        constexpr int i = decltype(ic)::value;
                        tmp[j][i] += F::apply(wv[i], x + std::size_t(c0 + j) * xs + kby, kqs);
                    });
            });
#if defined(__HIP_DEVICE_COMPILE__)
            // Talus: SG = every global load of the iteration (the R weight blocks, every column's
            // activations) scheduled ahead of the arithmetic, whatever register count the scheduler would aim for
            if constexpr (SG != 0) {
                __builtin_amdgcn_sched_group_barrier(0x020, 256, 0);
                __builtin_amdgcn_sched_group_barrier(0x001, 8192, 0);
            }
#endif
        }
#pragma unroll
        for (int j = 0; j < CG; ++j)
            if (c0 + j < NCOLS)
#pragma unroll
                for (int i = 0; i < R; ++i) {
                    float v = tmp[j][i];
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
                    if (lane == ((c0 + j) * R + i) % 32 && row0 + i < n_out) y[std::size_t(c0 + j) * n_out + row0 + i] = v;
                }
    }
}
template<typename F, int NCOLS, int R, int U = 1, int CG = NCOLS>
__launch_bounds__(128)
__global__ void native_mmvq_w32_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out) {
    w32_body<F, NCOLS, R, U, CG>(w, x, y, n_in, n_out, int(blockIdx.x));
}
// Talus: 6..8 columns with a lower occupancy target.  At 16 waves per SIMD (the default target, <= 96
// VGPRs) the scheduler gives up the 6..8-column loop's load-level parallelism - every column's activation loads then
// wait in turn (IQ4_XS: 39 VGPRs at 6 columns); capping occupancy at <= 12 waves keeps the loads in flight.
// Talus: the cap W is a template parameter: the (type, column count) table below picks R and W per arch.
template<typename F, int NCOLS, int R, int W, int CG = NCOLS, int SG = 0>
__launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(1, W)))
__global__ void native_mmvq_w32_wide_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                            float* __restrict__ y, int n_in, int n_out) {
    w32_body<F, NCOLS, R, 1, CG, SG>(w, x, y, n_in, n_out, int(blockIdx.x));
}
// Talus: up to 3 matrices of one type that read the same activations in one launch (gate + up, qkv + z,
// q / k / v): block b runs matrix i's block b - start[i], each matrix's rows exactly as its own launch would.
struct W32Multi {
    const void* w[3];
    float* y[3];
    int n_out[3];
    int start[3];
};
template<typename F, int NCOLS, int R>
__launch_bounds__(128)
__global__ void native_mmvq_w32_multi_kernel(W32Multi m, const Q81Block* __restrict__ x, int n_in) {
    const int b = int(blockIdx.x);
    const int i = b >= m.start[2] ? 2 : (b >= m.start[1] ? 1 : 0);
    w32_body<F, NCOLS, R, 1>(static_cast<const typename F::Block*>(m.w[i]), x, m.y[i], n_in, m.n_out[i], b - m.start[i]);
}
template<typename F, int NCOLS, int R, int W>
__launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(1, W)))
__global__ void native_mmvq_w32_multi_wide_kernel(W32Multi m, const Q81Block* __restrict__ x, int n_in) {
    const int b = int(blockIdx.x);
    const int i = b >= m.start[2] ? 2 : (b >= m.start[1] ? 1 : 0);
    w32_body<F, NCOLS, R, 1>(static_cast<const typename F::Block*>(m.w[i]), x, m.y[i], n_in, m.n_out[i], b - m.start[i]);
}
// Talus: the instance a (type, column count) runs - R rows per wave and W, the waves-per-EU cap of the
// instance (0: the compiler's default occupancy target, the plain kernel).  The valid instances: W 0 at 1..8
// columns, W 12 at 5..8, W 8 at 5..16 (R 1 / 2 above 8 columns).  Every instance's columns are bitwise the
// single-column R 1 result (--mmvq-bench checks each (R, W) the table can pick).
// cg (above 8 columns): 0 = every column in one pass, 8 = the columns in groups of 8 inside the kernel (the wave's
// weight blocks loaded again per group), -8 = one launch per 8 columns.
struct W32Cfg { int r, w, cg = 0, sg = 0; };   // sg: the load-first schedule (W 8 instances)
constexpr bool w32_valid(int n, int r, int w) {
    return w == 0 ? n <= 8 : w == 12 ? (n >= 5 && n <= 8) : w == 8 ? (n >= 2 && (n <= 8 || r <= 2))
         : w == 6 ? (n >= 11 && r == 2) : false;   // (W 6: up to 256 VGPRs, the 11..16-column R 2 single pass)
}
W32Cfg w32_sanitize(int n, W32Cfg c) {
    if (n <= 8 || (c.cg != 8 && c.cg != -8)) c.cg = 0;
    if (c.r != 1 && c.r != 2 && c.r != 4) c.r = 1;
    if (n > 8 && c.r == 4) c.r = 2;
    if (c.w != 0 && c.w != 6 && c.w != 8 && c.w != 12) c.w = 0;
    if (!w32_valid(n, c.r, c.w)) c.w = n > 8 ? 8 : (n >= 5 && c.w == 12 ? 12 : (n >= 2 && c.w ? 8 : 0));
    if (!w32_valid(n, c.r, c.w)) c.w = n > 8 ? 8 : 0;
    return c;
}
W32Cfg w32_cfg(int type, int ncols);
template<typename F, int N, int R, int W>
void w32_one(const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out, unsigned blocks, cudaStream_t s,
             int cg, int sg) {
    if constexpr (!w32_valid(N, R, W)) throw std::invalid_argument("native MMVQ (w32): no such instance");
    else if constexpr (W == 0) native_mmvq_w32_kernel<F, N, R, 1><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out);
    else if constexpr (W == 8) {
        if (N > 8 && cg == 8) {
            if (sg) native_mmvq_w32_wide_kernel<F, N, R, W, (N > 8 ? 8 : N), 1><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out);
            else native_mmvq_w32_wide_kernel<F, N, R, W, (N > 8 ? 8 : N)><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out);
        } else if (sg) native_mmvq_w32_wide_kernel<F, N, R, W, N, 1><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out);
        else native_mmvq_w32_wide_kernel<F, N, R, W><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out);
    } else native_mmvq_w32_wide_kernel<F, N, R, W><<<blocks, 128, 0, s>>>(w, x, y, n_in, n_out);
}
template<typename F, int R, int W>
void w32_launch_rw(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s,
                   int cg = 0, int sg = 0) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const unsigned blocks = unsigned((std::size_t(n_out) + 4 * R - 1) / (4 * R));
    if (cg == -8 && ncols > 8) {   // one launch per 8 columns: column j of a launch is column c0 + j of the batch
        for (int c0 = 0; c0 < ncols; c0 += 8) {
            const int nc = std::min(8, ncols - c0);
            if (nc >= 2) w32_launch_rw<F, R, W>(weights, x + std::size_t(c0) * (n_in / Q8K), y + std::size_t(c0) * n_out,
                                                n_in, n_out, nc, s, 0, sg);
            else w32_launch_rw<F, R, 0>(weights, x + std::size_t(c0) * (n_in / Q8K), y + std::size_t(c0) * n_out, n_in,
                                        n_out, nc, s);
        }
        return;
    }
    switch (ncols) {
#define STRATA_W32_CASE(N) case N: w32_one<F, N, R, W>(w, x, y, n_in, n_out, blocks, s, cg, sg); break;
        STRATA_W32_CASE(1) STRATA_W32_CASE(2) STRATA_W32_CASE(3) STRATA_W32_CASE(4)
        STRATA_W32_CASE(5) STRATA_W32_CASE(6) STRATA_W32_CASE(7) STRATA_W32_CASE(8)
        STRATA_W32_CASE(9) STRATA_W32_CASE(10) STRATA_W32_CASE(11) STRATA_W32_CASE(12)
        STRATA_W32_CASE(13) STRATA_W32_CASE(14) STRATA_W32_CASE(15) STRATA_W32_CASE(16)
#undef STRATA_W32_CASE
        default: throw std::invalid_argument("native MMVQ (w32) requires 1 <= ncols <= 16");
    }
}
template<typename F, int N, int R, int W>
void w32_multi_one(W32Multi m, int blocks, const Q81Block* x, int n_in, cudaStream_t s) {
    if constexpr (!w32_valid(N, R, W) || N > 8) throw std::invalid_argument("native MMVQ (w32 multi): no such instance");
    else if constexpr (W == 0) native_mmvq_w32_multi_kernel<F, N, R><<<unsigned(blocks), 128, 0, s>>>(m, x, n_in);
    else native_mmvq_w32_multi_wide_kernel<F, N, R, W><<<unsigned(blocks), 128, 0, s>>>(m, x, n_in);
}
template<typename F, int R, int W>
void w32_multi_launch_rw(W32Multi m, int blocks, const void* x_q8_1, int n_in, int ncols, cudaStream_t s) {
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    switch (ncols) {
#define STRATA_W32M_CASE(N) case N: w32_multi_one<F, N, R, W>(m, blocks, x, n_in, s); break;
        STRATA_W32M_CASE(1) STRATA_W32M_CASE(2) STRATA_W32M_CASE(3) STRATA_W32M_CASE(4)
        STRATA_W32M_CASE(5) STRATA_W32M_CASE(6) STRATA_W32M_CASE(7) STRATA_W32M_CASE(8)
#undef STRATA_W32M_CASE
        default: throw std::invalid_argument("native MMVQ (w32 multi) requires 1 <= ncols <= 8");
    }
}
#define STRATA_W32_RW(FN, ...) \
    if (c.w == 0) { if (c.r >= 4) FN<F, 4, 0>(__VA_ARGS__); else if (c.r == 2) FN<F, 2, 0>(__VA_ARGS__); else FN<F, 1, 0>(__VA_ARGS__); } \
    else if (c.w == 12) { if (c.r >= 4) FN<F, 4, 12>(__VA_ARGS__); else if (c.r == 2) FN<F, 2, 12>(__VA_ARGS__); else FN<F, 1, 12>(__VA_ARGS__); } \
    else if (c.w == 6) { FN<F, 2, 6>(__VA_ARGS__); } \
    else { if (c.r >= 4) FN<F, 4, 8>(__VA_ARGS__); else if (c.r == 2) FN<F, 2, 8>(__VA_ARGS__); else FN<F, 1, 8>(__VA_ARGS__); }
template<typename F>
void w32_multi_launch(W32Cfg c, int n, const void* const* w, float* const* y, const int* n_out, const void* x, int n_in,
                      int ncols, cudaStream_t s) {
    W32Multi m{};
    int blocks = 0;
    for (int i = 0; i < 3; ++i) {
        m.start[i] = blocks;
        if (i < n) {
            m.w[i] = w[i];
            m.y[i] = y[i];
            m.n_out[i] = n_out[i];
            blocks += (n_out[i] + 4 * c.r - 1) / (4 * c.r);
        } else {
            m.start[i] = 1 << 30;
        }
    }
    STRATA_W32_RW(w32_multi_launch_rw, m, blocks, x, n_in, ncols, s)
}
template<typename F>
void w32_launch(W32Cfg c, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    STRATA_W32_RW(w32_launch_rw, weights, x_q8_1, y, n_in, n_out, ncols, s, c.cg, c.w == 8 ? c.sg : 0)
}
#undef STRATA_W32_RW
int g_kq8 = [] { const char* e = std::getenv("STRATA_MMVQ_KQ8"); return e ? std::atoi(e) : 1; }();   // Talus: KQ8Traits for Q4_K / Q5_K
int g_w32_mode = -1;   // -1: unset (the device and STRATA_MMVQ_W32 decide), 0 off, 1 on
int g_w32_r = 0;       // 0: the table
bool w32_on() {
    if (g_w32_mode < 0) {
        int on = 0;
#if defined(__HIPCC__)
        cudaDeviceProp p;
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess)
            on = strata::kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName) ? 1 : 0;
        (void) cudaGetLastError();
#endif
        const char* e = std::getenv("STRATA_MMVQ_W32");
        if (e && e[0] == '0') on = 0;
        if (const char* r = std::getenv("STRATA_MMVQ_W32_R")) g_w32_r = std::atoi(r);
        g_w32_mode = on;
    }
    return g_w32_mode == 1;
}
// rows per wave by type and column count (read off --mmvq-bench on gfx1151, real UD-Q4_K_XL / UD-IQ4_XS layers, cold)
// One R for every column count of a type: a sum's operations do not depend on R in the source, but the compiler's
// contractions may differ between the R instances, and a verify column must be bitwise the decode step's.
// Per type, read off the in-engine verify (5 columns) on gfx1151 (UD-IQ4_XS, MTP n4 rounds, rocprofv3): IQ4_XS 40.6 /
// 42.5 / 46.3 ms at R = 1 / 2 / 4, IQ3_S 11.8 / 12.6 / 10.3, Q3_K 2.33 / 2.27 / 2.12, Q4_K 11.2 / 9.3 / 10.1, Q5_K
// (incl. the head) 15.9 / 13.1 / 15.9; the single-column cost moves less (+0.1-0.5 ms per pass).
// Talus: R by column count.  --mmvq-bench checks every column of every width AND every R against the
// single-column R = 1 result ("exact yes" for all of UD-IQ4_XS / UD-Q4_K_XL / the MTP file), so R may follow the column
// count without breaking a verify row's bitwise match.  In-engine (gfx1151, UD-IQ4_XS, MTP n4 at 2K, us per call,
// R1 / R2 / R4): 1 column Q4_K 124 / 127 / 131, Q5_K 203 / 209 / 211, IQ3_S 165 / 168 / 177, IQ2_S 167 / 154 / 214;
// 5 columns Q4_K 215 / 177 / 192, Q5_K 314 / 254 / 311, IQ3_S 256 / 272 / 223.  STRATA_MMVQ_W32_RTAB=0: one R per type.
// Talus: (type, column count) -> (R, W) by arch.  A rule list per arch, first match wins (type -1 = any);
// gfx1151 is the only tuned entry, other gfx11 parts take the generic rules (generic rules).
// STRATA_MMVQ_W32_TAB="type:cols:R:W[:CG[:SG]],..." (type or cols '*', cols also "lo-hi") goes first, for in-engine sweeps;
// STRATA_MMVQ_W32_R=1|2|4 forces R; STRATA_MMVQ_W32_RTAB=0 drops the gfx1151 rules.
struct W32Rule { int type, n0, n1, r, w, cg = 0, sg = 0; };
const W32Rule kW32Generic[] = {
    {10, 1, 1, 2, 0}, {17, 1, 1, 2, 0}, {22, 1, 1, 2, 0}, {11, 1, 1, 4, 0}, {-1, 1, 1, 1, 0},
    {12, 2, 5, 2, 0}, {13, 2, 5, 2, 0}, {14, 2, 5, 2, 0}, {21, 2, 5, 4, 0}, {11, 2, 5, 4, 0}, {10, 2, 5, 4, 0},
    {-1, 2, 5, 1, 0},
    {12, 6, 7, 2, 12}, {13, 6, 7, 2, 12}, {14, 6, 7, 2, 12}, {21, 6, 7, 4, 12}, {11, 6, 7, 4, 12}, {10, 6, 7, 4, 12},
    {-1, 6, 7, 1, 12},
    {12, 8, 8, 2, 8}, {13, 8, 8, 2, 8}, {14, 8, 8, 2, 8}, {21, 8, 8, 4, 8}, {11, 8, 8, 4, 8}, {10, 8, 8, 4, 8},
    {-1, 8, 8, 1, 8}, {-1, 9, 16, 1, 8},
};
// gfx1151 (Strix Halo): in-engine, UD-IQ4_XS verify windows at 2K (gfx1151 verify-window table)
// (STRATA_MMVQ_W32_TAB tuning table, ms per verify pass by type): 2..3 columns R 1 (R 2 for the IQ3 / IQ2
// grids), 4 columns R 2 at <= 8 waves (IQ4_XS R 1); 5..9 columns R 2 at <= 8 waves for every type (IQ4_XS 8 columns
// 35.9 ms against 42.4 at R 1, 33.0 at 1 column), Q4_K R 4.
const W32Rule kW32Gfx1151[] = {
    // Q4_K (KQ8) at 4..5 columns (MiMo DFlash verify) R 1 W 0, as at 2..3: MiMo 8192x4096 4 / 5 cols 132 / 132
    // -> 111 / 113 us, 4096x12288 156 / 163 -> 145 / 150, 4096x14848 186 / 192 -> 173 / 187 (mvb graph sweep with a
    // dependent kernel; every column bitwise the 1-column result, mvbx);
    // STRATA_MMVQ_W32_TAB="12:4-5:2:8" restores
    {12, 4, 5, 1, 0},
    // Graph sweep with a dependent kernel, 27B shapes 5120x17408 / 17408x5120):
    // IQ2_S / IQ2_XS at 3..5 columns R 4 W 8 (IQ2_S 5 cols 206 / 187 -> 187 / 179 us, IQ2_XS 4 cols 175 / 173 -> 167 /
    // 153), Q3_K at 3..6 R 4 W 8 (6 cols 375 / 354 -> 345 / 345) and at 8 R 4 W 0 (458 / 434 -> 431 / 409). Bitwise
    // (every instance's columns are the 1-column result, mb/mvbx.cpp); STRATA_MMVQ_W32_TAB overrides.
    {22, 3, 5, 4, 8}, {17, 3, 5, 4, 8}, {11, 3, 6, 4, 8}, {11, 8, 8, 4, 0},
    {21, 2, 3, 2, 0}, {18, 2, 3, 2, 0}, {22, 2, 3, 2, 0}, {17, 2, 3, 2, 0}, {11, 2, 3, 2, 0}, {-1, 2, 3, 1, 0},
    // IQ3_XXS at 4 columns R 2 at <= 8 waves (W 8), not the default occupancy: 27B 5120x17408 / 17408x5120
    // 240-250 -> 176-188 us (mvb graph sweep); STRATA_MMVQ_W32_TAB="18:4:2:0" restores
    {23, 4, 4, 1, 8}, {18, 4, 4, 2, 8}, {-1, 4, 4, 2, 8},
    {-1, 5, 10, 2, 8},   // (Q4_K / Q5_K: KQ8Traits R 2 selected on gfx1151)
    // 11..13 columns: R 2 in one pass at <= 6 waves (<= 256 VGPRs; 12-row window 117 ms against 128 in groups); 16:
    // that for IQ4_XS / Q4_K / IQ2_XS / Q2_K / IQ4_NL (IQ4_XS 53.7 ms against 60.7), else and at 14..15 R 2 in column
    // groups of 8 (16-row window 155 ms against 198 at R 1)
    {-1, 11, 13, 2, 6}, {23, 16, 16, 2, 6}, {12, 16, 16, 2, 6}, {13, 16, 16, 2, 6}, {17, 16, 16, 2, 6}, {10, 16, 16, 2, 6}, {20, 16, 16, 2, 6},
    {-1, 14, 16, 2, 8, 8},
};
std::vector<W32Rule> w32_env_rules() {
    std::vector<W32Rule> r;
    const char* e = std::getenv("STRATA_MMVQ_W32_TAB");
    if (!e) return r;
    std::string v(e);
    for (size_t p0 = 0; p0 < v.size();) {
        size_t p1 = v.find(',', p0);
        if (p1 == std::string::npos) p1 = v.size();
        const std::string it = v.substr(p0, p1 - p0);
        p0 = p1 + 1;
        int f[6] = {-1, -1, 1, 0, 0, 0};
        int n0 = 1, n1 = 16;
        size_t q0 = 0;
        for (int k = 0; k < 6 && q0 <= it.size(); ++k) {
            size_t q1 = it.find(':', q0);
            if (q1 == std::string::npos) q1 = it.size();
            const std::string t = it.substr(q0, q1 - q0);
            q0 = q1 + 1;
            if (k == 1) {
                if (t != "*") { const size_t d = t.find('-'); n0 = std::atoi(t.c_str()); n1 = d == std::string::npos ? n0 : std::atoi(t.c_str() + d + 1); }
            } else if (t != "*") f[k] = std::atoi(t.c_str());
        }
        r.push_back({f[0], n0, n1, f[2], f[3], f[4], f[5]});
    }
    return r;
}
int w32_arch() {   // 1: gfx1151, 0: other
    static const int a = [] {
        int r = 0;
#if defined(__HIPCC__)
        cudaDeviceProp p;
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess)
            r = strata::kernels::gfx_arch_is_gfx1151(p.gcnArchName) ? 1 : 0;
        (void) cudaGetLastError();
#endif
        return r;
    }();
    return a;
}
int g_w32_rtab = [] { const char* e = std::getenv("STRATA_MMVQ_W32_RTAB"); return e ? std::atoi(e) : 1; }();
int g_w32_wf = -1;   // native_mmvq_set_w32_waves (the bench): forces W when >= 0
W32Cfg w32_cfg(int type, int ncols) {
    static const std::vector<W32Rule> env = w32_env_rules();
    auto find = [&](const W32Rule* b, const W32Rule* e, W32Cfg& c) {
        for (const W32Rule* q = b; q != e; ++q)
            if ((q->type < 0 || q->type == type) && ncols >= q->n0 && ncols <= q->n1 && q->w >= 0) { c = {q->r, q->w, q->cg, q->sg}; return true; }
        return false;
    };
    W32Cfg c{1, 0};
    bool hit = !env.empty() && find(env.data(), env.data() + env.size(), c);
    if (!hit && g_w32_rtab && w32_arch() == 1) hit = find(std::begin(kW32Gfx1151), std::end(kW32Gfx1151), c);
    if (!hit) find(std::begin(kW32Generic), std::end(kW32Generic), c);
    if (g_w32_r > 0) c.r = g_w32_r;
    if (g_w32_wf >= 0) { c.w = g_w32_wf % 100; c.sg = g_w32_wf >= 100; }   // (100 + W: with the load-first schedule)
    return w32_sanitize(ncols, c);
}
}  // namespace

bool native_mmvq_w32_supported(int t) noexcept {
    switch (t) {
    case 2: case 6: case 8: case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 20: case 21:
    case 22: case 23: case 29: case 42: return true;
    default: return false;
    }
}
bool native_mmvq_w32_active() { return w32_on(); }
void native_mmvq_set_w32(int mode, int rows) {
    (void) w32_on();
    g_w32_mode = mode;
    g_w32_r = rows;
}
void native_mmvq_set_w32_waves(int w) { g_w32_wf = w; }   // Talus: -1 = the table's W
void native_mmvq_w32_config(int type, int ncols, int* r, int* w) { const W32Cfg c = w32_cfg(type, ncols); *r = c.r; *w = c.w; }
bool native_mmvq_w32_multi(int t, int n, const void* const* w, float* const* y, const int* n_out, const void* x,
                           int n_in, int ncols, void* stream) {
    static const bool off = [] { const char* e = std::getenv("STRATA_MMVQ_MULTI"); return e && e[0] == '0'; }();
    if (off || n < 1 || n > 3 || !w32_on() || !native_mmvq_w32_supported(t) || ncols < 1 || ncols > 8) return false;
    auto s = static_cast<cudaStream_t>(stream);
    const W32Cfg r = w32_cfg(t, ncols);
    switch (t) {
    case 2: w32_multi_launch<SmallTraits<Q40Block, 4>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 6: w32_multi_launch<SmallTraits<Q50Block, 4>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 8: w32_multi_launch<SmallTraits<Q80Block, 8>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 20: w32_multi_launch<SmallTraits<IQ4NLBlock, 4>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 11: w32_multi_launch<Q3KTraits>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 12: if (g_kq8) w32_multi_launch<KQ8Traits<false>>(r, n, w, y, n_out, x, n_in, ncols, s);
             else w32_multi_launch<Q4KTraits>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 13: if (g_kq8) w32_multi_launch<KQ8Traits<true>>(r, n, w, y, n_out, x, n_in, ncols, s);
             else w32_multi_launch<Q5KTraits>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 14: w32_multi_launch<Q6KTraits>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 23: w32_multi_launch<IQ4XSTraits>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 10: w32_multi_launch<IqTraits<10>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 16: w32_multi_launch<IqTraits<16>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 17: w32_multi_launch<IqTraits<17>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 18: w32_multi_launch<IqTraits<18>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 21: w32_multi_launch<IqTraits<21>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    case 22: w32_multi_launch<IqTraits<22>>(r, n, w, y, n_out, x, n_in, ncols, s); break;
    default: return false;
    }
    launch_check();
    return true;
}
// ---- wide dense mat-vec with the activations staged in LDS (Laguna verify windows of up to 64 columns) ----
// The wave32 kernels above read every column's activations from the cache per row group (R rows) and run out of
// registers to keep those loads in flight above 8 columns (16 columns: 227 VGPRs, ~6 waves, a wait per load): at
// verify(32) the dense mat-vecs cost ~4 ms per token, about 15x their ALU time.  Here a block of WAVES waves stages the
// columns' Q8_1 blocks of one K chunk in LDS (a padded copy, the 32 quants 16-byte aligned for apply's b128 reads), every
// wave runs w32_body's lane walk over that chunk for its R rows, and the column passes (NC columns each) run inside the
// block, so a row block's weights come from DRAM once per call (the passes re-read them from the cache).  A column's sum
// is w32_body's operations in its order: lane l takes blocks kbx = l / T + i * (32 / T) in i order with kqs = kqs(l),
// tmp += apply(load(row, kbx), x, kqs), then the same XOR butterfly - bitwise the single-column wave32 result.
struct alignas(16) Q81Lds { half2 ds; int pad[3]; int8_t qs[32]; };
static_assert(sizeof(Q81Lds) == 48 && offsetof(Q81Lds, qs) == 16);
namespace {
constexpr int kLdsQ = 32;   // Q8_1 blocks (1024 values) per column per chunk
template<typename F, int NC, int R, int WAVES>
__launch_bounds__(WAVES * 32)
__global__ void native_mmvq_lds_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out, int ncols, int groups, int cpg) {
    constexpr int BPIW = 32 / F::T;
    static_assert(32 % F::T == 0 && (kLdsQ % F::KBY) == 0 && ((kLdsQ / F::KBY) % BPIW) == 0, "chunk shape");
    constexpr int CK = kLdsQ / F::KBY;   // weight blocks per row per chunk
    constexpr int U = CK / BPIW;         // lane iterations per chunk
    __shared__ Q81Lds xl[NC * kLdsQ];
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5;
    const int rbk = int(blockIdx.x) / groups, grp = int(blockIdx.x) % groups;   // a row block's column groups adjacent
    const int row0 = (rbk * WAVES + wave) * R;
    const bool live = row0 < n_out;
    const int bpr = n_in / F::DIV, xs = n_in / Q8K;
    const int kqs = F::kqs(lane);
    const int nchunks = (bpr + CK - 1) / CK;
    const typename F::Block* wr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) wr[i] = w + std::size_t(min(row0 + i, n_out - 1)) * bpr;
    const int cb0 = grp * cpg, cb1 = min(ncols, cb0 + cpg);
    // each thread stages PT Q8_1 blocks per chunk (block bi = tid + p * WAVES * 32 of the NC x kLdsQ chunk); the next
    // chunk's words are loaded into registers before the current chunk's arithmetic, so their latency overlaps it
    constexpr int PT = NC * kLdsQ / (WAVES * 32);
    static_assert(PT * WAVES * 32 == NC * kLdsQ, "staging split");
    int pre[PT][9];
    auto fetch = [&](int c0, int nc, int ch) {
        const int q0 = ch * kLdsQ, nq = min(kLdsQ, xs - q0);
#pragma unroll
        for (int p = 0; p < PT; ++p) {
            const int bi = tid + p * WAVES * 32, c = bi / kLdsQ, q = bi % kLdsQ;
            if (c < nc && q < nq) {
                const int* src = reinterpret_cast<const int*>(x + std::size_t(c0 + c) * xs + q0 + q);
#pragma unroll
                for (int k = 0; k < 9; ++k) pre[p][k] = src[k];
            }
        }
    };
    auto stage = [&](int nc, int ch) {
        const int q0 = ch * kLdsQ, nq = min(kLdsQ, xs - q0);
#pragma unroll
        for (int p = 0; p < PT; ++p) {
            const int bi = tid + p * WAVES * 32, c = bi / kLdsQ, q = bi % kLdsQ;
            if (c < nc && q < nq) {
                Q81Lds& d = xl[c * kLdsQ + q];
                *reinterpret_cast<int*>(&d.ds) = pre[p][0];
                reinterpret_cast<int4*>(d.qs)[0] = make_int4(pre[p][1], pre[p][2], pre[p][3], pre[p][4]);
                reinterpret_cast<int4*>(d.qs)[1] = make_int4(pre[p][5], pre[p][6], pre[p][7], pre[p][8]);
            }
        }
    };
    if (cb0 < cb1) fetch(cb0, min(NC, cb1 - cb0), 0);
    for (int c0 = cb0; c0 < cb1; c0 += NC) {
        const int nc = min(NC, cb1 - c0);
        float tmp[NC][R] = {};
        for (int ch = 0; ch < nchunks; ++ch) {
            __syncthreads();   // the previous chunk's readers are done
            stage(nc, ch);
            __syncthreads();
            // prefetch the next chunk (this pass's next, or the next pass's first)
            if (ch + 1 < nchunks) fetch(c0, nc, ch + 1);
            else if (c0 + NC < cb1) fetch(c0 + NC, min(NC, cb1 - c0 - NC), 0);
            if (live) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int kl = u * BPIW + lane / F::T;
                    const int kbx = ch * CK + kl;
                    if (kbx < bpr) {
                        const int kby = kl * F::KBY;
                        typename F::W wv[R];
#pragma unroll
                        for (int i = 0; i < R; ++i) wv[i] = F::load(wr[i] + kbx, kqs);
                        w32_static_for<0, NC>([&](auto jc) {
                            constexpr int j = decltype(jc)::value;
                            if (j < nc)
                                w32_static_for<0, R>([&](auto ic) {
                                    constexpr int i = decltype(ic)::value;
                                    tmp[j][i] += F::apply(wv[i], xl + j * kLdsQ + kby, kqs);
                                });
                        });
                    }
                }
            }
        }
        if (live) {
#pragma unroll
            for (int j = 0; j < NC; ++j)
                if (j < nc)
#pragma unroll
                    for (int i = 0; i < R; ++i) {
                        float v = tmp[j][i];
#pragma unroll
                        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
                        if (lane == (j * R + i) % 32 && row0 + i < n_out) y[std::size_t(c0 + j) * n_out + row0 + i] = v;
                    }
        }
    }
}
template<typename F, int NC, int R, int WAVES>
void lds_launch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, int target, cudaStream_t s) {
    const int rb = (n_out + WAVES * R - 1) / (WAVES * R);
    const int passes = (ncols + NC - 1) / NC;
    const int groups = std::max(1, std::min(passes, (target + rb - 1) / rb));
    const int cpg = NC * ((passes + groups - 1) / groups);
    const int g = (ncols + cpg - 1) / cpg;
    native_mmvq_lds_kernel<F, NC, R, WAVES><<<unsigned(rb * g), WAVES * 32, 0, s>>>(
        static_cast<const typename F::Block*>(w), static_cast<const Q81Block*>(x), y, n_in, n_out, ncols, g, cpg);
}
// STRATA_MMVQ_LDS: the column count from which tk_mmvq takes this kernel (unset / 0 = never; the Laguna CLI sets 9);
// STRATA_MMVQ_LDS_CFG="NC:R:WAVES:TARGET" (16:2:4:320) for sweeps.  Types: Q4_K / Q5_K (KQ8 traits), Q6_K, Q8_0.
int lds_min() {   // read on first use: the Laguna CLI sets it after the library is loaded
    static const int m = [] { const char* e = std::getenv("STRATA_MMVQ_LDS"); return e ? std::atoi(e) : 0; }();
    return m;
}
struct LdsCfg { int nc = 16, r = 2, waves = 4, target = 320; };
LdsCfg lds_cfg() {
    static const LdsCfg c = [] {
        LdsCfg c;
        if (const char* e = std::getenv("STRATA_MMVQ_LDS_CFG")) {
            int a[4] = {c.nc, c.r, c.waves, c.target}, n = 0;
            for (const char* p = e; *p && n < 4; ++n) {
                a[n] = std::atoi(p);
                while (*p && *p != ':') ++p;
                if (*p == ':') ++p;
            }
            c.nc = a[0] == 8 ? 8 : 16;
            c.r = a[1] == 1 ? 1 : a[1] == 4 ? 4 : 2;
            c.waves = a[2] == 8 ? 8 : 4;
            c.target = std::max(1, a[3]);
        }
        return c;
    }();
    return c;
}
template<typename F>
void lds_dispatch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const LdsCfg c = lds_cfg();
#define STRATA_LDS_RW(NC, R) \
    if (c.waves == 8) lds_launch<F, NC, R, 8>(w, x, y, n_in, n_out, ncols, c.target, s); \
    else lds_launch<F, NC, R, 4>(w, x, y, n_in, n_out, ncols, c.target, s);
    if (c.nc == 8) {
        if (c.r == 4) { STRATA_LDS_RW(8, 4) } else if (c.r == 1) { STRATA_LDS_RW(8, 1) } else { STRATA_LDS_RW(8, 2) }
    } else {
        if (c.r == 1) { STRATA_LDS_RW(16, 1) } else { STRATA_LDS_RW(16, 2) }
    }
#undef STRATA_LDS_RW
}
}  // namespace

bool native_mmvq_lds(int t, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    // only where the single-column call takes the wave32 kernels with these traits (the rows must be its rows)
    const int lmin = lds_min();
    if (lmin <= 0 || ncols < lmin || !w32_on() || !native_mmvq_w32_supported(t)) return false;
    const auto s = static_cast<cudaStream_t>(stream);
    switch (t) {
    case 12: if (!g_kq8) return false; validate_shape(n_in, 1, 256); lds_dispatch<KQ8Traits<false>>(w, x, y, n_in, n_out, ncols, s); break;
    case 13: if (!g_kq8) return false; validate_shape(n_in, 1, 256); lds_dispatch<KQ8Traits<true>>(w, x, y, n_in, n_out, ncols, s); break;
    case 14: validate_shape(n_in, 1, 256); lds_dispatch<Q6KTraits>(w, x, y, n_in, n_out, ncols, s); break;
    case 8:  validate_shape(n_in, 1, 32);  lds_dispatch<SmallTraits<Q80Block, 8>>(w, x, y, n_in, n_out, ncols, s); break;
    default: return false;
    }
    launch_check();
    return true;
}

// --- - exact wide dense mat-vec on gfx11 matrix cores (Laguna verify windows, Q4_K / Q5_K) ----
// A column's sum in the wave32 kernel (KQ8Traits, R any): lane l = 8 c + j takes sub-block j of superblocks
// sb = c + 4 i in i order, tmp += v(sb, j) with v = dm.x * (d8 * (dot1 * sc)) - dm.y * (d8 * (dot2 * m)) (dot1 the
// 32-value int dot, dot2 the activation sub-block's quant sum, both exact), then an XOR butterfly 16, 8, 4, 2, 1:
//   q_j = (p(0,j) + p(2,j)) + (p(1,j) + p(3,j)),   y = ((q0 + q4) + (q2 + q6)) + ((q1 + q5) + (q3 + q7))
// (p(c,j) = lane 8 c + j's sequential sum; fp add is commutative, so every lane ends with these bits).  Here the int
// dots come from v_wmma_i32_16x16x16_iu8 (exact) and the float part runs the same expression in the same order: a
// block of 4 waves takes 16 rows x 16 columns; wave w owns the sub-blocks {0,4} / {2,6} / {1,5} / {3,7} (w = 0..3),
// sweeps the chains c = 0, 2, 1, 3 (6 live sums per output) and leaves q_j0 + q_j1; the y tree is finished in LDS.
// WMMA operands (gfx11): A = activations (lane l: column l % 16, 16 k), B = weights (lane l: row l % 16), D lane l
// holds D[col 2 v + l / 16][row l % 16].  Every column bitwise the single-column wave32 result (talus-laguna-wide-bench).
namespace {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_XT_W11 1
#else
#define STRATA_XT_W11 0
#endif
typedef int xt_i4 __attribute__((ext_vector_type(4)));
typedef int xt_i8 __attribute__((ext_vector_type(8)));
// The float part of one (row, column, sub-block) as the wave32 kernel computes it (gfx1151 ISA of KQ8Traits::apply:
// sumf_d = d8 * float(dot1 * sc), sumf_m = d8 * float(dot2 * m), r = fma(dm.x, sumf_d, -(dm.y * sumf_m))), pinned with
// non-contracting ops.  float(dot1) * float(sc) is float(dot1 * sc) exactly (|dot1 * sc| <= 32 * 31 * 127 * 63 < 2^24,
// |dot2 * m| <= 4064 * 63), up to the sign of a zero, which cannot reach a sum: a chain starts at +0 and +0 + -0 = +0.
// ndmy = -dm.y (exact), so the last step is one fmac.
__device__ __forceinline__ float xt_kq8_val(float dot1f, float dot2f, float scf, float mf, float d8, float dmx, float ndmy) {
    const float sumf_d = __fmul_rn(d8, __fmul_rn(dot1f, scf));
    const float sumf_m = __fmul_rn(d8, __fmul_rn(dot2f, mf));
    return __builtin_fmaf(dmx, sumf_d, __fmul_rn(ndmy, sumf_m));
}
template<bool Q5>
__launch_bounds__(128)
__global__ void native_mmvq_xt_kq8_kernel(const void* __restrict__ wv, const Q81Block* __restrict__ x, float* __restrict__ y,
                                          int n_in, int n_out, int ncols, int nct) {
#if STRATA_XT_W11
    using Block = std::conditional_t<Q5, Q5KBlock, Q4KBlock>;
    const Block* __restrict__ w = static_cast<const Block*>(wv);
    __shared__ float2 cs[4][16];          // per wave: (d8, dot2 as float bits) of the tile's 16 columns, slot (col & 1) * 8 + col / 2
    __shared__ float part[3][8][32];      // waves 1..3: their q_j0 + q_j1 per (v, lane)
    const int tid = int(threadIdx.x), lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int ct = int(blockIdx.x) % nct, rt = int(blockIdx.x) / nct;
    const int row = rt * 16 + l16, col0 = ct * 16;
    const int bpr = n_in / 256, xs = n_in / 32;
    const Block* __restrict__ wr = w + std::size_t(min(row, n_out - 1)) * bpr;
    const int acol = col0 + l16;                          // this lane's A column
    const bool alive = acol < ncols;
    const Q81Block* __restrict__ xc = x + std::size_t(alive ? acol : 0) * xs;
    // wave -> its two sub-blocks: {0,4} {2,6} {1,5} {3,7}
    const int ja = (wave & 1) * 2 + (wave >> 1), jb = ja + 4;
    float res[8];
    float t[2][8], p1[2][8];
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
        const int c = cc == 0 ? 0 : cc == 1 ? 2 : cc == 2 ? 1 : 3;
        float acc[2][8];
#pragma unroll
        for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int v = 0; v < 8; ++v) acc[q][v] = 0.0f;
        for (int sb = c; sb < bpr; sb += 4) {
            const Block* __restrict__ b = wr + sb;
            const float2 dm = __half22float2(b->dm);
            const float dmx = dm.x, ndmy = -dm.y;
            int qh[8];
            if constexpr (Q5) {
                const int4* h = reinterpret_cast<const int4*>(reinterpret_cast<const Q5KBlock*>(b)->qh);
                const int4 h0 = h[0], h1 = h[1];
                qh[0] = h0.x; qh[1] = h0.y; qh[2] = h0.z; qh[3] = h0.w; qh[4] = h1.x; qh[5] = h1.y; qh[6] = h1.z; qh[7] = h1.w;
            }
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int j = q ? jb : ja;
                // B: row's sub-block j as 32 int8 (KQ8Traits::load)
                const int k = j >> 1, sh = 4 * (j & 1);
                const int4* qq = reinterpret_cast<const int4*>(b->qs + 32 * k);
                const int4 q0 = qq[0], q1 = qq[1];
                int bv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    bv[i] = (bv[i] >> sh) & 0x0f0f0f0f;
                    if constexpr (Q5) bv[i] |= ((qh[i] >> j) & 0x01010101) << 4;
                }
                int sc, m;
                kq_scale_min(b->scales, j, sc, m);
                const float scf = float(sc), mf = float(m);
                // A: column's activation block 8 sb + j
                const Q81Block* __restrict__ xb = xc + 8 * sb + j;
                int av[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                float d8 = 0.0f;
                if (alive) {
                    const int4* u4 = reinterpret_cast<const int4*>(xb->qs);
                    const int4 a0 = u4[0], a1 = u4[1];
                    av[0] = a0.x; av[1] = a0.y; av[2] = a0.z; av[3] = a0.w; av[4] = a1.x; av[5] = a1.y; av[6] = a1.z; av[7] = a1.w;
                    d8 = __low2float(xb->ds);
                }
                int dot2 = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) dot2 = STRATA_DP4A(0x01010101, av[i], dot2);
                if (lane < 16) cs[wave][(l16 & 1) * 8 + (l16 >> 1)] = make_float2(d8, float(dot2));
                xt_i8 d = {0, 0, 0, 0, 0, 0, 0, 0};
                d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, xt_i4{av[0], av[1], av[2], av[3]}, true,
                                                                xt_i4{bv[0], bv[1], bv[2], bv[3]}, d, false);
                d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, xt_i4{av[4], av[5], av[6], av[7]}, true,
                                                                xt_i4{bv[4], bv[5], bv[6], bv[7]}, d, false);
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
                const float4* cp = reinterpret_cast<const float4*>(&cs[wave][hi * 8]);
                const float4 c0 = cp[0], c1 = cp[1], c2 = cp[2], c3 = cp[3];
                const float cd8[8] = {c0.x, c0.z, c1.x, c1.z, c2.x, c2.z, c3.x, c3.z};
                const float cs2[8] = {c0.y, c0.w, c1.y, c1.w, c2.y, c2.w, c3.y, c3.w};
#pragma unroll
                for (int v = 0; v < 8; ++v)
                    acc[q][v] = __fadd_rn(acc[q][v], xt_kq8_val(float(d[v]), cs2[v], scf, mf, cd8[v], dmx, ndmy));
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
            }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                if (cc == 0) t[q][v] = acc[q][v];                        // p(0)
                else if (cc == 1) t[q][v] = t[q][v] + acc[q][v];         // p(0) + p(2)
                else if (cc == 2) p1[q][v] = acc[q][v];                  // p(1)
                else t[q][v] = t[q][v] + (p1[q][v] + acc[q][v]);         // q_j = (p0 + p2) + (p1 + p3)
            }
    }
#pragma unroll
    for (int v = 0; v < 8; ++v) res[v] = t[0][v] + t[1][v];             // q_ja + q_jb (q0 + q4, ...)
    if (wave > 0)
#pragma unroll
        for (int v = 0; v < 8; ++v) part[wave - 1][v][lane] = res[v];
    __syncthreads();
    if (wave == 0 && row < n_out) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int col = col0 + 2 * v + hi;
            // wave 0 {0,4}, 1 {2,6}, 2 {1,5}, 3 {3,7}: ((q0+q4) + (q2+q6)) + ((q1+q5) + (q3+q7))
            const float s = (res[v] + part[0][v][lane]) + (part[1][v][lane] + part[2][v][lane]);
            if (col < ncols) y[std::size_t(col) * n_out + row] = s;
        }
    }
#endif
}
// --- - exact wide dense mat-vec for Q6_K / Q8_0 (per-4 / per-8 value rounding: no matrix-core gain on gfx11) ----
// w32_body's lane walk and sums, regrouped for wide calls: a wave takes R rows x C columns (the columns of one call in
// groups of C, the group in the grid), every column's activations read through a wave-uniform column base and a 32-bit
// lane offset (no per-lane 64-bit address math), Q6_K's weight decode (vi, scales) done once per (row, block) and its
// int products on v_mul_i32_i24 (|dot4 * sc| <= 4 * 32 * 127 * 128 < 2^23: the same integer as the 32-bit multiply).
// A column's float operations and their order are Q6KTraits / SmallTraits<Q80Block, 8>::apply's and w32_body's
// (sequential over the lane's blocks, then the XOR butterfly 16 .. 1): bitwise the single-column wave32 result.
struct XtQ6K {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8;
    struct W { int vi[2]; int sc[2]; float d; int bq8; };
    __device__ static int kqs(int lane) { return lane; }
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8 = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        const int vl = load_int_b2(w->ql, iqs);
        const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            r.sc[i] = w->scales[scale_offset + 4 * i];
            const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
            const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
            r.vi[i] = __vsubss4(vil | vih, 0x20202020);
        }
        r.d = w->d;
        return r;
    }
    // q6_q8_dot_impl's expression and order
    __device__ static float apply(const W& r, const char* __restrict__ xc, uint32_t kby, int iqs) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t o = (kby + uint32_t(r.bq8 + 2 * i)) * uint32_t(sizeof(Q81Block));
            const int u = *reinterpret_cast<const int*>(xc + o + 4 + 4 * (iqs % 8));
            const float d8 = __low2float(*reinterpret_cast<const half2*>(xc + o));
            sumf += d8 * STRATA_IMUL24(STRATA_DP4A(r.vi[i], u, 0), r.sc[i]);
        }
        return r.d * sumf;
    }
};
struct XtQ80 {
    using Block = Q80Block;
    static constexpr int DIV = 32, T = 4, KBY = 1;
    struct W { int v[2]; float d0; };
    __device__ static int kqs(int lane) { return 2 * (lane % 4); }
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int i = 0; i < 2; ++i) r.v[i] = load_int_b2(w->qs, iqs + i);
        r.d0 = w->d;
        return r;
    }
    // SmallTraits<Q80Block, 8>::apply's expression and order
    __device__ static float apply(const W& r, const char* __restrict__ xc, uint32_t kby, int iqs) {
        const uint32_t o = kby * uint32_t(sizeof(Q81Block));
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = *reinterpret_cast<const int*>(xc + o + 4 + 4 * (iqs + i));
            sumi = STRATA_DP4A(r.v[i], u, sumi);
        }
        const float d1 = __low2float(*reinterpret_cast<const half2*>(xc + o));
        return r.d0 * d1 * float(sumi);
    }
};
template<typename F, int R, int C>
__launch_bounds__(128)
__global__ void native_mmvq_xt2_kernel(const typename F::Block* __restrict__ w, const Q81Block* __restrict__ x,
                                       float* __restrict__ y, int n_in, int n_out, int ncols, int ncg) {
    constexpr int BPIW = 32 / F::T;
    const int lane = int(threadIdx.x) & 31;
    const int cg = int(blockIdx.x) % ncg, rbk = int(blockIdx.x) / ncg;   // a row block's column groups adjacent
    const int row0 = (rbk * 4 + (int(threadIdx.x) >> 5)) * R;
    if (row0 >= n_out) return;
    const int c0 = cg * C, nc = min(C, ncols - c0);
    const int bpr = n_in / F::DIV;
    const std::size_t xsb = std::size_t(n_in / Q8K) * sizeof(Q81Block);   // a column's bytes
    const int kqs = F::kqs(lane);
    const typename F::Block* wr[R];
#pragma unroll
    for (int i = 0; i < R; ++i) wr[i] = w + std::size_t(min(row0 + i, n_out - 1)) * bpr;
    const char* xc[C];
#pragma unroll
    for (int j = 0; j < C; ++j) xc[j] = reinterpret_cast<const char*>(x) + std::size_t(c0 + min(j, nc - 1)) * xsb;
    float tmp[C][R] = {};
    for (int kbx = lane / F::T; kbx < bpr; kbx += BPIW) {
        const uint32_t kby = uint32_t(kbx * F::KBY);
        typename F::W wv[R];
#pragma unroll
        for (int i = 0; i < R; ++i) wv[i] = F::load(wr[i] + kbx, kqs);
#pragma unroll
        for (int j = 0; j < C; ++j)
#pragma unroll
            for (int i = 0; i < R; ++i) tmp[j][i] += F::apply(wv[i], xc[j], kby, kqs);
    }
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float v = tmp[j][i];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
            if (lane == (j * R + i) % 32 && j < nc && row0 + i < n_out) y[std::size_t(c0 + j) * n_out + row0 + i] = v;
        }
}
template<typename F, int R, int C>
void xt2_launch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int rb = (n_out + 4 * R - 1) / (4 * R), ncg = (ncols + C - 1) / C;
    native_mmvq_xt2_kernel<F, R, C><<<unsigned(rb * ncg), 128, 0, s>>>(static_cast<const typename F::Block*>(w),
                                                                       static_cast<const Q81Block*>(x), y, n_in, n_out, ncols, ncg);
}
}  // namespace

// STRATA_MMVQ_XT: the column count from which tk_mmvq takes the exact WMMA kernel (unset / 0 = never; the Laguna engine
// sets it).  gfx11 only (other archs: false, the wave32 / generic paths).  Types: Q4_K / Q5_K (KQ8 traits).
namespace {
int xt_min() {
    static const int m = [] { const char* e = std::getenv("STRATA_MMVQ_XT"); return e ? std::atoi(e) : 0; }();
    return m;
}
}  // namespace
bool native_mmvq_xt(int t, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    const int lmin = xt_min();
    if (lmin <= 0 || ncols < lmin || ncols > 64 || !w32_on()) return false;
    const auto s = static_cast<cudaStream_t>(stream);
    if (t == 14 || t == 8) {
        // per-type floor (talus-laguna-wide-bench, gfx1151, Laguna shapes): the dp4a tiles win over the 16-column wave32 calls from
        // 16 columns for Q8_0 (head 100352 rows: n 16 1956 vs 2191 us, n 9 1901 vs 1513) and from 32 for Q6_K (1024 rows:
        // n 16 51 vs 33 us, n 32 58 vs 65); STRATA_MMVQ_XT_Q8 / STRATA_MMVQ_XT_Q6 override
        static const int q8_min = [] { const char* e = std::getenv("STRATA_MMVQ_XT_Q8"); return e ? std::atoi(e) : 16; }();
        static const int q6_min = [] { const char* e = std::getenv("STRATA_MMVQ_XT_Q6"); return e ? std::atoi(e) : 32; }();
        const int tmin = t == 14 ? q6_min : q8_min;
        if (tmin <= 0 || ncols < tmin) return false;
        // STRATA_MMVQ_XT2_CFG="R:C" (rows per wave : columns per pass) for sweeps; default 4:8
        static const int cfg0 = [] {
            const char* e = std::getenv("STRATA_MMVQ_XT2_CFG");
            if (!e) return 0;
            const int r = std::atoi(e);
            const char* c = std::strchr(e, ':');
            return 10 * r + (c ? std::atoi(c + 1) : 8);
        }();
        validate_shape(n_in, 1, t == 14 ? 256 : 32);
#define STRATA_XT2_CASES(F) \
        switch (cfg) { \
        case 28: xt2_launch<F, 2, 8>(w, x, y, n_in, n_out, ncols, s); break; \
        case 44: xt2_launch<F, 4, 4>(w, x, y, n_in, n_out, ncols, s); break; \
        case 216: xt2_launch<F, 2, 16>(w, x, y, n_in, n_out, ncols, s); break; \
        case 116: xt2_launch<F, 1, 16>(w, x, y, n_in, n_out, ncols, s); break; \
        case 88: xt2_launch<F, 8, 8>(w, x, y, n_in, n_out, ncols, s); break; \
        default: xt2_launch<F, 4, 8>(w, x, y, n_in, n_out, ncols, s); break; \
        }
        if (t == 14) { const int cfg = cfg0 ? cfg0 : 48; STRATA_XT2_CASES(XtQ6K) }
        else {   // Q8_0: R 8 for the head (6896 vs 7268 us at n 64), R 4 for short matrices (attn_gate 72 rows: 10 vs 27 us)
            const int cfg = cfg0 ? cfg0 : n_out >= 8192 ? 88 : 48;
            STRATA_XT2_CASES(XtQ80)
        }
#undef STRATA_XT2_CASES
        launch_check();
        return true;
    }
    if (!g_kq8 || (t != 12 && t != 13)) return false;
    validate_shape(n_in, 1, 256);
    const int nct = (ncols + 15) / 16, nrt = (n_out + 15) / 16;
    if (t == 13) native_mmvq_xt_kq8_kernel<true><<<unsigned(nct * nrt), 128, 0, s>>>(w, static_cast<const Q81Block*>(x), y, n_in, n_out, ncols, nct);
    else native_mmvq_xt_kq8_kernel<false><<<unsigned(nct * nrt), 128, 0, s>>>(w, static_cast<const Q81Block*>(x), y, n_in, n_out, ncols, nct);
    launch_check();
    return true;
}

bool native_mmvq_w32(int t, const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!w32_on() || !native_mmvq_w32_supported(t) || ncols < 1 || ncols > 16) return false;
    // (a packed Q8_0 copy is not used here: every column count must take this kernel, or a verify column would not be
    // bitwise the decode step's)
    const auto s = static_cast<cudaStream_t>(stream);
    const W32Cfg r = w32_cfg(t, ncols);
    switch (t) {
    case 2: w32_launch<SmallTraits<Q40Block, 4>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 6: w32_launch<SmallTraits<Q50Block, 4>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 8: w32_launch<SmallTraits<Q80Block, 8>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 20: w32_launch<SmallTraits<IQ4NLBlock, 4>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 11: w32_launch<Q3KTraits>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 12: if (g_kq8) w32_launch<KQ8Traits<false>>(r, w, x, y, n_in, n_out, ncols, s);
             else w32_launch<Q4KTraits>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 13: if (g_kq8) w32_launch<KQ8Traits<true>>(r, w, x, y, n_in, n_out, ncols, s);
             else w32_launch<Q5KTraits>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 14: w32_launch<Q6KTraits>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 23: w32_launch<IQ4XSTraits>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 42: w32_launch<Q20Traits>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 10: w32_launch<IqTraits<10>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 16: w32_launch<IqTraits<16>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 17: w32_launch<IqTraits<17>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 18: w32_launch<IqTraits<18>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 21: w32_launch<IqTraits<21>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 22: w32_launch<IqTraits<22>>(r, w, x, y, n_in, n_out, ncols, s); break;
    case 29: w32_launch<IqTraits<29>>(r, w, x, y, n_in, n_out, ncols, s); break;
    default: return false;
    }
    launch_check();
    return true;
}

} // namespace strata::kernels
