#include "strata/kernels/glm_spike_kernels.hpp"
#include "strata/hip_compat/intrinsics.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/gfx_arch.hpp"

#define GGML_COMMON_DECL_HIP
#define GGML_COMMON_IMPL_HIP
#include "../../../third_party/ggml/ggml-common.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <type_traits>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>

namespace talus {
namespace glm_spike {

namespace {

// ============================================================================
// 0. Fast Q8_1 Input Quantization (1 wave32 per 32-element block)
// ============================================================================

__global__ __launch_bounds__(256)
void glm_quantize_q8_1_kernel(
    const float* __restrict__ x_f32,
    block_q8_1*  __restrict__ out_q8,
    int64_t                   total_blocks)
{
    const int64_t blk = static_cast<int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (blk >= total_blocks) return;

    const int lane = threadIdx.x & 31;
    const float xi = x_f32[blk * 32 + lane];
    float amax = fabsf(xi);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    }
    const float d  = amax / 127.0f;
    const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
    const int8_t q = static_cast<int8_t>(__float2int_rn(xi * id));
    int sumi = static_cast<int>(q);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sumi += __shfl_xor(sumi, off, 32);
    }
    out_q8[blk].qs[lane] = q;
    if (lane == 0) {
        out_q8[blk].ds = __halves2half2(__float2half(d), __float2half(d * static_cast<float>(sumi)));
    }
}

// ============================================================================
// 1. IQ2_S (22) & IQ3_S (21) MoE Expert-ID Mat-Vec (glm_mmvq_ids)
//    - Adapted from WP-1.6 mimo_kernels.cu (4 rows/warp, 4 warps/WG = 16 rows/WG)
//    - Uses __builtin_amdgcn_sudot4 (STRATA_DP4A), RegQ8, and 4-way wave-ballot
//      duplicate-expert coalescing across all active (slot, token) pairs.
// ============================================================================

__device__ __forceinline__ int talus_signed_bytes(uint32_t g, uint32_t sbits) {
    const uint32_t one = ((sbits + 0x7f7f7f7fu) & 0x80808080u) >> 7;
    return static_cast<int>((g ^ (one * 0xffu)) + one);
}

__device__ __forceinline__ int load_i32_align2(const void* p) {
    int v;
    __builtin_memcpy(&v, p, sizeof(int));
    return v;
}

__device__ __forceinline__ int2 load_i64_align2(const void* p) {
    int2 v;
    __builtin_memcpy(&v, p, sizeof(int2));
    return v;
}

template <int ROWS>
__device__ __forceinline__ float glm_reduce_rows(float (&acc)[ROWS], int lane) {
    static_assert(ROWS == 4, "glm_reduce_rows requires ROWS == 4");
    const float send01 = (lane & 1) ? acc[0] : acc[1];
    const float keep01 = (lane & 1) ? acc[1] : acc[0];
    const float s01    = keep01 + __shfl_xor(send01, 1, 32);

    const float send23 = (lane & 1) ? acc[2] : acc[3];
    const float keep23 = (lane & 1) ? acc[3] : acc[2];
    const float s23    = keep23 + __shfl_xor(send23, 1, 32);

    const float send0123 = (lane & 2) ? s01 : s23;
    const float keep0123 = (lane & 2) ? s23 : s01;
    float v = keep0123 + __shfl_xor(send0123, 2, 32);

    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 16, 32);
    return v;
}

template <int TY>
struct GlmIqSplit;

template <>
struct GlmIqSplit<22> { // IQ2_S
    using blk    = block_iq2_s;
    using grid_t = uint2;
    static constexpr int QK       = 256;
    static constexpr int GRID_LEN = 1024;

    struct Raw {
        int     qs_packed;
        int     signs_packed_32;
        uint8_t qh;
        uint8_t sc;
        half    d;
    };

    struct W {
        int   g[8];
        int   ls0;
        int   ls1;
        float dw;
    };

    __device__ __forceinline__ static const grid_t* global_grid() {
        return reinterpret_cast<const grid_t*>(iq2s_grid);
    }

    __device__ __forceinline__ static Raw load_raw(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub)
    {
        const blk* bq2 = bq2_row + kbx;
        return {
            load_i32_align2(bq2->qs + 4 * sub),
            load_i32_align2(bq2->qs + 32 + 4 * sub),
            bq2->qh[sub],
            bq2->scales[sub],
            bq2->d
        };
    }

    __device__ __forceinline__ static W unpack_w(
        const Raw&                 raw,
        const grid_t* __restrict__ grid_lut)
    {
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&raw.qs_packed);
        const int qh = raw.qh;
        const uint8_t* signs_packed_8 = reinterpret_cast<const uint8_t*>(&raw.signs_packed_32);
        W r;
        r.ls0 = raw.sc & 0x0F;
        r.ls1 = raw.sc >> 4;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid_lut[qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)];
            const uint32_t sp = signs_packed_8[l0 / 2];
            r.g[l0 + 0] = talus_signed_bytes(grid_pos.x, ((sp & 0x03u) << 7) | ((sp & 0x0Cu) << 21));
            r.g[l0 + 1] = talus_signed_bytes(grid_pos.y, ((sp & 0x30u) << 3) | ((sp & 0xC0u) << 17));
        }
        r.dw = __half2float(raw.d);
        return r;
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = STRATA_DP4A(r.g[j], u[j], sumi0);
        #pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = STRATA_DP4A(r.g[j], u[j], sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        return (r.dw * d8) * static_cast<float>(sumi);
    }
};

template <>
struct GlmIqSplit<21> { // IQ3_S
    using blk    = block_iq3_s;
    using grid_t = uint32_t;
    static constexpr int QK       = 256;
    static constexpr int GRID_LEN = 512;

    struct Raw {
        int2    qs_packed;
        int     signs_packed_32;
        uint8_t qh;
        uint8_t sc_byte;
        uint8_t sub_idx;
        half    d;
    };

    struct W {
        int   g[8];
        int   ls;
        float dw;
    };

    __device__ __forceinline__ static const grid_t* global_grid() {
        return iq3s_grid;
    }

    __device__ __forceinline__ static Raw load_raw(
        const blk* __restrict__ bq3_row,
        int                     kbx,
        int                     sub)
    {
        const blk* bq3 = bq3_row + kbx;
        return {
            load_i64_align2(bq3->qs + 8 * sub),
            load_i32_align2(bq3->signs + 4 * sub),
            bq3->qh[sub],
            bq3->scales[sub / 2],
            static_cast<uint8_t>(sub),
            bq3->d
        };
    }

    __device__ __forceinline__ static W unpack_w(
        const Raw&                 raw,
        const grid_t* __restrict__ grid_lut)
    {
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&raw.qs_packed);
        const int qh = raw.qh;
        const uint8_t* signs_packed_8 = reinterpret_cast<const uint8_t*>(&raw.signs_packed_32);
        W r;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t gp0 = grid_lut[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
            const uint32_t gp1 = grid_lut[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
            const uint32_t sp = signs_packed_8[l0 / 2];
            r.g[l0 + 0] = talus_signed_bytes(gp0, ((sp & 0x03u) << 7) | ((sp & 0x0Cu) << 21));
            r.g[l0 + 1] = talus_signed_bytes(gp1, ((sp & 0x30u) << 3) | ((sp & 0xC0u) << 17));
        }
        r.ls = 1 + 2 * ((raw.sc_byte >> ((raw.sub_idx & 1) * 4)) & 0x0F);
        r.dw = __half2float(raw.d);
        return r;
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sumi = STRATA_DP4A(r.g[j], u[j], sumi);
        }
        return (r.dw * d8) * static_cast<float>(sumi * r.ls);
    }
};

struct RegQ8 {
    int   u[8];
    float d8;

    __device__ __forceinline__ void load(const block_q8_1* __restrict__ bq8, int sub) {
        const int* q32 = reinterpret_cast<const int*>(bq8[sub].qs);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            u[j] = q32[j];
        }
        d8 = __low2float(bq8[sub].ds);
    }
};

__device__ __forceinline__ int glm_read_id(const int32_t* __restrict__ ids, int flat_k, int n_expert_used, int ids_s1) {
    if (ids_s1 == n_expert_used) return ids[flat_k];
    const int tok  = flat_k / n_expert_used;
    const int slot = flat_k % n_expert_used;
    return ids[tok * ids_s1 + slot];
}

__device__ __forceinline__ int glm_slot_to_xcol(int flat_k, int n_expert_used, int x_group) {
    if (x_group <= 1) {
        return flat_k / n_expert_used;
    }
    if (x_group == n_expert_used) {
        return flat_k;
    }
    const int tok  = flat_k / n_expert_used;
    const int slot = flat_k % n_expert_used;
    return tok * x_group + (slot % x_group);
}

// 1-vector kernel (used for T=1 where top_k experts of a single token are distinct):
// stages the IQ2_S (8 KB) or IQ3_S (2 KB) lookup table in LDS (__shared__) to avoid
// L0 vector cache thrashing and vmcnt stalls, and keeps VGPR pressure low (~68 VGPRs).
template <int TY, int WARPS = 4, int ROWS = 4>
__global__ __launch_bounds__(WARPS * 32, 4)
void glm_mmvq_ids_1vec_kernel(
    const char*    __restrict__ weights,
    size_t                      expert_bytes,
    const int32_t* __restrict__ ids,
    int                         n_ids,
    int                         n_expert_used,
    int                         ids_s1,
    const char*    __restrict__ x_q8_1,
    float*         __restrict__ y,
    int                         n_in,
    int                         n_out,
    int                         x_group)
{
    using S   = GlmIqSplit<TY>;
    using blk = typename S::blk;
    using Raw = typename S::Raw;
    using W   = typename S::W;

    const int k = static_cast<int>(blockIdx.y);
    if (k >= n_ids) return;
    const int e = glm_read_id(ids, k, n_expert_used, ids_s1);
    if (e < 0) return;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const typename S::grid_t* g_grid = S::global_grid();

    const int row0 = (static_cast<int>(blockIdx.x) * WARPS + warp) * ROWS;
    if (row0 >= n_out) return;

    const int bpr   = n_in / S::QK;
    const int q8_pr = n_in / QK8_1;
    const int grp   = lane >> 3;
    const int sub   = lane & 7;

    const blk* w_exp = reinterpret_cast<const blk*>(weights + static_cast<size_t>(e) * expert_bytes);
    const blk* row_b[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = (row0 + r < n_out) ? (row0 + r) : (n_out - 1);
        row_b[r] = w_exp + static_cast<size_t>(row) * bpr;
    }

    const auto* x_all = reinterpret_cast<const block_q8_1*>(x_q8_1);
    const int x_col0  = glm_slot_to_xcol(k, n_expert_used, x_group);
    const block_q8_1* x0 = x_all + static_cast<size_t>(x_col0) * q8_pr;
    float acc0[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kbx = grp; kbx < bpr; kbx += 4) {
        Raw raw[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            raw[r] = S::load_raw(row_b[r], kbx, sub);
        }
        RegQ8 rq8;
        rq8.load(x0 + kbx * 8, sub);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const W wv = S::unpack_w(raw[r], g_grid);
            acc0[r] += S::dot(wv, rq8.u, rq8.d8);
        }
    }

    const float sum0 = glm_reduce_rows<ROWS>(acc0, lane);
    if (lane < ROWS && row0 + lane < n_out) {
        y[static_cast<size_t>(k) * n_out + row0 + lane] = sum0;
    }
}

// Talus GLM port: grouped expert mat-vec for prefill. Rows of x (q8_1, one row per routed (token, slot)) are sorted by
// expert (bounds[e] .. bounds[e+1]); every workgroup takes one expert and 16 weight rows, unpacks each weight sub-block
// once per pass of NT routed rows and dots it against all of them; y[ids_dst[r] * ld_y + row].
template <int TY, int NT, int WARPS = 4, int ROWS = 4>
__global__ __launch_bounds__(WARPS * 32)
void glm_mmvq_grouped_kernel(
    const char*    __restrict__ weights,
    size_t                      expert_bytes,
    const int32_t* __restrict__ bounds,
    const char*    __restrict__ x_q8_1,
    const int32_t* __restrict__ ids_dst,
    float*         __restrict__ y,
    int                         n_in,
    int                         n_out,
    int                         ld_y,
    int                         rows_per_wg)
{
    using S   = GlmIqSplit<TY>;
    using blk = typename S::blk;
    using W   = typename S::W;

    const int e  = static_cast<int>(blockIdx.y);
    const int r0 = bounds[e], r1 = bounds[e + 1];
    if (r0 >= r1) return;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ typename S::grid_t sgrid[S::GRID_LEN];
    for (int i = tid; i < S::GRID_LEN; i += WARPS * 32) sgrid[i] = S::global_grid()[i];
    __syncthreads();
    const typename S::grid_t* g_grid = sgrid;

    const int bpr   = n_in / S::QK;
    const int q8_pr = n_in / QK8_1;
    const int grp   = lane >> 3;
    const int sub   = lane & 7;
    const blk* w_exp = reinterpret_cast<const blk*>(weights + static_cast<size_t>(e) * expert_bytes);
    const auto* x_all = reinterpret_cast<const block_q8_1*>(x_q8_1);
    const int wg_row0 = static_cast<int>(blockIdx.x) * rows_per_wg;
    const int wg_row1 = min(n_out, wg_row0 + rows_per_wg);

    for (int row0 = wg_row0 + warp * ROWS; row0 < wg_row1; row0 += WARPS * ROWS) {
        const blk* row_b[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = (row0 + r < n_out) ? (row0 + r) : (n_out - 1);
            row_b[r] = w_exp + static_cast<size_t>(row) * bpr;
        }
        for (int t0 = r0; t0 < r1; t0 += NT) {
            const int nt = min(NT, r1 - t0);
            float acc[NT][ROWS];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;
            for (int kbx = grp; kbx < bpr; kbx += 4) {
                RegQ8 q[NT];
                #pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const int tt = t < nt ? t : 0;
                    q[t].load(x_all + static_cast<size_t>(t0 + tt) * q8_pr + kbx * 8, sub);
                }
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    const W wv = S::unpack_w(S::load_raw(row_b[r], kbx, sub), g_grid);
                    #pragma unroll
                    for (int t = 0; t < NT; ++t) acc[t][r] += S::dot(wv, q[t].u, q[t].d8);
                }
            }
            #pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float sum = glm_reduce_rows<ROWS>(acc[t], lane);
                if (t < nt && lane < ROWS && row0 + lane < n_out) {
                    y[static_cast<size_t>(ids_dst[t0 + t]) * ld_y + row0 + lane] = sum;
                }
            }
        }
    }
}

// v2: the routed rows' q8_1 blocks of the 4 weight blocks in flight are staged in LDS once per workgroup (all warps
// work on the same expert and tokens), so a pass covers NT = 16 rows with each weight sub-block unpacked once.
template <int TY, int NT, int WARPS, int ROWS>
__global__ __launch_bounds__(WARPS * 32)
void glm_mmvq_grouped_v2_kernel(
    const char*    __restrict__ weights,
    size_t                      expert_bytes,
    const int32_t* __restrict__ bounds,
    const char*    __restrict__ x_q8_1,
    const int32_t* __restrict__ ids_dst,
    float*         __restrict__ y,
    int                         n_in,
    int                         n_out,
    int                         ld_y,
    int                         rows_per_wg)
{
    using S   = GlmIqSplit<TY>;
    using blk = typename S::blk;
    using W   = typename S::W;

    const int e  = static_cast<int>(blockIdx.y);
    const int r0 = bounds[e], r1 = bounds[e + 1];
    if (r0 >= r1) return;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ typename S::grid_t sgrid[S::GRID_LEN];
    // [NT tokens][4 lane groups][8 subs] q8_1 blocks
    __shared__ block_q8_1 sx[NT][4][8];
    for (int i = tid; i < S::GRID_LEN; i += WARPS * 32) sgrid[i] = S::global_grid()[i];

    const int bpr   = n_in / S::QK;
    const int q8_pr = n_in / QK8_1;
    const int grp   = lane >> 3;
    const int sub   = lane & 7;
    const blk* w_exp = reinterpret_cast<const blk*>(weights + static_cast<size_t>(e) * expert_bytes);
    const auto* x_all = reinterpret_cast<const block_q8_1*>(x_q8_1);
    const int wg_row0 = static_cast<int>(blockIdx.x) * rows_per_wg;
    const int wg_row1 = min(n_out, wg_row0 + rows_per_wg);

    for (int rb = wg_row0; rb < wg_row1; rb += WARPS * ROWS) {
        const int row0 = rb + warp * ROWS;
        const blk* row_b[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int row = (row0 + r < n_out) ? (row0 + r) : (n_out - 1);
            row_b[r] = w_exp + static_cast<size_t>(row) * bpr;
        }
        for (int t0 = r0; t0 < r1; t0 += NT) {
            const int nt = min(NT, r1 - t0);
            float acc[NT][ROWS];
            #pragma unroll
            for (int t = 0; t < NT; ++t)
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) acc[t][r] = 0.0f;
            for (int kb0 = 0; kb0 < bpr; kb0 += 4) {
                // stage x: NT x 4 blocks x 8 subs of 36 bytes (9 dwords each)
                __syncthreads();
                {
                    const int total = NT * 4 * 8 * 9;
                    for (int i = tid; i < total; i += WARPS * 32) {
                        const int dw = i % 9, sb = (i / 9) % 8, g = (i / 72) % 4, t = i / 288;
                        int v = 0;
                        if (t < nt && kb0 + g < bpr) {
                            const int* src = reinterpret_cast<const int*>(x_all + static_cast<size_t>(t0 + t) * q8_pr + (kb0 + g) * 8 + sb);
                            v = src[dw];
                        }
                        reinterpret_cast<int*>(&sx[t][g][sb])[dw] = v;
                    }
                }
                __syncthreads();
                const int kbx = kb0 + grp;
                if (row0 < wg_row1 && kbx < bpr) {
                    W wv[ROWS];
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) wv[r] = S::unpack_w(S::load_raw(row_b[r], kbx, sub), sgrid);
                    #pragma unroll
                    for (int t = 0; t < NT; ++t) {
                        if (t < nt) {
                            RegQ8 q;
                            q.load(&sx[t][grp][0], sub);
                            #pragma unroll
                            for (int r = 0; r < ROWS; ++r) acc[t][r] += S::dot(wv[r], q.u, q.d8);
                        }
                    }
                }
            }
            if (row0 < wg_row1) {
                #pragma unroll
                for (int t = 0; t < NT; ++t) {
                    const float sum = glm_reduce_rows<ROWS>(acc[t], lane);
                    if (t < nt && lane < ROWS && row0 + lane < n_out) {
                        y[static_cast<size_t>(ids_dst[t0 + t]) * ld_y + row0 + lane] = sum;
                    }
                }
            }
        }
    }
}

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define GLM_WMMA_I8 1
#endif
typedef int glm_v4i __attribute__((ext_vector_type(4)));
typedef int glm_v8i __attribute__((ext_vector_type(8)));

__device__ __forceinline__ glm_v8i glm_wmma_i8(glm_v4i a, glm_v4i b) {
#if defined(GLM_WMMA_I8)
    glm_v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
#else
    (void) a; (void) b;
    return glm_v8i{0, 0, 0, 0, 0, 0, 0, 0};
#endif
}

__device__ __forceinline__ int glm_swap16(int v) {   // value of lane ^ 16
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
}

// Grouped expert product on the gfx11 matrix cores (v_wmma_i32_16x16x16_iu8): a wave owns 16 weight rows x 16 routed
// rows of one expert. Per pair of 32-value sub-blocks, lane half `hi` unpacks sub-block sb + hi of row l16 (the codebook
// unpack once per weight), the halves exchange through permlanex16 so every 16-value chunk has its A operand in both
// halves, B = the routed rows' q8_1 values; per chunk C (int32, rows 2 i + hi, column l16) is scaled by the row's
// sub-block scale (LDS) and the token's q8_1 scale.
template <int TY>
__global__ __launch_bounds__(256)
void glm_grouped_wmma_kernel(
    const char*    __restrict__ weights,
    size_t                      expert_bytes,
    const int32_t* __restrict__ bounds,
    const char*    __restrict__ x_q8_1,
    const int32_t* __restrict__ ids_dst,
    float*         __restrict__ y,
    int                         n_in,
    int                         n_out,
    int                         ld_y)
{
    using S   = GlmIqSplit<TY>;
    using blk = typename S::blk;
    using W   = typename S::W;
    constexpr int WAVES = 8;

    const int e  = static_cast<int>(blockIdx.y);
    const int r0 = bounds[e], r1 = bounds[e + 1];
    if (r0 >= r1) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    __shared__ typename S::grid_t sgrid[S::GRID_LEN];
    __shared__ float sscale[WAVES][16][4];   // [row][chunk] of the current sub-block pair
    for (int i = tid; i < S::GRID_LEN; i += WAVES * 32) sgrid[i] = S::global_grid()[i];
    __syncthreads();

    const int row0 = static_cast<int>(blockIdx.x) * (WAVES * 16) + w * 16;
    if (row0 >= n_out) return;
    const int my_row = min(row0 + l16, n_out - 1);
    const int bpr = n_in / S::QK;
    const int nsb = n_in / 32;
    const int q8_pr = n_in / QK8_1;
    const blk* wrow = reinterpret_cast<const blk*>(weights + static_cast<size_t>(e) * expert_bytes) +
                      static_cast<size_t>(my_row) * bpr;
    const auto* x_all = reinterpret_cast<const block_q8_1*>(x_q8_1);
    (void) bpr;

    for (int tt0 = r0; tt0 < r1; tt0 += 16) {
        const int ntok = min(16, r1 - tt0);
        const int tr = tt0 + min(l16, ntok - 1);   // B column: this lane's routed row (clamped)
        const block_q8_1* xr = x_all + static_cast<size_t>(tr) * q8_pr;
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
        for (int sb = 0; sb < nsb; sb += 2) {
            const int msb = sb + hi;
            const W wv = S::unpack_w(S::load_raw(wrow, msb >> 3, msb & 7), sgrid);
            // this lane's row scales for the two 16-value halves of sub-block msb
            float s0, s1;
            if constexpr (TY == 22) {
                s0 = wv.dw * (static_cast<float>(wv.ls0) + 0.5f) * 0.25f;
                s1 = wv.dw * (static_cast<float>(wv.ls1) + 0.5f) * 0.25f;
            } else {
                s0 = s1 = wv.dw * static_cast<float>(wv.ls);
            }
            sscale[w][l16][2 * hi + 0] = s0;
            sscale[w][l16][2 * hi + 1] = s1;
            int gs[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) gs[j] = glm_swap16(wv.g[j]);
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const bool own = (c >> 1) == hi;
                const int o = (c & 1) * 4;
                glm_v4i A;
                A.x = own ? wv.g[o + 0] : gs[o + 0];
                A.y = own ? wv.g[o + 1] : gs[o + 1];
                A.z = own ? wv.g[o + 2] : gs[o + 2];
                A.w = own ? wv.g[o + 3] : gs[o + 3];
                const block_q8_1& xb = xr[sb + (c >> 1)];
                const int* q32 = reinterpret_cast<const int*>(xb.qs) + (c & 1) * 4;
                glm_v4i B;
                B.x = q32[0]; B.y = q32[1]; B.z = q32[2]; B.w = q32[3];
                const float d8 = __low2float(xb.ds);
                const glm_v8i C = glm_wmma_i8(A, B);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[i] += static_cast<float>(C[i]) * (sscale[w][2 * i + hi][c] * d8);
                }
            }
            __builtin_amdgcn_wave_barrier();
        }
        if (l16 < ntok) {
            float* yr = y + static_cast<size_t>(ids_dst[tt0 + l16]) * ld_y;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = row0 + 2 * i + hi;
                if (row < n_out) yr[row] = acc[i];
            }
        }
    }
}

// v4: coalesced unpack into LDS + matrix cores. Per 256-value block step a workgroup stages the 16 routed rows' q8_1
// blocks (shared by its 8 waves) and every wave unpacks its 16 weight rows the decode way (8 lanes per row, coalesced
// raw loads) into LDS int8 [16][256] + sub-block scales, then runs 16 WMMA chunks (A row l16 from LDS, B token l16).
template <int TY>
__global__ __launch_bounds__(256)
void glm_grouped_wmma_v4_kernel(
    const char*    __restrict__ weights,
    size_t                      expert_bytes,
    const int32_t* __restrict__ bounds,
    const char*    __restrict__ x_q8_1,
    const int32_t* __restrict__ ids_dst,
    float*         __restrict__ y,
    int                         n_in,
    int                         n_out,
    int                         ld_y)
{
    using S   = GlmIqSplit<TY>;
    using blk = typename S::blk;
    using W   = typename S::W;
    constexpr int WAVES = 8;

    const int e  = static_cast<int>(blockIdx.y);
    const int r0 = bounds[e], r1 = bounds[e + 1];
    if (r0 >= r1) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int grp = lane >> 3, sub = lane & 7;
    __shared__ typename S::grid_t sgrid[S::GRID_LEN];
    __shared__ __attribute__((aligned(16))) int8_t swq[WAVES][16][256];
    __shared__ float ssc[WAVES][16][16];             // [row][16-value chunk]
    __shared__ __attribute__((aligned(16))) int8_t sxq[16][256];
    __shared__ float sxd[16][8];                     // [token][sub-block] q8_1 scale
    for (int i = tid; i < S::GRID_LEN; i += WAVES * 32) sgrid[i] = S::global_grid()[i];

    const int row0 = static_cast<int>(blockIdx.x) * (WAVES * 16) + w * 16;
    const bool wave_ok = row0 < n_out;
    const int bpr = n_in / S::QK;
    const int q8_pr = n_in / QK8_1;
    const blk* w_exp = reinterpret_cast<const blk*>(weights + static_cast<size_t>(e) * expert_bytes);
    const auto* x_all = reinterpret_cast<const block_q8_1*>(x_q8_1);

    for (int tt0 = r0; tt0 < r1; tt0 += 16) {
        const int ntok = min(16, r1 - tt0);
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
        for (int kbx = 0; kbx < bpr; ++kbx) {
            __syncthreads();
            // stage x: 16 tokens x 8 sub-blocks (int8 [16][256], scales [16][8])
            for (int i = tid; i < 16 * 8 * 9; i += WAVES * 32) {
                const int dw = i % 9, sb = (i / 9) % 8, t = i / 72;
                const int tr = tt0 + min(t, ntok - 1);
                const int v = reinterpret_cast<const int*>(x_all + static_cast<size_t>(tr) * q8_pr + kbx * 8 + sb)[dw];
                if (dw == 0) {
                    union { int i; __half2 h; } u; u.i = v;
                    sxd[t][sb] = __low2float(u.h);
                } else {
                    reinterpret_cast<int*>(&sxq[t][sb * 32])[dw - 1] = v;
                }
            }
            // unpack: 4 rounds of 4 rows, lane = (row grp, sub-block sub)
            if (wave_ok) {
                #pragma unroll
                for (int rr = 0; rr < 4; ++rr) {
                    const int r = rr * 4 + grp;
                    const int row = min(row0 + r, n_out - 1);
                    const W wv = S::unpack_w(S::load_raw(w_exp + static_cast<size_t>(row) * bpr, kbx, sub), sgrid);
                    int* dst = reinterpret_cast<int*>(&swq[w][r][sub * 32]);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) dst[j] = wv.g[j];
                    if constexpr (TY == 22) {
                        ssc[w][r][2 * sub + 0] = wv.dw * (static_cast<float>(wv.ls0) + 0.5f) * 0.25f;
                        ssc[w][r][2 * sub + 1] = wv.dw * (static_cast<float>(wv.ls1) + 0.5f) * 0.25f;
                    } else {
                        ssc[w][r][2 * sub + 0] = ssc[w][r][2 * sub + 1] = wv.dw * static_cast<float>(wv.ls);
                    }
                }
            }
            __syncthreads();
            if (wave_ok) {
                #pragma unroll 4
                for (int c = 0; c < 16; ++c) {
                    const glm_v4i A = *reinterpret_cast<const glm_v4i*>(&swq[w][l16][c * 16]);
                    const glm_v4i B = *reinterpret_cast<const glm_v4i*>(&sxq[l16][c * 16]);
                    const glm_v8i C = glm_wmma_i8(A, B);
                    const float d8 = sxd[l16][c >> 1];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) acc[i] += static_cast<float>(C[i]) * (ssc[w][2 * i + hi][c] * d8);
                }
            }
        }
        if (wave_ok && l16 < ntok) {
            float* yr = y + static_cast<size_t>(ids_dst[tt0 + l16]) * ld_y;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = row0 + 2 * i + hi;
                if (row < n_out) yr[row] = acc[i];
            }
        }
    }
}

template <int TY, int WARPS = 4, int ROWS = 4>
__global__ __launch_bounds__(WARPS * 32, 3)
void glm_mmvq_ids_kernel(
    const char*    __restrict__ weights,
    size_t                      expert_bytes,
    const int32_t* __restrict__ ids,
    int                         n_ids,
    int                         n_expert_used,
    int                         ids_s1,
    const char*    __restrict__ x_q8_1,
    float*         __restrict__ y,
    int                         n_in,
    int                         n_out,
    int                         x_group)
{
    using S   = GlmIqSplit<TY>;
    using blk = typename S::blk;
    using Raw = typename S::Raw;
    using W   = typename S::W;

    const int k = static_cast<int>(blockIdx.y);
    if (k >= n_ids) return;
    const int e = glm_read_id(ids, k, n_expert_used, ids_s1);
    if (e < 0) return;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // 4-way wave-ballot duplicate-expert coalescing across ids[0 .. n_ids - 1] (n_ids <= 64).
    // Up to 4 occurrences of the same expert across tokens share a single weight load + unpack pass.
    int s0 = k, s1 = -1, s2 = -1, s3 = -1;
    if (n_ids > 1 && n_ids <= 64) {
        const int id0 = (lane < n_ids) ? glm_read_id(ids, lane, n_expert_used, ids_s1) : -2;
        const int id1 = (lane + 32 < n_ids) ? glm_read_id(ids, lane + 32, n_expert_used, ids_s1) : -2;
        const uint32_t m0 = static_cast<uint32_t>(__ballot(id0 == e));
        const uint32_t m1 = static_cast<uint32_t>(__ballot(id1 == e));
        const uint64_t mask = static_cast<uint64_t>(m0) | (static_cast<uint64_t>(m1) << 32);
        const uint64_t below_mask = (k == 0) ? 0ULL : ((1ULL << k) - 1ULL);
        const int rank = __popcll(mask & below_mask);
        if ((rank & 3) != 0) return;
        uint64_t rem = mask & (~((below_mask << 1) | 1ULL));
        if (rem != 0ULL) {
            s1 = __ffsll(static_cast<unsigned long long>(rem)) - 1;
            rem &= (rem - 1ULL);
            if (rem != 0ULL) {
                s2 = __ffsll(static_cast<unsigned long long>(rem)) - 1;
                rem &= (rem - 1ULL);
                if (rem != 0ULL) {
                    s3 = __ffsll(static_cast<unsigned long long>(rem)) - 1;
                }
            }
        }
    }

    const typename S::grid_t* g_grid = S::global_grid();

    const int row0 = (static_cast<int>(blockIdx.x) * WARPS + warp) * ROWS;
    if (row0 >= n_out) return;

    const int bpr   = n_in / S::QK;
    const int q8_pr = n_in / QK8_1;
    const int grp   = lane >> 3;
    const int sub   = lane & 7;

    const blk* w_exp = reinterpret_cast<const blk*>(weights + static_cast<size_t>(e) * expert_bytes);
    const blk* row_b[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = (row0 + r < n_out) ? (row0 + r) : (n_out - 1);
        row_b[r] = w_exp + static_cast<size_t>(row) * bpr;
    }

    const auto* x_all = reinterpret_cast<const block_q8_1*>(x_q8_1);
    const int x_col0  = glm_slot_to_xcol(s0, n_expert_used, x_group);
    const int x_col1  = (s1 >= 0) ? glm_slot_to_xcol(s1, n_expert_used, x_group) : x_col0;

    // Fast 1-vector path
    if (s1 < 0 || (s2 < 0 && x_col0 == x_col1)) {
        const block_q8_1* x0 = x_all + static_cast<size_t>(x_col0) * q8_pr;
        float acc0[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int kbx = grp; kbx < bpr; kbx += 4) {
            Raw raw[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                raw[r] = S::load_raw(row_b[r], kbx, sub);
            }
            RegQ8 rq8;
            rq8.load(x0 + kbx * 8, sub);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const W wv = S::unpack_w(raw[r], g_grid);
                acc0[r] += S::dot(wv, rq8.u, rq8.d8);
            }
        }

        const float sum0 = glm_reduce_rows<ROWS>(acc0, lane);
        if (lane < ROWS && row0 + lane < n_out) {
            y[static_cast<size_t>(s0) * n_out + row0 + lane] = sum0;
            if (s1 >= 0) {
                y[static_cast<size_t>(s1) * n_out + row0 + lane] = sum0;
            }
        }
        return;
    }

    // Fast 2-vector path
    if (s2 < 0) {
        const block_q8_1* x0 = x_all + static_cast<size_t>(x_col0) * q8_pr;
        const block_q8_1* x1 = x_all + static_cast<size_t>(x_col1) * q8_pr;
        float acc0[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc1[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int kbx = grp; kbx < bpr; kbx += 4) {
            Raw raw[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                raw[r] = S::load_raw(row_b[r], kbx, sub);
            }
            RegQ8 rq0, rq1;
            rq0.load(x0 + kbx * 8, sub);
            rq1.load(x1 + kbx * 8, sub);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                const W wv = S::unpack_w(raw[r], g_grid);
                acc0[r] += S::dot(wv, rq0.u, rq0.d8);
                acc1[r] += S::dot(wv, rq1.u, rq1.d8);
            }
        }

        const float sum0 = glm_reduce_rows<ROWS>(acc0, lane);
        const float sum1 = glm_reduce_rows<ROWS>(acc1, lane);
        if (lane < ROWS && row0 + lane < n_out) {
            y[static_cast<size_t>(s0) * n_out + row0 + lane] = sum0;
            y[static_cast<size_t>(s1) * n_out + row0 + lane] = sum1;
        }
        return;
    }

    // 3-or-4-vector coalesced path: unpack one row at a time to keep VGPR count low
    const int x_col2 = glm_slot_to_xcol(s2, n_expert_used, x_group);
    const int x_col3 = (s3 >= 0) ? glm_slot_to_xcol(s3, n_expert_used, x_group) : x_col2;
    const block_q8_1* x_ptrs[4] = {
        x_all + static_cast<size_t>(x_col0) * q8_pr,
        x_all + static_cast<size_t>(x_col1) * q8_pr,
        x_all + static_cast<size_t>(x_col2) * q8_pr,
        x_all + static_cast<size_t>(x_col3) * q8_pr
    };
    const int n_vecs = (s3 >= 0) ? 4 : 3;

    float acc0[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc2[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc3[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kbx = grp; kbx < bpr; kbx += 4) {
        Raw raw[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            raw[r] = S::load_raw(row_b[r], kbx, sub);
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const W wv = S::unpack_w(raw[r], g_grid);
            for (int v = 0; v < n_vecs; ++v) {
                RegQ8 rq;
                rq.load(x_ptrs[v] + kbx * 8, sub);
                const float d = S::dot(wv, rq.u, rq.d8);
                if (v == 0) acc0[r] += d;
                else if (v == 1) acc1[r] += d;
                else if (v == 2) acc2[r] += d;
                else acc3[r] += d;
            }
        }
    }

    const float sum0 = glm_reduce_rows<ROWS>(acc0, lane);
    const float sum1 = glm_reduce_rows<ROWS>(acc1, lane);
    const float sum2 = glm_reduce_rows<ROWS>(acc2, lane);
    const float sum3 = (s3 >= 0) ? glm_reduce_rows<ROWS>(acc3, lane) : 0.0f;
    if (lane < ROWS && row0 + lane < n_out) {
        y[static_cast<size_t>(s0) * n_out + row0 + lane] = sum0;
        y[static_cast<size_t>(s1) * n_out + row0 + lane] = sum1;
        y[static_cast<size_t>(s2) * n_out + row0 + lane] = sum2;
        if (s3 >= 0) {
            y[static_cast<size_t>(s3) * n_out + row0 + lane] = sum3;
        }
    }
}

// ============================================================================
// 1b. GLM HIP v1 expert mat-vec (gfx1151 tuned).  Same per-lane dot as v0 (one lane per 32-value sub-block, the
//     dp4a chain, integer scale step and float expression of ggml's vec_dot_iq{2,3}_s_q8_1), with:
//     - the IQ2_S / IQ3_S grid staged in LDS (ds_load instead of divergent global gathers through L0),
//     - signs from a 4-bit nibble with a bit-spread v_mul_u32_u24 + v_perm_b32 byte mask (talus_sign4;
//       no quarter-rate v_mul_lo_u32), integer scales as v_mul_i32_i24,
//     - weight / activation addresses as an SGPR base + 32-bit lane offset (no 64-bit VALU address math),
//     - R rows per wave and NW waves per workgroup from a per-arch table (gfx1151 tuned, generic = v0's 4 x 4),
//     - expert dedup across tokens (wave ballot) with the 1..4 vector bodies compiled per count, x loaded once per
//       sub-block per vector (v0 reloaded it per row in the 3/4-vector path).
//     Within a lane the integer and float operations are v0's in v0's order; only the cross-lane reduction order
//     differs (R-dependent butterfly), so results match v0 to float rounding of the final sum.
// ============================================================================

// Signed grid word from a 4-bit sign nibble (bit i negates byte i), 4 VALU per word with the negated word at hand:
//   t   = nib * 0x810204         (v_mul_u32_u24: bit i -> bit 8i+2, the four copies do not overlap)
//   sel = (t & 0x04040404) | 0x03020100   (v_and_or_b32, constants in SGPRs: byte i selects byte i of g or of ng)
//   w   = v_perm_b32(ng, g, sel)
// ng = the bytewise negation of g (grid bytes are 1..127, so (g ^ ~0) + 0x01010101 never carries between bytes).
// Same bytes as v0's talus_signed_bytes (exactness: the bench's v1-v0 check).
__device__ __forceinline__ int glm_signed_word(uint32_t g, uint32_t ng, uint32_t nib) {
    const uint32_t t = (nib & 0x0fu) * 0x00810204u;
    uint32_t sel;
#if defined(__HIP_DEVICE_COMPILE__)
    asm("v_and_or_b32 %0, %1, %2, %3" : "=v"(sel) : "v"(t), "s"(0x04040404u), "s"(0x03020100u));
#else
    sel = (t & 0x04040404u) | 0x03020100u;
#endif
    return static_cast<int>(__builtin_amdgcn_perm(ng, g, sel));
}
__device__ __forceinline__ uint32_t glm_neg_bytes(uint32_t g) { return (g ^ 0xffffffffu) + 0x01010101u; }

__device__ __forceinline__ uint32_t glm_ld_u8(const char* __restrict__ base, uint32_t off) {
    return *reinterpret_cast<const uint8_t*>(base + off);
}
__device__ __forceinline__ uint32_t glm_ld_u16(const char* __restrict__ base, uint32_t off) {
    return *reinterpret_cast<const uint16_t*>(base + off);
}
__device__ __forceinline__ uint32_t glm_ld_u32a2(const char* __restrict__ base, uint32_t off) {
    uint32_t v;
    __builtin_memcpy(&v, base + off, 4);
    return v;
}
__device__ __forceinline__ uint2 glm_ld_u64a2(const char* __restrict__ base, uint32_t off) {
    uint2 v;
    __builtin_memcpy(&v, base + off, 8);
    return v;
}

template <int TY> struct GlmV1;

template <> struct GlmV1<22> {   // IQ2_S: 82-byte block {d, qs[32], signs[32], qh[8], scales[8]}
    static constexpr int BLK = 82;
    static constexpr int LDS_WORDS = 2048;   // 1024 x uint64 grid entries (negations computed in registers)
    struct W { int g[8]; int ls0, ls1; float dw; };
    __device__ __forceinline__ static void stage(uint32_t* sg, int tid, int nthr) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(iq2s_grid);
        for (int i = tid; i < LDS_WORDS; i += nthr) sg[i] = src[i];
    }
    __device__ __forceinline__ static W load(const char* __restrict__ wb, uint32_t boff, int sub,
                                            const uint32_t* __restrict__ sg) {
        const uint32_t qs  = glm_ld_u32a2(wb, boff + 2 + 4 * sub);
        const uint32_t sgn = glm_ld_u32a2(wb, boff + 34 + 4 * sub);
        const uint32_t qh  = glm_ld_u8(wb, boff + 66 + sub);
        const uint32_t sc  = glm_ld_u8(wb, boff + 74 + sub);
        const uint32_t dh  = glm_ld_u16(wb, boff);
        W r;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx = ((qs >> (8 * l)) & 0xffu) | ((qh << (8 - 2 * l)) & 0x300u);
            const uint2 gp = *reinterpret_cast<const uint2*>(sg + 2 * idx);
            r.g[2 * l + 0] = glm_signed_word(gp.x, glm_neg_bytes(gp.x), sgn >> (8 * l));
            r.g[2 * l + 1] = glm_signed_word(gp.y, glm_neg_bytes(gp.y), sgn >> (8 * l + 4));
        }
        r.ls0 = sc & 0x0f;
        r.ls1 = sc >> 4;
        r.dw  = __half2float(__ushort_as_half(static_cast<unsigned short>(dh)));
        return r;
    }
    __device__ __forceinline__ static float dot(const W& r, const int (&u)[8], float d8) {
        int s0 = 0, s1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) s0 = STRATA_DP4A(r.g[j], u[j], s0);
#pragma unroll
        for (int j = 4; j < 8; ++j) s1 = STRATA_DP4A(r.g[j], u[j], s1);
        const int sumi = (STRATA_IMUL24(s0, r.ls0) + STRATA_IMUL24(s1, r.ls1) + (s0 + s1) / 2) / 4;
        return (r.dw * d8) * static_cast<float>(sumi);
    }
};

template <> struct GlmV1<21> {   // IQ3_S: 110-byte block {d, qs[64], qh[8], signs[32], scales[4]}
    static constexpr int BLK = 110;
    static constexpr int LDS_WORDS = 1024;   // 512 x {g, -g} pairs
    struct W { int g[8]; int ls; float dw; };
    __device__ __forceinline__ static void stage(uint32_t* sg, int tid, int nthr) {
        for (int i = tid; i < 512; i += nthr) {
            const uint32_t g = iq3s_grid[i];
            *reinterpret_cast<uint2*>(sg + 2 * i) = make_uint2(g, glm_neg_bytes(g));
        }
    }
    __device__ __forceinline__ static W load(const char* __restrict__ wb, uint32_t boff, int sub,
                                            const uint32_t* __restrict__ sg) {
        const uint2    qs  = glm_ld_u64a2(wb, boff + 2 + 8 * sub);
        const uint32_t qh  = glm_ld_u8(wb, boff + 66 + sub);
        const uint32_t sgn = glm_ld_u32a2(wb, boff + 74 + 4 * sub);
        const uint32_t sc  = glm_ld_u8(wb, boff + 106 + (sub >> 1));
        const uint32_t dh  = glm_ld_u16(wb, boff);
        W r;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q   = (l < 2) ? qs.x : qs.y;
            const uint32_t sh  = 16 * (l & 1);
            const uint32_t i0  = ((q >> sh) & 0xffu)       | ((qh << (8 - 2 * l)) & 0x100u);
            const uint32_t i1  = ((q >> (sh + 8)) & 0xffu) | ((qh << (7 - 2 * l)) & 0x100u);
            const uint2 e0 = *reinterpret_cast<const uint2*>(sg + 2 * i0);
            const uint2 e1 = *reinterpret_cast<const uint2*>(sg + 2 * i1);
            r.g[2 * l + 0] = glm_signed_word(e0.x, e0.y, sgn >> (8 * l));
            r.g[2 * l + 1] = glm_signed_word(e1.x, e1.y, sgn >> (8 * l + 4));
        }
        r.ls = 1 + 2 * ((sc >> ((sub & 1) * 4)) & 0x0f);
        r.dw = __half2float(__ushort_as_half(static_cast<unsigned short>(dh)));
        return r;
    }
    __device__ __forceinline__ static float dot(const W& r, const int (&u)[8], float d8) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = STRATA_DP4A(r.g[j], u[j], sumi);
        return (r.dw * d8) * static_cast<float>(STRATA_IMUL24(sumi, r.ls));
    }
};

// Variant 2 (TY + 100): the two v_perm selectors of a sign byte (8 weights) from a 256-entry LDS table (2 KiB) placed
// after the grid: 2 VALU (index) + 1 ds_load_b64 per 8 weights instead of 3 VALU per word.
__device__ __forceinline__ void glm_stage_sel(uint32_t* sl, int tid, int nthr) {
    for (int i = tid; i < 256; i += nthr) {
        uint32_t lo = 0x03020100u, hi = 0x03020100u;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            lo |= ((i >> j) & 1u) << (8 * j + 2);
            hi |= ((i >> (j + 4)) & 1u) << (8 * j + 2);
        }
        *reinterpret_cast<uint2*>(sl + 2 * i) = make_uint2(lo, hi);
    }
}

template <> struct GlmV1<122> : GlmV1<22> {
    static constexpr int LDS_WORDS = 2048 + 512;
    __device__ __forceinline__ static void stage(uint32_t* sg, int tid, int nthr) {
        GlmV1<22>::stage(sg, tid, nthr);
        glm_stage_sel(sg + 2048, tid, nthr);
    }
    __device__ __forceinline__ static W load(const char* __restrict__ wb, uint32_t boff, int sub,
                                            const uint32_t* __restrict__ sg) {
        const uint32_t qs  = glm_ld_u32a2(wb, boff + 2 + 4 * sub);
        const uint32_t sgn = glm_ld_u32a2(wb, boff + 34 + 4 * sub);
        const uint32_t qh  = glm_ld_u8(wb, boff + 66 + sub);
        const uint32_t sc  = glm_ld_u8(wb, boff + 74 + sub);
        const uint32_t dh  = glm_ld_u16(wb, boff);
        W r;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t idx = ((qs >> (8 * l)) & 0xffu) | ((qh << (8 - 2 * l)) & 0x300u);
            const uint2 gp = *reinterpret_cast<const uint2*>(sg + 2 * idx);
            const uint2 se = *reinterpret_cast<const uint2*>(sg + 2048 + 2 * ((sgn >> (8 * l)) & 0xffu));
            r.g[2 * l + 0] = static_cast<int>(__builtin_amdgcn_perm(glm_neg_bytes(gp.x), gp.x, se.x));
            r.g[2 * l + 1] = static_cast<int>(__builtin_amdgcn_perm(glm_neg_bytes(gp.y), gp.y, se.y));
        }
        r.ls0 = sc & 0x0f;
        r.ls1 = sc >> 4;
        r.dw  = __half2float(__ushort_as_half(static_cast<unsigned short>(dh)));
        return r;
    }
};

template <> struct GlmV1<121> : GlmV1<21> {
    static constexpr int LDS_WORDS = 1024 + 512;
    __device__ __forceinline__ static void stage(uint32_t* sg, int tid, int nthr) {
        GlmV1<21>::stage(sg, tid, nthr);
        glm_stage_sel(sg + 1024, tid, nthr);
    }
    __device__ __forceinline__ static W load(const char* __restrict__ wb, uint32_t boff, int sub,
                                            const uint32_t* __restrict__ sg) {
        const uint2    qs  = glm_ld_u64a2(wb, boff + 2 + 8 * sub);
        const uint32_t qh  = glm_ld_u8(wb, boff + 66 + sub);
        const uint32_t sgn = glm_ld_u32a2(wb, boff + 74 + 4 * sub);
        const uint32_t sc  = glm_ld_u8(wb, boff + 106 + (sub >> 1));
        const uint32_t dh  = glm_ld_u16(wb, boff);
        W r;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t q   = (l < 2) ? qs.x : qs.y;
            const uint32_t sh  = 16 * (l & 1);
            const uint32_t i0  = ((q >> sh) & 0xffu)       | ((qh << (8 - 2 * l)) & 0x100u);
            const uint32_t i1  = ((q >> (sh + 8)) & 0xffu) | ((qh << (7 - 2 * l)) & 0x100u);
            const uint2 e0 = *reinterpret_cast<const uint2*>(sg + 2 * i0);
            const uint2 e1 = *reinterpret_cast<const uint2*>(sg + 2 * i1);
            const uint2 se = *reinterpret_cast<const uint2*>(sg + 1024 + 2 * ((sgn >> (8 * l)) & 0xffu));
            r.g[2 * l + 0] = static_cast<int>(__builtin_amdgcn_perm(e0.y, e0.x, se.x));
            r.g[2 * l + 1] = static_cast<int>(__builtin_amdgcn_perm(e1.y, e1.x, se.y));
        }
        r.ls = 1 + 2 * ((sc >> ((sub & 1) * 4)) & 0x0f);
        r.dw = __half2float(__ushort_as_half(static_cast<unsigned short>(dh)));
        return r;
    }
};

struct GlmX8 {
    int   u[8];
    float d8;
    __device__ __forceinline__ void load(const char* __restrict__ xb, uint32_t off) {   // off: block_q8_1 byte offset
        int4 a, b;
        __builtin_memcpy(&a, xb + off + 4, 16);
        __builtin_memcpy(&b, xb + off + 20, 16);
        u[0] = a.x; u[1] = a.y; u[2] = a.z; u[3] = a.w;
        u[4] = b.x; u[5] = b.y; u[6] = b.z; u[7] = b.w;
        d8 = __half2float(__ushort_as_half(static_cast<unsigned short>(glm_ld_u16(xb, off))));
    }
};

// R-row butterfly: after it, lane L holds row glm_rrow<R>(L) summed over all 32 lanes.
template <int R>
__device__ __forceinline__ float glm_reduce_r(float (&v)[R], int lane) {
#pragma unroll
    for (int s = 0; (1 << s) < R; ++s) {
        const int half = R >> (s + 1);
        const bool up = (lane >> s) & 1;
#pragma unroll
        for (int j = 0; j < half; ++j) {
            const float send = up ? v[j] : v[j + half];
            const float keep = up ? v[j + half] : v[j];
            v[j] = keep + __shfl_xor(send, 1 << s, 32);
        }
    }
    float x = v[0];
#pragma unroll
    for (int o = R; o < 32; o <<= 1) x += __shfl_xor(x, o, 32);
    return x;
}
template <int R>
__device__ __forceinline__ int glm_rrow(int lane) {
    int row = 0;
#pragma unroll
    for (int s = 0; (1 << s) < R; ++s) row += ((lane >> s) & 1) * (R >> (s + 1));
    return row;
}

template <int TY, int R, int NV>
__device__ __forceinline__ void glm_v1_body(
    const char* __restrict__ wb, const uint32_t* __restrict__ sg, const char* __restrict__ xb,
    const int (&slot)[4], const uint32_t (&xoff)[4], float* __restrict__ y,
    int row0, int n_out, int bpr, int lane)
{
    using S = GlmV1<TY>;
    const int grp = lane >> 3;
    const int sub = lane & 7;
    uint32_t roff[R];
#pragma unroll
    for (int r = 0; r < R; ++r) roff[r] = static_cast<uint32_t>(min(row0 + r, n_out - 1) * bpr) * S::BLK;
    float acc[NV][R];
#pragma unroll
    for (int v = 0; v < NV; ++v)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[v][r] = 0.0f;

    for (int kbx = grp; kbx < bpr; kbx += 4) {
        typename S::W w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) w[r] = S::load(wb, roff[r] + kbx * S::BLK, sub, sg);
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            GlmX8 x;
            x.load(xb, xoff[v] + static_cast<uint32_t>(kbx * 8 + sub) * 36u);
#pragma unroll
            for (int r = 0; r < R; ++r) acc[v][r] += S::dot(w[r], x.u, x.d8);
        }
    }
    const int rr = glm_rrow<R>(lane);
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        const float s = glm_reduce_r<R>(acc[v], lane);
        if (lane < R && row0 + rr < n_out) y[static_cast<size_t>(slot[v]) * n_out + row0 + rr] = s;
    }
}

// One (expert slot k, row block rb) tile: false when slot k is not the first occurrence of its expert (dedup) or
// unused (id < 0).
template <bool DEDUP>
__device__ __forceinline__ bool glm_v1_tile_active(const int32_t* __restrict__ ids, int k, int n_ids, int n_expert_used,
                                                   int ids_s1, int lane, int& e, uint64_t& rem) {
    e = glm_read_id(ids, k, n_expert_used, ids_s1);
    rem = 0;
    if (e < 0) return false;
    if constexpr (DEDUP) {
        const int id0 = (lane < n_ids) ? glm_read_id(ids, lane, n_expert_used, ids_s1) : -2;
        const int id1 = (lane + 32 < n_ids) ? glm_read_id(ids, lane + 32, n_expert_used, ids_s1) : -2;
        const uint64_t mask = static_cast<uint64_t>(static_cast<uint32_t>(__ballot(id0 == e))) |
                              (static_cast<uint64_t>(static_cast<uint32_t>(__ballot(id1 == e))) << 32);
        const uint64_t below = (k == 0) ? 0ull : ((1ull << k) - 1ull);
        if ((mask & below) != 0ull) return false;
        rem = mask & ~(below | (1ull << k));
    }
    return true;
}

template <int TY, int R, bool DEDUP>
__device__ __forceinline__ void glm_v1_tile(
    const char* __restrict__ weights, size_t expert_bytes, const uint32_t* __restrict__ sg,
    const char* __restrict__ x_q8_1, float* __restrict__ y,
    int k, int e, uint64_t rem, int row0, int n_in, int n_out, int n_expert_used, int x_group, int lane)
{
    if (row0 >= n_out) return;
    const int bpr   = n_in / 256;
    const int q8_pr = n_in / QK8_1;
    const char* wb = weights + static_cast<size_t>(e) * expert_bytes;
    int slot[4] = {k, k, k, k};
    int nv = 1;
    for (;;) {
        if constexpr (DEDUP) {
            while (nv < 4 && rem != 0ull) {
                slot[nv++] = __ffsll(static_cast<unsigned long long>(rem)) - 1;
                rem &= rem - 1ull;
            }
        }
        uint32_t xoff[4];
#pragma unroll
        for (int v = 0; v < 4; ++v)
            xoff[v] = static_cast<uint32_t>(glm_slot_to_xcol(slot[v], n_expert_used, x_group) * q8_pr) * 36u;
        if (nv == 1)      glm_v1_body<TY, R, 1>(wb, sg, x_q8_1, slot, xoff, y, row0, n_out, bpr, lane);
        else if (nv == 2) glm_v1_body<TY, R, 2>(wb, sg, x_q8_1, slot, xoff, y, row0, n_out, bpr, lane);
        else if (nv == 3) glm_v1_body<TY, R, 3>(wb, sg, x_q8_1, slot, xoff, y, row0, n_out, bpr, lane);
        else              glm_v1_body<TY, R, 4>(wb, sg, x_q8_1, slot, xoff, y, row0, n_out, bpr, lane);
        if (!DEDUP || rem == 0ull) break;
        nv = 0;   // more than 4 occurrences: another pass over the same rows (rare: 5 tokens on one expert)
    }
}

// PERS = false: one workgroup per tile (grid = row blocks x slots).  PERS = true: a fixed grid of workgroups stages
// the grid table once and walks tiles rb-fastest (consecutive workgroups on adjacent rows of one expert).
// FQ (persistent, x_group 1, <= 4 token vectors of <= 4096): x arrives as f32 and each workgroup quantizes it into
// LDS with glm_quantize_q8_1_kernel's operations (same q8_1 bytes), so the f32 entry point is one launch.
template <int TY, int R, int NW, int WPE, bool DEDUP, bool PERS, bool FQ = false>
__global__ __launch_bounds__(NW * 32, WPE)
void glm_mmvq_ids_v1_kernel(
    const char*    __restrict__ weights,
    size_t                      expert_bytes,
    const int32_t* __restrict__ ids,
    int                         n_ids,
    int                         n_expert_used,
    int                         ids_s1,
    const char*    __restrict__ x_q8_1,
    float*         __restrict__ y,
    int                         n_in,
    int                         n_out,
    int                         x_group,
    const char*    __restrict__ weights2,   // second matrix of a pair (gate + up, same ids / x), or null
    float*         __restrict__ y2)
{
    using S = GlmV1<TY>;
    __shared__ __attribute__((aligned(16))) uint32_t sg[S::LDS_WORDS];
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int e;
    uint64_t rem;
    if constexpr (!PERS) {
        int k = static_cast<int>(blockIdx.y);
        const bool second = k >= n_ids;          // pair launch: grid.y = 2 n_ids, the upper half runs matrix 2
        if (second) k -= n_ids;
        if (!glm_v1_tile_active<DEDUP>(ids, k, n_ids, n_expert_used, ids_s1, lane, e, rem)) return;
        S::stage(sg, tid, NW * 32);
        __syncthreads();
        const int row0 = (static_cast<int>(blockIdx.x) * NW + warp) * R;
        glm_v1_tile<TY, R, DEDUP>(second ? weights2 : weights, expert_bytes, sg, x_q8_1, second ? y2 : y, k, e, rem,
                                  row0, n_in, n_out, n_expert_used, x_group, lane);
    } else {
        S::stage(sg, tid, NW * 32);
        const char* xsrc = x_q8_1;
        if constexpr (FQ) {
            __shared__ __attribute__((aligned(16))) char xq_sh[4 * 4608];
            const float* xf = reinterpret_cast<const float*>(x_q8_1);
            const int nblk = (n_ids / n_expert_used) * (n_in / 32);
            for (int blk = tid >> 5; blk < nblk; blk += NW) {
                const float xi = xf[blk * 32 + lane];
                float amax = fabsf(xi);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
                const float d  = amax / 127.0f;
                const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
                const int8_t q = static_cast<int8_t>(__float2int_rn(xi * id));
                int sumi = static_cast<int>(q);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) sumi += __shfl_xor(sumi, off, 32);
                xq_sh[blk * 36 + 4 + lane] = static_cast<char>(q);
                if (lane == 0) {
                    const __half2 ds = __halves2half2(__float2half(d), __float2half(d * static_cast<float>(sumi)));
                    *reinterpret_cast<__half2*>(xq_sh + blk * 36) = ds;
                }
            }
            xsrc = xq_sh;
        }
        __syncthreads();
        const int nrb = (n_out + NW * R - 1) / (NW * R);
        const int ntiles = nrb * n_ids * (weights2 ? 2 : 1);
        for (int tile = static_cast<int>(blockIdx.x); tile < ntiles; tile += static_cast<int>(gridDim.x)) {
            int k  = tile / nrb;
            const int rb = tile - k * nrb;
            const bool second = k >= n_ids;
            if (second) k -= n_ids;
            if (!glm_v1_tile_active<DEDUP>(ids, k, n_ids, n_expert_used, ids_s1, lane, e, rem)) continue;
            const int row0 = (rb * NW + warp) * R;
            glm_v1_tile<TY, R, DEDUP>(second ? weights2 : weights, expert_bytes, sg, xsrc, second ? y2 : y, k, e, rem,
                                      row0, n_in, n_out, n_expert_used, x_group, lane);
        }
    }
}

// ============================================================================
// 2. KDA Recurrent Decode Step with Fused Prologue (glm_kda_step)
//    - Direct port of ggml-vulkan/vulkan-shaders/gated_delta_net.comp (S_V=128, KDA=1)
//    - Plus multi-column-per-wave template (COLS_PER_WAVE = 1, 2, 4) that amortizes
//      the KDA prologue (q/k L2 norm, gate sigmoid, beta sigmoid) across columns!
// ============================================================================

__device__ __forceinline__ float wave32_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, 32);
    }
    return v;
}

template <int COLS_PER_WAVE>
__global__ __launch_bounds__(128)
void glm_kda_step_kernel(
    const float*   __restrict__ data_q,
    const float*   __restrict__ data_k,
    const float*   __restrict__ data_v,
    const float*   __restrict__ data_g,
    const float*   __restrict__ data_beta,
    float*         __restrict__ data_state,
    const int32_t* __restrict__ data_ids,
    const float*   __restrict__ data_dtb,
    const float*   __restrict__ data_a,
    float*         __restrict__ data_dst,
    uint32_t H,
    uint32_t S_v,
    uint32_t n_tokens,
    uint32_t K_snap,
    uint32_t n_B,
    float    scale,
    uint32_t prologue,
    float    pro_lb,
    float    pro_l2_eps)
{
    constexpr uint32_t LANES_PER_COL = 32;
    constexpr uint32_t ROWS_PER_LANE = 4; // S_v = 128 = 32 * 4

    const uint32_t head_id  = blockIdx.x;
    const uint32_t col_grp  = blockIdx.y;
    const uint32_t b_idx    = blockIdx.z;

    const uint32_t warp_id  = threadIdx.x >> 5; // 0..3 (4 waves per 128-thread WG)
    const uint32_t lane     = threadIdx.x & 31;
    const uint32_t col_base = (col_grp * 4 + warp_id) * COLS_PER_WAVE;
    if (col_base >= S_v) return;

    const uint32_t sv2 = S_v * S_v;
    const uint32_t n_seq_tokens = n_tokens / n_B;

    const uint32_t q_stride_b    = S_v * H;
    const uint32_t v_stride_b    = S_v * H;
    const uint32_t g_stride_b    = S_v * H;
    const uint32_t beta_stride_b = H;
    const uint32_t state_stride_b = sv2 * H;

    const int32_t slot_in = (K_snap > 0) ? data_ids[b_idx] : static_cast<int32_t>(b_idx);
    const uint32_t state_in_base = static_cast<uint32_t>(slot_in) * state_stride_b + head_id * sv2;

    float s_shard[COLS_PER_WAVE][ROWS_PER_LANE];
    #pragma unroll
    for (int c = 0; c < COLS_PER_WAVE; ++c) {
        const uint32_t col = col_base + c;
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
            const uint32_t i = r * LANES_PER_COL + lane;
            s_shard[c][r] = data_state[state_in_base + col * S_v + i];
        }
    }

    for (uint32_t t = 0; t < n_seq_tokens; ++t) {
        const uint32_t token_idx = b_idx * n_seq_tokens + t;

        const uint32_t q_base    = token_idx * q_stride_b + head_id * S_v;
        const uint32_t k_base    = token_idx * q_stride_b + head_id * S_v;
        const uint32_t v_base    = token_idx * v_stride_b + head_id * S_v;
        const uint32_t g_base    = token_idx * g_stride_b + head_id * S_v;
        const uint32_t beta_base = token_idx * beta_stride_b + head_id;

        float q_reg[ROWS_PER_LANE];
        float k_reg[ROWS_PER_LANE];
        float g_exp[ROWS_PER_LANE];

        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
            const uint32_t i = r * LANES_PER_COL + lane;
            q_reg[r] = data_q[q_base + i];
            k_reg[r] = data_k[k_base + i];
        }

        float beta_val = data_beta[beta_base];

        if (prologue != 0) {
            float qq = 0.0f, kk = 0.0f;
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
                qq = fmaf(q_reg[r], q_reg[r], qq);
                kk = fmaf(k_reg[r], k_reg[r], kk);
            }
            qq = wave32_sum(qq);
            kk = wave32_sum(kk);
            const float n          = static_cast<float>(S_v);
            const float eps        = pro_l2_eps / n;
            const float fq         = rsqrtf(qq / n + eps);
            const float fk         = rsqrtf(kk / n + eps);
            const float inv_sqrt_n = rsqrtf(n);
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
                q_reg[r] = fq * q_reg[r] * inv_sqrt_n;
                k_reg[r] = fk * k_reg[r] * inv_sqrt_n;
            }
            beta_val = 1.0f / (1.0f + expf(-beta_val));
            const float a_h = data_a[head_id];
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
                const uint32_t i = r * LANES_PER_COL + lane;
                const float u  = (data_g[g_base + i] + data_dtb[head_id * S_v + i]) * a_h;
                const float gv = (1.0f / (1.0f + expf(u))) * pro_lb;
                g_exp[r] = expf(gv);
            }
        } else {
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
                const uint32_t i = r * LANES_PER_COL + lane;
                g_exp[r] = expf(data_g[g_base + i]);
            }
        }

        const uint32_t attn_off = (token_idx * H + head_id) * S_v;

        #pragma unroll
        for (int c = 0; c < COLS_PER_WAVE; ++c) {
            const uint32_t col = col_base + c;
            const float v_val = data_v[v_base + col];

            float kv_partial = 0.0f;
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
                s_shard[c][r] *= g_exp[r];
                kv_partial = fmaf(s_shard[c][r], k_reg[r], kv_partial);
            }

            const float kv_col    = wave32_sum(kv_partial);
            const float delta_col = (v_val - kv_col) * beta_val;

            float attn_partial = 0.0f;
            #pragma unroll
            for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
                s_shard[c][r] = fmaf(k_reg[r], delta_col, s_shard[c][r]);
                attn_partial  = fmaf(s_shard[c][r], q_reg[r], attn_partial);
            }

            const float attn_col = wave32_sum(attn_partial);
            if (lane == 0) {
                data_dst[attn_off + col] = attn_col * scale;
            }
        }

        if (K_snap > 0 && t >= n_seq_tokens - K_snap) {
            const uint32_t k_idx = t - (n_seq_tokens - K_snap);
            const int32_t slot_out = data_ids[k_idx * n_B + b_idx];
            if (slot_out >= 0) {
                const uint32_t state_out_base = static_cast<uint32_t>(slot_out) * state_stride_b + head_id * sv2;
                #pragma unroll
                for (int c = 0; c < COLS_PER_WAVE; ++c) {
                    const uint32_t col = col_base + c;
                    #pragma unroll
                    for (uint32_t r = 0; r < ROWS_PER_LANE; ++r) {
                        const uint32_t i = r * LANES_PER_COL + lane;
                        data_state[state_out_base + col * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }
    }
}

// GLM HIP v1 KDA step: a workgroup of NWV waves covers NWV * CPW state columns of one head; per token, wave 0
// computes the prologue once (q / k l2 norm, gate sigmoid + exp, beta sigmoid: v0's operations in v0's order) into
// LDS (double-buffered by token parity, one barrier per token) and every wave reads its rows from there.  v0 redid
// the prologue in each of the 128 / CPW column waves of a head.  Bitwise equal to v0.
// Wave32 sum through DPP + v_permlanex16 instead of five ds_bpermute shuffles.  quad_perm [1,0,3,2] / [2,3,0,1] are
// xor 1 / 2; row_half_mirror then row_mirror pair lanes whose partial sums equal those of the xor-4 / xor-8 partners;
// permlanex16 swaps the two rows (xor 16).  Same partial sums as the xor butterfly; measured within parity (cos 1.0)
// but not bitwise against v0's shuffle version (KDA v1 is therefore 'within tolerance', not bitwise).
template <int CTRL>
__device__ __forceinline__ float glm_dpp(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xf, 0xf, false));
}
__device__ __forceinline__ float glm_wave32_sum_dpp(float v) {
    v += glm_dpp<0xB1>(v);    // quad_perm(1,0,3,2)
    v += glm_dpp<0x4E>(v);    // quad_perm(2,3,0,1)
    v += glm_dpp<0x141>(v);   // row_half_mirror
    v += glm_dpp<0x140>(v);   // row_mirror
    v += __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    return v;
}

template <int CPW, int NWV>
__global__ __launch_bounds__(32 * NWV)
void glm_kda_step_v1_kernel(
    const float*   __restrict__ data_q,
    const float*   __restrict__ data_k,
    const float*   __restrict__ data_v,
    const float*   __restrict__ data_g,
    const float*   __restrict__ data_beta,
    float*         __restrict__ data_state,
    const int32_t* __restrict__ data_ids,
    const float*   __restrict__ data_dtb,
    const float*   __restrict__ data_a,
    float*         __restrict__ data_dst,
    uint32_t H,
    uint32_t n_tokens,
    uint32_t K_snap,
    uint32_t n_B,
    float    scale,
    uint32_t prologue,
    float    pro_lb,
    float    pro_l2_eps)
{
    constexpr uint32_t S_v = 128, ROWS = 4;
    __shared__ float sq[2][S_v], sk[2][S_v], sgx[2][S_v], sbeta[2];

    const uint32_t head_id = blockIdx.x;
    const uint32_t b_idx   = blockIdx.z;
    const uint32_t w       = threadIdx.x >> 5;
    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t col_base = (blockIdx.y * NWV + w) * CPW;
    const uint32_t sv2 = S_v * S_v;
    const uint32_t n_seq_tokens = n_tokens / n_B;
    const uint32_t state_stride_b = sv2 * H;

    const int32_t slot_in = (K_snap > 0) ? data_ids[b_idx] : static_cast<int32_t>(b_idx);
    const uint32_t state_in_base = static_cast<uint32_t>(slot_in) * state_stride_b + head_id * sv2;

    float s_shard[CPW][ROWS];
#pragma unroll
    for (int c = 0; c < CPW; ++c)
#pragma unroll
        for (uint32_t r = 0; r < ROWS; ++r)
            s_shard[c][r] = data_state[state_in_base + (col_base + c) * S_v + r * 32u + lane];

    for (uint32_t t = 0; t < n_seq_tokens; ++t) {
        const uint32_t token_idx = b_idx * n_seq_tokens + t;
        const uint32_t vb   = token_idx * S_v * H + head_id * S_v;
        const uint32_t buf  = t & 1u;
        if (w == 0) {
            float q_reg[ROWS], k_reg[ROWS], g_exp[ROWS];
#pragma unroll
            for (uint32_t r = 0; r < ROWS; ++r) {
                q_reg[r] = data_q[vb + r * 32u + lane];
                k_reg[r] = data_k[vb + r * 32u + lane];
            }
            float beta_val = data_beta[token_idx * H + head_id];
            if (prologue != 0) {
                float qq = 0.0f, kk = 0.0f;
#pragma unroll
                for (uint32_t r = 0; r < ROWS; ++r) {
                    qq = fmaf(q_reg[r], q_reg[r], qq);
                    kk = fmaf(k_reg[r], k_reg[r], kk);
                }
                qq = glm_wave32_sum_dpp(qq);
                kk = glm_wave32_sum_dpp(kk);
                const float n          = static_cast<float>(S_v);
                const float eps        = pro_l2_eps / n;
                const float fq         = rsqrtf(qq / n + eps);
                const float fk         = rsqrtf(kk / n + eps);
                const float inv_sqrt_n = rsqrtf(n);
#pragma unroll
                for (uint32_t r = 0; r < ROWS; ++r) {
                    q_reg[r] = fq * q_reg[r] * inv_sqrt_n;
                    k_reg[r] = fk * k_reg[r] * inv_sqrt_n;
                }
                beta_val = 1.0f / (1.0f + expf(-beta_val));
                const float a_h = data_a[head_id];
#pragma unroll
                for (uint32_t r = 0; r < ROWS; ++r) {
                    const uint32_t i = r * 32u + lane;
                    const float u  = (data_g[vb + i] + data_dtb[head_id * S_v + i]) * a_h;
                    const float gv = (1.0f / (1.0f + expf(u))) * pro_lb;
                    g_exp[r] = expf(gv);
                }
            } else {
#pragma unroll
                for (uint32_t r = 0; r < ROWS; ++r) g_exp[r] = expf(data_g[vb + r * 32u + lane]);
            }
#pragma unroll
            for (uint32_t r = 0; r < ROWS; ++r) {
                sq[buf][r * 32u + lane]  = q_reg[r];
                sk[buf][r * 32u + lane]  = k_reg[r];
                sgx[buf][r * 32u + lane] = g_exp[r];
            }
            if (lane == 0) sbeta[buf] = beta_val;
        }
        __syncthreads();
        float q_reg[ROWS], k_reg[ROWS], g_exp[ROWS];
#pragma unroll
        for (uint32_t r = 0; r < ROWS; ++r) {
            q_reg[r] = sq[buf][r * 32u + lane];
            k_reg[r] = sk[buf][r * 32u + lane];
            g_exp[r] = sgx[buf][r * 32u + lane];
        }
        const float beta_val = sbeta[buf];
        const uint32_t attn_off = (token_idx * H + head_id) * S_v;
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const uint32_t col = col_base + c;
            const float v_val = data_v[vb + col];
            float kv_partial = 0.0f;
#pragma unroll
            for (uint32_t r = 0; r < ROWS; ++r) {
                s_shard[c][r] *= g_exp[r];
                kv_partial = fmaf(s_shard[c][r], k_reg[r], kv_partial);
            }
            const float kv_col    = glm_wave32_sum_dpp(kv_partial);
            const float delta_col = (v_val - kv_col) * beta_val;
            float attn_partial = 0.0f;
#pragma unroll
            for (uint32_t r = 0; r < ROWS; ++r) {
                s_shard[c][r] = fmaf(k_reg[r], delta_col, s_shard[c][r]);
                attn_partial  = fmaf(s_shard[c][r], q_reg[r], attn_partial);
            }
            const float attn_col = glm_wave32_sum_dpp(attn_partial);
            if (lane == 0) data_dst[attn_off + col] = attn_col * scale;
        }
        if (K_snap > 0 && t >= n_seq_tokens - K_snap) {
            const uint32_t k_idx = t - (n_seq_tokens - K_snap);
            const int32_t slot_out = data_ids[k_idx * n_B + b_idx];
            if (slot_out >= 0) {
                const uint32_t ob = static_cast<uint32_t>(slot_out) * state_stride_b + head_id * sv2;
#pragma unroll
                for (int c = 0; c < CPW; ++c)
#pragma unroll
                    for (uint32_t r = 0; r < ROWS; ++r)
                        data_state[ob + (col_base + c) * S_v + r * 32u + lane] = s_shard[c][r];
            }
        }
    }
}

// GLM HIP v2 KDA step: as v1, but lane L owns rows 4L..4L+3 of every column, so a column's state is one 16-byte
// load / store per lane (v1: four 4-byte accesses at stride 32), and q / k / g come as float4.  Same math per element;
// the per-lane partial sums run over 4 consecutive rows instead of 4 strided ones (within tolerance, not bitwise).
template <int CPW, int NWV>
__global__ __launch_bounds__(32 * NWV)
void glm_kda_step_v2_kernel(
    const float*   __restrict__ data_q,
    const float*   __restrict__ data_k,
    const float*   __restrict__ data_v,
    const float*   __restrict__ data_g,
    const float*   __restrict__ data_beta,
    float*         __restrict__ data_state,
    const int32_t* __restrict__ data_ids,
    const float*   __restrict__ data_dtb,
    const float*   __restrict__ data_a,
    float*         __restrict__ data_dst,
    uint32_t H,
    uint32_t n_tokens,
    uint32_t K_snap,
    uint32_t n_B,
    float    scale,
    uint32_t prologue,
    float    pro_lb,
    float    pro_l2_eps)
{
    constexpr uint32_t S_v = 128;
    __shared__ float4 sq[2][32], sk[2][32], sgx[2][32];
    __shared__ float sbeta[2];

    const uint32_t head_id = blockIdx.x;
    const uint32_t b_idx   = blockIdx.z;
    const uint32_t w       = threadIdx.x >> 5;
    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t col_base = (blockIdx.y * NWV + w) * CPW;
    const uint32_t sv2 = S_v * S_v;
    const uint32_t n_seq_tokens = n_tokens / n_B;
    const uint32_t state_stride_b = sv2 * H;

    const int32_t slot_in = (K_snap > 0) ? data_ids[b_idx] : static_cast<int32_t>(b_idx);
    const float4* st_in = reinterpret_cast<const float4*>(data_state + static_cast<size_t>(slot_in) * state_stride_b + head_id * sv2);

    float4 s4[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) s4[c] = st_in[(col_base + c) * 32u + lane];

    for (uint32_t t = 0; t < n_seq_tokens; ++t) {
        const uint32_t token_idx = b_idx * n_seq_tokens + t;
        const uint32_t vb  = token_idx * S_v * H + head_id * S_v;
        const uint32_t buf = t & 1u;
        if (w == 0) {
            float4 q = reinterpret_cast<const float4*>(data_q + vb)[lane];
            float4 k = reinterpret_cast<const float4*>(data_k + vb)[lane];
            float4 g4;
            float beta_val = data_beta[token_idx * H + head_id];
            if (prologue != 0) {
                float qq = 0.0f, kk = 0.0f;
                qq = fmaf(q.x, q.x, qq); qq = fmaf(q.y, q.y, qq); qq = fmaf(q.z, q.z, qq); qq = fmaf(q.w, q.w, qq);
                kk = fmaf(k.x, k.x, kk); kk = fmaf(k.y, k.y, kk); kk = fmaf(k.z, k.z, kk); kk = fmaf(k.w, k.w, kk);
                qq = glm_wave32_sum_dpp(qq);
                kk = glm_wave32_sum_dpp(kk);
                const float n = static_cast<float>(S_v);
                const float eps = pro_l2_eps / n;
                const float fq = rsqrtf(qq / n + eps), fk = rsqrtf(kk / n + eps), isn = rsqrtf(n);
                q = make_float4(fq * q.x * isn, fq * q.y * isn, fq * q.z * isn, fq * q.w * isn);
                k = make_float4(fk * k.x * isn, fk * k.y * isn, fk * k.z * isn, fk * k.w * isn);
                beta_val = 1.0f / (1.0f + expf(-beta_val));
                const float a_h = data_a[head_id];
                const float4 gr = reinterpret_cast<const float4*>(data_g + vb)[lane];
                const float4 db = reinterpret_cast<const float4*>(data_dtb + head_id * S_v)[lane];
                auto gx = [&](float gv, float dv) { return expf((1.0f / (1.0f + expf((gv + dv) * a_h))) * pro_lb); };
                g4 = make_float4(gx(gr.x, db.x), gx(gr.y, db.y), gx(gr.z, db.z), gx(gr.w, db.w));
            } else {
                const float4 gr = reinterpret_cast<const float4*>(data_g + vb)[lane];
                g4 = make_float4(expf(gr.x), expf(gr.y), expf(gr.z), expf(gr.w));
            }
            sq[buf][lane] = q; sk[buf][lane] = k; sgx[buf][lane] = g4;
            if (lane == 0) sbeta[buf] = beta_val;
        }
        __syncthreads();
        const float4 q = sq[buf][lane], k = sk[buf][lane], g = sgx[buf][lane];
        const float beta_val = sbeta[buf];
        const uint32_t attn_off = (token_idx * H + head_id) * S_v;
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const uint32_t col = col_base + c;
            const float v_val = data_v[vb + col];
            float4& sc = s4[c];
            sc.x *= g.x; sc.y *= g.y; sc.z *= g.z; sc.w *= g.w;
            float kv = 0.0f;
            kv = fmaf(sc.x, k.x, kv); kv = fmaf(sc.y, k.y, kv); kv = fmaf(sc.z, k.z, kv); kv = fmaf(sc.w, k.w, kv);
            const float kv_col = glm_wave32_sum_dpp(kv);
            const float delta = (v_val - kv_col) * beta_val;
            sc.x = fmaf(k.x, delta, sc.x); sc.y = fmaf(k.y, delta, sc.y); sc.z = fmaf(k.z, delta, sc.z); sc.w = fmaf(k.w, delta, sc.w);
            float at = 0.0f;
            at = fmaf(sc.x, q.x, at); at = fmaf(sc.y, q.y, at); at = fmaf(sc.z, q.z, at); at = fmaf(sc.w, q.w, at);
            const float attn_col = glm_wave32_sum_dpp(at);
            if (lane == 0) data_dst[attn_off + col] = attn_col * scale;
        }
        if (K_snap > 0 && t >= n_seq_tokens - K_snap) {
            const uint32_t k_idx = t - (n_seq_tokens - K_snap);
            const int32_t slot_out = data_ids[k_idx * n_B + b_idx];
            if (slot_out >= 0) {
                float4* st_out = reinterpret_cast<float4*>(data_state + static_cast<size_t>(slot_out) * state_stride_b + head_id * sv2);
#pragma unroll
                for (int c = 0; c < CPW; ++c) st_out[(col_base + c) * 32u + lane] = s4[c];
            }
        }
    }
}

// v3 KDA step: v2's arithmetic expression for expression (bitwise v2), software-pipelined for prompt windows:
// wave 0 holds the next token's raw q / k / g / beta in registers (loaded one token ahead) and every wave its next
// v values, so a token step waits for no global load (v2: two dependent global round trips per token).
template <int CPW, int NWV>
__global__ __launch_bounds__(32 * NWV)
void glm_kda_step_v3_kernel(
    const float*   __restrict__ data_q,
    const float*   __restrict__ data_k,
    const float*   __restrict__ data_v,
    const float*   __restrict__ data_g,
    const float*   __restrict__ data_beta,
    float*         __restrict__ data_state,
    const int32_t* __restrict__ data_ids,
    const float*   __restrict__ data_dtb,
    const float*   __restrict__ data_a,
    float*         __restrict__ data_dst,
    uint32_t H,
    uint32_t n_tokens,
    uint32_t K_snap,
    uint32_t n_B,
    float    scale,
    uint32_t prologue,
    float    pro_lb,
    float    pro_l2_eps)
{
    constexpr uint32_t S_v = 128;
    __shared__ float4 sq[2][32], sk[2][32], sgx[2][32];
    __shared__ float sbeta[2];

    const uint32_t head_id = blockIdx.x;
    const uint32_t b_idx   = blockIdx.z;
    const uint32_t w       = threadIdx.x >> 5;
    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t col_base = (blockIdx.y * NWV + w) * CPW;
    const uint32_t sv2 = S_v * S_v;
    const uint32_t n_seq_tokens = n_tokens / n_B;
    const uint32_t state_stride_b = sv2 * H;

    const int32_t slot_in = (K_snap > 0) ? data_ids[b_idx] : static_cast<int32_t>(b_idx);
    const float4* st_in = reinterpret_cast<const float4*>(data_state + static_cast<size_t>(slot_in) * state_stride_b + head_id * sv2);

    float4 s4[CPW];
#pragma unroll
    for (int c = 0; c < CPW; ++c) s4[c] = st_in[(col_base + c) * 32u + lane];

    // the raw inputs of the next token (wave 0) and the next v values (all waves)
    float4 nq = make_float4(0.f, 0.f, 0.f, 0.f), nk = nq, ng = nq;
    float nbeta = 0.0f;
    float nv[CPW];
    float4 db = nq;
    float a_h = 0.0f;
    auto load_tok = [&](uint32_t t) {
        const uint32_t vb = (b_idx * n_seq_tokens + t) * S_v * H + head_id * S_v;
        if (w == 0) {
            nq = reinterpret_cast<const float4*>(data_q + vb)[lane];
            nk = reinterpret_cast<const float4*>(data_k + vb)[lane];
            ng = reinterpret_cast<const float4*>(data_g + vb)[lane];
            nbeta = data_beta[(b_idx * n_seq_tokens + t) * H + head_id];
        }
#pragma unroll
        for (int c = 0; c < CPW; ++c) nv[c] = data_v[vb + col_base + c];
    };
    if (w == 0 && prologue != 0) {
        a_h = data_a[head_id];
        db = reinterpret_cast<const float4*>(data_dtb + head_id * S_v)[lane];
    }
    if (n_seq_tokens > 0) load_tok(0);

    for (uint32_t t = 0; t < n_seq_tokens; ++t) {
        const uint32_t token_idx = b_idx * n_seq_tokens + t;
        const uint32_t buf = t & 1u;
        float vcur[CPW];
#pragma unroll
        for (int c = 0; c < CPW; ++c) vcur[c] = nv[c];
        if (w == 0) {
            float4 q = nq, k = nk;
            float4 g4;
            float beta_val = nbeta;
            const float4 gr = ng;
            if (prologue != 0) {
                float qq = 0.0f, kk = 0.0f;
                qq = fmaf(q.x, q.x, qq); qq = fmaf(q.y, q.y, qq); qq = fmaf(q.z, q.z, qq); qq = fmaf(q.w, q.w, qq);
                kk = fmaf(k.x, k.x, kk); kk = fmaf(k.y, k.y, kk); kk = fmaf(k.z, k.z, kk); kk = fmaf(k.w, k.w, kk);
                qq = glm_wave32_sum_dpp(qq);
                kk = glm_wave32_sum_dpp(kk);
                const float n = static_cast<float>(S_v);
                const float eps = pro_l2_eps / n;
                const float fq = rsqrtf(qq / n + eps), fk = rsqrtf(kk / n + eps), isn = rsqrtf(n);
                q = make_float4(fq * q.x * isn, fq * q.y * isn, fq * q.z * isn, fq * q.w * isn);
                k = make_float4(fk * k.x * isn, fk * k.y * isn, fk * k.z * isn, fk * k.w * isn);
                beta_val = 1.0f / (1.0f + expf(-beta_val));
                auto gx = [&](float gv, float dv) { return expf((1.0f / (1.0f + expf((gv + dv) * a_h))) * pro_lb); };
                g4 = make_float4(gx(gr.x, db.x), gx(gr.y, db.y), gx(gr.z, db.z), gx(gr.w, db.w));
            } else {
                g4 = make_float4(expf(gr.x), expf(gr.y), expf(gr.z), expf(gr.w));
            }
            sq[buf][lane] = q; sk[buf][lane] = k; sgx[buf][lane] = g4;
            if (lane == 0) sbeta[buf] = beta_val;
        }
        if (t + 1 < n_seq_tokens) load_tok(t + 1);
        __syncthreads();
        const float4 q = sq[buf][lane], k = sk[buf][lane], g = sgx[buf][lane];
        const float beta_val = sbeta[buf];
        const uint32_t attn_off = (token_idx * H + head_id) * S_v;
#pragma unroll
        for (int c = 0; c < CPW; ++c) {
            const uint32_t col = col_base + c;
            const float v_val = vcur[c];
            float4& sc = s4[c];
            sc.x *= g.x; sc.y *= g.y; sc.z *= g.z; sc.w *= g.w;
            float kv = 0.0f;
            kv = fmaf(sc.x, k.x, kv); kv = fmaf(sc.y, k.y, kv); kv = fmaf(sc.z, k.z, kv); kv = fmaf(sc.w, k.w, kv);
            const float kv_col = glm_wave32_sum_dpp(kv);
            const float delta = (v_val - kv_col) * beta_val;
            sc.x = fmaf(k.x, delta, sc.x); sc.y = fmaf(k.y, delta, sc.y); sc.z = fmaf(k.z, delta, sc.z); sc.w = fmaf(k.w, delta, sc.w);
            float at = 0.0f;
            at = fmaf(sc.x, q.x, at); at = fmaf(sc.y, q.y, at); at = fmaf(sc.z, q.z, at); at = fmaf(sc.w, q.w, at);
            const float attn_col = glm_wave32_sum_dpp(at);
            if (lane == 0) data_dst[attn_off + col] = attn_col * scale;
        }
        if (K_snap > 0 && t >= n_seq_tokens - K_snap) {
            const uint32_t k_idx = t - (n_seq_tokens - K_snap);
            const int32_t slot_out = data_ids[k_idx * n_B + b_idx];
            if (slot_out >= 0) {
                float4* st_out = reinterpret_cast<float4*>(data_state + static_cast<size_t>(slot_out) * state_stride_b + head_id * sv2);
#pragma unroll
                for (int c = 0; c < CPW; ++c) st_out[(col_base + c) * 32u + lane] = s4[c];
            }
        }
    }
}

// ============================================================================
// 3A. DSA Decode Lightning Indexer (glm_dsa_lightning_indexer)
//     - Direct port of ggml-vulkan/vulkan-shaders/lightning_indexer_vec.comp
//       using wave-uniform scalar Q loads (s_load_b512 into SGPRs) and RDNA3.5
//       VOPD dual-issue FMAs (v_dual_fmac_f32) across 8 independent accumulators
//       with zero LDS, zero barriers, and zero wave shuffles.
//     - For T=3 at large n_kv (>= 8192), TOKENS_FUSED=3 reuses the 32 float4
//       key registers across all 3 tokens so keys are read from memory only once.
// ============================================================================

template <bool K_IS_F16, int TOKENS_FUSED>
__global__ __launch_bounds__(64)
void glm_dsa_lightning_indexer_kernel(
    const float4* __restrict__ q4,
    const void*   __restrict__ data_k,
    const float*  __restrict__ weights,
    float*        __restrict__ dst,
    uint32_t n_heads,
    uint32_t n_kv,
    uint32_t n_tokens)
{
    constexpr uint32_t WG_SIZE = 64u;
    constexpr uint32_t D4      = 32u;

    const uint32_t s     = blockIdx.z;
    const uint32_t t0    = blockIdx.y * static_cast<uint32_t>(TOKENS_FUSED);
    const uint32_t ik    = blockIdx.x * WG_SIZE + threadIdx.x;
    // Out-of-range lanes clamp to n_kv - 1 so control flow around wave-uniform
    // scalar Q/weight loads stays wave-uniform (enabling s_load_b512 + v_dual_fmac_f32).
    const bool     valid = ik < n_kv;
    const uint32_t ikc   = valid ? ik : (n_kv - 1u);

    float4 kr[D4];
    const uint32_t k_base = (s * n_kv + ikc) * D4;
    #pragma unroll
    for (uint32_t c4 = 0; c4 < D4; ++c4) {
        if constexpr (!K_IS_F16) {
            const float4* k_v4 = static_cast<const float4*>(data_k);
            kr[c4] = k_v4[k_base + c4];
        } else {
            const uint2* k_u2 = static_cast<const uint2*>(data_k);
            const uint2 u = k_u2[k_base + c4];
            const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            kr[c4] = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
    }

    #pragma unroll
    for (int tf = 0; tf < TOKENS_FUSED; ++tf) {
        const uint32_t t = t0 + static_cast<uint32_t>(tf);
        if (t < n_tokens) {
            const uint32_t ts_idx = s * n_tokens + t;
            const uint32_t q_t    = ts_idx * n_heads * D4;
            const uint32_t w_t    = ts_idx * n_heads;

            float score = 0.0f;
            for (uint32_t h = 0; h < n_heads; h += 2u) {
                const uint32_t qa = q_t + h * D4;
                const uint32_t qb = qa + D4;
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                float bx = 0.0f, by = 0.0f, bz = 0.0f, bw = 0.0f;
                #pragma unroll
                for (uint32_t c4 = 0; c4 < D4; ++c4) {
                    const float4 qa4 = q4[qa + c4];
                    const float4 qb4 = q4[qb + c4];
                    ax = fmaf(kr[c4].x, qa4.x, ax);
                    ay = fmaf(kr[c4].y, qa4.y, ay);
                    az = fmaf(kr[c4].z, qa4.z, az);
                    aw = fmaf(kr[c4].w, qa4.w, aw);
                    bx = fmaf(kr[c4].x, qb4.x, bx);
                    by = fmaf(kr[c4].y, qb4.y, by);
                    bz = fmaf(kr[c4].z, qb4.z, bz);
                    bw = fmaf(kr[c4].w, qb4.w, bw);
                }
                const float da = (ax + ay) + (az + aw);
                const float db = (bx + by) + (bz + bw);
                score += fmaxf(da, 0.0f) * weights[w_t + h];
                score += fmaxf(db, 0.0f) * weights[w_t + h + 1u];
            }

            if (valid) {
                dst[ts_idx * n_kv + ik] = score;
            }
        }
    }
}

// GLM HIP v1 indexer: 32 keys per workgroup (one per lane), the heads split over NWV waves (n_heads / NWV each, q
// and the head weights still wave-uniform scalar loads), TF tokens per workgroup reusing the key registers, and the
// per-wave partial scores summed through LDS in wave order.  At 2K keys this is 64 x NWV waves instead of v0's 64,
// so the GPU is filled at short context; at 16K NWV 1 / TF 3 keeps v0's single pass over the keys.
template <bool K_IS_F16, int NWV, int TF, bool QLDS = false>
__global__ __launch_bounds__(32 * NWV)
void glm_dsa_li_v1_kernel(
    const float4* __restrict__ q4,
    const void*   __restrict__ data_k,
    const float*  __restrict__ weights,
    float*        __restrict__ dst,
    uint32_t n_heads,
    uint32_t n_kv,
    uint32_t n_tokens)
{
    constexpr uint32_t D4 = 32u;
    __shared__ float part[NWV > 1 ? NWV : 1][TF][32];

    const uint32_t s     = blockIdx.z;
    const uint32_t t0    = blockIdx.y * static_cast<uint32_t>(TF);
    const uint32_t lane  = threadIdx.x & 31u;
    const uint32_t w     = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const uint32_t ik    = blockIdx.x * 32u + lane;
    const bool     valid = ik < n_kv;
    const uint32_t ikc   = valid ? ik : (n_kv - 1u);
    const uint32_t hpw   = n_heads / NWV;
    const uint32_t hb    = w * hpw;

    // QLDS (v3): the TF tokens' q rows and head weights staged in LDS once per workgroup, read back as broadcast
    // ds_load_b128 (every lane the same address) instead of the scalar-cache s_load stream of v0 / v1
    extern __shared__ float4 qsh4[];
    if constexpr (QLDS) {
        const uint32_t nq4 = n_heads * D4;
        for (int tf = 0; tf < TF; ++tf) {
            const uint32_t t = t0 + static_cast<uint32_t>(tf);
            if (t >= n_tokens) break;
            const uint32_t ts_idx = s * n_tokens + t;
            for (uint32_t i = threadIdx.x; i < nq4; i += 32u * NWV) qsh4[tf * nq4 + i] = q4[ts_idx * nq4 + i];
            float* wsh = reinterpret_cast<float*>(qsh4 + TF * nq4);
            for (uint32_t i = threadIdx.x; i < n_heads; i += 32u * NWV) wsh[tf * n_heads + i] = weights[ts_idx * n_heads + i];
        }
        __syncthreads();
    }

    float4 kr[D4];
    const uint32_t k_base = (s * n_kv + ikc) * D4;
#pragma unroll
    for (uint32_t c4 = 0; c4 < D4; ++c4) {
        if constexpr (!K_IS_F16) {
            kr[c4] = static_cast<const float4*>(data_k)[k_base + c4];
        } else {
            const uint2 u = static_cast<const uint2*>(data_k)[k_base + c4];
            const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
            const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
            kr[c4] = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
    }

    float sc[TF];
#pragma unroll
    for (int tf = 0; tf < TF; ++tf) {
        sc[tf] = 0.0f;
        const uint32_t t = t0 + static_cast<uint32_t>(tf);
        if (t < n_tokens) {
            const uint32_t ts_idx = s * n_tokens + t;
            const float4* qb4 = QLDS ? qsh4 : q4;
            const float*  wb  = QLDS ? reinterpret_cast<const float*>(qsh4 + TF * n_heads * D4) : weights;
            const uint32_t q_t    = QLDS ? static_cast<uint32_t>(tf) * n_heads * D4 : ts_idx * n_heads * D4;
            const uint32_t w_t    = QLDS ? static_cast<uint32_t>(tf) * n_heads : ts_idx * n_heads;
            float score = 0.0f;
            for (uint32_t h = hb; h < hb + hpw; h += 2u) {
                const uint32_t qa = q_t + h * D4;
                const uint32_t qb = qa + D4;
                float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
                float bx = 0.0f, by = 0.0f, bz = 0.0f, bw = 0.0f;
#pragma unroll
                for (uint32_t c8 = 0; c8 < D4; c8 += 8u) {
                  float4 qa8[8], qb8[8];
#pragma unroll
                  for (uint32_t j = 0; j < 8u; ++j) { qa8[j] = qb4[qa + c8 + j]; qb8[j] = qb4[qb + c8 + j]; }
#pragma unroll
                  for (uint32_t j = 0; j < 8u; ++j) {
                    const uint32_t c4 = c8 + j;
                    const float4 qa4 = qa8[j];
                    const float4 qbb4 = qb8[j];
                    ax = fmaf(kr[c4].x, qa4.x, ax);
                    ay = fmaf(kr[c4].y, qa4.y, ay);
                    az = fmaf(kr[c4].z, qa4.z, az);
                    aw = fmaf(kr[c4].w, qa4.w, aw);
                    bx = fmaf(kr[c4].x, qbb4.x, bx);
                    by = fmaf(kr[c4].y, qbb4.y, by);
                    bz = fmaf(kr[c4].z, qbb4.z, bz);
                    bw = fmaf(kr[c4].w, qbb4.w, bw);
                  }
#if defined(__HIP_DEVICE_COMPILE__)
                  if constexpr (QLDS) {   // the chunk's 16 LDS reads issued together, then its 64 FMAs
                      __builtin_amdgcn_sched_group_barrier(0x100, 16, 0);
                      __builtin_amdgcn_sched_group_barrier(0x002, 64, 0);
                  }
#endif
                }
                const float da = (ax + ay) + (az + aw);
                const float db = (bx + by) + (bz + bw);
                score += fmaxf(da, 0.0f) * wb[w_t + h];
                score += fmaxf(db, 0.0f) * wb[w_t + h + 1u];
            }
            sc[tf] = score;
        }
    }
    if constexpr (NWV == 1) {
#pragma unroll
        for (int tf = 0; tf < TF; ++tf) {
            const uint32_t t = t0 + static_cast<uint32_t>(tf);
            if (t < n_tokens && valid) dst[(s * n_tokens + t) * n_kv + ik] = sc[tf];
        }
    } else {
#pragma unroll
        for (int tf = 0; tf < TF; ++tf) part[w][tf][lane] = sc[tf];
        __syncthreads();
        if (w == 0) {
#pragma unroll
            for (int tf = 0; tf < TF; ++tf) {
                float v = part[0][tf][lane];
#pragma unroll
                for (int j = 1; j < NWV; ++j) v += part[j][tf][lane];
                const uint32_t t = t0 + static_cast<uint32_t>(tf);
                if (t < n_tokens && valid) dst[(s * n_tokens + t) * n_kv + ik] = v;
            }
        }
    }
}

// fma(f16 half of k (widened exactly), q, acc) in one v_fma_mix_f32; asm so the widening is not hoisted out of the
// head loop (LICM would otherwise keep both keys as 256 f32 VGPRs and spill).  Same value as fmaf(float(k), q, acc).
__device__ __forceinline__ float glm_fma_mix_lo(uint32_t k, float q, float acc) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("v_fma_mix_f32 %0, %1, %2, %0 op_sel_hi:[1,0,0]" : "+v"(acc) : "v"(k), "s"(q));
    return acc;
#else
    return acc;
#endif
}
__device__ __forceinline__ float glm_fma_mix_hi(uint32_t k, float q, float acc) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("v_fma_mix_f32 %0, %1, %2, %0 op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "+v"(acc) : "v"(k), "s"(q));
    return acc;
#else
    return acc;
#endif
}

// GLM HIP v4 indexer (f16 keys): two keys per lane (k and k + 32), kept packed as f16 (64 VGPRs each) and widened
// inside v_fma_mix_f32, heads over NWV waves.  Each q value from the scalar cache now feeds two keys, halving the
// s_load stream per FMA that bounds v0 / v1 at 8K-16K keys.  Per (key, head) the four partial sums and their combine
// are v0's, so per-key results equal v1 with the same NWV.
template <int NWV>
__global__ __launch_bounds__(32 * NWV)
void glm_dsa_li_v4_kernel(
    const float4* __restrict__ q4,
    const uint2*  __restrict__ k2,
    const float*  __restrict__ weights,
    float*        __restrict__ dst,
    uint32_t n_heads,
    uint32_t n_kv,
    uint32_t n_tokens)
{
    constexpr uint32_t D4 = 32u;
    __shared__ float part[NWV > 1 ? NWV : 1][64];
    const uint32_t s    = blockIdx.z;
    const uint32_t t    = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t w    = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const uint32_t ik0  = blockIdx.x * 64u + lane;
    const uint32_t ik1  = ik0 + 32u;
    const uint32_t c0   = min(ik0, n_kv - 1u), c1 = min(ik1, n_kv - 1u);
    const uint32_t hpw  = n_heads / NWV;
    const uint32_t hb   = w * hpw;

    uint2 ka[D4], kb[D4];
#pragma unroll
    for (uint32_t c4 = 0; c4 < D4; ++c4) {
        ka[c4] = k2[(s * n_kv + c0) * D4 + c4];
        kb[c4] = k2[(s * n_kv + c1) * D4 + c4];
    }
    const uint32_t ts_idx = s * n_tokens + t;
    const uint32_t q_t = ts_idx * n_heads * D4;
    const uint32_t w_t = ts_idx * n_heads;
    float sa = 0.0f, sb = 0.0f;
    for (uint32_t h = hb; h < hb + hpw; ++h) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
        float bx = 0.0f, by = 0.0f, bz = 0.0f, bw = 0.0f;
#pragma unroll
        for (uint32_t c4 = 0; c4 < D4; ++c4) {
            const float4 q = q4[q_t + h * D4 + c4];
            ax = glm_fma_mix_lo(ka[c4].x, q.x, ax); ay = glm_fma_mix_hi(ka[c4].x, q.y, ay);
            az = glm_fma_mix_lo(ka[c4].y, q.z, az); aw = glm_fma_mix_hi(ka[c4].y, q.w, aw);
            bx = glm_fma_mix_lo(kb[c4].x, q.x, bx); by = glm_fma_mix_hi(kb[c4].x, q.y, by);
            bz = glm_fma_mix_lo(kb[c4].y, q.z, bz); bw = glm_fma_mix_hi(kb[c4].y, q.w, bw);
        }
        const float wh = weights[w_t + h];
        sa += fmaxf((ax + ay) + (az + aw), 0.0f) * wh;
        sb += fmaxf((bx + by) + (bz + bw), 0.0f) * wh;
    }
    if constexpr (NWV == 1) {
        if (ik0 < n_kv) dst[ts_idx * n_kv + ik0] = sa;
        if (ik1 < n_kv) dst[ts_idx * n_kv + ik1] = sb;
    } else {
        part[w][lane] = sa;
        part[w][lane + 32u] = sb;
        __syncthreads();
        if (w == 0) {
            float va = part[0][lane], vb = part[0][lane + 32u];
#pragma unroll
            for (int j = 1; j < NWV; ++j) { va += part[j][lane]; vb += part[j][lane + 32u]; }
            if (ik0 < n_kv) dst[ts_idx * n_kv + ik0] = va;
            if (ik1 < n_kv) dst[ts_idx * n_kv + ik1] = vb;
        }
    }
}

template <int I, int N, typename Fn>
__device__ __forceinline__ void glm_static_for(Fn&& f) {
    if constexpr (I < N) {
        f(std::integral_constant<int, I>{});
        glm_static_for<I + 1, N>(f);
    }
}

// GLM HIP v2 indexer (n_heads == 32): lane = head.  Each lane keeps its head's q row in VGPRs (loaded once per wave)
// and the keys are wave-uniform (scalar loads), so per key a lane does the 128-term dot for its head (v0's four
// partial sums and combine order), relu x head weight, and a transposing butterfly over 32 keys leaves lane L with
// key L's sum over the 32 heads.  q traffic is per wave instead of per 32 keys; the key stream is contiguous.
template <bool K_IS_F16, int KB, int NWV>
__global__ __launch_bounds__(32 * NWV)
void glm_dsa_li_v2_kernel(
    const float4* __restrict__ q4,
    const void*   __restrict__ data_k,
    const float*  __restrict__ weights,
    float*        __restrict__ dst,
    uint32_t n_kv,
    uint32_t n_tokens)
{
    constexpr uint32_t D4 = 32u;
    const uint32_t s    = blockIdx.z;
    const uint32_t t    = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t w    = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const uint32_t k0   = (blockIdx.x * NWV + w) * KB;
    if (k0 >= n_kv) return;
    const uint32_t ts_idx = s * n_tokens + t;

    float4 qr[D4];
#pragma unroll
    for (uint32_t c4 = 0; c4 < D4; ++c4) qr[c4] = q4[(ts_idx * 32u + lane) * D4 + c4];
    const float wv = weights[ts_idx * 32u + lane];

    for (uint32_t kb = 0; kb < KB; kb += 32u) {
        float part[32];
        glm_static_for<0, 32>([&](auto jc) {
            constexpr uint32_t j = decltype(jc)::value;
            const uint32_t key = min(k0 + kb + j, n_kv - 1u);
            float ax = 0.0f, ay = 0.0f, az = 0.0f, aw = 0.0f;
#pragma unroll
            for (uint32_t c4 = 0; c4 < D4; ++c4) {
                float4 kv;
                if constexpr (!K_IS_F16) {
                    kv = static_cast<const float4*>(data_k)[(s * n_kv + key) * D4 + c4];
                } else {
                    const uint2 u = static_cast<const uint2*>(data_k)[(s * n_kv + key) * D4 + c4];
                    const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
                    const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
                    kv = make_float4(lo.x, lo.y, hi.x, hi.y);
                }
                ax = fmaf(kv.x, qr[c4].x, ax);
                ay = fmaf(kv.y, qr[c4].y, ay);
                az = fmaf(kv.z, qr[c4].z, az);
                aw = fmaf(kv.w, qr[c4].w, aw);
            }
            part[j] = fmaxf((ax + ay) + (az + aw), 0.0f) * wv;
        });
        // transposing butterfly: after step with offset o, lanes with bit o set hold the upper half of the keys
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            const bool up = (lane & static_cast<uint32_t>(o)) != 0u;
#pragma unroll
            for (int j = 0; j < o; ++j) {
                const float send = up ? part[j] : part[j + o];
                const float keep = up ? part[j + o] : part[j];
                part[j] = keep + __shfl_xor(send, o, 32);
            }
        }
        const uint32_t key = k0 + kb + lane;   // offsets 16,8,4,2,1 map lane bits straight to the key index
        if (key < n_kv && kb + lane < KB) dst[ts_idx * n_kv + key] = part[0];
    }
}

// ============================================================================
// 3B. DSA Sparse MLA Decode Attention
//     Stage 1: glm_dsa_sparse_compact_kernel (port of flash_attn_sparse_compact.comp
//              with 8-way unrolled global prefetch + 1-pass LDS ballot caching)
//     Stage 2: glm_dsa_sparse_fa_dec_kernel (port of flash_attn_sparse_dec.comp
//              with __builtin_amdgcn_fdot2, lds_sync, and fast_expf)
//     Stage 3: glm_dsa_sparse_fa_split_k_reduce_kernel (coalesced float4 per-(h, t) WG)
// ============================================================================

// lane ^ K for a wave32 value through DPP row_xmask (K < 16) or v_permlanex16 (K == 16): the same data movement as
// __shfl_xor(v, K, 32) without the ds_bpermute round trip.
template <int K>
__device__ __forceinline__ float glm_xor_dpp(float v) {
    static_assert(K >= 1 && K <= 16, "lane xor 1..16");
    if constexpr (K == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 | K, 0xf, 0xf, false));
    }
}

__device__ __forceinline__ void lds_sync() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ float fast_expf(float x) {
    return __builtin_amdgcn_exp2f(x * 1.4426950408889634f);
}

__device__ __forceinline__ bool is_selected_fp16_bits(uint16_t bits) {
    return (bits & 0x7C00u) != 0x7C00u;
}

__global__ __launch_bounds__(1024)
void glm_dsa_sparse_compact_kernel(
    const uint16_t* __restrict__ data_m,
    int32_t*        __restrict__ data_i,
    uint32_t KV,
    uint32_t nem1,
    uint32_t nem2,
    uint32_t nbm1,
    uint32_t nbm2,
    uint32_t nbm3,
    uint32_t n_kv_max)
{
    constexpr uint32_t BLOCK_SIZE    = 1024;
    constexpr uint32_t NUM_SUBGROUPS = 32;
    constexpr uint32_t SUBGROUP_SIZE = 32;
    constexpr uint32_t MAX_CHUNKS    = 64; // 32 * 64 * 32 = 65536 KV positions cached in LDS

    __shared__ uint32_t counts_sh[NUM_SUBGROUPS];
    __shared__ uint32_t prefix_sh[NUM_SUBGROUPS];
    __shared__ uint32_t total_sh;
    __shared__ uint32_t ballots_sh[NUM_SUBGROUPS][MAX_CHUNKS];

    const uint32_t i1  = blockIdx.x;
    const uint32_t i2  = blockIdx.y;
    const uint32_t i3  = blockIdx.z;
    const uint32_t tid = threadIdx.x;

    const uint32_t m_base   = i3 * nbm3 + i2 * nbm2 + i1 * nbm1;
    const uint32_t out_base = ((i3 * nem2 + i2) * nem1 + i1) * n_kv_max;

    const uint32_t sg   = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t raw_seg   = (KV + NUM_SUBGROUPS - 1) / NUM_SUBGROUPS;
    const uint32_t seg       = ((raw_seg + 31u) & ~31u);
    const uint32_t seg_begin = min(sg * seg, KV);
    const uint32_t seg_end   = min(seg_begin + seg, KV);
    const uint32_t n_chunks  = (seg_end - seg_begin + 31u) >> 5;

    uint32_t count = 0;

    if (seg <= MAX_CHUNKS * SUBGROUP_SIZE) {
        constexpr uint32_t UNROLL = 8;
        for (uint32_t c0 = 0; c0 < n_chunks; c0 += UNROLL) {
            uint16_t m_vals[UNROLL];
            #pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
                const uint32_t k = seg_begin + (c0 + u) * SUBGROUP_SIZE + lane;
                m_vals[u] = (k < seg_end) ? data_m[m_base + k] : 0xFC00u;
            }
            #pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
                const uint32_t c = c0 + u;
                if (c < n_chunks) {
                    const uint32_t b = static_cast<uint32_t>(__ballot(is_selected_fp16_bits(m_vals[u])));
                    if (lane == 0u) {
                        ballots_sh[sg][c] = b;
                    }
                    count += __popc(b);
                }
            }
        }
        if (lane == 0) {
            counts_sh[sg] = count;
        }
        lds_sync();

        if (sg == 0) {
            uint32_t v = counts_sh[lane];
            uint32_t inclusive = v;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const uint32_t n = __shfl_up(inclusive, off, 32);
                if (lane >= static_cast<uint32_t>(off)) inclusive += n;
            }
            prefix_sh[lane] = inclusive - v;
            if (lane == 31) {
                total_sh = inclusive;
            }
        }
        lds_sync();

        uint32_t slot = prefix_sh[sg];
        const uint32_t lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
        for (uint32_t c = 0; c < n_chunks && slot < n_kv_max; ++c) {
            const uint32_t b = ballots_sh[sg][c];
            if (b == 0u) continue;
            const uint32_t pos = slot + __popc(b & lane_mask);
            if (((b >> lane) & 1u) != 0u && pos < n_kv_max) {
                const uint32_t k = seg_begin + c * SUBGROUP_SIZE + lane;
                data_i[out_base + pos] = static_cast<int32_t>(k);
            }
            slot += __popc(b);
        }
    } else {
        constexpr uint32_t UNROLL = 16;
        constexpr uint32_t STEP   = SUBGROUP_SIZE * UNROLL;
        for (uint32_t k0 = seg_begin; k0 < seg_end; k0 += STEP) {
            #pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
                const uint32_t k = k0 + u * SUBGROUP_SIZE + lane;
                const bool sel = (k < seg_end) && is_selected_fp16_bits(data_m[m_base + k]);
                count += __popc(static_cast<uint32_t>(__ballot(sel)));
            }
        }
        if (lane == 0) {
            counts_sh[sg] = count;
        }
        lds_sync();

        if (sg == 0) {
            uint32_t v = counts_sh[lane];
            uint32_t inclusive = v;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const uint32_t n = __shfl_up(inclusive, off, 32);
                if (lane >= static_cast<uint32_t>(off)) inclusive += n;
            }
            prefix_sh[lane] = inclusive - v;
            if (lane == 31) {
                total_sh = inclusive;
            }
        }
        lds_sync();

        uint32_t slot = prefix_sh[sg];
        const uint32_t lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
        for (uint32_t k0 = seg_begin; k0 < seg_end && slot < n_kv_max; k0 += STEP) {
            #pragma unroll
            for (uint32_t u = 0; u < UNROLL; ++u) {
                const uint32_t k = k0 + u * SUBGROUP_SIZE + lane;
                const bool sel = (k < seg_end) && is_selected_fp16_bits(data_m[m_base + k]);
                const uint32_t b = static_cast<uint32_t>(__ballot(sel));
                const uint32_t pos = slot + __popc(b & lane_mask);
                if (sel && pos < n_kv_max) {
                    data_i[out_base + pos] = static_cast<int32_t>(k);
                }
                slot += __popc(b);
            }
        }
    }

    const uint32_t total = total_sh;
    for (uint32_t s = min(total, n_kv_max) + tid; s < n_kv_max; s += BLOCK_SIZE) {
        data_i[out_base + s] = -1;
    }
}

struct alignas(8) Half4 {
    __half2 xy;
    __half2 zw;
};

__device__ __forceinline__ __half2 u32_to_h2(uint32_t u) {
    union { uint32_t u32; __half2 h2; } cvt;
    cvt.u32 = u;
    return cvt.h2;
}

__device__ __forceinline__ uint32_t h2_to_u32(__half2 h) {
    union { __half2 h2; uint32_t u32; } cvt;
    cvt.h2 = h;
    return cvt.u32;
}

__device__ __forceinline__ float dot2_f16_acc(__half2 a, __half2 b, float acc) {
    using f16x2_t = _Float16 __attribute__((ext_vector_type(2)));
    union { __half2 h2; f16x2_t v2; } ua, ub;
    ua.h2 = a;
    ub.h2 = b;
    return __builtin_amdgcn_fdot2(ua.v2, ub.v2, acc, false);
}

__device__ __forceinline__ float dot4_f16_acc(Half4 a, Half4 b, float acc) {
    acc = dot2_f16_acc(a.xy, b.xy, acc);
    return dot2_f16_acc(a.zw, b.zw, acc);
}

// GLM HIP: FUSED = the compaction folded in.  Every workgroup scans the whole mask row (16-byte loads, 16 waves on
// contiguous segments, per-lane selection bytes cached in LDS), takes the per-wave prefix and keeps only the slots
// [slot0, slot0 + n_blk * BC) of the global compacted order: exactly the index slice the separate compaction pass
// would have handed it, so the attention math is v0's bit for bit, one launch fewer and no idx_buf round trip.
// Needs KV % 8 == 0, m_stride % 8 == 0 and KV <= 16 * 16 * 256 (65536); the launcher falls back otherwise.
template <bool V_SAME, bool FUSED = false, int WPE = 1>
__global__ __launch_bounds__(512, WPE)
void glm_dsa_sparse_fa_dec_kernel(
    const float4*  __restrict__ data_q4,
    const uint4*   __restrict__ data_k8,
    const Half4*   __restrict__ data_v4,
    const __half*  __restrict__ data_m,
    const int32_t* __restrict__ data_i,
    float*         __restrict__ data_o,
    uint32_t N,
    uint32_t n_kv_max,
    uint32_t k_num,
    uint32_t per_blocks,
    uint32_t q_nb1,
    uint32_t q_nb2,
    uint32_t k_stride,
    uint32_t v_stride,
    uint32_t m_stride,
    uint32_t ne1,
    float    scale,
    uint32_t n_kv_total = 0,
    uint32_t skip_pad = 0)
{
    constexpr uint32_t D         = 512u;
    constexpr uint32_t MAX_SLOTS = 2304u;
    constexpr uint32_t WG        = 512u;
    constexpr uint32_t HG        = 32u;
    constexpr uint32_t BC        = 32u;
    constexpr uint32_t DS        = 8u;
    constexpr uint32_t KPT       = 8u;

    constexpr uint32_t D4  = D / 4u;        // 128
    constexpr uint32_t D8  = D / 8u;        // 64
    constexpr uint32_t KS4 = D4 + 8u;       // 136
    constexpr uint32_t RPI = WG / D8;       // 8
    constexpr uint32_t NIT = BC / RPI;      // 4
    constexpr uint32_t PVH = WG / (D / 4u); // 4
    constexpr uint32_t HPV = HG / PVH;      // 8

    __shared__ Half4   ksh[BC * KS4];
    __shared__ float   ssh[BC * HG];
    __shared__ uint4   psh[(BC / 2) * (HG / 4)];
    __shared__ float   msh[BC];
    __shared__ float   esh[HG];
    __shared__ int32_t idxsh[MAX_SLOTS];

    const float NEG = __uint_as_float(0xFEFFFFFFu);

    const uint32_t tid   = threadIdx.x;
    const uint32_t h0    = blockIdx.x * HG;
    const uint32_t t     = blockIdx.y;
    const uint32_t split = blockIdx.z;

    const uint32_t total_blocks = (n_kv_max + BC - 1u) / BC;
    const uint32_t b_begin = min(split * per_blocks, total_blocks);
    const uint32_t b_end   = min(b_begin + per_blocks, total_blocks);
    const uint32_t n_blk   = b_end - b_begin;
    const uint32_t slot0   = b_begin * BC;

    if constexpr (!FUSED) {
        for (uint32_t s = tid; s < n_blk * BC; s += WG) {
            const uint32_t slot = slot0 + s;
            idxsh[s] = (slot < n_kv_max) ? data_i[t * n_kv_max + slot] : -1;
        }
    } else {
        constexpr uint32_t NWAVE = WG / 32u;   // 16
        constexpr uint32_t MAXC  = 16u;        // chunks of 256 positions per wave segment
        __shared__ uint8_t  selsh[NWAVE][MAXC][32];
        __shared__ uint32_t wcnt[NWAVE];
        const uint32_t w    = tid >> 5;
        const uint32_t lane = tid & 31u;
        const uint32_t KV   = n_kv_total;
        const uint32_t seg  = (((KV + NWAVE - 1u) / NWAVE) + 255u) & ~255u;
        const uint32_t sb   = min(w * seg, KV);
        const uint32_t se   = min(sb + seg, KV);
        const uint32_t nch  = (se - sb + 255u) >> 8;
        const uint4* mrow = reinterpret_cast<const uint4*>(data_m + t * m_stride);
        uint32_t cnt = 0;
        for (uint32_t c = 0; c < nch; ++c) {
            const uint32_t p = sb + c * 256u + lane * 8u;
            uint32_t bits = 0;
            if (p < se) {
                const uint4 v = mrow[p >> 3];
                const uint32_t h[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    bits |= (is_selected_fp16_bits(static_cast<uint16_t>(h[j] & 0xffffu)) ? 1u : 0u) << (2 * j);
                    bits |= (is_selected_fp16_bits(static_cast<uint16_t>(h[j] >> 16)) ? 1u : 0u) << (2 * j + 1);
                }
            }
            selsh[w][c][lane] = static_cast<uint8_t>(bits);
            cnt += __popc(bits);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) cnt += __shfl_xor(cnt, o, 32);
        if (lane == 0) wcnt[w] = cnt;
        for (uint32_t s = tid; s < n_blk * BC; s += WG) idxsh[s] = -1;
        __syncthreads();
        uint32_t base = 0;
        for (uint32_t i = 0; i < w; ++i) base += wcnt[i];
        const uint32_t lo = slot0, hi = min(slot0 + n_blk * BC, n_kv_max);
        if (base < hi && base + wcnt[w] > lo) {
            uint32_t slot = base;
            for (uint32_t c = 0; c < nch && slot < hi; ++c) {
                const uint32_t bits = selsh[w][c][lane];
                const uint32_t n = __popc(bits);
                uint32_t inc = n;   // inclusive lane scan
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const uint32_t y = __shfl_up(inc, o, 32);
                    if (lane >= static_cast<uint32_t>(o)) inc += y;
                }
                uint32_t my = slot + inc - n;
                const uint32_t p0 = sb + c * 256u + lane * 8u;
                uint32_t b = bits;
                while (b) {
                    const uint32_t j = __ffs(b) - 1u;
                    b &= b - 1u;
                    if (my >= lo && my < hi) idxsh[my - lo] = static_cast<int32_t>(p0 + j);
                    ++my;
                }
                slot += __shfl(inc, 31, 32);
            }
        }
    }

    const uint32_t qk_hp = tid >> 5;           // 0..15
    const uint32_t qk_kg = (tid / DS) & 3u;    // 0..3
    const uint32_t qk_ds = tid % DS;           // 0..7

    Half4 qr[2][D4 / DS];
    #pragma unroll
    for (uint32_t hh = 0; hh < 2; ++hh) {
        const uint32_t qb = (t * q_nb1 + (h0 + 2u * qk_hp + hh) * q_nb2) / 4u;
        #pragma unroll
        for (uint32_t i = 0; i < D4 / DS; ++i) {
            const float4 qv = data_q4[qb + qk_ds + DS * i];
            qr[hh][i].xy = __floats2half2_rn(qv.x * scale, qv.y * scale);
            qr[hh][i].zw = __floats2half2_rn(qv.z * scale, qv.w * scale);
        }
    }
    lds_sync();

    const uint32_t gc8 = tid % D8;
    const uint32_t gr0 = tid / D8;

    uint4 kreg[NIT];
    float mreg = NEG;

    if (n_blk > 0) {
        #pragma unroll
        for (uint32_t it = 0; it < NIT; ++it) {
            const int32_t idx = idxsh[it * RPI + gr0];
            kreg[it] = (idx >= 0) ? data_k8[(static_cast<uint32_t>(idx) * k_stride) / 8u + gc8] : make_uint4(0, 0, 0, 0);
        }
        if (tid < BC) {
            const int32_t idx = idxsh[tid];
            mreg = (idx >= 0) ? __half2float(data_m[t * m_stride + static_cast<uint32_t>(idx)]) : NEG;
        }
    }

    const uint32_t sm_h = tid >> 4;  // 0..31
    const uint32_t sm_c = tid & 15u; // 0..15
    float Mh = NEG;
    float Lh = 0.0f;

    const uint32_t pv_d  = tid % (D / 4u); // 0..127
    const uint32_t pv_hh = tid / (D / 4u); // 0..3
    float4 Of[HPV];
    #pragma unroll
    for (uint32_t h = 0; h < HPV; ++h) {
        Of[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (uint32_t b = 0; b < n_blk; ++b) {
        // Sparse skip (GLM_SPARSE_SKIP=1, index-list path): the list is ascending rows then -1 padding, so a block whose first
        // slot is padding and every later one are all masked; they would leave M, L and O exactly unchanged (p = 0,
        // e = 2^0 = 1): stopping here is bitwise the same result.
        if (skip_pad && idxsh[b * BC] < 0) break;
        if (b > 0u) {
            lds_sync();
        }

        #pragma unroll
        for (uint32_t it = 0; it < NIT; ++it) {
            const uint32_t r = it * RPI + gr0;
            const uint4 u = kreg[it];
            ksh[r * KS4 + 2u * gc8 + 0u] = Half4{ u32_to_h2(u.x), u32_to_h2(u.y) };
            ksh[r * KS4 + 2u * gc8 + 1u] = Half4{ u32_to_h2(u.z), u32_to_h2(u.w) };
        }
        if (tid < BC) {
            msh[tid] = mreg;
        }
        lds_sync();

        if (b + 1u < n_blk) {
            #pragma unroll
            for (uint32_t it = 0; it < NIT; ++it) {
                const int32_t idx = idxsh[(b + 1u) * BC + it * RPI + gr0];
                kreg[it] = (idx >= 0) ? data_k8[(static_cast<uint32_t>(idx) * k_stride) / 8u + gc8] : make_uint4(0, 0, 0, 0);
            }
            if (tid < BC) {
                const int32_t idx = idxsh[(b + 1u) * BC + tid];
                mreg = (idx >= 0) ? __half2float(data_m[t * m_stride + static_cast<uint32_t>(idx)]) : NEG;
            }
        }

        float acc[2 * KPT];
        #pragma unroll
        for (uint32_t j = 0; j < 2 * KPT; ++j) {
            acc[j] = 0.0f;
        }
        #pragma unroll
        for (uint32_t i = 0; i < D4 / DS; ++i) {
            #pragma unroll
            for (uint32_t c = 0; c < KPT; ++c) {
                const Half4 kv = ksh[(c * 4u + qk_kg) * KS4 + qk_ds + DS * i];
                acc[c]       = dot4_f16_acc(qr[0][i], kv, acc[c]);
                acc[KPT + c] = dot4_f16_acc(qr[1][i], kv, acc[KPT + c]);
            }
        }

        float r8[8];
        {
            const bool hi = (qk_ds & 4u) != 0u;
            #pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                const float mine  = hi ? acc[8 + k] : acc[k];
                const float other = hi ? acc[k] : acc[8 + k];
                r8[k] = mine + glm_xor_dpp<4>(other);
            }
        }
        float r4[4];
        {
            const bool hi = (qk_ds & 2u) != 0u;
            #pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
                const float mine  = hi ? r8[4 + k] : r8[k];
                const float other = hi ? r8[k] : r8[4 + k];
                r4[k] = mine + glm_xor_dpp<2>(other);
            }
        }
        float r2[2];
        {
            const bool hi = (qk_ds & 1u) != 0u;
            #pragma unroll
            for (uint32_t k = 0; k < 2; ++k) {
                const float mine  = hi ? r4[2 + k] : r4[k];
                const float other = hi ? r4[k] : r4[2 + k];
                r2[k] = mine + glm_xor_dpp<1>(other);
            }
        }
        {
            const uint32_t h  = 2u * qk_hp + qk_ds / 4u;
            const uint32_t c0 = (2u * (qk_ds % 4u)     ) * 4u + qk_kg;
            const uint32_t c1 = (2u * (qk_ds % 4u) + 1u) * 4u + qk_kg;
            const float m0 = msh[c0];
            const float m1 = msh[c1];
            ssh[c0 * HG + h] = (m0 <= NEG) ? NEG : (r2[0] + m0);
            ssh[c1 * HG + h] = (m1 <= NEG) ? NEG : (r2[1] + m1);
        }
        lds_sync();

        // Online softmax across 16 lanes per head (within half-wave32) using hardware v_exp_f32
        {
            const float s0 = ssh[(2u * sm_c     ) * HG + sm_h];
            const float s1 = ssh[(2u * sm_c + 1u) * HG + sm_h];
            float bmax = fmaxf(s0, s1);
            bmax = fmaxf(bmax, glm_xor_dpp<8>(bmax));
            bmax = fmaxf(bmax, glm_xor_dpp<4>(bmax));
            bmax = fmaxf(bmax, glm_xor_dpp<2>(bmax));
            bmax = fmaxf(bmax, glm_xor_dpp<1>(bmax));
            const float Mnew = fmaxf(Mh, bmax);
            const float p0 = (s0 <= NEG) ? 0.0f : fast_expf(s0 - Mnew);
            const float p1 = (s1 <= NEG) ? 0.0f : fast_expf(s1 - Mnew);
            uint32_t* psh_u32 = reinterpret_cast<uint32_t*>(&psh[sm_c * (HG / 4u) + sm_h / 4u]);
            psh_u32[sm_h % 4u] = h2_to_u32(__floats2half2_rn(p0, p1));

            float ls = p0 + p1;
            ls += glm_xor_dpp<8>(ls);
            ls += glm_xor_dpp<4>(ls);
            ls += glm_xor_dpp<2>(ls);
            ls += glm_xor_dpp<1>(ls);
            const float eM = fast_expf(Mh - Mnew);
            Lh = Lh * eM + ls;
            Mh = Mnew;
            if (sm_c == 0u) {
                esh[sm_h] = eM;
            }
        }
        lds_sync();

        #pragma unroll
        for (uint32_t h = 0; h < HPV; ++h) {
            const float e = esh[pv_hh * HPV + h];
            Of[h].x *= e;
            Of[h].y *= e;
            Of[h].z *= e;
            Of[h].w *= e;
        }
        #pragma unroll
        for (uint32_t cp = 0; cp < BC / 2u; ++cp) {
            Half4 va, vb;
            if constexpr (V_SAME) {
                va = ksh[(2u * cp     ) * KS4 + pv_d];
                vb = ksh[(2u * cp + 1u) * KS4 + pv_d];
            } else {
                const int32_t ia = idxsh[b * BC + 2u * cp];
                const int32_t ib = idxsh[b * BC + 2u * cp + 1u];
                const Half4 zero_h4{ __floats2half2_rn(0.0f, 0.0f), __floats2half2_rn(0.0f, 0.0f) };
                va = (ia >= 0) ? data_v4[(static_cast<uint32_t>(ia) * v_stride) / 4u + pv_d] : zero_h4;
                vb = (ib >= 0) ? data_v4[(static_cast<uint32_t>(ib) * v_stride) / 4u + pv_d] : zero_h4;
            }
            const __half2 vx = __halves2half2(__low2half(va.xy), __low2half(vb.xy));
            const __half2 vy = __halves2half2(__high2half(va.xy), __high2half(vb.xy));
            const __half2 vz = __halves2half2(__low2half(va.zw), __low2half(vb.zw));
            const __half2 vw = __halves2half2(__high2half(va.zw), __high2half(vb.zw));

            #pragma unroll
            for (uint32_t h4 = 0; h4 < HPV / 4u; ++h4) {
                const uint4 pk = psh[cp * (HG / 4u) + pv_hh * (HPV / 4u) + h4];
                const uint32_t pk_arr[4] = { pk.x, pk.y, pk.z, pk.w };
                #pragma unroll
                for (uint32_t k = 0; k < 4u; ++k) {
                    const __half2 pp = u32_to_h2(pk_arr[k]);
                    const uint32_t h = 4u * h4 + k;
                    Of[h].x = dot2_f16_acc(pp, vx, Of[h].x);
                    Of[h].y = dot2_f16_acc(pp, vy, Of[h].y);
                    Of[h].z = dot2_f16_acc(pp, vz, Of[h].z);
                    Of[h].w = dot2_f16_acc(pp, vw, Of[h].w);
                }
            }
        }
    }

    float4* data_o4 = reinterpret_cast<float4*>(data_o);
    if (k_num == 1u) {
        lds_sync();
        if (sm_c == 0u) {
            esh[sm_h] = (Lh == 0.0f) ? 0.0f : (1.0f / Lh);
        }
        lds_sync();
        const uint32_t d_base = (t * ne1 + h0 + pv_hh * HPV) * D;
        #pragma unroll
        for (uint32_t h = 0; h < HPV; ++h) {
            const float inv_l = esh[pv_hh * HPV + h];
            data_o4[(d_base + h * D) / 4u + pv_d] = make_float4(
                Of[h].x * inv_l, Of[h].y * inv_l, Of[h].z * inv_l, Of[h].w * inv_l);
        }
        return;
    }

    const uint32_t o_base = D * ne1 * (split + k_num * t) + (h0 + pv_hh * HPV) * D;
    #pragma unroll
    for (uint32_t h = 0; h < HPV; ++h) {
        data_o4[(o_base + h * D) / 4u + pv_d] = Of[h];
    }
    if (sm_c == 0u) {
        const uint32_t lm_base = D * ne1 * k_num * N + ne1 * 2u * (split + k_num * t);
        data_o[lm_base + h0 + sm_h]       = Lh;
        data_o[lm_base + ne1 + h0 + sm_h] = Mh;
    }
}

// Coalesced float4 split-K reduction: 1 workgroup of 128 threads per (head, token).
// Computes M and L ONCE per (head, token) on wave 0, then all 128 threads reduce
// the D=512 dimensions using coalesced float4 loads!
__global__ __launch_bounds__(128)
void glm_dsa_sparse_fa_split_k_reduce_kernel(
    const float* __restrict__ split_buf,
    float4*      __restrict__ dst4,
    uint32_t D,
    uint32_t H,
    uint32_t N,
    uint32_t k_num)
{
    __shared__ float s_weights[64];

    const uint32_t h   = blockIdx.x;
    const uint32_t t   = blockIdx.y;
    const uint32_t tid = threadIdx.x; // 0..127 (each tid owns float4 d4 = tid of D=512)

    const uint32_t lm_base = D * H * k_num * N + H * 2u * k_num * t + h;

    if (tid < 32u) {
        float m_max = -INFINITY;
        for (uint32_t k = tid; k < k_num; k += 32u) {
            const float m = split_buf[lm_base + k * 2u * H + H];
            m_max = fmaxf(m_max, m);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m_max = fmaxf(m_max, __shfl_xor(m_max, off, 32));
        }

        float l_sum = 0.0f;
        for (uint32_t k = tid; k < k_num; k += 32u) {
            const float l = split_buf[lm_base + k * 2u * H];
            const float m = split_buf[lm_base + k * 2u * H + H];
            const float w = (m > -1e30f) ? fast_expf(m - m_max) : 0.0f;
            s_weights[k] = w;
            l_sum += l * w;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            l_sum += __shfl_xor(l_sum, off, 32);
        }

        const float inv_l = (l_sum == 0.0f) ? 0.0f : (1.0f / l_sum);
        for (uint32_t k = tid; k < k_num; k += 32u) {
            s_weights[k] *= inv_l;
        }
    }
    lds_sync();

    const float4* split_o4 = reinterpret_cast<const float4*>(split_buf);
    const uint32_t D4 = D / 4u; // 128
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint32_t k = 0; k < k_num; ++k) {
        const float w = s_weights[k];
        const uint32_t o4_idx = (H * (k + k_num * t) + h) * D4 + tid;
        const float4 v = split_o4[o4_idx];
        acc.x = fmaf(v.x, w, acc.x);
        acc.y = fmaf(v.y, w, acc.y);
        acc.z = fmaf(v.z, w, acc.z);
        acc.w = fmaf(v.w, w, acc.w);
    }

    dst4[(t * H + h) * D4 + tid] = acc;
}

} // namespace

// ============================================================================
// Public Host Launchers
// ============================================================================

hipError_t glm_quantize_q8_1(
    const float* x_f32,
    void* x_q8_1,
    int n_elements,
    hipStream_t stream)
{
    const int64_t total_blocks = n_elements / 32;
    const uint32_t grid = static_cast<uint32_t>((total_blocks + 7) / 8);
    hipLaunchKernelGGL(
        glm_quantize_q8_1_kernel,
        dim3(grid), dim3(256), 0, stream,
        x_f32, static_cast<block_q8_1*>(x_q8_1), total_blocks);
    return hipGetLastError();
}

// ---- per-arch tuning table for the v1 expert mat-vec: (qtype, K, tokens) -> (R rows/wave, NW waves/WG, waves/EU).
// gfx1151 is the only tuned entry (gfx1151 mmvq sweep); every other target takes v0's layout (4 x 4).
// GLM_MMVQ_CFG="variant:R:NW:WPE" forces one config (sweeps); variant 0 = the WP-1.7 spike kernels.
namespace {
struct MmvqRule { int qtype, k, t0, t1; GlmMmvqCfg c; };   // qtype / k = -1: any
const MmvqRule kMmvqGeneric[] = { {-1, -1, 1, 64, {1, 4, 4, 8, 0}} };
// gfx1151 per-shape dispatch table (selected from warm and cold MALL graph-captured sweeps):
// T=1: persistent grid (16 WGs per CU) stages the grid table once per WG; IQ3_S at 2 waves per WG.
// T>=2 (dedup over tokens): one WG per tile; IQ3_S gate 2 rows/wave x 8 waves, IQ3_S down 4 x 8, IQ2_S down persistent.
const MmvqRule kMmvqGfx1151[] = {
    {22, -1, 1, 1, {1, 4, 4, 8, 16}},
    {21, -1, 1, 1, {1, 4, 2, 8, 16}},
    {22, 2048, 2, 64, {1, 4, 4, 8, 16}},
    {22, -1, 2, 64, {1, 4, 4, 8, 0}},
    {21, 4096, 2, 64, {1, 2, 8, 8, 0}},
    {21, -1, 2, 64, {1, 4, 8, 8, 0}},
};
int glm_arch_id() {   // 1: gfx1151, 0: other
    static const int a = [] {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); return 0; }
        return strata::kernels::gfx_arch_is_gfx1151(p.gcnArchName) ? 1 : 0;
    }();
    return a;
}
bool glm_env_cfg(const char* name, GlmMmvqCfg& c) {
    const char* e = std::getenv(name);
    if (!e || !*e) return false;
    c.pers = 0;
    return std::sscanf(e, "%d:%d:%d:%d:%d", &c.variant, &c.r, &c.nw, &c.wpe, &c.pers) >= 4;
}
GlmMmvqCfg mmvq_pick(int qtype, int K, int T) {
    GlmMmvqCfg c{};
    static GlmMmvqCfg env{};
    static const bool has_env = glm_env_cfg("GLM_MMVQ_CFG", env);
    if (has_env) return env;
    auto find = [&](const MmvqRule* b, const MmvqRule* e) {
        for (const MmvqRule* q = b; q != e; ++q)
            if ((q->qtype < 0 || q->qtype == qtype) && (q->k < 0 || q->k == K) && T >= q->t0 && T <= q->t1) { c = q->c; return true; }
        return false;
    };
    if (!(glm_arch_id() == 1 && find(std::begin(kMmvqGfx1151), std::end(kMmvqGfx1151))))
        find(std::begin(kMmvqGeneric), std::end(kMmvqGeneric));
    return c;
}

int glm_num_cus() {
    static const int n = [] {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); return 20; }
        return p.multiProcessorCount;
    }();
    return n;
}

template <int TY, int R, int NW, int WPE>
hipError_t mmvq_v1_launch(const void* W, size_t expert_bytes, const void* xq, const int32_t* ids, float* y,
                          int N, int K, int n_expert_used, int n_tokens, int x_group, int ids_s1, int pers, hipStream_t stream,
                          const void* W2 = nullptr, float* y2 = nullptr) {
    const int n_ids = n_expert_used * n_tokens;
    const int np = W2 ? 2 : 1;
    const int nrb = (N + NW * R - 1) / (NW * R);
    const char* w = static_cast<const char*>(W);
    const char* w2 = static_cast<const char*>(W2);
    const char* x = static_cast<const char*>(xq);
    if (pers > 0) {
        const dim3 grid(static_cast<uint32_t>(std::min(nrb * n_ids * np, pers * glm_num_cus())), 1u, 1u);
        if (n_ids > 1)
            hipLaunchKernelGGL((glm_mmvq_ids_v1_kernel<TY, R, NW, WPE, true, true>), grid, dim3(NW * 32), 0, stream,
                               w, expert_bytes, ids, n_ids, n_expert_used, ids_s1, x, y, K, N, x_group, w2, y2);
        else
            hipLaunchKernelGGL((glm_mmvq_ids_v1_kernel<TY, R, NW, WPE, false, true>), grid, dim3(NW * 32), 0, stream,
                               w, expert_bytes, ids, n_ids, n_expert_used, ids_s1, x, y, K, N, x_group, w2, y2);
    } else {
        const dim3 grid(static_cast<uint32_t>(nrb), static_cast<uint32_t>(n_ids * np), 1u);
        if (n_ids > 1)
            hipLaunchKernelGGL((glm_mmvq_ids_v1_kernel<TY, R, NW, WPE, true, false>), grid, dim3(NW * 32), 0, stream,
                               w, expert_bytes, ids, n_ids, n_expert_used, ids_s1, x, y, K, N, x_group, w2, y2);
        else
            hipLaunchKernelGGL((glm_mmvq_ids_v1_kernel<TY, R, NW, WPE, false, false>), grid, dim3(NW * 32), 0, stream,
                               w, expert_bytes, ids, n_ids, n_expert_used, ids_s1, x, y, K, N, x_group, w2, y2);
    }
    return hipGetLastError();
}

template <int TY>
hipError_t mmvq_v1_dispatch(const GlmMmvqCfg& c, const void* W, size_t eb, const void* xq, const int32_t* ids, float* y,
                            int N, int K, int ne, int T, int xg, int s1, hipStream_t st,
                            const void* W2 = nullptr, float* y2 = nullptr) {
#define GLM_V1(R_, NW_, WPE_) \
    if (c.r == R_ && c.nw == NW_ && c.wpe == WPE_) return mmvq_v1_launch<TY, R_, NW_, WPE_>(W, eb, xq, ids, y, N, K, ne, T, xg, s1, c.pers, st, W2, y2);
#define GLM_V1_W(R_, NW_) GLM_V1(R_, NW_, 4) GLM_V1(R_, NW_, 8) GLM_V1(R_, NW_, 12) GLM_V1(R_, NW_, 16)
    GLM_V1_W(1, 2) GLM_V1_W(1, 4) GLM_V1_W(1, 8)
    GLM_V1_W(2, 2) GLM_V1_W(2, 4) GLM_V1_W(2, 8)
    GLM_V1_W(4, 1) GLM_V1_W(4, 2) GLM_V1_W(4, 4) GLM_V1_W(4, 8)
#undef GLM_V1_W
#undef GLM_V1
    return hipErrorInvalidValue;
}
template <int TY>
hipError_t mmvq_v1_fq_dispatch(const GlmMmvqCfg& c, const void* W, const float* xf, const int32_t* ids, float* y,
                               int N, int K, int ne, int T, int s1, hipStream_t st) {
    const int n_ids = ne * T;
    const size_t blk = (TY == 22) ? sizeof(block_iq2_s) : sizeof(block_iq3_s);
    const size_t eb = static_cast<size_t>(N) * static_cast<size_t>(K / 256) * blk;
#define GLM_FQ(R_, NW_, WPE_) \
    if (c.r == R_ && c.nw == NW_ && c.wpe == WPE_) { \
        const int nrb = (N + NW_ * R_ - 1) / (NW_ * R_); \
        const dim3 grid(static_cast<uint32_t>(std::min(nrb * n_ids, c.pers * glm_num_cus())), 1u, 1u); \
        if (n_ids > 1) hipLaunchKernelGGL((glm_mmvq_ids_v1_kernel<TY, R_, NW_, WPE_, true, true, true>), grid, dim3(NW_ * 32), 0, st, \
            static_cast<const char*>(W), eb, ids, n_ids, ne, s1, reinterpret_cast<const char*>(xf), y, K, N, 1, nullptr, nullptr); \
        else hipLaunchKernelGGL((glm_mmvq_ids_v1_kernel<TY, R_, NW_, WPE_, false, true, true>), grid, dim3(NW_ * 32), 0, st, \
            static_cast<const char*>(W), eb, ids, n_ids, ne, s1, reinterpret_cast<const char*>(xf), y, K, N, 1, nullptr, nullptr); \
        return hipGetLastError(); }
    GLM_FQ(4, 4, 8) GLM_FQ(4, 2, 8) GLM_FQ(2, 8, 8) GLM_FQ(4, 8, 8)
#undef GLM_FQ
    return hipErrorNotSupported;
}
} // namespace

hipError_t glm_mmvq_ids_cfg(
    int qtype,
    const void* W,
    const void* x_q8_1,
    const int32_t* ids,
    float* y,
    int N,
    int K,
    int n_expert_used,
    int n_tokens,
    int x_group,
    int ids_s1,
    const GlmMmvqCfg* cfg,
    hipStream_t stream)
{
    if (qtype != GLM_QTYPE_IQ2_S && qtype != GLM_QTYPE_IQ3_S) return hipErrorInvalidValue;
    if (n_expert_used * n_tokens > 64 || (K % 256) != 0) return hipErrorInvalidValue;
    const GlmMmvqCfg c = cfg ? *cfg : mmvq_pick(qtype, K, n_tokens);
    const size_t blk = (qtype == GLM_QTYPE_IQ2_S) ? sizeof(block_iq2_s) : sizeof(block_iq3_s);
    const size_t expert_bytes = static_cast<size_t>(N) * static_cast<size_t>(K / 256) * blk;
    if (c.variant == 1) {
        return (qtype == GLM_QTYPE_IQ2_S)
            ? mmvq_v1_dispatch<22>(c, W, expert_bytes, x_q8_1, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, stream)
            : mmvq_v1_dispatch<21>(c, W, expert_bytes, x_q8_1, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, stream);
    }
    if (c.variant == 2) {
        return (qtype == GLM_QTYPE_IQ2_S)
            ? mmvq_v1_dispatch<122>(c, W, expert_bytes, x_q8_1, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, stream)
            : mmvq_v1_dispatch<121>(c, W, expert_bytes, x_q8_1, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, stream);
    }

    // variant 0: the WP-1.7 spike kernels (4 rows/wave, 4 waves/WG)
    constexpr int WARPS  = 4;
    constexpr int ROWS   = 4;
    constexpr int B_ROWS = WARPS * ROWS; // 16 rows/WG

    const int n_ids = n_expert_used * n_tokens;
    const dim3 block(WARPS * 32);
    const dim3 grid(
        static_cast<uint32_t>((N + B_ROWS - 1) / B_ROWS),
        static_cast<uint32_t>(n_ids),
        1u);

    if (qtype == GLM_QTYPE_IQ2_S) {
        if (n_tokens == 1) {
            hipLaunchKernelGGL(
                (glm_mmvq_ids_1vec_kernel<22, WARPS, ROWS>),
                grid, block, 0, stream,
                static_cast<const char*>(W), expert_bytes, ids, n_ids, n_expert_used, ids_s1,
                static_cast<const char*>(x_q8_1), y, K, N, x_group);
        } else {
            hipLaunchKernelGGL(
                (glm_mmvq_ids_kernel<22, WARPS, ROWS>),
                grid, block, 0, stream,
                static_cast<const char*>(W), expert_bytes, ids, n_ids, n_expert_used, ids_s1,
                static_cast<const char*>(x_q8_1), y, K, N, x_group);
        }
    } else {
        if (n_tokens == 1) {
            hipLaunchKernelGGL(
                (glm_mmvq_ids_1vec_kernel<21, WARPS, ROWS>),
                grid, block, 0, stream,
                static_cast<const char*>(W), expert_bytes, ids, n_ids, n_expert_used, ids_s1,
                static_cast<const char*>(x_q8_1), y, K, N, x_group);
        } else {
            hipLaunchKernelGGL(
                (glm_mmvq_ids_kernel<21, WARPS, ROWS>),
                grid, block, 0, stream,
                static_cast<const char*>(W), expert_bytes, ids, n_ids, n_expert_used, ids_s1,
                static_cast<const char*>(x_q8_1), y, K, N, x_group);
        }
    }
    return hipGetLastError();
}

hipError_t glm_mmvq_ids(
    int qtype,
    const void* W,
    const void* x_q8_1,
    const int32_t* ids,
    float* y,
    int N,
    int K,
    int n_expert_used,
    int n_tokens,
    int x_group,
    int ids_s1,
    hipStream_t stream)
{
    return glm_mmvq_ids_cfg(qtype, W, x_q8_1, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, nullptr, stream);
}

// gate + up (same type, shape, ids and activations) in one launch; each output row is computed exactly as
// by its own glm_mmvq_ids call.  hipErrorNotSupported (nothing launched) when the config is not a v1 one.
hipError_t glm_mmvq_ids_pair(
    int qtype, const void* W, const void* W2, const void* x_q8_1, const int32_t* ids, float* y, float* y2,
    int N, int K, int n_expert_used, int n_tokens, int x_group, int ids_s1, hipStream_t stream)
{
    if (qtype != GLM_QTYPE_IQ2_S && qtype != GLM_QTYPE_IQ3_S) return hipErrorInvalidValue;
    if (n_expert_used * n_tokens > 64 || (K % 256) != 0) return hipErrorInvalidValue;
    const GlmMmvqCfg c = mmvq_pick(qtype, K, n_tokens);
    if (c.variant != 1) return hipErrorNotSupported;
    const size_t blk = (qtype == GLM_QTYPE_IQ2_S) ? sizeof(block_iq2_s) : sizeof(block_iq3_s);
    const size_t eb = static_cast<size_t>(N) * static_cast<size_t>(K / 256) * blk;
    return (qtype == GLM_QTYPE_IQ2_S)
        ? mmvq_v1_dispatch<22>(c, W, eb, x_q8_1, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, stream, W2, y2)
        : mmvq_v1_dispatch<21>(c, W, eb, x_q8_1, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, stream, W2, y2);
}

hipError_t glm_mmvq_ids_f32(
    int qtype,
    const void* W,
    const float* x_f32,
    void* x_q8_1_scratch,
    const int32_t* ids,
    float* y,
    int N,
    int K,
    int n_expert_used,
    int n_tokens,
    int x_group,
    int ids_s1,
    hipStream_t stream)
{
    const int n_vecs = (x_group <= 1) ? n_tokens : (x_group * n_tokens);
    // fused quantization is slower at T=1 because each WG re-quantizes the whole vector: opt-in only
    static const int fq_env = [] { const char* e = std::getenv("GLM_MMVQ_FQ"); return e ? std::atoi(e) : 0; }();
    if (fq_env && x_group <= 1 && n_tokens <= 4 && K <= 4096 && (K % 256) == 0 && n_expert_used * n_tokens <= 64 &&
        (qtype == GLM_QTYPE_IQ2_S || qtype == GLM_QTYPE_IQ3_S)) {
        const GlmMmvqCfg c = mmvq_pick(qtype, K, n_tokens);
        if (c.variant == 1 && c.pers > 0) {
            const hipError_t e = (qtype == GLM_QTYPE_IQ2_S)
                ? mmvq_v1_fq_dispatch<22>(c, W, x_f32, ids, y, N, K, n_expert_used, n_tokens, ids_s1, stream)
                : mmvq_v1_fq_dispatch<21>(c, W, x_f32, ids, y, N, K, n_expert_used, n_tokens, ids_s1, stream);
            if (e != hipErrorNotSupported) return e;
        }
    }
    hipError_t err = glm_quantize_q8_1(x_f32, x_q8_1_scratch, K * n_vecs, stream);
    if (err != hipSuccess) return err;
    return glm_mmvq_ids(qtype, W, x_q8_1_scratch, ids, y, N, K, n_expert_used, n_tokens, x_group, ids_s1, stream);
}

hipError_t glm_kda_step(
    const float* q,
    const float* k,
    const float* v,
    const float* g,
    const float* beta,
    float* state,
    const int32_t* ids,
    const float* dtb,
    const float* a,
    float* dst,
    int H,
    int S_v,
    int T,
    int K_snap,
    int n_B,
    float scale,
    bool prologue,
    float pro_lb,
    float pro_l2_eps,
    int cols_per_wave,
    hipStream_t stream)
{
    if (S_v != 128) return hipErrorInvalidValue;

    const uint32_t pro_flag = prologue ? 1u : 0u;
    if (cols_per_wave <= 0) {
        // GLM HIP auto (per-arch); other targets keep v0's 2 columns per wave.
        static const int env = [] { const char* e = std::getenv("GLM_KDA_CPW"); return e ? std::atoi(e) : 0; }();
        // gfx1151: v2 16 waves x 1 column for single-token decode, v2 8 waves x 8 columns for multi-token
        // MTP verify windows
        cols_per_wave = env > 0 ? env : (glm_arch_id() == 1 ? (T / n_B <= 1 ? 11601 : 10808) : 2);
    }
    if (cols_per_wave >= 10000) {   // GLM HIP v2: 10000 + 100 * waves per WG + columns per wave
        const int nwv = (cols_per_wave - 10000) / 100, cpw = cols_per_wave % 100;
        // v3 (bitwise v2, pipelined loads) for prompt windows (T > 8 per sequence); GLM_KDA_V3=0: v2
        static const bool v3 = [] { const char* e = std::getenv("GLM_KDA_V3"); return e == nullptr || e[0] != '0'; }();
        if (v3 && T / n_B > 8 && cpw == 8 && nwv == 8) {
            const dim3 grid(static_cast<uint32_t>(H), static_cast<uint32_t>(S_v / 64), static_cast<uint32_t>(n_B));
            hipLaunchKernelGGL((glm_kda_step_v3_kernel<8, 8>), grid, dim3(256), 0, stream,
                q, k, v, g, beta, state, ids, dtb, a, dst, static_cast<uint32_t>(H), static_cast<uint32_t>(T),
                static_cast<uint32_t>(K_snap), static_cast<uint32_t>(n_B), scale, pro_flag, pro_lb, pro_l2_eps);
            return hipGetLastError();
        }
#define GLM_KDA2(CPW_, NWV_) \
        if (cpw == CPW_ && nwv == NWV_) { \
            const dim3 grid(static_cast<uint32_t>(H), static_cast<uint32_t>(S_v / (CPW_ * NWV_)), static_cast<uint32_t>(n_B)); \
            hipLaunchKernelGGL((glm_kda_step_v2_kernel<CPW_, NWV_>), grid, dim3(32 * NWV_), 0, stream, \
                q, k, v, g, beta, state, ids, dtb, a, dst, static_cast<uint32_t>(H), static_cast<uint32_t>(T), \
                static_cast<uint32_t>(K_snap), static_cast<uint32_t>(n_B), scale, pro_flag, pro_lb, pro_l2_eps); \
            return hipGetLastError(); }
        GLM_KDA2(1, 1) GLM_KDA2(2, 1) GLM_KDA2(4, 1) GLM_KDA2(1, 4) GLM_KDA2(1, 8) GLM_KDA2(1, 16)
        GLM_KDA2(2, 4) GLM_KDA2(2, 8) GLM_KDA2(2, 16) GLM_KDA2(4, 4) GLM_KDA2(4, 8) GLM_KDA2(4, 16) GLM_KDA2(8, 8)
#undef GLM_KDA2
        return hipErrorInvalidValue;
    }
    if (cols_per_wave >= 100) {   // GLM HIP v1: 100 * waves per WG + columns per wave
        const int nwv = cols_per_wave / 100, cpw = cols_per_wave % 100;
#define GLM_KDA1(CPW_, NWV_) \
        if (cpw == CPW_ && nwv == NWV_) { \
            const dim3 grid(static_cast<uint32_t>(H), static_cast<uint32_t>(S_v / (CPW_ * NWV_)), static_cast<uint32_t>(n_B)); \
            hipLaunchKernelGGL((glm_kda_step_v1_kernel<CPW_, NWV_>), grid, dim3(32 * NWV_), 0, stream, \
                q, k, v, g, beta, state, ids, dtb, a, dst, static_cast<uint32_t>(H), static_cast<uint32_t>(T), \
                static_cast<uint32_t>(K_snap), static_cast<uint32_t>(n_B), scale, pro_flag, pro_lb, pro_l2_eps); \
            return hipGetLastError(); }
        GLM_KDA1(1, 1) GLM_KDA1(2, 1) GLM_KDA1(4, 1) GLM_KDA1(1, 2) GLM_KDA1(2, 2)
        GLM_KDA1(1, 4) GLM_KDA1(1, 8) GLM_KDA1(1, 16) GLM_KDA1(2, 4) GLM_KDA1(2, 8) GLM_KDA1(2, 16)
        GLM_KDA1(4, 4) GLM_KDA1(4, 8) GLM_KDA1(4, 16) GLM_KDA1(8, 4) GLM_KDA1(8, 8) GLM_KDA1(8, 16)
#undef GLM_KDA1
        return hipErrorInvalidValue;
    }
    if (cols_per_wave == 4) {
        const uint32_t cols_per_wg = 4 * 4; // 16
        const dim3 grid(static_cast<uint32_t>(H), static_cast<uint32_t>((S_v + cols_per_wg - 1) / cols_per_wg), static_cast<uint32_t>(n_B));
        hipLaunchKernelGGL(
            (glm_kda_step_kernel<4>),
            grid, dim3(128), 0, stream,
            q, k, v, g, beta, state, ids, dtb, a, dst,
            static_cast<uint32_t>(H), static_cast<uint32_t>(S_v),
            static_cast<uint32_t>(T), static_cast<uint32_t>(K_snap),
            static_cast<uint32_t>(n_B), scale, pro_flag, pro_lb, pro_l2_eps);
    } else if (cols_per_wave == 2) {
        const uint32_t cols_per_wg = 4 * 2; // 8
        const dim3 grid(static_cast<uint32_t>(H), static_cast<uint32_t>((S_v + cols_per_wg - 1) / cols_per_wg), static_cast<uint32_t>(n_B));
        hipLaunchKernelGGL(
            (glm_kda_step_kernel<2>),
            grid, dim3(128), 0, stream,
            q, k, v, g, beta, state, ids, dtb, a, dst,
            static_cast<uint32_t>(H), static_cast<uint32_t>(S_v),
            static_cast<uint32_t>(T), static_cast<uint32_t>(K_snap),
            static_cast<uint32_t>(n_B), scale, pro_flag, pro_lb, pro_l2_eps);
    } else {
        const uint32_t cols_per_wg = 4 * 1; // 4
        const dim3 grid(static_cast<uint32_t>(H), static_cast<uint32_t>((S_v + cols_per_wg - 1) / cols_per_wg), static_cast<uint32_t>(n_B));
        hipLaunchKernelGGL(
            (glm_kda_step_kernel<1>),
            grid, dim3(128), 0, stream,
            q, k, v, g, beta, state, ids, dtb, a, dst,
            static_cast<uint32_t>(H), static_cast<uint32_t>(S_v),
            static_cast<uint32_t>(T), static_cast<uint32_t>(K_snap),
            static_cast<uint32_t>(n_B), scale, pro_flag, pro_lb, pro_l2_eps);
    }
    return hipGetLastError();
}

namespace {
// (n_kv upper bound, tokens) -> indexer config, gfx1151 tuned (gfx1151 li sweep); generic = v0.
struct LiRule { int kv_max, t0, t1; GlmLiCfg c; };
const LiRule kLiGeneric[] = { {1 << 30, 1, 1 << 30, {0, 1, 1}} };
// gfx1151 (gfx1151 graph timing): heads over 4 waves up to 4K keys (any T) and for one token; two waves
// around 8K; T >= 2 at >= 8K stays on v0 (one wave of 64 keys per WG, all heads).
const LiRule kLiGfx1151[] = {
    {4096, 1, 1 << 30, {1, 4, 1}},
    {12288, 1, 1, {1, 2, 1}},
    {1 << 30, 1, 1, {1, 4, 1}},
    {1 << 30, 2, 1 << 30, {0, 1, 1}},
};
GlmLiCfg li_pick(int n_kv, int T) {
    static GlmLiCfg env{};
    static const bool has_env = [] {
        const char* e = std::getenv("GLM_LI_CFG");
        return e && std::sscanf(e, "%d:%d:%d", &env.variant, &env.nwv, &env.tf) == 3;
    }();
    if (has_env) return env;
    auto find = [&](const LiRule* b, const LiRule* e, GlmLiCfg& c) {
        for (const LiRule* r = b; r != e; ++r)
            if (n_kv <= r->kv_max && T >= r->t0 && T <= r->t1) { c = r->c; return true; }
        return false;
    };
    GlmLiCfg c{0, 1, 1};
    if (!(glm_arch_id() == 1 && find(std::begin(kLiGfx1151), std::end(kLiGfx1151), c)))
        find(std::begin(kLiGeneric), std::end(kLiGeneric), c);
    return c;
}
template <bool F16, int NWV, int TF, bool QLDS>
void li_v1_launch(const float* q, const void* k, const float* w, float* dst, int H, int n_kv, int T, int S, hipStream_t st) {
    const dim3 grid(static_cast<uint32_t>((n_kv + 31) / 32), static_cast<uint32_t>((T + TF - 1) / TF), static_cast<uint32_t>(S));
    const size_t shm = QLDS ? static_cast<size_t>(TF) * H * (128 + 1) * sizeof(float) : 0;
    hipLaunchKernelGGL((glm_dsa_li_v1_kernel<F16, NWV, TF, QLDS>), grid, dim3(32 * NWV), shm, st,
                       reinterpret_cast<const float4*>(q), k, w, dst,
                       static_cast<uint32_t>(H), static_cast<uint32_t>(n_kv), static_cast<uint32_t>(T));
}
template <bool F16>
bool li_v1_dispatch(const GlmLiCfg& c, const float* q, const void* k, const float* w, float* dst, int H, int n_kv, int T, int S, hipStream_t st) {
#define GLM_LI(NWV_, TF_) if (c.nwv == NWV_ && c.tf == TF_) { \
        if (c.variant == 3) li_v1_launch<F16, NWV_, TF_, true>(q, k, w, dst, H, n_kv, T, S, st); \
        else li_v1_launch<F16, NWV_, TF_, false>(q, k, w, dst, H, n_kv, T, S, st); return true; }
    GLM_LI(1, 1) GLM_LI(2, 1) GLM_LI(4, 1) GLM_LI(8, 1)
    GLM_LI(1, 3) GLM_LI(2, 3) GLM_LI(4, 3) GLM_LI(8, 3)
#undef GLM_LI
    return false;
}
} // namespace

// N3 (default ON, GLM_N3=0 = the fp32 indexer kernels; output-changing): prompt-window lightning indexer as fp16 WMMA. score[t][key] =
// sum_h w[t][h] relu(q[t][h] . k[key]); q rounded to fp16 (k is the f16 pooled cache), products accumulated in fp32.
// Block: 4 tokens x 64 keys; wave w: token w / 2, keys 32 (w & 1) .. + 31 (2 tiles), all 32 heads (2 tiles).
typedef _Float16 n3h16 __attribute__((ext_vector_type(16)));
typedef float n3f8 __attribute__((ext_vector_type(8)));
constexpr int N3_RS = 136;   // halves per staged row (128 + 8)
__global__ __launch_bounds__(256) void glm_dsa_li_wmma_kernel(const float* __restrict__ q, const __half* __restrict__ kc,
                                                              const float* __restrict__ w, float* __restrict__ dst,
                                                              uint32_t n_kv, uint32_t n_tokens) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    __shared__ __attribute__((aligned(16))) _Float16 sq[4 * 32][N3_RS];
    __shared__ __attribute__((aligned(16))) _Float16 sk[64][N3_RS];
    const uint32_t k0 = blockIdx.x * 64, t0 = blockIdx.y * 4, s = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    // stage q (4 tokens x 32 heads x 128, fp32 -> fp16) and k (64 keys x 128 f16)
    for (int e = tid; e < 4 * 32 * 32; e += 256) {   // float4 units
        const int row = e >> 5, c4 = e & 31;          // row = token_local * 32 + head
        const uint32_t t = min(t0 + (uint32_t) (row >> 5), n_tokens - 1);
        const float4 v = reinterpret_cast<const float4*>(q + ((size_t) (s * n_tokens + t) * 32 + (row & 31)) * 128)[c4];
        _Float16* d = &sq[row][4 * c4];
        d[0] = (_Float16) v.x; d[1] = (_Float16) v.y; d[2] = (_Float16) v.z; d[3] = (_Float16) v.w;
    }
    for (int e = tid; e < 64 * 16; e += 256) {   // uint4 units (8 halves)
        const int row = e >> 4, c8 = e & 15;
        const uint32_t key = min(k0 + (uint32_t) row, n_kv - 1);
        *reinterpret_cast<uint4*>(&sk[row][8 * c8]) = reinterpret_cast<const uint4*>(kc + ((size_t) s * n_kv + key) * 128)[c8];
    }
    __syncthreads();
    const int tl = wv >> 1, kb = (wv & 1) * 32;
    n3f8 acc[2][2];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 2; ++b) acc[a][b] = (n3f8){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
        n3h16 A[2], B[2];
#pragma unroll
        for (int a = 0; a < 2; ++a) A[a] = *reinterpret_cast<const n3h16*>(&sq[tl * 32 + 16 * a + l16][16 * ks]);
#pragma unroll
        for (int b = 0; b < 2; ++b) B[b] = *reinterpret_cast<const n3h16*>(&sk[kb + 16 * b + l16][16 * ks]);
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int b = 0; b < 2; ++b) acc[a][b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A[a], B[b], acc[a][b]);
    }
    const uint32_t t = t0 + tl;
    if (t >= n_tokens) return;
    const float* wt = w + (size_t) (s * n_tokens + t) * 32;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        float sc = 0.0f;
#pragma unroll
        for (int a = 0; a < 2; ++a)
#pragma unroll
            for (int i = 0; i < 8; ++i) sc += fmaxf(acc[a][b][i], 0.0f) * wt[16 * a + 2 * i + hi];
        sc += __shfl_xor(sc, 16, 32);
        const uint32_t key = k0 + kb + 16 * b + l16;
        if (hi == 0 && key < n_kv) dst[(size_t) (s * n_tokens + t) * n_kv + key] = sc;
    }
#else
    (void) q; (void) kc; (void) w; (void) dst; (void) n_kv; (void) n_tokens;
#endif
}

hipError_t glm_dsa_lightning_indexer_cfg(
    const float* q,
    const void* k,
    const float* w,
    float* dst,
    int D,
    int H,
    int n_kv,
    int n_tokens,
    int n_streams,
    bool k_is_f16,
    const GlmLiCfg* cfg,
    hipStream_t stream)
{
    if (D != 128) return hipErrorInvalidValue;
    static const bool n3 = [] { const char* e = std::getenv("GLM_N3"); return !e || e[0] != '0'; }();   // default ON (GLM_N3=0 = fp32 indexer)
    if (n3 && !cfg && k_is_f16 && H == 32 && n_tokens > 8) {   // N3 (default on, output-changing)
        const dim3 grid(static_cast<uint32_t>((n_kv + 63) / 64), static_cast<uint32_t>((n_tokens + 3) / 4), static_cast<uint32_t>(n_streams));
        hipLaunchKernelGGL(glm_dsa_li_wmma_kernel, grid, dim3(256), 0, stream, q, static_cast<const __half*>(k), w, dst,
                           static_cast<uint32_t>(n_kv), static_cast<uint32_t>(n_tokens));
        return hipGetLastError();
    }
    const GlmLiCfg c = cfg ? *cfg : li_pick(n_kv, n_tokens);
    if (c.variant == 4 && k_is_f16 && (H % c.nwv) == 0) {
        const dim3 grid(static_cast<uint32_t>((n_kv + 63) / 64), static_cast<uint32_t>(n_tokens), static_cast<uint32_t>(n_streams));
#define GLM_LI4(NWV_) if (c.nwv == NWV_) { hipLaunchKernelGGL((glm_dsa_li_v4_kernel<NWV_>), grid, dim3(32 * NWV_), 0, stream, \
            reinterpret_cast<const float4*>(q), static_cast<const uint2*>(k), w, dst, static_cast<uint32_t>(H), \
            static_cast<uint32_t>(n_kv), static_cast<uint32_t>(n_tokens)); return hipGetLastError(); }
        GLM_LI4(1) GLM_LI4(2) GLM_LI4(4) GLM_LI4(8)
#undef GLM_LI4
        return hipErrorInvalidValue;
    }
    if (c.variant == 2 && H == 32) {
        const int kb = c.tf;   // variant 2: tf field = keys per wave (32 / 64 / 128)
        const int per_wg = kb * c.nwv;
        const dim3 grid(static_cast<uint32_t>((n_kv + per_wg - 1) / per_wg), static_cast<uint32_t>(n_tokens), static_cast<uint32_t>(n_streams));
#define GLM_LI2(F16_, KB_, NWV_) \
        if (k_is_f16 == F16_ && kb == KB_ && c.nwv == NWV_) { \
            hipLaunchKernelGGL((glm_dsa_li_v2_kernel<F16_, KB_, NWV_>), grid, dim3(32 * NWV_), 0, stream, \
                reinterpret_cast<const float4*>(q), k, w, dst, static_cast<uint32_t>(n_kv), static_cast<uint32_t>(n_tokens)); \
            return hipGetLastError(); }
        GLM_LI2(true, 32, 1) GLM_LI2(true, 32, 2) GLM_LI2(true, 32, 4) GLM_LI2(true, 64, 1) GLM_LI2(true, 64, 2) GLM_LI2(true, 64, 4) GLM_LI2(true, 128, 1) GLM_LI2(true, 128, 2)
        GLM_LI2(false, 32, 1) GLM_LI2(false, 32, 2) GLM_LI2(false, 32, 4) GLM_LI2(false, 64, 1) GLM_LI2(false, 64, 2) GLM_LI2(false, 64, 4) GLM_LI2(false, 128, 1) GLM_LI2(false, 128, 2)
#undef GLM_LI2
        return hipErrorInvalidValue;
    }
    if ((c.variant == 1 || c.variant == 3) && (H % (2 * c.nwv)) == 0) {
        const bool ok = k_is_f16 ? li_v1_dispatch<true>(c, q, k, w, dst, H, n_kv, n_tokens, n_streams, stream)
                                 : li_v1_dispatch<false>(c, q, k, w, dst, H, n_kv, n_tokens, n_streams, stream);
        if (!ok) return hipErrorInvalidValue;
        return hipGetLastError();
    }

    const uint32_t n_tiles = static_cast<uint32_t>((n_kv + 63) / 64);
    const dim3 grid(n_tiles, static_cast<uint32_t>(n_tokens), static_cast<uint32_t>(n_streams));
    if (k_is_f16) {
        hipLaunchKernelGGL(
            (glm_dsa_lightning_indexer_kernel<true, 1>),
            grid, dim3(64), 0, stream,
            reinterpret_cast<const float4*>(q), k, w, dst,
            static_cast<uint32_t>(H), static_cast<uint32_t>(n_kv), static_cast<uint32_t>(n_tokens));
    } else {
        hipLaunchKernelGGL(
            (glm_dsa_lightning_indexer_kernel<false, 1>),
            grid, dim3(64), 0, stream,
            reinterpret_cast<const float4*>(q), k, w, dst,
            static_cast<uint32_t>(H), static_cast<uint32_t>(n_kv), static_cast<uint32_t>(n_tokens));
    }
    return hipGetLastError();
}

hipError_t glm_dsa_lightning_indexer(
    const float* q,
    const void* k,
    const float* w,
    float* dst,
    int D,
    int H,
    int n_kv,
    int n_tokens,
    int n_streams,
    bool k_is_f16,
    hipStream_t stream)
{
    return glm_dsa_lightning_indexer_cfg(q, k, w, dst, D, H, n_kv, n_tokens, n_streams, k_is_f16, nullptr, stream);
}

namespace { int g_fa_variant = -1; }
void glm_set_fa_variant(int v) { g_fa_variant = v; }

// split-K plan shared by the mask and index-list launchers (same targets as glm_dsa_sparse_fa)
static void glm_sparse_fa_split_plan(int H, int N, int n_kv_max, uint32_t& k_num, uint32_t& per_blocks) {
    constexpr uint32_t HG = 32u, BC = 32u, MAX_SLOTS = 2304u;
    const uint32_t total_blocks = (static_cast<uint32_t>(n_kv_max) + BC - 1u) / BC;
    const uint32_t base_wgs     = (static_cast<uint32_t>(H) / HG) * static_cast<uint32_t>(N);
    static const int fa_occ = [] { const char* e = std::getenv("GLM_FA_WGS_PER_CU"); return e ? std::atoi(e) : 2; }();
    const uint32_t target_wgs = static_cast<uint32_t>(glm_num_cus()) * static_cast<uint32_t>(fa_occ);
    k_num = std::max(std::max(1u, target_wgs / std::max(1u, base_wgs)), (static_cast<uint32_t>(n_kv_max) + MAX_SLOTS - 1u) / MAX_SLOTS);
    k_num = std::min(k_num, total_blocks);
    per_blocks = (total_blocks + k_num - 1u) / k_num;
    k_num = (total_blocks + per_blocks - 1u) / per_blocks;
}

hipError_t glm_dsa_sparse_fa(
    const float* q,
    const __half* k,
    const __half* v,
    const __half* mask,
    int32_t* idx_buf,
    float* split_buf,
    float* dst,
    int D,
    int H,
    int N,
    int KV,
    int n_kv_max,
    uint32_t q_nb1,
    uint32_t q_nb2,
    uint32_t k_stride,
    uint32_t v_stride,
    uint32_t m_stride,
    float scale,
    bool v_same,
    int force_k_num,
    hipStream_t stream)
{
    constexpr uint32_t HG        = 32u;
    constexpr uint32_t BC        = 32u;
    constexpr uint32_t MAX_SLOTS = 2304u;

    if (D != 512 || (H % HG) != 0) return hipErrorInvalidValue;

    const uint32_t total_blocks = (static_cast<uint32_t>(n_kv_max) + BC - 1u) / BC;
    const uint32_t base_wgs     = (static_cast<uint32_t>(H) / HG) * static_cast<uint32_t>(N);
    // GLM HIP: the fused kernel is built for 8 waves per SIMD (<= 192 VGPRs, 2 x 58.7 KiB LDS per WGP), so two
    // 512-thread workgroups share a WGP and the splits target 2 x CUs workgroups (gfx1151: 40).
    static const int fa_occ = [] { const char* e = std::getenv("GLM_FA_WGS_PER_CU"); return e ? std::atoi(e) : 2; }();
    const uint32_t target_wgs = static_cast<uint32_t>(glm_num_cus()) *
        ((g_fa_variant >= 0 ? g_fa_variant : 1) >= 1 ? static_cast<uint32_t>(fa_occ) : 1u);

    uint32_t k_num = (force_k_num > 0)
        ? static_cast<uint32_t>(force_k_num)
        : std::max(std::max(1u, target_wgs / base_wgs), (static_cast<uint32_t>(n_kv_max) + MAX_SLOTS - 1u) / MAX_SLOTS);
    k_num = std::min(k_num, total_blocks);
    uint32_t per_blocks = (total_blocks + k_num - 1u) / k_num;
    k_num = (total_blocks + per_blocks - 1u) / per_blocks;

    static const int fa_env = [] { const char* e = std::getenv("GLM_FA_VARIANT"); return e ? std::atoi(e) : 1; }();
    const int fa_variant = g_fa_variant >= 0 ? g_fa_variant : fa_env;
    const bool fused = fa_variant >= 1 && (KV % 8) == 0 && (m_stride % 8) == 0 && KV <= 65536 &&
                       (reinterpret_cast<uintptr_t>(mask) & 15u) == 0;
    float* out_target = (k_num == 1u) ? dst : split_buf;
    const dim3 dec_grid(static_cast<uint32_t>(H) / HG, static_cast<uint32_t>(N), k_num);

    if (fused) {
        // GLM HIP: compaction inside the decode kernel (one launch fewer; idx_buf unused)
        if (v_same) {
            hipLaunchKernelGGL(
                (glm_dsa_sparse_fa_dec_kernel<true, true, 8>),
                dec_grid, dim3(512), 0, stream,
                reinterpret_cast<const float4*>(q), reinterpret_cast<const uint4*>(k), reinterpret_cast<const Half4*>(k),
                mask, idx_buf, out_target,
                static_cast<uint32_t>(N), static_cast<uint32_t>(n_kv_max), k_num, per_blocks,
                q_nb1, q_nb2, k_stride, v_stride, m_stride, static_cast<uint32_t>(H), scale, static_cast<uint32_t>(KV));
        } else {
            hipLaunchKernelGGL(
                (glm_dsa_sparse_fa_dec_kernel<false, true, 8>),
                dec_grid, dim3(512), 0, stream,
                reinterpret_cast<const float4*>(q), reinterpret_cast<const uint4*>(k), reinterpret_cast<const Half4*>(v),
                mask, idx_buf, out_target,
                static_cast<uint32_t>(N), static_cast<uint32_t>(n_kv_max), k_num, per_blocks,
                q_nb1, q_nb2, k_stride, v_stride, m_stride, static_cast<uint32_t>(H), scale, static_cast<uint32_t>(KV));
        }
    } else {
    // 1. Compact sparse mask into idx_buf
    hipLaunchKernelGGL(
        glm_dsa_sparse_compact_kernel,
        dim3(static_cast<uint32_t>(N), 1u, 1u), dim3(1024), 0, stream,
        reinterpret_cast<const uint16_t*>(mask), idx_buf,
        static_cast<uint32_t>(KV), static_cast<uint32_t>(N), 1u,
        m_stride, m_stride * static_cast<uint32_t>(N), m_stride * static_cast<uint32_t>(N),
        static_cast<uint32_t>(n_kv_max));

    // 2. Sparse decode FlashAttention
    if (v_same) {
        hipLaunchKernelGGL(
            (glm_dsa_sparse_fa_dec_kernel<true>),
            dec_grid, dim3(512), 0, stream,
            reinterpret_cast<const float4*>(q),
            reinterpret_cast<const uint4*>(k),
            reinterpret_cast<const Half4*>(k),
            mask, idx_buf, out_target,
            static_cast<uint32_t>(N), static_cast<uint32_t>(n_kv_max), k_num, per_blocks,
            q_nb1, q_nb2, k_stride, v_stride, m_stride,
            static_cast<uint32_t>(H), scale);
    } else {
        hipLaunchKernelGGL(
            (glm_dsa_sparse_fa_dec_kernel<false>),
            dec_grid, dim3(512), 0, stream,
            reinterpret_cast<const float4*>(q),
            reinterpret_cast<const uint4*>(k),
            reinterpret_cast<const Half4*>(v),
            mask, idx_buf, out_target,
            static_cast<uint32_t>(N), static_cast<uint32_t>(n_kv_max), k_num, per_blocks,
            q_nb1, q_nb2, k_stride, v_stride, m_stride,
            static_cast<uint32_t>(H), scale);
    }
    }

    // 3. Split-K reduce if k_num > 1
    if (k_num > 1u) {
        hipLaunchKernelGGL(
            glm_dsa_sparse_fa_split_k_reduce_kernel,
            dim3(static_cast<uint32_t>(H), static_cast<uint32_t>(N), 1u), dim3(128), 0, stream,
            split_buf, reinterpret_cast<float4*>(dst),
            static_cast<uint32_t>(D), static_cast<uint32_t>(H), static_cast<uint32_t>(N), k_num);
    }

    return hipGetLastError();
}


// Opt-in WMMA sparse attention (GLM_SPARSE_WMMA=1; prefill windows, the single-split case): the index-list sparse attention on
// v_wmma_f32_16x16x16_f16.  One 512-thread workgroup per (token, 64 heads): wave w = (head tile w >> 2 of 16 heads,
// head-dim quarter w & 3 of 128).  Per 16 selected rows: the latent rows go to LDS twice (rows for S^T = K Q^T, transposed
// for O^T += V^T P^T, V = K), each wave scores its quarter of the 512 dims, the four quarters are summed through LDS in a
// fixed order (all four waves hold the same bits), online softmax in log2 units, P in f16.  Same operand precision as
// glm_dsa_sparse_fa_dec_kernel (f16 Q / K / P / V, f32 accumulation); the summation order differs -> not bitwise.
namespace {
typedef _Float16 gh16 __attribute__((ext_vector_type(16)));
typedef float gf8 __attribute__((ext_vector_type(8)));
constexpr int kGwD = 512, kGwKT = 16, kGwKS = kGwD + 16;
template <int PD>   // global prefetch distance in tiles (1 or 2)
__global__ __launch_bounds__(512) void glm_sparse_fa_wmma_kernel(const float* __restrict__ q, const _Float16* __restrict__ kv,
                                                                 const int32_t* __restrict__ idx, float* __restrict__ dst,
                                                                 int H, int n_kv_max, uint32_t q_nb1, uint32_t q_nb2,
                                                                 uint32_t k_stride, float scale) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
    __shared__ __attribute__((aligned(32))) _Float16 Ks[kGwKT][kGwKS];
    __shared__ __attribute__((aligned(32))) _Float16 Vt[kGwD][kGwKT];
    __shared__ float Sx[16][8][32];
    const int t = blockIdx.x, h0 = blockIdx.y * 64;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15, hi = lane >> 4;
    const int ht = w >> 2, dq = w & 3;
    const unsigned sel_lo = hi ? 0x01000504u : 0x05040100u, sel_hi = hi ? 0x03020706u : 0x07060302u;
    const int32_t* ix = idx + (size_t) t * n_kv_max;
    const int head = h0 + ht * 16 + l16;
    gh16 qf[8];
    {
        const float sc = scale * 1.44269504088896340736f;
        const float* qr = q + (size_t) t * q_nb1 + (size_t) head * q_nb2 + dq * 128;
#pragma unroll
        for (int ds = 0; ds < 8; ++ds)
#pragma unroll
            for (int e4 = 0; e4 < 4; ++e4) {
                const float4 a = *(const float4*) (qr + ds * 16 + e4 * 4);
                qf[ds][e4 * 4 + 0] = (_Float16) (a.x * sc);
                qf[ds][e4 * 4 + 1] = (_Float16) (a.y * sc);
                qf[ds][e4 * 4 + 2] = (_Float16) (a.z * sc);
                qf[ds][e4 * 4 + 3] = (_Float16) (a.w * sc);
            }
    }
    gf8 o[8];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) o[dt][i] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;
    // loader: thread = (key pair kp = tid >> 6: keys 2 kp, 2 kp + 1; 8-dim unit u = tid & 63)
    const int kp = tid >> 6, u = tid & 63;
    uint4 r0, r1, n0 = make_uint4(0u, 0u, 0u, 0u), n1 = n0;
    auto fetch = [&](int s0, uint4& a0, uint4& a1) {
        const int i0 = s0 + 2 * kp < n_kv_max ? ix[s0 + 2 * kp] : -1;
        const int i1 = s0 + 2 * kp + 1 < n_kv_max ? ix[s0 + 2 * kp + 1] : -1;
        a0 = i0 >= 0 ? *(const uint4*) (kv + (size_t) i0 * k_stride + u * 8) : make_uint4(0u, 0u, 0u, 0u);
        a1 = i1 >= 0 ? *(const uint4*) (kv + (size_t) i1 * k_stride + u * 8) : make_uint4(0u, 0u, 0u, 0u);
    };
    const int n_tiles = (n_kv_max + kGwKT - 1) / kGwKT;
    fetch(0, r0, r1);
    if (PD == 2 && n_tiles > 1) fetch(kGwKT, n0, n1);
    for (int tile = 0; tile < n_tiles; ++tile) {
        const int s0 = tile * kGwKT;
        if (ix[s0] < 0) break;   // the list is ascending rows, then -1 padding (k_dsa_select): the rest is padding
        __syncthreads();   // the previous tile's LDS reads are done
        *(uint4*) &Ks[2 * kp][u * 8] = r0;
        *(uint4*) &Ks[2 * kp + 1][u * 8] = r1;
        {
            const unsigned a[4] = {r0.x, r0.y, r0.z, r0.w}, b[4] = {r1.x, r1.y, r1.z, r1.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {   // dims u 8 + 2 e, + 1: (key 2 kp, key 2 kp + 1) pairs
                *(unsigned*) &Vt[u * 8 + 2 * e][2 * kp] = __builtin_amdgcn_perm(b[e], a[e], 0x05040100u);
                *(unsigned*) &Vt[u * 8 + 2 * e + 1][2 * kp] = __builtin_amdgcn_perm(b[e], a[e], 0x07060302u);
            }
        }
        __syncthreads();
        // keys of this tile valid? (padding is -1); the whole block reads the same 16 entries
        bool keyok[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) keyok[i] = s0 + 2 * i + hi < n_kv_max && ix[s0 + 2 * i + hi] >= 0;
        if constexpr (PD == 2) {
            r0 = n0; r1 = n1;
            if (tile + 2 < n_tiles) fetch(s0 + 2 * kGwKT, n0, n1);
        } else if (tile + 1 < n_tiles) fetch(s0 + kGwKT, r0, r1);
        gf8 sacc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int ds = 0; ds < 8; ++ds) {
            const gh16 ka = *(const gh16*) &Ks[l16][dq * 128 + ds * 16];
            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ka, qf[ds], sacc);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) Sx[w][i][lane] = sacc[i];
        __syncthreads();
        float sv[8], mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = ((Sx[ht * 4 + 0][i][lane] + Sx[ht * 4 + 1][i][lane]) + Sx[ht * 4 + 2][i][lane]) + Sx[ht * 4 + 3][i][lane];
            sv[i] = keyok[i] ? v : -INFINITY;
            mx = fmaxf(mx, sv[i]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn = fmaxf(m_run, mx);
        const float mb = mn == -INFINITY ? 0.f : mn;
        const float alpha = __builtin_amdgcn_exp2f(m_run - mb);
        float p[8], ls = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) { p[i] = __builtin_amdgcn_exp2f(sv[i] - mb); ls += p[i]; }
        ls += __shfl_xor(ls, 16, 32);
        l_run = l_run * alpha + ls;
        m_run = mn;
        union { gh16 v; unsigned u[8]; } pb;
#pragma unroll
        for (int mm = 0; mm < 4; ++mm) {
            union { _Float16 h[2]; unsigned u; } own;
            own.h[0] = (_Float16) p[2 * mm];
            own.h[1] = (_Float16) p[2 * mm + 1];
            const unsigned par = (unsigned) __shfl_xor((int) own.u, 16, 32);
            pb.u[2 * mm] = __builtin_amdgcn_perm(par, own.u, sel_lo);
            pb.u[2 * mm + 1] = __builtin_amdgcn_perm(par, own.u, sel_hi);
        }
        const bool rescale = __any(alpha != 1.f);
#pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            if (rescale)
#pragma unroll
                for (int i = 0; i < 8; ++i) o[dt][i] *= alpha;
            const gh16 va = *(const gh16*) &Vt[dq * 128 + dt * 16 + l16][0];
            o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, pb.v, o[dt]);
        }
    }
    const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
    float* orow = dst + ((size_t) t * H + head) * kGwD + dq * 128;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt)
#pragma unroll
        for (int i = 0; i < 8; ++i) orow[dt * 16 + 2 * i + hi] = o[dt][i] * inv;
#else
    (void) q; (void) kv; (void) idx; (void) dst; (void) H; (void) n_kv_max; (void) q_nb1; (void) q_nb2; (void) k_stride;
    (void) scale;
    __builtin_trap();
#endif
}
int g_sparse_wmma = -1;   // test hook (talus_glm_sparse_wmma), -1 = GLM_SPARSE_WMMA
int g_sparse_skip = -1;   // test hook (talus_glm_sparse_skip), -1 = GLM_SPARSE_SKIP
int glm_sparse_wmma_on() {   // 0 off, 1 = WMMA kernel (prefetch 1 tile), 2 = prefetch 2 tiles
    static const int env = [] { const char* e = std::getenv("GLM_SPARSE_WMMA"); const int v = e ? std::atoi(e) : 0; return v == 1 || v == 2 ? v : 0; }();
    return g_sparse_wmma >= 0 ? g_sparse_wmma : env;
}
bool glm_is_gfx11() {
    static const bool ok = [] {
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return false;
        return strata::kernels::gfx_arch_is_gfx11_wmma(p.gcnArchName);
    }();
    return ok;
}
}  // namespace

// Two-pass WMMA sparse prompt attention (default on; GLM_SPARSE_2P=0 = kill switch; prompt windows N >= GLM_SPARSE_2P_MIN, default 9 = past
// the decode / verify tile): one 512-thread workgroup per (token, 64 heads), all heads of the token as the GEMM M side
// (the selected latent rows are shared by the heads).
//   pass 1: S^T = K_sel Q^T on v_wmma_f32_16x16x16_f16 (Q f16 x scale x log2e in LDS, K rows straight from global into
//           the A operand; wave w scores key tiles w, w + 16, ...), S (fp32, log2 units) to a per-workgroup global
//           scratch slot [64 heads][2080 keys] (the engine passes n_kv_max = top-k + kpool - 1 = 2051, rounded up to the 32-key tile), exact row max per head (no online rescaling);
//   pass 2: P = exp2(S - max) in f16 (the numerics change vs the fdot2 kernel: same f16 operands, exact max instead
//           of a running max, different summation order), O^T = V^T P^T (V = K) with V^T and P staged in LDS 32 keys at
//           a time, wave w owns dims [32 w, 32 w + 32) for all 64 heads; O = O / sum(p) (fp32 sum).
// Scratch slots come from a small pool (2 x CUs, <= 1 resident workgroup per CU at 64 KiB LDS) taken with an atomic
// lock per slot. The index list must be ascending rows then -1 padding (k_dsa_select), as GLM_SPARSE_SKIP assumes.
namespace {
constexpr int kG2D = 512, kG2HB = 64, kG2SMAX = 2080, kG2KT = 32, kG2VS = kG2KT + 8;   // V^T / P row stride (halfs)
__device__ __forceinline__ void g2p_xchg8(const gf8& c, bool hi, float (&d)[8]) {
    // lane holds rows 2 i + hi (i 0..7) of a 16-row column; returns rows 8 hi .. 8 hi + 7 contiguous
    float send[4], recv[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) send[m] = hi ? c[m] : c[4 + m];
#pragma unroll
    for (int m = 0; m < 4; ++m) recv[m] = __shfl_xor(send[m], 16, 32);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        d[2 * m]     = hi ? recv[m] : c[m];
        d[2 * m + 1] = hi ? c[4 + m] : recv[m];
    }
}
__global__ __launch_bounds__(512) void glm_sparse_fa_2p_kernel(const float* __restrict__ q, const _Float16* __restrict__ kv,
                                                               const int32_t* __restrict__ idx, float* __restrict__ dst,
                                                               int H, int n_kv_max, uint32_t q_nb1, uint32_t q_nb2,
                                                               uint32_t k_stride, float scale, float* __restrict__ pool,
                                                               int* __restrict__ locks, int nslot) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
    __shared__ __attribute__((aligned(32))) _Float16 lds[kG2HB * kG2D];   // 64 KiB: Q (pass 1), V^T + P + stats (pass 2)
    float* ldsf = reinterpret_cast<float*>(lds);
    const int t = blockIdx.x, h0 = blockIdx.y * kG2HB;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, l16 = lane & 15;
    const bool hi = lane >= 16;
    const int32_t* ix = idx + (size_t) t * n_kv_max;
    // valid rows: the list is ascending then -1 padded -> first padding slot by bisection (uniform scalar loads)
    int nv;
    {
        int lo = 0, up = n_kv_max;
        while (lo < up) {
            const int mid = (lo + up) >> 1;
            if (ix[mid] >= 0) lo = mid + 1; else up = mid;
        }
        nv = lo;
    }
    // scratch slot
    if (tid == 0) {
        int s = (int) ((blockIdx.x * gridDim.y + blockIdx.y) % (unsigned) nslot);
        while (atomicCAS(&locks[s], 0, 1) != 0) s = s + 1 == nslot ? 0 : s + 1;
        reinterpret_cast<int*>(lds)[0] = s;
    }
    __syncthreads();
    const int slot = reinterpret_cast<const int*>(lds)[0];
    __threadfence();
    __syncthreads();
    float* Sg = pool + (size_t) slot * kG2HB * kG2SMAX;
    // Q -> LDS f16 (x scale x log2e), row h, 16-half chunk c stored at chunk c ^ (h & 15)
    {
        const float sc = scale * 1.44269504088896340736f;
#pragma unroll 4
        for (int r = 0; r < kG2HB * kG2D / 4 / 512; ++r) {
            const int e = (r * 512 + tid) * 4, h = e / kG2D, d = e % kG2D;
            const float4 a = *reinterpret_cast<const float4*>(q + (size_t) t * q_nb1 + (size_t) (h0 + h) * q_nb2 + d);
            const int c = (d >> 4) ^ (h & 15);
            _Float16* p = &lds[h * kG2D + c * 16 + (d & 15)];
            p[0] = (_Float16) (a.x * sc); p[1] = (_Float16) (a.y * sc); p[2] = (_Float16) (a.z * sc); p[3] = (_Float16) (a.w * sc);
        }
    }
    __syncthreads();
    // ---- pass 1: scores
    float mx[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    const int n_tiles = (nv + 15) >> 4;
    for (int tile = w; tile < n_tiles; tile += 16) {
        const int s0 = tile * 16, ks = s0 + l16;
        const int row = ks < nv ? ix[ks] : -1;
        // padding slots (keys >= nv) read row 0 (always valid memory): their scores are replaced by -inf below, so no
        // masking of the loaded values is needed (a select between loads would become a flat load through a pointer)
        const uint4* kr = reinterpret_cast<const uint4*>(kv + (size_t) (row >= 0 ? row : 0) * k_stride);
        gf8 acc[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.f;
        auto kload = [&](uint4 (&bf)[4], int c) {
#pragma unroll
            for (int u = 0; u < 4; ++u) bf[u] = kr[c * 4 + u];
        };
        auto kmma = [&](const uint4 (&bf)[4], int c) {   // 32 dims: chunks 2 c, 2 c + 1 of 16
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                union { uint4 u[2]; gh16 v; } a;
                a.u[0] = bf[2 * s]; a.u[1] = bf[2 * s + 1];
                const int ch = c * 2 + s;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int h = 16 * j + l16;
                    const gh16 b = *reinterpret_cast<const gh16*>(&lds[h * kG2D + ((ch ^ l16) * 16)]);
                    acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a.v, b, acc[j]);
                }
            }
        };
        // 16 chunks of 32 dims, ping-pong buffers: the next chunk's loads are in flight during this chunk's WMMAs
        uint4 ba[4], bb[4];
        kload(ba, 0);
#pragma unroll 1
        for (int c = 0; c < 16; c += 2) {
            kload(bb, c + 1);
            kmma(ba, c);
            if (c + 2 < 16) kload(ba, c + 2);
            kmma(bb, c + 1);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (s0 + 2 * i + (hi ? 1 : 0) >= nv) acc[j][i] = -INFINITY;
                mx[j] = fmaxf(mx[j], acc[j][i]);
            }
            float d[8];
            g2p_xchg8(acc[j], hi, d);
            float4* o = reinterpret_cast<float4*>(Sg + (size_t) (16 * j + l16) * kG2SMAX + s0 + (hi ? 8 : 0));
            o[0] = make_float4(d[0], d[1], d[2], d[3]);
            o[1] = make_float4(d[4], d[5], d[6], d[7]);
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) mx[j] = fmaxf(mx[j], __shfl_xor(mx[j], 16, 32));
    __threadfence();   // S stores visible (and no stale L0 lines) for the other waves' pass-2 reads
    __syncthreads();   // all Q reads done: LDS reusable
    float* mred = ldsf;                       // [16 waves][64]
    float* mfin = ldsf + 16 * kG2HB;          // [64]
    float* linv = mfin + kG2HB;               // [64]
    _Float16* Vt = lds + 4096;                // byte offset 8 KiB: [512 dims][kG2VS]
    _Float16* Pt = Vt + kG2D * kG2VS;         // [64 heads][kG2VS]
    int32_t* ixs = reinterpret_cast<int32_t*>(Pt + kG2HB * kG2VS);   // [kG2SMAX] the token's index list (pass 2)
    for (int i = tid; i < kG2SMAX; i += 512) ixs[i] = i < nv ? ix[i] : 0;   // padding -> row 0 (p = 0 there)
    if (!hi) {
#pragma unroll
        for (int j = 0; j < 4; ++j) mred[w * kG2HB + 16 * j + l16] = mx[j];
    }
    __syncthreads();
    if (tid < kG2HB) {
        float m = -INFINITY;
        for (int ww = 0; ww < 16; ++ww) m = fmaxf(m, mred[ww * kG2HB + tid]);
        mfin[tid] = m == -INFINITY ? 0.f : m;
    }
    __syncthreads();
    // ---- pass 2: P V
    // loader: lane -> key pair kp = l16 (keys 2 kp, 2 kp + 1), dim group dg = 2 w + hi (16 dims)
    const int kp = l16, dg = 2 * w + (hi ? 1 : 0);
    // P producer: head ph = tid >> 3, keys 4 (tid & 7) .. + 3 of the step
    const int ph = tid >> 3, pk = (tid & 7) * 4;
    const float pm = mfin[ph];
    float lsum = 0.f;
    gf8 o[2][4];
#pragma unroll
    for (int dt = 0; dt < 2; ++dt)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[dt][j][i] = 0.f;
    const int n_steps = (nv + kG2KT - 1) / kG2KT;
    uint4 va[2], vb[2];
    float4 sv = make_float4(0.f, 0.f, 0.f, 0.f);
    auto fetch = [&](int s0) {
        // keys >= nv read row 0 (finite data) with p = 0: no masking needed
        const int2 rr = *reinterpret_cast<const int2*>(&ixs[s0 + 2 * kp]);
        const uint4* p0 = reinterpret_cast<const uint4*>(kv + (size_t) rr.x * k_stride + dg * 16);
        const uint4* p1 = reinterpret_cast<const uint4*>(kv + (size_t) rr.y * k_stride + dg * 16);
        va[0] = p0[0]; va[1] = p0[1];
        vb[0] = p1[0]; vb[1] = p1[1];
        sv = *reinterpret_cast<const float4*>(Sg + (size_t) ph * kG2SMAX + s0 + pk);
    };
    if (n_steps > 0) fetch(0);
    for (int st = 0; st < n_steps; ++st) {
        const int s0 = st * kG2KT;
        if (st > 0) __syncthreads();   // previous step's LDS reads done
        {
            const unsigned a[8] = {va[0].x, va[0].y, va[0].z, va[0].w, va[1].x, va[1].y, va[1].z, va[1].w};
            const unsigned b[8] = {vb[0].x, vb[0].y, vb[0].z, vb[0].w, vb[1].x, vb[1].y, vb[1].z, vb[1].w};
#pragma unroll
            for (int e = 0; e < 8; ++e) {   // dims dg 16 + 2 e, + 1: (key 2 kp, key 2 kp + 1)
                *reinterpret_cast<unsigned*>(&Vt[(dg * 16 + 2 * e) * kG2VS + 2 * kp]) = __builtin_amdgcn_perm(b[e], a[e], 0x05040100u);
                *reinterpret_cast<unsigned*>(&Vt[(dg * 16 + 2 * e + 1) * kG2VS + 2 * kp]) = __builtin_amdgcn_perm(b[e], a[e], 0x07060302u);
            }
            const float s4[4] = {sv.x, sv.y, sv.z, sv.w};
            _Float16 p4[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float p = s0 + pk + m < nv ? __builtin_amdgcn_exp2f(s4[m] - pm) : 0.f;
                lsum += p;
                p4[m] = (_Float16) p;
            }
            union { _Float16 h[4]; uint2 u; } pp;
            pp.h[0] = p4[0]; pp.h[1] = p4[1]; pp.h[2] = p4[2]; pp.h[3] = p4[3];
            *reinterpret_cast<uint2*>(&Pt[ph * kG2VS + pk]) = pp.u;
        }
        __syncthreads();
        if (st + 1 < n_steps) fetch(s0 + kG2KT);
#pragma unroll 1
        for (int kc = 0; kc < 2; ++kc) {
            gh16 av[2], bv[4];
#pragma unroll
            for (int dt = 0; dt < 2; ++dt) av[dt] = *reinterpret_cast<const gh16*>(&Vt[((2 * w + dt) * 16 + l16) * kG2VS + kc * 16]);
#pragma unroll
            for (int j = 0; j < 4; ++j) bv[j] = *reinterpret_cast<const gh16*>(&Pt[(16 * j + l16) * kG2VS + kc * 16]);
#pragma unroll
            for (int dt = 0; dt < 2; ++dt)
#pragma unroll
                for (int j = 0; j < 4; ++j) o[dt][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av[dt], bv[j], o[dt][j]);
        }
    }
    // row sums: the 8 producer threads of a head are 8 consecutive lanes
    lsum += __shfl_xor(lsum, 1, 32);
    lsum += __shfl_xor(lsum, 2, 32);
    lsum += __shfl_xor(lsum, 4, 32);
    if ((tid & 7) == 0) linv[ph] = lsum > 0.f ? 1.f / lsum : 0.f;
    __syncthreads();
    if (tid == 0) { __threadfence(); atomicExch(&locks[slot], 0); }   // all S reads of this workgroup are done
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int h = 16 * j + l16;
        const float iv = linv[h];
        float* orow = dst + ((size_t) t * H + h0 + h) * kG2D;
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
            float d[8];
            g2p_xchg8(o[dt][j], hi, d);
            float4* op = reinterpret_cast<float4*>(orow + (2 * w + dt) * 16 + (hi ? 8 : 0));
            op[0] = make_float4(d[0] * iv, d[1] * iv, d[2] * iv, d[3] * iv);
            op[1] = make_float4(d[4] * iv, d[5] * iv, d[6] * iv, d[7] * iv);
        }
    }
#else
    (void) q; (void) kv; (void) idx; (void) dst; (void) H; (void) n_kv_max; (void) q_nb1; (void) q_nb2; (void) k_stride;
    (void) scale; (void) pool; (void) locks; (void) nslot;
    __builtin_trap();
#endif
}
}  // namespace
namespace {
int g_sparse_2p = -1;   // test hook (talus_glm_sparse_2p), -1 = GLM_SPARSE_2P
int glm_sparse_2p_on() {
    // default ON; GLM_SPARSE_2P=0 = kill switch
    static const int env = [] { const char* e = std::getenv("GLM_SPARSE_2P"); return e && std::atoi(e) == 0 ? 0 : 1; }();
    return g_sparse_2p >= 0 ? g_sparse_2p : env;
}
int glm_sparse_2p_min() {   // smallest window (tokens per call) that takes the two-pass kernel: past the decode tile (8)
    static const int v = [] { const char* e = std::getenv("GLM_SPARSE_2P_MIN"); return e ? std::max(1, std::atoi(e)) : 9; }();
    return v;
}
struct G2pPool { float* pool = nullptr; int* locks = nullptr; int nslot = 0; bool tried = false; };
// The scratch pool: one [64][2080] fp32 slot per possibly resident workgroup (64 KiB LDS -> at most 2 per WGP;
// multiProcessorCount reports WGPs on gfx11), x2 margin. Allocated once, outside stream capture.
G2pPool* g2p_pool(hipStream_t stream) {
    static G2pPool p;
    if (p.tried) return p.pool ? &p : nullptr;
    hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &cs) != hipSuccess) { (void) hipGetLastError(); return nullptr; }
    if (cs != hipStreamCaptureStatusNone) return nullptr;
    p.tried = true;
    const int nslot = 4 * std::max(1, glm_num_cus());
    const size_t bytes = (size_t) nslot * kG2HB * kG2SMAX * sizeof(float);
    if (hipMalloc(&p.pool, bytes) != hipSuccess || hipMalloc(&p.locks, (size_t) nslot * sizeof(int)) != hipSuccess ||
        hipMemset(p.locks, 0, (size_t) nslot * sizeof(int)) != hipSuccess) {
        (void) hipGetLastError();
        if (p.pool) (void) hipFree(p.pool);
        if (p.locks) (void) hipFree(p.locks);
        p.pool = nullptr; p.locks = nullptr;
        std::fprintf(stderr, "glm: GLM_SPARSE_2P scratch allocation failed (%zu MiB): fdot2 kernel kept\n", bytes >> 20);
        return nullptr;
    }
    p.nslot = nslot;
    std::fprintf(stderr, "glm: GLM_SPARSE_2P two-pass WMMA sparse prompt attention active (windows >= %d tokens, %d scratch slots, %zu MiB)\n",
                 glm_sparse_2p_min(), nslot, bytes >> 20);
    return &p;
}
}  // namespace

// Talus GLM port: sparse decode attention from a caller-built row-index list (idx [N][n_kv_max], ascending rows,
// -1 padded), e.g. the indexer top-k expansion. The mask is not scanned: data_m is a zero row read at the selected
// rows (m_stride 0). Same tile kernel and split-K reduce as the 3-launch path.
hipError_t glm_dsa_sparse_fa_idx(
    const float* q, const __half* kv, const __half* zero_mask, const int32_t* idx, float* split_buf, float* dst,
    int D, int H, int N, int n_kv_max, uint32_t q_nb1, uint32_t q_nb2, uint32_t k_stride, float scale,
    hipStream_t stream)
{
    constexpr uint32_t HG = 32u, BC = 32u, MAX_SLOTS = 2304u;
    if (D != 512 || (H % HG) != 0 || n_kv_max <= 0 || static_cast<uint32_t>(n_kv_max) > MAX_SLOTS * 64u) return hipErrorInvalidValue;
    uint32_t k_num, per_blocks;
    if (glm_sparse_2p_on() && N >= glm_sparse_2p_min() && D == kG2D && H % kG2HB == 0 && n_kv_max <= kG2SMAX &&
        glm_is_gfx11() && (k_stride % 8u) == 0 && (q_nb1 % 4u) == 0 && (q_nb2 % 4u) == 0 &&
        (reinterpret_cast<uintptr_t>(kv) & 15u) == 0 && (reinterpret_cast<uintptr_t>(q) & 15u) == 0 &&
        (reinterpret_cast<uintptr_t>(dst) & 15u) == 0) {
        if (G2pPool* p = g2p_pool(stream)) {
            hipLaunchKernelGGL(glm_sparse_fa_2p_kernel, dim3(static_cast<uint32_t>(N), static_cast<uint32_t>(H) / kG2HB),
                dim3(512), 0, stream, q, reinterpret_cast<const _Float16*>(kv), idx, dst, H, n_kv_max, q_nb1, q_nb2,
                k_stride, scale, p->pool, p->locks, p->nslot);
            return hipGetLastError();
        }
    }
    glm_sparse_fa_split_plan(H, N, n_kv_max, k_num, per_blocks);
    if (glm_sparse_wmma_on() && k_num == 1u && D == kGwD && H % 64 == 0 && glm_is_gfx11() &&
        (k_stride % 8u) == 0 && (q_nb1 % 4u) == 0 && (q_nb2 % 4u) == 0) {
        hipLaunchKernelGGL(glm_sparse_wmma_on() == 2 ? glm_sparse_fa_wmma_kernel<2> : glm_sparse_fa_wmma_kernel<1>, dim3(static_cast<uint32_t>(N), static_cast<uint32_t>(H) / 64u),
            dim3(512), 0, stream, q, reinterpret_cast<const _Float16*>(kv), idx, dst, H, n_kv_max, q_nb1, q_nb2, k_stride,
            scale);
        return hipGetLastError();
    }
    float* out_target = (k_num == 1u) ? dst : split_buf;
    const dim3 dec_grid(static_cast<uint32_t>(H) / HG, static_cast<uint32_t>(N), k_num);
    static const uint32_t skip_pad = [] { const char* e = std::getenv("GLM_SPARSE_SKIP"); return (!e || std::atoi(e) != 0) ? 1u : 0u; }();
    hipLaunchKernelGGL((glm_dsa_sparse_fa_dec_kernel<true>), dec_grid, dim3(512), 0, stream,
        reinterpret_cast<const float4*>(q), reinterpret_cast<const uint4*>(kv), reinterpret_cast<const Half4*>(kv),
        zero_mask, idx, out_target, static_cast<uint32_t>(N), static_cast<uint32_t>(n_kv_max), k_num, per_blocks,
        q_nb1, q_nb2, k_stride, k_stride, 0u, static_cast<uint32_t>(H), scale, 0u,
        g_sparse_skip >= 0 ? static_cast<uint32_t>(g_sparse_skip) : skip_pad);
    if (k_num > 1u) {
        hipLaunchKernelGGL(glm_dsa_sparse_fa_split_k_reduce_kernel,
            dim3(static_cast<uint32_t>(H), static_cast<uint32_t>(N), 1u), dim3(128), 0, stream,
            split_buf, reinterpret_cast<float4*>(dst), static_cast<uint32_t>(D), static_cast<uint32_t>(H),
            static_cast<uint32_t>(N), k_num);
    }
    return hipGetLastError();
}

size_t glm_dsa_sparse_fa_split_bytes(int D, int H, int N, int n_kv_max) {
    uint32_t k_num, per_blocks;
    glm_sparse_fa_split_plan(H, N, n_kv_max, k_num, per_blocks);
    return k_num > 1u ? static_cast<size_t>(D * H + 2 * H) * k_num * static_cast<size_t>(N) * sizeof(float) : 0;
}

static int g_grouped_ver = -1;
void glm_set_grouped_ver(int v) { g_grouped_ver = v; }

hipError_t glm_mmvq_grouped(int qtype, const void* W, size_t expert_bytes, int n_experts, const int32_t* bounds,
                            const void* x_q8_1, const int32_t* ids_dst, float* y, int N, int K, int ld_y,
                            hipStream_t stream) {
    if ((qtype != GLM_QTYPE_IQ2_S && qtype != GLM_QTYPE_IQ3_S) || (K % 256) != 0) return hipErrorInvalidValue;
    // default on gfx1151 (512-token chunks): IQ2_S on the matrix cores (v3), IQ3_S on the
    // dp4a kernel with LDS-staged rows (v2); v4 (coalesced unpack through LDS) is opt-in
    static const int ver_env = [] { const char* e = std::getenv("GLM_GROUPED_VER"); return e ? std::atoi(e) : 0; }();
    int ver = g_grouped_ver >= 0 ? g_grouped_ver : ver_env;
    if (ver == 0) ver = (qtype == GLM_QTYPE_IQ2_S) ? 3 : 2;
    if (ver >= 4 && glm_arch_id() == 1) {
        const dim3 grid(static_cast<uint32_t>((N + 127) / 128), static_cast<uint32_t>(n_experts));
        if (qtype == GLM_QTYPE_IQ2_S) {
            hipLaunchKernelGGL((glm_grouped_wmma_v4_kernel<22>), grid, dim3(256), 0, stream, static_cast<const char*>(W),
                               expert_bytes, bounds, static_cast<const char*>(x_q8_1), ids_dst, y, K, N, ld_y);
        } else {
            hipLaunchKernelGGL((glm_grouped_wmma_v4_kernel<21>), grid, dim3(256), 0, stream, static_cast<const char*>(W),
                               expert_bytes, bounds, static_cast<const char*>(x_q8_1), ids_dst, y, K, N, ld_y);
        }
        return hipGetLastError();
    }
    if (ver >= 3 && glm_arch_id() == 1) {
        const dim3 grid(static_cast<uint32_t>((N + 127) / 128), static_cast<uint32_t>(n_experts));
        if (qtype == GLM_QTYPE_IQ2_S) {
            hipLaunchKernelGGL((glm_grouped_wmma_kernel<22>), grid, dim3(256), 0, stream, static_cast<const char*>(W),
                               expert_bytes, bounds, static_cast<const char*>(x_q8_1), ids_dst, y, K, N, ld_y);
        } else {
            hipLaunchKernelGGL((glm_grouped_wmma_kernel<21>), grid, dim3(256), 0, stream, static_cast<const char*>(W),
                               expert_bytes, bounds, static_cast<const char*>(x_q8_1), ids_dst, y, K, N, ld_y);
        }
        return hipGetLastError();
    }
    static const int rpw = [] { const char* e = std::getenv("GLM_GROUPED_ROWS"); return e ? std::atoi(e) : 256; }();
    if (ver >= 2) {
        constexpr int WARPS = 8, ROWS = 4, NT = 16;
        const int rows_per_wg = std::max(WARPS * ROWS, (rpw / (WARPS * ROWS)) * (WARPS * ROWS));
        const dim3 grid(static_cast<uint32_t>((N + rows_per_wg - 1) / rows_per_wg), static_cast<uint32_t>(n_experts));
        if (qtype == GLM_QTYPE_IQ2_S) {
            hipLaunchKernelGGL((glm_mmvq_grouped_v2_kernel<22, NT, WARPS, ROWS>), grid, dim3(WARPS * 32), 0, stream,
                               static_cast<const char*>(W), expert_bytes, bounds, static_cast<const char*>(x_q8_1),
                               ids_dst, y, K, N, ld_y, rows_per_wg);
        } else {
            hipLaunchKernelGGL((glm_mmvq_grouped_v2_kernel<21, NT, WARPS, ROWS>), grid, dim3(WARPS * 32), 0, stream,
                               static_cast<const char*>(W), expert_bytes, bounds, static_cast<const char*>(x_q8_1),
                               ids_dst, y, K, N, ld_y, rows_per_wg);
        }
        return hipGetLastError();
    }
    constexpr int WARPS = 8, ROWS = 4, NT = 8;
    const int rows_per_wg = std::max(WARPS * ROWS, (rpw / (WARPS * ROWS)) * (WARPS * ROWS));
    const dim3 grid(static_cast<uint32_t>((N + rows_per_wg - 1) / rows_per_wg), static_cast<uint32_t>(n_experts));
    if (qtype == GLM_QTYPE_IQ2_S) {
        hipLaunchKernelGGL((glm_mmvq_grouped_kernel<22, NT, WARPS, ROWS>), grid, dim3(WARPS * 32), 0, stream,
                           static_cast<const char*>(W), expert_bytes, bounds, static_cast<const char*>(x_q8_1), ids_dst,
                           y, K, N, ld_y, rows_per_wg);
    } else {
        hipLaunchKernelGGL((glm_mmvq_grouped_kernel<21, NT, WARPS, ROWS>), grid, dim3(WARPS * 32), 0, stream,
                           static_cast<const char*>(W), expert_bytes, bounds, static_cast<const char*>(x_q8_1), ids_dst,
                           y, K, N, ld_y, rows_per_wg);
    }
    return hipGetLastError();
}

} // namespace glm_spike
} // namespace talus

// test hook for the opt-in WMMA sparse attention (1 / 2 on, 0 off, -1 = GLM_SPARSE_WMMA)
extern "C" void talus_glm_sparse_wmma(int v) { talus::glm_spike::g_sparse_wmma = v; }
extern "C" void talus_glm_sparse_skip(int v) { talus::glm_spike::g_sparse_skip = v; }
// test hook for the opt-in two-pass sparse prompt attention (1 on, 0 off, -1 = GLM_SPARSE_2P)
extern "C" void talus_glm_sparse_2p(int v) { talus::glm_spike::g_sparse_2p = v; }
