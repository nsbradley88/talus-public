#include "strata/kernels/mimo_kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "talus_iq_split.cuh"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {

// ============================================================================
// 0. Fast Q8_1 input quantization (1 wave32 per 32-element block)
// ============================================================================

__global__ __launch_bounds__(256)
void mimo_quantize_q8_1_kernel(
    const float* __restrict__ x_f32,
    block_q8_1*  __restrict__ out_q8,
    std::int64_t              total_blocks)
{
    const std::int64_t blk = static_cast<std::int64_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
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
    const std::int8_t q = static_cast<std::int8_t>(__float2int_rn(xi * id));
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
// 1. MiMo MoE Expert Mat-Vec (`mimo_mmvq_ids`)
//    - 4 rows per wave32 (`ROWS = 4`), 4 waves per workgroup (`WARPS = 4`, 16 rows/WG)
//    - LDS lookup-table staging for IQ2_XXS / IQ3_XXS / IQ2_XS / IQ3_S / IQ2_S / IQ1_M
//    - Register-cached `block_q8_1` activation (`RegQ8`) reused across all 4 rows
//    - Wave-ballot duplicate-expert coalescing across `ids[0 .. n_ids - 1]` (`n_ids <= 64`)
//      so shared experts in DFlash verify (`n_tokens = 5, top_k = 8`) load & unpack weights once
// ============================================================================

template <int ROWS>
__device__ __forceinline__ float mimo_reduce_rows(float (&acc)[ROWS], int lane) {
    static_assert(ROWS == 4, "mimo_reduce_rows requires ROWS == 4");
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
struct MimoSplit;

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

template <>
struct MimoSplit<16> { // IQ2_XXS
    using blk    = block_iq2_xxs;
    using grid_t = uint2;
    static constexpr int QK       = 256;
    static constexpr int GRID_LEN = 256;

    struct Raw {
        int2 q2_aux;
        half d;
    };

    struct W {
        int   g[8];
        int   ls;
        float dw;
    };

    __device__ __forceinline__ static const grid_t* global_grid() {
        return reinterpret_cast<const grid_t*>(iq2xxs_grid);
    }

    __device__ __forceinline__ static void init_grid(grid_t* s_grid, int tid, int nthreads) {
        const grid_t* g = global_grid();
        for (int i = tid; i < GRID_LEN; i += nthreads) {
            s_grid[i] = g[i];
        }
    }

    __device__ __forceinline__ static Raw load_raw(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub)
    {
        const blk* bq2 = bq2_row + kbx;
        return {load_i64_align2(bq2->qs + 4 * sub), bq2->d};
    }

    __device__ __forceinline__ static W unpack_w(
        const Raw&                 raw,
        const grid_t* __restrict__ grid_lut)
    {
        const int q2 = raw.q2_aux.x;
        const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(&q2);
        const uint32_t aux32 = static_cast<uint32_t>(raw.q2_aux.y);
        W r;
        #pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid_lut[aux8[k0 / 2]];
            const uint32_t signs = talus_iq::unpack_ksigns(aux32 >> (7 * k0 / 2));
            r.g[k0 + 0] = talus_iq::talus_signed(grid_pos.x, signs & 0x08040201u);
            r.g[k0 + 1] = talus_iq::talus_signed(grid_pos.y, signs & 0x80402010u);
        }
        r.ls = static_cast<int>((aux32 >> 27) | 1u);
        r.dw = __half2float(raw.d);
        return r;
    }

    __device__ __forceinline__ static W load_w(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub,
        const grid_t* __restrict__ grid_lut)
    {
        return unpack_w(load_raw(bq2_row, kbx, sub), grid_lut);
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sumi = STRATA_DP4A(r.g[j], u[j], sumi);
        }
        sumi = (sumi * r.ls) / 8;
        return (r.dw * d8) * static_cast<float>(sumi);
    }
};

template <>
struct MimoSplit<17> { // IQ2_XS
    using blk    = block_iq2_xs;
    using grid_t = uint2;
    static constexpr int QK       = 256;
    static constexpr int GRID_LEN = 512;

    struct Raw {
        int2    q2_packed;
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
        return reinterpret_cast<const grid_t*>(iq2xs_grid);
    }

    __device__ __forceinline__ static void init_grid(grid_t* s_grid, int tid, int nthreads) {
        const grid_t* g = global_grid();
        for (int i = tid; i < GRID_LEN; i += nthreads) {
            s_grid[i] = g[i];
        }
    }

    __device__ __forceinline__ static Raw load_raw(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub)
    {
        const blk* bq2 = bq2_row + kbx;
        return {load_i64_align2(bq2->qs + 4 * sub), bq2->scales[sub], bq2->d};
    }

    __device__ __forceinline__ static W unpack_w(
        const Raw&                 raw,
        const grid_t* __restrict__ grid_lut)
    {
        const uint16_t* q2 = reinterpret_cast<const uint16_t*>(&raw.q2_packed);
        W r;
        r.ls0 = raw.sc & 0x0F;
        r.ls1 = raw.sc >> 4;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid_lut[q2[l0 / 2] & 0x1FFu];
            const uint32_t signs = talus_iq::unpack_ksigns(q2[l0 / 2] >> 9);
            r.g[l0 + 0] = talus_iq::talus_signed(grid_pos.x, signs & 0x08040201u);
            r.g[l0 + 1] = talus_iq::talus_signed(grid_pos.y, signs & 0x80402010u);
        }
        r.dw = __half2float(raw.d);
        return r;
    }

    __device__ __forceinline__ static W load_w(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub,
        const grid_t* __restrict__ grid_lut)
    {
        return unpack_w(load_raw(bq2_row, kbx, sub), grid_lut);
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        int sumi0 = 0;
        int sumi1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = STRATA_DP4A(r.g[j], u[j], sumi0);
        #pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = STRATA_DP4A(r.g[j], u[j], sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        return (r.dw * d8) * static_cast<float>(sumi);
    }
};

template <>
struct MimoSplit<22> { // IQ2_S
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

    __device__ __forceinline__ static void init_grid(grid_t* s_grid, int tid, int nthreads) {
        const grid_t* g = global_grid();
        for (int i = tid; i < GRID_LEN; i += nthreads) {
            s_grid[i] = g[i];
        }
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
            r.g[l0 + 0] = talus_iq::talus_signed(grid_pos.x, ((sp & 0x03u) << 7) | ((sp & 0x0Cu) << 21));
            r.g[l0 + 1] = talus_iq::talus_signed(grid_pos.y, ((sp & 0x30u) << 3) | ((sp & 0xC0u) << 17));
        }
        r.dw = __half2float(raw.d);
        return r;
    }

    __device__ __forceinline__ static W load_w(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub,
        const grid_t* __restrict__ grid_lut)
    {
        return unpack_w(load_raw(bq2_row, kbx, sub), grid_lut);
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        return MimoSplit<17>::dot(
            MimoSplit<17>::W{{r.g[0], r.g[1], r.g[2], r.g[3], r.g[4], r.g[5], r.g[6], r.g[7]}, r.ls0, r.ls1, r.dw},
            u,
            d8);
    }
};

template <>
struct MimoSplit<18> { // IQ3_XXS
    using blk    = block_iq3_xxs;
    using grid_t = uint32_t;
    static constexpr int QK       = 256;
    static constexpr int GRID_LEN = 256;

    struct Raw {
        int2     q3_packed;
        uint32_t aux32;
        half     d;
    };

    struct W {
        int   g[8];
        int   ls;
        float dw;
    };

    __device__ __forceinline__ static const grid_t* global_grid() {
        return iq3xxs_grid;
    }

    __device__ __forceinline__ static void init_grid(grid_t* s_grid, int tid, int nthreads) {
        const grid_t* g = global_grid();
        for (int i = tid; i < GRID_LEN; i += nthreads) {
            s_grid[i] = g[i];
        }
    }

    __device__ __forceinline__ static Raw load_raw(
        const blk* __restrict__ bq3_row,
        int                     kbx,
        int                     sub)
    {
        const blk* bq3 = bq3_row + kbx;
        return {
            load_i64_align2(bq3->qs + 8 * sub),
            static_cast<uint32_t>(load_i32_align2(bq3->qs + 64 + 4 * sub)),
            bq3->d
        };
    }

    __device__ __forceinline__ static W unpack_w(
        const Raw&                 raw,
        const grid_t* __restrict__ grid_lut)
    {
        const uint8_t* q3 = reinterpret_cast<const uint8_t*>(&raw.q3_packed);
        const uint32_t aux32 = raw.aux32;
        W r;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t gp0 = grid_lut[q3[l0 + 0]];
            const uint32_t gp1 = grid_lut[q3[l0 + 1]];
            const uint32_t signs = talus_iq::unpack_ksigns(aux32 >> (7 * l0 / 2));
            r.g[l0 + 0] = talus_iq::talus_signed(gp0, signs & 0x08040201u);
            r.g[l0 + 1] = talus_iq::talus_signed(gp1, signs & 0x80402010u);
        }
        r.ls = static_cast<int>(aux32 >> 28);
        r.dw = __half2float(raw.d);
        return r;
    }

    __device__ __forceinline__ static W load_w(
        const blk* __restrict__ bq3_row,
        int                     kbx,
        int                     sub,
        const grid_t* __restrict__ grid_lut)
    {
        return unpack_w(load_raw(bq3_row, kbx, sub), grid_lut);
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sumi = STRATA_DP4A(r.g[j], u[j], sumi);
        }
        sumi = (r.ls * sumi + sumi / 2) / 2;
        return (r.dw * d8) * static_cast<float>(sumi);
    }
};

template <>
struct MimoSplit<21> { // IQ3_S
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

    __device__ __forceinline__ static void init_grid(grid_t* s_grid, int tid, int nthreads) {
        const grid_t* g = global_grid();
        for (int i = tid; i < GRID_LEN; i += nthreads) {
            s_grid[i] = g[i];
        }
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
            r.g[l0 + 0] = talus_iq::talus_signed(gp0, ((sp & 0x03u) << 7) | ((sp & 0x0Cu) << 21));
            r.g[l0 + 1] = talus_iq::talus_signed(gp1, ((sp & 0x30u) << 3) | ((sp & 0xC0u) << 17));
        }
        r.ls = 1 + 2 * ((raw.sc_byte >> ((raw.sub_idx & 1) * 4)) & 0x0F);
        r.dw = __half2float(raw.d);
        return r;
    }

    __device__ __forceinline__ static W load_w(
        const blk* __restrict__ bq3_row,
        int                     kbx,
        int                     sub,
        const grid_t* __restrict__ grid_lut)
    {
        return unpack_w(load_raw(bq3_row, kbx, sub), grid_lut);
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

template <>
struct MimoSplit<29> { // IQ1_M
    using blk    = block_iq1_m;
    using grid_t = uint32_t;
    static constexpr int QK       = 256;
    static constexpr int GRID_LEN = 2048;

    struct Raw {
        int      qs_packed;
        uint8_t  qh0;
        uint8_t  qh1;
        uint8_t  sub_idx;
        uint16_t sc[4];
    };

    struct W {
        int   g[8];
        float delta[4];
        int   sc0;
        int   sc1;
        float dw;
    };

    __device__ __forceinline__ static const grid_t* global_grid() {
        return iq1s_grid_gpu;
    }

    __device__ __forceinline__ static void init_grid(grid_t* s_grid, int tid, int nthreads) {
        const grid_t* g = global_grid();
        for (int i = tid; i < GRID_LEN; i += nthreads) {
            s_grid[i] = g[i];
        }
    }

    __device__ __forceinline__ static Raw load_raw(
        const blk* __restrict__ bq1_row,
        int                     kbx,
        int                     sub)
    {
        const blk* bq1 = bq1_row + kbx;
        const uint16_t* sc = reinterpret_cast<const uint16_t*>(bq1->scales);
        return {
            load_i32_align2(bq1->qs + 4 * sub),
            bq1->qh[2 * sub + 0],
            bq1->qh[2 * sub + 1],
            static_cast<uint8_t>(sub),
            {sc[0], sc[1], sc[2], sc[3]}
        };
    }

    __device__ __forceinline__ static W unpack_w(
        const Raw&                 raw,
        const grid_t* __restrict__ grid_lut)
    {
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(&raw.qs_packed);
        const uint8_t qh[2] = {raw.qh0, raw.qh1};
        W r;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = qh[l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = static_cast<int>(grid_lut[qs[l0 / 2] | ((qhl & 0x07) << 8)]);
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        iq1m_scale_t scale;
        scale.u16 = (raw.sc[0] >> 12) | ((raw.sc[1] >> 8) & 0x00F0) | ((raw.sc[2] >> 4) & 0x0F00) | (raw.sc[3] & 0xF000);
        r.dw = __half2float(scale.f16);
        const int tmp = raw.sc[raw.sub_idx / 2] >> (6 * (raw.sub_idx % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }

    __device__ __forceinline__ static W load_w(
        const blk* __restrict__ bq1_row,
        int                     kbx,
        int                     sub,
        const grid_t* __restrict__ grid_lut)
    {
        return unpack_w(load_raw(bq1_row, kbx, sub), grid_lut);
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = u[l0 + 0];
            const int u1 = u[l0 + 1];
            sumi[l0 / 4] = STRATA_DP4A(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = STRATA_DP4A(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = STRATA_DP4A(u0, 0x01010101, sumy);
            sumy = STRATA_DP4A(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * static_cast<float>(sumy);
        }
        return (r.dw * d8) * ((static_cast<float>(sumi[0]) + sumf[0]) * r.sc0 +
                              (static_cast<float>(sumi[1]) + sumf[1]) * r.sc1);
    }
};

template <>
struct MimoSplit<10> { // Q2_K
    using blk    = block_q2_K;
    using grid_t = uint32_t;
    static constexpr int QK       = 256;
    static constexpr int GRID_LEN = 0;

    struct W {
        int   g[8];
        float dl0;
        float ml0;
        float dl1;
        float ml1;
    };

    using Raw = W;

    __device__ __forceinline__ static const grid_t* global_grid() {
        return nullptr;
    }

    __device__ __forceinline__ static void init_grid(grid_t*, int, int) {}

    __device__ __forceinline__ static Raw load_raw(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub)
    {
        const blk* b = bq2_row + kbx;
        const float2 dm2 = __half22float2(b->dm);
        const uint8_t sc0 = b->scales[2 * sub + 0];
        const uint8_t sc1 = b->scales[2 * sub + 1];
        W r;
        r.dl0 = dm2.x * static_cast<float>(sc0 & 0x0F);
        r.ml0 = dm2.y * static_cast<float>(sc0 >> 4);
        r.dl1 = dm2.x * static_cast<float>(sc1 & 0x0F);
        r.ml1 = dm2.y * static_cast<float>(sc1 >> 4);
        const int shift = 2 * (sub & 3);
        const int base_i32 = 8 * (sub >> 2);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            r.g[j] = (load_i32_align2(b->qs + 4 * (base_i32 + j)) >> shift) & 0x03030303;
        }
        return r;
    }

    __device__ __forceinline__ static W unpack_w(
        const Raw& raw,
        const grid_t* __restrict__)
    {
        return raw;
    }

    __device__ __forceinline__ static W load_w(
        const blk* __restrict__ bq2_row,
        int                     kbx,
        int                     sub,
        const grid_t* __restrict__)
    {
        return load_raw(bq2_row, kbx, sub);
    }

    __device__ __forceinline__ static float dot(const W& r, const int u[8], float d8) {
        int sumi0 = 0, sumq0 = 0;
        int sumi1 = 0, sumq1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi0 = STRATA_DP4A(r.g[j], u[j], sumi0);
            sumq0 = STRATA_DP4A(0x01010101, u[j], sumq0);
        }
        #pragma unroll
        for (int j = 4; j < 8; ++j) {
            sumi1 = STRATA_DP4A(r.g[j], u[j], sumi1);
            sumq1 = STRATA_DP4A(0x01010101, u[j], sumq1);
        }
        return d8 * (r.dl0 * static_cast<float>(sumi0) - r.ml0 * static_cast<float>(sumq0) +
                     r.dl1 * static_cast<float>(sumi1) - r.ml1 * static_cast<float>(sumq1));
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

__device__ __forceinline__ int mimo_slot_to_xcol(int slot, bool x_per_id, int x_group) {
    if (!x_per_id) return 0;
    if (x_group <= 1) return slot;
    if (x_group == 8) return slot >> 3;
    return slot / x_group;
}

template <int TY, int WARPS = 4, int ROWS = 4, bool STAGE_GRID = false>
__global__ __launch_bounds__(WARPS * 32, 3)
void mimo_mmvq_ids_kernel(
    const char*         __restrict__ weights,
    std::size_t                      expert_bytes,
    const std::int32_t* __restrict__ ids,
    int                              n_ids,
    const char*         __restrict__ x_q8_1,
    bool                             x_per_id,
    float*              __restrict__ y,
    int                              n_in,
    int                              n_out,
    const char*         __restrict__ weights2,
    float*              __restrict__ y2,
    int                              x_group)
{
    using S   = MimoSplit<TY>;
    using blk = typename S::blk;
    using Raw = typename S::Raw;
    using W   = typename S::W;

    const int k = static_cast<int>(blockIdx.y);
    if (k >= n_ids) return;
    const int e = ids[k];
    if (e < 0) return;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // Pairwise duplicate-expert coalescing across `ids[0 .. n_ids - 1]` (`n_ids <= 64`).
    // Pairs `(0, 1)`, `(2, 3)`, ... of identical experts share a single weight load + unpack pass
    // with minimal VGPR overhead (only 4 extra VGPRs for `acc1[ROWS]`).
    int s0 = k;
    int s1 = -1;
    if (n_ids > 1 && n_ids <= 64) {
        const int id0 = (lane < n_ids) ? ids[lane] : -2;
        const int id1 = (lane + 32 < n_ids) ? ids[lane + 32] : -2;
        const std::uint32_t m0 = static_cast<std::uint32_t>(__ballot(id0 == e));
        const std::uint32_t m1 = static_cast<std::uint32_t>(__ballot(id1 == e));
        const std::uint64_t mask =
            static_cast<std::uint64_t>(m0) | (static_cast<std::uint64_t>(m1) << 32);
        const std::uint64_t below_mask = (k == 0) ? 0ULL : ((1ULL << k) - 1ULL);
        const int rank = __popcll(mask & below_mask);
        if ((rank & 1) != 0) return;
        const std::uint64_t above_mask = mask & (~((below_mask << 1) | 1ULL));
        if (above_mask != 0ULL) {
            s1 = __ffsll(static_cast<unsigned long long>(above_mask)) - 1;
        }
    }

    __shared__ typename S::grid_t s_grid[(STAGE_GRID && S::GRID_LEN > 0) ? S::GRID_LEN : 1];
    if constexpr (STAGE_GRID && S::GRID_LEN > 0) {
        S::init_grid(s_grid, tid, WARPS * 32);
        __syncthreads();
    }
    const typename S::grid_t* grid = (STAGE_GRID && S::GRID_LEN > 0) ? s_grid : S::global_grid();

    const int row0 = (static_cast<int>(blockIdx.x) * WARPS + warp) * ROWS;
    if (row0 >= n_out) return;

    const char* w_base = (blockIdx.z == 0) ? weights : weights2;
    float*      y_base = (blockIdx.z == 0) ? y : y2;
    if (w_base == nullptr || y_base == nullptr) return;

    const int bpr   = n_in / S::QK;
    const int q8_pr = n_in / QK8_1;
    const int grp   = lane >> 3;
    const int sub   = lane & 7;

    const blk* w_exp = reinterpret_cast<const blk*>(w_base + static_cast<std::size_t>(e) * expert_bytes);
    const blk* row_b[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = (row0 + r < n_out) ? (row0 + r) : (n_out - 1);
        row_b[r] = w_exp + static_cast<std::size_t>(row) * bpr;
    }

    const auto* x_all = reinterpret_cast<const block_q8_1*>(x_q8_1);
    const int x_col0  = mimo_slot_to_xcol(s0, x_per_id, x_group);
    const int x_col1  = (s1 >= 0) ? mimo_slot_to_xcol(s1, x_per_id, x_group) : x_col0;

    // Fast 1-vector path (`s1 < 0` or both slots share the same activation vector `x_col0 == x_col1`).
    if (s1 < 0 || x_col0 == x_col1) {
        const block_q8_1* x0 = x_all + static_cast<std::size_t>(x_col0) * q8_pr;
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
                const W wv = S::unpack_w(raw[r], grid);
                acc0[r] += S::dot(wv, rq8.u, rq8.d8);
            }
        }

        const float sum0 = mimo_reduce_rows<ROWS>(acc0, lane);
        if (lane < ROWS && row0 + lane < n_out) {
            y_base[static_cast<std::size_t>(s0) * n_out + row0 + lane] = sum0;
            if (s1 >= 0) {
                y_base[static_cast<std::size_t>(s1) * n_out + row0 + lane] = sum0;
            }
        }
        return;
    }

    // Coalesced 2-vector path (`s1 >= 0` and `x_col0 != x_col1`):
    // Load & unpack each weight block ONCE, dot against `x0` and `x1` sequentially reusing `RegQ8`.
    const block_q8_1* x0 = x_all + static_cast<std::size_t>(x_col0) * q8_pr;
    const block_q8_1* x1 = x_all + static_cast<std::size_t>(x_col1) * q8_pr;
    float acc0[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kbx = grp; kbx < bpr; kbx += 4) {
        Raw raw[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            raw[r] = S::load_raw(row_b[r], kbx, sub);
        }
        RegQ8 rq0;
        RegQ8 rq1;
        rq0.load(x0 + kbx * 8, sub);
        rq1.load(x1 + kbx * 8, sub);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const W wv = S::unpack_w(raw[r], grid);
            acc0[r] += S::dot(wv, rq0.u, rq0.d8);
            acc1[r] += S::dot(wv, rq1.u, rq1.d8);
        }
    }

    const float sum0 = mimo_reduce_rows<ROWS>(acc0, lane);
    const float sum1 = mimo_reduce_rows<ROWS>(acc1, lane);
    if (lane < ROWS && row0 + lane < n_out) {
        y_base[static_cast<std::size_t>(s0) * n_out + row0 + lane] = sum0;
        y_base[static_cast<std::size_t>(s1) * n_out + row0 + lane] = sum1;
    }
}

template <int TY>
void launch_mimo_mmvq_ids(
    const void*         weights,
    std::size_t         expert_bytes,
    const std::int32_t* ids,
    int                 n_ids,
    const void*         x_q8_1,
    bool                x_per_id,
    float*              y,
    int                 n_in,
    int                 n_out,
    void*               stream,
    const void*         weights2,
    float*              y2,
    int                 x_group)
{
    constexpr int WARPS = 4;
    constexpr int ROWS  = 4;
    constexpr int B_ROWS = WARPS * ROWS; // 16 rows per workgroup
    const dim3 block(WARPS * 32);
    const dim3 grid(
        static_cast<unsigned int>((n_out + B_ROWS - 1) / B_ROWS),
        static_cast<unsigned int>(n_ids),
        (weights2 != nullptr && y2 != nullptr) ? 2u : 1u);

    mimo_mmvq_ids_kernel<TY, WARPS, ROWS, false><<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
        static_cast<const char*>(weights),
        expert_bytes,
        ids,
        n_ids,
        static_cast<const char*>(x_q8_1),
        x_per_id,
        y,
        n_in,
        n_out,
        static_cast<const char*>(weights2),
        y2,
        x_group > 0 ? x_group : 1);
}

// ============================================================================
// 2. Vectorized K192 / V128 Load Helpers (F16 and Q8 KV caches)
//    - Each wave32 (`lane = 0..31`) owns:
//      * K[0..127]   at `lane * 4 + 0..3` (1 coalesced 64-bit load in F16)
//      * K[128..191] at `128 + lane * 2 + 0..1` (1 coalesced 32-bit load in F16)
//      * V[0..127]   at `lane * 4 + 0..3` (1 coalesced 64-bit load in F16)
// ============================================================================

struct RegK192 {
    float k0, k1, k2, k3, k4, k5;
};

struct RegV128 {
    float v0, v1, v2, v3;
};

struct RegQ192 {
    float4 q0123;
    float2 q45;

    template <int HDK = 192>
    __device__ __forceinline__ void load(const float* __restrict__ q_head, int lane) {
        q0123 = reinterpret_cast<const float4*>(q_head)[lane];
        if constexpr (HDK == 192) {
            q45 = reinterpret_cast<const float2*>(q_head + 128)[lane];
        } else {
            q45 = make_float2(0.0f, 0.0f);
        }
    }

    __device__ __forceinline__ float dot_local(const RegK192& rk) const {
        return q0123.x * rk.k0 + q0123.y * rk.k1
             + q0123.z * rk.k2 + q0123.w * rk.k3
             + q45.x   * rk.k4 + q45.y   * rk.k5;
    }
};

template <bool Q8_KV, int HDK = 192>
__device__ __forceinline__ RegK192 load_k192(
    const void*          __restrict__ k_pool,
    const std::uint16_t* __restrict__ k_scale,
    std::size_t                       slot,
    int                               kv_h,
    int                               n_kv,
    int                               lane)
{
    RegK192 rk;
    const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kv_h);
    if constexpr (!Q8_KV) {
        const half* k_row = static_cast<const half*>(k_pool) + head_idx * HDK;
        const half2* k_h2 = reinterpret_cast<const half2*>(k_row);
        const float2 f01  = __half22float2(k_h2[lane * 2 + 0]);
        const float2 f23  = __half22float2(k_h2[lane * 2 + 1]);
        rk.k0 = f01.x; rk.k1 = f01.y;
        rk.k2 = f23.x; rk.k3 = f23.y;
        if constexpr (HDK == 192) {
            const float2 f45 = __half22float2(k_h2[64 + lane]);
            rk.k4 = f45.x; rk.k5 = f45.y;
        } else {
            rk.k4 = 0.0f; rk.k5 = 0.0f;
        }
    } else {
        const std::int8_t* k_row = static_cast<const std::int8_t*>(k_pool) + head_idx * HDK;
        const float sk = __half2float(reinterpret_cast<const half*>(k_scale)[head_idx]);
        const std::uint32_t p0123 = reinterpret_cast<const std::uint32_t*>(k_row)[lane];
        rk.k0 = static_cast<float>(static_cast<std::int8_t>(p0123 & 0xFFu)) * sk;
        rk.k1 = static_cast<float>(static_cast<std::int8_t>((p0123 >> 8) & 0xFFu)) * sk;
        rk.k2 = static_cast<float>(static_cast<std::int8_t>((p0123 >> 16) & 0xFFu)) * sk;
        rk.k3 = static_cast<float>(static_cast<std::int8_t>((p0123 >> 24) & 0xFFu)) * sk;
        if constexpr (HDK == 192) {
            const std::uint16_t p45 = reinterpret_cast<const std::uint16_t*>(k_row + 128)[lane];
            rk.k4 = static_cast<float>(static_cast<std::int8_t>(p45 & 0xFFu)) * sk;
            rk.k5 = static_cast<float>(static_cast<std::int8_t>((p45 >> 8) & 0xFFu)) * sk;
        } else {
            rk.k4 = 0.0f; rk.k5 = 0.0f;
        }
    }
    return rk;
}

template <bool Q8_KV>
__device__ __forceinline__ RegV128 load_v128(
    const void*          __restrict__ v_pool,
    const std::uint16_t* __restrict__ v_scale,
    std::size_t                       slot,
    int                               kv_h,
    int                               n_kv,
    int                               lane)
{
    RegV128 rv;
    const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kv_h);
    if constexpr (!Q8_KV) {
        const half* v_row = static_cast<const half*>(v_pool) + head_idx * 128;
        const half2* v_h2 = reinterpret_cast<const half2*>(v_row);
        const float2 f01  = __half22float2(v_h2[lane * 2 + 0]);
        const float2 f23  = __half22float2(v_h2[lane * 2 + 1]);
        rv.v0 = f01.x; rv.v1 = f01.y;
        rv.v2 = f23.x; rv.v3 = f23.y;
    } else {
        const std::int8_t* v_row = static_cast<const std::int8_t*>(v_pool) + head_idx * 128;
        const float sv = __half2float(reinterpret_cast<const half*>(v_scale)[head_idx]);
        const std::uint32_t p0123 = reinterpret_cast<const std::uint32_t*>(v_row)[lane];
        rv.v0 = static_cast<float>(static_cast<std::int8_t>(p0123 & 0xFFu)) * sv;
        rv.v1 = static_cast<float>(static_cast<std::int8_t>((p0123 >> 8) & 0xFFu)) * sv;
        rv.v2 = static_cast<float>(static_cast<std::int8_t>((p0123 >> 16) & 0xFFu)) * sv;
        rv.v3 = static_cast<float>(static_cast<std::int8_t>((p0123 >> 24) & 0xFFu)) * sv;
    }
    return rv;
}

__device__ __forceinline__ RegK192 load_k192_smem(const half* k_row, int lane) {
    RegK192 rk;
    const half2* k_h2 = reinterpret_cast<const half2*>(k_row);
    const float2 f01  = __half22float2(k_h2[lane * 2 + 0]);
    const float2 f23  = __half22float2(k_h2[lane * 2 + 1]);
    const float2 f45  = __half22float2(k_h2[64 + lane]);
    rk.k0 = f01.x; rk.k1 = f01.y;
    rk.k2 = f23.x; rk.k3 = f23.y;
    rk.k4 = f45.x; rk.k5 = f45.y;
    return rk;
}

__device__ __forceinline__ RegV128 load_v128_smem(const half* v_row, int lane) {
    RegV128 rv;
    const half2* v_h2 = reinterpret_cast<const half2*>(v_row);
    const float2 f01  = __half22float2(v_h2[lane * 2 + 0]);
    const float2 f23  = __half22float2(v_h2[lane * 2 + 1]);
    rv.v0 = f01.x; rv.v1 = f01.y;
    rv.v2 = f23.x; rv.v3 = f23.y;
    return rv;
}

__device__ __forceinline__ float warp_sum_f32(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        x += __shfl_xor(x, off, 32);
    }
    return x;
}

// ============================================================================
// 3. Sliding-Window Attention with Learned Sinks (`mimo_attn_swa_sinks`)
//    - Supports `HQ = 4` query heads per wave (sharing K and V loads in registers
//      within the same `kv_h` GQA group) as well as `HQ = 1` for arbitrary `n_head / n_kv`.
//    - Incorporates `sinks[h]` into the online softmax denominator as `exp(sink[h] - m_final)`.
// ============================================================================

template <bool Q8_KV, int HQ = 4, int WARPS = 2, int HDK = 192>
__global__ __launch_bounds__(WARPS * 32)
void mimo_attn_swa_sinks_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    const std::uint16_t* __restrict__ k_scale,
    const std::uint16_t* __restrict__ v_scale,
    const float*         __restrict__ sinks,
    float*               __restrict__ out,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               window,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev,
    float                             value_scale,
    bool                              causal = true)
{
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int h0   = (static_cast<int>(blockIdx.x) * WARPS + warp) * HQ;
    const int qt   = static_cast<int>(blockIdx.y);
    if (h0 >= n_head || qt >= n_tok) return;

    const int g_ratio = n_head / n_kv;
    const int kv_h    = h0 / g_ratio;

    RegQ192 rq[HQ];
    float   m[HQ];
    float   l[HQ];
    float4  acc[HQ];

    #pragma unroll
    for (int j = 0; j < HQ; ++j) {
        const int h = (h0 + j < n_head) ? (h0 + j) : h0;
        rq[j].template load<HDK>(q + (static_cast<std::size_t>(qt) * n_head + h) * HDK, lane);
        m[j]   = -1e30f;
        l[j]   = 0.0f;
        acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const int q_pos   = causal ? (pos_dev[0] + qt) : pos_dev[0];
    const int eff_win = (stride > 0 && stride < window) ? stride : window;
    const int p_start = (q_pos - eff_win + 1 > 0) ? (q_pos - eff_win + 1) : 0;

    for (int p = p_start; p <= q_pos; ++p) {
        const std::size_t slot = (stride > 0)
            ? static_cast<std::size_t>(p % stride)
            : static_cast<std::size_t>(p);
        const RegK192 rk = load_k192<Q8_KV, HDK>(k_pool, k_scale, slot, kv_h, n_kv, lane);
        const RegV128 rv = load_v128<Q8_KV>(v_pool, v_scale, slot, kv_h, n_kv, lane);

        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const float score = warp_sum_f32(rq[j].dot_local(rk)) * scale;
            const float m_new = fmaxf(m[j], score);
            const float alpha = __expf(m[j] - m_new);
            const float w     = __expf(score - m_new);
            l[j] = l[j] * alpha + w;
            m[j] = m_new;
            acc[j].x = acc[j].x * alpha + w * rv.v0;
            acc[j].y = acc[j].y * alpha + w * rv.v1;
            acc[j].z = acc[j].z * alpha + w * rv.v2;
            acc[j].w = acc[j].w * alpha + w * rv.v3;
        }
    }

    #pragma unroll
    for (int j = 0; j < HQ; ++j) {
        const int h = h0 + j;
        if (h < n_head) {
            float scale_out = 0.0f;
            if (sinks != nullptr) {
                const float s_val   = sinks[h];
                const float m_final = fmaxf(m[j], s_val);
                const float alpha   = __expf(m[j] - m_final);
                const float denom   = l[j] * alpha + __expf(s_val - m_final);
                scale_out = (denom > 0.0f) ? ((alpha / denom) * value_scale) : 0.0f;
            } else {
                scale_out = (l[j] > 0.0f) ? ((1.0f / l[j]) * value_scale) : 0.0f;
            }
            float* out_head = out + (static_cast<std::size_t>(qt) * n_head + h) * 128;
            reinterpret_cast<float4*>(out_head)[lane] = make_float4(
                acc[j].x * scale_out,
                acc[j].y * scale_out,
                acc[j].z * scale_out,
                acc[j].w * scale_out);
        }
    }
}

template <int HQ = 4, int WARPS = 8, int TILE_P = 16>
__global__ __launch_bounds__(WARPS * 32)
void mimo_attn_swa_sinks_prefill_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    const float*         __restrict__ sinks,
    float*               __restrict__ out,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               window,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev,
    float                             value_scale)
{
    __shared__ half s_k[TILE_P][192];
    __shared__ half s_v[TILE_P][128];

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int q_local  = warp >> 1; // 0..3
    const int h_local  = warp & 1;  // 0..1
    const int h_wg     = static_cast<int>(blockIdx.x) * 8;
    const int h0       = h_wg + h_local * HQ;
    const int qt0      = static_cast<int>(blockIdx.y) * 4;
    const int qt       = qt0 + q_local;
    const int qt_last  = (qt0 + 3 < n_tok) ? (qt0 + 3) : (n_tok - 1);
    if (h_wg >= n_head || qt0 >= n_tok) return;

    const int g_ratio = n_head / n_kv;
    const int kv_h    = h_wg / g_ratio;
    const bool valid  = (h0 < n_head && qt < n_tok);

    RegQ192 rq[HQ];
    float   m[HQ];
    float   l[HQ];
    float4  acc[HQ];

    if (valid) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = (h0 + j < n_head) ? (h0 + j) : h0;
            rq[j].load(q + (static_cast<std::size_t>(qt) * n_head + h) * 192, lane);
            m[j]   = -1e30f;
            l[j]   = 0.0f;
            acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const int base_pos   = pos_dev[0];
    const int eff_win    = (stride > 0 && stride < window) ? stride : window;
    const int q_pos      = base_pos + qt;
    const int p_start    = (q_pos - eff_win + 1 > 0) ? (q_pos - eff_win + 1) : 0;
    const int q_pos0     = base_pos + qt0;
    const int q_pos_last = base_pos + qt_last;
    const int p_wg_beg   = (q_pos0 - eff_win + 1 > 0) ? (q_pos0 - eff_win + 1) : 0;
    const int p_wg_end   = q_pos_last + 1;

    for (int p_tile = p_wg_beg; p_tile < p_wg_end; p_tile += TILE_P) {
        const int tile_cnt = (p_wg_end - p_tile < TILE_P) ? (p_wg_end - p_tile) : TILE_P;

        for (int idx = tid; idx < tile_cnt * 96; idx += WARPS * 32) {
            const int t = idx / 96;
            const int c = idx % 96;
            const int p = p_tile + t;
            const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(p % stride) : static_cast<std::size_t>(p);
            const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kv_h);
            const half2* src = reinterpret_cast<const half2*>(static_cast<const half*>(k_pool) + head_idx * 192);
            reinterpret_cast<half2*>(s_k[t])[c] = src[c];
        }
        for (int idx = tid; idx < tile_cnt * 64; idx += WARPS * 32) {
            const int t = idx / 64;
            const int c = idx % 64;
            const int p = p_tile + t;
            const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(p % stride) : static_cast<std::size_t>(p);
            const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kv_h);
            const half2* src = reinterpret_cast<const half2*>(static_cast<const half*>(v_pool) + head_idx * 128);
            reinterpret_cast<half2*>(s_v[t])[c] = src[c];
        }
        __syncthreads();

        if (valid) {
            for (int t = 0; t < tile_cnt; ++t) {
                const int p = p_tile + t;
                if (p < p_start || p > q_pos) continue;
                const RegK192 rk = load_k192_smem(s_k[t], lane);
                const RegV128 rv = load_v128_smem(s_v[t], lane);

                #pragma unroll
                for (int j = 0; j < HQ; ++j) {
                    const float score = warp_sum_f32(rq[j].dot_local(rk)) * scale;
                    const float m_new = fmaxf(m[j], score);
                    const float alpha = __expf(m[j] - m_new);
                    const float w     = __expf(score - m_new);
                    l[j] = l[j] * alpha + w;
                    m[j] = m_new;
                    acc[j].x = acc[j].x * alpha + w * rv.v0;
                    acc[j].y = acc[j].y * alpha + w * rv.v1;
                    acc[j].z = acc[j].z * alpha + w * rv.v2;
                    acc[j].w = acc[j].w * alpha + w * rv.v3;
                }
            }
        }
        __syncthreads();
    }

    if (valid) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                float scale_out = 0.0f;
                if (sinks != nullptr) {
                    const float s_val   = sinks[h];
                    const float m_final = fmaxf(m[j], s_val);
                    const float alpha   = __expf(m[j] - m_final);
                    const float denom   = l[j] * alpha + __expf(s_val - m_final);
                    scale_out = (denom > 0.0f) ? ((alpha / denom) * value_scale) : 0.0f;
                } else {
                    scale_out = (l[j] > 0.0f) ? ((1.0f / l[j]) * value_scale) : 0.0f;
                }
                float* out_head = out + (static_cast<std::size_t>(qt) * n_head + h) * 128;
                reinterpret_cast<float4*>(out_head)[lane] = make_float4(
                    acc[j].x * scale_out,
                    acc[j].y * scale_out,
                    acc[j].z * scale_out,
                    acc[j].w * scale_out);
            }
        }
    }
}

/// SWA prefill v2 (TALUS_MIMO_SWA_VER=2): bitwise mimo_attn_swa_sinks_prefill_kernel<4, 8, 16> per
/// (token, head), with the per-key work spread over lanes instead of repeated in every lane.
/// The v1 per (token, head) arithmetic, read from its source:
///   per key p ascending in [p_start, q_pos]: lane L's partial P_L = q[4L..4L+3] . k[4L..4L+3] + q[128+2L..] . k[128+2L..]
///   (dot_local, one C expression), score = (xor-16, 8, 4, 2, 1 butterfly of the P_L) * scale, m_new = fmaxf(m, score),
///   alpha = __expf(m - m_new), w = __expf(score - m_new), l = l * alpha + w, acc = acc * alpha + w * v (4 dims per lane);
///   then the sinks epilogue.
/// v2: a wave owns one head and NTOK tokens; per 32-key chunk lane j owns key base + j: its K row (192 halves) sits in
/// the lane's registers, q comes in as wave-uniform values, and the lane evaluates the same dot_local expression for
/// all 32 virtual lanes L and the same butterfly tree (float + commutes: x_L + x_{L^off} is the pair the shuffle adds).
/// m over the chunk's keys is an inclusive max-scan (fmaxf is exact and associative), so alpha / w per key are the very
/// values of the sequential loop; l and acc then run the sequential chain over the chunk's keys in ascending order with
/// alpha / w broadcast from the key's lane.  Keys outside a token's window are skipped (v1 `continue`s over them).
template <int NTOK>
__global__ __launch_bounds__(256)
void mimo_attn_swa_sinks_prefill_v2_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    const float*         __restrict__ sinks,
    float*               __restrict__ out,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               window,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev,
    float                             value_scale,
    void*                __restrict__ q8_mmq,
    int                               q8_layout)
{
    __shared__ __attribute__((aligned(16))) half s_v[32][128];
    const int tid  = static_cast<int>(threadIdx.x);
    const int w    = __builtin_amdgcn_readfirstlane(tid >> 5);   // head within the KV group (G = 8); wave-uniform
    const int lane = tid & 31;
    const int G    = n_head / n_kv;
    const int kv_h = static_cast<int>(blockIdx.y);
    const int h    = kv_h * G + w;
    const int t0   = static_cast<int>(blockIdx.x) * NTOK;
    const int t1   = (t0 + NTOK < n_tok) ? (t0 + NTOK) : n_tok;   // exclusive
    const bool hok = (w < G);

    const int base_pos = pos_dev[0];
    const int eff_win  = (stride > 0 && stride < window) ? stride : window;
    const int kb_beg   = (base_pos + t0 - eff_win + 1 > 0) ? (base_pos + t0 - eff_win + 1) : 0;
    const int kb_end   = base_pos + t1;   // exclusive

    float  m[NTOK], l[NTOK];
    float4 acc[NTOK];
    #pragma unroll
    for (int i = 0; i < NTOK; ++i) {
        m[i] = -1e30f;
        l[i] = 0.0f;
        acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for (int base = kb_beg; base < kb_end; base += 32) {
        const int  pk   = base + lane;            // this lane's key
        const bool kok  = pk < kb_end;
        // V chunk -> LDS (32 keys x 128 halves; 256 threads x 2 x 8 halves)
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = tid + 256 * j;
            const int key = idx >> 4;
            const int u   = idx & 15;
            const int p   = base + key;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (p < kb_end) {
                const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(p % stride) : static_cast<std::size_t>(p);
                v = *reinterpret_cast<const uint4*>(static_cast<const half*>(v_pool) + (slot * static_cast<std::size_t>(n_kv) + kv_h) * 128 + u * 8);
            }
            *reinterpret_cast<uint4*>(&s_v[key][u * 8]) = v;
        }
        // this lane's K row (192 halves) in registers
        half2 kr[96];
        {
            const std::size_t slot = kok ? ((stride > 0) ? static_cast<std::size_t>(pk % stride) : static_cast<std::size_t>(pk)) : 0;
            const uint4* src = reinterpret_cast<const uint4*>(static_cast<const half*>(k_pool) + (slot * static_cast<std::size_t>(n_kv) + kv_h) * 192);
            #pragma unroll
            for (int c = 0; c < 24; ++c) {
                const uint4 u = kok ? src[c] : make_uint4(0u, 0u, 0u, 0u);
                kr[4 * c + 0] = __builtin_bit_cast(half2, u.x);
                kr[4 * c + 1] = __builtin_bit_cast(half2, u.y);
                kr[4 * c + 2] = __builtin_bit_cast(half2, u.z);
                kr[4 * c + 3] = __builtin_bit_cast(half2, u.w);
            }
        }
        __syncthreads();
        if (!hok) continue;

        #pragma unroll
        for (int i = 0; i < NTOK; ++i) {
            const int qt = t0 + i;
            if (qt >= t1) break;
            const int q_pos   = base_pos + qt;
            const int p_start = (q_pos - eff_win + 1 > 0) ? (q_pos - eff_win + 1) : 0;
            const int lo = (p_start > base) ? p_start : base;
            const int hi = (q_pos < base + 31) ? q_pos : (base + 31);
            if (lo > hi) continue;
            const float* qh = q + (static_cast<std::size_t>(qt) * n_head + h) * 192;
            // the 32 virtual lanes' partials, then the butterfly tree (pairs L, L ^ off)
            float a[32];
            #pragma unroll
            for (int L = 0; L < 32; ++L) {
                const float4 q0123 = reinterpret_cast<const float4*>(qh)[L];
                const float2 q45   = reinterpret_cast<const float2*>(qh + 128)[L];
                const float2 f01 = __half22float2(kr[2 * L + 0]);
                const float2 f23 = __half22float2(kr[2 * L + 1]);
                const float2 f45 = __half22float2(kr[64 + L]);
                a[L] = q0123.x * f01.x + q0123.y * f01.y
                     + q0123.z * f23.x + q0123.w * f23.y
                     + q45.x   * f45.x + q45.y   * f45.y;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                #pragma unroll
                for (int L = 0; L < off; ++L) a[L] = a[L] + a[L + off];
            }
            const bool in = (pk >= lo && pk <= hi);
            const float score = a[0] * scale;
            // inclusive max-scan over the chunk's lanes (out-of-window lanes: -inf, the identity), seeded with m
            float mx = in ? score : -INFINITY;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float o = __shfl_up(mx, off, 32);
                if (lane >= off) mx = fmaxf(mx, o);
            }
            const float m_new  = fmaxf(m[i], mx);
            float m_prev = __shfl_up(m_new, 1, 32);
            if (lane == 0) m_prev = m[i];
            // (lanes before lo: m_new == m_prev == m[i]; the first in-window lane's predecessor is m[i])
            const float alpha = __expf(m_prev - m_new);
            const float wgt   = __expf(score - m_new);
            float  li = l[i];
            float4 ai = acc[i];
            for (int p = lo; p <= hi; ++p) {
                const int   j  = p - base;
                // j is wave-uniform: v_readlane (an SGPR result) instead of a ds_bpermute round trip per key
                const float al = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(alpha), j));
                const float wj = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(wgt), j));
                li = li * al + wj;
                const float2 v01 = __half22float2(reinterpret_cast<const half2*>(s_v[j])[lane * 2 + 0]);
                const float2 v23 = __half22float2(reinterpret_cast<const half2*>(s_v[j])[lane * 2 + 1]);
                ai.x = ai.x * al + wj * v01.x;
                ai.y = ai.y * al + wj * v01.y;
                ai.z = ai.z * al + wj * v23.x;
                ai.w = ai.w * al + wj * v23.y;
            }
            l[i]   = li;
            acc[i] = ai;
            m[i]   = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(m_new), hi - base));
        }
    }

    if (!hok) return;
    #pragma unroll
    for (int i = 0; i < NTOK; ++i) {
        const int qt = t0 + i;
        if (qt >= t1) break;
        float scale_out = 0.0f;
        if (sinks != nullptr) {
            const float s_val   = sinks[h];
            const float m_final = fmaxf(m[i], s_val);
            const float alpha   = __expf(m[i] - m_final);
            const float denom   = l[i] * alpha + __expf(s_val - m_final);
            scale_out = (denom > 0.0f) ? ((alpha / denom) * value_scale) : 0.0f;
        } else {
            scale_out = (l[i] > 0.0f) ? ((1.0f / l[i]) * value_scale) : 0.0f;
        }
        float* out_head = out + (static_cast<std::size_t>(qt) * n_head + h) * 128;
        const float4 xi = make_float4(
            acc[i].x * scale_out,
            acc[i].y * scale_out,
            acc[i].z * scale_out,
            acc[i].w * scale_out);
        reinterpret_cast<float4*>(out_head)[lane] = xi;
        if (q8_mmq != nullptr) {
            // the wo projection's MMQ q8_1 row block (128 values = this head; layout D4 (0) or DS4 (1)), bitwise
            // llama.cpp's quantize_mmq_q8_1: thread t of that kernel holds values 4t..4t+3 - this lane's - and reduces
            // amax / sum over xor 4, 2, 1 (32 values), d_inv = 127 / amax, q = roundf(x d_inv), d = 1 / d_inv
            float amax = fabsf(xi.x);
            amax = fmaxf(amax, fabsf(xi.y));
            amax = fmaxf(amax, fabsf(xi.z));
            amax = fmaxf(amax, fabsf(xi.w));
            #pragma unroll
            for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off, 32));
            float sum = 0.0f;
            if (q8_layout == 1) {
                sum = xi.x + xi.y + xi.z + xi.w;
                #pragma unroll
                for (int off = 4; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
            }
            const float d_inv = 127.0f / amax;
            char4 qv;
            qv.x = roundf(xi.x * d_inv);
            qv.y = roundf(xi.y * d_inv);
            qv.z = roundf(xi.z * d_inv);
            qv.w = roundf(xi.w * d_inv);
            const float d = 1.0f / d_inv;
            char* blk = static_cast<char*>(q8_mmq) + (static_cast<std::size_t>(h) * n_tok + qt) * 144;
            reinterpret_cast<char4*>(blk + 16)[lane] = qv;
            if ((lane & 7) == 0) {
                if (q8_layout == 1) reinterpret_cast<half2*>(blk)[lane >> 3] = __floats2half2_rn(d, sum);
                else reinterpret_cast<float*>(blk)[lane >> 3] = d;
            }
        }
    }
}

// ============================================================================
// 4. Full Causal Attention K192 / V128 (`mimo_attn_k192_v128`)
//    - Supports GQA 16:1 (`n_head = 64, n_kv = 4`) with `HQ = 4` query heads per wave
//      and `WARPS = 4` waves per workgroup (all 16 query heads of a KV head in 1 workgroup!)
//    - Supports Split-K over long contexts (`kv_len = 512 .. 32768+`) with exact
//      log-sum-exp combine across splits.
// ============================================================================

// Q8_1 of one head's 128 attention outputs as tk_quantize_q8_1 (native_quantize_q8_1_kernel) would write them:
// lane L holds outputs 4L..4L+3, block b = lanes 8b..8b+7; the quantizer's per-block xor tree over 32 lanes (16, 8, 4,
// 2, 1) maps to lane xor 4, 2, 1 and then the lane's own pairs (c ^ 2, c ^ 1): same pairs, so the same sum bits.
struct Att2Q81 { half2 ds; std::int8_t qs[32]; };
static_assert(sizeof(Att2Q81) == 36, "q8_1 block");

__device__ __forceinline__ void att2_q8_head(const float4 o, int lane, Att2Q81* __restrict__ head_blocks) {
    float am = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
    am = fmaxf(am, __shfl_xor(am, 4, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    float a0 = o.x + __shfl_xor(o.x, 4, 32), a1 = o.y + __shfl_xor(o.y, 4, 32);
    float a2 = o.z + __shfl_xor(o.z, 4, 32), a3 = o.w + __shfl_xor(o.w, 4, 32);
    a0 = a0 + __shfl_xor(a0, 2, 32); a1 = a1 + __shfl_xor(a1, 2, 32);
    a2 = a2 + __shfl_xor(a2, 2, 32); a3 = a3 + __shfl_xor(a3, 2, 32);
    a0 = a0 + __shfl_xor(a0, 1, 32); a1 = a1 + __shfl_xor(a1, 1, 32);
    a2 = a2 + __shfl_xor(a2, 1, 32); a3 = a3 + __shfl_xor(a3, 1, 32);
    const float f0 = a0 + a2, f1 = a1 + a3;
    const float sum = f0 + f1;
    const float d = q8_1_finite(am / 127.0f);
    Att2Q81* blk = head_blocks + (lane >> 3);
    const int v0 = 4 * (lane & 7);
    blk->qs[v0 + 0] = q8_1_quant(o.x, d, am);
    blk->qs[v0 + 1] = q8_1_quant(o.y, d, am);
    blk->qs[v0 + 2] = q8_1_quant(o.z, d, am);
    blk->qs[v0 + 3] = q8_1_quant(o.w, d, am);
    if ((lane & 7) == 0) blk->ds = q8_1_ds(d, sum);
}

constexpr int K192_SPLIT_STRIDE_F32 = 132; // 128 floats for V + m + l + 2 pad (16B-aligned)

template <bool Q8_KV, int HQ = 4, int WARPS = 4>
__global__ __launch_bounds__(WARPS * 32)
void mimo_attn_k192_v128_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    const std::uint16_t* __restrict__ k_scale,
    const std::uint16_t* __restrict__ v_scale,
    float*               __restrict__ out_or_scratch,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev,
    int                               n_splits)
{
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int h0        = (static_cast<int>(blockIdx.x) * WARPS + warp) * HQ;
    const int qt        = static_cast<int>(blockIdx.y);
    const int split_idx = static_cast<int>(blockIdx.z);
    if (h0 >= n_head || qt >= n_tok || split_idx >= n_splits) return;

    const int g_ratio = n_head / n_kv;
    const int kv_h    = h0 / g_ratio;

    RegQ192 rq[HQ];
    float   m[HQ];
    float   l[HQ];
    float4  acc[HQ];

    #pragma unroll
    for (int j = 0; j < HQ; ++j) {
        const int h = (h0 + j < n_head) ? (h0 + j) : h0;
        rq[j].load(q + (static_cast<std::size_t>(qt) * n_head + h) * 192, lane);
        m[j]   = -1e30f;
        l[j]   = 0.0f;
        acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const int q_pos  = pos_dev[0] + qt;
    const int kv_len = (stride > 0 && q_pos + 1 > stride) ? stride : (q_pos + 1);
    const int chunk  = (kv_len + n_splits - 1) / n_splits;
    const int p_beg  = split_idx * chunk;
    const int p_end  = (p_beg + chunk < kv_len) ? (p_beg + chunk) : kv_len;

    for (int p = p_beg; p < p_end; ++p) {
        const std::size_t slot = (stride > 0)
            ? static_cast<std::size_t>(p % stride)
            : static_cast<std::size_t>(p);
        const RegK192 rk = load_k192<Q8_KV>(k_pool, k_scale, slot, kv_h, n_kv, lane);
        const RegV128 rv = load_v128<Q8_KV>(v_pool, v_scale, slot, kv_h, n_kv, lane);

        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const float score = warp_sum_f32(rq[j].dot_local(rk)) * scale;
            const float m_new = fmaxf(m[j], score);
            const float alpha = __expf(m[j] - m_new);
            const float w     = __expf(score - m_new);
            l[j] = l[j] * alpha + w;
            m[j] = m_new;
            acc[j].x = acc[j].x * alpha + w * rv.v0;
            acc[j].y = acc[j].y * alpha + w * rv.v1;
            acc[j].z = acc[j].z * alpha + w * rv.v2;
            acc[j].w = acc[j].w * alpha + w * rv.v3;
        }
    }

    if (n_splits == 1) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                const float inv_l = (l[j] > 0.0f) ? (1.0f / l[j]) : 0.0f;
                float* out_head = out_or_scratch + (static_cast<std::size_t>(qt) * n_head + h) * 128;
                reinterpret_cast<float4*>(out_head)[lane] = make_float4(
                    acc[j].x * inv_l,
                    acc[j].y * inv_l,
                    acc[j].z * inv_l,
                    acc[j].w * inv_l);
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                const std::size_t idx =
                    ((static_cast<std::size_t>(qt) * n_head + h) * n_splits + split_idx)
                    * K192_SPLIT_STRIDE_F32;
                float* dst = out_or_scratch + idx;
                reinterpret_cast<float4*>(dst)[lane] = acc[j];
                if (lane == 0) {
                    dst[128] = m[j];
                    dst[129] = l[j];
                }
            }
        }
    }
}

__global__ __launch_bounds__(128)
void mimo_attn_k192_v128_combine_kernel(
    const float* __restrict__ scratch,
    float*       __restrict__ out,
    int                       n_tok,
    int                       n_head,
    int                       n_splits,
    Att2Q81*     __restrict__ q8_out = nullptr)
{
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int h    = static_cast<int>(blockIdx.x) * 4 + warp;
    const int qt   = static_cast<int>(blockIdx.y);
    if (h >= n_head || qt >= n_tok) return;

    float  m   = -1e30f;
    float  l   = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const float* base = scratch + (static_cast<std::size_t>(qt) * n_head + h) * n_splits * K192_SPLIT_STRIDE_F32;
    for (int s = 0; s < n_splits; ++s) {
        const float* sp = base + static_cast<std::size_t>(s) * K192_SPLIT_STRIDE_F32;
        const float  ms = sp[128];
        const float  ls = sp[129];
        if (ls <= 0.0f) continue;

        const float4 vs    = reinterpret_cast<const float4*>(sp)[lane];
        const float  m_new = fmaxf(m, ms);
        const float  a_old = __expf(m - m_new);
        const float  a_new = __expf(ms - m_new);
        l = l * a_old + ls * a_new;
        m = m_new;
        acc.x = acc.x * a_old + vs.x * a_new;
        acc.y = acc.y * a_old + vs.y * a_new;
        acc.z = acc.z * a_old + vs.z * a_new;
        acc.w = acc.w * a_old + vs.w * a_new;
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float* out_head = out + (static_cast<std::size_t>(qt) * n_head + h) * 128;
    const float4 o4 = make_float4(
        acc.x * inv_l,
        acc.y * inv_l,
        acc.z * inv_l,
        acc.w * inv_l);
    reinterpret_cast<float4*>(out_head)[lane] = o4;
    if (q8_out != nullptr) att2_q8_head(o4, lane, q8_out + (static_cast<std::size_t>(qt) * n_head + h) * 4);
}

// ============================================================================
// 4b. Decode / verify attention v2 (F16 KV, HDK 192): bitwise the v1 kernels above
//     (`mimo_attn_k192_v128_kernel<false>`, `mimo_attn_swa_sinks_kernel<false, *, *, 192>`), faster:
//     - U positions in flight per step: their K/V loads are issued one step ahead (register double
//       buffer) and their U * HQ score reductions are interleaved, instead of one load -> reduce -> update
//       chain per position;
//     - the wave32 butterfly (xor 16, 8, 4, 2, 1; x + x') runs on permlanex16 / DPP instead of ds_bpermute;
//     - no 32-bit modulo per position (ring slot advanced incrementally);
//     - HQ / WARPS per launch from a per-arch table (fewer query heads per wave = more waves at T = 1).
//     The arithmetic per (row, head, split) is written out with explicit fmaf in the order the v1 kernels
//     compile to on gfx11 (dot: fma chain seeded with q1*k1; w = exp(fma(scale, sum, -m_new)); l and acc
//     as fma(old, alpha, .)); the split ranges, the combine and the epilogues are unchanged.
//     Kill switch TALUS_MIMO_ATT2=0 (v1 kernels).
// ============================================================================

struct RawKV192 {
    uint2         k0123;
    std::uint32_t k45;
    uint2         v0123;
};

template <int HDK = 192>
__device__ __forceinline__ RawKV192 att2_load(const half* __restrict__ kp, const half* __restrict__ vp,
                                              std::size_t head_idx, int lane) {
    RawKV192 r;
    const half* k_row = kp + head_idx * HDK;
    const half* v_row = vp + head_idx * 128;
    r.k0123 = reinterpret_cast<const uint2*>(k_row)[lane];
    if constexpr (HDK == 192) {
        r.k45 = reinterpret_cast<const std::uint32_t*>(k_row + 128)[lane];
    } else {
        r.k45 = 0u;
    }
    r.v0123 = reinterpret_cast<const uint2*>(v_row)[lane];
    return r;
}

__device__ __forceinline__ float2 att2_h2f(std::uint32_t u) {
    union { std::uint32_t u32; half2 h2; } cvt;
    cvt.u32 = u;
    return __half22float2(cvt.h2);
}

// HDK 128 (DFlash drafter): v1 runs dot_local with q45 = k45 = 0; it compiles to the same chain as HDK 192 seeded with
// q1*k1 (then q0, q2, q3), the two zero products folded away (one head keeps a "+ 0", which only touches the sign of
// an exact zero). Written out explicitly: the expression form lets the compiler seed the chain with q0*k0 instead.
__device__ __forceinline__ float att2_dot128(const RegQ192& q, const RawKV192& r) {
    const float2 k01 = att2_h2f(r.k0123.x);
    const float2 k23 = att2_h2f(r.k0123.y);
    float d = q.q0123.y * k01.y;
    d = __builtin_fmaf(q.q0123.x, k01.x, d);
    d = __builtin_fmaf(q.q0123.z, k23.x, d);
    d = __builtin_fmaf(q.q0123.w, k23.y, d);
    return d;
}

__device__ __forceinline__ float att2_dot(const RegQ192& q, const RawKV192& r) {
    const float2 k01 = att2_h2f(r.k0123.x);
    const float2 k23 = att2_h2f(r.k0123.y);
    const float2 k45 = att2_h2f(r.k45);
    float d = q.q0123.y * k01.y;
    d = __builtin_fmaf(q.q0123.x, k01.x, d);
    d = __builtin_fmaf(q.q0123.z, k23.x, d);
    d = __builtin_fmaf(q.q0123.w, k23.y, d);
    d = __builtin_fmaf(q.q45.x, k45.x, d);
    d = __builtin_fmaf(q.q45.y, k45.y, d);
    return d;
}

template <int XOR>
__device__ __forceinline__ float att2_xor(float x) {
    const int xi = __float_as_int(x);
    int yi;
    if constexpr (XOR == 16) {
        yi = __builtin_amdgcn_permlanex16(xi, xi, 0x76543210u, 0xfedcba98u, false, false);
    } else {
        yi = __builtin_amdgcn_update_dpp(xi, xi, 0x160 | XOR, 0xF, 0xF, false);   // row_xmask:XOR
    }
    return __int_as_float(yi);
}

__device__ __forceinline__ void att2_update(float sum, float scale, const RawKV192& r,
                                            float& m, float& l, float4& acc) {
    const float2 v01   = att2_h2f(r.v0123.x);
    const float2 v23   = att2_h2f(r.v0123.y);
    const float  score = sum * scale;
    const float  m_new = fmaxf(m, score);
    const float  alpha = __expf(m - m_new);
    const float  w     = __expf(__builtin_fmaf(scale, sum, -m_new));
    l     = __builtin_fmaf(l, alpha, w);
    m     = m_new;
    acc.x = __builtin_fmaf(acc.x, alpha, w * v01.x);
    acc.y = __builtin_fmaf(acc.y, alpha, w * v01.y);
    acc.z = __builtin_fmaf(acc.z, alpha, w * v23.x);
    acc.w = __builtin_fmaf(acc.w, alpha, w * v23.y);
}

// U consecutive positions: scores (interleaved butterflies), then the online-softmax updates in order.
template <int HQ, int U, int HDK = 192>
__device__ __forceinline__ void att2_step(const RegQ192 (&rq)[HQ], const RawKV192 (&r)[U], float scale,
                                          float (&m)[HQ], float (&l)[HQ], float4 (&acc)[HQ]) {
    float s[U][HQ];
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) s[u][j] = (HDK == 192) ? att2_dot(rq[j], r[u]) : att2_dot128(rq[j], r[u]);
    }
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) s[u][j] = s[u][j] + att2_xor<16>(s[u][j]);
    }
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) s[u][j] = s[u][j] + att2_xor<8>(s[u][j]);
    }
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) s[u][j] = s[u][j] + att2_xor<4>(s[u][j]);
    }
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) s[u][j] = s[u][j] + att2_xor<2>(s[u][j]);
    }
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) s[u][j] = s[u][j] + att2_xor<1>(s[u][j]);
    }
    #pragma unroll
    for (int u = 0; u < U; ++u) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) att2_update(s[u][j], scale, r[u], m[j], l[j], acc[j]);
    }
}

// Positions [p_beg, p_end) of one (row, kv head); slot(p) = p for stride == 0, else p % stride (ring).
template <int HQ, int U, int HDK = 192>
__device__ __forceinline__ void att2_range(const RegQ192 (&rq)[HQ], const half* __restrict__ kp,
                                           const half* __restrict__ vp, int p_beg, int p_end, int stride,
                                           int kv_h, int n_kv, int lane, float scale,
                                           float (&m)[HQ], float (&l)[HQ], float4 (&acc)[HQ]) {
    if (p_end <= p_beg) return;
    int slot = (stride > 0) ? (p_beg % stride) : p_beg;
    auto next_slot = [&](int s0) { return (stride > 0 && s0 + 1 == stride) ? 0 : s0 + 1; };
    auto head = [&](int s0) {
        return static_cast<std::size_t>(s0) * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kv_h);
    };
    const int n_main = ((p_end - p_beg) / U) * U;
    if (n_main > 0) {
        RawKV192 cur[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) { cur[u] = att2_load<HDK>(kp, vp, head(slot), lane); slot = next_slot(slot); }
        for (int base = 0; base < n_main; base += U) {
            RawKV192 nxt[U];
            const bool more = base + U < n_main;
            if (more) {
                #pragma unroll
                for (int u = 0; u < U; ++u) { nxt[u] = att2_load<HDK>(kp, vp, head(slot), lane); slot = next_slot(slot); }
            }
            att2_step<HQ, U, HDK>(rq, cur, scale, m, l, acc);
            if (more) {
                #pragma unroll
                for (int u = 0; u < U; ++u) cur[u] = nxt[u];
            }
        }
    }
    for (int p = p_beg + n_main; p < p_end; ++p) {
        RawKV192 one[1] = {att2_load<HDK>(kp, vp, head(slot), lane)};
        slot = next_slot(slot);
        att2_step<HQ, 1, HDK>(rq, one, scale, m, l, acc);
    }
}

template <int HQ, int WARPS, int U>
__global__ __launch_bounds__(WARPS * 32)
void mimo_attn_k192_v128_v2_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    float*               __restrict__ out_or_scratch,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev,
    int                               n_splits)
{
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int h0        = (static_cast<int>(blockIdx.x) * WARPS + warp) * HQ;
    const int qt        = static_cast<int>(blockIdx.y);
    const int split_idx = static_cast<int>(blockIdx.z);
    if (h0 >= n_head || qt >= n_tok || split_idx >= n_splits) return;

    const int g_ratio = n_head / n_kv;
    const int kv_h    = h0 / g_ratio;

    RegQ192 rq[HQ];
    float   m[HQ];
    float   l[HQ];
    float4  acc[HQ];
    #pragma unroll
    for (int j = 0; j < HQ; ++j) {
        const int h = (h0 + j < n_head) ? (h0 + j) : h0;
        rq[j].load(q + (static_cast<std::size_t>(qt) * n_head + h) * 192, lane);
        m[j]   = -1e30f;
        l[j]   = 0.0f;
        acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const int q_pos  = pos_dev[0] + qt;
    const int kv_len = (stride > 0 && q_pos + 1 > stride) ? stride : (q_pos + 1);
    const int chunk  = (kv_len + n_splits - 1) / n_splits;
    const int p_beg  = split_idx * chunk;
    const int p_end  = (p_beg + chunk < kv_len) ? (p_beg + chunk) : kv_len;

    att2_range<HQ, U>(rq, static_cast<const half*>(k_pool), static_cast<const half*>(v_pool), p_beg, p_end,
                      stride, kv_h, n_kv, lane, scale, m, l, acc);

    if (n_splits == 1) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                const float inv_l = (l[j] > 0.0f) ? (1.0f / l[j]) : 0.0f;
                float* out_head = out_or_scratch + (static_cast<std::size_t>(qt) * n_head + h) * 128;
                reinterpret_cast<float4*>(out_head)[lane] = make_float4(
                    acc[j].x * inv_l,
                    acc[j].y * inv_l,
                    acc[j].z * inv_l,
                    acc[j].w * inv_l);
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                const std::size_t idx =
                    ((static_cast<std::size_t>(qt) * n_head + h) * n_splits + split_idx)
                    * K192_SPLIT_STRIDE_F32;
                float* dst = out_or_scratch + idx;
                reinterpret_cast<float4*>(dst)[lane] = acc[j];
                if (lane == 0) {
                    dst[128] = m[j];
                    dst[129] = l[j];
                }
            }
        }
    }
}

template <int HQ, int WARPS, int U, int HDK = 192>
__global__ __launch_bounds__(WARPS * 32)
void mimo_attn_swa_sinks_v2_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    const float*         __restrict__ sinks,
    float*               __restrict__ out,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               window,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev,
    float                             value_scale,
    bool                              causal = true,
    Att2Q81*             __restrict__ q8_out = nullptr)
{
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int h0   = (static_cast<int>(blockIdx.x) * WARPS + warp) * HQ;
    const int qt   = static_cast<int>(blockIdx.y);
    if (h0 >= n_head || qt >= n_tok) return;

    const int g_ratio = n_head / n_kv;
    const int kv_h    = h0 / g_ratio;

    RegQ192 rq[HQ];
    float   m[HQ];
    float   l[HQ];
    float4  acc[HQ];
    #pragma unroll
    for (int j = 0; j < HQ; ++j) {
        const int h = (h0 + j < n_head) ? (h0 + j) : h0;
        rq[j].template load<HDK>(q + (static_cast<std::size_t>(qt) * n_head + h) * HDK, lane);
        m[j]   = -1e30f;
        l[j]   = 0.0f;
        acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const int q_pos   = causal ? (pos_dev[0] + qt) : pos_dev[0];
    const int eff_win = (stride > 0 && stride < window) ? stride : window;
    const int p_start = (q_pos - eff_win + 1 > 0) ? (q_pos - eff_win + 1) : 0;

    att2_range<HQ, U, HDK>(rq, static_cast<const half*>(k_pool), static_cast<const half*>(v_pool), p_start, q_pos + 1,
                      stride, kv_h, n_kv, lane, scale, m, l, acc);

    #pragma unroll
    for (int j = 0; j < HQ; ++j) {
        const int h = h0 + j;
        if (h < n_head) {
            float scale_out = 0.0f;
            if (sinks != nullptr) {
                const float s_val   = sinks[h];
                const float m_final = fmaxf(m[j], s_val);
                const float alpha   = __expf(m[j] - m_final);
                const float denom   = l[j] * alpha + __expf(s_val - m_final);
                scale_out = (denom > 0.0f) ? ((alpha / denom) * value_scale) : 0.0f;
            } else {
                scale_out = (l[j] > 0.0f) ? ((1.0f / l[j]) * value_scale) : 0.0f;
            }
            float* out_head = out + (static_cast<std::size_t>(qt) * n_head + h) * 128;
            const float4 o4 = make_float4(
                acc[j].x * scale_out,
                acc[j].y * scale_out,
                acc[j].z * scale_out,
                acc[j].w * scale_out);
            reinterpret_cast<float4*>(out_head)[lane] = o4;
            if (q8_out != nullptr) att2_q8_head(o4, lane, q8_out + (static_cast<std::size_t>(qt) * n_head + h) * 4);
        }
    }
}

// ============================================================================
// 4c. Full-layer decode / verify attention v3 (F16 KV, K192 / V128): lanes = positions.
//     Bitwise `mimo_attn_k192_v128_kernel<false>` (same split ranges, combine, epilogue). Per (row, head, split) the v1
//     kernel computes, position by position: a 32-lane dot (lane L: fma chain over q/k[4L..4L+3], [128+2L, +1], seeded
//     with q[4L+1]*k[4L+1]) summed by an xor butterfly (16, 8, 4, 2, 1), then the online-softmax update. Here a wave
//     takes 32 positions at once (lane i = position i): each lane forms the 32 partial chains of its position itself
//     and adds them in the butterfly's tree (bit-reversed leaf order, pairwise adds commute), the running max is an
//     inclusive lane scan (max is exact), alpha / w are computed per lane, and only the l / acc recurrences run
//     position by position (alpha, w broadcast with readlane). The K / V tiles of a group are staged once in LDS for the
//     NW waves (HG heads each) of the workgroup, the next group's tiles are prefetched into registers.
//     Kill switch TALUS_MIMO_ATT3=0 (v2 / v1).
// ============================================================================

typedef unsigned int att3_u4 __attribute__((ext_vector_type(4)));   // SROA-friendly 16-B chunk
constexpr int ATT3_G  = 32;    // positions per group (one per lane)
constexpr int ATT3_KS = 200;   // K row stride in LDS (halves): 400 B, 16-B aligned, spreads banks

// l = fma(l, a, w) on the VALU (v1 runs it there): a, w and l are wave-uniform here, and the compiler would otherwise
// move the chain to the gfx11.5 SALU float unit; keep the exact v1 instruction.
__device__ __forceinline__ float att3_vfma(float l, float a, float w) {
    float r;
    asm("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(l), "s"(a), "s"(w));
    return r;
}

__device__ __forceinline__ constexpr int att3_rev5(int i) {
    return ((i & 1) << 4) | ((i & 2) << 2) | (i & 4) | ((i & 8) >> 2) | ((i & 16) >> 4);
}

template <int HG, int NW>
__global__ __launch_bounds__(NW * 32)
void mimo_attn_k192_v128_v3_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    float*               __restrict__ out_or_scratch,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev,
    int                               n_splits)
{
    constexpr int NT = NW * 32;
    constexpr int KCH = 24, VCH = 16;                 // 16-B chunks per K / V row
    __shared__ __align__(16) half s_k[ATT3_G * ATT3_KS];
    __shared__ __align__(16) half s_v[ATT3_G * 128];

    const int tid       = static_cast<int>(threadIdx.x);
    const int lane      = tid & 31;
    const int warp      = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int qt        = static_cast<int>(blockIdx.y);
    const int split_idx = static_cast<int>(blockIdx.z);
    const int h_wg      = static_cast<int>(blockIdx.x) * (HG * NW);
    const int h0        = h_wg + warp * HG;
    if (h_wg >= n_head || qt >= n_tok || split_idx >= n_splits) return;
    const int g_ratio = n_head / n_kv;
    const int kv_h    = h_wg / g_ratio;
    const bool wave_ok = h0 < n_head;

    const int q_pos  = pos_dev[0] + qt;
    const int kv_len = (stride > 0 && q_pos + 1 > stride) ? stride : (q_pos + 1);
    const int chunk  = (kv_len + n_splits - 1) / n_splits;
    const int p_beg  = split_idx * chunk;
    const int p_end  = (p_beg + chunk < kv_len) ? (p_beg + chunk) : kv_len;
    const int n_pos  = p_end - p_beg;

    float  m[HG], l[HG];
    float4 acc[HG];
    #pragma unroll
    for (int j = 0; j < HG; ++j) {
        m[j]   = -1e30f;
        l[j]   = 0.0f;
        acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    const float* qh[HG];
    #pragma unroll
    for (int j = 0; j < HG; ++j) {
        const int h = (h0 + j < n_head) ? (h0 + j) : (n_head - 1);
        qh[j] = q + (static_cast<std::size_t>(qt) * n_head + h) * 192;
    }

    const half* kp = static_cast<const half*>(k_pool);
    const half* vp = static_cast<const half*>(v_pool);
    constexpr int KPER = ATT3_G * KCH / NT;               // K chunks per thread
    constexpr int VPER = ATT3_G * VCH / NT;               // V chunks per thread
    static_assert((ATT3_G * KCH) % NT == 0 && (ATT3_G * VCH) % NT == 0, "whole chunks per thread");
    // full layers: positions < kv_len <= stride, so slot == position (no ring wrap); rows past p_end clamp to the
    // last row (loaded, never used)
    auto slot_of = [&](int p) -> std::size_t { return static_cast<std::size_t>(p < p_end ? p : p_end - 1); };
    att3_u4 pre_k[KPER], pre_v[VPER];
    if (n_pos > 0) {
            const int gf = p_beg;
            #pragma unroll
            for (int r = 0; r < KPER; ++r) {
                const int c = tid + r * NT;
                pre_k[r] = reinterpret_cast<const att3_u4*>(kp + (slot_of(gf + c / KCH) * n_kv + kv_h) * 192)[c % KCH];
            }
            #pragma unroll
            for (int r = 0; r < VPER; ++r) {
                const int c = tid + r * NT;
                pre_v[r] = reinterpret_cast<const att3_u4*>(vp + (slot_of(gf + c / VCH) * n_kv + kv_h) * 128)[c % VCH];
            }
        }

    for (int g0 = p_beg; g0 < p_end; g0 += ATT3_G) {
        const int n_valid = (p_end - g0 < ATT3_G) ? (p_end - g0) : ATT3_G;
        __syncthreads();                                   // previous group's LDS reads are done
        #pragma unroll
        for (int r = 0; r < KPER; ++r) {
            const int c = tid + r * NT;
            reinterpret_cast<att3_u4*>(s_k + (c / KCH) * ATT3_KS)[c % KCH] = pre_k[r];
        }
        #pragma unroll
        for (int r = 0; r < VPER; ++r) {
            const int c = tid + r * NT;
            reinterpret_cast<att3_u4*>(s_v + (c / VCH) * 128)[c % VCH] = pre_v[r];
        }
        __syncthreads();
        if (g0 + ATT3_G < p_end) {
            const int gf = g0 + ATT3_G;
            #pragma unroll
            for (int r = 0; r < KPER; ++r) {
                const int c = tid + r * NT;
                pre_k[r] = reinterpret_cast<const att3_u4*>(kp + (slot_of(gf + c / KCH) * n_kv + kv_h) * 192)[c % KCH];
            }
            #pragma unroll
            for (int r = 0; r < VPER; ++r) {
                const int c = tid + r * NT;
                pre_v[r] = reinterpret_cast<const att3_u4*>(vp + (slot_of(gf + c / VCH) * n_kv + kv_h) * 128)[c % VCH];
            }
        }   // prefetch the next group
        if (!wave_ok) continue;

        // ---- scores: lane = position g0 + lane ----
        float sum[HG];
        {
            const half* krow = s_k + lane * ATT3_KS;
            float t0[HG], t1[HG], t2[HG], t3[HG], t4[HG];
            #pragma unroll
            for (int ii = 0; ii < 32; ++ii) {
                const int jj = att3_rev5(ii);
                const float2 k01 = __half22float2(reinterpret_cast<const half2*>(krow)[jj * 2 + 0]);
                const float2 k23 = __half22float2(reinterpret_cast<const half2*>(krow)[jj * 2 + 1]);
                const float2 k45 = __half22float2(reinterpret_cast<const half2*>(krow)[64 + jj]);
                #pragma unroll
                for (int h = 0; h < HG; ++h) {
                    const float4 q0 = reinterpret_cast<const float4*>(qh[h])[jj];
                    const float2 q1 = reinterpret_cast<const float2*>(qh[h] + 128)[jj];
                    float d = q0.y * k01.y;
                    d = __builtin_fmaf(q0.x, k01.x, d);
                    d = __builtin_fmaf(q0.z, k23.x, d);
                    d = __builtin_fmaf(q0.w, k23.y, d);
                    d = __builtin_fmaf(q1.x, k45.x, d);
                    d = __builtin_fmaf(q1.y, k45.y, d);
                    // the butterfly's tree: leaves in bit-reversed order, fold while the low bits of ii are ones
                    if ((ii & 1) == 0) { t0[h] = d; continue; }
                    d = t0[h] + d;
                    if ((ii & 2) == 0) { t1[h] = d; continue; }
                    d = t1[h] + d;
                    if ((ii & 4) == 0) { t2[h] = d; continue; }
                    d = t2[h] + d;
                    if ((ii & 8) == 0) { t3[h] = d; continue; }
                    d = t3[h] + d;
                    if ((ii & 16) == 0) { t4[h] = d; continue; }
                    sum[h] = t4[h] + d;
                }
            }
        }

        // ---- online softmax, lane-parallel parts ----
        float alpha[HG], wgt[HG];
        const bool valid = lane < n_valid;
        #pragma unroll
        for (int h = 0; h < HG; ++h) {
            const float sc = sum[h] * scale;
            float mx = valid ? sc : -INFINITY;
            #pragma unroll
            for (int off = 1; off < 32; off <<= 1) {
                const float t = __shfl_up(mx, off, 32);
                if (lane >= off) mx = fmaxf(mx, t);
            }
            const float mi = fmaxf(m[h], mx);
            float mp = __shfl_up(mi, 1, 32);
            if (lane == 0) mp = m[h];
            alpha[h] = __expf(mp - mi);
            wgt[h]   = __expf(__builtin_fmaf(scale, sum[h], -mi));
            m[h] = __shfl(mi, n_valid - 1, 32);
        }

        // ---- l / acc recurrences, position by position ----
        const half* vbase = s_v + lane * 4;
        if (n_valid == ATT3_G) {
            #pragma unroll
            for (int i = 0; i < ATT3_G; ++i) {
                const uint2 vr = *reinterpret_cast<const uint2*>(vbase + i * 128);
                const float2 v01 = att2_h2f(vr.x), v23 = att2_h2f(vr.y);
                #pragma unroll
                for (int h = 0; h < HG; ++h) {
                    const float a  = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(alpha[h]), i));
                    const float wv = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(wgt[h]), i));
                    l[h]     = att3_vfma(l[h], a, wv);
                    acc[h].x = __builtin_fmaf(acc[h].x, a, wv * v01.x);
                    acc[h].y = __builtin_fmaf(acc[h].y, a, wv * v01.y);
                    acc[h].z = __builtin_fmaf(acc[h].z, a, wv * v23.x);
                    acc[h].w = __builtin_fmaf(acc[h].w, a, wv * v23.y);
                }
            }
        } else {
            for (int i = 0; i < n_valid; ++i) {
                const uint2 vr = *reinterpret_cast<const uint2*>(vbase + i * 128);
                const float2 v01 = att2_h2f(vr.x), v23 = att2_h2f(vr.y);
                #pragma unroll
                for (int h = 0; h < HG; ++h) {
                    const float a  = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(alpha[h]), i));
                    const float wv = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(wgt[h]), i));
                    l[h]     = att3_vfma(l[h], a, wv);
                    acc[h].x = __builtin_fmaf(acc[h].x, a, wv * v01.x);
                    acc[h].y = __builtin_fmaf(acc[h].y, a, wv * v01.y);
                    acc[h].z = __builtin_fmaf(acc[h].z, a, wv * v23.x);
                    acc[h].w = __builtin_fmaf(acc[h].w, a, wv * v23.y);
                }
            }
        }
    }

    if (!wave_ok) return;
    if (n_splits == 1) {
        #pragma unroll
        for (int j = 0; j < HG; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                const float inv_l = (l[j] > 0.0f) ? (1.0f / l[j]) : 0.0f;
                float* out_head = out_or_scratch + (static_cast<std::size_t>(qt) * n_head + h) * 128;
                reinterpret_cast<float4*>(out_head)[lane] = make_float4(
                    acc[j].x * inv_l,
                    acc[j].y * inv_l,
                    acc[j].z * inv_l,
                    acc[j].w * inv_l);
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < HG; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                const std::size_t idx =
                    ((static_cast<std::size_t>(qt) * n_head + h) * n_splits + split_idx)
                    * K192_SPLIT_STRIDE_F32;
                float* dst = out_or_scratch + idx;
                reinterpret_cast<float4*>(dst)[lane] = acc[j];
                if (lane == 0) {
                    dst[128] = m[j];
                    dst[129] = l[j];
                }
            }
        }
    }
}

// Launch shapes for the v2 decode attention: query heads per wave (hq), waves per workgroup, positions in flight
// (u). Per-arch table keyed by the device's gfx target (gfx1151 measured; gfx1201 and others start from the same
// entries until measured). Env override for tuning: TALUS_MIMO_ATT2_FULL / TALUS_MIMO_ATT2_SWA = "hq:warps:u".
struct Att2Shape { int hq; int warps; int u; };

inline int att2_gfx_target() {
    static const int tgt = [] {
        int dev = 0;
        hipDeviceProp_t prop;
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return 0;
        int v = 0;
        if (std::sscanf(prop.gcnArchName, "gfx%x", &v) != 1) return 0;
        return v;   // e.g. 0x1151, 0x1201
    }();
    return tgt;
}

inline Att2Shape att2_env_shape(const char* name, Att2Shape def) {
    const char* e = std::getenv(name);
    Att2Shape s = def;
    if (e != nullptr && std::sscanf(e, "%d:%d:%d", &s.hq, &s.warps, &s.u) == 3) return s;
    return def;
}

inline Att2Shape att2_shape_full(int n_tok) {
    Att2Shape def{2, 8, 4};
    switch (att2_gfx_target()) {
        case 0x1151: def = (n_tok == 1) ? Att2Shape{2, 8, 4} : Att2Shape{2, 8, 4}; break;
        default:     break;
    }
    static const Att2Shape ov = att2_env_shape("TALUS_MIMO_ATT2_FULL", Att2Shape{0, 0, 0});
    return ov.hq > 0 ? ov : def;
}

inline Att2Shape att2_shape_swa(int n_tok) {
    Att2Shape def{1, 4, 4};
    switch (att2_gfx_target()) {
        case 0x1151: def = (n_tok == 1) ? Att2Shape{1, 4, 4} : Att2Shape{1, 4, 4}; break;
        default:     break;
    }
    static const Att2Shape ov = att2_env_shape("TALUS_MIMO_ATT2_SWA", Att2Shape{0, 0, 0});
    return ov.hq > 0 ? ov : def;
}

struct Att3Shape { int hg; int nw; };
inline Att3Shape att3_shape_full(int n_tok) {
    Att3Shape def = (n_tok == 1) ? Att3Shape{4, 4} : Att3Shape{2, 8};   // other targets: gfx1151's until measured
    switch (att2_gfx_target()) {
        // attb (random K/V, f16, stride 196608), us per call v1 -> v3: T=1 2K 154 -> 73 (4:4) / 70 (1:8) / 101 (2:8),
        // 32K 1355 -> 463 (4:4) / 436 (2:8); T=5 2K 247 -> 143 (2:8), 32K 3441 -> 1443 (2:8)
        case 0x1151: def = (n_tok == 1) ? Att3Shape{4, 4} : Att3Shape{2, 8}; break;
        default:     break;
    }
    static const Att3Shape ov = [] {
        Att3Shape s{0, 0};
        if (const char* e = std::getenv("TALUS_MIMO_ATT3_SHAPE")) std::sscanf(e, "%d:%d", &s.hg, &s.nw);
        return s;
    }();
    return ov.hg > 0 ? ov : def;
}

inline bool att3_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TALUS_MIMO_ATT3");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

inline bool att3_launch_full(Att3Shape sh, int g_ratio, const float* q, const void* k, const void* v, float* target,
                             int n_tok, int n_head, int n_kv, int stride, float scale, const std::int32_t* pos,
                             int splits, hipStream_t s) {
    if (sh.hg <= 0 || sh.nw <= 0 || (g_ratio % (sh.hg * sh.nw)) != 0) return false;
    const dim3 grid(static_cast<unsigned int>(n_head / (sh.hg * sh.nw)), static_cast<unsigned int>(n_tok),
                    static_cast<unsigned int>(splits));
#define X(H, W) if (sh.hg == H && sh.nw == W) { \
        mimo_attn_k192_v128_v3_kernel<H, W><<<grid, dim3(W * 32), 0, s>>>(q, k, v, target, n_tok, n_head, n_kv, \
                                                                         stride, scale, pos, splits); return true; }
    X(4, 4) X(2, 8) X(2, 4) X(1, 4) X(1, 8)
#undef X
    return false;
}

inline bool att2_drafter_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TALUS_MIMO_ATT2_DRAFT");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

inline bool att2_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("TALUS_MIMO_ATT2");
        return e == nullptr || e[0] != '0';
    }();
    return on;
}

// Returns false (nothing launched) for a shape outside the instantiated set: the caller keeps v1.
template <int HQ, int WARPS, int U>
void att2_launch_full_t(const float* q, const void* k, const void* v, float* target, int n_tok, int n_head, int n_kv,
                        int stride, float scale, const std::int32_t* pos, int splits, hipStream_t s) {
    const int h_per_wg = HQ * WARPS;
    const dim3 grid(static_cast<unsigned int>((n_head + h_per_wg - 1) / h_per_wg), static_cast<unsigned int>(n_tok),
                    static_cast<unsigned int>(splits));
    mimo_attn_k192_v128_v2_kernel<HQ, WARPS, U><<<grid, dim3(WARPS * 32), 0, s>>>(
        q, k, v, target, n_tok, n_head, n_kv, stride, scale, pos, splits);
}

template <int HQ, int WARPS, int U>
void att2_launch_swa_t(const float* q, const void* k, const void* v, const float* sinks, float* out, int n_tok,
                       int n_head, int n_kv, int window, int stride, float scale, const std::int32_t* pos,
                       float value_scale, hipStream_t s, int hdk = 192, bool causal = true, void* q8 = nullptr) {
    const int h_per_wg = HQ * WARPS;
    const dim3 grid(static_cast<unsigned int>((n_head + h_per_wg - 1) / h_per_wg), static_cast<unsigned int>(n_tok));
    Att2Q81* q8b = static_cast<Att2Q81*>(q8);
    if (hdk == 128) {
        mimo_attn_swa_sinks_v2_kernel<HQ, WARPS, U, 128><<<grid, dim3(WARPS * 32), 0, s>>>(
            q, k, v, sinks, out, n_tok, n_head, n_kv, window, stride, scale, pos, value_scale, causal, q8b);
    } else {
        mimo_attn_swa_sinks_v2_kernel<HQ, WARPS, U, 192><<<grid, dim3(WARPS * 32), 0, s>>>(
            q, k, v, sinks, out, n_tok, n_head, n_kv, window, stride, scale, pos, value_scale, causal, q8b);
    }
}

#define ATT2_SHAPES(X) X(1, 4, 4) X(1, 8, 4) X(2, 4, 4) X(2, 8, 4) X(4, 4, 4) X(1, 4, 2) X(2, 8, 2) X(2, 4, 2) \
                       X(1, 4, 8) X(2, 8, 8) X(2, 4, 8) X(4, 4, 2)

inline bool att2_launch_full(Att2Shape sh, int g_ratio, const float* q, const void* k, const void* v, float* target,
                             int n_tok, int n_head, int n_kv, int stride, float scale, const std::int32_t* pos,
                             int splits, hipStream_t s) {
    if (sh.hq <= 0 || (g_ratio % sh.hq) != 0) return false;
#define X(H, W, U) if (sh.hq == H && sh.warps == W && sh.u == U) { \
        att2_launch_full_t<H, W, U>(q, k, v, target, n_tok, n_head, n_kv, stride, scale, pos, splits, s); return true; }
    ATT2_SHAPES(X)
#undef X
    return false;
}

inline bool att2_launch_swa(Att2Shape sh, int g_ratio, const float* q, const void* k, const void* v,
                            const float* sinks, float* out, int n_tok, int n_head, int n_kv, int window, int stride,
                            float scale, const std::int32_t* pos, float value_scale, hipStream_t s, int hdk = 192,
                            bool causal = true, void* q8 = nullptr) {
    if (sh.hq <= 0 || (g_ratio % sh.hq) != 0) return false;
#define X(H, W, U) if (sh.hq == H && sh.warps == W && sh.u == U) { \
        att2_launch_swa_t<H, W, U>(q, k, v, sinks, out, n_tok, n_head, n_kv, window, stride, scale, pos, \
                                   value_scale, s, hdk, causal, q8); return true; }
    ATT2_SHAPES(X)
#undef X
    return false;
}


template <int HQ = 4, int WARPS = 8, int TILE_P = 16>
__global__ __launch_bounds__(WARPS * 32)
void mimo_attn_k192_v128_prefill_kernel(
    const float*         __restrict__ q,
    const void*          __restrict__ k_pool,
    const void*          __restrict__ v_pool,
    float*               __restrict__ out,
    int                               n_tok,
    int                               n_head,
    int                               n_kv,
    int                               stride,
    float                             scale,
    const std::int32_t*  __restrict__ pos_dev)
{
    __shared__ half s_k[TILE_P][192];
    __shared__ half s_v[TILE_P][128];

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int q_local  = warp >> 1; // 0..3
    const int h_local  = warp & 1;  // 0..1
    const int h_wg     = static_cast<int>(blockIdx.x) * 8;
    const int h0       = h_wg + h_local * HQ;
    const int qt0      = static_cast<int>(blockIdx.y) * 4;
    const int qt       = qt0 + q_local;
    const int qt_last  = (qt0 + 3 < n_tok) ? (qt0 + 3) : (n_tok - 1);
    if (h_wg >= n_head || qt0 >= n_tok) return;

    const int g_ratio = n_head / n_kv;
    const int kv_h    = h_wg / g_ratio;
    const bool valid  = (h0 < n_head && qt < n_tok);

    RegQ192 rq[HQ];
    float   m[HQ];
    float   l[HQ];
    float4  acc[HQ];

    if (valid) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = (h0 + j < n_head) ? (h0 + j) : h0;
            rq[j].load(q + (static_cast<std::size_t>(qt) * n_head + h) * 192, lane);
            m[j]   = -1e30f;
            l[j]   = 0.0f;
            acc[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const int base_pos   = pos_dev[0];
    const int q_pos      = base_pos + qt;
    const int kv_len     = (stride > 0 && q_pos + 1 > stride) ? stride : (q_pos + 1);
    const int q_pos_last = base_pos + qt_last;
    const int p_wg_end   = (stride > 0 && q_pos_last + 1 > stride) ? stride : (q_pos_last + 1);

    for (int p_tile = 0; p_tile < p_wg_end; p_tile += TILE_P) {
        const int tile_cnt = (p_wg_end - p_tile < TILE_P) ? (p_wg_end - p_tile) : TILE_P;

        for (int idx = tid; idx < tile_cnt * 96; idx += WARPS * 32) {
            const int t = idx / 96;
            const int c = idx % 96;
            const int p = p_tile + t;
            const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(p % stride) : static_cast<std::size_t>(p);
            const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kv_h);
            const half2* src = reinterpret_cast<const half2*>(static_cast<const half*>(k_pool) + head_idx * 192);
            reinterpret_cast<half2*>(s_k[t])[c] = src[c];
        }
        for (int idx = tid; idx < tile_cnt * 64; idx += WARPS * 32) {
            const int t = idx / 64;
            const int c = idx % 64;
            const int p = p_tile + t;
            const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(p % stride) : static_cast<std::size_t>(p);
            const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kv_h);
            const half2* src = reinterpret_cast<const half2*>(static_cast<const half*>(v_pool) + head_idx * 128);
            reinterpret_cast<half2*>(s_v[t])[c] = src[c];
        }
        __syncthreads();

        if (valid) {
            for (int t = 0; t < tile_cnt; ++t) {
                const int p = p_tile + t;
                if (p >= kv_len) break;
                const RegK192 rk = load_k192_smem(s_k[t], lane);
                const RegV128 rv = load_v128_smem(s_v[t], lane);

                #pragma unroll
                for (int j = 0; j < HQ; ++j) {
                    const float score = warp_sum_f32(rq[j].dot_local(rk)) * scale;
                    const float m_new = fmaxf(m[j], score);
                    const float alpha = __expf(m[j] - m_new);
                    const float w     = __expf(score - m_new);
                    l[j] = l[j] * alpha + w;
                    m[j] = m_new;
                    acc[j].x = acc[j].x * alpha + w * rv.v0;
                    acc[j].y = acc[j].y * alpha + w * rv.v1;
                    acc[j].z = acc[j].z * alpha + w * rv.v2;
                    acc[j].w = acc[j].w * alpha + w * rv.v3;
                }
            }
        }
        __syncthreads();
    }

    if (valid) {
        #pragma unroll
        for (int j = 0; j < HQ; ++j) {
            const int h = h0 + j;
            if (h < n_head) {
                const float inv_l = (l[j] > 0.0f) ? (1.0f / l[j]) : 0.0f;
                float* out_head = out + (static_cast<std::size_t>(qt) * n_head + h) * 128;
                reinterpret_cast<float4*>(out_head)[lane] = make_float4(
                    acc[j].x * inv_l,
                    acc[j].y * inv_l,
                    acc[j].z * inv_l,
                    acc[j].w * inv_l);
            }
        }
    }
}

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define MIMO_FA_WMMA 1
#endif

typedef _Float16 mimo_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 mimo_h8  __attribute__((ext_vector_type(8)));
typedef float    mimo_f8  __attribute__((ext_vector_type(8)));

__device__ __forceinline__ mimo_f8 mimo_wmma(mimo_h16 a, mimo_h16 b, mimo_f8 c) {
#if defined(MIMO_FA_WMMA)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a;
    (void) b;
    return c;
#endif
}

__device__ __forceinline__ mimo_h16 mimo_cat(mimo_h8 lo, mimo_h8 hi) {
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

__device__ __forceinline__ int mimo_swz_k192(int row, int unit) {
    return row * 192 + ((unit ^ (row & 7)) << 3);
}

/// RDNA3.5 WMMA FlashAttention prefill kernel for MiMo full-attention layers (`K192 / V128`):
/// - `grid = (ceil(n_tok * G / 64), n_kv)`, `block = 128` (`4` waves x `16` query columns = `64` query columns/WG)
/// - Keeps each wave's 16 query columns (`qf[12]` = `192` FP16 dims) and output accumulators (`o[8]` = `128` FP32 dims)
///   in registers across the entire key sequence `0 .. nkeys`, loading `kBK = 32` keys of `K` and `V` per tile into LDS
///   once per `64` query columns (`4` tokens x `16` GQA query heads).
__global__ __launch_bounds__(128)
void mimo_attn_k192_v128_wmma_prefill_kernel(
    const float*        __restrict__ q,
    const void*         __restrict__ k_pool,
    const void*         __restrict__ v_pool,
    float*              __restrict__ out,
    int                              n_tok,
    int                              n_head,
    int                              n_kv,
    int                              stride,
    float                            scale,
    const std::int32_t* __restrict__ pos_dev)
{
    constexpr int kBQ = 64;
    constexpr int kBK = 32;
    __shared__ __attribute__((aligned(16))) _Float16 Ks[kBK * 192];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[2 * 128 * 16];

    const int G    = n_head / n_kv;
    const int kvh  = static_cast<int>(blockIdx.y);
    const int c0   = static_cast<int>(blockIdx.x) * kBQ;
    const int ncol = n_tok * G;
    const int tid  = static_cast<int>(threadIdx.x);
    const int w    = tid >> 5;
    const int lane = tid & 31;
    const int l16  = lane & 15;
    const int hi   = lane >> 4;
    const unsigned sel_lo = hi ? 0x01000504u : 0x05040100u;
    const unsigned sel_hi = hi ? 0x03020706u : 0x07060302u;

    const int  p0     = pos_dev[0];
    const int  col    = c0 + w * 16 + l16;
    const bool colok  = (col < ncol);
    const int  qt_col = colok ? (col / G) : 0;
    const int  h_col  = kvh * G + (colok ? (col % G) : 0);
    const int  q_pos  = p0 + qt_col;
    const int  plim   = colok ? ((stride > 0 && q_pos + 1 > stride) ? (stride - 1) : q_pos) : -1;

    const int t_lo     = c0 / G;
    const int t_hi     = (c0 + kBQ - 1 < ncol) ? ((c0 + kBQ - 1) / G) : (n_tok - 1);
    const int nkeys_hi = p0 + t_hi + 1;
    const int nkeys    = (stride > 0 && nkeys_hi > stride) ? stride : nkeys_hi;
    const int kv_end   = nkeys;
    const int nkeys_lo = p0 + t_lo + 1;
    const int lim_lo   = (stride > 0 && nkeys_lo > stride) ? stride : nkeys_lo;
    const int full_end = (c0 + kBQ <= ncol) ? ((lim_lo < nkeys ? lim_lo : nkeys) / kBK) * kBK : 0;

    mimo_h16 qf[12];
    const float q_scale = scale * 1.44269504088896340736f; // log2(e)
    {
        const float* src = q + (static_cast<std::size_t>(qt_col) * n_head + h_col) * 192;
        #pragma unroll
        for (int ds = 0; ds < 12; ++ds) {
            #pragma unroll
            for (int e4 = 0; e4 < 4; ++e4) {
                float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (colok) {
                    a = *reinterpret_cast<const float4*>(src + ds * 16 + e4 * 4);
                }
                qf[ds][e4 * 4 + 0] = static_cast<_Float16>(a.x * q_scale);
                qf[ds][e4 * 4 + 1] = static_cast<_Float16>(a.y * q_scale);
                qf[ds][e4 * 4 + 2] = static_cast<_Float16>(a.z * q_scale);
                qf[ds][e4 * 4 + 3] = static_cast<_Float16>(a.w * q_scale);
            }
        }
    }

    mimo_f8 o[8];
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[dt][i] = 0.0f;
        }
    }
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const _Float16* k_pool_f16 = static_cast<const _Float16*>(k_pool);
    const _Float16* v_pool_f16 = static_cast<const _Float16*>(v_pool);
    const mimo_h8 z8 = {};

    auto run_tile = [&](int base, auto mask_tag) {
        constexpr bool kMask = decltype(mask_tag)::value;
        __syncthreads();

        #pragma unroll 2
        for (int j = 0; j < 6; ++j) {
            const int idx  = tid + 128 * j;
            const int key  = idx / 24;
            const int u    = idx % 24;
            const int cell = base + key;
            mimo_h8 v = z8;
            if (!kMask || cell < kv_end) {
                const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(cell % stride) : static_cast<std::size_t>(cell);
                const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kvh);
                v = *reinterpret_cast<const mimo_h8*>(k_pool_f16 + head_idx * 192 + u * 8);
            }
            *reinterpret_cast<mimo_h8*>(&Ks[mimo_swz_k192(key, u)]) = v;
        }

        #pragma unroll 2
        for (int j = 0; j < 4; ++j) {
            const int idx  = tid + 128 * j;
            const int key  = idx >> 4;
            const int u    = idx & 15;
            const int cell = base + key;
            mimo_h8 v = z8;
            if (!kMask || cell < kv_end) {
                const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(cell % stride) : static_cast<std::size_t>(cell);
                const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kvh);
                v = *reinterpret_cast<const mimo_h8*>(v_pool_f16 + head_idx * 128 + u * 8);
            }
            const int kt  = key >> 4;
            const int k16 = key & 15;
            #pragma unroll
            for (int e0 = 0; e0 < 8; ++e0) {
                const int e = e0 ^ (u & 3);
                Vs[(kt * 128 + u * 8 + e) * 16 + k16] = v[e];
            }
        }
        __syncthreads();

        mimo_f8 s[2];
        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[kt][i] = 0.0f;
            }
        }

        #pragma unroll
        for (int ds = 0; ds < 12; ++ds) {
            #pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                const int key = kt * 16 + l16;
                const mimo_h16 a = mimo_cat(
                    *reinterpret_cast<const mimo_h8*>(&Ks[mimo_swz_k192(key, 2 * ds)]),
                    *reinterpret_cast<const mimo_h8*>(&Ks[mimo_swz_k192(key, 2 * ds + 1)]));
                s[kt] = mimo_wmma(a, qf[ds], s[kt]);
            }
        }

        float mx = -INFINITY;
        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = s[kt][i];
                if (kMask && (base + kt * 16 + 2 * i + hi > plim)) {
                    v = -INFINITY;
                }
                s[kt][i] = v;
                mx = fmaxf(mx, v);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn    = fmaxf(m_run, mx);
        const float alpha = (mn == -INFINITY) ? 1.0f : __builtin_amdgcn_exp2f(m_run - mn);
        float ls = 0.0f;
        union PK { _Float16 h[2]; int i; } pk[2][4];
        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = (s[kt][i] == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(s[kt][i] - mn);
                ls += p;
                pk[kt][i >> 1].h[i & 1] = static_cast<_Float16>(p);
            }
        }
        ls += __shfl_xor(ls, 16, 32);
        l_run = l_run * alpha + ls;
        m_run = mn;

        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            union { mimo_h16 v; unsigned u[8]; } pb;
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                const unsigned own = static_cast<unsigned>(pk[kt][m].i);
                const unsigned par = static_cast<unsigned>(__shfl_xor(pk[kt][m].i, 16, 32));
                pb.u[2 * m]     = __builtin_amdgcn_perm(par, own, sel_lo);
                pb.u[2 * m + 1] = __builtin_amdgcn_perm(par, own, sel_hi);
            }
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                if (kt == 0) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        o[dt][i] *= alpha;
                    }
                }
                const mimo_h16 a = *reinterpret_cast<const mimo_h16*>(&Vs[(kt * 128 + dt * 16 + l16) * 16]);
                o[dt] = mimo_wmma(a, pb.v, o[dt]);
            }
        }
    };

    int base = 0;
    for (; base < full_end; base += kBK) {
        run_tile(base, std::false_type());
    }
    for (; base < nkeys; base += kBK) {
        run_tile(base, std::true_type());
    }

    if (!colok) return;
    const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    float* dst = out + (static_cast<std::size_t>(qt_col) * n_head + h_col) * 128;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = dt * 16 + 2 * i + hi;
            dst[d] = o[dt][i] * inv;
        }
    }
}

/// v2 of the WMMA FlashAttention prefill kernel above (TALUS_MIMO_FA_VER 2..5; 1 = v1).  Per query column the
/// instruction sequence is v1's (same 32-key tiles from key 0 in order, same LDS layouts / WMMA operands / softmax).
/// Changes: (a) the transposed V tile is written as whole dwords - lane pairs (keys 2m, 2m + 1) swap halves by DPP and
/// each writes 4 dwords - instead of 8 half stores with a per-lane element select (v1: ~110 VALU per tile); the LDS
/// contents are v1's; (b) NW waves per workgroup share each K / V tile (NW 8: 128 query columns); (c) kSkip: `o *= alpha`
/// is skipped when alpha == 1 for the whole wave (x * 1.0f == x bitwise).  Extra fully-masked tiles a column may run with
/// NW 8 (the workgroup's last token sets the key range) add exact zeros: all scores -inf -> mx = -inf, mn = m_run,
/// alpha = exp2(0) = 1, p = +0, l_run += +0, o += V^T * 0 (V finite).
template <int NW, bool kSkip, bool kPad = false>
__global__ __launch_bounds__(NW * 32)
void mimo_attn_k192_v128_wmma_prefill_v2_kernel(
    const float*        __restrict__ q,
    const void*         __restrict__ k_pool,
    const void*         __restrict__ v_pool,
    float*              __restrict__ out,
    int                              n_tok,
    int                              n_head,
    int                              n_kv,
    int                              stride,
    float                            scale,
    const std::int32_t* __restrict__ pos_dev)
{
    constexpr int kBQ = NW * 16;
    constexpr int kBK = 32;
    constexpr int kT  = NW * 32;
    // phase 2: 8 / 16 waves + padded rows issue all of a tile's global loads before its LDS stores (VMEM waits
    // at 30K: 5.3G -> 1.7G cycles; 30K chunk 124 -> 113 ms; the same values stored)
    constexpr bool kBatch = kPad && NW >= 8;
    // kPad (from the counters: LDS-wait bound with bank conflicts): K rows 200 halves apart (100 dwords: the 16
    // rows of a fragment read start in 16 distinct 4-bank groups) and V^T rows 24 halves apart (12 dwords: likewise),
    // instead of v1's 192 + XOR swizzle and 16; the operands read are the same values
    constexpr int KRS = kPad ? 200 : 192;   // halves per K row
    constexpr int VRS = kPad ? 24 : 16;     // halves per V^T row
    __shared__ __attribute__((aligned(16))) _Float16 Ks[kBK * KRS];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[2 * 128 * VRS];
    auto kidx = [](int row, int unit) { return kPad ? row * KRS + unit * 8 : mimo_swz_k192(row, unit); };

    const int G    = n_head / n_kv;
    const int kvh  = static_cast<int>(blockIdx.y);
    const int c0   = static_cast<int>(blockIdx.x) * kBQ;
    const int ncol = n_tok * G;
    const int tid  = static_cast<int>(threadIdx.x);
    const int w    = tid >> 5;
    const int lane = tid & 31;
    const int l16  = lane & 15;
    const int hi   = lane >> 4;
    const unsigned sel_lo = hi ? 0x01000504u : 0x05040100u;
    const unsigned sel_hi = hi ? 0x03020706u : 0x07060302u;

    const int  p0     = pos_dev[0];
    const int  col    = c0 + w * 16 + l16;
    const bool colok  = (col < ncol);
    const int  qt_col = colok ? (col / G) : 0;
    const int  h_col  = kvh * G + (colok ? (col % G) : 0);
    const int  q_pos  = p0 + qt_col;
    const int  plim   = colok ? ((stride > 0 && q_pos + 1 > stride) ? (stride - 1) : q_pos) : -1;

    const int t_lo     = c0 / G;
    const int t_hi     = (c0 + kBQ - 1 < ncol) ? ((c0 + kBQ - 1) / G) : (n_tok - 1);
    const int nkeys_hi = p0 + t_hi + 1;
    const int nkeys    = (stride > 0 && nkeys_hi > stride) ? stride : nkeys_hi;
    const int kv_end   = nkeys;
    const int nkeys_lo = p0 + t_lo + 1;
    const int lim_lo   = (stride > 0 && nkeys_lo > stride) ? stride : nkeys_lo;
    const int full_end = (c0 + kBQ <= ncol) ? ((lim_lo < nkeys ? lim_lo : nkeys) / kBK) * kBK : 0;
    // This wave reproduces v1's view exactly: v1's 4-wave workgroup of its columns (64 columns) has key range
    // [0, nkeys_w) - cells past it zero in v1's LDS tile, tiles past it not run.  A wider workgroup holds real data
    // there, and masked keys' values do reach the result at the ulp level (measured: talus-mimo-attn-bench p0 3, the
    // served greedy text), so the wave zeroes those K / V operands in masked tiles and skips tiles past nkeys_w.
    const int c1      = ((c0 + w * 16) / 64) * 64;
    const int t_hi1   = (c1 + 63 < ncol) ? ((c1 + 63) / G) : (n_tok - 1);
    const int nk1     = p0 + t_hi1 + 1;
    const int nkeys_w = (stride > 0 && nk1 > stride) ? stride : nk1;

    mimo_h16 qf[12];
    const float q_scale = scale * 1.44269504088896340736f; // log2(e)
    {
        const float* src = q + (static_cast<std::size_t>(qt_col) * n_head + h_col) * 192;
        #pragma unroll
        for (int ds = 0; ds < 12; ++ds) {
            #pragma unroll
            for (int e4 = 0; e4 < 4; ++e4) {
                float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (colok) {
                    a = *reinterpret_cast<const float4*>(src + ds * 16 + e4 * 4);
                }
                qf[ds][e4 * 4 + 0] = static_cast<_Float16>(a.x * q_scale);
                qf[ds][e4 * 4 + 1] = static_cast<_Float16>(a.y * q_scale);
                qf[ds][e4 * 4 + 2] = static_cast<_Float16>(a.z * q_scale);
                qf[ds][e4 * 4 + 3] = static_cast<_Float16>(a.w * q_scale);
            }
        }
    }

    mimo_f8 o[8];
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            o[dt][i] = 0.0f;
        }
    }
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // tile-load trip counts as runtime values (n_tok >= 1 always): keeps them loops, one unit's registers live at a time
    const int nku = (kBK * 24 + kT - 1) / kT + (n_tok <= 0 ? 1 : 0);   // (NW 16: 768 units over 512 threads)
    const int nvu = (kBK * 16) / kT + (n_tok <= 0 ? 1 : 0);
    const _Float16* k_pool_f16 = static_cast<const _Float16*>(k_pool);
    const _Float16* v_pool_f16 = static_cast<const _Float16*>(v_pool);
    const mimo_h8 z8 = {};

    auto run_tile = [&](int base, auto mask_tag) {
        constexpr bool kMask = decltype(mask_tag)::value;
        __syncthreads();

        if constexpr (kBatch) {
            // the tile's global loads all in flight before the LDS stores (compile-time trip counts, fully unrolled)
            constexpr int KU = (kBK * 24 + kT - 1) / kT, VU = (kBK * 16 + kT - 1) / kT;
            mimo_h8 kreg[KU];
            uint4 vreg[VU];
            #pragma unroll
            for (int j = 0; j < KU; ++j) {
                const int idx  = tid + kT * j;
                const int key  = idx / 24;
                const int cell = base + key;
                kreg[j] = z8;
                if (((kBK * 24) % kT == 0 || idx < kBK * 24) && (!kMask || cell < kv_end)) {
                    const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(cell % stride) : static_cast<std::size_t>(cell);
                    const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kvh);
                    kreg[j] = *reinterpret_cast<const mimo_h8*>(k_pool_f16 + head_idx * 192 + (idx % 24) * 8);
                }
            }
            #pragma unroll
            for (int j = 0; j < VU; ++j) {
                const int idx  = tid + kT * j;
                const int key  = (idx >> 4 & 1) * 16 + (idx & 15);
                const int cell = base + key;
                vreg[j] = make_uint4(0u, 0u, 0u, 0u);
                if (!kMask || cell < kv_end) {
                    const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(cell % stride) : static_cast<std::size_t>(cell);
                    const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kvh);
                    vreg[j] = *reinterpret_cast<const uint4*>(v_pool_f16 + head_idx * 128 + (idx >> 5) * 8);
                }
            }
            #pragma unroll
            for (int j = 0; j < KU; ++j) {
                const int idx  = tid + kT * j;
                if ((kBK * 24) % kT != 0 && idx >= kBK * 24) break;
                *reinterpret_cast<mimo_h8*>(&Ks[kidx(idx / 24, idx % 24)]) = kreg[j];
            }

            // V -> the transposed tile Vs[(kt * 128 + dim) * 16 + k16] (v1's layout): a unit = 8 dims of one key; the 16
            // lanes of a half-wave hold 16 consecutive keys of one dim group, lane pairs (k16 even / odd) swap halves so
            // each lane writes whole dwords {dim d of key 2m, of key 2m + 1}: the even lane dims 0-3, the odd lane 4-7.
            #pragma unroll
            for (int j = 0; j < VU; ++j) {
                const int idx  = tid + kT * j;
                const int k16  = idx & 15;
                const int rest = idx >> 4;
                const int kt   = rest & 1;
                const int u    = rest >> 1;
                const uint4 v  = vreg[j];
                const bool odd = (k16 & 1) != 0;
                const unsigned send0 = odd ? v.x : v.z, send1 = odd ? v.y : v.w;
                const unsigned mine0 = odd ? v.z : v.x, mine1 = odd ? v.w : v.y;
                // partner lane (lane ^ 1): DPP quad_perm [1, 0, 3, 2]
                const unsigned recv0 = static_cast<unsigned>(__builtin_amdgcn_mov_dpp(static_cast<int>(send0), 0xB1, 0xF, 0xF, true));
                const unsigned recv1 = static_cast<unsigned>(__builtin_amdgcn_mov_dpp(static_cast<int>(send1), 0xB1, 0xF, 0xF, true));
                // dword of dim (2c + h): lo half = the even key's, hi half = the odd key's
                //   even lane: lo from mine, hi from recv; odd lane: lo from recv, hi from mine
                // v_perm_b32(a, b, sel): bytes 0-3 = b, 4-7 = a
                const unsigned sel_l = odd ? 0x01000504u : 0x05040100u;   // {lo half of the even key, lo half of the odd key}
                const unsigned sel_h = odd ? 0x03020706u : 0x07060302u;   // {hi half of the even key, hi half of the odd key}
                unsigned* vd = reinterpret_cast<unsigned*>(Vs) + ((kt * 128 + u * 8 + (odd ? 4 : 0)) * VRS + (k16 & ~1)) / 2;
                vd[0 * (VRS / 2)] = __builtin_amdgcn_perm(recv0, mine0, sel_l);
                vd[1 * (VRS / 2)] = __builtin_amdgcn_perm(recv0, mine0, sel_h);
                vd[2 * (VRS / 2)] = __builtin_amdgcn_perm(recv1, mine1, sel_l);
                vd[3 * (VRS / 2)] = __builtin_amdgcn_perm(recv1, mine1, sel_h);
            }
        } else {   // (4 waves / unpadded: one unit's registers live at a time - the batched form spills there)
            #pragma unroll 1
            for (int j = 0; j < nku; ++j) {
                const int idx  = tid + kT * j;
                if ((kBK * 24) % kT != 0 && idx >= kBK * 24) break;
                const int key  = idx / 24;
                const int u    = idx % 24;
                const int cell = base + key;
                mimo_h8 v = z8;
                if (!kMask || cell < kv_end) {
                    const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(cell % stride) : static_cast<std::size_t>(cell);
                    const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kvh);
                    v = *reinterpret_cast<const mimo_h8*>(k_pool_f16 + head_idx * 192 + u * 8);
                }
                *reinterpret_cast<mimo_h8*>(&Ks[kidx(key, u)]) = v;
            }

            // V -> the transposed tile Vs[(kt * 128 + dim) * 16 + k16] (v1's layout): a unit = 8 dims of one key; the 16
            // lanes of a half-wave hold 16 consecutive keys of one dim group, lane pairs (k16 even / odd) swap halves so
            // each lane writes whole dwords {dim d of key 2m, of key 2m + 1}: the even lane dims 0-3, the odd lane 4-7.
            #pragma unroll 1
            for (int j = 0; j < nvu; ++j) {
                const int idx  = tid + kT * j;
                const int k16  = idx & 15;
                const int rest = idx >> 4;
                const int kt   = rest & 1;
                const int u    = rest >> 1;
                const int key  = kt * 16 + k16;
                const int cell = base + key;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (!kMask || cell < kv_end) {
                    const std::size_t slot = (stride > 0) ? static_cast<std::size_t>(cell % stride) : static_cast<std::size_t>(cell);
                    const std::size_t head_idx = slot * static_cast<std::size_t>(n_kv) + static_cast<std::size_t>(kvh);
                    v = *reinterpret_cast<const uint4*>(v_pool_f16 + head_idx * 128 + u * 8);
                }
                const bool odd = (k16 & 1) != 0;
                const unsigned send0 = odd ? v.x : v.z, send1 = odd ? v.y : v.w;
                const unsigned mine0 = odd ? v.z : v.x, mine1 = odd ? v.w : v.y;
                // partner lane (lane ^ 1): DPP quad_perm [1, 0, 3, 2]
                const unsigned recv0 = static_cast<unsigned>(__builtin_amdgcn_mov_dpp(static_cast<int>(send0), 0xB1, 0xF, 0xF, true));
                const unsigned recv1 = static_cast<unsigned>(__builtin_amdgcn_mov_dpp(static_cast<int>(send1), 0xB1, 0xF, 0xF, true));
                // dword of dim (2c + h): lo half = the even key's, hi half = the odd key's
                //   even lane: lo from mine, hi from recv; odd lane: lo from recv, hi from mine
                // v_perm_b32(a, b, sel): bytes 0-3 = b, 4-7 = a
                const unsigned sel_l = odd ? 0x01000504u : 0x05040100u;   // {lo half of the even key, lo half of the odd key}
                const unsigned sel_h = odd ? 0x03020706u : 0x07060302u;   // {hi half of the even key, hi half of the odd key}
                unsigned* vd = reinterpret_cast<unsigned*>(Vs) + ((kt * 128 + u * 8 + (odd ? 4 : 0)) * VRS + (k16 & ~1)) / 2;
                vd[0 * (VRS / 2)] = __builtin_amdgcn_perm(recv0, mine0, sel_l);
                vd[1 * (VRS / 2)] = __builtin_amdgcn_perm(recv0, mine0, sel_h);
                vd[2 * (VRS / 2)] = __builtin_amdgcn_perm(recv1, mine1, sel_l);
                vd[3 * (VRS / 2)] = __builtin_amdgcn_perm(recv1, mine1, sel_h);
            }
        }
        __syncthreads();
        if (base >= nkeys_w) return;   // (see nkeys_w)

        mimo_f8 s[2];
        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                s[kt][i] = 0.0f;
            }
        }

        #pragma unroll
        for (int ds = 0; ds < 12; ++ds) {
            #pragma unroll
            for (int kt = 0; kt < 2; ++kt) {
                const int key = kt * 16 + l16;
                mimo_h16 a = mimo_cat(
                    *reinterpret_cast<const mimo_h8*>(&Ks[kidx(key, 2 * ds)]),
                    *reinterpret_cast<const mimo_h8*>(&Ks[kidx(key, 2 * ds + 1)]));
                // masked tiles: keys past this wave's v1 range read as zero, as v1's tile holds them (the masked keys'
                // values reach the result at the ulp level: measured)
                if (kMask && base + key >= nkeys_w) a = mimo_h16{};
                s[kt] = mimo_wmma(a, qf[ds], s[kt]);
            }
        }

        float mx = -INFINITY;
        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = s[kt][i];
                if (kMask && (base + kt * 16 + 2 * i + hi > plim)) {
                    v = -INFINITY;
                }
                s[kt][i] = v;
                mx = fmaxf(mx, v);
            }
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float mn    = fmaxf(m_run, mx);
        const float alpha = (mn == -INFINITY) ? 1.0f : __builtin_amdgcn_exp2f(m_run - mn);
        float ls = 0.0f;
        union PK { _Float16 h[2]; int i; } pk[2][4];
        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = (s[kt][i] == -INFINITY) ? 0.0f : __builtin_amdgcn_exp2f(s[kt][i] - mn);
                ls += p;
                pk[kt][i >> 1].h[i & 1] = static_cast<_Float16>(p);
            }
        }
        ls += __shfl_xor(ls, 16, 32);
        l_run = l_run * alpha + ls;
        m_run = mn;
        // x * 1.0f == x for every float: skipping the rescale when the whole wave has alpha == 1 is bitwise
        // kSkip: x * 1.0f == x for every float, so the rescale is skipped when the whole wave has alpha == 1
        if (kSkip && !__all(alpha == 1.0f)) {
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    o[dt][i] *= alpha;
                }
            }
        }

        #pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            union { mimo_h16 v; unsigned u[8]; } pb;
            #pragma unroll
            for (int m = 0; m < 4; ++m) {
                const unsigned own = static_cast<unsigned>(pk[kt][m].i);
                const unsigned par = static_cast<unsigned>(__shfl_xor(pk[kt][m].i, 16, 32));
                pb.u[2 * m]     = __builtin_amdgcn_perm(par, own, sel_lo);
                pb.u[2 * m + 1] = __builtin_amdgcn_perm(par, own, sel_hi);
            }
            #pragma unroll
            for (int dt = 0; dt < 8; ++dt) {
                if (!kSkip && kt == 0) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        o[dt][i] *= alpha;
                    }
                }
                const _Float16* vrow = &Vs[(kt * 128 + dt * 16 + l16) * VRS];
                mimo_h16 a = mimo_cat(*reinterpret_cast<const mimo_h8*>(vrow), *reinterpret_cast<const mimo_h8*>(vrow + 8));
                if (kMask) {   // keys (elements) past this wave's v1 range: zero, as in v1's tile
                    const int nv = nkeys_w - base - kt * 16;
                    #pragma unroll
                    for (int e = 0; e < 16; ++e) {
                        if (e >= nv) a[e] = static_cast<_Float16>(0.0f);
                    }
                }
                o[dt] = mimo_wmma(a, pb.v, o[dt]);
            }
        }
    };

    int base = 0;
    for (; base < full_end; base += kBK) {
        run_tile(base, std::false_type());
    }
    for (; base < nkeys; base += kBK) {
        run_tile(base, std::true_type());
    }

    if (!colok) return;
    const float inv = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
    float* dst = out + (static_cast<std::size_t>(qt_col) * n_head + h_col) * 128;
    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int d = dt * 16 + 2 * i + hi;
            dst[d] = o[dt][i] * inv;
        }
    }
}

// ============================================================================
// 5. Grouped MoE Prefill MMQ for IQ2 / IQ3 / IQ1 / Q2_K Experts
//    - Workgroup tile: `M_TILE = 16` weight rows (`4` waves x `4` rows/wave)
//      x `N_TILE = 8` token rows
//    - Stages IQ lookup grid in LDS (`s_grid`) AND stages `N_TILE = 8` token rows'
//      `block_q8_1` activations in conflict-free LDS (`s_u[n][j][lane]`, `s_d8[n][lane]`)
//    - Each weight sub-block is loaded from DRAM & unpacked from `s_grid` ONCE
//      and dotted via `dp4a` (`__builtin_amdgcn_sudot4`) against all `nt <= 8` token rows!
// ============================================================================

template <int TY, int WARPS = 4, int ROWS = 4, int N_TILE = 8>
__global__ __launch_bounds__(WARPS * 32, 2)
void mimo_grouped_mmq_kernel(
    const char*         __restrict__ w,
    std::int64_t                     w_cols,
    std::int64_t                     w_rows,
    std::size_t                      expert_stride_bytes,
    const block_q8_1*   __restrict__ x_q8_base,
    const std::int32_t* __restrict__ ids_dev,
    const std::int32_t* __restrict__ bounds_dev,
    float*              __restrict__ dst_f32,
    std::int64_t                     ld_dst,
    const std::int32_t* __restrict__ expert_map_dev)
{
    using S   = MimoSplit<TY>;
    using blk = typename S::blk;
    using Raw = typename S::Raw;
    using W   = typename S::W;
    constexpr bool STAGE_GRID = (TY == 16 || TY == 17 || TY == 22 || TY == 29);

    const int q     = static_cast<int>(blockIdx.z);
    const int r_beg = bounds_dev[q];
    const int r_end = bounds_dev[q + 1];
    const int r0    = r_beg + static_cast<int>(blockIdx.y) * N_TILE;
    if (r0 >= r_end) return;

    const int nt = (r_end - r0 < N_TILE) ? (r_end - r0) : N_TILE;
    const int e  = (expert_map_dev != nullptr) ? expert_map_dev[q] : q;
    if (e < 0) return;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ typename S::grid_t s_grid[(STAGE_GRID && S::GRID_LEN > 0) ? S::GRID_LEN : 1];
    __shared__ int                s_u[N_TILE][8][32];
    __shared__ float              s_d8[N_TILE][32];

    if constexpr (STAGE_GRID && S::GRID_LEN > 0) {
        S::init_grid(s_grid, tid, WARPS * 32);
    }
    const typename S::grid_t* grid = (STAGE_GRID && S::GRID_LEN > 0) ? s_grid : S::global_grid();

    const int row0  = (static_cast<int>(blockIdx.x) * WARPS + warp) * ROWS;
    const int bpr   = static_cast<int>(w_cols / S::QK);
    const int q8_pr = static_cast<int>(w_cols / QK8_1);
    const int grp   = lane >> 3;
    const int sub   = lane & 7;
    const int q8_col_in_tile = grp * 8 + sub;

    const blk* w_exp = reinterpret_cast<const blk*>(w + static_cast<std::size_t>(e) * expert_stride_bytes);
    const blk* row_b[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        const int row = (row0 + r < static_cast<int>(w_rows)) ? (row0 + r) : (static_cast<int>(w_rows) - 1);
        row_b[r] = w_exp + static_cast<std::size_t>(row) * bpr;
    }

    float acc[N_TILE][ROWS];
    #pragma unroll
    for (int n = 0; n < N_TILE; ++n) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            acc[n][r] = 0.0f;
        }
    }

    for (int kbx_base = 0; kbx_base < bpr; kbx_base += 4) {
        // Cooperatively stage `nt` token rows x 32 `block_q8_1` columns into conflict-free LDS
        const int q8_idx = kbx_base * 8 + lane;
        if (q8_idx < q8_pr) {
            #pragma unroll
            for (int n = warp; n < N_TILE; n += WARPS) {
                if (n < nt) {
                    const block_q8_1& bq8 = x_q8_base[static_cast<std::size_t>(r0 + n) * q8_pr + q8_idx];
                    const int* q32 = reinterpret_cast<const int*>(bq8.qs);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        s_u[n][j][lane] = q32[j];
                    }
                    s_d8[n][lane] = __low2float(bq8.ds);
                }
            }
        }
        __syncthreads();

        const int kbx = kbx_base + grp;
        if (kbx < bpr && row0 < static_cast<int>(w_rows)) {
            Raw raw[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                raw[r] = S::load_raw(row_b[r], kbx, sub);
            }
            W wv[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                wv[r] = S::unpack_w(raw[r], grid);
            }
            #pragma unroll
            for (int n = 0; n < N_TILE; ++n) {
                if (n < nt) {
                    int u_local[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        u_local[j] = s_u[n][j][q8_col_in_tile];
                    }
                    const float d8_local = s_d8[n][q8_col_in_tile];
                    #pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        acc[n][r] += S::dot(wv[r], u_local, d8_local);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row0 >= static_cast<int>(w_rows)) return;

    #pragma unroll
    for (int n = 0; n < N_TILE; ++n) {
        if (n < nt) {
            const float sum = mimo_reduce_rows<ROWS>(acc[n], lane);
            if (lane < ROWS && row0 + lane < static_cast<int>(w_rows)) {
                const int dst_row = (ids_dev != nullptr) ? ids_dev[r0 + n] : (r0 + n);
                dst_f32[static_cast<std::size_t>(dst_row) * ld_dst + row0 + lane] = sum;
            }
        }
    }
}

template <int TY>
void launch_mimo_grouped_mmq(
    const void*         w,
    std::int64_t        w_cols,
    std::int64_t        w_rows,
    std::size_t         expert_stride_bytes,
    const block_q8_1*   x_q8_base,
    const std::int32_t* ids_dev,
    const std::int32_t* bounds_dev,
    float*              dst_f32,
    std::int64_t        ld_dst,
    std::int32_t        n_experts_active,
    std::int64_t        max_rows,
    void*               stream,
    const std::int32_t* expert_map_dev)
{
    constexpr int WARPS  = 4;
    constexpr int ROWS   = 4;
    constexpr int M_TILE = WARPS * ROWS; // 16 weight rows
    constexpr int N_TILE = 8;            // 8 token rows

    const dim3 block(WARPS * 32);
    const dim3 grid(
        static_cast<unsigned int>((w_rows + M_TILE - 1) / M_TILE),
        static_cast<unsigned int>((max_rows + N_TILE - 1) / N_TILE),
        static_cast<unsigned int>(n_experts_active));

    mimo_grouped_mmq_kernel<TY, WARPS, ROWS, N_TILE><<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
        static_cast<const char*>(w),
        w_cols,
        w_rows,
        expert_stride_bytes,
        x_q8_base,
        ids_dev,
        bounds_dev,
        dst_f32,
        ld_dst,
        expert_map_dev);
}

} // namespace

// ============================================================================
// Public Launcher Implementations
// ============================================================================

void mimo_quantize_q8_1(
    const float* x_f32,
    void*        x_q8_1,
    std::int64_t n_rows,
    std::int64_t n_cols,
    void*        stream)
{
    if (n_rows <= 0 || n_cols <= 0) return;
    const std::int64_t total_blocks = n_rows * (n_cols / 32);
    if (total_blocks <= 0) return;
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned int>((total_blocks + 7) / 8));
    mimo_quantize_q8_1_kernel<<<grid, block, 0, static_cast<hipStream_t>(stream)>>>(
        x_f32,
         static_cast<block_q8_1*>(x_q8_1),
        total_blocks);
}

void mimo_mmvq_ids(
    int                 ggml_type,
    const void*         weights,
    std::size_t         expert_bytes,
    const std::int32_t* ids,
    int                 n_ids,
    const void*         x_q8_1,
    bool                x_per_id,
    float*              y,
    int                 n_in,
    int                 n_out,
    void*               stream,
    const void*         weights2,
    float*              y2,
    int                 x_group)
{
    if (n_ids <= 0 || n_in <= 0 || n_out <= 0) return;
    switch (ggml_type) {
        case 16: launch_mimo_mmvq_ids<16>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, stream, weights2, y2, x_group); break;
        case 17: launch_mimo_mmvq_ids<17>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, stream, weights2, y2, x_group); break;
        case 18: launch_mimo_mmvq_ids<18>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, stream, weights2, y2, x_group); break;
        case 21: launch_mimo_mmvq_ids<21>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, stream, weights2, y2, x_group); break;
        case 22: launch_mimo_mmvq_ids<22>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, stream, weights2, y2, x_group); break;
        case 29: launch_mimo_mmvq_ids<29>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, stream, weights2, y2, x_group); break;
        case 10: launch_mimo_mmvq_ids<10>(weights, expert_bytes, ids, n_ids, x_q8_1, x_per_id, y, n_in, n_out, stream, weights2, y2, x_group); break;
        default: break;
    }
}

void mimo_attn_swa_sinks(
    const float*         q,
    const void*          k_pool,
    const void*          v_pool,
    const std::uint16_t* k_scale,
    const std::uint16_t* v_scale,
    const float*         sinks,
    float*               out,
    float*               scratch,
    int                  n_tok,
    int                  n_head,
    int                  n_kv,
    int                  head_dim_k,
    int                  head_dim_v,
    int                  window,
    int                  stride,
    float                scale,
    const std::int32_t*  pos_dev,
    int                  n_split,
    void*                stream,
    float                value_scale,
    void*                q8_out,
    void*                q8_mmq,
    int                  q8_mmq_layout,
    bool*                q8_mmq_done)
{
    (void)scratch;
    if (q8_mmq_done != nullptr) *q8_mmq_done = false;
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0 || (head_dim_k != 192 && head_dim_k != 128) || head_dim_v != 128) return;
    // q8_out: also write the outputs as Q8_1 (tk_quantize_q8_1's bytes): fused into the v2 epilogue, else a
    // native_quantize_q8_1 launch after the kernel.
    auto q8_after = [&]() {
        if (q8_out != nullptr) native_quantize_q8_1(out, q8_out, n_tok * n_head * head_dim_v, 1, stream);
    };

    const bool q8_kv   = (k_scale != nullptr && v_scale != nullptr);
    const bool causal  = (n_split != -2);
    const int  g_ratio = n_head / n_kv;
    const auto hip_s   = static_cast<hipStream_t>(stream);

    // the v2 prefill kernel (bitwise; G = 8 heads per KV head) by default; TALUS_MIMO_SWA_VER=1: the v1 kernel
    static const int swa_ver = [] {
        const char* e = std::getenv("TALUS_MIMO_SWA_VER");
        return e ? std::atoi(e) : 2;
    }();
    if (swa_ver == 2 && causal && head_dim_k == 192 && !q8_kv && n_tok >= 8 && g_ratio == 8) {
        const dim3 block(256);
        const dim3 grid(static_cast<unsigned int>((n_tok + 7) / 8), static_cast<unsigned int>(n_kv));
        const bool mmq = q8_mmq != nullptr && q8_out == nullptr && (q8_mmq_layout == 0 || q8_mmq_layout == 1) &&
                         n_head * head_dim_v % 512 == 0;
        mimo_attn_swa_sinks_prefill_v2_kernel<8><<<grid, block, 0, hip_s>>>(
            q, k_pool, v_pool, sinks, out, n_tok, n_head, n_kv, window, stride, scale, pos_dev, value_scale,
            mmq ? q8_mmq : nullptr, q8_mmq_layout);
        if (mmq && q8_mmq_done != nullptr) *q8_mmq_done = true;
        q8_after();
        return;
    }
    if (causal && head_dim_k == 192 && !q8_kv && n_tok >= 8 && (g_ratio % 8) == 0) {
        const dim3 block(8 * 32);
        const dim3 grid(
            static_cast<unsigned int>((n_head + 7) / 8),
            static_cast<unsigned int>((n_tok + 3) / 4));
        mimo_attn_swa_sinks_prefill_kernel<4, 8, 16><<<grid, block, 0, hip_s>>>(
            q, k_pool, v_pool, sinks, out,
            n_tok, n_head, n_kv, window, stride, scale, pos_dev, value_scale);
        q8_after();
        return;
    }

    if (causal && head_dim_k == 192 && !q8_kv && att2_enabled() &&
        att2_launch_swa(att2_shape_swa(n_tok), g_ratio, q, k_pool, v_pool, sinks, out, n_tok, n_head, n_kv, window,
                        stride, scale, pos_dev, value_scale, hip_s, 192, true, q8_out)) {
        return;
    }
    // DFlash drafter (HDK 128, non-causal block over the window): v2 as well
    if (head_dim_k == 128 && !q8_kv && att2_enabled() && att2_drafter_enabled() &&
        att2_launch_swa(att2_shape_swa(n_tok), g_ratio, q, k_pool, v_pool, sinks, out, n_tok, n_head, n_kv, window,
                        stride, scale, pos_dev, value_scale, hip_s, 128, causal, q8_out)) {
        return;
    }

    auto launch_decode = [&]<int HDK>() {
        if ((g_ratio % 4) == 0) {
            constexpr int HQ    = 4;
            constexpr int WARPS = 2;
            const int h_per_wg  = HQ * WARPS; // 8 query heads per workgroup
            const dim3 block(WARPS * 32);
            const dim3 grid(
                static_cast<unsigned int>((n_head + h_per_wg - 1) / h_per_wg),
                static_cast<unsigned int>(n_tok));
            if (q8_kv) {
                mimo_attn_swa_sinks_kernel<true, HQ, WARPS, HDK><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, k_scale, v_scale, sinks, out,
                    n_tok, n_head, n_kv, window, stride, scale, pos_dev, value_scale, causal);
            } else {
                mimo_attn_swa_sinks_kernel<false, HQ, WARPS, HDK><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, k_scale, v_scale, sinks, out,
                    n_tok, n_head, n_kv, window, stride, scale, pos_dev, value_scale, causal);
            }
        } else {
            constexpr int HQ    = 1;
            constexpr int WARPS = 2;
            const int h_per_wg  = HQ * WARPS;
            const dim3 block(WARPS * 32);
            const dim3 grid(
                static_cast<unsigned int>((n_head + h_per_wg - 1) / h_per_wg),
                static_cast<unsigned int>(n_tok));
            if (q8_kv) {
                mimo_attn_swa_sinks_kernel<true, HQ, WARPS, HDK><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, k_scale, v_scale, sinks, out,
                    n_tok, n_head, n_kv, window, stride, scale, pos_dev, value_scale, causal);
            } else {
                mimo_attn_swa_sinks_kernel<false, HQ, WARPS, HDK><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, k_scale, v_scale, sinks, out,
                    n_tok, n_head, n_kv, window, stride, scale, pos_dev, value_scale, causal);
            }
        }
    };

    if (head_dim_k == 128) {
        launch_decode.template operator()<128>();
    } else {
        launch_decode.template operator()<192>();
    }
    q8_after();
}

void mimo_attn_k192_v128(
    const float*         q,
    const void*          k_pool,
    const void*          v_pool,
    const std::uint16_t* k_scale,
    const std::uint16_t* v_scale,
    float*               out,
    float*               scratch,
    int                  n_tok,
    int                  n_head,
    int                  n_kv,
    int                  stride,
    float                scale,
    const std::int32_t*  pos_dev,
    int                  n_split,
    void*                stream,
    void*                q8_out)
{
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0) return;
    auto q8_after = [&]() {
        if (q8_out != nullptr) native_quantize_q8_1(out, q8_out, n_tok * n_head * 128, 1, stream);
    };

    const bool q8_kv   = (k_scale != nullptr && v_scale != nullptr);
    const int  g_ratio = n_head / n_kv;
    const auto hip_s   = static_cast<hipStream_t>(stream);

    if (!q8_kv && n_tok >= 8 && n_split <= 1 && (g_ratio % 8) == 0) {
        static const bool use_wmma = [] {
            const char* e = std::getenv("TALUS_MIMO_FA");
            return e == nullptr || e[0] != '0';
        }();
        // TALUS_MIMO_FA_VER 7 (default) = v2 with 8 waves + padded LDS rows + batched tile loads (bitwise; 30K: 150
        // -> 113 ms per chunk);
        // 1 = the v1 kernel; 2 / 3 = 4 waves (+ alpha skip), 4 / 5 = 8 waves (+ skip), 6 = 4 waves + padded rows
        static const int fa_ver = [] {
            const char* e = std::getenv("TALUS_MIMO_FA_VER");
            return e ? std::atoi(e) : 7;
        }();
        if (use_wmma && (n_head % n_kv) == 0 && fa_ver == 8) {   // v2 + padded LDS rows, 16 waves (256 query columns)
            const dim3 block(16 * 32);
            const dim3 grid(static_cast<unsigned int>((n_tok * g_ratio + 255) / 256), static_cast<unsigned int>(n_kv));
            mimo_attn_k192_v128_wmma_prefill_v2_kernel<16, false, true><<<grid, block, 0, hip_s>>>(
                q, k_pool, v_pool, out, n_tok, n_head, n_kv, stride, scale, pos_dev);
            q8_after();
            return;
        }
        if (use_wmma && (n_head % n_kv) == 0 && (fa_ver == 6 || fa_ver == 7)) {   // v2 + padded LDS rows (4 / 8 waves)
            const int nw = fa_ver == 6 ? 4 : 8;
            const dim3 block(nw * 32);
            const dim3 grid(
                static_cast<unsigned int>((n_tok * g_ratio + nw * 16 - 1) / (nw * 16)),
                static_cast<unsigned int>(n_kv));
            if (fa_ver == 6)
                mimo_attn_k192_v128_wmma_prefill_v2_kernel<4, false, true><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, out, n_tok, n_head, n_kv, stride, scale, pos_dev);
            else
                mimo_attn_k192_v128_wmma_prefill_v2_kernel<8, false, true><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, out, n_tok, n_head, n_kv, stride, scale, pos_dev);
            q8_after();
            return;
        }
        if (use_wmma && (n_head % n_kv) == 0 && fa_ver >= 2 && fa_ver <= 5) {
            const int nw = fa_ver <= 3 ? 4 : 8;
            const dim3 block(nw * 32);
            const dim3 grid(
                static_cast<unsigned int>((n_tok * g_ratio + nw * 16 - 1) / (nw * 16)),
                static_cast<unsigned int>(n_kv));
            if (fa_ver == 2)
                mimo_attn_k192_v128_wmma_prefill_v2_kernel<4, false><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, out, n_tok, n_head, n_kv, stride, scale, pos_dev);
            else if (fa_ver == 3)
                mimo_attn_k192_v128_wmma_prefill_v2_kernel<4, true><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, out, n_tok, n_head, n_kv, stride, scale, pos_dev);
            else if (fa_ver == 4)
                mimo_attn_k192_v128_wmma_prefill_v2_kernel<8, false><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, out, n_tok, n_head, n_kv, stride, scale, pos_dev);
            else
                mimo_attn_k192_v128_wmma_prefill_v2_kernel<8, true><<<grid, block, 0, hip_s>>>(
                    q, k_pool, v_pool, out, n_tok, n_head, n_kv, stride, scale, pos_dev);
            q8_after();
            return;
        }
        if (use_wmma && (n_head % n_kv) == 0) {
            const dim3 block(128);
            const dim3 grid(
                static_cast<unsigned int>((n_tok * g_ratio + 63) / 64),
                static_cast<unsigned int>(n_kv));
            mimo_attn_k192_v128_wmma_prefill_kernel<<<grid, block, 0, hip_s>>>(
                q, k_pool, v_pool, out,
                n_tok, n_head, n_kv, stride, scale, pos_dev);
            q8_after();
            return;
        }
        const dim3 block(8 * 32);
        const dim3 grid(
            static_cast<unsigned int>((n_head + 7) / 8),
            static_cast<unsigned int>((n_tok + 3) / 4));
        mimo_attn_k192_v128_prefill_kernel<4, 8, 16><<<grid, block, 0, hip_s>>>(
            q, k_pool, v_pool, out,
            n_tok, n_head, n_kv, stride, scale, pos_dev);
        q8_after();
        return;
    }

    int splits = 1;
    if (scratch != nullptr) {
        if (n_split > 1) {
            splits = (n_split <= 32) ? n_split : 32;
        } else if (n_split <= 0) {
            if (stride >= 16384) {
                splits = 16;
            } else if (stride >= 2048) {
                splits = 8;
            } else if (stride >= 512) {
                splits = 4;
            }
        }
    }

    float* target = (splits > 1) ? scratch : out;

    if (!q8_kv && att3_enabled() &&
        att3_launch_full(att3_shape_full(n_tok), g_ratio, q, k_pool, v_pool, target, n_tok, n_head, n_kv, stride,
                         scale, pos_dev, splits, hip_s)) {
        // v3 written; combine below
    } else if (!q8_kv && att2_enabled() &&
        att2_launch_full(att2_shape_full(n_tok), g_ratio, q, k_pool, v_pool, target, n_tok, n_head, n_kv, stride,
                         scale, pos_dev, splits, hip_s)) {
        // v2 written; combine below
    } else if ((g_ratio % 4) == 0) {
        constexpr int HQ    = 4;
        constexpr int WARPS = 4;
        const int h_per_wg  = HQ * WARPS; // 16 query heads per workgroup
        const dim3 block(WARPS * 32);
        const dim3 grid(
            static_cast<unsigned int>((n_head + h_per_wg - 1) / h_per_wg),
            static_cast<unsigned int>(n_tok),
            static_cast<unsigned int>(splits));
        if (q8_kv) {
            mimo_attn_k192_v128_kernel<true, HQ, WARPS><<<grid, block, 0, hip_s>>>(
                q, k_pool, v_pool, k_scale, v_scale, target,
                n_tok, n_head, n_kv, stride, scale, pos_dev, splits);
        } else {
            mimo_attn_k192_v128_kernel<false, HQ, WARPS><<<grid, block, 0, hip_s>>>(
                q, k_pool, v_pool, k_scale, v_scale, target,
                n_tok, n_head, n_kv, stride, scale, pos_dev, splits);
        }
    } else {
        constexpr int HQ    = 1;
        constexpr int WARPS = 4;
        const int h_per_wg  = HQ * WARPS;
        const dim3 block(WARPS * 32);
        const dim3 grid(
            static_cast<unsigned int>((n_head + h_per_wg - 1) / h_per_wg),
            static_cast<unsigned int>(n_tok),
            static_cast<unsigned int>(splits));
        if (q8_kv) {
            mimo_attn_k192_v128_kernel<true, HQ, WARPS><<<grid, block, 0, hip_s>>>(
                q, k_pool, v_pool, k_scale, v_scale, target,
                n_tok, n_head, n_kv, stride, scale, pos_dev, splits);
        } else {
            mimo_attn_k192_v128_kernel<false, HQ, WARPS><<<grid, block, 0, hip_s>>>(
                q, k_pool, v_pool, k_scale, v_scale, target,
                n_tok, n_head, n_kv, stride, scale, pos_dev, splits);
        }
    }

    if (splits > 1) {
        const dim3 c_block(128);
        const dim3 c_grid(
            static_cast<unsigned int>((n_head + 3) / 4),
            static_cast<unsigned int>(n_tok));
        mimo_attn_k192_v128_combine_kernel<<<c_grid, c_block, 0, hip_s>>>(
            scratch, out, n_tok, n_head, splits, static_cast<Att2Q81*>(q8_out));
    } else {
        q8_after();
    }
}

void mimo_prefill_moe_mmq_q8(
    int                 ggml_type,
    const void*         w,
    std::int64_t        w_cols,
    std::int64_t        w_rows,
    std::size_t         expert_stride_bytes,
    const void*         x_q8_1,
    const std::int32_t* ids_dev,
    const std::int32_t* bounds_dev,
    float*              dst_f32,
    std::int64_t        total_rows,
    std::int64_t        ld_dst,
    std::int32_t        n_experts_active,
    std::int64_t        max_rows,
    void*               stream,
    const std::int32_t* expert_map_dev)
{
    if (total_rows <= 0 || max_rows <= 0 || n_experts_active <= 0 || w_cols <= 0 || w_rows <= 0 || x_q8_1 == nullptr) {
        return;
    }
    const auto* x_q8_base = static_cast<const block_q8_1*>(x_q8_1);
    switch (ggml_type) {
        case 16: launch_mimo_grouped_mmq<16>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 17: launch_mimo_grouped_mmq<17>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 18: launch_mimo_grouped_mmq<18>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 21: launch_mimo_grouped_mmq<21>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 22: launch_mimo_grouped_mmq<22>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 29: launch_mimo_grouped_mmq<29>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 10: launch_mimo_grouped_mmq<10>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        default: break;
    }
}

void mimo_prefill_moe_mmq(
    int                 ggml_type,
    const void*         w,
    std::int64_t        w_cols,
    std::int64_t        w_rows,
    std::size_t         expert_stride_bytes,
    const float*        x_f32,
    const std::int32_t* ids_dev,
    const std::int32_t* bounds_dev,
    float*              dst_f32,
    std::int64_t        total_rows,
    std::int64_t        ld_dst,
    std::int32_t        n_experts_active,
    std::int64_t        max_rows,
    void*               scratch_xq,
    std::size_t         scratch_bytes,
    void*               stream,
    const std::int32_t* expert_map_dev)
{
    if (total_rows <= 0 || max_rows <= 0 || n_experts_active <= 0 || w_cols <= 0 || w_rows <= 0) {
        return;
    }
    const std::size_t needed_q8_bytes =
        static_cast<std::size_t>(total_rows) * static_cast<std::size_t>(w_cols / 32) * sizeof(block_q8_1);
    if (scratch_xq == nullptr || scratch_bytes < needed_q8_bytes) {
        return;
    }

    mimo_quantize_q8_1(x_f32, scratch_xq, total_rows, w_cols, stream);
    const auto* x_q8_base = static_cast<const block_q8_1*>(scratch_xq);

    switch (ggml_type) {
        case 16: launch_mimo_grouped_mmq<16>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 17: launch_mimo_grouped_mmq<17>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 18: launch_mimo_grouped_mmq<18>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 21: launch_mimo_grouped_mmq<21>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 22: launch_mimo_grouped_mmq<22>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 29: launch_mimo_grouped_mmq<29>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        case 10: launch_mimo_grouped_mmq<10>(w, w_cols, w_rows, expert_stride_bytes, x_q8_base, ids_dev, bounds_dev, dst_f32, ld_dst, n_experts_active, max_rows, stream, expert_map_dev); break;
        default: break;
    }
}


// ============================================================================
// 4d. decode / verify q + k NeoX rope (partial, n_rot of head_dim) and the f16 K / V cache store in one launch.
//     Bitwise rope_neox_par_kernel on q and k (same per-pair rope_neox_pair, same mrope table lookup) followed by
//     k_kv_store_f16 (cell = (store_pos[0] + t) % ring, __float2half of the rotated k / v).  Replaces three launches.
// ============================================================================
__global__ void mimo_rope_qk_store_kernel(float* __restrict__ q, float* __restrict__ k, const float* __restrict__ v,
                                          int n_tok, int n_head_q, int n_head_kv, int head_dim, int v_dim, int n_rot,
                                          const float* __restrict__ cos_tab, const float* __restrict__ sin_tab,
                                          const std::int32_t* __restrict__ pos_q, const std::int32_t* __restrict__ pos_k,
                                          half* __restrict__ k_pool, half* __restrict__ v_pool,
                                          const std::int32_t* __restrict__ store_pos, int ring,
                                          const std::int32_t* __restrict__ mtab) {
    const int half_r = n_rot / 2, per = half_r + (head_dim - n_rot);
    const long long rows_q = (long long) n_tok * n_head_q, rows_k = (long long) n_tok * n_head_kv;
    const long long nq = rows_q * per, nk = rows_k * per, nv = (long long) n_tok * n_head_kv * v_dim;
    const long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (g < nq) {                                                  // q: rope in place (rope_neox_par_kernel)
        const long long r = g / per;
        const int i = (int) (g - r * per);
        float* xr = q + r * head_dim;
        if (i >= half_r) return;                                   // tail dims: in place, unchanged
        const size_t toff = (size_t) mrope_pos(mtab, pos_q[r], i) * half_r;
        float oa, ob;
        rope_neox_pair(xr[i], xr[half_r + i], cos_tab[toff + i], sin_tab[toff + i], oa, ob);
        xr[i] = oa; xr[half_r + i] = ob;
        return;
    }
    if (g < nq + nk) {                                             // k: rope in place + f16 store
        const long long gk = g - nq;
        const long long r = gk / per;
        const int i = (int) (gk - r * per);
        const int t = (int) (r / n_head_kv), h = (int) (r % n_head_kv);
        float* xr = k + r * head_dim;
        const size_t cell = (size_t) ((*store_pos + t) % ring);
        half* dst = k_pool + cell * ((size_t) n_head_kv * head_dim) + (size_t) h * head_dim;
        if (i >= half_r) {
            const int d = n_rot + (i - half_r);
            dst[d] = __float2half(xr[d]);
            return;
        }
        const size_t toff = (size_t) mrope_pos(mtab, pos_k[r], i) * half_r;
        float oa, ob;
        rope_neox_pair(xr[i], xr[half_r + i], cos_tab[toff + i], sin_tab[toff + i], oa, ob);
        xr[i] = oa; xr[half_r + i] = ob;
        // the f32 values themselves go to f16 (as kv_store_f16 reads them back): keep the compiler from fusing the
        // rope fma into an f16-output fma (v_fma_mixlo_f16 rounds once, straight to f16)
        asm volatile("" : "+v"(oa), "+v"(ob));
        dst[i] = __float2half(oa); dst[half_r + i] = __float2half(ob);
        return;
    }
    if (g < nq + nk + nv) {                                        // v: f16 store
        const long long gv = g - nq - nk;
        const int vw = n_head_kv * v_dim;
        const int t = (int) (gv / vw), i = (int) (gv % vw);
        const size_t cell = (size_t) ((*store_pos + t) % ring);
        v_pool[cell * vw + i] = __float2half(v[(size_t) t * vw + i]);
    }
}

void mimo_rope_qk_store(float* q, float* k, const float* v, int n_tok, int n_head_q, int n_head_kv, int head_dim,
                        int v_dim, int n_rot, const float* cos_tab, const float* sin_tab, const std::int32_t* pos_q,
                        const std::int32_t* pos_k, void* k_pool, void* v_pool, const std::int32_t* store_pos, int ring,
                        void* stream) {
    if (n_tok <= 0) return;
    const long long per = n_rot / 2 + (head_dim - n_rot);
    const long long n = (long long) n_tok * (n_head_q + n_head_kv) * per + (long long) n_tok * n_head_kv * v_dim;
    const int threads = 128;
    mimo_rope_qk_store_kernel<<<(unsigned) ((n + threads - 1) / threads), threads, 0, static_cast<hipStream_t>(stream)>>>(
        q, k, v, n_tok, n_head_q, n_head_kv, head_dim, v_dim, n_rot, cos_tab, sin_tab, pos_q, pos_k,
        static_cast<half*>(k_pool), static_cast<half*>(v_pool), store_pos, ring, mrope_table());
}

} // namespace strata::kernels
