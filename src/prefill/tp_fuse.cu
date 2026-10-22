// src/prefill/tp_fuse.cu - see include/strata/prefill/tp_fuse.hpp.  Built into strata_mmq with the pinned ggml-cuda
// sources: the quantizer below is llama.cpp's quantize_mmq_q8_1 (quantize.cu, MIT) with its input load replaced by
// silu(g) * u (the engine's k_swiglu expression), so with the same 4 values per thread and the same shuffles its output
// is bitwise MMQ::quantize of k_swiglu's.
//
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include "strata/prefill/tp_fuse.hpp"

namespace talus::prefill {
namespace {

__device__ __forceinline__ float silu_f(float v) { return v / (1.0f + expf(-v)); }   // = q36_engine's silu

template <mmq_q8_1_ds_layout ds_layout>
__global__ void k_swiglu_q8(const float* __restrict__ g, const float* __restrict__ u, void* __restrict__ vy,
                            const int64_t ne00, const int64_t s01, const int64_t ne0, const int ne1) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    const int64_t base_idx = (int64_t) blockIdx.x * s01;
    block_q8_1_mmq* y = (block_q8_1_mmq*) vy;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs = i0 % QK8_1_MMQ;
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < ne00) {
        const float4 gv = ((const float4*) g)[(base_idx + i0) / 4], uv = ((const float4*) u)[(base_idx + i0) / 4];
        xi.x = silu_f(gv.x) * uv.x;
        xi.y = silu_f(gv.y) * uv.y;
        xi.z = silu_f(gv.z) * uv.z;
        xi.w = silu_f(gv.w) * uv.w;
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    const int64_t ib = k_block * ne1 + blockIdx.x;
    char4* yqs4 = (char4*) y[ib].qs;
    yqs4[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) y[ib].d2s6[iqs / 64] = d;
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) y[ib].ds4[iqs / 32] = make_half2(d, sum);
        else y[ib].d4[iqs / 32] = d;
    }
}

}  // namespace

void swiglu_q8_mmq(const float* g, const float* u, void* xq, int wtype, int64_t cols, int64_t rows, void* stream) {
    if (rows <= 0) return;
    const int64_t ne0 = (cols + 511) / 512 * 512;   // = moe_mmq.cu's pad512
    const int64_t block_num_y = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 nb((unsigned) rows, (unsigned) block_num_y, 1), bs(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const cudaStream_t s = (cudaStream_t) stream;
    switch (mmq_get_q8_1_ds_layout((ggml_type) wtype)) {
        case MMQ_Q8_1_DS_LAYOUT_D4: k_swiglu_q8<MMQ_Q8_1_DS_LAYOUT_D4><<<nb, bs, 0, s>>>(g, u, xq, cols, cols, ne0, (int) rows); break;
        case MMQ_Q8_1_DS_LAYOUT_DS4: k_swiglu_q8<MMQ_Q8_1_DS_LAYOUT_DS4><<<nb, bs, 0, s>>>(g, u, xq, cols, cols, ne0, (int) rows); break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6: k_swiglu_q8<MMQ_Q8_1_DS_LAYOUT_D2S6><<<nb, bs, 0, s>>>(g, u, xq, cols, cols, ne0, (int) rows); break;
        default: break;
    }
}

}  // namespace talus::prefill

extern "C" int talus_pf_swiglu_q8_mmq(const float* g, const float* u, void* xq, int wtype, int64_t cols, int64_t rows,
                                      void* stream) {
    if (cols % 4 != 0) return -1;
    talus::prefill::swiglu_q8_mmq(g, u, xq, wtype, cols, rows, stream);
    return 0;
}
