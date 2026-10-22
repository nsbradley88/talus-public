// src/prefill/fattn_prefill_f32.cu - fattn_prefill's kernel with FP32 output accumulators at head size 256
// (fattn-mma-f16-f32acc.cuh: llama.cpp's fattn-mma-f16.cuh with that one tile choice changed; its RDNA3 default
// accumulates P V in FP16).  Selected by STRATA_FA_F32ACC=1 (see fattn_prefill.cu).
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16-f32acc.cuh"

template void strata_fa_mma_case_f32acc<256, 256, 16, 2>(ggml_backend_cuda_context& ctx, ggml_tensor* dst);
template void strata_fa_mma_case_f32acc<256, 256, 32, 2>(ggml_backend_cuda_context& ctx, ggml_tensor* dst);

void strata_fa_f32acc_launch(ggml_backend_cuda_context& ctx, ggml_tensor* dst, int T) {
    if (T <= 16) strata_fa_mma_case_f32acc<256, 256, 16, 2>(ctx, dst);
    else strata_fa_mma_case_f32acc<256, 256, 32, 2>(ctx, dst);
}
