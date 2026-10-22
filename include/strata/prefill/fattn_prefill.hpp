// include/strata/prefill/fattn_prefill.hpp - Talus prefill: a prompt chunk's causal attention on llama.cpp's
// flash-attention kernel (ggml-cuda fattn-mma-f16, MIT; on gfx11 its WMMA path), instead of hipBLAS score / value
// GEMMs with an FP32 score buffer.  Built into strata_mmq (the ggml-cuda sources are there).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::fattn {

/// This build has the kernel for head size `head_dim` (256 only).
bool supported(int head_dim);
/// The key count the kernel is given: n rounded up to its 256-key stride (the cells past n are masked).
int64_t kv_padded(int64_t n);
/// Bytes of the causal mask of T query rows over kv_padded(n_kv) keys (FP16).
size_t mask_bytes(int64_t T, int64_t n_kv);

/// out [T][H][D] (FP32) = softmax(scale q k^T + causal mask) v for query rows at positions p0 .. p0 + T - 1:
/// q [T][H][D] FP32; k, v FP16 [HKV][kv_stride][D] (head h reads KV head h / (H / HKV)), positions 0 .. p0 + T - 1
/// written; the buffers must be readable up to kv_padded(p0 + T) rows past each head's start (finite values).
/// `mask` is scratch of mask_bytes(T, p0 + T).  Stream-ordered on `stream`.
void attend(const float* q, const void* k, const void* v, float* out, int T, int H, int HKV, int D,
            int64_t kv_stride, int64_t p0, float scale, void* mask, void* stream);

}  // namespace strata::prefill::fattn
