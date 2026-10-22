// include/strata/prefill/tp_fuse.hpp - Talus: small exact fusions of the prompt path.
//
// swiglu_q8_mmq: silu(g) * u straight into MMQ's q8_1 activation blocks for weights of `ggml_type` (the down projection's
// input), bitwise what k_swiglu then strata::prefill::mmq::quantize produce, without the [T][n_ff] FP32 round trip.
//
// Kernel-library entry (libtalus_kernels via strata_mmq); pointer, shape and stream arguments.
#pragma once

#include <cstdint>

namespace talus::prefill {

/// g, u [rows][cols] FP32 (cols % 4 == 0) -> xq (strata::prefill::mmq::q8_bytes(rows, cols) bytes).
void swiglu_q8_mmq(const float* g, const float* u, void* xq, int ggml_type, int64_t cols, int64_t rows, void* stream);

}  // namespace talus::prefill

extern "C" {
/// C ABI (Rust port) of swiglu_q8_mmq: 0 on success, -1 when unsupported.
int talus_pf_swiglu_q8_mmq(const float* g, const float* u, void* xq, int ggml_type, int64_t cols, int64_t rows, void* stream);
}
