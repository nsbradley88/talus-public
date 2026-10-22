// include/strata/prefill/tp_fa.hpp - Talus: a prompt chunk's causal flash attention on the gfx11
// matrix cores, head dim 256, grouped-query heads packed per KV head.  Replaces llama.cpp's fattn-mma-f16 kernel for
// prompt chunks (that kernel spills registers on gfx1151 at D 256 on long prompts; see docs/RESULTS.md).
//
// Formulation: per KV head, the T * G query rows (G = H / HKV query heads per KV head) in tiles of 64; per tile of 32
// keys S^T = K Q^T (16 x 16 WMMA tiles, keys x queries: a lane holds one query's column), the online softmax per lane
// (one cross-half exchange), O^T += V^T P^T with V^T read from the engine's cell-blocked V copy (`vt`, the decode
// attention's [kv head][cell / 16][dim][cell % 16]), so every operand is one contiguous 32-byte row.  FP16 Q / K / V / P,
// FP32 scores, softmax statistics and accumulators.  Cells past the chunk's last position are read as zeros (never the
// cache's history), so a chunk's output depends on its inputs only.
//
// Kernel-library entry (libtalus_kernels via strata_prefill); pointer, shape and stream arguments only.
#pragma once

#include <cstdint>

namespace talus::prefill {

/// The kernel exists for this device and shape (gfx11 WMMA, head dim 256, H a multiple of HKV).
bool fa_prefill_ok(int head_dim, int H, int HKV);

/// out [T][H][D] (FP32) = softmax(scale q k^T, causal) v, optionally times sigmoid(gate) (gate [T][H][D] or null), for
/// the query rows at positions p0 .. p0 + T - 1.
///   q   [T][H][D] FP32;  k  FP16 [HKV][kv_stride][D] (cells 0 .. p0 + T - 1 written);
///   vt  FP16 the V cache's cell-blocked copy, [HKV][vt_cells / 16][D][16] (cells 0 .. p0 + T - 1 written).
/// `variant` 0 = the arch's tuned choice (tp_tune), 1 = K / V through LDS, 2 = K / V fragments from global memory,
/// 3 = Q in registers with the head dim split over a wave pair (the gfx1151 default).
void fa_prefill(const float* q, const void* k, const void* vt, const float* gate, float* out, int T, int H, int HKV,
                int D, int64_t kv_stride, int64_t vt_cells, int64_t p0, float scale, int variant, void* stream);

}  // namespace talus::prefill

extern "C" {
/// C ABI of talus::prefill::fa_prefill (for the Rust port): returns 0 on success, -1 for an unsupported shape.
int talus_pf_fa_prefill(const float* q, const void* k, const void* vt, const float* gate, float* out, int T, int H,
                        int HKV, int D, int64_t kv_stride, int64_t vt_cells, int64_t p0, float scale, void* stream);
}
