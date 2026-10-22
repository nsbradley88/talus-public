// include/strata/kernels/fnp_qsa_tile_attn.hpp - the prompt path's QSA attention over tiles of
// neighbouring queries (opt-in STRATA_FNP_TILE_ATTN=1; gfx11 matrix cores, int8 K/V).
//
// qsa_prompt_attn_batch runs one block per (query, KV head) and reads each query's ~2,051 selected cells on its own;
// neighbouring queries select mostly the same cells (STRATA_SEL_OVERLAP on a 32K prompt), so that kernel is bound by
// re-reading the same K/V rows.  Here a block takes
// TQ = 4 consecutive queries (48 query-head rows, three 16-row WMMA tiles) and walks the UNION of their selections
// in ascending cell order, 32 cells a chunk, each row masked to its own query's cells:
//   1. fnp_qsa_tile_union: per tile, the 4 sorted cell lists merged (merge path, in LDS) into one ascending list of
//      (cell << 4 | query mask) entries in global scratch, and its length;
//   2. the attention: K and V of a chunk's cells staged once in LDS for all 48 rows; q.k with FP16 q (one image, as
//      STRATA_PA_FAST), the int8 codes exact in FP16, per-64 scales in FP32; online softmax per row; p.v with FP16 p.
// Same arithmetic class as the PA_FAST kernel but another chunking of each query's cells (the union's), so the
// softmax rescaling order differs: rounding-level, not bitwise - opt-in and quality-gated.
//
// C ABI for the shared libtalus_kernels: fnp_qsa_tile_attn_i8 (below, extern "C").
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// Queries per tile.
inline constexpr int kFnpTileQ = 4;

/// q [n_q, 24, 256] FP32, int8 pools (k_q / v_q / k_scale / v_scale / page_table), ids [n_q, cap] ascending per query,
/// steps [n_q, kStepCount] (kStepWidth = the query's cell count), attn [n_q, 24, 256].  `scratch` device memory of
/// fnp_qsa_tile_scratch_bytes(n_q, cap) bytes.  false: not applicable (another geometry, not int8 K/V, not a gfx11
/// device); nothing was launched.
bool fnp_qsa_tile_attn(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                       const QsaShapes& s, float* attn, int64_t n_q, void* scratch, void* stream);
uint64_t fnp_qsa_tile_scratch_bytes(int64_t n_q, int64_t cap);

}  // namespace strata::kernels

extern "C" {
/// The same as strata::kernels::fnp_qsa_tile_attn for int8 K/V, flat arguments (n_head 24, n_head_kv 2, head_dim 256).
/// Returns 0 on success, nonzero when not applicable.
int fnp_qsa_tile_attn_i8(const float* q, const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale,
                         const uint16_t* v_scale, const int32_t* page_table, int64_t page_size, const int32_t* ids,
                         const int32_t* steps, int64_t cap, float* attn, int64_t n_q, void* scratch, void* stream);
uint64_t fnp_qsa_tile_scratch_bytes_c(int64_t n_q, int64_t cap);
}
