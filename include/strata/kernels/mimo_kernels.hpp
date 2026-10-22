#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// Returns true if `ggml_type` is one of the MiMo MoE expert quantization types
/// supported by `mimo_mmvq_ids` and `mimo_prefill_moe_mmq`:
/// IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22), IQ1_M (29), Q2_K (10).
inline constexpr bool mimo_is_iq_qtype(int ggml_type) noexcept {
    switch (ggml_type) {
        case 10: // GGML_TYPE_Q2_K
        case 16: // GGML_TYPE_IQ2_XXS
        case 17: // GGML_TYPE_IQ2_XS
        case 18: // GGML_TYPE_IQ3_XXS
        case 21: // GGML_TYPE_IQ3_S
        case 22: // GGML_TYPE_IQ2_S
        case 29: // GGML_TYPE_IQ1_M
            return true;
        default:
            return false;
    }
}

/// Quantize `n_rows x n_cols` FP32 activations into contiguous `block_q8_1` blocks (`n_cols % 32 == 0`).
void mimo_quantize_q8_1(
    const float* x_f32,
    void*        x_q8_1,
    std::int64_t n_rows,
    std::int64_t n_cols,
    void*        stream);

/// 4-rows-per-workgroup wave32 MoE expert mat-vec (`MUL_MAT_ID`) for MiMo IQ2/IQ3/IQ1/Q2_K experts.
/// Uses `__builtin_amdgcn_sudot4` (`dp4a`) against register-cached `block_q8_1` inputs and wave-ballot
/// duplicate-expert coalescing across multi-token verify batches (`n_tokens = 1..8`).
/// When `x_per_id` is true, slot `k` reads input column `k / x_group` (`x_group == 1` by default).
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
    const void*         weights2 = nullptr,
    float*              y2 = nullptr,
    int                 x_group = 1);

/// MiMo sliding-window attention (e.g. `window = 128`, `head_dim_k = 192`, `head_dim_v = 128`,
/// `n_head = 64`, `n_kv = 8`) with learned per-head attention sinks (`f32[n_head]`) incorporated
/// into the online softmax denominator (`exp(sink[h] - max_score)`) and optional `value_scale`
/// (e.g. `0.70710678f` = `1/sqrt(2)`). Supports both F16 (`k_scale == nullptr`) and Q8 KV caches.
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
    float                value_scale = 1.0f,
    void*                q8_out = nullptr,    // also write Q8_1 of the outputs (tk_quantize_q8_1's bytes)
    void*                q8_mmq = nullptr,    // also the MMQ q8_1 rows of the outputs (layout 0 D4 / 1 DS4) ...
    int                  q8_mmq_layout = 0,   // ... where the prefill v2 kernel runs; *q8_mmq_done says whether it did
    bool*                q8_mmq_done = nullptr);

/// MiMo full causal GQA attention with asymmetric head dimensions (`head_dim_k = 192`, `head_dim_v = 128`,
/// `n_head = 64`, `n_kv = 4`, GQA 16:1). Supports multi-token verify (`n_tok = 1..8`), F16 and Q8 KV caches,
/// and split-K decoding over long contexts (`kv_len = 512 .. 32768+`).
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
    void*                q8_out = nullptr);   // also write Q8_1 of the outputs (tk_quantize_q8_1's bytes)

/// q + k partial NeoX rope (bitwise rope_neox_apply) and the f16 K / V cache store (bitwise
/// q36 kv_store_f16, cell = (store_pos[0] + t) % ring) in one launch.
void mimo_rope_qk_store(float* q, float* k, const float* v, int n_tok, int n_head_q, int n_head_kv, int head_dim,
                        int v_dim, int n_rot, const float* cos_tab, const float* sin_tab, const std::int32_t* pos_q,
                        const std::int32_t* pos_k, void* k_pool, void* v_pool, const std::int32_t* store_pos, int ring,
                        void* stream);

/// Grouped MoE prefill MMQ over already-quantized `block_q8_1` activations (`x_q8_1`, as produced by
/// `tk_quantize_q8_1` or `tk_swiglu_quantize_q8_1`). Used by `tk_prefill_moe_mmq`.
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
    const std::int32_t* expert_map_dev = nullptr);

/// Grouped MoE prefill MMQ for MiMo IQ2_XXS, IQ3_XXS, IQ2_XS, IQ3_S, IQ2_S, IQ1_M, and Q2_K experts.
/// Operates on token activations grouped by expert (`bounds_dev[0 .. n_experts_active]`), quantizing
/// `x_f32` into `scratch_xq` (`block_q8_1`) and executing a tiled shared-memory + `dp4a` grouped GEMM.
/// When `expert_map_dev != nullptr`, active slot `q` selects expert `expert_map_dev[q]` in `w`;
/// otherwise slot `q` indexes `w + q * expert_stride_bytes` directly.
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
    const std::int32_t* expert_map_dev = nullptr);

} // namespace strata::kernels
