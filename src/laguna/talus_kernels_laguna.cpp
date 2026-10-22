// src/laguna/talus_kernels_laguna.cpp
// C-ABI shim (Contract v0) wrapping strata::laguna::* kernel launchers.

#include "talus_kernels.h"
#include "laguna/laguna_kernels.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>

namespace {

template <typename Fn>
tk_status invoke_noexcept(Fn&& fn) noexcept {
    try {
        fn();
        return TK_OK;
    } catch (const std::bad_alloc&) {
        return TK_ERR_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
        return TK_ERR_INVALID_ARG;
    } catch (const std::exception&) {
        return TK_ERR_EXCEPTION;
    } catch (...) {
        return TK_ERR_EXCEPTION;
    }
}

}  // namespace

extern "C" {

/* ==========================================================================
 * 1. FP32 Matrix-Vector & Matrix-Matrix Kernels
 * ========================================================================== */

tk_status tk_f32_matvec(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!w || !x || !y) {
        return TK_ERR_NULL_PTR;
    }
    if (n_in <= 0 || n_out <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::f32_matvec(w, x, y, n_in, n_out, stream);
    });
}

tk_status tk_f32_matvec_fast(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!w || !x || !y) {
        return TK_ERR_NULL_PTR;
    }
    if (n_in <= 0 || n_out <= 0 || (n_in % 128) != 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::f32_matvec_fast(w, x, y, n_in, n_out, stream);
    });
}

tk_status tk_f32_matvec_rows(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!w || !x || !y) {
        return TK_ERR_NULL_PTR;
    }
    if (n_in <= 0 || n_out <= 0 || (n_in % 4) != 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::f32_matvec_rows(w, x, y, n_in, n_out, stream);
    });
}

tk_status tk_f32_matvec_rows_n(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!w || !x || !y) {
        return TK_ERR_NULL_PTR;
    }
    if (n_in <= 0 || n_out <= 0 || (n_in % 4) != 0 || n_tok < 1 || n_tok > 8) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::f32_matvec_rows_n(w, x, y, n_in, n_out, n_tok, stream);
    });
}

tk_status tk_f32_matmul(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!w || !x || !y) {
        return TK_ERR_NULL_PTR;
    }
    if (n_in <= 0 || n_out <= 0 || n_tok <= 0 || (n_in % 16) != 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::f32_matmul(w, x, y, n_in, n_out, n_tok, stream);
    });
}

/* ==========================================================================
 * 2. Elementwise, Gating, Position & MoE Combine Kernels
 * ========================================================================== */

tk_status tk_softplus_head_gate(
    tk_ctx* ctx,
    float* o,
    const float* g,
    int32_t n_head,
    int32_t head_dim,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!o || !g) {
        return TK_ERR_NULL_PTR;
    }
    if (n_head <= 0 || head_dim <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::softplus_head_gate(o, g, n_head, head_dim, stream);
    });
}

tk_status tk_axpy(
    tk_ctx* ctx,
    float* y,
    const float* x,
    float alpha,
    int32_t n,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!y || !x) {
        return TK_ERR_NULL_PTR;
    }
    if (n < 0) {
        return TK_ERR_INVALID_ARG;
    }
    if (n == 0) {
        return TK_OK;
    }
    return invoke_noexcept([&] {
        strata::laguna::axpy(y, x, alpha, n, stream);
    });
}

extern "C" __attribute__((weak)) void tk_unregister_mmq_q8_buf(const void* xq) noexcept;

tk_status tk_gather_rows(
    tk_ctx* ctx,
    const void* src,
    void* dst,
    const int32_t* idx_dev,
    int32_t n_rows,
    size_t row_bytes,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!src || !dst || !idx_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_rows < 0 || row_bytes == 0 || (row_bytes % 4) != 0) {
        return TK_ERR_INVALID_ARG;
    }
    if (n_rows == 0) {
        return TK_OK;
    }
    if (tk_unregister_mmq_q8_buf != nullptr) {
        tk_unregister_mmq_q8_buf(dst);
    }
    return invoke_noexcept([&] {
        strata::laguna::gather_rows(src, dst, idx_dev, n_rows, row_bytes, stream);
    });
}

tk_status tk_moe_add_slots(
    tk_ctx* ctx,
    float* x,
    const float* shared,
    const float* y,
    const int32_t* map_dev,
    const float* weights_dev,
    int32_t n_tok,
    int32_t n_embd,
    int32_t k,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!x || !shared || !y || !map_dev || !weights_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_tok <= 0 || n_embd <= 0 || k < 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::moe_add_slots(x, shared, y, map_dev, weights_dev, n_tok, n_embd, k, stream);
    });
}

tk_status tk_silu_mul(
    tk_ctx* ctx,
    float* gate,
    const float* up,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!gate || !up) {
        return TK_ERR_NULL_PTR;
    }
    if (n < 0 || n > INT_MAX) {
        return TK_ERR_INVALID_ARG;
    }
    if (n == 0) {
        return TK_OK;
    }
    return invoke_noexcept([&] {
        strata::laguna::silu_mul(gate, up, static_cast<int>(n), stream);
    });
}

tk_status tk_fill_pos(
    tk_ctx* ctx,
    int32_t* pos_dev,
    int32_t n_tok,
    int32_t per_tok,
    int32_t p0,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!pos_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_tok <= 0 || per_tok <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::fill_pos(pos_dev, n_tok, per_tok, p0, stream);
    });
}

tk_status tk_fill_pos_dev(
    tk_ctx* ctx,
    int32_t* out_dev,
    int32_t n,
    const int32_t* pos_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!out_dev || !pos_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::fill_pos_dev(out_dev, n, pos_dev, stream);
    });
}

/* ==========================================================================
 * 3. MoE Routing Kernels
 * ========================================================================== */

tk_status tk_route_sigmoid_topk(
    tk_ctx* ctx,
    const float* logits,
    const float* bias,
    int32_t* ids_dev,
    float* weights_dev,
    int32_t n_expert,
    int32_t k,
    float scale,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!logits || !bias || !ids_dev || !weights_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_expert <= 0 || n_expert > 256 || k <= 0 || k > 16 || k > n_expert) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::route_sigmoid_topk(logits, bias, ids_dev, weights_dev, n_expert, k, scale, stream);
    });
}

tk_status tk_route_warp(
    tk_ctx* ctx,
    const float* logits,
    const float* bias,
    int32_t* ids_dev,
    float* weights_dev,
    int32_t k,
    float scale,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!logits || !bias || !ids_dev || !weights_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (k <= 0 || k > 32) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::route_warp(logits, bias, ids_dev, weights_dev, k, scale, stream);
    });
}

tk_status tk_route_topk(
    tk_ctx* ctx,
    const float* logits,
    const float* bias,
    int32_t n_tok,
    int32_t k,
    float scale,
    int32_t* ids_dev,
    float* weights_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!logits || !bias || !ids_dev || !weights_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_tok <= 0 || k <= 0 || k > 32) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::route_topk(logits, bias, n_tok, k, scale, ids_dev, weights_dev, stream);
    });
}

tk_status tk_route_group(
    tk_ctx* ctx,
    const float* logits,
    const float* bias,
    int32_t n_tok,
    int32_t k,
    float scale,
    int32_t* sel_e_dev,
    float* sel_w_dev,
    int32_t* cnt_dev,
    int32_t* bounds_dev,
    int32_t* rtok_dev,
    int32_t* smap_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!logits || !bias || !sel_e_dev || !sel_w_dev || !cnt_dev || !bounds_dev || !rtok_dev || !smap_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_tok <= 0 || k <= 0 || k > 32) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::route_group(
            logits, bias, n_tok, k, scale, sel_e_dev, sel_w_dev, cnt_dev, bounds_dev, rtok_dev, smap_dev, stream);
    });
}

tk_status tk_router_route(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* logits,
    const float* bias,
    int32_t* ids_dev,
    float* weights_dev,
    int32_t n_in,
    int32_t n_expert,
    int32_t k,
    float scale,
    uint32_t* counter_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!w || !x || !logits || !bias || !ids_dev || !weights_dev || !counter_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_in <= 0 || (n_in % 128) != 0 || n_expert != 256 || k <= 0 || k > 32) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::router_route(
            w, x, logits, bias, ids_dev, weights_dev, n_in, n_expert, k, scale, counter_dev, stream);
    });
}

/* ==========================================================================
 * 4. Embedding, Sampling, Norms & RoPE
 * ========================================================================== */

tk_status tk_embed_q8_0(
    tk_ctx* ctx,
    const void* table,
    const int32_t* tok_dev,
    float* x,
    int32_t n_embd,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!table || !tok_dev || !x) {
        return TK_ERR_NULL_PTR;
    }
    if (n_embd <= 0 || (n_embd % 32) != 0 || n_tok <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::embed_q8_0(table, tok_dev, x, n_embd, stream, n_tok);
    });
}

tk_status tk_argmax_advance(
    tk_ctx* ctx,
    const float* logits,
    int32_t n,
    int32_t* tok_dev,
    int32_t* pos_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!logits || !tok_dev || !pos_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::argmax_advance(logits, n, tok_dev, pos_dev, stream);
    });
}

tk_status tk_argmax_rows(
    tk_ctx* ctx,
    const float* logits,
    int32_t n_rows,
    int32_t n_vocab,
    int32_t* out_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!logits || !out_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n_rows <= 0 || n_vocab <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::argmax_rows(logits, n_rows, n_vocab, out_dev, stream);
    });
}

tk_status tk_head_norm_rope(
    tk_ctx* ctx,
    float* x,
    const float* w,
    int32_t rows,
    float eps,
    int32_t n_rot,
    const float* cos_tab,
    const float* sin_tab,
    const int32_t* pos_dev,
    int32_t pos_div,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!x || !w || !cos_tab || !sin_tab || !pos_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (rows <= 0 || n_rot < 0 || n_rot > 128 || (n_rot & 1) != 0 || pos_div <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::head_norm_rope(x, w, rows, eps, n_rot, cos_tab, sin_tab, pos_dev, pos_div, stream);
    });
}

tk_status tk_rms_norm_to(
    tk_ctx* ctx,
    float* dst,
    const float* src,
    const float* w,
    int32_t rows,
    int32_t cols,
    float eps,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!dst || !src || !w) {
        return TK_ERR_NULL_PTR;
    }
    if (rows <= 0 || cols <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::rms_norm_to(dst, src, w, rows, cols, eps, stream);
    });
}

tk_status tk_add_norm_quant(
    tk_ctx* ctx,
    float* x,
    const float* add,
    const float* yexp,
    const float* weights_dev,
    int32_t k,
    const float* norm_w,
    float* h_out,
    void* q8_out,
    int32_t n,
    float eps,
    int32_t n_rows,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!x || !norm_w || !q8_out) {
        return TK_ERR_NULL_PTR;
    }
    if (add && k > 0 && (!yexp || !weights_dev)) {
        return TK_ERR_NULL_PTR;
    }
    if (k < 0 || n <= 0 || (n % 32) != 0 || n > 4096 || n_rows <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::add_norm_quant(x, add, yexp, weights_dev, k, norm_w, h_out, q8_out, n, eps, stream, n_rows);
    });
}

tk_status tk_axpy_norm_quant(
    tk_ctx* ctx,
    float* x,
    const float* ax,
    float alpha,
    const float* norm_w,
    float* h_out,
    void* q8_out,
    int32_t n,
    float eps,
    int32_t n_rows,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!x || !ax || !norm_w || !q8_out) {
        return TK_ERR_NULL_PTR;
    }
    if (n <= 0 || (n % 32) != 0 || n > 4096 || n_rows <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::axpy_norm_quant(x, ax, alpha, norm_w, h_out, q8_out, n, eps, stream, n_rows);
    });
}

tk_status tk_qkv_norm_rope_store(
    tk_ctx* ctx,
    float* q,
    const float* kc,
    const float* vc,
    const float* q_norm,
    const float* k_norm,
    int32_t n_head,
    int32_t n_kv,
    float eps,
    int32_t n_rot,
    const float* cos_tab,
    const float* sin_tab,
    const int32_t* pos_dev,
    void* k_pool,
    void* v_pool,
    int32_t vt_stride,
    uint16_t* k_scale,
    uint16_t* v_scale,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!q || !kc || !vc || !q_norm || !k_norm || !cos_tab || !sin_tab || !pos_dev || !k_pool || !v_pool) {
        return TK_ERR_NULL_PTR;
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return TK_ERR_NULL_PTR;
    }
    if (n_head <= 0 || n_kv <= 0 || n_rot < 0 || n_rot > 128 || (n_rot & 1) != 0 || n_tok <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    if (k_scale && vt_stride <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::qkv_norm_rope_store(
            q,
            const_cast<float*>(kc),
            vc,
            q_norm,
            k_norm,
            n_head,
            n_kv,
            eps,
            n_rot,
            cos_tab,
            sin_tab,
            pos_dev,
            static_cast<uint16_t*>(k_pool),
            static_cast<uint16_t*>(v_pool),
            stream,
            vt_stride,
            k_scale,
            v_scale,
            n_tok);
    });
}

/* ==========================================================================
 * 5. Laguna Attention & Merge Kernels
 * ========================================================================== */

tk_status tk_laguna_prompt_attn(
    tk_ctx* ctx,
    const float* q,
    const uint16_t* k_pool,
    const uint16_t* v_pool,
    float* out,
    int32_t nq,
    int32_t q_pos0,
    int32_t n_head,
    int32_t n_kv,
    int32_t window,
    const int32_t* q_pos0_dev,
    int32_t n_split,
    float* part_acc,
    float* part_ml,
    bool merge,
    int32_t vt_stride,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!q || !k_pool || !v_pool) {
        return TK_ERR_NULL_PTR;
    }
    if ((n_split <= 1 || merge) && !out) {
        return TK_ERR_NULL_PTR;
    }
    if (n_split > 1 && (!part_acc || !part_ml)) {
        return TK_ERR_NULL_PTR;
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return TK_ERR_NULL_PTR;
    }
    if (nq <= 0 || n_head <= 0 || n_kv <= 0 || (n_head % n_kv) != 0 || (n_head / n_kv) > 9) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::prompt_attn(
            q,
            k_pool,
            v_pool,
            out,
            nq,
            q_pos0,
            n_head,
            n_kv,
            window,
            stream,
            q_pos0_dev,
            n_split,
            part_acc,
            part_ml,
            merge,
            vt_stride,
            k_scale,
            v_scale);
    });
}

tk_status tk_laguna_prompt_attn_wmma(
    tk_ctx* ctx,
    const float* q,
    const uint16_t* k_pool,
    const uint16_t* v_pool,
    float* out,
    int32_t nq,
    int32_t q_pos0,
    int32_t n_head,
    int32_t n_kv,
    int32_t window,
    int32_t vt_stride,
    const int32_t* q_pos0_dev,
    int32_t n_split,
    float* part_acc,
    float* part_ml,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!q || !k_pool || !v_pool) {
        return TK_ERR_NULL_PTR;
    }
    if (n_split <= 1 && !out) {
        return TK_ERR_NULL_PTR;
    }
    if (n_split > 1 && (!part_acc || !part_ml)) {
        return TK_ERR_NULL_PTR;
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return TK_ERR_NULL_PTR;
    }
    if (nq <= 0 || n_head <= 0 || n_kv <= 0 || (n_head % n_kv) != 0 || vt_stride <= 0 || (vt_stride % 16) != 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::prompt_attn_wmma(
            q,
            k_pool,
            v_pool,
            out,
            nq,
            q_pos0,
            n_head,
            n_kv,
            window,
            stream,
            vt_stride,
            q_pos0_dev,
            n_split,
            part_acc,
            part_ml,
            k_scale,
            v_scale);
    });
}

tk_status tk_laguna_decode_attn(
    tk_ctx* ctx,
    const float* q,
    const uint16_t* k_pool,
    const uint16_t* v_pool,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    int32_t n_head,
    int32_t n_kv,
    int32_t window,
    int32_t stride,
    const int32_t* pos_dev,
    int32_t n_split,
    float* part_acc,
    float* part_ml,
    int32_t n_tok,
    const int32_t* sel_dev,
    int32_t sel_k,
    bool sel_per_kv,
    int32_t sel_bs,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!q || !k_pool || !v_pool || !pos_dev || !part_acc || !part_ml) {
        return TK_ERR_NULL_PTR;
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return TK_ERR_NULL_PTR;
    }
    if (n_head <= 0 || n_kv <= 0 || (n_head % n_kv) != 0 || (n_head / n_kv) > 9 || n_split <= 0 || n_tok <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    if (sel_dev && (sel_k <= 0 || sel_bs <= 0)) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::decode_attn(
            q,
            k_pool,
            v_pool,
            k_scale,
            v_scale,
            n_head,
            n_kv,
            window,
            stride,
            pos_dev,
            n_split,
            part_acc,
            part_ml,
            stream,
            n_tok,
            sel_dev,
            sel_k,
            sel_per_kv,
            sel_bs);
    });
}

tk_status tk_attn_merge_gate_quant(
    tk_ctx* ctx,
    const float* part_acc,
    const float* part_ml,
    const float* g,
    void* q8_out,
    int32_t n_head,
    int32_t n_split,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!part_acc || !part_ml || !g || !q8_out) {
        return TK_ERR_NULL_PTR;
    }
    if (n_head <= 0 || n_split <= 0 || n_tok <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::attn_merge_gate_quant(part_acc, part_ml, g, q8_out, n_head, n_split, stream, n_tok);
    });
}

/* ==========================================================================
 * 6. KV Cache Store Kernels
 * ========================================================================== */

tk_status tk_kv_store_f16(
    tk_ctx* ctx,
    uint16_t* k_pool,
    uint16_t* v_pool,
    const float* kc,
    const float* vc,
    int32_t n,
    const int32_t* pos_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!k_pool || !v_pool || !kc || !vc || !pos_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (n <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::kv_store(k_pool, v_pool, kc, vc, n, pos_dev, stream);
    });
}

tk_status tk_v_store_t(
    tk_ctx* ctx,
    const float* vc,
    uint16_t* v_pool,
    int32_t nb,
    int32_t n_kv,
    int32_t p0,
    int32_t stride,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!vc || !v_pool) {
        return TK_ERR_NULL_PTR;
    }
    if (nb <= 0 || n_kv <= 0 || p0 < 0 || stride <= 0 || (stride % 16) != 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::v_store_t(vc, v_pool, nb, n_kv, p0, stride, stream);
    });
}

tk_status tk_kv_store_q8(
    tk_ctx* ctx,
    const float* kc,
    const float* vc,
    void* k_pool,
    void* v_pool,
    uint16_t* k_scale,
    uint16_t* v_scale,
    int32_t nb,
    int32_t n_kv,
    int32_t p0,
    int32_t stride,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!kc || !vc || !k_pool || !v_pool || !k_scale || !v_scale) {
        return TK_ERR_NULL_PTR;
    }
    if (nb <= 0 || n_kv <= 0 || p0 < 0 || stride <= 0 || (stride % 16) != 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::kv_store_q8(kc, vc, k_pool, v_pool, k_scale, v_scale, nb, n_kv, p0, stride, stream);
    });
}

/* ==========================================================================
 * 7. Quest Sparse Attention Kernels
 * ========================================================================== */

tk_status tk_kbound_update(
    tk_ctx* ctx,
    const void* k_pool,
    const uint16_t* k_scale,
    void* bnd,
    int32_t nb_max,
    int32_t n_kv,
    int32_t b_first,
    int32_t b_last,
    int32_t pos_last,
    const int32_t* pos_dev,
    int32_t n_tok,
    int32_t bs,
    bool f16,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!k_pool || !k_scale || !bnd) {
        return TK_ERR_NULL_PTR;
    }
    if (nb_max <= 0 || n_kv <= 0 || (bs != 16 && bs != 32 && bs != 64)) {
        return TK_ERR_INVALID_ARG;
    }
    if (pos_dev && n_tok <= 0) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::kbound_update(
            k_pool, k_scale, bnd, nb_max, n_kv, b_first, b_last, pos_last, pos_dev, n_tok, stream, bs, f16);
    });
}

tk_status tk_sparse_scores(
    tk_ctx* ctx,
    const float* q,
    const void* bnd,
    float* ub,
    int32_t nb_max,
    int32_t n_head,
    int32_t n_kv,
    const int32_t* pos_dev,
    int32_t n_tok,
    int32_t bs,
    bool f16,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!q || !bnd || !ub || !pos_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (nb_max <= 0 || n_kv != 8 || n_head <= 0 || (n_head % n_kv) != 0 || (n_head / n_kv) > 8 || n_tok <= 0 ||
        (bs != 16 && bs != 32 && bs != 64)) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::sparse_scores(q, bnd, ub, nb_max, n_head, n_kv, pos_dev, n_tok, stream, bs, f16);
    });
}

tk_status tk_sparse_select(
    tk_ctx* ctx,
    const float* ub,
    int32_t nb_max,
    int32_t n_head,
    const int32_t* pos_dev,
    int32_t n_tok,
    int32_t k,
    int32_t sink,
    int32_t recent,
    int32_t* sel_dev,
    int32_t groups,
    int32_t bs,
    uint32_t* scr_dev,
    tk_stream stream) TK_NOEXCEPT {
    (void) ctx;
    if (!ub || !pos_dev || !sel_dev) {
        return TK_ERR_NULL_PTR;
    }
    if (nb_max <= 0 || n_head <= 0 || n_head > 64 || n_tok <= 0 || k <= 0 || groups <= 0 || (n_head % groups) != 0 ||
        (bs != 16 && bs != 32 && bs != 64) || (!scr_dev && nb_max > 12288)) {
        return TK_ERR_INVALID_ARG;
    }
    return invoke_noexcept([&] {
        strata::laguna::sparse_select(
            ub, nb_max, n_head, pos_dev, n_tok, k, sink, recent, sel_dev, stream, groups, bs, scr_dev);
    });
}

}  // extern "C"
