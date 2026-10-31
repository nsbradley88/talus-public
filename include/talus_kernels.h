/*
 * include/talus_kernels.h - Talus HIP C++ kernel library C ABI (Contract v0).
 *
 * Copyright (c) 2026 Niko1221 and the Strata contributors
 * Copyright (c) 2023-2026 The ggml authors
 * Copyright (c) 2026 nsbradley88
 * SPDX-License-Identifier: MIT
 *
 * ============================================================================
 * ABI CONTRACT RULES
 * ============================================================================
 *
 * 1. Exception safety (`noexcept` boundary):
 *    Every `tk_*` entry point is declared `TK_NOEXCEPT` (`noexcept` in C++)
 *    and MUST catch all C++ exceptions internally, translating them into a
 *    non-zero `tk_status` code (`TK_ERR_EXCEPTION`, `TK_ERR_OUT_OF_MEMORY`,
 *    `TK_ERR_INVALID_ARG`, etc.) and storing a diagnostic message retrievable
 *    via `tk_last_error(ctx)`. Unwinding across the C FFI boundary into Rust
 *    is undefined behavior and is strictly prohibited.
 *
 * 2. Configuration (`tk_config` and environment overrides):
 *    Runtime switches (`STRATA_*` and `TALUS_*` environment knobs and
 *    per-architecture speed switches from `arch_defaults.cpp`) are configured
 *    via `tk_config` handles passed to `tk_ctx_create` along with cached
 *    environment overrides on initialization. `tk_config_create_for_arch`
 *    populates architecture defaults (only `TK_ARCH_GFX1151` is populated in
 *    v0); `tk_config_from_env` overlays process environment variables so the
 *    C++ oracle and Rust host resolve identical knob sets.
 *
 * 3. Explicit packed-weight handles (`tk_packed_weights`):
 *    Load-time repacked decode weights (lossless Q8_0 `qs`+`d` plane splits
 *    and Q6_K `ql`/`qh`/`scales`/`d` plane splits) are owned by explicit
 *    `tk_packed_weights*` handles returned by `tk_pack_q8` / `tk_pack_q6` and
 *    consumed by `tk_mmvq_packed`, replacing hidden global weight-pointer
 *    registries (`q8_packed_registry`, `q6_packed_registry`).
 *
 * 4. HIP stream ownership and graph-capture contract:
 *    - `tk_stream` is a `hipStream_t` created and owned by the host.
 *    - Both the Rust host (`hip-sys`) and `libtalus_kernels.so` MUST link
 *      `libamdhip64.so` dynamically so `hipStreamBeginCapture` on a host
 *      stream records kernel launches made across this FFI boundary.
 *    - Entry points marked [CAPTURABLE] enqueue only device work on `stream`
 *      and perform no host allocations, blocking synchronizations, or host
 *      waits.
 *    - Device-varying scalars (token position, token ID, dynamic KV lengths,
 *      QSA step descriptors, router atomic counters) are passed as device
 *      pointers (`const int32_t* pos_dev`, `const int32_t* tok_dev`, etc.) at
 *      fixed device addresses so captured HIP graphs read updated values on
 *      replay without `hipGraphExecUpdate`.
 *    - Host scalar dimensions (`n_in`, `n_out`, `ncols`, `n_head`, `n_kv`,
 *      `n_split`, `window`, `stride`, etc.) are baked into launch grids and
 *      kernel arguments at capture time; the host keys captured graphs by
 *      `(LayerKind, T)` or `(D, kind)`.
 */

#ifndef TALUS_KERNELS_H
#define TALUS_KERNELS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define TK_NOEXCEPT noexcept
extern "C" {
#else
#define TK_NOEXCEPT
#endif

/** Contract v0 ABI version macro. */
#define TALUS_KERNELS_ABI_VERSION 0

/* ==========================================================================
 * Opaque Handles & Status Types
 * ========================================================================== */

/** Per-device kernel context (hipBLAS/hipBLASLt handles, algo cache, tuning tables, error buffer). */
typedef struct tk_ctx tk_ctx;

/** Opaque HIP stream handle (`hipStream_t`), created and owned by the host. */
typedef void* tk_stream;

/** Status return code (`0` == `TK_OK`; non-zero == `tk_status_code`). */
typedef int32_t tk_status;

/** Opaque kernel configuration knob set. */
typedef struct tk_config tk_config;

/** Opaque handle to device-resident repacked decode weights (Q8_0 or Q6_K plane layout). */
typedef struct tk_packed_weights tk_packed_weights;

/** Status codes returned by `tk_*` entry points. */
typedef enum tk_status_code {
    TK_OK                = 0,
    TK_ERR_INVALID_ARG   = 1,
    TK_ERR_NULL_PTR      = 2,
    TK_ERR_UNSUPPORTED   = 3,
    TK_ERR_OUT_OF_MEMORY = 4,
    TK_ERR_HIP           = 5,
    TK_ERR_HIPBLAS       = 6,
    TK_ERR_IO            = 7,
    TK_ERR_EXCEPTION     = 8,
    TK_ERR_INTERNAL      = 9
} tk_status_code;

/* ==========================================================================
 * Target Architecture & Quantization Enums
 * ========================================================================== */

/** GPU architecture family for tuning table and kernel selection. */
typedef enum tk_arch {
    TK_ARCH_GFX1151 = 0, /**< AMD Strix Halo (RDNA3.5, wave32, WMMA gfx11, unified memory). */
    TK_ARCH_GFX1201 = 1, /**< AMD Radeon AI PRO R9700 / RDNA4 (wave32, WMMA gfx12, FP8). */
    TK_ARCH_OTHER   = 2  /**< Any other GCN/CDNA/RDNA target (portable fallback path). */
} tk_arch;

/**
 * Tensor quantization and element types.
 * Enumerator values match GGML / Strata type IDs (`include/strata/artifact/gguf_reader.hpp`).
 */
typedef enum tk_qtype {
    TK_QTYPE_F32     = 0,
    TK_QTYPE_F16     = 1,
    TK_QTYPE_Q4_0    = 2,
    TK_QTYPE_Q4_1    = 3,
    TK_QTYPE_Q5_0    = 6,
    TK_QTYPE_Q5_1    = 7,
    TK_QTYPE_Q8_0    = 8,
    TK_QTYPE_Q8_1    = 9,
    TK_QTYPE_Q2_K    = 10,
    TK_QTYPE_Q3_K    = 11,
    TK_QTYPE_Q4_K    = 12,
    TK_QTYPE_Q5_K    = 13,
    TK_QTYPE_Q6_K    = 14,
    TK_QTYPE_Q8_K    = 15,
    TK_QTYPE_IQ2_XXS = 16,
    TK_QTYPE_IQ2_XS  = 17,
    TK_QTYPE_IQ3_XXS = 18,
    TK_QTYPE_IQ1_S   = 19,
    TK_QTYPE_IQ4_NL  = 20,
    TK_QTYPE_IQ3_S   = 21,
    TK_QTYPE_IQ2_S   = 22,
    TK_QTYPE_IQ4_XS  = 23,
    TK_QTYPE_I8      = 24,
    TK_QTYPE_IQ1_M   = 29,
    TK_QTYPE_BF16    = 30,
    TK_QTYPE_TQ1_0   = 34,
    TK_QTYPE_TQ2_0   = 35,
    TK_QTYPE_MXFP4   = 39,
    TK_QTYPE_NVFP4   = 40,
    TK_QTYPE_Q1_0    = 41,
    TK_QTYPE_Q2_0    = 42
} tk_qtype;

/* ==========================================================================
 * Descriptor Structs for Batched / Grouped Operations
 * ========================================================================== */

/**
 * Single work item for `tk_mmvq_batch` (mirrors `strata::kernels::MmvqItem`).
 * All device pointers inside each item must remain fixed across graph replay.
 */
typedef struct tk_mmvq_item {
    tk_qtype qtype;          /**< Weight quantization type. */
    const void* w;           /**< Device pointer to raw GGUF weight blocks (expert 0). */
    size_t expert_bytes;     /**< Byte stride between experts (0 for dense). */
    const int32_t* ids_dev;  /**< Device pointer to expert IDs (`n_ids`), or NULL for dense. */
    int32_t n_ids;           /**< Number of expert IDs (1 for dense). */
    const void* x_q8_1;      /**< Device pointer to Q8_1 input activations. */
    int32_t x_per_id;        /**< Non-zero if each expert reads its own input column `k / x_group`. */
    int32_t x_group;         /**< Number of expert IDs sharing one input column. */
    float* y;                /**< Device pointer to FP32 output `[n_ids, n_out]`. */
    int32_t expert_major;    /**< 0: row-block-major; 1: expert-major. */
    int32_t n_in;            /**< Input reduction dimension (baked at capture). */
    int32_t n_out;           /**< Output row dimension (baked at capture). */
} tk_mmvq_item;

/**
 * Grouped MoE MMQ product descriptor for `tk_prefill_moe_mmq`
 * (mirrors `strata::prefill::mmq::Product`).
 */
typedef struct tk_moe_mmq_product {
    const void* w;            /**< Device pointer to expert 0 weights (or gathered group buffer). */
    tk_qtype qtype;           /**< Weight quantization type. */
    int64_t w_rows;           /**< Output rows per expert (`2 * n_ff` for gate+up, `n_embd` for down). */
    int64_t w_cols;           /**< Reduction columns per expert (`n_embd` for gate+up, `n_ff` for down). */
    size_t expert_bytes;      /**< Byte stride between consecutive experts in `w`. */
    int32_t n_experts;        /**< Number of experts in this product launch. */
    const void* x_q8_1;       /**< Device pointer to MMQ-layout Q8_1 activations. */
    const int32_t* bounds_dev;/**< Device pointer `[n_experts + 1]` row bounds in `x_q8_1`. */
    const int32_t* ids_dev;   /**< Device pointer `[total_rows]` destination row mapping. */
    int64_t total_rows;       /**< Total routed activation rows across `n_experts`. */
    int64_t max_rows;         /**< Upper bound on rows assigned to any single expert (launch grid). */
    float* dst;               /**< Device pointer to FP32 output buffer. */
    int64_t ld_dst;           /**< Row stride (in floats) of `dst`. */
    const float* y_scale_dev; /**< Optional per-row activation scale (NULL unless FP4 activations). */
} tk_moe_mmq_product;

/* ==========================================================================
 * 1. Context & Configuration Entry Points
 *    [NOT CAPTURABLE] Host setup and teardown only.
 * ========================================================================== */

/** Create a configuration initialized with portable defaults. */
tk_status tk_config_create(tk_config** out_cfg) TK_NOEXCEPT;

/** Create a configuration initialized with architecture-specific defaults (e.g. gfx1151 switches). */
tk_status tk_config_create_for_arch(tk_arch arch, tk_config** out_cfg) TK_NOEXCEPT;

/** Overlay `STRATA_*` / `TALUS_*` environment variables onto `cfg`. */
tk_status tk_config_from_env(tk_config* cfg) TK_NOEXCEPT;

/** Destroy a `tk_config` handle (no-op if `cfg == NULL`). */
void tk_config_destroy(tk_config* cfg) TK_NOEXCEPT;

/** Set an integer knob on `cfg` by key name. */
tk_status tk_config_set_int(tk_config* cfg, const char* key, int64_t value) TK_NOEXCEPT;

/** Read an integer knob from `cfg` by key name. */
tk_status tk_config_get_int(const tk_config* cfg, const char* key, int64_t* out_value) TK_NOEXCEPT;

/**
 * Create a per-device kernel context (`hipBLASLt` handles, MMQ scratch, tuning tables).
 * If `cfg == NULL`, architecture defaults for `device` are used.
 */
tk_status tk_ctx_create(int32_t device, const tk_config* cfg, tk_ctx** out_ctx) TK_NOEXCEPT;

/** Destroy a `tk_ctx` and release all associated hipBLASLt/MMQ resources (no-op if `ctx == NULL`). */
void tk_ctx_destroy(tk_ctx* ctx) TK_NOEXCEPT;

/** Query the HIP device ordinal bound to `ctx`. */
tk_status tk_ctx_device(const tk_ctx* ctx, int32_t* out_device) TK_NOEXCEPT;

/** Query the detected `tk_arch` for `ctx`. */
tk_status tk_ctx_arch(const tk_ctx* ctx, tk_arch* out_arch) TK_NOEXCEPT;

/**
 * Load per-architecture tuning tables (`<arch>-hipblaslt-<ver>.txt`, MMVQ row tables) from `dir`.
 * Preserves the version-sibling hipBLASLt fallback from commit 5c7e9867.
 */
tk_status tk_ctx_load_tuning(tk_ctx* ctx, const char* dir) TK_NOEXCEPT;

/**
 * Return the last error message recorded on `ctx` (or thread-local error buffer if `ctx == NULL`).
 * The returned pointer is valid until the next `tk_*` call on the same `ctx` or thread.
 */
const char* tk_last_error(const tk_ctx* ctx) TK_NOEXCEPT;

/** Return a static human-readable name for a `tk_status` code. Never returns NULL. */
const char* tk_status_string(tk_status status) TK_NOEXCEPT;

/* ==========================================================================
 * 2. Packing & Activation Quantization
 * ========================================================================== */

/**
 * Quantize `ncols` contiguous columns of `n_in` FP32 activations into `x_q8_1` (36 bytes per 32 floats).
 * [CAPTURABLE] `x` and `x_q8_1` are fixed device pointers; `n_in` (multiple of 32) and `ncols` are
 * host scalars baked at capture time.
 */
tk_status tk_quantize_q8_1(
    tk_ctx* ctx,
    const float* x,
    void* x_q8_1,
    int32_t n_in,
    int32_t ncols,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Repack a GGUF Q8_0 matrix `[n_out, n_in]` into separate `qs` (`int8`) and `d` (`fp16`) planes.
 * [NOT CAPTURABLE] Allocates device memory and returns an explicit `tk_packed_weights*` handle.
 */
tk_status tk_pack_q8(
    tk_ctx* ctx,
    const void* w_gguf,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream,
    tk_packed_weights** out_packed) TK_NOEXCEPT;

/**
 * Repack a GGUF Q6_K matrix `[n_out, n_in]` into separate `ql`, `qh`, `scales`, and `d` planes.
 * [NOT CAPTURABLE] Allocates device memory and returns an explicit `tk_packed_weights*` handle.
 */
tk_status tk_pack_q6(
    tk_ctx* ctx,
    const void* w_gguf,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream,
    tk_packed_weights** out_packed) TK_NOEXCEPT;

/**
 * Free a `tk_packed_weights` handle created by `tk_pack_q8` or `tk_pack_q6` (no-op if `packed == NULL`).
 * [NOT CAPTURABLE]
 */
void tk_packed_free(tk_ctx* ctx, tk_packed_weights* packed) TK_NOEXCEPT;

/* ==========================================================================
 * 3. MMVQ & MoE Decode / Verify Entry Points
 * ========================================================================== */

/**
 * Quantized matrix-vector / multi-column verify product: `y[c, r] = dot(w[r], x_q8_1[c])`.
 * Supports `ncols` in `1..64` (each column bitwise equal to a single-column `ncols == 1` call; previously 1..16).
 * [CAPTURABLE] `w`, `x_q8_1`, `y` are fixed device addresses; `qtype`, `n_in`, `n_out`, `ncols` are
 * host scalars baked at capture time.
 */
tk_status tk_mmvq(
    tk_ctx* ctx,
    tk_qtype qtype,
    const void* w,
    const void* x_q8_1,
    float* y,
    int32_t n_in,
    int32_t n_out,
    int32_t ncols,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Quantized mat-vec reading from an explicit repacked `tk_packed_weights` handle (Q8_0 or Q6_K).
 * Output is bitwise identical to `tk_mmvq` on the original GGUF weight blocks.
 * [CAPTURABLE] `packed`, `x_q8_1`, `y` are fixed device addresses; `ncols` is baked at capture time.
 */
tk_status tk_mmvq_packed(
    tk_ctx* ctx,
    const tk_packed_weights* packed,
    const void* x_q8_1,
    float* y,
    int32_t ncols,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Indirect MoE expert quantized mat-vec (`mul_mat_vec_q` with device expert IDs).
 * Expert `ids_dev[k]` lies at `w + ids_dev[k] * expert_bytes`. When `w2 != NULL` and `y2 != NULL`,
 * computes the paired projection (e.g. gate and up) for the same `ids_dev` in one launch.
 * Covers Q4_K, Q5_K, Q6_K, Q8_0, and (after WP-1.6/1.7) IQ2_XXS, IQ3_XXS, IQ2_XS, IQ2_S, IQ3_S.
 * [CAPTURABLE] `ids_dev` is a device array of `n_ids` expert indices read on the device during replay;
 * `n_ids`, `expert_bytes`, `x_per_id`, `n_in`, `n_out` are baked at capture time.
 */
tk_status tk_mmvq_ids(
    tk_ctx* ctx,
    tk_qtype qtype,
    const void* w,
    size_t expert_bytes,
    const int32_t* ids_dev,
    int32_t n_ids,
    const void* x_q8_1,
    bool x_per_id,
    float* y,
    int32_t n_in,
    int32_t n_out,
    const void* w2,
    float* y2,
    tk_stream stream) TK_NOEXCEPT;

/**
 * FP32 matrix-vector product with FP64 accumulation (`strata::laguna::f32_matvec`, reference router).
 * [CAPTURABLE] `w`, `x`, `y` are device pointers; `n_in`, `n_out` are baked at capture time.
 */
tk_status tk_f32_matvec(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fast FP32 matrix-vector product, one warp per row, FP32 accumulation (`n_in % 128 == 0`).
 * [CAPTURABLE] `w`, `x`, `y` are device pointers; `n_in`, `n_out` are baked at capture time.
 */
tk_status tk_f32_matvec_fast(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream) TK_NOEXCEPT;

/**
 * FP32 matrix-vector product, one 128-thread block per row (`strata::laguna::f32_matvec_rows`).
 * [CAPTURABLE] `w`, `x`, `y` are device pointers; `n_in`, `n_out` are baked at capture time.
 */
tk_status tk_f32_matvec_rows(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Multi-token FP32 router mat-vec for `n_tok` (`1..8`) tokens (`strata::laguna::f32_matvec_rows_n`).
 * Each weight row `W[o]` is read once across all `n_tok` columns.
 * [CAPTURABLE] `w`, `x`, `y` are device pointers; `n_in`, `n_out`, `n_tok` are baked at capture time.
 */
tk_status tk_f32_matvec_rows_n(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Batched FP32 matrix-matrix product `y[t, o] = dot(w[o], x[t])` for `n_tok` rows.
 * [CAPTURABLE] `w`, `x`, `y` are device pointers; `n_in`, `n_out`, `n_tok` are baked at capture time.
 */
tk_status tk_f32_matmul(
    tk_ctx* ctx,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Sigmoid + bias top-k routing for one token on the device (`n_expert <= 256`, `k <= 16`).
 * Ties go to the lower expert index; weights are `p / max(sum, 6.1e-5) * scale`.
 * [CAPTURABLE] `logits`, `bias`, `ids_dev`, `weights_dev` are device pointers; `n_expert`, `k`, `scale`
 * are baked at capture time.
 */
tk_status tk_route_sigmoid_topk(
    tk_ctx* ctx,
    const float* logits,
    const float* bias,
    int32_t* ids_dev,
    float* weights_dev,
    int32_t n_expert,
    int32_t k,
    float scale,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Single-warp sigmoid top-k routing for 256 experts (`strata::laguna::route_warp`).
 * [CAPTURABLE] `logits`, `bias`, `ids_dev`, `weights_dev` are device pointers; `k`, `scale` are baked.
 */
tk_status tk_route_warp(
    tk_ctx* ctx,
    const float* logits,
    const float* bias,
    int32_t* ids_dev,
    float* weights_dev,
    int32_t k,
    float scale,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Multi-token sigmoid top-k routing for `n_tok` tokens x 256 experts (`strata::laguna::route_topk`).
 * [CAPTURABLE] `logits`, `bias`, `ids_dev`, `weights_dev` are device pointers; `n_tok`, `k`, `scale`
 * are baked at capture time.
 */
tk_status tk_route_topk(
    tk_ctx* ctx,
    const float* logits,
    const float* bias,
    int32_t n_tok,
    int32_t k,
    float scale,
    int32_t* ids_dev,
    float* weights_dev,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Prefill device routing and expert grouping for `n_tok` tokens (256 experts):
 * computes top-k (`sel_e_dev`, `sel_w_dev`: `n_tok * k`), then groups rows by expert in `(token, slot)`
 * order into `bounds_dev` (257), `rtok_dev` (row -> token), and `smap_dev` (`(token, slot)` -> row).
 * `cnt_dev` is 256 `int32_t` of device scratch.
 * [CAPTURABLE] All pointer arguments are device pointers; `n_tok`, `k`, `scale` are baked.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fused router projection + `route_warp` top-k in one kernel (`strata::laguna::router_route`).
 * `counter_dev` is a single device `uint32_t` initialized to 0 and reset to 0 by the last block.
 * [CAPTURABLE] `counter_dev`, `ids_dev`, `weights_dev` are device pointers safe for graph replay.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Accumulate shared expert output and routed expert slots into residual `x`:
 * `x[t] += shared[t] + sum_{j=0..k-1} weights_dev[t*k+j] * y[map_dev[t*k+j]]`.
 * [CAPTURABLE] `map_dev` and `weights_dev` are device arrays; `n_tok`, `n_embd`, `k` are baked.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/* ==========================================================================
 * 4. Norms, RoPE, Elementwise, Embedding & Sampling Helpers
 * ========================================================================== */

/**
 * In-place weighted RMSNorm over the last axis: `x[r, c] = (x[r, c] / rms(x[r])) * w[c]`.
 * `w` may be NULL for unweighted RMSNorm.
 * [CAPTURABLE] `x` and `w` are device pointers; `rows`, `cols`, `eps` are baked at capture time.
 */
tk_status tk_rms_norm(
    tk_ctx* ctx,
    float* x,
    const float* w,
    int64_t rows,
    int64_t cols,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Out-of-place (or in-place when `dst == src`) weighted RMSNorm: `dst = RMSNorm(src) * w`.
 * [CAPTURABLE] `dst`, `src`, `w` are device pointers; `rows`, `cols`, `eps` are baked at capture time.
 */
tk_status tk_rms_norm_to(
    tk_ctx* ctx,
    float* dst,
    const float* src,
    const float* w,
    int32_t rows,
    int32_t cols,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Per-head RMSNorm (`w[128]`, shared across heads) followed by in-place NEOX RoPE on the first
 * `n_rot` dims of each 128-dim head row (`strata::laguna::head_norm_rope`).
 * `pos_dev[row / pos_div]` supplies the token position (`pos_div == 0` uses `pos_dev[0]` for all rows).
 * [CAPTURABLE] `pos_dev` is a device pointer so token position advances on graph replay.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fused decode/verify QKV per-head RMSNorm + NEOX RoPE + KV cache store (`qkv_norm_rope_store`).
 * Normalizes and rotates `q` in place; normalizes, rotates, and stores `kc` to `k_pool` at `*pos_dev`;
 * stores `vc` to `v_pool` at `*pos_dev` (with optional transposed V `vt_stride` and Q8 scales).
 * [CAPTURABLE] `pos_dev` is a fixed-address device pointer read by the kernel on replay.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Partial NEOX RoPE rotation over `[rows, head_dim]`, rotating the first `n_rot` dims of each row
 * using device position array `pos_dev[row]` (`strata::kernels::rope_neox_apply`). `out` may equal `x`.
 * [CAPTURABLE] `pos_dev` is a device array (`[rows]`), never a host scalar.
 */
tk_status tk_rope_neox(
    tk_ctx* ctx,
    const float* x,
    float* out,
    int64_t rows,
    int32_t head_dim,
    int32_t n_rot,
    const float* cos_tab,
    const float* sin_tab,
    const int32_t* pos_dev,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Interleaved multimodal RoPE (M-RoPE, e.g. sections `{11, 11, 10, 0}`).
 * When `mrope_table_dev != NULL` (`[cells][3]` `(t, h, w)` on device), `pos_dev` entries index cells
 * in `mrope_table_dev`; when `NULL`, behaves identically to `tk_rope_neox`.
 * [CAPTURABLE] `pos_dev` and `mrope_table_dev` are device pointers.
 */
tk_status tk_mrope(
    tk_ctx* ctx,
    const float* x,
    float* out,
    int64_t rows,
    int32_t head_dim,
    int32_t n_rot,
    const int32_t sections[4],
    const float* cos_tab,
    const float* sin_tab,
    const int32_t* pos_dev,
    const int32_t* mrope_table_dev,
    tk_stream stream) TK_NOEXCEPT;

/**
 * In-place SwiGLU elementwise product: `gate[i] = silu(gate[i]) * up[i]` for `n` elements.
 * [CAPTURABLE] `gate` and `up` are device pointers; `n` is baked at capture time.
 */
tk_status tk_silu_mul(
    tk_ctx* ctx,
    float* gate,
    const float* up,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Scaled vector addition: `y[i] += alpha * x[i]` for `n` FP32 elements.
 * [CAPTURABLE] `y` and `x` are device pointers; `alpha` and `n` are baked at capture time.
 */
tk_status tk_axpy(
    tk_ctx* ctx,
    float* y,
    const float* x,
    float alpha,
    int32_t n,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fused MoE combine + residual add + RMSNorm + Q8_1 quantization (`strata::laguna::add_norm_quant`):
 * `x += add + sum_{j=0..k-1} weights_dev[j] * yexp[j*n]`, then writes `h_out = RMSNorm(x) * norm_w`
 * (when `h_out != NULL`) and quantizes `h_out` into `q8_out` (`n % 32 == 0`, `n <= 4096`).
 * [CAPTURABLE] `weights_dev` is a device pointer; `k`, `n`, `eps`, `n_rows` are baked at capture time.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * residual axpy + RMSNorm + Q8_1 in one launch: `x = fma(alpha, ax, x)` (bitwise `tk_axpy`), then the same
 * `h_out = RMSNorm(x) * norm_w` / Q8_1 as `tk_add_norm_quant` (`n % 32 == 0`, `n <= 4096`).
 * [CAPTURABLE]
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Dequantize `n_tok` rows (`tok_dev[0..n_tok)`) from a Q8_0 embedding table into FP32 `x`.
 * `table` may reside in device memory or zero-copy host-mapped memory.
 * [CAPTURABLE] `tok_dev` is a fixed-address device pointer holding token IDs.
 */
tk_status tk_embed_q8_0(
    tk_ctx* ctx,
    const void* table,
    const int32_t* tok_dev,
    float* x,
    int32_t n_embd,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Gather `n_rows` rows of `row_bytes` (multiple of 4) from `src` into `dst`: `dst[r] = src[idx_dev[r]]`.
 * [CAPTURABLE] `idx_dev` is a device array of row indices.
 */
tk_status tk_gather_rows(
    tk_ctx* ctx,
    const void* src,
    void* dst,
    const int32_t* idx_dev,
    int32_t n_rows,
    size_t row_bytes,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fill `pos_dev[i] = p0 + i / per_tok` for `n_tok * per_tok` entries from host scalar `p0`.
 * [NOT FOR DYNAMIC REPLAY] Bakes `p0` at launch time; use `tk_fill_pos_dev` inside captured decode graphs.
 */
tk_status tk_fill_pos(
    tk_ctx* ctx,
    int32_t* pos_dev,
    int32_t n_tok,
    int32_t per_tok,
    int32_t p0,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fill `out_dev[i] = *pos_dev` for `n` entries reading the position from device pointer `pos_dev`.
 * [CAPTURABLE] `pos_dev` is a fixed device pointer.
 */
tk_status tk_fill_pos_dev(
    tk_ctx* ctx,
    int32_t* out_dev,
    int32_t n,
    const int32_t* pos_dev,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Compute `*tok_dev = argmax(logits[0..n))` (ties to lower index) and increment `*pos_dev += 1`.
 * [CAPTURABLE] `tok_dev` and `pos_dev` are fixed device pointers updated in place on the device.
 */
tk_status tk_argmax_advance(
    tk_ctx* ctx,
    const float* logits,
    int32_t n,
    int32_t* tok_dev,
    int32_t* pos_dev,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Compute `out_dev[r] = argmax(logits[r * n_vocab .. (r + 1) * n_vocab))` for `n_rows` rows.
 * [CAPTURABLE] `logits` and `out_dev` are device pointers; `n_rows` and `n_vocab` are baked.
 */
tk_status tk_argmax_rows(
    tk_ctx* ctx,
    const float* logits,
    int32_t n_rows,
    int32_t n_vocab,
    int32_t* out_dev,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Apply Laguna per-head softplus attention output gate in place: `o[h, d] *= softplus(g[h])`.
 * [CAPTURABLE] `o` and `g` are device pointers; `n_head` and `head_dim` are baked at capture time.
 */
tk_status tk_softplus_head_gate(
    tk_ctx* ctx,
    float* o,
    const float* g,
    int32_t n_head,
    int32_t head_dim,
    tk_stream stream) TK_NOEXCEPT;

/* ==========================================================================
 * 5. Attention & KV Cache (Strata, q36, Laguna, MiMo, GLM)
 * ========================================================================== */

/**
 * Split-K decode / verify window causal GQA attention (q36 WMMA / Strata decode attention).
 * Static split-K chunk boundaries (128 cells below 4K, 512 above) guarantee verify window rows
 * `T = 1..16` are bitwise identical to `T = 1` decode rows.
 * [CAPTURABLE] `pos_dev` is a fixed device pointer supplying the starting token position;
 * `n_tok`, `n_head`, `n_kv`, `head_dim`, `head_dim_v`, `kv_stride`, `window`, `scale`, `n_split`
 * are host scalars baked at capture time.
 */
tk_status tk_attn_decode(
    tk_ctx* ctx,
    const float* q,
    const void* k_pool,
    const void* v_pool,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    float* out,
    float* scratch,
    int32_t n_tok,
    int32_t n_head,
    int32_t n_kv,
    int32_t head_dim,
    int32_t head_dim_v,
    int32_t kv_stride,
    int32_t window,
    float scale,
    const int32_t* pos_dev,
    int32_t n_split,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Prompt chunk causal flash attention on WMMA (`strata::prefill::fattn::attend`).
 * Caller MUST zero K and V padding cells `[p0 + n_tok, kv_padded(p0 + n_tok))` before launch so
 * WMMA `P * V` products over masked cells cannot read stale cache values (commit 468e5be1).
 * [CAPTURABLE] Stream-ordered on `stream`; `p0` and `n_tok` are baked per prompt chunk.
 */
tk_status tk_attn_prefill_fa(
    tk_ctx* ctx,
    const float* q,
    const void* k_pool,
    const void* v_pool,
    float* out,
    int32_t n_tok,
    int32_t n_head,
    int32_t n_kv,
    int32_t head_dim,
    int64_t kv_stride,
    int64_t p0,
    float scale,
    void* mask,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Laguna prompt/decode causal attention (`strata::laguna::prompt_attn`).
 * When `q_pos0_dev != NULL`, replaces host scalar `q_pos0` with `*q_pos0_dev` for graph replay.
 * [CAPTURABLE when `q_pos0_dev != NULL`]
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Laguna RDNA3/3.5 WMMA prompt/decode causal attention (`strata::laguna::prompt_attn_wmma`).
 * Requires tile-transposed V (`vt_stride > 0`, multiple of 16).
 * [CAPTURABLE when `q_pos0_dev != NULL`]
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Laguna prefetching split-K decode/verify attention (`strata::laguna::decode_attn`).
 * Supports optional Quest-style block-sparse selection indices in `sel_dev`.
 * [CAPTURABLE] `pos_dev` and optional `sel_dev` are device pointers; `n_split`, `n_tok`, `sel_k`,
 * `sel_per_kv`, `sel_bs` are baked at capture time.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Merge split-K attention partials (`part_acc`, `part_ml`), apply `softplus(g[h])` per head,
 * and quantize the result to Q8_1 (`strata::laguna::attn_merge_gate_quant`).
 * [CAPTURABLE] All pointer arguments are device pointers; `n_head`, `n_split`, `n_tok` are baked.
 */
tk_status tk_attn_merge_gate_quant(
    tk_ctx* ctx,
    const float* part_acc,
    const float* part_ml,
    const float* g,
    void* q8_out,
    int32_t n_head,
    int32_t n_split,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Store one token's FP32 `kc` and `vc` (`n` floats each) into FP16 `k_pool` and `v_pool` at cell `*pos_dev`.
 * [CAPTURABLE] `pos_dev` is a fixed device pointer.
 */
tk_status tk_kv_store_f16(
    tk_ctx* ctx,
    uint16_t* k_pool,
    uint16_t* v_pool,
    const float* kc,
    const float* vc,
    int32_t n,
    const int32_t* pos_dev,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Store `nb` tokens' FP32 V rows (`[nb, n_kv, 128]`) into tile-transposed FP16 `v_pool`
 * (`[kv_head][pos/16][dim][pos%16]`) starting at position `p0`.
 * [CAPTURABLE] `p0`, `nb`, `n_kv`, `stride` are baked per chunk.
 */
tk_status tk_v_store_t(
    tk_ctx* ctx,
    const float* vc,
    uint16_t* v_pool,
    int32_t nb,
    int32_t n_kv,
    int32_t p0,
    int32_t stride,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Quantize and store `nb` tokens' FP32 K and V rows into Q8 KV pools (`int8` codes + FP16 scale per 32 dims,
 * V tile-transposed) starting at position `p0` (`strata::laguna::kv_store_q8`).
 * [CAPTURABLE] `p0`, `nb`, `n_kv`, `stride` are baked per chunk.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Recompute Quest-style per-block elementwise min/max key bounds (`strata::laguna::kbound_update`).
 * When `pos_dev != NULL`, updates the blocks covering positions `*pos_dev .. *pos_dev + n_tok - 1`.
 * [CAPTURABLE when `pos_dev != NULL`]
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Compute Quest-style per-block upper-bound query-key scores `ub` (`[n_tok, n_head, nb_max]`).
 * [CAPTURABLE] `pos_dev` is a fixed device pointer.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/**
 * Select top-k key blocks per token from `ub` with forced `sink` and `recent` blocks (`sparse_select`).
 * [CAPTURABLE] `pos_dev`, `sel_dev`, and optional `scr_dev` are device pointers.
 */
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
    tk_stream stream) TK_NOEXCEPT;

/** tk_attn_swa_sinks (F16 KV, n_split 0, value_scale 1) that, where the prefill v2 kernel runs, also writes
 * the outputs as the wo projection's MMQ q8_1 rows into `q8_mmq` (q8_layout 0 = D4, 1 = DS4, as tk_q36_mmq_q8_layout;
 * bitwise tk_q36_mmq_quantize of `out`); *q8_written = 1 when it did, else 0 (out is always written). */
tk_status tk_attn_swa_sinks_mmq(
    tk_ctx* ctx,
    const float* q,
    const void* k_pool,
    const void* v_pool,
    const float* sinks,
    float* out,
    float* scratch,
    int32_t n_tok,
    int32_t n_head,
    int32_t n_kv,
    int32_t head_dim,
    int32_t head_dim_v,
    int32_t window,
    int32_t stride,
    float scale,
    const int32_t* pos_dev,
    void* q8_mmq,
    int32_t q8_layout,
    int32_t* q8_written,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Sliding-window attention (e.g. window=128) with per-head learned attention sinks (MiMo, WP-1.6).
 * [CAPTURABLE] `pos_dev` is a fixed device pointer; `window`, `stride`, `n_split`, `n_tok` are baked.
 */
tk_status tk_attn_swa_sinks(
    tk_ctx* ctx,
    const float* q,
    const void* k_pool,
    const void* v_pool,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    const float* sinks,
    float* out,
    float* scratch,
    int32_t n_tok,
    int32_t n_head,
    int32_t n_kv,
    int32_t head_dim,
    int32_t head_dim_v,
    int32_t window,
    int32_t stride,
    float scale,
    const int32_t* pos_dev,
    int32_t n_split,
    tk_stream stream) TK_NOEXCEPT;

/** as `tk_attn_swa_sinks`, and also writes the outputs as Q8_1 to `q8_out` (the bytes `tk_quantize_q8_1` of `out`
 * would write; fused into the decode kernels' epilogue). [CAPTURABLE] */
tk_status tk_attn_swa_sinks_q8(
    tk_ctx* ctx,
    const float* q,
    const void* k_pool,
    const void* v_pool,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    const float* sinks,
    float* out,
    float* scratch,
    int32_t n_tok,
    int32_t n_head,
    int32_t n_kv,
    int32_t head_dim,
    int32_t head_dim_v,
    int32_t window,
    int32_t stride,
    float scale,
    const int32_t* pos_dev,
    int32_t n_split,
    tk_stream stream,
    void* q8_out) TK_NOEXCEPT;

/**
 * Asymmetric full causal GQA attention with `head_dim_k = 192` and `head_dim_v = 128` (MiMo, WP-1.6).
 * [CAPTURABLE] `pos_dev` is a fixed device pointer; `stride`, `scale`, `n_split`, `n_tok` are baked.
 */
tk_status tk_attn_k192_v128(
    tk_ctx* ctx,
    const float* q,
    const void* k_pool,
    const void* v_pool,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    float* out,
    float* scratch,
    int32_t n_tok,
    int32_t n_head,
    int32_t n_kv,
    int32_t stride,
    float scale,
    const int32_t* pos_dev,
    int32_t n_split,
    tk_stream stream) TK_NOEXCEPT;

/** as `tk_attn_k192_v128`, and also writes the outputs as Q8_1 to `q8_out` (the bytes `tk_quantize_q8_1` of `out`
 * would write; fused into the decode kernels' epilogue). [CAPTURABLE] */
tk_status tk_attn_k192_v128_q8(
    tk_ctx* ctx,
    const float* q,
    const void* k_pool,
    const void* v_pool,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    float* out,
    float* scratch,
    int32_t n_tok,
    int32_t n_head,
    int32_t n_kv,
    int32_t stride,
    float scale,
    const int32_t* pos_dev,
    int32_t n_split,
    tk_stream stream,
    void* q8_out) TK_NOEXCEPT;

/**
 * MiMo decode: q + k partial NeoX rope in place (bitwise `tk_rope_neox` on the q rows and the k rows) and the
 * f16 K / V cache store (bitwise `tk_q36_dflash_kv_store_f16`: cell = (store_pos_dev[0] + t) % ring) in one launch.
 * q: [n_tok][n_head_q][head_dim], k: [n_tok][n_head_kv][head_dim], v: [n_tok][n_head_kv][v_dim] (f32);
 * pos_q_dev: n_tok * n_head_q positions, pos_k_dev: n_tok * n_head_kv positions.
 * [CAPTURABLE] all pointers are fixed device pointers.
 */
tk_status tk_mimo_rope_qk_store(
    tk_ctx* ctx,
    float* q,
    float* k,
    const float* v,
    int32_t n_tok,
    int32_t n_head_q,
    int32_t n_head_kv,
    int32_t head_dim,
    int32_t v_dim,
    int32_t n_rot,
    const float* cos_tab,
    const float* sin_tab,
    const int32_t* pos_q_dev,
    const int32_t* pos_k_dev,
    void* k_pool,
    void* v_pool,
    const int32_t* store_pos_dev,
    int32_t ring,
    tk_stream stream) TK_NOEXCEPT;

/**
 * DeepSeek/GLM Dynamic Sparse Attention (DSA) lightning indexer decode scorer + top-k selector (WP-1.7).
 * Scores candidate keys up to `*pos_dev` and writes `top_k` selected indices to `topk_ids_dev`.
 * [CAPTURABLE] `pos_dev` and `topk_ids_dev` are fixed device pointers.
 */
tk_status tk_dsa_decode_indexer(
    tk_ctx* ctx,
    const float* q_idx,
    const void* k_idx_pool,
    const float* weights,
    const int32_t* pos_dev,
    int32_t n_tok,
    int32_t idx_heads,
    int32_t idx_dim,
    int32_t top_k,
    int32_t capacity,
    float* scores_scratch,
    int32_t* topk_ids_dev,
    tk_stream stream) TK_NOEXCEPT;

/* ==========================================================================
 * 6. GDN & KDA Recurrent Layers
 * ========================================================================== */

/**
 * Gated DeltaNet (GDN) preprocessing: 4-tap causal conv1d + SiLU (`conv_state` updated in place),
 * L2 normalization on Q and K heads, softplus decay gate `gate_out = softplus(alpha + dt) * ssm_a`,
 * and in-place sigmoid on `beta_inout`.
 * [CAPTURABLE] All pointer arguments are device pointers; dimensions are baked at capture time.
 */
tk_status tk_gdn_preprocess(
    tk_ctx* ctx,
    float* conv_state,
    const float* qkv,
    const float* conv_w,
    float* conv_out,
    const float* alpha,
    const float* dt,
    const float* ssm_a,
    float* gate_out,
    float* beta_inout,
    int64_t channels,
    int64_t d_conv,
    int64_t h_k,
    int64_t h_v,
    int64_t state_size,
    float l2_eps,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Single-token Gated DeltaNet recurrence step in place on `state` (`[S, h_v, S]`, `j` fastest).
 * [CAPTURABLE] All pointer arguments are device pointers; `state_size`, `h_k`, `h_v` are baked.
 */
tk_status tk_gdn_step(
    tk_ctx* ctx,
    float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* out,
    int64_t state_size,
    int64_t h_k,
    int64_t h_v,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Multi-token sequential/chunked GDN recurrence over `n_tok` consecutive tokens (prompt chunks and
 * verify windows `T = 1..16`, including fused output RMSNorm + gate `z`).
 * `silu_gate`: `true` for SiLU gate (`qwen35`), `false` for sigmoid gate (`qwen4exp`).
 * [CAPTURABLE] All pointer arguments are device pointers; `n_tok` and dimensions are baked.
 */
tk_status tk_gdn_seq(
    tk_ctx* ctx,
    float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    const float* z,
    const float* gamma,
    float* out,
    int32_t n_tok,
    int64_t state_size,
    int64_t h_k,
    int64_t h_v,
    float norm_eps,
    bool silu_gate,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fused single-token GDN recurrence step + per-head RMSNorm + output gate (`fused_gdn_step_norm` /
 * `fused_gdn_step_norm_silu`):
 * `y = rmsnorm((S^T q) / sqrt(128), eps) * gamma * (silu_gate ? silu(z) : sigmoid(z))`.
 * [CAPTURABLE] All pointer arguments are device pointers; `h_k`, `h_v`, `eps`, `silu_gate` are baked.
 */
tk_status tk_gdn_fused(
    tk_ctx* ctx,
    float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    const float* z,
    const float* gamma,
    float eps,
    float* y,
    int32_t h_k,
    int32_t h_v,
    bool silu_gate,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Key-driven / Kimi Delta Attention (KDA) recurrent step with per-channel decay `gk` (`[h_v, state_size]`)
 * and fused output RMSNorm + gate (GLM-5.3-Flash spike, WP-1.7).
 * [CAPTURABLE] All pointer arguments are device pointers; `n_tok`, `state_size`, `h_k`, `h_v` are baked.
 */
tk_status tk_kda_step(
    tk_ctx* ctx,
    float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* gk,
    const float* beta,
    const float* z,
    const float* gamma,
    float eps,
    float* out,
    int32_t n_tok,
    int32_t state_size,
    int32_t h_k,
    int32_t h_v,
    tk_stream stream) TK_NOEXCEPT;

/* ==========================================================================
 * 7. Prefill GEMM & MMQ Entry Points
 * ========================================================================== */

/**
 * Batched prompt projection `Y[n_tok, n_out] = X[n_tok, n_in] * W[n_out, n_in]^T + beta * Y`
 * via hipBLASLt / tuned WMMA (`strata::prefill::Gemm::bf16` / `f16` / `native`).
 * [STREAM-ORDERED] Uses `ctx`'s hipBLASLt handle, workspace, and tuning table.
 */
tk_status tk_prefill_gemm(
    tk_ctx* ctx,
    tk_qtype w_qtype,
    const void* w,
    const void* x,
    tk_qtype x_qtype,
    float* y,
    int64_t n_tok,
    int64_t n_out,
    int64_t n_in,
    int64_t ldy,
    int64_t ldx,
    float beta,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Dense quantized prompt projection using llama.cpp int8/WMMA MMQ tiles (`STRATA_DENSE_MMQ`).
 * `w_blocks` remains in native GGUF block layout; `x_q8_1` holds MMQ-layout quantized activations.
 * [STREAM-ORDERED]
 */
tk_status tk_prefill_mmq(
    tk_ctx* ctx,
    tk_qtype w_qtype,
    const void* w_blocks,
    const void* x_q8_1,
    float* y,
    int64_t n_tok,
    int64_t n_out,
    int64_t n_in,
    int64_t ldy,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Grouped MoE expert MMQ product over `product->n_experts` experts (`strata::prefill::mmq::Context::run`).
 * [STREAM-ORDERED] `product->bounds_dev` and `product->ids_dev` are device arrays.
 */
tk_status tk_prefill_moe_mmq(
    tk_ctx* ctx,
    const tk_moe_mmq_product* product,
    tk_stream stream) TK_NOEXCEPT;

/**
 * as tk_prefill_moe_mmq_gu, but the SwiGLU rows go straight into `xq_out` as the down projection's MMQ q8_1
 * activations (`xq_rows` rows, the layout tk_q36_mmq_quantize writes for `down_qtype`; the buffer is registered as
 * tk_register_mmq_q8_buf would) - bitwise the FP32 rows followed by tk_q36_mmq_quantize(rows, down_qtype).
 * TK_ERR_UNSUPPORTED (nothing launched) unless the gate/up type is a one-scale-per-32 split-decoder type (IQ2_XXS,
 * IQ3_XXS, IQ3_S), the down type uses the D4 activation layout, w_rows % 128 == 0, the balanced kernel (TP_MOE_VER >= 73)
 * and the moe_gu row rule hold; TALUS_PF_MOE_GU_Q8=0 disables it.  [STREAM-ORDERED]
 */
tk_status tk_prefill_moe_mmq_gu_q8(
    tk_ctx* ctx,
    const tk_moe_mmq_product* gate,
    const void* w_up,
    float swiglu_limit,
    int32_t down_qtype,
    void* xq_out,
    int64_t xq_rows,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Fused grouped gate/up + SwiGLU : `gate` is the gate experts' product exactly as for tk_prefill_moe_mmq (MMQ-layout
 * q8_1 activations registered with tk_register_mmq_q8_buf, n_experts > 1), `w_up` the up experts (same qtype, shape
 * and expert_bytes).  Writes gate->dst[ids[row]][f] = g / (1 + exp(-g)) * u for f < w_rows (g = min(g, L) and
 * u = clamp(u, -L, L) when swiglu_limit L > 0) - bitwise the two tk_prefill_moe_mmq products followed by the engines'
 * SwiGLU (glm swiglu_clamp, tk_silu_mul).  TK_ERR_UNSUPPORTED (nothing launched) when the type / shape / device is not
 * covered (IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M; w_rows % 64, w_cols % 256; gfx11), not measured faster on this
 * GPU (tp_tune.hpp moe_gu: on gfx1151 >= 8 routed rows per expert, <= 32 for IQ2_S / IQ2_XS / IQ1_M), or TALUS_PF_MOE_GU=0
 * (=2 forces it where covered): the caller then runs its separate path.  [STREAM-ORDERED]
 */
tk_status tk_prefill_moe_mmq_gu(
    tk_ctx* ctx,
    const tk_moe_mmq_product* gate,
    const void* w_up,
    float swiglu_limit,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Grouped MoE expert MMQ / fused int8 product for IQ2/IQ3/IQ4 native expert blobs
 * (`strata::prefill::fused::experts_native` and WP-1.6 grouped IQ2/IQ3 MMQ).
 * [STREAM-ORDERED] `expert_blobs` (`[e1 - e0]`), `group_scratch`, `x_act`, `src_rows_dev`, `h_act`,
 * and `down_out` are device buffers.
 */
tk_status tk_prefill_moe_mmq_iq(
    tk_ctx* ctx,
    tk_qtype gu_qtype,
    tk_qtype down_qtype,
    const void* const* expert_blobs,
    int32_t e0,
    int32_t e1,
    int32_t n_expert,
    int64_t n_routed_rows,
    int64_t n_embd,
    int64_t n_ff,
    const void* group_scratch,
    const void* x_act,
    const int32_t* src_rows_dev,
    void* h_act,
    float* down_out,
    tk_stream stream) TK_NOEXCEPT;

/* ==========================================================================
 * 8. Additional Batch / Multi-Matrix MMVQ & Quantization Helpers
 * ========================================================================== */

/** Return the required byte size for `ncols` columns of `n_in` Q8_1 activations (`36 * (n_in / 32) * ncols`). */
size_t tk_q8_1_bytes(int32_t n_in, int32_t ncols) TK_NOEXCEPT;

/** Fused SwiGLU (`silu(gate) * up`) + Q8_1 quantization (`strata::kernels::native_swiglu_quantize_q8_1`). */
tk_status tk_swiglu_quantize_q8_1(
    tk_ctx* ctx,
    const float* gate,
    const float* up,
    void* x_q8_1,
    int32_t n_in,
    int32_t ncols,
    tk_stream stream) TK_NOEXCEPT;

/**
 * Multi-matrix wave32 MMVQ (`strata::kernels::native_mmvq_w32_multi`): `n_mats` (`1..3`) matrices of
 * identical `qtype` sharing `x_q8_1` in a single launch (e.g. `gate`+`up`, `q`+`k`+`v`).
 * [CAPTURABLE]
 */
tk_status tk_mmvq_w32_multi(
    tk_ctx* ctx,
    tk_qtype qtype,
    int32_t n_mats,
    const void* const* w_ptrs,
    float* const* y_ptrs,
    const int32_t* n_out_per_mat,
    const void* x_q8_1,
    int32_t n_in,
    int32_t ncols,
    tk_stream stream) TK_NOEXCEPT;

/**
 * the exact wide verify tiles (`strata::kernels::native_mmvq_tw`; default on, gfx1151; TALUS_MMVQ_WIDE=0 is
 * the kill switch) and their tile-plane weight copies (+13.0 GB on the 27B), which the engine places by its memory plan:
 * `talus_mmvq_wide_note` every matrix at load (1: a copy candidate), then `talus_mmvq_wide_pack_noted` (number of
 * copies made) once `talus_mmvq_wide_noted_bytes` fit.  `talus_mmvq_wide_pack_mode`: TALUS_MMVQ_WIDE_PACK, -1 unset
 * (auto), 0 never, 1 force.  Packing allocates and synchronizes. [NOT CAPTURABLE]
 */
int32_t talus_mmvq_wide_active(void) TK_NOEXCEPT;
int32_t talus_mmvq_wide_pack_mode(void) TK_NOEXCEPT;
int32_t talus_mmvq_wide_note(int32_t ggml_type, const void* weights, int32_t n_in, int32_t n_out) TK_NOEXCEPT;
uint64_t talus_mmvq_wide_noted_bytes(void) TK_NOEXCEPT;
int32_t talus_mmvq_wide_pack_noted(void) TK_NOEXCEPT;
void talus_mmvq_wide_release(const void* weights) TK_NOEXCEPT;
uint64_t talus_mmvq_wide_packed_bytes(void) TK_NOEXCEPT;

/**
 * Batched heterogeneous MMVQ launch (`strata::kernels::native_mmvq_batch`), up to 8 items concurrently.
 * [CAPTURABLE]
 */
tk_status tk_mmvq_batch(
    tk_ctx* ctx,
    const tk_mmvq_item* items,
    int32_t n_items,
    tk_stream stream) TK_NOEXCEPT;

/* ==========================================================================
 * 9. q36 / Qwen3.5 / Qwen3.8 Entry Points (Inline Kernel Split + Adraft/DFlash2)
 * ========================================================================== */

/** Sampling parameters passed to `tk_q36_argmax_samp` / `tk_q36_lse` (`talus::q36::DraftSamp`). */
typedef struct tk_q36_draft_samp {
    float tau;
    uint32_t seed;
    uint32_t ctr;
    float lmin;
    float topp;
    int32_t topk;
} tk_q36_draft_samp;

/** Pending GDN replay buffers (`talus::q36::GdnPend`). */
typedef struct tk_q36_gdn_pend {
    float* q[2];
    float* h[2];
    float* a[2];
    float* b[2];
    const int32_t* ctl;
    bool store;
} tk_q36_gdn_pend;

tk_status tk_q36_rms(
    tk_ctx* ctx,
    int32_t rows,
    const float* x,
    const float* w,
    float* y,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_rms_ld(
    tk_ctx* ctx,
    int32_t rows,
    const float* x,
    int32_t ldx,
    const float* w,
    float* out,
    int32_t ld_out,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_add_rms(
    tk_ctx* ctx,
    int32_t rows,
    float* x,
    const float* y,
    const float* w,
    float* h,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_add_rms_q8(
    tk_ctx* ctx,
    int32_t rows,
    float* x,
    const float* y,
    const float* w,
    float* h,
    void* q8,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_moe_out(
    tk_ctx* ctx,
    int32_t rows,
    float* x,
    const float* parts,
    const int32_t* prow,
    const float* wts,
    int32_t k,
    const float* shared,
    const float* sgate,
    int32_t sg_stride,
    const float* wn,
    float* h,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_rms_q8(
    tk_ctx* ctx,
    int32_t rows,
    const float* x,
    const float* w,
    float* y,
    void* q8,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_rms_ld_q8(
    tk_ctx* ctx,
    int32_t rows,
    const float* x,
    int32_t ldx,
    const float* w,
    float* out,
    int32_t ld_out,
    void* q8,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_add_rms_exact_q8(
    tk_ctx* ctx,
    int32_t rows,
    float* x,
    const float* y,
    const float* w,
    float* h,
    void* q8,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_moe_out_q8(
    tk_ctx* ctx,
    int32_t rows,
    float* x,
    const float* parts,
    const int32_t* prow,
    const float* wts,
    int32_t k,
    const float* shared,
    const float* sgate,
    int32_t sg_stride,
    const float* wn,
    float* h,
    void* q8,
    int32_t n,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_ab(
    tk_ctx* ctx,
    int32_t n_tok,
    const float* wa,
    const float* wb,
    const float* x,
    const float* dt,
    const float* ssm_a,
    float* alpha,
    float* beta,
    int32_t n_in,
    int32_t hv,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_ab2(
    tk_ctx* ctx,
    const float* wa,
    const float* wb,
    const float* x,
    const float* dt,
    const float* ssm_a,
    float* alpha,
    float* beta,
    int32_t n_in,
    int32_t hv,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_add(
    tk_ctx* ctx,
    float* x,
    const float* y,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_swiglu(
    tk_ctx* ctx,
    const float* g,
    const float* u,
    float* h,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gemv_f32(
    tk_ctx* ctx,
    int32_t n_tok,
    const float* w,
    const float* x,
    float* y,
    int32_t n_in,
    int32_t n_out,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_gb(
    tk_ctx* ctx,
    float* alpha,
    const float* dt,
    const float* ssm_a,
    float* beta,
    int32_t hv,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_gb2(
    tk_ctx* ctx,
    const float* ab,
    const float* dt,
    const float* ssm_a,
    float* alpha,
    float* beta,
    int32_t hv,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gate_rows(
    tk_ctx* ctx,
    float* sh,
    const float* gate,
    int32_t n,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_attn_prep(
    tk_ctx* ctx,
    int32_t n_tok,
    const float* qg,
    const float* kx,
    const float* vx,
    const float* qw,
    const float* kw,
    float* q,
    float* gate,
    void* kc,
    void* vc,
    int32_t h,
    int32_t hkv,
    int32_t hd,
    int32_t n_rot,
    const int32_t* pos0,
    int32_t ctx_len,
    float base,
    float eps,
    void* vt,
    int32_t vt_ctx,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_attn_part(
    tk_ctx* ctx,
    int32_t grid_x,
    int32_t grid_y,
    const float* q,
    const void* kc,
    const void* vc,
    const int32_t* pos0,
    int32_t t0,
    int32_t h,
    int32_t hkv,
    int32_t ctx_len,
    float scale,
    int32_t n_chunk,
    float* pacc,
    float* pm,
    float* pl,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_attn_merge(
    tk_ctx* ctx,
    int32_t rows,
    int32_t hd,
    const float* pacc,
    const float* pm,
    const float* pl,
    const float* gate,
    const int32_t* pos0,
    int32_t t0,
    int32_t h,
    int32_t n_chunk,
    float* out,
    int32_t chunk,
    void* q8,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_attn_wmma(
    tk_ctx* ctx,
    bool w8,
    int32_t grid_x,
    int32_t grid_y,
    const float* q,
    const void* kc,
    const void* vt,
    const int32_t* pos0,
    int32_t n_tok,
    int32_t h,
    int32_t hkv,
    int32_t ctx_len,
    int32_t vt_ctx,
    float sc,
    int32_t chunk,
    int32_t n_chunk,
    float* pacc,
    float* pm,
    float* pl,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_group(
    tk_ctx* ctx,
    const int32_t* ids,
    int32_t n_ent,
    int32_t k,
    int32_t n_expert,
    const uint8_t* blobs,
    uint64_t blob,
    uint64_t* grp_ptr,
    int32_t* grp_start,
    int32_t* n_groups,
    int32_t* ent_dst,
    int32_t* ent_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_group_hy(
    tk_ctx* ctx,
    const int32_t* ids,
    int32_t n_ent,
    int32_t k,
    int32_t n_expert,
    const int32_t* slot_of,
    const uint8_t* blobs,
    uint64_t blob,
    uint64_t* grp_ptr,
    int32_t* grp_start,
    int32_t* n_groups,
    int32_t* ent_dst,
    int32_t* ent_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_merge_cpu(
    tk_ctx* ctx,
    int32_t n_ent,
    float* parts,
    const float* cpu,
    const int32_t* miss,
    int32_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_attn_softmax16(
    tk_ctx* ctx,
    int32_t h,
    const float* s_in,
    void* s16_out,
    int32_t ld,
    int32_t n_tok,
    int32_t p_cols,
    int32_t pos0,
    float scale,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gate_mul(
    tk_ctx* ctx,
    float* att,
    const float* gate,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_route_count(
    tk_ctx* ctx,
    const int32_t* ids,
    int32_t n_ent,
    int32_t* cnt,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_route_scan(
    tk_ctx* ctx,
    const int32_t* cnt,
    int32_t n_expert,
    int32_t* bounds,
    int32_t* cur,
    int32_t* max_rows,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_route_scatter(
    tk_ctx* ctx,
    const int32_t* ids,
    int32_t n_ent,
    int32_t k,
    int32_t* cur,
    int32_t* dst,
    int32_t* src,
    int32_t* inv,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_argmax(
    tk_ctx* ctx,
    int32_t grid_x,
    int32_t grid_y,
    const float* x,
    int32_t n,
    int32_t* out,
    float* pv,
    int32_t* pi,
    bool stage2,
    const int32_t* map,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_argmax_samp(
    tk_ctx* ctx,
    const float* x,
    int32_t n,
    float* pv,
    int32_t* pi,
    const tk_q36_draft_samp* ds_dev,
    int32_t step,
    const float* thr,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_lse(
    tk_ctx* ctx,
    const float* x,
    int32_t n,
    const tk_q36_draft_samp* ds_dev,
    float* out,
    float* thr_out,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_ctl_end(
    tk_ctx* ctx,
    int32_t* ctl,
    bool flip,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_round_prep(
    tk_ctx* ctx,
    const int32_t* rctl,
    int32_t* tok,
    int32_t* tok_m,
    int32_t* pos0,
    int32_t* pos0_m,
    int32_t* gctl,
    int32_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_row_copy(
    tk_ctx* ctx,
    float* dst,
    const float* src,
    const int32_t* idx,
    int32_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_chain_prep(
    tk_ctx* ctx,
    const int32_t* tok,
    int32_t* tok_m,
    const int32_t* pos0,
    int32_t* pos0_m,
    int32_t k,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_sgemm_nt(
    tk_ctx* ctx,
    int32_t tm,
    int32_t grid_x,
    int32_t grid_y,
    int32_t grid_z,
    const float* w,
    const float* x,
    float* out,
    int32_t k_dim,
    int32_t n_out,
    int32_t n_tok,
    int32_t ldo,
    int32_t kz,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_sum_splits(
    tk_ctx* ctx,
    const float* part,
    float* out,
    int64_t n,
    int32_t splits,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_to_f16(
    tk_ctx* ctx,
    const float* x,
    void* y_f16,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_conv_seq(
    tk_ctx* ctx,
    float* hist,
    const float* qkv,
    const float* w,
    float* h,
    int32_t c_channels,
    int32_t qk_heads,
    float eps,
    int32_t n_tok,
    bool write_state,
    const tk_q36_gdn_pend* pd,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_conv_ab(
    tk_ctx* ctx,
    float* hist,
    const float* qkv,
    const float* w,
    float* hc,
    int32_t c_channels,
    int32_t qk_heads,
    float eps,
    int32_t n_tok,
    bool write_state,
    const tk_q36_gdn_pend* pd,
    const float* wa,
    const float* wb,
    const float* x,
    const float* dt,
    const float* ssm_a,
    float* alpha,
    float* beta,
    int32_t n_in,
    int32_t hv,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_conv_par(
    tk_ctx* ctx,
    const float* hist,
    const float* qkv,
    const float* w,
    float* h,
    int32_t c_channels,
    int32_t qk_heads,
    float eps,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_conv_hist(
    tk_ctx* ctx,
    float* hist,
    const float* qkv,
    int32_t c_channels,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_seq(
    tk_ctx* ctx,
    float* state,
    const float* h,
    int32_t c_channels,
    const float* gate,
    const float* beta,
    const float* z,
    const float* gamma,
    float eps,
    float* y,
    int32_t h_k,
    int32_t h_v,
    int32_t n_tok,
    bool write_state,
    const tk_q36_gdn_pend* pd,
    void* yq,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_seq_cols(
    tk_ctx* ctx,
    float* state,
    const float* h,
    int32_t c_channels,
    const float* gate,
    const float* beta,
    float* o_raw,
    int32_t h_k,
    int32_t h_v,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_seq_warp(
    tk_ctx* ctx,
    int32_t ncol,
    float* state,
    const float* h,
    int32_t c_channels,
    const float* gate,
    const float* beta,
    float* o_raw,
    int32_t h_k,
    int32_t h_v,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_gdn_norm(
    tk_ctx* ctx,
    int32_t n_tok,
    const float* o_raw,
    const float* z,
    const float* gamma,
    float eps,
    float* y,
    int32_t h_v,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_flush_read(
    tk_ctx* ctx,
    const void* p,
    size_t n,
    float* out,
    tk_stream stream) TK_NOEXCEPT;

/* Adaptive Draft Length (q36_adraft) */
tk_status tk_q36_draft_conf(
    tk_ctx* ctx,
    const float* x,
    int32_t n,
    const float* tau,
    const float* lmin,
    const int32_t* pick,
    const int32_t* map,
    float* conf,
    float* top,
    int32_t step,
    const float* lse,
    const float* thr,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_draft_support(
    tk_ctx* ctx,
    const float* x,
    int32_t n,
    const float* tau,
    const float* lmin,
    const float* top_p,
    const int32_t* top_k,
    float* lse,
    float* thr,
    tk_stream stream) TK_NOEXCEPT;

/* DFlash2 Block-Diffusion Drafter (q36_dflash) */
int32_t tk_q36_dflash_attn_chunks(int32_t window) TK_NOEXCEPT;
size_t tk_q36_dflash_attn_scratch_floats(int32_t heads, int32_t hd, int32_t window) TK_NOEXCEPT;

tk_status tk_q36_dflash_rms_rows(
    tk_ctx* ctx,
    const float* x,
    const float* w,
    float* out,
    int32_t n,
    int32_t rows,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_rms_rows_q8(
    tk_ctx* ctx,
    const float* x,
    const float* w,
    float* out,
    void* q8,
    int32_t n,
    int32_t rows,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_sum5_rms(
    tk_ctx* ctx,
    const float* const* ys,
    const float* w,
    float* out,
    int32_t n,
    int32_t rows,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_sum5_rms_q8(
    tk_ctx* ctx,
    const float* const* ys,
    const float* w,
    float* out,
    void* q8,
    int32_t n,
    int32_t rows,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_dyn_conv(
    tk_ctx* ctx,
    const float* x_in,
    const float* dyn,
    const float* base,
    float* out,
    int32_t n,
    int32_t rows,
    int32_t group,
    int32_t side,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_dyn_conv_q8(
    tk_ctx* ctx,
    const float* x_in,
    const float* dyn,
    const float* base,
    float* out,
    void* q8,
    int32_t n,
    int32_t rows,
    int32_t group,
    int32_t side,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_qk_norm_rope(
    tk_ctx* ctx,
    float* x,
    const float* w,
    int32_t heads,
    int32_t hd,
    int32_t rows,
    const int32_t* pos,
    float theta,
    float eps,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_kv_store(
    tk_ctx* ctx,
    const float* k,
    const float* v,
    float* kring,
    float* vring,
    int32_t width,
    int32_t rows,
    const int32_t* pos,
    int32_t ring,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_kv_store_f16(
    tk_ctx* ctx,
    const float* k,
    const float* v,
    void* kring_f16,
    void* vring_f16,
    int32_t k_width,
    int32_t v_width,
    int32_t rows,
    const int32_t* pos,
    int32_t ring,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_block_attn(
    tk_ctx* ctx,
    const float* q,
    const float* kring,
    const float* vring,
    const int32_t* pos,
    int32_t heads,
    int32_t heads_kv,
    int32_t hd,
    int32_t ring,
    int32_t window,
    float scale,
    float* part,
    float* out,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_swiglu(
    tk_ctx* ctx,
    const float* g,
    const float* u,
    float* out,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_swiglu_q8(
    tk_ctx* ctx,
    const float* g,
    const float* u,
    float* out,
    void* q8,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_add_inplace(
    tk_ctx* ctx,
    float* x,
    const float* y,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_block_tokens(
    tk_ctx* ctx,
    const int32_t* anchor,
    int32_t* out,
    int32_t n,
    int32_t mask,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_set_i32(
    tk_ctx* ctx,
    int32_t* p,
    int32_t v,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_top16(
    tk_ctx* ctx,
    const float* logits,
    int32_t nv,
    int32_t rows,
    const int32_t* map,
    int32_t* cand,
    float* unary,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_select_chain(
    tk_ctx* ctx,
    const int32_t* cand,
    const float* unary,
    const float* gate,
    const void* pred,
    const void* succ,
    const int32_t* anchor,
    int32_t* out,
    float* score,
    int32_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_dflash_select_chain_q(
    tk_ctx* ctx,
    const int32_t* cand,
    const float* unary,
    const float* gate,
    const void* pred,
    const void* succ,
    const int32_t* anchor,
    int32_t* out,
    float* score,
    float* q,
    const float* tau,
    const uint32_t* seed,
    const uint32_t* ctr,
    int32_t n,
    tk_stream stream) TK_NOEXCEPT;

/* Additional q36 Engine / Prefill / MoE / Embedding Helpers */
bool tk_q36_mmvq_supported(int32_t ggml_type) TK_NOEXCEPT;
bool tk_q36_q8_0_pack_should(int32_t ggml_type, int32_t n_in, int32_t n_out) TK_NOEXCEPT;
void tk_q36_q8_0_pack_host(const void* gguf_blocks, void* out, int32_t n_in, int32_t n_out) TK_NOEXCEPT;
void tk_q36_q8_0_packed_register(const void* gguf_weights, const void* packed, int32_t n_in, int32_t n_out) TK_NOEXCEPT;
void tk_q36_q8_0_packed_unregister(const void* gguf_weights) TK_NOEXCEPT;

tk_status tk_q36_embed_rows(
    tk_ctx* ctx,
    int32_t ggml_type,
    const void* table,
    size_t row_bytes,
    const int32_t* tok_dev,
    int64_t n_tok,
    int64_t n_embd,
    float* out,
    tk_stream stream) TK_NOEXCEPT;

bool tk_q36_mmq_built(void) TK_NOEXCEPT;
bool tk_q36_mmq_supported_fits(int32_t ggml_type, int64_t w_rows) TK_NOEXCEPT;
int32_t tk_q36_mmq_q8_layout(int32_t ggml_type) TK_NOEXCEPT;
size_t tk_q36_mmq_q8_bytes(int64_t rows, int64_t cols) TK_NOEXCEPT;
size_t tk_q36_mmq_matrix_bytes(int32_t ggml_type, int64_t rows, int64_t cols) TK_NOEXCEPT;

tk_status tk_q36_mmq_iota(
    tk_ctx* ctx,
    int32_t* dst,
    int64_t n,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_mmq_set_bounds(
    tk_ctx* ctx,
    int32_t* bounds,
    int32_t rows,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_mmq_quantize(
    tk_ctx* ctx,
    const float* x,
    const int32_t* ids,
    void* xq,
    int32_t ggml_type,
    int64_t cols,
    int64_t ld,
    int64_t rows,
    tk_stream stream) TK_NOEXCEPT;

/** bitwise tk_q36_mmq_quantize(x, rtok, ...) of the routed rows, with each of the n_tok token rows of x
 * (`cols` floats apart) quantized once and written to its k routed rows smap[t * k + s] (tk_route_group's smap). */
tk_status tk_q36_mmq_quantize_scatter(
    tk_ctx* ctx,
    const float* x,
    const int32_t* smap,
    void* xq,
    int32_t ggml_type,
    int64_t cols,
    int64_t n_tok,
    int64_t n_rows,
    int32_t k,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_mmq_swiglu(
    tk_ctx* ctx,
    const float* gu,
    float* h,
    int64_t rows,
    int64_t n_ff,
    bool interleaved,
    tk_stream stream) TK_NOEXCEPT;

bool tk_q36_fattn_supported(int32_t head_dim) TK_NOEXCEPT;
size_t tk_q36_fattn_mask_bytes(int64_t n_tok, int64_t n_kv) TK_NOEXCEPT;

tk_status tk_q36_prefill_init_scratch(
    tk_ctx* ctx,
    int64_t scratch_elems) TK_NOEXCEPT;

tk_status tk_q36_attn_blas_prefill(
    tk_ctx* ctx,
    const float* q,
    void* q16,
    const void* kc,
    const void* vc,
    float* att_s,
    void* att_p16,
    float* att,
    const float* gate,
    int32_t T,
    int32_t H,
    int32_t HKV,
    int32_t HD,
    int32_t ctx_len,
    int64_t p0,
    size_t score_floats,
    tk_stream stream) TK_NOEXCEPT;

tk_status tk_q36_router_topk(
    tk_ctx* ctx,
    const float* logits,
    int32_t* ids,
    float* weights,
    int32_t n_expert,
    int32_t k,
    int32_t n_tok,
    tk_stream stream) TK_NOEXCEPT;

bool tk_q36_expert_supported(int32_t gu_type, int32_t d_type, int64_t n_embd, int64_t n_ff) TK_NOEXCEPT;

void tk_q36_expert_layout(
    int32_t gu_type,
    int32_t d_type,
    int64_t n_embd,
    int64_t n_ff,
    size_t* out_gu_row,
    size_t* out_d_row,
    size_t* out_up_off,
    size_t* out_down_off,
    size_t* out_bytes,
    size_t* out_tail_off) TK_NOEXCEPT;

size_t tk_q36_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff) TK_NOEXCEPT;

tk_status tk_q36_expert_grouped(
    tk_ctx* ctx,
    int32_t gu_type,
    int32_t d_type,
    int64_t n_embd,
    int64_t n_ff,
    const uint64_t* grp_ptr,
    const int32_t* grp_start,
    const int32_t* n_groups,
    const int32_t* ent_dst,
    const int32_t* ent_tok,
    int64_t cap_groups,
    int64_t cap_entries,
    const void* x_q8_1,
    void* scratch,
    float* out,
    int64_t grid_groups,
    tk_stream stream) TK_NOEXCEPT;

/* ==========================================================================
 * 10. Talus prefill Prompt Attention, Expanded INT8 & W4A4 Entry Points
 * ========================================================================== */

int talus_pf_fa_prefill(
    const float* q,
    const void* k,
    const void* vt,
    const float* gate,
    float* out,
    int32_t T,
    int32_t H,
    int32_t HKV,
    int32_t D,
    int64_t kv_stride,
    int64_t vt_cells,
    int64_t p0,
    float scale,
    void* stream) TK_NOEXCEPT;

int talus_pf_i8x_expand(
    const void* w,
    int32_t ggml_type,
    int32_t n,
    int32_t k,
    int8_t* qs,
    float* ws,
    void* stream) TK_NOEXCEPT;

int talus_pf_i8gemm(
    const int8_t* qs,
    const float* ws,
    int32_t n,
    int32_t k,
    int32_t mode,
    const void* xq,
    int32_t T,
    float* dst,
    int64_t ld_dst,
    void* stream) TK_NOEXCEPT;

/**
 * grouped expert GEMM (talus::prefill::moe_gemm, src/prefill/tp_moe.cu).  The contract of
 * tk_prefill_moe_mmq (expert e's sorted MMQ-layout q8_1 rows [bounds[e], bounds[e+1]) of xq times its [n][k] GGUF matrix
 * at w + e * expert_bytes, written to dst rows ids[row]) with bitwise-identical output; tk_prefill_moe_mmq already takes
 * this kernel when it applies (TALUS_PF_MOE=0 disables).  Types IQ4_XS IQ4_NL IQ3_S IQ3_XXS IQ2_XXS Q8_0 Q5_0 IQ2_XS
 * IQ2_S Q3_K Q4_K Q5_K; n % 128 == 0, k % 256 == 0; gfx11.  0 on success, -1 unsupported (nothing launched).
 */
int talus_pf_moe_gemm(
    const void* w,
    int32_t ggml_type,
    size_t expert_bytes,
    int32_t n,
    int32_t k,
    int32_t n_experts,
    const void* xq,
    const int32_t* bounds,
    const int32_t* ids,
    int64_t total_rows,
    int64_t max_rows,
    float* dst,
    int64_t ld_dst,
    void* stream) TK_NOEXCEPT;

int talus_pf_swiglu_q8_mmq(
    const float* g,
    const float* u,
    void* xq,
    int32_t ggml_type,
    int64_t cols,
    int64_t rows,
    void* stream) TK_NOEXCEPT;

/**
 * The up projection on its expanded matrix (talus_pf_i8gemm's operands) with the SwiGLU fused into the epilogue: writes
 * yq = MMQ q8_1 blocks of silu(g) * u (layout = mmq_q8_1_ds_layout of the down weights: 0 D4, 1 DS4), bitwise
 * talus_pf_i8gemm into u followed by talus_pf_swiglu_q8_mmq(g, u, yq).  g: the gate projection's FP32 output
 * [T][ldg].  yq must not alias xq.  0 launched, -1 not covered (n % 512, k % 256, layout D2S6, non-gfx11,
 * TALUS_PF_SWQ8=0): nothing launched, the caller runs the separate path.
 */
int talus_pf_i8gemm_swiglu_q8(
    const int8_t* qs,
    const float* ws,
    int32_t n,
    int32_t k,
    int32_t mode,
    const void* xq,
    int32_t T,
    const float* g,
    int64_t ldg,
    void* yq,
    int32_t layout,
    void* stream) TK_NOEXCEPT;

int talus_pf_w4_pack(
    const float* x,
    int32_t T,
    int32_t k,
    int64_t ldx,
    const int8_t* signs,
    uint8_t* xq,
    float* xs,
    void* stream) TK_NOEXCEPT;

int talus_pf_w4_gemm(
    const uint8_t* wq,
    const float* ws,
    int32_t n,
    int32_t k,
    const uint8_t* xq,
    const float* xs,
    int32_t T,
    float* dst,
    int64_t ld_dst,
    void* stream) TK_NOEXCEPT;

/* ==========================================================================
 * 11. QSA tile attention (fnp_qsa_tile_attn) Prompt Tile Attention Entry Points
 * ========================================================================== */

int fnp_qsa_tile_attn_i8(
    const float* q,
    const int8_t* k_q,
    const int8_t* v_q,
    const uint16_t* k_scale,
    const uint16_t* v_scale,
    const int32_t* page_table,
    int64_t page_size,
    const int32_t* ids,
    const int32_t* steps,
    int64_t cap,
    float* attn,
    int64_t n_q,
    void* scratch,
    void* stream);

uint64_t fnp_qsa_tile_scratch_bytes_c(
    int64_t n_q,
    int64_t cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TALUS_KERNELS_H */
