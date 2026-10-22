// include/strata/kernels/tk_internal.hpp - Shared internal C++ structures for libtalus_kernels (Contract v0, WP-1.1a).
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#pragma once

#include "talus_kernels.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

/** Pack `(tk_qtype, ncols)` into a 64-bit lookup key for the per-arch MMVQ rows table. */
inline constexpr uint64_t tk_mmvq_table_key(tk_qtype qtype, int32_t ncols) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(qtype)) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(ncols));
}

/**
 * Default wave32 MMVQ rows-per-workgroup tuning table for `(arch, qtype, ncols)`.
 * Transcribed from `w32_rows` in `src/kernels/cuda/native_mmvq.cu`.
 */
inline constexpr int32_t tk_default_mmvq_rows(tk_arch arch, tk_qtype qtype, int32_t ncols) noexcept {
    if (ncols <= 0 || ncols > 8) {
        return 1;
    }
    if (arch != TK_ARCH_GFX1151 && arch != TK_ARCH_GFX1201) {
        return 1;
    }
    if (ncols == 1) {
        switch (qtype) {
            case TK_QTYPE_Q2_K:
            case TK_QTYPE_IQ2_XS:
            case TK_QTYPE_IQ2_S:
                return 2;
            case TK_QTYPE_Q3_K:
                return 4;
            default:
                return 1;
        }
    }
    switch (qtype) {
        case TK_QTYPE_Q4_K:
        case TK_QTYPE_Q5_K:
        case TK_QTYPE_Q6_K:
            return 2;
        case TK_QTYPE_IQ3_S:
        case TK_QTYPE_Q3_K:
        case TK_QTYPE_Q2_K:
            return 4;
        default:
            return 1;
    }
}

/**
 * Default split-K decode attention chunk cells for `(arch, kv_len)`.
 * For `TK_ARCH_GFX1151`: `128` below 4K (`kv_len < 4096`), `512` at or above 4K.
 */
inline constexpr int32_t tk_default_attn_chunk_cells(tk_arch arch, int32_t kv_len) noexcept {
    (void)arch;
    return (kv_len < 4096) ? 128 : 512;
}

/**
 * Internal configuration dictionary and per-architecture tuning tables (`tk_config`).
 */
struct tk_config {
    tk_arch arch = TK_ARCH_GFX1151;
    std::unordered_map<std::string, int> int_knobs;
    std::unordered_map<uint64_t, int32_t> mmvq_rows_table;
    int32_t attn_chunk_below_4k = 128;
    int32_t attn_chunk_4k_boundary = 4096;
    int32_t attn_chunk_above_4k = 512;

    int32_t mmvq_rows(tk_qtype qtype, int32_t ncols) const noexcept {
        if (ncols <= 0) {
            return 1;
        }
        const auto it = mmvq_rows_table.find(tk_mmvq_table_key(qtype, ncols));
        if (it != mmvq_rows_table.end()) {
            return it->second;
        }
        return tk_default_mmvq_rows(arch, qtype, ncols);
    }

    int32_t attn_chunk_cells(int32_t kv_len) const noexcept {
        return (kv_len < attn_chunk_4k_boundary) ? attn_chunk_below_4k : attn_chunk_above_4k;
    }
};

/**
 * Populate `cfg` with the per-architecture default knobs (from `src/core/arch_defaults.cpp`)
 * and per-architecture MMVQ / attention tuning tables.
 */
inline void tk_populate_arch_defaults(tk_config& cfg, tk_arch arch) {
    cfg.arch = arch;
    cfg.int_knobs.clear();
    cfg.mmvq_rows_table.clear();
    cfg.attn_chunk_below_4k = 128;
    cfg.attn_chunk_4k_boundary = 4096;
    cfg.attn_chunk_above_4k = 512;

    if (arch == TK_ARCH_GFX1151) {
        // 7 prompt switches from src/core/arch_defaults.cpp (kGfx1151Prompt)
        cfg.int_knobs["STRATA_GDN_HEAD"] = 1;
        cfg.int_knobs["STRATA_GDN_PP"] = 3;   // gdn_rec_dpp2c_kernel (same bits as 2)
        cfg.int_knobs["STRATA_GDN_CONVL2"] = 1;
        cfg.int_knobs["STRATA_GDN_NOY"] = 1;
        cfg.int_knobs["STRATA_CVEC_FUSE"] = 1;
        cfg.int_knobs["STRATA_HCD_EXACT"] = 1;
        cfg.int_knobs["STRATA_PF_PAD"] = 1;

        // 11 decode switches from src/core/arch_defaults.cpp (kGfx1151Decode)
        cfg.int_knobs["STRATA_Q8_PACKED"] = 1;
        cfg.int_knobs["STRATA_Q6_PACKED"] = 1;
        cfg.int_knobs["STRATA_MMVF_ROWS"] = 1;
        cfg.int_knobs["STRATA_ATTN_LANECELL"] = 1;
        cfg.int_knobs["STRATA_EXPERT_V2"] = 1;
        cfg.int_knobs["STRATA_TSUM"] = 1;
        cfg.int_knobs["STRATA_LFUSE"] = 1;
        cfg.int_knobs["STRATA_GDN_SPLIT"] = 1;
        cfg.int_knobs["STRATA_QFUSE"] = 1;
        cfg.int_knobs["STRATA_PLE_BATCH"] = 1;
        cfg.int_knobs["STRATA_SH_STREAM"] = 1;
    } else if (arch == TK_ARCH_GFX1201) {
        // 5 prompt switches from src/core/arch_defaults.cpp (kGfx12Prompt)
        cfg.int_knobs["STRATA_GDN_HEAD"] = 1;
        cfg.int_knobs["STRATA_GDN_PP"] = 2;
        cfg.int_knobs["STRATA_GDN_CONVL2"] = 1;
        cfg.int_knobs["STRATA_GDN_NOY"] = 1;
        cfg.int_knobs["STRATA_CVEC_FUSE"] = 1;
    }

    static constexpr tk_qtype kAllQtypes[] = {
        TK_QTYPE_F32,     TK_QTYPE_F16,     TK_QTYPE_Q4_0,    TK_QTYPE_Q4_1,
        TK_QTYPE_Q5_0,    TK_QTYPE_Q5_1,    TK_QTYPE_Q8_0,    TK_QTYPE_Q8_1,
        TK_QTYPE_Q2_K,    TK_QTYPE_Q3_K,    TK_QTYPE_Q4_K,    TK_QTYPE_Q5_K,
        TK_QTYPE_Q6_K,    TK_QTYPE_Q8_K,    TK_QTYPE_IQ2_XXS, TK_QTYPE_IQ2_XS,
        TK_QTYPE_IQ3_XXS, TK_QTYPE_IQ1_S,   TK_QTYPE_IQ4_NL,  TK_QTYPE_IQ3_S,
        TK_QTYPE_IQ2_S,   TK_QTYPE_IQ4_XS,  TK_QTYPE_BF16,    TK_QTYPE_Q2_0,
    };
    for (tk_qtype qt : kAllQtypes) {
        for (int32_t ncols = 1; ncols <= 16; ++ncols) {
            cfg.mmvq_rows_table[tk_mmvq_table_key(qt, ncols)] = tk_default_mmvq_rows(arch, qt, ncols);
        }
    }
}

/**
 * Opaque per-device execution context (`tk_ctx`).
 */
struct tk_ctx {
    int device = 0;
    tk_arch arch = TK_ARCH_GFX1151;
    tk_config config{};
    mutable std::mutex err_mu;
    mutable std::string last_error;
    void* prefill_state = nullptr;
    bool has_gpu = false;

    tk_ctx() {
        tk_populate_arch_defaults(config, TK_ARCH_GFX1151);
    }
};

/**
 * Opaque handle for pre-packed Q8_0 or Q6_K weight matrices (`tk_packed_weights`).
 */
struct tk_packed_weights {
    tk_qtype qtype = TK_QTYPE_Q8_0;
    int n_in = 0;
    int n_out = 0;
    void* dev_ptr = nullptr;
    size_t byte_size = 0;
    int device = 0;
    const void* mmvq_key = nullptr;
    bool host_allocated = false;
    bool registered_q8 = false;
    bool registered_q6 = false;
};

/**
 * Thread-local fallback error string used when `ctx == nullptr` or for returning a stable `const char*`.
 */
inline std::string& tk_thread_last_error() noexcept {
    thread_local std::string g_tk_last_error;
    return g_tk_last_error;
}

/**
 * Record an error message on `ctx` (if non-null) and in the thread-local fallback buffer.
 */
inline void tk_set_error(const tk_ctx* ctx, const char* msg) noexcept {
    try {
        const char* safe_msg = (msg != nullptr) ? msg : "";
        tk_thread_last_error() = safe_msg;
        if (ctx != nullptr) {
            std::lock_guard<std::mutex> lock(ctx->err_mu);
            ctx->last_error = safe_msg;
        }
    } catch (...) {
        // Never allow an exception to escape error recording.
    }
}

/**
 * Clear the recorded error on `ctx` (if non-null) and in the thread-local fallback buffer.
 */
inline void tk_clear_error(const tk_ctx* ctx) noexcept {
    tk_set_error(ctx, "");
}

/**
 * Exception-catching FFI boundary macros.
 */
#define TK_TRY_BEGIN try {

#define TK_TRY_END(ctx)                                              \
    } catch (const std::bad_alloc& e) {                              \
        tk_set_error((ctx), e.what());                               \
        return TK_ERR_OUT_OF_MEMORY;                                 \
    } catch (const std::invalid_argument& e) {                       \
        tk_set_error((ctx), e.what());                               \
        return TK_ERR_INVALID_ARG;                                   \
    } catch (const std::exception& e) {                              \
        tk_set_error((ctx), e.what());                               \
        return TK_ERR_EXCEPTION;                                     \
    } catch (...) {                                                  \
        tk_set_error((ctx), "unhandled non-std C++ exception");      \
        return TK_ERR_EXCEPTION;                                     \
    }

#ifdef __cplusplus
extern "C" {
#endif

/** Prefill lifecycle hooks (weakly defined in `talus_kernels_decode.cpp`, overridden by WP-1.1b). */
tk_status tk_prefill_ctx_init(tk_ctx* ctx) TK_NOEXCEPT;
void tk_prefill_ctx_destroy(tk_ctx* ctx) TK_NOEXCEPT;
tk_status tk_prefill_ctx_load_tuning(tk_ctx* ctx, const char* dir) TK_NOEXCEPT;
const char* tk_prefill_last_error(const tk_ctx* ctx) TK_NOEXCEPT;
void tk_ctx_set_error(tk_ctx* ctx, const char* msg) TK_NOEXCEPT;
void tk_set_last_error(tk_ctx* ctx, const char* msg) TK_NOEXCEPT;

/** Return the compiled `TALUS_KERNELS_ABI_VERSION` (`0`). */
uint32_t tk_abi_version(void) TK_NOEXCEPT;

/** Query the wave32 MMVQ rows-per-workgroup tuning table for `(arch, qtype, ncols)`. */
int32_t tk_mmvq_rows_for_arch(tk_arch arch, tk_qtype qtype, int32_t ncols) TK_NOEXCEPT;

/** Query the decode attention split-K chunk cell size for `(arch, kv_len)`. */
int32_t tk_attn_chunk_cells_for_arch(tk_arch arch, int32_t kv_len) TK_NOEXCEPT;

/** Grouped RMS normalization (`x_dev` `[n_rows, dim]`, `w_dev` `[n_rows * dim]`). */
tk_status tk_gr_norm(
    tk_ctx*       ctx,
    const float*  x_dev,
    const float*  w_dev,
    float*        dst_dev,
    int32_t       n_rows,
    int32_t       dim,
    int32_t       groups,
    float         eps,
    tk_stream     stream) TK_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif
