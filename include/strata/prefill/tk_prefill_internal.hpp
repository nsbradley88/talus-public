// include/strata/prefill/tk_prefill_internal.hpp - Internal prefill state & tuning helpers for Contract v0 (WP-1.1b).
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#pragma once

#include "talus_kernels.h"

#if __has_include("hipblaslt_tuning.hpp")
#include "hipblaslt_tuning.hpp"
#else
#include "../../../src/prefill/hipblaslt_tuning.hpp"
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>

namespace strata::prefill::tk {

/// Canonical GCN architecture token for a `tk_arch` enumerator.
inline const char* arch_name(tk_arch arch) noexcept {
    switch (arch) {
        case TK_ARCH_GFX1151:
            return "gfx1151";
        case TK_ARCH_GFX1201:
            return "gfx1201";
        case TK_ARCH_OTHER:
        default:
            return "";
    }
}

/// Strip target feature suffixes (e.g. "gfx1151:sramecc-:xnack-" -> "gfx1151").
inline std::string strip_gcn_arch_suffix(std::string gcn_arch) {
    const auto colon = gcn_arch.find(':');
    if (colon != std::string::npos) {
        gcn_arch.resize(colon);
    }
    return gcn_arch;
}

/// Map a GCN architecture name (with or without `:feature` suffixes) to `tk_arch`.
inline tk_arch arch_from_gcn_name(const std::string& gcn_arch) noexcept {
    const std::string base = strip_gcn_arch_suffix(gcn_arch);
    if (base == "gfx1151") {
        return TK_ARCH_GFX1151;
    }
    if (base == "gfx1201" || base == "gfx1200") {
        return TK_ARCH_GFX1201;
    }
    return TK_ARCH_OTHER;
}

/// Construct the commit 5c7e9867 version-sibling tuning path `<dir>/<arch>-hipblaslt-<runtime_version>.txt`
/// next to `path`.
inline std::string sibling_tuning_path(const std::string& path, const std::string& arch, int runtime_version) {
    const std::string base_arch = strip_gcn_arch_suffix(arch);
    const auto slash = path.find_last_of("/\\");
    const std::string dir = (slash == std::string::npos) ? std::string() : path.substr(0, slash + 1);
    return dir + base_arch + "-hipblaslt-" + std::to_string(runtime_version) + ".txt";
}

/// Result of resolving and loading a per-arch hipBLASLt tuning table.
struct TuningLoadResult {
    tk_status status = TK_OK;
    std::string resolved_path;
    std::string error;
    bool used_sibling_fallback = false;
};

/// Resolve and load a hipBLASLt tuning table for `(runtime_arch, runtime_version)` from `dir_or_file`,
/// preserving the commit 5c7e9867 runtime version-sibling fallback (`<arch>-hipblaslt-<runtime_version>.txt`).
///
/// Resolution order:
/// 1. If `dir_or_file` is NULL or empty, consult `STRATA_HIPBLASLT_TUNING`, falling back to `"tools/hip"`.
/// 2. If the target path is a directory, load `<dir>/<arch>-hipblaslt-<runtime_version>.txt`.
/// 3. If the target path is a file, first attempt `table.load(path, arch, runtime_version)`.
/// 4. If that fails (e.g. `gfx1151-hipblaslt-100401.txt` on runtime `100500`), attempt the sibling file
///    `<dir_of_path>/<arch>-hipblaslt-<runtime_version>.txt` in the same directory.
inline TuningLoadResult load_tuning_with_fallback(
    strata::prefill::hipblaslt::TuningTable& out_table,
    const char* dir_or_file,
    const std::string& runtime_arch,
    int runtime_version) {
    const std::string arch = strip_gcn_arch_suffix(runtime_arch);
    if (arch.empty() || runtime_version <= 0) {
        return {
            TK_ERR_INVALID_ARG,
            {},
            "invalid runtime architecture ('" + arch + "') or hipBLASLt version (" +
                std::to_string(runtime_version) + ")",
            false,
        };
    }

    std::string input_path;
    if (dir_or_file != nullptr && dir_or_file[0] != '\0') {
        input_path = dir_or_file;
    } else {
        const char* env_path = std::getenv("STRATA_HIPBLASLT_TUNING");
        if (env_path != nullptr && env_path[0] != '\0') {
            input_path = env_path;
        } else {
            input_path = "tools/hip";
        }
    }

    std::error_code ec;
    const bool is_dir = std::filesystem::is_directory(input_path, ec) ||
                        (!input_path.empty() && (input_path.back() == '/' || input_path.back() == '\\'));

    if (is_dir) {
        std::string dir = input_path;
        if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') {
            dir.push_back('/');
        }
        const std::string candidate = dir + arch + "-hipblaslt-" + std::to_string(runtime_version) + ".txt";
        strata::prefill::hipblaslt::TuningTable table;
        std::string err;
        if (table.load(candidate, arch, runtime_version, err)) {
            out_table = std::move(table);
            return {TK_OK, candidate, {}, false};
        }
        const tk_status code = (err.rfind("cannot open", 0) == 0) ? TK_ERR_IO : TK_ERR_INVALID_ARG;
        return {code, {}, "failed to load tuning table '" + candidate + "': " + err, false};
    }

    strata::prefill::hipblaslt::TuningTable table;
    std::string err1;
    if (table.load(input_path, arch, runtime_version, err1)) {
        out_table = std::move(table);
        return {TK_OK, input_path, {}, false};
    }

    // Commit 5c7e9867 version-sibling table fallback:
    // Calibrations sit side by side as <arch>-hipblaslt-<version>.txt, so when a config names
    // another runtime's table (e.g. 100401 on ROCm 10.2 runtime 100500), resolve the sibling
    // table for the active runtime version in the same directory.
    const std::string sibling = sibling_tuning_path(input_path, arch, runtime_version);
    if (sibling != input_path) {
        strata::prefill::hipblaslt::TuningTable sibling_table;
        std::string err2;
        if (sibling_table.load(sibling, arch, runtime_version, err2)) {
            out_table = std::move(sibling_table);
            return {TK_OK, sibling, {}, true};
        }
        const tk_status code =
            (err1.rfind("cannot open", 0) == 0 && err2.rfind("cannot open", 0) == 0)
                ? TK_ERR_IO
                : TK_ERR_INVALID_ARG;
        return {
            code,
            {},
            "failed to load tuning table '" + input_path + "' (" + err1 + ") and version-sibling '" +
                sibling + "' (" + err2 + ")",
            false,
        };
    }

    const tk_status code = (err1.rfind("cannot open", 0) == 0) ? TK_ERR_IO : TK_ERR_INVALID_ARG;
    return {code, {}, "failed to load tuning table '" + input_path + "': " + err1, false};
}

}  // namespace strata::prefill::tk

/// Stateful per-device prefill context holding hipBLAS/hipBLASLt handles, workspace,
/// dequantization scratch, MMQ context, algo cache, and per-arch tuning table metadata.
struct tk_prefill_state {
    int32_t device_id = 0;
    tk_arch arch = TK_ARCH_GFX1151;
    std::string gcn_arch_name = "gfx1151";
    int hipblaslt_version = 100500;
    bool has_gpu = false;

    void* hipblas_handle = nullptr;
    void* hipblaslt_handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;

    uint16_t* dequant_scratch = nullptr;
    int64_t dequant_scratch_elems = 0;

    void* mmq_ctx = nullptr;
    void* mmq_dense_buf = nullptr;
    int64_t mmq_dense_cap_rows = 0;
    bool mmq_dense_failed = false;
    bool f16_io = false;

    void* algo_cache = nullptr;
    uint64_t lt_launches = 0;
    uint64_t lt_fallbacks = 0;

    strata::prefill::hipblaslt::TuningTable tuning_table;
    bool tuning_loaded = false;
    bool used_sibling_fallback = false;
    std::string tuning_path;
    std::string last_error;

    /// Load a tuning table for `(gcn_arch_name, hipblaslt_version)` from `dir_or_file`
    /// with the commit 5c7e9867 version-sibling fallback.
    tk_status load_tuning_host(const char* dir_or_file) {
        const std::string effective_arch =
            !gcn_arch_name.empty() ? gcn_arch_name : std::string(strata::prefill::tk::arch_name(arch));
        auto res = strata::prefill::tk::load_tuning_with_fallback(
            tuning_table, dir_or_file, effective_arch, hipblaslt_version);
        if (res.status == TK_OK) {
            tuning_loaded = true;
            used_sibling_fallback = res.used_sibling_fallback;
            tuning_path = std::move(res.resolved_path);
            last_error.clear();
        } else {
            last_error = std::move(res.error);
        }
        return res.status;
    }
};

#ifdef __cplusplus
extern "C" {
#endif

/** Initialize the prefill state associated with `ctx` (hipBLAS, hipBLASLt, workspace, MMQ context). */
tk_status tk_prefill_ctx_init(tk_ctx* ctx) TK_NOEXCEPT;

/** Destroy and release all prefill state associated with `ctx` (no-op if `ctx == NULL`). */
void tk_prefill_ctx_destroy(tk_ctx* ctx) TK_NOEXCEPT;

/**
 * Load a per-architecture hipBLASLt tuning table into `ctx`'s prefill state from `dir_or_file`.
 * Preserves the commit 5c7e9867 version-sibling fallback (`<arch>-hipblaslt-<runtime_ver>.txt`).
 */
tk_status tk_prefill_ctx_load_tuning(tk_ctx* ctx, const char* dir_or_file) TK_NOEXCEPT;

/** Return the last error message recorded by the prefill subsystem for `ctx` (or thread-local if `ctx == NULL`). */
const char* tk_prefill_last_error(const tk_ctx* ctx) TK_NOEXCEPT;

#ifdef __cplusplus
}  // extern "C"
#endif

/// Return the mutable `tk_prefill_state` associated with `ctx`, or `nullptr` if none is registered.
tk_prefill_state* tk_prefill_get_state(tk_ctx* ctx) noexcept;

/// Return the const `tk_prefill_state` associated with `ctx`, or `nullptr` if none is registered.
const tk_prefill_state* tk_prefill_get_state_const(const tk_ctx* ctx) noexcept;
