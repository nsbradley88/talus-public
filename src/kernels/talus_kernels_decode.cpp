// src/kernels/talus_kernels_decode.cpp - Strata decode kernels C-ABI shim (Contract v0, WP-1.1a).
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT

#include "strata/kernels/tk_internal.hpp"

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/gdn.hpp"
#include "strata/kernels/mimo_kernels.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"

#if __has_include(<hip/hip_runtime_api.h>)
#include <hip/hip_runtime_api.h>
#endif

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

extern char** environ;

namespace {

constexpr int32_t kMaxStubDeviceId = 64;

inline void* tk_effective_stream(tk_stream stream) noexcept {
    if (stream != nullptr) {
        return stream;
    }
#if defined(hipStreamLegacy)
    return reinterpret_cast<void*>(hipStreamLegacy);
#else
    return reinterpret_cast<void*>(static_cast<uintptr_t>(1));
#endif
}

inline tk_status tk_null_err(const tk_ctx* ctx, const char* msg) noexcept {
    tk_set_error(ctx, msg);
    return TK_ERR_NULL_PTR;
}

inline tk_status tk_arg_err(const tk_ctx* ctx, const char* msg) noexcept {
    tk_set_error(ctx, msg);
    return TK_ERR_INVALID_ARG;
}

inline tk_status tk_unsupported_err(const tk_ctx* ctx, const char* msg) noexcept {
    tk_set_error(ctx, msg);
    return TK_ERR_UNSUPPORTED;
}

inline tk_status tk_hip_err(const tk_ctx* ctx, const char* msg) noexcept {
    tk_set_error(ctx, msg);
    return TK_ERR_HIP;
}

inline bool tk_is_valid_arch(tk_arch arch) noexcept {
    switch (arch) {
        case TK_ARCH_GFX1151:
        case TK_ARCH_GFX1201:
        case TK_ARCH_OTHER:
            return true;
        default:
            return false;
    }
}

inline tk_arch tk_parse_arch_str(const char* s, tk_arch fallback) noexcept {
    if (s == nullptr || s[0] == '\0') {
        return fallback;
    }
    std::string lower;
    for (const char* p = s; *p != '\0' && *p != ':'; ++p) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
    }
    if (lower == "gfx1151") {
        return TK_ARCH_GFX1151;
    }
    if (lower == "gfx1201" || lower == "gfx1200") {
        return TK_ARCH_GFX1201;
    }
    return TK_ARCH_OTHER;
}

inline std::string tk_canonical_knob_key(const char* key) {
    std::string upper;
    for (const char* p = key; *p != '\0'; ++p) {
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(*p))));
    }
    if (upper.rfind("STRATA_", 0) == 0 || upper.rfind("TALUS_", 0) == 0) {
        return upper;
    }
    return "STRATA_" + upper;
}

inline bool tk_parse_int_strict(const char* s, int32_t& out) noexcept {
    if (s == nullptr || s[0] == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || (end != nullptr && *end != '\0')) {
        return false;
    }
    if (v < static_cast<long>(std::numeric_limits<int32_t>::min()) ||
        v > static_cast<long>(std::numeric_limits<int32_t>::max())) {
        return false;
    }
    out = static_cast<int32_t>(v);
    return true;
}

inline int32_t tk_qtype_block_size(tk_qtype qtype) noexcept {
    switch (qtype) {
        case TK_QTYPE_Q4_0:
        case TK_QTYPE_Q5_0:
        case TK_QTYPE_Q8_0:
        case TK_QTYPE_IQ4_NL:
            return 32;
        case TK_QTYPE_Q2_0:
            return 64;
        case TK_QTYPE_Q2_K:
        case TK_QTYPE_Q3_K:
        case TK_QTYPE_Q4_K:
        case TK_QTYPE_Q5_K:
        case TK_QTYPE_Q6_K:
        case TK_QTYPE_IQ2_XXS:
        case TK_QTYPE_IQ2_XS:
        case TK_QTYPE_IQ3_XXS:
        case TK_QTYPE_IQ1_S:
        case TK_QTYPE_IQ1_M:
        case TK_QTYPE_IQ3_S:
        case TK_QTYPE_IQ2_S:
        case TK_QTYPE_IQ4_XS:
            return 256;
        default:
            return 0;
    }
}

inline bool tk_is_supported_ids_qtype(tk_qtype qtype) noexcept {
    switch (qtype) {
        case TK_QTYPE_Q2_K:
        case TK_QTYPE_Q4_K:
        case TK_QTYPE_Q5_K:
        case TK_QTYPE_Q6_K:
        case TK_QTYPE_Q8_0:
        case TK_QTYPE_IQ2_XXS:
        case TK_QTYPE_IQ2_XS:
        case TK_QTYPE_IQ3_XXS:
        case TK_QTYPE_IQ3_S:
        case TK_QTYPE_IQ2_S:
        case TK_QTYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

inline void tk_dispatch_mmvq_cols(
    int ggml_type,
    const void* w,
    const void* x_q8_1,
    float* y,
    int32_t n_in,
    int32_t n_out,
    int32_t ncols,
    void* stream) {
    // wide calls (Laguna verify windows) take the matrix-core or LDS-staged kernel in one launch when it covers the type
    if (strata::kernels::native_mmvq_xt(ggml_type, w, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    if (strata::kernels::native_mmvq_lds(ggml_type, w, x_q8_1, y, n_in, n_out, ncols, stream)) return;
    const size_t col_q8_bytes = strata::kernels::native_q8_1_bytes(n_in, 1);
    const int32_t cmax = (strata::kernels::native_mmvq_w32_supported(ggml_type) &&
                          strata::kernels::native_mmvq_w32_active())
                             ? 16
                             : 8;
    int32_t col_off = 0;
    while (col_off < ncols) {
        const int32_t chunk = std::min<int32_t>(ncols - col_off, cmax);
        const void* chunk_x = static_cast<const uint8_t*>(x_q8_1) + static_cast<size_t>(col_off) * col_q8_bytes;
        float* chunk_y = y + static_cast<size_t>(col_off) * static_cast<size_t>(n_out);
        strata::kernels::native_mmvq(ggml_type, w, chunk_x, chunk_y, n_in, n_out, chunk, stream);
        col_off += chunk;
    }
}

}  // namespace

extern "C" {

/* ==========================================================================
 * Weak Default Prefill Hooks (overridden when WP-1.1b is linked)
 * ========================================================================== */

__attribute__((weak)) tk_status tk_prefill_ctx_init(tk_ctx* ctx) TK_NOEXCEPT {
    if (ctx == nullptr) {
        return TK_ERR_NULL_PTR;
    }
    ctx->prefill_state = nullptr;
    return TK_OK;
}

__attribute__((weak)) void tk_prefill_ctx_destroy(tk_ctx* ctx) TK_NOEXCEPT {
    if (ctx != nullptr) {
        ctx->prefill_state = nullptr;
    }
}

__attribute__((weak)) tk_status tk_prefill_ctx_load_tuning(tk_ctx* ctx, const char* dir) TK_NOEXCEPT {
    if (ctx == nullptr || dir == nullptr) {
        tk_set_error(ctx, "tk_prefill_ctx_load_tuning: null argument");
        return TK_ERR_NULL_PTR;
    }
    if (dir[0] == '\0') {
        tk_set_error(ctx, "tk_prefill_ctx_load_tuning: empty tuning path");
        return TK_ERR_INVALID_ARG;
    }
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        tk_set_error(ctx, "tk_prefill_ctx_load_tuning: path does not exist");
        return TK_ERR_IO;
    }
    return TK_OK;
}

__attribute__((weak)) const char* tk_prefill_last_error(const tk_ctx* ctx) TK_NOEXCEPT {
    if (ctx != nullptr) {
        std::lock_guard<std::mutex> lock(ctx->err_mu);
        return ctx->last_error.c_str();
    }
    return tk_thread_last_error().c_str();
}

void tk_ctx_set_error(tk_ctx* ctx, const char* msg) TK_NOEXCEPT {
    tk_set_error(ctx, msg);
}

void tk_set_last_error(tk_ctx* ctx, const char* msg) TK_NOEXCEPT {
    tk_set_error(ctx, msg);
}

/* ==========================================================================
 * 1. Context, Config & Error Handling
 * ========================================================================== */

uint32_t tk_abi_version(void) TK_NOEXCEPT {
    return TALUS_KERNELS_ABI_VERSION;
}

int32_t tk_mmvq_rows_for_arch(tk_arch arch, tk_qtype qtype, int32_t ncols) TK_NOEXCEPT {
    return tk_default_mmvq_rows(arch, qtype, ncols);
}

int32_t tk_attn_chunk_cells_for_arch(tk_arch arch, int32_t kv_len) TK_NOEXCEPT {
    return tk_default_attn_chunk_cells(arch, kv_len);
}

const char* tk_status_string(tk_status status) TK_NOEXCEPT {
    switch (status) {
        case TK_OK:
            return "TK_OK: success";
        case TK_ERR_INVALID_ARG:
            return "TK_ERR_INVALID_ARG: invalid argument";
        case TK_ERR_NULL_PTR:
            return "TK_ERR_NULL_PTR: null pointer argument";
        case TK_ERR_UNSUPPORTED:
            return "TK_ERR_UNSUPPORTED: unsupported operation or quantization type";
        case TK_ERR_OUT_OF_MEMORY:
            return "TK_ERR_OUT_OF_MEMORY: out of memory";
        case TK_ERR_HIP:
            return "TK_ERR_HIP: HIP runtime error";
        case TK_ERR_HIPBLAS:
            return "TK_ERR_HIPBLAS: hipBLAS / hipBLASLt error";
        case TK_ERR_IO:
            return "TK_ERR_IO: I/O error";
        case TK_ERR_EXCEPTION:
            return "TK_ERR_EXCEPTION: unhandled C++ exception at FFI boundary";
        case TK_ERR_INTERNAL:
            return "TK_ERR_INTERNAL: internal error";
        default:
            return "TK_ERR_UNKNOWN: unknown status code";
    }
}

const char* tk_last_error(const tk_ctx* ctx) TK_NOEXCEPT {
    if (ctx != nullptr) {
        try {
            std::lock_guard<std::mutex> lock(ctx->err_mu);
            tk_thread_last_error() = ctx->last_error;
        } catch (...) {
            return "";
        }
    }
    return tk_thread_last_error().c_str();
}

tk_status tk_config_create(tk_config** out_cfg) TK_NOEXCEPT {
    return tk_config_create_for_arch(TK_ARCH_GFX1151, out_cfg);
}

tk_status tk_config_create_for_arch(tk_arch arch, tk_config** out_cfg) TK_NOEXCEPT {
    if (out_cfg == nullptr) {
        return tk_null_err(nullptr, "tk_config_create_for_arch: out_cfg is null");
    }
    *out_cfg = nullptr;
    if (!tk_is_valid_arch(arch)) {
        return tk_arg_err(nullptr, "tk_config_create_for_arch: invalid tk_arch");
    }
    TK_TRY_BEGIN
    auto* cfg = new tk_config();
    tk_populate_arch_defaults(*cfg, arch);
    *out_cfg = cfg;
    return TK_OK;
    TK_TRY_END(nullptr)
}

tk_status tk_config_from_env(tk_config* cfg) TK_NOEXCEPT {
    if (cfg == nullptr) {
        return tk_null_err(nullptr, "tk_config_from_env: cfg is null");
    }
    TK_TRY_BEGIN
    const char* arch_env = std::getenv("STRATA_ARCH");
    if (arch_env != nullptr && arch_env[0] != '\0') {
        const tk_arch parsed_arch = tk_parse_arch_str(arch_env, cfg->arch);
        tk_populate_arch_defaults(*cfg, parsed_arch);
    }

    // Overlay all STRATA_* and TALUS_* integer environment variables onto cfg->int_knobs.
    if (environ != nullptr) {
        for (char** envp = environ; *envp != nullptr; ++envp) {
            const char* entry = *envp;
            if (std::strncmp(entry, "STRATA_", 7) != 0 && std::strncmp(entry, "TALUS_", 6) != 0) {
                continue;
            }
            const char* eq = std::strchr(entry, '=');
            if (eq == nullptr || eq == entry) {
                continue;
            }
            int32_t parsed = 0;
            if (tk_parse_int_strict(eq + 1, parsed)) {
                std::string key(entry, static_cast<size_t>(eq - entry));
                cfg->int_knobs[key] = parsed;
            }
        }
    }
    return TK_OK;
    TK_TRY_END(nullptr)
}

void tk_config_destroy(tk_config* cfg) TK_NOEXCEPT {
    delete cfg;
}

tk_status tk_config_set_int(tk_config* cfg, const char* key, int64_t value) TK_NOEXCEPT {
    if (cfg == nullptr || key == nullptr) {
        return tk_null_err(nullptr, "tk_config_set_int: null argument");
    }
    if (key[0] == '\0') {
        return tk_arg_err(nullptr, "tk_config_set_int: empty key");
    }
    if (value < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        value > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        return tk_arg_err(nullptr, "tk_config_set_int: value out of 32-bit int range");
    }
    TK_TRY_BEGIN
    const int int_val = static_cast<int>(value);
    cfg->int_knobs[key] = int_val;
    cfg->int_knobs[tk_canonical_knob_key(key)] = int_val;
    return TK_OK;
    TK_TRY_END(nullptr)
}

tk_status tk_config_get_int(const tk_config* cfg, const char* key, int64_t* out_value) TK_NOEXCEPT {
    if (cfg == nullptr || key == nullptr || out_value == nullptr) {
        return tk_null_err(nullptr, "tk_config_get_int: null argument");
    }
    if (key[0] == '\0') {
        return tk_arg_err(nullptr, "tk_config_get_int: empty key");
    }
    TK_TRY_BEGIN
    auto it = cfg->int_knobs.find(key);
    if (it != cfg->int_knobs.end()) {
        *out_value = static_cast<int64_t>(it->second);
        return TK_OK;
    }
    const std::string canon = tk_canonical_knob_key(key);
    it = cfg->int_knobs.find(canon);
    if (it != cfg->int_knobs.end()) {
        *out_value = static_cast<int64_t>(it->second);
        return TK_OK;
    }
    return tk_arg_err(nullptr, "tk_config_get_int: key not found");
    TK_TRY_END(nullptr)
}

tk_status tk_ctx_create(int32_t device_id, const tk_config* cfg, tk_ctx** out_ctx) TK_NOEXCEPT {
    if (out_ctx == nullptr) {
        return tk_null_err(nullptr, "tk_ctx_create: out_ctx is null");
    }
    *out_ctx = nullptr;
    if (device_id < 0 || device_id >= kMaxStubDeviceId) {
        return tk_arg_err(nullptr, "tk_ctx_create: device_id out of range");
    }

    TK_TRY_BEGIN
    auto* ctx = new tk_ctx();
    ctx->device = device_id;
    if (cfg != nullptr) {
        ctx->config = *cfg;
        ctx->arch = cfg->arch;
    } else {
        ctx->arch = TK_ARCH_GFX1151;
        tk_populate_arch_defaults(ctx->config, ctx->arch);
    }

    int dev_count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&dev_count);
    if (count_err == cudaSuccess && dev_count > 0) {
        if (device_id >= dev_count) {
            delete ctx;
            return tk_arg_err(nullptr, "tk_ctx_create: device_id >= HIP device count");
        }
        const cudaError_t set_err = cudaSetDevice(device_id);
        if (set_err != cudaSuccess) {
            (void)cudaGetLastError();
            delete ctx;
            return tk_hip_err(nullptr, "tk_ctx_create: cudaSetDevice failed");
        }
        ctx->has_gpu = true;
        if (cfg == nullptr) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
                const tk_arch detected = tk_parse_arch_str(prop.gcnArchName, ctx->arch);
                ctx->arch = detected;
                tk_populate_arch_defaults(ctx->config, detected);
            } else {
                (void)cudaGetLastError();
            }
        }
    } else {
        // Headless build/test host: clear sticky HIP error and operate in host-validation mode.
        (void)cudaGetLastError();
        ctx->has_gpu = false;
    }

    if (ctx->has_gpu) {
        const tk_status prefill_st = tk_prefill_ctx_init(ctx);
        if (prefill_st != TK_OK) {
            const char* p_err = tk_prefill_last_error(ctx);
            tk_set_error(nullptr, (p_err != nullptr && p_err[0] != '\0') ? p_err : "tk_prefill_ctx_init failed");
            delete ctx;
            return prefill_st;
        }
    }

    *out_ctx = ctx;
    return TK_OK;
    TK_TRY_END(nullptr)
}

void tk_ctx_destroy(tk_ctx* ctx) TK_NOEXCEPT {
    if (ctx == nullptr) {
        return;
    }
    tk_prefill_ctx_destroy(ctx);
    delete ctx;
}

tk_status tk_ctx_device(const tk_ctx* ctx, int32_t* out_device) TK_NOEXCEPT {
    if (ctx == nullptr || out_device == nullptr) {
        return tk_null_err(ctx, "tk_ctx_device: null argument");
    }
    *out_device = ctx->device;
    return TK_OK;
}

tk_status tk_ctx_arch(const tk_ctx* ctx, tk_arch* out_arch) TK_NOEXCEPT {
    if (ctx == nullptr || out_arch == nullptr) {
        return tk_null_err(ctx, "tk_ctx_arch: null argument");
    }
    *out_arch = ctx->arch;
    return TK_OK;
}

tk_status tk_ctx_load_tuning(tk_ctx* ctx, const char* dir) TK_NOEXCEPT {
    if (ctx == nullptr || dir == nullptr) {
        return tk_null_err(ctx, "tk_ctx_load_tuning: null argument");
    }
    if (dir[0] == '\0') {
        return tk_arg_err(ctx, "tk_ctx_load_tuning: empty dir");
    }
    TK_TRY_BEGIN
    const tk_status st = tk_prefill_ctx_load_tuning(ctx, dir);
    if (st != TK_OK) {
        const char* p_err = tk_prefill_last_error(ctx);
        if (p_err != nullptr && p_err[0] != '\0') {
            tk_set_error(ctx, p_err);
        }
        return st;
    }
    return TK_OK;
    TK_TRY_END(ctx)
}

/* ==========================================================================
 * 2. Weight Packing & Activation Quantization
 * ========================================================================== */

extern "C" __attribute__((weak)) void tk_unregister_mmq_q8_buf(const void* xq) noexcept;

tk_status tk_quantize_q8_1(
    tk_ctx*       ctx,
    const float*  x,
    void*         x_q8_1,
    int32_t       n_in,
    int32_t       ncols,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || x == nullptr || x_q8_1 == nullptr) {
        return tk_null_err(ctx, "tk_quantize_q8_1: null pointer argument");
    }
    if (n_in <= 0 || (n_in % 32) != 0 || ncols <= 0) {
        return tk_arg_err(ctx, "tk_quantize_q8_1: invalid n_in or ncols");
    }
    if ((reinterpret_cast<uintptr_t>(x) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(x_q8_1) & 3u) != 0) {
        return tk_arg_err(ctx, "tk_quantize_q8_1: unaligned device pointer");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_quantize_q8_1: no HIP device available");
    }
    TK_TRY_BEGIN
    if (tk_unregister_mmq_q8_buf != nullptr) {
        tk_unregister_mmq_q8_buf(x_q8_1);
    }
    void* const s = tk_effective_stream(stream);
    strata::kernels::native_quantize_q8_1(x, x_q8_1, n_in * ncols, 1, s);
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_pack_q8(
    tk_ctx*              ctx,
    const void*          w_gguf,
    int32_t              n_in,
    int32_t              n_out,
    tk_stream            stream,
    tk_packed_weights**  out_packed) TK_NOEXCEPT {
    (void)stream;
    if (ctx == nullptr || w_gguf == nullptr || out_packed == nullptr) {
        return tk_null_err(ctx, "tk_pack_q8: null pointer argument");
    }
    *out_packed = nullptr;
    if (n_in <= 0 || (n_in % 32) != 0 || n_out <= 0) {
        return tk_arg_err(ctx, "tk_pack_q8: invalid dimensions (n_in must be a positive multiple of 32)");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_pack_q8: no HIP device available");
    }
    TK_TRY_BEGIN
    const size_t n_blocks = static_cast<size_t>(n_in / 32);
    const size_t total_bytes = static_cast<size_t>(n_out) * n_blocks * 34u;

    std::vector<uint8_t> host_src(total_bytes);
    if (cudaMemcpy(host_src.data(), w_gguf, total_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        std::memcpy(host_src.data(), w_gguf, total_bytes);
    }

    std::vector<uint8_t> host_packed(total_bytes);
    strata::kernels::native_q8_0_pack_host(host_src.data(), host_packed.data(), n_in, n_out);

    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, total_bytes) != cudaSuccess || dev_ptr == nullptr) {
        (void)cudaGetLastError();
        return tk_hip_err(ctx, "tk_pack_q8: cudaMalloc failed");
    }

    auto* pw = new tk_packed_weights();
    pw->qtype = TK_QTYPE_Q8_0;
    pw->n_in = n_in;
    pw->n_out = n_out;
    pw->dev_ptr = dev_ptr;
    pw->byte_size = total_bytes;
    pw->device = ctx->device;
    pw->mmvq_key = dev_ptr;

    if (strata::kernels::native_q8_0_packed_eligible(n_in, n_out)) {
        if (cudaMemcpy(dev_ptr, host_packed.data(), total_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaFree(dev_ptr);
            delete pw;
            return tk_hip_err(ctx, "tk_pack_q8: cudaMemcpyHostToDevice failed");
        }
        strata::kernels::native_q8_0_packed_register(dev_ptr, dev_ptr, n_in, n_out);
        pw->registered_q8 = true;
    } else {
        if (cudaMemcpy(dev_ptr, host_src.data(), total_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaFree(dev_ptr);
            delete pw;
            return tk_hip_err(ctx, "tk_pack_q8: cudaMemcpyHostToDevice failed");
        }
    }

    *out_packed = pw;
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_pack_q6(
    tk_ctx*              ctx,
    const void*          w_gguf,
    int32_t              n_in,
    int32_t              n_out,
    tk_stream            stream,
    tk_packed_weights**  out_packed) TK_NOEXCEPT {
    (void)stream;
    if (ctx == nullptr || w_gguf == nullptr || out_packed == nullptr) {
        return tk_null_err(ctx, "tk_pack_q6: null pointer argument");
    }
    *out_packed = nullptr;
    if (n_in <= 0 || (n_in % 256) != 0 || n_out <= 0) {
        return tk_arg_err(ctx, "tk_pack_q6: invalid dimensions (n_in must be a positive multiple of 256)");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_pack_q6: no HIP device available");
    }
    TK_TRY_BEGIN
    const size_t n_blocks = static_cast<size_t>(n_in / 256);
    const size_t raw_bytes = static_cast<size_t>(n_out) * n_blocks * 210u;

    std::vector<uint8_t> host_src(raw_bytes);
    if (cudaMemcpy(host_src.data(), w_gguf, raw_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        std::memcpy(host_src.data(), w_gguf, raw_bytes);
    }

    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, raw_bytes) != cudaSuccess || dev_ptr == nullptr) {
        (void)cudaGetLastError();
        return tk_hip_err(ctx, "tk_pack_q6: cudaMalloc failed");
    }
    if (cudaMemcpy(dev_ptr, host_src.data(), raw_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaFree(dev_ptr);
        return tk_hip_err(ctx, "tk_pack_q6: cudaMemcpyHostToDevice failed");
    }

    const bool packed_ok = strata::kernels::native_q6_k_pack(dev_ptr, n_in, n_out, "tk_pack_q6");

    auto* pw = new tk_packed_weights();
    pw->qtype = TK_QTYPE_Q6_K;
    pw->n_in = n_in;
    pw->n_out = n_out;
    pw->dev_ptr = dev_ptr;
    pw->byte_size = raw_bytes;
    pw->device = ctx->device;
    pw->mmvq_key = dev_ptr;
    pw->registered_q6 = packed_ok;
    *out_packed = pw;
    return TK_OK;
    TK_TRY_END(ctx)
}

void tk_packed_free(tk_ctx* ctx, tk_packed_weights* packed) TK_NOEXCEPT {
    (void)ctx;
    if (packed == nullptr) {
        return;
    }
    if (packed->registered_q8 && packed->mmvq_key != nullptr) {
        strata::kernels::native_q8_0_packed_unregister(packed->mmvq_key);
    }
    if (packed->registered_q6 && packed->mmvq_key != nullptr) {
        strata::kernels::native_q6_k_unpack(packed->mmvq_key);
    }
    if (packed->dev_ptr != nullptr) {
        (void)cudaFree(packed->dev_ptr);
        (void)cudaGetLastError();
    }
    delete packed;
}

/* ==========================================================================
 * 3. MMVQ & MoE Decode / Verify Entry Points
 * ========================================================================== */

tk_status tk_mmvq(
    tk_ctx*       ctx,
    tk_qtype      qtype,
    const void*   w,
    const void*   x_q8_1,
    float*        y,
    int32_t       n_in,
    int32_t       n_out,
    int32_t       ncols,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || w == nullptr || x_q8_1 == nullptr || y == nullptr) {
        return tk_null_err(ctx, "tk_mmvq: null pointer argument");
    }
    if (!strata::kernels::native_mmvq_supported(static_cast<int>(qtype))) {
        return tk_unsupported_err(ctx, "tk_mmvq: unsupported quantization type");
    }
    const int32_t blk = tk_qtype_block_size(qtype);
    // (up to 64 columns; above 16 the dispatch runs 16-column chunks unless the LDS kernel takes the call)
    if (n_in <= 0 || n_out <= 0 || ncols < 1 || ncols > 64 || blk <= 0 || (n_in % blk) != 0) {
        return tk_arg_err(ctx, "tk_mmvq: invalid dimensions");
    }
    if ((reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(x_q8_1) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0) {
        return tk_arg_err(ctx, "tk_mmvq: unaligned device pointer");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_mmvq: no HIP device available");
    }
    TK_TRY_BEGIN
    tk_dispatch_mmvq_cols(static_cast<int>(qtype), w, x_q8_1, y, n_in, n_out, ncols, tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_mmvq_packed(
    tk_ctx*                   ctx,
    const tk_packed_weights*  packed,
    const void*               x_q8_1,
    float*                    y,
    int32_t                   ncols,
    tk_stream                 stream) TK_NOEXCEPT {
    if (ctx == nullptr || packed == nullptr || packed->dev_ptr == nullptr ||
        x_q8_1 == nullptr || y == nullptr) {
        return tk_null_err(ctx, "tk_mmvq_packed: null pointer argument");
    }
    if (ncols < 1 || ncols > 16 || packed->n_in <= 0 || packed->n_out <= 0) {
        return tk_arg_err(ctx, "tk_mmvq_packed: invalid dimensions");
    }
    if (packed->qtype != TK_QTYPE_Q8_0 && packed->qtype != TK_QTYPE_Q6_K) {
        return tk_unsupported_err(ctx, "tk_mmvq_packed: unsupported packed qtype");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_mmvq_packed: no HIP device available");
    }
    TK_TRY_BEGIN
    const void* key = (packed->mmvq_key != nullptr) ? packed->mmvq_key : packed->dev_ptr;
    tk_dispatch_mmvq_cols(
        static_cast<int>(packed->qtype),
        key,
        x_q8_1,
        y,
        packed->n_in,
        packed->n_out,
        ncols,
        tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_mmvq_ids(
    tk_ctx*         ctx,
    tk_qtype        qtype,
    const void*     w,
    size_t          expert_bytes,
    const int32_t*  ids_dev,
    int32_t         n_ids,
    const void*     x_q8_1,
    bool            x_per_id,
    float*          y,
    int32_t         n_in,
    int32_t         n_out,
    const void*     w2,
    float*          y2,
    tk_stream       stream) TK_NOEXCEPT {
    if (ctx == nullptr || w == nullptr || ids_dev == nullptr || x_q8_1 == nullptr || y == nullptr) {
        return tk_null_err(ctx, "tk_mmvq_ids: null pointer argument");
    }
    if ((w2 != nullptr) != (y2 != nullptr)) {
        return tk_null_err(ctx, "tk_mmvq_ids: w2 and y2 must both be non-null or both be null");
    }
    if (!tk_is_supported_ids_qtype(qtype)) {
        return tk_unsupported_err(ctx, "tk_mmvq_ids: unsupported quantization type");
    }
    const int32_t blk = tk_qtype_block_size(qtype);
    if (n_ids <= 0 || n_in <= 0 || n_out <= 0 || blk <= 0 || (n_in % blk) != 0) {
        return tk_arg_err(ctx, "tk_mmvq_ids: invalid dimensions");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_mmvq_ids: no HIP device available");
    }
    TK_TRY_BEGIN
    if (strata::kernels::mimo_is_iq_qtype(static_cast<int>(qtype))) {
        const int x_group =
            (w2 != nullptr && expert_bytes > 0 && x_per_id && n_ids > 8 && (n_ids % 8) == 0) ? 8 : 1;
        strata::kernels::mimo_mmvq_ids(
            static_cast<int>(qtype),
            w,
            expert_bytes,
            ids_dev,
            n_ids,
            x_q8_1,
            x_per_id,
            y,
            n_in,
            n_out,
            tk_effective_stream(stream),
            w2,
            y2,
            x_group);
    } else if (expert_bytes == 0 && (x_per_id || n_ids == 1) && n_ids >= 1 && n_ids <= 8 &&
               (qtype == TK_QTYPE_Q8_0 || qtype == TK_QTYPE_Q4_K ||
                qtype == TK_QTYPE_Q5_K || qtype == TK_QTYPE_Q6_K)) {
        static const bool ids_w32 = [] {
            const char* e = std::getenv("STRATA_IDS_W32");
            if (!e) e = std::getenv("TALUS_MIMO_W32_DENSE");
            return e != nullptr && e[0] == '1';
        }();
        void* const s = tk_effective_stream(stream);
        const int gtype = static_cast<int>(qtype);
        if (ids_w32) {
            if (w2 != nullptr) {
                const void* ws[2] = {w, w2};
                float* ys[2] = {y, y2};
                const int n_outs[2] = {n_out, n_out};
                if (strata::kernels::native_mmvq_w32_multi(gtype, 2, ws, ys, n_outs, x_q8_1, n_in, n_ids, s)) {
                    return TK_OK;
                }
            } else if (strata::kernels::native_mmvq_w32(gtype, w, x_q8_1, y, n_in, n_out, n_ids, s)) {
                return TK_OK;
            }
        }
        strata::kernels::native_mmvq_ids_cols(gtype, w, n_ids, x_q8_1, y, n_in, n_out, s, w2, y2);
    } else {
        strata::kernels::native_mmvq_ids(
            static_cast<int>(qtype),
            w,
            expert_bytes,
            ids_dev,
            n_ids,
            x_q8_1,
            x_per_id,
            y,
            n_in,
            n_out,
            tk_effective_stream(stream),
            w2,
            y2);
    }
    return TK_OK;
    TK_TRY_END(ctx)
}

/* ==========================================================================
 * 4. Norms & RoPE
 * ========================================================================== */

tk_status tk_rms_norm(
    tk_ctx*       ctx,
    float*        x,
    const float*  w,
    int64_t       rows,
    int64_t       cols,
    float         eps,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || x == nullptr) {
        return tk_null_err(ctx, "tk_rms_norm: null pointer argument");
    }
    if (rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
        return tk_arg_err(ctx, "tk_rms_norm: invalid dimensions or eps");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_rms_norm: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::rms_norm_weighted(x, w, rows, cols, eps, tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_gr_norm(
    tk_ctx*       ctx,
    const float*  x_dev,
    const float*  w_dev,
    float*        dst_dev,
    int32_t       n_rows,
    int32_t       dim,
    int32_t       groups,
    float         eps,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || x_dev == nullptr || w_dev == nullptr || dst_dev == nullptr) {
        return tk_null_err(ctx, "tk_gr_norm: null pointer argument");
    }
    if (n_rows <= 0 || dim <= 0 || groups <= 0 || (dim % groups) != 0 || !(eps >= 0.0f)) {
        return tk_arg_err(ctx, "tk_gr_norm: invalid dimensions or eps");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_gr_norm: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::native_gr_rms_norm_weighted(
        x_dev, w_dev, dst_dev, dim / groups, n_rows * groups, eps, tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_rope_neox(
    tk_ctx*         ctx,
    const float*    x,
    float*          out,
    int64_t         rows,
    int32_t         head_dim,
    int32_t         n_rot,
    const float*    cos_tab,
    const float*    sin_tab,
    const int32_t*  pos_dev,
    tk_stream       stream) TK_NOEXCEPT {
    if (ctx == nullptr || x == nullptr || out == nullptr ||
        cos_tab == nullptr || sin_tab == nullptr || pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_rope_neox: null pointer argument");
    }
    if (rows <= 0 || head_dim <= 0 || n_rot <= 0 || n_rot > head_dim || (n_rot & 1) != 0) {
        return tk_arg_err(ctx, "tk_rope_neox: invalid parameters");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_rope_neox: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::rope_neox_apply(
        x, out, rows, head_dim, n_rot, cos_tab, sin_tab, pos_dev, tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_mrope(
    tk_ctx*         ctx,
    const float*    x,
    float*          out,
    int64_t         rows,
    int32_t         head_dim,
    int32_t         n_rot,
    const int32_t   sections[4],
    const float*    cos_tab,
    const float*    sin_tab,
    const int32_t*  pos_dev,
    const int32_t*  mrope_table_dev,
    tk_stream       stream) TK_NOEXCEPT {
    if (ctx == nullptr || x == nullptr || out == nullptr || sections == nullptr ||
        cos_tab == nullptr || sin_tab == nullptr || pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_mrope: null pointer argument");
    }
    if (rows <= 0 || head_dim <= 0 || n_rot <= 0 || n_rot > head_dim || (n_rot & 1) != 0 ||
        sections[0] < 0 || sections[1] < 0 || sections[2] < 0 || sections[3] < 0) {
        return tk_arg_err(ctx, "tk_mrope: invalid parameters");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_mrope: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::mrope_table_set(mrope_table_dev);
    strata::kernels::rope_neox_apply(
        x, out, rows, head_dim, n_rot, cos_tab, sin_tab, pos_dev, tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

/* ==========================================================================
 * 5. Attention & KV Cache (Decode + Weak MiMo/GLM Stubs)
 * ========================================================================== */

tk_status tk_attn_decode(
    tk_ctx*          ctx,
    const float*     q,
    const void*      k_pool,
    const void*      v_pool,
    const uint16_t*  k_scale,
    const uint16_t*  v_scale,
    float*           out,
    float*           scratch,
    int32_t          n_tok,
    int32_t          n_head,
    int32_t          n_kv,
    int32_t          head_dim,
    int32_t          head_dim_v,
    int32_t          kv_stride,
    int32_t          window,
    float            scale,
    const int32_t*   pos_dev,
    int32_t          n_split,
    tk_stream        stream) TK_NOEXCEPT {
    (void)window;
    (void)scale;
    if (ctx == nullptr || q == nullptr || k_pool == nullptr ||
        v_pool == nullptr || out == nullptr || pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_attn_decode: null pointer argument");
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return tk_null_err(ctx, "tk_attn_decode: k_scale and v_scale must both be non-null or both be null");
    }
    if (n_split > 1 && scratch == nullptr) {
        return tk_null_err(ctx, "tk_attn_decode: scratch is required when n_split > 1");
    }
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0 || (n_head % n_kv) != 0 ||
        head_dim <= 0 || head_dim_v <= 0 || kv_stride <= 0 || !(scale > 0.0f) || n_split <= 0) {
        return tk_arg_err(ctx, "tk_attn_decode: invalid dimensions or scale");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_attn_decode: no HIP device available");
    }
    TK_TRY_BEGIN
    void* const s = tk_effective_stream(stream);
    if (head_dim == 256 && head_dim_v == 256 && scratch != nullptr) {
        strata::kernels::QsaShapes shapes = strata::kernels::qsa_real_shapes();
        shapes.n_head = n_head;
        shapes.n_head_kv = n_kv;
        shapes.head_dim = head_dim;
        shapes.page_size = 1;
        strata::kernels::QsaAttnPools pools{};
        if (k_scale == nullptr) {
            pools.k_pool = static_cast<const uint16_t*>(k_pool);
            pools.v_pool = static_cast<const uint16_t*>(v_pool);
        } else {
            pools.k_q = static_cast<const int8_t*>(k_pool);
            pools.v_q = static_cast<const int8_t*>(v_pool);
            pools.k_scale = k_scale;
            pools.v_scale = v_scale;
        }
        strata::kernels::qsa_decode_attn_step(
            q, pools, nullptr, pos_dev, kv_stride, shapes, scratch, out, s);
        return TK_OK;
    }
    return tk_unsupported_err(ctx, "tk_attn_decode: unsupported head_dim configuration");
    TK_TRY_END(ctx)
}

// tk_attn_swa_sinks that, where the prefill v2 kernel runs, also writes the outputs as the wo projection's
// MMQ q8_1 rows (q8_layout 0 = D4, 1 = DS4; bitwise tk_q36_mmq_quantize of `out`); *q8_written = 1 when it did.
tk_status tk_attn_swa_sinks_mmq(
    tk_ctx*          ctx,
    const float*     q,
    const void*      k_pool,
    const void*      v_pool,
    const float*     sinks,
    float*           out,
    float*           scratch,
    int32_t          n_tok,
    int32_t          n_head,
    int32_t          n_kv,
    int32_t          head_dim,
    int32_t          head_dim_v,
    int32_t          window,
    int32_t          stride,
    float            scale,
    const int32_t*   pos_dev,
    void*            q8_mmq,
    int32_t          q8_layout,
    int32_t*         q8_written,
    tk_stream        stream) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || k_pool == nullptr || v_pool == nullptr ||
        out == nullptr || pos_dev == nullptr || q8_written == nullptr) {
        return tk_null_err(ctx, "tk_attn_swa_sinks_mmq: null pointer argument");
    }
    *q8_written = 0;
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0 || window <= 0 || stride < 0) {
        return tk_arg_err(ctx, "tk_attn_swa_sinks_mmq: invalid dimensions");
    }
    if ((head_dim != 192 && head_dim != 128) || head_dim_v != 128 || (n_head % n_kv) != 0 ||
        q == out || k_pool == v_pool) {
        return tk_unsupported_err(ctx, "tk_attn_swa_sinks_mmq: unsupported head dimensions or aliased buffers");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_attn_swa_sinks_mmq: no HIP device available");
    }
    TK_TRY_BEGIN
    bool done = false;
    strata::kernels::mimo_attn_swa_sinks(
        q, k_pool, v_pool, nullptr, nullptr, sinks, out, scratch, n_tok, n_head, n_kv, head_dim, head_dim_v, window,
        stride, scale, pos_dev, 0, tk_effective_stream(stream), 1.0f, nullptr, q8_mmq, q8_layout, &done);
    *q8_written = done ? 1 : 0;
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_attn_swa_sinks(
    tk_ctx*          ctx,
    const float*     q,
    const void*      k_pool,
    const void*      v_pool,
    const uint16_t*  k_scale,
    const uint16_t*  v_scale,
    const float*     sinks,
    float*           out,
    float*           scratch,
    int32_t          n_tok,
    int32_t          n_head,
    int32_t          n_kv,
    int32_t          head_dim,
    int32_t          head_dim_v,
    int32_t          window,
    int32_t          stride,
    float            scale,
    const int32_t*   pos_dev,
    int32_t          n_split,
    tk_stream        stream) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || k_pool == nullptr || v_pool == nullptr ||
        out == nullptr || pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_attn_swa_sinks: null pointer argument");
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return tk_null_err(ctx, "tk_attn_swa_sinks: k_scale and v_scale must both be non-null or both be null");
    }
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0 || window <= 0 || stride < 0) {
        return tk_arg_err(ctx, "tk_attn_swa_sinks: invalid dimensions");
    }
    if ((head_dim != 192 && head_dim != 128) || head_dim_v != 128 || (n_head % n_kv) != 0 ||
        q == out || k_pool == v_pool) {
        return tk_unsupported_err(ctx, "tk_attn_swa_sinks: unsupported head dimensions or aliased buffers");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_attn_swa_sinks: no HIP device available");
    }
    TK_TRY_BEGIN
    const float value_scale = (n_split == -1) ? 0.7071067811865475244f : 1.0f;
    strata::kernels::mimo_attn_swa_sinks(
        q,
        k_pool,
        v_pool,
        k_scale,
        v_scale,
        sinks,
        out,
        scratch,
        n_tok,
        n_head,
        n_kv,
        head_dim,
        head_dim_v,
        window,
        stride,
        scale,
        pos_dev,
        n_split,
        tk_effective_stream(stream),
        value_scale);
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_attn_swa_sinks_q8(
    tk_ctx*          ctx,
    const float*     q,
    const void*      k_pool,
    const void*      v_pool,
    const uint16_t*  k_scale,
    const uint16_t*  v_scale,
    const float*     sinks,
    float*           out,
    float*           scratch,
    int32_t          n_tok,
    int32_t          n_head,
    int32_t          n_kv,
    int32_t          head_dim,
    int32_t          head_dim_v,
    int32_t          window,
    int32_t          stride,
    float            scale,
    const int32_t*   pos_dev,
    int32_t          n_split,
    tk_stream        stream,
    void*            q8_out) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || k_pool == nullptr || v_pool == nullptr ||
        out == nullptr || pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_attn_swa_sinks_q8: null pointer argument");
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return tk_null_err(ctx, "tk_attn_swa_sinks_q8: k_scale and v_scale must both be non-null or both be null");
    }
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0 || window <= 0 || stride < 0) {
        return tk_arg_err(ctx, "tk_attn_swa_sinks_q8: invalid dimensions");
    }
    if ((head_dim != 192 && head_dim != 128) || head_dim_v != 128 || (n_head % n_kv) != 0 ||
        q == out || k_pool == v_pool) {
        return tk_unsupported_err(ctx, "tk_attn_swa_sinks_q8: unsupported head dimensions or aliased buffers");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_attn_swa_sinks_q8: no HIP device available");
    }
    TK_TRY_BEGIN
    const float value_scale = (n_split == -1) ? 0.7071067811865475244f : 1.0f;
    strata::kernels::mimo_attn_swa_sinks(
        q,
        k_pool,
        v_pool,
        k_scale,
        v_scale,
        sinks,
        out,
        scratch,
        n_tok,
        n_head,
        n_kv,
        head_dim,
        head_dim_v,
        window,
        stride,
        scale,
        pos_dev,
        n_split,
        tk_effective_stream(stream),
        value_scale,
        q8_out);
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_attn_k192_v128(
    tk_ctx*          ctx,
    const float*     q,
    const void*      k_pool,
    const void*      v_pool,
    const uint16_t*  k_scale,
    const uint16_t*  v_scale,
    float*           out,
    float*           scratch,
    int32_t          n_tok,
    int32_t          n_head,
    int32_t          n_kv,
    int32_t          stride,
    float            scale,
    const int32_t*   pos_dev,
    int32_t          n_split,
    tk_stream        stream) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || k_pool == nullptr || v_pool == nullptr ||
        out == nullptr || pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_attn_k192_v128: null pointer argument");
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return tk_null_err(ctx, "tk_attn_k192_v128: k_scale and v_scale must both be non-null or both be null");
    }
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0 || stride < 0) {
        return tk_arg_err(ctx, "tk_attn_k192_v128: invalid dimensions");
    }
    if ((n_head % n_kv) != 0 || q == out || k_pool == v_pool) {
        return tk_unsupported_err(ctx, "tk_attn_k192_v128: unsupported GQA ratio or aliased buffers");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_attn_k192_v128: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::mimo_attn_k192_v128(
        q,
        k_pool,
        v_pool,
        k_scale,
        v_scale,
        out,
        scratch,
        n_tok,
        n_head,
        n_kv,
        stride,
        scale,
        pos_dev,
        n_split,
        tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_attn_k192_v128_q8(
    tk_ctx*          ctx,
    const float*     q,
    const void*      k_pool,
    const void*      v_pool,
    const uint16_t*  k_scale,
    const uint16_t*  v_scale,
    float*           out,
    float*           scratch,
    int32_t          n_tok,
    int32_t          n_head,
    int32_t          n_kv,
    int32_t          stride,
    float            scale,
    const int32_t*   pos_dev,
    int32_t          n_split,
    tk_stream        stream,
    void*            q8_out) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || k_pool == nullptr || v_pool == nullptr ||
        out == nullptr || pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_attn_k192_v128_q8: null pointer argument");
    }
    if ((k_scale != nullptr) != (v_scale != nullptr)) {
        return tk_null_err(ctx, "tk_attn_k192_v128_q8: k_scale and v_scale must both be non-null or both be null");
    }
    if (n_tok <= 0 || n_head <= 0 || n_kv <= 0 || stride < 0) {
        return tk_arg_err(ctx, "tk_attn_k192_v128_q8: invalid dimensions");
    }
    if ((n_head % n_kv) != 0 || q == out || k_pool == v_pool) {
        return tk_unsupported_err(ctx, "tk_attn_k192_v128_q8: unsupported GQA ratio or aliased buffers");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_attn_k192_v128_q8: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::mimo_attn_k192_v128(
        q,
        k_pool,
        v_pool,
        k_scale,
        v_scale,
        out,
        scratch,
        n_tok,
        n_head,
        n_kv,
        stride,
        scale,
        pos_dev,
        n_split,
        tk_effective_stream(stream),
        q8_out);
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_mimo_rope_qk_store(
    tk_ctx*         ctx,
    float*          q,
    float*          k,
    const float*    v,
    int32_t         n_tok,
    int32_t         n_head_q,
    int32_t         n_head_kv,
    int32_t         head_dim,
    int32_t         v_dim,
    int32_t         n_rot,
    const float*    cos_tab,
    const float*    sin_tab,
    const int32_t*  pos_q_dev,
    const int32_t*  pos_k_dev,
    void*           k_pool,
    void*           v_pool,
    const int32_t*  store_pos_dev,
    int32_t         ring,
    tk_stream       stream) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || k == nullptr || v == nullptr || cos_tab == nullptr || sin_tab == nullptr ||
        pos_q_dev == nullptr || pos_k_dev == nullptr || k_pool == nullptr || v_pool == nullptr ||
        store_pos_dev == nullptr) {
        return tk_null_err(ctx, "tk_mimo_rope_qk_store: null pointer argument");
    }
    if (n_tok <= 0 || n_head_q <= 0 || n_head_kv <= 0 || head_dim <= 0 || v_dim <= 0 || ring <= 0 || n_rot <= 0 ||
        (n_rot % 2) != 0 || n_rot > head_dim) {
        return tk_arg_err(ctx, "tk_mimo_rope_qk_store: invalid dimensions");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_mimo_rope_qk_store: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::mimo_rope_qk_store(q, k, v, n_tok, n_head_q, n_head_kv, head_dim, v_dim, n_rot, cos_tab, sin_tab,
                                        pos_q_dev, pos_k_dev, k_pool, v_pool, store_pos_dev, ring,
                                        tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

__attribute__((weak)) tk_status tk_dsa_decode_indexer(
    tk_ctx*         ctx,
    const float*    q_idx,
    const void*     k_idx_pool,
    const float*    weights,
    const int32_t*  pos_dev,
    int32_t         n_tok,
    int32_t         idx_heads,
    int32_t         idx_dim,
    int32_t         top_k,
    int32_t         capacity,
    float*          scores_scratch,
    int32_t*        topk_ids_dev,
    tk_stream       stream) TK_NOEXCEPT {
    (void)q_idx;
    (void)k_idx_pool;
    (void)weights;
    (void)pos_dev;
    (void)n_tok;
    (void)idx_heads;
    (void)idx_dim;
    (void)top_k;
    (void)capacity;
    (void)scores_scratch;
    (void)topk_ids_dev;
    (void)stream;
    return tk_unsupported_err(ctx, "tk_dsa_decode_indexer: GLM-5.1 DSA top-k indexer not yet enabled (WP-1.7)");
}

/* ==========================================================================
 * 6. GDN & KDA Recurrent Layers
 * ========================================================================== */

tk_status tk_gdn_preprocess(
    tk_ctx*       ctx,
    float*        conv_state,
    const float*  qkv,
    const float*  conv_w,
    float*        conv_out,
    const float*  alpha,
    const float*  dt,
    const float*  ssm_a,
    float*        gate_out,
    float*        beta_inout,
    int64_t       channels,
    int64_t       d_conv,
    int64_t       h_k,
    int64_t       h_v,
    int64_t       state_size,
    float         l2_eps,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || conv_state == nullptr || qkv == nullptr || conv_w == nullptr ||
        conv_out == nullptr || alpha == nullptr || dt == nullptr || ssm_a == nullptr ||
        gate_out == nullptr || beta_inout == nullptr) {
        return tk_null_err(ctx, "tk_gdn_preprocess: null pointer argument");
    }
    if (channels <= 0 || d_conv < 2 || h_k <= 0 || h_v <= 0 || (h_v % h_k) != 0 ||
        state_size <= 0 || !(l2_eps > 0.0f) || channels < (2 * h_k + h_v) * state_size) {
        return tk_arg_err(ctx, "tk_gdn_preprocess: invalid dimensions or l2_eps");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_gdn_preprocess: no HIP device available");
    }
    TK_TRY_BEGIN
    void* const s = tk_effective_stream(stream);
    if (d_conv == 4 && state_size == 128 && channels == (2 * h_k + h_v) * 128) {
        strata::kernels::fused_gdn_conv_l2(
            conv_state, qkv, conv_w, conv_out, static_cast<int>(channels), static_cast<int>(2 * h_k), l2_eps, s);
    } else {
        strata::kernels::gdn_conv_step(conv_state, qkv, conv_w, conv_out, channels, d_conv, s);
        strata::kernels::silu_inplace(conv_out, channels, s);
        strata::kernels::gdn_l2_norm(conv_out, 2 * h_k, state_size, l2_eps, s);
    }
    strata::kernels::gdn_gate(alpha, dt, ssm_a, gate_out, 1, h_v, s);
    strata::kernels::gdn_beta_gate(beta_inout, h_v, s);
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_gdn_step(
    tk_ctx*       ctx,
    float*        state,
    const float*  q,
    const float*  k,
    const float*  v,
    const float*  gate,
    const float*  beta,
    float*        out,
    int64_t       state_size,
    int64_t       h_k,
    int64_t       h_v,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || state == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || gate == nullptr || beta == nullptr || out == nullptr) {
        return tk_null_err(ctx, "tk_gdn_step: null pointer argument");
    }
    if (state_size <= 0 || h_k <= 0 || h_v <= 0 || (h_v % h_k) != 0) {
        return tk_arg_err(ctx, "tk_gdn_step: invalid dimensions");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_gdn_step: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::GdnShapes shapes{};
    shapes.S = state_size;
    shapes.h_k = h_k;
    shapes.h_v = h_v;
    strata::kernels::gdn_step(state, q, k, v, gate, beta, out, shapes, tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_gdn_seq(
    tk_ctx*       ctx,
    float*        state,
    const float*  q,
    const float*  k,
    const float*  v,
    const float*  gate,
    const float*  beta,
    const float*  z,
    const float*  gamma,
    float*        out,
    int32_t       n_tok,
    int64_t       state_size,
    int64_t       h_k,
    int64_t       h_v,
    float         norm_eps,
    bool          silu_gate,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || state == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || gate == nullptr || beta == nullptr || z == nullptr ||
        gamma == nullptr || out == nullptr) {
        return tk_null_err(ctx, "tk_gdn_seq: null pointer argument");
    }
    if (n_tok <= 0 || state_size <= 0 || h_k <= 0 || h_v <= 0 || (h_v % h_k) != 0 || !(norm_eps > 0.0f)) {
        return tk_arg_err(ctx, "tk_gdn_seq: invalid dimensions or norm_eps");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_gdn_seq: no HIP device available");
    }
    TK_TRY_BEGIN
    void* const s = tk_effective_stream(stream);
    const size_t qk_dim = static_cast<size_t>(h_k) * static_cast<size_t>(state_size);
    const size_t v_dim = static_cast<size_t>(h_v) * static_cast<size_t>(state_size);

    for (int32_t tok = 0; tok < n_tok; ++tok) {
        const float* tok_q = q + static_cast<size_t>(tok) * qk_dim;
        const float* tok_k = k + static_cast<size_t>(tok) * qk_dim;
        const float* tok_v = v + static_cast<size_t>(tok) * v_dim;
        const float* tok_gate = gate + static_cast<size_t>(tok) * static_cast<size_t>(h_v);
        const float* tok_beta = beta + static_cast<size_t>(tok) * static_cast<size_t>(h_v);
        const float* tok_z = z + static_cast<size_t>(tok) * v_dim;
        float* tok_out = out + static_cast<size_t>(tok) * v_dim;
        if (silu_gate) {
            strata::kernels::fused_gdn_step_norm_silu(
                state, tok_q, tok_k, tok_v, tok_gate, tok_beta, tok_z, gamma,
                norm_eps, tok_out, static_cast<int>(h_k), static_cast<int>(h_v), s);
        } else {
            strata::kernels::fused_gdn_step_norm(
                state, tok_q, tok_k, tok_v, tok_gate, tok_beta, tok_z, gamma,
                norm_eps, tok_out, static_cast<int>(h_k), static_cast<int>(h_v), s);
        }
    }
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_gdn_fused(
    tk_ctx*       ctx,
    float*        state,
    const float*  q,
    const float*  k,
    const float*  v,
    const float*  gate,
    const float*  beta,
    const float*  z,
    const float*  gamma,
    float         eps,
    float*        y,
    int32_t       h_k,
    int32_t       h_v,
    bool          silu_gate,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || state == nullptr || q == nullptr || k == nullptr ||
        v == nullptr || gate == nullptr || beta == nullptr || z == nullptr ||
        gamma == nullptr || y == nullptr) {
        return tk_null_err(ctx, "tk_gdn_fused: null pointer argument");
    }
    if (h_k <= 0 || h_v <= 0 || (h_v % h_k) != 0 || !(eps > 0.0f)) {
        return tk_arg_err(ctx, "tk_gdn_fused: invalid dimensions or eps");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_gdn_fused: no HIP device available");
    }
    TK_TRY_BEGIN
    void* const s = tk_effective_stream(stream);
    if (silu_gate) {
        strata::kernels::fused_gdn_step_norm_silu(
            state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, s);
    } else {
        strata::kernels::fused_gdn_step_norm(
            state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v, s);
    }
    return TK_OK;
    TK_TRY_END(ctx)
}

__attribute__((weak)) tk_status tk_kda_step(
    tk_ctx*       ctx,
    float*        state,
    const float*  q,
    const float*  k,
    const float*  v,
    const float*  gk,
    const float*  beta,
    const float*  z,
    const float*  gamma,
    float         eps,
    float*        out,
    int32_t       n_tok,
    int32_t       state_size,
    int32_t       h_k,
    int32_t       h_v,
    tk_stream     stream) TK_NOEXCEPT {
    (void)state;
    (void)q;
    (void)k;
    (void)v;
    (void)gk;
    (void)beta;
    (void)z;
    (void)gamma;
    (void)eps;
    (void)out;
    (void)n_tok;
    (void)state_size;
    (void)h_k;
    (void)h_v;
    (void)stream;
    return tk_unsupported_err(ctx, "tk_kda_step: Kimi KDA per-channel decay recurrent step not yet enabled (WP-1.6/1.7)");
}

/* ==========================================================================
 * 8. Additional Batch / Multi-Matrix MMVQ & Quantization Helpers
 * ========================================================================== */

size_t tk_q8_1_bytes(int32_t n_in, int32_t ncols) TK_NOEXCEPT {
    if (n_in <= 0 || (n_in % 32) != 0 || ncols <= 0) {
        return 0;
    }
    return strata::kernels::native_q8_1_bytes(n_in, 1) * static_cast<size_t>(ncols);
}

tk_status tk_swiglu_quantize_q8_1(
    tk_ctx*       ctx,
    const float*  gate,
    const float*  up,
    void*         x_q8_1,
    int32_t       n_in,
    int32_t       ncols,
    tk_stream     stream) TK_NOEXCEPT {
    if (ctx == nullptr || gate == nullptr || up == nullptr || x_q8_1 == nullptr) {
        return tk_null_err(ctx, "tk_swiglu_quantize_q8_1: null pointer argument");
    }
    if (n_in <= 0 || (n_in % 32) != 0 || ncols <= 0) {
        return tk_arg_err(ctx, "tk_swiglu_quantize_q8_1: invalid n_in or ncols");
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_swiglu_quantize_q8_1: no HIP device available");
    }
    TK_TRY_BEGIN
    if (tk_unregister_mmq_q8_buf != nullptr) {
        tk_unregister_mmq_q8_buf(x_q8_1);
    }
    void* const s = tk_effective_stream(stream);
    strata::kernels::native_swiglu_quantize_q8_1(gate, up, x_q8_1, n_in * ncols, 1, s);
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_mmvq_w32_multi(
    tk_ctx*              ctx,
    tk_qtype             qtype,
    int32_t              n_mats,
    const void* const*   w_ptrs,
    float* const*        y_ptrs,
    const int32_t*       n_out_per_mat,
    const void*          x_q8_1,
    int32_t              n_in,
    int32_t              ncols,
    tk_stream            stream) TK_NOEXCEPT {
    if (ctx == nullptr || w_ptrs == nullptr || y_ptrs == nullptr ||
        n_out_per_mat == nullptr || x_q8_1 == nullptr) {
        return tk_null_err(ctx, "tk_mmvq_w32_multi: null pointer argument");
    }
    if (!strata::kernels::native_mmvq_supported(static_cast<int>(qtype))) {
        return tk_unsupported_err(ctx, "tk_mmvq_w32_multi: unsupported quantization type");
    }
    const int32_t blk = tk_qtype_block_size(qtype);
    if (n_mats < 1 || n_mats > 3 || n_in <= 0 || ncols < 1 || ncols > 16 || blk <= 0 || (n_in % blk) != 0) {
        return tk_arg_err(ctx, "tk_mmvq_w32_multi: invalid dimensions or n_mats/ncols out of range");
    }
    for (int32_t m = 0; m < n_mats; ++m) {
        if (w_ptrs[m] == nullptr || y_ptrs[m] == nullptr) {
            return tk_null_err(ctx, "tk_mmvq_w32_multi: null matrix or destination pointer");
        }
        if (n_out_per_mat[m] <= 0) {
            return tk_arg_err(ctx, "tk_mmvq_w32_multi: non-positive n_out_per_mat");
        }
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_mmvq_w32_multi: no HIP device available");
    }
    TK_TRY_BEGIN
    void* const s = tk_effective_stream(stream);
    if (!strata::kernels::native_mmvq_w32_multi(
            static_cast<int>(qtype), n_mats, w_ptrs, y_ptrs, n_out_per_mat, x_q8_1, n_in, ncols, s)) {
        for (int32_t m = 0; m < n_mats; ++m) {
            tk_dispatch_mmvq_cols(
                static_cast<int>(qtype), w_ptrs[m], x_q8_1, y_ptrs[m], n_in, n_out_per_mat[m], ncols, s);
        }
    }
    return TK_OK;
    TK_TRY_END(ctx)
}

tk_status tk_mmvq_batch(
    tk_ctx*              ctx,
    const tk_mmvq_item*  items,
    int32_t              n_items,
    tk_stream            stream) TK_NOEXCEPT {
    if (ctx == nullptr || items == nullptr) {
        return tk_null_err(ctx, "tk_mmvq_batch: null pointer argument");
    }
    if (n_items < 1 || n_items > 8) {
        return tk_arg_err(ctx, "tk_mmvq_batch: n_items must be in [1..8]");
    }
    strata::kernels::MmvqItem native_items[8]{};
    for (int32_t i = 0; i < n_items; ++i) {
        const tk_mmvq_item& src = items[i];
        if (src.w == nullptr || src.x_q8_1 == nullptr || src.y == nullptr) {
            return tk_null_err(ctx, "tk_mmvq_batch: null pointer inside tk_mmvq_item");
        }
        if (!tk_is_supported_ids_qtype(src.qtype)) {
            return tk_unsupported_err(ctx, "tk_mmvq_batch: unsupported quantization type in item");
        }
        const int32_t blk = tk_qtype_block_size(src.qtype);
        if (src.n_ids <= 0 || src.n_in <= 0 || src.n_out <= 0 || src.x_group <= 0 ||
            blk <= 0 || (src.n_in % blk) != 0) {
            return tk_arg_err(ctx, "tk_mmvq_batch: invalid item dimensions");
        }
        native_items[i].type = static_cast<int>(src.qtype);
        native_items[i].w = src.w;
        native_items[i].expert_bytes = src.expert_bytes;
        native_items[i].ids = src.ids_dev;
        native_items[i].n_ids = src.n_ids;
        native_items[i].x = src.x_q8_1;
        native_items[i].x_per_id = src.x_per_id;
        native_items[i].x_group = src.x_group;
        native_items[i].y = src.y;
        native_items[i].expert_major = src.expert_major;
        native_items[i].n_in = src.n_in;
        native_items[i].n_out = src.n_out;
    }
    if (!ctx->has_gpu) {
        return tk_hip_err(ctx, "tk_mmvq_batch: no HIP device available");
    }
    TK_TRY_BEGIN
    strata::kernels::native_mmvq_batch(native_items, n_items, tk_effective_stream(stream));
    return TK_OK;
    TK_TRY_END(ctx)
}

}  // extern "C"
