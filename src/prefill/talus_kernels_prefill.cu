// src/prefill/talus_kernels_prefill.cu - Contract v0 C ABI wrapper for stateful prefill (WP-1.1b).
//
// Copyright (c) 2026 Niko1221 and the Strata contributors
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT

#include "talus_kernels.h"
#if __has_include("strata/kernels/tk_internal.hpp")
#include "strata/kernels/tk_internal.hpp"
#define TK_PREFILL_HAS_TK_INTERNAL 1
#endif
#include "strata/prefill/tk_prefill_internal.hpp"

#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/mimo_kernels.hpp"
#include "strata/prefill/fattn_prefill.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/moe_w16.hpp"
#include "strata/prefill/tp_moe.hpp"
#include "strata/prefill/tp_tune.hpp"

#if defined(__HIPCC__)
#include "strata/hip_compat/cuda_runtime.h"
#include "strata/hip_compat/cublas_v2.h"
#ifndef STRATA_USE_HIP
#define STRATA_USE_HIP 1
#endif
#if !defined(STRATA_HIPBLASLT_AVAILABLE) && __has_include(<hipblaslt/hipblaslt.h>)
#define STRATA_HIPBLASLT_AVAILABLE 1
#endif
#else
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#ifdef STRATA_USE_HIP
#include "wmma_gemm.h"
#endif

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#pragma clang diagnostic pop
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

// Optional hooks defined in WP-1.1a (talus_kernels_decode.cu).
// Declared weak so talus_kernels_prefill.cu links cleanly both standalone and inside libtalus_kernels.so.
extern "C" {
__attribute__((weak)) tk_status tk_ctx_device(const tk_ctx* ctx, int32_t* out_device) TK_NOEXCEPT;
__attribute__((weak)) tk_status tk_ctx_arch(const tk_ctx* ctx, tk_arch* out_arch) TK_NOEXCEPT;
__attribute__((weak)) void tk_ctx_set_error(tk_ctx* ctx, const char* msg) TK_NOEXCEPT;
__attribute__((weak)) void tk_set_last_error(tk_ctx* ctx, const char* msg) TK_NOEXCEPT;
}

namespace {

constexpr size_t kDefaultWorkspaceBytes = 32u << 20;        // 32 MiB fixed hipBLAS/hipBLASLt workspace
constexpr int64_t kDefaultDequantScratchElems = 32ll << 20; // 32M FP16 elements (64 MiB) max slice scratch

thread_local std::string g_tls_prefill_error;
std::mutex g_registry_mu;
std::mutex g_mmq_q8_buf_mu;
std::set<const void*> g_mmq_q8_bufs;

bool is_registered_mmq_q8_buf(const void* xq) noexcept {
    if (!xq) return false;
    std::lock_guard<std::mutex> lock(g_mmq_q8_buf_mu);
    return g_mmq_q8_bufs.find(xq) != g_mmq_q8_bufs.end();
}

bool dequant_f16_supported(int ggml_type) noexcept {
    if (strata::kernels::dequant_bf16_supported(ggml_type)) {
        return true;
    }
    return ggml_type == 10 || ggml_type == 16 || ggml_type == 17 ||
           ggml_type == 18 || ggml_type == 21 || ggml_type == 22 ||
           ggml_type == 29;
}

#if defined(__HIPCC__)
__device__ unsigned long long g_tk_f16_nonfinite = 0;

__global__ void tk_widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + static_cast<int64_t>(blockIdx.x) * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) {
            atomicAdd(&g_tk_f16_nonfinite, 1ull);
        }
        __syncthreads();
        if (c < n) {
            y[c] = v;
        }
        __syncthreads();
    }
}

__global__ void tk_bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float f = __uint_as_float(static_cast<uint32_t>(s[i]) << 16);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));
    }
}
#endif

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtAlgoCache {
    std::mutex mu;
    std::map<HipLtCallKey, HipLtCachedAlgo> entries;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    void clear() {
        std::lock_guard<std::mutex> lock(mu);
        entries.clear();
        fallback_shapes.clear();
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

HipLtCachedAlgo resolve_hipblaslt_algo(
    tk_prefill_state& state,
    strata::prefill::hipblaslt::InputType type,
    int t,
    int n,
    int k,
    int ldy,
    float beta) {
    auto* cache = static_cast<HipLtAlgoCache*>(state.algo_cache);
    auto lt_handle = reinterpret_cast<hipblasLtHandle_t>(state.hipblaslt_handle);
    if (!cache || !lt_handle || !state.tuning_loaded) {
        return {};
    }

    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};

    std::lock_guard<std::mutex> lock(cache->mu);
    const auto found = cache->entries.find(key);
    if (found != cache->entries.end()) {
        return found->second;
    }

    HipLtCachedAlgo resolved;
    const auto* row = state.tuning_table.closest(type, n, k, ldy, t);
    if (!row) {
        return cache->entries.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type =
        type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return cache->entries.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(lt_handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() ||
        candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        return cache->entries.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(
            lt_handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c, algo, required_workspace) !=
        HIPBLAS_STATUS_SUCCESS) {
        return cache->entries.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    return cache->entries.emplace(key, resolved).first->second;
}

int hipblaslt_solution_for(
    tk_prefill_state& state,
    strata::prefill::hipblaslt::InputType type,
    int64_t t,
    int64_t n,
    int64_t k,
    int64_t ldy,
    float beta) {
    if (!state.tuning_loaded || !state.hipblaslt_handle ||
        t <= 0 || n <= 0 || k <= 0 ||
        t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) {
        return -1;
    }
    const auto resolved =
        resolve_hipblaslt_algo(state, type, static_cast<int>(t), static_cast<int>(n), static_cast<int>(k),
                               static_cast<int>(ldy), beta);
    if (!resolved.supported || resolved.workspace_bytes > state.workspace_bytes) {
        return -1;
    }
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(
    tk_prefill_state& state,
    strata::prefill::hipblaslt::InputType type,
    const uint16_t* x,
    const uint16_t* w,
    float* y,
    int64_t t,
    int64_t n,
    int64_t k,
    int64_t ldy,
    float beta,
    void* stream,
    int64_t ldx = 0) {
    if (!state.tuning_loaded || !state.hipblaslt_handle || !state.algo_cache ||
        t <= 0 || n <= 0 || k <= 0 ||
        t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) {
        return false;
    }
    auto* cache = static_cast<HipLtAlgoCache*>(state.algo_cache);
    auto lt_handle = reinterpret_cast<hipblasLtHandle_t>(state.hipblaslt_handle);

    const auto resolved =
        resolve_hipblaslt_algo(state, type, static_cast<int>(t), static_cast<int>(n), static_cast<int>(k),
                               static_cast<int>(ldy), beta);
    if (!resolved.supported || resolved.workspace_bytes > state.workspace_bytes) {
        std::lock_guard<std::mutex> lock(cache->mu);
        ++state.lt_fallbacks;
        cache->fallback_shapes.emplace(
            type, static_cast<int>(t), static_cast<int>(n), static_cast<int>(k), static_cast<int>(ldy));
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type =
        type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, static_cast<int>(t), static_cast<int>(n), static_cast<int>(k),
                   static_cast<int>(ldy), static_cast<int>(ldx))) {
        return false;
    }

    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(
        lt_handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y, desc.c, y, desc.c,
        &resolved.algo, state.workspace, state.workspace_bytes, reinterpret_cast<hipStream_t>(stream));
    if (status == HIPBLAS_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(cache->mu);
        ++state.lt_launches;
        return true;
    }

    if (beta != 0.0f) {
        throw std::runtime_error("hipBLASLt launch failed with nonzero beta (status " +
                                 std::to_string(static_cast<int>(status)) + ")");
    }

    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    std::lock_guard<std::mutex> lock(cache->mu);
    auto it = cache->entries.find(HipLtCallKey{
        type, static_cast<int>(t), static_cast<int>(n), static_cast<int>(k), static_cast<int>(ldy), beta_bits});
    if (it != cache->entries.end()) {
        it->second.supported = false;
    }
    ++state.lt_fallbacks;
    cache->fallback_shapes.emplace(
        type, static_cast<int>(t), static_cast<int>(n), static_cast<int>(k), static_cast<int>(ldy));
    return false;
}
#endif

struct PrefillStateDeleter {
    void operator()(tk_prefill_state* state) const noexcept {
        if (!state) return;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
        delete static_cast<HipLtAlgoCache*>(state->algo_cache);
        state->algo_cache = nullptr;
        if (state->hipblaslt_handle) {
            (void) hipblasLtDestroy(reinterpret_cast<hipblasLtHandle_t>(state->hipblaslt_handle));
            state->hipblaslt_handle = nullptr;
        }
#endif
        if (state->mmq_ctx) {
            delete static_cast<strata::prefill::mmq::Context*>(state->mmq_ctx);
            state->mmq_ctx = nullptr;
        }
        if (state->mmq_dense_buf) {
            (void) cudaFree(state->mmq_dense_buf);
            state->mmq_dense_buf = nullptr;
        }
        if (state->dequant_scratch) {
            (void) cudaFree(state->dequant_scratch);
            state->dequant_scratch = nullptr;
        }
        if (state->hipblas_handle) {
            (void) cublasDestroy(reinterpret_cast<cublasHandle_t>(state->hipblas_handle));
            state->hipblas_handle = nullptr;
        }
        if (state->workspace) {
            (void) cudaFree(state->workspace);
            state->workspace = nullptr;
        }
        delete state;
    }
};

using StatePtr = std::unique_ptr<tk_prefill_state, PrefillStateDeleter>;

std::unordered_map<const tk_ctx*, StatePtr>& state_registry() {
    static auto* reg = new std::unordered_map<const tk_ctx*, StatePtr>();
    return *reg;
}

void record_error(tk_ctx* ctx, std::string msg) noexcept {
    try {
        g_tls_prefill_error = msg;
        if (ctx != nullptr) {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            auto& reg = state_registry();
            auto it = reg.find(ctx);
            if (it != reg.end() && it->second) {
                it->second->last_error = msg;
            }
        }
#if defined(TK_PREFILL_HAS_TK_INTERNAL)
        tk_set_error(ctx, g_tls_prefill_error.c_str());
#endif
        if (tk_ctx_set_error != nullptr) {
            tk_ctx_set_error(ctx, g_tls_prefill_error.c_str());
        } else if (tk_set_last_error != nullptr) {
            tk_set_last_error(ctx, g_tls_prefill_error.c_str());
        }
    } catch (...) {
    }
}

tk_status translate_current_exception(tk_ctx* ctx, const char* fn_name) noexcept {
    try {
        throw;
    } catch (const std::bad_alloc& e) {
        record_error(ctx, std::string(fn_name) + ": out of memory: " + e.what());
        return TK_ERR_OUT_OF_MEMORY;
    } catch (const std::invalid_argument& e) {
        record_error(ctx, std::string(fn_name) + ": invalid argument: " + e.what());
        return TK_ERR_INVALID_ARG;
    } catch (const std::exception& e) {
        record_error(ctx, std::string(fn_name) + ": exception: " + e.what());
        return TK_ERR_EXCEPTION;
    } catch (...) {
        record_error(ctx, std::string(fn_name) + ": unknown C++ exception");
        return TK_ERR_EXCEPTION;
    }
}

tk_prefill_state* ensure_state(tk_ctx* ctx, tk_status* out_status) {
    {
        std::lock_guard<std::mutex> lock(g_registry_mu);
        auto& reg = state_registry();
        auto it = reg.find(ctx);
        if (it != reg.end() && it->second) {
            *out_status = TK_OK;
            return it->second.get();
        }
    }
    const tk_status st = tk_prefill_ctx_init(ctx);
    *out_status = st;
    if (st != TK_OK) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_registry_mu);
    auto& reg = state_registry();
    auto it = reg.find(ctx);
    return (it != reg.end()) ? it->second.get() : nullptr;
}

tk_status ensure_dequant_scratch(tk_prefill_state& state, int64_t want_elems, tk_ctx* ctx) {
    if (state.dequant_scratch_elems >= want_elems) {
        return TK_OK;
    }
    if (state.dequant_scratch != nullptr) {
        (void) cudaFree(state.dequant_scratch);
        state.dequant_scratch = nullptr;
        state.dequant_scratch_elems = 0;
    }
    uint16_t* ptr = nullptr;
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), static_cast<size_t>(want_elems) * sizeof(uint16_t));
    if (err != cudaSuccess) {
        record_error(ctx, std::string("tk_prefill_gemm: cudaMalloc dequant scratch failed: ") + cudaGetErrorString(err));
        return (err == cudaErrorMemoryAllocation) ? TK_ERR_OUT_OF_MEMORY : TK_ERR_HIP;
    }
    state.dequant_scratch = ptr;
    state.dequant_scratch_elems = want_elems;
    return TK_OK;
}

strata::prefill::mmq::Context* ensure_mmq_context(tk_prefill_state& state) {
    if (!state.mmq_ctx) {
        state.mmq_ctx = new strata::prefill::mmq::Context();
    }
    return static_cast<strata::prefill::mmq::Context*>(state.mmq_ctx);
}

tk_status run_f16_inplace(
    tk_prefill_state& state,
    const uint16_t* X,
    const uint16_t* W,
    float* Y,
    int64_t T,
    int64_t N,
    int64_t K,
    int64_t ldy,
    tk_stream stream,
    tk_ctx* ctx) {
#if defined(__HIPCC__)
    auto blas_h = reinterpret_cast<cublasHandle_t>(state.hipblas_handle);
    if (!blas_h) {   // (STRATA_PREFILL_NO_BLAS=1)
        record_error(ctx, "tk_prefill_gemm: hipBLAS handle is not initialized");
        return TK_ERR_HIPBLAS;
    }
    (void) cublasSetStream(blas_h, reinterpret_cast<cudaStream_t>(stream));
    const float one = 1.0f;
    const float zero = 0.0f;
    const cublasStatus_t st = cublasGemmEx(
        blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(T), static_cast<int>(K),
        &one, W, CUDA_R_16F, static_cast<int>(K),
        X, CUDA_R_16F, static_cast<int>(K),
        &zero, Y, CUDA_R_16F, static_cast<int>(2 * ldy),
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        record_error(ctx, "tk_prefill_gemm: cublasGemmEx f16_inplace failed with status " +
                              std::to_string(static_cast<int>(st)));
        return TK_ERR_HIPBLAS;
    }
    tk_widen_rows_f16<<<static_cast<unsigned>(T), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        Y, N, ldy, 0);
    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        record_error(ctx, std::string("tk_prefill_gemm: tk_widen_rows_f16 failed: ") +
                              cudaGetErrorString(launch_err));
        return TK_ERR_HIP;
    }
    return TK_OK;
#else
    (void) state; (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) stream; (void) ctx;
    return TK_ERR_UNSUPPORTED;
#endif
}

tk_status run_f16_gemm(
    tk_prefill_state& state,
    const uint16_t* X,
    const uint16_t* W,
    float* Y,
    int64_t T,
    int64_t N,
    int64_t K,
    int64_t ldy,
    int64_t ldx,
    float beta,
    tk_stream stream,
    tk_ctx* ctx) {
    if (T <= 0 || N <= 0) {
        return TK_OK;
    }
    if (ldy <= 0) {
        ldy = N;
    }
    if (ldx <= K) {
        ldx = 0;
    }
#ifdef STRATA_USE_HIP
    static const bool wmma_on = [] {
        const char* v = std::getenv("STRATA_WMMA_GEMM");
        return v && v[0] != 0 && v[0] != '0';
    }();
    static const bool pf_on = [] {
        const char* v = std::getenv("STRATA_PF_GEMM");
        return v && v[0] == '1';
    }();
    static const int64_t pf_min_t = [] {
        const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T");
        return v ? static_cast<int64_t>(std::atoll(v)) : int64_t(0);
    }();
    if (pf_on && T >= pf_min_t && ldx == 0 && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream)) {
        return TK_OK;
    }
    if (wmma_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream)) {
        return TK_OK;
    }
#endif
#if defined(__HIPCC__)
    if (state.f16_io && beta == 0.0f && ldx == 0) {
        return run_f16_inplace(state, X, W, Y, T, N, K, ldy, stream, ctx);
    }
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(state, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy, beta, stream, ldx)) {
        return TK_OK;
    }
#endif
    auto blas_h = reinterpret_cast<cublasHandle_t>(state.hipblas_handle);
    if (!blas_h) {
        record_error(ctx, "tk_prefill_gemm: hipBLAS handle is not initialized");
        return TK_ERR_HIPBLAS;
    }
    (void) cublasSetStream(blas_h, reinterpret_cast<cudaStream_t>(stream));
    const float alpha = 1.0f;
    const cublasStatus_t st = cublasGemmEx(
        blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(T), static_cast<int>(K),
        &alpha, W, CUDA_R_16F, static_cast<int>(K),
        X, CUDA_R_16F, static_cast<int>(ldx ? ldx : K),
        &beta, Y, CUDA_R_32F, static_cast<int>(ldy),
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        record_error(ctx, "tk_prefill_gemm: cublasGemmEx f16 failed with status " +
                              std::to_string(static_cast<int>(st)));
        return TK_ERR_HIPBLAS;
    }
    return TK_OK;
}

tk_status run_bf16_gemm(
    tk_prefill_state& state,
    const uint16_t* X,
    const uint16_t* W,
    float* Y,
    int64_t T,
    int64_t N,
    int64_t K,
    int64_t ldy,
    int64_t ldx,
    float beta,
    tk_stream stream,
    tk_ctx* ctx) {
    if (T <= 0 || N <= 0) {
        return TK_OK;
    }
    if (ldy <= 0) {
        ldy = N;
    }
    if (ldx <= K) {
        ldx = 0;
    }
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    static const bool hcd_exact = [] {
        const char* v = std::getenv("STRATA_HCD_EXACT");
        return v && v[0] == '1';
    }();
    if (hcd_exact && N == 320 && K == 10240 && beta == 0.0f && state.tuning_loaded) {
        const int id = hipblaslt_solution_for(
            state, strata::prefill::hipblaslt::InputType::bf16, T, N, K, ldy, 0.0f);
        if (strata_pf_hcdown_exact_solution(id, state.hipblaslt_version) &&
            strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream)) {
            return TK_OK;
        }
    }
#endif
#ifdef STRATA_USE_HIP
    static const bool wmma_on = [] {
        const char* v = std::getenv("STRATA_WMMA_GEMM");
        return v && v[0] != 0 && v[0] != '0';
    }();
    static const bool bf16_on = [] {
        const char* v = std::getenv("STRATA_WMMA_BF16");
        return !v || (v[0] != 0 && v[0] != '0');
    }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream)) {
        return TK_OK;
    }
#endif
#if defined(__HIPCC__)
    if (state.f16_io) {
        if (beta != 0.0f) {
            record_error(ctx, "tk_prefill_gemm: accumulating BF16 GEMM unsupported on FP16 prompt path");
            return TK_ERR_UNSUPPORTED;
        }
        if (state.dequant_scratch_elems == 0) {
            const int64_t want_elems = std::min<int64_t>(N * K, std::max<int64_t>(K * 64, kDefaultDequantScratchElems));
            if (const tk_status sc_st = ensure_dequant_scratch(state, want_elems, ctx); sc_st != TK_OK) {
                return sc_st;
            }
        }
        const int64_t rows = std::min<int64_t>(N, state.dequant_scratch_elems / K);
        if (rows <= 0) {
            record_error(ctx, "tk_prefill_gemm: dequant scratch too small for K");
            return TK_ERR_OUT_OF_MEMORY;
        }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n_slice = std::min<int64_t>(rows, N - r0);
            tk_bf16_to_f16_rows<<<1024, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
                W + r0 * K, reinterpret_cast<__half*>(state.dequant_scratch), n_slice * K);
            const tk_status st =
                run_f16_inplace(state, X, state.dequant_scratch, Y + r0, T, n_slice, K, ldy, stream, ctx);
            if (st != TK_OK) {
                return st;
            }
        }
        return TK_OK;
    }
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(state, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy, beta, stream, ldx)) {
        return TK_OK;
    }
#endif
    auto blas_h = reinterpret_cast<cublasHandle_t>(state.hipblas_handle);
    if (!blas_h) {
        record_error(ctx, "tk_prefill_gemm: hipBLAS handle is not initialized");
        return TK_ERR_HIPBLAS;
    }
    (void) cublasSetStream(blas_h, reinterpret_cast<cudaStream_t>(stream));
    const float alpha = 1.0f;
    const cublasStatus_t st = cublasGemmEx(
        blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(T), static_cast<int>(K),
        &alpha, W, CUDA_R_16BF, static_cast<int>(K),
        X, CUDA_R_16BF, static_cast<int>(ldx ? ldx : K),
        &beta, Y, CUDA_R_32F, static_cast<int>(ldy),
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) {
        record_error(ctx, "tk_prefill_gemm: cublasGemmEx bf16 failed with status " +
                              std::to_string(static_cast<int>(st)));
        return TK_ERR_HIPBLAS;
    }
    return TK_OK;
}

bool try_dense_native_mmq(
    tk_prefill_state& state,
    const uint16_t* X,
    int ggml_type,
    const void* W,
    float* Y,
    int64_t T,
    int64_t N,
    int64_t K,
    int64_t ldy,
    tk_stream stream) {
#if defined(__HIPCC__)
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024;
    constexpr int64_t kMaxK = 32768;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || state.mmq_dense_failed || !mmq::built() || !mmq::fits(ggml_type, N)) {
        return false;
    }
    if (K > kMaxK || (K % 256) != 0) {
        return false;
    }
    int64_t R = state.dequant_scratch_elems / (2 * K);
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) {
        return false;
    }
    if (ldy <= 0) {
        ldy = N;
    }

    const auto up256 = [](size_t v) { return (v + 255u) & ~size_t(255u); };
    // offsets follow the buffer's capacity (tk_prefill_mmq may have grown it past kRows)
    const auto ident_off_for = [&](int64_t rows) { return up256(mmq::q8_bytes(rows, kMaxK)); };
    const auto bounds_off_for = [&](int64_t rows) {
        return ident_off_for(rows) + up256(static_cast<size_t>(rows) * sizeof(int32_t));
    };
    if (!state.mmq_dense_buf || state.mmq_dense_cap_rows < kRows) {
        if (state.mmq_dense_buf) {
            (void) cudaFree(state.mmq_dense_buf);
            state.mmq_dense_buf = nullptr;
        }
        if (cudaMalloc(&state.mmq_dense_buf, bounds_off_for(kRows) + 256u) != cudaSuccess) {
            (void) cudaGetLastError();
            state.mmq_dense_failed = true;
            return false;
        }
        state.mmq_dense_cap_rows = kRows;
        mmq::iota(reinterpret_cast<int32_t*>( static_cast<uint8_t*>(state.mmq_dense_buf) + ident_off_for(kRows)),
                  kRows, stream);
    }
    const size_t ident_off = ident_off_for(state.mmq_dense_cap_rows);
    const size_t bounds_off = bounds_off_for(state.mmq_dense_cap_rows);

    void* xq = state.mmq_dense_buf;
    const int32_t* ident =
        reinterpret_cast<const int32_t*>(static_cast<const uint8_t*>(state.mmq_dense_buf) + ident_off);
    int32_t* bounds = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(state.mmq_dense_buf) + bounds_off);
    float* xf = reinterpret_cast<float*>(state.dequant_scratch);
    auto* mmq_ctx = ensure_mmq_context(state);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = std::min<int64_t>(R, T - r0);
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream);
        mmq::quantize(xf, nullptr, xq, ggml_type, K, K, rows, stream);
        mmq::set_bounds(bounds, static_cast<int32_t>(rows), stream);
        mmq::Product p;
        p.w = W;
        p.type = ggml_type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(ggml_type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        mmq_ctx->run(p, stream);
    }
    return true;
#else
    (void) state; (void) X; (void) ggml_type; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy; (void) stream;
    return false;
#endif
}

tk_status run_native_gemm(
    tk_prefill_state& state,
    const uint16_t* X,
    int ggml_type,
    const void* W_blocks,
    float* Y,
    int64_t T,
    int64_t N,
    int64_t K,
    int64_t ldy,
    int64_t ldx,
    float beta,
    tk_stream stream,
    tk_ctx* ctx) {
    if (!dequant_f16_supported(ggml_type)) {
        record_error(ctx, "tk_prefill_gemm: unsupported quantized weight type " + std::to_string(ggml_type));
        return TK_ERR_UNSUPPORTED;
    }
    if (T <= 0 || N <= 0) {
        return TK_OK;
    }
    if (ldy <= 0) {
        ldy = N;
    }

    if (state.dequant_scratch_elems == 0) {
        const int64_t padded_k = K + 64;
        const int64_t want_elems =
            std::min<int64_t>(N * padded_k, std::max<int64_t>(padded_k * 64, kDefaultDequantScratchElems));
        if (const tk_status sc_st = ensure_dequant_scratch(state, want_elems, ctx); sc_st != TK_OK) {
            return sc_st;
        }
    }

#ifdef STRATA_USE_HIP
    static const bool pad_on = [] {
        const char* v = std::getenv("STRATA_PF_PAD");
        return v && v[0] == '1';
    }();
    static const bool pf_on = [] {
        const char* v = std::getenv("STRATA_PF_GEMM");
        return v && v[0] == '1';
    }();
    static const int64_t pf_min_t = [] {
        const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T");
        return v ? static_cast<int64_t>(std::atoll(v)) : int64_t(0);
    }();
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && (K % 32) == 0 &&
        N * (K + 64) <= state.dequant_scratch_elems) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, state.dequant_scratch, stream) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, state.dequant_scratch, ldw, Y, T, N, K, ldy, beta, stream)) {
            return TK_OK;
        }
    }
#endif

    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        const int64_t rows = state.dequant_scratch_elems / K;
        bool ok = rows > 0;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n_slice = std::min<int64_t>(rows, N - r0);
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n_slice, K, state.dequant_scratch, stream);
            ok = strata_pf_gemm_f16_ld(X, ldx, state.dequant_scratch, K, Y + r0, T, n_slice, K, ldy, beta, stream);
        }
        if (ok) {
            return TK_OK;
        }
#endif
        return run_f16_gemm(state, X, state.dequant_scratch, Y, T, N, K, ldy, ldx, beta, stream, ctx);
    }

    if (beta == 0.0f && (ldx == 0 || ldx == K) &&
        try_dense_native_mmq(state, X, ggml_type, W_blocks, Y, T, N, K, ldy, stream)) {
        return TK_OK;
    }

    if (N * K > state.dequant_scratch_elems) {
        const int64_t rows = state.dequant_scratch_elems / K;
        if (rows <= 0) {
            record_error(ctx, "tk_prefill_gemm: dequant scratch too small for K=" + std::to_string(K));
            return TK_ERR_OUT_OF_MEMORY;
        }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n_slice = std::min<int64_t>(rows, N - r0);
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n_slice, K, state.dequant_scratch, stream);
            const tk_status st =
                run_f16_gemm(state, X, state.dequant_scratch, Y + r0, T, n_slice, K, ldy, 0, beta, stream, ctx);
            if (st != TK_OK) {
                return st;
            }
        }
        return TK_OK;
    }

    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, state.dequant_scratch, stream);
    return run_f16_gemm(state, X, state.dequant_scratch, Y, T, N, K, ldy, 0, beta, stream, ctx);
}

}  // namespace

tk_prefill_state* tk_prefill_get_state(tk_ctx* ctx) noexcept {
    if (!ctx) return nullptr;
    try {
        std::lock_guard<std::mutex> lock(g_registry_mu);
        auto& reg = state_registry();
        auto it = reg.find(ctx);
        return (it != reg.end()) ? it->second.get() : nullptr;
    } catch (...) {
        return nullptr;
    }
}

const tk_prefill_state* tk_prefill_get_state_const(const tk_ctx* ctx) noexcept {
    if (!ctx) return nullptr;
    try {
        std::lock_guard<std::mutex> lock(g_registry_mu);
        auto& reg = state_registry();
        auto it = reg.find(ctx);
        return (it != reg.end()) ? it->second.get() : nullptr;
    } catch (...) {
        return nullptr;
    }
}

extern "C" const char* tk_prefill_last_error(const tk_ctx* ctx) TK_NOEXCEPT {
    try {
        if (ctx != nullptr) {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            auto& reg = state_registry();
            auto it = reg.find(ctx);
            if (it != reg.end() && it->second && !it->second->last_error.empty()) {
                return it->second->last_error.c_str();
            }
        }
        return g_tls_prefill_error.c_str();
    } catch (...) {
        return "tk_prefill_last_error: internal exception";
    }
}

extern "C" tk_status tk_prefill_ctx_init(tk_ctx* ctx) TK_NOEXCEPT {
    if (ctx == nullptr) {
        record_error(nullptr, "tk_prefill_ctx_init: ctx is NULL");
        return TK_ERR_NULL_PTR;
    }
    try {
        StatePtr state(new tk_prefill_state());

        int32_t dev = 0;
#if defined(TK_PREFILL_HAS_TK_INTERNAL)
        dev = ctx->device;
#endif
        if (tk_ctx_device != nullptr && tk_ctx_device(ctx, &dev) == TK_OK) {
            state->device_id = dev;
        } else {
            state->device_id = dev;
        }

        tk_arch requested_arch = TK_ARCH_GFX1151;
#if defined(TK_PREFILL_HAS_TK_INTERNAL)
        requested_arch = ctx->arch;
#endif
        if (tk_ctx_arch != nullptr) {
            tk_arch queried_arch = requested_arch;
            if (tk_ctx_arch(ctx, &queried_arch) == TK_OK) {
                requested_arch = queried_arch;
            }
        }
        if (requested_arch == TK_ARCH_OTHER) {
            state->arch = TK_ARCH_OTHER;
            state->gcn_arch_name.clear();
        } else {
            state->arch = requested_arch;
            state->gcn_arch_name = strata::prefill::tk::arch_name(requested_arch);
        }

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
        state->algo_cache = new HipLtAlgoCache();
#if defined(HIPBLASLT_VERSION_MAJOR) && defined(HIPBLASLT_VERSION_MINOR) && defined(HIPBLASLT_VERSION_PATCH)
        state->hipblaslt_version =
            HIPBLASLT_VERSION_MAJOR * 100000 + HIPBLASLT_VERSION_MINOR * 100 + HIPBLASLT_VERSION_PATCH;
#endif
#endif

        int dev_count = 0;
        const cudaError_t count_err = cudaGetDeviceCount(&dev_count);
        if (count_err != cudaSuccess || dev_count <= 0) {
            (void) cudaGetLastError();
            state->has_gpu = false;
        } else {
            if (state->device_id < 0 || state->device_id >= dev_count) {
                record_error(ctx, "tk_prefill_ctx_init: invalid device ordinal " +
                                      std::to_string(state->device_id));
                return TK_ERR_HIP;
            }
            const cudaError_t set_err = cudaSetDevice(state->device_id);
            if (set_err != cudaSuccess) {
                record_error(ctx, std::string("tk_prefill_ctx_init: cudaSetDevice failed: ") +
                                      cudaGetErrorString(set_err));
                return TK_ERR_HIP;
            }

#if defined(__HIPCC__)
            hipDeviceProp_t props{};
            if (hipGetDeviceProperties(&props, state->device_id) == hipSuccess && props.gcnArchName[0] != '\0') {
                const std::string detected_gcn = strata::prefill::tk::strip_gcn_arch_suffix(props.gcnArchName);
                if (requested_arch == TK_ARCH_OTHER || state->gcn_arch_name.empty()) {
                    state->gcn_arch_name = detected_gcn;
                    state->arch = strata::prefill::tk::arch_from_gcn_name(detected_gcn);
                }
            }
#endif

            // STRATA_PREFILL_NO_BLAS=1 skips the hipBLAS / hipBLASLt handles and the
            // 32 MiB workspace for engines that never call the BLAS paths (they then return TK_ERR_HIPBLAS).
            static const bool no_blas = [] {
                const char* v = std::getenv("STRATA_PREFILL_NO_BLAS");
                return v != nullptr && v[0] == '1';
            }();
            cublasHandle_t blas_h = nullptr;
            if (no_blas) {
                state->f16_io = strata::prefill::prompt_f16();
                state->has_gpu = true;
            } else if (cublasCreate(&blas_h) != CUBLAS_STATUS_SUCCESS) {
                record_error(ctx, "tk_prefill_ctx_init: cublasCreate/hipblasCreate failed");
                return TK_ERR_HIPBLAS;
            }
            if (!no_blas) {
            state->hipblas_handle = blas_h;

            void* ws = nullptr;
            const cudaError_t ws_err = cudaMalloc(&ws, kDefaultWorkspaceBytes);
            if (ws_err != cudaSuccess) {
                record_error(ctx, std::string("tk_prefill_ctx_init: failed to allocate 32 MiB workspace: ") +
                                      cudaGetErrorString(ws_err));
                return (ws_err == cudaErrorMemoryAllocation) ? TK_ERR_OUT_OF_MEMORY : TK_ERR_HIP;
            }
            state->workspace = ws;
            state->workspace_bytes = kDefaultWorkspaceBytes;
            (void) cublasSetWorkspace(blas_h, ws, kDefaultWorkspaceBytes);
            (void) cublasSetMathMode(blas_h, CUBLAS_DEFAULT_MATH);

            state->f16_io = strata::prefill::prompt_f16();

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
            hipblasLtHandle_t lt_h = nullptr;
            if (hipblasLtCreate(&lt_h) == HIPBLAS_STATUS_SUCCESS) {
                state->hipblaslt_handle = lt_h;
                int lt_ver = 0;
                if (hipblasLtGetVersion(lt_h, &lt_ver) == HIPBLAS_STATUS_SUCCESS && lt_ver > 0) {
                    state->hipblaslt_version = lt_ver;
                }
            }
#endif
            state->has_gpu = true;
            }
        }

        if (const char* env_ver = std::getenv("STRATA_HIPBLASLT_VERSION")) {
            const int parsed_ver = std::atoi(env_ver);
            if (parsed_ver > 0) {
                state->hipblaslt_version = parsed_ver;
            }
        }

        const char* env_tuning = std::getenv("STRATA_HIPBLASLT_TUNING");
        if (env_tuning != nullptr && env_tuning[0] != '\0' && state->hipblaslt_version > 0) {
            (void) state->load_tuning_host(env_tuning);
        }

        {
            std::lock_guard<std::mutex> lock(g_registry_mu);
#if defined(TK_PREFILL_HAS_TK_INTERNAL)
            ctx->prefill_state = state.get();
#endif
            state_registry()[ctx] = std::move(state);
        }
        return TK_OK;
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_ctx_init");
    }
}

extern "C" void tk_prefill_ctx_destroy(tk_ctx* ctx) TK_NOEXCEPT {
    if (ctx == nullptr) {
        return;
    }
    try {
        StatePtr removed;
        {
            std::lock_guard<std::mutex> lock(g_registry_mu);
#if defined(TK_PREFILL_HAS_TK_INTERNAL)
            ctx->prefill_state = nullptr;
#endif
            auto& reg = state_registry();
            auto it = reg.find(ctx);
            if (it != reg.end()) {
                removed = std::move(it->second);
                reg.erase(it);
            }
        }
    } catch (...) {
    }
}

extern "C" tk_status tk_prefill_ctx_load_tuning(tk_ctx* ctx, const char* dir_or_file) TK_NOEXCEPT {
    if (ctx == nullptr) {
        record_error(nullptr, "tk_prefill_ctx_load_tuning: ctx is NULL");
        return TK_ERR_NULL_PTR;
    }
    try {
        tk_prefill_state* state = tk_prefill_get_state(ctx);
        if (state == nullptr) {
            tk_status init_st = TK_OK;
            state = ensure_state(ctx, &init_st);
            if (state == nullptr) {
                return init_st;
            }
        }

#if defined(TK_PREFILL_HAS_TK_INTERNAL)
        if (ctx->arch != TK_ARCH_OTHER && ctx->arch != state->arch) {
            state->arch = ctx->arch;
            state->gcn_arch_name = strata::prefill::tk::arch_name(ctx->arch);
        }
#endif
        if (tk_ctx_arch != nullptr) {
            tk_arch queried_arch = state->arch;
            if (tk_ctx_arch(ctx, &queried_arch) == TK_OK &&
                queried_arch != TK_ARCH_OTHER && queried_arch != state->arch) {
                state->arch = queried_arch;
                state->gcn_arch_name = strata::prefill::tk::arch_name(queried_arch);
            }
        }
        if (const char* env_ver = std::getenv("STRATA_HIPBLASLT_VERSION")) {
            const int parsed_ver = std::atoi(env_ver);
            if (parsed_ver > 0) {
                state->hipblaslt_version = parsed_ver;
            }
        }

        const tk_status st = state->load_tuning_host(dir_or_file);
        if (st == TK_OK) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
            if (state->algo_cache != nullptr) {
                static_cast<HipLtAlgoCache*>(state->algo_cache)->clear();
            }
#endif
        } else {
            record_error(ctx, state->last_error);
        }
        return st;
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_ctx_load_tuning");
    }
}

extern "C" tk_status tk_prefill_gemm(
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
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr || w == nullptr || x == nullptr || y == nullptr) {
        record_error(ctx, "tk_prefill_gemm: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    if (n_tok < 0 || n_out <= 0 || n_in <= 0 ||
        (ldy != 0 && ldy < n_out) ||
        (ldx != 0 && ldx < n_in) ||
        !std::isfinite(beta)) {
        record_error(ctx, "tk_prefill_gemm: invalid dimensions or non-finite beta");
        return TK_ERR_INVALID_ARG;
    }
    if (n_tok == 0) {
        return TK_OK;
    }

    try {
        tk_status st = TK_OK;
        tk_prefill_state* state = ensure_state(ctx, &st);
        if (!state) {
            return st;
        }

        const auto* x_u16 = static_cast<const uint16_t*>(x);
        if (w_qtype == TK_QTYPE_BF16) {
            if (x_qtype != TK_QTYPE_BF16 && !(state->f16_io && x_qtype == TK_QTYPE_F16)) {
                record_error(ctx, "tk_prefill_gemm: BF16 weights require BF16 activations");
                return TK_ERR_INVALID_ARG;
            }
            if (!state->has_gpu) {
                record_error(ctx, "tk_prefill_gemm: no HIP GPU device initialized on context");
                return TK_ERR_HIP;
            }
            return run_bf16_gemm(
                *state, x_u16, static_cast<const uint16_t*>(w), y,
                n_tok, n_out, n_in, ldy, ldx, beta, stream, ctx);
        }

        if (w_qtype == TK_QTYPE_F16) {
            if (x_qtype != TK_QTYPE_F16) {
                record_error(ctx, "tk_prefill_gemm: F16 weights require F16 activations");
                return TK_ERR_INVALID_ARG;
            }
            if (!state->has_gpu) {
                record_error(ctx, "tk_prefill_gemm: no HIP GPU device initialized on context");
                return TK_ERR_HIP;
            }
            return run_f16_gemm(
                *state, x_u16, static_cast<const uint16_t*>(w), y,
                n_tok, n_out, n_in, ldy, ldx, beta, stream, ctx);
        }

        if (x_qtype != TK_QTYPE_F16) {
            record_error(ctx, "tk_prefill_gemm: quantized weights require F16 activations");
            return TK_ERR_INVALID_ARG;
        }
        if (!dequant_f16_supported(static_cast<int>(w_qtype))) {
            record_error(ctx, "tk_prefill_gemm: unsupported quantized weight type " +
                                  std::to_string(static_cast<int>(w_qtype)));
            return TK_ERR_UNSUPPORTED;
        }
        if (!state->has_gpu) {
            record_error(ctx, "tk_prefill_gemm: no HIP GPU device initialized on context");
            return TK_ERR_HIP;
        }
        return run_native_gemm(
            *state, x_u16, static_cast<int>(w_qtype), w, y,
            n_tok, n_out, n_in, ldy, ldx, beta, stream, ctx);
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_gemm");
    }
}

extern "C" tk_status tk_prefill_mmq(
    tk_ctx* ctx,
    tk_qtype w_qtype,
    const void* w_blocks,
    const void* x_q8_1,
    float* y,
    int64_t n_tok,
    int64_t n_out,
    int64_t n_in,
    int64_t ldy,
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr || w_blocks == nullptr || x_q8_1 == nullptr || y == nullptr) {
        record_error(ctx, "tk_prefill_mmq: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    if (n_tok < 0 || n_out <= 0 || n_in <= 0 || (n_in % 256) != 0 || (ldy != 0 && ldy < n_out)) {
        record_error(ctx, "tk_prefill_mmq: invalid dimensions (n_in must be a multiple of 256)");
        return TK_ERR_INVALID_ARG;
    }
    if (n_tok == 0) {
        return TK_OK;
    }

    try {
        namespace mmq = strata::prefill::mmq;
        const int ggml_t = static_cast<int>(w_qtype);
        if (!mmq::built() || !mmq::fits(ggml_t, n_out)) {
            record_error(ctx, "tk_prefill_mmq: weight type or matrix shape unsupported by MMQ");
            return TK_ERR_UNSUPPORTED;
        }

        tk_status st = TK_OK;
        tk_prefill_state* state = ensure_state(ctx, &st);
        if (!state) {
            return st;
        }
        if (!state->has_gpu) {
            record_error(ctx, "tk_prefill_mmq: no HIP GPU device initialized on context");
            return TK_ERR_HIP;
        }

        if (ldy <= 0) {
            ldy = n_out;
        }

        constexpr int64_t kMaxK = 32768;
        const auto up256 = [](size_t v) { return (v + 255u) & ~size_t(255u); };
        const int64_t need_rows = std::max<int64_t>(n_tok, 1024);
        // the layout (ident / bounds offsets) follows the buffer's capacity, not this call's rows: a smaller call
        // after a larger one must find the identity rows where the allocating call wrote them
        const auto ident_off_for = [&](int64_t rows) { return up256(mmq::q8_bytes(rows, kMaxK)); };
        const auto bounds_off_for = [&](int64_t rows) {
            return ident_off_for(rows) + up256(static_cast<size_t>(rows) * sizeof(int32_t));
        };
        if (!state->mmq_dense_buf || state->mmq_dense_cap_rows < need_rows) {
            if (state->mmq_dense_buf) {
                (void) cudaFree(state->mmq_dense_buf);
                state->mmq_dense_buf = nullptr;
                state->mmq_dense_cap_rows = 0;
            }
            const cudaError_t alloc_err = cudaMalloc(&state->mmq_dense_buf, bounds_off_for(need_rows) + 256u);
            if (alloc_err != cudaSuccess) {
                record_error(ctx, std::string("tk_prefill_mmq: cudaMalloc failed: ") + cudaGetErrorString(alloc_err));
                return (alloc_err == cudaErrorMemoryAllocation) ? TK_ERR_OUT_OF_MEMORY : TK_ERR_HIP;
            }
            state->mmq_dense_cap_rows = need_rows;
            mmq::iota(
                reinterpret_cast<int32_t*>(static_cast<uint8_t*>(state->mmq_dense_buf) + ident_off_for(need_rows)),
                need_rows,
                stream);
        }
        const size_t ident_off = ident_off_for(state->mmq_dense_cap_rows);
        const size_t bounds_off = bounds_off_for(state->mmq_dense_cap_rows);

        const int32_t* ident =
            reinterpret_cast<const int32_t*>(static_cast<const uint8_t*>(state->mmq_dense_buf) + ident_off);
        int32_t* bounds =
            reinterpret_cast<int32_t*>(static_cast<uint8_t*>(state->mmq_dense_buf) + bounds_off);
        mmq::set_bounds(bounds, static_cast<int32_t>(n_tok), stream);

        mmq::Product p;
        p.w = w_blocks;
        p.type = ggml_t;
        p.w_rows = n_out;
        p.w_cols = n_in;
        p.expert_bytes = mmq::matrix_bytes(ggml_t, n_out, n_in);
        p.n = 1;
        p.xq = x_q8_1;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = n_tok;
        p.max_rows = n_tok;
        p.dst = y;
        p.ld_dst = ldy;

        // the GGUF-direct int8 WMMA kernel (tp_moe with one expert) - the same output bits as MMQ, x1.06-1.44
        // at the GLM / Laguna dense shapes (talus-pf-bench moe E 1).
        // TALUS_PF_DENSE_MOE=0 (or TALUS_PF_MOE=0): MMQ only.
        static const bool use_dense_moe = [] {
            const char* v = std::getenv("TALUS_PF_DENSE_MOE");
            const char* m = std::getenv("TALUS_PF_MOE");
            return (v == nullptr || v[0] != '0') && (m == nullptr || m[0] != '0');
        }();
        if (use_dense_moe && talus::prefill::tp_gfx11_wmma() &&
            talus::prefill::moe_supported(ggml_t, (int) n_out, (int) n_in)) {
            talus::prefill::MoeProduct mp;
            mp.w = w_blocks;
            mp.type = ggml_t;
            mp.expert_bytes = p.expert_bytes;
            mp.n = (int) n_out;
            mp.k = (int) n_in;
            mp.n_experts = 1;
            mp.xq = x_q8_1;
            mp.bounds = bounds;
            mp.ids = nullptr;   // dst row = activation row (MMQ's identity ids)
            mp.total_rows = n_tok;
            mp.max_rows = n_tok;
            mp.dst = y;
            mp.ld_dst = ldy;
            if (talus::prefill::moe_gemm(mp, stream)) return TK_OK;
        }

        if (use_dense_moe && talus::prefill::tp_gfx11_wmma() && ggml_t == (int) TK_QTYPE_Q6_K) {   // rb-glm (opt-in TP_Q6K=1)
            talus::prefill::MoeProduct mp;
            mp.w = w_blocks;
            mp.type = ggml_t;
            mp.expert_bytes = p.expert_bytes;
            mp.n = (int) n_out;
            mp.k = (int) n_in;
            mp.n_experts = 1;
            mp.xq = x_q8_1;
            mp.bounds = bounds;
            mp.ids = nullptr;
            mp.total_rows = n_tok;
            mp.max_rows = n_tok;
            mp.dst = y;
            mp.ld_dst = ldy;
            if (talus::prefill::moe_gemm_q6k_dense(mp, stream)) return TK_OK;
        }
        auto* mmq_ctx = ensure_mmq_context(*state);
        mmq_ctx->run(p, stream);
        return TK_OK;
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_mmq");
    }
}

extern "C" tk_status tk_prefill_moe_mmq(
    tk_ctx* ctx,
    const tk_moe_mmq_product* product,
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr || product == nullptr) {
        record_error(ctx, "tk_prefill_moe_mmq: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    if (product->n_experts < 0 || product->w_rows <= 0 || product->w_cols <= 0 ||
        product->total_rows < 0 || product->max_rows < 0 ||
        product->ld_dst < product->w_rows) {
        record_error(ctx, "tk_prefill_moe_mmq: invalid product dimensions");
        return TK_ERR_INVALID_ARG;
    }
    if (product->n_experts == 0 || product->max_rows == 0 || product->total_rows == 0) {
        return TK_OK;
    }
    if (product->w == nullptr || product->x_q8_1 == nullptr ||
        product->bounds_dev == nullptr || product->ids_dev == nullptr || product->dst == nullptr) {
        record_error(ctx, "tk_prefill_moe_mmq: NULL device buffer in product descriptor");
        return TK_ERR_NULL_PTR;
    }

    try {
        namespace mmq = strata::prefill::mmq;
        const int ggml_t = static_cast<int>(product->qtype);
        const bool is_mmq_layout = (product->n_experts <= 1) || is_registered_mmq_q8_buf(product->x_q8_1);
        const bool mimo_iq =
            !is_mmq_layout && strata::kernels::mimo_is_iq_qtype(ggml_t) && ((product->w_cols % 256) == 0);
        // IQ1_M has no MMQ path; with MMQ-layout activations (the IQ2_S / IQ2_XS D4 layout) and gfx11 it runs on
        // tp_moe (exact int8 representation of the IQ1_M weights, mode 1); otherwise unsupported as before
        if (ggml_t == (int) TK_QTYPE_IQ1_M && is_mmq_layout && product->n_experts > 1 && product->y_scale_dev == nullptr &&
            talus::prefill::tp_gfx11_wmma() &&
            talus::prefill::moe_supported(ggml_t, (int) product->w_rows, (int) product->w_cols)) {
            talus::prefill::MoeProduct mp;
            mp.w = product->w;
            mp.type = ggml_t;
            mp.expert_bytes = product->expert_bytes;
            mp.n = (int) product->w_rows;
            mp.k = (int) product->w_cols;
            mp.n_experts = product->n_experts;
            mp.xq = product->x_q8_1;
            mp.bounds = product->bounds_dev;
            mp.ids = product->ids_dev;
            mp.total_rows = product->total_rows;
            mp.max_rows = product->max_rows;
            mp.dst = product->dst;
            mp.ld_dst = product->ld_dst;
            if (talus::prefill::moe_gemm(mp, stream)) return TK_OK;
        }
        // IQ1_M on plain q8_1 rows (MiMo's IQ1_M gate / up): int8 WMMA with the dp4a kernel's output bits
        // (tp_moe.cu k_moe1m; TP_MOE_IQ1M_X=0 keeps the dp4a kernel)
        if (ggml_t == (int) TK_QTYPE_IQ1_M && !is_mmq_layout && product->n_experts > 1 && product->y_scale_dev == nullptr &&
            talus::prefill::moe_iq1m_q8_1_supported(ggml_t, (int) product->w_rows, (int) product->w_cols)) {
            talus::prefill::MoeProduct mp;
            mp.w = product->w;
            mp.type = ggml_t;
            mp.expert_bytes = product->expert_bytes;
            mp.n = (int) product->w_rows;
            mp.k = (int) product->w_cols;
            mp.n_experts = product->n_experts;
            mp.xq = product->x_q8_1;
            mp.bounds = product->bounds_dev;
            mp.ids = product->ids_dev;
            mp.total_rows = product->total_rows;
            mp.max_rows = product->max_rows;
            mp.dst = product->dst;
            mp.ld_dst = product->ld_dst;
            if (talus::prefill::moe_gemm_iq1m_q8_1(mp, stream)) return TK_OK;
        }
        if (!mimo_iq && (!mmq::built() || !mmq::fits(ggml_t, product->w_rows))) {
            record_error(ctx, "tk_prefill_moe_mmq: quantization type or matrix shape unsupported by MMQ");
            return TK_ERR_UNSUPPORTED;
        }

        tk_status st = TK_OK;
        tk_prefill_state* state = ensure_state(ctx, &st);
        if (!state) {
            return st;
        }
        if (!state->has_gpu) {
            record_error(ctx, "tk_prefill_moe_mmq: no HIP GPU device initialized on context");
            return TK_ERR_HIP;
        }

        static const bool force_legacy_mmq = [] {
            const char* v = std::getenv("STRATA_LEGACY_MMQ");
            return v != nullptr && v[0] == '1';
        }();
        if (mimo_iq && (!force_legacy_mmq || !mmq::fits(ggml_t, product->w_rows))) {
            strata::kernels::mimo_prefill_moe_mmq_q8(
                ggml_t,
                product->w,
                product->w_cols,
                product->w_rows,
                product->expert_bytes,
                product->x_q8_1,
                product->ids_dev,
                product->bounds_dev,
                product->dst,
                product->total_rows,
                product->ld_dst,
                product->n_experts,
                product->max_rows,
                stream,
                nullptr);
            return TK_OK;
        }

        mmq::Product p;
        p.w = product->w;
        p.type = ggml_t;
        p.w_rows = product->w_rows;
        p.w_cols = product->w_cols;
        p.expert_bytes = product->expert_bytes;
        p.n = product->n_experts;
        p.xq = product->x_q8_1;
        p.bounds = product->bounds_dev;
        p.ids = product->ids_dev;
        p.total_rows = product->total_rows;
        p.max_rows = product->max_rows;
        p.dst = product->dst;
        p.ld_dst = product->ld_dst;
        p.y_scale = product->y_scale_dev;

        // the grouped int8 WMMA kernel (talus::prefill::moe_gemm) - the same contract and the same
        // output bits as MMQ's grouped launch, faster at the ~10-100 rows per expert of a MoE prompt (talus-pf-bench
        // moe: x1.06-1.21 at 2K-row chunks, x1.0-1.65 at 512).  TALUS_PF_MOE=0: MMQ only.
        static const bool use_tp_moe = [] {
            const char* v = std::getenv("TALUS_PF_MOE");
            return v == nullptr || v[0] != '0';
        }();
        if (use_tp_moe && product->n_experts > 1 && product->y_scale_dev == nullptr && talus::prefill::tp_gfx11_wmma() &&
            talus::prefill::moe_supported(ggml_t, (int) product->w_rows, (int) product->w_cols)) {
            talus::prefill::MoeProduct mp;
            mp.w = product->w;
            mp.type = ggml_t;
            mp.expert_bytes = product->expert_bytes;
            mp.n = (int) product->w_rows;
            mp.k = (int) product->w_cols;
            mp.n_experts = product->n_experts;
            mp.xq = product->x_q8_1;
            mp.bounds = product->bounds_dev;
            mp.ids = product->ids_dev;
            mp.total_rows = product->total_rows;
            mp.max_rows = product->max_rows;
            mp.dst = product->dst;
            mp.ld_dst = product->ld_dst;
            if (talus::prefill::moe_gemm(mp, stream)) return TK_OK;
        }

        auto* mmq_ctx = ensure_mmq_context(*state);
        mmq_ctx->run(p, stream);
        return TK_OK;
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_moe_mmq");
    }
}

// fused gate/up + SwiGLU on tp_moe (see talus_kernels.h).  TALUS_PF_MOE_GU=0: always TK_ERR_UNSUPPORTED (the
// caller keeps its separate gate / up / SwiGLU path).
// tk_prefill_moe_mmq_gu with the SwiGLU rows written straight as the down projection's MMQ q8_1 activations
// (bitwise the FP32 rows + tk_q36_mmq_quantize(rows, down_qtype)); TK_ERR_UNSUPPORTED = run the separate path.
// TALUS_PF_MOE_GU_Q8=0: always unsupported.
extern "C" tk_status tk_prefill_moe_mmq_gu_q8(
    tk_ctx* ctx,
    const tk_moe_mmq_product* gate,
    const void* w_up,
    float swiglu_limit,
    int32_t down_qtype,
    void* xq_out,
    int64_t xq_rows,
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr || gate == nullptr || w_up == nullptr || xq_out == nullptr) {
        record_error(ctx, "tk_prefill_moe_mmq_gu_q8: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    static const bool on = [] {
        const char* v = std::getenv("TALUS_PF_MOE_GU_Q8");
        const char* g = std::getenv("TALUS_PF_MOE_GU");
        const char* m = std::getenv("TALUS_PF_MOE");
        return (v == nullptr || v[0] != '0') && (g == nullptr || g[0] != '0') && (m == nullptr || m[0] != '0');
    }();
    const int ggml_t = static_cast<int>(gate->qtype);
    if (!on || gate->n_experts <= 1 || gate->y_scale_dev != nullptr || !talus::prefill::tp_gfx11_wmma() ||
        !is_registered_mmq_q8_buf(gate->x_q8_1) || xq_rows < gate->total_rows ||
        !talus::prefill::moe_gu_q8_supported(ggml_t, (int) down_qtype, (int) gate->w_rows, (int) gate->w_cols)) {
        return TK_ERR_UNSUPPORTED;
    }
    {   // the same per-arch rule as tk_prefill_moe_mmq_gu (mode-0 types: >= moe_gu_min_rows rows per expert)
        const talus::prefill::TpTune& tu = talus::prefill::tp_tune_current();
        const int64_t mean = gate->total_rows / gate->n_experts;
        if (tu.moe_gu == 0 || mean < tu.moe_gu_min_rows) return TK_ERR_UNSUPPORTED;
    }
    if (gate->max_rows == 0 || gate->total_rows == 0) return TK_ERR_UNSUPPORTED;
    if (gate->w == nullptr || gate->bounds_dev == nullptr) {
        record_error(ctx, "tk_prefill_moe_mmq_gu_q8: NULL device buffer in product descriptor");
        return TK_ERR_NULL_PTR;
    }
    try {
        talus::prefill::MoeProduct mp;
        mp.w = gate->w;
        mp.type = ggml_t;
        mp.expert_bytes = gate->expert_bytes;
        mp.n = (int) gate->w_rows;
        mp.k = (int) gate->w_cols;
        mp.n_experts = gate->n_experts;
        mp.xq = gate->x_q8_1;
        mp.bounds = gate->bounds_dev;
        mp.ids = gate->ids_dev;
        mp.total_rows = gate->total_rows;
        mp.max_rows = gate->max_rows;
        mp.dst = nullptr;
        mp.ld_dst = 0;
        mp.w_up = w_up;
        mp.swiglu_limit = swiglu_limit;
        mp.q8_out = xq_out;
        mp.q8_rows = xq_rows;
        if (!talus::prefill::moe_gemm(mp, stream)) return TK_ERR_UNSUPPORTED;
        {
            std::lock_guard<std::mutex> lock(g_mmq_q8_buf_mu);
            g_mmq_q8_bufs.insert(xq_out);
        }
        return TK_OK;
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_moe_mmq_gu_q8");
    }
}

extern "C" tk_status tk_prefill_moe_mmq_gu(
    tk_ctx* ctx,
    const tk_moe_mmq_product* gate,
    const void* w_up,
    float swiglu_limit,
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr || gate == nullptr || w_up == nullptr) {
        record_error(ctx, "tk_prefill_moe_mmq_gu: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    static const bool on = [] {
        const char* v = std::getenv("TALUS_PF_MOE_GU");
        const char* m = std::getenv("TALUS_PF_MOE");
        return (v == nullptr || v[0] != '0') && (m == nullptr || m[0] != '0');
    }();
    const int ggml_t = static_cast<int>(gate->qtype);
    // IQ1_M on plain q8_1 rows (MiMo): the exact fused k_moe1m (tp_moe.cu; TP_MOE_IQ1M_GU=0 / TP_MOE_IQ1M_X=0: unsupported)
    if (on && ggml_t == (int) TK_QTYPE_IQ1_M && gate->n_experts > 1 && gate->y_scale_dev == nullptr && swiglu_limit == 0.0f &&
        !is_registered_mmq_q8_buf(gate->x_q8_1) && gate->ld_dst >= gate->w_rows && gate->w != nullptr &&
        gate->bounds_dev != nullptr && gate->dst != nullptr && gate->x_q8_1 != nullptr &&
        talus::prefill::moe_iq1m_q8_1_supported(ggml_t, (int) gate->w_rows, (int) gate->w_cols)) {
        if (gate->max_rows == 0 || gate->total_rows == 0) return TK_OK;
        talus::prefill::MoeProduct mp;
        mp.w = gate->w;
        mp.type = ggml_t;
        mp.expert_bytes = gate->expert_bytes;
        mp.n = (int) gate->w_rows;
        mp.k = (int) gate->w_cols;
        mp.n_experts = gate->n_experts;
        mp.xq = gate->x_q8_1;
        mp.bounds = gate->bounds_dev;
        mp.ids = gate->ids_dev;
        mp.total_rows = gate->total_rows;
        mp.max_rows = gate->max_rows;
        mp.dst = gate->dst;
        mp.ld_dst = gate->ld_dst;
        mp.w_up = w_up;
        mp.swiglu_limit = 0.0f;
        if (talus::prefill::moe_gemm_iq1m_q8_1(mp, stream)) return TK_OK;
        return TK_ERR_UNSUPPORTED;
    }
    if (!on || gate->n_experts <= 1 || gate->y_scale_dev != nullptr || !talus::prefill::tp_gfx11_wmma() ||
        !is_registered_mmq_q8_buf(gate->x_q8_1) ||
        !talus::prefill::moe_gu_supported(ggml_t, (int) gate->w_rows, (int) gate->w_cols) ||
        gate->ld_dst < gate->w_rows) {
        return TK_ERR_UNSUPPORTED;
    }
    {   // the per-arch table: where the fused launch is measured faster
        const talus::prefill::TpTune& tu = talus::prefill::tp_tune_current();
        const bool force = std::getenv("TALUS_PF_MOE_GU") != nullptr && std::getenv("TALUS_PF_MOE_GU")[0] == '2';
        const int64_t mean = gate->total_rows / gate->n_experts;
        // the mode-1 cap is the int8 fused kernel's; where the fp16 kernels take the product (moe_gemm ->
        // moe_gemm_w16: GLM's IQ2_S gate/up from pf_w16_moe_min_rows) the fused launch is bitwise the separate gate,
        // up and SwiGLU and saves their round trips (TALUS_W16_GU=0: keep the cap)
        static const bool w16gu = [] { const char* v = std::getenv("TALUS_W16_GU"); return v == nullptr || v[0] != '0'; }();
        const bool w16_takes = w16gu && talus::prefill::moe_w16_enabled(ggml_t) && mean >= tu.pf_w16_moe_min_rows;
        if (!force && (tu.moe_gu == 0 || mean < tu.moe_gu_min_rows ||
                       (talus::prefill::moe_dot_mode(ggml_t) == 1 && mean > tu.moe_gu_mode1_max_rows && !w16_takes)))
            return TK_ERR_UNSUPPORTED;
    }
    if (gate->max_rows == 0 || gate->total_rows == 0) return TK_OK;
    if (gate->w == nullptr || gate->bounds_dev == nullptr || gate->dst == nullptr) {
        record_error(ctx, "tk_prefill_moe_mmq_gu: NULL device buffer in product descriptor");
        return TK_ERR_NULL_PTR;
    }
    try {
        talus::prefill::MoeProduct mp;
        mp.w = gate->w;
        mp.type = ggml_t;
        mp.expert_bytes = gate->expert_bytes;
        mp.n = (int) gate->w_rows;
        mp.k = (int) gate->w_cols;
        mp.n_experts = gate->n_experts;
        mp.xq = gate->x_q8_1;
        mp.bounds = gate->bounds_dev;
        mp.ids = gate->ids_dev;
        mp.total_rows = gate->total_rows;
        mp.max_rows = gate->max_rows;
        mp.dst = gate->dst;
        mp.ld_dst = gate->ld_dst;
        mp.w_up = w_up;
        mp.swiglu_limit = swiglu_limit;
        if (talus::prefill::moe_gemm(mp, stream)) return TK_OK;
        record_error(ctx, "tk_prefill_moe_mmq_gu: launch failed");
        return TK_ERR_HIP;
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_moe_mmq_gu");
    }
}

extern "C" tk_status tk_prefill_moe_mmq_iq(
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
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr) {
        record_error(nullptr, "tk_prefill_moe_mmq_iq: ctx is NULL");
        return TK_ERR_NULL_PTR;
    }
    if (e0 < 0 || e1 < e0 || n_expert <= 0 || e1 > n_expert ||
        (e1 - e0) > strata::prefill::fused::kMaxBatch ||
        n_routed_rows < 0 || n_embd <= 0 || n_ff <= 0) {
        record_error(ctx, "tk_prefill_moe_mmq_iq: invalid expert range or dimensions");
        return TK_ERR_INVALID_ARG;
    }
    if (e1 == e0 || n_routed_rows == 0) {
        return TK_OK;
    }
    if (expert_blobs == nullptr || group_scratch == nullptr || x_act == nullptr ||
        src_rows_dev == nullptr || h_act == nullptr || down_out == nullptr) {
        record_error(ctx, "tk_prefill_moe_mmq_iq: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    if ((reinterpret_cast<uintptr_t>(down_out) & 15u) != 0u) {
        record_error(ctx, "tk_prefill_moe_mmq_iq: down_out must be 16-byte aligned");
        return TK_ERR_INVALID_ARG;
    }

    try {
        const int gu_type = static_cast<int>(gu_qtype);
        const int d_type = static_cast<int>(down_qtype);
        if (n_embd != 2560 || n_ff != 640 || !strata::prefill::fused::native_supported(gu_type, d_type)) {
            record_error(ctx, "tk_prefill_moe_mmq_iq: unsupported IQ fused expert format or geometry");
            return TK_ERR_UNSUPPORTED;
        }

        const auto layout = strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff);
        if (layout.gu_row == 0 || layout.d_row == 0) {
            record_error(ctx, "tk_prefill_moe_mmq_iq: invalid native expert block layout");
            return TK_ERR_UNSUPPORTED;
        }

        tk_status st = TK_OK;
        tk_prefill_state* state = ensure_state(ctx, &st);
        if (!state) {
            return st;
        }
        if (!state->has_gpu) {
            record_error(ctx, "tk_prefill_moe_mmq_iq: no HIP GPU device initialized on context");
            return TK_ERR_HIP;
        }

        strata::prefill::fused::Batch batch{};
        batch.e0 = e0;
        batch.e1 = e1;
        for (int32_t i = 0; i < (e1 - e0); ++i) {
            batch.blob[i] = static_cast<const uint8_t*>(expert_blobs[i]);
        }

        const strata::prefill::fused::NativeGeom geom{
            gu_type, d_type, layout.gu_row, layout.d_row, layout.up_off, layout.down_off};
        strata::prefill::fused::experts_native(
            batch, geom, n_expert, n_routed_rows, group_scratch, x_act, src_rows_dev, h_act, down_out, stream);
        return TK_OK;
    } catch (...) {
        return translate_current_exception(ctx, "tk_prefill_moe_mmq_iq");
    }
}

extern "C" tk_status tk_attn_prefill_fa(
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
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || k_pool == nullptr || v_pool == nullptr ||
        out == nullptr || mask == nullptr) {
        record_error(ctx, "tk_attn_prefill_fa: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    if (n_tok < 0 || n_head <= 0 || n_kv <= 0 || head_dim <= 0 ||
        kv_stride <= 0 || p0 < 0 || kv_stride < p0 + n_tok || !std::isfinite(scale)) {
        record_error(ctx, "tk_attn_prefill_fa: invalid dimensions, kv_stride, or scale");
        return TK_ERR_INVALID_ARG;
    }
    if (n_tok == 0) {
        return TK_OK;
    }
    if (!strata::prefill::fattn::supported(head_dim) ||
        (n_head % n_kv) != 0 ||
        ((n_head / n_kv) % 2) != 0) {
        record_error(ctx, "tk_attn_prefill_fa: unsupported flash-attention head configuration");
        return TK_ERR_UNSUPPORTED;
    }

    try {
        tk_status st = TK_OK;
        tk_prefill_state* state = ensure_state(ctx, &st);
        if (!state) {
            return st;
        }
        if (!state->has_gpu) {
            record_error(ctx, "tk_attn_prefill_fa: no HIP GPU device initialized on context");
            return TK_ERR_HIP;
        }

        strata::prefill::fattn::attend(
            q, k_pool, v_pool, out, n_tok, n_head, n_kv, head_dim, kv_stride, p0, scale, mask, stream);
        return TK_OK;
    } catch (...) {
        return translate_current_exception(ctx, "tk_attn_prefill_fa");
    }
}

extern "C" void tk_register_mmq_q8_buf(const void* xq) noexcept {
    if (!xq) return;
    try {
        std::lock_guard<std::mutex> lock(g_mmq_q8_buf_mu);
        g_mmq_q8_bufs.insert(xq);
    } catch (...) {
    }
}

extern "C" void tk_unregister_mmq_q8_buf(const void* xq) noexcept {
    if (!xq) return;
    try {
        std::lock_guard<std::mutex> lock(g_mmq_q8_buf_mu);
        g_mmq_q8_bufs.erase(xq);
    } catch (...) {
    }
}

extern "C" tk_status tk_q36_prefill_init_scratch(
    tk_ctx* ctx,
    int64_t scratch_elems) TK_NOEXCEPT {
    if (ctx == nullptr) {
        record_error(nullptr, "tk_q36_prefill_init_scratch: ctx is NULL");
        return TK_ERR_NULL_PTR;
    }
    if (scratch_elems < 0) {
        record_error(ctx, "tk_q36_prefill_init_scratch: negative scratch_elems");
        return TK_ERR_INVALID_ARG;
    }
    try {
        tk_status st = TK_OK;
        tk_prefill_state* state = ensure_state(ctx, &st);
        if (!state) {
            return st;
        }
        if (!state->has_gpu || scratch_elems == 0) {
            return TK_OK;
        }
        return ensure_dequant_scratch(*state, scratch_elems, ctx);
    } catch (...) {
        return translate_current_exception(ctx, "tk_q36_prefill_init_scratch");
    }
}

extern "C" tk_status tk_q36_attn_blas_prefill(
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
    tk_stream stream) TK_NOEXCEPT {
    if (ctx == nullptr || q == nullptr || q16 == nullptr || kc == nullptr ||
        vc == nullptr || att_s == nullptr || att_p16 == nullptr || att == nullptr || gate == nullptr) {
        record_error(ctx, "tk_q36_attn_blas_prefill: NULL pointer argument");
        return TK_ERR_NULL_PTR;
    }
    if (T <= 0 || H <= 0 || HKV <= 0 || (H % HKV) != 0 || HD <= 0 || ctx_len <= 0 || p0 < 0 || score_floats == 0) {
        record_error(ctx, "tk_q36_attn_blas_prefill: invalid dimensions");
        return TK_ERR_INVALID_ARG;
    }
    try {
        tk_status st = TK_OK;
        tk_prefill_state* state = ensure_state(ctx, &st);
        if (!state) {
            return st;
        }
        if (!state->has_gpu) {
            record_error(ctx, "tk_q36_attn_blas_prefill: no HIP GPU device initialized on context");
            return TK_ERR_HIP;
        }
        auto blas_h = reinterpret_cast<cublasHandle_t>(state->hipblas_handle);
        if (!blas_h) {
            record_error(ctx, "tk_q36_attn_blas_prefill: hipBLAS handle is not initialized");
            return TK_ERR_HIPBLAS;
        }
        (void) cublasSetStream(blas_h, reinterpret_cast<cudaStream_t>(stream));

        const int32_t P = static_cast<int32_t>(p0) + T;
        const int32_t ldp = (P + 7) / 8 * 8;
        const int32_t tq = static_cast<int32_t>(std::max<int64_t>(
            1, std::min<int64_t>(T, static_cast<int64_t>(score_floats / (static_cast<size_t>(H) * static_cast<size_t>(ldp))))));
        const int32_t G = H / HKV;
        const float one = 1.0f;
        const float zero = 0.0f;
        const float sc = 1.0f / std::sqrt(static_cast<float>(HD));

        st = tk_q36_to_f16(ctx, q, q16, static_cast<int64_t>(T) * H * HD, stream);
        if (st != TK_OK) return st;

        const auto* kc_h = static_cast<const __half*>(kc);
        const auto* vc_h = static_cast<const __half*>(vc);
        const auto* q16_h = static_cast<const __half*>(q16);
        auto* p16_h = static_cast<__half*>(att_p16);

        for (int32_t t0 = 0; t0 < T; t0 += tq) {
            const int32_t nt = std::min(tq, T - t0);
            for (int32_t kvh = 0; kvh < HKV; ++kvh) {
                const __half* Kh = kc_h + static_cast<size_t>(kvh) * ctx_len * HD;
                const __half* Qh = q16_h + (static_cast<size_t>(t0) * H + kvh * G) * HD;
                float* Sh = att_s + static_cast<size_t>(kvh) * G * nt * ldp;
                if (cublasGemmStridedBatchedEx(
                        blas_h, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one,
                        Kh, CUDA_R_16F, HD, 0,
                        Qh, CUDA_R_16F, H * HD, HD, &zero,
                        Sh, CUDA_R_32F, ldp, static_cast<long long>(nt) * ldp, G,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                    record_error(ctx, "tk_q36_attn_blas_prefill: attention scores GEMM failed");
                    return TK_ERR_HIPBLAS;
                }
            }
            st = tk_q36_attn_softmax16(ctx, H, att_s, p16_h, ldp, nt, P, static_cast<int32_t>(p0) + t0, sc, stream);
            if (st != TK_OK) return st;
            for (int32_t kvh = 0; kvh < HKV; ++kvh) {
                const __half* Vh = vc_h + static_cast<size_t>(kvh) * ctx_len * HD;
                const __half* Sh = p16_h + static_cast<size_t>(kvh) * G * nt * ldp;
                float* Oh = att + (static_cast<size_t>(t0) * H + kvh * G) * HD;
                if (cublasGemmStridedBatchedEx(
                        blas_h, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one,
                        Vh, CUDA_R_16F, HD, 0,
                        Sh, CUDA_R_16F, ldp, static_cast<long long>(nt) * ldp, &zero,
                        Oh, CUDA_R_32F, H * HD, HD, G,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                    record_error(ctx, "tk_q36_attn_blas_prefill: attention values GEMM failed");
                    return TK_ERR_HIPBLAS;
                }
            }
        }
        return tk_q36_gate_mul(ctx, att, gate, static_cast<int64_t>(T) * H * HD, stream);
    } catch (...) {
        return translate_current_exception(ctx, "tk_q36_attn_blas_prefill");
    }
}
