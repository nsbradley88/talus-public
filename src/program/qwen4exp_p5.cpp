// src/program/qwen4exp_p5.cpp - qwen4exp Pieces 5 + 6: launch-only host primitives for the Rust-owned conversation
// state (rust/crates/talus-model/src/qwen4exp/conv_state*). The primitives that need the engine's state (the
// session layout, the elastic K/V, the drafter) are serve-context lambdas in generate.cpp (`qwen4exp-p5` blocks).
#include "talus_qwen4exp_kernels.h"
#include "strata/core/progress.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace {
void put_error(char* err, size_t cap, const char* what) {
    if (err == nullptr || cap == 0) return;
    const size_t n = std::min(std::strlen(what), cap - 1);
    std::memcpy(err, what, n);
    err[n] = '\0';
}
}  // namespace

// conversation_detail::copy_nonblocking: one non-blocking stream per device and thread (process lifetime), the copy
// synchronized there; each landed copy is progress for the watchdog (#2049).
extern "C" int tk_qwen4exp_p5_copy(void* dst, const void* src, size_t n, char* err, size_t err_cap) {
    if (n == 0) return 0;
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e == cudaSuccess) {
        thread_local std::vector<cudaStream_t> streams;
        if (dev < 0) dev = 0;
        if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);
        if (!streams[(size_t) dev]) {
            e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
            if (e != cudaSuccess) streams[(size_t) dev] = nullptr;
        }
        if (e == cudaSuccess) {
            cudaStream_t s = streams[(size_t) dev];
            e = cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, s);
            if (e == cudaSuccess) {
                e = cudaStreamSynchronize(s);
                strata::core::progress_beat();
            }
        }
    }
    if (e == cudaSuccess) return 0;
    put_error(err, err_cap, cudaGetErrorString(e));
    return (int) e;
}

extern "C" int tk_qwen4exp_p5_sync(char* err, size_t err_cap) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) return 0;
    put_error(err, err_cap, cudaGetErrorString(e));
    return (int) e;
}

extern "C" void tk_qwen4exp_p5_progress_at(const char* text, int64_t detail) {
    // the watchdog keeps the pointer: only string literals the Rust side passes for the whole process
    strata::core::progress_at(text, detail);
}

extern "C" void tk_qwen4exp_p5_progress_allow(int64_t seconds) { strata::core::progress_allow(seconds); }

extern "C" void tk_qwen4exp_p5_progress_beat(void) { strata::core::progress_beat(); }
