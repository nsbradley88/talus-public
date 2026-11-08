// src/kernels/talus_rowpad.cpp - the padded-row registry (include/strata/kernels/rowpad.hpp).
#include "strata/kernels/rowpad.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace {
struct Entry {
    void* copy = nullptr;
    int64_t rows = 0, row_bytes = 0, ld_bytes = 0;
    int refs = 0;
};
std::shared_mutex g_mu;
std::unordered_map<const void*, Entry> g_map;
std::atomic<int> g_n{0};
std::atomic<int64_t> g_bytes{0};
int64_t pad_bytes() {
    static const int64_t p = [] {
        const char* e = std::getenv("TALUS_ROUTER_PAD");
        if (!e || !*e) return (int64_t) 0;
        const long v = std::atol(e);
        if (v <= 0) return (int64_t) 0;
        if (v == 1) return (int64_t) 512;
        return (int64_t) ((v + 15) / 16 * 16);
    }();
    return p;
}
}  // namespace

namespace strata::kernels {
const float* rowpad_lookup(const float* w, int64_t n_in, int64_t n_out, int64_t* ld) noexcept {
    *ld = n_in;
    if (g_n.load(std::memory_order_acquire) == 0 || w == nullptr) return w;
    std::shared_lock<std::shared_mutex> lk(g_mu);
    const auto it = g_map.find(w);
    if (it == g_map.end()) return w;
    const Entry& e = it->second;
    if (e.row_bytes != n_in * (int64_t) sizeof(float) || n_out > e.rows) return w;   // another view: the original
    *ld = e.ld_bytes / (int64_t) sizeof(float);
    return static_cast<const float*>(e.copy);
}
}  // namespace strata::kernels

extern "C" int talus_rowpad_prepare(const void* w, int64_t rows, int64_t row_bytes, void* stream) {
    const int64_t pad = pad_bytes();
    if (pad == 0 || w == nullptr || rows <= 0 || row_bytes <= 0 || row_bytes % 16 != 0) return 0;
    std::unique_lock<std::shared_mutex> lk(g_mu);
    auto it = g_map.find(w);
    if (it != g_map.end()) {
        if (it->second.rows != rows || it->second.row_bytes != row_bytes) return 0;
        ++it->second.refs;
        return 1;
    }
    Entry e;
    e.rows = rows;
    e.row_bytes = row_bytes;
    e.ld_bytes = row_bytes + pad;
    if (cudaMalloc(&e.copy, (size_t) (e.ld_bytes * rows)) != cudaSuccess) {
        (void) cudaGetLastError();
        return -1;
    }
    if (cudaMemcpy2DAsync(e.copy, (size_t) e.ld_bytes, w, (size_t) row_bytes, (size_t) row_bytes, (size_t) rows,
                          cudaMemcpyDeviceToDevice, (cudaStream_t) stream) != cudaSuccess ||
        cudaStreamSynchronize((cudaStream_t) stream) != cudaSuccess) {
        (void) cudaGetLastError();
        cudaFree(e.copy);
        return -2;
    }
    e.refs = 1;
    g_bytes += e.ld_bytes * rows;
    g_map.emplace(w, e);
    g_n.store((int) g_map.size(), std::memory_order_release);
    return 1;
}

extern "C" void talus_rowpad_release(const void* w) {
    std::unique_lock<std::shared_mutex> lk(g_mu);
    auto it = g_map.find(w);
    if (it == g_map.end() || --it->second.refs > 0) return;
    cudaDeviceSynchronize();   // no launch may still read the copy
    cudaFree(it->second.copy);
    g_bytes -= it->second.ld_bytes * it->second.rows;
    g_map.erase(it);
    g_n.store((int) g_map.size(), std::memory_order_release);
}

extern "C" int64_t talus_rowpad_bytes(void) { return g_bytes.load(); }
extern "C" int64_t talus_rowpad_pad_bytes(void) { return pad_bytes(); }
