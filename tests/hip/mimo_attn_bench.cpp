// MiMo prompt attention micro-benchmark on synthetic f16 KV pools (no model needed).
//   talus-mimo-attn-bench MODE P0 [N_TOK] [ITERS]
//     MODE full : mimo_attn_k192_v128 (64 q heads, 4 KV heads, K192 / V128), keys 0 .. P0 + t (causal)
//     MODE router: strata::laguna::f32_matmul 4096 -> 256 over N_TOK rows (MiMo router; TALUS_F32MM_V2 selects)
//     MODE swa  : mimo_attn_swa_sinks (64 q heads, 8 KV heads, window 128, sinks, value_scale 0.707)
//   P0 = position of the chunk's first token (the engine's pos_dev[0]); N_TOK = chunk size (default 2048).
// Prints the time per launch, the effective TFLOPS and an FNV-1a hash of the output bytes: variants selected by
// environment switches inside the library must print the same hash (bitwise check).  OUT=file writes the output.
#include "strata/kernels/mimo_kernels.hpp"
#include "../../src/laguna/laguna_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static uint16_t f2h(float f) {   // round-to-nearest-even float -> half (normal range only; inputs are ~N(0,1))
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int e = (int) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) return (uint16_t) sign;            // flush tiny values
    if (e >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s full|swa P0 [N_TOK] [ITERS]\n", argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "router") == 0) {
        const int nt = argc > 3 ? std::atoi(argv[3]) : 2048, it = argc > 4 ? std::atoi(argv[4]) : 10;
        const int ni = 4096, no = 256;
        std::mt19937 rg(3);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> wh((size_t) no * ni), xh((size_t) nt * ni);
        for (auto& v : wh) v = 0.02f * nd(rg);
        for (auto& v : xh) v = nd(rg);
        float *w, *x, *y;
        cudaMalloc(&w, wh.size() * 4); cudaMalloc(&x, xh.size() * 4); cudaMalloc(&y, (size_t) nt * no * 4);
        cudaMemcpy(w, wh.data(), wh.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(x, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice);
        cudaStream_t st; cudaStreamCreate(&st);
        strata::laguna::f32_matmul(w, x, y, ni, no, nt, st);
        cudaStreamSynchronize(st);
        std::vector<float> yh((size_t) nt * no);
        cudaMemcpy(yh.data(), y, yh.size() * 4, cudaMemcpyDeviceToHost);
        uint64_t hh = 1469598103934665603ull;
        for (size_t i = 0; i < yh.size() * 4; ++i) hh = (hh ^ ((const unsigned char*) yh.data())[i]) * 1099511628211ull;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a, st);
        for (int i = 0; i < it; ++i) strata::laguna::f32_matmul(w, x, y, ni, no, nt, st);
        cudaEventRecord(b, st); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= it;
        std::printf("router n_tok %d: %.3f ms  %.2f TFLOPS  hash %016llx\n", nt, ms, 2.0 * nt * ni * no / ms / 1e9,
                    (unsigned long long) hh);
        return 0;
    }
    const bool full = std::strcmp(argv[1], "full") == 0;
    const int p0 = std::atoi(argv[2]);
    const int n_tok = argc > 3 ? std::atoi(argv[3]) : 2048;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 10;
    const int n_head = 64, n_kv = full ? 4 : 8, window = 128;
    const int stride = full ? (std::getenv("STRIDE") ? std::atoi(std::getenv("STRIDE")) : ((p0 + n_tok + 4095) / 4096) * 4096)
                            : (16384 + window);
    // STALE=1: every cell of the pool holds data (a reused cache), not just the ones before p0 + n_tok
    const int cells = (full && !std::getenv("STALE")) ? (p0 + n_tok) : stride;
    const float scale = 1.0f / std::sqrt(192.0f);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> kh((size_t) stride * n_kv * 192, 0), vh((size_t) stride * n_kv * 128, 0);
    for (size_t i = 0; i < (size_t) cells * n_kv * 192; ++i) kh[i] = f2h(nd(rng));
    for (size_t i = 0; i < (size_t) cells * n_kv * 128; ++i) vh[i] = f2h(nd(rng));
    std::vector<float> qh((size_t) n_tok * n_head * 192);
    for (auto& v : qh) v = 2.0f * nd(rng);
    std::vector<float> sk(n_head);
    for (auto& v : sk) v = nd(rng);

    void *kp, *vp;
    float *q, *out, *scratch, *sinks;
    int* pos;
    cudaMalloc(&kp, kh.size() * 2);
    cudaMalloc(&vp, vh.size() * 2);
    cudaMemcpy(kp, kh.data(), kh.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(vp, vh.data(), vh.size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&q, qh.size() * 4);
    cudaMemcpy(q, qh.data(), qh.size() * 4, cudaMemcpyHostToDevice);
    const size_t on = (size_t) n_tok * n_head * 128;
    cudaMalloc(&out, on * 4);
    cudaMalloc(&scratch, 64 << 20);
    cudaMalloc(&sinks, n_head * 4);
    cudaMemcpy(sinks, sk.data(), n_head * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&pos, 4);
    cudaMemcpy(pos, &p0, 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);

    auto run = [&] {
        if (full)
            strata::kernels::mimo_attn_k192_v128(q, kp, vp, nullptr, nullptr, out, scratch, n_tok, n_head, n_kv, stride,
                                                 scale, pos, 0, s);
        else
            strata::kernels::mimo_attn_swa_sinks(q, kp, vp, nullptr, nullptr, sinks, out, scratch, n_tok, n_head, n_kv,
                                                 192, 128, window, stride, scale, pos, 0, s, 0.707f);
    };
    cudaMemsetAsync(out, 0, on * 4, s);
    run();
    cudaStreamSynchronize(s);
    std::vector<float> oh(on);
    cudaMemcpy(oh.data(), out, on * 4, cudaMemcpyDeviceToHost);
    uint64_t hsh = 1469598103934665603ull;
    const unsigned char* b = (const unsigned char*) oh.data();
    for (size_t i = 0; i < on * 4; ++i) hsh = (hsh ^ b[i]) * 1099511628211ull;
    if (const char* f = std::getenv("OUT")) {
        FILE* fp = std::fopen(f, "wb");
        if (fp) {
            std::fwrite(oh.data(), 4, on, fp);
            std::fclose(fp);
        }
    }
    for (int i = 0; i < 2; ++i) run();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) run();
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= iters;
    // useful work: QK (192) + PV (128) MACs over the causal / windowed key ranges
    double pairs = 0;
    for (int t = 0; t < n_tok; ++t) {
        const double keys = full ? (p0 + t + 1) : std::min(window, p0 + t + 1);
        pairs += keys;
    }
    const double flop = pairs * n_head * (192 + 128) * 2.0;
    std::printf("%s p0 %d n_tok %d: %.3f ms  %.2f TFLOPS  hash %016llx\n", full ? "full" : "swa", p0, n_tok, ms,
                flop / ms / 1e9, (unsigned long long) hsh);
    return 0;
}
