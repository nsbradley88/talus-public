// tests/hip/router_pad_bench.cpp - the real router kernels on padded vs plain f32 rows (TALUS_ROUTER_PAD).
//   talus-router-pad [PAD_BYTES] [ITERS] [MODELS]   PAD default 512, ITERS passes per round (default 50), MODELS a comma
//                                                   list of laguna,mimo,glm (default all)
// Per model: L layers of router weights (the model's shape, random), two identical copies: A (plain, not registered)
// and B (registered with talus_rowpad_prepare, so the kernels read its padded copy).  A "pass" = one router call per
// layer back to back, as a decode step does (each layer's rows cold: L x rows x row bytes >> MALL).  Rounds interleave
// A and B; reported: us per call (best of rounds), GB/s of weight bytes, B/A, and the outputs of the last A and B
// passes compared bitwise (logits; for tk_router_route also ids and weights).
//   laguna  3072 x 256 x 47 layers: tk_router_route (decode, T 1) and tk_f32_matvec_rows_n (verify, T 2 4 8)
//   mimo    4096 x 256 x 47: tk_f32_matmul (decode / verify window: the few-token kernel, T 1 2 4 8)
//   glm     4096 x 288 x 43: tkg_f32_matvec (T 1 2 3 4 8)
// Exit 1 on any bitwise mismatch.  Idea from libstrix (asf0/libstrix, MIT), no code from it.
#include "strata/kernels/rowpad.hpp"
#include "talus_glm_kernels.h"
#include "talus_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
struct Model { const char* name; int n_in, n_out, layers; };
const Model kModels[] = {{"laguna", 3072, 256, 47}, {"mimo", 4096, 256, 47}, {"glm", 4096, 288, 43}};
int bad = 0;
}   // namespace

int main(int argc, char** argv) {
    const std::string pad = argc > 1 ? argv[1] : "512";
    setenv("TALUS_ROUTER_PAD", pad.c_str(), 1);
    const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    const std::string want = argc > 3 ? argv[3] : "laguna,mimo,glm";
    std::printf("router-pad bench: pad %lld B (TALUS_ROUTER_PAD=%s), %d passes per round, 4 rounds A/B interleaved\n",
                (long long) talus_rowpad_pad_bytes(), pad.c_str(), iters);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "ev");
    ck(cudaEventCreate(&e1), "ev");
    for (const Model& m : kModels) {
        if (("," + want + ",").find("," + std::string(m.name) + ",") == std::string::npos) continue;
        const size_t per = (size_t) m.n_in * m.n_out;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 0.05f);
        std::vector<float> h(per * m.layers);
        for (auto& v : h) v = nd(rng);
        float *wa = nullptr, *wb = nullptr, *x = nullptr, *bias = nullptr, *ya = nullptr, *yb = nullptr, *wts = nullptr;
        int32_t* ids = nullptr;
        unsigned* cnt = nullptr;
        ck(cudaMalloc(&wa, per * m.layers * 4), "wa");
        ck(cudaMalloc(&wb, per * m.layers * 4), "wb");
        ck(cudaMemcpy(wa, h.data(), per * m.layers * 4, cudaMemcpyHostToDevice), "up");
        ck(cudaMemcpy(wb, h.data(), per * m.layers * 4, cudaMemcpyHostToDevice), "up");
        std::vector<float> hx((size_t) 16 * m.n_in), hb(m.n_out);
        for (auto& v : hx) v = nd(rng) * 20.f;
        for (auto& v : hb) v = nd(rng);
        ck(cudaMalloc(&x, hx.size() * 4), "x");
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMalloc(&bias, hb.size() * 4), "b");
        ck(cudaMemcpy(bias, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice), "b");
        const size_t ny = (size_t) m.layers * 16 * m.n_out;
        ck(cudaMalloc(&ya, ny * 4), "ya");
        ck(cudaMalloc(&yb, ny * 4), "yb");
        ck(cudaMalloc(&ids, (size_t) 2 * m.layers * 32 * 4), "ids");
        ck(cudaMalloc(&wts, (size_t) 2 * m.layers * 32 * 4), "wts");
        ck(cudaMalloc(&cnt, 4), "cnt");
        ck(cudaMemset(cnt, 0, 4), "cnt");
        int reg = 0;
        for (int l = 0; l < m.layers; ++l)
            reg += talus_rowpad_prepare(wb + per * l, m.n_out, (int64_t) m.n_in * 4, st) == 1;
        std::printf("\n%s: %d x %d f32 rows (%d B), %d layers (%.1f MiB per pass); %d layers registered (%.1f MiB padded)\n",
                    m.name, m.n_out, m.n_in, m.n_in * 4, m.layers, per * 4.0 * m.layers / 1048576.0, reg,
                    talus_rowpad_bytes() / 1048576.0);
        struct Cfg { const char* kern; int T; };
        std::vector<Cfg> cfgs;
        if (!std::strcmp(m.name, "laguna")) cfgs = {{"tk_router_route", 1}, {"tk_f32_matvec_rows_n", 2}, {"tk_f32_matvec_rows_n", 4}, {"tk_f32_matvec_rows_n", 8}};
        else if (!std::strcmp(m.name, "mimo")) cfgs = {{"tk_f32_matmul", 1}, {"tk_f32_matmul", 2}, {"tk_f32_matmul", 4}, {"tk_f32_matmul", 8}, {"tk_f32_matmul", 16}};
        else cfgs = {{"tkg_f32_matvec", 1}, {"tkg_f32_matvec", 2}, {"tkg_f32_matvec", 3}, {"tkg_f32_matvec", 4}, {"tkg_f32_matvec", 8}};
        std::printf("  %-22s %-2s %10s %10s %8s %8s %7s  %s\n", "kernel", "T", "plain us", "pad us", "plain", "pad", "pad/pl",
                    "bitwise");
        for (const Cfg& c : cfgs) {
            auto pass = [&](const float* w, float* y, int slot) {
                for (int l = 0; l < m.layers; ++l) {
                    const float* wl = w + per * l;
                    float* yl = y + (size_t) l * 16 * m.n_out;
                    int rc = 0;
                    if (!std::strcmp(c.kern, "tk_router_route"))
                        rc = tk_router_route(nullptr, wl, x, yl, bias, ids + (slot * m.layers + l) * 32,
                                             wts + (slot * m.layers + l) * 32, m.n_in, m.n_out, 8, 2.5f, cnt, st);
                    else if (!std::strcmp(c.kern, "tk_f32_matvec_rows_n"))
                        rc = tk_f32_matvec_rows_n(nullptr, wl, x, yl, m.n_in, m.n_out, c.T, st);
                    else if (!std::strcmp(c.kern, "tk_f32_matmul"))
                        rc = tk_f32_matmul(nullptr, wl, x, yl, m.n_in, m.n_out, c.T, st);
                    else
                        rc = tkg_f32_matvec(wl, x, yl, m.n_in, m.n_out, c.T, st);
                    if (rc) { std::fprintf(stderr, "%s rc %d\n", c.kern, rc); std::exit(2); }
                }
            };
            float best[2] = {1e30f, 1e30f};
            for (int round = 0; round < 4; ++round)
                for (int arm = 0; arm < 2; ++arm) {
                    const float* w = arm ? wb : wa;
                    float* y = arm ? yb : ya;
                    pass(w, y, arm);   // warm-up (launch path), not timed
                    ck(cudaEventRecord(e0, st), "rec");
                    for (int i = 0; i < iters; ++i) pass(w, y, arm);
                    ck(cudaEventRecord(e1, st), "rec");
                    ck(cudaEventSynchronize(e1), "sync");
                    float ms = 0;
                    ck(cudaEventElapsedTime(&ms, e0, e1), "el");
                    best[arm] = std::min(best[arm], ms * 1000.f / (iters * m.layers));
                }
            std::vector<float> a(ny), b(ny);
            ck(cudaMemset(ya, 0, ny * 4), "z");
            ck(cudaMemset(yb, 0, ny * 4), "z");
            pass(wa, ya, 0);
            pass(wb, yb, 1);
            ck(cudaStreamSynchronize(st), "sync");
            ck(cudaMemcpy(a.data(), ya, ny * 4, cudaMemcpyDeviceToHost), "dl");
            ck(cudaMemcpy(b.data(), yb, ny * 4, cudaMemcpyDeviceToHost), "dl");
            long diff = 0;
            for (size_t i = 0; i < ny; ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
            if (!std::strcmp(c.kern, "tk_router_route")) {
                std::vector<int32_t> ia((size_t) 2 * m.layers * 32);
                std::vector<float> wa2((size_t) 2 * m.layers * 32);
                ck(cudaMemcpy(ia.data(), ids, ia.size() * 4, cudaMemcpyDeviceToHost), "dl");
                ck(cudaMemcpy(wa2.data(), wts, wa2.size() * 4, cudaMemcpyDeviceToHost), "dl");
                for (int l = 0; l < m.layers; ++l)
                    for (int j = 0; j < 8; ++j) {
                        diff += ia[l * 32 + j] != ia[(m.layers + l) * 32 + j];
                        diff += std::memcmp(&wa2[l * 32 + j], &wa2[(m.layers + l) * 32 + j], 4) != 0;
                    }
            }
            bad += diff != 0;
            // FNV-1a of the plain arm's outputs, to compare kernel variants (env) across runs bitwise
            uint64_t hsh = 1469598103934665603ull;
            for (size_t i = 0; i < ny; ++i) {
                uint32_t u;
                std::memcpy(&u, &a[i], 4);
                hsh = (hsh ^ u) * 1099511628211ull;
            }
            std::printf("  hash %s T%d %016llx\n", c.kern, c.T, (unsigned long long) hsh);
            const double bytes = (double) per * 4;
            std::printf("  %-22s %-2d %10.2f %10.2f %6.0fGB/s %6.0fGB/s %7.3f  %s\n", c.kern, c.T, best[0], best[1],
                        bytes / (best[0] * 1e3), bytes / (best[1] * 1e3), best[1] / best[0],
                        diff ? ("MISMATCH " + std::to_string(diff)).c_str() : "identical");
            std::fflush(stdout);
        }
        for (int l = 0; l < m.layers; ++l) talus_rowpad_release(wb + per * l);
        cudaFree(wa); cudaFree(wb); cudaFree(x); cudaFree(bias); cudaFree(ya); cudaFree(yb); cudaFree(ids);
        cudaFree(wts); cudaFree(cnt);
    }
    std::printf("\nrouter-pad: %s\n", bad ? "FAIL (bitwise mismatch)" : "PASS (all outputs bitwise identical)");
    return bad ? 1 : 0;
}
