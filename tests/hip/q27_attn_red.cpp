// tests/hip/q27_attn_red.cpp - the 27B decode / verify attention (k_attn_wmma, k_attn_wmma8) softmax
// reduction variants outside the engine.
//   talus-q27-attn-red exact [CTX]        every instance (k_attn_wmma<1..3>, k_attn_wmma8<1..3>) x RED 1..3 bitwise
//                                          RED 0 (pacc, pm, pl words) at T 1..16 and positions around tile / chunk
//                                          edges up to CTX, on four input sets (random, ties, zero keys, wide scores)
//   talus-q27-attn-red time [ITERS] [FILLS] [TS] [REDS]   the production launcher (launch_attn_wmma, RT1G rows) per
//                                          RED, us per call; FILLS / TS / REDS comma lists (default 2048,8192,32768,
//                                          65536 / 1,2,4,6,8,12,16 / 0,1,2,3), rounds interleaved
// 27B shape: H 24, HKV 4, head dim 256, chunks 512 cells (128-cell chunks below 4K), as the engine sets them.
// Exit 1 on any bitwise mismatch.  Synthetic inputs; idea from libstrix (asf0/libstrix, MIT), no code from it.
#include "strata/qwen35/q36_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr int H = 24, HKV = 4, HD = 256, G = H / HKV;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
__global__ void k_cmp(const uint32_t* a, const uint32_t* b, size_t n, unsigned long long* cnt) {
    unsigned long long c = 0;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x)
        c += a[i] != b[i];
    if (c) atomicAdd(cnt, c);
}
std::vector<int> list(const char* s, std::vector<int> d) {
    if (!s) return d;
    std::vector<int> v;
    for (const char* p = s; *p;) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}
struct Bufs {
    int ctx = 0, vt_ctx = 0, ckc = 0, n_chunk = 0;
    float* q = nullptr; __half* kc = nullptr; __half* vt = nullptr; int32_t* pos0 = nullptr;
    float* pacc[2] = {}; float* pm[2] = {}; float* pl[2] = {};
    size_t n_acc = 0, n_m = 0;
    unsigned long long* cnt = nullptr;
};
void alloc(Bufs& b, int ctx) {
    b.ctx = ctx;
    b.ckc = talus::q36::kAttChunk | ((talus::q36::kAttSmallCells / talus::q36::kAttSmall) << 16);
    b.vt_ctx = talus::q36::att_vt_cells(ctx, b.ckc);
    b.n_chunk = talus::q36::att_chunk_count(ctx, b.ckc);
    ck(cudaMalloc(&b.q, sizeof(float) * 16 * H * HD), "q");
    ck(cudaMalloc(&b.kc, sizeof(__half) * (size_t) HKV * ctx * HD), "kc");
    ck(cudaMalloc(&b.vt, sizeof(__half) * (size_t) HKV * b.vt_ctx * HD), "vt");
    ck(cudaMalloc(&b.pos0, 4), "pos0");
    b.n_m = (size_t) 16 * H * b.n_chunk;
    b.n_acc = b.n_m * HD;
    for (int i = 0; i < 2; ++i) {
        ck(cudaMalloc(&b.pacc[i], sizeof(float) * b.n_acc), "pacc");
        ck(cudaMalloc(&b.pm[i], sizeof(float) * b.n_m), "pm");
        ck(cudaMalloc(&b.pl[i], sizeof(float) * b.n_m), "pl");
    }
    ck(cudaMalloc(&b.cnt, 8), "cnt");
}
// input sets: 0 random, 1 ties (all keys of a head equal: equal scores, max ties), 2 random with every 7th key zero
// (scores exactly +0) and q rows with zero halves, 3 wide scores (q x 12: exp underflow to 0 in most of a tile)
void fill(Bufs& b, int set, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t nk = (size_t) HKV * b.ctx * HD;
    std::vector<__half> k(nk), v((size_t) HKV * b.vt_ctx * HD);
    std::vector<float> key0(HD);
    for (auto& x : key0) x = nd(rng);
    for (size_t i = 0; i < nk; ++i) {
        const size_t cell = (i / HD) % b.ctx;
        float x = set == 1 ? key0[i % HD] : nd(rng);
        if (set == 2 && cell % 7 == 3) x = 0.f;
        k[i] = __float2half(x);
    }
    for (auto& x : v) x = __float2half(nd(rng));
    std::vector<float> q((size_t) 16 * H * HD);
    for (size_t i = 0; i < q.size(); ++i) {
        float x = nd(rng);
        if (set == 2 && (i / HD) % 5 == 1 && (i % HD) < 128) x = 0.f;
        if (set == 3) x *= 12.f;
        q[i] = x;
    }
    ck(cudaMemcpy(b.kc, k.data(), k.size() * 2, cudaMemcpyHostToDevice), "up k");
    ck(cudaMemcpy(b.vt, v.data(), v.size() * 2, cudaMemcpyHostToDevice), "up v");
    ck(cudaMemcpy(b.q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "up q");
}
unsigned long long diff(Bufs& b, const void* x, const void* y, size_t n) {
    ck(cudaMemset(b.cnt, 0, 8), "cnt0");
    k_cmp<<<1024, 256>>>((const uint32_t*) x, (const uint32_t*) y, n, b.cnt);
    unsigned long long c = 0;
    ck(cudaMemcpy(&c, b.cnt, 8, cudaMemcpyDeviceToHost), "cnt");
    return c;
}
void run_inst(Bufs& b, int slot, int w8, int rt, int red, int T) {
    const float sc = 1.0f / 16.0f;
    ck(cudaMemset(b.pacc[slot], 0xff, sizeof(float) * b.n_acc), "clr");
    ck(cudaMemset(b.pm[slot], 0xff, sizeof(float) * b.n_m), "clr");
    ck(cudaMemset(b.pl[slot], 0xff, sizeof(float) * b.n_m), "clr");
    const int R = T * G;
    const int ng = w8 ? 1 : (R + 16 * rt - 1) / (16 * rt);
    dim3 grid((unsigned) (b.n_chunk * ng), HKV);
    talus::q36::launch_attn_wmma_inst(w8, rt, red, grid, nullptr, b.q, b.kc, b.vt, b.pos0, T, H, HKV, b.ctx, b.vt_ctx, sc,
                                      b.ckc, b.n_chunk, b.pacc[slot], b.pm[slot], b.pl[slot], 0, ng);
    ck(cudaGetLastError(), "launch");
}
int exact(int ctx) {
    Bufs b;
    alloc(b, ctx);
    const std::vector<int> pos = {0, 1, 5, 26, 31, 32, 100, 127, 128, 511, 512, 1000, 4090, 4096, 4133, 8191, ctx / 2 + 17,
                                  ctx - 16 - 3, ctx - 16};
    long checks = 0, bad = 0;
    for (int set = 0; set < 4; ++set) {
        fill(b, set, 1234 + set);
        for (int p : pos) {
            if (p + 16 > ctx) continue;
            ck(cudaMemcpy(b.pos0, &p, 4, cudaMemcpyHostToDevice), "pos");
            for (int w8 = 0; w8 < 2; ++w8)
                for (int rt = 1; rt <= 3; ++rt)
                    for (int T = 1; T <= 16; ++T) {
                        if (w8 && T * G > 16 * rt) continue;      // k_attn_wmma8 takes all rows in one block
                        if (!w8 && rt > 1 && T * G <= 16 * (rt - 1)) continue;   // a smaller instance covers it
                        run_inst(b, 0, w8, rt, 0, T);
                        for (int red = 1; red <= 3; ++red) {
                            run_inst(b, 1, w8, rt, red, T);
                            const unsigned long long d = diff(b, b.pacc[0], b.pacc[1], b.n_acc) +
                                                         diff(b, b.pm[0], b.pm[1], b.n_m) + diff(b, b.pl[0], b.pl[1], b.n_m);
                            ++checks;
                            if (d) {
                                ++bad;
                                if (bad < 20)
                                    std::printf("MISMATCH set %d pos %d %s<%d> RED %d T %d: %llu words differ\n", set, p,
                                                w8 ? "k_attn_wmma8" : "k_attn_wmma", rt, red, T, d);
                            }
                        }
                    }
        }
        std::printf("set %d done: %ld checks, %ld mismatches so far\n", set, checks, bad);
        std::fflush(stdout);
    }
    std::printf("exact ctx %d: %ld instance checks (set x pos x inst x T x RED), %ld MISMATCH -> %s\n", ctx, checks, bad,
                bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
int timing(int iters, std::vector<int> fills, std::vector<int> ts, std::vector<int> reds) {
    int mx = 0;
    for (int f : fills) mx = std::max(mx, f);
    Bufs b;
    alloc(b, mx + 512);
    fill(b, 0, 99);
    const float sc = 1.0f / 16.0f;
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "ev");
    ck(cudaEventCreate(&e1), "ev");
    std::printf("%-7s %-3s", "fill", "T");
    for (int r : reds) std::printf("   RED%d us", r);
    for (size_t i = 1; i < reds.size(); ++i) std::printf("  %d/%d", reds[i], reds[0]);
    std::printf("\n");
    for (int f : fills)
        for (int T : ts) {
            const int p = f - T;
            ck(cudaMemcpy(b.pos0, &p, 4, cudaMemcpyHostToDevice), "pos");
            std::vector<float> best(reds.size(), 1e30f);
            for (int round = 0; round < 3; ++round)
                for (size_t ri = 0; ri < reds.size(); ++ri) {
                    talus::q36::g_attn_red = reds[ri];
                    auto call = [&] {
                        talus::q36::launch_attn_wmma(false, dim3((unsigned) b.n_chunk, HKV), nullptr, b.q, b.kc, b.vt,
                                                     b.pos0, T, H, HKV, b.ctx, b.vt_ctx, sc, b.ckc, b.n_chunk, b.pacc[0],
                                                     b.pm[0], b.pl[0]);
                    };
                    for (int i = 0; i < 10; ++i) call();
                    ck(cudaEventRecord(e0), "rec");
                    for (int i = 0; i < iters; ++i) call();
                    ck(cudaEventRecord(e1), "rec");
                    ck(cudaEventSynchronize(e1), "sync");
                    float ms = 0;
                    ck(cudaEventElapsedTime(&ms, e0, e1), "el");
                    best[ri] = std::min(best[ri], ms * 1000.f / iters);
                }
            std::printf("%-7d %-3d", f, T);
            for (float x : best) std::printf(" %9.1f", x);
            for (size_t i = 1; i < best.size(); ++i) std::printf(" %5.3f", best[i] / best[0]);
            std::printf("\n");
            std::fflush(stdout);
        }
    return 0;
}
}   // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "exact";
    if (mode == "exact") return exact(argc > 2 ? std::atoi(argv[2]) : 33280);
    if (mode == "time")
        return timing(argc > 2 ? std::atoi(argv[2]) : 200, list(argc > 3 ? argv[3] : nullptr, {2048, 8192, 32768, 65536}),
                      list(argc > 4 ? argv[4] : nullptr, {1, 2, 4, 6, 8, 12, 16}),
                      list(argc > 5 ? argv[5] : nullptr, {0, 1, 2, 3}));
    std::fprintf(stderr, "usage: talus-q27-attn-red exact [CTX] | time [ITERS] [FILLS] [TS] [REDS]\n");
    return 2;
}
