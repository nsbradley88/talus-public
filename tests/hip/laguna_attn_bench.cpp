// decode attention micro-benchmark on synthetic f16 pools (Laguna shapes), no model needed.
//   talus-attn-bench [ctx] [n_split] [mode] [pools]   mode 0 = prompt_attn dot2, 1 = WMMA split-K, 2 = decode_attn,
//   3 = decode_attn q8, -1 = WMMA q8; pools 12 (cold, default) or 1 (hot: in the 32 MB on-die cache at 8K)
#include "../../src/laguna/laguna_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int ctx = argc > 1 ? std::atoi(argv[1]) : 8192;
    const int split = argc > 2 ? std::atoi(argv[2]) : 16;
    const int mode = argc > 3 ? std::atoi(argv[3]) : 0;
    const int n_pools = argc > 4 ? std::atoi(argv[4]) : 12;
    const int ctx16 = ((ctx + 15) & ~15) + 64, n_kv = 8, cell = n_kv * 128;
    std::vector<uint16_t> pool((size_t) ctx16 * cell);
    std::mt19937 rng(1);
    for (auto& v : pool) v = (uint16_t) (0x3000 + (rng() & 0x7ff));   // small positive halves
    // 12 separate pools (the 12 full-attention layers), cycled so the 32 MB on-die cache cannot hold them
    constexpr int kPools = 12;
    uint16_t *kps[kPools], *vps[kPools]; float *q, *out, *part, *ml; int* pos;
    for (int i = 0; i < kPools; ++i) {
        cudaMalloc(&kps[i], pool.size() * 2); cudaMalloc(&vps[i], pool.size() * 2);
        cudaMemcpy(kps[i], pool.data(), pool.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(vps[i], pool.data(), pool.size() * 2, cudaMemcpyHostToDevice);
    }
    int cur = 0;
    uint16_t *ks, *vs;   // q8 scales (shared by the pools; values are synthetic)
    cudaMalloc(&ks, (size_t) ctx16 * n_kv * 4 * 2); cudaMalloc(&vs, (size_t) ctx16 * n_kv * 4 * 2);
    {
        std::vector<uint16_t> sc((size_t) ctx16 * n_kv * 4, 0x2000);
        cudaMemcpy(ks, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(vs, sc.data(), sc.size() * 2, cudaMemcpyHostToDevice);
    }
    cudaMalloc(&q, 72 * 128 * 4); cudaMemset(q, 0, 72 * 128 * 4);
    cudaMalloc(&out, 72 * 128 * 4); cudaMalloc(&part, (size_t) split * 72 * 128 * 4); cudaMalloc(&ml, (size_t) split * 72 * 2 * 4);
    cudaMalloc(&pos, 4); const int p = ctx - 1; cudaMemcpy(pos, &p, 4, cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreate(&s);
    for (int full = 1; full >= 0; --full) {
        const int H = full ? 48 : 72, window = full ? 0 : 512;
        auto run = [&] {
            uint16_t* kp = kps[cur]; uint16_t* vp = vps[cur]; cur = (cur + 1) % (n_pools < kPools ? n_pools : kPools);
            if (mode >= 2) strata::laguna::decode_attn(q, kp, vp, mode == 3 ? ks : nullptr, mode == 3 ? vs : nullptr, H, n_kv,
                                                       window, ctx16, pos, split, part, ml, s);
            else if (mode == 1 || mode == -1)
                strata::laguna::prompt_attn_wmma(q, kp, vp, out, 1, 0, H, n_kv, window, s, ctx16, pos, split, part, ml,
                                                 mode == -1 ? ks : nullptr, mode == -1 ? vs : nullptr);
            else strata::laguna::prompt_attn(q, kp, vp, out, 1, 0, H, n_kv, window, s, pos, split, part, ml, false, ctx16);
        };
        for (int i = 0; i < 5; ++i) run();
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, s);
        const int it = 200;
        for (int i = 0; i < it; ++i) run();
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        const double keys = full ? ctx : (ctx < 512 ? ctx : 512);
        const double bytes = keys * cell * 2 * ((mode == 3 || mode == -1) ? 1 : 2);
        std::printf("%s ctx %d split %d mode %d: %.1f us/call, %.0f GB/s\n", full ? "full" : "swa ", ctx, split, mode,
                    ms * 1000 / it, bytes / (ms * 1e-3 / it) / 1e9);
    }
    return 0;
}
