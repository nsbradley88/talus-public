// Laguna verify attention, grouped kernel vs the per-token kernel on synthetic q8 pools (no model needed).
//   talus-vattn-bench [ctx] [ntoks,...] [sparse 0/1/2] [iters]
// For each layer kind (full H 48 / SWA H 72 window 512 on a 1024-position ring) and window width n: the per-token
// kernel (old grid), the per-token kernel token-major, and the grouped kernel (NT 2/4/8) - partials compared bitwise
// against the per-token kernel, then each token's rows against a separate 1-token call at its own position (the decode
// step).  Timing: 12 pools cycled (the 12 full layers; cold in the 32 MB on-die cache), us per call.
// sparse 1: per (token, KV head) selections of 1024 blocks of 16 (sink + recent + a shared top set with per-token
// drift, as a window's selections look); sparse 2: independent random selections (worst case for sharing).
#include "../../src/laguna/laguna_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

extern "C" void talus_laguna_decode_attn_grp(int grp_min, int nt, int tokmajor, int swa);

static uint16_t f2h(float f) {   // f32 -> f16 bits, round to nearest even (normal range only)
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000; int e = (int) ((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
    if (e <= 0) return (uint16_t) s;
    uint32_t h = s | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t r = m & 0x1fff;
    if (r > 0x1000 || (r == 0x1000 && (h & 1))) ++h;
    return (uint16_t) h;
}

int main(int argc, char** argv) {
    const int ctx = argc > 1 ? std::atoi(argv[1]) : 32768;
    std::vector<int> ntoks;
    {
        std::string a = argc > 2 ? argv[2] : "2,5,8,16,32,64";
        for (size_t i = 0; i < a.size();) { size_t j = a.find(',', i); if (j == std::string::npos) j = a.size(); ntoks.push_back(std::atoi(a.substr(i, j - i).c_str())); i = j + 1; }
    }
    const int sparse = argc > 3 ? std::atoi(argv[3]) : 0;
    const int iters = argc > 4 ? std::atoi(argv[4]) : 50;
    const int n_kv = 8, kD = 128, split = 16, maxn = *std::max_element(ntoks.begin(), ntoks.end());
    const int full_stride = ((ctx + maxn + 15) & ~15) + 64, swa_stride = 1024;
    const int sel_k = 1024, bs = 16;
    constexpr int kPools = 12;
    std::mt19937 rng(7);
    auto alloc_pool = [&](int stride, int8_t** k, int8_t** v, uint16_t** ks, uint16_t** vs) {
        const size_t n = (size_t) stride * n_kv * kD;
        std::vector<int8_t> h(n);
        for (auto& x : h) x = (int8_t) ((int) (rng() % 255) - 127);
        cudaMalloc(k, n); cudaMalloc(v, n);
        cudaMemcpy(*k, h.data(), n, cudaMemcpyHostToDevice);
        for (auto& x : h) x = (int8_t) ((int) (rng() % 255) - 127);
        cudaMemcpy(*v, h.data(), n, cudaMemcpyHostToDevice);
        const size_t ns = (size_t) stride * n_kv * 4;
        std::vector<uint16_t> sc(ns);
        for (auto& x : sc) x = f2h(0.002f + 0.02f * (float) (rng() % 1000) / 1000.0f);
        cudaMalloc(ks, ns * 2); cudaMalloc(vs, ns * 2);
        cudaMemcpy(*ks, sc.data(), ns * 2, cudaMemcpyHostToDevice);
        for (auto& x : sc) x = f2h(0.002f + 0.02f * (float) (rng() % 1000) / 1000.0f);
        cudaMemcpy(*vs, sc.data(), ns * 2, cudaMemcpyHostToDevice);
    };
    int8_t *fk[kPools], *fv[kPools]; uint16_t *fks[kPools], *fvs[kPools];
    for (int i = 0; i < kPools; ++i) alloc_pool(full_stride, &fk[i], &fv[i], &fks[i], &fvs[i]);
    int8_t *sk, *sv; uint16_t *sks, *svs;
    alloc_pool(swa_stride, &sk, &sv, &sks, &svs);
    float *q, *part_a, *ml_a, *part_b, *ml_b;
    const size_t qn = (size_t) maxn * 72 * kD, pn = (size_t) maxn * split * 72 * kD, mn = (size_t) maxn * split * 72 * 2;
    {
        std::vector<float> hq(qn);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& x : hq) x = nd(rng);
        cudaMalloc(&q, qn * 4); cudaMemcpy(q, hq.data(), qn * 4, cudaMemcpyHostToDevice);
    }
    cudaMalloc(&part_a, pn * 4); cudaMalloc(&ml_a, mn * 4); cudaMalloc(&part_b, pn * 4); cudaMalloc(&ml_b, mn * 4);
    int* pos; cudaMalloc(&pos, 4 * (maxn + 1));
    const int p0 = ctx - maxn;
    // selections [maxn][n_kv][sel_k]
    int* sel = nullptr;
    if (sparse) {
        std::vector<int> hs((size_t) maxn * n_kv * sel_k);
        for (int kvh = 0; kvh < n_kv; ++kvh) {
            const int nb0 = (p0) / bs + 1;
            std::vector<int> base;
            {   // a shared set: sink 8 + recent 64 + random rest
                std::set<int> b;
                for (int x = 0; x < 8; ++x) b.insert(x);
                for (int x = nb0 - 64; x < nb0; ++x) b.insert(x);
                while ((int) b.size() < sel_k) b.insert((int) (rng() % nb0));
                base.assign(b.begin(), b.end());
            }
            for (int t = 0; t < maxn; ++t) {
                const int nb = (p0 + t) / bs + 1;
                std::set<int> b;
                if (sparse == 1) {
                    b.insert(base.begin(), base.end());
                    for (int x = std::max(0, nb - 64); x < nb; ++x) b.insert(x);
                    for (int d = 0; d < 6; ++d) b.erase(std::next(b.begin(), 8 + (int) (rng() % (b.size() - 80))));   // drift
                    while ((int) b.size() > sel_k) b.erase(std::next(b.begin(), 8 + (int) (rng() % (b.size() - 80))));
                    while ((int) b.size() < sel_k) b.insert((int) (rng() % nb));
                } else {
                    while ((int) b.size() < sel_k) b.insert((int) (rng() % nb));
                }
                std::vector<int> v(b.begin(), b.end());
                std::copy(v.begin(), v.end(), hs.begin() + ((size_t) t * n_kv + kvh) * sel_k);
            }
        }
        cudaMalloc(&sel, hs.size() * 4);
        cudaMemcpy(sel, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice);
    }
    cudaStream_t s; cudaStreamCreate(&s);
    bool all_ok = true;
    for (int full = 1; full >= 0; --full) {
        const int H = full ? 48 : 72, window = full ? 0 : 512, stride = full ? full_stride : swa_stride;
        const int* sl = (full && sparse) ? sel : nullptr;
        for (int n : ntoks) {
            cudaMemcpy(pos, &p0, 4, cudaMemcpyHostToDevice);
            int cur = 0;
            auto run = [&](float* pa, float* ma, bool cycle) {
                const int8_t* kp = full ? fk[cur] : sk; const int8_t* vp = full ? fv[cur] : sv;
                const uint16_t* ks = full ? fks[cur] : sks; const uint16_t* vs = full ? fvs[cur] : svs;
                if (cycle) cur = (cur + 1) % kPools;
                strata::laguna::decode_attn(q, (const uint16_t*) kp, (const uint16_t*) vp, ks, vs, H, n_kv, window, stride,
                                            pos, split, pa, ma, s, n, sl, sel_k, true, bs);
            };
            auto timeit = [&](float* pa, float* ma) {
                for (int i = 0; i < 3; ++i) run(pa, ma, true);
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                cudaEventRecord(e0, s);
                for (int i = 0; i < iters; ++i) run(pa, ma, true);
                cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                cudaEventDestroy(e0); cudaEventDestroy(e1);
                return ms * 1000.0f / iters;
            };
            const size_t np = (size_t) n * split * H * kD, nm = (size_t) n * split * H * 2;
            std::vector<float> ra(np), rma(nm), rb(np), rmb(nm);
            // reference: per-token kernel, old grid, pool 0
            talus_laguna_decode_attn_grp(0, 0, 0, 0);
            cur = 0; run(part_a, ml_a, false); cudaStreamSynchronize(s);
            cudaMemcpy(ra.data(), part_a, np * 4, cudaMemcpyDeviceToHost); cudaMemcpy(rma.data(), ml_a, nm * 4, cudaMemcpyDeviceToHost);
            const float t_old = timeit(part_b, ml_b);
            // decode-step check (1 token at its own position), the reference's property
            int dec_bad = 0;
            for (int t = 0; t < n; ++t) {
                const int pt = p0 + t;
                cudaMemcpy(pos, &pt, 4, cudaMemcpyHostToDevice);
                cur = 0;
                const int8_t* kp = full ? fk[0] : sk; const int8_t* vp = full ? fv[0] : sv;
                const uint16_t* ks = full ? fks[0] : sks; const uint16_t* vs = full ? fvs[0] : svs;
                strata::laguna::decode_attn(q + (size_t) t * H * kD, (const uint16_t*) kp, (const uint16_t*) vp, ks, vs, H, n_kv,
                                            window, stride, pos, split, part_b, ml_b, s, 1,
                                            sl ? sl + (size_t) t * n_kv * sel_k : nullptr, sel_k, true, bs);
                cudaStreamSynchronize(s);
                std::vector<float> d((size_t) split * H * kD), dm((size_t) split * H * 2);
                cudaMemcpy(d.data(), part_b, d.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(dm.data(), ml_b, dm.size() * 4, cudaMemcpyDeviceToHost);
                dec_bad += std::memcmp(d.data(), ra.data() + (size_t) t * split * H * kD, d.size() * 4) != 0;
                dec_bad += std::memcmp(dm.data(), rma.data() + (size_t) t * split * H * 2, dm.size() * 4) != 0;
            }
            cudaMemcpy(pos, &p0, 4, cudaMemcpyHostToDevice);
            std::printf("%s ctx %d sparse %d n %2d: old %8.1f us  dec-check %s", full ? "full" : "swa ", ctx, sl ? sparse : 0, n,
                        t_old, dec_bad ? "FAIL" : "ok");
            struct V { const char* name; int gm, nt, tm; } vars[] = {{"tokmaj", 0, 0, 1}, {"grp4", 2, 4, 0}, {"g4c2", 2, 24, 0}, {"grp8", 2, 8, 0}, {"g8c2", 2, 28, 0}};
            for (const V& v : vars) {
                if (v.nt % 10 == 8 && !full) continue;
                talus_laguna_decode_attn_grp(v.gm, v.nt, v.tm, 1);
                cudaMemset(part_b, 0xff, np * 4); cudaMemset(ml_b, 0xff, nm * 4);
                cur = 0; run(part_b, ml_b, false); cudaStreamSynchronize(s);
                cudaMemcpy(rb.data(), part_b, np * 4, cudaMemcpyDeviceToHost); cudaMemcpy(rmb.data(), ml_b, nm * 4, cudaMemcpyDeviceToHost);
                const bool ok = std::memcmp(ra.data(), rb.data(), np * 4) == 0 && std::memcmp(rma.data(), rmb.data(), nm * 4) == 0;
                all_ok &= ok;
                const float tv = timeit(part_b, ml_b);
                std::printf("  | %s %8.1f us %s", v.name, tv, ok ? "bitwise" : "DIFF");
            }
            std::printf("\n");
            talus_laguna_decode_attn_grp(-1, -1, -1, 0);
        }
    }
    std::printf("%s\n", all_ok ? "ALL BITWISE" : "SOME DIFFER");
    return all_ok ? 0 : 1;
}
