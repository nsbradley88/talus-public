// src/prefill/tp_bench.cu - talus-pf-bench: parity and timing of the Talus prefill kernels against the kernels they
// replace, on synthetic data at Qwen3.8-27B's shapes.
//
//   talus-pf-bench fa [T] [depth ...]      tp_fa (both variants) vs llama.cpp's fattn-mma-f16 (strata::prefill::fattn)
//   talus-pf-bench i8 [T] [type ...]       tp_i8gemm vs llama.cpp's MMQ at the 27B's projection shapes (bitwise + time)
//   talus-pf-bench w4 [T]                  tp_w4a4 pack + GEMM against a host model of the same arithmetic, time vs MMQ
//   talus-pf-bench fuse [T]                swiglu_q8_mmq vs the engine's k_swiglu + MMQ quantize (bitwise + time)
//
// Copyright (c) 2026 nsbradley88
// SPDX-License-Identifier: MIT
#include "strata/prefill/fattn_prefill.hpp"
#include "strata/prefill/tp_fa.hpp"
#include "strata/prefill/tp_fuse.hpp"
#include "strata/prefill/tp_i8gemm.hpp"
#include "strata/prefill/tp_w4a4.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/tp_moe.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_w16.hpp"
#include "strata/prefill/gemm.hpp"
#include "wmma_gemm.h"
#include <dlfcn.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                 \
    do {                                                                                      \
        hipError_t e_ = (x);                                                                  \
        if (e_ != hipSuccess) {                                                               \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                     \
        }                                                                                     \
    } while (0)

namespace {

template <class F> float time_ms(hipStream_t s, int reps, F f) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    f();   // warm
    CK(hipStreamSynchronize(s));
    CK(hipEventRecord(a, s));
    for (int i = 0; i < reps; ++i) f();
    CK(hipEventRecord(b, s));
    CK(hipEventSynchronize(b));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return ms / reps;
}

struct Diff { double max_abs = 0, mean_abs = 0, max_ref = 0; size_t nan = 0; };
Diff diff(const std::vector<float>& ref, const std::vector<float>& x) {
    Diff d;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(x[i])) { ++d.nan; continue; }
        const double e = std::fabs((double) ref[i] - x[i]);
        d.max_abs = std::max(d.max_abs, e);
        d.mean_abs += e;
        d.max_ref = std::max(d.max_ref, (double) std::fabs(ref[i]));
    }
    d.mean_abs /= (double) ref.size();
    return d;
}

int bench_fa(int argc, char** argv) {
    const int H = 24, HKV = 4, D = 256;
    int T = argc > 0 ? std::atoi(argv[0]) : 512;
    std::vector<int> depths;
    for (int i = 1; i < argc; ++i) depths.push_back(std::atoi(argv[i]));
    if (depths.empty()) depths = {0, 2048, 8192, 16384, 31744};
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fails = 0;
    for (int p0 : depths) {
        const int64_t ctx = (int64_t) p0 + T + 512;   // + the llama.cpp kernel's 256-key stride padding
        const int64_t vt_cells = (ctx + 31) / 32 * 32;
        std::vector<__half> hk((size_t) HKV * ctx * D), hv((size_t) HKV * ctx * D), hvt((size_t) HKV * vt_cells * D);
        for (auto& x : hk) x = __float2half(nd(rng));
        for (auto& x : hv) x = __float2half(nd(rng));
        for (int kvh = 0; kvh < HKV; ++kvh)
            for (int64_t c = 0; c < ctx; ++c)
                for (int d = 0; d < D; ++d)
                    hvt[(((size_t) kvh * (vt_cells / 16) + c / 16) * D + d) * 16 + c % 16] = hv[((size_t) kvh * ctx + c) * D + d];
        std::vector<float> hq((size_t) T * H * D), hg((size_t) T * H * D);
        for (auto& x : hq) x = nd(rng);
        for (auto& x : hg) x = nd(rng);
        void *k, *v, *vt, *mask;
        float *q, *g, *o_ref, *o_new;
        CK(hipMalloc(&k, hk.size() * 2));
        CK(hipMalloc(&v, hv.size() * 2));
        CK(hipMalloc(&vt, hvt.size() * 2));
        CK(hipMalloc(&q, hq.size() * 4));
        CK(hipMalloc(&g, hg.size() * 4));
        CK(hipMalloc(&o_ref, hq.size() * 4));
        CK(hipMalloc(&o_new, hq.size() * 4));
        CK(hipMalloc(&mask, strata::prefill::fattn::mask_bytes(T, p0 + T)));
        CK(hipMemcpy(k, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(v, hv.data(), hv.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(vt, hvt.data(), hvt.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(q, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(g, hg.data(), hg.size() * 4, hipMemcpyHostToDevice));
        const float scale = 1.0f / 16.0f;
        // the reference zeroes K / V past p0 + T (the new kernel reads them as zeros): run it first
        auto ref = [&] { strata::prefill::fattn::attend(q, k, v, o_ref, T, H, HKV, D, ctx, p0, scale, mask, s); };
        const float t_ref = std::getenv("TP_BENCH_FA_NOREF") ? 1.f : time_ms(s, 5, ref);
        if (std::getenv("TP_BENCH_FA_NOREF")) ref();
        std::vector<float> r(hq.size()), x(hq.size());
        CK(hipMemcpy(r.data(), o_ref, r.size() * 4, hipMemcpyDeviceToHost));
        const double flop = 4.0 * T * H * D * ((double) p0 + (T + 1) / 2.0);
        std::printf("fa T %d depth %6d: llama.cpp fattn %8.3f ms (%5.1f TFLOPS)\n", T, p0, t_ref, flop / t_ref / 1e9);
        std::vector<float> xs[8];
        std::vector<int> vars = {1, 2, 3};
        if (const char* e = std::getenv("TP_BENCH_FA_VARIANTS")) {
            vars.clear();
            for (const char* c = e; *c; ++c)
                if (*c >= '1' && *c <= '7') vars.push_back(*c - '0');
        }
        for (int var : vars) {
            auto run = [&] {
                talus::prefill::fa_prefill(q, k, vt, nullptr, o_new, T, H, HKV, D, ctx, vt_cells, p0, scale, var, s);
            };
            const float t = time_ms(s, 5, run);
            CK(hipMemcpy(x.data(), o_new, x.size() * 4, hipMemcpyDeviceToHost));
            xs[var] = x;
            const Diff d = diff(r, x);
            const bool ok = d.nan == 0 && d.max_abs < 2e-2 * std::max(1.0, d.max_ref);
            fails += !ok;
            std::printf("fa T %d depth %6d: tp_fa v%d      %8.3f ms (%5.1f TFLOPS, x%.2f)  max|d| %.3g mean|d| %.3g max|ref| %.3g nan %zu  %s\n",
                        T, p0, var, t, flop / t / 1e9, t_ref / t, d.max_abs, d.mean_abs, d.max_ref, d.nan, ok ? "ok" : "FAIL");
            if (var != vars[0]) {   // bitwise against the first listed variant (longctx: exact rewrites of v3)
                const bool same = std::memcmp(x.data(), xs[vars[0]].data(), x.size() * 4) == 0;
                std::printf("fa T %d depth %6d: v%d vs v%d: %s\n", T, p0, var, vars[0], same ? "bitwise the same" : "DIFFERS");
                fails += !same && std::getenv("TP_BENCH_FA_EXACT") != nullptr;
            }
        }
        // the fused gate: tp_fa(gate) == tp_fa then x * sigmoid(gate)
        talus::prefill::fa_prefill(q, k, vt, g, o_new, T, H, HKV, D, ctx, vt_cells, p0, scale, vars.back(), s);
        std::vector<float> xg(hq.size());
        CK(hipMemcpy(xg.data(), o_new, xg.size() * 4, hipMemcpyDeviceToHost));
        talus::prefill::fa_prefill(q, k, vt, nullptr, o_new, T, H, HKV, D, ctx, vt_cells, p0, scale, vars.back(), s);
        CK(hipMemcpy(x.data(), o_new, x.size() * 4, hipMemcpyDeviceToHost));
        size_t gbad = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const float want = x[i] * (1.0f / (1.0f + std::exp(-hg[i])));
            if (std::fabs(want - xg[i]) > 1e-5f * std::max(1.0f, std::fabs(want))) ++gbad;
        }
        std::printf("fa T %d depth %6d: fused gate %zu mismatches  %s\n", T, p0, gbad, gbad ? "FAIL" : "ok");
        fails += gbad != 0;
        // determinism: the same output when the cells past the chunk hold other values
        {
            std::vector<__half> junk((size_t) 512 * D);
            for (auto& j : junk) j = __float2half(nd(rng) * 100.f);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                CK(hipMemcpy((__half*) k + ((size_t) kvh * ctx + p0 + T) * D, junk.data(), junk.size() * 2, hipMemcpyHostToDevice));
                for (int c = 0; c < 16 && p0 + T + c < vt_cells; ++c) {
                    const int64_t cell = p0 + T + c;
                    for (int d = 0; d < D; ++d)
                        CK(hipMemcpy((__half*) vt + (((size_t) kvh * (vt_cells / 16) + cell / 16) * D + d) * 16 + cell % 16,
                                     &junk[(size_t) c * D + d], 2, hipMemcpyHostToDevice));
                }
            }
            for (int var : vars) {
                talus::prefill::fa_prefill(q, k, vt, nullptr, o_new, T, H, HKV, D, ctx, vt_cells, p0, scale, var, s);
                std::vector<float> y(hq.size());
                CK(hipMemcpy(y.data(), o_new, y.size() * 4, hipMemcpyDeviceToHost));
                const bool same = std::memcmp(y.data(), xs[var].data(), y.size() * 4) == 0;
                std::printf("fa T %d depth %6d: v%d with junk past the chunk: %s\n", T, p0, var,
                            same ? "bitwise the same" : "DIFFERS");
                fails += !same;
            }
        }
        CK(hipFree(k)); CK(hipFree(v)); CK(hipFree(vt)); CK(hipFree(q)); CK(hipFree(g));
        CK(hipFree(o_ref)); CK(hipFree(o_new)); CK(hipFree(mask));
    }
    std::printf("fa: %d failures\n", fails);
    return fails ? 1 : 0;
}

struct I8Acc { int fails = 0; double t_ref = 0, t_new = 0, ops = 0; };
int i8_block_bytes(int t) {
    switch (t) { case 23: return 136; case 21: return 110; case 18: return 98; case 16: return 66; case 20: return 18;
                 case 11: return 110; case 17: return 74; case 22: return 82; case 12: return 144; case 13: return 176;
                 case 8: return 34; case 6: return 22; case 29: return 56; case 14: return 210; default: return 0; }
}
// one dense case: MMQ (when it covers the type) vs tp_i8gemm, bitwise + time.  ref: compare against MMQ (else only time)
void i8_case(int type, int N, int K, int T, I8Acc& acc, std::mt19937& rng, hipStream_t s) {
    namespace MMQ = strata::prefill::mmq;
    static MMQ::Context ctx;
    const int bb = i8_block_bytes(type);
    const bool has_mmq = MMQ::supported(type);
    if (!bb || !talus::prefill::i8x_supported(type)) {
        std::printf("i8: type %d not covered\n", type);
        return;
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    const size_t wb = (size_t) N * (K / (type == 8 || type == 6 || type == 20 ? 32 : 256)) * bb;
    std::vector<uint8_t> hw(wb + 4096, 0);
    for (size_t i = 0; i < wb; ++i) hw[i] = (uint8_t) (rng() & 0xff);
    for (size_t b = 0; b < wb / bb; ++b) {   // every block's FP16 scale(s) to sane values
        __half d = __float2half((rng() & 1 ? 1.f : -1.f) * ud(rng)), m = __float2half(ud(rng));
        if (type == 11 || type == 14) std::memcpy(&hw[b * bb + bb - 2], &d, 2);   // Q3_K / Q6_K: d is the last field
        else if (type == 29) {   // IQ1_M: d in the top nibbles of the 4 scale u16s
            uint16_t u; std::memcpy(&u, &d, 2);
            uint16_t* sc = (uint16_t*) &hw[b * bb + 48];
            for (int j = 0; j < 4; ++j) sc[j] = (uint16_t) ((sc[j] & 0x0fff) | (((u >> (4 * j)) & 0xf) << 12));
        } else std::memcpy(&hw[b * bb], &d, 2);
        if (type == 12 || type == 13) std::memcpy(&hw[b * bb + 2], &m, 2);   // Q4_K / Q5_K: dmin
    }
    std::vector<float> hx((size_t) T * K);
    for (auto& v : hx) v = nd(rng);
    void *w, *xq;
    float *x, *d_ref, *d_new;
    int32_t *bounds, *ids;
    CK(hipMalloc(&w, hw.size()));
    CK(hipMalloc(&x, hx.size() * 4));
    CK(hipMalloc(&xq, MMQ::q8_bytes(T, K)));
    CK(hipMalloc(&d_ref, (size_t) T * N * 4));
    CK(hipMalloc(&d_new, (size_t) T * N * 4));
    CK(hipMalloc(&bounds, 8));
    CK(hipMalloc(&ids, (size_t) T * 4));
    CK(hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    const int32_t hb[2] = {0, T};
    CK(hipMemcpy(bounds, hb, 8, hipMemcpyHostToDevice));
    MMQ::iota(ids, T, s);
    MMQ::quantize(x, nullptr, xq, has_mmq ? type : 22, K, K, T, s);   // (IQ1_M: the IQ2_S D4 layout)
    float t_ref = -1.f;
    if (has_mmq) {
        MMQ::Product p;
        p.w = w; p.type = type; p.w_rows = N; p.w_cols = K; p.expert_bytes = wb; p.n = 1; p.xq = xq;
        p.bounds = bounds; p.ids = ids; p.total_rows = T; p.max_rows = T; p.dst = d_ref; p.ld_dst = N;
        t_ref = time_ms(s, 5, [&] { ctx.run(p, s); });
    }
    talus::prefill::I8Cache cache;
    if (!cache.add(w, type, N, K, s)) { std::printf("i8: expansion failed (type %d N %d K %d)\n", type, N, K); ++acc.fails; return; }
    const talus::prefill::I8Mat* m = cache.find(w);
    const float t_new = time_ms(s, 5, [&] { talus::prefill::i8gemm(*m, xq, T, d_new, N, s); });
    size_t ndiff = 0;
    double maxd = 0;
    if (has_mmq) {
        std::vector<float> r((size_t) T * N), y((size_t) T * N);
        CK(hipMemcpy(r.data(), d_ref, r.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(y.data(), d_new, y.size() * 4, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < r.size(); ++i) ndiff += std::memcmp(&r[i], &y[i], 4) != 0;
        maxd = diff(r, y).max_abs;
    }
    const double ops = 2.0 * T * N * K;
    acc.fails += ndiff != 0;
    if (has_mmq) acc.t_ref += t_ref;
    acc.t_new += t_new;
    acc.ops += ops;
    std::printf("i8 type %2d N %5d K %5d T %d: MMQ %7.3f ms (%5.1f TOPS)  tp_i8gemm %7.3f ms (%5.1f TOPS, %4.1f%% of 45, x%.2f)  "
                "%zu of %zu outputs differ (max|d| %.3g)  %s\n",
                type, N, K, T, t_ref, has_mmq ? ops / t_ref / 1e9 : 0.0, t_new, ops / t_new / 1e9, ops / t_new / 1e9 / 45.0 * 100.0,
                has_mmq ? t_ref / t_new : 0.0, ndiff, (size_t) T * N, maxd, !has_mmq ? "no-ref" : ndiff ? "DIFF" : "bitwise");
    CK(hipFree(w)); CK(hipFree(x)); CK(hipFree(xq)); CK(hipFree(d_ref)); CK(hipFree(d_new));
    CK(hipFree(bounds)); CK(hipFree(ids));
}

int bench_i8(int argc, char** argv) {
    int T = argc > 0 ? std::atoi(argv[0]) : 512;
    std::vector<int> types;
    for (int i = 1; i < argc; ++i) types.push_back(std::atoi(argv[i]));
    if (types.empty()) types = {23, 21, 18, 11, 17, 22, 12, 13};   // IQ4_XS IQ3_S IQ3_XXS Q3_K IQ2_XS IQ2_S Q4_K Q5_K
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {6144, 5120}, {5120, 6144}, {12288, 5120}, {1024, 5120}};
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(77);
    I8Acc acc;
    for (int type : types)
        for (const auto& sh : shapes) i8_case(type, sh[0], sh[1], T, acc, rng, s);
    std::printf("i8: MMQ %.2f ms, tp_i8gemm %.2f ms over all shapes (x%.2f, %.1f TOPS); %d failures\n", acc.t_ref, acc.t_new,
                acc.t_ref / std::max(1e-9, acc.t_new), acc.ops / acc.t_new / 1e9, acc.fails);
    return acc.fails ? 1 : 0;
}

// one tp_i8gemm launch after one warmup launch (thread-trace target): talus-pf-bench i8one TYPE N K T [REPS]
int bench_i8one(int argc, char** argv) {
    namespace MMQ = strata::prefill::mmq;
    if (argc < 4) { std::fprintf(stderr, "i8one TYPE N K T [REPS]\n"); return 2; }
    const int type = std::atoi(argv[0]), N = std::atoi(argv[1]), K = std::atoi(argv[2]), T = std::atoi(argv[3]);
    const int reps = argc > 4 ? std::atoi(argv[4]) : 1;
    const int bb = i8_block_bytes(type);
    if (!bb || !talus::prefill::i8x_supported(type)) { std::printf("i8one: type %d not covered\n", type); return 2; }
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t wb = (size_t) N * (K / (type == 8 || type == 6 || type == 20 ? 32 : 256)) * bb;
    std::vector<uint8_t> hw(wb + 4096, 0);
    for (size_t i = 0; i < wb; ++i) hw[i] = (uint8_t) (rng() & 0xff);
    for (size_t b = 0; b < wb / bb; ++b) {
        __half d = __float2half((rng() & 1 ? 1.f : -1.f) * ud(rng)), m = __float2half(ud(rng));
        if (type == 11 || type == 14) std::memcpy(&hw[b * bb + bb - 2], &d, 2);
        else std::memcpy(&hw[b * bb], &d, 2);
        if (type == 12 || type == 13) std::memcpy(&hw[b * bb + 2], &m, 2);
    }
    std::vector<float> hx((size_t) T * K);
    for (auto& v : hx) v = nd(rng);
    void *w, *xq;
    float *x, *dst;
    CK(hipMalloc(&w, hw.size()));
    CK(hipMalloc(&x, hx.size() * 4));
    CK(hipMalloc(&xq, MMQ::q8_bytes(T, K)));
    CK(hipMalloc(&dst, (size_t) T * N * 4));
    CK(hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    MMQ::quantize(x, nullptr, xq, MMQ::supported(type) ? type : 22, K, K, T, s);
    talus::prefill::I8Cache cache;
    if (!cache.add(w, type, N, K, s)) { std::printf("i8one: expansion failed\n"); return 1; }
    const talus::prefill::I8Mat* m = cache.find(w);
    const float t = time_ms(s, reps, [&] { talus::prefill::i8gemm(*m, xq, T, dst, N, s); });
    std::printf("i8one type %d N %d K %d T %d: %.3f ms (%.1f TOPS)\n", type, N, K, T, t, 2.0 * T * N * K / t / 1e9);
    CK(hipFree(w)); CK(hipFree(x)); CK(hipFree(xq)); CK(hipFree(dst));
    return 0;
}

// per-model dense shapes (type N K): talus-pf-bench i8m MODEL [T]
//   MODEL: 27b | fn | glm | laguna | all
int bench_i8m(int argc, char** argv) {
    const std::string model = argc > 0 ? argv[0] : "all";
    const int T = argc > 1 ? std::atoi(argv[1]) : 2048;
    struct C { const char* m; int type, n, k; };
    static const C cases[] = {
        // Qwen3.8-27B (UD-IQ4_XS): ffn gate/up/down, attn output, qkv; the mix of types the model uses
        {"27b", 23, 17408, 5120}, {"27b", 23, 5120, 17408}, {"27b", 21, 17408, 5120}, {"27b", 21, 5120, 17408},
        {"27b", 18, 17408, 5120}, {"27b", 13, 17408, 5120}, {"27b", 13, 5120, 17408}, {"27b", 12, 5120, 17408},
        {"27b", 11, 17408, 5120}, {"27b", 22, 17408, 5120}, {"27b", 17, 17408, 5120},
        {"27b", 23, 5120, 6144}, {"27b", 13, 10240, 5120}, {"27b", 12, 1024, 5120},
        {"27b", 14, 17408, 5120}, {"27b", 14, 5120, 17408}, {"27b", 14, 5120, 6144},
        // Flash-Next (Q8_0 dense): attn qkv / q / output, shared expert gate/up
        {"fn", 8, 10240, 2560}, {"fn", 8, 12288, 2560}, {"fn", 8, 2560, 6144}, {"fn", 8, 640, 2560},
        // GLM-5.3-Flash (Q4_K dense): attn q/k/v, output, shared expert, layer 0-2 FFN, q_a
        {"glm", 12, 8192, 4096}, {"glm", 12, 4096, 8192}, {"glm", 12, 4096, 16384}, {"glm", 12, 2048, 4096},
        {"glm", 12, 4096, 2048}, {"glm", 12, 1536, 4096}, {"glm", 12, 12288, 4096}, {"glm", 12, 4096, 12288},
        // Laguna-S-2.1 (Q5_K attention, Q8_0 layer-0 FFN)
        {"laguna", 13, 9216, 3072}, {"laguna", 13, 3072, 9216}, {"laguna", 13, 6144, 3072}, {"laguna", 13, 3072, 6144},
        {"laguna", 8, 12288, 3072}, {"laguna", 8, 3072, 12288},
        // MiMo-V2.6-Flash dense (Q4_K attention qkv / output, Q6_K layer-0 FFN)
        {"mimo", 12, 13568, 4096}, {"mimo", 12, 14848, 4096}, {"mimo", 12, 4096, 8192}, {"mimo", 14, 16384, 4096},
        {"mimo", 14, 4096, 16384},
        // MiMo IQ1_M (no MMQ reference: time only)
        {"iq1m", 29, 2048, 4096}, {"iq1m", 29, 4096, 2048},
    };
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(77);
    I8Acc all;
    for (const char* mm : {"27b", "fn", "glm", "laguna", "mimo", "iq1m"}) {
        if (model != "all" && model != mm) continue;
        I8Acc acc;
        for (const C& c : cases)
            if (!std::strcmp(c.m, mm)) i8_case(c.type, c.n, c.k, T, acc, rng, s);
        std::printf("i8m %s T %d: MMQ %.2f ms, tp_i8gemm %.2f ms (%.1f TOPS, %.1f%% of 45); %d failures\n", mm, T, acc.t_ref,
                    acc.t_new, acc.ops / acc.t_new / 1e9, acc.ops / acc.t_new / 1e9 / 45.0 * 100.0, acc.fails);
        all.fails += acc.fails;
    }
    return all.fails ? 1 : 0;
}

// host model of tp_w4a4: R = H D / 16 per 256-block, symmetric int4 per (row, group)
static void host_rotate(float* v, const int8_t* sg) {   // 256 values in place
    for (int e = 0; e < 256; ++e) v[e] *= (float) sg[e];
    for (int h = 1; h < 256; h <<= 1)
        for (int i = 0; i < 256; ++i)
            if (!(i & h)) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
    for (int e = 0; e < 256; ++e) v[e] *= 0.0625f;
}
static void host_quant_rows(const float* x, int rows, int k, const int8_t* sg, std::vector<int8_t>& q, std::vector<float>& sc) {
    const int G = k / 256;
    q.assign((size_t) rows * k, 0);
    sc.assign((size_t) rows * G, 0.f);
    std::vector<float> v(256);
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < G; ++g) {
            std::memcpy(v.data(), x + (size_t) r * k + g * 256, 1024);
            host_rotate(v.data(), sg + g * 256);
            float amax = 0.f;
            for (float a : v) amax = std::max(amax, std::fabs(a));
            const float inv = amax > 0.f ? 7.0f / amax : 0.f;
            for (int e = 0; e < 256; ++e) q[(size_t) r * k + g * 256 + e] = (int8_t) std::min(7.f, std::max(-7.f, std::rint(v[e] * inv)));
            sc[(size_t) r * G + g] = amax / 7.0f;
        }
}


// the engines' SwiGLU (glm k_swiglu_clamp / laguna silu_mul_kernel expression), the fused gate/up reference
__global__ void k_bench_swiglu(const float* g, const float* __restrict__ u, float* o, int64_t n, float limit) {
    const int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.0f) { gv = fminf(gv, limit); uv = fminf(fmaxf(uv, -limit), limit); }
    o[i] = gv / (1.0f + expf(-gv)) * uv;
}

// ---------------------------------------------------------------------------------------------------------------------
// Prototype (bench only, not in the library): a W4A16-style grouped expert GEMM for IQ4_NL GGUF
// experts - bf16 activations, the weights decoded per 128-value K step into a bf16 LDS tile as bf16(d * kvalue), bf16
// WMMA accumulating in fp32 over the whole K (no per-block scale epilogue).  Numerics differ from MMQ (bf16 rounding of
// x and of d * q).  Layout as tp_moe v6: block = 128 weight rows x up to 128 routed rows, waves 4 row pairs x 2 token
// groups.  Activations bf16 [rows][k] (sorted by expert).
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
__constant__ float kIq4nlValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__global__ void __launch_bounds__(256) k_bench_moe_bf16_iq4nl(const uint8_t* __restrict__ w, size_t expert_bytes, int k,
                                                              const __bf16* __restrict__ x, const int32_t* __restrict__ bounds,
                                                              const int32_t* __restrict__ ids, float* __restrict__ dst, int64_t ldd) {
#if defined(__gfx1151__) || defined(__gfx1100__)
    constexpr int SS = 136;   // bf16 per tile row (128 + 8)
    __shared__ __attribute__((aligned(16))) __bf16 tile[128 * SS];
    const int e = blockIdx.z;
    const int r0 = bounds[e] + blockIdx.x * 128, r1 = min(bounds[e + 1], r0 + 128);
    if (r0 >= r1) return;
    const int nt = (r1 - r0 + 15) >> 4;
    const int lane = threadIdx.x, wv = threadIdx.y, l16 = lane & 15, hi = lane >> 4, tid = wv * 32 + lane;
    const int rp = wv & 3, tg = wv >> 2;
    const int ntw = (nt + 1 - tg) >> 1;
    const int row0 = blockIdx.y * 128;
    const size_t rowbytes = (size_t) (k / 32) * 18;
    const uint8_t* we = w + (size_t) e * expert_bytes;
    v8f acc[2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
    const __bf16* xr[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) xr[b] = x + (size_t) min(r0 + (2 * b + tg) * 16 + l16, r1 - 1) * k;
    for (int k0 = 0; k0 < k; k0 += 128) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 2; ++i) {   // 512 (row, 32-value block) units
            const int u = tid + 256 * i, row = u >> 2, bl = u & 3;
            const uint8_t* bp = we + (size_t) (row0 + row) * rowbytes + (size_t) (k0 / 32 + bl) * 18;
            const uint16_t* hp = (const uint16_t*) bp;
            const float d = __half2float(__ushort_as_half(hp[0]));
            uint16_t qw[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) qw[j] = hp[1 + j];
            __bf16* o = tile + row * SS + bl * 32;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int q = (qw[j >> 1] >> (8 * (j & 1))) & 0xff;
                o[j] = (__bf16) (d * kIq4nlValues[q & 15]);
                o[j + 16] = (__bf16) (d * kIq4nlValues[q >> 4]);
            }
        }
        __syncthreads();
#pragma unroll 2
        for (int kk = 0; kk < 8; ++kk) {
            v16bf A[2];
#pragma unroll
            for (int a = 0; a < 2; ++a) A[a] = *(const v16bf*) &tile[(rp * 32 + a * 16 + l16) * SS + kk * 16];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                if (b >= ntw) break;
                const v16bf B = *(const v16bf*) (xr[b] + k0 + kk * 16);
#pragma unroll
                for (int a = 0; a < 2; ++a) acc[a][b] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(A[a], B, acc[a][b]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        if (b >= ntw) break;
        const int r = r0 + (2 * b + tg) * 16 + l16;
        if (r >= r1) continue;
#pragma unroll
        for (int a = 0; a < 2; ++a) {
            const int64_t o = (int64_t) (ids ? ids[r] : r) * ldd + row0 + rp * 32 + a * 16 + hi;
#pragma unroll
            for (int i = 0; i < 8; ++i) dst[o + 2 * i] = acc[a][b][i];
        }
    }
#else
    (void) w; (void) expert_bytes; (void) k; (void) x; (void) bounds; (void) ids; (void) dst; (void) ldd;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// Flash-Next's native-pack prompt experts (gate/up IQ3_S, down IQ4_NL, n_embd 2560, n_ff 640, top 10 of 512)
// on the int8 kernels (fused::experts_native: native_w11) against the bf16 ones (experts_native_w16), with an FP64
// reference for a few rows of one expert.
//   talus-pf-bench w16 T [E TOPK]
namespace w16b {
constexpr int NE = 2560, NF = 640, GU_ROW = 1100, D_ROW = 360;
constexpr size_t UP_OFF = (size_t) NF * GU_ROW, DOWN_OFF = 2 * UP_OFF, BLOB = DOWN_OFF + (size_t) NE * D_ROW;
constexpr int KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
double h2d(const uint8_t* p) { __half h; std::memcpy(&h, p, 2); return (double) __half2float(h); }
// ggml's dequantize_row_iq3_s / iq4_nl of one row
void deq_iq3s(const uint8_t* row, int k, const uint32_t* grid, double* y) {
    for (int i = 0; i < k / 256; ++i) {
        const uint8_t* x = row + 110 * i;
        const double d = h2d(x);
        const uint8_t *qs = x + 2, *qh = x + 66, *signs = x + 74, *scales = x + 106;
        for (int j = 0; j < 8; ++j) {
            const double db = d * (1 + 2 * ((scales[j / 2] >> (4 * (j & 1))) & 0xf));
            for (int e = 0; e < 8; ++e) {
                const uint32_t g = grid[qs[8 * j + e] | (((qh[j] >> e) & 1) << 8)];
                const uint8_t sb = signs[4 * j + e / 2];
                for (int v = 0; v < 4; ++v) {
                    const int bit = 4 * (e & 1) + v;
                    y[256 * i + 32 * j + 4 * e + v] = db * (double) ((g >> (8 * v)) & 0xff) * (((sb >> bit) & 1) ? -1.0 : 1.0);
                }
            }
        }
    }
}
void deq_iq4nl(const uint8_t* row, int k, double* y) {
    for (int i = 0; i < k / 32; ++i) {
        const uint8_t* x = row + 18 * i;
        const double d = h2d(x);
        for (int j = 0; j < 16; ++j) { y[32 * i + j] = d * KV[x[2 + j] & 15]; y[32 * i + j + 16] = d * KV[x[2 + j] >> 4]; }
    }
}
// a native activation row (k values) dequantized
void deq_act(const uint8_t* a, int k, double* y) {
    for (int u = 0; u < k / 64; ++u) {
        const uint8_t* p = a + 80 * u;
        float d[2]; std::memcpy(d, p + 64, 8);
        for (int j = 0; j < 64; ++j) y[64 * u + j] = (double) (int8_t) p[j] * d[j / 32];
    }
}
double rel(const std::vector<double>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (a[i] - b[i]); d += b[i] * b[i]; }
    return std::sqrt(n / (d + 1e-300));
}
}  // namespace w16b


// the hyper-connection down projection (N 320, K 10240, BF16 -> FP32, token stride K + 64) on hipBLASLt (the
// loaded tuning table's solution, as the engine's Gemm::bf16) against pfg::kernel_hcd_exact forced (no solution-id check):
// bitwise compare + timing per T.   talus-pf-bench hcd T...   (needs STRATA_HIPBLASLT_TUNING)
int bench_hcd(int argc, char** argv) {
    constexpr int64_t N = 320, K = 10240, LDX = K + 64;
    hipStream_t s;
    CK(hipStreamCreate(&s));
    strata::prefill::Gemm gemm;
    std::string err;
    if (!gemm.init(s, 1 << 20, err)) { std::fprintf(stderr, "Gemm::init: %s\n", err.c_str()); return 2; }
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t) (u >> 16); };
    std::vector<uint16_t> w((size_t) N * K);
    for (auto& v : w) v = bf(0.05f * dist(rng));
    uint16_t* dw;
    CK(hipMalloc(&dw, w.size() * 2));
    CK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    for (int i = 0; i < argc; ++i) {
        const int64_t T = std::atoll(argv[i]);
        std::vector<uint16_t> x((size_t) T * LDX, 0);
        for (int64_t t = 0; t < T; ++t) for (int64_t k = 0; k < K; ++k) x[(size_t) t * LDX + k] = bf(dist(rng));
        uint16_t* dx; float *ya, *yb;
        CK(hipMalloc(&dx, x.size() * 2)); CK(hipMalloc(&ya, (size_t) T * N * 4)); CK(hipMalloc(&yb, (size_t) T * N * 4));
        CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemset(ya, 0xff, (size_t) T * N * 4)); CK(hipMemset(yb, 0xee, (size_t) T * N * 4));
        auto blt = [&] { gemm.bf16(dx, dw, ya, T, N, K, 0, 0.0f, LDX); };
        auto ex = [&] { strata_pf_hcdown_exact_bf16(dx, LDX, dw, yb, T, N, K, s); };
        blt(); ex();
        CK(hipStreamSynchronize(s));
        std::vector<float> a((size_t) T * N), b((size_t) T * N);
        CK(hipMemcpy(a.data(), ya, a.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), yb, b.size() * 4, hipMemcpyDeviceToHost));
        int64_t diff = 0;
        for (size_t j = 0; j < a.size(); ++j) diff += std::memcmp(&a[j], &b[j], 4) != 0;
        float ta = 1e30f, tb = 1e30f;
        for (int r = 0; r < 3; ++r) { ta = std::min(ta, time_ms(s, 10, blt)); tb = std::min(tb, time_ms(s, 10, ex)); }
        const double fl = 2.0 * T * N * K;
        CK(hipMemset(yb, 0xee, (size_t) T * N * 4));
        const bool took = gemm.bf16_hcd_exact(dx, LDX, dw, yb, T, N, K);   // the engine's dispatch (the solution-id check)
        CK(hipStreamSynchronize(s));
        int64_t diff2 = 0;
        if (took) {
            CK(hipMemcpy(b.data(), yb, b.size() * 4, hipMemcpyDeviceToHost));
            for (size_t j = 0; j < a.size(); ++j) diff2 += std::memcmp(&a[j], &b[j], 4) != 0;
        }
        std::printf("hcd T %6lld: dispatch %s (differing %lld)\n", (long long) T, took ? "TAKES the exact kernel" : "declines (hipBLASLt runs)",
                    (long long) diff2);
        std::printf("hcd T %6lld: hipBLASLt %.3f ms (%.1f TFLOPS)  exact kernel %.3f ms (%.1f TFLOPS)  outputs differing %lld of %lld%s\n",
                    (long long) T, ta, fl / ta / 1e9, tb, fl / tb / 1e9, (long long) diff, (long long) a.size(), diff ? "" : "  BITWISE");
        CK(hipFree(dx)); CK(hipFree(ya)); CK(hipFree(yb));
    }
    return 0;
}

int bench_w16(int argc, char** argv) {
    using namespace w16b;
    namespace F = strata::prefill::fused;
    if (argc < 1) { std::fprintf(stderr, "w16 T [E TOPK]\n"); return 2; }
    const int T = std::atoi(argv[0]), E = argc > 1 ? std::atoi(argv[1]) : 512, TK = argc > 2 ? std::atoi(argv[2]) : 10;
    const int64_t n = (int64_t) T * TK;
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(7);
    // experts: random codes, sane fp16 scales at every block's start
    std::vector<uint8_t> hb((size_t) E * BLOB);
    for (auto& v : hb) v = (uint8_t) rng();
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (int e = 0; e < E; ++e) {
        uint8_t* bl = hb.data() + (size_t) e * BLOB;
        for (size_t o = 0; o < DOWN_OFF; o += 110) { __half h = __float2half(ud(rng)); std::memcpy(bl + o, &h, 2); }
        for (size_t o = DOWN_OFF; o < BLOB; o += 18) { __half h = __float2half(ud(rng)); std::memcpy(bl + o, &h, 2); }
    }
    uint8_t* db;
    CK(hipMalloc(&db, hb.size() + 256));
    // TP_BENCH_W16_MIS=2: the blobs 2 bytes off a 4-byte boundary (the GGUF blocks' worst case)
    const size_t mis = std::getenv("TP_BENCH_W16_MIS") ? (size_t) std::atoi(std::getenv("TP_BENCH_W16_MIS")) : 0;
    uint8_t* dbo = db + mis;
    CK(hipMemcpy(dbo, hb.data(), hb.size(), hipMemcpyHostToDevice));
    // routing
    std::vector<int32_t> ids((size_t) n);
    for (int t = 0; t < T; ++t) {
        std::vector<int> sel;
        while ((int) sel.size() < TK) { int e = (int) (rng() % E); if (std::find(sel.begin(), sel.end(), e) == sel.end()) sel.push_back(e); }
        for (int k = 0; k < TK; ++k) ids[(size_t) t * TK + k] = sel[k];
    }
    int32_t *dids, *dslot, *dsrc;
    CK(hipMalloc(&dids, n * 4)); CK(hipMalloc(&dslot, n * 4)); CK(hipMalloc(&dsrc, n * 4));
    CK(hipMemcpy(dids, ids.data(), n * 4, hipMemcpyHostToDevice));
    const size_t gb = F::group_bytes(n, E);
    void* scr;
    CK(hipMalloc(&scr, gb));
    CK(hipMemset(scr, 0, gb));
    F::group(dids, n, TK, E, scr, dslot, dsrc, s);
    // activations
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) T * NE);
    for (auto& v : hx) v = nd(rng);
    float* dx;
    uint8_t* xa;
    CK(hipMalloc(&dx, hx.size() * 4));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&xa, F::act_bytes(T, NE)));
    F::quantize_act_native(dx, T, NE, xa, s);
    uint8_t *ha0, *ha1;
    float *dm0, *dm1;
    CK(hipMalloc(&ha0, F::act_bytes(n, NF))); CK(hipMalloc(&ha1, F::act_bytes(n, NF)));
    CK(hipMalloc(&dm0, (size_t) n * NE * 4)); CK(hipMalloc(&dm1, (size_t) n * NE * 4));
    CK(hipMemset(dm0, 0, (size_t) n * NE * 4)); CK(hipMemset(dm1, 0, (size_t) n * NE * 4));
    const F::NativeGeom g{21, 20, (size_t) GU_ROW, (size_t) D_ROW, UP_OFF, DOWN_OFF};
    std::vector<F::Batch> batches;
    for (int e0 = 0; e0 < E; e0 += F::kMaxBatch) {
        F::Batch b;
        b.e0 = e0; b.e1 = std::min(E, e0 + F::kMaxBatch);
        for (int e = b.e0; e < b.e1; ++e) b.blob[e - e0] = dbo + (size_t) e * BLOB;
        batches.push_back(b);
    }
    auto native = [&] { for (auto& b : batches) F::experts_native(b, g, E, n, scr, xa, dsrc, ha0, dm0, s); };
    auto w16 = [&](int ph) { for (auto& b : batches) F::experts_native_w16(b, g, E, n, scr, xa, dsrc, ha1, dm1, s, ph, true); };
    native();
    w16(0);
    CK(hipStreamSynchronize(s));
    const float tn = time_ms(s, 5, native), tw = time_ms(s, 5, [&] { w16(0); });
    const float tg = time_ms(s, 5, [&] { w16(1); }), td = time_ms(s, 5, [&] { w16(2); });
    w16(0);   // (the down-only runs read H written by the last gate/up run: the same values)
    CK(hipStreamSynchronize(s));
    const double fgu = 2.0 * n * 1280 * NE, fd = 2.0 * n * NE * NF;
    std::printf("w16 T %d E %d top %d: int8 native %.3f ms (%.1f TOPS)  bf16 %.3f ms (%.1f TOPS, x%.2f): gate/up %.3f ms (%.1f TOPS), "
                "down %.3f ms (%.1f TOPS)\n", T, E, TK, tn, (fgu + fd) / tn / 1e9, tw, (fgu + fd) / tw / 1e9, tn / tw, tg, fgu / tg / 1e9,
                td, fd / td / 1e9);
    // numerics: whole outputs bf16 vs int8, and an FP64 reference for up to 16 rows of the first expert with rows
    std::vector<float> o0((size_t) n * NE), o1((size_t) n * NE);
    CK(hipMemcpy(o0.data(), dm0, o0.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(o1.data(), dm1, o1.size() * 4, hipMemcpyDeviceToHost));
    {
        std::vector<double> a(o1.begin(), o1.end()), b(o0.begin(), o0.end());
        std::printf("  down rows: bf16 vs int8 rel L2 %.3e\n", rel(a, b));
        // the rows that disagree most (a correctness check: the two paths differ by ~1e-2 relative per row)
        int shown = 0;
        size_t nbad = 0;
        for (int64_t p = 0; p < n; ++p) {
            double nn = 0, dd = 0;
            for (int c = 0; c < NE; ++c) { const double x = o1[(size_t) p * NE + c], y = o0[(size_t) p * NE + c]; nn += (x - y) * (x - y); dd += y * y; }
            const double r = std::sqrt(nn / (dd + 1e-300));
            if (r > 0.1) { ++nbad; if (shown++ < 6) std::printf("    row %lld rel %.3e (|int8| %.3e)\n", (long long) p, r, std::sqrt(dd)); }
        }
        std::printf("  rows with rel > 0.1: %zu of %lld\n", nbad, (long long) n);
    }
    if (mis != 0) {   // the same experts 4-byte aligned: the bf16 outputs must be bitwise the misaligned run's
        uint8_t* da;
        CK(hipMalloc(&da, hb.size() + 256));
        CK(hipMemcpy(da, hb.data(), hb.size(), hipMemcpyHostToDevice));
        std::vector<F::Batch> ba = batches;
        for (auto& b : ba) for (int e = b.e0; e < b.e1; ++e) b.blob[e - b.e0] = da + (size_t) e * BLOB;
        for (auto& b : ba) F::experts_native_w16(b, g, E, n, scr, xa, dsrc, ha1, dm1, s, 0, true);
        std::vector<uint8_t> ha((size_t) F::act_bytes(n, NF)), hm((size_t) F::act_bytes(n, NF));
        std::vector<float> oa((size_t) n * NE);
        CK(hipMemcpy(oa.data(), dm1, oa.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(ha.data(), ha1, ha.size(), hipMemcpyDeviceToHost));
        size_t nd = 0; int64_t first = -1;
        for (size_t i = 0; i < oa.size(); ++i) if (std::memcmp(&oa[i], &o1[i], 4)) { ++nd; if (first < 0) first = (int64_t) i; }
        std::printf("  aligned vs misaligned blobs: down outputs differing %zu of %zu (first row %lld col %lld)\n", nd, oa.size(),
                    (long long) (first < 0 ? -1 : first / NE), (long long) (first < 0 ? -1 : first % NE));
        // H: gate/up alone (run the misaligned gate/up again into ha0 and compare codes)
        for (auto& b : batches) F::experts_native_w16(b, g, E, n, scr, xa, dsrc, ha0, dm0, s, 1, true);
        CK(hipMemcpy(hm.data(), ha0, hm.size(), hipMemcpyDeviceToHost));
        size_t nh = 0; int64_t fh = -1;
        for (int64_t r = 0; r < n; ++r) for (int u = 0; u < NF / 64; ++u) for (int j = 0; j < 64; ++j) {
            const size_t o = (size_t) r * (NF / 64) * 80 + (size_t) u * 80 + j;
            if (ha[o] != hm[o]) { ++nh; if (fh < 0) fh = r * NF + 64 * u + j; }
        }
        std::printf("  aligned vs misaligned blobs: H codes differing %zu (first row %lld feature %lld)\n", nh,
                    (long long) (fh < 0 ? -1 : fh / NF), (long long) (fh < 0 ? -1 : fh % NF));
        CK(hipFree(da));
    }
    std::vector<int32_t> tab(4 * E + 2), src((size_t) n);
    CK(hipMemcpy(tab.data(), scr, tab.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(src.data(), dsrc, n * 4, hipMemcpyDeviceToHost));
    const int32_t* cnt = tab.data();
    const int32_t* off = tab.data() + E;
    int ex = 0;
    while (ex < E && cnt[ex] == 0) ++ex;
    const int nr = std::min(16, cnt[ex]);
    std::vector<uint32_t> grid(512);
    F::w16_host_iq3s_grid(grid.data());
    std::vector<uint8_t> xah(F::act_bytes(T, NE)), h0(F::act_bytes(n, NF)), h1(F::act_bytes(n, NF));
    CK(hipMemcpy(xah.data(), xa, xah.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(h0.data(), ha0, h0.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(h1.data(), ha1, h1.size(), hipMemcpyDeviceToHost));
    const uint8_t* bl = hb.data() + (size_t) ex * BLOB;
    std::vector<double> wg((size_t) NF * NE), wu((size_t) NF * NE), wd((size_t) NE * NF);
    for (int f = 0; f < NF; ++f) { deq_iq3s(bl + (size_t) f * GU_ROW, NE, grid.data(), &wg[(size_t) f * NE]); deq_iq3s(bl + UP_OFF + (size_t) f * GU_ROW, NE, grid.data(), &wu[(size_t) f * NE]); }
    for (int c = 0; c < NE; ++c) deq_iq4nl(bl + DOWN_OFF + (size_t) c * D_ROW, NF, &wd[(size_t) c * NF]);
    std::vector<double> href, hq0, hq1, dr0, dr1, dg0, dg1;
    for (int r = 0; r < nr; ++r) {
        const int p = off[ex] + r;
        std::vector<double> x(NE), h(NF), a0(NF), a1(NF);
        deq_act(xah.data() + (size_t) src[p] * (NE / 64) * 80, NE, x.data());
        for (int f = 0; f < NF; ++f) {
            double gt = 0, up = 0;
            for (int k = 0; k < NE; ++k) { gt += wg[(size_t) f * NE + k] * x[k]; up += wu[(size_t) f * NE + k] * x[k]; }
            h[f] = gt / (1 + std::exp(-gt)) * up;
        }
        deq_act(h0.data() + (size_t) p * (NF / 64) * 80, NF, a0.data());
        deq_act(h1.data() + (size_t) p * (NF / 64) * 80, NF, a1.data());
        href.insert(href.end(), h.begin(), h.end()); hq0.insert(hq0.end(), a0.begin(), a0.end()); hq1.insert(hq1.end(), a1.begin(), a1.end());
        for (int c = 0; c < NE; ++c) {
            double s0 = 0, s1 = 0;
            for (int k = 0; k < NF; ++k) { s0 += wd[(size_t) c * NF + k] * a0[k]; s1 += wd[(size_t) c * NF + k] * a1[k]; }
            dr0.push_back(s0); dr1.push_back(s1);
            dg0.push_back(o0[(size_t) p * NE + c]); dg1.push_back(o1[(size_t) p * NE + c]);
        }
    }
    std::printf("  FP64 reference, expert %d, %d rows: H (int8) vs exact: int8 path %.3e, bf16 path %.3e; down vs its own H: "
                "int8 %.3e, bf16 %.3e\n", ex, nr, rel(hq0, href), rel(hq1, href), rel(dg0, dr0), rel(dg1, dr1));
    CK(hipFree(db)); CK(hipFree(dids)); CK(hipFree(dslot)); CK(hipFree(dsrc)); CK(hipFree(scr)); CK(hipFree(dx)); CK(hipFree(xa));
    CK(hipFree(ha0)); CK(hipFree(ha1)); CK(hipFree(dm0)); CK(hipFree(dm1));
    return 0;
}

// grouped experts: MMQ's grouped launch (strata::prefill::mmq::Context::run, bounds + ids) on random top-k routing
//   talus-pf-bench moe TYPE N K E TOPK T [skew]  (N = w_rows per expert, K = w_cols)
// relative L2 of b against the reference a, and the rows (of n values) with rel > 0.1
static void w16_cmp(const char* what, const std::vector<float>& a, const std::vector<float>& b, size_t n) {
    double num = 0, den = 0;
    size_t bad = 0;
    for (size_t r = 0; r * n < a.size(); ++r) {
        double nn = 0, dd = 0;
        for (size_t i = r * n; i < (r + 1) * n; ++i) { nn += (a[i] - b[i]) * (double) (a[i] - b[i]); dd += (double) a[i] * a[i]; }
        num += nn; den += dd;
        bad += nn > 0.01 * dd;
    }
    std::printf("  %s: rel L2 vs int8 %.3e, rows with rel > 0.1: %zu of %zu\n", what, std::sqrt(num / (den + 1e-30)), bad, a.size() / n);
}

// IQ1_M experts on plain q8_1 rows: the int8 WMMA kernel (tp_moe k_moe1m) against the dp4a grouped kernel
// (strata::kernels::mimo_prefill_moe_mmq_q8, MiMo's IQ1_M gate / up path), bitwise.
//   talus-pf-bench moe1m N K E TOPK T [skew]   (MiMo: 2048 4096 256 8 2048)
int bench_moe1m(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "moe1m N K E TOPK T [skew]\n"); return 2; }
    const int N = std::atoi(argv[0]), K = std::atoi(argv[1]), E = std::atoi(argv[2]), topk = std::atoi(argv[3]), T = std::atoi(argv[4]);
    const double skew = argc > 5 ? std::atof(argv[5]) : 0.0;
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(77);
    std::vector<double> wgt(E);
    for (int e = 0; e < E; ++e) wgt[e] = 1.0 / std::pow(1.0 + e, skew);
    std::discrete_distribution<int> pick(wgt.begin(), wgt.end());
    std::vector<int> cnt(E, 0);
    for (int t = 0; t < T; ++t) {
        std::vector<int> sel;
        while ((int) sel.size() < topk) { const int e = pick(rng); if (std::find(sel.begin(), sel.end(), e) == sel.end()) sel.push_back(e); }
        for (int e : sel) ++cnt[e];
    }
    std::vector<int32_t> hb(E + 1, 0);
    for (int e = 0; e < E; ++e) hb[e + 1] = hb[e] + cnt[e];
    const int R = hb[E], mx = *std::max_element(cnt.begin(), cnt.end());
    const int bb = 56;   // block_iq1_m
    const size_t eb = (size_t) N * (K / 256) * bb;
    std::vector<uint8_t> hw(eb * E + 4096);
    for (auto& b : hw) b = (uint8_t) (rng() & 0xff);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    for (size_t b = 0; b < hw.size() / bb; ++b) {   // a sane fp16 superblock scale in the scales' top nibbles
        const __half d = __float2half((rng() & 1 ? 1.f : -1.f) * ud(rng));
        uint16_t u;
        std::memcpy(&u, &d, 2);
        uint16_t* sc = (uint16_t*) &hw[b * bb + 48];
        for (int j = 0; j < 4; ++j) sc[j] = (uint16_t) ((sc[j] & 0x0fff) | (((u >> (4 * j)) & 0xf) << 12));
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) R * K);
    for (auto& v : hx) v = nd(rng);
    // a few rows of extreme range (the fragile cases: tiny / huge d)
    for (int r = 0; r < std::min(R, 4); ++r)
        for (int c = 0; c < K; ++c) hx[(size_t) r * K + c] *= (r & 1) ? 1e-4f : 3e3f;
    void *w, *xq;
    float *x, *d0, *d1;
    int32_t *bounds, *ids;
    CK(hipMalloc(&w, hw.size()));
    CK(hipMalloc(&x, hx.size() * 4));
    CK(hipMalloc(&xq, (size_t) R * (K / 32) * 36));
    CK(hipMalloc(&d0, (size_t) R * N * 4));
    CK(hipMalloc(&d1, (size_t) R * N * 4));
    CK(hipMalloc(&bounds, (E + 1) * 4));
    CK(hipMalloc(&ids, (size_t) R * 4));
    CK(hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(bounds, hb.data(), (E + 1) * 4, hipMemcpyHostToDevice));
    {   // ids: a permutation (the scattered store path)
        std::vector<int32_t> hi(R);
        for (int i = 0; i < R; ++i) hi[i] = i;
        std::shuffle(hi.begin(), hi.end(), rng);
        CK(hipMemcpy(ids, hi.data(), (size_t) R * 4, hipMemcpyHostToDevice));
    }
    // the dp4a kernel and the q8_1 quantizer from libtalus_kernels.so (TP_BENCH_TK_LIB overrides the path)
    const char* lp = std::getenv("TP_BENCH_TK_LIB") ? std::getenv("TP_BENCH_TK_LIB") : "libtalus_kernels.so";
    void* lib = dlopen(lp, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::printf("moe1m: dlopen %s: %s\n", lp, dlerror()); return 1; }
    using QFn = void (*)(const float*, void*, int64_t, int64_t, void*);
    using MFn = void (*)(int, const void*, int64_t, int64_t, size_t, const void*, const int32_t*, const int32_t*, float*, int64_t,
                         int64_t, int32_t, int64_t, void*, const int32_t*);
    const QFn qf = (QFn) dlsym(lib, "_ZN6strata7kernels18mimo_quantize_q8_1EPKfPvllS3_");
    const MFn mf = (MFn) dlsym(lib, "_ZN6strata7kernels23mimo_prefill_moe_mmq_q8EiPKvllmS2_PKiS4_PfllilPvS4_");
    if (!qf || !mf) { std::printf("moe1m: symbols not found in %s\n", lp); return 1; }
    qf(x, xq, R, K, s);
    auto ref = [&](int64_t maxr) { mf(29, w, K, N, eb, xq, ids, bounds, d0, R, N, E, maxr, s, nullptr); };
    talus::prefill::MoeProduct mp;
    mp.w = w; mp.type = 29; mp.expert_bytes = eb; mp.n = N; mp.k = K; mp.n_experts = E; mp.xq = xq; mp.bounds = bounds;
    mp.ids = ids; mp.total_rows = R; mp.max_rows = T; mp.dst = d1; mp.ld_dst = N;
    if (!talus::prefill::moe_iq1m_q8_1_supported(29, N, K)) { std::printf("moe1m: shape not covered\n"); return 1; }
    CK(hipMemset(d0, 0xff, (size_t) R * N * 4));
    CK(hipMemset(d1, 0x7f, (size_t) R * N * 4));
    ref(T);
    if (!talus::prefill::moe_gemm_iq1m_q8_1(mp, s)) { std::printf("moe1m: launch refused (TP_MOE_IQ1M_X=0?)\n"); return 1; }
    CK(hipStreamSynchronize(s));
    std::vector<float> a((size_t) R * N), b((size_t) R * N);
    CK(hipMemcpy(a.data(), d0, a.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), d1, b.size() * 4, hipMemcpyDeviceToHost));
    size_t ndiff = 0, first = (size_t) -1;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], 4) != 0) { ++ndiff; if (first == (size_t) -1) first = i; }
        num += (a[i] - b[i]) * (double) (a[i] - b[i]);
        den += (double) a[i] * a[i];
    }
    const double ops = 2.0 * N * K * R;
    const float t0 = time_ms(s, 10, [&] { ref(T); }), t1 = time_ms(s, 10, [&] { talus::prefill::moe_gemm_iq1m_q8_1(mp, s); });
    std::printf("moe1m N %d K %d E %d top %d T %d skew %.2f: rows %d (mean %.1f max %d)  dp4a %.3f ms (%.1f TOPS)  wmma %.3f ms "
                "(%.1f TOPS, x%.2f)  %zu differ (rel L2 %.2e) %s\n",
                N, K, E, topk, T, skew, R, (double) R / E, mx, t0, ops / t0 / 1e9, t1, ops / t1 / 1e9, t0 / t1, ndiff,
                std::sqrt(num / (den + 1e-30)), ndiff ? "DIFF" : "bitwise");
    if (ndiff) std::printf("  first diff at row %zu col %zu: dp4a %.9g wmma %.9g\n", first / N, first % N, a[first], b[first]);
    for (int tk : {64, 128}) {   // k_moe1m rows per block: bitwise + time each
        talus::prefill::moe_iq1m_set_tok(tk);
        CK(hipMemset(d1, 0x7f, (size_t) R * N * 4));
        talus::prefill::moe_gemm_iq1m_q8_1(mp, s);
        CK(hipStreamSynchronize(s));
        CK(hipMemcpy(b.data(), d1, b.size() * 4, hipMemcpyDeviceToHost));
        size_t nd = 0;
        for (size_t i = 0; i < a.size(); ++i) nd += std::memcmp(&a[i], &b[i], 4) != 0;
        const float tt = time_ms(s, 10, [&] { talus::prefill::moe_gemm_iq1m_q8_1(mp, s); });
        std::printf("  tok %d: %.3f ms (%.1f TOPS) %zu differ\n", tk, tt, ops / tt / 1e9, nd);
        ndiff += nd;
    }
    talus::prefill::moe_iq1m_set_tok(0);
    size_t ngu = 0;
    {   // fused gate / up + SwiGLU (k_moe1m<GU>) against dp4a gate, dp4a up and the SwiGLU expression
        void* wu;
        CK(hipMalloc(&wu, hw.size()));
        for (size_t i = 0; i < hw.size(); ++i) hw[i] = (uint8_t) (hw[i] * 7 + 13);   // another expert set, same scales trick
        for (size_t blk = 0; blk < hw.size() / bb; ++blk) {
            const __half d = __float2half((rng() & 1 ? 1.f : -1.f) * ud(rng));
            uint16_t u;
            std::memcpy(&u, &d, 2);
            uint16_t* sc = (uint16_t*) &hw[blk * bb + 48];
            for (int j = 0; j < 4; ++j) sc[j] = (uint16_t) ((sc[j] & 0x0fff) | (((u >> (4 * j)) & 0xf) << 12));
        }
        CK(hipMemcpy(wu, hw.data(), hw.size(), hipMemcpyHostToDevice));
        float *gu, *hh;
        CK(hipMalloc(&gu, (size_t) R * N * 4));
        CK(hipMalloc(&hh, (size_t) R * N * 4));
        auto sep = [&] {
            mf(29, w, K, N, eb, xq, ids, bounds, d0, R, N, E, T, s, nullptr);
            mf(29, wu, K, N, eb, xq, ids, bounds, gu, R, N, E, T, s, nullptr);
            const int64_t nn = (int64_t) R * N;
            k_bench_swiglu<<<(unsigned) ((nn + 255) / 256), 256, 0, s>>>(d0, gu, hh, nn, 0.f);
        };
        talus::prefill::MoeProduct pg = mp;
        pg.w_up = wu;
        sep();
        CK(hipMemset(d1, 0x7f, (size_t) R * N * 4));
        if (talus::prefill::moe_gemm_iq1m_q8_1(pg, s)) {
            CK(hipStreamSynchronize(s));
            CK(hipMemcpy(a.data(), hh, a.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(b.data(), d1, b.size() * 4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < a.size(); ++i) ngu += std::memcmp(&a[i], &b[i], 4) != 0;
            const float ts = time_ms(s, 5, sep), tf = time_ms(s, 5, [&] { talus::prefill::moe_gemm_iq1m_q8_1(pg, s); });
            const float tg = time_ms(s, 5, [&] { talus::prefill::moe_gemm_iq1m_q8_1(mp, s); });
            std::printf("moe1m gu: dp4a gate+up+swiglu %.3f ms, wmma separate x2 %.3f ms, wmma fused %.3f ms (%.1f TOPS)  %zu differ %s\n",
                        ts, 2 * tg, tf, 2 * ops / tf / 1e9, ngu, ngu ? "DIFF" : "bitwise");
        } else {
            std::printf("moe1m gu: not launched (TP_MOE_IQ1M_GU=0?)\n");
        }
        CK(hipFree(wu)); CK(hipFree(gu)); CK(hipFree(hh));
    }
    CK(hipFree(w)); CK(hipFree(x)); CK(hipFree(xq)); CK(hipFree(d0)); CK(hipFree(d1)); CK(hipFree(bounds)); CK(hipFree(ids));
    return ndiff || ngu ? 3 : 0;
}

int bench_moe(int argc, char** argv) {
    namespace MMQ = strata::prefill::mmq;
    if (argc < 6) { std::fprintf(stderr, "moe TYPE N K E TOPK T [skew]\n"); return 2; }
    const int type = std::atoi(argv[0]), N = std::atoi(argv[1]), K = std::atoi(argv[2]), E = std::atoi(argv[3]);
    const int topk = std::atoi(argv[4]), T = std::atoi(argv[5]);
    const double skew = argc > 6 ? std::atof(argv[6]) : 0.0;   // Zipf-like exponent of the expert popularity
    const bool has_mmq = MMQ::supported(type);   // (IQ1_M: no MMQ; checked against tp_i8gemm on the first experts)
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(99);
    std::vector<double> wgt(E);
    for (int e = 0; e < E; ++e) wgt[e] = 1.0 / std::pow(1.0 + e, skew);
    std::discrete_distribution<int> pick(wgt.begin(), wgt.end());
    std::vector<int> cnt(E, 0);
    for (int t = 0; t < T; ++t) {
        std::vector<int> sel;
        while ((int) sel.size() < topk) { const int e = pick(rng); if (std::find(sel.begin(), sel.end(), e) == sel.end()) sel.push_back(e); }
        for (int e : sel) ++cnt[e];
    }
    std::vector<int32_t> hb(E + 1, 0);
    for (int e = 0; e < E; ++e) hb[e + 1] = hb[e] + cnt[e];
    const int R = hb[E];
    const int mx = *std::max_element(cnt.begin(), cnt.end());
    const size_t eb = MMQ::matrix_bytes(type, N, K);
    std::vector<uint8_t> hw(eb * E + 4096, 0);
    for (auto& b : hw) b = (uint8_t) (rng() & 0xff);
    // sane FP16 scales in every block's first two bytes (Q3_K: last two) - the same trick as bench_i8
    auto block_bytes = [](int t) {
        switch (t) { case 23: return 136; case 21: return 110; case 18: return 98; case 16: return 66; case 20: return 18;
                     case 11: return 110; case 17: return 74; case 22: return 82; case 12: return 144; case 13: return 176;
                     case 14: return 210; case 10: return 84; case 8: return 34; case 29: return 56; default: return 0; }
    };
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    const int bb = block_bytes(type);
    if (bb)
        for (size_t b = 0; b < hw.size() / bb; ++b) {
            __half d = __float2half((rng() & 1 ? 1.f : -1.f) * ud(rng)), m = __float2half(ud(rng));
            if (type == 11) std::memcpy(&hw[b * bb + bb - 2], &d, 2);
            else if (type == 14) std::memcpy(&hw[b * bb + bb - 2], &d, 2);
            else if (type == 10) { std::memcpy(&hw[b * bb + bb - 4], &d, 2); std::memcpy(&hw[b * bb + bb - 2], &m, 2); }
            else if (type == 29) {
                uint16_t u; std::memcpy(&u, &d, 2);
                uint16_t* sc = (uint16_t*) &hw[b * bb + 48];
                for (int j = 0; j < 4; ++j) sc[j] = (uint16_t) ((sc[j] & 0x0fff) | (((u >> (4 * j)) & 0xf) << 12));
            }
            else std::memcpy(&hw[b * bb], &d, 2);
            if (type == 12 || type == 13) std::memcpy(&hw[b * bb + 2], &m, 2);
        }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx((size_t) R * K);
    for (auto& v : hx) v = nd(rng);
    void *w, *xq;
    float *x, *d;
    int32_t *bounds, *ids;
    CK(hipMalloc(&w, hw.size()));
    CK(hipMalloc(&x, hx.size() * 4));
    CK(hipMalloc(&xq, MMQ::q8_bytes(R, K)));
    CK(hipMalloc(&d, (size_t) R * N * 4));
    CK(hipMalloc(&bounds, (E + 1) * 4));
    CK(hipMalloc(&ids, (size_t) R * 4));
    CK(hipMemcpy(w, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(bounds, hb.data(), (E + 1) * 4, hipMemcpyHostToDevice));
    MMQ::iota(ids, R, s);
    MMQ::quantize(x, nullptr, xq, has_mmq ? type : 22, K, K, R, s);
    MMQ::Context ctx;
    MMQ::Product p;
    p.w = w; p.type = type; p.w_rows = N; p.w_cols = K; p.expert_bytes = eb; p.n = E; p.xq = xq;
    p.bounds = bounds; p.ids = ids; p.total_rows = R; p.max_rows = mx; p.dst = d; p.ld_dst = N;
    static const bool fast = std::getenv("TP_BENCH_MOE_FAST") != nullptr;   // tp_moe timings only (reference still checked)
    const float t = has_mmq && !fast ? time_ms(s, 10, [&] { ctx.run(p, s); }) : -1.f;
    const double ops = 2.0 * N * K * R;
    // the same rows as one dense matrix (expert 0 for all): the no-grouping reference
    MMQ::Product q = p;
    int32_t* b1;
    CK(hipMalloc(&b1, 8));
    MMQ::set_bounds(b1, R, s);
    q.n = 1; q.bounds = b1; q.max_rows = R;
    const float t1 = has_mmq && !fast ? time_ms(s, 10, [&] { ctx.run(q, s); }) : -1.f;
    // tp_moe: same product, bitwise against MMQ grouped
    float* d2;
    CK(hipMalloc(&d2, (size_t) R * N * 4));
    talus::prefill::MoeProduct mp;
    mp.w = w; mp.type = type; mp.expert_bytes = eb; mp.n = N; mp.k = K; mp.n_experts = E; mp.xq = xq; mp.bounds = bounds;
    mp.ids = ids; mp.total_rows = R; mp.max_rows = mx; mp.dst = d2; mp.ld_dst = N;
    float t2 = -1.f, t3 = -1.f;
    size_t ndiff = 0;
    if (talus::prefill::moe_supported(type, N, K)) {
        if (has_mmq) ctx.run(p, s);   // reference into d
        if (const char* vl = std::getenv("TP_BENCH_MOE_VERS")) {   // interleaved A/B of versions, median of rounds
            std::vector<int> vs;
            for (const char* q = vl; *q;) { vs.push_back(std::atoi(q)); while (*q && *q != ',') ++q; if (*q) ++q; }
            const int rounds = std::getenv("TP_BENCH_ROUNDS") ? std::atoi(std::getenv("TP_BENCH_ROUNDS")) : 7;
            std::vector<std::vector<float>> tv(vs.size());
            for (int r = 0; r < rounds; ++r)
                for (size_t i = 0; i < vs.size(); ++i) {
                    talus::prefill::moe_set_version(vs[i]);
                    tv[i].push_back(time_ms(s, 5, [&] { talus::prefill::moe_gemm(mp, s); }));
                }
            std::printf("ab %s T %d:", argv[0], T);
            for (size_t i = 0; i < vs.size(); ++i) {
                std::sort(tv[i].begin(), tv[i].end());
                const float med = tv[i][tv[i].size() / 2];
                // bitwise check of this version against the reference
                talus::prefill::moe_set_version(vs[i]);
                CK(hipMemset(d2, 0, (size_t) R * N * 4));
                talus::prefill::moe_gemm(mp, s);
                std::vector<float> ry((size_t) R * N), rr((size_t) R * N);
                CK(hipMemcpy(ry.data(), d2, ry.size() * 4, hipMemcpyDeviceToHost));
                size_t nd = 0;
                if (has_mmq) {
                    CK(hipMemcpy(rr.data(), d, rr.size() * 4, hipMemcpyDeviceToHost));
                    for (size_t j = 0; j < rr.size(); ++j) nd += std::memcmp(&rr[j], &ry[j], 4) != 0;
                }
                std::printf("  v%d %.1f%s", vs[i], ops / med / 1e9, nd ? "*" : "");
            }
            std::printf("\n");
            talus::prefill::moe_set_version(-1);
        }
        t2 = time_ms(s, 10, [&] { talus::prefill::moe_gemm(mp, s); });
        {   // as served (the engines pass max_rows = T, an upper bound)
            talus::prefill::MoeProduct ms = mp;
            ms.max_rows = T;
            t3 = time_ms(s, 10, [&] { talus::prefill::moe_gemm(ms, s); });
        }
        std::vector<float> r((size_t) R * N), y((size_t) R * N);
        CK(hipMemcpy(y.data(), d2, y.size() * 4, hipMemcpyDeviceToHost));
        if (has_mmq) {
            CK(hipMemcpy(r.data(), d, r.size() * 4, hipMemcpyDeviceToHost));
            for (size_t i = 0; i < r.size(); ++i) ndiff += std::memcmp(&r[i], &y[i], 4) != 0;
        } else {   // the first 4 experts' rows against tp_i8gemm on the expanded expert (rows re-quantized alone)
            for (int e = 0; e < std::min(E, 4); ++e) {
                const int r0 = hb[e], re = hb[e + 1] - hb[e];
                if (re == 0) continue;
                talus::prefill::I8Cache cache;
                const void* we = (const char*) w + (size_t) e * eb;
                if (!cache.add(we, type, N, K, s)) { std::printf("moe: expansion failed\n"); ndiff = 1; break; }
                void* xe;
                float* de;
                CK(hipMalloc(&xe, MMQ::q8_bytes(re, K)));
                CK(hipMalloc(&de, (size_t) re * N * 4));
                MMQ::quantize(x + (size_t) r0 * K, nullptr, xe, 22, K, K, re, s);
                talus::prefill::i8gemm(*cache.find(we), xe, re, de, N, s);
                std::vector<float> z((size_t) re * N);
                CK(hipMemcpy(z.data(), de, z.size() * 4, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < z.size(); ++i) ndiff += std::memcmp(&z[i], &y[(size_t) r0 * N + i], 4) != 0;
                CK(hipFree(xe)); CK(hipFree(de));
            }
        }
    }
    if (std::getenv("TP_BENCH_W16") && t2 > 0) {   // the fp16 WMMA kernels on the same product
        float* d3;
        CK(hipMalloc(&d3, (size_t) R * N * 4));
        talus::prefill::MoeProduct mw = mp;
        mw.dst = d3;
        mw.max_rows = T;   // as served
        if (talus::prefill::moe_gemm_w16(mw, s, true)) {
            const float tw = time_ms(s, 10, [&] { talus::prefill::moe_gemm_w16(mw, s, true); });
            talus::prefill::moe_gemm(mp, s);
            talus::prefill::moe_gemm_w16(mw, s, true);
            std::vector<float> a((size_t) R * N), c((size_t) R * N);
            CK(hipMemcpy(a.data(), d2, a.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(c.data(), d3, c.size() * 4, hipMemcpyDeviceToHost));
            std::printf("w16 %d T %d: int8 tp_moe %.3f ms (%.1f TOPS)  fp16 %.3f ms (%.1f TOPS, x%.2f)\n", type, T, t3, ops / t3 / 1e9, tw,
                        ops / tw / 1e9, t3 / tw);
            w16_cmp("w16", a, c, N);
        } else {
            std::printf("w16 %d: not covered\n", type);
        }
        CK(hipFree(d3));
    }
    if (std::getenv("TP_BENCH_BF16") && type == 20 && K % 128 == 0) {
        // Prototype: W4A16-style IQ4_NL experts (bf16 activations) against tp_moe's output (not bitwise)
        std::vector<__bf16> hb((size_t) R * K);
        for (size_t i = 0; i < hb.size(); ++i) hb[i] = (__bf16) hx[i];
        __bf16* xb;
        float* db;
        CK(hipMalloc(&xb, hb.size() * 2));
        CK(hipMemcpy(xb, hb.data(), hb.size() * 2, hipMemcpyHostToDevice));
        CK(hipMalloc(&db, (size_t) R * N * 4));
        const dim3 g((unsigned) ((mx + 127) / 128), (unsigned) (N / 128), (unsigned) E);
        auto run = [&] { k_bench_moe_bf16_iq4nl<<<g, dim3(32, 8), 0, s>>>((const uint8_t*) w, eb, K, xb, bounds, ids, db, N); };
        const float tb = time_ms(s, 10, run);
        talus::prefill::moe_gemm(mp, s);
        std::vector<float> a((size_t) R * N), c((size_t) R * N);
        CK(hipMemcpy(a.data(), d2, a.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(c.data(), db, c.size() * 4, hipMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - c[i]) * (double) (a[i] - c[i]); den += (double) a[i] * a[i]; }
        std::printf("bf16 %d T %d: W4A16 prototype %.3f ms (%.1f TOPS), rel L2 vs tp_moe int8 %.2e\n", type, T, tb, ops / tb / 1e9,
                    std::sqrt(num / (den + 1e-30)));
        CK(hipFree(xb)); CK(hipFree(db));
    }
    if (std::getenv("TP_BENCH_GU") && talus::prefill::moe_gu_supported(type, N, K)) {
        // fused gate/up + SwiGLU (gate = w, up = a second random expert set) against two tp_moe launches + SwiGLU
        void* wu;
        CK(hipMalloc(&wu, hw.size()));
        for (auto& b : hw) b = (uint8_t) (rng() & 0xff);
        if (bb)
            for (size_t b = 0; b < hw.size() / bb; ++b) {
                __half dd = __float2half((rng() & 1 ? 1.f : -1.f) * ud(rng));
                if (type == 29) {
                    uint16_t u; std::memcpy(&u, &dd, 2);
                    uint16_t* sc = (uint16_t*) &hw[b * bb + 48];
                    for (int j = 0; j < 4; ++j) sc[j] = (uint16_t) ((sc[j] & 0x0fff) | (((u >> (4 * j)) & 0xf) << 12));
                } else std::memcpy(&hw[b * bb], &dd, 2);
            }
        CK(hipMemcpy(wu, hw.data(), hw.size(), hipMemcpyHostToDevice));
        const float lim = std::getenv("TP_BENCH_GU_LIMIT") ? std::atof(std::getenv("TP_BENCH_GU_LIMIT")) : 0.f;
        float *g, *u, *h, *h2;
        CK(hipMalloc(&g, (size_t) R * N * 4)); CK(hipMalloc(&u, (size_t) R * N * 4));
        CK(hipMalloc(&h, (size_t) R * N * 4)); CK(hipMalloc(&h2, (size_t) R * N * 4));
        talus::prefill::MoeProduct pg = mp, pu = mp, pf = mp;
        pg.dst = g; pu.w = wu; pu.dst = u;
        pf.w_up = wu; pf.dst = h2; pf.swiglu_limit = lim;
        auto sep = [&] {
            talus::prefill::moe_gemm(pg, s);
            talus::prefill::moe_gemm(pu, s);
            const int64_t n = (int64_t) R * N;
            k_bench_swiglu<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(g, u, h, n, lim);
        };
        const float ts = time_ms(s, 5, sep), tf = time_ms(s, 5, [&] { talus::prefill::moe_gemm(pf, s); });
        sep();
        talus::prefill::moe_gemm(pf, s);
        std::vector<float> a((size_t) R * N), c((size_t) R * N);
        CK(hipMemcpy(a.data(), h, a.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(c.data(), h2, c.size() * 4, hipMemcpyDeviceToHost));
        size_t nd2 = 0;
        for (size_t i = 0; i < a.size(); ++i) nd2 += std::memcmp(&a[i], &c[i], 4) != 0;
        std::printf("gu %d T %d: gate+up+swiglu separate %.3f ms, fused %.3f ms (x%.2f, %.1f TOPS) %zu differ %s\n", type, T, ts, tf,
                    ts / tf, 2 * ops / tf / 1e9, nd2, nd2 ? "DIFF" : "bitwise");
        if (std::getenv("TP_BENCH_W16")) {   // fused gate/up on the fp16 kernels against the int8 fused
            talus::prefill::MoeProduct pw = pf;
            pw.dst = h;
            pw.max_rows = T;
            if (talus::prefill::moe_gemm_w16(pw, s, true)) {
                const float tw = time_ms(s, 10, [&] { talus::prefill::moe_gemm_w16(pw, s, true); });
                talus::prefill::moe_gemm_w16(pw, s, true);
                CK(hipMemcpy(a.data(), h, a.size() * 4, hipMemcpyDeviceToHost));
                std::printf("w16 gu %d T %d: int8 fused %.3f ms  fp16 %.3f ms (%.1f TOPS, x%.2f)\n", type, T, tf, tw, 2 * ops / tw / 1e9, tf / tw);
                w16_cmp("w16 gu", c, a, N);
            } else {
                std::printf("w16 gu %d: not covered\n", type);
            }
        }
        CK(hipFree(wu)); CK(hipFree(g)); CK(hipFree(u)); CK(hipFree(h)); CK(hipFree(h2));
    }
    std::printf("moe type %2d N %5d K %5d E %3d top %2d T %5d: rows %d (mean %.1f max %d)  MMQ grouped %7.3f ms (%5.1f TOPS)  "
                "dense-same-rows %7.3f ms (%5.1f TOPS)  tp_moe %7.3f ms (%5.1f TOPS, x%.2f) %zu differ %s  served(max=T) %7.3f ms (%5.1f TOPS)\n",
                type, N, K, E, topk, T, R,
                (double) R / E, mx, t, ops / t / 1e9, t1, ops / t1 / 1e9, t2, t2 > 0 ? ops / t2 / 1e9 : 0.0, t2 > 0 ? t / t2 : 0.0,
                ndiff, t2 < 0 ? "n/a" : ndiff ? "DIFF" : "bitwise", t3, t3 > 0 ? ops / t3 / 1e9 : 0.0);
    CK(hipFree(d2));
    CK(hipFree(w)); CK(hipFree(x)); CK(hipFree(xq)); CK(hipFree(d)); CK(hipFree(bounds)); CK(hipFree(ids)); CK(hipFree(b1));
    return 0;
}

int bench_w4(int argc, char** argv) {
    namespace MMQ = strata::prefill::mmq;
    const int T = argc > 0 ? std::atoi(argv[0]) : 512;
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {10240, 5120}, {6144, 5120}, {5120, 6144}, {12288, 5120}, {1024, 5120}};
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    MMQ::Context ctx;
    int fails = 0;
    double t_ref_all = 0, t_new_all = 0;
    for (int si = 0; si < 7; ++si) {
        const int N = shapes[si][0], K = shapes[si][1], G = K / 256;
        const bool check = si == 0 || si == 1 || si == 6;   // the host model is slow: check three shapes
        std::vector<int8_t> sg(K);
        for (auto& v : sg) v = rng() & 1 ? 1 : -1;
        std::vector<float> hw((size_t) N * K), hx((size_t) T * K);
        for (auto& v : hw) v = nd(rng) * 0.02f;
        for (auto& v : hx) v = nd(rng);
        for (int t = 0; t < T; ++t) hx[(size_t) t * K + (t * 37) % K] *= 30.f;   // an outlier channel per token
        std::vector<int8_t> wq8;
        std::vector<float> wsc;
        host_quant_rows(hw.data(), N, K, sg.data(), wq8, wsc);
        // the sidecar layout
        std::vector<uint8_t> wq((size_t) N * K / 2);
        std::vector<float> wsl((size_t) (N / 16) * G * 16);
        const int order[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
        for (int r = 0; r < N; ++r)
            for (int g = 0; g < G; ++g) {
                uint8_t* dst = &wq[(((size_t) (r / 16) * G + g) * 16 + r % 16) * 128];
                for (int j = 0; j < 128; ++j) {
                    const int lo = wq8[(size_t) r * K + g * 256 + 2 * j] & 0xF, hi = wq8[(size_t) r * K + g * 256 + 2 * j + 1] & 0xF;
                    dst[j] = (uint8_t) (lo | (hi << 4));
                }
            }
        for (int r16 = 0; r16 < N / 16; ++r16)
            for (int g = 0; g < G; ++g)
                for (int j = 0; j < 16; ++j) wsl[((size_t) r16 * G + g) * 16 + j] = wsc[(size_t) (r16 * 16 + order[j]) * G + g];
        int8_t* dsg;
        uint8_t *dwq, *dxq;
        float *dws, *dx, *dxs, *dy;
        CK(hipMalloc(&dsg, K));
        CK(hipMalloc(&dwq, wq.size()));
        CK(hipMalloc(&dws, wsl.size() * 4));
        CK(hipMalloc(&dx, hx.size() * 4));
        CK(hipMalloc(&dxq, (size_t) T * K / 2));
        CK(hipMalloc(&dxs, (size_t) T * G * 4));
        CK(hipMalloc(&dy, (size_t) T * N * 4));
        CK(hipMemcpy(dsg, sg.data(), K, hipMemcpyHostToDevice));
        CK(hipMemcpy(dwq, wq.data(), wq.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dws, wsl.data(), wsl.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
        talus::prefill::W4Mat m;
        m.q = dwq; m.s = dws; m.n = N; m.k = K;
        const float t_pack = time_ms(s, 5, [&] { talus::prefill::w4_pack(dx, T, K, K, dsg, dxq, dxs, s); });
        const float t_new = time_ms(s, 5, [&] { talus::prefill::w4_gemm(m, dxq, dxs, T, dy, N, s); });
        // MMQ (IQ4_XS, random blocks) at the same shape for the time
        float t_ref = 0.f;
        {
            const size_t wb = MMQ::matrix_bytes(23, N, K);
            std::vector<uint8_t> hb(wb + 4096, 0);
            for (size_t i = 0; i < wb; ++i) hb[i] = (uint8_t) (rng() & 0xff);
            for (size_t b = 0; b < wb / 136; ++b) { __half d = __float2half(ud(rng)); std::memcpy(&hb[b * 136], &d, 2); }
            void *w, *xq; float* d; int32_t *bounds, *ids;
            CK(hipMalloc(&w, hb.size())); CK(hipMalloc(&xq, MMQ::q8_bytes(T, K))); CK(hipMalloc(&d, (size_t) T * N * 4));
            CK(hipMalloc(&bounds, 8)); CK(hipMalloc(&ids, (size_t) T * 4));
            CK(hipMemcpy(w, hb.data(), hb.size(), hipMemcpyHostToDevice));
            const int32_t b2[2] = {0, T};
            CK(hipMemcpy(bounds, b2, 8, hipMemcpyHostToDevice));
            MMQ::iota(ids, T, s);
            MMQ::quantize(dx, nullptr, xq, 23, K, K, T, s);
            MMQ::Product p;
            p.w = w; p.type = 23; p.w_rows = N; p.w_cols = K; p.expert_bytes = wb; p.n = 1; p.xq = xq;
            p.bounds = bounds; p.ids = ids; p.total_rows = T; p.max_rows = T; p.dst = d; p.ld_dst = N;
            t_ref = time_ms(s, 5, [&] { ctx.run(p, s); });
            CK(hipFree(w)); CK(hipFree(xq)); CK(hipFree(d)); CK(hipFree(bounds)); CK(hipFree(ids));
        }
        t_ref_all += t_ref;
        t_new_all += t_new;
        const double ops = 2.0 * T * N * K;
        std::printf("w4 N %5d K %5d T %d: MMQ(IQ4_XS) %7.3f ms (%5.1f TOPS)  w4_gemm %7.3f ms (%5.1f TOPS, x%.2f)  pack %.3f ms\n",
                    N, K, T, t_ref, ops / t_ref / 1e9, t_new, ops / t_new / 1e9, t_ref / t_new, t_pack);
        if (check) {
            std::vector<uint8_t> gxq((size_t) T * K / 2);
            std::vector<float> gxs((size_t) T * G), y((size_t) T * N);
            CK(hipMemcpy(gxq.data(), dxq, gxq.size(), hipMemcpyDeviceToHost));
            CK(hipMemcpy(gxs.data(), dxs, gxs.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
            std::vector<int8_t> hq8;
            std::vector<float> hsc;
            host_quant_rows(hx.data(), T, K, sg.data(), hq8, hsc);
            size_t qmis = 0;
            for (size_t i = 0; i < (size_t) T * K; ++i) {
                const int8_t gq = (int8_t) (((gxq[i / 2] >> (4 * (i & 1))) & 0xF) << 4) >> 4;
                qmis += gq != hq8[i];
            }
            double smax = 0;
            for (size_t i = 0; i < gxs.size(); ++i) smax = std::max(smax, std::fabs((double) gxs[i] - hsc[i]) / std::max(1e-30, (double) hsc[i]));
            // the GEMM against the host model on the GPU's own packed activations; and W4A4 against fp32 W x
            double e_sim = 0, n_sim = 0, e_true = 0, n_true = 0;
            for (int t = 0; t < T; t += 7)
                for (int n = 0; n < N; n += 3) {
                    double acc = 0, ref = 0;
                    for (int g = 0; g < G; ++g) {
                        long dot = 0;
                        for (int e = 0; e < 256; ++e) {
                            const size_t ix = (size_t) t * K + g * 256 + e;
                            const int8_t gq = (int8_t) (((gxq[ix / 2] >> (4 * (ix & 1))) & 0xF) << 4) >> 4;
                            dot += (long) gq * wq8[(size_t) n * K + g * 256 + e];
                        }
                        acc += (double) dot * wsc[(size_t) n * G + g] * gxs[(size_t) t * G + g];
                    }
                    for (int c = 0; c < K; ++c) ref += (double) hw[(size_t) n * K + c] * hx[(size_t) t * K + c];
                    const double yv = y[(size_t) t * N + n];
                    e_sim += (yv - acc) * (yv - acc); n_sim += acc * acc;
                    e_true += (yv - ref) * (yv - ref); n_true += ref * ref;
                }
            const double rs = std::sqrt(e_sim / std::max(1e-30, n_sim)), rt = std::sqrt(e_true / std::max(1e-30, n_true));
            const bool ok = rs < 1e-5 && qmis < (size_t) T * K / 10000 && smax < 1e-5;
            fails += !ok;
            std::printf("w4 N %5d K %5d: pack vs host %zu of %zu int4 differ, scales max rel %.2g; GEMM vs host model rel rms %.2g; "
                        "W4A4 vs fp32 W x rel rms %.4f  %s\n", N, K, qmis, (size_t) T * K, smax, rs, rt, ok ? "ok" : "FAIL");
        }
        CK(hipFree(dsg)); CK(hipFree(dwq)); CK(hipFree(dws)); CK(hipFree(dx)); CK(hipFree(dxq)); CK(hipFree(dxs)); CK(hipFree(dy));
    }
    std::printf("w4: MMQ(IQ4_XS) %.2f ms, w4_gemm %.2f ms over all shapes (x%.2f); %d failures\n", t_ref_all, t_new_all,
                t_ref_all / std::max(1e-9, t_new_all), fails);
    return fails ? 1 : 0;
}

__device__ __forceinline__ float bench_silu(float v) { return v / (1.0f + expf(-v)); }   // = q36_engine's
__global__ void bench_swiglu(const float* g, const float* u, float* h, int64_t n) {   // = q36_engine's k_swiglu
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = bench_silu(g[i]) * u[i];
}

int bench_fuse(int argc, char** argv) {
    namespace MMQ = strata::prefill::mmq;
    const int T = argc > 0 ? std::atoi(argv[0]) : 512, K = 17408;
    hipStream_t s;
    CK(hipStreamCreate(&s));
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hg((size_t) T * K), hu((size_t) T * K);
    for (auto& v : hg) v = nd(rng) * 2.f;
    for (auto& v : hu) v = nd(rng);
    float *g, *u, *h;
    void *qa, *qb;
    const size_t qn = MMQ::q8_bytes(T, K);
    CK(hipMalloc(&g, hg.size() * 4)); CK(hipMalloc(&u, hu.size() * 4)); CK(hipMalloc(&h, hg.size() * 4));
    CK(hipMalloc(&qa, qn)); CK(hipMalloc(&qb, qn));
    CK(hipMemcpy(g, hg.data(), hg.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(u, hu.data(), hu.size() * 4, hipMemcpyHostToDevice));
    int fails = 0;
    for (int type : {23, 12, 14, 10, 11}) {   // IQ4_XS (D4), Q4_K (DS4), Q6_K, Q2_K, Q3_K
        CK(hipMemset(qa, 0, qn)); CK(hipMemset(qb, 0, qn));
        const int64_t n = (int64_t) T * K;
        const float t_ref = time_ms(s, 5, [&] {
            bench_swiglu<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(g, u, h, n);
            MMQ::quantize(h, nullptr, qa, type, K, K, T, s);
        });
        const float t_new = time_ms(s, 5, [&] { talus::prefill::swiglu_q8_mmq(g, u, qb, type, K, T, s); });
        std::vector<uint8_t> a(qn), b(qn);
        CK(hipMemcpy(a.data(), qa, qn, hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), qb, qn, hipMemcpyDeviceToHost));
        size_t nd2 = 0;
        for (size_t i = 0; i < qn; ++i) nd2 += a[i] != b[i];
        fails += nd2 != 0;
        std::printf("fuse type %2d (layout %d) T %d K %d: swiglu + quantize %.3f ms, fused %.3f ms (x%.2f); %zu of %zu bytes differ  %s\n",
                    type, MMQ::q8_layout(type), T, K, t_ref, t_new, t_ref / t_new, nd2, qn, nd2 ? "DIFF" : "bitwise");
    }
    std::printf("fuse: %d failures\n", fails);
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: talus-pf-bench fa [T] [depth ...]\n");
        return 2;
    }
    const std::string m = argv[1];
    if (m == "fa") return bench_fa(argc - 2, argv + 2);
    if (m == "i8") return bench_i8(argc - 2, argv + 2);
    if (m == "i8m") return bench_i8m(argc - 2, argv + 2);
    if (m == "i8one") return bench_i8one(argc - 2, argv + 2);
    if (m == "w4") return bench_w4(argc - 2, argv + 2);
    if (m == "fuse") return bench_fuse(argc - 2, argv + 2);
    if (m == "moe") return bench_moe(argc - 2, argv + 2);
    if (m == "moe1m") return bench_moe1m(argc - 2, argv + 2);
    if (m == "w16") return bench_w16(argc - 2, argv + 2);
    if (m == "hcd") return bench_hcd(argc - 2, argv + 2);
    std::fprintf(stderr, "talus-pf-bench: unknown mode %s\n", m.c_str());
    return 2;
}
