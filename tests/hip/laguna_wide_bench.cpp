// Laguna-S-2.1 wide-verify kernels at the engine's shapes, outside the engine.
//   experts: the routed gate+up / down items of a verify window of n tokens x 10 experts through native_mmvq_batch,
//            per-id (expert_major 0, today's verify path) against grouped (expert_major 3): bitwise check + us per call
//   dense:   native_mmvq (the wave32 path with STRATA_MMVQ_W32=1) at the dense shapes, n columns in 16-column calls
// Weights are random bytes (2 layer pools, cycled; the 32 MB on-die cache holds little), activations Q8_1 of random
// floats, expert ids n x 10 distinct per token (uniform, or SKEW=1: a few popular experts as in real routing).
//   talus-laguna-wide-bench [experts|dense|all] [iters] [n,n,..]
#include "strata/kernels/native_mmvq.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

static size_t row_bytes(int type, int n_in) {
    switch (type) {
    case 12: return (size_t) n_in / 256 * 144;   // Q4_K
    case 13: return (size_t) n_in / 256 * 176;   // Q5_K
    case 14: return (size_t) n_in / 256 * 210;   // Q6_K
    case 8:  return (size_t) n_in / 32 * 34;     // Q8_0
    }
    std::fprintf(stderr, "type %d\n", type); std::exit(1);
}

static void* rand_dev(size_t bytes, std::mt19937& rng) {
    std::vector<uint8_t> h(bytes);
    for (auto& b : h) b = (uint8_t) (rng() & 0x1f);   // small bytes: the f16 block scales stay finite
    void* d = nullptr;
    CK(hipMalloc(&d, bytes));
    CK(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice));
    return d;
}

// Q8_1 of random floats (n_in values per column, ncols columns)
static void* rand_q8(int n_in, int ncols, std::mt19937& rng, hipStream_t st) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> h((size_t) n_in * ncols);
    for (auto& v : h) v = nd(rng);
    float* xf = nullptr;
    CK(hipMalloc(&xf, h.size() * 4));
    CK(hipMemcpy(xf, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    void* q = nullptr;
    CK(hipMalloc(&q, k::native_q8_1_bytes(n_in, 1) * ncols));
    const size_t cb = k::native_q8_1_bytes(n_in, 1);
    for (int c0 = 0; c0 < ncols; c0 += 16)
        k::native_quantize_q8_1(xf + (size_t) c0 * n_in, (uint8_t*) q + c0 * cb, n_in, std::min(16, ncols - c0), st);
    CK(hipStreamSynchronize(st));
    CK(hipFree(xf));
    return q;
}

struct Mat { int type, n_in, n_out, experts; void* w; size_t expert_bytes; };
static Mat make(int type, int n_in, int n_out, int experts, std::mt19937& rng) {
    Mat m{type, n_in, n_out, experts, nullptr, row_bytes(type, n_in) * n_out};
    m.w = rand_dev(m.expert_bytes * experts, rng);
    return m;
}

static std::vector<int> parse_ns(const char* s) {
    std::vector<int> v;
    if (!s) return {1, 2, 4, 5, 8, 12, 16, 24, 32, 48, 64};
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "all";
    const int iters = argc > 2 ? std::atoi(argv[2]) : 50;
    const std::vector<int> ns = parse_ns(argc > 3 ? argv[3] : nullptr);
    const bool skew = std::getenv("SKEW") && std::atoi(std::getenv("SKEW")) != 0;
    std::mt19937 rng(1);
    constexpr int kPools = 2, kEmbd = 3072, kFf = 1024, kTopK = 10, kExp = 256, kMaxN = 64;
    hipStream_t s; CK(hipStreamCreate(&s));
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    auto timed = [&](auto&& fn) {
        for (int i = 0; i < 3; ++i) fn(i);
        CK(hipEventRecord(e0, s));
        for (int i = 0; i < iters; ++i) fn(i);
        CK(hipEventRecord(e1, s)); CK(hipEventSynchronize(e1));
        float ms = 0; CK(hipEventElapsedTime(&ms, e0, e1));
        return 1000.0 * ms / iters;
    };

    if (what == "experts" || what == "all") {
        struct EL { Mat g, u, d; };
        std::vector<EL> L;
        for (int p = 0; p < kPools; ++p)
            L.push_back({make(12, kEmbd, kFf, kExp, rng), make(12, kEmbd, kFf, kExp, rng), make(13, kFf, kEmbd, kExp, rng)});
        void* xq = rand_q8(kEmbd, kMaxN, rng, s);                // a token's normed hidden state (x_group 10)
        void* xff = rand_q8(kFf, kMaxN * kTopK, rng, s);         // one swiglu column per id
        const size_t ysz = (size_t) kMaxN * kTopK * kEmbd;
        float *ya = nullptr, *yb = nullptr;
        CK(hipMalloc(&ya, ysz * 4 * 2)); CK(hipMalloc(&yb, ysz * 4 * 2));
        // id sets: kSets windows of kMaxN tokens x 10 distinct experts
        constexpr int kSets = 8;
        std::vector<int32_t> hid((size_t) kSets * kMaxN * kTopK);
        std::vector<double> wexp(kExp, 1.0);
        if (skew) for (int e = 0; e < kExp; ++e) wexp[e] = 1.0 / std::pow(1.0 + (double) ((e * 37) % kExp), 0.6);
        std::discrete_distribution<int> pick_e(wexp.begin(), wexp.end());
        for (int t = 0; t < kSets * kMaxN; ++t) {
            std::vector<int> pick;
            while ((int) pick.size() < kTopK) {
                const int e = pick_e(rng);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            for (int j = 0; j < kTopK; ++j) hid[(size_t) t * kTopK + j] = pick[j];
        }
        int32_t* ids = nullptr;
        CK(hipMalloc(&ids, hid.size() * 4));
        CK(hipMemcpy(ids, hid.data(), hid.size() * 4, hipMemcpyHostToDevice));
        auto items = [&](int i, int n, int mode, float* y, bool down) {
            const EL& l = L[i % kPools];
            const int32_t* id = ids + (size_t) (i % kSets) * kMaxN * kTopK;
            std::vector<k::MmvqItem> v;
            auto one = [&](const Mat& m, const void* x, float* yo, int xg) {
                k::MmvqItem it;
                it.type = m.type; it.w = m.w; it.expert_bytes = m.expert_bytes; it.ids = id; it.n_ids = n * kTopK;
                it.x = x; it.x_per_id = 1; it.x_group = xg; it.y = yo; it.n_in = m.n_in; it.n_out = m.n_out;
                static const int nexp = [] { const char* e = std::getenv("BENCH_NEXP"); return e ? std::atoi(e) : kExp; }();
                it.expert_major = mode == 3 ? (3 | (nexp << 8)) : (n == 1 ? 1 : 0);   // the engine's choices
                v.push_back(it);
            };
            if (down) one(l.d, xff, y, 1);
            else { one(l.g, xq, y, kTopK); one(l.u, xq, y + (size_t) kMaxN * kTopK * kFf, kTopK); }
            return v;
        };
        std::printf("experts (%s ids): n  distinct  gate+up per-id / grouped us   down per-id / grouped us   bitwise diffs\n",
                    skew ? "skewed" : "uniform");
        for (int n : ns) {
            if (n < 1 || n > kMaxN) continue;
            int distinct = 0;
            {
                std::vector<char> seen(kExp, 0);
                for (int j = 0; j < n * kTopK; ++j) distinct += !seen[hid[j]]++;
            }
            size_t diffs = 0;
            for (int down = 0; down < 2; ++down) {
                CK(hipMemset(ya, 0xff, ysz * 8)); CK(hipMemset(yb, 0xfe, ysz * 8));
                auto va = items(0, n, 0, ya, down), vb = items(0, n, 3, yb, down);
                k::native_mmvq_batch(va.data(), (int) va.size(), s);
                k::native_mmvq_batch(vb.data(), (int) vb.size(), s);
                CK(hipStreamSynchronize(s));
                const size_t per = down ? (size_t) n * kTopK * kEmbd : (size_t) n * kTopK * kFf;
                std::vector<uint32_t> ha(per), hb(per);
                for (int part = 0; part < (down ? 1 : 2); ++part) {
                    const size_t off = part ? (size_t) kMaxN * kTopK * kFf : 0;
                    CK(hipMemcpy(ha.data(), ya + off, per * 4, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(hb.data(), yb + off, per * 4, hipMemcpyDeviceToHost));
                    for (size_t q = 0; q < per; ++q) diffs += ha[q] != hb[q];
                }
            }
            double us[2][2];
            for (int mode = 0; mode < 2; ++mode)
                for (int down = 0; down < 2; ++down)
                    us[mode][down] = timed([&](int i) {
                        auto v = items(i, n, mode ? 3 : 0, mode ? yb : ya, down);
                        k::native_mmvq_batch(v.data(), (int) v.size(), s);
                    });
            std::printf("experts n %2d  %3d  %9.1f / %9.1f   %9.1f / %9.1f   %zu\n", n, distinct, us[0][0], us[1][0], us[0][1],
                        us[1][1], diffs);
        }
    }

    if (what == "dense" || what == "all") {
        struct DM { const char* name; Mat m; };
        std::vector<std::vector<DM>> P;
        for (int p = 0; p < kPools; ++p)
            P.push_back({{"q swa Q5_K 3072x9216", make(13, kEmbd, 9216, 1, rng)},
                         {"k Q6_K 3072x1024", make(14, kEmbd, 1024, 1, rng)},
                         {"gate Q8_0 3072x72", make(8, kEmbd, 72, 1, rng)},
                         {"o swa Q5_K 9216x3072", make(13, 9216, kEmbd, 1, rng)},
                         {"sh gate Q6_K 3072x1024", make(14, kEmbd, kFf, 1, rng)},
                         {"sh down Q6_K 1024x3072", make(14, kFf, kEmbd, 1, rng)},
                         // the DFlash drafter's prompt injection (32-column chunks): fc and per-layer k / v
                         {"df fc Q8_0 18432x3072", make(8, 18432, kEmbd, 1, rng)},
                         {"df k Q8_0 3072x1024", make(8, kEmbd, 1024, 1, rng)}});
        Mat head = make(8, kEmbd, 100352, 1, rng);   // the real head is Q8_0
        void* x = rand_q8(18432, kMaxN, rng, s);   // wide enough for the DFlash fc input
        float *y = nullptr, *y2 = nullptr, *yr = nullptr;
        CK(hipMalloc(&yr, (size_t) kMaxN * 100352 * 4));
        CK(hipMalloc(&y, (size_t) kMaxN * 100352 * 4));
        CK(hipMalloc(&y2, (size_t) kMaxN * 100352 * 4));
        // needs STRATA_MMVQ_W32=1 and STRATA_MMVQ_LDS=N (the LDS kernel from N columns) in the environment
        std::printf("dense: us per n-column product, wave32 in 16-column calls / LDS kernel in one call [bitwise diffs]"
                    " (STRATA_MMVQ_W32=%s STRATA_MMVQ_LDS=%s)\n",
                    std::getenv("STRATA_MMVQ_W32") ? std::getenv("STRATA_MMVQ_W32") : "unset",
                    std::getenv("STRATA_MMVQ_LDS") ? std::getenv("STRATA_MMVQ_LDS") : "unset");
        const int nm = (int) P[0].size();
        for (int j = 0; j <= nm; ++j) {
            std::printf("%-24s", j < nm ? P[0][j].name : "head Q8_0 3072x100352");
            for (int n : ns) {
                auto w32 = [&](int i, float* yo) {
                    const Mat& m = j < nm ? P[i % kPools][j].m : head;
                    const size_t xs = k::native_q8_1_bytes(m.n_in, 1);
                    for (int c0 = 0; c0 < n; c0 += 16)
                        k::native_mmvq(m.type, m.w, (const uint8_t*) x + c0 * xs, yo + (size_t) c0 * m.n_out, m.n_in, m.n_out,
                                       std::min(16, n - c0), s);
                };
                auto lds = [&](int i, float* yo) {
                    const Mat& m = j < nm ? P[i % kPools][j].m : head;
                    return k::native_mmvq_lds(m.type, m.w, x, yo, m.n_in, m.n_out, n, s);
                };
                // the exact matrix-core kernel (STRATA_MMVQ_XT); reference = n single-column calls (decode's kernel)
                auto xt = [&](int i, float* yo) {
                    const Mat& m = j < nm ? P[i % kPools][j].m : head;
                    return k::native_mmvq_xt(m.type, m.w, x, yo, m.n_in, m.n_out, n, s);
                };
                const Mat& m0 = j < nm ? P[0][j].m : head;
                const size_t xs0 = k::native_q8_1_bytes(m0.n_in, 1);
                for (int c = 0; c < n; ++c)
                    k::native_mmvq(m0.type, m0.w, (const uint8_t*) x + c * xs0, yr + (size_t) c * m0.n_out, m0.n_in, m0.n_out, 1, s);
                auto diff = [&](float* yo) {
                    std::vector<uint32_t> ha((size_t) n * m0.n_out), hb(ha.size());
                    CK(hipMemcpy(ha.data(), yr, ha.size() * 4, hipMemcpyDeviceToHost));
                    CK(hipMemcpy(hb.data(), yo, hb.size() * 4, hipMemcpyDeviceToHost));
                    size_t d = 0;
                    for (size_t q = 0; q < ha.size(); ++q) d += ha[q] != hb[q];
                    return d;
                };
                CK(hipMemset(y, 0xff, (size_t) n * m0.n_out * 4)); CK(hipMemset(y2, 0xfe, (size_t) n * m0.n_out * 4));
                w32(0, y);
                const bool has = lds(0, y2) || xt(0, y2);
                CK(hipStreamSynchronize(s));
                const size_t d32 = diff(y), diffs = has ? diff(y2) : 0;
                const double ua = timed([&](int i) { w32(i, y); });
                const double ub = has ? timed([&](int i) { if (!lds(i, y2)) xt(i, y2); }) : 0.0;
                if (has) std::printf("  n%d %.0f/%.0f%s", n, ua, ub, diffs ? " [DIFF]" : "");
                else std::printf("  n%d %.0f/-", n, ua);
                if (diffs) std::printf("(%zu)", diffs);
                if (d32) std::printf("[w32 DIFF %zu]", d32);
            }
            std::printf("\n");
        }
    }
    return 0;
}
