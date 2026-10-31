// tests/hip/gdn_repeat.cpp - the prompt GDN recurrences on long runs of one repeated token and on
// pathological gates (the failure hipfire fixed in its chunked WY prefill: F16 WY-inverse overflow -> NaN when keys
// repeat; idea from hipfire / warpfront, no code from it), and the GDN prompt kernel's residency (occupancy) cliff.
//   talus-gdn-repeat check [T] [CHUNK]   27B k_gdn_seq_warp<8> (launch_gdn_seq_warp) and Flash-Next gdn_rec_dpp2c
//                                         (+ out norm) over T tokens (default 4096) in CHUNK-token calls (default 512,
//                                         the engines' chunk), state carried; per scenario: non-finite counts, max |o|,
//                                         max |state|, and the 27B kernel vs an FP64 CPU recurrence (2 heads)
//   talus-gdn-repeat occ [T] [NCOLS]     k_gdn_seq_warp<NCOL> us per call vs the value-head count (8..96): a step
//                                         where the grid stops fitting resident (all waves of a sequential scan must
//                                         be resident at once, or the rest runs as a second pass)
// Exit 1 when a production scenario (not the negative control) has a non-finite value.
#include "strata/prefill/kernels.hpp"
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
constexpr int S = 128, HK = 16, HV = 48, C = 2 * HK * S + HV * S;   // the 27B / Flash-Next GDN shape (C 10240)
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 16), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
float h2f(uint16_t b) { __half x; std::memcpy(&x, &b, 2); return __half2float(x); }
void unit(float* p, float scale) {
    double ss = 0;
    for (int d = 0; d < S; ++d) ss += (double) p[d] * p[d];
    for (int d = 0; d < S; ++d) p[d] = (float) (p[d] / std::sqrt(ss + 1e-6)) * scale;
}
struct Scn { const char* name; bool control; };
const Scn kScn[] = {
    {"repeat (one token, gate -0.1, beta 0.6)", false},
    {"repeat, no decay (gate 0) beta 1", false},
    {"repeat, no decay, beta 1, |v| x1000", false},
    {"repeat, gate alternating -1e4 / -1e-7, beta 1", false},
    {"repeat, q = k = 0 (L2 of a zero row)", false},
    {"two tokens alternating, no decay, beta 1", false},
    {"random tokens (reference)", false},
    {"NEG CONTROL: repeat, |k| = 4 (no L2 norm), no decay, beta 1", true},
};
constexpr int kNScn = sizeof(kScn) / sizeof(kScn[0]);
// inputs for T tokens: h [T][C], gate [T][HV] (log decay), beta [T][HV]
void make(int sc, int64_t T, std::vector<float>& h, std::vector<float>& gate, std::vector<float>& beta) {
    std::mt19937 rng(77 + sc);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    h.assign((size_t) T * C, 0.f);
    gate.assign((size_t) T * HV, 0.f);
    beta.assign((size_t) T * HV, 0.f);
    auto token = [&](float* ht, float kscale, float vscale, bool zero_qk) {
        for (int i = 0; i < C; ++i) ht[i] = nd(rng);
        for (int hh = 0; hh < 2 * HK; ++hh) unit(ht + hh * S, hh >= HK ? kscale : 1.f);
        if (zero_qk) std::fill(ht, ht + 2 * HK * S, 0.f);
        for (int i = 2 * HK * S; i < C; ++i) ht[i] *= vscale;
    };
    std::vector<float> a(C), b(C);
    const float ks = sc == 7 ? 4.f : 1.f, vs = sc == 2 ? 1000.f : 1.f;
    token(a.data(), ks, vs, sc == 4);
    token(b.data(), ks, vs, sc == 4);
    for (int64_t t = 0; t < T; ++t) {
        float* ht = h.data() + t * C;
        if (sc == 6) token(ht, 1.f, 1.f, false);
        else std::memcpy(ht, (sc == 5 && (t & 1)) ? b.data() : a.data(), C * 4);
        for (int j = 0; j < HV; ++j) {
            float g = -0.1f, be = 0.6f;
            if (sc == 1 || sc == 2 || sc == 5 || sc == 7) { g = 0.f; be = 1.f; }
            if (sc == 3) { g = (t & 1) ? -1e-7f : -1e4f; be = 1.f; }
            if (sc == 6) { g = -2.f * ud(rng); be = ud(rng); }
            gate[(size_t) t * HV + j] = g;
            beta[(size_t) t * HV + j] = be;
        }
    }
}
// FP64 recurrence for heads [0, nh): o (= the kernel's o_raw: o / sqrt(S)) and the final state [head][row][col]
void cpu_ref(const std::vector<float>& h, const std::vector<float>& gate, const std::vector<float>& beta, int64_t T,
             int nh, std::vector<double>& oc, std::vector<double>& st) {
    oc.assign((size_t) T * nh * S, 0.0);
    st.assign((size_t) nh * S * S, 0.0);
    std::vector<double> kv(S);
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h.data() + t * C;
        for (int hd = 0; hd < nh; ++hd) {
            const int qh = hd % HK;
            const float *q = ht + qh * S, *k = ht + HK * S + qh * S, *v = ht + 2 * HK * S + hd * S;
            const double g = std::exp((double) gate[t * HV + hd]), b = beta[t * HV + hd];
            double* s = st.data() + (size_t) hd * S * S;
            for (int c = 0; c < S; ++c) {
                double a = 0;
                for (int r = 0; r < S; ++r) a += s[r * S + c] * k[r];
                kv[c] = a;
            }
            for (int c = 0; c < S; ++c) {
                const double d = (v[c] - g * kv[c]) * b;
                double o = 0;
                for (int r = 0; r < S; ++r) {
                    s[r * S + c] = g * s[r * S + c] + k[r] * d;
                    o += s[r * S + c] * q[r];
                }
                oc[((size_t) t * nh + hd) * S + c] = o / std::sqrt((double) S);
            }
        }
    }
}
struct Stat { size_t nonfin = 0; double mx = 0; };
template <typename F> Stat stat(size_t n, F get) {
    Stat s;
    for (size_t i = 0; i < n; ++i) {
        const double x = get(i);
        if (!std::isfinite(x)) ++s.nonfin;
        else s.mx = std::max(s.mx, std::fabs(x));
    }
    return s;
}
int check(int64_t T, int chunk) {
    int fails = 0;
    float *dst = nullptr, *dy = nullptr, *dor = nullptr;
    uint16_t* dy16 = nullptr;
    ck(cudaMalloc(&dst, (size_t) S * HV * S * 4), "st");
    ck(cudaMalloc(&dy, (size_t) chunk * HV * S * 4), "y");
    ck(cudaMalloc(&dor, (size_t) T * HV * S * 4), "or");
    ck(cudaMalloc(&dy16, (size_t) chunk * HV * S * 2), "y16");
    std::vector<float> z((size_t) chunk * HV * S), gamma(S);
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : z) x = nd(rng);
    for (auto& x : gamma) x = 1.f;
    float* dz = dev(z);
    float* dgm = dev(gamma);
    for (int sc = 0; sc < kNScn; ++sc) {
        std::vector<float> h, gate, beta;
        make(sc, T, h, gate, beta);
        float* dh = dev(h);
        float* dg = dev(gate);
        float* db = dev(beta);
        // 27B: k_gdn_seq_warp<8>, o_raw for all T
        ck(cudaMemset(dst, 0, (size_t) S * HV * S * 4), "st0");
        for (int64_t t0 = 0; t0 < T; t0 += chunk) {
            const int n = (int) std::min<int64_t>(chunk, T - t0);
            talus::q36::launch_gdn_seq_warp(8, nullptr, dst, dh + t0 * C, C, dg + t0 * HV, db + t0 * HV,
                                            dor + t0 * HV * S, HK, HV, n);
        }
        ck(cudaDeviceSynchronize(), "q27 run");
        const auto o27 = host(dor, (size_t) T * HV * S);
        const auto s27 = host(dst, (size_t) S * HV * S);
        const Stat so = stat(o27.size(), [&](size_t i) { return (double) o27[i]; });
        const Stat ss = stat(s27.size(), [&](size_t i) { return (double) s27[i]; });
        // Flash-Next: gdn_rec_dpp2c + out norm (variant 6), y16 / y per chunk
        ck(cudaMemset(dst, 0, (size_t) S * HV * S * 4), "st0");
        Stat fy, fy16;
        for (int64_t t0 = 0; t0 < T; t0 += chunk) {
            const int n = (int) std::min<int64_t>(chunk, T - t0);
            strata::prefill::gdn_recurrence_variant(6, dst, dh + t0 * C, dg + t0 * HV, db + t0 * HV, dz, dgm, 1e-6f, dy,
                                                    dy16, n, nullptr);
            ck(cudaDeviceSynchronize(), "fn run");
            const auto y = host(dy, (size_t) n * HV * S);
            const auto y16 = host(dy16, (size_t) n * HV * S);
            const Stat a = stat(y.size(), [&](size_t i) { return (double) y[i]; });
            const Stat b = stat(y16.size(), [&](size_t i) { return (double) h2f(y16[i]); });
            fy.nonfin += a.nonfin; fy.mx = std::max(fy.mx, a.mx);
            fy16.nonfin += b.nonfin; fy16.mx = std::max(fy16.mx, b.mx);
        }
        const auto sfn = host(dst, (size_t) S * HV * S);
        const Stat sf = stat(sfn.size(), [&](size_t i) { return (double) sfn[i]; });
        // FP64 reference, 2 heads
        std::vector<double> oc, st;
        const int nh = 2;
        cpu_ref(h, gate, beta, T, nh, oc, st);
        double eo = 0, es = 0, mo = 0, ms = 0;
        for (int64_t t = 0; t < T; ++t)
            for (int hd = 0; hd < nh; ++hd)
                for (int c = 0; c < S; ++c) {
                    const double r = oc[((size_t) t * nh + hd) * S + c], g = o27[((size_t) t * HV + hd) * S + c];
                    eo = std::max(eo, std::fabs(g - r));
                    mo = std::max(mo, std::fabs(r));
                }
        for (int hd = 0; hd < nh; ++hd)
            for (int r = 0; r < S; ++r)
                for (int c = 0; c < S; ++c) {
                    const double x = st[((size_t) hd * S + r) * S + c], g = s27[((size_t) r * HV + hd) * S + c];
                    es = std::max(es, std::fabs(g - x));
                    ms = std::max(ms, std::fabs(x));
                }
        const size_t nf = so.nonfin + ss.nonfin + fy.nonfin + fy16.nonfin + sf.nonfin;
        const bool ok = nf == 0;
        if (!ok && !kScn[sc].control) ++fails;
        std::printf("%s [%d] %s, T %lld in %d-token calls\n", kScn[sc].control ? (ok ? "CONTROL-FINITE" : "CONTROL-NONFINITE (expected)")
                                                                 : (ok ? "PASS" : "FAIL"),
                    sc, kScn[sc].name, (long long) T, chunk);
        std::printf("   27B seq_warp<8>: o non-finite %zu (max|o| %.3e), state non-finite %zu (max %.3e); vs FP64 (2 heads): "
                    "o max|d| %.3e (max|o| %.3e), state max|d| %.3e (max %.3e)\n",
                    so.nonfin, so.mx, ss.nonfin, ss.mx, eo, mo, es, ms);
        std::printf("   FN dpp2c+norm:   y non-finite %zu (max %.3e), y16 non-finite %zu (max %.3e), state non-finite %zu "
                    "(max %.3e)\n",
                    fy.nonfin, fy.mx, fy16.nonfin, fy16.mx, sf.nonfin, sf.mx);
        std::fflush(stdout);
        cudaFree(dh); cudaFree(dg); cudaFree(db);
    }
    std::printf("gdn-repeat: %s (%d production scenario failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
int occ(int T, std::vector<int> ncols) {
    const int hvmax = 96, hkmax = hvmax / 3, Cm = 2 * hkmax * S + hvmax * S;
    std::vector<float> h((size_t) T * Cm), gate((size_t) T * hvmax, -0.1f), beta((size_t) T * hvmax, 0.5f);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 0.1f);
    for (auto& x : h) x = nd(rng);
    float* dh = dev(h);
    float* dg = dev(gate);
    float* db = dev(beta);
    float *dst = nullptr, *dor = nullptr;
    ck(cudaMalloc(&dst, (size_t) S * hvmax * S * 4), "st");
    ck(cudaMalloc(&dor, (size_t) T * hvmax * S * 4), "or");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int dev_id = 0, cus = 0;
    cudaGetDevice(&dev_id);
    cudaDeviceGetAttribute(&cus, cudaDevAttrMultiProcessorCount, dev_id);
    std::printf("device CUs %d; k_gdn_seq_warp<NCOL> us per call at T %d (h_k = h_v / 3; 27B is h_v 48)\n", T, T);
    std::printf("%-5s", "h_v");
    for (int n : ncols) std::printf("  NCOL%-2d us  us/head", n);
    std::printf("\n");
    for (int hv = 8; hv <= hvmax; hv += 8) {
        const int hk = std::max(1, hv / 3), Cx = 2 * hk * S + hv * S;
        std::printf("%-5d", hv);
        for (int n : ncols) {
            float best = 1e30f;
            for (int r = 0; r < 3; ++r) {
                talus::q36::launch_gdn_seq_warp(n, nullptr, dst, dh, Cx, dg, db, dor, hk, hv, T);
                ck(cudaEventRecord(e0), "rec");
                for (int i = 0; i < 5; ++i) talus::q36::launch_gdn_seq_warp(n, nullptr, dst, dh, Cx, dg, db, dor, hk, hv, T);
                ck(cudaEventRecord(e1), "rec");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, ms * 1000.f / 5);
            }
            std::printf("  %9.1f %8.2f", best, best / hv);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    return 0;
}
std::vector<int> list(const char* s, std::vector<int> d) {
    if (!s) return d;
    std::vector<int> v;
    for (const char* p = s; *p;) { v.push_back(std::atoi(p)); while (*p && *p != ',') ++p; if (*p) ++p; }
    return v;
}
}   // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "check";
    if (mode == "check") return check(argc > 2 ? std::atoll(argv[2]) : 4096, argc > 3 ? std::atoi(argv[3]) : 512);
    if (mode == "occ") return occ(argc > 2 ? std::atoi(argv[2]) : 512, list(argc > 3 ? argv[3] : nullptr, {4, 8, 16}));
    std::fprintf(stderr, "usage: talus-gdn-repeat check [T] [CHUNK] | occ [T] [NCOLS]\n");
    return 2;
}
