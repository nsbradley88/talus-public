// tests/hip/gdn_scan_bench.cpp (rb-fn) - the Flash-Next prompt GDN recurrence variants outside the engine.
//   talus-gdn-bench [REF] [VARIANTS] [LENS] [REPS]
//     REF       reference variant (default 5 = gdn_rec_quad2c_kernel, production on gfx1151 before rb-fn)
//     VARIANTS  comma list of gdn_recurrence_variant ids (default 6,7,8,9): 6 dpp2c, 7 dpp2f<4>, 8 dpp2f<8>,
//               9 chunked (WY, fp32; NOT bitwise by construction - reported as error stats)
//     LENS      comma list of chunk lengths (default 1,3,4,5,7,8,9,17,100,2048,2051,16384)
//     REPS      timing repetitions at T >= 2048 (default 5)
// Per variant: y16 (what the out projection reads) and the final state compared BITWISE with REF from the same nonzero
// start state; for a non-bitwise variant also max |d| / max rel and the same against an FP64 CPU recurrence (T <= 2051,
// first 4 heads) next to REF's own FP64 error, so "rounding level" is a measured statement.  Synthetic inputs in the
// model's ranges (unit q/k rows, log-decay gates in [-2, 0), beta in [0, 1)).  Exit 1 on any bitwise failure of a
// variant < 9.
#include "strata/prefill/kernels.hpp"

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
constexpr int S = 128, HK = 16, HV = 48, C = 10240;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
std::vector<long> parse(const char* s, std::vector<long> def) {
    if (s == nullptr) return def;
    std::vector<long> v;
    std::string t(s);
    size_t p = 0;
    while (p <= t.size()) {
        size_t q = t.find(',', p);
        if (q == std::string::npos) q = t.size();
        if (q > p) v.push_back(std::atol(t.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}
float h2f(uint16_t b) {
    __half x;
    std::memcpy(&x, &b, 2);
    return __half2float(x);
}
// FP64 recurrence for heads [0, nh): the pre-norm output oc (o / sqrt(S)) and the final state, from the same inputs
void cpu_ref(const std::vector<float>& h, const std::vector<float>& gate, const std::vector<float>& beta,
             const std::vector<float>& st0, int64_t T, int nh, std::vector<double>& oc, std::vector<double>& st) {
    oc.assign((size_t) T * nh * S, 0.0);
    st.assign((size_t) nh * S * S, 0.0);   // [head][row][col]
    for (int hd = 0; hd < nh; ++hd)
        for (int r = 0; r < S; ++r)
            for (int c = 0; c < S; ++c) st[((size_t) hd * S + r) * S + c] = st0[((size_t) r * HV + hd) * S + c];
    std::vector<double> kv(S);
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h.data() + t * C;
        for (int hd = 0; hd < nh; ++hd) {
            const int qh = hd % HK;
            const float* q = ht + qh * S;
            const float* k = ht + HK * S + qh * S;
            const float* v = ht + 2 * HK * S + hd * S;
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
}  // namespace

int main(int argc, char** argv) {
    const int ref_v = argc > 1 ? std::atoi(argv[1]) : 5;
    const auto variants = parse(argc > 2 ? argv[2] : nullptr, {6, 7, 8, 9});
    const auto lens = parse(argc > 3 ? argv[3] : nullptr, {1, 3, 4, 5, 7, 8, 9, 17, 100, 2048, 2051, 16384});
    const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    int failures = 0;
    for (const long T : lens) {
        std::vector<float> h((size_t) T * C), gate((size_t) T * HV), beta((size_t) T * HV), z((size_t) T * HV * S),
            gamma(S), state((size_t) S * HV * S);
        for (int64_t t = 0; t < T; ++t) {
            float* ht = h.data() + t * C;
            for (int i = 0; i < C; ++i) ht[i] = nd(rng);
            for (int hh = 0; hh < 2 * HK; ++hh) {   // q and k heads: unit rows, as after gdn_conv's L2 norm
                double ss = 0;
                for (int d = 0; d < S; ++d) ss += (double) ht[hh * S + d] * ht[hh * S + d];
                for (int d = 0; d < S; ++d) ht[hh * S + d] = (float) (ht[hh * S + d] / std::sqrt(ss + 1e-6));
            }
        }
        for (auto& x : gate) x = -2.0f * ud(rng);
        for (auto& x : beta) x = ud(rng);
        for (auto& x : z) x = nd(rng);
        for (auto& x : gamma) x = 0.5f + ud(rng);
        for (auto& x : state) x = 0.1f * nd(rng);
        const float* dh = dev(h);
        const float* dg = dev(gate);
        const float* db = dev(beta);
        const float* dz = dev(z);
        const float* dgm = dev(gamma);
        float* dst = nullptr;
        float* dy = nullptr;
        uint16_t* dy16 = nullptr;
        ck(cudaMalloc(&dst, state.size() * 4), "malloc");
        ck(cudaMalloc(&dy, (size_t) T * HV * S * 4), "malloc");
        ck(cudaMalloc(&dy16, (size_t) T * HV * S * 2), "malloc");
        auto run = [&](int v, std::vector<float>* y, std::vector<uint16_t>* y16, std::vector<float>* st) {
            ck(cudaMemcpy(dst, state.data(), state.size() * 4, cudaMemcpyHostToDevice), "state");
            ck(cudaMemset(dy, 0, (size_t) T * HV * S * 4), "memset");
            strata::prefill::gdn_recurrence_variant(v, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, nullptr);
            ck(cudaDeviceSynchronize(), "run");
            if (y) *y = host(dy, (size_t) T * HV * S);
            if (y16) *y16 = host(dy16, (size_t) T * HV * S);
            if (st) *st = host(dst, state.size());
        };
        std::vector<float> y_ref, st_ref;
        std::vector<uint16_t> y16_ref;
        run(ref_v, &y_ref, &y16_ref, &st_ref);
        // FP64 reference (pre-norm output of variant 3..9 is in y when STRATA_GDN_NOY is unset? no: y holds the
        // normalized value for WY=true kernels; compare the state, and y16 against REF)
        const bool fp64 = T <= 2051;
        std::vector<double> oc64, st64;
        const int nh = 4;
        if (fp64) cpu_ref(h, gate, beta, state, T, nh, oc64, st64);
        auto st_err64 = [&](const std::vector<float>& st) {
            double mx = 0;
            for (int hd = 0; hd < nh; ++hd)
                for (int r = 0; r < S; ++r)
                    for (int c = 0; c < S; ++c)
                        mx = std::max(mx, std::fabs((double) st[((size_t) r * HV + hd) * S + c] -
                                                    st64[((size_t) hd * S + r) * S + c]));
            return mx;
        };
        if (fp64) std::printf("T %ld ref %d: state max|d| vs FP64 %.3e\n", T, ref_v, st_err64(st_ref));
        for (const long vl : variants) {
            const int v = (int) vl;
            std::vector<float> y, st;
            std::vector<uint16_t> y16;
            run(v, &y, &y16, &st);
            size_t d16 = 0, dsn = 0;
            double m16 = 0, mst = 0, r16 = 0;
            for (size_t i = 0; i < y16.size(); ++i) {
                if (y16[i] != y16_ref[i]) {
                    ++d16;
                    const double a = h2f(y16[i]), b = h2f(y16_ref[i]), d = std::fabs(a - b);
                    m16 = std::max(m16, d);
                    r16 = std::max(r16, d / std::max(1e-3, std::fabs(b)));
                }
            }
            for (size_t i = 0; i < st.size(); ++i)
                if (std::memcmp(&st[i], &st_ref[i], 4) != 0) {
                    ++dsn;
                    mst = std::max(mst, (double) std::fabs(st[i] - st_ref[i]));
                }
            const bool ok = d16 == 0 && dsn == 0;
            if (v < 9) failures += !ok;
            std::printf("%s T %ld variant %d vs %d: y16 %zu (max|d| %.3e, rel %.3e), state %zu (max|d| %.3e) of %zu / %zu differ",
                        ok ? "PASS" : (v < 9 ? "FAIL" : "DIFF"), T, v, ref_v, d16, m16, r16, dsn, mst, y16.size(),
                        st.size());
            if (fp64 && !ok) std::printf("; state max|d| vs FP64 %.3e", st_err64(st));
            std::printf("\n");
        }
        if (T >= 2048 && reps > 0) {
            std::vector<long> tv = {ref_v};
            tv.insert(tv.end(), variants.begin(), variants.end());
            for (const long vl : tv) {
                const int v = (int) vl;
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                strata::prefill::gdn_recurrence_variant(v, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, nullptr);
                cudaEventRecord(a);
                for (int r = 0; r < reps; ++r)
                    strata::prefill::gdn_recurrence_variant(v, dst, dh, dg, db, dz, dgm, 1e-6f, dy, dy16, T, nullptr);
                cudaEventRecord(b);
                ck(cudaEventSynchronize(b), "time");
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                std::printf("  time T %ld variant %d: %.3f ms per call (%.3f us per token)\n", T, v, ms / reps,
                            ms / reps * 1000 / (double) T);
                cudaEventDestroy(a);
                cudaEventDestroy(b);
            }
        }
        cudaFree(dst); cudaFree(dy); cudaFree(dy16);
        cudaFree((void*) dh); cudaFree((void*) dg); cudaFree((void*) db); cudaFree((void*) dz); cudaFree((void*) dgm);
    }
    std::printf("FAILURES: %d\n", failures);
    return failures ? 1 : 0;
}
