// tests/hip/tp_w16_test.cpp - tests and bench of src/prefill/tp_w16.cu (16-bit weight mat-vec + GEMM).
//
//   tp_w16_test --cpu      CPU only (no GPU): the mat-vec's lane code (gemv_lane, the same function the kernel runs)
//                          over 32 emulated lanes + the butterfly, against an fp64 reference on dequantized weights;
//                          every column bitwise the 1-column result for every (R, U); tails (N, K not multiples);
//                          bf16 rounding (ggml rule); GEMM tile order covers every tile once; a mutation check.
//   tp_w16_test            GPU: the kernels bitwise the CPU emulation (mat-vec), the GEMM within fp32-accumulation
//                          rounding of an fp64 reference on bf16-rounded activations; C ABI checks.
//   tp_w16_test --bench    GPU: mat-vec GB/s per (shape, columns, R, U, W) on the 27B projection shapes (weights
//                          rotated through > 1 GiB so every read comes from DRAM), GEMM TFLOPS at 512 / 2048 rows.
#include "../../src/prefill/tp_w16.cu"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace talus::prefill::w16;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                   \
    do {                                                \
        if (!(c)) {                                     \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                   \
            std::printf("\n");                          \
            ++g_fail;                                   \
        }                                               \
    } while (0)

float bf16_to_f(uint16_t h) { return u2f((uint32_t) h << 16); }
float f16_to_f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }
uint16_t f_to_f16(float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }

struct Prob {
    int wt, N, K, T;
    int64_t ldw, ldx, ldy;
    std::vector<uint16_t> W;
    std::vector<float> X;
};

Prob make(int wt, int N, int K, int T, uint32_t seed, int64_t ldw_pad = 0, int64_t ldx_pad = 0) {
    Prob p;
    p.wt = wt; p.N = N; p.K = K; p.T = T;
    p.ldw = K + ldw_pad; p.ldx = K + ldx_pad; p.ldy = N;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    p.W.assign((size_t) N * p.ldw, 0);
    p.X.assign((size_t) T * p.ldx, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const float v = 0.02f * nd(rng) * (k % 97 == 0 ? 30.0f : 1.0f);   // a few outlier columns
            p.W[(size_t) n * p.ldw + k] = wt == kBF16 ? f32_to_bf16(v) : f_to_f16(v);
        }
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) p.X[(size_t) t * p.ldx + k] = nd(rng) * (k % 251 == 0 ? 50.0f : 1.0f);
    return p;
}

float wval(const Prob& p, int n, int k) {
    const uint16_t h = p.W[(size_t) n * p.ldw + k];
    return p.wt == kBF16 ? bf16_to_f(h) : f16_to_f(h);
}

// fp64 reference and the |w||x| scale of each output (for the tolerance)
void reference(const Prob& p, bool round_x, std::vector<double>& y, std::vector<double>& mag) {
    y.assign((size_t) p.T * p.N, 0.0);
    mag.assign((size_t) p.T * p.N, 0.0);
    for (int t = 0; t < p.T; ++t)
        for (int n = 0; n < p.N; ++n) {
            double s = 0.0, m = 0.0;
            for (int k = 0; k < p.K; ++k) {
                float x = p.X[(size_t) t * p.ldx + k];
                if (round_x) x = p.wt == kBF16 ? bf16_to_f(f32_to_bf16(x)) : f16_to_f(f_to_f16(x));
                const double w = wval(p, n, k);
                s += w * x;
                m += std::fabs(w * x);
            }
            y[(size_t) t * p.N + n] = s;
            mag[(size_t) t * p.N + n] = m;
        }
}

float host_butterfly(float (&v)[32]) {
    for (int s = 16; s >= 1; s >>= 1) {
        float nv[32];
        for (int l = 0; l < 32; ++l) nv[l] = v[l] + v[l ^ s];
        std::memcpy(v, nv, sizeof(nv));
    }
    return v[0];
}

// Y[t][n] for columns [t0, t0 + T) exactly as the kernel computes them (gemv_lane for each lane + the butterfly)
template <int WT, int T, int R, int U>
void emulate_cols(const Prob& p, int t0, std::vector<float>& Y) {
    const int K8 = p.K / 8;
    const float* X = p.X.data() + (size_t) t0 * p.ldx;
    for (int row0 = 0; row0 < p.N; row0 += R) {
        float part[32][R][T];
        for (int lane = 0; lane < 32; ++lane) {
            float acc[R][T];
            gemv_lane<WT, T, R, U>(p.W.data(), p.ldw, p.N, K8, X, p.ldx, row0, lane, acc);
            std::memcpy(part[lane], acc, sizeof(acc));
        }
        for (int r = 0; r < R; ++r)
            for (int t = 0; t < T; ++t) {
                float v[32];
                for (int l = 0; l < 32; ++l) v[l] = part[l][r][t];
                if (row0 + r < p.N) Y[(size_t) (t0 + t) * p.N + row0 + r] = host_butterfly(v);
            }
    }
}

template <int WT, int R, int U>
void emulate(const Prob& p, std::vector<float>& Y) {
    Y.assign((size_t) p.T * p.N, NAN);
    for (int t0 = 0; t0 < p.T; t0 += kMaxCols) {
        const int c = std::min(kMaxCols, p.T - t0);
        switch (c) {
#define E(n) case n: emulate_cols<WT, n, R, U>(p, t0, Y); break;
            E(1) E(2) E(3) E(4) E(5) E(6) E(7) E(8) E(9) E(10) E(11) E(12) E(13) E(14) E(15) E(16)
#undef E
        }
    }
}

template <int WT>
void emulate_cfg(int r, int u, const Prob& p, std::vector<float>& Y) {
    if (r == 1 && u == 1) emulate<WT, 1, 1>(p, Y);
    else if (r == 1 && u == 2) emulate<WT, 1, 2>(p, Y);
    else if (r == 1 && u == 4) emulate<WT, 1, 4>(p, Y);
    else if (r == 2 && u == 1) emulate<WT, 2, 1>(p, Y);
    else if (r == 2 && u == 2) emulate<WT, 2, 2>(p, Y);
    else if (r == 2 && u == 4) emulate<WT, 2, 4>(p, Y);
    else if (r == 4 && u == 1) emulate<WT, 4, 1>(p, Y);
    else if (r == 4 && u == 2) emulate<WT, 4, 2>(p, Y);
    else emulate<WT, 4, 4>(p, Y);
}

void emulate_any(int r, int u, const Prob& p, std::vector<float>& Y) {
    if (p.wt == kBF16) emulate_cfg<kBF16>(r, u, p, Y);
    else emulate_cfg<kF16>(r, u, p, Y);
}

double max_rel(const std::vector<float>& y, const std::vector<double>& ref, const std::vector<double>& mag) {
    double worst = 0.0;
    for (size_t i = 0; i < y.size(); ++i) worst = std::max(worst, std::fabs(y[i] - ref[i]) / std::max(mag[i], 1e-30));
    return worst;
}

bool bitwise_eq(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

// ------------------------------------------------------------------------------------------------ CPU tests

void cpu_round() {
    struct C { uint32_t in; uint16_t out; } cs[] = {
        {0x3f800000u, 0x3f80}, {0x3f808000u, 0x3f80},   // tie -> even (down)
        {0x3f818000u, 0x3f82},                           // tie -> even (up)
        {0x3f808001u, 0x3f81}, {0x3f807fffu, 0x3f80},
        {0x7f7fffffu, 0x7f80},                           // max float rounds to inf (as ggml)
        {0x7f800000u, 0x7f80}, {0xff800000u, 0xff80},    // inf
        {0x7fc00000u, 0x7fc0}, {0x7f800001u, 0x7fc0},    // NaN stays NaN (quiet bit)
        {0x00000001u, 0x0000}, {0x80000000u, 0x8000},
    };
    for (const C& c : cs) {
        const uint16_t got = f32_to_bf16(u2f(c.in));
        CHECK(got == c.out, "f32_to_bf16(0x%08x) = 0x%04x, want 0x%04x", c.in, got, c.out);
    }
}

void cpu_gemv() {
    struct S { int wt, N, K, T; int64_t padw, padx; };
    const S shapes[] = {
        {kBF16, 37, 8 * 32 * 4 * 3 + 8 * 5, 16, 0, 0},   // N odd, K8 not a multiple of 32 * U
        {kBF16, 64, 5120, 5, 0, 0},
        {kBF16, 9, 6144, 16, 24, 12},                    // padded row strides
        {kBF16, 3, 17408, 3, 0, 0},
        {kBF16, 5, 24, 2, 0, 0},                         // K8 = 3 < 32: most lanes idle
        {kF16, 21, 2048, 7, 0, 4},
        {kBF16, 11, 1024, 21, 0, 0},                     // more than 16 columns: two launches (16 + 5)
    };
    for (const S& s : shapes) {
        const Prob p = make(s.wt, s.N, s.K, s.T, 1234u + s.N * 7 + s.K, s.padw, s.padx);
        std::vector<double> ref, mag;
        reference(p, false, ref, mag);
        std::vector<float> base;
        emulate_any(2, 2, p, base);
        const double rel = max_rel(base, ref, mag);
        // fp32 accumulation of K products: error <= ~K * 2^-24 of sum |w x|; observed far below
        CHECK(rel < 2e-6, "type %d N %d K %d T %d: max rel error %.3g", s.wt, s.N, s.K, s.T, rel);
        if (s.wt == kBF16) {
            for (int r : {1, 2, 4})
                for (int u : {1, 2, 4}) {
                    std::vector<float> y;
                    emulate_any(r, u, p, y);
                    CHECK(bitwise_eq(y, base), "N %d K %d T %d: R %d U %d not bitwise R 2 U 2", s.N, s.K, s.T, r, u);
                }
        }
        // each column alone (1-column launch) bitwise the same column of the window
        for (int t = 0; t < s.T; ++t) {
            Prob one = p;
            one.T = 1;
            one.X.assign(p.X.begin() + (size_t) t * p.ldx, p.X.begin() + (size_t) (t + 1) * p.ldx);
            std::vector<float> y1;
            emulate_any(2, 2, one, y1);
            CHECK(std::memcmp(y1.data(), base.data() + (size_t) t * p.N, (size_t) p.N * 4) == 0,
                  "N %d K %d: column %d of %d not bitwise its 1-column result", s.N, s.K, t, s.T);
        }
        std::printf("cpu gemv type %2d N %5d K %5d T %2d (ldw %lld ldx %lld): max rel %.2e, R/U/T bitwise ok\n", s.wt,
                    s.N, s.K, s.T, (long long) p.ldw, (long long) p.ldx, rel);
    }
    // mutation check: the comparison catches a single wrong weight
    {
        Prob p = make(kBF16, 8, 512, 2, 99u);
        std::vector<double> ref, mag;
        reference(p, false, ref, mag);
        p.W[3 * p.ldw + 129] ^= 0x0040;   // flip a mantissa bit of one weight
        std::vector<float> y;
        emulate_any(2, 2, p, y);
        CHECK(max_rel(y, ref, mag) > 1e-5, "mutation not detected (rel %.3g)", max_rel(y, ref, mag));
    }
}

void cpu_tiles() {
    for (int M : {1, 17, 128, 129, 1000, 2048})
        for (int N : {48, 256, 1024, 5120, 10240, 17408, 248320}) {
            const int num_m = (M + g::BM - 1) / g::BM, num_n = (N + g::BN - 1) / g::BN;
            std::vector<int> hit((size_t) num_m * num_n, 0);
            for (int b = 0; b < num_m * num_n; ++b) {
                const int group = b / (g::GM * num_n), first_m = group * g::GM;
                const int gsize = std::min(g::GM, num_m - first_m);
                const int m0 = (first_m + (b % (g::GM * num_n)) % gsize) * g::BM;
                const int n0 = ((b % (g::GM * num_n)) / gsize) * g::BN;
                if (m0 / g::BM < num_m && n0 / g::BN < num_n) ++hit[(size_t) (m0 / g::BM) * num_n + n0 / g::BN];
            }
            int bad = 0;
            for (int h : hit) bad += h != 1;
            CHECK(bad == 0, "GEMM tile order M %d N %d: %d tiles not covered exactly once", M, N, bad);
        }
    std::printf("cpu gemm tile order: every tile once for all tested (M, N)\n");
}

// ------------------------------------------------------------------------------------------------ GPU

#define HC(x)                                                                          \
    do {                                                                               \
        hipError_t e_ = (x);                                                           \
        if (e_ != hipSuccess) {                                                        \
            std::printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(3);                                                              \
        }                                                                              \
    } while (0)

template <typename T>
T* dev_copy(const std::vector<T>& h) {
    T* d = nullptr;
    HC(hipMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)));
    HC(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

void gpu_gemv() {
    struct S { int wt, N, K, T; int64_t padw, padx; };
    const S shapes[] = {
        {kBF16, 5120, 5120, 1, 0, 0},  {kBF16, 6144, 5120, 2, 0, 0},  {kBF16, 1024, 5120, 5, 0, 0},
        {kBF16, 5120, 17408, 8, 0, 0}, {kBF16, 17408, 5120, 16, 0, 0}, {kBF16, 4097, 6144, 13, 0, 0},
        {kBF16, 37, 2088, 16, 24, 12}, {kBF16, 777, 1024, 21, 0, 0},  {kF16, 513, 5120, 7, 0, 4},
        {kF16, 1024, 5120, 1, 0, 0},
    };
    for (const S& s : shapes) {
        const Prob p = make(s.wt, s.N, s.K, s.T, 77u + s.N + s.K * 3, s.padw, s.padx);
        std::vector<float> emu;
        const Tables& tb = tables();
        // the library's table config for these column counts: emulate with its (R, U) (bitwise any (R, U) anyway)
        emulate_any(s.wt == kBF16 ? tb.cfg[std::min(s.T, kMaxCols)].r : 2, s.wt == kBF16 ? tb.cfg[std::min(s.T, kMaxCols)].u : 2, p, emu);
        uint16_t* dW = dev_copy(p.W);
        float* dX = dev_copy(p.X);
        float* dY = nullptr;
        HC(hipMalloc(&dY, (size_t) p.T * p.N * 4));
        HC(hipMemset(dY, 0xff, (size_t) p.T * p.N * 4));
        const int rc = talus_w16_gemv(s.wt, dW, p.ldw, p.N, p.K, dX, p.ldx, dY, p.N, p.T, nullptr);
        HC(hipDeviceSynchronize());
        std::vector<float> y((size_t) p.T * p.N);
        HC(hipMemcpy(y.data(), dY, y.size() * 4, hipMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t i = 0; i < y.size(); ++i) diff += std::memcmp(&y[i], &emu[i], 4) != 0;
        CHECK(rc == 0 && diff == 0, "gpu gemv type %d N %d K %d T %d: rc %d, %zu of %zu outputs differ from the CPU emulation",
              s.wt, s.N, s.K, s.T, rc, diff, y.size());
        // every (R, U, W) config bitwise the table's
        if (s.wt == kBF16 && s.T <= kMaxCols) {
            for (int r : {1, 2, 4})
                for (int u : {1, 2, 4})
                    for (int w : {1, 4, 8}) {
                        GemvCfg c{r, u, w};
                        HC(hipMemset(dY, 0xff, (size_t) p.T * p.N * 4));
                        launch_gemv<kBF16>(p.T, c, dW, p.ldw, p.N, p.K / 8, dX, p.ldx, dY, p.N, nullptr);
                        HC(hipDeviceSynchronize());
                        std::vector<float> y2((size_t) p.T * p.N);
                        HC(hipMemcpy(y2.data(), dY, y2.size() * 4, hipMemcpyDeviceToHost));
                        CHECK(bitwise_eq(y2, emu), "gpu gemv N %d K %d T %d: R %d U %d W %d not bitwise", s.N, s.K,
                              s.T, r, u, w);
                    }
        }
        std::printf("gpu gemv type %2d N %5d K %5d T %2d: %s\n", s.wt, s.N, s.K, s.T,
                    diff == 0 && rc == 0 ? "bitwise the CPU emulation" : "MISMATCH");
        HC(hipFree(dW)); HC(hipFree(dX)); HC(hipFree(dY));
    }
}

void gpu_gemm() {
    struct S { int wt, N, K, T; };
    const S shapes[] = {{kBF16, 1024, 5120, 17}, {kBF16, 5120, 6144, 200}, {kBF16, 777, 1024, 513},
                        {kBF16, 17408, 5120, 64}, {kBF16, 300, 17408, 129}, {kF16, 1024, 2048, 100}};
    const bool wmma = talus_w16_info(1) == 1;
    std::printf("gpu gemm: WMMA %s on this device\n", wmma ? "yes" : "no (mat-vec slices)");
    for (const S& s : shapes) {
        const Prob p = make(s.wt, s.N, s.K, s.T, 5u + s.N + s.T);
        std::vector<double> ref, mag;
        reference(p, wmma, ref, mag);
        uint16_t* dW = dev_copy(p.W);
        float* dX = dev_copy(p.X);
        float* dY = nullptr;
        uint16_t* dS = nullptr;
        const int64_t ldy = p.N + 3;   // an output row stride wider than N
        HC(hipMalloc(&dY, (size_t) p.T * ldy * 4));
        HC(hipMemset(dY, 0, (size_t) p.T * ldy * 4));
        HC(hipMalloc(&dS, (size_t) p.T * p.K * 2));
        const int rc = talus_w16_gemm(s.wt, dW, p.ldw, p.N, p.K, dX, p.ldx, dY, ldy, p.T, dS, (int64_t) p.T * p.K, nullptr);
        HC(hipDeviceSynchronize());
        std::vector<float> yl((size_t) p.T * ldy), y((size_t) p.T * p.N);
        HC(hipMemcpy(yl.data(), dY, yl.size() * 4, hipMemcpyDeviceToHost));
        for (int t = 0; t < p.T; ++t) std::memcpy(&y[(size_t) t * p.N], &yl[(size_t) t * ldy], (size_t) p.N * 4);
        const double rel = max_rel(y, ref, mag);
        CHECK(rc == 0 && rel < 4e-6, "gpu gemm type %d N %d K %d T %d: rc %d max rel %.3g", s.wt, s.N, s.K, s.T, rc, rel);
        bool pad_ok = true;
        for (int t = 0; t < p.T; ++t)
            for (int64_t j = p.N; j < ldy; ++j) pad_ok &= yl[(size_t) t * ldy + j] == 0.0f;
        CHECK(pad_ok, "gpu gemm wrote past N in a row of stride %lld", (long long) ldy);
        std::printf("gpu gemm type %2d N %5d K %5d T %4d: max rel %.2e vs fp64 on %s activations\n", s.wt, s.N, s.K,
                    s.T, rel, wmma ? "16-bit-rounded" : "fp32");
        HC(hipFree(dW)); HC(hipFree(dX)); HC(hipFree(dY)); HC(hipFree(dS));
    }
    // ABI refusals
    CHECK(talus_w16_gemv(8, (void*) 16, 0, 16, 64, (const float*) 16, 0, (float*) 16, 0, 1, nullptr) == 1, "Q8_0 not refused");
    CHECK(talus_w16_gemv(kBF16, (void*) 16, 0, 16, 60, (const float*) 16, 0, (float*) 16, 0, 1, nullptr) == 1, "K %% 8 not refused");
    CHECK(talus_w16_supported(kBF16, 248320, 5120) == 1 && talus_w16_supported(2, 16, 64) == 0, "supported()");
}

// ------------------------------------------------------------------------------------------------ bench

void bench() {
    // 27B projection shapes (N x K) and the bytes of one weight pass
    struct Shp { const char* name; int N, K; };
    const Shp shp[] = {{"ffn_up/gate 17408x5120", 17408, 5120}, {"ffn_down 5120x17408", 5120, 17408},
                       {"attn_qkv 10240x5120", 10240, 5120},    {"attn_gate 6144x5120", 6144, 5120},
                       {"ssm_out 5120x6144", 5120, 6144},       {"attn_k 1024x5120", 1024, 5120},
                       {"head 248320x5120", 248320, 5120}};
    const double bus = 256.0;   // GB/s: LPDDR5X-8000 x 256 bit
    float* dX = nullptr;
    float* dY = nullptr;
    HC(hipMalloc(&dX, (size_t) 16 * 17408 * 4));
    HC(hipMemset(dX, 0, (size_t) 16 * 17408 * 4));
    HC(hipMalloc(&dY, (size_t) 16 * 248320 * 4));
    hipEvent_t e0, e1;
    HC(hipEventCreate(&e0)); HC(hipEventCreate(&e1));
    const char* only = std::getenv("W16_BENCH_CFGS");   // "R:U:W,..." to try (default: a small grid)
    std::vector<GemvCfg> cfgs;
    if (only) {
        const char* q = only;
        int r, u, w, n;
        while (std::sscanf(q, "%d:%d:%d%n", &r, &u, &w, &n) == 3) { cfgs.push_back({r, u, w}); q += n; while (*q == ',') ++q; }
    } else {
        for (int r : {1, 2, 4}) for (int u : {1, 2, 4}) for (int w : {2, 4, 8}) cfgs.push_back({r, u, w});
    }
    const int colsets[] = {1, 2, 4, 5, 8, 12, 16};
    for (const Shp& s : shp) {
        const size_t bytes = (size_t) s.N * s.K * 2;
        const int copies = (int) std::max<size_t>(1, (size_t(1) << 30) / bytes + 1);   // rotate > 1 GiB: DRAM reads
        std::vector<uint16_t*> ws(copies);
        for (auto& w : ws) { HC(hipMalloc(&w, bytes)); HC(hipMemset(w, 0x3c, bytes)); }
        for (int T : colsets) {
            double best = 0; GemvCfg bc{};
            std::string row;
            for (const GemvCfg& c : cfgs) {
                for (int i = 0; i < copies; ++i) launch_gemv<kBF16>(T, c, ws[i], s.K, s.N, s.K / 8, dX, s.K, dY, s.N, nullptr);
                const int iters = std::max(copies * 3, 12);
                HC(hipEventRecord(e0));
                for (int i = 0; i < iters; ++i)
                    launch_gemv<kBF16>(T, c, ws[i % copies], s.K, s.N, s.K / 8, dX, s.K, dY, s.N, nullptr);
                HC(hipEventRecord(e1));
                HC(hipEventSynchronize(e1));
                float ms = 0;
                HC(hipEventElapsedTime(&ms, e0, e1));
                const double gbs = (double) bytes * iters / (ms * 1e-3) / 1e9;
                if (gbs > best) { best = gbs; bc = c; }
                char b[48]; std::snprintf(b, sizeof b, " %d:%d:%d=%.0f", c.r, c.u, c.w, gbs); row += b;
            }
            const GemvCfg& tc = tables().cfg[T];
            std::printf("bench %-24s T %2d best %6.1f GB/s (%5.1f%% of %.0f) at R %d U %d W %d; table %d:%d:%d |%s\n",
                        s.name, T, best, 100.0 * best / bus, bus, bc.r, bc.u, bc.w, tc.r, tc.u, tc.w, row.c_str());
        }
        for (auto& w : ws) HC(hipFree(w));
    }
    // GEMM
    if (talus_w16_info(1) == 1) {
        for (int T : {512, 2048}) {
            for (const Shp& s : shp) {
                if (s.N > 20000) continue;
                uint16_t* w = nullptr; float* x = nullptr; float* y = nullptr; uint16_t* sc = nullptr;
                HC(hipMalloc(&w, (size_t) s.N * s.K * 2)); HC(hipMemset(w, 0x3c, (size_t) s.N * s.K * 2));
                HC(hipMalloc(&x, (size_t) T * s.K * 4)); HC(hipMemset(x, 0, (size_t) T * s.K * 4));
                HC(hipMalloc(&y, (size_t) T * s.N * 4)); HC(hipMalloc(&sc, (size_t) T * s.K * 2));
                talus_w16_gemm(kBF16, w, s.K, s.N, s.K, x, s.K, y, s.N, T, sc, (int64_t) T * s.K, nullptr);
                HC(hipEventRecord(e0));
                const int it = 10;
                for (int i = 0; i < it; ++i) talus_w16_gemm(kBF16, w, s.K, s.N, s.K, x, s.K, y, s.N, T, sc, (int64_t) T * s.K, nullptr);
                HC(hipEventRecord(e1)); HC(hipEventSynchronize(e1));
                float ms = 0; HC(hipEventElapsedTime(&ms, e0, e1));
                std::printf("bench gemm %-24s T %4d %7.3f ms  %5.1f TFLOPS (incl. the x rounding pass)\n", s.name, T, ms / it,
                            2.0 * T * s.N * s.K * it / (ms * 1e-3) / 1e12);
                HC(hipFree(w)); HC(hipFree(x)); HC(hipFree(y)); HC(hipFree(sc));
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    cpu_round();
    if (mode == "--cpu") {
        cpu_gemv();
        cpu_tiles();
    } else if (mode == "--bench") {
        bench();
    } else {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess || n == 0) {
            std::printf("no GPU: skipped (run --cpu for the CPU tests)\n");
            return 77;
        }
        hipDeviceProp_t pr{};
        HC(hipGetDeviceProperties(&pr, 0));
        std::printf("device %s\n", pr.gcnArchName);
        cpu_gemv();
        gpu_gemv();
        gpu_gemm();
    }
    std::printf("%s (%d failure%s)\n", g_fail ? "FAIL" : "PASS", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
