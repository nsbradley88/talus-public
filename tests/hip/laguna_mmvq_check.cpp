// native_mmvq on HIP for the weight formats Laguna-S-2.1 uses (Q4_K, Q5_K, Q6_K, Q8_0), at its
// shapes, against a CPU float reference.  Weights are random floats quantized by ggml's own quantizer and
// dequantized back for the reference, so the only expected difference is the kernel's Q8_1 activation rounding.
// Exit 0 when every case is within tolerance.
#include "strata/kernels/native_mmvq.hpp"

#include <hip/hip_runtime.h>

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); return false; } } while (0)

using MmvqFn = void (*)(const void*, const float*, void*, float*, int, int, int, void*);

static bool run_case(const char* name, ggml_type type, MmvqFn fn, int n_in, int n_out, int ncols, hipStream_t s,
                     std::mt19937& rng, double& worst) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> w((size_t) n_in * n_out), x((size_t) n_in * ncols);
    for (float& v : w) v = dist(rng) * 0.05f;
    for (float& v : x) v = dist(rng);
    const size_t row_bytes = ggml_row_size(type, n_in);
    std::vector<uint8_t> wq(row_bytes * n_out);
    ggml_quantize_chunk(type, w.data(), wq.data(), 0, n_out, n_in, nullptr);
    std::vector<float> wd((size_t) n_in * n_out);
    ggml_get_type_traits(type)->to_float(wq.data(), wd.data(), (int64_t) n_in * n_out);

    std::vector<double> ref((size_t) n_out * ncols);
    for (int c = 0; c < ncols; ++c)
        for (int o = 0; o < n_out; ++o) {
            double acc = 0;
            for (int i = 0; i < n_in; ++i) acc += (double) wd[(size_t) o * n_in + i] * x[(size_t) c * n_in + i];
            ref[(size_t) c * n_out + o] = acc;
        }

    void *dw = nullptr, *dq = nullptr;
    float *dx = nullptr, *dy = nullptr;
    CHECK(hipMalloc(&dw, wq.size()));
    CHECK(hipMalloc(&dx, x.size() * sizeof(float)));
    CHECK(hipMalloc(&dq, strata::kernels::native_q8_1_bytes(n_in, ncols)));
    CHECK(hipMalloc(&dy, ref.size() * sizeof(float)));
    CHECK(hipMemcpy(dw, wq.data(), wq.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice));
    fn(dw, dx, dq, dy, n_in, n_out, ncols, s);
    CHECK(hipStreamSynchronize(s));
    std::vector<float> y(ref.size());
    CHECK(hipMemcpy(y.data(), dy, y.size() * sizeof(float), hipMemcpyDeviceToHost));
    hipFree(dw); hipFree(dx); hipFree(dq); hipFree(dy);

    // error relative to the output's RMS (individual outputs can be near zero)
    double ss = 0, err = 0;
    for (size_t i = 0; i < ref.size(); ++i) { ss += ref[i] * ref[i]; err = std::max(err, std::fabs(y[i] - ref[i])); }
    const double rms = std::sqrt(ss / (double) ref.size());
    const double rel = err / rms;
    worst = std::max(worst, rel);
    const bool ok = std::isfinite(rel) && rel < 0.05;
    std::printf("%-6s n_in %5d n_out %5d ncols %d  max|err|/rms %.5f  %s\n", name, n_in, n_out, ncols, rel, ok ? "ok" : "FAIL");
    return ok;
}

int main() {
    ggml_cpu_init();
    hipStream_t s;
    if (hipStreamCreate(&s) != hipSuccess) { std::fprintf(stderr, "no HIP device\n"); return 2; }
    std::mt19937 rng(7);
    struct Fmt { const char* name; ggml_type type; MmvqFn fn; };
    const Fmt fmts[] = {
        {"Q4_K", GGML_TYPE_Q4_K, strata::kernels::native_q4_k_f32},
        {"Q5_K", GGML_TYPE_Q5_K, strata::kernels::native_q5_k_f32},
        {"Q6_K", GGML_TYPE_Q6_K, strata::kernels::native_q6_k_f32},
        {"Q8_0", GGML_TYPE_Q8_0, strata::kernels::native_q8_0_f32},
    };
    // Laguna shapes: expert gate/up and k/v (3072 -> 1024), expert down (1024 -> 3072), SWA q (3072 -> 9216)
    const int shapes[][2] = {{3072, 1024}, {1024, 3072}, {3072, 9216}};
    bool all = true;
    double worst = 0;
    for (const Fmt& f : fmts)
        for (const auto& sh : shapes)
            for (int ncols : {1, 8}) all &= run_case(f.name, f.type, f.fn, sh[0], sh[1], ncols, s, rng, worst);
    std::printf("%s (worst max|err|/rms %.5f)\n", all ? "ALL OK" : "FAILURES", worst);
    return all ? 0 : 1;
}
