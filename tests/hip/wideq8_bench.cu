// wideq8_bench (glm-wideq8): the Q8_0 wide WMMA kernel at the GLM prompt shapes through libtalus_kernels' C ABI.
// Run once with GLM_N2_ST=0 and once with GLM_N2_ST=1: equal "HASH" lines = bitwise; "TIME" = median of 20 launches;
// "FLOOR" = hipMemsetAsync of the same output bytes (the store floor). Last line "WIDEQ8 DONE".
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "talus_glm_kernels.h"
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s line %d\n", hipGetErrorString(e_), __LINE__); return 2; } } while (0)
static uint64_t fnv(const std::vector<float>& v) { uint64_t h = 1469598103934665603ull; const uint8_t* p = (const uint8_t*) v.data();
  for (size_t i = 0; i < v.size() * 4; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }
static void q8w(std::vector<uint8_t>& w, size_t blocks, std::mt19937& rng) {   // block_q8_0: fp16 d + 32 int8
  std::uniform_int_distribution<int> qd(-127, 127); std::uniform_real_distribution<float> dd(0.001f, 0.02f);
  w.resize(blocks * 34);
  for (size_t b = 0; b < blocks; ++b) { _Float16 d = (_Float16) dd(rng); memcpy(&w[b * 34], &d, 2); for (int i = 0; i < 32; ++i) w[b * 34 + 2 + i] = (uint8_t) (int8_t) qd(rng); }
}
template <class F> static float timeit(F f, int& rc) {
  hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); std::vector<float> ms;
  for (int r = 0; r < 25; ++r) { hipEventRecord(a, 0); if (f() != 0) rc = 1; hipEventRecord(b, 0); hipEventSynchronize(b);
    float t = 0; hipEventElapsedTime(&t, a, b); if (r >= 5) ms.push_back(t); }
  std::sort(ms.begin(), ms.end()); return ms[ms.size() / 2] * 1e3f;
}
int main() {
  std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f); int rc = 0;
  for (int T : {512, 64, 37}) {
    // KDA f_b + g_b: 2 matrices, n_in 128 -> n_out 8192
    const int n_in = 128, n_out = 8192;
    std::vector<uint8_t> w0, w1; q8w(w0, (size_t) n_out * n_in / 32, rng); q8w(w1, (size_t) n_out * n_in / 32, rng);
    std::vector<float> x0((size_t) T * n_in), x1(x0.size()); for (auto& e : x0) e = nd(rng); for (auto& e : x1) e = nd(rng);
    void *dw0, *dw1; float *dx0, *dx1, *dy0, *dy1; size_t yb = (size_t) T * n_out * 4;
    CK(hipMalloc(&dw0, w0.size())); CK(hipMalloc(&dw1, w1.size())); CK(hipMalloc(&dx0, x0.size() * 4)); CK(hipMalloc(&dx1, x1.size() * 4));
    CK(hipMalloc(&dy0, yb)); CK(hipMalloc(&dy1, yb));
    CK(hipMemcpy(dw0, w0.data(), w0.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dw1, w1.data(), w1.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(dx0, x0.data(), x0.size() * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(dx1, x1.data(), x1.size() * 4, hipMemcpyHostToDevice));
    const void* ws[2] = {dw0, dw1}; const float* xs[2] = {dx0, dx1}; float* ys[2] = {dy0, dy1}; int no[2] = {n_out, n_out};
    auto run = [&]() { return tkg_multi_q8_0(2, ws, xs, ys, no, n_in, T, 0); };
    CK(hipMemset(dy0, 0, yb)); CK(hipMemset(dy1, 0, yb)); if (run()) { printf("multi_q8 rc\n"); rc = 1; } CK(hipDeviceSynchronize());
    std::vector<float> y(T * (size_t) n_out * 2); CK(hipMemcpy(y.data(), dy0, yb, hipMemcpyDeviceToHost)); CK(hipMemcpy(y.data() + y.size() / 2, dy1, yb, hipMemcpyDeviceToHost));
    printf("HASH kda_fb_gb T=%d %016llx\n", T, (unsigned long long) fnv(y));
    const float tk = timeit(run, rc);
    const float tf = timeit([&]() { return (int) (hipMemsetAsync(dy0, 0, yb, 0) | hipMemsetAsync(dy1, 0, yb, 0)); }, rc);
    printf("TIME kda_fb_gb T=%d %.1f us FLOOR %.1f us ratio %.2f\n", T, tk, tf, tk / tf);
    hipFree(dw0); hipFree(dw1); hipFree(dx0); hipFree(dx1); hipFree(dy0); hipFree(dy1);
    // MLA per-head wk_b / wv_b: 64 heads, (n_in 256 -> n_out 512) and (512 -> 256)
    for (auto sh : {std::pair<int, int>{256, 512}, std::pair<int, int>{512, 256}}) {
      const int H = 64, ni = sh.first, nout = sh.second;
      std::vector<uint8_t> w; q8w(w, (size_t) H * nout * ni / 32, rng);
      std::vector<float> x((size_t) T * H * ni); for (auto& e : x) e = nd(rng);
      void* dw; float *dx, *dy; size_t yb2 = (size_t) T * H * nout * 4;
      CK(hipMalloc(&dw, w.size())); CK(hipMalloc(&dx, x.size() * 4)); CK(hipMalloc(&dy, yb2));
      CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
      auto run2 = [&]() { return tkg_heads_q8_0(dw, dx, dy, H, ni, nout, T, 0); };
      CK(hipMemset(dy, 0, yb2)); if (run2()) { printf("heads_q8 rc\n"); rc = 1; } CK(hipDeviceSynchronize());
      std::vector<float> yy((size_t) T * H * nout); CK(hipMemcpy(yy.data(), dy, yb2, hipMemcpyDeviceToHost));
      printf("HASH mla_heads_%dx%d T=%d %016llx\n", ni, nout, T, (unsigned long long) fnv(yy));
      const float t2 = timeit(run2, rc), f2 = timeit([&]() { return (int) hipMemsetAsync(dy, 0, yb2, 0); }, rc);
      printf("TIME mla_heads_%dx%d T=%d %.1f us FLOOR %.1f us ratio %.2f\n", ni, nout, T, t2, f2, t2 / f2);
      hipFree(dw); hipFree(dx); hipFree(dy);
    }
  }
  printf("WIDEQ8 DONE rc=%d\n", rc);
  return rc;
}
