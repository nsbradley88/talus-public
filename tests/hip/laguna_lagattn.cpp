// lagattn: Laguna prompt attention versions (env LAG_VERS, default "124"): time, and bitwise vs the first one.
// usage: lagattn [nq] [p0 ...] (links libtalus_kernels.so); env LAG_H="48,72" LAG_Q8="1,0" LAG_WIN="0,512"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "talus_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
extern "C" void talus_laguna_prompt_attn_version(int v);
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)
int main(int argc, char** argv) {
  const int nq = argc > 1 ? atoi(argv[1]) : 2048, n_kv = 8, D = 128;
  int fails = 0;
  auto ilist = [](const char* env, std::vector<int> def) {
    const char* e = getenv(env); if (!e) return def; std::vector<int> v; const char* c = e;
    while (*c) { v.push_back(atoi(c)); while (*c && *c != ',') ++c; if (*c) ++c; } return v; };
  std::vector<int> p0s; for (int i = 2; i < argc; ++i) p0s.push_back(atoi(argv[i]));
  if (p0s.empty()) p0s = {0, 6144, 30000};
  std::vector<int> vers; { const char* e = getenv("LAG_VERS"); for (const char* c = e ? e : "124"; *c; ++c) vers.push_back(*c - '0'); }
  for (int H : ilist("LAG_H", {48, 72}))
  for (int q8 : ilist("LAG_Q8", {1, 0}))
  for (int window : ilist("LAG_WIN", {0, 512}))
  for (int p0 : p0s) {
    const int ctx = (p0 + nq + 15) / 16 * 16 + 16, stride = ctx;   // vt_stride = positions per row (no ring)
    std::mt19937 rng(p0 + H + window + q8);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t kel = (size_t) ctx * n_kv * D;
    std::vector<uint16_t> k16(kel), v16(kel);
    std::vector<int8_t> k8(kel), v8(kel);
    std::vector<uint16_t> ks((size_t) ctx * n_kv * 4), vs((size_t) n_kv * (stride / 16) * 4 * 16);
    for (size_t i = 0; i < kel; ++i) { __half a = __float2half(nd(rng)), b = __float2half(nd(rng)); memcpy(&k16[i], &a, 2); memcpy(&v16[i], &b, 2);
                                       k8[i] = (int8_t) (int) (nd(rng) * 40); v8[i] = (int8_t) (int) (nd(rng) * 40); }
    for (auto& x : ks) { __half a = __float2half(0.01f + 0.02f * (rng() % 100) / 100.f); memcpy(&x, &a, 2); }
    for (auto& x : vs) { __half a = __float2half(0.01f + 0.02f * (rng() % 100) / 100.f); memcpy(&x, &a, 2); }
    std::vector<float> hq((size_t) nq * H * D);
    for (auto& x : hq) x = nd(rng);
    void *dk, *dv, *dks, *dvs; float *dq, *o1, *o2;
    const size_t kb = q8 ? kel : kel * 2;
    CK(hipMalloc(&dk, kb)); CK(hipMalloc(&dv, kb)); CK(hipMalloc(&dks, ks.size() * 2)); CK(hipMalloc(&dvs, vs.size() * 2));
    CK(hipMalloc(&dq, hq.size() * 4)); CK(hipMalloc(&o1, hq.size() * 4)); CK(hipMalloc(&o2, hq.size() * 4));
    CK(hipMemcpy(dk, q8 ? (void*) k8.data() : (void*) k16.data(), kb, hipMemcpyHostToDevice));
    CK(hipMemcpy(dv, q8 ? (void*) v8.data() : (void*) v16.data(), kb, hipMemcpyHostToDevice));
    CK(hipMemcpy(dks, ks.data(), ks.size() * 2, hipMemcpyHostToDevice)); CK(hipMemcpy(dvs, vs.data(), vs.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
    hipStream_t s; CK(hipStreamCreate(&s));
    auto run = [&](int ver, float* o) {
      talus_laguna_prompt_attn_version(ver);
      tk_status st = tk_laguna_prompt_attn_wmma(nullptr, dq, (const uint16_t*) dk, (const uint16_t*) dv, o, nq, p0, H, n_kv, window,
                                                stride, nullptr, 1, nullptr, nullptr, q8 ? (const uint16_t*) dks : nullptr,
                                                q8 ? (const uint16_t*) dvs : nullptr, (tk_stream) s);
      if (st != TK_OK) { printf("status %d\n", (int) st); exit(1); }
    };
    std::vector<float*> outs(vers.size()); std::vector<float> tv(vers.size());
    for (size_t vi = 0; vi < vers.size(); ++vi) {
      CK(hipMalloc(&outs[vi], hq.size() * 4)); CK(hipMemset(outs[vi], 0xff, hq.size() * 4));
      const int ver = vers[vi]; float* o = outs[vi];
      run(ver, o); CK(hipStreamSynchronize(s));
      hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
      const int reps = p0 > 60000 ? 2 : 5;
      CK(hipEventRecord(a, s)); for (int r = 0; r < reps; ++r) run(ver, o); CK(hipEventRecord(b, s)); CK(hipEventSynchronize(b));
      CK(hipEventElapsedTime(&tv[vi], a, b)); tv[vi] /= reps;
    }
    std::vector<float> a(hq.size()), b(hq.size());
    CK(hipMemcpy(a.data(), outs[0], a.size() * 4, hipMemcpyDeviceToHost));
    const double flop = 4.0 * nq * H * D * (window ? (double) window : (double) p0 + nq / 2.0);
    printf("H %d q8 %d window %4d p0 %6d nq %d: v%d %8.3f ms (%5.1f TF)", H, q8, window, p0, nq, vers[0], tv[0], flop / tv[0] / 1e9);
    for (size_t vi = 1; vi < vers.size(); ++vi) {
      CK(hipMemcpy(b.data(), outs[vi], b.size() * 4, hipMemcpyDeviceToHost));
      size_t nd_ = 0; for (size_t i = 0; i < a.size(); ++i) nd_ += memcmp(&a[i], &b[i], 4) != 0;
      fails += nd_ != 0;
      printf(" | v%d %8.3f ms x%.2f %s", vers[vi], tv[vi], tv[0] / tv[vi], nd_ ? "DIFF" : "bitwise");
      if (nd_) printf("(%zu)", nd_);
    }
    printf("\n"); fflush(stdout);
    for (float* o : outs) hipFree(o);
    hipFree(o1); hipFree(o2);
    hipFree(dk); hipFree(dv); hipFree(dks); hipFree(dvs); hipFree(dq); hipStreamDestroy(s);
  }
  printf("lagattn: %d failures\n", fails);
  return fails ? 1 : 0;
}
