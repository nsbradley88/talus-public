// kda_step_bench: GLM KDA prompt-window step, v4 (one column per lane) vs v3, bitwise and timing (glm-kda4).
// Inputs: seeded random q / k / v / raw g / raw beta / dt bias / A, state evolved by v3 over two windows first.
// Checks dst and the whole state buffer (current slot + K_snap snapshot slots) bit for bit, several T / n_B / K_snap.
// Last line: "KDABENCH DONE ok=<0|1>".
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "strata/kernels/glm_spike_kernels.hpp"
using namespace talus::glm_spike;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); return 2; } } while (0)

struct Bufs { float *q, *k, *v, *g, *beta, *state, *dtb, *a, *dst; int32_t* ids; };

static int run_case(int T, int n_B, int K_snap, bool pro, uint32_t seed, bool time_it, int& bad) {
    const int H = 64, S = 128, NT = T * n_B, slots = 1 + n_B * std::max(K_snap, 1) + n_B;
    std::mt19937 rng(seed); std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& x, float sc) { for (auto& e : x) e = nd(rng) * sc; };
    std::vector<float> q((size_t) NT * H * S), k(q.size()), v(q.size()), g(q.size()), beta((size_t) NT * H), dtb(H * S), a(H);
    std::vector<float> st((size_t) slots * H * S * S);
    fill(q, 1.f); fill(k, 1.f); fill(v, 1.f); fill(g, 1.5f); fill(beta, 1.f); fill(dtb, 0.5f); fill(a, 1.f); fill(st, 0.05f);
    std::vector<int32_t> ids(std::max(1, K_snap) * n_B + n_B);
    for (int b = 0; b < n_B; ++b) ids[b] = b;                                   // slot_in
    for (int i = 0; i < K_snap * n_B; ++i) ids[i] = (i < n_B) ? i : n_B + i;  // snapshot slots (first = in-place)
    Bufs d[2];
    const size_t vb = q.size() * 4, stb = st.size() * 4;
    for (auto& b : d) {
        CK(hipMalloc(&b.q, vb)); CK(hipMalloc(&b.k, vb)); CK(hipMalloc(&b.v, vb)); CK(hipMalloc(&b.g, vb));
        CK(hipMalloc(&b.beta, beta.size() * 4)); CK(hipMalloc(&b.state, stb)); CK(hipMalloc(&b.dtb, dtb.size() * 4));
        CK(hipMalloc(&b.a, a.size() * 4)); CK(hipMalloc(&b.dst, vb)); CK(hipMalloc(&b.ids, ids.size() * 4));
        CK(hipMemcpy(b.q, q.data(), vb, hipMemcpyHostToDevice)); CK(hipMemcpy(b.k, k.data(), vb, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.v, v.data(), vb, hipMemcpyHostToDevice)); CK(hipMemcpy(b.g, g.data(), vb, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.beta, beta.data(), beta.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.dtb, dtb.data(), dtb.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.a, a.data(), a.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.ids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(b.state, st.data(), stb, hipMemcpyHostToDevice));
        CK(hipMemset(b.dst, 0, vb));
    }
    const float scale = 1.0f / std::sqrt(128.0f), lb = -5.0f, eps = 1e-6f;
    // evolve the state twice with v3 (no snapshots) on both copies, so the compared window starts from a used state
    for (int r = 0; r < 2; ++r)
        for (auto& b : d) CK(glm_kda_step_v3(b.q, b.k, b.v, b.g, b.beta, b.state, b.ids, b.dtb, b.a, b.dst, H, NT, 0, n_B, scale, pro, lb, eps, 0));
    CK(glm_kda_step_v3(d[0].q, d[0].k, d[0].v, d[0].g, d[0].beta, d[0].state, d[0].ids, d[0].dtb, d[0].a, d[0].dst, H, NT, K_snap, n_B, scale, pro, lb, eps, 0));
    CK(glm_kda_step_v4(d[1].q, d[1].k, d[1].v, d[1].g, d[1].beta, d[1].state, d[1].ids, d[1].dtb, d[1].a, d[1].dst, H, NT, K_snap, n_B, scale, pro, lb, eps, 0));
    CK(hipDeviceSynchronize());
    std::vector<float> o0(q.size()), o1(q.size()), s0(st.size()), s1(st.size());
    CK(hipMemcpy(o0.data(), d[0].dst, vb, hipMemcpyDeviceToHost)); CK(hipMemcpy(o1.data(), d[1].dst, vb, hipMemcpyDeviceToHost));
    CK(hipMemcpy(s0.data(), d[0].state, stb, hipMemcpyDeviceToHost)); CK(hipMemcpy(s1.data(), d[1].state, stb, hipMemcpyDeviceToHost));
    size_t dd = 0, ds = 0, nonfin = 0; double maxrel = 0;
    for (size_t i = 0; i < o0.size(); ++i) { if (memcmp(&o0[i], &o1[i], 4)) { ++dd; maxrel = std::max(maxrel, (double) std::fabs(o0[i] - o1[i]) / (std::fabs(o0[i]) + 1e-30)); } if (!std::isfinite(o0[i])) ++nonfin; }
    for (size_t i = 0; i < s0.size(); ++i) if (memcmp(&s0[i], &s1[i], 4)) ++ds;
    const bool ok = dd == 0 && ds == 0;
    if (!ok) ++bad;
    printf("CASE T=%d n_B=%d K_snap=%d prologue=%d seed=%u: dst diff %zu / %zu (max rel %.3g), state diff %zu / %zu, nonfinite %zu -> %s\n",
           T, n_B, K_snap, (int) pro, seed, dd, o0.size(), maxrel, ds, s0.size(), nonfin, ok ? "BITWISE" : "DIFF");
    if (time_it) {
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        for (int ver = 3; ver <= 4; ++ver) {
            std::vector<float> ms;
            for (int r = 0; r < 25; ++r) {
                CK(hipEventRecord(e0, 0));
                if (ver == 3) CK(glm_kda_step_v3(d[0].q, d[0].k, d[0].v, d[0].g, d[0].beta, d[0].state, d[0].ids, d[0].dtb, d[0].a, d[0].dst, H, NT, 0, n_B, scale, pro, lb, eps, 0));
                else CK(glm_kda_step_v4(d[1].q, d[1].k, d[1].v, d[1].g, d[1].beta, d[1].state, d[1].ids, d[1].dtb, d[1].a, d[1].dst, H, NT, 0, n_B, scale, pro, lb, eps, 0));
                CK(hipEventRecord(e1, 0)); CK(hipEventSynchronize(e1));
                float t = 0; CK(hipEventElapsedTime(&t, e0, e1)); if (r >= 5) ms.push_back(t);
            }
            std::sort(ms.begin(), ms.end());
            printf("TIME T=%d v%d median %.1f us min %.1f us max %.1f us\n", T, ver, ms[ms.size() / 2] * 1e3, ms.front() * 1e3, ms.back() * 1e3);
        }
    }
    for (auto& b : d) { hipFree(b.q); hipFree(b.k); hipFree(b.v); hipFree(b.g); hipFree(b.beta); hipFree(b.state); hipFree(b.dtb); hipFree(b.a); hipFree(b.dst); hipFree(b.ids); }
    return 0;
}

int main() {
    int bad = 0, rc = 0;
    struct C { int T, n_B, K; bool pro; bool time_it; } cs[] = {
        {512, 1, 0, true, true}, {512, 1, 1, true, false}, {512, 1, 3, true, false}, {9, 1, 0, true, false},
        {37, 1, 2, true, false}, {100, 1, 0, false, false}, {2048, 1, 0, true, true}, {4096, 1, 0, true, true},
        {256, 2, 0, true, false}, {17, 3, 2, true, false}, {513, 1, 0, true, false}};
    uint32_t seed = 1;
    for (auto& c : cs) { rc |= run_case(c.T, c.n_B, c.K, c.pro, seed++, c.time_it, bad); if (rc) break; }
    for (uint32_t s2 = 100; s2 < 104 && !rc; ++s2) rc |= run_case(512, 1, 0, true, s2, false, bad);
    printf("KDABENCH DONE ok=%d bad=%d rc=%d\n", (bad == 0 && rc == 0) ? 1 : 0, bad, rc);
    return rc;
}
