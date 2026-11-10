// q8p_bench  , GLM prompt lever 4 "quantize inside producers"): for each fused producer, the separate path
// (fp32 producer + tk_q36_mmq_quantize, what the engine runs with GLM_Q8P*=0) against the producer that writes the MMQ
// q8_1 rows itself, through libtalus_kernels' C ABI, on seeded random inputs.
//   HASH <case> T=<T> ref <h> fused <h>  : FNV-1a of the q8 bytes each path writes (only the bytes quantize writes)
//   CMP <case> T=<T> ok|DIFF            : bitwise equality of those bytes (DIFF also prints the first differing byte)
//   TIME <case> T=<T> prod <us> quant <us> fused <us> added <us> ratio <r> saved <us>
//        median of 20 launches each (5 warm-up), inputs restored before every launch (outside the timed span).
//        added = fused - prod (the cost of the store moved into the producer), ratio = added / quant
//   RULE <case> T=<T> ok|FAIL            : the lever's pre-registered stop rule: added <= 0.60 x quant
// Last lines: "Q8PRULE pass=P fail=F", then "Q8PBENCH DONE ok=N bad=M" (N = CMP ok, M = CMP DIFF + errors).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include "talus_glm_kernels.h"
#include "talus_kernels.h"

static int g_ok = 0, g_bad = 0, g_rule_pass = 0, g_rule_fail = 0;
static tk_ctx* g_ctx = nullptr;
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP error %s line %d\n", hipGetErrorString(e_), __LINE__); ++g_bad; return; } } while (0)
#define CKR(x, what) do { int r_ = (x); if (r_ != 0) { printf("ERR %s rc=%d line %d\n", what, r_, __LINE__); ++g_bad; return; } } while (0)

constexpr int Q4_K = 12, Q8_0 = 8, IQ2_S = 22;
static size_t mmq_used(size_t rows, size_t cols) { return rows * cols / 128 * 144; }   // cols % 512 == 0
static uint64_t fnv(const uint8_t* p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; } return h; }

struct Dev {   // device buffer with a pristine copy for restoring
    void* p = nullptr; void* keep = nullptr; size_t n = 0;
    ~Dev() { if (p) hipFree(p); if (keep) hipFree(keep); }
};
static bool dalloc(Dev& d, size_t n) { d.n = n; return hipMalloc(&d.p, n) == hipSuccess && hipMemset(d.p, 0, n) == hipSuccess; }
static bool dput(Dev& d, const void* h, size_t n, bool keep = false) {
    if (!dalloc(d, n) || hipMemcpy(d.p, h, n, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (keep) return hipMalloc(&d.keep, n) == hipSuccess && hipMemcpy(d.keep, d.p, n, hipMemcpyDeviceToDevice) == hipSuccess;
    return true;
}
static void restore(std::initializer_list<Dev*> ds) { for (Dev* d : ds) if (d->keep) hipMemcpyAsync(d->p, d->keep, d->n, hipMemcpyDeviceToDevice, 0); }

// median of 20 timed launches of f, setup() before each (not timed)
static float timeit(const std::function<int()>& f, const std::function<void()>& setup, int& rc) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); std::vector<float> ms;
    for (int r = 0; r < 25; ++r) {
        setup();
        hipEventRecord(a, 0); if (f() != 0) rc = 1; hipEventRecord(b, 0); hipEventSynchronize(b);
        float t = 0; hipEventElapsedTime(&t, a, b); if (r >= 5) ms.push_back(t);
    }
    hipEventDestroy(a); hipEventDestroy(b);
    std::sort(ms.begin(), ms.end()); return ms[ms.size() / 2] * 1e3f;
}

static void compare(const std::string& name, int T, const Dev& ref, const Dev& fus, size_t off, size_t n) {
    std::vector<uint8_t> a(n), b(n);
    hipMemcpy(a.data(), (uint8_t*) ref.p + off, n, hipMemcpyDeviceToHost);
    hipMemcpy(b.data(), (uint8_t*) fus.p + off, n, hipMemcpyDeviceToHost);
    printf("HASH %s T=%d ref %016llx fused %016llx\n", name.c_str(), T, (unsigned long long) fnv(a.data(), n), (unsigned long long) fnv(b.data(), n));
    size_t i = 0; while (i < n && a[i] == b[i]) ++i;
    if (i == n) { printf("CMP %s T=%d ok\n", name.c_str(), T); ++g_ok; }
    else { printf("CMP %s T=%d DIFF first byte %zu (block %zu, offset %zu) ref %02x fused %02x\n", name.c_str(), T, i, i / 144, i % 144, a[i], b[i]); ++g_bad; }
}
static void report(const std::string& name, int T, float prod, float quant, float fused) {
    const float added = fused - prod, ratio = added / quant, saved = prod + quant - fused;
    printf("TIME %s T=%d prod %.1f us quant %.1f us fused %.1f us added %.1f us ratio %.2f saved %.1f us\n", name.c_str(), T, prod, quant, fused, added, ratio, saved);
    const bool ok = ratio <= 0.60f;
    printf("RULE %s T=%d %s\n", name.c_str(), T, ok ? "ok" : "FAIL");
    (ok ? g_rule_pass : g_rule_fail)++;
}
static int quant(const float* x, void* xq, int type, int cols, int rows) {
    return (int) tk_q36_mmq_quantize(g_ctx, x, nullptr, xq, type, cols, cols, rows, (tk_stream) 0);
}

// ---- piece 1: KDA output gate -> wo (Q4_K, DS4). slices > 1: the E4 form (tw rows, gate per 512-row slice into rows
// r0.. of the tw-row matrix) against one quantize of all tw rows
static void kda_gate(int T, int slices, std::mt19937& rng) {
    const int H = 64, D = 128, cols = H * D;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> o((size_t) T * cols), g2(o.size()), w(D);
    for (auto& e : o) e = nd(rng); for (auto& e : g2) e = 2.f * nd(rng); for (auto& e : w) e = 1.f + 0.1f * nd(rng);
    for (int r = 0; r < T * H; r += 97) std::fill(o.begin() + (size_t) r * D, o.begin() + (size_t) (r + 1) * D, 0.f);   // zero groups
    Dev dO, dG, dW, dOut, dRef, dFus;
    if (!dput(dO, o.data(), o.size() * 4) || !dput(dG, g2.data(), g2.size() * 4) || !dput(dW, w.data(), D * 4) ||
        !dalloc(dOut, o.size() * 4) || !dalloc(dRef, tk_q36_mmq_q8_bytes(T, cols)) || !dalloc(dFus, tk_q36_mmq_q8_bytes(T, cols))) { ++g_bad; printf("ERR alloc\n"); return; }
    const float eps = 1e-6f;
    const std::string name = slices > 1 ? "kda_gate_e4" : "kda_gate";
    const int S = (T + slices - 1) / slices;
    auto prod = [&]() { return tkg_kda_gate((float*) dO.p, (float*) dG.p, (float*) dW.p, (float*) dOut.p, T, H, D, eps, 0); };
    auto qnt = [&]() { return quant((float*) dOut.p, dRef.p, Q4_K, cols, T); };
    auto fused = [&]() {
        for (int r0 = 0; r0 < T; r0 += S) {
            const int tt = std::min(S, T - r0);
            const int r = tkg_kda_gate_mmq((float*) dO.p + (size_t) r0 * cols, (float*) dG.p + (size_t) r0 * cols, (float*) dW.p,
                                           dFus.p, 1, r0, T, tt, H, D, eps, 0);
            if (r) return r;
        }
        return 0;
    };
    CKR(prod(), "tkg_kda_gate"); CKR(qnt(), "quantize"); CKR(fused(), "tkg_kda_gate_mmq"); CK(hipDeviceSynchronize());
    compare(name, T, dRef, dFus, 0, mmq_used(T, cols));
    int rc = 0; auto none = []() {};
    const float tp = timeit(prod, none, rc), tq = timeit(qnt, none, rc), tf = timeit(fused, none, rc);
    if (rc) { printf("ERR %s timing rc\n", name.c_str()); ++g_bad; }
    report(name, T, tp, tq, tf);
}

// ---- piece 2: MLA v_b (per head Q8_0 512 -> 256, 64 heads) -> wo (Q4_K, DS4); slices > 1: E4 (per 512-row slice into
// rows r0.. of the tw-row matrix)
static void mla_vb(int T, int slices, std::mt19937& rng) {
    const int H = 64, ni = 512, no = 256, cols = H * no;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> qd(-127, 127); std::uniform_real_distribution<float> dd(0.001f, 0.02f);
    std::vector<uint8_t> wq((size_t) H * no * ni / 32 * 34);
    for (size_t b = 0; b < wq.size() / 34; ++b) { _Float16 d = (_Float16) dd(rng); memcpy(&wq[b * 34], &d, 2); for (int i = 0; i < 32; ++i) wq[b * 34 + 2 + i] = (uint8_t) (int8_t) qd(rng); }
    std::vector<float> x((size_t) T * H * ni); for (auto& e : x) e = nd(rng);
    for (int t = 5; t < T; t += 211) std::fill(x.begin() + (size_t) t * H * ni, x.begin() + (size_t) t * H * ni + ni, 0.f);   // a zero head row
    Dev dW, dX, dY, dRef, dFus;
    if (!dput(dW, wq.data(), wq.size()) || !dput(dX, x.data(), x.size() * 4) || !dalloc(dY, (size_t) T * cols * 4) ||
        !dalloc(dRef, tk_q36_mmq_q8_bytes(T, cols)) || !dalloc(dFus, tk_q36_mmq_q8_bytes(T, cols))) { ++g_bad; printf("ERR alloc\n"); return; }
    const std::string name = slices > 1 ? "mla_vb_e4" : "mla_vb";
    const int S = (T + slices - 1) / slices;
    auto prod = [&]() { return tkg_heads_q8_0(dW.p, (float*) dX.p, (float*) dY.p, H, ni, no, T, 0); };
    auto qnt = [&]() { return quant((float*) dY.p, dRef.p, Q4_K, cols, T); };
    auto fused = [&]() {
        for (int r0 = 0; r0 < T; r0 += S) {
            const int tt = std::min(S, T - r0);
            const int r = tkg_heads_q8_0_mmq(dW.p, (float*) dX.p + (size_t) r0 * H * ni, dFus.p, 1, r0, T, H, ni, no, tt, 0);
            if (r) return r;
        }
        return 0;
    };
    CKR(prod(), "tkg_heads_q8_0"); CKR(qnt(), "quantize"); CKR(fused(), "tkg_heads_q8_0_mmq"); CK(hipDeviceSynchronize());
    compare(name, T, dRef, dFus, 0, mmq_used(T, cols));
    int rc = 0; auto none = []() {};
    const float tp = timeit(prod, none, rc), tq = timeit(qnt, none, rc), tf = timeit(fused, none, rc);
    if (rc) { printf("ERR %s timing rc\n", name.c_str()); ++g_bad; }
    report(name, T, tp, tq, tf);
}

// ---- piece 4: SwiGLU (clamped) -> shared expert / dense down (Q4_K, DS4)
static void swiglu(int T, int ff, float limit, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 3.f);
    std::vector<float> g((size_t) T * ff), u(g.size()); for (auto& e : g) e = nd(rng); for (auto& e : u) e = nd(rng);
    for (size_t i = 0; i + 32 <= g.size(); i += 32 * 101) std::fill(u.begin() + i, u.begin() + i + 32, 0.f);   // zero groups
    Dev dG, dU, dH, dRef, dFus;
    if (!dput(dG, g.data(), g.size() * 4) || !dput(dU, u.data(), u.size() * 4) || !dalloc(dH, g.size() * 4) ||
        !dalloc(dRef, tk_q36_mmq_q8_bytes(T, ff)) || !dalloc(dFus, tk_q36_mmq_q8_bytes(T, ff))) { ++g_bad; printf("ERR alloc\n"); return; }
    const std::string name = "swiglu_" + std::to_string(ff);
    auto prod = [&]() { return tkg_swiglu_clamp((float*) dG.p, (float*) dU.p, (float*) dH.p, T * ff, limit, 0); };
    auto qnt = [&]() { return quant((float*) dH.p, dRef.p, Q4_K, ff, T); };
    auto fused = [&]() { return tkg_swiglu_clamp_mmq((float*) dG.p, (float*) dU.p, dFus.p, 1, T, ff, limit, 0); };
    CKR(prod(), "tkg_swiglu_clamp"); CKR(qnt(), "quantize"); CKR(fused(), "tkg_swiglu_clamp_mmq"); CK(hipDeviceSynchronize());
    compare(name, T, dRef, dFus, 0, mmq_used(T, ff));
    int rc = 0; auto none = []() {};
    const float tp = timeit(prod, none, rc), tq = timeit(qnt, none, rc), tf = timeit(fused, none, rc);
    if (rc) { printf("ERR %s timing rc\n", name.c_str()); ++g_bad; }
    report(name, T, tp, tq, tf);
}

// ---- piece 3: mHC post-mix + pre2 (y) -> MMQ copies. moe: k_hc_post_mix_moe_pre (k = 8 expert rows), else
// k_hc_post_mix_pre mode 0. Copies (type, S, lim) as the engine plans them (plan_attn / plan_ffn).
struct CopySpec { int type, layout, S, lim; };
static void hc_pre(const std::string& name, int T, bool moe, std::vector<CopySpec> cps, std::mt19937& rng) {
    const int e = 4096, K = 4 * e, k = 8, HC_NS = 8, HC_PS = 64;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.05f, 0.4f);
    std::uniform_int_distribution<int> qd(-127, 127); std::uniform_real_distribution<float> dd(0.0005f, 0.004f);
    std::vector<float> xo((size_t) T * e), xhc((size_t) T * K), pc((size_t) T * 32), scale(3), base(24), nw(e);
    for (auto& v : xo) v = nd(rng); for (auto& v : xhc) v = nd(rng); for (auto& v : pc) v = ud(rng);
    for (auto& v : scale) v = 0.5f + 0.1f * nd(rng); for (auto& v : base) v = 0.3f * nd(rng); for (auto& v : nw) v = 1.f + 0.1f * nd(rng);
    std::vector<uint8_t> w((size_t) 24 * K / 32 * 34);
    for (size_t b = 0; b < w.size() / 34; ++b) { _Float16 d = (_Float16) dd(rng); memcpy(&w[b * 34], &d, 2); for (int i = 0; i < 32; ++i) w[b * 34 + 2 + i] = (uint8_t) (int8_t) qd(rng); }
    std::vector<float> ey, ew, sh;
    if (moe) {
        ey.resize((size_t) T * k * e); ew.resize((size_t) T * k); sh.resize((size_t) T * e);
        for (auto& v : ey) v = nd(rng); for (auto& v : ew) v = ud(rng); for (auto& v : sh) v = nd(rng);
    }
    Dev dXo, dXhc, dPc, dW, dParts, dScale, dBase, dNw, dY, dYq, dCnt, dEy, dEw, dSh;
    const size_t yqb = (size_t) T * e / 32 * 36;
    bool okA = dput(dXhc, xhc.data(), xhc.size() * 4, true) && dput(dPc, pc.data(), pc.size() * 4, true) && dput(dW, w.data(), w.size()) &&
               dalloc(dParts, (size_t) T * HC_NS * HC_PS * 4) && dput(dScale, scale.data(), 12) && dput(dBase, base.data(), 96) &&
               dput(dNw, nw.data(), e * 4) && dalloc(dY, (size_t) T * e * 4) && dalloc(dYq, yqb) && dalloc(dCnt, (size_t) T * 4);
    if (moe) okA = okA && dput(dEy, ey.data(), ey.size() * 4) && dput(dEw, ew.data(), ew.size() * 4) && dput(dSh, sh.data(), sh.size() * 4);
    else okA = okA && dput(dXo, xo.data(), xo.size() * 4);
    // copies: one area each, as many bytes as the whole window
    std::vector<Dev> ref(cps.size()), fus(cps.size());
    for (size_t c = 0; c < cps.size(); ++c) okA = okA && dalloc(ref[c], tk_q36_mmq_q8_bytes(T, e)) && dalloc(fus[c], tk_q36_mmq_q8_bytes(T, e));
    if (!okA) { ++g_bad; printf("ERR alloc\n"); return; }
    const float rms_eps = 1e-6f, hc_eps = 1e-6f; const int iters = 20;
    auto setup = [&]() { restore({&dXhc, &dPc}); };
    auto prod = [&]() {
        if (moe) return tkg_hc_post_mix_moe_pre((float*) dEy.p, (float*) dEw.p, (float*) dSh.p, (float*) dXhc.p, (float*) dPc.p, dW.p,
                                                (float*) dParts.p, T, k, e, (float*) dScale.p, (float*) dBase.p, (float*) dNw.p, (float*) dY.p,
                                                rms_eps, rms_eps, hc_eps, iters, dYq.p, (int*) dCnt.p, 0);
        return tkg_hc_post_mix_pre((float*) dXo.p, (float*) dXhc.p, (float*) dPc.p, dW.p, (float*) dParts.p, T, e, 0, (float*) dScale.p,
                                   (float*) dBase.p, (float*) dNw.p, (float*) dY.p, rms_eps, rms_eps, hc_eps, iters, dYq.p, (int*) dCnt.p, 0);
    };
    auto qnt = [&]() {   // what the engine's separate path quantizes: each copy's S-row slices below lim
        for (size_t c = 0; c < cps.size(); ++c)
            for (int r0 = 0; r0 < cps[c].lim; r0 += cps[c].S) {
                const int rows = std::min(cps[c].S, cps[c].lim - r0);
                const int r = quant((float*) dY.p + (size_t) r0 * e, (uint8_t*) ref[c].p + (size_t) r0 * (e / 128) * 144, cps[c].type, e, rows);
                if (r) return r;
            }
        return 0;
    };
    void* mp[2] = {nullptr, nullptr}; int ml[2] = {0, 0}, ms[2] = {1, 1}, mlim[2] = {0, 0};
    for (size_t c = 0; c < cps.size(); ++c) { mp[c] = fus[c].p; ml[c] = cps[c].layout; ms[c] = cps[c].S; mlim[c] = cps[c].lim; }
    auto fused = [&]() {
        if (moe) return tkg_hc_post_mix_moe_pre_mmq((float*) dEy.p, (float*) dEw.p, (float*) dSh.p, (float*) dXhc.p, (float*) dPc.p, dW.p,
                                                    (float*) dParts.p, T, k, e, (float*) dScale.p, (float*) dBase.p, (float*) dNw.p, (float*) dY.p,
                                                    rms_eps, rms_eps, hc_eps, iters, dYq.p, (int*) dCnt.p, (int) cps.size(),
                                                    mp[0], ml[0], ms[0], mlim[0], mp[1], ml[1], ms[1], mlim[1], 0);
        return tkg_hc_post_mix_pre_mmq((float*) dXo.p, (float*) dXhc.p, (float*) dPc.p, dW.p, (float*) dParts.p, T, e, 0, (float*) dScale.p,
                                       (float*) dBase.p, (float*) dNw.p, (float*) dY.p, rms_eps, rms_eps, hc_eps, iters, dYq.p, (int*) dCnt.p,
                                       (int) cps.size(), mp[0], ml[0], ms[0], mlim[0], mp[1], ml[1], ms[1], mlim[1], 0);
    };
    // reference: producer then the separate quantizes; also check the fused kernel's fp32 / q8_1 outputs are unchanged
    setup(); CKR(prod(), "post_mix_pre"); CKR(qnt(), "quantize"); CK(hipDeviceSynchronize());
    std::vector<uint8_t> y0((size_t) T * e * 4), yq0(yqb), x0(dXhc.n), pc0(dPc.n);
    hipMemcpy(y0.data(), dY.p, y0.size(), hipMemcpyDeviceToHost); hipMemcpy(yq0.data(), dYq.p, yqb, hipMemcpyDeviceToHost);
    hipMemcpy(x0.data(), dXhc.p, x0.size(), hipMemcpyDeviceToHost); hipMemcpy(pc0.data(), dPc.p, pc0.size(), hipMemcpyDeviceToHost);
    setup(); CK(hipMemset(dY.p, 0, dY.n)); CK(hipMemset(dYq.p, 0, yqb));
    CKR(fused(), "post_mix_pre_mmq"); CK(hipDeviceSynchronize());
    std::vector<uint8_t> y1(y0.size()), yq1(yqb), x1(x0.size()), pc1(pc0.size());
    hipMemcpy(y1.data(), dY.p, y1.size(), hipMemcpyDeviceToHost); hipMemcpy(yq1.data(), dYq.p, yqb, hipMemcpyDeviceToHost);
    hipMemcpy(x1.data(), dXhc.p, x1.size(), hipMemcpyDeviceToHost); hipMemcpy(pc1.data(), dPc.p, pc1.size(), hipMemcpyDeviceToHost);
    const bool same = y0 == y1 && yq0 == yq1 && x0 == x1 && pc0 == pc1;
    printf("CMP %s_outs T=%d %s\n", name.c_str(), T, same ? "ok" : "DIFF (y / yq / streams / pc of the mmq kernel)");
    (same ? g_ok : g_bad)++;
    for (size_t c = 0; c < cps.size(); ++c)
        compare(name + "_copy" + std::to_string(c), T, ref[c], fus[c], 0, mmq_used(cps[c].lim, e));
    int rc = 0;
    const float tp = timeit(prod, setup, rc), tq = timeit(qnt, []() {}, rc), tf = timeit(fused, setup, rc);
    if (rc) { printf("ERR %s timing rc\n", name.c_str()); ++g_bad; }
    report(name, T, tp, tq, tf);
}

int main() {
    if (tk_ctx_create(0, nullptr, &g_ctx) != TK_OK) { printf("ERR tk_ctx_create\nQ8PBENCH DONE ok=0 bad=1\n"); return 2; }
    printf("layouts Q4_K %d Q8_0 %d IQ2_S %d (0 D4, 1 DS4)\n", tk_q36_mmq_q8_layout(Q4_K), tk_q36_mmq_q8_layout(Q8_0), tk_q36_mmq_q8_layout(IQ2_S));
    std::mt19937 rng(11);
    for (int T : {512, 4096, 37}) {
        kda_gate(T, 1, rng);
        mla_vb(T, 1, rng);
        swiglu(T, 2048, 7.0f, rng);
        swiglu(T, 12288, 0.0f, rng);
        // FFN plan after the attention (k_hc_post_mix_pre): routed rows D4 (IQ2_S, s = T), shared expert DS4 (Q4_K, 2048-row slices)
        hc_pre("hc_ffn", T, false, {{IQ2_S, 0, T, T}, {Q4_K, 1, 2048, T}}, rng);
        // attention plan after a MoE FFN (k_hc_post_mix_moe_pre): KDA q/k/v DS4 (E4: s = T); MLA wq_a DS4 + wkv_a D4 per 512-row slice
        hc_pre("hc_kda", T, true, {{Q4_K, 1, T, T}}, rng);
        hc_pre("hc_mla", T, true, {{Q4_K, 1, 512, T}, {Q8_0, 0, 512, T}}, rng);
    }
    kda_gate(4096, 8, rng);   // E4 slices
    mla_vb(4096, 8, rng);
    // last chunk of a prompt: 2053 rows, the 5-row tail runs mat-vecs (lim 2048)
    hc_pre("hc_mla_tail", 2053, true, {{Q4_K, 1, 512, 2048}, {Q8_0, 0, 512, 2048}}, rng);
    printf("Q8PRULE pass=%d fail=%d\n", g_rule_pass, g_rule_fail);
    printf("Q8PBENCH DONE ok=%d bad=%d\n", g_ok, g_bad);
    tk_ctx_destroy(g_ctx);
    return g_bad ? 1 : 0;
}
