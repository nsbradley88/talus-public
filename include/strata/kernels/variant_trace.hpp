// include/strata/kernels/variant_trace.hpp - S1: the decision trace of the kernel library's host-side
// selectors.
//
// TALUS_VARIANT_TRACE=<path> appends one line per first use of each (site, variant, keys) tuple:
//     <site>\t<variant>\t<k0>,<k1>,<k2>,<k3>
// `site` names the selector (export or launcher), `variant` the kernel family / template it picked, the keys the
// shape or plan values that picked it (types, column counts, rows-per-wave, split counts...). The Rust port moves a
// selector only when its trace over the model's corpus stays identical, so the trace documents every decision that
// can change which kernel runs or how it reduces.
//
// Unset (the default) costs one relaxed atomic load per call site. Header-only: every translation unit of the shared
// library shares the one state (inline functions, C++17).
#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>

namespace strata::kernels::vtrace {

struct State {
    std::mutex mu;
    std::FILE* f = nullptr;
    std::unordered_set<std::string> seen;
};

inline State& state() {
    static State s;
    return s;
}

// 0 = not checked yet, 1 = off, 2 = on
inline std::atomic<int>& mode() {
    static std::atomic<int> m{0};
    return m;
}

inline bool init_slow() {
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    int m = mode().load(std::memory_order_relaxed);
    if (m != 0) return m == 2;
    const char* p = std::getenv("TALUS_VARIANT_TRACE");
    if (p != nullptr && p[0] != '\0') s.f = std::fopen(p, "a");
    m = s.f != nullptr ? 2 : 1;
    mode().store(m, std::memory_order_release);
    return m == 2;
}

inline bool enabled() {
    const int m = mode().load(std::memory_order_acquire);
    if (m != 0) return m == 2;
    return init_slow();
}

inline void record(const char* site, const char* variant, long long k0, long long k1, long long k2, long long k3) {
    char line[256];
    const int n = std::snprintf(line, sizeof line, "%s\t%s\t%lld,%lld,%lld,%lld\n", site, variant, k0, k1, k2, k3);
    if (n <= 0) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.f == nullptr) return;
    if (!s.seen.emplace(line, static_cast<size_t>(n < (int) sizeof line ? n : (int) sizeof line - 1)).second) return;
    std::fputs(line, s.f);
    std::fflush(s.f);
}

// S6: a host-computed launch choice, checked against the library's own for one release. Returns the value
// to run: `lib` unless they agree or TALUS_PLAN_TRUST=1 (then `host`). A disagreement is recorded (TALUS_VARIANT_TRACE
// "plan_mismatch") and reported on stderr the first 8 times per process.
inline int plan_choose(const char* site, int host, int lib) {
    if (host == lib) return host;
    static const bool trust = [] { const char* e = std::getenv("TALUS_PLAN_TRUST"); return e != nullptr && e[0] == '1'; }();
    if (enabled()) record("plan_mismatch", site, host, lib, trust ? 1 : 0, 0);
    static std::atomic<int> warned{0};
    if (warned.fetch_add(1) < 8)
        std::fprintf(stderr, "talus: %s: host plan %d differs from the library's %d; running the %s\n", site, host, lib,
                     trust ? "host's" : "library's");
    return trust ? host : lib;
}

}  // namespace strata::kernels::vtrace

// TALUS_VTRACE(site, variant[, k0[, k1[, k2[, k3]]]]): site/variant are string literals (or const char*), keys integers.
#define TALUS_VTRACE_N(site, variant, k0, k1, k2, k3, ...)                                                         \
    do {                                                                                                            \
        if (::strata::kernels::vtrace::enabled())                                                                   \
            ::strata::kernels::vtrace::record((site), (variant), (long long) (k0), (long long) (k1),                 \
                                              (long long) (k2), (long long) (k3));                                  \
    } while (0)
#define TALUS_VTRACE(...) TALUS_VTRACE_N(__VA_ARGS__, 0, 0, 0, 0, 0)
