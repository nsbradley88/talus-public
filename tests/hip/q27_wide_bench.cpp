// the 27B's weight pass at verify widths on the real model, outside the engine.
//   talus-q27w-bench MODEL.gguf [table|exact|all] [--widths 1,2,..] [--iters N] [--variants w32,tw,twp] [--layers LO-HI]
// table: every 2-D weight of blk.0-63 + output.weight (the --mmvq-bench / wpass.py set: no token_embd, no MTP layer
//        64, no conv1d) as the engine launches them, one call per matrix with n columns; per type and width the kernel
//        time (hipEvents around the type's matrices back to back, median of N passes: cold, the pass is 13 GB) and GB/s,
//        the whole pass in ms and % of 256 GB/s.  Variants: w32 = today's wave32 kernels (native_mmvq), tw = the wide
//        tiles on the GGUF rows (native_mmvq_tw mode 1), twp = the tiles on the tile-plane copies (mode 2); 
//        w32ts = the wave32 kernels with the tsum epilogue (native_mmvq_set_w32_ts(1)), tws = the tiles with the split
//        decode (mode 4, packed), trp = one row per lane (mode 3, packed; types / widths it does not take run w32).
// exact: for the matrices of layers LO-HI (default 0-3) and one matrix of every (type, shape), 16 columns of random
//        activations; reference = each column alone through native_mmvq (1 column, the decode path); every variant at
//        every width 1..16 must give each column's rows bitwise equal (also the 3-matrix multi launch for matrices
//        sharing an input: multi = tiles mode 2, multi-tws, multi-trp, multi-w32ts = native_mmvq_w32_multi with the
//        tsum epilogue at 2..8 columns; 1 column takes the EP 0 kernel).  The reference always runs with the tsum epilogue off.  Prints per-row hashes of the reference (fnv1a over the rows) so builds can be compared.
// Needs MemAvailable >= 20 GiB at start (weights 13.3 GB + tile copies up to 12.7 GB); aborts below 8 GiB.
#include "strata/kernels/native_mmvq.hpp"

#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

static long mem_avail_kb() {
    FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %ld kB", &v) == 1) break;
    std::fclose(f);
    return v;
}
static void mem_guard(const char* where, long min_gib) {
    const long kb = mem_avail_kb();
    if (kb >= 0 && kb < min_gib * 1024 * 1024) {
        std::fprintf(stderr, "MemAvailable %.1f GiB < %ld GiB at %s: abort\n", kb / 1048576.0, min_gib, where);
        std::exit(2);
    }
}

// ---------------------------------------------------------------- minimal GGUF reader (header + mmap)
struct Tensor { std::string name; int type = 0; int64_t ne0 = 0, ne1 = 1, ne2 = 1; uint64_t off = 0; std::size_t bytes = 0; void* d = nullptr; };
static std::size_t type_bytes(int t, int64_t ne) {
    switch (t) {
    case 0: return ne * 4; case 1: case 30: return ne * 2;
    case 2: return ne / 32 * 18; case 8: return ne / 32 * 34; case 20: return ne / 32 * 18;
    case 10: return ne / 256 * 84; case 11: return ne / 256 * 110; case 12: return ne / 256 * 144; case 13: return ne / 256 * 176;
    case 14: return ne / 256 * 210; case 16: return ne / 256 * 66; case 17: return ne / 256 * 74; case 18: return ne / 256 * 98;
    case 21: return ne / 256 * 110; case 22: return ne / 256 * 82; case 23: return ne / 256 * 136;
    default: return 0;
    }
}
static const char* type_name(int t) {
    switch (t) {
    case 2: return "Q4_0"; case 8: return "Q8_0"; case 10: return "Q2_K"; case 11: return "Q3_K"; case 12: return "Q4_K";
    case 13: return "Q5_K"; case 14: return "Q6_K"; case 16: return "IQ2_XXS"; case 17: return "IQ2_XS"; case 18: return "IQ3_XXS";
    case 20: return "IQ4_NL"; case 21: return "IQ3_S"; case 22: return "IQ2_S"; case 23: return "IQ4_XS";
    default: return "?";
    }
}
struct Gguf {
    const uint8_t* map = nullptr;
    std::size_t size = 0, p = 0, data0 = 0;
    std::vector<Tensor> t;
    template<typename T> T rd() { T v; std::memcpy(&v, map + p, sizeof v); p += sizeof v; return v; }
    std::string str() { const uint64_t n = rd<uint64_t>(); std::string s(reinterpret_cast<const char*>(map + p), n); p += n; return s; }
    void skip(uint32_t ty) {
        static const int sz[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (ty == 8) { (void) str(); return; }
        if (ty == 9) { const uint32_t at = rd<uint32_t>(); const uint64_t n = rd<uint64_t>(); for (uint64_t i = 0; i < n; ++i) skip(at); return; }
        p += sz[ty];
    }
    bool open(const char* path) {
        const int fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        fstat(fd, &st);
        size = std::size_t(st.st_size);
        map = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        ::close(fd);
        if (map == MAP_FAILED) return false;
        p = 8;   // magic, version
        const uint64_t nt = rd<uint64_t>(), nkv = rd<uint64_t>();
        uint64_t align = 32;
        for (uint64_t i = 0; i < nkv; ++i) {
            const std::string key = str();
            const uint32_t ty = rd<uint32_t>();
            if (key == "general.alignment" && ty == 4) align = rd<uint32_t>();
            else skip(ty);
        }
        for (uint64_t i = 0; i < nt; ++i) {
            Tensor x;
            x.name = str();
            const uint32_t nd = rd<uint32_t>();
            int64_t ne[4] = {1, 1, 1, 1};
            for (uint32_t d = 0; d < nd; ++d) ne[d] = int64_t(rd<uint64_t>());
            x.type = int(rd<uint32_t>());
            x.off = rd<uint64_t>();
            x.ne0 = ne[0]; x.ne1 = ne[1]; x.ne2 = ne[2];
            x.bytes = type_bytes(x.type, ne[0] * ne[1] * ne[2]);
            t.push_back(x);
        }
        data0 = (p + align - 1) / align * align;
        return true;
    }
};

static bool in_pass(const Tensor& x) {
    if (x.ne1 <= 1 || x.ne2 != 1 || x.bytes == 0) return false;
    if (x.name == "token_embd.weight") return false;
    if (x.name.rfind("blk.64.", 0) == 0) return false;
    if (x.name.find("conv1d") != std::string::npos) return false;
    if (!(x.name.rfind("blk.", 0) == 0 || x.name == "output.weight")) return false;
    return k::native_mmvq_supported(x.type);
}
static int layer_of(const std::string& n) { return n.rfind("blk.", 0) == 0 ? std::atoi(n.c_str() + 4) : -1; }

static uint64_t fnv(const void* p, std::size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [table|exact|all] [--widths 1,2,..] [--iters N] [--variants w32,tw,twp] [--layers LO-HI]\n", argv[0]);
        return 1;
    }
    // the tiles on for this process (no model loader opts it in here; the library reads the switch once)
    setenv("TALUS_MMVQ_WIDE", "1", 0);
    std::string mode = argc > 2 && argv[2][0] != '-' ? argv[2] : "all";
    std::vector<int> widths = {1, 2, 4, 5, 6, 8, 9, 10, 12, 14, 16};
    std::vector<std::string> variants = {"w32", "w32ts", "twp", "tws", "trp"};
    int iters = 5, lo = 0, hi = 3;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto list = [&](const char* s) { std::vector<std::string> r; std::string cur; for (const char* c = s; ; ++c) { if (*c == ',' || !*c) { if (!cur.empty()) r.push_back(cur); cur.clear(); if (!*c) break; } else cur += *c; } return r; };
        if (a == "--widths" && i + 1 < argc) { widths.clear(); for (auto& s : list(argv[++i])) widths.push_back(std::atoi(s.c_str())); }
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--variants" && i + 1 < argc) variants = list(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) { lo = std::atoi(argv[++i]); const char* d = std::strchr(argv[i], '-'); hi = d ? std::atoi(d + 1) : lo; }
    }
    mem_guard("start", 20);
    Gguf g;
    if (!g.open(argv[1])) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    std::vector<Tensor*> pass;
    std::size_t total = 0;
    for (auto& x : g.t)
        if (in_pass(x)) { pass.push_back(&x); total += x.bytes; }
    std::printf("model %s: %zu matrices in the pass, %.3f GB\n", argv[1], pass.size(), total / 1e9);
    hipStream_t st;
    CK(hipStreamCreate(&st));
    for (Tensor* x : pass) {
        CK(hipMalloc(&x->d, x->bytes));
        CK(hipMemcpy(x->d, g.map + g.data0 + x->off, x->bytes, hipMemcpyHostToDevice));
    }
    mem_guard("weights", 12);
    std::size_t packed = 0;
    bool want_twp = false;
    for (const char* v : {"twp", "tws", "trp"}) want_twp |= std::find(variants.begin(), variants.end(), v) != variants.end();
    if (want_twp) {
        for (Tensor* x : pass) {
            if (k::native_mmvq_tw_prepare(x->type, x->d, int(x->ne0), int(x->ne1))) packed += 1;
            if ((packed & 63) == 0) mem_guard("packing", 10);
        }
        std::printf("tile-plane copies: %zu matrices, %.3f GB extra device memory\n", packed, k::native_mmvq_tw_packed_bytes() / 1e9);
    }
    mem_guard("packed", 8);
    // activations: Q8_1 of random normals, 16 columns per distinct n_in
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::map<int, void*> xq;
    for (Tensor* x : pass) {
        const int n_in = int(x->ne0);
        if (xq.count(n_in)) continue;
        std::vector<float> h(std::size_t(n_in) * 16);
        for (auto& v : h) v = nd(rng);
        float* xf;
        CK(hipMalloc(&xf, h.size() * 4));
        CK(hipMemcpy(xf, h.data(), h.size() * 4, hipMemcpyHostToDevice));
        void* q;
        CK(hipMalloc(&q, k::native_q8_1_bytes(n_in, 16)));
        k::native_quantize_q8_1(xf, q, n_in, 16, st);
        CK(hipStreamSynchronize(st));
        CK(hipFree(xf));
        xq[n_in] = q;
    }
    std::size_t ymax = 0;
    for (Tensor* x : pass) ymax = std::max<std::size_t>(ymax, std::size_t(x->ne1) * 16);
    float *y, *yr;
    CK(hipMalloc(&y, ymax * 4));
    CK(hipMalloc(&yr, ymax * 4));
    auto tw_mode_of = [](const std::string& v) { return v == "twp" ? 2 : v == "tws" ? 4 : v == "trp" ? 3 : 1; };
    auto run = [&](const std::string& v, Tensor* x, int n, const void* xin, float* out) {
        k::native_mmvq_set_w32_ts(v == "w32ts" ? 1 : 0);
        if (v == "w32" || v == "w32ts") { k::native_mmvq_tw_set(0); k::native_mmvq(x->type, x->d, xin, out, int(x->ne0), int(x->ne1), n, st); return; }
        k::native_mmvq_tw_set(tw_mode_of(v));
        if (!k::native_mmvq_tw(x->type, x->d, xin, out, int(x->ne0), int(x->ne1), n, st)) {
            k::native_mmvq_tw_set(0);
            k::native_mmvq(x->type, x->d, xin, out, int(x->ne0), int(x->ne1), n, st);
        }
    };
    if (mode == "table" || mode == "all") {
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        std::map<int, std::vector<Tensor*>> by_type;
        for (Tensor* x : pass) by_type[x->type].push_back(x);
        std::printf("\n%-6s %-4s | %-8s %8s | %9s %8s %6s | per type GB/s (%% of 256)\n", "width", "var", "", "", "pass ms", "GB/s", "%");
        for (const int n : widths)
            for (const auto& v : variants) {
                double pass_ms = 0;
                std::string per;
                for (auto& [t, xs] : by_type) {
                    std::size_t b = 0;
                    for (Tensor* x : xs) b += x->bytes;
                    std::vector<float> ms;
                    for (int it = 0; it < iters + 1; ++it) {
                        CK(hipEventRecord(e0, st));
                        for (Tensor* x : xs) run(v, x, n, xq[int(x->ne0)], y);
                        CK(hipEventRecord(e1, st));
                        CK(hipEventSynchronize(e1));
                        float m;
                        CK(hipEventElapsedTime(&m, e0, e1));
                        if (it > 0) ms.push_back(m);   // (the first pass warms the instruction cache / code objects)
                    }
                    std::sort(ms.begin(), ms.end());
                    const double med = ms[ms.size() / 2];
                    pass_ms += med;
                    char buf[96];
                    std::snprintf(buf, sizeof buf, " %s %.1f (%.1f)", type_name(t), b / med / 1e6, 100.0 * b / med / 1e6 / 256.0);
                    per += buf;
                }
                std::printf("n=%-4d %-4s | pass %8.2f ms %7.1f GB/s %5.1f%% |%s\n", n, v.c_str(), pass_ms, total / pass_ms / 1e6,
                            100.0 * total / pass_ms / 1e6 / 256.0, per.c_str());
                std::fflush(stdout);
            }
        k::native_mmvq_tw_set(-1);
        k::native_mmvq_set_w32_ts(0);
    }
    int bad = 0, checks = 0;
    if (mode == "exact" || mode == "all") {
        std::map<std::string, Tensor*> sel;
        for (Tensor* x : pass) {
            const int l = layer_of(x->name);
            const std::string key = std::string(type_name(x->type)) + " " + std::to_string(x->ne0) + "x" + std::to_string(x->ne1);
            if ((l >= lo && l <= hi) || x->name == "output.weight") sel[x->name] = x;
            else if (!sel.count("#" + key)) sel["#" + key] = x;
        }
        std::printf("\nexact: %zu matrices, widths 1..16, variants", sel.size());
        for (auto& v : variants) std::printf(" %s", v.c_str());
        std::printf(" (+ multi)\n");
        std::vector<float> hr, hy;
        std::map<std::string, int> fails;
        for (auto& [key, x] : sel) {
            const int n_in = int(x->ne0), n_out = int(x->ne1);
            const std::size_t cb = k::native_q8_1_bytes(n_in, 1);
            const uint8_t* xin = static_cast<const uint8_t*>(xq[n_in]);
            k::native_mmvq_tw_set(0);
            k::native_mmvq_set_w32_ts(0);
            for (int c = 0; c < 16; ++c) k::native_mmvq(x->type, x->d, xin + c * cb, yr + std::size_t(c) * n_out, n_in, n_out, 1, st);
            CK(hipStreamSynchronize(st));
            hr.resize(std::size_t(n_out) * 16);
            CK(hipMemcpy(hr.data(), yr, hr.size() * 4, hipMemcpyDeviceToHost));
            std::printf("  %-34s %-7s %6d x %-6d ref fnv %016llx\n", x->name.c_str(), type_name(x->type), n_in, n_out,
                        (unsigned long long) fnv(hr.data(), hr.size() * 4));
            std::vector<std::string> vs = variants;
            for (const char* m : {"multi", "multi-tws", "multi-trp", "multi-w32ts"}) vs.push_back(m);
            for (const auto& v : vs)
                for (int n = 1; n <= 16; ++n) {
                    CK(hipMemset(y, 0xff, std::size_t(n_out) * n * 4));
                    if (v.rfind("multi", 0) == 0) {   // the matrix three times in one launch (as gate + up / q, k, v would be)
                        const void* ws[3] = {x->d, x->d, x->d};
                        float* ys[3] = {y, y, y};
                        const int no[3] = {n_out, n_out, n_out};
                        bool ok;
                        if (v == "multi-w32ts") {
                            k::native_mmvq_tw_set(0);
                            k::native_mmvq_set_w32_ts(1);
                            ok = n <= 8 && k::native_mmvq_w32_multi(x->type, 1, ws, ys, no, xin, n_in, n, st);
                            k::native_mmvq_set_w32_ts(0);
                        } else {
                            k::native_mmvq_tw_set(v == "multi" ? 2 : v == "multi-tws" ? 4 : 3);
                            ok = k::native_mmvq_tw_multi(x->type, 1, ws, ys, no, xin, n_in, n, st);
                        }
                        if (!ok) { k::native_mmvq_tw_set(-1); break; }
                    } else run(v, x, n, xin, y);
                    CK(hipStreamSynchronize(st));
                    hy.resize(std::size_t(n_out) * n);
                    CK(hipMemcpy(hy.data(), y, hy.size() * 4, hipMemcpyDeviceToHost));
                    ++checks;
                    if (std::memcmp(hy.data(), hr.data(), hy.size() * 4) != 0) {
                        int col = -1, row = -1;
                        for (std::size_t i = 0; i < hy.size() && col < 0; ++i)
                            if (std::memcmp(&hy[i], &hr[i], 4) != 0) { col = int(i / n_out); row = int(i % n_out); }
                        if (fails[v + std::string(type_name(x->type))]++ < 3)
                            std::printf("    MISMATCH %s n=%d: col %d row %d: %.9g vs ref %.9g\n", v.c_str(), n, col, row,
                                        hy[std::size_t(col) * n_out + row], hr[std::size_t(col) * n_out + row]);
                        ++bad;
                    }
                }
            k::native_mmvq_tw_set(-1);
            k::native_mmvq_set_w32_ts(0);
        }
        std::printf("exact: %d of %d (matrix, variant, width) checks bitwise equal to the 1-column reference: %s\n", checks - bad,
                    checks, bad ? "FAIL" : "PASS");
    }
    std::printf("MemAvailable at end: %.1f GiB\n", mem_avail_kb() / 1048576.0);
    return bad ? 3 : 0;
}
