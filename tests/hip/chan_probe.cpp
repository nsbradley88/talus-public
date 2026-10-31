// tests/hip/chan_probe.cpp - DRAM channel-camping probe: read bandwidth of the layouts where many blocks
// read segments a power of two apart in lockstep, against the same reads with the segments padded (a bitwise-safe
// layout change: only addresses move).  No engine code; synthetic buffers.  Idea from libstrix (asf0/libstrix, MIT:
// non-power-of-two split strides), no code from it.
//   talus-chan-probe [REPS] [CASES]   CASES: comma list of case names (default all)
// Each case: NSEG blocks, block b reads SEG bytes from  (b / INNER) * (OUTER + pad) + (b % INNER) * (STRIDE + pad),
// front to back, STEP bytes per block per iteration (16-byte loads), all blocks resident at once (lockstep).  Every
// launch reads a different region of a 2 GiB buffer (regions 2 MiB aligned), so the data comes from DRAM, not MALL/L2.
// Pads: 0 (today), 256, 512, 1024, 4352 (4 KiB + 256) bytes.  Verdict per case: gain = best padded / pad 0 - 1;
// CAMPING when gain > 5% and > 3x the run-to-run spread -> propose that padding; else none.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <int TH, int LPI>
__global__ void __launch_bounds__(TH) k_read(const uint4* __restrict__ base, size_t inner, size_t stride, size_t outer,
                                             size_t seg, uint32_t* out) {
    const size_t b = blockIdx.x;
    const uint4* p = (const uint4*) ((const char*) base + (b / inner) * outer + (b % inner) * stride);
    const size_t n = seg / 16;
    uint32_t acc = 0;
    for (size_t i = threadIdx.x; i < n; i += (size_t) TH * LPI) {
        uint4 v[LPI];
#pragma unroll
        for (int j = 0; j < LPI; ++j) v[j] = i + (size_t) j * TH < n ? p[i + (size_t) j * TH] : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int j = 0; j < LPI; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
    }
    if (acc == 0x9e3779b9u) out[b] = acc;   // (never true for the fill: keeps the loads)
}
struct Case {
    const char* name;
    const char* what;
    size_t seg, stride, inner, outer;   // outer 0: one group (inner = nseg)
    int nseg, lpi;                      // loads per thread per iteration (step = 128 threads x 16 B x lpi)
};
// 27B: HKV 4, head dim 256 f16 (512 B per cell), 512-cell split chunks = 256 KiB; kvh bases ctx x 512 B apart
// (ctx 65536 -> 32 MiB).  At 32K depth n_chunk ~ 64 + the 128-cell chunks below 4K; modelled as 64 x 4.
const Case kCases[] = {
    {"attn_k32k", "k_attn_wmma K/V^T: 64 chunks x 4 kvh, 256 KiB segments, kvh 32 MiB apart (32K deep, ctx 64K)",
     256 << 10, 256 << 10, 64, 32u << 20, 256, 4},
    {"attn_k64k", "k_attn_wmma K/V^T at 64K: 128 chunks x 4 kvh, 256 KiB, kvh 32 MiB apart", 256 << 10, 256 << 10, 128,
     32u << 20, 512, 4},
    {"attn_k8k", "k_attn_wmma at 8K: 16 chunks x 4 kvh (8 KiB per iteration)", 256 << 10, 256 << 10, 16, 32u << 20, 64, 4},
    {"kv8_32k", "int8 KV (27B kq / FN kv_q8): 256 B per cell, 512-cell segments 128 KiB, 64 x 4 kvh, kvh 16 MiB apart",
     128 << 10, 128 << 10, 64, 16u << 20, 256, 4},
    {"router12k", "f32 router rows 12 KiB (Laguna), 256 rows, a block per row", 12 << 10, 12 << 10, 256, 0, 256, 1},
    {"router16k", "f32 router rows 16 KiB (MiMo), 256 rows", 16 << 10, 16 << 10, 256, 0, 256, 1},
    {"router24k", "f32 router rows 24 KiB (GLM), 256 rows", 24 << 10, 24 << 10, 256, 0, 256, 1},
    {"pcache80k", "int8 prefill cache row tiles K x 16 B (K 5120: 80 KiB), 320 tiles", 80 << 10, 80 << 10, 320, 0, 320, 1},
    {"pcache272k", "int8 prefill cache row tiles (K 17408: 272 KiB), 320 tiles", 272 << 10, 272 << 10, 320, 0, 320, 1},
    {"control_odd", "control: 256 segments of 250 KiB at a 250 KiB stride (not a power of two)", 250 << 10, 250 << 10, 256,
     0, 256, 4},
};
constexpr size_t kBuf = (size_t) 2 << 30, kAlign = 2u << 20;
}   // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 40;
    const std::string want = argc > 2 ? argv[2] : "";
    char* buf = nullptr;
    uint32_t* out = nullptr;
    ck(cudaMalloc(&buf, kBuf), "buf");
    ck(cudaMalloc(&out, 4096 * 4), "out");
    ck(cudaMemset(buf, 0x5a, kBuf), "fill");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "ev");
    ck(cudaEventCreate(&e1), "ev");
    const size_t pads[] = {0, 256, 512, 1024, 4352};
    std::printf("channel-camping probe: %d launches per point (each on a cold 2 MiB-aligned region), GB/s best / spread "
                "of 3 rounds\n",
                reps);
    int camping = 0;
    for (const Case& c : kCases) {
        if (!want.empty() && ("," + want + ",").find("," + std::string(c.name) + ",") == std::string::npos) continue;
        std::printf("\n%s: %s\n", c.name, c.what);
        double base_gbs = 0, base_spread = 0, best_gain = -1;
        size_t best_pad = 0;
        for (size_t pad : pads) {
            const size_t stride = c.stride + pad, outer = c.outer ? c.outer + pad * (size_t) c.inner : 0;
            const size_t inner = c.outer ? c.inner : (size_t) c.nseg;
            const size_t groups = (c.nseg + inner - 1) / inner;
            const size_t foot = (groups - 1) * outer + (inner - 1) * stride + c.seg;
            const size_t region = (foot + kAlign - 1) / kAlign * kAlign;
            const int nreg = (int) std::max<size_t>(1, kBuf / region);
            if (nreg < 4) { std::printf("  pad %5zu: footprint %zu MiB too big\n", pad, foot >> 20); continue; }
            double r[3];
            int launch = 0;
            for (int round = 0; round < 3; ++round) {
                auto go = [&] {
                    const uint4* b = (const uint4*) (buf + (size_t) (launch++ % nreg) * region);
                    if (c.lpi == 4) k_read<128, 4><<<c.nseg, 128>>>(b, inner, stride, outer ? outer : stride * inner, c.seg, out);
                    else k_read<128, 1><<<c.nseg, 128>>>(b, inner, stride, outer ? outer : stride * inner, c.seg, out);
                };
                for (int i = 0; i < 4; ++i) go();
                ck(cudaEventRecord(e0), "rec");
                for (int i = 0; i < reps; ++i) go();
                ck(cudaEventRecord(e1), "rec");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, e0, e1), "el");
                r[round] = (double) c.seg * c.nseg * reps / (ms * 1e-3) / 1e9;
            }
            const double best = std::max({r[0], r[1], r[2]}), worst = std::min({r[0], r[1], r[2]});
            const double spread = (best - worst) / best;
            if (pad == 0) { base_gbs = best; base_spread = spread; }
            const double gain = pad == 0 ? 0 : best / base_gbs - 1;
            if (pad && gain > best_gain) { best_gain = gain; best_pad = pad; }
            std::printf("  pad %5zu B: %7.1f GB/s (spread %.1f%%)%s\n", pad, best, spread * 100,
                        pad ? (std::string("  ") + (gain >= 0 ? "+" : "") + std::to_string(gain * 100).substr(0, 5) + "%").c_str() : "");
            std::fflush(stdout);
        }
        const bool hit = best_gain > 0.05 && best_gain > 3 * base_spread;
        camping += hit && std::string(c.name).rfind("control", 0) != 0;
        std::printf("  VERDICT %s: %s (best pad %zu B: %+.1f%%, pad-0 spread %.1f%%)\n", c.name,
                    hit ? "CAMPING -> propose this padding (bitwise: addresses only)" : "no camping (< 5% or within noise)",
                    best_pad, best_gain * 100, base_spread * 100);
    }
    std::printf("\nchan-probe: %d layout(s) over the 5%% bar\n", camping);
    return 0;
}
