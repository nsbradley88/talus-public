// include/strata/prefill/tp_tune.hpp - Talus: tuning choices of the Talus prefill kernels, keyed by
// GPU architecture (hipDeviceProp_t::gcnArchName).  Only gfx1151 (Strix Halo) is measured and populated; any other
// architecture gets the conservative defaults (row `kTpTuneDefault`).
#pragma once

#include <cstring>

namespace talus::prefill {

struct TpTune {
    const char* arch;
    int fa_variant;       // tp_fa: 1 = K / V through LDS, 2 = K / V from global, 3 = Q in registers, head dim split
    int i8_variant;       // tp_i8gemm: 1 = 2 x 4 tiles per wave, 2 = the same with the next block prefetched, 3 = 4 x 2
                          // prefetched, 4 = 1 x 8 prefetched, 5 = 4 x 2, 6 = LDS-staged double-buffered 2 x 4
    int w4_bm, w4_bn;     // tp_w4a4: output tile
    int w4_min_rows;      // W4A4: the smallest chunk (rows) that may take the int4 path
    // tp_i8gemm variant per expansion mode (0 q8_0, 1 q8_0_16, 2 q8_1); 0 = i8_variant.  7 = LDS
    // kernel with the bias epilogue, 90-95 = v9 (A from global, B through LDS), 80-87 = v8 (both through LDS),
    // 100-107 = v10 (v9 with buffer loads and an unrolled K loop; 106 = 128 x 128, KS 4, A scales loaded at use)
    int i8_mode_variant[3];
    // tp_moe (grouped experts) kernel when TP_MOE_VER is unset: 0 = v6 (k_moe5, up to 4 token tiles per
    // wave), 72 = v6 with at most 3 token tiles per wave (96-row passes, ~210 VGPRs: no spills) and, for the split
    // decoders, the sign masks computed instead of looked up; 73 = 72 with an odd last token tile shared by the two
    // wave halves; 74 = 73 on 128-row passes (experiment: faster at ~128 rows per expert, slower at 64)
    int moe_ver = 0;
    // tk_prefill_moe_mmq_gu (fused gate/up + SwiGLU) runs when moe_gu != 0 and the mean rows per expert are at
    // least moe_gu_min_rows and, for the per-16-scale types (dot mode 1), at most moe_gu_mode1_max_rows
    int moe_gu = 0, moe_gu_min_rows = 0, moe_gu_mode1_max_rows = 0;
    // the native packs' experts on fp16 / bf16 WMMA (moe_w16.cu) on this GPU (pf_w16: 0 not allowed, 1 opt-in
    // with STRATA_PF_W16=1, 2 on by default with STRATA_PF_W16=0 the kill switch), persistent blocks per multiprocessor for 64-row and 128-row work items (0: a block per work item),
    // routed rows per work item for gate/up and down (64 / 128;
    // 0 = by the mean rows per expert: 128 from 64)
    int pf_w16 = 0, pf_w16_blocks = 2, pf_w16_bm_gu = 64, pf_w16_bm_d = 128, pf_w16_blocks128 = 2;
    // The tp_moe contract (STRATA_PF_W16_MOE=1: GLM / MiMo products): the fp16 kernels from this many routed
    // rows per expert on average (below this threshold the int8 path is preferred)
    int pf_w16_moe_min_rows = 32;
    // the native packs' fp16 kernels (Flash-Next) from this many routed rows per expert on average (decode /
    // MTP verify windows stay on int8)
    int pf_w16_min_rows = 16;
};

inline constexpr TpTune kTpTuneDefault{"default", 3, 6, 128, 128, 64, {0, 0, 0}, 0, 0, 0, 0, 0, 2, 0, 0};

inline constexpr TpTune kTpTune[] = {
    // gfx1151: Radeon 8060S, 20 WGPs
    // i8 per mode (talus-pf-bench i8m, all bitwise MMQ): v106 (buffer loads with scalar K offsets, no 64-bit
    // address VALU, 2-block unrolled K loop), measured faster on gfx1151 across all three expansion modes
    // moe 72 (talus-pf-bench moe, all bitwise MMQ): 96-row passes without register spills on gfx1151
    // moe_gu (fused gate/up, bitwise): enabled for >= 8 routed rows per expert (mode 1 up to 32 rows)
    // pf_w16 2 : Flash-Next's native-pack experts on the fp16 kernels by default
    // (STRATA_PF_W16=0 = kill switch); the tp_moe contract
    // (GLM / MiMo, STRATA_PF_W16_MOE=1) stays opt-in
    // moe 73 (bitwise): v72 + the odd last token tile shared by both wave halves (helps at skewed
    // routing, neutral at uniform); Q2_K keeps the v72 body
    // i8 modes 1 / 2 v117 (bitwise MMQ): v106 with the epilogue's fmas emitted diagonally (VOPD pairs) under an
    // occupancy-6 register budget; the fused swq8 up GEMM follows. Mode 0 stays v106 (its epilogue already pairs).
    {"gfx1151", 3, 6, 128, 128, 64, {106, 117, 117}, 73, 1, 8, 32, 2, 2, 0, 0, 0},
};

/// The row for `gcn` ("gfx1151" or "gfx1151:sramecc-:xnack-"), else the defaults.
inline const TpTune& tp_tune(const char* gcn) {
    if (gcn != nullptr)
        for (const TpTune& t : kTpTune) {
            const std::size_t n = std::strlen(t.arch);
            if (std::strncmp(gcn, t.arch, n) == 0 && (gcn[n] == '\0' || gcn[n] == ':')) return t;
        }
    return kTpTuneDefault;
}

/// The current device's row (cached after the first call).
const TpTune& tp_tune_current();
/// The current device is a gfx11 part the Talus prefill WMMA kernels are compiled for.
bool tp_gfx11_wmma();

}  // namespace talus::prefill
