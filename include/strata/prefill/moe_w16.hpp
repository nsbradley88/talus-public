// include/strata/prefill/moe_w16.hpp - the native packs' prompt experts with bf16 WMMA
// (src/prefill/moe_w16.cu).  Same contract, buffers and layouts as fused::experts_native (moe_fused_iq.hpp): the int8
// activation rows in, H as int8 per 32 values, the down rows out in FP32 - but each product is a bf16 GEMM with the
// weight scale folded into the decode (w = bf16(d * code)), the activations bf16(d * q), and fp32 accumulation over the
// whole K, instead of the int8 dot with a float epilogue per 32 values.  The numbers differ from the int8 path's by
// the bf16 rounding of w and x (KLD-gated; default off).
//
// On by default where tp_tune.hpp's row says so (gfx1151: pf_w16 2) and the formats are covered (gate/up IQ3_S, down
// IQ4_NL: Flash-Next's UD-IQ4_XS pack); STRATA_PF_W16=0 is the kill switch (=1 turns it on where the row only allows it).
#pragma once

#include "strata/prefill/moe_fused_iq.hpp"

#include <cstdint>

namespace strata::prefill::fused {

/// STRATA_PF_W16 (read once; unset = the tune row's default), the GPU's tune row allows it, and these formats are covered.
bool w16_enabled(int gu_type, int d_type);

/// experts_native's products on the bf16 kernels; false (nothing launched) when not covered.  phase: 0 both, 1 gate/up
/// (+ SwiGLU + H) only, 2 down only (talus-pf-bench).  force: ignore STRATA_PF_W16 (talus-pf-bench).
bool experts_native_w16(const Batch& b, const NativeGeom& g, int n_expert, int64_t n, const void* scratch,
                        const void* xa, const int32_t* src, void* ha, float* dm, void* stream, int phase = 0,
                        bool force = false);

/// talus-pf-bench: ggml's iq3s_grid (512 words) on the host.
void w16_host_iq3s_grid(uint32_t* out);

}  // namespace strata::prefill::fused

namespace talus::prefill {
struct MoeProduct;
/// STRATA_PF_W16_MOE=1 (read once), the tune row allows it, and the type has a bf16 kernel (IQ2_S, IQ3_S, IQ4_NL).
bool moe_w16_enabled(int ggml_type);
/// tp_moe's grouped product (moe_gemm's contract: MMQ-layout q8_1
/// activations, FP32 dst rows; w_up = fused gate/up + clamped SwiGLU) on the fp16 / bf16 WMMA kernels of moe_w16.cu;
/// false (nothing launched) when off or not covered.  force: ignore STRATA_PF_W16_MOE (talus-pf-bench).
bool moe_gemm_w16(const MoeProduct& p, void* stream, bool force = false);
}  // namespace talus::prefill
