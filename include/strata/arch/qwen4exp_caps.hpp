// include/strata/arch/qwen4exp_caps.hpp - the shapes this build's qwen4exp kernels are COMPILED for.
//
// The model's description comes from its GGUF (strata/core/model_desc.hpp: `describe_model`).  Most of the
// engine reads it at run time.  Some kernels cannot: they are templated on a head size, size shared memory for
// one residual width, or fuse a 512x10 router.  Their constraint is written down HERE, once, and the arch's
// capability check (`qwen4exp_kernel_check` in model_desc.hpp) compares the description against it at load.  A
// model that differs is refused with "the kernels require X", not mis-run.
//
// Rules:
//   * nothing reads these in place of the description where a run-time value works;
//   * every value here is checked against the description (model_desc.hpp), so adding one means adding its check;
//   * the per-kernel `constexpr`s that predate this header (qsa_decode_attn.cu HD/G, fused_gr.cu, native_router.cu,
//     ...) are covered by the same check; folding them in here is a follow-up, not a semantic change.
//
// Header-only and std-only, so CPU tools (strata-gguf, strata-plan) can include it.
#pragma once

#include <cstdint>

namespace strata::arch::qwen4exp {

inline constexpr const char* kArch = "qwen4exp";

// residual and hyper-connections (fused_gr.cu, ple.cu, prefill)
inline constexpr int64_t kEmbd = 2560;
inline constexpr int64_t kHc = 4;
inline constexpr int64_t kHcLr = 320;

// Gated DeltaNet (fused_gdn.cu, native_gdn*.cu, verify_kernels.cu: S = 128; prefill's C/ZV/HV)
inline constexpr int64_t kSsmState = 128;
inline constexpr int64_t kSsmKHeads = 16;
inline constexpr int64_t kSsmVHeads = 48;
inline constexpr int64_t kSsmConv = 4;
inline constexpr int64_t kSsmValueDim = kSsmState * kSsmVHeads;                         // 6144
inline constexpr int64_t kSsmConvChannels = 2 * kSsmState * kSsmKHeads + kSsmValueDim;  // 10240

// QSA attention and its indexer (qsa_decode_attn.cu / qsa_prompt_attn.cu: HD 256, G 12; qsa_select.cu: 4 x 128,
// block 4; kv_q4.cu: head_dim 256 exactly)
inline constexpr int64_t kHead = 24;
inline constexpr int64_t kHeadKv = 2;
inline constexpr int64_t kHeadDim = 256;
inline constexpr int64_t kRot = 64;
inline constexpr int64_t kIdxHeads = 4;
inline constexpr int64_t kIdxDim = 128;
inline constexpr int64_t kIdxBlock = 4;
inline constexpr int64_t kIdxTopK = 2048;
inline constexpr float kRmsEps = 1e-6f;   // every RMSNorm / L2 norm kernel takes this constant

// MoE (native_router.cu 512 -> 10, prefill K = 10, moe_mmq.cu 1280/640, the canonical Q2_0 CPU kernels)
inline constexpr int64_t kFf = 640;
inline constexpr int64_t kExpertUsed = 10;
inline constexpr int64_t kExpertMax = 512;   // the prefill path's per-expert arrays; pruned variants have fewer

// depth: the only one the qwen4exp path (PLE at layer 1, expert profiles, MTP) has been run at
inline constexpr int64_t kLayers = 48;
inline constexpr int64_t kPleLayer = 1;
inline constexpr int64_t kPleConv = 4;   // the PLE block's causal conv (native_ple_postops.cu)

}  // namespace strata::arch::qwen4exp
