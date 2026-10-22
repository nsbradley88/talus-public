// src/program/qwen4exp_r2.hpp - Launch helpers that the Rust plane's wrappers (qwen4exp_p2/p3/p4.cpp) used to
// reach in src/core/layer.cpp, as thin launch wrappers of their own (src/program/qwen4exp_r2.cpp).  Each is the
// native-pack branch of the layer.cpp function it replaces, kernel for kernel; a branch only a non-native (S-form)
// pack reaches still calls the layer.cpp original.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>
#include <string>

namespace strata::core::qwen4exp_r2 {

/// generate.cpp sets it next to layer_set_native_bf16 (the router reads FP32 activations; every native pack).
void set_native_bf16(bool enabled);
bool native_bf16();

/// moe_route without a doorbell (every Rust-plane caller passes none): the router projection + the top-k.
bool moe_route(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, void* stream, std::string& err);
/// moe_combine_parts: sum of w[i] * parts[i] plus the shared output.
bool moe_combine_parts(const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b, const float* parts,
                       float* out, void* stream, std::string& err);
/// lm_head_mix: the output hyper-connection read into bb.mixed.
bool lm_head_mix(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, void* stream,
                 std::string& err);
/// lm_head: native packs never take it (their head is NativeHead); the S-form head stays layer.cpp's.
bool lm_head(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, float* logits, void* stream,
             std::string& err);
/// qsa_attn_pools / qsa_kv_resolve
strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState& st);
void qsa_kv_resolve(const QsaState& st, const ModelGeometry& g, const int32_t* ids, const int32_t* steps, int64_t n_q,
                    int64_t cap, void* stream);
/// embed_row: one token's embedding row (the GGUF-form NativeEmbed table; the S-form pack table stays layer.cpp's).
bool embed_row(const WeightTable& tables, const ModelGeometry& g, int64_t token, float* out_dev, void* stream,
               std::string& err);

}  // namespace strata::core::qwen4exp_r2
