// include/strata/prefill/tp_moe.hpp - grouped expert GEMM for MoE prompts (src/prefill/tp_moe.cu).
//
// One launch over n_experts experts whose GGUF weights lie expert_bytes apart from w: for expert e, the activation rows
// [bounds[e], bounds[e+1]) of xq (MMQ's q8_1 rows for this weight type, strata::prefill::mmq::quantize; rows sorted by
// expert) times its [n][k] matrix into dst rows ids[row] (null: row), ld_dst floats apart.  The same contract and the
// same output bits as strata::prefill::mmq::Context::run (llama.cpp's grouped MMQ), at the int8 WMMA rate of
// tp_i8gemm instead of MMQ's 64/128-row token tiles.  Types: tp_i8gemm's (i8x_mode >= 0): IQ4_XS, IQ4_NL, IQ3_S,
// IQ3_XXS, IQ2_XXS, Q8_0, Q5_0, IQ2_XS, IQ2_S, Q3_K, Q4_K, Q5_K; n % 128 == 0, k % 256 == 0.  gfx11 only.
#pragma once

#include <cstddef>
#include <cstdint>

namespace talus::prefill {

struct MoeProduct {
    const void* w = nullptr;
    int type = -1;
    size_t expert_bytes = 0;
    int n = 0, k = 0, n_experts = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;   ///< device, n_experts + 1 absolute row offsets into xq
    const int32_t* ids = nullptr;      ///< device, dst row of each xq row (null: identity)
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
    /// non-null = fused gate/up: `w` the gate experts, `w_up` the up experts (same type, layout and expert_bytes),
    /// dst[row][f] = g / (1 + exp(-g)) * u (with g = min(g, L), u = clamp(u, -L, L) when swiglu_limit L > 0): the
    /// expression of glm's k_swiglu_clamp / laguna's silu_mul_kernel.  Split-decoder types only (moe_gu_supported).
    const void* w_up = nullptr;
    float swiglu_limit = 0.f;
    /// non-null (with w_up) = write the SwiGLU rows as the down projection's MMQ q8_1 activations (D4 layout,
    /// q8_rows rows) instead of FP32 dst - bitwise llama.cpp's quantize_mmq_q8_1 of those rows (moe_gu_q8_supported)
    void* q8_out = nullptr;
    int64_t q8_rows = 0;
};

bool moe_supported(int ggml_type, int n, int k);
/// dense (one-expert) Q6_K on tp_moe (bitwise MMQ); false = not covered, use MMQ
bool moe_gemm_q6k_dense(const MoeProduct& p, void* stream);
/// the dot family of the type (0: one scale per 32 values, 1: per 16, 2: scale + min; -1 unsupported).
int moe_dot_mode(int ggml_type);
/// the fused gate/up product (MoeProduct::w_up) covers this type and shape (n = features, k = input).
bool moe_gu_supported(int ggml_type, int n, int k);
/// the fused gate/up can write the down projection's q8_1 rows directly (down type with the D4 activation
/// layout, gate/up type on the split decoders, the balanced v73 kernel)
bool moe_gu_q8_supported(int gu_type, int down_type, int n, int k);
/// talus-pf-bench only: force kernel version v (TP_MOE_VER values; -1 = the environment / default).
void moe_set_version(int v);
/// false: unsupported (nothing launched; use mmq::Context::run).
bool moe_gemm(const MoeProduct& p, void* stream);
/// IQ1_M experts on PLAIN q8_1 rows (block_q8_1, 36 bytes per 32 values: the activations of the dp4a grouped kernel
/// strata::kernels::mimo_prefill_moe_mmq_q8), int8 WMMA, output bits identical to that dp4a kernel (its per-lane block
/// order and butterfly reduction reproduced).  Covered: n % 64 == 0, k == 4096, no w_up.  TP_MOE_IQ1M_X=0: off.
bool moe_iq1m_q8_1_supported(int ggml_type, int n, int k);
bool moe_gemm_iq1m_q8_1(const MoeProduct& p, void* stream);
/// talus-pf-bench only: k_moe1m routed rows per block (64 / 128; 0 = TP_MOE_IQ1M_TOK / default)
void moe_iq1m_set_tok(int t);

}  // namespace talus::prefill

extern "C" {
/// C ABI of moe_gemm.  0 on success, -1 when unsupported (nothing launched).
int talus_pf_moe_gemm(const void* w, int ggml_type, size_t expert_bytes, int n, int k, int n_experts, const void* xq,
                      const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                      int64_t ld_dst, void* stream) noexcept;
}
