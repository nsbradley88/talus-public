#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace talus {
namespace glm_spike {

enum GlmQuantType : int {
    GLM_QTYPE_IQ3_S = 21,
    GLM_QTYPE_IQ2_S = 22,
};

// Quantize contiguous FP32 vector [n_elements] into block_q8_1 [n_elements / 32].
hipError_t glm_quantize_q8_1(
    const float* x_f32,
    void* x_q8_1,
    int n_elements,
    hipStream_t stream = nullptr);

// GLM-5.3-Flash MoE expert-ID matrix-vector multiply (IQ2_S / IQ3_S).
// W: [K, N, E] quantized in GGML block layout (row_bytes = (K / 256) * block_bytes, expert_stride = N * row_bytes)
// x_q8_1: [K/32, x_group, T] quantized Q8_1 blocks (x_group = top_k for ffn_down or 8-stream gate/up, or 1 for shared x)
// ids: [top_k, T] int32 expert indices in [0, E) (with stride ids_s1 ints between tokens)
// y: [N, top_k, T] FP32 output
hipError_t glm_mmvq_ids(
    int qtype,
    const void* W,
    const void* x_q8_1,
    const int32_t* ids,
    float* y,
    int N,
    int K,
    int n_expert_used,
    int n_tokens,
    int x_group,
    int ids_s1,
    hipStream_t stream = nullptr);

// gate + up pair (same type / shape / ids / x) in one launch, rows bitwise those of glm_mmvq_ids.
// hipErrorNotSupported (nothing launched) when the selected config is not a v1 kernel: call glm_mmvq_ids twice.
hipError_t glm_mmvq_ids_pair(
    int qtype, const void* W, const void* W2, const void* x_q8_1, const int32_t* ids, float* y, float* y2,
    int N, int K, int n_expert_used, int n_tokens, int x_group, int ids_s1, hipStream_t stream = nullptr);

// GLM HIP: explicit kernel config for the expert mat-vec (sweeps / A-B). variant 0 = WP-1.7 spike kernels,
// 1 = v1 (LDS grid, v_perm signs, R rows per wave, NW waves per WG, min WPE waves per SIMD). glm_mmvq_ids()
// takes the per-arch table (gfx1151 tuned) or GLM_MMVQ_CFG="variant:R:NW:WPE[:pers]".
struct GlmMmvqCfg { int variant, r, nw, wpe, pers; };   // pers > 0: persistent grid of pers WGs per CU
hipError_t glm_mmvq_ids_cfg(
    int qtype,
    const void* W,
    const void* x_q8_1,
    const int32_t* ids,
    float* y,
    int N,
    int K,
    int n_expert_used,
    int n_tokens,
    int x_group,
    int ids_s1,
    const GlmMmvqCfg* cfg,
    hipStream_t stream = nullptr);

// Convenience end-to-end wrapper: quantize x_f32 -> x_q8_1 and run glm_mmvq_ids.
hipError_t glm_mmvq_ids_f32(
    int qtype,
    const void* W,
    const float* x_f32,
    void* x_q8_1_scratch,
    const int32_t* ids,
    float* y,
    int N,
    int K,
    int n_expert_used,
    int n_tokens,
    int x_group,
    int ids_s1,
    hipStream_t stream = nullptr);

// GLM-5.3-Flash KDA recurrent decode step with fused prologue, state read by ids,
// and in-place rollback snapshot writeback (K=1 for T=1, K=3 for T=3).
// Shapes:
//   q, k:  [S_v, H, T] FP32 (S_v = 128, H = 64)
//   v:     [S_v, H, T] FP32
//   g:     [S_v, H, T] FP32 (raw gate before prologue)
//   beta:  [1,   H, T] FP32 (raw beta before sigmoid)
//   state: [S_v, S_v, H, n_seqs] FP32 (in-place read via ids[0..n_B-1] and write to ids[0..n_B*K-1])
//   ids:   [n_B * K] int32 slot indices into state
//   dtb:   [S_v, H] FP32 (dt_bias for prologue)
//   a:     [H] FP32 (A_log/A scale for prologue)
//   dst:   [S_v, H, T] FP32 attention output
hipError_t glm_kda_step(
    const float* q,
    const float* k,
    const float* v,
    const float* g,
    const float* beta,
    float* state,
    const int32_t* ids,
    const float* dtb,
    const float* a,
    float* dst,
    int H,
    int S_v,
    int T,
    int K_snap,
    int n_B,
    float scale,
    bool prologue,
    float pro_lb,
    float pro_l2_eps,
    int cols_per_wave = 0,   // GLM HIP: <= 0 = per-arch auto; 1/2/4 = v0; 100*waves+cols = v1 (shared prologue)
    hipStream_t stream = nullptr);

// GLM-5.3-Flash DSA decode lightning indexer.
// Shapes:
//   q:   [D, H, T, S] FP32 (D = 128, H = 32)
//   k:   [D, n_kv, 1, S] FP32 (k_is_f16 = false) or FP16 (k_is_f16 = true)
//   w:   [H, T, S] FP32
//   dst: [n_kv, T, S] FP32
hipError_t glm_dsa_lightning_indexer(
    const float* q,
    const void* k,
    const float* w,
    float* dst,
    int D,
    int H,
    int n_kv,
    int n_tokens,
    int n_streams,
    bool k_is_f16,
    hipStream_t stream = nullptr);

// GLM HIP: explicit indexer config (variant 0 = WP-1.7 kernel; 1 = heads split over nwv waves, tf tokens per WG).
// glm_dsa_lightning_indexer() takes the per-arch table or GLM_LI_CFG="variant:nwv:tf".
struct GlmLiCfg { int variant, nwv, tf; };
hipError_t glm_dsa_lightning_indexer_cfg(
    const float* q,
    const void* k,
    const float* w,
    float* dst,
    int D,
    int H,
    int n_kv,
    int n_tokens,
    int n_streams,
    bool k_is_f16,
    const GlmLiCfg* cfg,
    hipStream_t stream = nullptr);

// GLM-5.3-Flash DSA sparse MLA decode attention (compaction + sparse decode FA + split-K reduce).
// Shapes:
//   q:         [D, N, H] FP32 with strides q_nb1 (token stride in floats), q_nb2 (head stride in floats)
//   k:         [D, KV, 1] FP16 with row stride k_stride (in halfs)
//   v:         [D, KV, 1] FP16 with row stride v_stride (in halfs); if v_same == true, v aliases k
//   mask:      [KV, N] FP16 with row stride m_stride (in halfs)
//   idx_buf:   [n_kv_max * N] int32 scratch buffer
//   split_buf: [(D * H + 2 * H) * k_num * N] FP32 scratch buffer
//   dst:       [D, H, N] FP32 contiguous output
hipError_t glm_dsa_sparse_fa(
    const float* q,
    const __half* k,
    const __half* v,
    const __half* mask,
    int32_t* idx_buf,
    float* split_buf,
    float* dst,
    int D,
    int H,
    int N,
    int KV,
    int n_kv_max,
    uint32_t q_nb1,
    uint32_t q_nb2,
    uint32_t k_stride,
    uint32_t v_stride,
    uint32_t m_stride,
    float scale,
    bool v_same = true,
    int force_k_num = 0,
    hipStream_t stream = nullptr);

// GLM HIP bench hook: sparse FA variant (0 = WP-1.7 three launches, 1 = compaction fused into the decode kernel,
// -1 = default / GLM_FA_VARIANT).
void glm_set_fa_variant(int v);

// Talus GLM port: sparse decode attention from a row-index list (idx [N][n_kv_max] ascending, -1 padded);
// zero_mask = f16 zeros covering every row index (read at the selected rows). split_buf: glm_dsa_sparse_fa_split_bytes.
hipError_t glm_dsa_sparse_fa_idx(
    const float* q, const __half* kv, const __half* zero_mask, const int32_t* idx, float* split_buf, float* dst,
    int D, int H, int N, int n_kv_max, uint32_t q_nb1, uint32_t q_nb2, uint32_t k_stride, float scale,
    hipStream_t stream = nullptr);
size_t glm_dsa_sparse_fa_split_bytes(int D, int H, int N, int n_kv_max);

// Talus GLM port: grouped expert mat-vec (prefill): q8_1 rows sorted by expert (bounds [n_experts + 1]), output row r
// of the sorted order goes to y[ids_dst[r] * ld_y + n].
hipError_t glm_mmvq_grouped(int qtype, const void* W, size_t expert_bytes, int n_experts, const int32_t* bounds,
                            const void* x_q8_1, const int32_t* ids_dst, float* y, int N, int K, int ld_y,
                            hipStream_t stream = nullptr);

void glm_set_grouped_ver(int v);   // test hook: 1 = dp4a v1, 2 = dp4a v2 (LDS x), 3 = WMMA (gfx11)

} // namespace glm_spike
} // namespace talus
