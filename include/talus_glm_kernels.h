/*
 * talus_glm_kernels.h - GLM-5.3-Flash (glm5-next) kernels for the Talus Rust host (C ABI, part of libtalus_kernels).
 *
 * Plain device pointers + hipStream_t (passed as void*), return 0 on success or a hipError_t code. No tk_ctx.
 * Activations are row-major [T][dim] (T = tokens of the window). "ctl" is a device int32 control block:
 *   ctl[0] = pos0, the position of token 0 of the window (single sequence; cache rows are positions).
 * Every entry point is graph-capturable: device pointers fixed, host scalars baked at capture time.
 *
 * (c) nsbradley88; ggml-derived formats (MIT, ggml authors). See THIRD_PARTY_NOTICES.md.
 */
#ifndef TALUS_GLM_KERNELS_H
#define TALUS_GLM_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TKG_ABI_VERSION 7

int tkg_abi_version(void);

/* ---- mHC (hyper-connections, 4 streams) ---- */
/* xhc[t][s][i] = x[t][i] for s < 4 */
int tkg_hc_repeat(const float* x, float* xhc, int T, int n_embd, void* stream);
/* split-K mixes: parts[t][p][0..M-1] = w(Q8_0 [M rows x K]) . xhc[t][p*K/nsplit ..], parts[t][p][M] = sum of squares of
   the slice; parts row stride 32 floats (M <= 31) */
int tkg_hc_mix(const void* w_q8_0, const float* xhc, float* parts, int T, int K, int M, int nsplit, void* stream);
/* pre-mix: sums the slices, r = rsqrt(sumsq/K + rms_eps), pre/post/comb (Sinkhorn), y = rmsnorm(sum_h pre_h x_h) * norm_w.
   pc[t][0..3] = post, pc[t][4..19] = comb[dst + 4 src]; pc row stride 32. */
int tkg_hc_pre(const float* xhc, const float* parts, int nsplit, const float* scale, const float* base,
               const float* norm_w, float* y, float* pc, int T, int n_embd, float rms_eps, float norm_eps,
               float hc_eps, int sinkhorn_iters, void* yq /* optional q8_1 copy of y */, void* stream);
/* post: out[t][d][i] = xo[t][i] * post[d] + sum_s comb[d + 4 s] * res[t][s][i]; out may alias res */
int tkg_hc_post(const float* xo, const float* res, const float* pc, float* out, int T, int n_embd, void* stream);
/* head: h[t][i] = rmsnorm(mean_s xhc[t][s][i]) * norm_w */
int tkg_hc_head(const float* xhc, const float* norm_w, float* h, int T, int n_embd, float eps, void* stream);

/* fused v2: post (first = 0), the embedding repeated into 4 streams (first = 1) or mixes only (first = 2) + partial mixes for the next pre (w_next may be
   NULL) + Gram partials; parts: [T][8][64]. pre2: Sinkhorn + collapse + rmsnorm (+ q8_1 copy), parallel over 8 slices. */
int tkg_hc_post_mix(const float* xo, float* xhc, const float* pc, const void* w_next, float* parts, int T, int n_embd,
                    int first, void* stream);
int tkg_hc_post_mix_moe(const float* y, const float* w, const float* shexp, float* xhc, const float* pc,
                        const void* w_next, float* parts, int T, int k, int n_embd, void* stream);
int tkg_hc_pre2(const float* xhc, const float* parts, const float* scale, const float* base, const float* norm_w,
                float* y, float* pc, int T, int n_embd, float rms_eps, float norm_eps, float hc_eps, int sinkhorn_iters,
                void* yq, void* stream);
/* rb-num GLM_HC_BF16 (opt-in, prompt windows): the same three launches + the head with the 4 streams stored in 16 bits
   (fmt 1 = fp16, saturating at +-65504; 2 = bf16), fp32 math, rounding once at the store (the Gram and the next mixes use
   the rounded values). xhc: [T][4][n_embd] 16-bit elements. */
int tkg_hc_post_mix16(const float* xo, void* xhc, const float* pc, const void* w_next, float* parts, int T, int n_embd,
                      int first, int fmt, void* stream);
int tkg_hc_post_mix_moe16(const float* y, const float* w, const float* shexp, void* xhc, const float* pc,
                          const void* w_next, float* parts, int T, int k, int n_embd, int fmt, void* stream);
int tkg_hc_pre2_16(const void* xhc, const float* parts, const float* scale, const float* base, const float* norm_w,
                   float* y, float* pc, int T, int n_embd, float rms_eps, float norm_eps, float hc_eps, int sinkhorn_iters,
                   void* yq, int fmt, void* stream);
int tkg_hc_head16(const void* xhc, const float* norm_w, float* h, int T, int n_embd, float eps, int fmt, void* stream);

/* ---- generic ---- */
/* y[r][i] = x[r][i] * rsqrt(mean(x[r]^2) + eps) * w[i]  (strides in floats; w may be NULL) */
int tkg_rms_norm(const float* x, const float* w, float* y, int rows, int cols, int x_stride, int y_stride,
                 float eps, void* stream);
/* y[i] += x[i] */
int tkg_add(float* y, const float* x, int n, void* stream);
/* out = silu(min(g, limit)) * clamp(u, -limit, limit); limit <= 0: plain silu(g) * u */
int tkg_swiglu_clamp(const float* g, const float* u, float* out, int n, float limit, void* stream);
/* token embedding rows (ggml type 12 = Q4_K, 0 = F32) -> out[t][n_embd]; tok = device int32 [T] */
int tkg_embed(int ggml_type, const void* table, const int32_t* tok, float* out, int T, int n_embd, void* stream);

/* ---- KDA (linear attention) ---- */
/* causal conv1d (kernel 4) + SiLU over q/k/v [T][C], conv state [slot][3*C][3] (oldest first, channel groups q,k,v).
   Reads slot ids[0]; token t >= T-K writes its post-token state to slot ids[t - (T-K)]. w*: [C][4]. */
int tkg_kda_conv(const float* q, const float* k, const float* v, float* conv_state, const int32_t* ids, int K,
                 const float* wq, const float* wk, const float* wv, float* oq, float* ok, float* ov, int T, int C,
                 void* stream);
/* out[t][h][d] = rmsnorm_d(o[t][h]) * w[d] * sigmoid(g2[t][h][d]) */
int tkg_kda_gate(const float* o, const float* g2, const float* w, float* out, int T, int H, int D, float eps,
                 void* stream);

/* ---- MLA + DSA ---- */
/* cache[(pos0 + t)][d] = f16(rmsnorm(kv[t]) * w) */
int tkg_mla_store(const float* kv, const float* w, float eps, void* cache_f16, const int32_t* ctl, int T, int D,
                  void* stream);
/* the key|gate rows live in a ring of TKG_KG_RING rows (row = position & (TKG_KG_RING - 1)): pooling reads only the
   last kpool rows, and a window never exceeds the ring */
#define TKG_KG_RING 4096
/* key = LayerNorm(ik) * ln_w + ln_b (eps), keygate[(pos0 + t) % ring] = f16(key[128] | ig[128]) */
int tkg_idx_store(const float* ik, const float* ln_w, const float* ln_b, float ln_eps, const float* ig,
                  void* keygate_f16, const int32_t* ctl, int T, void* stream);
/* for each window token at p with (p + 1) % kpool == 0: pooled[p / kpool][d] =
   f16(sum_k softmax_k(gate[k][d] + ape[d + 128 k]) key[k][d]) over rows p-kpool+1..p */
int tkg_idx_pool(const void* keygate_f16, const float* ape, void* pooled_f16, const int32_t* ctl, int T, int kpool,
                 void* stream);
/* selection: token t at p = pos0 + t sees n_vis = (p + 1) / kpool pools; keeps the top min(n_vis, top_pools) pool
   scores (scores[t * score_stride + j]), expands each to its kpool rows, adds the (p + 1) % kpool tail rows, writes
   the rows in ascending order to idx[t * n_kv_max ..], -1 padded */
int tkg_dsa_select(const float* scores, int score_stride, const int32_t* ctl, int T, int kpool, int top_pools,
                   int n_kv_max, int32_t* idx, void* stream);

/* ---- MoE ---- */
/* sigmoid gating: sel = sigmoid(l) + bias, top-k of sel; w = sigmoid(l)[sel] / sum * scale */
int tkg_router(const float* logits, const float* bias, int32_t* ids, float* w, int T, int n_exp, int k,
               float scale, void* stream);
/* out[t][i] = sum_j w[t][j] y[t][j][i] + (shexp ? shexp[t][i] : 0) */
int tkg_moe_combine(const float* y, const float* w, const float* shexp, float* out, int T, int k, int D,
                    void* stream);
/* f32-activation expert mat-vec for Q3_K / Q2_K / any ggml K-quant ids path: y[i][r] = W[ids[i]] row r . x[x_per_id ? i : i / k] */
int tkg_mmvq_ids_f32(int ggml_type, const void* w, size_t expert_bytes, const int32_t* ids, int n_ids, int k,
                     const float* x, int x_per_id, float* y, int n_in, int n_out, void* stream);

/* dst[r] = src row ids[r] (row_bytes % 4 == 0), e.g. the draft-vocab head */
int tkg_gather_rows(const void* src, size_t row_bytes, const int32_t* ids, int n, void* dst, void* stream);
/* greedy argmax per row (row stride ld floats): first index of the max, NaN / -inf never win (host
   sample::argmax); out_tok[r] = map ? map[idx] : idx, out_idx[r] = idx (either may be NULL) */
int tkg_argmax_rows(const float* x, int n, int rows, long ld, const int32_t* map, int32_t* out_tok, int32_t* out_idx,
                    void* stream);

/* ---- decode helpers (v2) ---- */
/* y[t][r] = W[r] . x[t], F32 weights [n_out][n_in], T <= 8 */
int tkg_f32_matvec(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* stream);
/* per-head Q8_0 mat-vec with f32 activations: y[t][h][r] = W[h][r] . x[t][h], W [H][n_out][n_in], T <= 8 */
int tkg_heads_q8_0(const void* W, const float* x, float* y, int H, int n_in, int n_out, int T, void* stream);
/* clamp-SwiGLU to q8_1 blocks (n % 32 == 0) */
int tkg_swiglu_clamp_q8(const float* g, const float* u, void* oq, int n, float limit, void* stream);
/* KDA output gate to q8_1 blocks */
int tkg_kda_gate_q8(const float* o, const float* g2, const float* w, void* oq, int T, int H, int D, float eps, void* stream);
/* Q3_K (11) / Q2_K (10) expert mat-vec v2, f32 activations, n_in = 2048 or 4096 */
int tkg_kq_ids(int ggml_type, const void* w, size_t expert_bytes, const int32_t* ids, int n_ids, int k, const float* x,
               int x_per_id, float* y, int n_in, int n_out, void* stream);
/* tkg_kq_ids for one or two matrices (w1 / y1 NULL = one) sharing ids and x; the id's activation row is staged in
   LDS once per block; bitwise tkg_kq_ids per (id, row). Returns -1 for unsupported shapes (use tkg_kq_ids). */
/* KDA decode block in one launch: tkg_kda_conv + tkg_glm_kda_step + tkg_kda_gate_q8 (bitwise the three); cnt =
   H zeroed ints (scratch, left zero). Returns -1 when no fused kernel matches the arch's step layout (use the three). */
int tkg_kda_fused(const float* qr, const float* kr, const float* vr, float* conv_state, const float* wq, const float* wk,
                  const float* wv, const float* g, const float* beta, float* state, const int32_t* ids, const float* dtb,
                  const float* a, float* o, const float* g2, const float* onorm_w, void* oq, int* cnt, int H, int D,
                  int T, int K_snap, float scale, float pro_lb, float pro_l2_eps, float gate_eps, void* stream);
/* router logits (tkg_f32_matvec over n_exp rows, T <= 4) + tkg_router in one launch, bitwise the two; cnt = 1
   zeroed int (scratch, left zero). -1 = unsupported shape. */
int tkg_f32mv_router(const float* W, const float* x, float* logits, int n_in, int n_exp, int T, const float* bias,
                     int32_t* ids, float* w, int k, float scale, int* cnt, void* stream);
/* tkg_hc_post_mix (first 0 / 1) or tkg_hc_post_mix_moe, then tkg_hc_pre2 for the next sub-block, in one launch
   (bitwise the two); cnt = T zeroed ints (scratch, left zero). -1 = unsupported (no next mix, inject mode). */
int tkg_hc_post_mix_pre(const float* xo, float* xhc, float* pc, const void* w_next, float* parts, int T, int n_embd,
                        int first, const float* scale, const float* base, const float* norm_w, float* y, float rms_eps,
                        float norm_eps, float hc_eps, int iters, void* yq, int* cnt, void* stream);
int tkg_hc_post_mix_moe_pre(const float* ey, const float* ew, const float* shexp, float* xhc, float* pc,
                            const void* w_next, float* parts, int T, int k, int n_embd, const float* scale,
                            const float* base, const float* norm_w, float* y, float rms_eps, float norm_eps,
                            float hc_eps, int iters, void* yq, int* cnt, void* stream);
int tkg_kq_ids2(int ggml_type, const void* w0, const void* w1, size_t expert_bytes, const int32_t* ids, int n_ids, int k,
                const float* x, int x_per_id, float* y0, float* y1, int n_in, int n_out, void* stream);

/* up to 3 Q8_0 mat-vecs (own x / y / n_out each, common n_in % 32 == 0) with f32 activations in one launch, T <= 8 */
int tkg_multi_q8_0(int n_mats, const void* const* w, const float* const* x, float* const* y, const int* n_out, int n_in,
                   int T, void* stream);

/* MoE routing sort for the grouped prefill MMQ (rows grouped by expert, stable in token order) */
int tkg_moe_route_sort(const int32_t* ids, int T, int k, int n_exp, int32_t* rowsrc, int32_t* rownat, int32_t* bounds,
                       void* stream);

/* q8_1 rows gathered through ids (ids NULL = identity) */
int tkg_gather_q8_1(const float* x, const int32_t* ids, void* xq, int n_in, int rows, void* stream);
/* grouped expert mat-vec (prefill, IQ2_S / IQ3_S): q8_1 rows sorted by expert, outputs to rows ids_dst */
int tkg_glm_mmvq_grouped(int qtype, const void* w, size_t expert_bytes, int n_experts, const int32_t* bounds,
                         const void* xq, const int32_t* ids_dst, float* y, int N, int K, int ld_y, void* stream);

void tkg_glm_set_grouped_ver(int v); /* test hook */

/* ---- wrappers over the tuned GLM HIP kernels (talus::glm_spike) ---- */
int tkg_quantize_q8_1(const float* x, void* xq, int n, void* stream);
int tkg_glm_mmvq_ids(int qtype, const void* w, const void* xq, const int32_t* ids, float* y, int N, int K,
                     int n_expert_used, int n_tokens, int x_group, int ids_s1, void* stream);
/* gate + up pair in one launch (rows bitwise those of two tkg_glm_mmvq_ids calls); returns -1 when not launched
 * (config without a pair kernel): the caller then runs tkg_glm_mmvq_ids twice. */
int tkg_glm_mmvq_ids_pair(int qtype, const void* w, const void* w2, const void* xq, const int32_t* ids, float* y,
                          float* y2, int N, int K, int n_expert_used, int n_tokens, int x_group, int ids_s1, void* stream);
int tkg_glm_kda_step(const float* q, const float* k, const float* v, const float* g, const float* beta, float* state,
                     const int32_t* ids, const float* dtb, const float* a, float* dst, int H, int S_v, int T,
                     int K_snap, float scale, float pro_lb, float pro_l2_eps, void* stream);
int tkg_glm_indexer(const float* q, const void* k_f16, const float* w, float* dst, int D, int H, int n_kv,
                    int n_tokens, void* stream);
/* sparse MLA attention from a row-index list (idx [N][n_kv_max], -1 padded); zero_mask = device f16 zeros >= ctx rows */
int tkg_glm_sparse_fa_idx(const float* q, const void* kv_f16, const void* zero_mask_f16, const int32_t* idx,
                          float* split_buf, float* dst, int D, int H, int N, int n_kv_max, uint32_t q_nb1,
                          uint32_t q_nb2, uint32_t k_stride, float scale, void* stream);
size_t tkg_glm_sparse_fa_split_bytes(int D, int H, int N, int n_kv_max);

#ifdef __cplusplus
}
#endif

#endif /* TALUS_GLM_KERNELS_H */
