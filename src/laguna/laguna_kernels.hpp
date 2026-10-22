// src/laguna/laguna_kernels.hpp - kernels the eager Laguna driver needs.
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define TALUS_HD __host__ __device__
#else
#define TALUS_HD
#endif

namespace strata::laguna {

// Transposed V pool, in 16-key tiles: [kv_head][pos / 16][dim][pos % 16], `stride` (a multiple of 16) positions per
// head.  A dim's 16 keys are one 32-byte run (WMMA's B fragment) and a 16-key tile is one contiguous 4 KB block.
TALUS_HD inline size_t vt_index(int kvh, int pos, int d, int stride) {
    return (((size_t) kvh * (stride / 16) + pos / 16) * 128 + d) * 16 + (pos & 15);
}

// q8 KV (int8 values, one f16 scale per 32 dims, as llama.cpp's q8_0): K int8 at the f16 layout's element index, its
// scales [pos][kv_head][4]; V int8 at vt_index, its scales [kv_head][pos/16][4][pos%16] so a 16-key fragment's scales
// are one 32-byte run.
TALUS_HD inline size_t ks_index(int pos, int kvh, int blk, int n_kv) { return ((size_t) pos * n_kv + kvh) * 4 + blk; }
TALUS_HD inline size_t vs_index(int kvh, int pos, int blk, int stride) {
    return (((size_t) kvh * (stride / 16) + pos / 16) * 4 + blk) * 16 + (pos & 15);
}

/// y[o] = sum_i w[o*n_in + i] * x[i], F32 weights (the router), f64 accumulation.
void f32_matvec(const float* w, const float* x, float* y, int n_in, int n_out, void* stream);

/// o[h][d] *= softplus(g[h]): Laguna's per-head attention output gate.
void softplus_head_gate(float* o, const float* g, int n_head, int head_dim, void* stream);

/// y[i] += w * x[i]
void axpy(float* y, const float* x, float w, int n, void* stream);

/// dst[r] = src[idx[r]], rows of row_bytes (a multiple of 4); idx is a device array.
void gather_rows(const void* src, void* dst, const int* idx, int n_rows, size_t row_bytes, void* stream);

/// x[t] += shared[t] + sum_j w[t*k+j] * y[map[t*k+j]], j ascending; map, w are device arrays.
void moe_add_slots(float* x, const float* shared, const float* y, const int* map, const float* w, int n_tok,
                   int n_embd, int k, void* stream);

/// Prompt attention for nq query tokens at positions q_pos0.., causal, over the f16 pools ([pos][n_kv][128]) that
/// already hold every key up to the last query.  window > 0: keys within `window` positions (self included).
/// q, out: (nq, n_head, 128) f32.  GQA group n_head/n_kv <= 9.
/// q_pos0_dev, when non-null, replaces q_pos0 with a device value (graph replay).  n_split > 1 splits each block's
/// key span n_split ways (flash-decoding, for decode's few query rows) with partials in part_acc
/// (n_split * nq * n_head * 128 floats) and part_ml (n_split * nq * n_head * 2), merged into out (merge = false:
/// the caller merges, e.g. attn_merge_gate_quant).  vt_stride > 0: V is stored tile-transposed,
/// [kv_head][pos/16][dim][pos%16] with vt_stride (a multiple of 16) positions per head (see v_store_t).
void prompt_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, float* out, int nq, int q_pos0,
                 int n_head, int n_kv, int window, void* stream, const int* q_pos0_dev = nullptr, int n_split = 1,
                 float* part_acc = nullptr, float* part_ml = nullptr, bool merge = true, int vt_stride = 0,
                 const uint16_t* k_scale = nullptr, const uint16_t* v_scale = nullptr);

/// y[t][o] = sum_i w[o][i] * x[t][i] for n_tok rows, F32 weights and accumulation (the batched router).
void f32_matmul(const float* w, const float* x, float* y, int n_in, int n_out, int n_tok, void* stream);

/// g = silu(g) * u
void silu_mul(float* g, const float* u, int64_t n, void* stream);

/// pos[i] = p0 + i / per_tok for n_tok * per_tok entries (RoPE positions per (token, head) row)
void fill_pos(int* pos, int n_tok, int per_tok, int p0, void* stream);

// ---- decode: device-side state, so a token step can be captured once and replayed ----

/// Laguna's router for one token on the device: sigmoid, top-k of p + bias (ties: lower index), weights
/// p / max(sum, 6.1e-5) * scale.  n_expert <= 256, k <= 16.
void route_sigmoid_topk(const float* logits, const float* bias, int32_t* ids, float* w, int n_expert, int k,
                        float scale, void* stream);

/// x = row *tok of a Q8_0 table, dequantized
void embed_q8_0(const void* table, const int* tok, float* x, int n_embd, void* stream, int n_tok = 1);

/// one token's K/V (n floats each) into f16 pools laid out [cell][n] at cell *pos
void kv_store(uint16_t* k_pool, uint16_t* v_pool, const float* kc, const float* vc, int n, const int* pos,
              void* stream);

/// out[i] = *pos for n entries (RoPE positions of one token's rows)
void fill_pos_dev(int* out, int n, const int* pos, void* stream);

/// *tok = argmax(logits) (ties: lower index), then *pos += 1
void argmax_advance(const float* logits, int n, int* tok, int* pos, void* stream);

/// Per-head RMSNorm (w[128], shared by all heads) then NEOX RoPE on the first n_rot dims, in place, for `rows`
/// rows of 128; pos[row] indexes the (max_pos, n_rot/2) tables.  Replaces rms_norm_weighted + rope_neox_apply.
/// pos[row / pos_div] is the row's position (pos_div 0: pos[0] for every row).
void head_norm_rope(float* x, const float* w, int rows, float eps, int n_rot, const float* cos_tab,
                    const float* sin_tab, const int* pos, int pos_div, void* stream);

/// dst = RMSNorm(src) * w per row of `cols` (dst may equal src)
void rms_norm_to(float* dst, const float* src, const float* w, int rows, int cols, float eps, void* stream);

/// y = W . x, F32, one warp per row, f32 accumulation (the decode router); n_in % 128 == 0
void f32_matvec_fast(const float* w, const float* x, float* y, int n_in, int n_out, void* stream);

/// x += add + sum_j w[j] * yexp[j*n] (add null: no update), then h = RMSNorm(x) * norm_w written as f32 (h_out,
/// when non-null) and Q8_1 (q8, native_quantize_q8_1's bits).  One block; n % 32 == 0, n <= 4096.
void add_norm_quant(float* x, const float* add, const float* yexp, const float* w, int k, const float* norm_w,
                    float* h_out, void* q8, int n, float eps, void* stream, int n_rows = 1);
/// x = fma(alpha, ax, x) (axpy's y += w * x), then add_norm_quant's RMSNorm + Q8_1 (bitwise axpy + add_norm_quant).
void axpy_norm_quant(float* x, const float* ax, float alpha, const float* norm_w, float* h_out, void* q8, int n,
                     float eps, void* stream, int n_rows);

/// y = W . x, F32, one 128-thread block per row
void f32_matvec_rows(const float* w, const float* x, float* y, int n_in, int n_out, void* stream);

/// route_sigmoid_topk for 256 experts in one warp
void route_warp(const float* logits, const float* bias, int32_t* ids, float* w, int k, float scale, void* stream);

/// Decode: q heads normed + rotated in place; k heads normed + rotated and stored to k_pool at *pos; v stored to
/// v_pool at *pos.  Replaces two head_norm_rope and a kv_store.
void qkv_norm_rope_store(float* q, const float* kc, const float* vc, const float* q_norm, const float* k_norm, int H,
                         int n_kv, float eps, int n_rot, const float* cos_tab, const float* sin_tab, const int* pos,
                         uint16_t* k_pool, uint16_t* v_pool, void* stream, int vt_stride = 0,
                         uint16_t* k_scale = nullptr, uint16_t* v_scale = nullptr, int n_tok = 1);

/// Decode: prompt_attn's split partials (one token) merged, times softplus(g[h]), quantized to Q8_1 (n_head x 128).
void attn_merge_gate_quant(const float* part_acc, const float* part_ml, const float* g, void* q8, int n_head,
                           int n_split, void* stream, int n_tok = 1);

/// Decode: router logits then (last block) route_warp's top-k.  counter: a device uint, zero, reset by the kernel.
void router_route(const float* w, const float* x, float* logits, const float* bias, int32_t* ids, float* wout,
                  int n_in, int n_expert, int k, float scale, unsigned int* counter, void* stream);

/// prompt_attn for prompts on RDNA3/3.5 WMMA (f16 inputs, f32 accumulate and softmax); same arguments, no split.
/// V must be transposed (vt_stride: positions per row, a multiple of 16).
/// n_split > 1 (decode): split-K, one wave per (KV head, chunk), partials into part_acc/part_ml as prompt_attn's;
/// q_pos0_dev replaces q_pos0 when non-null.
void prompt_attn_wmma(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, float* out, int nq, int q_pos0,
                      int n_head, int n_kv, int window, void* stream, int vt_stride, const int* q_pos0_dev = nullptr,
                      int n_split = 1, float* part_acc = nullptr, float* part_ml = nullptr,
                      const uint16_t* k_scale = nullptr, const uint16_t* v_scale = nullptr);

/// prompt_attn_wmma's prompt kernel: 2 (default, TALUS_LAGUNA_ATTN) = K / V tiles shared by the block,
/// 1 = per-wave tiles (the original; bitwise the same output); -1 = back to the environment's choice.
void prompt_attn_set_version(int v);

/// Prefill routing on the device for n_tok tokens (256 experts): top-k per token (sel_e, sel_w: n_tok*k), then the
/// rows grouped by expert in (token, slot) order: bounds (257), rtok (row -> token), smap ((token, slot) -> row).
/// cnt: 256 ints of scratch.
void route_group(const float* logits, const float* bias, int n_tok, int k, float scale, int32_t* sel_e, float* sel_w,
                 int* cnt, int32_t* bounds, int32_t* rtok, int32_t* smap, void* stream);

/// V rows (nb x n_kv x 128 f32) into the tile-transposed f16 pool [kv_head][pos/16][dim][pos%16] at p0..
void v_store_t(const float* vc, uint16_t* v_pool, int nb, int n_kv, int p0, int stride, void* stream);

/// q8 KV (k_scale/v_scale non-null in the attention and store calls): int8 values, one f16 scale per 32 dims; K at the
/// f16 layout's index with scales [pos][kv_head][4], V tile-transposed with scales [kv_head][pos/16][4][pos%16].
/// Prefill store of nb tokens' K and V rows at p0.. (stride: positions per head, a multiple of 16).
void kv_store_q8(const float* kc, const float* vc, void* k_pool, void* v_pool, uint16_t* k_scale, uint16_t* v_scale,
                 int nb, int n_kv, int p0, int stride, void* stream);

/// Decode attention for one query token (V tile-transposed; q8 when k_scale is non-null), split-K into n_split chunks,
/// the next key tile prefetched into registers.  Reads whole tiles (32 or 64 keys): the pools need 64 positions of slack
/// past the context (stride includes them).  Partials as prompt_attn's, merged by attn_merge_gate_quant.
void decode_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const uint16_t* k_scale,
                 const uint16_t* v_scale, int n_head, int n_kv, int window, int stride, const int* pos_dev, int n_split,
                 float* part_acc, float* part_ml, void* stream, int n_tok = 1, const int* sel = nullptr,
                 int sel_k = 0, bool sel_per_kv = false, int sel_bs = 64);
/// sel set (sparse_select's output: n_tok x sel_k block indices, or n_tok x n_kv x sel_k with sel_per_kv): attend
/// only to those 64-key blocks.

// ---- multi-token decode (speculative verify): the n_tok arguments above run several consecutive tokens at once ----

/// Laguna's top-k for n_tok tokens' router logits (n_tok x 256): ids, w (n_tok x k).
void route_topk(const float* logits, const float* bias, int n_tok, int k, float scale, int32_t* ids, float* w,
                void* stream);

/// y[t][o] = W[o] . x[t] for n_tok <= 8 tokens (F32 router rows, each read once)
void f32_matvec_rows_n(const float* w, const float* x, float* y, int n_in, int n_out, int n_tok, void* stream);

/// out[r] = argmax of row r (n floats), ties to the lower index
void argmax_rows(const float* logits, int n_rows, int n, int* out, void* stream);

// ---- block-sparse decode attention (Quest-style) ----
// Per full-attention layer: bnd [n_kv][nb_max][256] f32, the elementwise max (dims 0..127) and min (128..255) of the
// keys of each block of bs keys (16, 32 or 64).  kbound_update recomputes blocks b_first..b_last from the stored q8 keys up to pos_last
// (pos_dev set: the blocks holding positions *pos_dev .. *pos_dev + n_tok - 1).
void kbound_update(const void* k_pool, const uint16_t* k_scale, void* bnd, int nb_max, int n_kv, int b_first,
                   int b_last, int pos_last, const int* pos_dev, int n_tok, void* stream, int bs = 64, bool f16 = false);
/// ub [n_tok][n_head][nb_max]: the per-block bound of q.k (scaled by 1/sqrt(128)) for blocks 0 .. pos/64.
void sparse_scores(const float* q, const void* bnd, float* ub, int nb_max, int n_head, int n_kv,
                   const int* pos_dev, int n_tok, void* stream, int bs = 64, bool f16 = false);
/// sel [n_tok][k]: per token the top-k blocks by the sum over heads of softmax(ub) over blocks, the blocks holding the
/// first `sink` and the last `recent` positions forced; ascending.  Needs pos/64 + 1 >= k.
void sparse_select(const float* ub, int nb_max, int n_head, const int* pos_dev, int n_tok, int k, int sink, int recent,
                   int* sel, void* stream, int groups = 1, int bs = 64, uint32_t* scr = nullptr);
/// scr set ([n_tok][groups][nb_max] u32 scratch): the radix-select kernel (no context cap; the same selections);
/// otherwise the LDS kernel (nb_max <= 12288).  f16 (kbound_update / sparse_scores): bounds stored as f16, rounded
/// outward.
/// groups > 1: a selection per group of n_head / groups heads (groups = n_kv: per KV head), sel [n_tok][groups][k].

}  // namespace strata::laguna
