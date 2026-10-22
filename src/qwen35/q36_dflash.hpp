// src/qwen35/q36_dflash.hpp - Talus (talus-next, lever 6): the kernels of the DFlash2 block-diffusion drafter
// (z-lab's Qwen3.8-27B-DFlash2; the host side is q36_dflash_model.{hpp,cpp}).  Launchers only; every launch goes to
// `stream`, reads positions from the device and has a fixed grid, so the draft and the context injection can be
// captured in the engine's round graphs.  Reference: llama.cpp (branch halo) src/models/dflash.cpp.
#pragma once

#include <cstdint>

namespace talus::q36::dflash {

/// out[r] = rms(x[r]) * w  (rows of n; out may be x); optional q8_1 output (n % 32 == 0)
void rms_rows(const float* x, const float* w, float* out, int n, int rows, float eps, void* stream,
              void* q8 = nullptr);
/// out[r] = rms(y0[r] + .. + y4[r]) * w: the context features' five fc slices summed, then enc.output_norm
void sum5_rms(const float* const* ys, const float* w, float* out, int n, int rows, float eps, void* stream,
              void* q8 = nullptr);
/// DFlash2's per-token dynamic depthwise conv over a block of `rows` tokens (kernel 2, causal inside the block):
/// out[i][c] = sum_t (base[side][t][c] + dyn[i][side][t][c / group]) * in[i - t][c]   (in[-1] = 0)
/// dyn rows are [2 sides][2 taps][n / group]; base is [2 sides][2 taps][n]
void dyn_conv(const float* in, const float* dyn, const float* base, float* out, int n, int rows, int group, int side,
              void* stream, void* q8 = nullptr);
/// per (row, head): rms over hd (weight w[hd]), then NEOX RoPE over the whole head at position *pos + row
void qk_norm_rope(float* x, const float* w, int heads, int hd, int rows, const int32_t* pos, float theta, float eps,
                  void* stream);
/// the K / V rows (heads_kv * hd floats each) into the ring caches at cells (*pos + r) mod ring
void kv_store(const float* k, const float* v, float* kring, float* vring, int width, int rows, const int32_t* pos,
              int ring, void* stream);
/// the K / V rows (k_width / v_width floats each) into F16 ring caches at cells (*pos + r) mod ring
void kv_store_f16(const float* k, const float* v, void* kring_f16, void* vring_f16, int k_width, int v_width,
                  int rows, const int32_t* pos, int ring, void* stream);
/// The block's attention: queries q [8][heads][hd] at positions *pos .. *pos + 7 over the ring cells of positions
/// max(0, *pos - window + 1) .. *pos + 7 (key kp visible to query qp when qp - kp < window; non-causal inside the
/// block).  part / pm / pl: split-K scratch (attn_scratch_floats).  out [8][heads][hd].
int attn_chunks(int window);
size_t attn_scratch_floats(int heads, int hd, int window);
void block_attn(const float* q, const float* kring, const float* vring, const int32_t* pos, int heads, int heads_kv,
                int hd, int ring, int window, float scale, float* part, float* out, void* stream);
/// out = silu(g) * u
void swiglu(const float* g, const float* u, float* out, int64_t n, void* stream, void* q8 = nullptr);
/// x += y
void add_inplace(float* x, const float* y, int64_t n, void* stream);
/// the block's tokens: [*anchor, mask x (n - 1)]
void block_tokens(const int32_t* anchor, int32_t* out, int n, int32_t mask, void* stream);
/// *p = v (eager positions for the prompt's context injection)
void set_i32(int32_t* p, int32_t v, void* stream);
/// rows 1 .. rows - 1 of the head's logits [rows][nv]: the top 16 (cand: token ids through map, or the row index;
/// unary: their logits), descending, ties to the lower index
void top16(const float* logits, int nv, int rows, const int32_t* map, int32_t* cand, float* unary, void* stream);
/// DFlash2's selector walk on the device: d_0 = *anchor; for i = 1 .. n: d_i = cand_i[argmax_j unary_i[j] +
/// sum_r succ[cand_i[j]][r] * pred[d_{i-1}][r] * gate_i[r]].  pred / succ: Q4_K tables of 256-wide rows (one block
/// each, by token id); gate [rows][256].  The drafts into out[0 .. n - 1]; score[i - 1] the winner's softmax weight.
void select_chain(const int32_t* cand, const float* unary, const float* gate, const void* pred, const void* succ,
                  const int32_t* anchor, int32_t* out, float* score, int n, void* stream);
/// The same walk for a sampled request (talus-next): *tau > 0 draws d_i from q_i = softmax(score_i / tau) over the 16
/// candidates (Gumbel-max, noise from (*seed, *ctr, i, j)); q[(i - 1) * 16 + j] gets q_i, score[i - 1] max_j q_i (known
/// before the draw: the stop rule's input).  *tau <= 0: the greedy walk (q = softmax at T 1).
void select_chain_q(const int32_t* cand, const float* unary, const float* gate, const void* pred, const void* succ,
                    const int32_t* anchor, int32_t* out, float* score, float* q, const float* tau, const uint32_t* seed,
                    const uint32_t* ctr, int n, void* stream);

}  // namespace talus::q36::dflash
