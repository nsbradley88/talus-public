// src/program/qwen4exp_p2.cpp - qwen4exp-p2: launch-only C ABI wrappers for the Rust Flash-Next prompt path.
//
// The orchestration of a batched prompt (the chunk loop, the expert stream and its issuer thread, the layer and half
// sequence, the host grouping of the routed rows, the MTP draft K/V batches) is rust/crates/talus-model/src/qwen4exp/
// prefill.rs.  Every function here is one kernel launch, one runtime call or one host query, with the arguments
// translated from the C ABI's flat structs (include/talus_qwen4exp_kernels.h, the qwen4exp-p2 block) to the kernels'
// C++ ones.  No control flow of the prompt path lives here.
#include "qwen4exp_r2.hpp"   // qwen4exp-r2: the layer.cpp launch helpers as thin wrappers
#include "talus_qwen4exp_kernels.h"

#include "wmma_gemm.h"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/progress.hpp"
#include "strata/core/session.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/fnp_qsa_tile_attn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/tp_moe.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {
namespace pf = strata::prefill;
namespace sk = strata::kernels;
namespace sc = strata::core;

inline void put_err(char* err, size_t cap, const char* msg) {
    if (err != nullptr && cap > 0) std::snprintf(err, cap, "%s", msg != nullptr ? msg : "");
}
inline sk::QsaShapes shapes_of(const tk_qwen4exp_p2_qsa_shapes_t* s) {
    sk::QsaShapes r;
    r.n_head = s->n_head; r.n_head_kv = s->n_head_kv; r.head_dim = s->head_dim; r.n_rot = s->n_rot;
    r.idx_n_head = s->idx_n_head; r.idx_dim = s->idx_dim; r.idx_block = s->idx_block; r.idx_top_k = s->idx_top_k;
    r.page_size = s->page_size;
    return r;
}
inline sk::KvHostPools host_of(const tk_qwen4exp_p2_kv_pools_t* h) {
    sk::KvHostPools r;
    static_assert(sizeof(r) == sizeof(*h), "qwen4exp-p2: KvHostPools mirror");
    std::memcpy(&r, h, sizeof(r));
    return r;
}
inline sk::QsaAttnPools attn_of(const tk_qwen4exp_p2_attn_pools_t* p) {
    sk::QsaAttnPools r;
    static_assert(sizeof(r) == sizeof(*p), "qwen4exp-p2: QsaAttnPools mirror");
    std::memcpy(&r, p, sizeof(r));
    return r;
}
inline void qsa_info_of(const sc::QsaState& st, tk_qwen4exp_p2_qsa_info_t* o) {
    *o = tk_qwen4exp_p2_qsa_info_t{};
    o->kv_mode = st.kv_mode;
    o->kv_int8 = st.kv_int8;
    o->kv_q4 = st.kv_q4;
    o->kv_hybrid = st.kv_hybrid;
    o->kv_rot = st.kv_rot;
    o->host_present = st.host.present();
    o->fmt = sc::qsa_kv_format(st);
    o->n_pages = st.n_pages;
    o->max_cells = st.max_cells;
    o->idx_pooled_rows = st.idx_pooled_rows;
    o->n_slots = st.n_slots;
    o->k_pool = st.k_pool; o->v_pool = st.v_pool; o->k_q = st.k_q; o->v_q = st.v_q;
    o->k_scale = st.k_scale; o->v_scale = st.v_scale; o->k_q4 = st.k_q4; o->v_q4 = st.v_q4;
    o->page_table = st.page_table;
    std::memcpy(&o->host, &st.host, sizeof(o->host));
    const sk::QsaAttnPools ap = strata::core::qwen4exp_r2::qsa_attn_pools(st);   // qwen4exp-r2
    std::memcpy(&o->attn, &ap, sizeof(o->attn));
    o->idx = {st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
}
}  // namespace

extern "C" {

// ---------------------------------------------------------------- runtime
void tk_qwen4exp_p2_abi_sizes(int64_t out[8]) {
    out[0] = sizeof(tk_qwen4exp_p2_layout_t);
    out[1] = sizeof(tk_qwen4exp_p2_qsa_info_t);
    out[2] = sizeof(tk_qwen4exp_p2_weight_t);
    out[3] = sizeof(tk_qwen4exp_p2_expert_fmt_t);
    out[4] = sizeof(tk_qwen4exp_p2_stats_t);
    out[5] = sizeof(tk_qwen4exp_p2_mtp_info_t);
    out[6] = sizeof(tk_qwen4exp_p2_cvec_t);
    out[7] = sizeof(tk_qwen4exp_p2_stager_job_t);
}
int tk_qwen4exp_p2_set_device(int32_t device) { return cudaSetDevice(device) == cudaSuccess ? 1 : 0; }
int tk_qwen4exp_p2_memcpy(void* dst, const void* src, uint64_t bytes, int32_t kind, void* stream) {
    return cudaMemcpyAsync(dst, src, (size_t) bytes, static_cast<decltype(cudaMemcpyHostToDevice)>(kind), (cudaStream_t) stream) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_p2_memset(void* dst, int32_t value, uint64_t bytes, void* stream) {
    return cudaMemsetAsync(dst, value, (size_t) bytes, (cudaStream_t) stream) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_p2_event_create(void** ev, int32_t timing) {
    cudaEvent_t e = nullptr;
    const cudaError_t r = timing ? cudaEventCreate(&e) : cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    *ev = (void*) e;
    return r == cudaSuccess ? 1 : 0;
}
void tk_qwen4exp_p2_event_destroy(void* ev) { if (ev) cudaEventDestroy((cudaEvent_t) ev); }
int tk_qwen4exp_p2_event_record(void* ev, void* stream) {
    return cudaEventRecord((cudaEvent_t) ev, (cudaStream_t) stream) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_p2_event_sync(void* ev) { return cudaEventSynchronize((cudaEvent_t) ev) == cudaSuccess ? 1 : 0; }
int tk_qwen4exp_p2_event_elapsed(float* ms, void* a, void* b) {
    return cudaEventElapsedTime(ms, (cudaEvent_t) a, (cudaEvent_t) b) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_p2_stream_wait_event(void* stream, void* ev) {
    return cudaStreamWaitEvent((cudaStream_t) stream, (cudaEvent_t) ev, 0) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_p2_stream_sync(void* stream) {
    return cudaStreamSynchronize((cudaStream_t) stream) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_p2_peek_error(char* err, size_t err_cap) {
    const cudaError_t e = cudaGetLastError();
    put_err(err, err_cap, cudaGetErrorString(e));
    return e == cudaSuccess ? 1 : 0;
}
void tk_qwen4exp_p2_progress_at(const char* what, int64_t detail, int64_t chunk) { sc::progress_at(what, detail, chunk); }
void tk_qwen4exp_p2_progress_beat(void) { sc::progress_beat(); }
void tk_qwen4exp_p2_pin_thread(void) { strata::aux_cpus::pin_current_thread(); }

// ---------------------------------------------------------------- weights, embedding
int tk_qwen4exp_p2_weight(void* wt, int64_t layer, const char* suffix, tk_qwen4exp_p2_weight_t* out) {
    *out = tk_qwen4exp_p2_weight_t{};
    const sc::LayerView v(*(const sc::WeightTable*) wt, layer);
    const sc::WeightRef* r = v.get(suffix);
    if (r == nullptr) return 0;
    out->ref = r;
    out->data = r->data;
    out->bf16 = r->kind == sc::WeightKind::Bf16InF32;
    out->hc_q8 = r->hc_q8 != nullptr;
    out->native_type = r->native_type;
    out->native_data = r->native_data;
    out->ne0 = r->ne0;
    out->ne1 = r->ne1;
    return 1;
}
const uint16_t* tk_qwen4exp_p2_hc_bf16_view(const void* wref, void* stream, int32_t slot, int32_t cls) {
    return sc::NativeDense::hc_bf16_view(*(const sc::WeightRef*) wref, stream, slot, cls);
}
void tk_qwen4exp_p2_hc_bf16_release(void* stream, int32_t slot, int32_t cls) {
    sc::NativeDense::hc_bf16_release(stream, slot, cls);
}
int tk_qwen4exp_p2_embed_kind(void* wt, void* g, int64_t* out_rows) {
    *out_rows = 0;
    if (sc::native_embed() != nullptr) return 1;
    const sc::ModelGeometry& gg = *(const sc::ModelGeometry*) g;
    const sc::WeightRef* w = ((const sc::WeightTable*) wt)->find("token_embd.weight");
    if (w == nullptr || w->codebook_iq4nl || w->ne0 != gg.n_embd || w->group_elems <= 0 ||
        (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8))
        return 0;
    *out_rows = w->ne1;
    return 2;
}
void tk_qwen4exp_p2_embed_gather(void* wt, const int32_t* tok_dev, int64_t n, float* out, void* stream) {
    if (const sc::NativeEmbed* nemb = sc::native_embed()) {
        nemb->gather_dev(tok_dev, n, out, stream);
        return;
    }
    const sc::WeightRef* w = ((const sc::WeightTable*) wt)->find("token_embd.weight");
    const auto* codes = (const uint8_t*) w->data;
    const auto* scales = (const float*) (codes + w->codes_bytes);
    const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
    sk::embedding_gather_dev(codes, scales, offsets, tok_dev, (int) n, w->ne0, w->code_bits, w->code_bias,
                             w->group_elems, (uint64_t) (w->ne0 / (8 / w->code_bits)),
                             (uint64_t) (w->ne0 / w->group_elems), out, stream);
}
int tk_qwen4exp_p2_embed_row(void* wt, void* g, int64_t token, float* out, void* stream, char* err, size_t err_cap) {
    std::string e;
    if (!strata::core::qwen4exp_r2::embed_row(*(const sc::WeightTable*) wt, *(const sc::ModelGeometry*) g, token, out, stream, e)) {
        put_err(err, err_cap, e.c_str());
        return 0;
    }
    return 1;
}

// ---------------------------------------------------------------- GEMMs
void tk_qwen4exp_p2_gemm_bf16(void* gemm, const uint16_t* x, const uint16_t* w, float* y, int64_t t, int64_t n,
                              int64_t k, int64_t ldy, float beta, int64_t ldx) {
    ((pf::Gemm*) gemm)->bf16(x, w, y, t, n, k, ldy, beta, ldx);
}
void tk_qwen4exp_p2_gemm_f16(void* gemm, const uint16_t* x, const uint16_t* w, float* y, int64_t t, int64_t n,
                             int64_t k) {
    ((pf::Gemm*) gemm)->f16(x, w, y, t, n, k);
}
void tk_qwen4exp_p2_gemm_native(void* gemm, const uint16_t* x, int32_t type, const void* w, float* y, int64_t t,
                                int64_t n, int64_t k, int64_t ldy, float beta, int64_t ldx) {
    ((pf::Gemm*) gemm)->native(x, type, w, y, t, n, k, ldy, beta, ldx);
}
int tk_qwen4exp_p2_gemm_bf16_hcd_exact(void* gemm, const uint16_t* x, int64_t ldx, const uint16_t* w, float* y,
                                       int64_t t, int64_t n, int64_t k) {
    return ((pf::Gemm*) gemm)->bf16_hcd_exact(x, ldx, w, y, t, n, k) ? 1 : 0;
}

// ---------------------------------------------------------------- hyper-connection
void tk_qwen4exp_p2_gr_broadcast(const float* e, float* r, int64_t t, void* stream) { pf::gr_broadcast(e, r, t, stream); }
void tk_qwen4exp_p2_gr_norm(const float* r, const float* w, float eps, float* xn, uint16_t* xn16, int64_t t,
                            void* stream, uint16_t* xn16_lo) {
    pf::gr_norm(r, w, eps, xn, xn16, t, stream, xn16_lo);
}
void tk_qwen4exp_p2_gr_norm_rs(const float* r, const float* w, float eps, float* rs, uint16_t* xn16, int64_t t,
                               void* stream, uint16_t* xn16_lo, int64_t ldx) {
    pf::gr_norm_rs(r, w, eps, rs, xn16, t, stream, xn16_lo, ldx);
}
void tk_qwen4exp_p2_gr_mix_r(const float* r, const float* rs, const float* w, const float* gated, float* mixed,
                             uint16_t* mixed16, int64_t t, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    pf::gr_mix_r(r, rs, w, gated, mixed, mixed16, t, stream, mixed_h, mixed16_lo);
}
void tk_qwen4exp_p2_gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t t,
                           void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    pf::gr_mix(xn, gated, mixed, mixed16, t, stream, mixed_h, mixed16_lo);
}
int tk_qwen4exp_p2_gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* r, const float* rs,
                            const float* w, float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t t,
                            void* stream) {
    return pf::gr_upmix(lo16, w_up, r, rs, w, mixed, mixed16, mixed_h, t, stream) ? 1 : 0;
}
void tk_qwen4exp_p2_gr_silu(const float* lo, uint16_t* lo16, int64_t t, void* stream, uint16_t* lo16_lo) {
    pf::gr_silu(lo, lo16, t, stream, lo16_lo);
}
void tk_qwen4exp_p2_gr_write(float* r, const float* bo, const float* inj, int64_t inj_ld, int64_t t, void* stream) {
    pf::gr_write(r, bo, inj, inj_ld, t, stream);
}
void tk_qwen4exp_p2_gr_write_norm_rs(float* r, const float* bo, const float* inj, int64_t inj_ld, const float* w_next,
                                     float eps, float* rs, uint16_t* xn16, int64_t t, void* stream,
                                     uint16_t* xn16_lo, int64_t ldx) {
    pf::gr_write_norm_rs(r, bo, inj, inj_ld, w_next, eps, rs, xn16, t, stream, xn16_lo, ldx);
}
// rb-num (opt-in TALUS_FN_HC_BF16=1 BF16 / =2 FP16): the hyper-connection kernels on a 16-bit residual R
void tk_qwen4exp_p2_gr16_broadcast(const float* e, uint16_t* r, int64_t t, int32_t fmt, void* stream) {
    pf::gr16_broadcast(e, r, t, fmt, stream);
}
void tk_qwen4exp_p2_gr16_norm_rs(const uint16_t* r, const float* w, float eps, float* rs, uint16_t* xn16, int64_t t,
                                 int32_t fmt, void* stream, uint16_t* xn16_lo, int64_t ldx) {
    pf::gr16_norm_rs(r, w, eps, rs, xn16, t, fmt, stream, xn16_lo, ldx);
}
void tk_qwen4exp_p2_gr16_mix_r(const uint16_t* r, const float* rs, const float* w, const float* gated, float* mixed,
                               uint16_t* mixed16, int64_t t, int32_t fmt, void* stream, uint16_t* mixed_h,
                               uint16_t* mixed16_lo) {
    pf::gr16_mix_r(r, rs, w, gated, mixed, mixed16, t, fmt, stream, mixed_h, mixed16_lo);
}
int tk_qwen4exp_p2_gr16_upmix(const uint16_t* lo16, const uint16_t* w_up, const uint16_t* r, const float* rs,
                              const float* w, float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t t,
                              int32_t fmt, void* stream) {
    return pf::gr16_upmix(lo16, w_up, r, rs, w, mixed, mixed16, mixed_h, t, fmt, stream) ? 1 : 0;
}
void tk_qwen4exp_p2_gr16_write_norm_rs(uint16_t* r, const float* bo, const float* inj, int64_t inj_ld,
                                       const float* w_next, float eps, float* rs, uint16_t* xn16, int64_t t,
                                       int32_t fmt, void* stream, uint16_t* xn16_lo, int64_t ldx) {
    pf::gr16_write_norm_rs(r, bo, inj, inj_ld, w_next, eps, rs, xn16, t, fmt, stream, xn16_lo, ldx);
}
void tk_qwen4exp_p2_gr16_write(uint16_t* r, const float* bo, const float* inj, int64_t inj_ld, int64_t t, int32_t fmt,
                               void* stream) {
    pf::gr16_write(r, bo, inj, inj_ld, t, fmt, stream);
}
void tk_qwen4exp_p2_gr16_widen(const uint16_t* r16, float* r, int64_t t, int32_t fmt, void* stream) {
    pf::gr16_widen(r16, r, t, fmt, stream);
}
void tk_qwen4exp_p2_gr16_narrow(const float* r, uint16_t* r16, int64_t t, int32_t fmt, void* stream) {
    pf::gr16_narrow(r, r16, t, fmt, stream);
}
void tk_qwen4exp_p2_gr_write_cvec_norm_rs(float* r, const float* bo, const float* inj, int64_t inj_ld,
                                          const float* v_l, const float* s_l, const int* on, int32_t mode,
                                          const float* w_next, float eps, float* rs, uint16_t* xn16, int64_t t,
                                          void* stream, uint16_t* xn16_lo, int64_t ldx) {
    pf::gr_write_cvec_norm_rs(r, bo, inj, inj_ld, v_l, s_l, on, mode, w_next, eps, rs, xn16, t, stream, xn16_lo, ldx);
}
int tk_qwen4exp_p2_pf_hcdown_bf16(const uint16_t* xn16, int64_t ldx, const uint16_t* wd, const uint16_t* wi,
                                  int64_t lr, int64_t hc, float* lo, float* inj, int64_t t, int64_t d, void* stream) {
    return strata_pf_hcdown_bf16(xn16, ldx, wd, wi, lr, hc, lo, inj, t, d, stream) ? 1 : 0;
}
void tk_qwen4exp_p2_cvec_info(int64_t layer, tk_qwen4exp_p2_cvec_t* out) {
    *out = tk_qwen4exp_p2_cvec_t{};
    const sk::Cvec& c = sk::cvec();
    out->covers = c.covers(layer);
    out->mode = c.mode;
    out->n_embd = c.n_embd;
    out->hc = c.hc;
    if (out->covers) out->tables_ok = sk::cvec_tables(&out->dir, &out->s, &out->on);
}
void tk_qwen4exp_p2_cvec_apply(float* r, int64_t layer, int64_t t, int64_t ld, void* stream) {
    sk::cvec_apply(r, layer, t, ld, nullptr, 0, nullptr, 0, false, stream);
}

// ---------------------------------------------------------------- PLE
void tk_qwen4exp_p2_ple_ngram_rows(void* consts, const int32_t* tokens, const int32_t* prev, int32_t n,
                                   uint32_t* out) {
    sk::ngram_rows(tokens, prev, n, *(const sk::PleConsts*) consts, out);
}
int tk_qwen4exp_p2_ple_gather_batch(void* table, const uint32_t* rows, uint64_t n, float* out, char* err,
                                    size_t err_cap) {
    std::string e;
    if (!((sk::PleTable*) table)->gather_batch(rows, (size_t) n, out, e)) {
        put_err(err, err_cap, e.c_str());
        return 0;
    }
    return 1;
}
int tk_qwen4exp_p2_ple_postops_batch(void* weights, float* key, float* hidden, const float* value, float* hist,
                                     float* qn, float* gated, float* gate, int32_t n, void* stream, char* err,
                                     size_t err_cap) {
    try {
        sk::native_ple_postops_batch(key, hidden, value, hist, *(const sk::PleWeights*) weights, qn, gated, gate, n,
                                     stream);
    } catch (const std::exception& e) {
        put_err(err, err_cap, (std::string("prefill PLE: ") + e.what()).c_str());
        return 0;
    }
    return 1;
}
int tk_qwen4exp_p2_ple_block(void* weights, const float* emb, const float* hidden, const float* hist,
                             float* normalized, float* result, void* scratch, void* stream, char* err,
                             size_t err_cap) {
    sk::PleOut po;
    po.normalized = normalized;
    po.result = result;
    try {
        sk::ple_block(emb, hidden, hist, *(const sk::PleWeights*) weights, po, scratch, stream);
    } catch (const std::exception& e) {
        put_err(err, err_cap, (std::string("prefill PLE: ") + e.what()).c_str());
        return 0;
    }
    return 1;
}
void tk_qwen4exp_p2_ple_history_advance(float* hist, const float* normalized, void* stream) {
    sk::ple_history_advance(hist, normalized, stream);
}
void tk_qwen4exp_p2_to_f16(const float* x, uint16_t* y, int64_t n, void* stream) { pf::to_f16(x, y, n, stream); }
void tk_qwen4exp_p2_to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    pf::to_bf16(x, y, n, stream, ylo);
}

// ---------------------------------------------------------------- GDN
void tk_qwen4exp_p2_gdn_gates(const float* ab, const float* dt, const float* a, float* gate, float* beta, int64_t t,
                              void* stream) {
    pf::gdn_gates(ab, dt, a, gate, beta, t, stream);
}
void tk_qwen4exp_p2_gdn_conv(float* hist, const float* qkv, const float* w, float* h, int64_t t, float eps,
                             void* stream) {
    pf::gdn_conv(hist, qkv, w, h, t, eps, stream);
}
void tk_qwen4exp_p2_gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                                   const float* gamma, float eps, float* y, uint16_t* y16, int64_t t, void* stream,
                                   int64_t ld16) {
    pf::gdn_recurrence(state, h, gate, beta, z, gamma, eps, y, y16, t, stream, ld16);
}

// ---------------------------------------------------------------- QSA
int tk_qwen4exp_p2_qsa_info(void* ss, int64_t ord, tk_qwen4exp_p2_qsa_info_t* out) {
    const sc::SessionState& s = *(const sc::SessionState*) ss;
    if (ord < s.qsa_ord0 || ord >= s.qsa_ord0 + s.qsa_alloc) return 0;
    qsa_info_of(s.qsa_states[ord], out);
    return 1;
}
void tk_qwen4exp_p2_rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps,
                             void* stream) {
    pf::rms_rows(x, w, rows, cols, ld, eps, stream);
}
void tk_qwen4exp_p2_rope(float* x, int64_t t, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, void* stream) {
    pf::rope(x, t, heads, dim, ld, pos0, sk::rope_scaling(), stream);
}
void tk_qwen4exp_p2_split_q(const float* q_full, float* q, int64_t t, void* stream) { pf::split_q(q_full, q, t, stream); }
void tk_qwen4exp_p2_gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t t, void* stream,
                              int64_t ld16) {
    pf::gate_attn(attn, q_full, out16, t, stream, ld16);
}
void tk_qwen4exp_p2_fwht256(float* x, int64_t rows, void* stream) { sk::fwht256_inplace_cuda(x, rows, stream); }
void tk_qwen4exp_p2_kv_stage_from_host(const tk_qwen4exp_p2_attn_pools_t* stage,
                                       const tk_qwen4exp_p2_kv_pools_t* host, int32_t fmt, int64_t n_blocks,
                                       const tk_qwen4exp_p2_qsa_shapes_t* s, void* stream) {
    sk::kv_stage_from_host(attn_of(stage), host_of(host), fmt, n_blocks, shapes_of(s), stream);
}
void tk_qwen4exp_p2_kv_unstage_to_host(const tk_qwen4exp_p2_attn_pools_t* stage,
                                       const tk_qwen4exp_p2_kv_pools_t* host, int32_t fmt, int64_t b0, int64_t b1,
                                       const tk_qwen4exp_p2_qsa_shapes_t* s, void* stream) {
    sk::kv_unstage_to_host(attn_of(stage), host_of(host), fmt, b0, b1, shapes_of(s), stream);
}
void tk_qwen4exp_p2_kv_append(const float* k, const float* v, int64_t t, int64_t pos0, const int32_t* page_table,
                              int64_t page_size, uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q,
                              uint16_t* k_scale, uint16_t* v_scale, void* stream,
                              const tk_qwen4exp_p2_kv_pools_t* host, const tk_qwen4exp_p2_kv_pools_t* stage) {
    sk::KvHostPools h, st;
    if (host) h = host_of(host);
    if (stage) st = host_of(stage);
    pf::kv_append(k, v, t, pos0, page_table, page_size, k_pool, v_pool, k_q, v_q, k_scale, v_scale, stream,
                  host ? &h : nullptr, stage ? &st : nullptr);
}
void tk_qwen4exp_p2_kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t t,
                                 const float* k, const float* v, const tk_qwen4exp_p2_qsa_shapes_t* s, void* stream,
                                 const tk_qwen4exp_p2_kv_pools_t* host, const tk_qwen4exp_p2_kv_pools_t* stage) {
    sk::KvHostPools h, st;
    if (host) h = host_of(host);
    if (stage) st = host_of(stage);
    sk::kv_append_q4(k_q4, v_q4, page_table, pos0, t, k, v, shapes_of(s), stream, host ? &h : nullptr,
                     stage ? &st : nullptr);
}
int tk_qwen4exp_p2_indexer_append_batch(const float* raw, int64_t n, int64_t p0, const float* gamma, float eps,
                                        const tk_qwen4exp_p2_idx_bufs_t* b, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                        int64_t max_cells, void* stream, char* err, size_t err_cap) {
    const sk::QsaIndexerBuffers ib{b->tail, b->dead, b->pooled, b->block_pos};
    try {
        sk::native_qsa_indexer_append_batch(raw, n, p0, 0, gamma, eps, ib, shapes_of(s), max_cells, sk::rope_scaling(),
                                            stream);
    } catch (const std::exception& e) {
        put_err(err, err_cap, (std::string("prefill indexer: ") + e.what()).c_str());
        return 0;
    }
    return 1;
}
int tk_qwen4exp_p2_indexer_append(const float* raw, const int32_t* rel_pos_dev, const float* gamma, float eps,
                                  const tk_qwen4exp_p2_idx_bufs_t* b, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                  int64_t max_cells, void* stream, char* err, size_t err_cap) {
    const sk::QsaIndexerBuffers ib{b->tail, b->dead, b->pooled, b->block_pos};
    try {
        sk::native_qsa_indexer_append(raw, rel_pos_dev, 0, gamma, eps, ib, shapes_of(s), max_cells, sk::rope_scaling(),
                                      stream);
    } catch (const std::exception& e) {
        put_err(err, err_cap, (std::string("prefill indexer: ") + e.what()).c_str());
        return 0;
    }
    return 1;
}
int tk_qwen4exp_p2_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                                   int64_t nq, int64_t max_blocks, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                   float* scores, void* stream, int64_t active) {
    return sk::qsa_block_scores_tc(pooled, dead, q_idx, steps, nq, max_blocks, shapes_of(s), scores, stream, active) ? 1 : 0;
}
void tk_qwen4exp_p2_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                                 int64_t nq, int64_t max_blocks, const tk_qwen4exp_p2_qsa_shapes_t* s, float* scores,
                                 void* stream, int64_t active) {
    sk::qsa_block_scores(pooled, dead, q_idx, steps, nq, max_blocks, shapes_of(s), scores, stream, active);
}
void tk_qwen4exp_p2_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                               int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s, int32_t* ids, void* stream,
                               int64_t active) {
    sk::qsa_block_topk(scores, steps, nq, max_blocks, cap, shapes_of(s), ids, stream, active);
}
int tk_qwen4exp_p2_fnp_tile_attn(const float* q, const tk_qwen4exp_p2_attn_pools_t* pools, const int32_t* ids,
                                 const int32_t* steps, int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s, float* attn,
                                 int64_t n_q, void* stream) {
    // its union scratch: one device buffer per process, grown to the largest chunk's need (the prompt stream is
    // serial) - prefill.cpp's fnp_tile_scratch
    static void* p = nullptr;
    static uint64_t have = 0;
    const uint64_t bytes = sk::fnp_qsa_tile_scratch_bytes(n_q, cap);
    if (bytes > have) {
        if (p != nullptr) { cudaDeviceSynchronize(); cudaFree(p); p = nullptr; have = 0; }
        if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); p = nullptr; return 0; }
        have = bytes;
    }
    return sk::fnp_qsa_tile_attn(q, attn_of(pools), ids, steps, cap, shapes_of(s), attn, n_q, p, stream) ? 1 : 0;
}
int tk_qwen4exp_p2_prompt_attn_batch(const float* q, const tk_qwen4exp_p2_attn_pools_t* pools, const int32_t* ids,
                                     const int32_t* steps, int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                     float* attn, int64_t n_q, void* stream) {
    return sk::qsa_prompt_attn_batch(q, attn_of(pools), ids, steps, cap, shapes_of(s), attn, n_q, stream) ? 1 : 0;
}
void tk_qwen4exp_p2_decode_attn_batch(const float* q, const tk_qwen4exp_p2_attn_pools_t* pools, const int32_t* ids,
                                      const int32_t* steps, int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                      float* scratch, float* attn, int64_t n_q, void* stream) {
    sk::qsa_decode_attn_batch(q, attn_of(pools), ids, steps, cap, shapes_of(s), scratch, attn, n_q, stream);
}
uint64_t tk_qwen4exp_p2_decode_attn_scratch_floats(int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s) {
    return sk::qsa_decode_attn_scratch_floats(cap, shapes_of(s));
}

// ---------------------------------------------------------------- MoE
void tk_qwen4exp_p2_route(const float* logits, int32_t* ids, float* w, int64_t t, int64_t n_expert, void* stream) {
    pf::route(logits, ids, w, t, n_expert, stream);
}
void tk_qwen4exp_p2_swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    pf::swiglu_pair(g, u, h16, n, stream);
}
void tk_qwen4exp_p2_swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    pf::swiglu_interleaved(gu, h16, n, stream);
}
void tk_qwen4exp_p2_copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) { pf::copy_i32(dst, src, n, stream); }
void tk_qwen4exp_p2_gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width,
                                  void* stream) {
    pf::gather_rows16(x16, src, dst16, n, width, stream);
}
void tk_qwen4exp_p2_moe_combine(const float* d, const int32_t* slot, const float* w, const float* shared,
                                const float* sg, float* bo, int64_t t, void* stream, const float* row_sd) {
    pf::moe_combine(d, slot, w, shared, sg, bo, t, stream, row_sd);
}
int tk_qwen4exp_p2_expert_fmt(int64_t layer, tk_qwen4exp_p2_expert_fmt_t* out) {
    *out = tk_qwen4exp_p2_expert_fmt_t{};
    const sk::cpu::ExpertLayout& lay = sk::cpu::expert_layout();
    out->native = lay.native;
    out->blob_bytes = lay.blob_bytes(layer);
    if (!lay.native) {
        out->gu_type = 42;
        out->d_type = 42;
        return 1;
    }
    if (layer < 0 || layer >= (int64_t) lay.fmt.size()) return 0;
    const sk::cpu::NativeFmt& f = lay.fmt[(size_t) layer];
    out->gu_type = f.gu_type;
    out->d_type = f.d_type;
    out->n_embd = f.n_embd;
    out->n_ff = f.n_ff;
    out->gu_row = f.gu_row;
    out->d_row = f.d_row;
    out->up_off = f.up_off;
    out->down_off = f.down_off;
    out->tail_off = f.tail_off;
    out->native_supported = pf::fused::native_supported(f.gu_type, f.d_type);
    return 1;
}
int tk_qwen4exp_p2_fused_native_supported(int32_t gu_type, int32_t d_type) {
    return pf::fused::native_supported(gu_type, d_type) ? 1 : 0;
}
int tk_qwen4exp_p2_fused_enabled(void) { return pf::fused::enabled() ? 1 : 0; }
void tk_qwen4exp_p2_fused_quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream,
                                       int32_t native) {
    if (native) pf::fused::quantize_act_native(x, rows, cols, xa, stream);
    else pf::fused::quantize_act(x, rows, cols, xa, stream);
}
void tk_qwen4exp_p2_fused_group(const int32_t* ids, int64_t n, int32_t k, int32_t n_expert, void* scratch,
                                int32_t* slot, int32_t* src, void* stream) {
    pf::fused::group(ids, n, k, n_expert, scratch, slot, src, stream);
}
void tk_qwen4exp_p2_fused_experts(int32_t native, const tk_qwen4exp_p2_expert_fmt_t* fmt, int32_t e0, int32_t e1,
                                  const uint8_t* const* blobs, int32_t n_expert, int64_t n, const void* scratch,
                                  const void* xa, const int32_t* src, void* ha, float* dm, void* stream) {
    pf::fused::Batch b;
    b.e0 = e0;
    b.e1 = e1;
    for (int32_t e = e0; e < e1; ++e) b.blob[e - e0] = blobs[e - e0];
    if (native) {
        const pf::fused::NativeGeom ng{fmt->gu_type, fmt->d_type, (size_t) fmt->gu_row, (size_t) fmt->d_row,
                                       (size_t) fmt->up_off, (size_t) fmt->down_off};
        pf::fused::experts_native(b, ng, n_expert, n, scratch, xa, src, ha, dm, stream);
    } else {
        pf::fused::experts(b, n_expert, n, scratch, xa, src, ha, dm, stream);
    }
}
int32_t tk_qwen4exp_p2_fused_max_batch(void) { return pf::fused::kMaxBatch; }
uint64_t tk_qwen4exp_p2_mmq_matrix_bytes(int32_t type, int64_t rows, int64_t cols) {
    return pf::mmq::matrix_bytes(type, rows, cols);
}
int tk_qwen4exp_p2_mmq_fp4_activations(int32_t type) { return pf::mmq::fp4_activations(type) ? 1 : 0; }
int tk_qwen4exp_p2_mmq_fits(int32_t type, int64_t rows) { return pf::mmq::fits(type, rows) ? 1 : 0; }
void tk_qwen4exp_p2_mmq_quantize(const float* x, const int32_t* ids, void* xq, int32_t type, int64_t cols, int64_t ld,
                                 int64_t rows, void* stream, float* yscale) {
    pf::mmq::quantize(x, ids, xq, type, cols, ld, rows, stream, yscale);
}
void tk_qwen4exp_p2_mmq_gather_native(const void* gate, const void* up, uint64_t gu_half, const void* down,
                                      uint64_t d_bytes, void* gu_dst, void* d_dst, void* stream, const void* tail,
                                      void* tail_dst, uint64_t zero_bytes) {
    pf::mmq::gather_native(gate, up, (size_t) gu_half, down, (size_t) d_bytes, gu_dst, d_dst, stream, tail, tail_dst,
                           (size_t) zero_bytes);
}
int tk_qwen4exp_p2_mmq_gather_native_group(const uint8_t* const* blobs, int32_t first, int32_t n, uint64_t up_off,
                                           uint64_t gu_half, uint64_t down_off, uint64_t d_bytes, void* gu_dst,
                                           uint64_t gu_stride, void* d_dst, uint64_t d_stride, void* stream,
                                           uint64_t tail_off, void* tail_dst, uint64_t zero_bytes) {
    pf::mmq::GatherGroup gg;
    for (int32_t i = 0; i < n && i < pf::mmq::kGatherGroupMax; ++i) gg.blob[i] = blobs[i];
    gg.first = first;
    gg.n = n;
    return pf::mmq::gather_native_group(gg, (size_t) up_off, (size_t) gu_half, (size_t) down_off, (size_t) d_bytes,
                                        gu_dst, (size_t) gu_stride, d_dst, (size_t) d_stride, stream, (size_t) tail_off,
                                        tail_dst, (size_t) zero_bytes) ? 1 : 0;
}
void tk_qwen4exp_p2_mmq_gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    pf::mmq::gather_strata_q2(blob, gu_dst, d_dst, stream);
}
void tk_qwen4exp_p2_mmq_product(void* mmq_ctx, const void* w, int32_t type, int64_t w_rows, int64_t w_cols,
                                uint64_t expert_bytes, int32_t n, const void* xq, const int32_t* bounds,
                                const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst, int64_t ld_dst,
                                const float* y_scale, int32_t allow_tp_moe, void* stream) {
    pf::mmq::Product p;
    p.w = w; p.type = type; p.w_rows = w_rows; p.w_cols = w_cols; p.expert_bytes = (size_t) expert_bytes; p.n = n;
    p.xq = xq; p.bounds = bounds; p.ids = ids; p.total_rows = total_rows; p.max_rows = max_rows; p.dst = dst;
    p.ld_dst = ld_dst; p.y_scale = y_scale;
#if defined(STRATA_USE_HIP) && defined(STRATA_PREFILL_MMQ)
    // prefill.cpp's run_mmq_product: tp_moe where it covers the product (TALUS_PF_MOE=0 turns it off)
    static const bool use_tp_moe = [] {
        const char* v = std::getenv("TALUS_PF_MOE");
        return v == nullptr || v[0] != '0';
    }();
    if (allow_tp_moe && use_tp_moe && p.n >= 1 && p.y_scale == nullptr &&
        talus::prefill::moe_supported(p.type, (int) p.w_rows, (int) p.w_cols)) {
        talus::prefill::MoeProduct mp;
        mp.w = p.w;
        mp.type = p.type;
        mp.expert_bytes = p.expert_bytes;
        mp.n = (int) p.w_rows;
        mp.k = (int) p.w_cols;
        mp.n_experts = p.n;
        mp.xq = p.xq;
        mp.bounds = p.bounds;
        mp.ids = p.ids;
        mp.total_rows = p.total_rows;
        mp.max_rows = p.max_rows;
        mp.dst = p.dst;
        mp.ld_dst = p.ld_dst;
        if (talus::prefill::moe_gemm(mp, stream)) return;
    }
#else
    (void) allow_tp_moe;
#endif
    ((pf::mmq::Context*) mmq_ctx)->run(p, stream);
}
void tk_qwen4exp_p2_mmq_swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, int32_t interleaved,
                               void* stream) {
    pf::mmq::swiglu(gu, h, rows, n_ff, interleaved != 0, stream);
}
void tk_qwen4exp_p2_mmq_swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds,
                                      int32_t n, const float* tails, int64_t row0, void* stream) {
    pf::mmq::swiglu_scaled(gu, h, rows, n_ff, bounds, n, tails, row0, stream);
}
void tk_qwen4exp_p2_mmq_down_row_scales(float* sd, const int32_t* bounds, int32_t n, const float* tails,
                                        int64_t nrows, void* stream) {
    pf::mmq::down_row_scales(sd, bounds, n, tails, nrows, stream);
}
void tk_qwen4exp_p2_mmq_iota(int32_t* dst, int64_t n, void* stream) { pf::mmq::iota(dst, n, stream); }
uint64_t tk_qwen4exp_p2_mmq_q8_bytes(int64_t rows, int64_t cols) { return pf::mmq::q8_bytes(rows, cols); }
void tk_qwen4exp_p2_iq_dequant_gu_f16(int32_t type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd,
                                      uint16_t* dst, void* stream, const float* scale) {
    sk::iq_dequant_gu_f16(type, gate, up, n_ff, n_embd, dst, stream, scale);
}
void tk_qwen4exp_p2_iq_dequant_f16(int32_t type, const void* src, int64_t n, uint16_t* dst, void* stream,
                                   const float* scale) {
    sk::iq_dequant_f16(type, src, n, dst, stream, scale);
}
void tk_qwen4exp_p2_blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    pf::blob_dequant_f16(blob, gu16, down16, stream);
}

// ---------------------------------------------------------------- expert source / cache
int tk_qwen4exp_p2_src_transient(void* src, int64_t l, int64_t e) { return ((sc::ExpertSource*) src)->transient(l, e) ? 1 : 0; }
int tk_qwen4exp_p2_src_pinned(void* src, int64_t l, int64_t e) { return ((sc::ExpertSource*) src)->pinned(l, e) ? 1 : 0; }
const uint8_t* tk_qwen4exp_p2_src_blob_stable(void* src, int64_t l, int64_t e) {
    return ((sc::ExpertSource*) src)->blob_stable(l, e);
}
const uint8_t* tk_qwen4exp_p2_cache_device_slot(void* cache, int32_t slot) {
    return ((const sc::ExpertCache*) cache)->device_slot(slot);
}

// ---------------------------------------------------------------- MTP draft layer
int tk_qwen4exp_p2_mtp_info(void* mtp, tk_qwen4exp_p2_mtp_info_t* out) {
    *out = tk_qwen4exp_p2_mtp_info_t{};
    sc::MtpDrafter& d = *(sc::MtpDrafter*) mtp;
    out->w_ne = d.tensor_f32("pre_fc_norm_embedding.weight");
    out->w_fe = d.tensor_q8("fc_embedding.weight");
    out->w_nh = d.tensor_f32("pre_fc_norm_hidden.weight");
    out->w_fh = d.tensor_q8("fc_hidden.weight");
    out->w_hn = d.tensor_f32("attn_hyper_connection.hc_norm.weight");
    out->w_dn = d.tensor_bf16("attn_hyper_connection.input_mix_weight_down.weight");
    out->w_up = d.tensor_bf16("attn_hyper_connection.input_mix_weight_up.weight");
    out->w_k = d.tensor_q8("self_attn.k_proj.weight");
    out->w_v = d.tensor_q8("self_attn.v_proj.weight");
    out->w_kn = d.tensor_f32("self_attn.k_norm.weight");
    out->first_needed = d.first_needed();
    out->device = d.device();
    out->hnorm_per_stream = d.hnorm_per_stream();
    qsa_info_of(d.kv_state_rw(), &out->kv);
    return 1;
}
int tk_qwen4exp_p2_mtp_idle(void* mtp, char* err, size_t err_cap) {
    std::string e;
    if (!((sc::MtpDrafter*) mtp)->idle(e)) {
        put_err(err, err_cap, e.c_str());
        return 0;
    }
    return 1;
}
void tk_qwen4exp_p2_mtp_add_ms_prefill(void* mtp, double ms) { ((sc::MtpDrafter*) mtp)->ms_prefill += ms; }
void tk_qwen4exp_p2_rms_norm_grouped(const float* x, const float* gamma, float* y, int32_t n, int32_t groups,
                                     int32_t rows, float eps, void* stream) {
    sk::native_qsa_rms_norm_grouped(x, gamma, y, n, groups, rows, eps, stream);
}
void tk_qwen4exp_p2_add_streams_broadcast(const float* h, const float* e, float* r, int64_t n_embd, int32_t hc,
                                          int32_t n_tok, void* stream) {
    sk::add_streams_broadcast(h, e, r, n_embd, hc, n_tok, stream);
}
void tk_qwen4exp_p2_mmq_run(void* mmq_ctx, const void* w, int32_t type, int64_t w_rows, int64_t w_cols,
                            uint64_t expert_bytes, int32_t n, const void* xq, const int32_t* bounds, const int32_t* ids,
                            int64_t total_rows, int64_t max_rows, float* dst, int64_t ld_dst, void* stream) {
    pf::mmq::Product p;
    p.w = w; p.type = type; p.w_rows = w_rows; p.w_cols = w_cols; p.expert_bytes = (size_t) expert_bytes; p.n = n;
    p.xq = xq; p.bounds = bounds; p.ids = ids; p.total_rows = total_rows; p.max_rows = max_rows; p.dst = dst;
    p.ld_dst = ld_dst;
    ((pf::mmq::Context*) mmq_ctx)->run(p, stream);
}

}  // extern "C"
