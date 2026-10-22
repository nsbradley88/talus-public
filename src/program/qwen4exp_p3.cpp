// src/program/qwen4exp_p3.cpp - qwen4exp Piece 3: launch-only wrappers for the Rust decode window recorder.
//
// The Rust side (rust/crates/talus-model/src/qwen4exp/decode.rs) owns the ORDER of a decode window: which layer,
// which group of rows, which variant of each step, and every buffer offset. Each function here enqueues exactly the
// kernel(s) Verifier::record_window enqueues at the same point - no control flow beyond picking the one kernel the
// Rust caller named - so a window recorded from Rust is the same graph as the C++ one, node for node.
#include "qwen4exp_r2.hpp"   // qwen4exp-r2: the layer.cpp launch helpers as thin wrappers
#include "talus_qwen4exp_kernels.h"

#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/session.hpp"
#include "strata/core/verify.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/pdl.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace strata::core {
// verify.cpp's hook (qwen4exp-p3 block): the registered Rust recorder, consulted by Verifier::capture
using Qwen4expP3Recorder = int (*)(void* user, void* ver, int t, void* stream, char* err, size_t cap);
extern Qwen4expP3Recorder g_qwen4exp_p3_recorder;
extern void* g_qwen4exp_p3_user;

/// Friend of Verifier (verify.hpp): reads the recorder's buffers and state, writes `groups_`.
struct Qwen4expP3Access {
    static void view(Verifier& v, tk_qwen4exp_p3_window_view_t& o) {
        using namespace strata::kernels;
        const ModelGeometry& g = *v.g_;
        SessionState& ss = *v.ss_;
        std::memset(&o, 0, sizeof o);
        o.ver = &v; o.ss = &ss; o.wt = v.wt_; o.g = &g;
        o.n_embd = g.n_embd; o.hc = g.hc; o.hc_lr = g.hc_lr; o.k = ss.k; o.conv_channels = g.ssm_conv_channels;
        o.value_dim = g.ssm_value_dim; o.v_heads = g.ssm_v_heads; o.k_heads = g.ssm_k_heads;
        o.state_size = g.ssm_state_size; o.d_conv = g.ssm_d_conv;
        o.n_head = g.n_head; o.head_dim = g.head_dim; o.n_head_kv = g.n_head_kv; o.idx_q_heads = g.idx_q_heads;
        o.idx_key_dim = g.idx_key_dim; o.n_expert = g.n_expert; o.n_ff = g.n_ff; o.n_layers = g.n_layers;
        o.n_vocab = v.n_vocab_;
        const QsaShapes s = qsa_shapes(g);
        o.idx_block = s.idx_block; o.n_rot = s.n_rot; o.step_count = kStepCount; o.step_pos = kStepPos;
        o.ng_hist = NG_HIST; o.ng_hc_dim = NG_HC_DIM; o.ng_n_embd = NG_N_EMBD;
        o.max_t = v.max_t_; o.plan_i32 = v.plan_i32_; o.cap = v.cap_; o.max_blocks = v.max_blocks_;
        o.attn_scratch_floats = v.attn_scratch_floats_; o.lb = v.lb_; o.le = v.le_; o.gdn_ord0 = ss.gdn_ord0;
        o.n_qsa_layers = g.n_qsa_layers();
        o.staging_blobs = Verifier::kStagingBlobs; o.pcie_group_rows = Verifier::kPcieGroupRows;
        o.gdn_floats = (int64_t) ((uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
                                  (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1));
        o.max_blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        o.ple_block_scratch_bytes = (int64_t) ple_block_scratch_bytes();
        o.eps = 1e-6f;
        o.batch_rec = v.batch_rec_; o.split = v.split_; o.prof_on = v.prof_on_; o.trace_on = v.trace_m_ != nullptr;
        o.ar_on = v.ar_on(); o.device_plan = v.device_plan_; o.remote_opt = v.remote_opt_ != nullptr;
        o.always_publish = v.always_publish_; o.df_branch = v.df_branch_;
        o.head_loaded = v.head_ != nullptr && v.head_->loaded();
        o.head_type = o.head_loaded ? v.head_->type() : -1;
        o.head_weights = o.head_loaded ? v.head_->weights() : nullptr;
        o.ple_ready = ss.ple.ready(); o.ple_stage = v.ple_stage();
        o.native_layout = strata::kernels::cpu::expert_layout().native;
        o.pcie_mode = v.sink_.pcie_mode;
        o.sh_fork_ready = v.sh_cs_ != nullptr && v.ev_fork_ != nullptr && v.ev_join_ != nullptr;
        o.ple_key_bf16 = ss.ple.w.key_bf16 != nullptr; o.ple_value_bf16 = ss.ple.w.value_bf16 != nullptr;
        o.ple_key_native = ss.ple.w.key_native_data != nullptr;
        o.has_hand_in = v.hand_in_ != nullptr;
        o.cs = v.cs_; o.sh_cs = v.sh_cs_; o.ev_fork = v.ev_fork_; o.ev_join = v.ev_join_;
        o.tok = v.tok_; o.step = v.step_; o.pos = v.pos_; o.m_tok = v.m_tok_; o.m_step = v.m_step_; o.m_pos = v.m_pos_;
        o.ple = v.ple_; o.emb = v.emb_; o.R = v.R_; o.mixed = v.mixed_; o.bo = v.bo_; o.inj = v.inj_; o.inj2 = v.inj2_;
        o.lo = v.lo_; o.rs = v.rs_; o.xn = v.xn_; o.m_ple = v.m_ple_;
        o.xq = v.xq_; o.xil = v.xil_; o.sh_xq = v.sh_xq_;
        o.qkv_L = v.qkv_L_; o.h_L = v.h_L_; o.gate_L = v.gate_L_; o.beta_L = v.beta_L_; o.z = v.z_; o.y = v.y_;
        o.qfull = v.qfull_; o.qcur = v.qcur_; o.kcur = v.kcur_; o.vcur = v.vcur_; o.idx_raw_L = v.idx_raw_L_;
        o.qidx = v.qidx_; o.scores = v.scores_; o.attn = v.attn_; o.attn32 = v.attn32_; o.attn_scratch = v.attn_scratch_;
        o.tail_snap = v.tail_snap_; o.sel = v.sel_;
        o.logits = v.logits_; o.w = v.w_; o.shared = v.shared_; o.parts = v.parts_; o.hit_out = v.hit_out_;
        o.ids = v.ids_; o.plan = v.plan_; o.staging = v.staging_; o.hit_xq = v.hit_xq_; o.nat_xq = v.nat_xq_;
        o.hit_xs = v.hit_xs_; o.hit_scratch = v.hit_scratch_;
        o.head_mixed = v.head_mixed_; o.head_inj = v.head_inj_; o.head_logits = v.head_logits_; o.sh_bf16 = v.sh_bf16_;
        o.arg_scratch = v.arg_scratch_; o.one = v.one_;
        o.sh_gate = v.sh_gate_; o.sh_up = v.sh_up_; o.sh_g = v.sh_g_; o.hist_snap = v.hist_snap_;
        o.qcnt = v.qcnt_; o.skip = v.skip_; o.slot_off_d = v.slot_off_d_; o.m_plan_err = v.m_plan_err_;
        o.m_x = v.m_x_; o.m_ids = v.m_ids_; o.m_w = v.m_w_; o.m_seq = v.m_seq_; o.m_flag = v.m_flag_;
        o.m_flagA = v.m_flagA_; o.m_flagB = v.m_flagB_; o.m_plan = v.m_plan_; o.m_ymiss = v.m_ymiss_;
        o.m_out = v.m_out_; o.m_nf = v.m_nf_;
        o.gdn_state = ss.gdn_state; o.ple_hist = ss.ple.hist; o.ple_scratch = ss.ple.scratch;
        o.d_res = v.hits_.d_res; o.cache_base = v.hits_.cache_base; o.blob = v.hits_.blob;
        o.ple_key = v.ple_key_; o.ple_val = v.ple_val_;
        o.ple_w_key_bf16 = ss.ple.w.key_bf16; o.ple_w_value_bf16 = ss.ple.w.value_bf16;
        o.ple_w_key_native = ss.ple.w.key_native_data; o.ple_w_key_native_type = ss.ple.w.key_native_type;
    }
    static void set_groups(Verifier& v, int t, int g) {
        if (t >= 0 && t < 9 && !v.batch_rec_) v.groups_[t] = g;
    }
};
}  // namespace strata::core

namespace {
using namespace strata::core;
using namespace strata::kernels;

thread_local std::string g_err;

#define P3_TRY(...)                                                     \
    try {                                                               \
        __VA_ARGS__;                                                    \
        return 0;                                                       \
    } catch (const std::exception& e) {                                 \
        g_err = e.what();                                               \
        return -1;                                                      \
    }

const ModelGeometry& G(const void* g) { return *(const ModelGeometry*) g; }
SessionState& SS(void* ss) { return *(SessionState*) ss; }
const QsaState& QST(void* ss, int64_t qi) { return SS(ss).qsa_states[qi]; }

int recorder_tramp(void* user, void* ver, int t, void* stream, char* err, size_t cap);
tk_qwen4exp_p3_recorder_fn g_rust_fn = nullptr;
void* g_rust_user = nullptr;

int recorder_tramp(void* user, void* ver, int t, void* stream, char* err, size_t cap) {
    return g_rust_fn ? g_rust_fn(user, ver, t, stream, err, cap) : 0;
}

bool gfx12_device() {
#if defined(STRATA_USE_HIP)
    int dev = 0;
    cudaDeviceProp p;
    return cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
           gfx_arch_is_gfx12(p.gcnArchName);
#else
    return false;
#endif
}
}  // namespace

extern "C" {

void tk_qwen4exp_p3_set_window_recorder(tk_qwen4exp_p3_recorder_fn fn, void* user) {
    g_rust_fn = fn;
    g_rust_user = user;
    strata::core::g_qwen4exp_p3_user = user;
    strata::core::g_qwen4exp_p3_recorder = fn ? &recorder_tramp : nullptr;
}

int tk_qwen4exp_p3_last_error(char* buf, size_t cap) {
    if (buf == nullptr || cap == 0) return (int) g_err.size();
    std::snprintf(buf, cap, "%s", g_err.c_str());
    return (int) g_err.size();
}

uint64_t tk_qwen4exp_p3_runtime_flags(void) {
    uint64_t f = 0;
    if (native_router_enabled()) f |= 1ull << 0;
    if (native_moe_combine_enabled()) f |= 1ull << 1;
    if (shared_expert_native_bf16_enabled()) f |= 1ull << 2;
    if (native_qsa_enabled()) f |= 1ull << 3;
    if (native_rope_enabled()) f |= 1ull << 4;
    if (ple_native_bf16_enabled()) f |= 1ull << 5;
    if (ple_native_postops_enabled()) f |= 1ull << 6;
    if (gr_native_mmvf_enabled()) f |= 1ull << 7;
    if (argmax_rows_wanted()) f |= 1ull << 8;
    if (Verifier::nan_guard_on()) f |= 1ull << 9;
    if (pdl_supported()) f |= 1ull << 10;
    if (gfx12_device()) f |= 1ull << 11;
#if defined(STRATA_USE_HIP)
    f |= 1ull << 12;
#endif
#if defined(STRATA_HIP_GFX906)
    f |= 1ull << 14;
#endif
    return f;
}

int tk_qwen4exp_p3_window_view(void* ver, tk_qwen4exp_p3_window_view_t* out) {
    if (ver == nullptr || out == nullptr) return -1;
    Qwen4expP3Access::view(*(Verifier*) ver, *out);
    return 0;
}

int tk_qwen4exp_p3_set_groups(void* ver, int t, int g) {
    if (ver == nullptr) return -1;
    Qwen4expP3Access::set_groups(*(Verifier*) ver, t, g);
    return 0;
}

int tk_qwen4exp_p3_qsa_view(void* ss, int64_t qi, tk_qwen4exp_p3_qsa_view_t* o) {
    if (ss == nullptr || o == nullptr) return -1;
    const QsaState& st = QST(ss, qi);
    std::memset(o, 0, sizeof *o);
    o->idx_tail = st.idx_tail; o->idx_dead = st.idx_dead; o->idx_pooled = st.idx_pooled;
    o->idx_block_pos = st.idx_block_pos; o->cos_tab = st.cos_tab; o->sin_tab = st.sin_tab; o->max_cells = st.max_cells;
    o->kv_rot = st.kv_rot; o->kv_hybrid = st.kv_hybrid; o->kv_q4 = st.kv_q4; o->kv_int8 = st.kv_int8;
    return 0;
}

int tk_qwen4exp_p3_layer_weights(const void* wt_, int32_t layer, int32_t is_qsa, tk_qwen4exp_p3_layer_weights_t* o) {
    const WeightTable& wt = *(const WeightTable*) wt_;
    const LayerView v(wt, layer);
    std::memset(o, 0, sizeof *o);
    o->is_qsa = is_qsa;
    std::string missing;
    auto need = [&](const char* suffix) -> const WeightRef* {
        const WeightRef* r = v.get(suffix);
        if (r == nullptr && missing.empty()) missing = v.name(suffix) + " is missing";
        return r;
    };
    auto native = [&](const WeightRef* r, const char* suffix) -> bool {
        if (r == nullptr) return false;
        if (r->native_data == nullptr) {
            if (missing.empty()) missing = "verify: " + v.name(suffix) + " is not served natively (run with --native)";
            return false;
        }
        return true;
    };
    const char* pfx[2] = {"hc_attn_", "hc_ffn_"};
    for (int h = 0; h < 2; ++h) {
        const WeightRef* n = need((std::string(pfx[h]) + "norm.weight").c_str());
        const WeightRef* d = need((std::string(pfx[h]) + "down.weight").c_str());
        const WeightRef* u = need((std::string(pfx[h]) + "up.weight").c_str());
        const WeightRef* i = need((std::string(pfx[h]) + "inject.weight").c_str());
        if (!n || !d || !u || !i) { g_err = missing; return -1; }
        o->hc_norm[h] = (const float*) n->data; o->hc_down[h] = (const uint16_t*) d->data;
        o->hc_up[h] = (const uint16_t*) u->data; o->hc_inject[h] = (const uint16_t*) i->data;
        o->hc_q8_down[h] = (const uint8_t*) d->hc_q8; o->hc_q8_up[h] = (const uint8_t*) u->hc_q8;
        o->hc_q8_inject[h] = (const uint8_t*) i->hc_q8;
    }
    if (!is_qsa) {
        const WeightRef *wqkv = need("attn_qkv.weight"), *wg = need("attn_gate.weight"), *wout = need("ssm_out.weight"),
                        *wa = need("ssm_alpha.weight"), *wb = need("ssm_beta.weight"), *wc = need("ssm_conv1d.weight"),
                        *wnm = need("ssm_norm.weight"), *wdt = need("ssm_dt.bias"), *wsa = need("ssm_a");
        if (!wqkv || !wg || !wout || !wa || !wb || !wc || !wnm || !wdt || !wsa) { g_err = missing; return -1; }
        if (!native(wqkv, "attn_qkv.weight") || !native(wg, "attn_gate.weight") || !native(wout, "ssm_out.weight")) {
            g_err = missing;
            return -1;
        }
        o->qkv_type = wqkv->native_type; o->qkv = wqkv->native_data;
        o->gate_type = wg->native_type; o->gate = wg->native_data;
        o->out_type = wout->native_type; o->out = wout->native_data;
        o->alpha = (const uint16_t*) wa->data; o->beta = (const uint16_t*) wb->data;
        o->conv1d = (const float*) wc->data; o->ssm_norm = (const float*) wnm->data;
        o->dt = (const float*) wdt->data; o->ssm_a = (const float*) wsa->data;
    } else {
        const WeightRef *wik = need("indexer.k_proj.weight"), *wq = need("attn_q.weight"), *wk = need("attn_k.weight"),
                        *wv = need("attn_v.weight"), *wo = need("attn_output.weight"), *wiq = need("indexer.q_proj.weight"),
                        *wqn = need("attn_q_norm.weight"), *wkn = need("attn_k_norm.weight"),
                        *wiqn = need("indexer.q_norm.weight"), *wikn = need("indexer.k_norm.weight");
        if (!wik || !wq || !wk || !wv || !wo || !wiq || !wqn || !wkn || !wiqn || !wikn) { g_err = missing; return -1; }
        if (!native(wq, "attn_q.weight") || !native(wk, "attn_k.weight") || !native(wv, "attn_v.weight") ||
            !native(wo, "attn_output.weight")) {
            g_err = missing;
            return -1;
        }
        o->idx_k = (const uint16_t*) wik->data; o->idx_q = (const uint16_t*) wiq->data;
        o->q_type = wq->native_type; o->q = wq->native_data; o->k_type = wk->native_type; o->k = wk->native_data;
        o->v_type = wv->native_type; o->v = wv->native_data; o->o_type = wo->native_type; o->o = wo->native_data;
        o->q_norm = (const float*) wqn->data; o->k_norm = (const float*) wkn->data;
        o->iq_norm = (const float*) wiqn->data; o->ik_norm = (const float*) wikn->data;
    }
    const WeightRef* wr = v.get("ffn_gate_inp.weight");
    o->router = wr ? (const uint16_t*) wr->data : nullptr;
    const WeightRef *wgi = need("ffn_gate_inp_shexp.weight"), *wsg = need("ffn_gate_shexp.weight"),
                    *wsu = need("ffn_up_shexp.weight"), *wsd = need("ffn_down_shexp.weight");
    if (!wgi || !wsg || !wsu || !wsd) { g_err = missing; return -1; }
    if (!native(wsg, "ffn_gate_shexp.weight") || !native(wsu, "ffn_up_shexp.weight") ||
        !native(wsd, "ffn_down_shexp.weight")) {
        g_err = missing;
        return -1;
    }
    o->shexp_gate_inp = (const uint16_t*) wgi->data;
    o->sg_type = wsg->native_type; o->sg = wsg->native_data;
    o->su_type = wsu->native_type; o->su = wsu->native_data;
    o->sd_type = wsd->native_type; o->sd = wsd->native_data;
    return 0;
}

int tk_qwen4exp_p3_head_weights(const void* wt_, tk_qwen4exp_p3_head_weights_t* o) {
    const WeightTable& wt = *(const WeightTable*) wt_;
    std::memset(o, 0, sizeof *o);
    const WeightRef *hn = wt.find("output_hc_norm.weight"), *hd = wt.find("output_hc_down.weight"),
                    *hu = wt.find("output_hc_up.weight");
    if (!hn || !hd || !hu) { g_err = "verify: an output_hc_* weight is missing"; return -1; }
    o->norm = (const float*) hn->data; o->down = (const uint16_t*) hd->data; o->up = (const uint16_t*) hu->data;
    o->q8_down = (const uint8_t*) hd->hc_q8; o->q8_up = (const uint8_t*) hu->hc_q8;
    o->forms_ok = hn->kind == WeightKind::F32 && hd->kind == WeightKind::Bf16InF32 && hu->kind == WeightKind::Bf16InF32;
    return 0;
}

int tk_qwen4exp_p3_pdl_scope(int on) {
    const bool prev = pdl_scope();
    pdl_scope() = on != 0;
    return prev ? 1 : 0;
}

int tk_qwen4exp_p3_cvec_covers(int64_t layer) { return cvec().covers(layer) ? 1 : 0; }
int tk_qwen4exp_p3_norm_rope_usable(int head_dim, int n_rot) { return native_norm_rope_usable(head_dim, n_rot) ? 1 : 0; }
int tk_qwen4exp_p3_mmvq_il_supported(int type, int ncols, int n_out) {
    return native_mmvq_il_supported(type, ncols, n_out) ? 1 : 0;
}
int tk_qwen4exp_p3_combine_hits_supported(const float* parts, const float* shared, const float* out, int64_t n_embd,
                                          int64_t k, int n) {
    return native_moe_combine_hits_supported(parts, shared, out, n_embd, k, n) ? 1 : 0;
}
int tk_qwen4exp_p3_expert_blob_bytes(int64_t layer, int64_t* out) {
    *out = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(layer);
    return 0;
}

int tk_qwen4exp_p3_event_fork(void* ev, void* from, void* to) {
    if ((from != nullptr && cudaEventRecord((cudaEvent_t) ev, (cudaStream_t) from) != cudaSuccess) ||
        (to != nullptr && cudaStreamWaitEvent((cudaStream_t) to, (cudaEvent_t) ev, 0) != cudaSuccess)) {
        g_err = "event fork/join failed";
        return -1;
    }
    return 0;
}
int tk_qwen4exp_p3_is_qsa_layer(const void* g, int64_t layer) { return is_qsa_layer(G(g), layer) ? 1 : 0; }

int tk_qwen4exp_p3_copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* s) {
    P3_TRY(copy_i32_from_mapped(dst, src, n, s))
}
int tk_qwen4exp_p3_copy_from_mapped(float* dst, const float* src, int64_t n, void* s) {
    P3_TRY(copy_from_mapped(dst, src, n, s))
}

int tk_qwen4exp_p3_embed_rows(const void* wt_, const int32_t* tok, int t, float* emb, void* s) {
    try {
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok, t, emb, s);
            return 0;
        }
        const WeightRef* w = ((const WeightTable*) wt_)->find("token_embd.weight");
        if (w == nullptr || w->codebook_iq4nl || (w->code_bits != 2 && w->code_bits != 4 && w->code_bits != 8)) {
            g_err = "verify: token_embd.weight is missing or not an S2/S4/S8 tensor";
            return -1;
        }
        const auto* codes = (const uint8_t*) w->data;
        const auto* scales = (const float*) (codes + w->codes_bytes);
        const auto* offsets = w->has_offset ? (const float*) (codes + w->codes_bytes + w->scales_bytes) : nullptr;
        const uint64_t row_codes = (uint64_t) (w->ne0 / (8 / w->code_bits));
        const uint64_t row_groups = (uint64_t) (w->ne0 / w->group_elems);
        embedding_gather_dev(codes, scales, offsets, tok, t, w->ne0, w->code_bits, w->code_bias, w->group_elems,
                             row_codes, row_groups, emb, s);
        return 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
}

int tk_qwen4exp_p3_broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int t, void* s) {
    P3_TRY(broadcast_streams(x, R, n_embd, hc, t, s))
}
int tk_qwen4exp_p3_wait_flag_ge(const uint32_t* flag, uint32_t v, void* s) { P3_TRY(wait_flag_ge(flag, v, s)) }
int tk_qwen4exp_p3_wait_flag_ge_or(const uint32_t* flag, uint32_t v, const uint32_t* skip, void* s) {
    P3_TRY(wait_flag_ge_or(flag, v, skip, s))
}
int tk_qwen4exp_p3_bf16_gemv(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* s) {
    P3_TRY(bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, s))
}
int tk_qwen4exp_p3_bf16_gemv_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in,
                                   int64_t n_out, int n, void* s) {
    P3_TRY(bf16_gemv_fp32_mmvf_multi(x, ldx, w, y, ldy, n_in, n_out, n, s))
}
int tk_qwen4exp_p3_bf16_gemv_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                                       int64_t n_in, int64_t n_out, int n, const uint16_t* w_aux, float* y_aux,
                                       int64_t ldy_aux, void* s) {
    try {
        return bf16_gemv_fp32_mmvf_multi_aux(x, ldx, w, y, ldy, n_in, n_out, n, w_aux, y_aux, ldy_aux, s) ? 1 : 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
}
int tk_qwen4exp_p3_quantize_q8_1(const float* x, void* xq, int n_in, int ncols, void* s) {
    P3_TRY(native_quantize_q8_1(x, xq, n_in, ncols, s))
}
int tk_qwen4exp_p3_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* s) {
    P3_TRY(native_mmvq(type, w, xq, y, n_in, n_out, ncols, s))
}
int tk_qwen4exp_p3_mmvq_il(int type, const void* w, const void* xq, const void* xil, float* y, int n_in, int n_out,
                           int ncols, void* s) {
    P3_TRY(native_mmvq_il(type, w, xq, xil, y, n_in, n_out, ncols, s))
}
int tk_qwen4exp_p3_q8_1_interleave(const void* xq, void* xil, int n_in, int ncols, void* s) {
    P3_TRY(native_q8_1_interleave(xq, xil, n_in, ncols, s))
}
int tk_qwen4exp_p3_gr_write(const float* R, const float* bo, const float* inj, int64_t n_embd, int64_t hc, int64_t hc_lr,
                            float* R_out, int n, int multi, void* s) {
    const GrShapes gs{n_embd, hc, hc_lr};
    P3_TRY(if (multi) gr_write_multi(R, bo, inj, gs, R_out, n, s); else gr_write(R, bo, inj, gs, R_out, s))
}
int tk_qwen4exp_p3_gr_post_multi(const float* R, const float* bo, const float* inj, float* out, int n_embd, int hc, int n,
                                 long long r_stride, long long b_stride, long long i_stride, void* s) {
    P3_TRY(native_gr_post_multi(R, bo, inj, out, n_embd, hc, n, r_stride, b_stride, i_stride, s))
}
int tk_qwen4exp_p3_fused_gr_read_multi(const tk_qwen4exp_p3_gr_args_t* a, int n, float* xn, void* s) {
    if (n < 1 || n > kFusedGrMaxT) { g_err = "fused_gr_read_multi: bad row count"; return -1; }
    FusedGrArgs fa[kFusedGrMaxT];
    for (int i = 0; i < n; ++i) {
        const tk_qwen4exp_p3_gr_args_t& r = a[i];
        FusedGrArgs& f = fa[i];
        f.R = r.R; f.R_out = r.R_out; f.apply = r.apply != 0; f.bo_prev = r.bo_prev; f.inj_prev = r.inj_prev;
        f.w_norm = r.w_norm; f.w_down = r.w_down; f.w_up = r.w_up; f.w_inject = r.w_inject; f.eps = r.eps;
        f.lo = r.lo; f.rs = r.rs; f.inject_out = r.inject_out; f.mixed = r.mixed;
        f.q8_down = r.q8_down; f.q8_up = r.q8_up; f.q8_inject = r.q8_inject; f.q8_mixed = r.q8_mixed; f.q8_cnt = r.q8_cnt;
    }
    try {
        return fused_gr_read_multi(fa, n, xn, s) ? 1 : 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
}

int tk_qwen4exp_p3_ple_postops_batch_snap(void* ss, float* xn, float* R, const float* z, float* qn, float* gated,
                                          float* pgate, int n, float* snap, void* s) {
    SessionState& S = SS(ss);
    P3_TRY(native_ple_postops_batch_snap(xn, R, z, S.ple.hist, S.ple.w, qn, gated, pgate, n, snap, s))
}
int tk_qwen4exp_p3_ple_block(void* ss, const float* emb, float* R, float* hist, float* normalized, const float* pre_key,
                             const float* pre_value, int use_pre, void* s) {
    SessionState& S = SS(ss);
    PleWeights pw = S.ple.w;
    if (use_pre) { pw.pre_key = pre_key; pw.pre_value = pre_value; }
    PleOut po;
    po.normalized = normalized;
    po.result = R;
    P3_TRY(ple_block(emb, R, hist, pw, po, S.ple.scratch, s))
}
int tk_qwen4exp_p3_ple_block_projected(void* ss, const float* key, const float* value, float* R, float* hist,
                                       float* normalized, void* s) {
    SessionState& S = SS(ss);
    PleOut po;
    po.normalized = normalized;
    po.result = R;
    P3_TRY(ple_block_projected(key, value, R, hist, S.ple.w, po, S.ple.scratch, s))
}
int tk_qwen4exp_p3_ple_history_advance(float* hist, const float* normalized, void* s) {
    P3_TRY(ple_history_advance(hist, normalized, s))
}

int tk_qwen4exp_p3_gdn_ab_multi(const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt, const float* sa,
                                float* gate, float* beta, int n_embd, int h_v, int n, void* s) {
    P3_TRY(gdn_ab_multi(x, wa, wb, dt, sa, gate, beta, n_embd, h_v, n, s))
}
int tk_qwen4exp_p3_gdn_conv_ab_multi(const float* hist, const float* qkv, const float* conv_w, float* h, int ch,
                                     int qk_heads, float eps, int n, int tb, int commit, const float* x,
                                     const uint16_t* wa, const uint16_t* wb, const float* dt, const float* sa,
                                     float* gate, float* beta, int n_embd, int h_v, void* s) {
    P3_TRY(gdn_conv_ab_multi(hist, qkv, conv_w, h, ch, qk_heads, eps, n, tb, commit != 0, x, wa, wb, dt, sa, gate, beta,
                             n_embd, h_v, s))
}
int tk_qwen4exp_p3_gdn_conv_l2_multi(const float* hist, const float* qkv, const float* conv_w, float* h, int ch,
                                     int qk_heads, float eps, int n, int tb, int commit, void* s) {
    P3_TRY(gdn_conv_l2_multi(hist, qkv, conv_w, h, ch, qk_heads, eps, n, s, tb, commit != 0))
}
int tk_qwen4exp_p3_gdn_step_norm_multi(float* state, const float* h, int ch, const float* gate, const float* beta,
                                       const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n,
                                       const int32_t* n_keep, int tb, void* xq_out, void* s) {
    P3_TRY(gdn_step_norm_multi(state, h, ch, gate, beta, z, gamma, eps, y, h_k, h_v, n, n_keep, s, tb, xq_out))
}

int tk_qwen4exp_p3_rms_norm(float* x, const float* w, int rows, int cols, float eps, int native, void* s) {
    P3_TRY(if (native) native_qsa_rms_norm_weighted(x, w, x, cols, rows, eps, s);
           else rms_norm_weighted(x, w, rows, cols, eps, s))
}
int tk_qwen4exp_p3_rope(float* x, int rows, int cols, int n_rot, const int32_t* pos, const float* cos_tab,
                        const float* sin_tab, int native, void* s) {
    P3_TRY(if (native) native_rope_apply(x, x, rows, cols, n_rot, rope_scaling(), pos, s);
           else rope_neox_apply(x, x, rows, cols, n_rot, cos_tab, sin_tab, pos, s))
}
int tk_qwen4exp_p3_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out, int rows, int head_dim,
                                 int n_rot, float eps, const int32_t* pos, void* s) {
    P3_TRY(native_qsa_rms_norm_rope(x, in_stride, gamma, out, rows, head_dim, n_rot, eps, rope_scaling(), pos, s))
}
int tk_qwen4exp_p3_copy_rows_strided(float* dst, const float* src, int64_t rows, int64_t w, int64_t src_w, void* s) {
    P3_TRY(copy_rows_strided(dst, src, rows, w, src_w, s))
}
int tk_qwen4exp_p3_copy_2d(float* dst, const float* src, int64_t head_dim, int64_t n_head, void* s) {
    if (cudaMemcpy2DAsync(dst, (size_t) head_dim * 4, src, (size_t) head_dim * 2 * 4, (size_t) head_dim * 4,
                          (size_t) n_head, cudaMemcpyDeviceToDevice, (cudaStream_t) s) != cudaSuccess) {
        g_err = "verify: the q/gate split failed";
        return -1;
    }
    return 0;
}
int tk_qwen4exp_p3_fwht256(float* x, int64_t rows, void* s) { P3_TRY(fwht256_inplace_cuda(x, rows, s)) }

int tk_qwen4exp_p3_kv_append(const void* g, void* ss, int64_t qi, const int32_t* step, int step_stride,
                             const float* kcur, const float* vcur, int cur_stride, int n, int batched, void* s) {
    const QsaState& st = QST(ss, qi);
    const QsaShapes sh = qsa_shapes(G(g));
    try {
        if (batched) {   // #783 PR-d: the window's cells in one launch per pool (verify.cpp)
            if (st.kv_hybrid) {
                const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                const bool mirror = st.host.present();
                kv_append_q8_steps(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step, step_stride, kcur, kcur,
                                   cur_stride, n, sh, s, mirror ? &hk : nullptr);
                kv_append_q4_steps(st.v_q4, st.v_q4, st.page_table, step, step_stride, n, vcur, vcur, sh, s,
                                   mirror ? &hv : nullptr);
            } else if (st.kv_q4) {
                kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step, step_stride, n, kcur, vcur, sh, s, &st.host);
            } else if (st.kv_int8) {
                kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step, step_stride, kcur, vcur,
                                   cur_stride, n, sh, s, &st.host);
            } else {
                for (int t = 0; t < n; ++t)
                    kv_append_step(st.k_pool, st.v_pool, st.page_table, step + (size_t) t * step_stride,
                                   kcur + (size_t) t * cur_stride, vcur + (size_t) t * cur_stride, sh, s, &st.host);
            }
            return 0;
        }
        for (int t = 0; t < n; ++t) {
            const int32_t* step_t = step + (size_t) t * step_stride;
            const float* kc = kcur + (size_t) t * cur_stride;
            const float* vc = vcur + (size_t) t * cur_stride;
            if (st.kv_hybrid) {
                const KvHostPools hk = kv_hybrid_k_half(st.host), hv = kv_hybrid_v_half(st.host);
                const bool mirror = st.host.present();
                kv_append_q8_step(st.k_q, st.k_q, st.k_scale, st.k_scale, st.page_table, step_t, kc, kc, sh, s,
                                  mirror ? &hk : nullptr);
                kv_append_q4_step(st.v_q4, st.v_q4, st.page_table, step_t, vc, vc, sh, s, mirror ? &hv : nullptr);
            } else if (st.kv_q4) {
                kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step_t, kc, vc, sh, s, &st.host);
            } else if (st.kv_int8) {
                kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step_t, kc, vc, sh, s, &st.host);
            } else {
                kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kc, vc, sh, s, &st.host);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
}

int tk_qwen4exp_p3_indexer_append(const void* g, void* ss, int64_t qi, const float* raw, int64_t raw_stride,
                                  const int32_t* step, int step_stride, int n, int batched, const float* gamma, float eps,
                                  void* s) {
    const QsaState& st = QST(ss, qi);
    const QsaShapes sh = qsa_shapes(G(g));
    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
    P3_TRY(if (batched) native_qsa_indexer_append_steps(raw, step, step_stride, n, 0, gamma, eps, ib, sh, st.max_cells,
                                                         rope_scaling(), s);
           else for (int t = 0; t < n; ++t)
               native_qsa_indexer_append(raw + (size_t) t * raw_stride, step + (size_t) t * step_stride, 0, gamma, eps, ib,
                                         sh, st.max_cells, rope_scaling(), s))
}
int tk_qwen4exp_p3_qsa_block_scores(const void* g, void* ss, int64_t qi, const float* qidx, const int32_t* steps,
                                    int64_t n, int64_t max_blocks, float* scores, void* s) {
    const QsaState& st = QST(ss, qi);
    P3_TRY(qsa_block_scores(st.idx_pooled, st.idx_dead, qidx, steps, n, max_blocks, qsa_shapes(G(g)), scores, s))
}
int tk_qwen4exp_p3_qsa_block_topk(const void* g, const float* scores, const int32_t* steps, int64_t n, int64_t max_blocks,
                                  int64_t cap, int32_t* sel, void* s) {
    P3_TRY(qsa_block_topk(scores, steps, n, max_blocks, cap, qsa_shapes(G(g)), sel, s))
}
int tk_qwen4exp_p3_qsa_kv_resolve(const void* g, void* ss, int64_t qi, const int32_t* sel, const int32_t* steps, int64_t n,
                                  int64_t cap, void* s) {
    P3_TRY(strata::core::qwen4exp_r2::qsa_kv_resolve(QST(ss, qi), G(g), sel, steps, n, cap, s))   // qwen4exp-r2
}
int tk_qwen4exp_p3_qsa_attn(const void* g, void* ss, int64_t qi, const float* q, const int32_t* sel, const int32_t* steps,
                            int64_t cap, float* scratch, float* attn, int64_t n, void* s) {
    const QsaAttnPools pools = strata::core::qwen4exp_r2::qsa_attn_pools(QST(ss, qi));   // qwen4exp-r2
    P3_TRY(qsa_decode_attn_batch(q, pools, sel, steps, cap, qsa_shapes(G(g)), scratch, attn, n, s))
}
int tk_qwen4exp_p3_qsa_gate(const float* attn, const float* qfull, float* out, int rows, int head_dim, int native,
                            const void* g, void* s) {
    P3_TRY(if (native) native_qsa_gate_apply(attn, qfull, out, rows, head_dim, s);
           else qsa_gate_apply_f32(attn, qfull, qsa_shapes(G(g)), out, s))
}

int tk_qwen4exp_p3_router_top10_multi(const float* logits, int32_t* ids, float* w, int n, void* s) {
    P3_TRY(native_router_top10_multi(logits, ids, w, n, s))
}
int tk_qwen4exp_p3_moe_route_row(const void* wt, const void* g, void* ss, int64_t layer, int64_t k, const float* x,
                                 float* logits, int32_t* ids, float* w, void* s) {
    MoEBuffers mb = SS(ss).moe;
    mb.logits = logits; mb.ids = ids; mb.weights = w;
    std::string err;
    try {
        if (!strata::core::qwen4exp_r2::moe_route(*(const WeightTable*) wt, G(g), layer, k, mb, x, s, err)) { g_err = err; return -1; }   // qwen4exp-r2
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
    return 0;
}
int tk_qwen4exp_p3_resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res, int n_expert,
                                 const uint8_t* base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                                 long long capx, uint32_t* skip, uint32_t ring, uint32_t* plan_err, void* s) {
    P3_TRY(resident_plan(ids, n_entries, k, res, n_expert, base, slot_off, blob, plan, capx, skip, ring, s, plan_err))
}
int tk_qwen4exp_p3_doorbell_publish(int variant, const float* x, const int32_t* ids, const float* w, const int32_t* res,
                                    int n_expert, int64_t n, int64_t k, float* x_out, int32_t* ids_out, float* w_out,
                                    uint32_t* seq, uint32_t value, void* s) {
    try {
        if (variant == 2) {
#if defined(STRATA_USE_HIP)
            doorbell_publish_value(x, ids, w, n, k, x_out, ids_out, w_out, seq, value, s);
#else
            g_err = "doorbell_publish_value is HIP only";
            return -1;
#endif
        } else if (variant == 1) {
            doorbell_publish_res(x, ids, res, n_expert, n, k, x_out, ids_out, seq, s);
        } else {
            doorbell_publish(x, ids, w, n, k, x_out, ids_out, w_out, seq, s);
        }
        return 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
}
int tk_qwen4exp_p3_f32_to_bf16(const float* x, uint16_t* y, int64_t n, void* s) { P3_TRY(f32_to_bf16_bulk(x, y, n, s)) }
int tk_qwen4exp_p3_shared_expert_multi(int n, const float* x, const uint16_t* x_bf16, int32_t gt, const void* gd,
                                       int32_t ut, const void* ud, int32_t dt_, const void* dd, void* q8_1,
                                       const uint16_t* gate_inp, float* gate, float* up, float* g_out, float* out,
                                       int64_t n_embd, int64_t n_ff, const void* xq_ready, int lfuse, void* s) {
    NativeSharedWeights nsw;
    nsw.gate_type = gt; nsw.gate_data = gd;
    nsw.up_type = ut; nsw.up_data = ud;
    nsw.down_type = dt_; nsw.down_data = dd;
    nsw.q8_1 = q8_1;
    P3_TRY(shared_expert_multi(n, x, x_bf16, nsw, gate_inp, gate, up, g_out, out, n_embd, n_ff, s, xq_ready, lfuse))
}
int tk_qwen4exp_p3_quantize_q8_1_rows(const float* x, int64_t rows, int64_t cols, void* y, void* s) {
    P3_TRY(quantize_q8_1_rows(x, rows, cols, y, s))
}
int tk_qwen4exp_p3_quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* s) {
    P3_TRY(quantize_q8_0_scaled(x, blocks, scales, n, s))
}
int tk_qwen4exp_p3_expert_grouped(int64_t layer, int native, const unsigned long long* gp, const int32_t* gs,
                                  const int32_t* gn, const int32_t* dst, const int32_t* tok, int64_t cap,
                                  const void* x_nat, const uint8_t* x_s2, const float* xs_s2, void* scratch, float* out,
                                  int64_t gy, void* s) {
    try {
        if (native) {
            const auto& f = strata::kernels::cpu::expert_layout().fmt[(size_t) layer];
            const NativeExpertLayout L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            native_expert_grouped(L, gp, gs, gn, dst, tok, cap, cap, x_nat, scratch, out, s, gy);
        } else {
            moe_grouped_s2(gp, gs, gn, dst, tok, cap, cap, x_s2, xs_s2, scratch, out, s);
        }
        return 0;
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
}
int tk_qwen4exp_p3_copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip,
                                               uint32_t v, void* s) {
    P3_TRY(copy_i32_from_mapped_unless(dst, src, n, skip, v, s))
}
int tk_qwen4exp_p3_wait_copy_i32_from_mapped(const uint32_t* flag, uint32_t v, int32_t* dst, const int32_t* src, int64_t n,
                                             void* s) {
    P3_TRY(wait_copy_i32_from_mapped(flag, v, dst, src, n, s))
}
int tk_qwen4exp_p3_fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap,
                               void* s) {
    P3_TRY(fetch_blobs(src, n, dst, blob_bytes, cap, s))
}
int tk_qwen4exp_p3_rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* s) {
    P3_TRY(rebase_ptrs(ptr, n, base, blob_bytes, s))
}
int tk_qwen4exp_p3_copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t v,
                                            void* s) {
    P3_TRY(copy_or_zero_from_mapped(dst, src, n, skip, v, s))
}
int tk_qwen4exp_p3_copy_rows_from_mapped(const uint32_t* flag_after, uint32_t v, float* dst, const float* src,
                                         int64_t rows, int64_t width, const int32_t* hit_rows, const int32_t* count,
                                         void* s) {
    P3_TRY(if (flag_after != nullptr) copy_rows_from_mapped_after(flag_after, v, dst, src, rows, width, hit_rows, count, s);
           else copy_rows_from_mapped(dst, src, rows, width, hit_rows, count, s))
}
int tk_qwen4exp_p3_moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                               int64_t n_embd, void* s) {
    P3_TRY(moe_hit_add(parts, hit_out, dst, count, cap, n_embd, s))
}
int tk_qwen4exp_p3_moe_combine(int variant, const float* parts, const float* hits, const int32_t* dst,
                               const int32_t* count, const float* w, const float* shared, const float* shared_gate,
                               float* out, int64_t n_embd, int64_t k, int n, void* s) {
    P3_TRY(if (variant == 2) native_moe_combine_multi_hits_gated(parts, w, shared, shared_gate, out, n_embd, k, n, s);
           else if (variant == 1)
               native_moe_combine_multi_hits_added(parts, hits, dst, count, w, shared, out, n_embd, k, n, s);
           else native_moe_combine_multi(parts, w, shared, out, n_embd, k, n, s))
}
int tk_qwen4exp_p3_moe_combine_row(const void* g, void* ss, int64_t layer, int64_t k, const float* w, const float* shared,
                                   const float* parts, float* out, void* s) {
    MoEBuffers mb = SS(ss).moe;
    mb.weights = const_cast<float*>(w);
    mb.shared = const_cast<float*>(shared);
    std::string err;
    try {
        if (!strata::core::qwen4exp_r2::moe_combine_parts(G(g), layer, k, mb, parts, out, s, err)) { g_err = err; return -1; }   // qwen4exp-r2
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
    return 0;
}
int tk_qwen4exp_p3_cvec_apply(float* R, int64_t layer, int64_t t, int64_t r_ld, const float* bo, int64_t bo_ld,
                              const float* inj, int64_t inj_ld, int write, void* s) {
    P3_TRY(cvec_apply(R, layer, t, r_ld, bo, bo_ld, inj, inj_ld, write != 0, s))
}
int tk_qwen4exp_p3_head_mix_row(const void* wt, const void* g, void* ss, float* R, float* mixed, float* logits,
                                int full_head, void* s) {
    BlockBuffers bb = SS(ss).block;
    bb.R = R;
    bb.mixed = mixed;
    std::string err;
    try {
        const bool ok = full_head ? strata::core::qwen4exp_r2::lm_head(*(const WeightTable*) wt, G(g), bb, logits, s, err)   // qwen4exp-r2
                                  : strata::core::qwen4exp_r2::lm_head_mix(*(const WeightTable*) wt, G(g), bb, s, err);
        if (!ok) { g_err = err; return -1; }
    } catch (const std::exception& e) {
        g_err = e.what();
        return -1;
    }
    return 0;
}
int tk_qwen4exp_p3_argmax_rows(const float* logits, int rows, int n, void* scratch, int32_t* out, void* s) {
    P3_TRY(argmax_rows(logits, rows, n, scratch, out, s))
}
int tk_qwen4exp_p3_sample_greedy(const float* logits, int rows, int n, int32_t* out, void* s) {
    SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.0f;
    P3_TRY(sample_tokens(logits, rows, n, nullptr, 0, sp, out, s))
}
int tk_qwen4exp_p3_logits_nonfinite_rows(const float* logits, int rows, int n, uint32_t* flags, void* s) {
    P3_TRY(logits_nonfinite_rows(logits, rows, n, flags, s))
}
int tk_qwen4exp_p3_gpu_noop(void* s) { P3_TRY(gpu_noop(s)) }

}  // extern "C"
