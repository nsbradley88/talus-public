// src/program/qwen4exp_p4.cpp - qwen4exp Piece 4: the C ABI under the Rust Verifier + MTP host plane
// (rust/crates/talus-model/src/qwen4exp/verify_mtp.rs).
//
// Rust runs Verifier::run / commit / wait_commit / copy_logits / window_logprobs / set_sampling and
// MtpDrafter::draft / draft_first / set_draft_sampling / set_draft_history.  The C++ objects keep their storage
// (mapped staging, the device arena, the captured graphs, the streams), so this file only hands Rust views of it
// (Qwen4expP4Access is a friend of both classes) and wraps what is not host control flow: graph capture (the T-token
// layer sequence, until Piece 3's per-layer launches exist), the PLE table, and the sampler kernel launches.
#include "qwen4exp_r2.hpp"   // qwen4exp-r2: the layer.cpp launch helpers as thin wrappers
#include "qwen4exp_p4.hpp"

#include "strata/core/coupled_draft.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/rope_scaling.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
void put_err(const std::string& e, char* err, size_t cap) {
    if (err == nullptr || cap == 0) return;
    std::snprintf(err, cap, "%s", e.c_str());
}
bool env_is(const char* name, char want) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == want;
}
}  // namespace

namespace strata::core {

// Piece 3's window recorder hook (src/core/verify.cpp qwen4exp-p3 block)
using Qwen4expP3Recorder = int (*)(void* user, void* ver, int t, void* stream, char* err, size_t cap);
extern Qwen4expP3Recorder g_qwen4exp_p3_recorder;
extern void* g_qwen4exp_p3_user;

struct Qwen4expP4Access {
    static int ver_view(Verifier& v, SessionState* ss, tk_qwen4exp_p4_ver_view_t* o) {
        if (v.g_ == nullptr || v.ss_ == nullptr) return -1;
        std::memset(o, 0, sizeof(*o));
        const ModelGeometry& g = *v.g_;
        const strata::kernels::QsaShapes s = qsa_shapes(g);
        o->ver = &v;
        // the Rust plane runs one whole-model stage with no remote opt; the debug traces stay on the C++ run()
        o->eligible = (v.next_ == nullptr && v.lb_ == 0 && v.le_ == g.n_layers && v.remote_opt_ == nullptr &&
                       v.trace_h_ == nullptr && !v.prof_on_ && v.ext_stream_ == nullptr)
                          ? 1 : 0;
        o->max_t = v.max_t_;
        o->n_layers = (int32_t) g.n_layers;
        o->lb = (int32_t) v.lb_;
        o->le = (int32_t) v.le_;
        o->n_embd = (int32_t) g.n_embd;
        o->hc = (int32_t) g.hc;
        o->k = (int32_t) v.ss_->k;
        o->n_head = (int32_t) g.n_head;
        o->n_head_kv = (int32_t) g.n_head_kv;
        o->idx_q_heads = (int32_t) g.idx_q_heads;
        o->step_count = strata::kernels::kStepCount;
        o->idx_block = s.idx_block;
        o->idx_top_k = s.idx_top_k;
        o->n_vocab = v.n_vocab_;
        o->h_tok = v.h_tok_; o->h_step = v.h_step_; o->h_pos = v.h_pos_; o->h_commit = v.h_commit_; o->h_out = v.h_out_; o->m_out = v.m_out_;
        o->h_ple = v.h_ple_; o->h_nf = v.h_nf_; o->h_x = v.h_x_; o->h_ids = v.h_ids_; o->h_ymiss = v.h_ymiss_;
        o->h_seq = v.h_seq_; o->h_flag = v.h_flag_; o->h_flagA = v.h_flagA_; o->h_flagB = v.h_flagB_;
        o->h_plan_err = v.h_plan_err_;
        o->h_plan = v.h_plan_; o->plan_i32 = v.plan_i32_;
        o->sink = &v.sink_;
        o->staging = v.staging_;
        o->staging_blobs = Verifier::kStagingBlobs;
        o->max_blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        o->cur_layer = &v.cur_layer_;
        o->groups = v.groups_;
        o->nonfinite_blocks = strata::kernels::kNonfiniteBlocks;
        o->verify_max_t = strata::kernels::kVerifyMaxT;
        o->head_logits = v.head_logits_;
        o->R = v.R_;
        o->stream = (void*) v.cs_;
        o->copy_stream = (void*) v.copy_;
        o->commit_done = (void*) v.commit_done_;
        o->commit_exec = (void**) &v.commit_exec_;
        o->last_t = &v.last_t_; o->last_pos0 = &v.last_pos0_; o->last_tokens = v.last_tokens_;
        static_assert(sizeof(bool) == 1, "bool views");
        o->staged = (uint8_t*) &v.staged_;
        o->last_batch = (uint8_t*) &v.last_batch_;
        o->copy_used = (uint8_t*) &v.copy_used_;
        o->commit_pending = (uint8_t*) &v.commit_pending_;
        o->always_publish = (uint8_t*) &v.always_publish_;
        o->head_sampling = (uint8_t*) &v.head_sampling_;
        o->nf_row = &v.nf_row_; o->nf_pos = &v.nf_pos_; o->nf_mask = &v.nf_mask_;
        o->ms_wait = &v.ms_wait; o->ms_pool = &v.ms_pool; o->ms_host = &v.ms_host; o->ms_commit = &v.ms_commit;
        o->ms_gate = &v.ms_gate;
        o->windows = &v.windows; o->spec_windows = &v.spec_windows;
        o->sampling = &v.sampling_;
        o->hist_d = &v.hist_d_; o->hist_len = &v.hist_len_;
        static_assert(sizeof(std::atomic<bool>) == 1, "atomic<bool> view");
        o->released = (const uint8_t*) &v.released_;
        o->adapt_gate = v.adapt_gate_;
        o->d_spec_slot = &v.d_spec_;
        (void) ss;
        SessionState& s2 = *v.ss_;   // run() reads its own session's PLE
        o->ple_prev = s2.ple_prev;
        o->ple_on = (s2.ple.ready() && v.ple_stage()) ? 1 : 0;
        o->ple_ss = &s2;
        o->max_cells = &v.ss_->qsa_states[v.ss_->qsa_primary()].max_cells;
        o->sizeof_sink = (int32_t) sizeof(GpuPlanSink);
        o->sizeof_sampler = (int32_t) sizeof(strata::kernels::SamplerParams);
        return 0;
    }

    static int ver_prepare(Verifier& v, int T, void** exec, int* ar_on, std::string& err) {
        const OnDevice on_device(v.device_);
        v.refresh_ar();
        if (!v.capture(T, err) || !v.capture_commit(err)) return -1;
        if (exec) *exec = (void*) (v.ar_off_ ? v.exec_nr_[T] : v.exec_[T]);
        if (ar_on) *ar_on = v.ar_on() ? 1 : 0;
        // capture_commit may have made the commit graph just now
        return 0;
    }

    static int mtp_view(MtpDrafter& m, tk_qwen4exp_p4_mtp_view_t* o) {
        if (m.g_ == nullptr) return -1;
        std::memset(o, 0, sizeof(*o));
        o->mtp = &m;
        // STRATA_MTP_PROFILE / STRATA_DRAFT_TIMING are mtp.cpp's own diagnostics: those runs keep its draft()
        o->eligible = (!env_is("STRATA_MTP_PROFILE", '1') && !env_is("STRATA_DRAFT_TIMING", '1')) ? 1 : 0;
        o->max_t = m.max_t_;
        o->max_drafts = &m.max_drafts_;
        o->n_head = (int32_t) m.g_->n_head;
        o->hcn = m.g_->hc * m.g_->n_embd;
        o->max_cells = &m.st_.max_cells;
        o->h_tok = m.h_tok_; o->h_step = m.h_step_; o->h_pos = m.h_pos_; o->h_row = m.h_row_; o->h_out = m.h_out_;
        o->h_prob = m.h_prob_; o->h_force = m.h_force_;
        o->stream = (void*) m.cs_;
        o->window_R = (float*) m.window_R_;
        o->src_R = &m.src_R_;
        o->coupled_active = (uint8_t*) &m.coupled_active_;
        o->coupled_ok = m.coupled_ok_ ? 1 : 0;
        o->chain_live = (uint8_t*) &m.chain_live_;
        o->cparams = m.h_cparams_;
        o->h_chist = m.h_chist_;
        o->h_q = m.h_q_;
        o->spec_q_stride = kSpecQStride;
        o->coupled_hist_cap = kCoupledHistCap;
        o->ms_draft = &m.ms_draft; o->ms_src = &m.ms_src; o->rounds = &m.rounds; o->n_src = &m.n_src;
        SessionState* const pss = m.ple_ss_ ? m.ple_ss_ : m.ss_;
        const bool ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
        o->ple_prev = pss != nullptr ? pss->ple_prev : nullptr;
        o->ple_on = ple ? 1 : 0;
        o->ple_ss = pss;
        return 0;
    }

    static int mtp_prepare(MtpDrafter& m, int T, bool cp, int max_steps, void** round, void** steps, std::string& err) {
        const OnDevice on_device(m.device_);
        if (!m.capture_round(T, cp, err)) return -1;
        for (int j = 1; j < max_steps; ++j)
            if (!m.capture_step(j, cp, err)) return -1;
        if (round) *round = (void*) (cp ? m.round_exec_c_[T] : m.round_exec_[T]);
        if (steps)
            for (int j = 1; j < max_steps && j < 9; ++j) steps[j] = (void*) (cp ? m.step_exec_c_[j] : m.step_exec_[j]);
        return 0;
    }
    // ---- phase 2a: capture in Rust

    static int cap_view(Verifier& v, tk_qwen4exp_p4_capture_view_t* o) {
        if (v.g_ == nullptr || v.ss_ == nullptr) return -1;
        std::memset(o, 0, sizeof(*o));
        const ModelGeometry& g = *v.g_;
        SessionState& ss = *v.ss_;
        static_assert(sizeof(cudaGraphExec_t) == sizeof(void*), "exec handles");
        o->exec = (void**) v.exec_;
        o->exec_nr = (void**) v.exec_nr_;
        o->all_resident = (uint8_t*) &v.all_resident_;
        o->ar_off = (uint8_t*) &v.ar_off_;
        o->h_res = v.h_res_;
        o->n_expert = g.n_expert;
        o->commit = v.commit_; o->m_commit = v.m_commit_;
        o->qkv_L = v.qkv_L_; o->h_L = v.h_L_; o->gate_L = v.gate_L_; o->beta_L = v.beta_L_; o->z = v.z_;
        o->y_dummy = v.y_dummy_; o->idx_raw_L = v.idx_raw_L_; o->tail_snap = v.tail_snap_; o->hist_snap = v.hist_snap_;
        o->ple_hist = ss.ple.hist;
        o->ple_copy = (ss.ple.ready() && v.ple_stage()) ? 1 : 0;
        o->gdn_state = ss.gdn_state; o->gdn_ord0 = ss.gdn_ord0;
        o->ssm_state_size = g.ssm_state_size; o->ssm_k_heads = g.ssm_k_heads; o->ssm_v_heads = g.ssm_v_heads;
        o->ssm_d_conv = g.ssm_d_conv; o->ssm_conv_channels = g.ssm_conv_channels; o->idx_key_dim = g.idx_key_dim;
        o->ng_hist_hc = (int64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM;
        static thread_local std::vector<uint8_t> types;
        types.assign((size_t) g.n_layers, 0);
        for (int64_t l = 0; l < g.n_layers; ++l) types[(size_t) l] = is_qsa_layer(g, l) ? 1 : 0;
        o->layer_is_qsa = types.data();
        return 0;
    }
    static int qsa_state(Verifier& v, int64_t qi, tk_qwen4exp_p4_qsa_state_t* o) {
        if (v.ss_ == nullptr || o == nullptr) return -1;
        const QsaState& st = v.ss_->qsa_states[qi];
        o->idx_tail = st.idx_tail; o->idx_dead = st.idx_dead; o->idx_pooled = st.idx_pooled;
        o->idx_block_pos = st.idx_block_pos; o->max_cells = st.max_cells;
        return 0;
    }
    static const float* weight(Verifier& v, int64_t l, const char* suffix, std::string& err) {
        const LayerView lv(*v.wt_, l);
        const WeightRef* r = lv.get(suffix);
        if (r == nullptr) { err = lv.name(suffix) + " is missing"; return nullptr; }
        return (const float*) r->data;
    }
    // the window's kernels: Piece 3's Rust recorder (decode.rs, installed with TALUS_FN_RUST=1) first, as
    // Verifier::capture does; declined (0): Verifier::record_window
    static bool record(Verifier& v, int T, std::string& err) {
        int fnp3 = 0;
        if (g_qwen4exp_p3_recorder != nullptr) {
            char eb[512] = "";
            fnp3 = g_qwen4exp_p3_recorder(g_qwen4exp_p3_user, &v, T, (void*) v.cs_, eb, sizeof eb);
            if (fnp3 < 0) err = eb;
        }
        return fnp3 > 0 || (fnp3 == 0 && v.record_window(T, v.cs_, err));
    }
    static bool instantiate(Verifier& v, int T, cudaGraph_t graph, std::string& err) {
        cudaGraphExec_t& exec_t = v.ar_off_ ? v.exec_nr_[T] : v.exec_[T];
        const bool made = v.instantiate_evicting(exec_t, graph, {}, "verify: instantiate: ", err);
        cudaGraphDestroy(graph);
        if (!made) return false;
        const cudaError_t ue = cudaGraphUpload(exec_t, v.cs_);
        const cudaError_t us = cudaStreamSynchronize(v.cs_);
        std::fprintf(stderr, "talus verify: captured the %d-token window (upload %s, sync %s)\n", T,
                     cudaGetErrorString(ue), cudaGetErrorString(us));
        return true;
    }
    static strata::kernels::QsaShapes shapes(Verifier& v) { return qsa_shapes(*v.g_); }

    // ---- phase 2b
    static int mtp_pf_view(MtpDrafter& m, tk_qwen4exp_p4_mtp_pf_view_t* o) {
        std::memset(o, 0, sizeof(*o));
        o->tok = m.tok_; o->step = m.step_; o->pos = m.pos_; o->rin = m.Rin_;
        o->window = m.window_; o->prompt_len = &m.prompt_len_; o->ms_prefill = &m.ms_prefill;
        return 0;
    }
    static int mtp_prefill_graph(MtpDrafter& m, int T, bool dev, void** exec, std::string& err) {
        const OnDevice on_device(m.device_);
        if (dev ? !m.capture_prefill_dev(T, err) : !m.capture_prefill(T, err)) return -1;
        *exec = (void*) (dev ? m.prefill_dev_exec_[T] : m.prefill_exec_[T]);
        return 0;
    }
    static int mtp_pf_upload(MtpDrafter& m, const int32_t* rec, int64_t n, int32_t** out, std::string& err) {
        const OnDevice on_device(m.device_);
        if (m.pf_cap_ < n) {
            if (m.pf_dev_) cudaFree(m.pf_dev_);
            m.pf_dev_ = nullptr;
            m.pf_cap_ = 0;
            if (cudaMalloc((void**) &m.pf_dev_, (size_t) n * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return -1;
            }
            m.pf_cap_ = n;
        }
        if (m.pf_ring_.h2d(m.pf_dev_, rec, (size_t) n * sizeof(int32_t), m.cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return -1;
        }
        *out = m.pf_dev_;
        return 0;
    }

    // ---- phase 3: the MTP layer recorded from Rust (verify_mtp.rs MtpRecorder) over these views and wrappers
    static int mtp_rec_view(MtpDrafter& m, tk_qwen4exp_p4_mtp_rec_view_t* o) {
        if (m.g_ == nullptr || m.ss_ == nullptr) return -1;
        std::memset(o, 0, sizeof(*o));
        const ModelGeometry& g = *m.g_;
        const strata::kernels::QsaShapes s = qsa_shapes(g);
        o->tok = m.tok_; o->step = m.step_; o->pos = m.pos_; o->row = m.row_; o->ident = m.ident_; o->ids = m.ids_;
        o->hit_slot = m.hit_slot_; o->hit_dst = m.hit_dst_; o->hit_count = m.hit_count_; o->out_ids = m.out_ids_;
        o->grp_start = m.grp_start_; o->grp_counts = m.grp_counts_; o->grp_ptr = m.grp_ptr_;
        o->rin = m.Rin_; o->R = m.R_; o->emb = m.emb_; o->en = m.en_; o->e2 = m.e2_; o->hn = m.hn_; o->h2 = m.h2_;
        o->mixed = m.mixed_; o->inj = m.inj_; o->inj2 = m.inj2_; o->lo = m.lo_; o->rs = m.rs_; o->bo = m.bo_; o->xn = m.xn_;
        o->qfull = m.qfull_; o->qcur = m.qcur_; o->kcur = m.kcur_; o->vcur = m.vcur_; o->attn = m.attn_; o->attn32 = m.attn32_;
        o->logits = m.logits_; o->w = m.w_; o->shared = m.shared_; o->parts = m.parts_; o->y = m.y_; o->sample = m.sample_;
        o->head_logits = m.head_logits_; o->dummy_inj = m.dummy_inj_; o->hit_xs = m.hit_xs_; o->probs = m.probs_;
        o->xq = m.xq_; o->hit_xq = m.hit_xq_; o->hit_scratch = m.hit_scratch_; o->arg_scratch = m.arg_scratch_;
        o->top_scratch = m.top_scratch_; o->x_bf16 = m.x_bf16_; o->experts = m.experts_;
        o->m_tok = m.m_tok_; o->m_step = m.m_step_; o->m_pos = m.m_pos_; o->m_row = m.m_row_; o->m_force = m.m_force_;
        o->m_out = m.m_out_; o->m_prob = m.m_prob_;
        o->N = (int32_t) g.n_embd; o->HC = (int32_t) g.hc; o->NH = (int32_t) g.n_head; o->HD = (int32_t) g.head_dim;
        o->NKV = (int32_t) g.n_head_kv; o->K = (int32_t) m.ss_->k; o->hc_lr = (int32_t) g.hc_lr; o->n_rot = (int32_t) s.n_rot;
        o->n_expert = (int32_t) g.n_expert; o->max_t = m.max_t_;
        o->window = m.window_; o->n_vocab = m.n_vocab_; o->n_dvocab = m.n_dvocab_;
        o->hnorm_stream = m.hnorm_stream_ ? 1 : 0; o->force_on = m.force_on_ ? 1 : 0; o->kv_rot = m.st_.kv_rot ? 1 : 0;
        o->coupled_ok = m.coupled_ok_ ? 1 : 0; o->spec_q = m.m_q_ != nullptr ? 1 : 0;
        o->cos_tab = m.st_.cos_tab; o->sin_tab = m.st_.sin_tab;
        o->stream = (void*) m.cs_;
        o->round_exec = (void**) m.round_exec_; o->round_exec_c = (void**) m.round_exec_c_;
        o->step_exec = (void**) m.step_exec_; o->step_exec_c = (void**) m.step_exec_c_;
        o->prefill_exec = (void**) m.prefill_exec_; o->prefill_dev_exec = (void**) m.prefill_dev_exec_;
        o->native_rope = strata::kernels::native_rope_enabled() ? 1 : 0;
        o->norm_rope_usable = strata::kernels::native_norm_rope_usable((int) g.head_dim, (int) s.n_rot) ? 1 : 0;
        o->head_type = m.head_ != nullptr ? m.head_->type() : -1;
        o->sub_head = m.dhead_ != nullptr ? 1 : 0;
        o->dhead_type = m.dhead_type_;
        return 0;
    }
    static const void* mtp_tensor(MtpDrafter& m, const char* name, int kind, int* type) {
        if (kind == 0) return m.f32(name);
        if (kind == 1) return m.bf16(name);
        int t = -1;
        const void* p = m.wq(name, t);
        if (type) *type = t;
        return p;
    }
    // record_front's embedding gather (the GGUF-form native table, else the pack's table)
    static int m_embed(MtpDrafter& m, int T, cudaStream_t cs, std::string& err) {
        const WeightRef* we = m.wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return -1; }
        if (const NativeEmbed* ne = native_embed()) {
            ne->gather_dev(m.tok_, T, m.emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            strata::kernels::embedding_gather_dev(codes, scales, offsets, m.tok_, T, we->ne0, we->code_bits, we->code_bias,
                                                  we->group_elems, (uint64_t) (we->ne0 / (8 / we->code_bits)),
                                                  (uint64_t) (we->ne0 / we->group_elems), m.emb_, cs);
        }
        return 0;
    }
    // the fused hyper-connection reads: mode 0 = attention (no apply), 1 = MLP (the attention write folded in),
    // 2 = the final mixer with the MoE write folded in (STRATA_FUSE_HEAD_GR), 3 = the final mixer, no apply
    static int m_gr_read(MtpDrafter& m, int mode, int T, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const int64_t N = g.n_embd, HC = g.hc;
        const char* pre = mode == 0 ? "attn_hyper_connection" : mode == 1 ? "mlp_hyper_connection" : "hyper_connection_mixer";
        const std::string wn = std::string(pre) + ".hc_norm.weight", wd = std::string(pre) + ".input_mix_weight_down.weight",
                          wu = std::string(pre) + ".input_mix_weight_up.weight", wi = std::string(pre) + ".block_inject_weight.weight";
        FusedGrArgs fa[kFusedGrMaxT];
        for (int t = 0; t < T; ++t) {
            fa[t].R = m.R_ + (size_t) t * HC * N; fa[t].R_out = m.R_ + (size_t) t * HC * N;
            fa[t].apply = mode == 1 || mode == 2;
            if (mode == 1) { fa[t].bo_prev = m.bo_ + t * N; fa[t].inj_prev = m.inj_ + t * HC; }
            if (mode == 2) { fa[t].bo_prev = m.y_ + t * N; fa[t].inj_prev = m.inj2_ + t * HC; }
            fa[t].w_norm = m.f32(wn.c_str());
            fa[t].w_down = m.bf16(wd.c_str());
            fa[t].w_up = m.bf16(wu.c_str());
            if (mode == 0 || mode == 1) fa[t].w_inject = m.bf16(wi.c_str());
            else if (mode == 2) fa[t].w_inject = nullptr;
            fa[t].eps = 1e-6f; fa[t].lo = m.lo_ + t * g.hc_lr; fa[t].rs = m.rs_ + t * HC;
            if (mode == 0) { fa[t].inject_out = m.inj_ + t * HC; fa[t].mixed = m.mixed_ + t * N; }
            else if (mode == 1) { fa[t].inject_out = m.inj2_ + t * HC; fa[t].mixed = m.mixed_ + t * N; }
            else if (mode == 2) { fa[t].inject_out = m.dummy_inj_; fa[t].mixed = m.sample_ + t * N; }
            else { fa[t].mixed = m.sample_ + t * N; }
        }
        fused_gr_read_multi(fa, T, m.xn_, cs);
        return 0;
    }
    // the final mixer per row with the drafter's own gr workspace (gr_read)
    static int m_gr_read_row(MtpDrafter& m, int t, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
        gr_read(m.R_ + (size_t) t * g.hc * g.n_embd, m.f32("hyper_connection_mixer.hc_norm.weight"),
                m.bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                m.bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, 1e-6f, gs, m.own_gr_,
                m.sample_ + t * g.n_embd, m.dummy_inj_, cs);
        return 0;
    }
    // record_front's K/V append in the state's own format (no_batch: one call per row)
    static int m_kv_append(MtpDrafter& m, const int32_t* step, int T, bool no_batch_kv, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const QsaShapes s = qsa_shapes(g);
        const int64_t NKV = g.n_head_kv, HD = g.head_dim;
        QsaState& st = m.st_;
        if (!no_batch_kv && st.kv_q4)
            kv_append_q4_steps(st.k_q4, st.v_q4, st.page_table, step, 4, T, m.kcur_, m.vcur_, s, cs, &st.host);
        else if (!no_batch_kv && st.kv_int8)
            kv_append_q8_steps(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step, 4, m.kcur_, m.vcur_,
                               (int) (NKV * HD), T, s, cs, &st.host);
        else
            for (int t = 0; t < T; ++t) {
                if (st.kv_q4)
                    kv_append_q4_step(st.k_q4, st.v_q4, st.page_table, step + t * 4, m.kcur_ + t * NKV * HD,
                                      m.vcur_ + t * NKV * HD, s, cs, &st.host);
                else if (st.kv_int8)
                    kv_append_q8_step(st.k_q, st.v_q, st.k_scale, st.v_scale, st.page_table, step + t * 4,
                                      m.kcur_ + t * NKV * HD, m.vcur_ + t * NKV * HD, s, cs, &st.host);
                else
                    kv_append_step(st.k_pool, st.v_pool, st.page_table, step + t * 4, m.kcur_ + t * NKV * HD,
                                   m.vcur_ + t * NKV * HD, s, cs, &st.host);
            }
        return 0;
    }
    // the dense attention over the drafter's cells (a sliding window: the last window_ cells)
    static int m_attn(MtpDrafter& m, const int32_t* step, int T, cudaStream_t cs) {
        using namespace strata::kernels;
        const QsaShapes s = qsa_shapes(*m.g_);
        const QsaAttnPools pools = strata::core::qwen4exp_r2::qsa_attn_pools(m.st_);   // qwen4exp-r2
        if (m.window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) m.window_, m.ident_, m.cap_, cs);
        qsa_decode_attn_batch(m.qcur_, pools, m.ident_, step, m.cap_, s, m.attn_scratch_, m.attn_, T, cs);
        return 0;
    }
    // the shared expert of each row (mixed_ -> shared_)
    static int m_shared(MtpDrafter& m, int T, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const int64_t N = g.n_embd;
        NativeSharedWeights nsw;
        nsw.gate_data = m.wq("mlp.shared_expert.gate_proj.weight", nsw.gate_type);
        nsw.up_data = m.wq("mlp.shared_expert.up_proj.weight", nsw.up_type);
        nsw.down_data = m.wq("mlp.shared_expert.down_proj.weight", nsw.down_type);
        nsw.q8_1 = m.xq_;
        const SForm none{};
        const bool need_bf16_x = !shared_expert_native_bf16_enabled();
        for (int t = 0; t < T; ++t) {
            if (need_bf16_x) f32_to_bf16_bulk(m.mixed_ + t * N, m.x_bf16_, N, cs);
            shared_expert(nullptr, nullptr, m.x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr, none,
                          nullptr, nullptr, nullptr, m.bf16("mlp.shared_expert_gate.weight"), m.sh_scratch_,
                          m.shared_ + t * N, N, g.n_ff, 32, cs, m.mixed_ + t * N, &nsw);
        }
        return 0;
    }
    // the router (mixed_ -> ids_, w_) and STRATA_ROUTE_RESIDENT_MTP
    static int m_route(MtpDrafter& m, int T, cudaStream_t cs, std::string& err) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const int64_t N = g.n_embd, K = m.ss_->k;
        if (T > 1 && native_router_enabled() && g.n_expert == 512 && K == 10) {
            bf16_gemv_fp32_mmvf_multi(m.mixed_, N, m.bf16("mlp.gate.weight"), m.logits_, g.n_expert, (int) N, (int) g.n_expert,
                                      T, cs);
            native_router_top10_multi(m.logits_, m.ids_, m.w_, T, cs);
        } else {
            for (int t = 0; t < T; ++t) {
                bf16_gemv_fp32_mmvf(m.mixed_ + t * N, m.bf16("mlp.gate.weight"), m.logits_ + t * g.n_expert, (int) N,
                                    (int) g.n_expert, cs);
                if (native_router_enabled() && g.n_expert == 512 && K == 10)
                    native_router_top10(m.logits_ + t * g.n_expert, m.ids_ + t * K, m.w_ + t * K, cs);
                else
                    router_top10(m.logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, m.ids_ + t * K, m.w_ + t * K, cs);
            }
        }
        if (m.rr_res_ != nullptr && m.rr_margin_ > 0.0f && g.n_expert == 512 && K == 10) {
            try {
                native_route_resident(m.logits_, m.ids_, m.w_, m.rr_res_, T, m.rr_margin_, m.rr_lo_, m.rr_hi_, nullptr, cs);
            } catch (const std::exception& e) { err = std::string("mtp route-resident: ") + e.what(); return -1; }
        }
        return 0;
    }
    // the 512 resident routed experts (ids_ -> parts_)
    static int m_experts(MtpDrafter& m, int T, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const int64_t N = g.n_embd, K = m.ss_->k;
        moe_group_resident(m.ids_, (int) (T * K), (int) K, m.experts_, (int64_t) strata::kernels::cpu::BLOB, m.grp_ptr_,
                           m.grp_start_, m.grp_counts_, m.hit_dst_, m.hit_slot_, cs);
        quantize_q8_0_scaled(m.mixed_, m.hit_xq_, m.hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(m.grp_ptr_, m.grp_start_, m.grp_counts_, m.hit_dst_, m.hit_slot_, (int64_t) T * K, (int64_t) T * K,
                       m.hit_xq_, m.hit_xs_, m.hit_scratch_, m.parts_, cs);
        return 0;
    }
    // the combine of row t (multi = -1: every row in one launch) and the residual writes (row t, or multi)
    static int m_combine(MtpDrafter& m, int t, int T, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const int64_t N = g.n_embd, K = m.ss_->k;
        if (t < 0) native_moe_combine_multi(m.parts_, m.w_, m.shared_, m.y_, N, K, T, cs);
        else if (native_moe_combine_enabled())
            native_moe_combine(m.parts_ + (size_t) t * K * N, m.w_ + t * K, m.shared_ + t * N, m.y_ + t * N, N, K, cs);
        else
            moe_combine(m.parts_ + (size_t) t * K * N, m.w_ + t * K, m.shared_ + t * N, m.y_ + t * N, N, K, cs);
        return 0;
    }
    static int m_gr_write(MtpDrafter& m, int t, int T, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
        const int64_t N = g.n_embd, HC = g.hc;
        if (t < 0) gr_write_multi(m.R_, m.y_, m.inj2_, gs, m.R_, T, cs);
        else gr_write(m.R_ + (size_t) t * HC * N, m.y_ + t * N, m.inj2_ + t * HC, gs, m.R_ + (size_t) t * HC * N, cs);
        return 0;
    }
    // the draft head (sample_ -> head_logits_) and the pick: coupled / probabilistic sampling, or the argmax and its
    // probability (mapped back from the draft vocabulary)
    static int m_head(MtpDrafter& m, int T, bool coupled, int j, const int32_t* step, cudaStream_t cs) {
        using namespace strata::kernels;
        const int64_t N = m.g_->n_embd;
        native_quantize_q8_1(m.sample_, m.xq_, (int) N, T, cs);
        const bool sub = m.dhead_ != nullptr;
        const int64_t nv = sub ? m.n_dvocab_ : m.n_vocab_;
        native_mmvq(sub ? m.dhead_type_ : m.head_->type(), sub ? m.dhead_ : m.head_->weights(), m.xq_, m.head_logits_,
                    (int) N, (int) nv, T, cs);
        if (coupled) {
            if (m.m_q_ != nullptr)
                spec_draft_sample(m.head_logits_, (int) nv, sub ? m.dvocab_ : nullptr, sub ? m.dinv_ : nullptr,
                                  (int) m.n_vocab_, m.cparams_, m.cring_, kCoupledHistCap, j, step, m.cscratch_, m.out_ids_,
                                  m.probs_, m.m_q_, spec_gate_pick(), spec_draft_temp_scale(), cs);
            else
                coupled_draft_sample(m.head_logits_, (int) nv, sub ? m.dvocab_ : nullptr, sub ? m.dinv_ : nullptr,
                                     (int) m.n_vocab_, m.cparams_, m.cring_, kCoupledHistCap, j, step, m.cscratch_, m.out_ids_,
                                     m.probs_, cs);
            return 0;
        }
        if (argmax_rows_wanted()) {
            argmax_rows(m.head_logits_, T, (int) nv, m.arg_scratch_, m.out_ids_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(m.head_logits_, T, (int) nv, nullptr, 0, sp, m.out_ids_, cs);
        }
        if (multi_block_head_ops()) row_top_prob_split(m.head_logits_, T, (int) nv, m.out_ids_, m.probs_, m.top_scratch_, cs);
        else row_top_prob(m.head_logits_, T, (int) nv, m.out_ids_, m.probs_, cs);
        if (sub) map_ids(m.out_ids_, m.dvocab_, T, cs);
        return 0;
    }
    static int m_round_tail(MtpDrafter& m, int which, int j, cudaStream_t cs) {
        using namespace strata::kernels;
        const ModelGeometry& g = *m.g_;
        const int64_t HCN = g.hc * g.n_embd, N = g.n_embd;
        switch (which) {
        case 0: coupled_draft_stage(m.m_cparams_, m.m_chist_, m.cparams_, m.cring_, kCoupledHistCap, cs); break;
        case 1:   // row a's intermediates to row 0, its q8_1
            copy_row_to_first(m.row_, m.R_, HCN, m.inj_, g.hc, m.mixed_, N, cs);
            native_quantize_q8_1(m.mixed_, m.xq_, (int) N, 1, cs);
            break;
        case 2: mtp_select(m.R_, HCN, m.out_ids_, m.row_ + 1, m.Rin_, m.tok_, m.m_out_, j, cs, m.probs_, m.m_prob_); break;
        case 3: force_token(m.tok_, m.m_force_, j, cs); break;
        default: return -1;
        }
        return 0;
    }
    static int mtp_instantiate(MtpDrafter& m, int which, int idx, void* graph, std::string& err) {
        cudaGraphExec_t* slot = nullptr;
        const char* what = "";
        switch (which) {
        case 0: slot = &m.round_exec_[idx]; what = "round"; break;
        case 1: slot = &m.round_exec_c_[idx]; what = "round (coupled)"; break;
        case 2: slot = &m.step_exec_[idx]; what = "step"; break;
        case 3: slot = &m.step_exec_c_[idx]; what = "step (coupled)"; break;
        case 4: slot = &m.prefill_exec_[idx]; what = "prefill"; break;
        case 5: slot = &m.prefill_dev_exec_[idx]; what = "prefill (device inputs)"; break;
        default: return -1;
        }
        cudaGraph_t gph = (cudaGraph_t) graph;
        if (cudaGraphInstantiate(slot, gph, 0) != cudaSuccess) {
            cudaGraphDestroy(gph);
            err = std::string("mtp: ") + what + " capture: instantiate failed";
            return -1;
        }
        cudaGraphDestroy(gph);
        cudaGraphUpload(*slot, m.cs_);   // an explicit upload: the first launch's implicit one blocked behind a spin
        cudaStreamSynchronize(m.cs_);
        return 0;
    }
};

}  // namespace strata::core

using strata::core::Qwen4expP4Access;

extern "C" int tk_qwen4exp_p4_hooks_info(tk_qwen4exp_p4_hooks_t* h, tk_qwen4exp_p4_hooks_info_t* out) {
    if (h == nullptr || out == nullptr) return -1;
    std::memset(out, 0, sizeof(*out));
    out->serve = h->serve ? 1 : 0;
    out->use_mtp = h->mtp_on() ? 1 : 0;
    out->spec_corrupt = h->spec_corrupt;
    out->commit_async = h->commit_async ? 1 : 0;
    out->hist_n = h->hist_n;
    out->d_hist = h->d_hist;
    out->hist_cap = h->hist_cap;
    out->pool = (void*) h->pool;
    out->pool_user = h->pool_user;
    return 0;
}

extern "C" int tk_qwen4exp_p4_hook(tk_qwen4exp_p4_hooks_t* h, int which, int64_t a, int64_t b) {
    if (h == nullptr) return -1;
    switch (which) {
    case TK_P4_HOOK_PRE_WINDOW: return h->pre_window ? h->pre_window() : 1;
    case TK_P4_HOOK_ENSURE_KV: return h->ensure_kv ? h->ensure_kv(a, (int) b) : 1;
    case TK_P4_HOOK_WINDOW_STARTED: if (h->window_started) h->window_started(); return 1;
    case TK_P4_HOOK_POST_WINDOW: if (h->post_window) h->post_window(); return 1;
    case TK_P4_HOOK_FIRST_VERIFIED: return h->first_verified ? h->first_verified() : 0;
    case TK_P4_HOOK_COMMIT_BEGIN: if (h->commit_begin) h->commit_begin(a); return 1;
    case TK_P4_HOOK_JOIN_ADAPT: return h->join_adapt ? h->join_adapt() : 1;
    case TK_P4_HOOK_COMMIT_DONE: if (h->commit_done) h->commit_done(); return 1;
    case TK_P4_HOOK_MARK_NONFINITE: if (h->mark_nonfinite) h->mark_nonfinite(); return 1;
    case TK_P4_HOOK_DRAFT_DONE: if (h->draft_done) h->draft_done((int) a); return 1;
    default: return -1;
    }
}

extern "C" const char* tk_qwen4exp_p4_drive_failed(tk_qwen4exp_p4_hooks_t* h) {
    return (h != nullptr && h->drive_failed) ? h->drive_failed() : nullptr;
}

extern "C" int tk_qwen4exp_p4_ver_view(tk_qwen4exp_p4_hooks_t* h, tk_qwen4exp_p4_ver_view_t* out) {
    if (h == nullptr || h->ver == nullptr || out == nullptr) return -1;
    return Qwen4expP4Access::ver_view(*h->ver, h->ss, out);
}

extern "C" int tk_qwen4exp_p4_mtp_view(tk_qwen4exp_p4_hooks_t* h, tk_qwen4exp_p4_mtp_view_t* out) {
    if (h == nullptr || h->mtp == nullptr || !h->mtp_on() || out == nullptr) return -1;
    return Qwen4expP4Access::mtp_view(*h->mtp, out);
}

extern "C" int tk_qwen4exp_p4_gate_view(void* adapt_gate, tk_qwen4exp_p4_gate_view_t* out) {
    if (adapt_gate == nullptr || out == nullptr) return -1;
    auto* g = (strata::core::AdaptGate*) adapt_gate;
    static_assert(sizeof(std::atomic<int32_t>) == 4, "atomic<int32_t> view");
    out->landed = (const int32_t*) &g->landed;
    out->active = (const uint8_t*) &g->active;
    out->hland = g->hland;
    out->has = g->has.data();
    out->n_has = g->has.size();
    return 0;
}

extern "C" int tk_qwen4exp_p4_ver_prepare(void* ver, int t, void** out_exec, int* out_ar_on, char* err, size_t err_cap) {
    if (ver == nullptr) return -1;
    std::string e;
    if (Qwen4expP4Access::ver_prepare(*(strata::core::Verifier*) ver, t, out_exec, out_ar_on, e) != 0) {
        put_err(e, err, err_cap);
        return -1;
    }
    return 0;
}

extern "C" int tk_qwen4exp_p4_ver_release(void* ver, int timeout_ms) {
    return (ver != nullptr && ((strata::core::Verifier*) ver)->release_gpu_waits(timeout_ms)) ? 1 : 0;
}

extern "C" int tk_qwen4exp_p4_mtp_prepare(void* mtp, int t, int coupled, int max_steps, void** out_round, void** out_steps,
                                          char* err, size_t err_cap) {
    if (mtp == nullptr) return -1;
    std::string e;
    if (Qwen4expP4Access::mtp_prepare(*(strata::core::MtpDrafter*) mtp, t, coupled != 0, max_steps, out_round, out_steps,
                                    e) != 0) {
        put_err(e, err, err_cap);
        return -1;
    }
    return 0;
}

extern "C" int tk_qwen4exp_p4_ple_rows(void* ss, const int32_t* tokens, int n, int32_t* prev, uint32_t* out_rows) {
    if (ss == nullptr || tokens == nullptr || prev == nullptr || out_rows == nullptr) return -1;
    strata::core::SessionState& s = *(strata::core::SessionState*) ss;
    if (s.ple.table == nullptr) return -1;
    for (int t = 0; t < n; ++t) {
        uint32_t* rows = out_rows + (size_t) t * strata::kernels::PLE_N_HEADS;
        strata::kernels::ngram_rows(&tokens[t], prev, 1, s.ple.consts, rows);
        prev[0] = prev[1];
        prev[1] = tokens[t];
        s.ple.table->prefetch_rows(rows);
    }
    return 0;
}

extern "C" int tk_qwen4exp_p4_ple_gather(void* ss, const uint32_t* rows, size_t n_tokens, float* out, char* err,
                                         size_t err_cap) {
    if (ss == nullptr) return -1;
    strata::core::SessionState& s = *(strata::core::SessionState*) ss;
    std::string e;
    if (s.ple.table == nullptr || !s.ple.table->gather_batch(rows, n_tokens, out, e)) {
        put_err(e.empty() ? std::string("the PLE table is not open") : e, err, err_cap);
        return -1;
    }
    return 0;
}

extern "C" int tk_qwen4exp_p4_sample_tokens(const float* logits, int n_tokens, int n_vocab, const int32_t* history,
                                            int history_len, const void* sampling, int32_t* out, void* stream) {
    if (sampling == nullptr) return -1;
    strata::kernels::sample_tokens(logits, n_tokens, n_vocab, history, history_len,
                                   *(const strata::kernels::SamplerParams*) sampling, out, stream);
    return 0;
}

extern "C" int tk_qwen4exp_p4_sample_tokens_spec(const float* logits, int n_tokens, int n_vocab, const int32_t* history,
                                                 int history_len, const void* sampling, const int32_t* d_drafts,
                                                 const int32_t* d_q, int n_q, int32_t* out, void* stream) {
    if (sampling == nullptr) return 0;
    return strata::kernels::sample_tokens_spec(logits, n_tokens, n_vocab, history, history_len,
                                               *(const strata::kernels::SamplerParams*) sampling, d_drafts, d_q, n_q, out,
                                               stream)
               ? 1 : 0;
}

extern "C" void tk_qwen4exp_p4_progress(const char* where, int64_t detail, int kind) {
    if (kind == 0) strata::core::progress_at(where, detail);
    else if (kind == 1) strata::core::progress_tick();
    else strata::core::progress_beat();
}

extern "C" int tk_qwen4exp_p4_env_flags(void) {
    int f = 0;
#if defined(STRATA_USE_HIP)
    f |= 1;
    const char* v = std::getenv("STRATA_ONE_TOKEN_COMMIT");   // HIP: opt-in (=1), verify.cpp one_token_self_commit
    if (v != nullptr && std::atoi(v) != 0) f |= 2;
#else
    const char* v = std::getenv("STRATA_ONE_TOKEN_COMMIT");
    if (v == nullptr || std::atoi(v) != 0) f |= 2;
#endif
    if (std::getenv("STRATA_COMMIT_SYNC") == nullptr) f |= 4;
    return f;
}

// ================================ phase 2a: capture in Rust ================================


#define P4_TRY(body)                                                    \
    try { body; return 0; } catch (const std::exception& e) {           \
        std::fprintf(stderr, "talus verify commit: %s\n", e.what());   \
        return -1;                                                      \
    }
#define P4_TRY_ERR(body)                                                \
    try { body; return 0; } catch (const std::exception& e) {           \
        put_err(std::string("verify commit: ") + e.what(), err, err_cap); \
        return -1;                                                      \
    }

extern "C" int tk_qwen4exp_p4_capture_view(void* ver, tk_qwen4exp_p4_capture_view_t* out) {
    return (ver && out) ? Qwen4expP4Access::cap_view(*(strata::core::Verifier*) ver, out) : -1;
}
extern "C" int tk_qwen4exp_p4_qsa_state(void* ver, int64_t qi, tk_qwen4exp_p4_qsa_state_t* out) {
    return ver ? Qwen4expP4Access::qsa_state(*(strata::core::Verifier*) ver, qi, out) : -1;
}
extern "C" const float* tk_qwen4exp_p4_layer_weight(void* ver, int64_t layer, const char* suffix, char* err, size_t err_cap) {
    if (ver == nullptr || suffix == nullptr) return nullptr;
    std::string e;
    const float* p = Qwen4expP4Access::weight(*(strata::core::Verifier*) ver, layer, suffix, e);
    if (p == nullptr) put_err(e, err, err_cap);
    return p;
}
extern "C" int tk_qwen4exp_p4_record_window(void* ver, int t, char* err, size_t err_cap) {
    if (ver == nullptr) return -1;
    std::string e;
    if (!Qwen4expP4Access::record(*(strata::core::Verifier*) ver, t, e)) { put_err(e, err, err_cap); return -1; }
    return 0;
}
extern "C" int tk_qwen4exp_p4_window_instantiate(void* ver, int t, void* graph, char* err, size_t err_cap) {
    if (ver == nullptr || graph == nullptr) return -1;
    std::string e;
    if (!Qwen4expP4Access::instantiate(*(strata::core::Verifier*) ver, t, (cudaGraph_t) graph, e)) { put_err(e, err, err_cap); return -1; }
    return 0;
}
extern "C" int tk_qwen4exp_p4_k_copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    P4_TRY(strata::kernels::copy_i32_from_mapped(dst, src, n, stream))
}
extern "C" int tk_qwen4exp_p4_k_copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    P4_TRY(strata::kernels::copy_from_mapped(dst, src, n, stream))
}
extern "C" int tk_qwen4exp_p4_k_copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n,
                                             void* stream) {
    P4_TRY(strata::kernels::copy_indexed(dst, src, stride, index, n, stream))
}
extern "C" int tk_qwen4exp_p4_k_gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep,
                                                void* stream) {
    P4_TRY(strata::kernels::gdn_conv_commit(history, qkv, channels, n_keep, stream))
}
extern "C" int tk_qwen4exp_p4_k_gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate,
                                                    const float* beta, const float* z, const float* gamma, float eps,
                                                    float* y, int h_k, int h_v, int n_tok, const int32_t* n_keep,
                                                    void* stream, int t_out_begin) {
    P4_TRY(strata::kernels::gdn_step_norm_multi(state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok,
                                                n_keep, stream, t_out_begin))
}
extern "C" int tk_qwen4exp_p4_k_gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv,
                                                  int64_t qkv_stride, const float* h, int64_t h_stride, const float* gate,
                                                  const float* beta, int64_t gb_stride, int conv_channels, int h_k, int h_v,
                                                  int n_max, int n_layers, const int32_t* n_keep, void* stream) {
    try {
        return strata::kernels::gdn_commit_layers(state, state_stride, conv_off, qkv, qkv_stride, h, h_stride, gate, beta,
                                                  gb_stride, conv_channels, h_k, h_v, n_max, n_layers, n_keep, stream)
                   ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "talus verify commit: %s\n", e.what());
        return -1;
    }
}
extern "C" int tk_qwen4exp_p4_k_indexer_append_steps_layers(void* ver, const float* raw0, int64_t raw_stride,
                                                            const int32_t* rel_pos, int pos_stride, int n_steps,
                                                            int32_t pos_base, float eps, const float* const* gamma,
                                                            const float* const* tail_src, float* const* tail,
                                                            float* const* dead, float* const* pooled,
                                                            int32_t* const* block_pos, int n_layers, int64_t max_cells,
                                                            void* stream, char* err, size_t err_cap) {
    if (ver == nullptr || n_layers < 0 || n_layers > strata::kernels::QsaIndexerLayers::kMax) return -1;
    strata::kernels::QsaIndexerLayers il;
    for (int i = 0; i < n_layers; ++i) {
        il.gamma[i] = gamma[i]; il.tail_src[i] = tail_src[i]; il.tail[i] = tail[i]; il.dead[i] = dead[i];
        il.pooled[i] = pooled[i]; il.block_pos[i] = block_pos[i];
    }
    const strata::kernels::QsaShapes s = Qwen4expP4Access::shapes(*(strata::core::Verifier*) ver);
    P4_TRY_ERR(strata::kernels::native_qsa_indexer_append_steps_layers(raw0, raw_stride, rel_pos, pos_stride, n_steps,
                                                                       pos_base, eps, il, n_layers, s, max_cells,
                                                                       strata::kernels::rope_scaling(), stream))
}
extern "C" int tk_qwen4exp_p4_k_indexer_append_steps(void* ver, const float* raw, const int32_t* rel_pos, int pos_stride,
                                                     int n_steps, int32_t pos_base, const float* gamma, float eps,
                                                     const tk_qwen4exp_p4_qsa_state_t* st, void* stream, char* err,
                                                     size_t err_cap) {
    if (ver == nullptr || st == nullptr) return -1;
    const strata::kernels::QsaIndexerBuffers ib{st->idx_tail, st->idx_dead, st->idx_pooled, st->idx_block_pos};
    const strata::kernels::QsaShapes s = Qwen4expP4Access::shapes(*(strata::core::Verifier*) ver);
    P4_TRY_ERR(strata::kernels::native_qsa_indexer_append_steps(raw, rel_pos, pos_stride, n_steps, pos_base, gamma, eps, ib,
                                                                s, st->max_cells, strata::kernels::rope_scaling(), stream))
}
extern "C" int tk_qwen4exp_p4_k_indexer_append(void* ver, const float* raw, const int32_t* rel_pos, int32_t pos_base,
                                               const float* gamma, float eps, const tk_qwen4exp_p4_qsa_state_t* st,
                                               void* stream, char* err, size_t err_cap) {
    if (ver == nullptr || st == nullptr) return -1;
    const strata::kernels::QsaIndexerBuffers ib{st->idx_tail, st->idx_dead, st->idx_pooled, st->idx_block_pos};
    const strata::kernels::QsaShapes s = Qwen4expP4Access::shapes(*(strata::core::Verifier*) ver);
    P4_TRY_ERR(strata::kernels::native_qsa_indexer_append(raw, rel_pos, pos_base, gamma, eps, ib, s, st->max_cells,
                                                          strata::kernels::rope_scaling(), stream))
}

// ================================ phase 2b: the C++ entry points hand over to Rust ================================

namespace {
tk_qwen4exp_p4_ops_t g_p4_ops{};
int ops_run(void* u, strata::core::Verifier* v, int T, const int32_t* tokens, int64_t pos0, void* pool, void* pool_user,
            int32_t* out, const int32_t* q, int nq, char* err, size_t cap) {
    return g_p4_ops.ver_run(u, v, T, tokens, pos0, pool, pool_user, out, q, nq, err, cap);
}
int ops_commit(void* u, strata::core::Verifier* v, int n_keep, int async, char* err, size_t cap) {
    return g_p4_ops.ver_commit(u, v, n_keep, async, err, cap);
}
int ops_draft(void* u, strata::core::MtpDrafter* m, int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts,
              float* probs, float min_p, int* n_drafts, char* err, size_t cap) {
    return g_p4_ops.mtp_draft(u, m, T, tokens, p, a, drafts, probs, min_p, n_drafts, err, cap);
}
int ops_prefill(void* u, strata::core::MtpDrafter* m, const float* r, const int32_t* nt, int64_t n, int64_t c0, int sync,
                char* err, size_t cap) {
    return g_p4_ops.mtp_prefill(u, m, r, nt, n, c0, sync, err, cap);
}
int ops_wait(void* u, strata::core::Verifier* v, char* err, size_t cap) { return g_p4_ops.ver_wait_commit(u, v, err, cap); }
int ops_copy(void* u, const strata::core::Verifier* v, int t, float* host) { return g_p4_ops.ver_copy_logits(u, v, t, host); }
int ops_dsamp(void* u, strata::core::MtpDrafter* m, const void* sp) { return g_p4_ops.mtp_set_draft_sampling(u, m, sp); }
int ops_dhist(void* u, strata::core::MtpDrafter* m, const int32_t* tail, int64_t n, int32_t next) {
    return g_p4_ops.mtp_set_draft_history(u, m, tail, n, next);
}
}  // namespace

extern "C" void tk_qwen4exp_p4_install(const tk_qwen4exp_p4_ops_t* ops) {
    auto& vo = strata::core::g_qwen4exp_p4_verify_ops;
    auto& mo = strata::core::g_qwen4exp_p4_mtp_ops;
    if (ops == nullptr) {
        vo = {};
        mo = {};
        g_p4_ops = {};
        return;
    }
    g_p4_ops = *ops;
    vo.user = ops->user;
    vo.run = ops->ver_run ? &ops_run : nullptr;
    vo.commit = ops->ver_commit ? &ops_commit : nullptr;
    mo.user = ops->user;
    mo.draft = ops->mtp_draft ? &ops_draft : nullptr;
    mo.prefill = ops->mtp_prefill ? &ops_prefill : nullptr;
    vo.wait_commit = ops->ver_wait_commit ? &ops_wait : nullptr;
    vo.copy_logits = ops->ver_copy_logits ? &ops_copy : nullptr;
    mo.set_draft_sampling = ops->mtp_set_draft_sampling ? &ops_dsamp : nullptr;
    mo.set_draft_history = ops->mtp_set_draft_history ? &ops_dhist : nullptr;
}
extern "C" int tk_qwen4exp_p4_ver_view_of(void* ver, tk_qwen4exp_p4_ver_view_t* out) {
    return (ver && out) ? Qwen4expP4Access::ver_view(*(strata::core::Verifier*) ver, nullptr, out) : -1;
}
extern "C" int tk_qwen4exp_p4_mtp_view_of(void* mtp, tk_qwen4exp_p4_mtp_view_t* out) {
    return (mtp && out) ? Qwen4expP4Access::mtp_view(*(strata::core::MtpDrafter*) mtp, out) : -1;
}
extern "C" int tk_qwen4exp_p4_mtp_pf_view(void* mtp, tk_qwen4exp_p4_mtp_pf_view_t* out) {
    return (mtp && out) ? Qwen4expP4Access::mtp_pf_view(*(strata::core::MtpDrafter*) mtp, out) : -1;
}
extern "C" int tk_qwen4exp_p4_mtp_prefill_graph(void* mtp, int t, int dev_inputs, void** out_exec, char* err, size_t err_cap) {
    if (mtp == nullptr || out_exec == nullptr) return -1;
    std::string e;
    if (Qwen4expP4Access::mtp_prefill_graph(*(strata::core::MtpDrafter*) mtp, t, dev_inputs != 0, out_exec, e) != 0) {
        put_err(e, err, err_cap);
        return -1;
    }
    return 0;
}
extern "C" int tk_qwen4exp_p4_mtp_pf_upload(void* mtp, const int32_t* rec, int64_t n_ints, int32_t** out_dev, char* err,
                                            size_t err_cap) {
    if (mtp == nullptr || out_dev == nullptr) return -1;
    std::string e;
    if (Qwen4expP4Access::mtp_pf_upload(*(strata::core::MtpDrafter*) mtp, rec, n_ints, out_dev, e) != 0) {
        put_err(e, err, err_cap);
        return -1;
    }
    return 0;
}

// ================================ phase 3: the MTP layer's recording from Rust ================================

extern "C" int tk_qwen4exp_p4_mtp_rec_view(void* mtp, tk_qwen4exp_p4_mtp_rec_view_t* out) {
    return (mtp && out) ? Qwen4expP4Access::mtp_rec_view(*(strata::core::MtpDrafter*) mtp, out) : -1;
}
extern "C" const void* tk_qwen4exp_p4_mtp_tensor(void* mtp, const char* name, int kind, int* out_type) {
    return (mtp && name) ? Qwen4expP4Access::mtp_tensor(*(strata::core::MtpDrafter*) mtp, name, kind, out_type) : nullptr;
}
#define P4M(expr)                                                           \
    try { return (expr); } catch (const std::exception& e) {                \
        put_err(std::string("mtp: ") + e.what(), err, err_cap);             \
        return -1;                                                          \
    }
extern "C" int tk_qwen4exp_p4_m_op(void* mtp, int op, int t, int n, int j, const int32_t* step, void* stream, char* err,
                                   size_t err_cap) {
    if (mtp == nullptr) return -1;
    auto& m = *(strata::core::MtpDrafter*) mtp;
    cudaStream_t cs = (cudaStream_t) stream;
    std::string e;
    int rc = -1;
    try {
        switch (op) {
        case TK_P4_M_EMBED: rc = Qwen4expP4Access::m_embed(m, n, cs, e); break;
        case TK_P4_M_GR_READ: rc = Qwen4expP4Access::m_gr_read(m, t, n, cs); break;
        case TK_P4_M_GR_READ_ROW: rc = Qwen4expP4Access::m_gr_read_row(m, t, cs); break;
        case TK_P4_M_KV_APPEND: rc = Qwen4expP4Access::m_kv_append(m, step, n, t != 0, cs); break;
        case TK_P4_M_ATTN: rc = Qwen4expP4Access::m_attn(m, step, n, cs); break;
        case TK_P4_M_SHARED: rc = Qwen4expP4Access::m_shared(m, n, cs); break;
        case TK_P4_M_ROUTE: rc = Qwen4expP4Access::m_route(m, n, cs, e); break;
        case TK_P4_M_EXPERTS: rc = Qwen4expP4Access::m_experts(m, n, cs); break;
        case TK_P4_M_COMBINE: rc = Qwen4expP4Access::m_combine(m, t, n, cs); break;
        case TK_P4_M_GR_WRITE: rc = Qwen4expP4Access::m_gr_write(m, t, n, cs); break;
        case TK_P4_M_HEAD: rc = Qwen4expP4Access::m_head(m, n, t != 0, j, step, cs); break;
        case TK_P4_M_ROUND_TAIL: rc = Qwen4expP4Access::m_round_tail(m, t, j, cs); break;
        default: e = "mtp: unknown op"; rc = -1;
        }
    } catch (const std::exception& ex) {
        e = std::string("mtp: ") + ex.what();
        rc = -1;
    }
    if (rc < 0) put_err(e.empty() ? std::string("mtp: op failed") : e, err, err_cap);
    return rc;
}
extern "C" int tk_qwen4exp_p4_mtp_instantiate(void* mtp, int which, int idx, void* graph, char* err, size_t err_cap) {
    if (mtp == nullptr || graph == nullptr) return -1;
    std::string e;
    if (Qwen4expP4Access::mtp_instantiate(*(strata::core::MtpDrafter*) mtp, which, idx, graph, e) != 0) {
        put_err(e, err, err_cap);
        return -1;
    }
    return 0;
}
// generic launch-only kernels the MTP layer uses (device pointers throughout)
extern "C" int tk_qwen4exp_p4_g_rms_norm(const float* x, const float* gamma, float* out, int cols, int rows, float eps,
                                         void* stream) {
    P4_TRY(strata::kernels::native_qsa_rms_norm_weighted(x, gamma, out, cols, rows, eps, stream))
}
extern "C" int tk_qwen4exp_p4_g_rms_norm_grouped(const float* x, const float* gamma, float* out, int cols, int groups,
                                                 int rows, float eps, void* stream) {
    P4_TRY(strata::kernels::native_qsa_rms_norm_grouped(x, gamma, out, cols, groups, rows, eps, stream))
}
extern "C" int tk_qwen4exp_p4_g_quantize_q8_1(const float* x, void* xq, int n_in, int ncols, void* stream) {
    P4_TRY(strata::kernels::native_quantize_q8_1(x, xq, n_in, ncols, stream))
}
extern "C" int tk_qwen4exp_p4_g_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols,
                                     void* stream) {
    P4_TRY(strata::kernels::native_mmvq(type, w, xq, y, n_in, n_out, ncols, stream))
}
extern "C" int tk_qwen4exp_p4_g_add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc,
                                                      int n_tok, void* stream) {
    P4_TRY(strata::kernels::add_streams_broadcast(h, e, R, n_embd, hc, n_tok, stream))
}
extern "C" int tk_qwen4exp_p4_g_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out, int rows,
                                              int head_dim, int n_rot, float eps, const int32_t* pos, void* stream) {
    P4_TRY(strata::kernels::native_qsa_rms_norm_rope(x, in_stride, gamma, out, rows, head_dim, n_rot, eps,
                                                     strata::kernels::rope_scaling(), pos, stream))
}
extern "C" int tk_qwen4exp_p4_g_rope(const float* x, float* out, int rows, int head_dim, int n_rot, const int32_t* pos,
                                     void* stream) {
    P4_TRY(strata::kernels::native_rope_apply(x, out, rows, head_dim, n_rot, strata::kernels::rope_scaling(), pos, stream))
}
extern "C" int tk_qwen4exp_p4_g_rope_neox(const float* x, float* out, int64_t rows, int head_dim, int n_rot,
                                          const float* cos_tab, const float* sin_tab, const int32_t* pos, void* stream) {
    P4_TRY(strata::kernels::rope_neox_apply(x, out, rows, head_dim, n_rot, cos_tab, sin_tab, pos, stream))
}
extern "C" int tk_qwen4exp_p4_g_fwht256(float* x, int64_t n_rows, void* stream) {
    P4_TRY(strata::kernels::fwht256_inplace_cuda(x, n_rows, stream))
}
extern "C" int tk_qwen4exp_p4_g_copy_2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                                        void* stream) {
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, (cudaStream_t) stream) ==
                   cudaSuccess
               ? 0 : -1;
}
extern "C" int tk_qwen4exp_p4_g_gate_apply(const float* attn, const float* qfull, float* out, int n_head, int head_dim,
                                           void* stream) {
    P4_TRY(strata::kernels::native_qsa_gate_apply(attn, qfull, out, n_head, head_dim, stream))
}
extern "C" int tk_qwen4exp_p4_env_mtp(void) {   // the kernel choices mtp.cpp reads once (bits)
    int f = 0;
    if (strata::kernels::native_moe_combine_enabled()) f |= 1;
    const char* v = std::getenv("STRATA_FUSE_HEAD_GR");
    if (v != nullptr && std::atoi(v) != 0) f |= 2;
#if defined(STRATA_USE_HIP)
    const char* hm = std::getenv("STRATA_HEAD_MIX_MULTI");   // HIP: opt-in (=1)
    if (hm != nullptr && std::atoi(hm) != 0) f |= 4;
#else
    const char* hm = std::getenv("STRATA_HEAD_MIX_MULTI");
    if (hm == nullptr || std::atoi(hm) != 0) f |= 4;
    const char* sb = std::getenv("STRATA_MTP_SHARED_BRANCH");
    if (sb == nullptr || std::atoi(sb) != 0) f |= 32;   // the shared expert's branch (CUDA only)
#endif
    const char* nm = std::getenv("STRATA_NO_MULTI_GR");
    if (nm != nullptr && nm[0] != '\0' && nm[0] != '0') f |= 8;
    const char* nb = std::getenv("STRATA_NO_BATCH_KV_STEP");
    if (nb != nullptr && nb[0] != '\0' && nb[0] != '0') f |= 16;
    return f;
}
