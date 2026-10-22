// src/program/qwen4exp_r2v.cpp - qwen4exp-r2 (Piece 4 residuals): the C ABI under the Rust setup of the Verifier and the
// MTP drafter (rust/crates/talus-model/src/qwen4exp/verify_mtp/setup.rs).
//
// Rust runs Verifier::init (checks, mapped staging, the plan layout, the device arena's carve, streams, events, the
// env switches, the all-resident decision), Verifier::set_logit_bias, MtpDrafter::load (dense.txt / dense.bin /
// experts.bin, the K/V state's flag dance and fallback, staging, the arena, streams) and MtpDrafter::bind (the draft
// head subset).  This file only hands Rust the facts it cannot compute (the objects' settings, the kernels' byte
// counts), lets the C++ objects ADOPT what Rust allocated (Qwen4expR2vAccess is a friend of both), and wraps what is the
// C++ objects' own: the K/V state carve (layer.cpp), the Q4_0 conversions, the coupled sampler setup.
//
// TALUS_FN_R2V_CHECK=1: before each Rust init / load / bind, the C++ body runs on a scratch object and the two layouts
// are compared buffer by buffer (offsets in the arena, sizes, scalars, the uploaded weights' bytes).
#include "talus_qwen4exp_kernels.h"

#include "strata/core/coupled_draft.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/mtp.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gfx_arch.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {
tk_qwen4exp_r2v_ops_t g_ops{};
thread_local bool g_bypass = false;   // the check mode's scratch object runs the C++ body
bool check_on() {
    static const bool v = [] { const char* e = std::getenv("TALUS_FN_R2V_CHECK"); return e && e[0] == '1'; }();
    return v;
}
void put_err(const std::string& e, char* err, size_t cap) {
    if (err != nullptr && cap > 0) std::snprintf(err, cap, "%s", e.c_str());
}
uint64_t alloc_size(const void* p) {
    if (p == nullptr) return 0;
    void* base = nullptr;
    size_t sz = 0;
#if defined(STRATA_USE_HIP)
    if (hipMemGetAddressRange(&base, &sz, (hipDeviceptr_t) p) != hipSuccess) { (void) cudaGetLastError(); return 0; }
#else
    (void) base;
#endif
    return (uint64_t) sz;
}
bool dev_bytes_equal(const void* a, const void* b, uint64_t n) {
    if (n == 0) return true;
    if (a == nullptr || b == nullptr) return a == b;
    std::vector<uint8_t> x(64u << 20), y(64u << 20);
    for (uint64_t off = 0; off < n;) {
        const uint64_t k = std::min<uint64_t>(x.size(), n - off);
        if (cudaMemcpy(x.data(), (const uint8_t*) a + off, k, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(y.data(), (const uint8_t*) b + off, k, cudaMemcpyDeviceToHost) != cudaSuccess)
            return false;
        if (std::memcmp(x.data(), y.data(), k) != 0) return false;
        off += k;
    }
    return true;
}
}  // namespace

// The Verifier's members in init's order: the mapped staging (host, device alias) and the device arena's carve.
#define R2V_VER_MAPPED(X)                                                                                          \
    X(h_tok_, m_tok_) X(h_step_, m_step_) X(h_pos_, m_pos_) X(h_commit_, m_commit_) X(h_ple_, m_ple_)             \
    X(h_out_, m_out_) X(h_nf_, m_nf_) X(h_x_, m_x_) X(h_ids_, m_ids_) X(h_w_, m_w_) X(h_seq_, m_seq_)             \
    X(h_flag_, m_flag_) X(h_flagA_, m_flagA_) X(h_flagB_, m_flagB_) X(h_plan_err_, m_plan_err_)                   \
    X(h_ymiss_, m_ymiss_) X(h_plan_, m_plan_)
#define R2V_VER_ARENA(X)                                                                                           \
    X(tok_) X(step_) X(pos_) X(commit_) X(ple_) X(emb_) X(R_) X(mixed_) X(bo_) X(inj_) X(inj2_) X(lo_) X(rs_)       \
    X(xn_) X(xq_) X(sh_xq_) X(xil_) X(qkv_L_) X(h_L_) X(gate_L_) X(beta_L_) X(z_) X(y_) X(y_dummy_) X(qfull_)        \
    X(qcur_) X(kcur_) X(vcur_) X(idx_raw_L_) X(qidx_) X(scores_) X(sel_) X(attn_) X(attn32_) X(attn_scratch_)       \
    X(tail_snap_) X(logits_) X(w_) X(ids_) X(shared_) X(parts_) X(hit_out_) X(hit_slot_) X(hit_dst_) X(hit_count_)  \
    X(plan_) X(staging_) X(hit_xq_) X(hit_xs_) X(nat_xq_) X(hit_scratch_) X(head_mixed_) X(head_inj_) X(sh_bf16_)   \
    X(sh_gate_) X(sh_up_) X(sh_g_) X(head_logits_) X(arg_scratch_) X(one_) X(hist_snap_) X(ple_key_) X(ple_val_)
// The drafter's: mapped staging, then the arena (gr_mem: the GR workspace's block, not a member)
#define R2V_MTP_MAPPED(X)                                                                                          \
    X(h_tok_, m_tok_) X(h_step_, m_step_) X(h_pos_, m_pos_) X(h_row_, m_row_) X(h_out_, m_out_) X(h_prob_, m_prob_) \
    X(h_force_, m_force_)
#define R2V_MTP_ARENA(X)                                                                                           \
    X(tok_) X(step_) X(pos_) X(row_) X(ident_) X(Rin_) X(R_) X(emb_) X(en_) X(e2_) X(hn_) X(h2_) X(mixed_) X(inj_)  \
    X(inj2_) X(lo_) X(rs_) X(bo_) X(xn_) X(xq_) X(qfull_) X(qcur_) X(kcur_) X(vcur_) X(attn_) X(attn32_)            \
    X(attn_scratch_) X(logits_) X(w_) X(ids_) X(shared_) X(parts_) X(y_) X(sample_)
#define R2V_MTP_ARENA2(X)                                                                                          \
    X(hit_slot_) X(hit_dst_) X(hit_count_) X(grp_ptr_) X(grp_start_) X(grp_counts_) X(hit_xq_) X(hit_xs_)            \
    X(hit_scratch_) X(sh_scratch_) X(x_bf16_) X(out_ids_) X(probs_) X(arg_scratch_) X(top_scratch_) X(dummy_inj_)

#define R2V_COUNT1(a) +1
#define R2V_COUNT2(a, b) +1
static_assert(0 R2V_VER_MAPPED(R2V_COUNT2) == TK_R2V_VER_MAPPED, "qwen4exp-r2v: Verifier mapped list");
static_assert(0 R2V_VER_ARENA(R2V_COUNT1) == TK_R2V_VER_ARENA, "qwen4exp-r2v: Verifier arena list");
static_assert(0 R2V_MTP_MAPPED(R2V_COUNT2) == TK_R2V_MTP_MAPPED, "qwen4exp-r2v: MTP mapped list");
static_assert(0 R2V_MTP_ARENA(R2V_COUNT1) + 1 R2V_MTP_ARENA2(R2V_COUNT1) == TK_R2V_MTP_ARENA, "qwen4exp-r2v: MTP arena list");

namespace strata::core {

// the hooks verify.cpp / mtp.cpp call (their qwen4exp-r2v blocks)
bool qwen4exp_r2v_ver_reference(Verifier& v, const WeightTable& wt, const ModelGeometry& g, SessionState& ss,
                              const VerifyHits& hits, const NativeHead* head, int max_t);
int qwen4exp_r2v_ver_init(Verifier& v, bool check, std::string& err);
int qwen4exp_r2v_ver_bias(Verifier& v, const std::vector<float>& bias, std::string& err);
bool qwen4exp_r2v_mtp_reference(MtpDrafter& m, const std::string& rt_dir, const ModelGeometry& g, SessionState& ss,
                              int max_t, int64_t window, const MtpDrafter* shared);
int qwen4exp_r2v_mtp_load(MtpDrafter& m, int64_t window, bool check, std::string& err);
bool qwen4exp_r2v_bind_reference(MtpDrafter& m, const WeightTable& wt, const NativeHead* head, const float* window_R,
                               const MtpDrafter* shared);
int qwen4exp_r2v_mtp_bind(MtpDrafter& m, bool check, std::string& err);

struct Qwen4expR2vAccess {
    // ---------------------------------------------------------------- Verifier
    static bool ver_eligible(const Verifier& v) {
        return v.g_ != nullptr && v.lb_ == 0 && (v.le_ < 0 || v.le_ == v.g_->n_layers) && v.hand_in_ == nullptr &&
               v.hand_out_ == nullptr && v.next_ == nullptr && v.ext_stream_ == nullptr;
    }
    static int ver_facts(Verifier& v, tk_qwen4exp_r2v_ver_facts_t* o) {
        std::memset(o, 0, sizeof(*o));
        if (v.g_ == nullptr || v.ss_ == nullptr || v.wt_ == nullptr) return -1;
        const ModelGeometry& g = *v.g_;
        const SessionState& ss = *v.ss_;
        o->eligible = ver_eligible(v) ? 1 : 0;
        o->max_t = v.max_t_;
        std::string why;
        o->layer_ok = layer_verify_compatible(why) ? 1 : 0;
        std::snprintf(o->why, sizeof o->why, "%s", why.c_str());
        o->fused_gr_ok = strata::kernels::fused_gr_supported(g.n_embd, g.hc, g.hc_lr) ? 1 : 0;
        o->always_publish = v.always_publish_ ? 1 : 0;
        o->peer_portable = peer_portable() ? 1 : 0;
#if defined(STRATA_USE_HIP)
        {
            int dev = 0;
            cudaDeviceProp p{};
            o->is_gfx12 = cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                          strata::kernels::gfx_arch_is_gfx12(p.gcnArchName);
        }
#endif
#if defined(STRATA_HIP_GFX906) || !defined(STRATA_USE_HIP)
        o->hip_gfx906 = 1;   // not the Rust plane's build: the C++ body runs
#endif
        o->device = v.device_;
        o->k = (int32_t) ss.k;
        o->ssm_state_size = (int32_t) g.ssm_state_size;
        o->ssm_d_conv = (int32_t) g.ssm_d_conv;
        o->n_layers = (int32_t) g.n_layers;
        o->n_expert = (int32_t) g.n_expert;
        o->n_embd = (int32_t) g.n_embd;
        o->hc = (int32_t) g.hc;
        o->hc_lr = (int32_t) g.hc_lr;
        o->conv_ch = (int32_t) g.ssm_conv_channels;
        o->value_dim = (int32_t) g.ssm_value_dim;
        o->v_heads = (int32_t) g.ssm_v_heads;
        o->n_head = (int32_t) g.n_head;
        o->head_dim = (int32_t) g.head_dim;
        o->n_head_kv = (int32_t) g.n_head_kv;
        o->idx_q_heads = (int32_t) g.idx_q_heads;
        o->idx_key_dim = (int32_t) g.idx_key_dim;
        o->n_ff = (int32_t) g.n_ff;
        o->n_gdn = (int32_t) g.n_gdn_layers();
        o->n_qsa = (int32_t) g.n_qsa_layers();
        o->verify_max_t = strata::kernels::kVerifyMaxT;
        o->cpu_maxt = strata::kernels::cpu::MAXT;
        o->step_count = strata::kernels::kStepCount;
        o->nonfinite_blocks = strata::kernels::kNonfiniteBlocks;
        o->ng_hist = strata::kernels::NG_HIST;
        o->ng_hc_dim = strata::kernels::NG_HC_DIM;
        const WeightRef* wo = v.wt_->find("output.weight");
        o->n_vocab = wo ? wo->ne1 : -1;
        const strata::kernels::QsaShapes s = qsa_shapes(g);
        o->max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
        o->idx_block = s.idx_block;
        o->cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
        o->attn_scratch_floats = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(o->cap, s);
        o->max_blob = (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
        const int T = v.max_t_ > 0 ? v.max_t_ : 1;
        const int max_in = (int) std::max<int64_t>(std::max<int64_t>(g.n_embd, g.ssm_value_dim), g.n_head * g.head_dim);
        o->q8_1_bytes = (uint64_t) strata::kernels::native_q8_1_bytes(max_in, T);
        o->q8_1_il_bytes = (uint64_t) strata::kernels::native_q8_1_il_bytes(max_in, 4);
        o->hit_grouped_scratch = strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) T * ss.k, g.n_embd, g.n_ff);
        o->native_expert_scratch = (uint64_t) strata::kernels::native_expert_scratch_bytes((int64_t) T * ss.k, g.n_ff);
        o->argmax_scratch = strata::kernels::argmax_rows_scratch_bytes(T);
        o->d_res = v.hits_.d_res;
        o->h_res = v.hits_.h_res;
        o->cache_base = v.hits_.cache_base;
        o->slot_off = v.hits_.slot_off;
        o->n_slots = v.hits_.n_slots;
        o->blob = v.hits_.blob;
        return 0;
    }
    static void ver_adopt(Verifier& v, const tk_qwen4exp_r2v_ver_state_t* st) {
        int i = 0;
#define R2V_ADOPT_M(h, m) v.h = (decltype(v.h)) st->mapped_h[i]; v.m = (decltype(v.m)) st->mapped_d[i]; ++i;
        R2V_VER_MAPPED(R2V_ADOPT_M)
#undef R2V_ADOPT_M
        v.plan_i32_ = st->plan_i32;
        v.sink_.counts = st->sink_counts;
        v.sink_.start = st->sink_start;
        v.sink_.dst = st->sink_dst;
        v.sink_.tok = st->sink_tok;
        v.sink_.ptr = st->sink_ptr;
        v.sink_.ptr2 = st->sink_ptr2;
        v.sink_.start2 = st->sink_start2;
        v.sink_.cap = st->sink_cap;
        v.sink_.publish = &Verifier::publish_plan;
        v.sink_.fetch = &Verifier::fetch_dma;
        v.sink_.ctx = &v;
        v.n_vocab_ = st->n_vocab;
        v.cap_ = st->cap;
        v.max_blocks_ = st->max_blocks;
        v.attn_scratch_floats_ = st->attn_scratch_floats;
        v.le_ = st->le;
        v.arena_ = st->arena;
        i = 0;
#define R2V_ADOPT_A(m) v.m = (decltype(v.m)) st->arena_ptrs[i++];
        R2V_VER_ARENA(R2V_ADOPT_A)
#undef R2V_ADOPT_A
        v.sink_.staging = (unsigned long long) v.staging_;
        v.sink_.staging_cap = st->staging_cap;
        v.copy_ = (cudaStream_t) st->copy;
        v.cs_ = (cudaStream_t) st->cs;
        v.sh_cs_ = (cudaStream_t) st->sh_cs;
        v.commit_done_ = (cudaEvent_t) st->commit_done;
        v.ev_fork_ = (cudaEvent_t) st->ev_fork;
        v.ev_join_ = (cudaEvent_t) st->ev_join;
        v.df_branch_ = st->df_branch != 0;
        v.df_fork_ = (cudaEvent_t) st->df_fork;
        for (int k = 0; k < 2; ++k) {
            v.df_side_[k] = (cudaStream_t) st->df_side[k];
            v.df_join_[k] = (cudaEvent_t) st->df_join[k];
        }
        v.all_resident_ = st->all_resident != 0;
        v.ar_off_ = false;
        v.h_res_ = v.hits_.h_res;
        v.device_plan_ = st->device_plan != 0;
        v.qcnt_ = (unsigned*) st->qcnt;
        v.skip_ = (uint32_t*) st->skip;
        v.slot_off_d_ = (unsigned long long*) st->slot_off_d;
    }
    // TALUS_FN_R2V_CHECK: the C++ body's layout on a scratch Verifier with the same settings
    struct VerRef {
        bool valid = false;
        uint64_t arena_bytes = 0;
        int64_t off[TK_R2V_VER_ARENA] = {};
        int64_t plan_i32 = 0, n_vocab = 0, cap = 0, max_blocks = 0, asf = 0, le = 0;
        int64_t sink[7] = {};
        int64_t sink_cap = 0, staging_cap = 0;
        bool all_resident = false, device_plan = false, df_branch = false, qcnt = false, skip = false, slot_off = false;
    };
    static VerRef& ver_ref() { static VerRef r; return r; }
    static void ver_capture(const Verifier& v, VerRef& r) {
        r.valid = true;
        r.arena_bytes = alloc_size(v.arena_);
        int i = 0;
#define R2V_OFF(m) r.off[i++] = v.m ? (int64_t) ((const uint8_t*) v.m - (const uint8_t*) v.arena_) : -1;
        R2V_VER_ARENA(R2V_OFF)
#undef R2V_OFF
        r.plan_i32 = v.plan_i32_; r.n_vocab = v.n_vocab_; r.cap = v.cap_; r.max_blocks = v.max_blocks_;
        r.asf = v.attn_scratch_floats_; r.le = v.le_;
        const uint8_t* hp = (const uint8_t*) v.h_plan_;
        const void* sp[7] = {v.sink_.counts, v.sink_.start, v.sink_.dst, v.sink_.tok, v.sink_.ptr, v.sink_.ptr2, v.sink_.start2};
        for (int k = 0; k < 7; ++k) r.sink[k] = sp[k] ? (int64_t) ((const uint8_t*) sp[k] - hp) : -1;
        r.sink_cap = v.sink_.cap; r.staging_cap = v.sink_.staging_cap;
        r.all_resident = v.all_resident_; r.device_plan = v.device_plan_; r.df_branch = v.df_branch_;
        r.qcnt = v.qcnt_ != nullptr; r.skip = v.skip_ != nullptr; r.slot_off = v.slot_off_d_ != nullptr;
    }
    static bool ver_reference(Verifier& v, const WeightTable& wt, const ModelGeometry& g, SessionState& ss,
                              const VerifyHits& hits, const NativeHead* head, int max_t) {
        ver_ref().valid = false;
        if (g_ops.ver_init == nullptr || g_bypass || !check_on()) return false;
        auto r = std::make_unique<Verifier>();
        r->lb_ = v.lb_; r->le_ = v.le_; r->hand_in_ = v.hand_in_; r->hand_out_ = v.hand_out_;
        r->always_publish_ = v.always_publish_; r->ext_stream_ = v.ext_stream_;
        r->sink_.pcie_mode = v.sink_.pcie_mode;
        if (!(v.lb_ == 0 && (v.le_ < 0 || v.le_ == g.n_layers) && v.hand_in_ == nullptr && v.hand_out_ == nullptr &&
              v.next_ == nullptr && v.ext_stream_ == nullptr))
            return false;
        std::string e;
        g_bypass = true;
        const bool ok = r->init(wt, g, ss, hits, head, max_t, e);
        g_bypass = false;
        if (!ok) {
            std::fprintf(stderr, "qwen4exp-r2v check: the C++ reference Verifier::init failed: %s\n", e.c_str());
            return false;
        }
        ver_capture(*r, ver_ref());
        return true;   // r's destructor frees the reference
    }
    static void ver_compare(const Verifier& v) {
        VerRef& a = ver_ref();
        if (!a.valid) return;
        VerRef b;
        ver_capture(v, b);
        int bad = 0;
        auto cmp = [&](const char* what, int64_t x, int64_t y) {
            if (x != y) {
                ++bad;
                std::fprintf(stderr, "qwen4exp-r2v check: Verifier %s: C++ %lld, Rust %lld\n", what, (long long) x, (long long) y);
            }
        };
        static const char* names[] = {
#define R2V_NAME(m) #m,
            R2V_VER_ARENA(R2V_NAME)
#undef R2V_NAME
        };
        for (int i = 0; i < TK_R2V_VER_ARENA; ++i) cmp(names[i], a.off[i], b.off[i]);
        cmp("arena bytes", (int64_t) a.arena_bytes, (int64_t) b.arena_bytes);
        cmp("plan_i32", a.plan_i32, b.plan_i32); cmp("n_vocab", a.n_vocab, b.n_vocab); cmp("cap", a.cap, b.cap);
        cmp("max_blocks", a.max_blocks, b.max_blocks); cmp("attn_scratch_floats", a.asf, b.asf); cmp("le", a.le, b.le);
        for (int k = 0; k < 7; ++k) cmp("sink offset", a.sink[k], b.sink[k]);
        cmp("sink cap", a.sink_cap, b.sink_cap); cmp("staging cap", a.staging_cap, b.staging_cap);
        cmp("all_resident", a.all_resident, b.all_resident); cmp("device_plan", a.device_plan, b.device_plan);
        cmp("df_branch", a.df_branch, b.df_branch); cmp("qcnt", a.qcnt, b.qcnt); cmp("skip", a.skip, b.skip);
        cmp("slot_off_d", a.slot_off, b.slot_off);
        std::fprintf(stderr, "qwen4exp-r2v check: Verifier::init layout %s (%d arena buffers, %.1f MiB)\n",
                     bad ? "DIFFERS" : "identical", TK_R2V_VER_ARENA, (double) b.arena_bytes / 1048576.0);
        a.valid = false;
    }
    static int bias_view(Verifier& v, const float** host, int64_t* n_host, float** device, int64_t* n_vocab,
                         int32_t* eligible) {
        *host = v.logit_bias_host_.data();
        *n_host = (int64_t) v.logit_bias_host_.size();
        *device = v.logit_bias_device_;
        *n_vocab = v.n_vocab_;
        int cur = -1;
        (void) cudaGetDevice(&cur);
        *eligible = (v.next_ == nullptr && v.g_ != nullptr && v.le_ == v.g_->n_layers && cur == v.device_) ? 1 : 0;
        return 0;
    }
    static void bias_adopt(Verifier& v, const float* bias, int64_t n, float* device) {
        v.logit_bias_host_.assign(bias, bias + n);
        v.logit_bias_device_ = device;
        v.sampling_.logit_bias = n == 0 ? nullptr : v.logit_bias_device_;
    }

    // ---------------------------------------------------------------- MtpDrafter
    static int mtp_facts(MtpDrafter& m, tk_qwen4exp_r2v_mtp_facts_t* o) {
        std::memset(o, 0, sizeof(*o));
        if (m.g_ == nullptr || m.ss_ == nullptr) return -1;
        const ModelGeometry& g = *m.g_;
        const SessionState& ss = *m.ss_;
        int cur = -1;
        (void) cudaGetDevice(&cur);
        o->eligible = cur == m.device_ ? 1 : 0;
        o->max_t = m.max_t_;
        o->device = m.device_;
        o->q4 = m.q4_; o->q4_head = m.q4_head_; o->force_on = m.force_on_;
#if defined(STRATA_USE_HIP)
        o->hip = 1;
#endif
#if defined(STRATA_HIP_GFX906)
        o->hip_gfx906 = 1;
#endif
        o->peer_portable = peer_portable() ? 1 : 0;
        o->n_embd = (int32_t) g.n_embd; o->hc = (int32_t) g.hc; o->hc_lr = (int32_t) g.hc_lr; o->k = (int32_t) ss.k;
        o->n_head = (int32_t) g.n_head; o->head_dim = (int32_t) g.head_dim; o->n_head_kv = (int32_t) g.n_head_kv;
        o->n_expert = (int32_t) g.n_expert; o->n_ff = (int32_t) g.n_ff;
        o->verify_max_t = strata::kernels::kVerifyMaxT;
        o->max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
        o->expert_blob = (int64_t) strata::kernels::cpu::BLOB;
        const int T = m.max_t_ > 0 ? m.max_t_ : 1;
        o->q8_1_bytes = (uint64_t) strata::kernels::native_q8_1_bytes((int) (g.n_head * g.head_dim), 8);
        const strata::kernels::GrShapes gsh{g.n_embd, g.hc, g.hc_lr};
        o->gr_ws_bytes = (uint64_t) strata::kernels::gr_workspace_bytes(gsh);
        o->hit_grouped_scratch = strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) T * ss.k, g.n_embd, g.n_ff);
        o->shared_scratch = strata::kernels::shared_expert_scratch_bytes(g.n_ff);
        o->argmax_scratch = strata::kernels::argmax_rows_scratch_bytes(T);
        o->top_scratch = strata::kernels::row_top_prob_scratch_bytes(T);
        o->rt_dir = m.rt_dir_.c_str();
        return 0;
    }
    static void mtp_set_weights(MtpDrafter& m, const tk_qwen4exp_r2v_tensor_t* t, int64_t n, void* dense, void* experts,
                                uint64_t vram_add) {
        if (t != nullptr) m.tensors_.clear();
        for (int64_t i = 0; t != nullptr && i < n; ++i) {
            MtpDrafter::Tensor x;
            x.name = t[i].name; x.kind = t[i].kind; x.rows = t[i].rows; x.cols = t[i].cols; x.off = t[i].off; x.bytes = t[i].bytes;
            m.tensors_.push_back(std::move(x));
        }
        if (dense != nullptr) m.dense_ = (uint8_t*) dense;
        if (experts != nullptr) m.experts_ = (uint8_t*) experts;
        m.vram_ += vram_add;
    }
    static uint64_t mtp_adopt(MtpDrafter& m, const tk_qwen4exp_r2v_mtp_state_t* st) {
        int i = 0;
#define R2V_ADOPT_M(h, d) m.h = (decltype(m.h)) st->mapped_h[i]; m.d = (decltype(m.d)) st->mapped_d[i]; ++i;
        R2V_MTP_MAPPED(R2V_ADOPT_M)
#undef R2V_ADOPT_M
        m.state_arena_ = st->state_arena;
        m.arena_ = st->arena;
        i = 0;
#define R2V_ADOPT_A(x) m.x = (decltype(m.x)) st->arena_ptrs[i++];
        R2V_MTP_ARENA(R2V_ADOPT_A)
        void* gr_mem = st->arena_ptrs[i++];
        R2V_MTP_ARENA2(R2V_ADOPT_A)
#undef R2V_ADOPT_A
        if (gr_mem != nullptr) {
            const strata::kernels::GrShapes gsh{m.g_->n_embd, m.g_->hc, m.g_->hc_lr};
            strata::kernels::gr_workspace_init(gsh, gr_mem, m.own_gr_);
        }
        m.window_ = st->window;
        m.cap_ = st->cap;
        m.attn_scratch_floats_ = st->attn_scratch_floats;
        m.vram_ += st->vram;
        m.cs_ = (cudaStream_t) st->cs;
        m.side_ = (cudaStream_t) st->side;
        return m.vram_;
    }
    static uint64_t attn_scratch_floats(MtpDrafter& m, int64_t cap) {
        return (uint64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap, qsa_shapes(*m.g_));
    }
    static bool make_q4_dense(MtpDrafter& m, const uint8_t* blob, uint64_t n, std::string& e) {
        const std::vector<uint8_t> b(blob, blob + n);
        return m.make_q4_dense(b, e);
    }
    static uint64_t state_bytes(MtpDrafter& m, int64_t max_cells, int64_t ring) {
        return qsa_state_bytes(*m.g_, max_cells, false, ring);
    }
    static uint64_t state_init(MtpDrafter& m, int64_t max_cells, void* arena, int64_t ring, int32_t* kv_mode) {
        SessionState& ss = *m.ss_;
        const uint64_t r = qsa_state_init(*m.g_, max_cells, arena, m.st_, &ss.qsa_states[ss.qsa_primary()], ring);
        *kv_mode = m.st_.kv_mode;
        return r;
    }
    static void state_reset(MtpDrafter& m) { m.st_ = QsaState{}; }
    static void state_zero(MtpDrafter& m) { qsa_state_zero(m.st_, *m.g_, nullptr); }
    static void bind_head_logits(MtpDrafter& m, float* logits, int64_t n_vocab) {
        m.n_vocab_ = n_vocab;
        if (logits != nullptr) m.head_logits_ = logits;
    }
    static void bind_dhead(MtpDrafter& m, uint8_t* dhead, int32_t* dvocab, const int32_t* dvocab_host, int64_t n,
                           int32_t type, uint64_t vram_add) {
        m.dhead_ = dhead;
        m.dvocab_ = dvocab;
        m.n_dvocab_ = n;
        m.dvocab_host_.assign(dvocab_host, dvocab_host + n);
        m.dhead_type_ = type;
        m.vram_ += vram_add;
    }
    static bool make_q4_head(MtpDrafter& m, std::string& e) { return m.make_q4_head(e); }
    static bool setup_coupled(MtpDrafter& m, std::string& e) { return m.setup_coupled(e); }
    struct MtpRef {
        bool valid = false;
        uint64_t arena_bytes = 0, state_bytes = 0, vram = 0;
        int64_t off[TK_R2V_MTP_ARENA] = {};
        int64_t window = 0, cap = 0, asf = 0;
        QsaState st{};
        uint64_t st_arena = 0;
        std::vector<std::string> tensors;
        uint64_t dense_bytes = 0, experts_bytes = 0;
        std::unique_ptr<MtpDrafter> keep;   // the reference stays alive until the compare (its weights are compared)
        bool side = false;
    };
    static MtpRef& mtp_ref() { static MtpRef r; return r; }
    static void mtp_offsets(const MtpDrafter& m, int64_t* off) {
        int i = 0;
#define R2V_OFF(x) off[i++] = m.x ? (int64_t) ((const uint8_t*) m.x - (const uint8_t*) m.arena_) : -1;
        R2V_MTP_ARENA(R2V_OFF)
        off[i++] = m.own_gr_.xn ? (int64_t) ((const uint8_t*) m.own_gr_.xn - (const uint8_t*) m.arena_) : -1;
        R2V_MTP_ARENA2(R2V_OFF)
#undef R2V_OFF
    }
    static bool mtp_reference(MtpDrafter& m, const std::string& rt_dir, const ModelGeometry& g, SessionState& ss,
                              int max_t, int64_t window, const MtpDrafter* shared) {
        MtpRef& R = mtp_ref();
        R.valid = false;
        R.keep.reset();
        if (g_ops.mtp_load == nullptr || g_bypass || !check_on() || shared != nullptr) return false;
        auto r = std::make_unique<MtpDrafter>();
        r->q4_ = m.q4_; r->q4_head_ = m.q4_head_; r->force_on_ = m.force_on_; r->dvocab_path_ = m.dvocab_path_;
        r->ple_ss_ = m.ple_ss_; r->hnorm_stream_ = m.hnorm_stream_;
        std::string e;
        g_bypass = true;
        const bool ok = r->load(rt_dir, g, ss, max_t, e, window, nullptr);
        g_bypass = false;
        if (!ok) {
            std::fprintf(stderr, "qwen4exp-r2v check: the C++ reference MtpDrafter::load failed: %s\n", e.c_str());
            return false;
        }
        R.valid = true;
        R.arena_bytes = alloc_size(r->arena_);
        R.state_bytes = alloc_size(r->state_arena_);
        R.vram = r->vram_;
        mtp_offsets(*r, R.off);
        R.window = r->window_; R.cap = r->cap_; R.asf = r->attn_scratch_floats_;
        R.st = r->st_;
        R.st_arena = (uint64_t) (uintptr_t) r->state_arena_;
        R.tensors.clear();
        for (const auto& t : r->tensors_)
            R.tensors.push_back(t.name + " " + t.kind + " " + std::to_string(t.rows) + " " + std::to_string(t.cols) + " " +
                                std::to_string(t.off) + " " + std::to_string(t.bytes));
        R.dense_bytes = alloc_size(r->dense_);
        R.experts_bytes = alloc_size(r->experts_);
        R.side = r->side_ != nullptr;
        R.keep = std::move(r);
        return true;
    }
    static void mtp_compare(const MtpDrafter& m) {
        MtpRef& a = mtp_ref();
        if (!a.valid) return;
        int bad = 0;
        auto cmp = [&](const char* what, int64_t x, int64_t y) {
            if (x != y) {
                ++bad;
                std::fprintf(stderr, "qwen4exp-r2v check: MtpDrafter %s: C++ %lld, Rust %lld\n", what, (long long) x, (long long) y);
            }
        };
        int64_t off[TK_R2V_MTP_ARENA];
        mtp_offsets(m, off);
        static const char* names[] = {
#define R2V_NAME(x) #x,
            R2V_MTP_ARENA(R2V_NAME) "gr_mem", R2V_MTP_ARENA2(R2V_NAME)
#undef R2V_NAME
        };
        for (int i = 0; i < TK_R2V_MTP_ARENA; ++i) cmp(names[i], a.off[i], off[i]);
        cmp("arena bytes", (int64_t) a.arena_bytes, (int64_t) alloc_size(m.arena_));
        cmp("state bytes", (int64_t) a.state_bytes, (int64_t) alloc_size(m.state_arena_));
        cmp("vram", (int64_t) a.vram, (int64_t) m.vram_);
        cmp("window", a.window, m.window_); cmp("cap", a.cap, m.cap_); cmp("attn_scratch_floats", a.asf, m.attn_scratch_floats_);
        cmp("side stream", a.side, m.side_ != nullptr);
        // the K/V state: every word that points into the state arena by its offset, the rest (flags, counts) as is;
        // pointers elsewhere (the pinned host copy, the shared RoPE table) are skipped
        {
            const uint64_t* x = (const uint64_t*) &a.st;
            const uint64_t* y = (const uint64_t*) &m.st_;
            const uint64_t xa = a.st_arena, ya = (uint64_t) (uintptr_t) m.state_arena_;
            int nbad = 0;
            for (size_t w = 0; w < sizeof(QsaState) / 8; ++w) {
                const bool xin = x[w] >= xa && x[w] < xa + a.state_bytes, yin = y[w] >= ya && y[w] < ya + a.state_bytes;
                if (xin || yin) { if (!(xin && yin && x[w] - xa == y[w] - ya)) ++nbad; }
                else if (x[w] < (1ull << 40) && y[w] < (1ull << 40) && x[w] != y[w]) ++nbad;
            }
            cmp("K/V state words", 0, nbad);
        }
        cmp("tensors", (int64_t) a.tensors.size(), (int64_t) m.tensors_.size());
        for (size_t i = 0; i < a.tensors.size() && i < m.tensors_.size(); ++i) {
            const auto& t = m.tensors_[i];
            const std::string s = t.name + " " + t.kind + " " + std::to_string(t.rows) + " " + std::to_string(t.cols) + " " +
                                  std::to_string(t.off) + " " + std::to_string(t.bytes);
            if (s != a.tensors[i]) { ++bad; std::fprintf(stderr, "qwen4exp-r2v check: tensor %zu: %s vs %s\n", i, a.tensors[i].c_str(), s.c_str()); }
        }
        const MtpDrafter& r = *a.keep;
        const uint64_t nd = std::min(a.dense_bytes, alloc_size(m.dense_));
        cmp("dense bytes", (int64_t) a.dense_bytes, (int64_t) alloc_size(m.dense_));
        cmp("dense equal", 1, dev_bytes_equal(r.dense_, m.dense_, nd) ? 1 : 0);
        const uint64_t ne = (uint64_t) m.g_->n_expert * strata::kernels::cpu::BLOB;
        cmp("experts equal", 1, dev_bytes_equal(r.experts_, m.experts_, ne) ? 1 : 0);
        std::fprintf(stderr, "qwen4exp-r2v check: MtpDrafter::load layout %s (%d arena buffers, %.1f MiB; K/V state %.1f MiB)\n",
                     bad ? "DIFFERS" : "identical", TK_R2V_MTP_ARENA, (double) alloc_size(m.arena_) / 1048576.0,
                     (double) alloc_size(m.state_arena_) / 1048576.0);
        a.valid = false;
        a.keep.reset();
    }
    static int bind_facts(MtpDrafter& m, tk_qwen4exp_r2v_bind_facts_t* o) {
        std::memset(o, 0, sizeof(*o));
        if (m.wt_ == nullptr) return -1;
        o->head_loaded = m.head_ != nullptr && m.head_->loaded();
        if (o->head_loaded) {
            o->head_type = m.head_->type();
            o->head_device = m.head_->device();
            o->head_row_bytes = (uint64_t) m.head_->row_bytes();
            o->head_weights = m.head_->weights();
        }
        o->device = m.device_;
        o->q4_head = m.q4_head_;
        o->coupled_env = coupled_draft_env() ? 1 : 0;
        o->have_cparams = m.cparams_ != nullptr;
        o->have_dhead = m.dhead_ != nullptr;
        o->q6_packed = strata::kernels::native_q6_k_packed_enabled() ? 1 : 0;
        o->max_t = m.max_t_;
        const WeightRef* wo = m.wt_->find("output.weight");
        o->n_vocab = wo ? wo->ne1 : -1;
        o->head_logits = m.head_logits_;
        static thread_local std::string vf;
        vf = m.vocab_file();
        o->vocab_file = vf.c_str();
        return 0;
    }
    struct BindRef {
        bool valid = false;
        std::vector<int32_t> dvocab;
        int64_t n_dvocab = 0, n_vocab = 0;
        int dtype = -1;
        uint64_t vram = 0;
        std::unique_ptr<MtpDrafter> keep;
    };
    static BindRef& bind_ref() { static BindRef r; return r; }
    static bool bind_reference(MtpDrafter& m, const WeightTable& wt, const NativeHead* head, const float* window_R,
                               const MtpDrafter* shared) {
        BindRef& R = bind_ref();
        R.valid = false;
        R.keep.reset();
        if (g_ops.mtp_bind == nullptr || g_bypass || !check_on() || shared != nullptr || m.dhead_ != nullptr ||
            m.g_ == nullptr)
            return false;
        auto r = std::make_unique<MtpDrafter>();
        r->q4_ = m.q4_; r->q4_head_ = m.q4_head_; r->force_on_ = m.force_on_; r->dvocab_path_ = m.dvocab_path_;
        r->ple_ss_ = m.ple_ss_; r->hnorm_stream_ = m.hnorm_stream_;
        std::string e;
        g_bypass = true;
        bool ok = r->load(m.rt_dir_, *m.g_, *m.ss_, m.max_t_, e, m.window_ > 0 ? m.window_ : 1ll << 40, nullptr) &&
                  r->bind(wt, head, window_R, e, nullptr);
        g_bypass = false;
        if (!ok) {
            std::fprintf(stderr, "qwen4exp-r2v check: the C++ reference MtpDrafter::bind failed: %s\n", e.c_str());
            return false;
        }
        R.valid = true;
        R.dvocab = r->dvocab_host_;
        R.n_dvocab = r->n_dvocab_;
        R.n_vocab = r->n_vocab_;
        R.dtype = r->dhead_type_;
        R.keep = std::move(r);
        return true;
    }
    static void bind_compare(const MtpDrafter& m) {
        BindRef& a = bind_ref();
        if (!a.valid) return;
        int bad = 0;
        auto cmp = [&](const char* what, int64_t x, int64_t y) {
            if (x != y) {
                ++bad;
                std::fprintf(stderr, "qwen4exp-r2v check: MtpDrafter::bind %s: C++ %lld, Rust %lld\n", what, (long long) x, (long long) y);
            }
        };
        const MtpDrafter& r = *a.keep;
        cmp("n_dvocab", a.n_dvocab, m.n_dvocab_);
        cmp("n_vocab", a.n_vocab, m.n_vocab_);
        cmp("draft head type", a.dtype, m.dhead_type_);
        cmp("vocab ids", 1, a.dvocab == m.dvocab_host_ ? 1 : 0);
        const uint64_t hb = m.head_ ? (uint64_t) m.n_dvocab_ * m.head_->row_bytes() : 0;
        cmp("draft head bytes equal", 1, dev_bytes_equal(r.dhead_, m.dhead_, hb) ? 1 : 0);
        cmp("vocab ids on device equal", 1, dev_bytes_equal(r.dvocab_, m.dvocab_, (uint64_t) m.n_dvocab_ * 4) ? 1 : 0);
        cmp("head logits", r.head_logits_ != nullptr, m.head_logits_ != nullptr);
        cmp("coupled", r.cparams_ != nullptr, m.cparams_ != nullptr);
        std::fprintf(stderr, "qwen4exp-r2v check: MtpDrafter::bind %s (draft head over %lld tokens)\n",
                     bad ? "DIFFERS" : "identical", (long long) m.n_dvocab_);
        a.valid = false;
        a.keep.reset();
    }
};

// ------------------------------------------------------------------------- the hooks
bool qwen4exp_r2v_ver_reference(Verifier& v, const WeightTable& wt, const ModelGeometry& g, SessionState& ss,
                              const VerifyHits& hits, const NativeHead* head, int max_t) {
    return Qwen4expR2vAccess::ver_reference(v, wt, g, ss, hits, head, max_t);
}
int qwen4exp_r2v_ver_init(Verifier& v, bool check, std::string& err) {
    if (g_ops.ver_init == nullptr || g_bypass) return 0;
    char eb[512] = {0};
    const int rc = g_ops.ver_init(g_ops.user, &v, eb, sizeof eb);
    if (rc < 0) err = eb;
    if (rc > 0 && check) Qwen4expR2vAccess::ver_compare(v);
    return rc;
}
int qwen4exp_r2v_ver_bias(Verifier& v, const std::vector<float>& bias, std::string& err) {
    if (g_ops.ver_set_logit_bias == nullptr || g_bypass) return 0;
    char eb[512] = {0};
    const int rc = g_ops.ver_set_logit_bias(g_ops.user, &v, bias.data(), (int64_t) bias.size(), eb, sizeof eb);
    if (rc < 0) err = eb;
    return rc;
}
bool qwen4exp_r2v_mtp_reference(MtpDrafter& m, const std::string& rt_dir, const ModelGeometry& g, SessionState& ss,
                              int max_t, int64_t window, const MtpDrafter* shared) {
    return Qwen4expR2vAccess::mtp_reference(m, rt_dir, g, ss, max_t, window, shared);
}
int qwen4exp_r2v_mtp_load(MtpDrafter& m, int64_t window, bool check, std::string& err) {
    if (g_ops.mtp_load == nullptr || g_bypass) return 0;
    char eb[512] = {0};
    const int rc = g_ops.mtp_load(g_ops.user, &m, window, eb, sizeof eb);
    if (rc < 0) err = eb;
    if (rc > 0 && check) Qwen4expR2vAccess::mtp_compare(m);
    if (rc == 0 && check) { Qwen4expR2vAccess::mtp_ref().valid = false; Qwen4expR2vAccess::mtp_ref().keep.reset(); }
    return rc;
}
bool qwen4exp_r2v_bind_reference(MtpDrafter& m, const WeightTable& wt, const NativeHead* head, const float* window_R,
                               const MtpDrafter* shared) {
    return Qwen4expR2vAccess::bind_reference(m, wt, head, window_R, shared);
}
int qwen4exp_r2v_mtp_bind(MtpDrafter& m, bool check, std::string& err) {
    if (g_ops.mtp_bind == nullptr || g_bypass) return 0;
    char eb[512] = {0};
    const int rc = g_ops.mtp_bind(g_ops.user, &m, eb, sizeof eb);
    if (rc < 0) err = eb;
    if (rc > 0 && check) Qwen4expR2vAccess::bind_compare(m);
    if (rc == 0 && check) { Qwen4expR2vAccess::bind_ref().valid = false; Qwen4expR2vAccess::bind_ref().keep.reset(); }
    return rc;
}

}  // namespace strata::core

using strata::core::Qwen4expR2vAccess;
using strata::core::MtpDrafter;
using strata::core::Verifier;

extern "C" {

void tk_qwen4exp_r2v_install(const tk_qwen4exp_r2v_ops_t* ops) { g_ops = ops ? *ops : tk_qwen4exp_r2v_ops_t{}; }

int tk_qwen4exp_r2v_ver_facts(void* ver, tk_qwen4exp_r2v_ver_facts_t* out) {
    return (ver && out) ? Qwen4expR2vAccess::ver_facts(*(Verifier*) ver, out) : -1;
}
void tk_qwen4exp_r2v_ver_adopt(void* ver, const tk_qwen4exp_r2v_ver_state_t* st) {
    if (ver && st) Qwen4expR2vAccess::ver_adopt(*(Verifier*) ver, st);
}
int tk_qwen4exp_r2v_ver_bias_view(void* ver, const float** host, int64_t* n_host, float** device, int64_t* n_vocab,
                                  int32_t* eligible) {
    return ver ? Qwen4expR2vAccess::bias_view(*(Verifier*) ver, host, n_host, device, n_vocab, eligible) : -1;
}
void tk_qwen4exp_r2v_ver_bias_adopt(void* ver, const float* bias, int64_t n, float* device) {
    if (ver) Qwen4expR2vAccess::bias_adopt(*(Verifier*) ver, bias, n, device);
}
int tk_qwen4exp_r2v_stream_high_prio(void** out) {
    *out = nullptr;
#if defined(STRATA_USE_HIP)
    int lo = 0, hi = 0;
    hipStream_t s = nullptr;
    if (hipDeviceGetStreamPriorityRange(&lo, &hi) == hipSuccess &&
        hipStreamCreateWithPriority(&s, hipStreamNonBlocking, hi) == hipSuccess) {
        *out = (void*) s;
        (void) cudaGetLastError();
        return 1;
    }
    (void) cudaGetLastError();
#endif
    return 0;
}
int tk_qwen4exp_r2v_arch_is(const char* prefix) {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 0;
#if defined(STRATA_USE_HIP)
    if (std::strcmp(prefix, "gfx12") == 0) return strata::kernels::gfx_arch_is_gfx12(prop.gcnArchName) ? 1 : 0;
    return std::strncmp(prop.gcnArchName, prefix, std::strlen(prefix)) == 0 ? 1 : 0;
#else
    (void) prefix;
    return 0;
#endif
}

int tk_qwen4exp_r2v_mtp_facts(void* mtp, tk_qwen4exp_r2v_mtp_facts_t* out) {
    return (mtp && out) ? Qwen4expR2vAccess::mtp_facts(*(MtpDrafter*) mtp, out) : -1;
}
uint64_t tk_qwen4exp_r2v_attn_scratch_floats(void* mtp, int64_t cap) {
    return mtp ? Qwen4expR2vAccess::attn_scratch_floats(*(MtpDrafter*) mtp, cap) : 0;
}
int tk_qwen4exp_r2v_mtp_make_q4_dense(void* mtp, const uint8_t* blob, uint64_t n, char* err, size_t err_cap) {
    std::string e;
    if (Qwen4expR2vAccess::make_q4_dense(*(MtpDrafter*) mtp, blob, n, e)) return 1;
    put_err(e, err, err_cap);
    return 0;
}
uint64_t tk_qwen4exp_r2v_mtp_state_bytes(void* mtp, int64_t max_cells, int64_t ring) {
    return Qwen4expR2vAccess::state_bytes(*(MtpDrafter*) mtp, max_cells, ring);
}
uint64_t tk_qwen4exp_r2v_mtp_state_init(void* mtp, int64_t max_cells, void* arena, int64_t ring, int32_t* kv_mode) {
    return Qwen4expR2vAccess::state_init(*(MtpDrafter*) mtp, max_cells, arena, ring, kv_mode);
}
void tk_qwen4exp_r2v_mtp_state_reset(void* mtp) { Qwen4expR2vAccess::state_reset(*(MtpDrafter*) mtp); }
void tk_qwen4exp_r2v_mtp_state_zero(void* mtp) { Qwen4expR2vAccess::state_zero(*(MtpDrafter*) mtp); }
void tk_qwen4exp_r2v_bind_head_logits(void* mtp, float* logits, int64_t n_vocab) {
    Qwen4expR2vAccess::bind_head_logits(*(MtpDrafter*) mtp, logits, n_vocab);
}
void tk_qwen4exp_r2v_bind_dhead(void* mtp, uint8_t* dhead, int32_t* dvocab, const int32_t* dvocab_host, int64_t n,
                                int32_t type, uint64_t vram_add) {
    Qwen4expR2vAccess::bind_dhead(*(MtpDrafter*) mtp, dhead, dvocab, dvocab_host, n, type, vram_add);
}
int tk_qwen4exp_r2v_mtp_make_q4_head(void* mtp, char* err, size_t err_cap) {
    std::string e;
    if (Qwen4expR2vAccess::make_q4_head(*(MtpDrafter*) mtp, e)) return 1;
    put_err(e, err, err_cap);
    return 0;
}
int tk_qwen4exp_r2v_mtp_setup_coupled(void* mtp, char* err, size_t err_cap) {
    std::string e;
    if (Qwen4expR2vAccess::setup_coupled(*(MtpDrafter*) mtp, e)) return 1;
    put_err(e, err, err_cap);
    return 0;
}
void tk_qwen4exp_r2v_mtp_set_weights(void* mtp, const tk_qwen4exp_r2v_tensor_t* t, int64_t n, void* dense, void* experts,
                                     uint64_t vram_add) {
    Qwen4expR2vAccess::mtp_set_weights(*(MtpDrafter*) mtp, t, n, dense, experts, vram_add);
}
uint64_t tk_qwen4exp_r2v_mtp_adopt(void* mtp, const tk_qwen4exp_r2v_mtp_state_t* st) {
    return (mtp && st) ? Qwen4expR2vAccess::mtp_adopt(*(MtpDrafter*) mtp, st) : 0;
}
int tk_qwen4exp_r2v_bind_facts(void* mtp, tk_qwen4exp_r2v_bind_facts_t* out) {
    return (mtp && out) ? Qwen4expR2vAccess::bind_facts(*(MtpDrafter*) mtp, out) : -1;
}
void tk_qwen4exp_r2v_gather_rows(const void* src, int64_t row_bytes, const int32_t* ids, int64_t n, void* dst) {
    strata::kernels::gather_rows((const uint8_t*) src, row_bytes, ids, n, (uint8_t*) dst, nullptr);
}
int tk_qwen4exp_r2v_q6_pack(void* w, int n_in, int n_out, const char* what) {
    return strata::kernels::native_q6_k_pack(w, n_in, n_out, what) ? 1 : 0;
}
void tk_qwen4exp_r2v_kv_flags(int32_t* hybrid, int32_t* int8) {
    *hybrid = strata::core::qsa_kv_hybrid() ? 1 : 0;
    *int8 = strata::core::qsa_kv_int8() ? 1 : 0;
}
void tk_qwen4exp_r2v_kv_set_flags(int32_t hybrid, int32_t int8) {
    strata::core::qsa_set_kv_hybrid(hybrid != 0);
    strata::core::qsa_set_kv_int8(int8 != 0);
}

}  // extern "C"
