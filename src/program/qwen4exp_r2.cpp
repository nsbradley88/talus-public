// src/program/qwen4exp_r2.cpp - qwen4exp-r2: residual launch helpers of Pieces 2/3 (see qwen4exp_r2.hpp) and the prompt
// path's setup wrappers (the `tk_qwen4exp_r2_*` block of include/talus_qwen4exp_kernels.h).
//
// The helpers below are the native-pack branches of src/core/layer.cpp's moe_route / moe_combine_parts / lm_head_mix /
// qsa_attn_pools / qsa_kv_resolve / embed_row, kernel for kernel, so the Rust plane's wrappers no longer reach
// layer.cpp on a native pack.  The layer.cpp originals stay for the C++ path until Group B is deleted.
#include "qwen4exp_r2.hpp"

#include "strata/core/layout.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <atomic>
#include <exception>
#include <string>

namespace strata::core::qwen4exp_r2 {
namespace {
std::atomic<bool> g_native_bf16{false};
constexpr float RMS_EPS = 1e-6f;   // layer.cpp's RMS_EPS (qwen4exp.attention.layer_norm_rms_epsilon)
}  // namespace

void set_native_bf16(bool enabled) { g_native_bf16.store(enabled); }
bool native_bf16() { return g_native_bf16.load(); }

bool moe_route(const WeightTable& tables, const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b,
               const float* x, void* stream, std::string& err) {
    // the S-form branch (BF16 activations, the split GEMV): a non-native pack only - layer.cpp's
    if (!native_bf16()) return strata::core::moe_route(tables, g, layer, k, b, x, stream, err, nullptr);
    using namespace strata::kernels;
    const LayerView v(tables, layer);
    const WeightRef* w_router = v.get("ffn_gate_inp.weight");
    if (w_router == nullptr) { err = v.name("ffn_gate_inp.weight") + " is missing"; return false; }
    if (k < 1 || k > 64) { err = "moe_route: k must be 1..64"; return false; }
    // project_bf16 with native_bf16_projections: the FP32-activation BF16 GEMV
    bf16_gemv_fp32_mmvf(x, (const uint16_t*) w_router->data, b.logits, g.n_embd, g.n_expert, stream);
    if (native_router_enabled() && g.n_expert == 512 && k == 10) {
        try { native_router_top10(b.logits, b.ids, b.weights, stream); }
        catch (const std::exception& error) { err = v.name("router") + ": " + error.what(); return false; }
    } else {
        router_top10(b.logits, 1, (int) g.n_expert, (int) k, b.ids, b.weights, stream);
    }
    return true;
}

bool moe_combine_parts(const ModelGeometry& g, int64_t layer, int64_t k, const MoEBuffers& b, const float* parts,
                       float* out, void* stream, std::string& err) {
    using namespace strata::kernels;
    if (k < 1 || k > 64) { err = "moe_finish: k must be 1..64"; return false; }
    try {
        if (native_moe_combine_enabled()) native_moe_combine(parts, b.weights, b.shared, out, g.n_embd, k, stream);
        else moe_combine(parts, b.weights, b.shared, out, g.n_embd, k, stream);
    } catch (const std::exception& error) {
        err = "blk." + std::to_string(layer) + ".moe_combine" + ": " + error.what();
        return false;
    }
    return true;
}

bool lm_head_mix(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, void* stream,
                 std::string& err) {
    const WeightRef* wn = tables.find("output_hc_norm.weight");
    const WeightRef* wd = tables.find("output_hc_down.weight");
    const WeightRef* wu = tables.find("output_hc_up.weight");
    if (!wn || !wd || !wu) {
        err = "lm_head: an output_hc_* weight is missing";
        return false;
    }
    if (wn->kind != WeightKind::F32 || wd->kind != WeightKind::Bf16InF32 || wu->kind != WeightKind::Bf16InF32) {
        err = "lm_head: the output_hc_* weights have the wrong engine forms";
        return false;
    }
    const strata::kernels::GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    strata::kernels::gr_read(bb.R, (const float*) wn->data, (const uint16_t*) wd->data, (const uint16_t*) wu->data,
                             nullptr, RMS_EPS, gs, bb.gr, bb.mixed, bb.inject, stream);
    return true;
}

bool lm_head(const WeightTable& tables, const ModelGeometry& g, const BlockBuffers& bb, float* logits, void* stream,
             std::string& err) {
    // the S-form head (quantize_q8_K + the S-form GEMV): never on a native pack (NativeHead) - layer.cpp's
    return strata::core::lm_head(tables, g, bb, logits, stream, err);
}

strata::kernels::QsaAttnPools qsa_attn_pools(const QsaState& st) {
    strata::kernels::QsaAttnPools pools;
    pools.page_table = st.page_table;
    if (st.kv_hybrid) { pools.k_q = st.k_q; pools.k_scale = st.k_scale; pools.v_q4 = st.v_q4; }
    else if (st.kv_q4) { pools.k_q4 = st.k_q4; pools.v_q4 = st.v_q4; }
    else if (st.kv_int8) { pools.k_q = st.k_q; pools.v_q = st.v_q; pools.k_scale = st.k_scale; pools.v_scale = st.v_scale; }
    else { pools.k_pool = st.k_pool; pools.v_pool = st.v_pool; }
    return pools;
}

void qsa_kv_resolve(const QsaState& st, const ModelGeometry& g, const int32_t* ids, const int32_t* steps, int64_t n_q,
                    int64_t cap, void* stream) {
    if (st.kv_mode != 1) return;
    strata::kernels::kv_stream_resolve(st.map, qwen4exp_r2::qsa_attn_pools(st), st.host, qsa_kv_format(st), ids, steps, n_q, cap,
                                       qsa_shapes(g), stream);
}

bool embed_row(const WeightTable& tables, const ModelGeometry& g, int64_t token, float* out_dev, void* stream,
               std::string& err) {
    if (const NativeEmbed* ne = native_embed()) {   // the GGUF-form table (every native pack)
        if (token < 0 || out_dev == nullptr) { err = "embed_row: invalid token or output"; return false; }
        ne->gather_one(token, out_dev, stream);
        return true;
    }
    return strata::core::embed_row(tables, g, token, out_dev, stream, err);   // the pack's S-form table
}

}  // namespace strata::core::qwen4exp_r2

// ================================================================ the prompt path's setup: facts and runtime calls
// (the logic is rust/crates/talus-model/src/qwen4exp/prefill/setup.rs; these only read what the kernels library knows
// and make the runtime calls)
#include "talus_qwen4exp_kernels.h"
#include "strata/core/peer_experts.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

extern "C" {

void tk_qwen4exp_r2_pf_facts(tk_qwen4exp_r2_pf_facts_t* o) {
    *o = tk_qwen4exp_r2_pf_facts_t{};
    const strata::kernels::cpu::ExpertLayout& lay = strata::kernels::cpu::expert_layout();
#if defined(STRATA_USE_HIP)
    o->is_hip = 1;
#endif
    o->layout_native = lay.native ? 1 : 0;
    o->fused_enabled = strata::prefill::fused::enabled() ? 1 : 0;
    o->peer_portable = strata::core::peer_portable() ? 1 : 0;
    o->mmq_built = strata::prefill::mmq::built() ? 1 : 0;
    o->prompt_f16 = strata::prefill::prompt_f16() ? 1 : 0;   // the current device's (a stage's own)
    o->lay_n_layers = lay.n_layers;
    o->n_fmt = (int64_t) lay.fmt.size();
    o->max_blob = (int64_t) lay.max_blob;
    o->ng_hc_dim = strata::kernels::NG_HC_DIM;
    o->step_count = strata::kernels::kStepCount;
    o->ring_max = 1024;                 // prefill.cpp RING_MAX (the event arrays)
    o->gemm_scratch_elems = 32ll << 20; // prefill.cpp GEMM_SCRATCH
    o->gemm_ws_bytes = 32ll << 20;      // prefill.cpp GEMM_WS
}

void tk_qwen4exp_r2_pf_session(const void* g_, const void* ss_, tk_qwen4exp_r2_pf_session_t* o) {
    *o = tk_qwen4exp_r2_pf_session_t{};
    const auto& g = *(const strata::core::ModelGeometry*) g_;
    const auto& ss = *(const strata::core::SessionState*) ss_;
    o->n_layers = g.n_layers; o->n_expert = g.n_expert; o->n_embd = g.n_embd; o->hc = g.hc; o->hc_lr = g.hc_lr;
    o->n_ff = g.n_ff; o->n_ff_shexp = g.n_ff_shexp; o->conv_ch = g.ssm_conv_channels; o->value_dim = g.ssm_value_dim;
    o->v_heads = g.ssm_v_heads; o->n_head = g.n_head; o->n_head_kv = g.n_head_kv; o->head_dim = g.head_dim;
    o->head_dim_v = g.head_dim_v; o->idx_heads = g.idx_q_heads; o->idx_dim = g.idx_key_dim; o->k = ss.k;
    o->rms_eps = g.rms_eps;
    const strata::kernels::QsaShapes s = strata::core::qsa_shapes(g);
    o->cap = strata::kernels::qsa_selection_width(strata::kernels::kTopkMaxCells, s);
    o->idx_block = s.idx_block;
    o->page_size = s.page_size;
    o->attn_scratch_floats = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(o->cap, s);
    o->kv_q4_bytes_per_head = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) s.head_dim);
    if (ss.qsa_alloc > 0) {
        const strata::core::QsaState& q0 = ss.qsa_states[ss.qsa_primary()];
        o->max_cells = q0.max_cells;
        o->n_pages = q0.n_pages;
        o->kv_mode = q0.kv_mode;
        o->kv_hybrid = q0.kv_hybrid ? 1 : 0;
        o->kv_q4 = q0.kv_q4 ? 1 : 0;
        o->kv_int8 = q0.kv_int8 ? 1 : 0;
    }
}
uint64_t tk_qwen4exp_r2_fused_group_bytes(int64_t n, int32_t n_expert) {
    return (uint64_t) strata::prefill::fused::group_bytes(n, (int) n_expert);
}
uint64_t tk_qwen4exp_r2_fused_act_bytes(int64_t rows, int64_t cols) {
    return (uint64_t) strata::prefill::fused::act_bytes(rows, cols);
}

int tk_qwen4exp_r2_malloc(void** out, uint64_t bytes) {
    *out = nullptr;
    if (cudaMalloc(out, (size_t) bytes) != cudaSuccess) { *out = nullptr; return 0; }
    return 1;
}
void tk_qwen4exp_r2_free(void* p) { if (p) cudaFree(p); }
int tk_qwen4exp_r2_memset_sync(void* dst, int32_t value, uint64_t bytes) {
    return cudaMemset(dst, value, (size_t) bytes) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_r2_memcpy_h2d_sync(void* dst, const void* src, uint64_t bytes) {
    return cudaMemcpy(dst, src, (size_t) bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 1 : 0;
}
int tk_qwen4exp_r2_host_alloc_mapped(void** host, void** dev, uint64_t bytes, int32_t portable) {
    *host = *dev = nullptr;
    void* h = nullptr;
    void* d = nullptr;
    if (cudaHostAlloc(&h, (size_t) bytes, cudaHostAllocMapped | (portable ? cudaHostAllocPortable : 0)) == cudaSuccess &&
        cudaHostGetDevicePointer(&d, h, 0) == cudaSuccess) {
        *host = h;
        *dev = d;
        return 1;
    }
    if (h) cudaFreeHost(h);
    cudaGetLastError();
    return 0;
}
void tk_qwen4exp_r2_free_host(void* p) { if (p) cudaFreeHost(p); }
int tk_qwen4exp_r2_stream_create(void** out) {
    cudaStream_t s = nullptr;
    const bool ok = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess;
    *out = ok ? (void*) s : nullptr;
    return ok ? 1 : 0;
}
int tk_qwen4exp_r2_get_device(int32_t* out) {
    int d = -1;
    const bool ok = cudaGetDevice(&d) == cudaSuccess;
    *out = d;
    return ok ? 1 : 0;
}
void tk_qwen4exp_r2_mem_info(uint64_t* free_b, uint64_t* total_b) {
    size_t fb = 0, tb = 0;
    cudaMemGetInfo(&fb, &tb);
    (void) cudaGetLastError();
    *free_b = fb;
    *total_b = tb;
}
void tk_qwen4exp_r2_clear_error(void) { (void) cudaGetLastError(); }

}  // extern "C"
