// include/strata/core/model_desc.hpp - the model's DESCRIPTION, derived from its GGUF, and the arch registry.
//
// Until this header the engine described its one model with a struct of hard-coded defaults (`ModelGeometry`
// in layout.hpp) plus a handful of GGUF overrides, a fixed-value guard (`check_architecture`: 48 blocks, n_embd
// 2560, 24/2 heads) and `layer % 4 == 3` for the layer type.  Now:
//
//   * `ModelGeometry` (the name is kept: some 150 call sites take one) holds no defaults.  It is filled at load
//     by the arch's `describe` from GGUF metadata, and carries an explicit per-layer `layer_kind` list instead
//     of interval arithmetic.
//   * The registry (`arch_registry`) maps `general.architecture` to an `ArchSpec`: how to describe the model
//     from its metadata, which tensors each layer type needs (and their shapes), and the CAPABILITY check - the
//     shapes this build's kernels can run, with the reason for each.  `qwen4exp` is the only registered arch.
//   * An unknown arch is refused with the list of supported ones; a known arch with a shape its kernels do not
//     support is refused with "the kernels require X"; a missing or mis-shaped tensor is refused at load with
//     its name (`check_gguf_tensors`, and `check_all` over a pack's index in layout.hpp).
//
// Header-only (std + the header-only GGUF reader), so CPU tools - strata-gguf --describe, the planner, tests -
// can derive a description without a GPU.
#pragma once

#include "strata/arch/laguna_caps.hpp"
#include "strata/arch/qwen4exp_caps.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

/// What a layer's token mixer is.  New kinds (dense GQA, sliding-window GQA, ...) are added here together with
/// the code that runs them; nothing else may infer a layer's kind.
enum class LayerKind : uint8_t {
    Gdn = 0,      ///< Gated DeltaNet linear attention (recurrent state, no KV)
    Qsa = 1,      ///< qwen4exp full attention with indexer selection (KV + indexer keys)
    Attn = 2,     ///< plain causal GQA attention over the whole context (KV)
    AttnSwa = 3,  ///< causal GQA attention over the last `sliding_window` positions (KV ring)
};

inline constexpr LayerKind kAllLayerKinds[] = {LayerKind::Gdn, LayerKind::Qsa, LayerKind::Attn, LayerKind::AttnSwa};

inline const char* layer_kind_name(LayerKind k) {
    switch (k) {
    case LayerKind::Gdn: return "gdn";
    case LayerKind::Qsa: return "qsa";
    case LayerKind::Attn: return "attn";
    case LayerKind::AttnSwa: return "attn_swa";
    }
    return "?";
}

/// The model's description.  Every field is filled by the arch's `describe` (from the GGUF, with the arch's
/// documented default for a key the file does not carry) or by `model_reference(arch)`; a default-constructed
/// one is EMPTY, not "the qwen4exp model".
struct ModelGeometry {
    std::string arch;                       ///< `general.architecture`
    int64_t n_embd = 0;
    int64_t n_layers = 0;
    std::vector<LayerKind> layer_kind;      ///< one per layer: THE layer-type predicate
    /// `full_attention_interval` as the file declares it.  NOT the layer-type predicate (that is `layer_kind`);
    /// kept because a conversation snapshot's geometry key carries it.  0 when the arch has none.
    int64_t qsa_interval = 0;

    // GDN layers
    int64_t ssm_state_size = 0;      ///< S, `ssm.state_size`
    int64_t ssm_k_heads = 0;         ///< `ssm.group_count`
    int64_t ssm_v_heads = 0;         ///< `ssm.time_step_rank`
    int64_t ssm_d_conv = 0;          ///< `ssm.conv_kernel`
    int64_t ssm_conv_channels = 0;   ///< 2*S*k_heads + value_dim (derived)
    int64_t ssm_value_dim = 0;       ///< `ssm.inner_size` = S * v_heads

    // attention layers (QSA)
    int64_t n_head = 0;
    int64_t n_head_kv = 0;
    int64_t head_dim = 0;            ///< `attention.key_length`
    int64_t head_dim_v = 0;          ///< `attention.value_length`
    int64_t n_rot = 0;               ///< `rope.dimension_count`: rotated dims per head
    int64_t idx_q_heads = 0;         ///< `attention.indexer.head_count` (QUERY heads; the cached key is one head)
    int64_t idx_key_dim = 0;         ///< `attention.indexer.key_length`
    int64_t idx_top_k = 0;           ///< `attention.indexer.top_k`
    double rms_eps = 0;              ///< `attention.layer_norm_rms_epsilon`
    double rope_freq_base = 0;       ///< `rope.freq_base`
    int64_t rope_sections[4] = {0, 0, 0, 0};   ///< `rope.dimension_sections` (M-RoPE)
    /// The file's own rope SCALING keys, as declared (empty / 0 = absent); the CLI resolves over them.
    std::string rope_scaling_type;
    double rope_scaling_factor = 0;
    double rope_scaling_orig_ctx = 0;

    // hyper-connections, every layer
    int64_t hc = 0;                  ///< `hyper_connection.count`
    int64_t hc_lr = 0;               ///< `hyper_connection.low_rank`

    // MoE, every layer
    int64_t n_expert = 0;
    int64_t n_expert_used = 0;       ///< top-k
    int64_t n_ff = 0;                ///< `expert_feed_forward_length`
    int64_t n_ff_shexp = 0;          ///< `expert_shared_feed_forward_length`

    // vocabulary, context, per-layer embedding
    int64_t n_vocab = 0;             ///< `tokenizer.ggml.tokens` count
    int64_t n_ctx_train = 0;         ///< `context_length`
    std::vector<int64_t> ple_layers; ///< `ple.layers`
    int64_t ple_conv_kernel = 0;     ///< `ple.conv_kernel`
    int64_t image_token_id = -1;     ///< `ple.image_token_id`

    // per-layer attention shapes and other-arch fields: 0 / empty when the arch has none (qwen4exp leaves all unset)
    std::vector<int64_t> n_head_per_layer;   ///< `attention.head_count` as an array; empty = `n_head` on every layer
    int64_t sliding_window = 0;              ///< `attention.sliding_window` (AttnSwa layers)
    double rope_freq_base_swa = 0;           ///< `rope.freq_base_swa`
    int64_t n_rot_swa = 0;                   ///< `rope.dimension_count_swa`
    int64_t n_ff_dense = 0;                  ///< `feed_forward_length`: the leading dense layers' FFN width
    int64_t n_dense_lead = 0;                ///< `leading_dense_block_count`: layers 0..n-1 have a dense FFN, not MoE
    int64_t expert_gating = 0;               ///< `expert_gating_func` (1 softmax, 2 sigmoid); 0 = the arch's own
    bool expert_weights_norm = false;        ///< `expert_weights_norm`: top-k weights renormalised to sum 1
    double expert_weights_scale = 0;         ///< `expert_weights_scale`: routed output scale after normalisation
    bool attn_gate_per_head = false;         ///< attention output gated by softplus(W_g x), one scalar per head

    // Strata 0.1.44's model families (geometry.hpp's fields, merged into the description)
    strata::ModelFamily family = strata::ModelFamily::Unknown;   ///< from `general.architecture`
    int64_t n_mtp_layers = 0;                ///< `nextn_predict_layers`: MTP blocks after the trunk (not in n_layers)
    int64_t idx_block = 0;                   ///< the non-zero `attention.compress_ratios`: indexer pooling block
    bool ffn_dense = false;                  ///< qwen35: a dense FFN (ffn_gate/up/down, width n_ff_dense) per layer

    int64_t heads(int64_t layer) const {
        return n_head_per_layer.empty() ? n_head : n_head_per_layer[(size_t) layer];
    }
    int64_t hc_dim() const { return hc * n_embd; }
    bool has_indexer() const { return idx_q_heads > 0; }               ///< false: dense attention only
    bool has_hyper_connections() const { return hc > 0; }             ///< false: plain pre-norm residual
    /// A full-attention layer of the GDN hybrids: QSA (qwen4exp) or plain gated attention (qwen35, qwen35moe).
    bool is_full_attention(int64_t layer) const {
        return layer >= 0 && layer < (int64_t) layer_kind.size() &&
               (layer_kind[(size_t) layer] == LayerKind::Qsa || layer_kind[(size_t) layer] == LayerKind::Attn);
    }
    LayerKind kind(int64_t layer) const { return layer_kind[(size_t) layer]; }
    bool is_qsa(int64_t layer) const {
        return layer >= 0 && layer < (int64_t) layer_kind.size() && layer_kind[(size_t) layer] == LayerKind::Qsa;
    }
    int64_t n_qsa_layers() const { return (int64_t) std::count(layer_kind.begin(), layer_kind.end(), LayerKind::Qsa); }
    /// Full-attention layers of any kind (QSA or plain gated attention): Strata's `n_qsa_layers` for the families.
    int64_t n_full_attention_layers() const {
        return (int64_t) std::count_if(layer_kind.begin(), layer_kind.end(),
                                       [](LayerKind k) { return k == LayerKind::Qsa || k == LayerKind::Attn; });
    }
    int64_t n_gdn_layers() const { return (int64_t) std::count(layer_kind.begin(), layer_kind.end(), LayerKind::Gdn); }
    /// How many QSA layers have an index below `bound` - a QSA layer's ordinal is `qsa_layers_below(layer)`.
    int64_t qsa_layers_below(int64_t bound) const {
        const int64_t b = std::clamp<int64_t>(bound, 0, (int64_t) layer_kind.size());
        return (int64_t) std::count(layer_kind.begin(), layer_kind.begin() + b, LayerKind::Qsa);
    }
    int64_t gdn_layers_below(int64_t bound) const {
        return std::clamp<int64_t>(bound, 0, (int64_t) layer_kind.size()) - qsa_layers_below(bound);
    }
};

/// True for the full-attention (QSA) layers - read from the description's per-layer list.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) { return g.is_qsa(layer); }

// ---- per-layer tensor requirements --------------------------------------------------------------------------

/// The engine form a pack must hold a 1-D tensor in (layout.cpp maps it onto `WeightKind`); `Any` = unchecked.
enum class PackForm : uint8_t { Any, F32, Bf16 };

/// One tensor a layer needs.  `ne` is GGUF order (ne[0] the contiguous axis); `ne[1] == 0` means 1-D (checked
/// by element count).  `gguf_only`: not in a pack's index (the experts, read from the GGUF in place).
/// `optional`: shape-checked when present, not required (tensors a variant may ship in another file).
struct TensorReq {
    std::string suffix;
    int64_t ne[3] = {0, 0, 0};
    PackForm form = PackForm::Any;
    bool gguf_only = false;
    bool optional = false;
};

/// The answer a tensor lookup gives the checker: found, and its shape (`ne[1] == 0` for 1-D).
struct TensorShape {
    bool found = false;
    int64_t ne[3] = {0, 0, 0};
    int64_t elements = 0;
};

using LayerTensorFn = std::vector<TensorReq> (*)(const ModelGeometry&, int64_t layer);
using DescribeFn = bool (*)(const GgufFile& meta, ModelGeometry& g, std::string& err);
using ReferenceFn = ModelGeometry (*)();
using CapabilityFn = std::string (*)(const ModelGeometry& g);

/// One registered architecture.
struct ArchSpec {
    const char* name;            ///< `general.architecture`
    const char* about;           ///< one line for listings
    DescribeFn describe;         ///< GGUF metadata -> description
    ReferenceFn reference;       ///< the description of the arch's reference model, for packs that carry no GGUF
    LayerTensorFn layer_tensors; ///< the tensors one layer needs, given its kind
    std::vector<TensorReq> (*global_tensors)(const ModelGeometry&);
    CapabilityFn capability;     ///< "" when this build's kernels run the description, else every reason they do not
};

// ---- qwen4exp ------------------------------------------------------------------------------------------------

namespace qwen4exp_detail {
namespace caps = strata::arch::qwen4exp;

inline bool meta_i64(const GgufFile& m, const std::string& key, int64_t& out) {
    const MetaValue* v = m.get(key);
    if (v == nullptr || !v->is_num()) return false;
    out = (v->type == MetaType::F32 || v->type == MetaType::F64) ? (int64_t) v->f : (int64_t) v->u;
    return true;
}
inline bool meta_f64(const GgufFile& m, const std::string& key, double& out) {
    const MetaValue* v = m.get(key);
    if (v == nullptr || !v->is_num()) return false;
    out = v->num();
    return true;
}
/// A numeric array, whole: false when absent, not numeric, or longer than the reader keeps.
inline bool meta_i64s(const GgufFile& m, const std::string& key, std::vector<int64_t>& out) {
    const MetaValue* v = m.get(key);
    if (v == nullptr || v->type != MetaType::ARRAY || v->items.size() != v->count) return false;
    out.clear();
    for (const MetaValue& e : v->items) {
        if (!e.is_num()) return false;
        out.push_back((int64_t) e.u);
    }
    return true;
}

/// The reference model: Qwen3.8-Flash-Next as its GGUF describes it (the values the engine hard-coded before
/// descriptions existed - model_desc_test compares a real file against this).
inline ModelGeometry reference() {
    ModelGeometry g;
    g.arch = caps::kArch;
    g.n_embd = 2560;
    g.n_layers = 48;
    g.qsa_interval = 4;
    for (int64_t l = 0; l < g.n_layers; ++l) g.layer_kind.push_back(l % 4 == 3 ? LayerKind::Qsa : LayerKind::Gdn);
    g.ssm_state_size = 128;
    g.ssm_k_heads = 16;
    g.ssm_v_heads = 48;
    g.ssm_d_conv = 4;
    g.ssm_value_dim = 6144;
    g.ssm_conv_channels = 10240;
    g.n_head = 24;
    g.n_head_kv = 2;
    g.head_dim = 256;
    g.head_dim_v = 256;
    g.n_rot = 64;
    g.idx_q_heads = 4;
    g.idx_key_dim = 128;
    g.idx_top_k = 2048;
    g.idx_block = 4;
    g.family = strata::ModelFamily::Qwen4Exp;
    g.rms_eps = (double) 1e-6f;   // the file stores an f32
    g.rope_freq_base = 1e7;
    g.rope_sections[0] = 11; g.rope_sections[1] = 11; g.rope_sections[2] = 10; g.rope_sections[3] = 0;
    g.hc = 4;
    g.hc_lr = 320;
    g.n_expert = 512;
    g.n_expert_used = 10;
    g.n_ff = 640;
    g.n_ff_shexp = 640;
    g.n_vocab = 248320;
    g.n_ctx_train = 262144;
    g.ple_layers = {1};
    g.ple_conv_kernel = 4;
    g.image_token_id = 248056;
    return g;
}

inline bool describe(const GgufFile& m, ModelGeometry& g, std::string& err) {
    const std::string p = std::string(caps::kArch) + ".";
    // The six keys the old guard required: their absence is an error, as before.
    const char* required[] = {"block_count", "embedding_length", "expert_count", "expert_used_count",
                              "attention.head_count", "attention.head_count_kv"};
    for (const char* k : required)
        if (m.get(p + k) == nullptr) { err = "missing " + p + k; return false; }
    // Every other key: the file's value when it carries one, the reference model's when it does not (some
    // variants' GGUFs predate a key; their tensors are checked against the result all the same).
    g = reference();
    meta_i64(m, p + "block_count", g.n_layers);
    meta_i64(m, p + "embedding_length", g.n_embd);
    meta_i64(m, p + "expert_count", g.n_expert);
    meta_i64(m, p + "expert_used_count", g.n_expert_used);
    meta_i64(m, p + "attention.head_count", g.n_head);
    meta_i64(m, p + "attention.head_count_kv", g.n_head_kv);
    meta_i64(m, p + "attention.key_length", g.head_dim);
    if (!meta_i64(m, p + "attention.value_length", g.head_dim_v)) g.head_dim_v = g.head_dim;
    meta_i64(m, p + "rope.dimension_count", g.n_rot);
    meta_i64(m, p + "attention.indexer.head_count", g.idx_q_heads);
    meta_i64(m, p + "attention.indexer.key_length", g.idx_key_dim);
    meta_i64(m, p + "attention.indexer.top_k", g.idx_top_k);
    meta_f64(m, p + "attention.layer_norm_rms_epsilon", g.rms_eps);
    meta_f64(m, p + "rope.freq_base", g.rope_freq_base);
    std::vector<int64_t> v;
    if (meta_i64s(m, p + "rope.dimension_sections", v))
        for (size_t i = 0; i < 4; ++i) g.rope_sections[i] = i < v.size() ? v[i] : 0;
    if (const MetaValue* t = m.get(p + "rope.scaling.type"); t != nullptr && t->type == MetaType::STRING)
        g.rope_scaling_type = t->s;
    meta_f64(m, p + "rope.scaling.factor", g.rope_scaling_factor);
    meta_f64(m, p + "rope.scaling.original_context_length", g.rope_scaling_orig_ctx);
    meta_i64(m, p + "ssm.state_size", g.ssm_state_size);
    meta_i64(m, p + "ssm.group_count", g.ssm_k_heads);
    meta_i64(m, p + "ssm.time_step_rank", g.ssm_v_heads);
    meta_i64(m, p + "ssm.conv_kernel", g.ssm_d_conv);
    if (!meta_i64(m, p + "ssm.inner_size", g.ssm_value_dim)) g.ssm_value_dim = g.ssm_state_size * g.ssm_v_heads;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    meta_i64(m, p + "hyper_connection.count", g.hc);
    meta_i64(m, p + "hyper_connection.low_rank", g.hc_lr);
    meta_i64(m, p + "expert_feed_forward_length", g.n_ff);
    if (!meta_i64(m, p + "expert_shared_feed_forward_length", g.n_ff_shexp)) g.n_ff_shexp = g.n_ff;
    meta_i64(m, p + "context_length", g.n_ctx_train);
    if (meta_i64s(m, p + "ple.layers", v)) g.ple_layers = v;
    meta_i64(m, p + "ple.conv_kernel", g.ple_conv_kernel);
    meta_i64(m, p + "ple.image_token_id", g.image_token_id);
    if (const MetaValue* t = m.get("tokenizer.ggml.tokens"); t != nullptr && t->type == MetaType::ARRAY)
        g.n_vocab = (int64_t) t->count;

    if (g.n_layers < 1 || g.n_layers > 4096) {
        err = p + "block_count = " + std::to_string(g.n_layers) + " is not a layer count";
        return false;
    }
    // THE LAYER TYPES.  The file says it twice: `full_attention_interval` and `attention.compress_ratios` (one
    // entry per layer, nonzero on the attention layers).  Both are read, and they must agree.
    int64_t interval = 0;
    const bool have_interval = meta_i64(m, p + "full_attention_interval", interval);
    std::vector<int64_t> ratios;
    const bool have_ratios = meta_i64s(m, p + "attention.compress_ratios", ratios) &&
                             (int64_t) ratios.size() == g.n_layers;
    if (have_interval && interval < 1) {
        err = p + "full_attention_interval = " + std::to_string(interval) + " is not an interval";
        return false;
    }
    if (!have_interval) interval = have_ratios ? 0 : 4;   // neither key: the reference model's layout
    g.qsa_interval = interval;
    g.layer_kind.assign((size_t) g.n_layers, LayerKind::Gdn);
    int64_t block = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const bool by_interval = interval > 0 && l % interval == interval - 1;
        const bool by_ratio = have_ratios && ratios[(size_t) l] != 0;
        if (interval > 0 && have_ratios && by_interval != by_ratio) {
            err = p + "full_attention_interval (" + std::to_string(interval) + ") and attention.compress_ratios "
                  "disagree about layer " + std::to_string(l);
            return false;
        }
        if (interval > 0 ? by_interval : by_ratio) g.layer_kind[(size_t) l] = LayerKind::Qsa;
        if (have_ratios && ratios[(size_t) l] != 0) {   // the indexer's pooling block: one for every QSA layer
            const int64_t r = ratios[(size_t) l];
            if (r < 2 || (block != 0 && r != block)) {
                err = p + "attention.compress_ratios[" + std::to_string(l) + "] = " + std::to_string(r) +
                      " is not one pooling block (>= 2) shared by every attention layer";
                return false;
            }
            block = r;
        }
    }
    if (block != 0) g.idx_block = block;
    return true;
}

inline std::vector<TensorReq> layer_tensors(const ModelGeometry& g, int64_t layer) {
    const int64_t D = g.hc_dim(), N = g.n_embd;
    std::vector<TensorReq> r = {
        // gated residual, EVERY layer - `gr_read` takes w_down (hc_lr, hc_dim) and w_up (hc_lr, hc_dim) after its
        // own transpose, so the orientation is [hc_dim, hc_lr] and [hc_lr, hc_dim]
        {"hc_attn_down.weight", {D, g.hc_lr, 0}},
        {"hc_attn_up.weight", {g.hc_lr, D, 0}},
        {"hc_attn_inject.weight", {D, g.hc, 0}},
        {"hc_ffn_down.weight", {D, g.hc_lr, 0}},
        {"hc_ffn_up.weight", {g.hc_lr, D, 0}},
        {"hc_ffn_inject.weight", {D, g.hc, 0}},
        {"hc_attn_norm.weight", {D, 0, 0}, PackForm::F32},
        {"hc_ffn_norm.weight", {D, 0, 0}, PackForm::F32},
        // MoE, EVERY layer.  The shared expert's scalar gate is BF16 in a pack - the one 1-D exception a reader
        // would not guess (layout.cpp has the story).
        {"ffn_gate_inp.weight", {N, g.n_expert, 0}},
        {"ffn_gate_shexp.weight", {N, g.n_ff_shexp, 0}},
        {"ffn_up_shexp.weight", {N, g.n_ff_shexp, 0}},
        {"ffn_down_shexp.weight", {g.n_ff_shexp, N, 0}},
        {"ffn_gate_inp_shexp.weight", {N, 0, 0}, PackForm::Bf16},
        {"ffn_gate_exps.weight", {N, g.n_ff, g.n_expert}, PackForm::Any, true},
        {"ffn_up_exps.weight", {N, g.n_ff, g.n_expert}, PackForm::Any, true},
        {"ffn_down_exps.weight", {g.n_ff, N, g.n_expert}, PackForm::Any, true},
    };
    if (g.kind(layer) == LayerKind::Gdn) {
        const TensorReq gdn[] = {
            {"attn_qkv.weight", {N, g.ssm_conv_channels, 0}},
            {"attn_gate.weight", {N, g.ssm_value_dim, 0}},
            {"ssm_out.weight", {g.ssm_value_dim, N, 0}},
            {"ssm_conv1d.weight", {g.ssm_d_conv, g.ssm_conv_channels, 0}},
            {"ssm_alpha.weight", {N, g.ssm_v_heads, 0}},
            {"ssm_beta.weight", {N, g.ssm_v_heads, 0}},
            {"ssm_a", {g.ssm_v_heads, 0, 0}, PackForm::F32},
            {"ssm_dt.bias", {g.ssm_v_heads, 0, 0}, PackForm::F32},
            {"ssm_norm.weight", {g.ssm_state_size, 0, 0}, PackForm::F32},
        };
        r.insert(r.end(), std::begin(gdn), std::end(gdn));
    } else {
        // attn_q carries the query AND its output gate: 2 * n_head * head_dim.  THE INDEXER: `q_proj` is the
        // QUERY count times the key width, `k_proj` ONE shared key head - conflating them made the planner's
        // indexer term four times too large when using the query head count.
        const TensorReq qsa[] = {
            {"attn_q.weight", {N, 2 * g.n_head * g.head_dim, 0}},
            {"attn_k.weight", {N, g.n_head_kv * g.head_dim, 0}},
            {"attn_v.weight", {N, g.n_head_kv * g.head_dim_v, 0}},
            {"attn_output.weight", {g.n_head * g.head_dim_v, N, 0}},
            {"attn_q_norm.weight", {g.head_dim, 0, 0}, PackForm::F32},
            {"attn_k_norm.weight", {g.head_dim, 0, 0}, PackForm::F32},
            {"indexer.q_proj.weight", {N, g.idx_q_heads * g.idx_key_dim, 0}},
            {"indexer.k_proj.weight", {N, g.idx_key_dim, 0}},
            {"indexer.q_norm.weight", {g.idx_key_dim, 0, 0}, PackForm::F32},
            {"indexer.k_norm.weight", {g.idx_key_dim, 0, 0}, PackForm::F32},
        };
        r.insert(r.end(), std::begin(qsa), std::end(qsa));
    }
    if (std::find(g.ple_layers.begin(), g.ple_layers.end(), layer) != g.ple_layers.end()) {
        // the PLE block's projections; a variant may keep them beside the PLE table in another file
        const TensorReq ple[] = {
            {"ple_key.weight", {N, D, 0}, PackForm::Any, false, true},
            {"ple_value.weight", {N, N, 0}, PackForm::Any, false, true},
            {"ple_norm_key.weight", {D, 0, 0}, PackForm::Any, false, true},
            {"ple_norm_query.weight", {D, 0, 0}, PackForm::Any, false, true},
            {"ple_norm_conv.weight", {D, 0, 0}, PackForm::Any, false, true},
            {"ple_conv1d.weight", {g.ple_conv_kernel, D, 0}, PackForm::Any, false, true},
        };
        r.insert(r.end(), std::begin(ple), std::end(ple));
    }
    return r;
}

/// Model-level tensors: shape-checked where present (the embedding or the head may come from another file).
inline std::vector<TensorReq> global_tensors(const ModelGeometry& g) {
    return {
        {"token_embd.weight", {g.n_embd, g.n_vocab, 0}, PackForm::Any, false, true},
        {"output.weight", {g.n_embd, g.n_vocab, 0}, PackForm::Any, false, true},
        {"output_hc_down.weight", {g.hc_dim(), g.hc_lr, 0}, PackForm::Any, false, true},
        {"output_hc_up.weight", {g.hc_lr, g.hc_dim(), 0}, PackForm::Any, false, true},
        {"output_hc_norm.weight", {g.hc_dim(), 0, 0}, PackForm::Any, false, true},
    };
}

/// What THIS BUILD's qwen4exp kernels can run.  Each line is a kernel constraint (qwen4exp_caps.hpp), not a
/// property of the model; a description that breaks one is refused rather than mis-run.
inline std::string capability(const ModelGeometry& g) {
    std::string out;
    auto need = [&](bool ok, const std::string& what, int64_t have, int64_t want, const char* who) {
        if (ok) return;
        if (!out.empty()) out += "; ";
        out += what + " is " + std::to_string(have) + ", " + who + " require " + std::to_string(want);
    };
    auto eq = [&](const char* key, int64_t have, int64_t want, const char* who) {
        need(have == want, std::string(caps::kArch) + "." + key, have, want, who);
    };
    eq("block_count", g.n_layers, caps::kLayers, "the qwen4exp path (validated depth: expert profiles, MTP, PLE)");
    eq("embedding_length", g.n_embd, caps::kEmbd, "the fused GR/PLE and prefill kernels");
    eq("hyper_connection.count", g.hc, caps::kHc, "the fused GR kernels");
    eq("hyper_connection.low_rank", g.hc_lr, caps::kHcLr, "the fused GR kernels");
    eq("ssm.state_size", g.ssm_state_size, caps::kSsmState, "the GDN kernels");
    eq("ssm.group_count", g.ssm_k_heads, caps::kSsmKHeads, "the GDN kernels and prefill");
    eq("ssm.time_step_rank", g.ssm_v_heads, caps::kSsmVHeads, "the GDN kernels and prefill");
    eq("ssm.conv_kernel", g.ssm_d_conv, caps::kSsmConv, "the GDN conv kernels");
    eq("ssm.inner_size", g.ssm_value_dim, caps::kSsmValueDim, "the GDN kernels and prefill");
    eq("attention.head_count", g.n_head, caps::kHead, "the QSA attention kernels (GQA 12)");
    eq("attention.head_count_kv", g.n_head_kv, caps::kHeadKv, "the QSA attention kernels (GQA 12)");
    eq("attention.key_length", g.head_dim, caps::kHeadDim, "the QSA attention and q4 KV kernels (head 256)");
    eq("attention.value_length", g.head_dim_v, caps::kHeadDim, "the QSA attention kernels (head 256)");
    eq("rope.dimension_count", g.n_rot, caps::kRot, "the QSA rope kernels");
    eq("attention.indexer.head_count", g.idx_q_heads, caps::kIdxHeads, "the selection kernels");
    eq("attention.indexer.key_length", g.idx_key_dim, caps::kIdxDim, "the selection kernels");
    eq("attention.indexer.top_k", g.idx_top_k, caps::kIdxTopK, "the selection kernels");
    eq("expert_feed_forward_length", g.n_ff, caps::kFf, "the prefill MoE kernels");
    eq("expert_shared_feed_forward_length", g.n_ff_shexp, caps::kFf, "the shared-expert kernels");
    need(g.n_expert >= 1 && g.n_expert <= caps::kExpertMax, std::string(caps::kArch) + ".expert_count", g.n_expert,
         caps::kExpertMax, "the prefill path (at most)");
    need((float) g.rms_eps == caps::kRmsEps, std::string(caps::kArch) + ".attention.layer_norm_rms_epsilon (x1e9)",
         (int64_t) std::llround(g.rms_eps * 1e9), (int64_t) std::llround((double) caps::kRmsEps * 1e9),
         "the norm kernels");
    need(g.ple_layers.size() == 1 && g.ple_layers[0] == caps::kPleLayer, std::string(caps::kArch) + ".ple.layers[0]",
         g.ple_layers.empty() ? -1 : g.ple_layers[0], caps::kPleLayer, "the PLE block (one PLE layer)");
    eq("ple.conv_kernel", g.ple_conv_kernel, caps::kPleConv, "the PLE conv kernels");
    return out;
}
}  // namespace qwen4exp_detail

// ---- laguna (poolside Laguna-S-2.1) -------------------------------------------------------------------------------

namespace laguna_detail {
namespace caps = strata::arch::laguna;
using qwen4exp_detail::meta_f64;
using qwen4exp_detail::meta_i64;
using qwen4exp_detail::meta_i64s;

/// The reference model: Laguna-S-2.1 as its GGUF describes it (plans/laguna-spec.md section 1).
inline ModelGeometry reference() {
    ModelGeometry g;
    g.arch = caps::kArch;
    g.n_embd = 3072;
    g.n_layers = 48;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const bool full = l % caps::kFullEvery == 0;
        g.layer_kind.push_back(full ? LayerKind::Attn : LayerKind::AttnSwa);
        g.n_head_per_layer.push_back(full ? 48 : 72);
    }
    g.n_head = 48;
    g.n_head_kv = 8;
    g.head_dim = 128;
    g.head_dim_v = 128;
    g.n_rot = 64;
    g.n_rot_swa = 128;
    g.sliding_window = 512;
    g.rms_eps = (double) 1e-6f;
    g.rope_freq_base = 500000;
    g.rope_freq_base_swa = 10000;
    g.rope_scaling_type = "yarn";
    g.rope_scaling_factor = 32;
    g.rope_scaling_orig_ctx = 8192;
    g.n_expert = 256;
    g.n_expert_used = 10;
    g.n_ff = 1024;
    g.n_ff_shexp = 1024;
    g.n_ff_dense = 12288;
    g.n_dense_lead = 1;
    g.expert_gating = caps::kGatingSigmoid;
    g.expert_weights_norm = true;
    g.expert_weights_scale = 2.5;
    g.attn_gate_per_head = true;
    g.n_vocab = 100352;
    g.n_ctx_train = 262144;
    return g;
}

inline bool describe(const GgufFile& m, ModelGeometry& g, std::string& err) {
    const std::string p = std::string(caps::kArch) + ".";
    const char* required[] = {"block_count", "embedding_length", "expert_count", "expert_used_count",
                              "attention.head_count", "attention.head_count_kv", "attention.key_length",
                              "attention.sliding_window", "expert_feed_forward_length"};
    for (const char* k : required)
        if (m.get(p + k) == nullptr) { err = "missing " + p + k; return false; }
    g = reference();   // keys the file does not carry keep the reference model's value
    meta_i64(m, p + "block_count", g.n_layers);
    meta_i64(m, p + "embedding_length", g.n_embd);
    meta_i64(m, p + "attention.head_count_kv", g.n_head_kv);
    meta_i64(m, p + "attention.key_length", g.head_dim);
    meta_i64(m, p + "attention.value_length", g.head_dim_v);
    meta_i64(m, p + "attention.sliding_window", g.sliding_window);
    meta_i64(m, p + "rope.dimension_count", g.n_rot);
    meta_i64(m, p + "rope.dimension_count_swa", g.n_rot_swa);
    meta_f64(m, p + "rope.freq_base", g.rope_freq_base);
    meta_f64(m, p + "rope.freq_base_swa", g.rope_freq_base_swa);
    meta_f64(m, p + "attention.layer_norm_rms_epsilon", g.rms_eps);
    meta_i64(m, p + "context_length", g.n_ctx_train);
    meta_i64(m, p + "expert_count", g.n_expert);
    meta_i64(m, p + "expert_used_count", g.n_expert_used);
    meta_i64(m, p + "expert_feed_forward_length", g.n_ff);
    meta_i64(m, p + "expert_shared_feed_forward_length", g.n_ff_shexp);
    meta_i64(m, p + "feed_forward_length", g.n_ff_dense);
    meta_i64(m, p + "leading_dense_block_count", g.n_dense_lead);
    meta_i64(m, p + "expert_gating_func", g.expert_gating);
    if (const MetaValue* v = m.get(p + "expert_weights_norm")) g.expert_weights_norm = v->u != 0;
    meta_f64(m, p + "expert_weights_scale", g.expert_weights_scale);
    if (const MetaValue* v = m.get(p + "rope.scaling.type")) g.rope_scaling_type = v->s;
    meta_f64(m, p + "rope.scaling.factor", g.rope_scaling_factor);
    meta_f64(m, p + "rope.scaling.original_context_length", g.rope_scaling_orig_ctx);
    if (const MetaValue* v = m.get("tokenizer.ggml.tokens")) g.n_vocab = (int64_t) v->count;
    // per-layer query heads: an array (one per layer) or one number for every layer
    std::vector<int64_t> heads;
    if (meta_i64s(m, p + "attention.head_count", heads)) {
        if ((int64_t) heads.size() != g.n_layers) {
            err = p + "attention.head_count has " + std::to_string(heads.size()) + " entries for " +
                  std::to_string(g.n_layers) + " layers";
            return false;
        }
    } else {
        int64_t h = 0;
        meta_i64(m, p + "attention.head_count", h);
        heads.assign((size_t) g.n_layers, h);
    }
    g.n_head_per_layer = heads;
    // the layer pattern: full attention every kFullEvery layers starting at 0 (the file carries no pattern key)
    g.layer_kind.clear();
    for (int64_t l = 0; l < g.n_layers; ++l)
        g.layer_kind.push_back(l % caps::kFullEvery == 0 ? LayerKind::Attn : LayerKind::AttnSwa);
    g.n_head = heads.empty() ? 0 : heads[0];   // the first (full) layer's count; `heads(layer)` is per layer
    return true;
}

inline std::vector<TensorReq> layer_tensors(const ModelGeometry& g, int64_t layer) {
    const int64_t N = g.n_embd, H = g.heads(layer), D = g.head_dim, Dv = g.head_dim_v;
    std::vector<TensorReq> r = {
        {"attn_norm.weight", {N, 0, 0}, PackForm::F32},
        {"attn_q.weight", {N, H * D, 0}},
        {"attn_k.weight", {N, g.n_head_kv * D, 0}},
        {"attn_v.weight", {N, g.n_head_kv * Dv, 0}},
        {"attn_output.weight", {H * Dv, N, 0}},
        {"attn_q_norm.weight", {D, 0, 0}, PackForm::F32},
        {"attn_k_norm.weight", {D, 0, 0}, PackForm::F32},
        {"ffn_norm.weight", {N, 0, 0}, PackForm::F32},
    };
    if (g.attn_gate_per_head) r.push_back({"attn_gate.weight", {N, H, 0}});
    if (layer < g.n_dense_lead) {
        const TensorReq dense[] = {
            {"ffn_gate.weight", {N, g.n_ff_dense, 0}},
            {"ffn_up.weight", {N, g.n_ff_dense, 0}},
            {"ffn_down.weight", {g.n_ff_dense, N, 0}},
        };
        r.insert(r.end(), std::begin(dense), std::end(dense));
    } else {
        const TensorReq moe[] = {
            {"ffn_gate_inp.weight", {N, g.n_expert, 0}},
            {"exp_probs_b.bias", {g.n_expert, 0, 0}, PackForm::F32},
            {"ffn_gate_shexp.weight", {N, g.n_ff_shexp, 0}},
            {"ffn_up_shexp.weight", {N, g.n_ff_shexp, 0}},
            {"ffn_down_shexp.weight", {g.n_ff_shexp, N, 0}},
            {"ffn_gate_exps.weight", {N, g.n_ff, g.n_expert}, PackForm::Any, true},
            {"ffn_up_exps.weight", {N, g.n_ff, g.n_expert}, PackForm::Any, true},
            {"ffn_down_exps.weight", {g.n_ff, N, g.n_expert}, PackForm::Any, true},
        };
        r.insert(r.end(), std::begin(moe), std::end(moe));
    }
    return r;
}

inline std::vector<TensorReq> global_tensors(const ModelGeometry& g) {
    return {
        {"token_embd.weight", {g.n_embd, g.n_vocab, 0}},
        {"output_norm.weight", {g.n_embd, 0, 0}, PackForm::F32},
        {"output.weight", {g.n_embd, g.n_vocab, 0}},
    };
}

/// Laguna: the description and the tensor check only.  Nothing in this build runs a laguna forward pass yet.
inline std::string capability(const ModelGeometry& g) {
    std::string out = "the laguna forward pass is not implemented in this build (description and tensor "
                      "check only; M2 adds attention/attention_swa layers, sigmoid MoE and the K-quant experts)";
    if (g.expert_gating != caps::kGatingSigmoid)
        out += "; expert_gating_func " + std::to_string(g.expert_gating) + " (the port targets sigmoid)";
    return out;
}
}  // namespace laguna_detail

// ---- qwen35 / qwen35moe (Qwen3.8-27B dense, Qwen3.6-35B-A3B) -----------------------------------------------------
//
// The GDN + gated-attention hybrid without Flash-Next's extras: a plain pre-norm residual (no hyper-connections), dense
// attention (no indexer), no PLE; full attention every `full_attention_interval` layers, and an in-file MTP block
// (`nextn_predict_layers`, counted after the trunk: blk.<n_layers>).  qwen35moe has a 256-expert MoE with a gated
// shared expert on every layer; qwen35 a dense SwiGLU FFN (`feed_forward_length`).  Both run in src/qwen35.

namespace qwen35_detail {
using qwen4exp_detail::meta_f64;
using qwen4exp_detail::meta_i64;
using qwen4exp_detail::meta_i64s;

inline ModelGeometry common_reference(const char* arch, strata::ModelFamily family, int64_t n_embd, int64_t n_layers,
                                      int64_t n_head, int64_t n_head_kv, int64_t ssm_k, int64_t ssm_v) {
    ModelGeometry g;
    g.arch = arch;
    g.family = family;
    g.n_embd = n_embd;
    g.n_layers = n_layers;
    g.qsa_interval = 4;
    for (int64_t l = 0; l < n_layers; ++l) g.layer_kind.push_back(l % 4 == 3 ? LayerKind::Attn : LayerKind::Gdn);
    g.n_mtp_layers = 1;
    g.ssm_state_size = 128;
    g.ssm_k_heads = ssm_k;
    g.ssm_v_heads = ssm_v;
    g.ssm_d_conv = 4;
    g.ssm_value_dim = 128 * ssm_v;
    g.ssm_conv_channels = 2 * 128 * ssm_k + g.ssm_value_dim;
    g.n_head = n_head;
    g.n_head_kv = n_head_kv;
    g.head_dim = g.head_dim_v = 256;
    g.n_rot = 64;
    g.rms_eps = (double) 1e-6f;
    g.rope_freq_base = 1e7;
    g.rope_sections[0] = 11; g.rope_sections[1] = 11; g.rope_sections[2] = 10; g.rope_sections[3] = 0;
    g.n_vocab = 248320;
    g.n_ctx_train = 262144;
    return g;
}
/// Qwen3.6-35B-A3B
inline ModelGeometry reference_moe() {
    ModelGeometry g = common_reference("qwen35moe", strata::ModelFamily::Qwen35Moe, 2048, 40, 16, 2, 16, 32);
    g.n_expert = 256;
    g.n_expert_used = 8;
    g.n_ff = 512;
    g.n_ff_shexp = 512;
    return g;
}
/// Qwen3.8-27B
inline ModelGeometry reference_dense() {
    ModelGeometry g = common_reference("qwen35", strata::ModelFamily::Qwen35, 5120, 64, 24, 4, 16, 48);
    g.ffn_dense = true;
    g.n_ff_dense = 17408;
    return g;
}

inline bool describe(const GgufFile& m, ModelGeometry& g, std::string& err) {
    const MetaValue* av = m.get("general.architecture");
    const std::string arch = av ? av->s : "";
    const strata::ModelFamily family = strata::model_family_from_arch(arch);
    const bool moe = family == strata::ModelFamily::Qwen35Moe;
    const std::string p = arch + ".";
    std::vector<const char*> required = {
        "block_count", "embedding_length", "attention.head_count", "attention.head_count_kv", "attention.key_length",
        "attention.value_length", "ssm.conv_kernel", "ssm.state_size", "ssm.group_count", "ssm.time_step_rank",
        "ssm.inner_size", "full_attention_interval"};
    if (moe) {
        for (const char* k : {"expert_count", "expert_used_count", "expert_feed_forward_length",
                              "expert_shared_feed_forward_length", "nextn_predict_layers"})
            required.push_back(k);
    } else {
        required.push_back("feed_forward_length");
    }
    for (const char* k : required) {
        const MetaValue* v = m.get(p + k);
        if (v == nullptr) { err = "missing " + p + k; return false; }
        if (!v->is_num()) { err = p + k + " is not a number"; return false; }
    }
    g = moe ? reference_moe() : reference_dense();
    int64_t block_count = 0, interval = 0;
    meta_i64(m, p + "block_count", block_count);
    g.n_mtp_layers = 0;
    meta_i64(m, p + "nextn_predict_layers", g.n_mtp_layers);
    meta_i64(m, p + "embedding_length", g.n_embd);
    meta_i64(m, p + "attention.head_count", g.n_head);
    meta_i64(m, p + "attention.head_count_kv", g.n_head_kv);
    meta_i64(m, p + "attention.key_length", g.head_dim);
    meta_i64(m, p + "attention.value_length", g.head_dim_v);
    meta_i64(m, p + "rope.dimension_count", g.n_rot);
    meta_f64(m, p + "rope.freq_base", g.rope_freq_base);
    meta_f64(m, p + "attention.layer_norm_rms_epsilon", g.rms_eps);
    std::vector<int64_t> v;
    if (meta_i64s(m, p + "rope.dimension_sections", v))
        for (size_t i = 0; i < 4; ++i) g.rope_sections[i] = i < v.size() ? v[i] : 0;
    if (const MetaValue* t = m.get(p + "rope.scaling.type"); t != nullptr && t->type == MetaType::STRING)
        g.rope_scaling_type = t->s;
    meta_f64(m, p + "rope.scaling.factor", g.rope_scaling_factor);
    meta_f64(m, p + "rope.scaling.original_context_length", g.rope_scaling_orig_ctx);
    meta_i64(m, p + "ssm.conv_kernel", g.ssm_d_conv);
    meta_i64(m, p + "ssm.state_size", g.ssm_state_size);
    meta_i64(m, p + "ssm.group_count", g.ssm_k_heads);
    meta_i64(m, p + "ssm.time_step_rank", g.ssm_v_heads);
    meta_i64(m, p + "ssm.inner_size", g.ssm_value_dim);
    meta_i64(m, p + "full_attention_interval", interval);
    meta_i64(m, p + "context_length", g.n_ctx_train);
    if (moe) {
        meta_i64(m, p + "expert_count", g.n_expert);
        meta_i64(m, p + "expert_used_count", g.n_expert_used);
        meta_i64(m, p + "expert_feed_forward_length", g.n_ff);
        meta_i64(m, p + "expert_shared_feed_forward_length", g.n_ff_shexp);
    } else {
        meta_i64(m, p + "feed_forward_length", g.n_ff_dense);
    }
    if (const MetaValue* t = m.get("tokenizer.ggml.tokens"); t != nullptr && t->type == MetaType::ARRAY && t->count > 0)
        g.n_vocab = (int64_t) t->count;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_value_dim;
    g.qsa_interval = interval;
    g.n_layers = block_count - g.n_mtp_layers;

    auto bad = [&](const std::string& what) { err = p + what; return false; };
    if (g.n_mtp_layers < 0 || g.n_mtp_layers >= block_count) return bad("nextn_predict_layers is not smaller than block_count");
    if (interval < 2) return bad("full_attention_interval must be at least 2 (the GDN / attention pattern)");
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_head <= 0 || g.n_head_kv <= 0 || g.head_dim <= 0 ||
        g.ssm_state_size <= 0 || g.ssm_k_heads <= 0 || g.ssm_v_heads <= 0 || g.ssm_d_conv <= 0)
        return bad("*: a geometry key is zero or out of range");
    if (moe && (g.n_expert <= 0 || g.n_expert_used <= 0 || g.n_expert_used > g.n_expert || g.n_ff <= 0))
        return bad("expert_*: zero or out of range");
    if (!moe && g.n_ff_dense <= 0) return bad("feed_forward_length is zero");
    if (g.head_dim_v != g.head_dim) return bad("attention.key_length != attention.value_length");
    if (g.ssm_value_dim != g.ssm_state_size * g.ssm_v_heads) return bad("ssm.inner_size != ssm.state_size * ssm.time_step_rank");
    if (g.ssm_v_heads % g.ssm_k_heads) return bad("ssm.time_step_rank is not a multiple of ssm.group_count");
    if (g.n_head % g.n_head_kv) return bad("attention.head_count is not a multiple of attention.head_count_kv");
    if (g.n_layers % interval) return bad("block_count - nextn_predict_layers is not a multiple of full_attention_interval");
    g.layer_kind.assign((size_t) g.n_layers, LayerKind::Gdn);
    for (int64_t l = 0; l < g.n_layers; ++l)
        if (l % interval == interval - 1) g.layer_kind[(size_t) l] = LayerKind::Attn;
    return true;
}

inline std::vector<TensorReq> layer_tensors(const ModelGeometry& g, int64_t layer) {
    const int64_t N = g.n_embd;
    std::vector<TensorReq> r = {
        {"attn_norm.weight", {N, 0, 0}, PackForm::F32},
        {"post_attention_norm.weight", {N, 0, 0}, PackForm::F32},
    };
    if (g.ffn_dense) {
        const TensorReq ffn[] = {
            {"ffn_gate.weight", {N, g.n_ff_dense, 0}},
            {"ffn_up.weight", {N, g.n_ff_dense, 0}},
            {"ffn_down.weight", {g.n_ff_dense, N, 0}},
        };
        r.insert(r.end(), std::begin(ffn), std::end(ffn));
    } else {
        const TensorReq moe[] = {
            {"ffn_gate_inp.weight", {N, g.n_expert, 0}},
            {"ffn_gate_inp_shexp.weight", {N, 0, 0}},
            {"ffn_gate_shexp.weight", {N, g.n_ff_shexp, 0}},
            {"ffn_up_shexp.weight", {N, g.n_ff_shexp, 0}},
            {"ffn_down_shexp.weight", {g.n_ff_shexp, N, 0}},
            {"ffn_gate_exps.weight", {N, g.n_ff, g.n_expert}, PackForm::Any, true},
            {"ffn_up_exps.weight", {N, g.n_ff, g.n_expert}, PackForm::Any, true},
            {"ffn_down_exps.weight", {g.n_ff, N, g.n_expert}, PackForm::Any, true},
        };
        r.insert(r.end(), std::begin(moe), std::end(moe));
    }
    if (g.kind(layer) == LayerKind::Gdn) {
        const TensorReq gdn[] = {
            {"attn_qkv.weight", {N, g.ssm_conv_channels, 0}},
            {"attn_gate.weight", {N, g.ssm_value_dim, 0}},
            {"ssm_out.weight", {g.ssm_value_dim, N, 0}},
            {"ssm_conv1d.weight", {g.ssm_d_conv, g.ssm_conv_channels, 0}},
            {"ssm_alpha.weight", {N, g.ssm_v_heads, 0}},
            {"ssm_beta.weight", {N, g.ssm_v_heads, 0}},
            {"ssm_a", {g.ssm_v_heads, 0, 0}, PackForm::F32},
            {"ssm_dt.bias", {g.ssm_v_heads, 0, 0}, PackForm::F32},
            {"ssm_norm.weight", {g.ssm_state_size, 0, 0}, PackForm::F32},
        };
        r.insert(r.end(), std::begin(gdn), std::end(gdn));
    } else {
        const TensorReq attn[] = {
            {"attn_q.weight", {N, 2 * g.n_head * g.head_dim, 0}},   // the query and its output gate
            {"attn_k.weight", {N, g.n_head_kv * g.head_dim, 0}},
            {"attn_v.weight", {N, g.n_head_kv * g.head_dim, 0}},
            {"attn_output.weight", {g.n_head * g.head_dim, N, 0}},
            {"attn_q_norm.weight", {g.head_dim, 0, 0}, PackForm::F32},
            {"attn_k_norm.weight", {g.head_dim, 0, 0}, PackForm::F32},
        };
        r.insert(r.end(), std::begin(attn), std::end(attn));
    }
    return r;
}

inline std::vector<TensorReq> global_tensors(const ModelGeometry& g) {
    return {
        {"token_embd.weight", {g.n_embd, g.n_vocab, 0}},
        {"output_norm.weight", {g.n_embd, 0, 0}, PackForm::F32},
        {"output.weight", {g.n_embd, g.n_vocab, 0}},
    };
}

/// What src/qwen35's kernels run: GDN heads of 128 (square state), conv 4, attention heads of 256 with a 64-dim
/// rotated slice.  The widths (n_embd, FFN, head counts) are run-time values there.  The ENGINE calls this at load
/// (the registry's capability stays empty: Strata's family tests describe tiny synthetic qwen35moe files).
inline std::string kernel_check(const ModelGeometry& g) {
    std::string out;
    auto need = [&](bool ok, const std::string& what) {
        if (ok) return;
        out += (out.empty() ? "" : "; ") + what;
    };
    need(g.ssm_state_size == 128, g.arch + ".ssm.state_size must be 128 (the GDN kernels)");
    need(g.ssm_d_conv == 4, g.arch + ".ssm.conv_kernel must be 4 (the GDN conv kernels)");
    need(g.head_dim == 256, g.arch + ".attention.key_length must be 256 (the attention kernels)");
    need(g.n_rot == 64, g.arch + ".rope.dimension_count must be 64 (the rope kernels)");
    need(g.n_head % g.n_head_kv == 0, g.arch + ".attention.head_count must be a multiple of head_count_kv");
    return out;
}
inline std::string capability(const ModelGeometry&) { return {}; }
}  // namespace qwen35_detail

/// Every architecture this build can run.  A second arch is one more entry here plus its kernels.
inline const std::vector<ArchSpec>& arch_registry() {
    static const std::vector<ArchSpec> r = {
        {strata::arch::qwen4exp::kArch, "Qwen3.8-Flash-Next: GDN + QSA hybrid, hyper-connections, PLE, 512-expert MoE",
         &qwen4exp_detail::describe, &qwen4exp_detail::reference, &qwen4exp_detail::layer_tensors,
         &qwen4exp_detail::global_tensors, &qwen4exp_detail::capability},
        {strata::arch::laguna::kArch, "Laguna-S-2.1: GQA attention (full every 4th layer, SWA-512 elsewhere), "
                                      "256-expert sigmoid MoE + shared expert (description only)",
         &laguna_detail::describe, &laguna_detail::reference, &laguna_detail::layer_tensors,
         &laguna_detail::global_tensors, &laguna_detail::capability},
        {"qwen35moe", "Qwen3.6-35B-A3B: GDN + gated attention 3:1, 256-expert MoE + shared expert, nextn MTP (src/qwen35)",
         &qwen35_detail::describe, &qwen35_detail::reference_moe, &qwen35_detail::layer_tensors,
         &qwen35_detail::global_tensors, &qwen35_detail::capability},
        {"qwen35", "Qwen3.8-27B: GDN + gated attention 3:1, dense SwiGLU FFN, nextn MTP (src/qwen35, talus-qwen35)",
         &qwen35_detail::describe, &qwen35_detail::reference_dense, &qwen35_detail::layer_tensors,
         &qwen35_detail::global_tensors, &qwen35_detail::capability},
    };
    return r;
}

inline const ArchSpec* find_arch(const std::string& name) {
    for (const ArchSpec& a : arch_registry())
        if (name == a.name) return &a;
    return nullptr;
}

inline std::string supported_archs() {
    std::string s;
    for (const ArchSpec& a : arch_registry()) s += (s.empty() ? "" : ", ") + std::string(a.name);
    return s;
}

/// The registered arch's reference description (for a pack that carries no GGUF metadata).  Throws on an
/// unknown arch.
inline ModelGeometry model_reference(const std::string& arch) {
    const ArchSpec* a = find_arch(arch);
    if (a == nullptr) throw std::runtime_error("unknown architecture '" + arch + "' (supported: " + supported_archs() + ")");
    return a->reference();
}

/// Metadata -> description: looks the arch up, describes the model, and checks that this build's kernels can
/// run it.  On failure `err` says which of the three failed.
/// `cap_out`: when given, a description this build cannot run is still returned, with the reasons in `*cap_out`
/// ("" = runnable) - for listings and `--dump-model-description`; every run passes nullptr and is refused.
inline bool describe_model(const GgufFile& meta, ModelGeometry& g, std::string& err, std::string* cap_out = nullptr) {
    const MetaValue* arch = meta.get("general.architecture");
    if (arch == nullptr) { err = "missing general.architecture"; return false; }
    const ArchSpec* a = find_arch(arch->s);
    if (a == nullptr) {
        err = "architecture is '" + arch->s + "', this engine supports: " + supported_archs();
        return false;
    }
    ModelGeometry d;
    if (!a->describe(meta, d, err)) return false;
    const std::string cap = a->capability(d);
    if (cap_out != nullptr) *cap_out = cap;
    else if (!cap.empty()) { err = cap; return false; }
    g = std::move(d);
    return true;
}

using TensorLookup = std::function<TensorShape(const std::string&)>;

/// One requirement against what `lookup` found.  `in_pack`: the lookup is a pack index, which holds no
/// `gguf_only` tensor.  Returns false with the problem in `err`, naming the tensor, what it has and what is required.
inline bool check_tensor_req(const TensorReq& r, const std::string& name, const std::string& where,
                             const TensorLookup& lookup, bool in_pack, std::string& err) {
    if (in_pack && r.gguf_only) return true;
    const TensorShape s = lookup(name);
    if (!s.found) {
        if (r.optional) return true;
        err = where + ": missing " + name;
        return false;
    }
    char buf[512];
    if (r.ne[1] == 0) {   // 1-D: the element count
        if (s.elements != r.ne[0]) {
            std::snprintf(buf, sizeof buf, "%s: %s has %lld elements, the description requires %lld", where.c_str(),
                          name.c_str(), (long long) s.elements, (long long) r.ne[0]);
            err = buf;
            return false;
        }
        return true;
    }
    for (int d = 0; d < 3; ++d) {
        const int64_t want = r.ne[d], have = s.ne[d];
        if (d == 2 && want == 0 && (have == 0 || have == 1)) continue;   // a 2-D requirement
        if (have != want) {
            std::snprintf(buf, sizeof buf, "%s: %s ne%d is %lld, the description requires %lld", where.c_str(),
                          name.c_str(), d, (long long) have, (long long) want);
            err = buf;
            return false;
        }
    }
    return true;
}

/// Every tensor ONE layer needs, given its kind (the arch's `layer_tensors`), through `lookup`.
inline bool check_layer_tensors(const ModelGeometry& g, int64_t layer, const TensorLookup& lookup, bool in_pack,
                                std::string& err) {
    const ArchSpec* a = find_arch(g.arch);
    if (a == nullptr) { err = "unknown architecture '" + g.arch + "'"; return false; }
    if (layer < 0 || layer >= g.n_layers || (int64_t) g.layer_kind.size() != g.n_layers) {
        err = "layer " + std::to_string(layer) + " is outside the description's " + std::to_string(g.layer_kind.size()) +
              " layer kinds (" + std::to_string(g.n_layers) + " layers)";
        return false;
    }
    const std::string where = "layer " + std::to_string(layer) + " (" + layer_kind_name(g.kind(layer)) + ")";
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::vector<TensorReq> own = a->layer_tensors(g, layer);
    for (const TensorReq& r : own)
        if (!check_tensor_req(r, prefix + r.suffix, where, lookup, in_pack, err)) return false;
    // AND NOTHING OF ANOTHER KIND: a tensor only another layer kind has must be absent.  This is the check that
    // the description's layer kinds and the file agree, which no shape check on the layer's own set can see.
    ModelGeometry other = g;
    for (LayerKind k : kAllLayerKinds) {
        if (k == g.kind(layer)) continue;
        other.layer_kind[(size_t) layer] = k;
        for (const TensorReq& r : a->layer_tensors(other, layer)) {
            const bool shared = std::any_of(own.begin(), own.end(), [&](const TensorReq& o) { return o.suffix == r.suffix; });
            if (shared || (in_pack && r.gguf_only) || !lookup(prefix + r.suffix).found) continue;
            err = where + ": has " + prefix + r.suffix + ", a " + layer_kind_name(k) +
                  " tensor - the file and the description's layer kinds disagree";
            return false;
        }
    }
    return true;
}

/// Every layer's tensors and the model-level ones.  Returns false with the first problem.
inline bool check_model_tensors(const ModelGeometry& g, const TensorLookup& lookup, bool in_pack, std::string& err) {
    const ArchSpec* a = find_arch(g.arch);
    if (a == nullptr) { err = "unknown architecture '" + g.arch + "'"; return false; }
    for (int64_t l = 0; l < g.n_layers; ++l)
        if (!check_layer_tensors(g, l, lookup, in_pack, err)) return false;
    for (const TensorReq& r : a->global_tensors(g))
        if (!check_tensor_req(r, r.suffix, "model", lookup, in_pack, err)) return false;
    return true;
}

/// A GGUF tensor's shape as the checker wants it (1-D: ne[1] == 0).
inline TensorShape gguf_shape(const TensorInfo* t) {
    TensorShape s;
    if (t == nullptr) return s;
    s.found = true;
    s.elements = (int64_t) t->elements();
    for (size_t d = 0; d < 3 && d < t->shape.size(); ++d) s.ne[d] = (int64_t) t->shape[d];
    if (t->shape.size() == 1) s.ne[1] = 0;
    return s;
}

/// Every layer's tensors against the description, in a model's GGUF shards (and `extra` files, e.g. a PLE GGUF
/// kept beside them).  Metadata only: no tensor data is read.
inline bool check_gguf_tensors(const GgufModel& model, const ModelGeometry& g, std::string& err,
                               const std::vector<const GgufFile*>& extra = {}) {
    return check_model_tensors(
        g,
        [&](const std::string& name) {
            const TensorInfo* t = model.find(name);
            for (size_t i = 0; t == nullptr && i < extra.size(); ++i) t = extra[i]->find(name);
            return gguf_shape(t);
        },
        false, err);
}

/// The description, one `key value` per line - what `strata-gguf --describe` and `strata --dump-model-description`
/// print, and what model_desc_test compares.
inline std::string format_description(const ModelGeometry& g) {
    std::string s;
    auto kv = [&](const char* k, const std::string& v) { s += std::string(k) + " " + v + "\n"; };
    auto i = [&](const char* k, int64_t v) { kv(k, std::to_string(v)); };
    auto f = [&](const char* k, double v) {
        char b[64];
        std::snprintf(b, sizeof b, "%.9g", v);
        kv(k, b);
    };
    kv("arch", g.arch);
    i("n_layers", g.n_layers);
    i("n_embd", g.n_embd);
    std::string kinds, qsa;
    for (int64_t l = 0; l < (int64_t) g.layer_kind.size(); ++l) {
        kinds += (l ? "," : "") + std::string(layer_kind_name(g.kind(l)));
        if (g.kind(l) == LayerKind::Qsa) qsa += (qsa.empty() ? "" : ",") + std::to_string(l);
    }
    kv("layer_kinds", kinds);
    kv("qsa_layers", qsa);
    i("n_qsa_layers", g.n_qsa_layers());
    i("n_gdn_layers", g.n_gdn_layers());
    i("qsa_interval", g.qsa_interval);
    i("ssm_state_size", g.ssm_state_size);
    i("ssm_k_heads", g.ssm_k_heads);
    i("ssm_v_heads", g.ssm_v_heads);
    i("ssm_d_conv", g.ssm_d_conv);
    i("ssm_conv_channels", g.ssm_conv_channels);
    i("ssm_value_dim", g.ssm_value_dim);
    i("n_head", g.n_head);
    i("n_head_kv", g.n_head_kv);
    i("head_dim", g.head_dim);
    i("head_dim_v", g.head_dim_v);
    i("n_rot", g.n_rot);
    i("idx_q_heads", g.idx_q_heads);
    i("idx_key_dim", g.idx_key_dim);
    i("idx_top_k", g.idx_top_k);
    f("rms_eps", g.rms_eps);
    f("rope_freq_base", g.rope_freq_base);
    kv("rope_sections", std::to_string(g.rope_sections[0]) + "," + std::to_string(g.rope_sections[1]) + "," +
                            std::to_string(g.rope_sections[2]) + "," + std::to_string(g.rope_sections[3]));
    kv("rope_scaling_type", g.rope_scaling_type.empty() ? "(none)" : g.rope_scaling_type);
    f("rope_scaling_factor", g.rope_scaling_factor);
    f("rope_scaling_orig_ctx", g.rope_scaling_orig_ctx);
    i("hc", g.hc);
    i("hc_lr", g.hc_lr);
    i("n_expert", g.n_expert);
    i("n_expert_used", g.n_expert_used);
    i("n_ff", g.n_ff);
    i("n_ff_shexp", g.n_ff_shexp);
    i("n_vocab", g.n_vocab);
    i("n_ctx_train", g.n_ctx_train);
    std::string ple;
    for (int64_t l : g.ple_layers) ple += (ple.empty() ? "" : ",") + std::to_string(l);
    kv("ple_layers", ple);
    i("ple_conv_kernel", g.ple_conv_kernel);
    i("image_token_id", g.image_token_id);
    if (!g.n_head_per_layer.empty()) {
        std::string h;
        for (int64_t v : g.n_head_per_layer) h += (h.empty() ? "" : ",") + std::to_string(v);
        kv("n_head_per_layer", h);
    }
    if (g.sliding_window) i("sliding_window", g.sliding_window);
    if (g.rope_freq_base_swa) f("rope_freq_base_swa", g.rope_freq_base_swa);
    if (g.n_rot_swa) i("n_rot_swa", g.n_rot_swa);
    if (g.n_dense_lead) { i("n_dense_lead", g.n_dense_lead); i("n_ff_dense", g.n_ff_dense); }
    if (g.expert_gating) {
        i("expert_gating", g.expert_gating);
        i("expert_weights_norm", g.expert_weights_norm ? 1 : 0);
        f("expert_weights_scale", g.expert_weights_scale);
    }
    if (g.attn_gate_per_head) i("attn_gate_per_head", 1);
    return s;
}

}  // namespace strata::core

namespace strata {

/// The load-time architecture check on a GGUF's metadata shard (formerly the fixed-value `Qwen4ExpGuard`): the
/// arch must be registered, its description derivable, and within this build's kernel capabilities.  Empty == ok.
inline std::string check_architecture(const GgufFile& g) {
    core::ModelGeometry d;
    std::string err;
    return core::describe_model(g, d, err) ? std::string() : err;
}

/// The engine's guard (Strata 0.1.44): `check_architecture`, then a family this engine has a path for.  A
/// recognised family it does not run is refused with a message that says so.
inline std::string check_engine_architecture(const GgufFile& g) {
    std::string err = check_architecture(g);
    if (!err.empty()) return err;
    return model_family_not_supported(model_family(g));
}

}  // namespace strata
