// include/strata/core/family_layout.hpp - per model family: the geometry read from a GGUF's metadata, and the
// tensor layout table (every tensor of a layer, by name and GGUF shape) checked against a model file's directory.
//
// Header-only and GPU-free: the engine, the tools and the unit tests (tests/core/model_family_test.cpp) share it.
//
//   * geometry_from_gguf: fills core::ModelGeometry from "<arch>.*" keys.  For qwen4exp the keys a file does not
//     carry keep Flash-Next's values (the struct defaults, which every published qwen4exp file matches); for
//     qwen35moe every key is required (check_architecture_qwen35moe).
//   * compiled_geometry_mismatch: the first field where a geometry differs from the shapes this engine's kernels
//     are built for (the expert count and top-k are run-time values and are not compared).
//   * layer_tensors / global_tensors: the layout tables.  Names are GGUF names (blk.N.<suffix>), shapes are in
//     GGUF order (dim 0 contiguous), as tools/gguf_reader.py prints them.
//   * check_gguf_layout: every table entry present with its shape, inside its file, of a type with a known block
//     size; and, for a family whose table is complete (qwen35moe), no blk.* tensor the table does not name.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"
#include "strata/core/geometry.hpp"
#include "strata/core/model_desc.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace strata::core {

/// Fills `g` (and `top_k`, the experts per token) from the metadata shard `meta`.  False with `err` when the
/// architecture guard refuses the file or a key has an impossible value.  Talus: the arch registry's description
/// (model_desc.hpp `describe_model`), which fills the family fields too.
inline bool geometry_from_gguf(const strata::GgufFile& meta, ModelGeometry& g, int64_t& top_k, std::string& err) {
    if (!describe_model(meta, g, err)) {
        err = meta.path() + ": " + err;
        return false;
    }
    top_k = g.n_expert_used;
    return true;
}

/// The first geometry field that differs from the shapes this engine's kernels are built for (ModelGeometry's
/// defaults, Flash-Next's), as "<what> is <got>, this engine's kernels are built for <want>"; empty when none.
/// n_expert and top-k are run-time sizes (the Coder ships 256 experts) and n_vocab is the embedding's business.
inline std::string compiled_geometry_mismatch(const ModelGeometry& g) {
    const CompiledGeometry& c = kCompiledGeometry;
    if (g.family != c.family)
        return std::string("the model is ") + strata::model_family_title(g.family) + ", this engine's kernels are built "
               "for " + strata::model_family_title(c.family);
    struct F {
        const char* what;
        int64_t got, want;
    };
    const F fields[] = {
        {"embedding_length", g.n_embd, c.n_embd},
        {"the layer count", g.n_layers, c.n_layers},
        {"full_attention_interval", g.qsa_interval, c.qsa_interval},
        {"ssm.state_size", g.ssm_state_size, c.ssm_state_size},
        {"ssm.group_count", g.ssm_k_heads, c.ssm_k_heads},
        {"ssm.time_step_rank", g.ssm_v_heads, c.ssm_v_heads},
        {"ssm.conv_kernel", g.ssm_d_conv, c.ssm_d_conv},
        {"the GDN conv channels", g.ssm_conv_channels, c.ssm_conv_channels},
        {"ssm.inner_size", g.ssm_value_dim, c.ssm_value_dim},
        {"attention.head_count", g.n_head, c.n_head},
        {"attention.head_count_kv", g.n_head_kv, c.n_head_kv},
        {"attention.key_length", g.head_dim, c.head_dim},
        {"attention.indexer.head_count", g.idx_q_heads, c.idx_q_heads},
        {"attention.indexer.key_length", g.idx_key_dim, c.idx_key_dim},
        {"hyper_connection.count", g.hc, c.hc},
        {"hyper_connection.low_rank", g.hc_lr, c.hc_lr},
        {"expert_feed_forward_length", g.n_ff, c.n_ff},
        {"expert_shared_feed_forward_length", g.n_ff_shexp, c.n_ff_shexp},
        {"the in-file MTP layers", g.n_mtp_layers, c.n_mtp_layers},
    };
    for (const F& f : fields)
        if (f.got != f.want) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s is %lld, this engine's kernels are built for %lld", f.what,
                          (long long) f.got, (long long) f.want);
            return buf;
        }
    return {};
}

enum class FamilyLayerKind { Gdn, Attention, Mtp };   // Talus: core::LayerKind is the description's

inline FamilyLayerKind family_layer_kind(const ModelGeometry& g, int64_t layer) {
    if (layer >= g.n_layers) return FamilyLayerKind::Mtp;
    return g.is_full_attention(layer) ? FamilyLayerKind::Attention : FamilyLayerKind::Gdn;
}

/// One tensor of the layout table: its name (a blk.N. suffix for a layer's, the full name for a global one) and
/// its GGUF shape.
struct TensorWant {
    std::string name;
    std::vector<int64_t> shape;
};

/// Every tensor of layer `layer` (kind from `layer_kind`), by family.
inline std::vector<TensorWant> layer_tensors(const ModelGeometry& g, int64_t layer) {
    const FamilyLayerKind kind = family_layer_kind(g, layer);
    const int64_t N = g.n_embd, NE = g.n_expert, FF = g.n_ff, FS = g.n_ff_shexp;
    std::vector<TensorWant> t;
    auto add = [&](const char* s, std::vector<int64_t> shape) { t.push_back({s, std::move(shape)}); };
    if (g.ffn_dense) {   // qwen35: a dense SwiGLU FFN, every layer (the MTP block's too)
        add("ffn_gate.weight", {N, g.n_ff_dense});
        add("ffn_up.weight", {N, g.n_ff_dense});
        add("ffn_down.weight", {g.n_ff_dense, N});
    } else {
    // MoE, every layer (the MTP block's too)
    add("ffn_gate_inp.weight", {N, NE});
    add("ffn_gate_inp_shexp.weight", {N});
    add("ffn_gate_shexp.weight", {N, FS});
    add("ffn_up_shexp.weight", {N, FS});
    add("ffn_down_shexp.weight", {FS, N});
    add("ffn_gate_exps.weight", {N, FF, NE});
    add("ffn_up_exps.weight", {N, FF, NE});
    add("ffn_down_exps.weight", {FF, N, NE});
    }
    if (g.has_hyper_connections()) {
        // gated residual (Flash-Next): `gr_read` transposes, so the file's orientation is [hc_dim, hc_lr] / [hc_lr, hc_dim]
        const int64_t H = g.hc_dim();
        add("hc_attn_down.weight", {H, g.hc_lr});
        add("hc_attn_up.weight", {g.hc_lr, H});
        add("hc_attn_inject.weight", {H, g.hc});
        add("hc_attn_norm.weight", {H});
        add("hc_ffn_down.weight", {H, g.hc_lr});
        add("hc_ffn_up.weight", {g.hc_lr, H});
        add("hc_ffn_inject.weight", {H, g.hc});
        add("hc_ffn_norm.weight", {H});
    } else {
        // plain pre-norm residual (qwen35moe): x += mixer(rms(x)); x += moe(rms(x))
        add("attn_norm.weight", {N});
        add("post_attention_norm.weight", {N});
    }
    if (kind == FamilyLayerKind::Gdn) {
        add("attn_qkv.weight", {N, g.ssm_conv_channels});
        add("attn_gate.weight", {N, g.ssm_value_dim});
        add("ssm_out.weight", {g.ssm_value_dim, N});
        add("ssm_conv1d.weight", {g.ssm_d_conv, g.ssm_conv_channels});
        add("ssm_alpha.weight", {N, g.ssm_v_heads});
        add("ssm_beta.weight", {N, g.ssm_v_heads});
        add("ssm_a", {g.ssm_v_heads});
        add("ssm_dt.bias", {g.ssm_v_heads});
        add("ssm_norm.weight", {g.ssm_state_size});
    } else {
        // attention (query + output gate in attn_q), also the MTP block's
        add("attn_q.weight", {N, 2 * g.n_head * g.head_dim});
        add("attn_k.weight", {N, g.n_head_kv * g.head_dim});
        add("attn_v.weight", {N, g.n_head_kv * g.head_dim});
        add("attn_output.weight", {g.n_head * g.head_dim, N});
        add("attn_q_norm.weight", {g.head_dim});
        add("attn_k_norm.weight", {g.head_dim});
        if (g.has_indexer() && kind == FamilyLayerKind::Attention) {
            // `q_proj` is the QUERY count, `k_proj` the KEY width (round 194: they are different numbers)
            add("indexer.q_proj.weight", {N, g.idx_q_heads * g.idx_key_dim});
            add("indexer.k_proj.weight", {N, g.idx_key_dim});
            add("indexer.q_norm.weight", {g.idx_key_dim});
            add("indexer.k_norm.weight", {g.idx_key_dim});
        }
    }
    if (kind == FamilyLayerKind::Mtp) {
        // the GGUF-native MTP block: eh_proj over concat(enorm(embedding), hnorm(hidden))
        add("nextn.eh_proj.weight", {2 * N, N});
        add("nextn.enorm.weight", {N});
        add("nextn.hnorm.weight", {N});
        add("nextn.shared_head_norm.weight", {N});
    }
    return t;
}

/// The tensors outside the layers, by family.  (Flash-Next's output_norm, PLE and n-gram tables are not listed:
/// the published files place them differently, and their loaders check them.)
inline std::vector<TensorWant> global_tensors(const ModelGeometry& g) {
    std::vector<TensorWant> t;
    t.push_back({"token_embd.weight", {g.n_embd, g.n_vocab}});
    t.push_back({"output.weight", {g.n_embd, g.n_vocab}});
    if (g.has_hyper_connections()) {
        t.push_back({"output_hc_down.weight", {g.hc_dim(), g.hc_lr}});
        t.push_back({"output_hc_up.weight", {g.hc_lr, g.hc_dim()}});
        t.push_back({"output_hc_norm.weight", {g.hc_dim()}});
    } else {
        t.push_back({"output_norm.weight", {g.n_embd}});
    }
    return t;
}

/// Whether the family's table names every blk.* tensor its files carry (then an unnamed one is an error).
inline bool layout_table_complete(const ModelGeometry& g) {
    return g.family == strata::ModelFamily::Qwen35Moe || g.family == strata::ModelFamily::Qwen35;
}

/// Checks the model `m` (all shards) against the layout table of `g`.  `checked` counts the tensors found.
inline bool check_gguf_layout(const strata::GgufModel& m, const ModelGeometry& g, std::string& err,
                              size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const std::string& name, const std::vector<int64_t>& shape) {
        named.insert(name);
        size_t s = 0;
        const strata::TensorInfo* t = m.find(name, &s);
        if (!t) {
            err = "missing " + name;
            return false;
        }
        bool same = t->shape.size() == shape.size();
        for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == (uint64_t) shape[i];
        if (!same) {
            std::string got, want;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
            for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
            err = name + " is [" + got + "], the " + strata::model_family_title(g.family) + " layout wants [" + want + "]";
            return false;
        }
        if (!m.in_bounds(*t, s)) {
            err = name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
            return false;
        }
        ++n;
        return true;
    };
    for (const TensorWant& w : global_tensors(g))
        if (!check(w.name, w.shape)) return false;
    for (int64_t l = 0; l < g.n_layers + g.n_mtp_layers; ++l)
        for (const TensorWant& w : layer_tensors(g, l))
            if (!check("blk." + std::to_string(l) + "." + w.name, w.shape)) return false;
    if (layout_table_complete(g)) {
        for (size_t s = 0; s < m.size(); ++s)
            for (const strata::TensorInfo& t : m.shard(s).tensors())
                if (!named.count(t.name)) {
                    err = t.name + " is not in the " + std::string(strata::model_family_title(g.family)) + " layout";
                    return false;
                }
    }
    if (checked) *checked = n;
    return true;
}

}  // namespace strata::core
