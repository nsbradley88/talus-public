// include/strata/core/layout.hpp - the pack's LAYOUT, resolved by name and checked against the kernels.
//
// `WeightTable` answers "where are the bytes".  This answers "is this the tensor I think it is", and it is
// the layer where a pack and a kernel disagree - which is the failure this project keeps paying for.  Two
// examples, both real:
//
//   * round 194 sized the indexer key store from `indexer.head_count = 4`, which counts QUERY heads.  The
//     cached key is ONE shared head of 128 (`indexer.k_proj` is [2560, 128]), so the term was 4x too big and
//     the error survived a round because it moved in the direction that TIGHTENS the budget.
//   * `docs/semantics.md` and the tensor manifest agree on every dimension here, but nothing CHECKED that a
//     named tensor had the shape the kernel reading it assumes.
//
// So this header does two things: it names the tensors per layer type, and it asserts every shape the
// kernels depend on.  A mismatch is reported with the tensor name, the shape found and the shape required,
// at LOAD time - not as a wrong number inside a GEMV at token 4000.
#pragma once

#include "strata/core/model_desc.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/qsa.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

// `ModelGeometry` - the model's description - and `is_qsa_layer` live in model_desc.hpp: filled from the GGUF at
// load by the arch registry, with an explicit per-layer kind list.  This header keeps the pack-side checks.

/// The shapes the QSA kernels want, every one from the model's geometry; only the KV page size is the
/// kernels' own choice (`qsa_real_shapes`).
inline strata::kernels::QsaShapes qsa_shapes(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.n_rot = g.n_rot;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    s.idx_block = g.idx_block;
    s.idx_top_k = g.idx_top_k;
    return s;
}

/// One layer's tensors, resolved by NAME.  `get("attn_qkv.weight")` looks up `blk.<layer>.attn_qkv.weight`
/// and returns null if the pack does not have it - a null is information, because a GDN layer has no
/// `attn_q` and a QSA layer has no `attn_qkv`.
///
/// Holds a reference to the table; the table must outlive it.
class LayerView {
public:
    LayerView(const WeightTable& table, int64_t layer) : table_(&table), layer_(layer) {}

    int64_t layer() const { return layer_; }
    std::string name(const char* suffix) const;
    const WeightRef* get(const char* suffix) const { return table_->find(name(suffix)); }

private:
    const WeightTable* table_;
    int64_t layer_;
};

/// Every shape the kernels depend on, asserted for ONE layer of a PACK (its index): the arch's per-layer-kind
/// tensor list (model_desc.hpp `layer_tensors`), plus the engine form of the 1-D tensors.  Returns false and
/// fills `err` with the first mismatch, naming the tensor, what it has and what is required.
///
/// This is deliberately a separate function from `LayerView`: a caller that only wants the pointers should
/// not pay for the checks, and a caller that wants the checks should get ALL of them rather than the ones
/// its own call site happens to touch.
bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err);

/// `check_layer` over every layer, plus the cross-layer property no per-tensor check can see: a GDN layer
/// carries no QSA tensor and a QSA layer no GDN tensor (the description's layer kinds and the pack agree), and
/// the model-level tensors.  Run at load (generate.cpp).  Returns the first failure.
bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err);

}  // namespace strata::core
