// src/core/layout.cpp - the shape checks over a pack's index.  See the header for why this exists.
//
// The tensor list per layer kind is the arch's (model_desc.hpp `layer_tensors`), shared with the GGUF-side check
// (`check_gguf_tensors`), so the two cannot disagree about what a layer holds.  What only a PACK has is the
// engine form of its 1-D tensors, checked here.
#include "strata/core/layout.hpp"

#include <cstdio>

namespace strata::core {
namespace {

/// A pack row's shape as the checker wants it.  `ne0` is the CONTIGUOUS axis, matching the manifest and
/// `s_gemv`'s convention (`y[o] = sum_i x[i]*W[i][o]`, row o contiguous of length ne0); a 1-D row has ne1 0.
TensorShape pack_shape(const WeightRef* r) {
    TensorShape s;
    if (r == nullptr) return s;
    s.found = true;
    s.ne[0] = r->ne0;
    s.ne[1] = r->ne1;
    s.elements = r->elements;
    return s;
}

TensorLookup pack_lookup(const WeightTable& t) {
    return [&t](const std::string& name) { return pack_shape(t.find(name)); };
}

/// The ENGINE FORM of the 1-D tensors.
///
/// `form` is not decoration.  The engine form of a 1-D tensor is whatever `WeightKind` the loader applied -
/// `F32` copies 4 B/elem, `Bf16InF32` re-rounds to 2 - and a consumer that casts either one to `const float*`
/// reads 2x its length.  **THAT IS NOT HYPOTHETICAL: `ffn_gate_inp_shexp.weight` is BF16, so the arena holds
/// 5120 B for 2560 elements, and `shared_expert` read it as f32.  Every one of the 2560 MoE outputs came out
/// non-finite** - a wrong answer loud enough to notice, which is the lucky version.  Reading 5120 B past the
/// end of a tensor inside a 4.5 GiB arena does not fault, so the same mistake on a tensor followed by
/// plausible bytes would have produced plausible logits.
bool check_forms(const WeightTable& t, const ModelGeometry& g, int64_t layer, std::string& err) {
    const ArchSpec* a = find_arch(g.arch);
    if (a == nullptr || layer < 0 || layer >= (int64_t) g.layer_kind.size()) return true;   // check_layer_tensors says so
    const LayerView v(t, layer);
    for (const TensorReq& w : a->layer_tensors(g, layer)) {
        if (w.form == PackForm::Any) continue;
        const WeightRef* r = v.get(w.suffix.c_str());
        if (r == nullptr) continue;   // presence is check_layer_tensors'
        const WeightKind want = w.form == PackForm::Bf16 ? WeightKind::Bf16InF32 : WeightKind::F32;
        if (r->kind != want) {
            // The byte count is what makes this a REAL check and not a label: a 4 B/elem tensor holds
            // `elements * 4`, a 2 B/elem one holds `elements * 2`, and a consumer reading it as the wrong one
            // walks off the end.
            const uint64_t want_bytes = (uint64_t) r->elements * (want == WeightKind::Bf16InF32 ? 2u : 4u);
            char buf[512];
            std::snprintf(buf, sizeof buf,
                          "layer %lld: %s is engine form %d (%llu B), the kernels read it as form %d (%llu B)",
                          (long long) layer, v.name(w.suffix.c_str()).c_str(), (int) r->kind,
                          (unsigned long long) r->bytes, (int) want, (unsigned long long) want_bytes);
            err = buf;
            return false;
        }
    }
    return true;
}

}  // namespace

std::string LayerView::name(const char* suffix) const {
    return "blk." + std::to_string(layer_) + "." + suffix;
}

bool check_layer(const WeightTable& table, const ModelGeometry& g, int64_t layer, std::string& err) {
    return check_layer_tensors(g, layer, pack_lookup(table), true, err) && check_forms(table, g, layer, err);
}

bool check_all(const WeightTable& table, const ModelGeometry& g, std::string& err) {
    if (!check_model_tensors(g, pack_lookup(table), true, err)) return false;
    for (int64_t l = 0; l < g.n_layers; ++l)
        if (!check_forms(table, g, l, err)) return false;
    return true;
}

}  // namespace strata::core
