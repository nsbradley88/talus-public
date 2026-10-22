// src/program/qwen4exp_r1.cpp - qwen4exp-r1 (the residual of Piece 1): the C++ side of the Rust expert source and PLE table
// reader. Rust registers its factory (tk_qwen4exp_r1_set_rust_hooks); generate.cpp asks it for a source / a reader on
// the Rust data plane and the C++ objects adopt the handle (FileExpertSource::adopt_rust, PleTable::adopt_rust), after
// which every method of theirs forwards to Rust. Here: that registry, the adoption calls, and the few facts the Rust
// side reads from the C++ engine (the expert layout the pack loader set, the aux-CPU pinning, the ForesightSwap holds).
// The logic is Rust's: rust/crates/talus-model/src/qwen4exp/weights/{source,ple_reader,r1}.rs.
#include "qwen4exp_r1.hpp"

#include "strata/core/foresight_swap.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
std::atomic<const tk_qwen4exp_r1_rust_hooks_t*> g_hooks{nullptr};

// the layout's arrays, kept for the Rust reader (rebuilt on every call)
struct LayoutExport {
    std::vector<uint64_t> up_off, down_off, fmt_bytes, offset, bytes, gguf_off;
    std::vector<int32_t> gu_type, d_type;
    std::vector<int64_t> n_embd, n_ff;
    std::vector<const char*> gguf_file;
};
LayoutExport g_layout;

int svc_expert_layout(tk_qwen4exp_r1_layout_t* out) {
    if (out == nullptr) return -1;
    const auto& lay = strata::kernels::cpu::expert_layout();
    LayoutExport& x = g_layout;
    const size_t nl = lay.fmt.size();
    x.up_off.resize(nl); x.down_off.resize(nl); x.fmt_bytes.resize(nl); x.gu_type.resize(nl); x.d_type.resize(nl);
    x.n_embd.resize(nl); x.n_ff.resize(nl);
    for (size_t i = 0; i < nl; ++i) {
        const auto& f = lay.fmt[i];
        x.up_off[i] = (uint64_t) f.up_off;
        x.down_off[i] = (uint64_t) f.down_off;
        x.fmt_bytes[i] = (uint64_t) f.bytes;
        x.gu_type[i] = f.gu_type;
        x.d_type[i] = f.d_type;
        x.n_embd[i] = f.n_embd;
        x.n_ff[i] = f.n_ff;
    }
    x.offset = lay.offset;
    x.bytes = lay.bytes;
    x.gguf_off = lay.gguf_off;
    x.gguf_file.clear();
    for (const std::string& s : lay.gguf_file) x.gguf_file.push_back(s.c_str());
    *out = tk_qwen4exp_r1_layout_t{};
    out->native = lay.native ? 1 : 0;
    out->n_layers = lay.n_layers;
    out->n_expert = lay.n_expert;
    // every per-layer array has the fmt's length; offset / bytes may differ in a malformed layout, which Rust refuses
    out->n_per_layer = nl;
    if (lay.offset.size() != nl || lay.bytes.size() != nl) out->n_per_layer = 0;
    out->fmt_up_off = x.up_off.data();
    out->fmt_down_off = x.down_off.data();
    out->fmt_bytes = x.fmt_bytes.data();
    out->fmt_gu_type = x.gu_type.data();
    out->fmt_d_type = x.d_type.data();
    out->fmt_n_embd = x.n_embd.data();
    out->fmt_n_ff = x.n_ff.data();
    out->offset = x.offset.data();
    out->bytes = x.bytes.data();
    out->gguf_off = x.gguf_off.data();
    out->n_gguf_off = x.gguf_off.size();
    out->gguf_file = x.gguf_file.data();
    out->n_gguf_file = x.gguf_file.size();
    out->max_blob = lay.max_blob;
    out->total = lay.total;
    out->canonical_blob = (uint64_t) strata::kernels::cpu::BLOB;
    return 0;
}
void svc_pin_aux_thread() { strata::aux_cpus::pin_current_thread(); }
void svc_fs_hold(void* fs, int64_t layer, int32_t expert, int hold) {
    auto* f = static_cast<strata::core::ForesightSwap*>(fs);
    if (f == nullptr) return;
    if (hold) f->hold(layer, expert);
    else f->unhold(layer, expert);
}
int svc_ple_prefetch_enabled() { return strata::kernels::ple_prefetch_enabled() ? 1 : 0; }

const tk_qwen4exp_r1_services_t g_services = {
    sizeof(tk_qwen4exp_r1_services_t), svc_expert_layout, svc_pin_aux_thread, svc_fs_hold, svc_ple_prefetch_enabled,
};
}  // namespace

extern "C" const tk_qwen4exp_r1_services_t* tk_qwen4exp_r1_services(void) { return &g_services; }

extern "C" void tk_qwen4exp_r1_set_rust_hooks(const tk_qwen4exp_r1_rust_hooks_t* hooks) {
    g_hooks.store(hooks, std::memory_order_release);
}

extern "C" const tk_qwen4exp_r1_rust_hooks_t* tk_qwen4exp_r1_rust_hooks(void) {
    return g_hooks.load(std::memory_order_acquire);
}

namespace qwen4exp_r1 {

const tk_qwen4exp_r1_rust_hooks_t* hooks() {
    const char* v = std::getenv("TALUS_FN_R1");
    if (v != nullptr && v[0] == '0') return nullptr;
    const tk_qwen4exp_r1_rust_hooks_t* h = g_hooks.load(std::memory_order_acquire);
    if (h == nullptr || h->struct_size < sizeof(tk_qwen4exp_r1_rust_hooks_t)) return nullptr;
    return h;
}

bool adopt_source(strata::core::FileExpertSource& src, const std::string& pack_dir, const std::string& gguf) {
    const tk_qwen4exp_r1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->source_create == nullptr) return false;
    const tk_qwen4exp_r1_source_vt_t* vt = nullptr;
    char why[512] = {0};
    void* handle = h->source_create(pack_dir.c_str(), gguf.c_str(), &vt, why, sizeof why);
    if (handle == nullptr || vt == nullptr || vt->struct_size < sizeof(tk_qwen4exp_r1_source_vt_t)) {
        std::fprintf(stderr, "talus qwen4exp-r1: the expert source stays C++ (%s)\n", why[0] ? why : "no Rust source");
        return false;
    }
    src.adopt_rust(vt, handle);
    std::fprintf(stderr, "talus qwen4exp-r1: the expert source is Rust's (the GGUF shards in place)\n");
    return true;
}

bool adopt_ple(strata::kernels::PleTable& table) {
    const tk_qwen4exp_r1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->ple_create == nullptr) return false;
    const tk_qwen4exp_r1_ple_vt_t* vt = nullptr;
    char why[512] = {0};
    void* handle = h->ple_create(&vt, why, sizeof why);
    if (handle == nullptr || vt == nullptr || vt->struct_size < sizeof(tk_qwen4exp_r1_ple_vt_t)) {
        std::fprintf(stderr, "talus qwen4exp-r1: the PLE table reader stays C++ (%s)\n", why[0] ? why : "no Rust reader");
        return false;
    }
    table.adopt_rust(vt, handle);
    std::fprintf(stderr, "talus qwen4exp-r1: the PLE table reader is Rust's\n");
    return true;
}

void release_rust_state() {
    const tk_qwen4exp_r1_rust_hooks_t* h = hooks();
    if (h != nullptr && h->release_all != nullptr) h->release_all();
}

}  // namespace qwen4exp_r1
