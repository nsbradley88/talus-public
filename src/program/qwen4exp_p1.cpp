// src/program/qwen4exp_p1.cpp - qwen4exp-p1 (Piece 1): the Rust hook registry and the thin adapters generate.cpp calls.
#include "qwen4exp_p1.hpp"
#include "qwen4exp_r1.hpp"   // qwen4exp-r1: Rust keeps what it allocated

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/model_desc.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/aux_cpus.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace {
std::atomic<const tk_qwen4exp_p1_rust_hooks_t*> g_hooks{nullptr};

bool p1_disabled() {
    const char* v = std::getenv("TALUS_FN_P1");
    return v != nullptr && v[0] == '0';
}

struct SkipList {
    std::vector<const char*> ptrs;
    explicit SkipList(const std::set<std::string>* skip) {
        if (skip != nullptr) {
            ptrs.reserve(skip->size());
            for (const auto& s : *skip) ptrs.push_back(s.c_str());
        }
    }
};

uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const auto* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) b[i];
        h *= 0x00000100000001b3ULL;
    }
    return h;
}
}  // namespace

// ---- the kernel-library services the Rust loaders use (launch-free except the resident hc projections' Q8_0)
namespace {
int svc_mmvq_supported(int t) { return strata::kernels::native_mmvq_supported(t) ? 1 : 0; }
uint64_t svc_mmvq_weight_bytes(int t, int n_in, int n_out) {
    try { return (uint64_t) strata::kernels::native_mmvq_weight_bytes(t, n_in, n_out); } catch (...) { return 0; }
}
uint64_t svc_q8_1_bytes(int n_in, int ncols) { return (uint64_t) strata::kernels::native_q8_1_bytes(n_in, ncols); }
int svc_check_model(const char* const* paths, size_t n, int allow_embd, char* err, size_t cap) {
    try {
        std::vector<std::string> v;
        for (size_t i = 0; i < n; ++i) v.emplace_back(paths[i]);
        const strata::GgufModel model(v);
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        const std::string e = allow_embd && arch != nullptr && arch->s == "strata-embd"
                                  ? std::string() : strata::check_engine_architecture(model.meta());
        if (e.empty()) return 0;
        std::snprintf(err, cap, "%s", e.c_str());
    } catch (const std::exception& ex) {
        std::snprintf(err, cap, "%s", ex.what());
    }
    return -1;
}
int svc_check_shard_arch(const char* path, char* err, size_t cap) {
    try {
        strata::GgufFile g(path);
        const std::string e = strata::check_engine_architecture(g);
        if (e.empty()) return 0;
        std::snprintf(err, cap, "%s", e.c_str());
    } catch (const std::exception& ex) {
        std::snprintf(err, cap, "native dense: %s", ex.what());
    }
    return -1;
}
int svc_q8_packed_enabled() { return strata::kernels::native_q8_0_packed_enabled() ? 1 : 0; }
int svc_q8_packed_eligible(int n_in, int n_out) { return strata::kernels::native_q8_0_packed_eligible(n_in, n_out) ? 1 : 0; }
void svc_q8_pack_host(const void* src, void* dst, int n_in, int n_out) {
    strata::kernels::native_q8_0_pack_host(src, dst, n_in, n_out);
}
void svc_q8_packed_register(const void* w, const void* packed, int n_in, int n_out) {
    strata::kernels::native_q8_0_packed_register(w, packed, n_in, n_out);
}
int svc_hc_q8_requested() { return strata::core::NativeDense::hc_q8_requested() ? 1 : 0; }
void svc_fused_gr_quantize_q8_0(const uint16_t* bf16, uint8_t* out, int64_t nb, void* stream) {
    strata::kernels::fused_gr_quantize_q8_0(bf16, out, nb, stream);
}
int svc_hc_scratch_install(uint64_t elems, void** out4, char* err, size_t cap) {
    std::string e;
    const int rc = strata::core::NativeDense::install_hc_scratch(elems, out4, e);
    if (rc < 0) std::snprintf(err, cap, "%s", e.c_str());
    return rc;
}
int svc_embed_type_supported(int t) { return strata::kernels::embed_type_supported(t) ? 1 : 0; }
uint64_t svc_iq_row_bytes(int t, int64_t n) { return (uint64_t) strata::kernels::iq_row_bytes(t, n); }
int svc_q6_packed_enabled() { return strata::kernels::native_q6_k_packed_enabled() ? 1 : 0; }
int svc_q6_pack(const void* w, int n_in, int n_out, const char* what) {
    return strata::kernels::native_q6_k_pack(w, n_in, n_out, what) ? 1 : 0;
}
const tk_qwen4exp_p1_services_t g_services = {
    svc_mmvq_supported, svc_mmvq_weight_bytes, svc_q8_1_bytes, svc_check_model, svc_check_shard_arch,
    svc_q8_packed_enabled, svc_q8_packed_eligible, svc_q8_pack_host, svc_q8_packed_register, svc_hc_q8_requested,
    svc_fused_gr_quantize_q8_0, svc_hc_scratch_install, svc_embed_type_supported, svc_iq_row_bytes,
    svc_q6_packed_enabled, svc_q6_pack,
};

struct ShardList {
    std::vector<const char*> ptrs;
    explicit ShardList(const std::vector<std::string>& v) {
        for (const auto& s : v) ptrs.push_back(s.c_str());
    }
};
}  // namespace

extern "C" const tk_qwen4exp_p1_services_t* tk_qwen4exp_p1_services(void) { return &g_services; }

extern "C" void tk_qwen4exp_p1_set_rust_hooks(const tk_qwen4exp_p1_rust_hooks_t* hooks) {
    g_hooks.store(hooks, std::memory_order_release);
}

extern "C" const tk_qwen4exp_p1_rust_hooks_t* tk_qwen4exp_p1_rust_hooks(void) {
    return g_hooks.load(std::memory_order_acquire);
}

namespace qwen4exp_p1 {

const tk_qwen4exp_p1_rust_hooks_t* hooks() {
    if (p1_disabled()) return nullptr;
    return g_hooks.load(std::memory_order_acquire);
}

bool pack_pool_bytes(const std::string& pack_dir, const std::set<std::string>* skip, uint64_t& out, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->pack_pool_bytes == nullptr) { err = "qwen4exp-p1: no pack_pool_bytes hook"; return false; }
    SkipList sl(skip);
    char ebuf[512] = {0};
    if (h->pack_pool_bytes(h->user_data, pack_dir.c_str(), skip ? sl.ptrs.data() : nullptr, sl.ptrs.size(), &out, ebuf,
                           sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: pack_pool_bytes failed";
        return false;
    }
    return true;
}

bool pack_load(const std::string& pack_dir, void* arena, uint64_t arena_bytes, const std::set<std::string>* skip,
               strata::core::WeightTable& wt, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->pack_load == nullptr) { err = "qwen4exp-p1: no pack_load hook"; return false; }
    SkipList sl(skip);
    char ebuf[512] = {0};
    const tk_qwen4exp_p1_weight_row_t* rows = nullptr;
    size_t n_rows = 0;
    tk_qwen4exp_p1_load_report_t rep{};
    if (h->pack_load(h->user_data, pack_dir.c_str(), arena, arena_bytes, skip ? sl.ptrs.data() : nullptr, sl.ptrs.size(),
                     &rows, &n_rows, &rep, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: pack_load failed";
        return false;
    }
    std::map<std::string, strata::core::WeightRef> table;
    for (size_t i = 0; i < n_rows; ++i) {
        const tk_qwen4exp_p1_weight_row_t& r = rows[i];
        strata::core::WeightRef w;
        w.data = r.data;
        w.bytes = r.bytes;
        w.ne0 = r.ne0;
        w.ne1 = r.ne1;
        w.elements = r.elements;
        w.kind = (strata::core::WeightKind) r.kind;
        w.code_bits = r.code_bits;
        w.code_bias = r.code_bias;
        w.group_elems = r.group_elems;
        w.codebook_iq4nl = r.codebook_iq4nl != 0;
        w.has_offset = r.has_offset != 0;
        w.codes_bytes = r.codes_bytes;
        w.scales_bytes = r.scales_bytes;
        w.offset_bytes = r.offset_bytes;
        w.src_off = r.src_off;
        w.src_bytes = r.src_bytes;
        w.file_id = r.file_id;
        w.scales_fp16 = r.scales_fp16 != 0;
        w.act_kind = r.act_kind;
        w.resident = r.resident != 0;
        table[r.name] = w;
    }
    strata::core::LoadReport lr;
    lr.tensors = (size_t) rep.tensors;
    lr.arena_bytes = rep.arena_bytes;
    lr.re_rounded = (size_t) rep.re_rounded;
    lr.bytes_saved = rep.bytes_saved;
    lr.read_ms = rep.read_ms;
    lr.upload_ms = rep.upload_ms;
    wt.adopt(pack_dir, std::move(table), lr);
    std::fprintf(stderr, "talus qwen4exp-p1: pack loaded by Rust: %zu tensors, %llu MiB arena (read %.0f ms, upload %.0f ms)\n",
                 n_rows, (unsigned long long) (rep.arena_bytes >> 20), rep.read_ms, rep.upload_ms);
    return true;
}

bool read_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                  std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->profile_read == nullptr) { err = "qwen4exp-p1: no profile_read hook"; return false; }
    char ebuf[512] = {0};
    const int32_t* pairs = nullptr;
    size_t n = 0;
    if (h->profile_read(h->user_data, path.c_str(), n_layers, n_expert, &pairs, &n, &slots, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: profile_read failed";
        return false;
    }
    ranked.resize(n);
    for (size_t i = 0; i < n; ++i) ranked[i] = {pairs[2 * i], pairs[2 * i + 1]};
    return true;
}

bool cache_plan(const tk_qwen4exp_p1_cache_plan_in_t& in, tk_qwen4exp_p1_cache_plan_out_t& out,
                std::vector<int64_t>& sized) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->cache_plan == nullptr) return false;
    if (h->cache_plan(h->user_data, &in, &out) != 0) return false;
    sized.assign(out.sized_slots, out.sized_slots + out.n_sized);
    return true;
}

namespace {
struct FillCtx {
    strata::core::ExpertSource* src;
    strata::core::FileExpertSource* file;
};
const uint8_t* blob_of(void* ctx, int32_t layer, int32_t expert) {
    return static_cast<FillCtx*>(ctx)->src->blob_stable(layer, expert);
}
int advise_pairs_of(void* ctx, const int32_t* pairs, int64_t n) {
    return static_cast<FillCtx*>(ctx)->file->advise_pairs(
               reinterpret_cast<const std::pair<int32_t, int32_t>*>(pairs), n) ? 1 : 0;
}
void prefetch_pairs_of(void* ctx, const int32_t* pairs, int64_t n) {
    static_cast<FillCtx*>(ctx)->file->prefetch_pairs(reinterpret_cast<const std::pair<int32_t, int32_t>*>(pairs), n);
}
void pin_aux_of(void*) { strata::aux_cpus::pin_current_thread(); }
}  // namespace

bool cache_build(tk_qwen4exp_p1_cache_build_in_t in, strata::core::ExpertSource* src, strata::core::FileExpertSource* file,
                 strata::core::ExpertCache& xc, int& expert_cache, std::vector<int64_t>& sized, int64_t& prefilled,
                 std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->cache_build == nullptr) { err = "qwen4exp-p1: no cache_build hook"; return false; }
    FillCtx fc{src, file};
    in.blob_fn = src != nullptr ? &blob_of : nullptr;
    in.blob_ctx = &fc;
    in.file_source = file != nullptr ? 1 : 0;
    in.unbuffered = file != nullptr && file->unbuffered() ? 1 : 0;
    in.advise_pairs = &advise_pairs_of;
    in.prefetch_pairs = &prefetch_pairs_of;
    in.pin_aux_thread = &pin_aux_of;
    tk_qwen4exp_p1_cache_build_out_t out{};
    char ebuf[512] = {0};
    if (h->cache_build(h->user_data, &in, &out, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: cache_build failed";
        return false;
    }
    expert_cache = out.expert_cache;
    prefilled = out.prefilled;
    if (out.base == nullptr) {
        xc.close();
        sized.clear();
        return true;
    }
    if (!xc.adopt(out.base, out.slots, in.n_layers, in.n_expert, out.blob,
                  out.offsets ? std::vector<uint64_t>(out.offsets, out.offsets + out.n_offsets) : std::vector<uint64_t>(),
                  std::vector<int32_t>(out.residency, out.residency + out.n_residency),
                  std::vector<int32_t>(out.layer_next, out.layer_next + out.n_layer_next), out.next_free, out.admitted,
                  out.fills, in.per_layer != 0, err))
        return false;
    if (qwen4exp_r1::hooks() != nullptr) xc.r1_borrow();   // qwen4exp-r1
    (void) sized;   // (nothing reads the slot list after the build; an auto cache's shrink is in out.slots)
    return true;
}

bool res_create(const std::vector<int32_t>& host_res, int32_t*& d_res, int32_t*& d_hit_count, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->res_create == nullptr) { err = "qwen4exp-p1: no res_create hook"; return false; }
    char ebuf[512] = {0};
    if (h->res_create(h->user_data, host_res.data(), host_res.size(), &d_res, &d_hit_count, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "the device residency table could not be staged";
        return false;
    }
    return true;
}

namespace {
uint64_t cb_bytes_needed(void* u, int64_t c) { return static_cast<LendCallbacks*>(u)->bytes_needed(c); }
int64_t cb_sp_chunk(void* u) { return static_cast<LendCallbacks*>(u)->sp_chunk(); }
int cb_relayout(void* u, int64_t c, uint8_t* base, uint64_t bytes, char* err, size_t cap) {
    std::string e;
    if (static_cast<LendCallbacks*>(u)->relayout(c, base, bytes, e)) return 0;
    if (err != nullptr && cap > 0) std::snprintf(err, cap, "%s", e.c_str());
    return -1;
}
const uint8_t* cb_blob(void* u, int32_t l, int32_t e) { return static_cast<LendCallbacks*>(u)->blob(l, e); }
void cb_note_async_read(void* u, const uint8_t* b) {
    auto* cb = static_cast<LendCallbacks*>(u);
    if (cb->note_async_read) cb->note_async_read(b);
}
}  // namespace

bool lend_init(tk_qwen4exp_p1_lend_ctx_t ctx, LendCallbacks* cb, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->lend_init == nullptr) { err = "qwen4exp-p1: no lend_init hook"; return false; }
    ctx.user = cb;
    ctx.bytes_needed = &cb_bytes_needed;
    ctx.sp_chunk = &cb_sp_chunk;
    ctx.relayout = &cb_relayout;
    ctx.blob = &cb_blob;
    ctx.note_async_read = &cb_note_async_read;
    char ebuf[512] = {0};
    if (h->lend_init(h->user_data, &ctx, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: lend_init failed";
        return false;
    }
    return true;
}

bool lend(int64_t tokens, int32_t& lent_n, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->lend == nullptr) { err = "qwen4exp-p1: no lend hook"; return false; }
    char ebuf[512] = {0};
    if (h->lend(h->user_data, tokens, &lent_n, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: lend failed";
        return false;
    }
    return true;
}

int64_t gen_lend(tk_qwen4exp_p1_lend_ctx_t ctx, LendCallbacks* cb, int32_t first, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->gen_lend == nullptr) { err = "qwen4exp-p1: no gen_lend hook"; return -1; }
    ctx.user = cb;
    ctx.blob = &cb_blob;
    ctx.note_async_read = &cb_note_async_read;
    char ebuf[512] = {0};
    const int64_t k = h->gen_lend(h->user_data, &ctx, first, ebuf, sizeof ebuf);
    if (k < 0) err = ebuf[0] ? ebuf : "qwen4exp-p1: gen_lend failed";
    return k;
}

int64_t gen_refill(std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->gen_refill == nullptr) { err = "qwen4exp-p1: no gen_refill hook"; return -1; }
    char ebuf[512] = {0};
    const int64_t n = h->gen_refill(h->user_data, ebuf, sizeof ebuf);
    if (n < 0) err = ebuf[0] ? ebuf : "refilling the lent slots failed";
    return n;
}

bool refill(std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->refill == nullptr) { err = "qwen4exp-p1: no refill hook"; return false; }
    char ebuf[512] = {0};
    if (h->refill(h->user_data, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: refill failed";
        return false;
    }
    return true;
}

bool native_served_names(const std::vector<std::string>& shards, bool include_ple_key, std::set<std::string>& out,
                         std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->native_served_names == nullptr) { err = "qwen4exp-p1: no native_served_names hook"; return false; }
    ShardList sl(shards);
    const char* const* names = nullptr;
    size_t n = 0;
    char ebuf[512] = {0};
    if (h->native_served_names(h->user_data, sl.ptrs.data(), sl.ptrs.size(), include_ple_key ? 1 : 0, &names, &n, ebuf,
                               sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "native dense: served_names failed";
        return false;
    }
    for (size_t i = 0; i < n; ++i) out.insert(names[i]);
    return true;
}

bool native_dense_load(const std::vector<std::string>& shards, strata::core::WeightTable& wt,
                       strata::core::NativeDense& nd, bool include_ple_key, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->native_dense_load == nullptr) { err = "qwen4exp-p1: no native_dense_load hook"; return false; }
    ShardList sl(shards);
    tk_qwen4exp_p1_native_dense_out_t out{};
    char ebuf[512] = {0};
    if (h->native_dense_load(h->user_data, sl.ptrs.data(), sl.ptrs.size(), include_ple_key ? 1 : 0, &out, ebuf,
                             sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "native dense: load failed";
        return false;
    }
    for (size_t i = 0; i < out.n_attach; ++i) {
        const tk_qwen4exp_p1_native_attach_t& a = out.attach[i];
        strata::core::WeightRef* r = const_cast<strata::core::WeightRef*>(wt.find(a.name));
        if (r == nullptr) { err = std::string("native dense: tensor absent from canonical table: ") + a.name; return false; }
        if (a.native_data != nullptr) {
            r->native_data = a.native_data;
            r->native_type = a.native_type;
            r->native_q8_1 = a.native_q8_1;
        }
        if (a.hc_q8 != nullptr) r->hc_q8 = a.hc_q8;
    }
    nd.adopt(std::vector<void*>(out.weights, out.weights + out.n_weights),
             std::vector<const void*>(out.packed_keys, out.packed_keys + out.n_packed), out.scratch, out.bytes,
             out.allocated);
    if (qwen4exp_r1::hooks() != nullptr) nd.r1_borrow();   // qwen4exp-r1
    return true;
}

bool native_head_load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out,
                      strata::core::NativeHead& head, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->native_head_load == nullptr) { err = "qwen4exp-p1: no native_head_load hook"; return false; }
    ShardList sl(shards);
    tk_qwen4exp_p1_native_head_out_t out{};
    char ebuf[512] = {0};
    if (h->native_head_load(h->user_data, sl.ptrs.data(), sl.ptrs.size(), n_in, n_out, &out, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "native head: load failed";
        return false;
    }
    head.adopt(out.weights, out.scratch, out.bytes, out.n_in, out.n_out, out.type, out.device);
    if (qwen4exp_r1::hooks() != nullptr) head.r1_borrow();   // qwen4exp-r1
    return true;
}

bool native_embed_load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab,
                       strata::core::NativeEmbed& embed, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->native_embed_load == nullptr) { err = "qwen4exp-p1: no native_embed_load hook"; return false; }
    ShardList sl(shards);
    tk_qwen4exp_p1_native_embed_out_t out{};
    char ebuf[512] = {0};
    if (h->native_embed_load(h->user_data, sl.ptrs.data(), sl.ptrs.size(), n_embd, n_vocab, &out, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "native embedding: load failed";
        return false;
    }
    embed.adopt(out.host, out.dev, out.bytes, (size_t) out.row, out.n_embd, out.n_vocab, out.type);
    if (qwen4exp_r1::hooks() != nullptr) embed.r1_borrow();   // qwen4exp-r1
    return true;
}

namespace {
bool hash_dev(const void* p, uint64_t n, std::vector<uint8_t>& buf, uint64_t& h) {
    if (p == nullptr || n == 0) return true;
    buf.resize((size_t) n);
    const bool ok = cudaMemcpy(buf.data(), p, (size_t) n, cudaMemcpyDeviceToHost) == cudaSuccess;
    h = fnv(buf.data(), buf.size(), h);
    return ok;
}
}  // namespace

void print_native_hash(const strata::core::WeightTable& wt, const strata::core::NativeHead* head,
                       const strata::core::NativeEmbed* embed, const char* what) {
    const char* v = std::getenv("TALUS_FN_P1_HASH");
    if (v == nullptr || v[0] != '1') return;
    uint64_t h = 0xcbf29ce484222325ULL, total = 0;
    size_t n_native = 0, n_hc = 0;
    std::vector<uint8_t> buf;
    bool ok = true;
    for (const auto& [name, r] : wt.all()) {
        if (r.native_data != nullptr) {
            const uint64_t nb = (uint64_t) strata::kernels::native_mmvq_weight_bytes(r.native_type, (int) r.ne0, (int) r.ne1);
            h = fnv(name.data(), name.size(), h);
            h = fnv(&r.native_type, sizeof r.native_type, h);
            ok = hash_dev(r.native_data, nb, buf, h) && ok;
            total += nb;
            ++n_native;
        }
        if (r.hc_q8 != nullptr) {
            const uint64_t nb = (uint64_t) r.elements / 32 * 34;
            h = fnv(name.data(), name.size(), h);
            ok = hash_dev(r.hc_q8, nb, buf, h) && ok;
            total += nb;
            ++n_hc;
        }
    }
    if (head != nullptr && head->loaded()) {
        const int t = head->type();
        h = fnv(&t, sizeof t, h);
        ok = hash_dev(head->weights(), head->weight_bytes(), buf, h) && ok;
        total += head->weight_bytes();
    }
    if (embed != nullptr && embed->bytes() > 0) {
        const int t = embed->type();
        h = fnv(&t, sizeof t, h);
        ok = hash_dev(embed->device_table(), embed->bytes(), buf, h) && ok;
        total += embed->bytes();
    }
    std::fprintf(stderr, "p1: native %s fnv=%016llx projections=%zu hc_q8=%zu bytes=%llu%s\n", what, (unsigned long long) h,
                 n_native, n_hc, (unsigned long long) total, ok ? "" : " (A READ-BACK FAILED)");
}

bool ple_wire(bool native_ple_key, strata::kernels::PleWeights& w, std::string& err) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->ple_wire == nullptr) { err = "qwen4exp-p1: no ple_wire hook"; return false; }
    tk_qwen4exp_p1_ple_weights_t p{};
    char ebuf[512] = {0};
    if (h->ple_wire(h->user_data, native_ple_key ? 1 : 0, &p, ebuf, sizeof ebuf) != 0) {
        err = ebuf[0] ? ebuf : "qwen4exp-p1: ple_wire failed";
        return false;
    }
    w.key_codes = p.key_codes;
    w.key_scales = p.key_scales;
    w.value_bf16 = p.value_bf16;
    w.norm_key = p.norm_key;
    w.norm_query = p.norm_query;
    w.norm_conv = p.norm_conv;
    w.conv1d_f16 = p.conv1d_f16;
    w.key_native_data = p.key_native_data;
    w.key_native_type = p.key_native_type;
    w.key_native_q8_1 = p.key_native_q8_1;
    w.key_bf16 = p.key_bf16;
    return true;
}

namespace {
uint64_t pc_bytes_needed(void* u, int64_t c) { return static_cast<PlannerCallbacks*>(u)->bytes_needed(c); }
uint64_t pc_bytes_no_ring(void* u, int64_t c) { return static_cast<PlannerCallbacks*>(u)->bytes_needed_no_ring(c); }
void pc_set_ring(void* u, int64_t room, int64_t small) { static_cast<PlannerCallbacks*>(u)->set_ring_budget(room, small); }
int64_t pc_ring_cap(void* u, int64_t c) { return static_cast<PlannerCallbacks*>(u)->ring_cap_for(c); }
int64_t pc_ring_default(void* u) { return static_cast<PlannerCallbacks*>(u)->ring_default_slots(); }
int pc_ring_bytes(void* u) { return static_cast<PlannerCallbacks*>(u)->ring_bytes_enabled() ? 1 : 0; }
}  // namespace

void bind_planner(tk_qwen4exp_p1_planner_t& p, PlannerCallbacks* cb) {
    p.user = cb;
    p.bytes_needed = &pc_bytes_needed;
    p.bytes_needed_no_ring = &pc_bytes_no_ring;
    p.set_ring_budget = &pc_set_ring;
    p.ring_cap_for = &pc_ring_cap;
    p.ring_default_slots = &pc_ring_default;
    p.ring_bytes_enabled = &pc_ring_bytes;
}

int64_t plan_lend(const tk_qwen4exp_p1_planner_t& p, int64_t& chunk) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->plan_lend == nullptr) return -1;
    return h->plan_lend(h->user_data, &p, &chunk);
}

int64_t lend_slots(const tk_qwen4exp_p1_planner_t& p, int64_t chunk) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->lend_slots == nullptr) return -1;
    return h->lend_slots(h->user_data, &p, chunk);
}

bool serve_scan(const tk_qwen4exp_p1_planner_t& p, int64_t& chunk, int64_t& first) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->serve_scan == nullptr) return false;
    return h->serve_scan(h->user_data, &p, &chunk, &first) == 0;
}

bool ple_io_plan(const std::string& ple_io, int64_t inflight, int64_t row_cache, bool sync_submit,
                 strata::kernels::PleIoOptions& pio) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->ple_io_plan == nullptr) return false;
    tk_qwen4exp_p1_ple_io_t p{};
    if (h->ple_io_plan(h->user_data, ple_io.c_str(), inflight, row_cache, sync_submit ? 1 : 0, &p) != 0) return false;
    pio.mode = p.mode == 1 ? strata::kernels::PleIo::Mmap : strata::kernels::PleIo::Direct;
    pio.lock = p.lock != 0;
    pio.max_inflight = p.max_inflight;
    pio.cache_rows = p.cache_rows;
    pio.io_thread = p.io_thread != 0;
    pio.keepalive_ms = p.keepalive_ms;
    pio.keepalive_window_s = p.keepalive_window_s;
#if !defined(_WIN32)
    if (p.set_io_threads) setenv("STRATA_IO_THREADS", "64", 0);
#endif
    return true;
}

int ple_probe_decide(double rows_per_s, bool ple_io_explicit) {
    const tk_qwen4exp_p1_rust_hooks_t* h = hooks();
    if (h == nullptr || h->ple_probe_decide == nullptr) return -1;
    return h->ple_probe_decide(h->user_data, rows_per_s, ple_io_explicit ? 1 : 0);
}

void dump_slot_hashes(const strata::core::ExpertCache& xc, int64_t n_layers, int64_t n_expert,
                      const std::vector<uint64_t>& blob_bytes) {
    const char* path = std::getenv("TALUS_FN_P1_SLOTS");
    if (path == nullptr || path[0] == 0) return;
    std::FILE* f = std::fopen(path, "w");
    if (f == nullptr) return;
    std::vector<uint8_t> buf;
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e) {
            const int32_t s = xc.slot_of(l, e);
            if (s < 0) continue;
            buf.resize((size_t) blob_bytes[(size_t) l]);
            const bool ok = cudaMemcpy(buf.data(), xc.device_slot(s), buf.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
            std::fprintf(f, "%d %lld %lld %zu %016llx%s\n", s, (long long) l, (long long) e, buf.size(),
                         (unsigned long long) fnv(buf.data(), buf.size(), 0xcbf29ce484222325ULL), ok ? "" : " ERR");
        }
    std::fclose(f);
}

void print_table_hash(const strata::core::WeightTable& wt, const void* arena) {
    const char* v = std::getenv("TALUS_FN_P1_HASH");
    if (v == nullptr || v[0] != '1') return;
    uint64_t h = 0xcbf29ce484222325ULL;
    uint64_t total = 0;
    std::vector<uint8_t> buf;
    bool ok = true;
    for (const auto& [name, r] : wt.all()) {
        h = fnv(name.data(), name.size(), h);
        const int64_t off = r.data ? (int64_t) ((const uint8_t*) r.data - (const uint8_t*) arena) : -1;
        const int64_t meta[] = {off, (int64_t) r.bytes, r.ne0, r.ne1, r.elements, (int64_t) r.kind, r.code_bits,
                                r.code_bias, r.group_elems, r.codebook_iq4nl, r.has_offset, (int64_t) r.codes_bytes,
                                (int64_t) r.scales_bytes, (int64_t) r.offset_bytes, (int64_t) r.src_off,
                                (int64_t) r.src_bytes, r.file_id, r.scales_fp16, r.act_kind, r.resident};
        h = fnv(meta, sizeof meta, h);
        if (r.data != nullptr && r.bytes > 0) {
            buf.resize((size_t) r.bytes);
            if (cudaMemcpy(buf.data(), r.data, (size_t) r.bytes, cudaMemcpyDeviceToHost) != cudaSuccess) ok = false;
            h = fnv(buf.data(), buf.size(), h);
            total += r.bytes;
        }
    }
    std::fprintf(stderr, "p1: weight table fnv=%016llx tensors=%zu bytes=%llu%s\n", (unsigned long long) h,
                 wt.all().size(), (unsigned long long) total, ok ? "" : " (A READ-BACK FAILED)");
}

}  // namespace qwen4exp_p1
