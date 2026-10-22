// src/program/qwen4exp_p1.hpp - qwen4exp-p1 (Piece 1) C++ side of the Rust hook table: the narrow calls generate.cpp makes
// instead of its own Piece 1 code when the Rust data plane is on. Everything here is a thin adapter; the logic is Rust's
// (rust/crates/talus-model/src/qwen4exp/weights.rs).
#pragma once

#include "talus_qwen4exp_kernels.h"
#include "strata/core/weights.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/ngram.hpp"

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace qwen4exp_p1 {

/// The registered hook table, or null (not registered, or TALUS_FN_P1=0).
const tk_qwen4exp_p1_rust_hooks_t* hooks();

/// WeightTable::pool_bytes through Rust. False with `err` when the hook is absent or fails.
bool pack_pool_bytes(const std::string& pack_dir, const std::set<std::string>* skip, uint64_t& out, std::string& err);

/// WeightTable::load through Rust: the arena is filled by Rust and `wt` adopts the rows.
bool pack_load(const std::string& pack_dir, void* arena, uint64_t arena_bytes, const std::set<std::string>* skip,
               strata::core::WeightTable& wt, std::string& err);

/// read_expert_profile through Rust.
bool read_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                  std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

/// The cache sizing through Rust: fills `out` (sized_slots copied into `sized`). False when the hook is absent.
bool cache_plan(const tk_qwen4exp_p1_cache_plan_in_t& in, tk_qwen4exp_p1_cache_plan_out_t& out,
                std::vector<int64_t>& sized);

/// The cache open + profile fill through Rust; `xc` adopts the result. `src` is the expert source the blobs come from
/// (null: no fill); `file` is the mapped file tier when `src` is it (its read-ahead runs during the fill). On success
/// `expert_cache` / `sized` / `prefilled` are what the build ended with.
bool cache_build(tk_qwen4exp_p1_cache_build_in_t in, strata::core::ExpertSource* src, strata::core::FileExpertSource* file,
                 strata::core::ExpertCache& xc, int& expert_cache, std::vector<int64_t>& sized, int64_t& prefilled,
                 std::string& err);

/// d_res and the hit counter through Rust (staged from `host_res`).
bool res_create(const std::vector<int32_t>& host_res, int32_t*& d_res, int32_t*& d_hit_count, std::string& err);

/// The serve loan's C++ callbacks (the prompt path and the expert source stay C++); must outlive the serve loop.
struct LendCallbacks {
    std::function<uint64_t(int64_t)> bytes_needed;
    std::function<int64_t()> sp_chunk;
    std::function<bool(int64_t, uint8_t*, uint64_t, std::string&)> relayout;
    std::function<const uint8_t*(int32_t, int32_t)> blob;
    std::function<void(const uint8_t*)> note_async_read;
};
/// Hands the loan to Rust: `ctx` without the callback fields (filled here from `cb`).
bool lend_init(tk_qwen4exp_p1_lend_ctx_t ctx, LendCallbacks* cb, std::string& err);
bool lend(int64_t tokens, int32_t& lent_n, std::string& err);
/// generate's one-shot loan through Rust: the slots from `first` out of the table (count, -1 on error), and back.
int64_t gen_lend(tk_qwen4exp_p1_lend_ctx_t ctx, LendCallbacks* cb, int32_t first, std::string& err);
int64_t gen_refill(std::string& err);
bool refill(std::string& err);

/// The native GGUF tensors through Rust: the served names (the arena's skip set), the projections (attached to
/// `wt`'s rows, buffers adopted by `nd`), the output head, the token embedding.
bool native_served_names(const std::vector<std::string>& shards, bool include_ple_key, std::set<std::string>& out,
                         std::string& err);
bool native_dense_load(const std::vector<std::string>& shards, strata::core::WeightTable& wt,
                       strata::core::NativeDense& nd, bool include_ple_key, std::string& err);
bool native_head_load(const std::vector<std::string>& shards, int64_t n_in, int64_t n_out,
                      strata::core::NativeHead& head, std::string& err);
bool native_embed_load(const std::vector<std::string>& shards, int64_t n_embd, int64_t n_vocab,
                       strata::core::NativeEmbed& embed, std::string& err);

/// TALUS_FN_P1_HASH=1: one line `p1: weight table fnv=<hex> tensors=<n> bytes=<b>` over every row's metadata (name,
/// shape, kind, plane sizes, arena offset) and device bytes. Same code on both paths, so the C++ and Rust loads compare.
void print_table_hash(const strata::core::WeightTable& wt, const void* arena);

/// The PLE weight fields of `w` through Rust (blk.1.ple_* checked and wired).
bool ple_wire(bool native_ple_key, strata::kernels::PleWeights& w, std::string& err);

/// The prompt path's pricing for the loan planner (Prefill statics, bound to the session it prices).
struct PlannerCallbacks {
    std::function<uint64_t(int64_t)> bytes_needed;
    std::function<uint64_t(int64_t)> bytes_needed_no_ring;
    std::function<void(int64_t, int64_t)> set_ring_budget;
    std::function<int64_t(int64_t)> ring_cap_for;
    std::function<int64_t()> ring_default_slots;
    std::function<bool()> ring_bytes_enabled;
};
/// Fills the callback fields of `p` from `cb` (which must outlive the calls).
void bind_planner(tk_qwen4exp_p1_planner_t& p, PlannerCallbacks* cb);
/// plan_lend / lend_slots / the serve scan through Rust (-1 / false: the hook is absent).
int64_t plan_lend(const tk_qwen4exp_p1_planner_t& p, int64_t& chunk);
int64_t lend_slots(const tk_qwen4exp_p1_planner_t& p, int64_t chunk);
bool serve_scan(const tk_qwen4exp_p1_planner_t& p, int64_t& chunk, int64_t& first);

/// The PLE table's I/O options through Rust (false: no hook) and the probe decision (0 / 1 mmap / 2 warn; -1 no hook).
bool ple_io_plan(const std::string& ple_io, int64_t inflight, int64_t row_cache, bool sync_submit,
                 strata::kernels::PleIoOptions& pio);
int ple_probe_decide(double rows_per_s, bool ple_io_explicit);

/// TALUS_FN_P1_HASH=1: `p1: native <what> fnv=...` over the native projections, the hc Q8_0 copies, the head and the
/// embedding table (device bytes and types), on both paths.
void print_native_hash(const strata::core::WeightTable& wt, const strata::core::NativeHead* head,
                       const strata::core::NativeEmbed* embed, const char* what);

/// TALUS_FN_P1_SLOTS=<path>: one line per filled slot `slot layer expert bytes fnv` (both paths, the same code), to
/// locate a cache difference.
void dump_slot_hashes(const strata::core::ExpertCache& xc, int64_t n_layers, int64_t n_expert,
                      const std::vector<uint64_t>& blob_bytes);

}  // namespace qwen4exp_p1
