#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tk_qwen4exp_rust_driver {
    void* user_data;

    // Stdin line hook for --serve:
    // If block != 0, waits for the next non-STOP line or EOF.
    // If block == 0, polls without blocking (returns 0 if queue empty, 1 if line ready, -1 on EOF).
    // Returns: 1 = line pointer/len stored in *out_ptr / *out_len, 0 = no line (non-blocking), -1 = EOF.
    int (*next_line)(void* user_data, int block, const char** out_ptr, size_t* out_len);

    // Clear the stop flag at the beginning of a new request.
    void (*clear_stop)(void* user_data);

    // Poll whether a STOP command was received for the active request.
    int (*is_stopped)(void* user_data);

    // Parse native protocol logit_bias=id:value,id:value in Rust:
    // Returns 1 on success (filling out_bias[0..vocab]), 0 on error (writing NUL-terminated error to out_err).
    int (*parse_logit_bias)(
        void* user_data,
        const char* text,
        int vocab,
        float* out_bias,
        char* out_err,
        size_t err_cap
    );

    // Prompt cut & checkpoint boundary planner hook in Rust:
    // Computes turn_at, root_at, message_at, and sorted cuts[0..*out_n_cuts] (up to 6 entries).
    // Returns 1 if handled by Rust, 0 to fall back.
    int (*plan_prompt_cuts)(
        void* user_data,
        const int64_t* ids,
        size_t n_ids,
        int64_t resume,
        int64_t read_from,
        int resumed_from0,
        int64_t reread_to,
        int prompt_cache,
        int64_t turn_token,
        int64_t tail_role_token,
        int64_t prompt_cache_root,
        int req_ckpt,
        int multi_gpu,
        int message_checkpoint_env,
        int64_t pin_at,
        int64_t* out_turn_at,
        int64_t* out_root_at,
        int64_t* out_message_at,
        int64_t* out_cuts,
        size_t* out_n_cuts
    );

    // Initialize DraftPolicy + SuffixDrafter once at server / speculative-loop startup:
    void (*init_spec)(
        void* user_data,
        int max_t,
        int suffix_draft,
        int lookup_chain,
        int lookup_chain_min,
        size_t capacity_tokens
    );

    // Reset SuffixDrafter for a new request / prompt and append initial tokens:
    void (*reset_request_spec)(
        void* user_data,
        const int64_t* prompt_ids,
        size_t n_prompt,
        const int64_t* produced_ids,
        size_t n_produced
    );

    // Plan the next speculative decode window in Rust:
    // Computes T, T_mtp, from_sfx, sfx_match, chain_n, chain_match, and fills out_sbuf / out_cbuf.
    int (*plan_decode_window)(
        void* user_data,
        int first_window,
        int use_mtp,
        int initial_t,
        int s_mtp,
        int max_s,
        int room,
        int allow_chain,
        double spec_min_p,
        int chain_fixed,
        const int32_t* drafts,
        const float* dprob,
        int* out_t,
        int* out_t_mtp,
        int* out_from_sfx,
        int* out_sfx_match,
        int* out_chain_n,
        int* out_chain_match,
        int32_t* out_sbuf,
        int32_t* out_cbuf
    );

    // Append an emitted token to the Rust SuffixDrafter:
    void (*append_emitted_token)(void* user_data, int32_t token);

    // Observe the completed speculative decode round in the Rust DraftPolicy:
    void (*observe_decode_round)(
        void* user_data,
        int timed_round,
        int eos,
        int from_sfx,
        int t,
        int t_mtp,
        int chain_n,
        int accepted,
        int sfx_match,
        int chain_match,
        double round_ms
    );

    // Pure-Rust --serve execution loop hook (invoked once weights, caches, prefill, verifier, and MTP are initialized):
    // When non-null (default in pure-Rust mode; null under --cpp-engine / TALUS_FN_CPP=1), Rust owns the entire
    // --serve protocol loop, conversation state orchestration, prefill cut execution, and speculative decode loop.
    int (*run_serve_loop)(void* user_data, struct tk_qwen4exp_serve_ctx* ctx);

    // Pure-Rust single-request generate execution loop hook (invoked in non-serve mode):
    // When non-null (default in pure-Rust mode; null under --cpp-engine / TALUS_FN_CPP=1), Rust owns the entire
    // prefill, token loop, and speculative verify/MTP decode loop.
    int (*run_generate_loop)(void* user_data, struct tk_qwen4exp_gen_ctx* ctx);
} tk_qwen4exp_rust_driver_t;

typedef struct tk_qwen4exp_serve_ctx tk_qwen4exp_serve_ctx_t;
typedef struct tk_qwen4exp_gen_ctx tk_qwen4exp_gen_ctx_t;

typedef struct tk_qwen4exp_serve_info {
    int64_t n_vocab;
    int64_t n_layers;
    int64_t n_expert;
    int64_t max_context;
    int32_t spec;
    int32_t mtp_max_t;
    int32_t use_mtp;
    int32_t suffix_draft;
    int32_t lookup_chain;
    int32_t lookup_chain_min;
    double spec_min_p;
    double pcie_frac;
    int32_t prompt_cache;
    int64_t prompt_cache_every;
    int64_t prompt_cache_root;
    int32_t prompt_cache_tail;
    int64_t turn_token;
    int64_t tail_role_token;
    int64_t short_read;
    int32_t message_checkpoint_env;
    size_t n_eos;
    int64_t eos_ids[16];
} tk_qwen4exp_serve_info_t;

typedef struct tk_qwen4exp_gen_info {
    int64_t n_vocab;
    int64_t n_layers;
    int64_t n_expert;
    int64_t max_context;
    int64_t max_new;
    int64_t prefill_chunk;
    int64_t prefill_until;
    int32_t native_pack;
    int32_t spec;
    int32_t mtp_max_t;
    int32_t use_mtp;
    int32_t suffix_draft;
    int32_t lookup_chain;
    int32_t lookup_chain_min;
    int32_t stop_eos;
    int32_t spec_corrupt;
    double spec_min_p;
    size_t n_prompt;
    const int64_t* prompt_tokens;
    size_t n_eos;
    int64_t eos_ids[16];
} tk_qwen4exp_gen_info_t;

typedef struct tk_qwen4exp_req_sampling {
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    int32_t penalty_last_n;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    uint64_t seed;
    double pcie_frac;
    double spec_min_p;
    int32_t aux_cpus;
} tk_qwen4exp_req_sampling_t;

// Run the Flash-Next (qwen4exp) engine driven by the Rust host callbacks.
int tk_qwen4exp_run_with_rust_driver(
    int argc,
    const char* const* argv,
    const tk_qwen4exp_rust_driver_t* driver
);

// Trigger an immediate asynchronous stop of the active request (callable from any thread).
void tk_qwen4exp_request_stop(void);

// Fine-grained C-ABI primitives for the pure-Rust --serve loop:
void tk_qwen4exp_serve_get_info(
    tk_qwen4exp_serve_ctx_t* ctx,
    tk_qwen4exp_serve_info_t* out_info
);
int tk_qwen4exp_serve_pre_command(
    tk_qwen4exp_serve_ctx_t* ctx,
    const char* line,
    int is_quit
);
void tk_qwen4exp_serve_set_busy(
    tk_qwen4exp_serve_ctx_t* ctx,
    int busy
);
int tk_qwen4exp_serve_begin_request(
    tk_qwen4exp_serve_ctx_t* ctx,
    const int64_t* ids,
    size_t n_ids,
    int req_cache,
    int req_ckpt,
    int64_t req_pin,
    int req_cvec,
    const tk_qwen4exp_req_sampling_t* sampling,
    const float* bias,
    size_t n_bias,
    int64_t* out_resume,
    int64_t* out_read_from,
    int* out_resumed_from0,
    int64_t* out_reread_to,
    int64_t* out_pin_at
);
int tk_qwen4exp_serve_prefill_segment(
    tk_qwen4exp_serve_ctx_t* ctx,
    int64_t at,
    int64_t to,
    int use_windows,
    int* out_nf_hit,
    int* out_cancelled
);
int tk_qwen4exp_serve_checkpoint_at(
    tk_qwen4exp_serve_ctx_t* ctx,
    int64_t pos,
    int as_tail,
    int is_pin,
    int64_t turn_at
);
int tk_qwen4exp_serve_prefill_finish(
    tk_qwen4exp_serve_ctx_t* ctx
);
int tk_qwen4exp_serve_verify_window(
    tk_qwen4exp_serve_ctx_t* ctx,
    int t,
    int t_mtp,
    int from_sfx,
    int64_t pos,
    const int32_t* window,
    const int32_t* consumed,
    size_t n_consumed,
    int32_t* out_tokens
);
int tk_qwen4exp_serve_copy_logits(
    tk_qwen4exp_serve_ctx_t* ctx,
    int row,
    float* out_logits
);
int tk_qwen4exp_serve_commit_window(
    tk_qwen4exp_serve_ctx_t* ctx,
    int keep,
    int64_t rounds,
    int first_window,
    int* out_nf_hit
);
int tk_qwen4exp_serve_draft_window(
    tk_qwen4exp_serve_ctx_t* ctx,
    int t,
    int accepted,
    int64_t pos,
    int need_draft,
    double req_spec_min_p,
    const int32_t* out_tokens,
    const int32_t* consumed,
    size_t n_consumed,
    int32_t* out_drafts,
    float* out_dprob
);
int tk_qwen4exp_serve_finish_request(
    tk_qwen4exp_serve_ctx_t* ctx,
    int cancelled,
    int nf_hit,
    int req_ckpt,
    int64_t produced_n,
    int64_t n_prompt,
    int64_t resume,
    double prompt_ms,
    double decode_ms,
    const char* finish_reason,
    int64_t draft_accepted,
    int64_t draft_offered,
    int64_t sfx_windows,
    int64_t sfx_ok,
    int64_t sfx_drafts,
    int64_t chain_windows,
    int64_t chain_ok,
    int64_t chain_drafts,
    int64_t dec_windows,
    const int32_t* consumed,
    size_t n_consumed
);
void tk_qwen4exp_serve_on_exit(
    tk_qwen4exp_serve_ctx_t* ctx
);

// Fine-grained C-ABI primitives for the pure-Rust single-request generate loop:
void tk_qwen4exp_gen_get_info(
    tk_qwen4exp_gen_ctx_t* ctx,
    tk_qwen4exp_gen_info_t* out_info
);
int tk_qwen4exp_gen_prefill_batched(
    tk_qwen4exp_gen_ctx_t* ctx,
    int64_t* out_pos_start,
    int64_t* out_tok,
    double* out_prefill_ms
);
int tk_qwen4exp_gen_step_token(
    tk_qwen4exp_gen_ctx_t* ctx,
    int64_t pos,
    int64_t tok,
    int32_t* out_next,
    double* out_step_ms,
    double* out_ttft_ms
);
int tk_qwen4exp_gen_spec_init(
    tk_qwen4exp_gen_ctx_t* ctx,
    int64_t spec_pos,
    int32_t x,
    int* out_first_window,
    int64_t* out_max_new,
    int32_t* out_drafts,
    float* out_dprob
);
int tk_qwen4exp_gen_spec_verify(
    tk_qwen4exp_gen_ctx_t* ctx,
    int t,
    int t_mtp,
    int from_sfx,
    int64_t pos,
    const int32_t* window,
    int32_t* out_tokens
);
int tk_qwen4exp_gen_spec_copy_logits(
    tk_qwen4exp_gen_ctx_t* ctx,
    int row,
    float* out_logits
);
int tk_qwen4exp_gen_spec_commit(
    tk_qwen4exp_gen_ctx_t* ctx,
    int keep,
    int64_t rounds
);
int tk_qwen4exp_gen_spec_join_adapt(
    tk_qwen4exp_gen_ctx_t* ctx
);
int tk_qwen4exp_gen_spec_draft(
    tk_qwen4exp_gen_ctx_t* ctx,
    int t,
    int accepted,
    int64_t pos,
    int need_draft,
    const int32_t* out_tokens,
    int32_t* out_drafts,
    float* out_dprob
);
int tk_qwen4exp_gen_spec_finish(
    tk_qwen4exp_gen_ctx_t* ctx,
    int64_t rounds,
    int64_t drafts_ok,
    int64_t drafts_total,
    int64_t sfx_windows,
    int64_t sfx_ok,
    int64_t sfx_drafts,
    int64_t chain_windows,
    int64_t chain_ok,
    int64_t chain_drafts,
    const int64_t* window_hist,
    size_t n_window_hist,
    const int64_t* accepted_hist,
    size_t n_accepted_hist
);
int tk_qwen4exp_gen_finish(
    tk_qwen4exp_gen_ctx_t* ctx,
    const int64_t* produced,
    size_t n_produced,
    double total_ms,
    double prefill_ms,
    double ttft_ms
);

// ============================================================================
// Piece 0–6 C-ABI declarations for the parallel Flash-Next (`qwen4exp`) port
// ============================================================================

typedef struct tk_qwen4exp_weights_info {
    int32_t native_pack;
    int32_t mtp_loaded;
    int32_t ple_ready;
    int32_t cache_slots;
    int64_t resident_experts;
    uint64_t cache_bytes;
    int64_t max_blob_bytes;
} tk_qwen4exp_weights_info_t;

typedef struct tk_qwen4exp_prefill_stats {
    int64_t tokens;
    int64_t chunks;
    double ms_total;
    int64_t experts_streamed;
    int64_t experts_dma;
    int64_t experts_resident;
    int64_t experts_cpu;
    double ms_experts_host;
    double ms_ple;
    double cpu_share;
} tk_qwen4exp_prefill_stats_t;

typedef struct tk_qwen4exp_conv_state_sizes {
    uint64_t gdn_bytes;
    uint64_t ple_bytes;
    uint64_t qsa_tail_bytes;
    uint64_t qsa_dead_bytes;
    uint64_t qsa_block_pos_bytes;
    int64_t qsa_layers;
} tk_qwen4exp_conv_state_sizes_t;

typedef struct tk_qwen4exp_kld_report {
    int64_t positions_scored;
    double mean_kld;
    double median_kld;
    double p90_kld;
    double p99_kld;
    double max_kld;
    double top1_agreement;
    double flip_rate;
} tk_qwen4exp_kld_report_t;

// Piece 0: Per-layer bitwise dump hook (`TALUS_FN_DUMP=<dir>`).
// Copies `n_floats` from device pointer `d_ptr` (or host pointer when `is_device == 0`)
// on `stream` and writes `<TALUS_FN_DUMP>/<tag>.bin` + appends to `<TALUS_FN_DUMP>/manifest.tsv`.
int tk_qwen4exp_dump_tensor(
    const char* tag,
    int32_t layer,
    int64_t pos,
    const float* ptr,
    size_t n_floats,
    int is_device,
    void* stream
);

// Piece 1 (`qwen4exp-p1`): Weight loading, expert arena/cache, resident budget, PLE table.
int tk_qwen4exp_p1_query_weights_info(
    tk_qwen4exp_serve_ctx_t* ctx,
    tk_qwen4exp_weights_info_t* out_info
);
int tk_qwen4exp_p1_expert_cache_info(
    tk_qwen4exp_serve_ctx_t* ctx,
    int32_t* out_host_res,
    size_t cap_entries
);
int tk_qwen4exp_p1_borrow_slots(
    tk_qwen4exp_serve_ctx_t* ctx,
    int64_t want_chunk_tokens,
    int32_t* out_lent_slots
);
int tk_qwen4exp_p1_refill_slots(
    tk_qwen4exp_serve_ctx_t* ctx
);

// Piece 2 (`qwen4exp-p2`): Prefill orchestration (`prefill.cpp`), `tp_moe` (W16 default), `draft_kv`.
int tk_qwen4exp_p2_prefill_chunk(
    tk_qwen4exp_serve_ctx_t* ctx,
    int64_t at,
    int64_t to,
    tk_qwen4exp_prefill_stats_t* out_stats
);
int tk_qwen4exp_p2_tp_moe_gemm(
    const void* w,
    int type,
    size_t expert_bytes,
    int n,
    int k,
    int n_experts,
    const void* xq,
    const int32_t* bounds,
    const int32_t* ids,
    int total_rows,
    int max_rows,
    float* dst,
    int ld_dst,
    void* stream
);
int tk_qwen4exp_p2_draft_kv(
    tk_qwen4exp_serve_ctx_t* ctx,
    const float* r_rows,
    const int32_t* next_tokens,
    int64_t t_rows,
    int64_t pos0
);

// Piece 3 (`qwen4exp-p3`): Single-token decode layer (QSA, GDN, MoE, dense, PLE, LM head).
int tk_qwen4exp_p3_ple_step(
    tk_qwen4exp_gen_ctx_t* ctx,
    int32_t token,
    int phase
);
int tk_qwen4exp_p3_embed_token(
    tk_qwen4exp_gen_ctx_t* ctx,
    int64_t token,
    int64_t pos
);
int tk_qwen4exp_p3_run_layer(
    tk_qwen4exp_gen_ctx_t* ctx,
    int32_t layer,
    int64_t pos
);
int tk_qwen4exp_p3_run_head(
    tk_qwen4exp_gen_ctx_t* ctx,
    int64_t pos,
    int32_t* out_next_token
);

// Piece 4 (`qwen4exp-p4`): Multi-token speculative `Verifier` + `MtpDrafter`.
int tk_qwen4exp_p4_verify_run(
    tk_qwen4exp_serve_ctx_t* ctx,
    int t,
    int t_mtp,
    int from_sfx,
    int64_t pos,
    const int32_t* window,
    const int32_t* consumed,
    size_t n_consumed,
    int32_t* out_tokens
);
int tk_qwen4exp_p4_verify_commit(
    tk_qwen4exp_serve_ctx_t* ctx,
    int keep,
    int64_t rounds,
    int first_window,
    int* out_nf_hit
);
int tk_qwen4exp_p4_mtp_draft(
    tk_qwen4exp_serve_ctx_t* ctx,
    int t,
    int accepted,
    int64_t pos,
    int need_draft,
    double req_spec_min_p,
    const int32_t* out_tokens,
    const int32_t* consumed,
    size_t n_consumed,
    int32_t* out_drafts,
    float* out_dprob
);

// ---- qwen4exp-p4 begin ----
// Piece 4 (`qwen4exp-p4`): the Rust host plane of the speculative Verifier and the MTP drafter
// (rust/crates/talus-model/src/qwen4exp/verify_mtp.rs).  The C++ objects keep the storage (mapped staging,
// device arena, captured graphs, streams); Rust runs the control flow over the views below and calls back only
// for graph capture, the PLE table, and the sampler kernels.  Implemented in src/program/qwen4exp_p4.cpp
// (+ the two hook accessors in generate.cpp's qwen4exp-p4 block).

// The serve / generate loop's Verifier, MtpDrafter, pool and generate-local hooks (opaque; null: not available,
// the caller falls back to the coarse tk_qwen4exp_serve_* / tk_qwen4exp_gen_spec_* entry points).
typedef struct tk_qwen4exp_p4_hooks tk_qwen4exp_p4_hooks_t;
tk_qwen4exp_p4_hooks_t* tk_qwen4exp_p4_serve_hooks(tk_qwen4exp_serve_ctx_t* ctx);
tk_qwen4exp_p4_hooks_t* tk_qwen4exp_p4_gen_hooks(tk_qwen4exp_gen_ctx_t* ctx);

typedef struct tk_qwen4exp_p4_ver_view {
    void* ver;                 // strata::core::Verifier*
    int32_t eligible;          // 1: the Rust plane can run this verifier (one stage, no remote opt, no trace/profile)
    int32_t max_t;
    int32_t n_layers, lb, le;
    int32_t n_embd, hc, k;
    int32_t n_head, n_head_kv, idx_q_heads, step_count;
    int64_t idx_block, idx_top_k;
    int64_t n_vocab;
    // mapped staging (host side)
    int32_t* h_tok; int32_t* h_step; int32_t* h_pos; int32_t* h_commit; int32_t* h_out; int32_t* m_out;   // m_out: h_out's device alias
    float* h_ple; uint32_t* h_nf; float* h_x; int32_t* h_ids; float* h_ymiss;
    uint32_t* h_seq; uint32_t* h_flag; uint32_t* h_flagA; uint32_t* h_flagB; uint32_t* h_plan_err;
    int32_t* h_plan; int64_t plan_i32;
    void* sink;                // strata::core::GpuPlanSink*
    uint8_t* staging; int64_t staging_blobs; int64_t max_blob;
    uint32_t* cur_layer;
    int32_t* groups;           // [9]
    int32_t nonfinite_blocks, verify_max_t;
    // device
    float* head_logits; float* R;
    void* stream; void* copy_stream; void* commit_done; void** commit_exec;   // commit_exec: the member (made by prepare)
    // scalar members (read and written in place)
    int32_t* last_t; int64_t* last_pos0; int32_t* last_tokens;   // last_tokens: [8]
    uint8_t* staged; uint8_t* last_batch; uint8_t* copy_used; uint8_t* commit_pending; uint8_t* always_publish; uint8_t* head_sampling;
    int32_t* nf_row; int64_t* nf_pos; uint32_t* nf_mask;
    double* ms_wait; double* ms_pool; double* ms_host; double* ms_commit; double* ms_gate;
    int64_t* windows; int64_t* spec_windows;
    void* sampling;            // strata::kernels::SamplerParams*
    const int32_t** hist_d; int32_t* hist_len;
    const uint8_t* released;   // std::atomic<bool>
    void* adapt_gate;          // strata::core::AdaptGate* (null: off)
    int32_t** d_spec_slot;     // device: kVerifyMaxT draft ids, then kVerifyMaxT q rows (allocated on first use)
    // the session
    int32_t* ple_prev;         // [2]
    int32_t ple_on;            // ss.ple.ready() && the stage holds layer 1
    void* ple_ss;              // strata::core::SessionState* for the PLE table calls
    const int64_t* max_cells;
    int32_t sizeof_sink, sizeof_sampler;
} tk_qwen4exp_p4_ver_view_t;

typedef struct tk_qwen4exp_p4_gate_view {
    const int32_t* landed;     // std::atomic<int32_t>
    const uint8_t* active;     // std::atomic<bool>
    const volatile int32_t* hland;
    const uint8_t* has; size_t n_has;
} tk_qwen4exp_p4_gate_view_t;

typedef struct tk_qwen4exp_p4_mtp_view {
    void* mtp;                 // strata::core::MtpDrafter*
    int32_t eligible;          // 1: the Rust plane can run this drafter's draft() (no profile / timing diagnostics, no Metal)
    int32_t max_t, n_head;
    const int32_t* max_drafts;
    int64_t hcn;               // hc * n_embd
    const int64_t* max_cells;  // its K/V's max_cells
    int32_t* h_tok; int32_t* h_step; int32_t* h_pos; int32_t* h_row; int32_t* h_out; float* h_prob; int32_t* h_force;
    void* stream;
    float* window_R; const float** src_R;
    uint8_t* coupled_active; uint8_t coupled_ok; uint8_t* chain_live;
    void* cparams;             // mapped SamplerParams the coupled round graph reads (null: no coupled setup)
    int32_t* h_chist;          // mapped penalty ring (kCoupledHistCap + max_t)
    const int32_t* h_q;        // STRATA_SPEC_PROB: the drafts' q lists (null: off)
    int32_t spec_q_stride;
    int32_t coupled_hist_cap;
    double* ms_draft; double* ms_src; int64_t* rounds; int64_t* n_src;
    int32_t* ple_prev;         // the PLE session's [2] (null: no PLE)
    int32_t ple_on;
    void* ple_ss;
} tk_qwen4exp_p4_mtp_view_t;

typedef struct tk_qwen4exp_p4_hooks_info {
    int32_t serve;             // 1: the --serve loop (errors as "ERR ..." on stdout), 0: the generate loop
    int32_t use_mtp;
    int32_t spec_corrupt;      // generate: --spec-corrupt (no q lists then)
    int32_t commit_async;      // what the loop gave Verifier::set_commit_async
    int32_t* hist_n;           // serve: the request's penalty window (0: no penalties)
    int32_t* d_hist;           // serve: device rows for the window's penalty histories
    int32_t hist_cap;          // int32 slots in d_hist
    void* pool;                // strata::core::PoolMultiFn
    void* pool_user;
} tk_qwen4exp_p4_hooks_info_t;
int tk_qwen4exp_p4_hooks_info(tk_qwen4exp_p4_hooks_t* h, tk_qwen4exp_p4_hooks_info_t* out);
// The generate-local work around a window (the adaptive tier, the K/V growth, the timing and the drive):
enum {
    TK_P4_HOOK_PRE_WINDOW = 0,      // drive reset + the adaptive tier before a window; <0: failed (message printed)
    TK_P4_HOOK_ENSURE_KV = 1,       // a = pos, b = t: K/V room for the window; <0: failed (message printed)
    TK_P4_HOOK_WINDOW_STARTED = 2,  // the window's timing starts
    TK_P4_HOOK_POST_WINDOW = 3,     // after a good window (the adaptive overlap's flip)
    TK_P4_HOOK_FIRST_VERIFIED = 4,  // generate: 1 when this was the first verified window (its TTFT is taken), else 0
    TK_P4_HOOK_COMMIT_BEGIN = 5,    // a = rounds: timing + maybe the adaptive thread
    TK_P4_HOOK_JOIN_ADAPT = 6,      // join the adaptive thread; 1: it succeeded, -1: it failed
    TK_P4_HOOK_COMMIT_DONE = 7,     // a good commit (progress beat, foresight counter, timing)
    TK_P4_HOOK_MARK_NONFINITE = 8,  // serve: the request met non-finite logits
    TK_P4_HOOK_DRAFT_DONE = 9       // a = t: the draft's timing + the PCIe share step
};
int tk_qwen4exp_p4_hook(tk_qwen4exp_p4_hooks_t* h, int which, int64_t a, int64_t b);
// The expert pool's failure message after a window (null: the pool did not fail).
const char* tk_qwen4exp_p4_drive_failed(tk_qwen4exp_p4_hooks_t* h);

int tk_qwen4exp_p4_ver_view(tk_qwen4exp_p4_hooks_t* h, tk_qwen4exp_p4_ver_view_t* out);
int tk_qwen4exp_p4_mtp_view(tk_qwen4exp_p4_hooks_t* h, tk_qwen4exp_p4_mtp_view_t* out);
int tk_qwen4exp_p4_gate_view(void* adapt_gate, tk_qwen4exp_p4_gate_view_t* out);
// refresh_ar + capture(T) + capture_commit; the graph the window launches and whether it is the zero-doorbell one.
int tk_qwen4exp_p4_ver_prepare(void* ver, int t, void** out_exec, int* out_ar_on, char* err, size_t err_cap);
// release_gpu_waits (#267): 1 when the streams finished.
int tk_qwen4exp_p4_ver_release(void* ver, int timeout_ms);
// The MTP round graph for T rows and the step graphs 1..max_steps-1 (captured on first use).
int tk_qwen4exp_p4_mtp_prepare(void* mtp, int t, int coupled, int max_steps, void** out_round, void** out_steps,
                               char* err, size_t err_cap);
// The PLE n-gram rows of `n` tokens after prev[2] (advanced in place), each prefetched; out: n * 16 rows.
int tk_qwen4exp_p4_ple_rows(void* ss, const int32_t* tokens, int n, int32_t* prev, uint32_t* out_rows);
int tk_qwen4exp_p4_ple_gather(void* ss, const uint32_t* rows, size_t n_tokens, float* out, char* err, size_t err_cap);
// The head sampler (a kernel launch on `stream`): sampling = a SamplerParams (counter already set).
int tk_qwen4exp_p4_sample_tokens(const float* logits, int n_tokens, int n_vocab, const int32_t* history, int history_len,
                                 const void* sampling, int32_t* out, void* stream);
// Probabilistic acceptance (spec_prob.hpp): 1 when the rows were judged by rejection sampling.
int tk_qwen4exp_p4_sample_tokens_spec(const float* logits, int n_tokens, int n_vocab, const int32_t* history,
                                      int history_len, const void* sampling, const int32_t* d_drafts, const int32_t* d_q,
                                      int n_q, int32_t* out, void* stream);
void tk_qwen4exp_p4_progress(const char* where, int64_t detail, int kind);   // kind 0: at, 1: tick, 2: beat
int tk_qwen4exp_p4_env_flags(void);   // bit 0: HIP build, bit 1: STRATA_ONE_TOKEN_COMMIT on, bit 2: commit async
// ---- phase 2a: the window / commit graph capture in Rust (verify_mtp.rs RustVerifier::prepare) ----
typedef struct tk_qwen4exp_p4_capture_view {
    void** exec;               // Verifier::exec_[9]
    void** exec_nr;            // Verifier::exec_nr_[9] (#871: the doorbell variant)
    uint8_t* all_resident; uint8_t* ar_off;
    const int32_t* h_res;      // host residency table [n_layers * n_expert] (null: none)
    int64_t n_expert;
    // the commit graph's inputs
    int32_t* commit; const int32_t* m_commit;
    float* qkv_L; float* h_L; float* gate_L; float* beta_L; float* z; float* y_dummy; float* idx_raw_L;
    float* tail_snap; float* hist_snap; float* ple_hist; int32_t ple_copy;
    float* gdn_state; int64_t gdn_ord0;
    int64_t ssm_state_size, ssm_k_heads, ssm_v_heads, ssm_d_conv, ssm_conv_channels, idx_key_dim, ng_hist_hc;
    const uint8_t* layer_is_qsa;   // [n_layers]
} tk_qwen4exp_p4_capture_view_t;
typedef struct tk_qwen4exp_p4_qsa_state {
    float* idx_tail; float* idx_dead; float* idx_pooled; int32_t* idx_block_pos; int64_t max_cells;
} tk_qwen4exp_p4_qsa_state_t;
int tk_qwen4exp_p4_capture_view(void* ver, tk_qwen4exp_p4_capture_view_t* out);
int tk_qwen4exp_p4_qsa_state(void* ver, int64_t qsa_index, tk_qwen4exp_p4_qsa_state_t* out);
// A layer's weight (`blk.<l>.<suffix>`) on the device; null (err: the tensor name) when missing.
const float* tk_qwen4exp_p4_layer_weight(void* ver, int64_t layer, const char* suffix, char* err, size_t err_cap);
// Record the T-token window into the stream capture in progress (Verifier::record_window, until Piece 3's recorder).
int tk_qwen4exp_p4_record_window(void* ver, int t, char* err, size_t err_cap);
// instantiate (evicting batch layouts out of VRAM), upload and sync a captured window graph into exec_/exec_nr_[T].
int tk_qwen4exp_p4_window_instantiate(void* ver, int t, void* graph, char* err, size_t err_cap);
// the commit graph's kernels (launch-only; -1 with err when one throws)
int tk_qwen4exp_p4_k_copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream);
int tk_qwen4exp_p4_k_copy_from_mapped(float* dst, const float* src, int64_t n, void* stream);
int tk_qwen4exp_p4_k_copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream);
int tk_qwen4exp_p4_k_gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream);
int tk_qwen4exp_p4_k_gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate,
                                         const float* beta, const float* z, const float* gamma, float eps, float* y,
                                         int h_k, int h_v, int n_tok, const int32_t* n_keep, void* stream, int t_out_begin);
// 1: launched, 0: not taken (the caller keeps the per-layer loop), -1: error
int tk_qwen4exp_p4_k_gdn_commit_layers(float* state, int64_t state_stride, int64_t conv_off, const float* qkv,
                                       int64_t qkv_stride, const float* h, int64_t h_stride, const float* gate,
                                       const float* beta, int64_t gb_stride, int conv_channels, int h_k, int h_v,
                                       int n_max, int n_layers, const int32_t* n_keep, void* stream);
int tk_qwen4exp_p4_k_indexer_append_steps_layers(void* ver, const float* raw0, int64_t raw_stride,
                                                 const int32_t* rel_pos, int pos_stride, int n_steps, int32_t pos_base,
                                                 float eps, const float* const* gamma, const float* const* tail_src,
                                                 float* const* tail, float* const* dead, float* const* pooled,
                                                 int32_t* const* block_pos, int n_layers, int64_t max_cells, void* stream,
                                                 char* err, size_t err_cap);
int tk_qwen4exp_p4_k_indexer_append_steps(void* ver, const float* raw, const int32_t* rel_pos, int pos_stride,
                                          int n_steps, int32_t pos_base, const float* gamma, float eps,
                                          const tk_qwen4exp_p4_qsa_state_t* st, void* stream, char* err, size_t err_cap);
int tk_qwen4exp_p4_k_indexer_append(void* ver, const float* raw, const int32_t* rel_pos, int32_t pos_base,
                                    const float* gamma, float eps, const tk_qwen4exp_p4_qsa_state_t* st, void* stream,
                                    char* err, size_t err_cap);
// ---- phase 2b: every Verifier::run / commit and MtpDrafter::draft / prefill to the Rust plane (TALUS_FN_RUST=1):
// installed once from Rust; the C++ entry points call these first (1 done, 0 declined: the C++ body runs, -1 failed).
typedef struct tk_qwen4exp_p4_ops {
    void* user;
    int (*ver_run)(void* user, void* ver, int t, const int32_t* tokens, int64_t pos0, void* pool, void* pool_user,
                   int32_t* out, const int32_t* spec_q, int spec_nq, char* err, size_t err_cap);
    int (*ver_commit)(void* user, void* ver, int n_keep, int commit_async, char* err, size_t err_cap);
    int (*mtp_draft)(void* user, void* mtp, int t, const int32_t* tokens, int64_t p, int a, int32_t* drafts, float* probs,
                     float min_p, int* n_drafts, char* err, size_t err_cap);
    int (*mtp_prefill)(void* user, void* mtp, const float* r_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                       int sync, char* err, size_t err_cap);
    int (*ver_wait_commit)(void* user, void* ver, char* err, size_t err_cap);
    int (*ver_copy_logits)(void* user, const void* ver, int t, float* host);   // 1 / 0 as copy_logits, -1 declined
    int (*mtp_set_draft_sampling)(void* user, void* mtp, const void* sampler_params);
    int (*mtp_set_draft_history)(void* user, void* mtp, const int32_t* tail, int64_t n_tail, int32_t next);
} tk_qwen4exp_p4_ops_t;
void tk_qwen4exp_p4_install(const tk_qwen4exp_p4_ops_t* ops);   // null: uninstall
int tk_qwen4exp_p4_ver_view_of(void* ver, tk_qwen4exp_p4_ver_view_t* out);
int tk_qwen4exp_p4_mtp_view_of(void* mtp, tk_qwen4exp_p4_mtp_view_t* out);
typedef struct tk_qwen4exp_p4_mtp_pf_view {
    int32_t* tok; int32_t* step; int32_t* pos; float* rin;   // device inputs of the prefill graphs
    int64_t window; const int64_t* prompt_len; double* ms_prefill;
} tk_qwen4exp_p4_mtp_pf_view_t;
int tk_qwen4exp_p4_mtp_pf_view(void* mtp, tk_qwen4exp_p4_mtp_pf_view_t* out);
// the K/V-only prefill graph for T rows (dev_inputs: E-4's, inputs copied on the device; else from the mapped staging)
int tk_qwen4exp_p4_mtp_prefill_graph(void* mtp, int t, int dev_inputs, void** out_exec, char* err, size_t err_cap);
// E-4: the prompt's token / step / position records into the drafter's device block (grown as needed), from a pinned slot
int tk_qwen4exp_p4_mtp_pf_upload(void* mtp, const int32_t* rec, int64_t n_ints, int32_t** out_dev, char* err, size_t err_cap);
// ---- phase 3: the MTP layer recorded from Rust (verify_mtp.rs MtpRecorder): the drafter's buffers and the record ops
typedef struct tk_qwen4exp_p4_mtp_rec_view {
    int32_t *tok, *step, *pos, *row, *ident, *ids, *hit_slot, *hit_dst, *hit_count, *out_ids, *grp_start, *grp_counts;
    unsigned long long* grp_ptr;
    float *rin, *R, *emb, *en, *e2, *hn, *h2, *mixed, *inj, *inj2, *lo, *rs, *bo, *xn;
    float *qfull, *qcur, *kcur, *vcur, *attn, *attn32, *logits, *w, *shared, *parts, *y, *sample, *head_logits, *dummy_inj;
    float *hit_xs, *probs;
    uint8_t *xq, *hit_xq, *experts;
    void *hit_scratch, *arg_scratch, *top_scratch;
    uint16_t* x_bf16;
    const int32_t *m_tok, *m_step, *m_pos, *m_row, *m_force;   // mapped device aliases
    int32_t* m_out; float* m_prob;
    int32_t N, HC, NH, HD, NKV, K, hc_lr, n_rot, n_expert, max_t;
    int64_t window, n_vocab, n_dvocab;
    int32_t hnorm_stream, force_on, kv_rot, coupled_ok, spec_q;
    float *cos_tab, *sin_tab;
    void* stream;
    void **round_exec, **round_exec_c, **step_exec, **step_exec_c, **prefill_exec, **prefill_dev_exec;   // [9] each
    int32_t native_rope, norm_rope_usable;
    int32_t head_type, sub_head, dhead_type;
} tk_qwen4exp_p4_mtp_rec_view_t;
int tk_qwen4exp_p4_mtp_rec_view(void* mtp, tk_qwen4exp_p4_mtp_rec_view_t* out);
// a tensor of the draft layer: kind 0 f32, 1 bf16, 2 the projection weights (Q8_0, or Q4_0 under --mtp-q4: out_type)
const void* tk_qwen4exp_p4_mtp_tensor(void* mtp, const char* name, int kind, int* out_type);
enum {
    TK_P4_M_EMBED = 0,       // n rows: tok -> emb
    TK_P4_M_GR_READ = 1,     // t = mode (0 attention, 1 MLP, 2 final mixer + MoE write, 3 final mixer), n rows
    TK_P4_M_GR_READ_ROW = 2, // t = row: the final mixer by gr_read
    TK_P4_M_KV_APPEND = 3,   // step records, n rows, t = no_batch
    TK_P4_M_ATTN = 4,        // step records, n rows
    TK_P4_M_SHARED = 5,      // n rows
    TK_P4_M_ROUTE = 6,       // n rows
    TK_P4_M_EXPERTS = 7,     // n rows
    TK_P4_M_COMBINE = 8,     // t = row (-1: all n rows in one launch)
    TK_P4_M_GR_WRITE = 9,    // t = row (-1: all n rows)
    TK_P4_M_HEAD = 10,       // n rows, t = coupled, j = chain step, step record
    TK_P4_M_ROUND_TAIL = 11  // t = 0 coupled stage, 1 row a to row 0, 2 select output j, 3 force token j
};
int tk_qwen4exp_p4_m_op(void* mtp, int op, int t, int n, int j, const int32_t* step, void* stream, char* err, size_t err_cap);
// which: 0 round, 1 round (coupled), 2 step, 3 step (coupled), 4 prefill, 5 prefill (device inputs)
int tk_qwen4exp_p4_mtp_instantiate(void* mtp, int which, int idx, void* graph, char* err, size_t err_cap);
int tk_qwen4exp_p4_g_rms_norm(const float* x, const float* gamma, float* out, int cols, int rows, float eps, void* stream);
int tk_qwen4exp_p4_g_rms_norm_grouped(const float* x, const float* gamma, float* out, int cols, int groups, int rows,
                                      float eps, void* stream);
int tk_qwen4exp_p4_g_quantize_q8_1(const float* x, void* xq, int n_in, int ncols, void* stream);
int tk_qwen4exp_p4_g_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);
int tk_qwen4exp_p4_g_add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok,
                                           void* stream);
int tk_qwen4exp_p4_g_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out, int rows, int head_dim,
                                   int n_rot, float eps, const int32_t* pos, void* stream);
int tk_qwen4exp_p4_g_rope(const float* x, float* out, int rows, int head_dim, int n_rot, const int32_t* pos, void* stream);
int tk_qwen4exp_p4_g_rope_neox(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                               const float* sin_tab, const int32_t* pos, void* stream);
int tk_qwen4exp_p4_g_fwht256(float* x, int64_t n_rows, void* stream);
int tk_qwen4exp_p4_g_copy_2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                             void* stream);
int tk_qwen4exp_p4_g_gate_apply(const float* attn, const float* qfull, float* out, int n_head, int head_dim, void* stream);
int tk_qwen4exp_p4_env_mtp(void);   // bit 0 native combine, 1 STRATA_FUSE_HEAD_GR, 2 head mix multi, 3 no multi GR,
                                    // 4 no batched K/V step, 5 the shared expert's branch (CUDA)
// ---- qwen4exp-p4 end ----

// Piece 5 (`qwen4exp-p5`): Conversation state, snapshots, disk session eviction (`SAVE`/`RESTORE`).
int tk_qwen4exp_p5_state_sizes(
    tk_qwen4exp_serve_ctx_t* ctx,
    tk_qwen4exp_conv_state_sizes_t* out_sizes
);
int tk_qwen4exp_p5_snapshot_capture(
    tk_qwen4exp_serve_ctx_t* ctx,
    int64_t pos,
    int as_tail,
    int is_pin,
    int64_t turn_at
);
int tk_qwen4exp_p5_snapshot_restore(
    tk_qwen4exp_serve_ctx_t* ctx,
    int64_t target_pos
);
int tk_qwen4exp_p5_session_file_op(
    tk_qwen4exp_serve_ctx_t* ctx,
    int is_save,
    const char* path
);

// Piece 6 (`qwen4exp-p6`): `--logits-out` / `--kld-in` teacher-forced KLD (first-token KLD).
int tk_qwen4exp_p6_kld_eval(
    tk_qwen4exp_gen_ctx_t* ctx,
    tk_qwen4exp_kld_report_t* out_report
);

// ---- qwen4exp-p3 begin ----
// Piece 3: the per-layer decode window (Verifier::record_window, T rows; T = 1 is plain decode) recorded from Rust
// over launch-only wrappers (src/program/qwen4exp_p3.cpp). Every wrapper enqueues exactly the kernel(s) the C++
// recorder enqueues at the same point, so a window recorded by Rust is the same graph, node for node.
typedef struct tk_qwen4exp_p3_window_view {
    void* ver;            /* strata::core::Verifier* */
    void* ss;             /* strata::core::SessionState* */
    const void* wt;       /* strata::core::WeightTable* */
    const void* g;        /* strata::core::ModelGeometry* */
    int64_t n_embd, hc, hc_lr, k, conv_channels, value_dim, v_heads, k_heads, state_size, d_conv;
    int64_t n_head, head_dim, n_head_kv, idx_q_heads, idx_key_dim, n_expert, n_ff, n_layers, n_vocab;
    int64_t idx_block, n_rot, step_count, step_pos, ng_hist, ng_hc_dim, ng_n_embd;
    int64_t max_t, plan_i32, cap, max_blocks, attn_scratch_floats, lb, le, gdn_ord0, n_qsa_layers;
    int64_t staging_blobs, pcie_group_rows, gdn_floats, max_blob, ple_block_scratch_bytes;
    float eps;
    int32_t batch_rec, split, prof_on, trace_on, ar_on, device_plan, remote_opt, always_publish, df_branch;
    int32_t head_loaded, head_type, ple_ready, ple_stage, native_layout, pcie_mode, sh_fork_ready;
    int32_t ple_key_bf16, ple_value_bf16, ple_key_native, has_hand_in;
    const void* head_weights;
    void* cs; void* sh_cs; void* ev_fork; void* ev_join;
    int32_t *tok, *step, *pos; const int32_t *m_tok, *m_step, *m_pos;
    float *ple, *emb, *R, *mixed, *bo, *inj, *inj2, *lo, *rs, *xn;
    const float* m_ple;
    uint8_t *xq, *xil, *sh_xq;
    float *qkv_L, *h_L, *gate_L, *beta_L, *z, *y;
    float *qfull, *qcur, *kcur, *vcur, *idx_raw_L, *qidx, *scores, *attn, *attn32, *attn_scratch, *tail_snap;
    int32_t* sel;
    float *logits, *w, *shared, *parts, *hit_out;
    int32_t *ids, *plan;
    uint8_t *staging, *hit_xq, *nat_xq;
    float* hit_xs;
    void* hit_scratch;
    float *head_mixed, *head_inj, *head_logits;
    uint16_t* sh_bf16;
    uint8_t* arg_scratch;
    int32_t* one;
    float *sh_gate, *sh_up, *sh_g, *hist_snap;
    unsigned* qcnt;
    uint32_t* skip;
    unsigned long long* slot_off_d;
    uint32_t* m_plan_err;
    float* m_x; int32_t* m_ids; float* m_w;
    uint32_t *m_seq, *m_flag, *m_flagA, *m_flagB;
    int32_t* m_plan; float* m_ymiss; int32_t* m_out; uint32_t* m_nf;
    float* gdn_state; float* ple_hist; void* ple_scratch;
    const int32_t* d_res; const uint8_t* cache_base; int64_t blob;
    float *ple_key, *ple_val;
    const uint16_t *ple_w_key_bf16, *ple_w_value_bf16; const void* ple_w_key_native; int32_t ple_w_key_native_type;
} tk_qwen4exp_p3_window_view_t;

typedef struct tk_qwen4exp_p3_qsa_view {
    float *idx_tail, *idx_dead, *idx_pooled; int32_t* idx_block_pos;
    const float *cos_tab, *sin_tab;
    int64_t max_cells;
    int32_t kv_rot, kv_hybrid, kv_q4, kv_int8;
} tk_qwen4exp_p3_qsa_view_t;

typedef struct tk_qwen4exp_p3_layer_weights {
    int32_t is_qsa;
    const float* hc_norm[2]; const uint16_t* hc_down[2]; const uint16_t* hc_up[2]; const uint16_t* hc_inject[2];
    const uint8_t* hc_q8_down[2]; const uint8_t* hc_q8_up[2]; const uint8_t* hc_q8_inject[2];
    /* GDN */
    int32_t qkv_type, gate_type, out_type; const void *qkv, *gate, *out;
    const uint16_t *alpha, *beta; const float *conv1d, *ssm_norm, *dt, *ssm_a;
    /* QSA */
    const uint16_t *idx_k, *idx_q; int32_t q_type, k_type, v_type, o_type; const void *q, *k, *v, *o;
    const float *q_norm, *k_norm, *iq_norm, *ik_norm;
    /* MoE: router may be null (the per-row moe_route then reports it) */
    const uint16_t* router; const uint16_t* shexp_gate_inp;
    int32_t sg_type, su_type, sd_type; const void *sg, *su, *sd;
} tk_qwen4exp_p3_layer_weights_t;

typedef struct tk_qwen4exp_p3_head_weights {
    const float* norm; const uint16_t* down; const uint16_t* up; const uint8_t* q8_down; const uint8_t* q8_up;
    int32_t forms_ok;   /* norm F32, down/up bf16 */
} tk_qwen4exp_p3_head_weights_t;

typedef struct tk_qwen4exp_p3_gr_args {
    const float* R; float* R_out; int32_t apply; const float* bo_prev; const float* inj_prev;
    const float* w_norm; const uint16_t* w_down; const uint16_t* w_up; const uint16_t* w_inject;
    float eps; float* lo; float* rs; float* inject_out; float* mixed;
    const uint8_t* q8_down; const uint8_t* q8_up; const uint8_t* q8_inject; uint8_t* q8_mixed; unsigned* q8_cnt;
} tk_qwen4exp_p3_gr_args_t;

/* Rust's window recorder: returns 1 recorded, 0 declined (the C++ recorder runs), -1 error (message in err). */
typedef int (*tk_qwen4exp_p3_recorder_fn)(void* user, void* ver, int t, void* stream, char* err, size_t err_cap);
void tk_qwen4exp_p3_set_window_recorder(tk_qwen4exp_p3_recorder_fn fn, void* user);
int tk_qwen4exp_p3_last_error(char* buf, size_t cap);
/* bit 0 native_router, 1 native_moe_combine, 2 shared_expert_native_bf16, 3 native_qsa, 4 native_rope,
   5 ple_native_bf16, 6 ple_native_postops, 7 gr_native_mmvf, 8 argmax_rows_wanted, 9 nan_guard, 10 pdl_supported,
   11 device_is_gfx12, 12 hip_build, 13 nan_guard_on */
uint64_t tk_qwen4exp_p3_runtime_flags(void);
int tk_qwen4exp_p3_window_view(void* ver, tk_qwen4exp_p3_window_view_t* out);
int tk_qwen4exp_p3_set_groups(void* ver, int t, int g);
int tk_qwen4exp_p3_qsa_view(void* ss, int64_t qi, tk_qwen4exp_p3_qsa_view_t* out);
int tk_qwen4exp_p3_layer_weights(const void* wt, int32_t layer, int32_t is_qsa, tk_qwen4exp_p3_layer_weights_t* out);
int tk_qwen4exp_p3_head_weights(const void* wt, tk_qwen4exp_p3_head_weights_t* out);
int tk_qwen4exp_p3_pdl_scope(int on);
int tk_qwen4exp_p3_cvec_covers(int64_t layer);
int tk_qwen4exp_p3_norm_rope_usable(int head_dim, int n_rot);
int tk_qwen4exp_p3_mmvq_il_supported(int type, int ncols, int n_out);
int tk_qwen4exp_p3_combine_hits_supported(const float* parts, const float* shared, const float* out, int64_t n_embd, int64_t k, int n);
int tk_qwen4exp_p3_expert_blob_bytes(int64_t layer, int64_t* out);
/* launch-only wrappers: 0 ok, -1 error (tk_qwen4exp_p3_last_error) */
/* records `ev` on `from` (when not null), then makes `to` wait on it (when not null) */
int tk_qwen4exp_p3_event_fork(void* ev, void* from, void* to);
int tk_qwen4exp_p3_is_qsa_layer(const void* g, int64_t layer);
int tk_qwen4exp_p3_copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* s);
int tk_qwen4exp_p3_copy_from_mapped(float* dst, const float* src, int64_t n, void* s);
int tk_qwen4exp_p3_embed_rows(const void* wt, const int32_t* tok, int t, float* emb, void* s);
int tk_qwen4exp_p3_broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int t, void* s);
int tk_qwen4exp_p3_wait_flag_ge(const uint32_t* flag, uint32_t v, void* s);
int tk_qwen4exp_p3_wait_flag_ge_or(const uint32_t* flag, uint32_t v, const uint32_t* skip, void* s);
int tk_qwen4exp_p3_bf16_gemv(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* s);
int tk_qwen4exp_p3_bf16_gemv_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in, int64_t n_out, int n, void* s);
int tk_qwen4exp_p3_bf16_gemv_multi_aux(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy, int64_t n_in, int64_t n_out, int n, const uint16_t* w_aux, float* y_aux, int64_t ldy_aux, void* s);
int tk_qwen4exp_p3_quantize_q8_1(const float* x, void* xq, int n_in, int ncols, void* s);
int tk_qwen4exp_p3_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* s);
int tk_qwen4exp_p3_mmvq_il(int type, const void* w, const void* xq, const void* xil, float* y, int n_in, int n_out, int ncols, void* s);
int tk_qwen4exp_p3_q8_1_interleave(const void* xq, void* xil, int n_in, int ncols, void* s);
int tk_qwen4exp_p3_gr_write(const float* R, const float* bo, const float* inj, int64_t n_embd, int64_t hc, int64_t hc_lr, float* R_out, int n, int multi, void* s);
int tk_qwen4exp_p3_gr_post_multi(const float* R, const float* bo, const float* inj, float* out, int n_embd, int hc, int n, long long r_stride, long long b_stride, long long i_stride, void* s);
int tk_qwen4exp_p3_fused_gr_read_multi(const tk_qwen4exp_p3_gr_args_t* a, int n, float* xn, void* s);
int tk_qwen4exp_p3_ple_postops_batch_snap(void* ss, float* xn, float* R, const float* z, float* qn, float* gated, float* pgate, int n, float* snap, void* s);
int tk_qwen4exp_p3_ple_block(void* ss, const float* emb, float* R, float* hist, float* normalized, const float* pre_key, const float* pre_value, int use_pre, void* s);
int tk_qwen4exp_p3_ple_block_projected(void* ss, const float* key, const float* value, float* R, float* hist, float* normalized, void* s);
int tk_qwen4exp_p3_ple_history_advance(float* hist, const float* normalized, void* s);
int tk_qwen4exp_p3_gdn_ab_multi(const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt, const float* sa, float* gate, float* beta, int n_embd, int h_v, int n, void* s);
int tk_qwen4exp_p3_gdn_conv_ab_multi(const float* hist, const float* qkv, const float* conv_w, float* h, int ch, int qk_heads, float eps, int n, int tb, int commit, const float* x, const uint16_t* wa, const uint16_t* wb, const float* dt, const float* sa, float* gate, float* beta, int n_embd, int h_v, void* s);
int tk_qwen4exp_p3_gdn_conv_l2_multi(const float* hist, const float* qkv, const float* conv_w, float* h, int ch, int qk_heads, float eps, int n, int tb, int commit, void* s);
int tk_qwen4exp_p3_gdn_step_norm_multi(float* state, const float* h, int ch, const float* gate, const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n, const int32_t* n_keep, int tb, void* xq_out, void* s);
int tk_qwen4exp_p3_rms_norm(float* x, const float* w, int rows, int cols, float eps, int native, void* s);
int tk_qwen4exp_p3_rope(float* x, int rows, int cols, int n_rot, const int32_t* pos, const float* cos_tab, const float* sin_tab, int native, void* s);
int tk_qwen4exp_p3_rms_norm_rope(const float* x, int in_stride, const float* gamma, float* out, int rows, int head_dim, int n_rot, float eps, const int32_t* pos, void* s);
int tk_qwen4exp_p3_copy_rows_strided(float* dst, const float* src, int64_t rows, int64_t w, int64_t src_w, void* s);
int tk_qwen4exp_p3_copy_2d(float* dst, const float* src, int64_t head_dim, int64_t n_head, void* s);
int tk_qwen4exp_p3_fwht256(float* x, int64_t rows, void* s);
/* mode: 0 per-token loop, 1 batched steps; the K/V format is the layer's own */
int tk_qwen4exp_p3_kv_append(const void* g, void* ss, int64_t qi, const int32_t* step, int step_stride, const float* kcur, const float* vcur, int cur_stride, int n, int batched, void* s);
int tk_qwen4exp_p3_indexer_append(const void* g, void* ss, int64_t qi, const float* raw, int64_t raw_stride, const int32_t* step, int step_stride, int n, int batched, const float* gamma, float eps, void* s);
int tk_qwen4exp_p3_qsa_block_scores(const void* g, void* ss, int64_t qi, const float* qidx, const int32_t* steps, int64_t n, int64_t max_blocks, float* scores, void* s);
int tk_qwen4exp_p3_qsa_block_topk(const void* g, const float* scores, const int32_t* steps, int64_t n, int64_t max_blocks, int64_t cap, int32_t* sel, void* s);
int tk_qwen4exp_p3_qsa_kv_resolve(const void* g, void* ss, int64_t qi, const int32_t* sel, const int32_t* steps, int64_t n, int64_t cap, void* s);
int tk_qwen4exp_p3_qsa_attn(const void* g, void* ss, int64_t qi, const float* q, const int32_t* sel, const int32_t* steps, int64_t cap, float* scratch, float* attn, int64_t n, void* s);
int tk_qwen4exp_p3_qsa_gate(const float* attn, const float* qfull, float* out, int rows, int head_dim, int native, const void* g, void* s);
int tk_qwen4exp_p3_router_top10_multi(const float* logits, int32_t* ids, float* w, int n, void* s);
int tk_qwen4exp_p3_moe_route_row(const void* wt, const void* g, void* ss, int64_t layer, int64_t k, const float* x, float* logits, int32_t* ids, float* w, void* s);
int tk_qwen4exp_p3_resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res, int n_expert, const uint8_t* base, const unsigned long long* slot_off, long long blob, int32_t* plan, long long capx, uint32_t* skip, uint32_t ring, uint32_t* plan_err, void* s);
/* variant: 0 publish, 1 publish_res, 2 publish_value */
int tk_qwen4exp_p3_doorbell_publish(int variant, const float* x, const int32_t* ids, const float* w, const int32_t* res, int n_expert, int64_t n, int64_t k, float* x_out, int32_t* ids_out, float* w_out, uint32_t* seq, uint32_t value, void* s);
int tk_qwen4exp_p3_f32_to_bf16(const float* x, uint16_t* y, int64_t n, void* s);
int tk_qwen4exp_p3_shared_expert_multi(int n, const float* x, const uint16_t* x_bf16, int32_t gt, const void* gd, int32_t ut, const void* ud, int32_t dt_, const void* dd, void* q8_1, const uint16_t* gate_inp, float* gate, float* up, float* g_out, float* out, int64_t n_embd, int64_t n_ff, const void* xq_ready, int lfuse, void* s);
int tk_qwen4exp_p3_quantize_q8_1_rows(const float* x, int64_t rows, int64_t cols, void* y, void* s);
int tk_qwen4exp_p3_quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* s);
int tk_qwen4exp_p3_expert_grouped(int64_t layer, int native, const unsigned long long* gp, const int32_t* gs, const int32_t* gn, const int32_t* dst, const int32_t* tok, int64_t cap, const void* x_nat, const uint8_t* x_s2, const float* xs_s2, void* scratch, float* out, int64_t gy, void* s);
int tk_qwen4exp_p3_copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t v, void* s);
int tk_qwen4exp_p3_wait_copy_i32_from_mapped(const uint32_t* flag, uint32_t v, int32_t* dst, const int32_t* src, int64_t n, void* s);
int tk_qwen4exp_p3_fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* s);
int tk_qwen4exp_p3_rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* s);
int tk_qwen4exp_p3_copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t v, void* s);
int tk_qwen4exp_p3_copy_rows_from_mapped(const uint32_t* flag_after, uint32_t v, float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows, const int32_t* count, void* s);
int tk_qwen4exp_p3_moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap, int64_t n_embd, void* s);
/* variant: 0 multi, 1 hits_added, 2 hits_gated */
int tk_qwen4exp_p3_moe_combine(int variant, const float* parts, const float* hits, const int32_t* dst, const int32_t* count, const float* w, const float* shared, const float* shared_gate, float* out, int64_t n_embd, int64_t k, int n, void* s);
int tk_qwen4exp_p3_moe_combine_row(const void* g, void* ss, int64_t layer, int64_t k, const float* w, const float* shared, const float* parts, float* out, void* s);
int tk_qwen4exp_p3_cvec_apply(float* R, int64_t layer, int64_t t, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj, int64_t inj_ld, int write, void* s);
int tk_qwen4exp_p3_head_mix_row(const void* wt, const void* g, void* ss, float* R, float* mixed, float* logits, int full_head, void* s);
int tk_qwen4exp_p3_argmax_rows(const float* logits, int rows, int n, void* scratch, int32_t* out, void* s);
int tk_qwen4exp_p3_sample_greedy(const float* logits, int rows, int n, int32_t* out, void* s);
int tk_qwen4exp_p3_logits_nonfinite_rows(const float* logits, int rows, int n, uint32_t* flags, void* s);
int tk_qwen4exp_p3_gpu_noop(void* s);
// ---- qwen4exp-p3 end ----

// ---- qwen4exp-p5 begin ----
// Piece 6 (`kld_flags.rs`): the Rust host owns the `--dump-logits` writer of the per-token generate path.
// Takes the open (empty) dump file from the C++ side: returns 1 with the path (NUL-terminated), the row stride and the
// position count, 0 when no dump is open, -1 on error. The C++ step loop then writes nothing.
int tk_qwen4exp_p6_take_logits_dump(
    tk_qwen4exp_gen_ctx_t* ctx,
    char* out_path,
    size_t path_cap,
    int64_t* out_stride,
    int64_t* out_positions
);
// Copies the last single-token step's logits row (n_vocab floats, before sampling) to host memory. 1 = ok, -1 = error.
int tk_qwen4exp_p6_step_logits(
    tk_qwen4exp_gen_ctx_t* ctx,
    float* out_logits
);

// Piece 5 (`conv_state.rs`): the `--serve` conversation state is Rust-owned. The C++ serve lambdas call these hooks
// where they ran the conversation code (a request's begin, checkpoints, pins, the live update at the end, SAVE /
// RESTORE); Rust calls the primitives below for the device work.
typedef struct tk_qwen4exp_p5_qsa {
    void* pools[5];          // authoritative K, V, K scale, V scale, indexer pooled rows (host pools when streamed)
    void* idx_tail;
    void* idx_dead;
    void* idx_block_pos;
    void* idx_pooled;
    int64_t max_cells, n_pages, n_slots, idx_pooled_rows;
    int32_t kv_mode, kv_format, kv_int8, kv_q4, kv_hybrid, kv_rot, host_present, map_ok, ring_ok, reserved;
} tk_qwen4exp_p5_qsa_t;

#define TK_QWEN4EXP_P5_MAX_QSA 32
typedef struct tk_qwen4exp_p5_layout {
    int64_t geometry[18];
    int64_t layer_lo, layer_hi, n_gdn_layers, n_qsa_layers, gdn_alloc, qsa_ord0, qsa_alloc, max_cells;
    int64_t ssm_state_size, ssm_v_heads, ssm_conv_channels, ssm_d_conv, idx_key_dim, n_head_kv, head_dim;
    int64_t ng_hist, ng_hc_dim, idx_block, page_size, kv_q4_bytes_per_head;
    uint64_t memory_limit_bytes;
    void* gdn_state;
    void* ple_hist;
    int32_t has_draft;       // the draft layer's K/V is parked with a conversation (--mtp)
    int32_t n_qsa;           // owned QSA states in qsa[]
    tk_qwen4exp_p5_qsa_t qsa[TK_QWEN4EXP_P5_MAX_QSA];
    tk_qwen4exp_p5_qsa_t draft;
} tk_qwen4exp_p5_layout_t;

typedef struct tk_qwen4exp_p5_options {
    int32_t prompt_cache, prompt_cache_tail;
    int64_t prompt_cache_every, conversation_cache_mib, conversation_cache_slots, conversation_cache_min_tokens;
    int64_t conversation_cache_min_free_mib, session_min_free_mib;
    int32_t session_save_reclaim, trace;
    int64_t max_context;
    int32_t use_mtp, reserved;
} tk_qwen4exp_p5_options_t;

typedef struct tk_qwen4exp_p5_hooks {
    void* user;
    // the conversation part of a request's begin: 1 = resume / read_from / reread_to set, -1 = ERR printed, stop
    int (*begin)(void* user, const int64_t* ids, size_t n, int req_cache, int req_ckpt, int64_t req_pin,
                 int want_cvec, int64_t* out_resume, int64_t* out_read_from, int64_t* out_reread_to);
    int (*pin)(void* user, int64_t len);                                   // 1 = a checkpoint of that length
    // a checkpoint after cur[0, len): 1 = ok, 0 = failed (why: the reason suffix of the ERR line)
    int (*checkpoint)(void* user, const int64_t* ids, size_t n_ids, int64_t len, int as_tail, char* why,
                      size_t why_cap);
    void (*finish)(void* user, const int32_t* consumed, size_t n_consumed, const int64_t* cur, size_t n_cur,
                   int cancelled, int nf_hit, int req_ckpt, int pp_exact, int64_t pp_reached);
    int (*session)(void* user, const char* line);                          // SAVE / RESTORE: 0 answered, -1 stop
    size_t (*check_count)(void* user);
} tk_qwen4exp_p5_hooks_t;

int tk_qwen4exp_p5_register_hooks(tk_qwen4exp_serve_ctx_t* ctx, const tk_qwen4exp_p5_hooks_t* hooks);
int tk_qwen4exp_p5_layout(tk_qwen4exp_serve_ctx_t* ctx, tk_qwen4exp_p5_layout_t* out);
int tk_qwen4exp_p5_options(tk_qwen4exp_serve_ctx_t* ctx, tk_qwen4exp_p5_options_t* out);
// device <-> host copy (cudaMemcpyDefault on a non-blocking stream, synchronized): 0 = ok, else err filled
int tk_qwen4exp_p5_copy(void* dst, const void* src, size_t n, char* err, size_t err_cap);
int tk_qwen4exp_p5_sync(char* err, size_t err_cap);
// after a K/V restore: refill the VRAM slots (streamed) or the draft ring from the restored pools
// (which = owned QSA ordinal, -1 = the draft layer)
int tk_qwen4exp_p5_kv_residency(tk_qwen4exp_serve_ctx_t* ctx, int which, int64_t upto, char* err, size_t err_cap);
void tk_qwen4exp_p5_set_ple_prev(tk_qwen4exp_serve_ctx_t* ctx, int32_t a, int32_t b);
int tk_qwen4exp_p5_kvg_ensure(tk_qwen4exp_serve_ctx_t* ctx, int64_t cells);              // 1 ok, 0 no VRAM
int tk_qwen4exp_p5_kvg_trim(tk_qwen4exp_serve_ctx_t* ctx, int64_t cells, int64_t hold);  // 1 ok, 0 failed
void tk_qwen4exp_p5_session_zero(tk_qwen4exp_serve_ctx_t* ctx);
void tk_qwen4exp_p5_mtp_kv_restore(tk_qwen4exp_serve_ctx_t* ctx, int64_t resume);
int tk_qwen4exp_p5_take_nan_flush(tk_qwen4exp_serve_ctx_t* ctx);
int tk_qwen4exp_p5_wait_commit(tk_qwen4exp_serve_ctx_t* ctx, char* err, size_t err_cap);   // 1 ok, 0 failed
// the session identity's inputs as text (conv_state/serve.rs parse_session_inputs); returns the length needed
size_t tk_qwen4exp_p5_session_inputs(tk_qwen4exp_serve_ctx_t* ctx, char* buf, size_t cap);
void tk_qwen4exp_p5_progress_at(const char* text, int64_t detail);
void tk_qwen4exp_p5_progress_allow(int64_t seconds);
void tk_qwen4exp_p5_progress_beat(void);
// ---- qwen4exp-p5 end ----

// ---- qwen4exp-p1 begin ----
// Piece 1 (`qwen4exp-p1`): the Rust hook table. Rust (`qwen4exp::weights::install_rust_hooks`) registers it once per process;
// the C++ engine calls a hook instead of its own code only when the Rust data plane is on (the driver's run_serve_loop is
// set) and TALUS_FN_P1 is not 0. A null hook keeps the C++ code for that step.
typedef struct tk_qwen4exp_p1_weight_row {
    const char* name;
    const void* data;
    uint64_t bytes;
    int64_t ne0;
    int64_t ne1;
    int64_t elements;
    int32_t kind;
    int32_t code_bits;
    int32_t code_bias;
    int32_t group_elems;
    int32_t codebook_iq4nl;
    int32_t has_offset;
    uint64_t codes_bytes;
    uint64_t scales_bytes;
    uint64_t offset_bytes;
    uint64_t src_off;
    uint64_t src_bytes;
    int32_t file_id;
    int32_t scales_fp16;
    int32_t act_kind;
    int32_t resident;
} tk_qwen4exp_p1_weight_row_t;

typedef struct tk_qwen4exp_p1_load_report {
    uint64_t tensors;
    uint64_t arena_bytes;
    uint64_t re_rounded;
    uint64_t bytes_saved;
    double read_ms;
    double upload_ms;
} tk_qwen4exp_p1_load_report_t;

// the expert cache sizing (generate.cpp: --expert-cache auto / per-layer / the native pack's profile list)
typedef struct tk_qwen4exp_p1_cache_plan_in {
    int64_t n_layers;
    int64_t n_expert;
    int32_t expert_cache;        // as it stands (-1: auto)
    int32_t native_pack;
    int32_t per_layer;
    int32_t multi_gpu;
    int32_t vram_reserve_mib;
    int32_t vram_reserve_given;
    int64_t prefill_mib;         // auto only: the prompt path's own buffers
    int64_t mtp_bind;            // auto only: the draft heads' bind bytes
    int64_t min_slots;           // auto only: the smallest working cache
    int64_t session_mib;         // auto only: for the "no VRAM left" message
    int64_t max_context;
    int64_t pipe_first;
    uint64_t max_blob;
    const uint64_t* blob_bytes;  // n_layers
    const int32_t* profile;      // 2 * n_profile: (layer, expert) ranked
    size_t n_profile;
    const char* profile_path;
} tk_qwen4exp_p1_cache_plan_in_t;

typedef struct tk_qwen4exp_p1_cache_plan_out {
    int32_t expert_cache;
    int32_t vram_reserve_mib;
    int32_t reserve_adapted;
    const int64_t* sized_slots;  // Rust-owned until the next plan / build
    size_t n_sized;
    uint64_t per_layer_bytes;
} tk_qwen4exp_p1_cache_plan_out_t;

typedef const uint8_t* (*tk_qwen4exp_p1_blob_fn)(void* ctx, int32_t layer, int32_t expert);

// the expert cache: open (shrinking an auto cache that fails), fill from the profile, slot-0 check
typedef struct tk_qwen4exp_p1_cache_build_in {
    int64_t n_layers;
    int64_t n_expert;
    int32_t expert_cache;
    int32_t auto_cache;
    int32_t per_layer;
    int32_t vram_reserve_mib;
    int64_t pipe_first;
    uint64_t max_blob;
    const uint64_t* blob_bytes;
    const int64_t* sized_slots;
    size_t n_sized;
    uint64_t per_layer_bytes;
    const int32_t* profile;
    size_t n_profile;
    tk_qwen4exp_p1_blob_fn blob_fn;
    void* blob_ctx;
    // the mapped file tier (FileExpertSource): its read-ahead of the profile's pairs during the fill
    int32_t file_source;
    int32_t unbuffered;
    int (*advise_pairs)(void* ctx, const int32_t* pairs, int64_t n);      // 1: advised
    void (*prefetch_pairs)(void* ctx, const int32_t* pairs, int64_t n);   // unbuffered: read the batch now
    void (*pin_aux_thread)(void* ctx);
} tk_qwen4exp_p1_cache_build_in_t;

typedef struct tk_qwen4exp_p1_cache_build_out {
    uint8_t* base;               // null: no cache; else ExpertCache::adopt takes (and frees) it
    int64_t slots;
    int64_t blob;
    const uint64_t* offsets;     // slots + 1, or null (uniform slots)
    size_t n_offsets;
    const int32_t* residency;    // n_layers * n_expert
    size_t n_residency;
    const int32_t* layer_next;
    size_t n_layer_next;
    int64_t next_free;
    int64_t admitted;
    int64_t fills;
    int64_t prefilled;
    int32_t expert_cache;
    int32_t sized_changed;
} tk_qwen4exp_p1_cache_build_out_t;

// the serve loan: the one prompt path borrowing the tail slots of the one expert cache (generate.cpp lend / refill)
typedef struct tk_qwen4exp_p1_lend_ctx {
    void* user;
    int32_t* host_res;           // the C++ table, edited in place
    size_t n_res;
    int64_t n_layers;
    int64_t n_expert;
    int32_t* d_res;
    uint8_t* cache_base;
    int64_t cache_slots;
    int64_t cache_bytes;
    int64_t cache_blob;
    const uint64_t* slot_offsets;
    uint64_t max_blob;
    const uint64_t* blob_bytes;
    int32_t first;               // the part's first lendable slot (-1: no loan)
    int32_t first_now;
    int32_t active;              // 0: no prompt path borrows (lend is a no-op)
    int32_t head_alias;
    int64_t prefill_chunk;
    int64_t split_small;
    int64_t split_small_max;
    int32_t equal_chunks;
    int32_t refill_blocking;
    int64_t stream_all_min;
    uint64_t (*bytes_needed)(void* user, int64_t chunk);
    int64_t (*sp_chunk)(void* user);
    int (*relayout)(void* user, int64_t chunk, uint8_t* base, uint64_t bytes, char* err, size_t err_cap);
    const uint8_t* (*blob)(void* user, int32_t layer, int32_t expert);
    void (*note_async_read)(void* user, const uint8_t* blob);
} tk_qwen4exp_p1_lend_ctx_t;

// the native GGUF tensors (NativeDense / NativeHead / NativeEmbed), loaded by Rust, adopted by the C++ objects
typedef struct tk_qwen4exp_p1_native_attach {
    const char* name;
    void* native_data;
    int32_t native_type;
    void* native_q8_1;
    void* hc_q8;
} tk_qwen4exp_p1_native_attach_t;

typedef struct tk_qwen4exp_p1_native_dense_out {
    void* const* weights;        // every buffer the C++ NativeDense frees
    size_t n_weights;
    const void* const* packed_keys;
    size_t n_packed;
    void* scratch;
    uint64_t bytes;
    uint64_t allocated;
    const tk_qwen4exp_p1_native_attach_t* attach;
    size_t n_attach;
} tk_qwen4exp_p1_native_dense_out_t;

typedef struct tk_qwen4exp_p1_native_head_out {
    void* weights;
    void* scratch;
    uint64_t bytes;
    int32_t n_in;
    int32_t n_out;
    int32_t type;
    int32_t device;
} tk_qwen4exp_p1_native_head_out_t;

typedef struct tk_qwen4exp_p1_native_embed_out {
    void* host;                  // null: the table is in VRAM at `dev`
    const void* dev;
    uint64_t bytes;
    uint64_t row;
    int64_t n_embd;
    int64_t n_vocab;
    int32_t type;
} tk_qwen4exp_p1_native_embed_out_t;

// what the Rust loaders need from the kernel library (byte counts of the GPU formats, the architecture check, the
// optional repacks): tk_qwen4exp_p1_services()
typedef struct tk_qwen4exp_p1_services {
    int (*native_mmvq_supported)(int type);
    uint64_t (*native_mmvq_weight_bytes)(int type, int n_in, int n_out);
    uint64_t (*native_q8_1_bytes)(int n_in, int ncols);
    int (*check_model)(const char* const* paths, size_t n, int allow_embd, char* err, size_t err_cap);
    int (*check_shard_arch)(const char* path, char* err, size_t err_cap);
    int (*q8_0_packed_enabled)(void);
    int (*q8_0_packed_eligible)(int n_in, int n_out);
    void (*q8_0_pack_host)(const void* src, void* dst, int n_in, int n_out);
    void (*q8_0_packed_register)(const void* w, const void* packed, int n_in, int n_out);
    int (*hc_q8_requested)(void);
    void (*fused_gr_quantize_q8_0)(const uint16_t* bf16, uint8_t* out, int64_t n_blocks, void* stream);
    int (*hc_scratch_install)(uint64_t elems, void** out4, char* err, size_t err_cap);
    int (*embed_type_supported)(int type);
    uint64_t (*iq_row_bytes)(int type, int64_t n);
    int (*q6_k_packed_enabled)(void);
    int (*q6_k_pack)(const void* w, int n_in, int n_out, const char* what);
} tk_qwen4exp_p1_services_t;
const tk_qwen4exp_p1_services_t* tk_qwen4exp_p1_services(void);

// the PLE weights (strata::kernels::PleWeights' weight fields), wired by Rust from blk.1.ple_*
typedef struct tk_qwen4exp_p1_ple_weights {
    const uint8_t* key_codes;
    const float* key_scales;
    const uint16_t* value_bf16;
    const float* norm_key;
    const float* norm_query;
    const float* norm_conv;
    const uint16_t* conv1d_f16;
    const void* key_native_data;
    int32_t key_native_type;
    void* key_native_q8_1;
    const uint16_t* key_bf16;
} tk_qwen4exp_p1_ple_weights_t;

// the prompt path's loan planning (plan_lend / lend_slots / the serve chunk scan): the cache, the knobs, and the
// prompt path's pricing (Prefill statics) as callbacks
typedef struct tk_qwen4exp_p1_planner {
    void* user;
    int64_t cache_slots;
    int64_t cache_bytes;
    const uint64_t* slot_offsets;
    uint64_t max_blob;
    int32_t prefill_auto;
    int64_t prefill_chunk;
    int64_t prefill_auto_max;
    int64_t prefill_auto_cap;
    int64_t max_context;
    int64_t auto_ceiling;
    int64_t lend_pct;
    uint64_t (*bytes_needed)(void* user, int64_t chunk);
    uint64_t (*bytes_needed_no_ring)(void* user, int64_t chunk);
    void (*set_ring_budget)(void* user, int64_t room, int64_t small);
    int64_t (*ring_cap_for)(void* user, int64_t chunk);
    int64_t (*ring_default_slots)(void* user);
    int (*ring_bytes_enabled)(void* user);
} tk_qwen4exp_p1_planner_t;

// the PLE table's I/O options (strata::kernels::PleIoOptions), planned by Rust
typedef struct tk_qwen4exp_p1_ple_io {
    int32_t mode;                // 0 direct, 1 mmap
    int32_t lock;
    uint32_t max_inflight;
    uint64_t cache_rows;
    int32_t io_thread;
    double keepalive_ms;
    double keepalive_window_s;
    int32_t set_io_threads;      // set STRATA_IO_THREADS=64 (not overwriting) before the open
} tk_qwen4exp_p1_ple_io_t;

typedef struct tk_qwen4exp_p1_rust_hooks {
    void* user_data;
    // WeightTable::pool_bytes: the arena size of <pack_dir>/index.txt without the `skip` names (skip == NULL: no skip set)
    int (*pack_pool_bytes)(void* user, const char* pack_dir, const char* const* skip, size_t n_skip, uint64_t* out_bytes,
                           char* err, size_t err_cap);
    // WeightTable::load into `arena`; the rows stay valid for the process (Rust owns them)
    int (*pack_load)(void* user, const char* pack_dir, void* arena, uint64_t arena_bytes, const char* const* skip,
                     size_t n_skip, const tk_qwen4exp_p1_weight_row_t** out_rows, size_t* out_n_rows,
                     tk_qwen4exp_p1_load_report_t* out_report, char* err, size_t err_cap);
    // read_expert_profile: the ranked pairs (2 int32 each, Rust-owned) and the slot count the profile was built for
    int (*profile_read)(void* user, const char* path, int64_t n_layers, int64_t n_expert, const int32_t** out_pairs,
                        size_t* out_n, int64_t* out_slots, char* err, size_t err_cap);
    int (*cache_plan)(void* user, const tk_qwen4exp_p1_cache_plan_in_t* in, tk_qwen4exp_p1_cache_plan_out_t* out);
    int (*cache_build)(void* user, const tk_qwen4exp_p1_cache_build_in_t* in, tk_qwen4exp_p1_cache_build_out_t* out,
                       char* err, size_t err_cap);
    // d_res + the hit counter, allocated by Rust and staged from host_res
    int (*res_create)(void* user, const int32_t* host_res, size_t n, int32_t** out_d_res, int32_t** out_d_hit_count,
                      char* err, size_t err_cap);
    int (*lend_init)(void* user, const tk_qwen4exp_p1_lend_ctx_t* ctx, char* err, size_t err_cap);
    int (*lend)(void* user, int64_t tokens, int32_t* out_lent, char* err, size_t err_cap);
    int (*refill)(void* user, char* err, size_t err_cap);
    int (*native_served_names)(void* user, const char* const* shards, size_t n_shards, int include_ple_key,
                               const char* const** out_names, size_t* out_n, char* err, size_t err_cap);
    int (*native_dense_load)(void* user, const char* const* shards, size_t n_shards, int include_ple_key,
                             tk_qwen4exp_p1_native_dense_out_t* out, char* err, size_t err_cap);
    int (*native_head_load)(void* user, const char* const* shards, size_t n_shards, int64_t n_in, int64_t n_out,
                            tk_qwen4exp_p1_native_head_out_t* out, char* err, size_t err_cap);
    int (*native_embed_load)(void* user, const char* const* shards, size_t n_shards, int64_t n_embd, int64_t n_vocab,
                             tk_qwen4exp_p1_native_embed_out_t* out, char* err, size_t err_cap);
    int (*ple_wire)(void* user, int native_ple_key, tk_qwen4exp_p1_ple_weights_t* out, char* err, size_t err_cap);
    int64_t (*plan_lend)(void* user, const tk_qwen4exp_p1_planner_t* planner, int64_t* chunk_inout);
    int64_t (*lend_slots)(void* user, const tk_qwen4exp_p1_planner_t* planner, int64_t chunk);
    int (*serve_scan)(void* user, const tk_qwen4exp_p1_planner_t* planner, int64_t* out_chunk, int64_t* out_first);
    // generate's one-shot loan: take every slot from `first` (returns how many), and give them back after the prompt
    int64_t (*gen_lend)(void* user, const tk_qwen4exp_p1_lend_ctx_t* ctx, int32_t first, char* err, size_t err_cap);
    int64_t (*gen_refill)(void* user, char* err, size_t err_cap);
    int (*ple_io_plan)(void* user, const char* ple_io, int64_t inflight, int64_t row_cache, int sync_submit,
                       tk_qwen4exp_p1_ple_io_t* out);
    // after the direct-read probe: 0 nothing, 1 switch to mapped reads, 2 warn
    int (*ple_probe_decide)(void* user, double rows_per_s, int ple_io_explicit);
} tk_qwen4exp_p1_rust_hooks_t;

void tk_qwen4exp_p1_set_rust_hooks(const tk_qwen4exp_p1_rust_hooks_t* hooks);
const tk_qwen4exp_p1_rust_hooks_t* tk_qwen4exp_p1_rust_hooks(void);
// ---- qwen4exp-p1 end ----

// ---- qwen4exp-p2 begin ----
// Piece 2 (qwen4exp-p2): the batched prompt path's data plane in Rust (rust/.../qwen4exp/prefill.rs).  Rust owns the
// chunk loop, the expert stream plan and its issuer thread, the layer / half sequencing, the host grouping of routed
// rows and the MTP draft K/V batches; every call below is one kernel launch (or one runtime call / host query).
// The C++ `strata::prefill::Prefill` object still owns the buffers (init / carve / relayout, called by the serve lend
// in generate.cpp) and dispatches `run` / `draft_kv` to Rust when TALUS_FN_RUST=1 installed the callbacks.

/// run: 1 done, 0 failed (`err` set), -1 declined (an unported mode: the C++ path runs).
typedef int (*tk_qwen4exp_p2_run_fn)(void* user, void* prefill, const int64_t* tokens, int64_t n, int64_t pos0,
                                     char* err, size_t err_cap);
/// draft: 1 done, 0 declined (`why` set, `err` empty) or failed (`err` set), -1 not handled (the C++ path runs).
typedef int (*tk_qwen4exp_p2_draft_fn)(void* user, void* prefill, void* mtp, const float* r_rows,
                                       const int32_t* next_tokens, int64_t n, int64_t cell0, char* err,
                                       size_t err_cap, const char** why);
void tk_qwen4exp_p2_install(tk_qwen4exp_p2_run_fn run, tk_qwen4exp_p2_draft_fn draft, void* user);
int tk_qwen4exp_p2_installed(void);
/// sizeof of the ABI structs in this order: layout, qsa_info, weight, expert_fmt, stats, mtp_info, cvec, stager_job
void tk_qwen4exp_p2_abi_sizes(int64_t out[8]);

typedef struct tk_qwen4exp_p2_qsa_shapes {
    int64_t n_head, n_head_kv, head_dim, n_rot, idx_n_head, idx_dim, idx_block, idx_top_k, page_size;
} tk_qwen4exp_p2_qsa_shapes_t;
/// = strata::kernels::KvHostPools (8 pointers)
typedef struct tk_qwen4exp_p2_kv_pools {
    uint16_t* k_pool; uint16_t* v_pool; int8_t* k_q; int8_t* v_q; uint16_t* k_scale; uint16_t* v_scale;
    uint8_t* k_q4; uint8_t* v_q4;
} tk_qwen4exp_p2_kv_pools_t;
/// = strata::kernels::QsaAttnPools
typedef struct tk_qwen4exp_p2_attn_pools {
    const uint16_t* k_pool; const uint16_t* v_pool; const int8_t* k_q; const int8_t* v_q; const uint16_t* k_scale;
    const uint16_t* v_scale; const uint8_t* k_q4; const uint8_t* v_q4; const int32_t* page_table;
} tk_qwen4exp_p2_attn_pools_t;
/// = strata::kernels::QsaIndexerBuffers
typedef struct tk_qwen4exp_p2_idx_bufs {
    float* tail; float* dead; float* pooled; int32_t* block_pos;
} tk_qwen4exp_p2_idx_bufs_t;

/// One QSA state (a session's ordinal `ord`, or the drafter's).
typedef struct tk_qwen4exp_p2_qsa_info {
    int32_t kv_mode, kv_int8, kv_q4, kv_hybrid, kv_rot, host_present, fmt, pad0;
    int64_t n_pages, max_cells, idx_pooled_rows, n_slots;
    uint16_t* k_pool; uint16_t* v_pool; int8_t* k_q; int8_t* v_q; uint16_t* k_scale; uint16_t* v_scale;
    uint8_t* k_q4; uint8_t* v_q4; int32_t* page_table;
    tk_qwen4exp_p2_kv_pools_t host;
    tk_qwen4exp_p2_attn_pools_t attn;   ///< core::qsa_attn_pools(st)
    tk_qwen4exp_p2_idx_bufs_t idx;
} tk_qwen4exp_p2_qsa_info_t;

/// A resolved weight (LayerView::get): `ref` is the opaque WeightRef for hc_bf16_view.
typedef struct tk_qwen4exp_p2_weight {
    const void* ref; const void* data; int32_t bf16; int32_t hc_q8; int32_t native_type; int32_t pad0;
    const void* native_data; int64_t ne0, ne1;
} tk_qwen4exp_p2_weight_t;

/// A layer's routed-expert blob layout (ExpertLayout / NativeFmt) and the prompt path's plan for it.
typedef struct tk_qwen4exp_p2_expert_fmt {
    int32_t native, gu_type, d_type, mmq_layer, mmq_fo, native_supported, pad0, pad1;
    int64_t n_embd, n_ff;
    uint64_t gu_row, d_row, up_off, down_off, tail_off, blob_bytes;
} tk_qwen4exp_p2_expert_fmt_t;

/// = strata::prefill::PrefillStats (all 8-byte fields, in its order)
typedef struct tk_qwen4exp_p2_stats {
    int64_t tokens, chunks;
    double ms_total, ms_experts_host;
    int64_t experts_streamed, experts_dma, experts_resident, experts_cpu;
    double cpu_share, ms_ple;
} tk_qwen4exp_p2_stats_t;

/// Everything a run reads off the C++ Prefill object (its Impl), refreshed by `bind` before every run.
typedef struct tk_qwen4exp_p2_layout {
    // eligibility: the modes the Rust path does not run (the C++ path keeps them)
    int64_t stage_lb, stage_le;
    int32_t has_next, has_hand_in, has_peer, has_cpu_pool, has_embd_rows, has_on_chunk, has_on_stage_chunk,
            has_should_stop, metal, borrowed;
    // geometry / plan
    int32_t device, f16_io, fused_bufs, ring, fused_ring, mmq_any, mmq_fallback, mmq_built, n_layers, n_expert,
            cpu_share_on, peer_portable;
    int64_t T, T_max, cap, max_blocks, sel_batch, attn_batch, stream_all_min, max_blob;
    // the shape the prompt path is compiled for (prefill.cpp's constants, checked against the model by init)
    int64_t n_embd, hc, hc_lr, n_used, n_ff, conv_ch, value_dim, v_heads, n_head, n_head_kv, head_dim, idx_heads,
            idx_dim, gdn_floats, gdn_state_floats, ple_n_heads, ng_n_embd, ng_hc_dim;
    float eps; int32_t stage_ring, ring_max, pad3;
    const char* mmq_layer;   ///< [n_layers] 0/1
    const char* mmq_fo;      ///< [n_layers] 0/1
    const char* is_qsa;      ///< [n_layers] 0/1
    tk_qwen4exp_p2_qsa_shapes_t shapes;
    // streams
    void* cs; void* copy;
    // opaque C++ objects
    void* wt; void* g; void* ss; void* src; void* cache; void* gemm; void* mmq_ctx; void* ple_weights;
    void* ple_table; void* ple_consts; void* ple_scratch;
    const int32_t* host_res;
    tk_qwen4exp_p2_stats_t* stats;
    // session state
    float* gdn_state; int64_t gdn_ord0, gdn_alloc, qsa_ord0, n_qsa_states;
    int32_t* ple_prev;   ///< ss.ple_prev[2]
    int32_t ple_ready, ple_batch_ok, ple_direct, pad1;
    float* ple_hist;
    const uint16_t* ple_key_bf16; const void* ple_key_native_data; int32_t ple_key_native_type, pad2;
    const uint16_t* ple_value_bf16;
    // chunk buffers
    uint8_t* region; uint64_t region_bytes;
    float *emb, *R, *xn, *grs, *lo, *gated, *inj, *mixed, *bo;
    uint16_t *xn16, *lo16, *mixed_bf, *mixed_h, *xn16_lo, *lo16_lo, *mixed_bf_lo;
    float *qkv, *z, *ab, *gate, *beta, *hbuf, *y; uint16_t* y_h;
    float *Kc, *Vc, *Qf, *q, *idx_raw, *q_idx, *attn; uint16_t* attn_h;
    int32_t* steps_dev; int32_t* sel_ids; float* sel_scores; float* attn_scratch;
    float *logits, *w, *GU, *Dm, *sgate, *sup, *shared, *sg;
    int32_t *ids, *slot_dev, *src_dev;
    uint16_t *Xs, *Hh, *sh_h;
    void *Xq, *Hq; float* H;
    int32_t *ids_identity, *bounds_dev;
    uint8_t *grp_gu, *grp_d;
    float *Xscale, *Hscale, *grp_tail, *row_sd;
    int32_t* grp_host; int32_t* grp_dev; uint64_t grp_tk;
    uint16_t* dq_gu[2]; uint16_t* dq_d[2];
    uint8_t** stage_dev;   ///< [RING_MAX]
    void** copied;         ///< cudaEvent_t[RING_MAX]
    void** used;
    uint8_t* stage_live;   ///< bool[RING_MAX]
    int32_t* used_of;
    float* ple_emb; float* ple_norm;
    float* ple_emb_host[2]; int32_t ple_pageable[2]; void* ple_copied[2]; uint32_t* ple_rows[2];
    int32_t* tok_dev;
    tk_qwen4exp_p2_kv_pools_t stage;   ///< the KV streaming staging pool (identity layout)
    int32_t* ident_table;
} tk_qwen4exp_p2_layout_t;

// ---- the Prefill object (src/prefill/prefill.cpp, its qwen4exp-p2 block)
int tk_qwen4exp_p2_bind(void* prefill, tk_qwen4exp_p2_layout_t* out);
typedef struct tk_qwen4exp_p2_stager_job {
    const uint8_t* src; uint64_t bytes; int32_t from_source, l, e, pad0;
} tk_qwen4exp_p2_stager_job_t;
void tk_qwen4exp_p2_stager_start(void* prefill, const tk_qwen4exp_p2_stager_job_t* jobs, int64_t n);
const uint8_t* tk_qwen4exp_p2_stager_wait(void* prefill, int64_t j);
void tk_qwen4exp_p2_stager_issued_one(void* prefill, int64_t j, void* stream);
void tk_qwen4exp_p2_stager_finish(void* prefill);
int tk_qwen4exp_p2_steps_upload(void* prefill, int32_t* dst, const int32_t* src, uint64_t count, void* stream);
int tk_qwen4exp_p2_tok_upload(void* prefill, int32_t* dst, const int32_t* src, uint64_t count, void* stream);
int tk_qwen4exp_p2_upring_h2d(void* prefill, void* dst, const void* src, uint64_t bytes, void* stream);
int tk_qwen4exp_p2_upring_d2h_sync(void* prefill, void* dst, const void* src, uint64_t bytes, void* stream);
int tk_qwen4exp_p2_should_stop(void* prefill);
int tk_qwen4exp_p2_on_chunk(void* prefill, const float* r_rows, int64_t t, int64_t p0, char* err, size_t err_cap);
int tk_qwen4exp_p2_on_stage_chunk(void* prefill, int64_t done, char* err, size_t err_cap);
void* tk_qwen4exp_p2_ensure_mmq_ctx(void* prefill);

// ---- runtime (src/program/qwen4exp_p2.cpp)
int tk_qwen4exp_p2_set_device(int32_t device);
int tk_qwen4exp_p2_memcpy(void* dst, const void* src, uint64_t bytes, int32_t kind, void* stream);   // kind: cudaMemcpyKind
int tk_qwen4exp_p2_memset(void* dst, int32_t value, uint64_t bytes, void* stream);
int tk_qwen4exp_p2_event_create(void** ev, int32_t timing);
void tk_qwen4exp_p2_event_destroy(void* ev);
int tk_qwen4exp_p2_event_record(void* ev, void* stream);
int tk_qwen4exp_p2_event_sync(void* ev);
int tk_qwen4exp_p2_event_elapsed(float* ms, void* a, void* b);
int tk_qwen4exp_p2_stream_wait_event(void* stream, void* ev);
int tk_qwen4exp_p2_stream_sync(void* stream);
int tk_qwen4exp_p2_peek_error(char* err, size_t err_cap);
void tk_qwen4exp_p2_progress_at(const char* what, int64_t detail, int64_t chunk);
void tk_qwen4exp_p2_progress_beat(void);
void tk_qwen4exp_p2_pin_thread(void);
// ---- weights, embedding
int tk_qwen4exp_p2_weight(void* wt, int64_t layer, const char* suffix, tk_qwen4exp_p2_weight_t* out);
const uint16_t* tk_qwen4exp_p2_hc_bf16_view(const void* wref, void* stream, int32_t slot, int32_t cls);
void tk_qwen4exp_p2_hc_bf16_release(void* stream, int32_t slot, int32_t cls);
/// 1 native embed table, 2 the WeightTable's token_embd (gatherable), 0 neither (per-token rows); *out_rows = ne1
int tk_qwen4exp_p2_embed_kind(void* wt, void* g, int64_t* out_rows);
void tk_qwen4exp_p2_embed_gather(void* wt, const int32_t* tok_dev, int64_t n, float* out, void* stream);
int tk_qwen4exp_p2_embed_row(void* wt, void* g, int64_t token, float* out, void* stream, char* err, size_t err_cap);
// ---- GEMMs (strata::prefill::Gemm)
void tk_qwen4exp_p2_gemm_bf16(void* gemm, const uint16_t* x, const uint16_t* w, float* y, int64_t t, int64_t n,
                              int64_t k, int64_t ldy, float beta, int64_t ldx);
void tk_qwen4exp_p2_gemm_f16(void* gemm, const uint16_t* x, const uint16_t* w, float* y, int64_t t, int64_t n,
                             int64_t k);
void tk_qwen4exp_p2_gemm_native(void* gemm, const uint16_t* x, int32_t type, const void* w, float* y, int64_t t,
                                int64_t n, int64_t k, int64_t ldy, float beta, int64_t ldx);
int tk_qwen4exp_p2_gemm_bf16_hcd_exact(void* gemm, const uint16_t* x, int64_t ldx, const uint16_t* w, float* y,
                                       int64_t t, int64_t n, int64_t k);
// ---- hyper-connection (strata/prefill/kernels.hpp)
void tk_qwen4exp_p2_gr_broadcast(const float* e, float* r, int64_t t, void* stream);
void tk_qwen4exp_p2_gr_norm(const float* r, const float* w, float eps, float* xn, uint16_t* xn16, int64_t t,
                            void* stream, uint16_t* xn16_lo);
void tk_qwen4exp_p2_gr_norm_rs(const float* r, const float* w, float eps, float* rs, uint16_t* xn16, int64_t t,
                               void* stream, uint16_t* xn16_lo, int64_t ldx);
void tk_qwen4exp_p2_gr_mix_r(const float* r, const float* rs, const float* w, const float* gated, float* mixed,
                             uint16_t* mixed16, int64_t t, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo);
void tk_qwen4exp_p2_gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t t,
                           void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo);
int tk_qwen4exp_p2_gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* r, const float* rs,
                            const float* w, float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t t,
                            void* stream);
void tk_qwen4exp_p2_gr_silu(const float* lo, uint16_t* lo16, int64_t t, void* stream, uint16_t* lo16_lo);
void tk_qwen4exp_p2_gr_write(float* r, const float* bo, const float* inj, int64_t inj_ld, int64_t t, void* stream);
void tk_qwen4exp_p2_gr_write_norm_rs(float* r, const float* bo, const float* inj, int64_t inj_ld, const float* w_next,
                                     float eps, float* rs, uint16_t* xn16, int64_t t, void* stream,
                                     uint16_t* xn16_lo, int64_t ldx);
void tk_qwen4exp_p2_gr_write_cvec_norm_rs(float* r, const float* bo, const float* inj, int64_t inj_ld,
                                          const float* v_l, const float* s_l, const int* on, int32_t mode,
                                          const float* w_next, float eps, float* rs, uint16_t* xn16, int64_t t,
                                          void* stream, uint16_t* xn16_lo, int64_t ldx);
int tk_qwen4exp_p2_pf_hcdown_bf16(const uint16_t* xn16, int64_t ldx, const uint16_t* wd, const uint16_t* wi,
                                  int64_t lr, int64_t hc, float* lo, float* inj, int64_t t, int64_t d, void* stream);
/// the control vector: 1 when layer `l` is steered; tables (null when absent)
typedef struct tk_qwen4exp_p2_cvec {
    int32_t covers, mode, tables_ok, pad0; int64_t n_embd, hc; const float* dir; const float* s; const int* on;
} tk_qwen4exp_p2_cvec_t;
void tk_qwen4exp_p2_cvec_info(int64_t layer, tk_qwen4exp_p2_cvec_t* out);
void tk_qwen4exp_p2_cvec_apply(float* r, int64_t layer, int64_t t, int64_t ld, void* stream);
// ---- PLE
void tk_qwen4exp_p2_ple_ngram_rows(void* consts, const int32_t* tokens, const int32_t* prev, int32_t n,
                                   uint32_t* out);
int tk_qwen4exp_p2_ple_gather_batch(void* table, const uint32_t* rows, uint64_t n, float* out, char* err,
                                    size_t err_cap);
int tk_qwen4exp_p2_ple_postops_batch(void* weights, float* key, float* hidden, const float* value, float* hist,
                                     float* qn, float* gated, float* gate, int32_t n, void* stream, char* err,
                                     size_t err_cap);
int tk_qwen4exp_p2_ple_block(void* weights, const float* emb, const float* hidden, const float* hist,
                             float* normalized, float* result, void* scratch, void* stream, char* err,
                             size_t err_cap);
void tk_qwen4exp_p2_ple_history_advance(float* hist, const float* normalized, void* stream);
void tk_qwen4exp_p2_to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
void tk_qwen4exp_p2_to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo);
// ---- GDN
void tk_qwen4exp_p2_gdn_gates(const float* ab, const float* dt, const float* a, float* gate, float* beta, int64_t t,
                              void* stream);
void tk_qwen4exp_p2_gdn_conv(float* hist, const float* qkv, const float* w, float* h, int64_t t, float eps,
                             void* stream);
void tk_qwen4exp_p2_gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                                   const float* gamma, float eps, float* y, uint16_t* y16, int64_t t, void* stream,
                                   int64_t ld16);
// ---- QSA
int tk_qwen4exp_p2_qsa_info(void* ss, int64_t ord, tk_qwen4exp_p2_qsa_info_t* out);
void tk_qwen4exp_p2_rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps,
                             void* stream);
void tk_qwen4exp_p2_rope(float* x, int64_t t, int64_t heads, int64_t dim, int64_t ld, int64_t pos0, void* stream);
void tk_qwen4exp_p2_split_q(const float* q_full, float* q, int64_t t, void* stream);
void tk_qwen4exp_p2_gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t t, void* stream,
                              int64_t ld16);
void tk_qwen4exp_p2_fwht256(float* x, int64_t rows, void* stream);
void tk_qwen4exp_p2_kv_stage_from_host(const tk_qwen4exp_p2_attn_pools_t* stage,
                                       const tk_qwen4exp_p2_kv_pools_t* host, int32_t fmt, int64_t n_blocks,
                                       const tk_qwen4exp_p2_qsa_shapes_t* s, void* stream);
void tk_qwen4exp_p2_kv_unstage_to_host(const tk_qwen4exp_p2_attn_pools_t* stage,
                                       const tk_qwen4exp_p2_kv_pools_t* host, int32_t fmt, int64_t b0, int64_t b1,
                                       const tk_qwen4exp_p2_qsa_shapes_t* s, void* stream);
void tk_qwen4exp_p2_kv_append(const float* k, const float* v, int64_t t, int64_t pos0, const int32_t* page_table,
                              int64_t page_size, uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q,
                              uint16_t* k_scale, uint16_t* v_scale, void* stream,
                              const tk_qwen4exp_p2_kv_pools_t* host, const tk_qwen4exp_p2_kv_pools_t* stage);
void tk_qwen4exp_p2_kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t t,
                                 const float* k, const float* v, const tk_qwen4exp_p2_qsa_shapes_t* s, void* stream,
                                 const tk_qwen4exp_p2_kv_pools_t* host, const tk_qwen4exp_p2_kv_pools_t* stage);
int tk_qwen4exp_p2_indexer_append_batch(const float* raw, int64_t n, int64_t p0, const float* gamma, float eps,
                                        const tk_qwen4exp_p2_idx_bufs_t* b, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                        int64_t max_cells, void* stream, char* err, size_t err_cap);
int tk_qwen4exp_p2_indexer_append(const float* raw, const int32_t* rel_pos_dev, const float* gamma, float eps,
                                  const tk_qwen4exp_p2_idx_bufs_t* b, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                  int64_t max_cells, void* stream, char* err, size_t err_cap);
int tk_qwen4exp_p2_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                                   int64_t nq, int64_t max_blocks, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                   float* scores, void* stream, int64_t active);
void tk_qwen4exp_p2_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                                 int64_t nq, int64_t max_blocks, const tk_qwen4exp_p2_qsa_shapes_t* s, float* scores,
                                 void* stream, int64_t active);
void tk_qwen4exp_p2_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                               int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s, int32_t* ids, void* stream,
                               int64_t active);
int tk_qwen4exp_p2_fnp_tile_attn(const float* q, const tk_qwen4exp_p2_attn_pools_t* pools, const int32_t* ids,
                                 const int32_t* steps, int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s, float* attn,
                                 int64_t n_q, void* stream);
int tk_qwen4exp_p2_prompt_attn_batch(const float* q, const tk_qwen4exp_p2_attn_pools_t* pools, const int32_t* ids,
                                     const int32_t* steps, int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                     float* attn, int64_t n_q, void* stream);
void tk_qwen4exp_p2_decode_attn_batch(const float* q, const tk_qwen4exp_p2_attn_pools_t* pools, const int32_t* ids,
                                      const int32_t* steps, int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s,
                                      float* scratch, float* attn, int64_t n_q, void* stream);
uint64_t tk_qwen4exp_p2_decode_attn_scratch_floats(int64_t cap, const tk_qwen4exp_p2_qsa_shapes_t* s);
// ---- MoE
void tk_qwen4exp_p2_route(const float* logits, int32_t* ids, float* w, int64_t t, int64_t n_expert, void* stream);
void tk_qwen4exp_p2_swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream);
void tk_qwen4exp_p2_swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream);
void tk_qwen4exp_p2_copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream);
void tk_qwen4exp_p2_gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width,
                                  void* stream);
void tk_qwen4exp_p2_moe_combine(const float* d, const int32_t* slot, const float* w, const float* shared,
                                const float* sg, float* bo, int64_t t, void* stream, const float* row_sd);
int tk_qwen4exp_p2_expert_fmt(int64_t layer, tk_qwen4exp_p2_expert_fmt_t* out);
int tk_qwen4exp_p2_fused_native_supported(int32_t gu_type, int32_t d_type);
int tk_qwen4exp_p2_fused_enabled(void);
void tk_qwen4exp_p2_fused_quantize_act(const float* x, int64_t rows, int64_t cols, void* xa, void* stream,
                                       int32_t native);
void tk_qwen4exp_p2_fused_group(const int32_t* ids, int64_t n, int32_t k, int32_t n_expert, void* scratch,
                                int32_t* slot, int32_t* src, void* stream);
/// fused::experts_native (native = 1, `fmt` = the layer's geometry) / fused::experts (native = 0) over experts
/// [e0, e1) whose blobs are `blobs[0 .. e1 - e0)`
void tk_qwen4exp_p2_fused_experts(int32_t native, const tk_qwen4exp_p2_expert_fmt_t* fmt, int32_t e0, int32_t e1,
                                  const uint8_t* const* blobs, int32_t n_expert, int64_t n, const void* scratch,
                                  const void* xa, const int32_t* src, void* ha, float* dm, void* stream);
int32_t tk_qwen4exp_p2_fused_max_batch(void);
uint64_t tk_qwen4exp_p2_mmq_matrix_bytes(int32_t type, int64_t rows, int64_t cols);
int tk_qwen4exp_p2_mmq_fp4_activations(int32_t type);
int tk_qwen4exp_p2_mmq_fits(int32_t type, int64_t rows);
void tk_qwen4exp_p2_mmq_quantize(const float* x, const int32_t* ids, void* xq, int32_t type, int64_t cols, int64_t ld,
                                 int64_t rows, void* stream, float* yscale);
void tk_qwen4exp_p2_mmq_gather_native(const void* gate, const void* up, uint64_t gu_half, const void* down,
                                      uint64_t d_bytes, void* gu_dst, void* d_dst, void* stream, const void* tail,
                                      void* tail_dst, uint64_t zero_bytes);
int tk_qwen4exp_p2_mmq_gather_native_group(const uint8_t* const* blobs, int32_t first, int32_t n, uint64_t up_off,
                                           uint64_t gu_half, uint64_t down_off, uint64_t d_bytes, void* gu_dst,
                                           uint64_t gu_stride, void* d_dst, uint64_t d_stride, void* stream,
                                           uint64_t tail_off, void* tail_dst, uint64_t zero_bytes);
void tk_qwen4exp_p2_mmq_gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream);
/// run_mmq_product: tp_moe's moe_gemm (TALUS_PF_MOE, default on) where it covers the product, else MMQ (`mmq_ctx`)
void tk_qwen4exp_p2_mmq_product(void* mmq_ctx, const void* w, int32_t type, int64_t w_rows, int64_t w_cols,
                                uint64_t expert_bytes, int32_t n, const void* xq, const int32_t* bounds,
                                const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst, int64_t ld_dst,
                                const float* y_scale, int32_t allow_tp_moe, void* stream);
void tk_qwen4exp_p2_mmq_swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, int32_t interleaved,
                               void* stream);
void tk_qwen4exp_p2_mmq_swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds,
                                      int32_t n, const float* tails, int64_t row0, void* stream);
void tk_qwen4exp_p2_mmq_down_row_scales(float* sd, const int32_t* bounds, int32_t n, const float* tails,
                                        int64_t nrows, void* stream);
void tk_qwen4exp_p2_mmq_iota(int32_t* dst, int64_t n, void* stream);
uint64_t tk_qwen4exp_p2_mmq_q8_bytes(int64_t rows, int64_t cols);
void tk_qwen4exp_p2_iq_dequant_gu_f16(int32_t type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd,
                                      uint16_t* dst, void* stream, const float* scale);
void tk_qwen4exp_p2_iq_dequant_f16(int32_t type, const void* src, int64_t n, uint16_t* dst, void* stream,
                                   const float* scale);
void tk_qwen4exp_p2_blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream);
// ---- expert source / cache (host queries)
int tk_qwen4exp_p2_src_transient(void* src, int64_t l, int64_t e);
int tk_qwen4exp_p2_src_pinned(void* src, int64_t l, int64_t e);
const uint8_t* tk_qwen4exp_p2_src_blob_stable(void* src, int64_t l, int64_t e);
const uint8_t* tk_qwen4exp_p2_cache_device_slot(void* cache, int32_t slot);
// ---- MTP draft layer (E-9, draft_kv)
typedef struct tk_qwen4exp_p2_mtp_info {
    const float* w_ne; const void* w_fe; const float* w_nh; const void* w_fh; const float* w_hn;
    const uint16_t* w_dn; const uint16_t* w_up; const void* w_k; const void* w_v; const float* w_kn;
    int64_t first_needed; int32_t device, hnorm_per_stream;
    tk_qwen4exp_p2_qsa_info_t kv;
} tk_qwen4exp_p2_mtp_info_t;
int tk_qwen4exp_p2_mtp_info(void* mtp, tk_qwen4exp_p2_mtp_info_t* out);
int tk_qwen4exp_p2_mtp_idle(void* mtp, char* err, size_t err_cap);
void tk_qwen4exp_p2_mtp_add_ms_prefill(void* mtp, double ms);
void tk_qwen4exp_p2_rms_norm_grouped(const float* x, const float* gamma, float* y, int32_t n, int32_t groups,
                                     int32_t rows, float eps, void* stream);
void tk_qwen4exp_p2_add_streams_broadcast(const float* h, const float* e, float* r, int64_t n_embd, int32_t hc,
                                          int32_t n_tok, void* stream);
void tk_qwen4exp_p2_mmq_run(void* mmq_ctx, const void* w, int32_t type, int64_t w_rows, int64_t w_cols,
                            uint64_t expert_bytes, int32_t n, const void* xq, const int32_t* bounds, const int32_t* ids,
                            int64_t total_rows, int64_t max_rows, float* dst, int64_t ld_dst, void* stream);
// ---- qwen4exp-p2 end ----

// ---- qwen4exp-r1 begin ----
// qwen4exp-r1 (the residual of Piece 1): the expert source (strata::core::FileExpertSource, a native pack's GGUF shards
// read in place) and the PLE table reader (strata::kernels::PleTable / strata::ngram::PleReader) are Rust's
// (rust/crates/talus-model/src/qwen4exp/weights/{source,ple_reader,r1}.rs). The C++ objects adopt a Rust handle and
// forward every method to these tables (FileExpertSource::adopt_rust, PleTable::adopt_rust); src/program/qwen4exp_r1.cpp.

// strata::kernels::cpu::expert_layout() as the Rust source reads it (pointers valid until the next call)
typedef struct tk_qwen4exp_r1_layout {
    int32_t native;
    int64_t n_layers, n_expert;
    size_t n_per_layer;                       // length of every per-layer array below
    const uint64_t* fmt_up_off;
    const uint64_t* fmt_down_off;
    const uint64_t* fmt_bytes;
    const int32_t* fmt_gu_type;
    const int32_t* fmt_d_type;
    const int64_t* fmt_n_embd;
    const int64_t* fmt_n_ff;
    const uint64_t* offset;
    const uint64_t* bytes;
    const uint64_t* gguf_off;                 // 3 x n_layers (or empty)
    size_t n_gguf_off;
    const char* const* gguf_file;             // 3 x n_layers (or empty)
    size_t n_gguf_file;
    uint64_t max_blob, total, canonical_blob;
} tk_qwen4exp_r1_layout_t;

// what the Rust source and reader ask of the C++ side
typedef struct tk_qwen4exp_r1_services {
    size_t struct_size;
    int (*expert_layout)(tk_qwen4exp_r1_layout_t* out);
    void (*pin_aux_thread)(void);                                    // strata::aux_cpus::pin_current_thread
    void (*fs_hold)(void* foresight, int64_t layer, int32_t expert, int hold);   // ForesightSwap::hold / unhold
    int (*ple_prefetch_enabled)(void);                               // strata::kernels::ple_prefetch_enabled
} tk_qwen4exp_r1_services_t;
const tk_qwen4exp_r1_services_t* tk_qwen4exp_r1_services(void);

// tk_qwen4exp_r1_source_vt.stat
enum {
    TK_R1_S_MAPPED = 0, TK_R1_S_BLOBS, TK_R1_S_PINNED_BYTES, TK_R1_S_RESIDENT_BYTES, TK_R1_S_COMPLEMENT_PINNED,
    TK_R1_S_COMPLEMENT_READY, TK_R1_S_LOCKED_BYTES, TK_R1_S_RESIDENT_LENT_SLOTS, TK_R1_S_GGUF_MODE,
    TK_R1_S_EXCHANGE_CAPACITY, TK_R1_S_EXCHANGE_PINNED, TK_R1_S_EXCHANGES, TK_R1_S_FILE_READS, TK_R1_S_RAM_READS,
    TK_R1_S_FILE_READ_BYTES, TK_R1_S_FILE_BLOB_BYTES, TK_R1_S_FILE_US, TK_R1_S_UNBUFFERED, TK_R1_S_EXPERT_BYTES,
    TK_R1_S_WARMS, TK_R1_S_WARMED_HITS, TK_R1_S_WARMED, TK_R1_S_DIRECT_FALLBACKS, TK_R1_S_READS
};
// tk_qwen4exp_r1_source_vt.query / .pointer / .unbuffered / .exchange_op
enum { TK_R1_Q_PINNED = 0, TK_R1_Q_TRANSIENT, TK_R1_Q_HAS_RESIDENT, TK_R1_Q_PCIE_LAYER };
enum { TK_R1_P_DEVICE_ALIAS = 0, TK_R1_P_RESIDENT_BLOB, TK_R1_P_EXCHANGE_BUFFER };
enum { TK_R1_U_SET = 0, TK_R1_U_RECHECK, TK_R1_U_PROMPT_DIRECT };
enum { TK_R1_X_PREPARE_OVERRIDES = 0, TK_R1_X_COMMIT, TK_R1_X_SOFT_COMMIT, TK_R1_X_COMMIT_COPIES, TK_R1_X_COMMIT_FLIP };

// the GPU expert cache as pin_cache_complement reads it
typedef struct tk_qwen4exp_r1_cache_view {
    int32_t valid;
    int64_t fills, resident, slots;
    const int32_t* slot_of;                   // n_layers x n_expert: slot or -1
    size_t n_slot_of;
} tk_qwen4exp_r1_cache_view_t;

typedef struct tk_qwen4exp_r1_source_vt {
    size_t struct_size;
    void (*destroy)(void* h);
    int (*open)(void* h, const char* pack_dir, const char* gguf, int64_t n_layers, int64_t n_expert, char* err,
                size_t cap);
    void (*close)(void* h);
    uint64_t (*stat)(void* h, int what);
    size_t (*input)(void* h, size_t i, const char** role, const char** path);   // model_inputs(): the count
    const uint8_t* (*blob)(void* h, int64_t layer, int64_t expert, int stable);
    void (*note_async_read)(void* h, const uint8_t* blob, void* stream);
    int (*query)(void* h, int what, int64_t layer, int64_t expert);
    const uint8_t* (*pointer)(void* h, int what, int64_t layer, int64_t expert);
    double (*cached_share)(void* h, int64_t samples);
    int (*copy_blob)(void* h, int64_t layer, int64_t expert, uint8_t* dst);
    int (*copy_blobs)(void* h, const int32_t* layers, const int32_t* experts, uint8_t* const* dst, size_t n);
    void (*begin_layer)(void* h, int64_t layer);
    void (*prefetch)(void* h, int64_t layer, const int64_t* experts, int64_t n);
    void (*warm)(void* h, int64_t layer, const int64_t* experts, int64_t n);
    void (*prefetch_pairs)(void* h, const int32_t* pairs, int64_t n);
    int (*advise_pairs)(void* h, const int32_t* pairs, int64_t n);
    int (*unbuffered)(void* h, int what, uint64_t ram_bytes, char* why, size_t cap);
    int (*pin_cache_complement)(void* h, const tk_qwen4exp_r1_cache_view_t* cache, int pin, const int32_t* add_pairs,
                                size_t n_add, int64_t lend_from_slot, uint64_t headroom, uint64_t budget,
                                const int32_t* rank, size_t n_rank, int has_rank, char* err, size_t cap);
    void (*stage_lend_regions)(void* h, const int32_t* pairs, const size_t* lens, size_t n_regions);
    int (*reserve_exchanges)(void* h, int64_t n, char* err, size_t cap);
    int (*stage_exchange)(void* h, int64_t layer, int64_t in, int64_t out, int64_t q);
    int64_t (*exchange_op)(void* h, int what);
    void (*set_foresight)(void* h, void* foresight);
    void (*set_fetch_threads)(void* h, int n);
} tk_qwen4exp_r1_source_vt_t;

// tk_qwen4exp_r1_ple_vt.stat
enum { TK_R1_PLE_IS_OPEN = 0, TK_R1_PLE_LOCKED, TK_R1_PLE_ROWS, TK_R1_PLE_BYTES_READ, TK_R1_PLE_MMAP, TK_R1_PLE_FORMAT };

typedef struct tk_qwen4exp_r1_ple_vt {
    size_t struct_size;
    void (*destroy)(void* h);
    int (*open)(void* h, const char* path, const tk_qwen4exp_p1_ple_io_t* io, char* err, size_t cap);
    void (*close)(void* h);
    uint64_t (*stat)(void* h, int what);
    int (*issue)(void* h, const uint32_t* rows16);
    int (*collect)(void* h, float* out2560, char* err, size_t cap);
    void (*prefetch_rows)(void* h, const uint32_t* rows16);
    void (*wait_prefetches)(void* h);
    int (*gather_batch)(void* h, const uint32_t* rows, size_t n_tokens, float* out, char* err, size_t cap);
    void (*gather)(void* h, const uint32_t* rows16, float* out2560);
    void (*read_row)(void* h, uint32_t row, float* out160);
    void (*set_injected_delay_us)(void* h, double us);
    size_t (*io_report)(void* h, char* out, size_t cap);
} tk_qwen4exp_r1_ple_vt_t;

// Rust registers these (weights.rs); null `*vt` = declined (`why` says why) and the C++ object runs as before
typedef struct tk_qwen4exp_r1_rust_hooks {
    size_t struct_size;
    void* (*source_create)(const char* pack_dir, const char* gguf, const tk_qwen4exp_r1_source_vt_t** vt, char* why,
                           size_t cap);
    void* (*ple_create)(const tk_qwen4exp_r1_ple_vt_t** vt, char* why, size_t cap);
    // Rust owns what Piece 1 allocated for the C++ objects that adopted it (ExpertCache / NativeDense / NativeHead /
    // NativeEmbed borrow it while this factory is live): freed here, once the engine's C++ objects are gone
    void (*release_all)(void);
} tk_qwen4exp_r1_rust_hooks_t;
void tk_qwen4exp_r1_set_rust_hooks(const tk_qwen4exp_r1_rust_hooks_t* hooks);
const tk_qwen4exp_r1_rust_hooks_t* tk_qwen4exp_r1_rust_hooks(void);
// ---- qwen4exp-r1 end ----
// ---- qwen4exp-r2 begin ----
// qwen4exp-r2: the prompt path's buffer setup in Rust (rust/.../qwen4exp/prefill/setup.rs): the sizing statics of
// strata::prefill::Prefill (bytes_needed*, the ring rules and their state), its carve / relayout, and init's device
// half.  The C++ statics and Prefill::init / relayout dispatch to the hooks below once Rust installed them
// (TALUS_FN_RUST=1; TALUS_FN_R2=0 keeps the C++ code).  The C++ object adopts what Rust laid out.
typedef struct tk_qwen4exp_r2_pf_hooks {
    void* user;
    uint64_t (*bytes_needed)(void* user, const void* g, const void* ss, int64_t chunk, int32_t owned_pages, int32_t src);
    uint64_t (*bytes_needed_no_ring)(void* user, const void* g, const void* ss, int64_t chunk, int32_t src);
    int64_t (*ring_slots_for)(void* user, int64_t chunk);
    int64_t (*ring_default_slots)(void* user);
    int64_t (*ring_max_slots)(void* user);
    int64_t (*ring_cap_for)(void* user, int64_t old_chunk);
    int64_t (*stream_all_min)(void* user);
    int32_t (*ring_bytes_enabled)(void* user);
    double (*pinned_share)(void* user);
    void (*set_pinned_share)(void* user, double share);
    void (*set_ring_override)(void* user, int32_t slots);
    void (*set_ring_budget)(void* user, int32_t slots, int64_t small_max);
    void (*arm_cpu_share)(void* user, int32_t applies, int32_t by_default);
    /// 1 done, 0 failed (`err` set), -1 declined (the C++ init / relayout runs)
    int (*init)(void* user, void* prefill, const void* wt, const void* g, void* ss, void* src, const void* cache,
                const int32_t* host_res, int64_t chunk, void* stream, void* borrow, uint64_t borrow_bytes,
                int32_t borrow_spill, char* err, size_t err_cap);
    int (*relayout)(void* user, void* prefill, int64_t chunk, void* borrow, uint64_t borrow_bytes, char* err,
                    size_t err_cap);
} tk_qwen4exp_r2_pf_hooks_t;
/// src/prefill/prefill.cpp (its qwen4exp-r2 block): registers the hooks (null: back to the C++ code)
void tk_qwen4exp_r2_pf_install(const tk_qwen4exp_r2_pf_hooks_t* hooks);

/// Process facts the prompt path's sizing reads (the expert layout, the build, the current device).
typedef struct tk_qwen4exp_r2_pf_facts {
    int32_t is_hip, layout_native, fused_enabled, peer_portable, mmq_built, prompt_f16, pad0, pad1;
    int64_t lay_n_layers, n_fmt, max_blob, ng_hc_dim, step_count, ring_max, gemm_scratch_elems, gemm_ws_bytes;
} tk_qwen4exp_r2_pf_facts_t;
void tk_qwen4exp_r2_pf_facts(tk_qwen4exp_r2_pf_facts_t* out);
/// The model / session facts (geometry for init's checks, the primary QSA state for the sizes).
typedef struct tk_qwen4exp_r2_pf_session {
    int64_t n_layers, n_expert, n_embd, hc, hc_lr, n_ff, n_ff_shexp, conv_ch, value_dim, v_heads, n_head, n_head_kv,
            head_dim, head_dim_v, idx_heads, idx_dim, k;
    double rms_eps;
    int64_t cap, max_cells, idx_block, attn_scratch_floats, n_pages, page_size, kv_q4_bytes_per_head;
    int32_t kv_mode, kv_hybrid, kv_q4, kv_int8;
} tk_qwen4exp_r2_pf_session_t;
void tk_qwen4exp_r2_pf_session(const void* g, const void* ss, tk_qwen4exp_r2_pf_session_t* out);
uint64_t tk_qwen4exp_r2_fused_group_bytes(int64_t n, int32_t n_expert);
uint64_t tk_qwen4exp_r2_fused_act_bytes(int64_t rows, int64_t cols);
/// runtime calls: 1 ok
int tk_qwen4exp_r2_malloc(void** out, uint64_t bytes);
void tk_qwen4exp_r2_free(void* p);
int tk_qwen4exp_r2_memset_sync(void* dst, int32_t value, uint64_t bytes);
int tk_qwen4exp_r2_memcpy_h2d_sync(void* dst, const void* src, uint64_t bytes);
int tk_qwen4exp_r2_host_alloc_mapped(void** host, void** dev, uint64_t bytes, int32_t portable);
void tk_qwen4exp_r2_free_host(void* p);
int tk_qwen4exp_r2_stream_create(void** out);
int tk_qwen4exp_r2_get_device(int32_t* out);
void tk_qwen4exp_r2_mem_info(uint64_t* free_b, uint64_t* total_b);
void tk_qwen4exp_r2_clear_error(void);

/// The carved buffers of a chunk (Prefill::carve's), adopted by the object.  `owned`: device allocations the object
/// frees with its own (owned buffers, a spilled region).
typedef struct tk_qwen4exp_r2_pf_bufs {
    int64_t T, cap, max_blocks;
    int32_t ring, fused_bufs;
    uint8_t* region; uint64_t region_bytes;
    float *emb, *R, *xn, *grs; uint16_t* xn16; float* lo; uint16_t* lo16; float *gated, *inj, *mixed;
    uint16_t *mixed_bf, *mixed_h; float* bo; uint16_t *xn16_lo, *lo16_lo, *mixed_bf_lo; int32_t* steps_dev;
    float *qkv, *z, *ab, *gate, *beta, *hbuf, *y; uint16_t* y_h;
    float *Kc, *Vc, *Qf, *q, *idx_raw, *q_idx, *attn; uint16_t* attn_h; int32_t* sel_ids; float* sel_scores;
    float* attn_scratch;
    float *logits, *w; int32_t *ids, *slot_dev, *src_dev; uint16_t* Xs; float* GU; uint16_t* Hh; float* Dm;
    float *sgate, *sup; uint16_t* sh_h; float *shared, *sg; void* Xq; float* H; void* Hq;
    float *Xscale, *Hscale, *grp_tail, *row_sd;
    uint16_t* dq_gu[2]; uint16_t* dq_d[2];
    int32_t *ids_identity, *bounds_dev; uint8_t *grp_gu, *grp_d;
    uint8_t* const* stage_dev;   ///< [ring]
    float* ple_emb; float* ple_norm;
    tk_qwen4exp_p2_kv_pools_t stage;
    void* const* owned; int64_t n_owned;
} tk_qwen4exp_r2_pf_bufs_t;
/// init's device half, adopted by the object (T_max, the loan flags, the stream / events / tables Rust made)
typedef struct tk_qwen4exp_r2_pf_init {
    int64_t T_max;
    int32_t borrowed, borrow_spill, device, n_events;
    const void* wt; const void* g; void* ss; void* src; const void* cache; const int32_t* host_res;
    void* cs; void* copy;
    void* const* copied; void* const* used;   ///< [n_events] (ring_cap())
    int32_t* tok_dev;
    int32_t* ident_table;
    int32_t* grp_host; int32_t* grp_dev; uint64_t grp_n, grp_tk;
    void* const* owned; int64_t n_owned;
} tk_qwen4exp_r2_pf_init_t;
/// prefill.cpp (qwen4exp-r2 block): the object's side
int tk_qwen4exp_r2_pf_stage_ok(void* prefill, int64_t n_layers);   ///< 1: one stage (no split, no helper, no peer)
int tk_qwen4exp_r2_pf_adopt_init(void* prefill, const tk_qwen4exp_r2_pf_init_t* in);
/// the host resources init keeps in C++ (the Stager threads, the pinned uploads, the host vectors, the PLE host
/// buffers): 1 ok, 0 failed
int tk_qwen4exp_r2_pf_init_host(void* prefill, int64_t chunk);
/// Gemm::init_external on the carved scratch, then the stage's FP16 prompt decision (`f16_io`: prompt_f16()).
int tk_qwen4exp_r2_pf_gemm_init(void* prefill, void* stream, uint16_t* scratch, int64_t scratch_elems, void* ws,
                                uint64_t ws_bytes, int32_t* f16_io, char* err, size_t err_cap);
void tk_qwen4exp_r2_pf_gemm_rebind(void* prefill, uint16_t* scratch, int64_t scratch_elems, void* ws, uint64_t ws_bytes);
int tk_qwen4exp_r2_pf_adopt_bufs(void* prefill, const tk_qwen4exp_r2_pf_bufs_t* b);
/// What relayout reads off the object: 1 initialized
typedef struct tk_qwen4exp_r2_pf_state {
    int32_t borrowed, borrow_spill, has_src, f16_io;
    int64_t T, T_max;
    void* cs; void* copy; void* kv_copy;
    const void* g; const void* ss;
} tk_qwen4exp_r2_pf_state_t;
int tk_qwen4exp_r2_pf_state(void* prefill, tk_qwen4exp_r2_pf_state_t* out);
/// TALUS_FN_R2_CHECK: the C++ statics with the hooks bypassed: [bytes_needed, owned, no_ring, ring_slots_for,
/// ring_default, ring_max, ring_cap_for(chunk), stream_all_min, ring_bytes_enabled]; and the C++ carve on a fake base
void tk_qwen4exp_r2_pf_cpp_statics(const void* g, const void* ss, int64_t chunk, int32_t src, uint64_t out[9]);
int tk_qwen4exp_r2_pf_cpp_carve(void* prefill, int64_t chunk, uint8_t* fake_base, uint64_t cap,
                                tk_qwen4exp_p2_layout_t* out);
// ---- qwen4exp-r2 end ----

// ---- qwen4exp-r2v begin ----
// qwen4exp-r2 (Piece 4 residuals): Verifier::init, MtpDrafter::load / bind and Verifier::set_logit_bias run in Rust
// (rust/.../qwen4exp/verify_mtp/setup.rs) when installed (TALUS_FN_RUST=1; TALUS_FN_R2V=0 keeps the C++ bodies).  Rust
// makes every allocation, stream and event and lays out the arenas; the C++ objects adopt the result (their destructors
// free it).  Each hook returns 1 done, 0 declined (the C++ body runs), -1 failed (err).  Impl: src/program/qwen4exp_r2v.cpp.
typedef struct tk_qwen4exp_r2v_ops {
    void* user;
    int (*ver_init)(void* user, void* ver, char* err, size_t err_cap);
    int (*ver_set_logit_bias)(void* user, void* ver, const float* bias, int64_t n, char* err, size_t err_cap);
    int (*mtp_load)(void* user, void* mtp, int64_t window, char* err, size_t err_cap);
    int (*mtp_bind)(void* user, void* mtp, char* err, size_t err_cap);
} tk_qwen4exp_r2v_ops_t;
void tk_qwen4exp_r2v_install(const tk_qwen4exp_r2v_ops_t* ops);

// What Verifier::init needs to know (the object's settings after its C++ prologue, the geometry, the kernels' sizes).
typedef struct tk_qwen4exp_r2v_ver_facts {
    int32_t eligible;         // one whole-model stage: lb 0, le -1 / n_layers, no hand-off, no next stage, no set_stream
    int32_t max_t, layer_ok, fused_gr_ok, always_publish, peer_portable, is_gfx12, hip_gfx906, device;
    int32_t k, ssm_state_size, ssm_d_conv, n_layers, n_expert, n_embd, hc, hc_lr, conv_ch, value_dim, v_heads, n_head,
        head_dim, n_head_kv, idx_q_heads, idx_key_dim, n_ff, n_gdn, n_qsa;
    int32_t verify_max_t, cpu_maxt, step_count, nonfinite_blocks, ng_hist, ng_hc_dim;
    int64_t n_vocab;          // output.weight's rows, -1: missing
    int64_t max_cells, idx_block, cap, attn_scratch_floats;
    uint64_t max_blob, q8_1_bytes, q8_1_il_bytes, hit_grouped_scratch, native_expert_scratch, argmax_scratch;
    const int32_t* d_res;
    const int32_t* h_res;
    const uint8_t* cache_base;
    const uint64_t* slot_off;
    int64_t n_slots, blob;
    char why[256];            // layer_verify_compatible's reason
} tk_qwen4exp_r2v_ver_facts_t;
int tk_qwen4exp_r2v_ver_facts(void* ver, tk_qwen4exp_r2v_ver_facts_t* out);

#define TK_R2V_VER_MAPPED 17   // the mapped staging in init's order, the plan block last
#define TK_R2V_VER_ARENA 63    // the device arena's buffers in init's carve order
typedef struct tk_qwen4exp_r2v_ver_state {
    void* mapped_h[TK_R2V_VER_MAPPED];
    void* mapped_d[TK_R2V_VER_MAPPED];
    int64_t plan_i32;
    int32_t* sink_counts; int32_t* sink_start; int32_t* sink_dst; int32_t* sink_tok;
    unsigned long long* sink_ptr; unsigned long long* sink_ptr2; int32_t* sink_start2;
    int64_t sink_cap, staging_cap;
    void* arena;
    uint64_t arena_bytes;
    void* arena_ptrs[TK_R2V_VER_ARENA];
    int64_t n_vocab, cap, max_blocks, attn_scratch_floats, le;
    void *copy, *cs, *sh_cs, *commit_done, *ev_fork, *ev_join;
    int32_t df_branch;
    void* df_fork;
    void* df_side[2];
    void* df_join[2];
    int32_t all_resident, device_plan;
    void *qcnt, *skip, *slot_off_d;
} tk_qwen4exp_r2v_ver_state_t;
void tk_qwen4exp_r2v_ver_adopt(void* ver, const tk_qwen4exp_r2v_ver_state_t* st);
// set_logit_bias: the request's bias as the object holds it (host copy, device buffer), and the new one adopted
int tk_qwen4exp_r2v_ver_bias_view(void* ver, const float** host, int64_t* n_host, float** device, int64_t* n_vocab,
                                  int32_t* eligible);
void tk_qwen4exp_r2v_ver_bias_adopt(void* ver, const float* bias, int64_t n, float* device);
// a stream at the device's highest priority (ROCm hands out hardware queues per priority); 1 made, 0 not
int tk_qwen4exp_r2v_stream_high_prio(void** out);
// the current device's gcnArchName starts with `prefix` (gfx103 / gfx12 rules)
int tk_qwen4exp_r2v_arch_is(const char* prefix);

// MtpDrafter::load / bind
typedef struct tk_qwen4exp_r2v_mtp_facts {
    int32_t eligible;         // no shared weights (checked by the caller), the object's own device
    int32_t max_t, device, q4, q4_head, force_on, hip, hip_gfx906, peer_portable;
    int32_t n_embd, hc, hc_lr, k, n_head, head_dim, n_head_kv, n_expert, n_ff;
    int32_t verify_max_t;
    int64_t max_cells, expert_blob;     // cpu::BLOB
    uint64_t q8_1_bytes, gr_ws_bytes, hit_grouped_scratch, shared_scratch, argmax_scratch, top_scratch;
    const char* rt_dir;
} tk_qwen4exp_r2v_mtp_facts_t;
int tk_qwen4exp_r2v_mtp_facts(void* mtp, tk_qwen4exp_r2v_mtp_facts_t* out);
uint64_t tk_qwen4exp_r2v_attn_scratch_floats(void* mtp, int64_t cap);
typedef struct tk_qwen4exp_r2v_tensor {
    const char* name;
    const char* kind;
    int64_t rows, cols;
    uint64_t off, bytes;
} tk_qwen4exp_r2v_tensor_t;
// `t` null: the tensor table is kept (the experts' upload)
void tk_qwen4exp_r2v_mtp_set_weights(void* mtp, const tk_qwen4exp_r2v_tensor_t* t, int64_t n, void* dense, void* experts,
                                     uint64_t vram_add);
int tk_qwen4exp_r2v_mtp_make_q4_dense(void* mtp, const uint8_t* blob, uint64_t n, char* err, size_t err_cap);
// the draft layer's K/V state (layer.cpp's carve, Piece 1/5's): flags, bytes, init into the object's st_, zero, reset
void tk_qwen4exp_r2v_kv_flags(int32_t* hybrid, int32_t* int8);
void tk_qwen4exp_r2v_kv_set_flags(int32_t hybrid, int32_t int8);
uint64_t tk_qwen4exp_r2v_mtp_state_bytes(void* mtp, int64_t max_cells, int64_t ring);
uint64_t tk_qwen4exp_r2v_mtp_state_init(void* mtp, int64_t max_cells, void* arena, int64_t ring, int32_t* kv_mode);
void tk_qwen4exp_r2v_mtp_state_reset(void* mtp);
void tk_qwen4exp_r2v_mtp_state_zero(void* mtp);

#define TK_R2V_MTP_MAPPED 7
#define TK_R2V_MTP_ARENA 51   // the carve order, the GR workspace included
typedef struct tk_qwen4exp_r2v_mtp_state {
    void* mapped_h[TK_R2V_MTP_MAPPED];
    void* mapped_d[TK_R2V_MTP_MAPPED];
    void* state_arena;
    void* arena;
    void* arena_ptrs[TK_R2V_MTP_ARENA];
    int64_t window, cap, attn_scratch_floats;
    uint64_t vram;
    void *cs, *side;
} tk_qwen4exp_r2v_mtp_state_t;
// `vram` is ADDED to the object's count (the Q4_0 copies add theirs inside); returns the total
uint64_t tk_qwen4exp_r2v_mtp_adopt(void* mtp, const tk_qwen4exp_r2v_mtp_state_t* st);

typedef struct tk_qwen4exp_r2v_bind_facts {
    int32_t head_loaded, head_type, head_device, device, q4_head, coupled_env, have_cparams, have_dhead, q6_packed;
    int32_t max_t;
    int64_t n_vocab;          // -1: output.weight missing
    uint64_t head_row_bytes;
    const void* head_weights;
    float* head_logits;
    const char* vocab_file;   // valid until the next call
} tk_qwen4exp_r2v_bind_facts_t;
int tk_qwen4exp_r2v_bind_facts(void* mtp, tk_qwen4exp_r2v_bind_facts_t* out);
void tk_qwen4exp_r2v_bind_head_logits(void* mtp, float* logits, int64_t n_vocab);
void tk_qwen4exp_r2v_bind_dhead(void* mtp, uint8_t* dhead, int32_t* dvocab, const int32_t* dvocab_host, int64_t n,
                                int32_t type, uint64_t vram_add);
void tk_qwen4exp_r2v_gather_rows(const void* src, int64_t row_bytes, const int32_t* ids, int64_t n, void* dst);
int tk_qwen4exp_r2v_mtp_make_q4_head(void* mtp, char* err, size_t err_cap);
int tk_qwen4exp_r2v_q6_pack(void* w, int n_in, int n_out, const char* what);
int tk_qwen4exp_r2v_mtp_setup_coupled(void* mtp, char* err, size_t err_cap);
// ---- qwen4exp-r2v end ----

#ifdef __cplusplus
}
#endif
