//! Dynamic FFI bindings to `include/talus_qwen4exp_kernels.h` in `libtalus_kernels.so`.

use core::ffi::{c_char, c_double, c_float, c_int, c_void};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

#[repr(C)]
pub struct TkQwen4ExpServeCtx {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TkQwen4ExpGenCtx {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpServeInfo {
    pub n_vocab: i64,
    pub n_layers: i64,
    pub n_expert: i64,
    pub max_context: i64,
    pub spec: i32,
    pub mtp_max_t: i32,
    pub use_mtp: i32,
    pub suffix_draft: i32,
    pub lookup_chain: i32,
    pub lookup_chain_min: i32,
    pub spec_min_p: c_double,
    pub pcie_frac: c_double,
    pub prompt_cache: i32,
    pub prompt_cache_every: i64,
    pub prompt_cache_root: i64,
    pub prompt_cache_tail: i32,
    pub turn_token: i64,
    pub tail_role_token: i64,
    pub short_read: i64,
    pub message_checkpoint_env: i32,
    pub n_eos: usize,
    pub eos_ids: [i64; 16],
}

impl Default for TkQwen4ExpServeInfo {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpGenInfo {
    pub n_vocab: i64,
    pub n_layers: i64,
    pub n_expert: i64,
    pub max_context: i64,
    pub max_new: i64,
    pub prefill_chunk: i64,
    pub prefill_until: i64,
    pub native_pack: i32,
    pub spec: i32,
    pub mtp_max_t: i32,
    pub use_mtp: i32,
    pub suffix_draft: i32,
    pub lookup_chain: i32,
    pub lookup_chain_min: i32,
    pub stop_eos: i32,
    pub spec_corrupt: i32,
    pub spec_min_p: c_double,
    pub n_prompt: usize,
    pub prompt_tokens: *const i64,
    pub n_eos: usize,
    pub eos_ids: [i64; 16],
}

impl Default for TkQwen4ExpGenInfo {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpReqSampling {
    pub temperature: c_float,
    pub top_p: c_float,
    pub top_k: i32,
    pub min_p: c_float,
    pub penalty_last_n: i32,
    pub penalty_repeat: c_float,
    pub penalty_freq: c_float,
    pub penalty_present: c_float,
    pub seed: u64,
    pub pcie_frac: c_double,
    pub spec_min_p: c_double,
    pub aux_cpus: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkQwen4ExpWeightsInfo {
    pub native_pack: i32,
    pub mtp_loaded: i32,
    pub ple_ready: i32,
    pub cache_slots: i32,
    pub resident_experts: i64,
    pub cache_bytes: u64,
    pub max_blob_bytes: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkQwen4ExpPrefillStats {
    pub tokens: i64,
    pub chunks: i64,
    pub ms_total: c_double,
    pub experts_streamed: i64,
    pub experts_dma: i64,
    pub experts_resident: i64,
    pub experts_cpu: i64,
    pub ms_experts_host: c_double,
    pub ms_ple: c_double,
    pub cpu_share: c_double,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkQwen4ExpConvStateSizes {
    pub gdn_bytes: u64,
    pub ple_bytes: u64,
    pub qsa_tail_bytes: u64,
    pub qsa_dead_bytes: u64,
    pub qsa_block_pos_bytes: u64,
    pub qsa_layers: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkQwen4ExpKldReport {
    pub positions_scored: i64,
    pub mean_kld: c_double,
    pub median_kld: c_double,
    pub p90_kld: c_double,
    pub p99_kld: c_double,
    pub max_kld: c_double,
    pub top1_agreement: c_double,
    pub flip_rate: c_double,
}

/// `#[repr(C)]` mirror of `tk_qwen4exp_rust_driver_t` in `include/talus_qwen4exp_kernels.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkQwen4ExpRustDriver {
    pub user_data: *mut c_void,
    pub next_line: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            block: c_int,
            out_ptr: *mut *const c_char,
            out_len: *mut usize,
        ) -> c_int,
    >,
    pub clear_stop: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
    pub is_stopped: Option<unsafe extern "C" fn(user_data: *mut c_void) -> c_int>,
    pub parse_logit_bias: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            text: *const c_char,
            vocab: c_int,
            out_bias: *mut c_float,
            out_err: *mut c_char,
            err_cap: usize,
        ) -> c_int,
    >,
    pub plan_prompt_cuts: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            ids: *const i64,
            n_ids: usize,
            resume: i64,
            read_from: i64,
            resumed_from0: c_int,
            reread_to: i64,
            prompt_cache: c_int,
            turn_token: i64,
            tail_role_token: i64,
            prompt_cache_root: i64,
            req_ckpt: c_int,
            multi_gpu: c_int,
            message_checkpoint_env: c_int,
            pin_at: i64,
            out_turn_at: *mut i64,
            out_root_at: *mut i64,
            out_message_at: *mut i64,
            out_cuts: *mut i64,
            out_n_cuts: *mut usize,
        ) -> c_int,
    >,
    pub init_spec: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            max_t: c_int,
            suffix_draft: c_int,
            lookup_chain: c_int,
            lookup_chain_min: c_int,
            capacity_tokens: usize,
        ),
    >,
    pub reset_request_spec: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            prompt_ids: *const i64,
            n_prompt: usize,
            produced_ids: *const i64,
            n_produced: usize,
        ),
    >,
    pub plan_decode_window: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            first_window: c_int,
            use_mtp: c_int,
            initial_t: c_int,
            s_mtp: c_int,
            max_s: c_int,
            room: c_int,
            allow_chain: c_int,
            spec_min_p: c_double,
            chain_fixed: c_int,
            drafts: *const i32,
            dprob: *const c_float,
            out_t: *mut c_int,
            out_t_mtp: *mut c_int,
            out_from_sfx: *mut c_int,
            out_sfx_match: *mut c_int,
            out_chain_n: *mut c_int,
            out_chain_match: *mut c_int,
            out_sbuf: *mut i32,
            out_cbuf: *mut i32,
        ) -> c_int,
    >,
    pub append_emitted_token: Option<unsafe extern "C" fn(user_data: *mut c_void, token: i32)>,
    pub observe_decode_round: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            timed_round: c_int,
            eos: c_int,
            from_sfx: c_int,
            t: c_int,
            t_mtp: c_int,
            chain_n: c_int,
            accepted: c_int,
            sfx_match: c_int,
            chain_match: c_int,
            round_ms: c_double,
        ),
    >,
    pub run_serve_loop:
        Option<unsafe extern "C" fn(user_data: *mut c_void, ctx: *mut TkQwen4ExpServeCtx) -> c_int>,
    pub run_generate_loop:
        Option<unsafe extern "C" fn(user_data: *mut c_void, ctx: *mut TkQwen4ExpGenCtx) -> c_int>,
}

pub type FnRunWithRustDriver = unsafe extern "C" fn(
    argc: c_int,
    argv: *const *const c_char,
    driver: *const TkQwen4ExpRustDriver,
) -> c_int;

pub type FnRequestStop = unsafe extern "C" fn();

pub type FnServeGetInfo =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx, out_info: *mut TkQwen4ExpServeInfo);
pub type FnServePreCommand = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    line: *const c_char,
    is_quit: c_int,
) -> c_int;
pub type FnServeSetBusy = unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx, busy: c_int);
pub type FnServeBeginRequest = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    ids: *const i64,
    n_ids: usize,
    req_cache: c_int,
    req_ckpt: c_int,
    req_pin: i64,
    req_cvec: c_int,
    sampling: *const TkQwen4ExpReqSampling,
    bias: *const c_float,
    n_bias: usize,
    out_resume: *mut i64,
    out_read_from: *mut i64,
    out_resumed_from0: *mut c_int,
    out_reread_to: *mut i64,
    out_pin_at: *mut i64,
) -> c_int;
pub type FnServePrefillSegment = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    at: i64,
    to: i64,
    use_windows: c_int,
    out_nf_hit: *mut c_int,
    out_cancelled: *mut c_int,
) -> c_int;
pub type FnServeCheckpointAt = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    pos: i64,
    as_tail: c_int,
    is_pin: c_int,
    turn_at: i64,
) -> c_int;
pub type FnServePrefillFinish = unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx) -> c_int;
pub type FnServeVerifyWindow = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    t: c_int,
    t_mtp: c_int,
    from_sfx: c_int,
    pos: i64,
    window: *const i32,
    consumed: *const i32,
    n_consumed: usize,
    out_tokens: *mut i32,
) -> c_int;
pub type FnServeCopyLogits = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    row: c_int,
    out_logits: *mut c_float,
) -> c_int;
pub type FnServeCommitWindow = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    keep: c_int,
    rounds: i64,
    first_window: c_int,
    out_nf_hit: *mut c_int,
) -> c_int;
pub type FnServeDraftWindow = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    t: c_int,
    accepted: c_int,
    pos: i64,
    need_draft: c_int,
    req_spec_min_p: c_double,
    out_tokens: *const i32,
    consumed: *const i32,
    n_consumed: usize,
    out_drafts: *mut i32,
    out_dprob: *mut c_float,
) -> c_int;
pub type FnServeFinishRequest = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    cancelled: c_int,
    nf_hit: c_int,
    req_ckpt: c_int,
    produced_n: i64,
    n_prompt: i64,
    resume: i64,
    prompt_ms: c_double,
    decode_ms: c_double,
    finish_reason: *const c_char,
    draft_accepted: i64,
    draft_offered: i64,
    sfx_windows: i64,
    sfx_ok: i64,
    sfx_drafts: i64,
    chain_windows: i64,
    chain_ok: i64,
    chain_drafts: i64,
    dec_windows: i64,
    consumed: *const i32,
    n_consumed: usize,
) -> c_int;
pub type FnServeOnExit = unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx);

pub type FnGenGetInfo =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, out_info: *mut TkQwen4ExpGenInfo);
pub type FnGenPrefillBatched = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    out_pos_start: *mut i64,
    out_tok: *mut i64,
    out_prefill_ms: *mut c_double,
) -> c_int;
pub type FnGenStepToken = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    pos: i64,
    tok: i64,
    out_next: *mut i32,
    out_step_ms: *mut c_double,
    out_ttft_ms: *mut c_double,
) -> c_int;
pub type FnGenSpecInit = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    spec_pos: i64,
    x: i32,
    out_first_window: *mut c_int,
    out_max_new: *mut i64,
    out_drafts: *mut i32,
    out_dprob: *mut c_float,
) -> c_int;
pub type FnGenSpecVerify = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    t: c_int,
    t_mtp: c_int,
    from_sfx: c_int,
    pos: i64,
    window: *const i32,
    out_tokens: *mut i32,
) -> c_int;
pub type FnGenSpecCopyLogits =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, row: c_int, out_logits: *mut c_float) -> c_int;
pub type FnGenSpecCommit =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, keep: c_int, rounds: i64) -> c_int;
pub type FnGenSpecJoinAdapt = unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx) -> c_int;
pub type FnGenSpecDraft = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    t: c_int,
    accepted: c_int,
    pos: i64,
    need_draft: c_int,
    out_tokens: *const i32,
    out_drafts: *mut i32,
    out_dprob: *mut c_float,
) -> c_int;
pub type FnGenSpecFinish = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    rounds: i64,
    drafts_ok: i64,
    drafts_total: i64,
    sfx_windows: i64,
    sfx_ok: i64,
    sfx_drafts: i64,
    chain_windows: i64,
    chain_ok: i64,
    chain_drafts: i64,
    window_hist: *const i64,
    n_window_hist: usize,
    accepted_hist: *const i64,
    n_accepted_hist: usize,
) -> c_int;
pub type FnGenFinish = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    produced: *const i64,
    n_produced: usize,
    total_ms: c_double,
    prefill_ms: c_double,
    ttft_ms: c_double,
) -> c_int;

pub type FnDumpTensor = unsafe extern "C" fn(
    tag: *const c_char,
    layer: i32,
    pos: i64,
    ptr: *const c_float,
    n_floats: usize,
    is_device: c_int,
    stream: *mut c_void,
) -> c_int;

pub type FnP1QueryWeightsInfo = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    out_info: *mut TkQwen4ExpWeightsInfo,
) -> c_int;
pub type FnP1ExpertCacheInfo = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    out_host_res: *mut i32,
    cap_entries: usize,
) -> c_int;
pub type FnP1BorrowSlots = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    want_chunk_tokens: i64,
    out_lent_slots: *mut i32,
) -> c_int;
pub type FnP1RefillSlots = unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx) -> c_int;

pub type FnP2PrefillChunk = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    at: i64,
    to: i64,
    out_stats: *mut TkQwen4ExpPrefillStats,
) -> c_int;
pub type FnP2TpMoeGemm = unsafe extern "C" fn(
    w: *const c_void,
    type_: c_int,
    expert_bytes: usize,
    n: c_int,
    k: c_int,
    n_experts: c_int,
    xq: *const c_void,
    bounds: *const i32,
    ids: *const i32,
    total_rows: c_int,
    max_rows: c_int,
    dst: *mut c_float,
    ld_dst: c_int,
    stream: *mut c_void,
) -> c_int;
pub type FnP2DraftKv = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    r_rows: *const c_float,
    next_tokens: *const i32,
    t_rows: i64,
    pos0: i64,
) -> c_int;

pub type FnP3PleStep =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, token: i32, phase: c_int) -> c_int;
pub type FnP3EmbedToken =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, token: i64, pos: i64) -> c_int;
pub type FnP3RunLayer =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, layer: i32, pos: i64) -> c_int;
pub type FnP3RunHead =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, pos: i64, out_next_token: *mut i32) -> c_int;

pub type FnP4VerifyRun = FnServeVerifyWindow;
pub type FnP4VerifyCommit = FnServeCommitWindow;
pub type FnP4MtpDraft = FnServeDraftWindow;

pub type FnP5StateSizes = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    out_sizes: *mut TkQwen4ExpConvStateSizes,
) -> c_int;
pub type FnP5SnapshotCapture = FnServeCheckpointAt;
pub type FnP5SnapshotRestore =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx, target_pos: i64) -> c_int;
pub type FnP5SessionFileOp = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    is_save: c_int,
    path: *const c_char,
) -> c_int;

pub type FnP6KldEval =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, out_report: *mut TkQwen4ExpKldReport) -> c_int;

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

fn candidate_so_paths(explicit: Option<&Path>) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(p) = explicit {
        out.push(p.to_path_buf());
    }
    for var in ["TALUS_KERNELS_SO", "TALUS_KERNELS_LIB"] {
        if let Ok(env_p) = std::env::var(var) {
            if !env_p.is_empty() {
                out.push(PathBuf::from(env_p));
            }
        }
    }
    if let Ok(dir) = std::env::var("TALUS_KERNELS_PREBUILT") {
        if !dir.is_empty() {
            out.push(Path::new(&dir).join("libtalus_kernels.so"));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.join("libtalus_kernels.so"));
            if let Some(parent) = dir.parent() {
                out.push(parent.join("libtalus_kernels.so"));
                if let Some(grand) = parent.parent() {
                    out.push(grand.join("build/libtalus_kernels.so"));
                }
            }
        }
    }
    out.push(PathBuf::from("build/libtalus_kernels.so"));
    out.push(PathBuf::from("libtalus_kernels.so"));
    out
}

/// Loaded handle to `libtalus_kernels.so` with the Flash-Next (`qwen4exp`) entry points.
#[derive(Clone, Copy)]
pub struct Qwen4ExpKernels {
    _handle: *mut c_void,
    pub run_with_rust_driver: FnRunWithRustDriver,
    pub request_stop: FnRequestStop,
    pub serve_get_info: FnServeGetInfo,
    pub serve_pre_command: FnServePreCommand,
    pub serve_set_busy: FnServeSetBusy,
    pub serve_begin_request: FnServeBeginRequest,
    pub serve_prefill_segment: FnServePrefillSegment,
    pub serve_checkpoint_at: FnServeCheckpointAt,
    pub serve_prefill_finish: FnServePrefillFinish,
    pub serve_verify_window: FnServeVerifyWindow,
    pub serve_copy_logits: FnServeCopyLogits,
    pub serve_commit_window: FnServeCommitWindow,
    pub serve_draft_window: FnServeDraftWindow,
    pub serve_finish_request: FnServeFinishRequest,
    pub serve_on_exit: FnServeOnExit,
    pub gen_get_info: FnGenGetInfo,
    pub gen_prefill_batched: FnGenPrefillBatched,
    pub gen_step_token: FnGenStepToken,
    pub gen_spec_init: FnGenSpecInit,
    pub gen_spec_verify: FnGenSpecVerify,
    pub gen_spec_copy_logits: FnGenSpecCopyLogits,
    pub gen_spec_commit: FnGenSpecCommit,
    pub gen_spec_join_adapt: FnGenSpecJoinAdapt,
    pub gen_spec_draft: FnGenSpecDraft,
    pub gen_spec_finish: FnGenSpecFinish,
    pub gen_finish: FnGenFinish,
    pub dump_tensor: FnDumpTensor,
    pub p1_query_weights_info: FnP1QueryWeightsInfo,
    pub p1_expert_cache_info: FnP1ExpertCacheInfo,
    pub p1_borrow_slots: FnP1BorrowSlots,
    pub p1_refill_slots: FnP1RefillSlots,
    pub p2_prefill_chunk: FnP2PrefillChunk,
    pub p2_tp_moe_gemm: FnP2TpMoeGemm,
    pub p2_draft_kv: FnP2DraftKv,
    pub p3_ple_step: FnP3PleStep,
    pub p3_embed_token: FnP3EmbedToken,
    pub p3_run_layer: FnP3RunLayer,
    pub p3_run_head: FnP3RunHead,
    pub p4_verify_run: FnP4VerifyRun,
    pub p4_verify_commit: FnP4VerifyCommit,
    pub p4_mtp_draft: FnP4MtpDraft,
    pub p5_state_sizes: FnP5StateSizes,
    pub p5_snapshot_capture: FnP5SnapshotCapture,
    pub p5_snapshot_restore: FnP5SnapshotRestore,
    pub p5_session_file_op: FnP5SessionFileOp,
    pub p6_kld_eval: FnP6KldEval,
}

unsafe impl Send for Qwen4ExpKernels {}
unsafe impl Sync for Qwen4ExpKernels {}

unsafe fn load_sym<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let c_name = CString::new(name).unwrap();
    let ptr = dlsym(handle, c_name.as_ptr());
    if ptr.is_null() {
        let err_ptr = dlerror();
        let detail = if !err_ptr.is_null() {
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        } else {
            format!("missing symbol {name}")
        };
        return Err(detail);
    }
    Ok(std::mem::transmute_copy(&ptr))
}

// ---- qwen4exp-p4 begin ----
impl Qwen4ExpKernels {
    /// The `dlopen` handle of `libtalus_kernels.so`: Piece 4 (`verify_mtp.rs`) resolves its own
    /// `tk_qwen4exp_p4_*` table from it.
    pub fn raw_handle(&self) -> *mut c_void {
        self._handle
    }
}

/// `dlsym` of one entry point of the loaded library (Piece 4's table, `verify_mtp.rs`).
pub unsafe fn load_sym_from<T>(handle: *mut c_void, name: &str) -> Result<T, String> {
    load_sym(handle, name)
}
// ---- qwen4exp-p4 end ----

impl Qwen4ExpKernels {
    pub fn load(explicit: Option<&Path>) -> Result<Self, String> {
        let candidates = candidate_so_paths(explicit);
        let mut last_err = String::new();
        for cand in candidates {
            let Ok(c_path) = CString::new(cand.to_string_lossy().as_bytes()) else {
                continue;
            };
            unsafe {
                let _ = dlerror();
                let handle = dlopen(c_path.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
                if handle.is_null() {
                    let err_ptr = dlerror();
                    if !err_ptr.is_null() {
                        last_err = CStr::from_ptr(err_ptr).to_string_lossy().into_owned();
                    }
                    continue;
                }
                let res = (|| -> Result<Self, String> {
                    Ok(Self {
                        _handle: handle,
                        run_with_rust_driver: load_sym(handle, "tk_qwen4exp_run_with_rust_driver")?,
                        request_stop: load_sym(handle, "tk_qwen4exp_request_stop")?,
                        serve_get_info: load_sym(handle, "tk_qwen4exp_serve_get_info")?,
                        serve_pre_command: load_sym(handle, "tk_qwen4exp_serve_pre_command")?,
                        serve_set_busy: load_sym(handle, "tk_qwen4exp_serve_set_busy")?,
                        serve_begin_request: load_sym(handle, "tk_qwen4exp_serve_begin_request")?,
                        serve_prefill_segment: load_sym(
                            handle,
                            "tk_qwen4exp_serve_prefill_segment",
                        )?,
                        serve_checkpoint_at: load_sym(handle, "tk_qwen4exp_serve_checkpoint_at")?,
                        serve_prefill_finish: load_sym(handle, "tk_qwen4exp_serve_prefill_finish")?,
                        serve_verify_window: load_sym(handle, "tk_qwen4exp_serve_verify_window")?,
                        serve_copy_logits: load_sym(handle, "tk_qwen4exp_serve_copy_logits")?,
                        serve_commit_window: load_sym(handle, "tk_qwen4exp_serve_commit_window")?,
                        serve_draft_window: load_sym(handle, "tk_qwen4exp_serve_draft_window")?,
                        serve_finish_request: load_sym(handle, "tk_qwen4exp_serve_finish_request")?,
                        serve_on_exit: load_sym(handle, "tk_qwen4exp_serve_on_exit")?,
                        gen_get_info: load_sym(handle, "tk_qwen4exp_gen_get_info")?,
                        gen_prefill_batched: load_sym(handle, "tk_qwen4exp_gen_prefill_batched")?,
                        gen_step_token: load_sym(handle, "tk_qwen4exp_gen_step_token")?,
                        gen_spec_init: load_sym(handle, "tk_qwen4exp_gen_spec_init")?,
                        gen_spec_verify: load_sym(handle, "tk_qwen4exp_gen_spec_verify")?,
                        gen_spec_copy_logits: load_sym(handle, "tk_qwen4exp_gen_spec_copy_logits")?,
                        gen_spec_commit: load_sym(handle, "tk_qwen4exp_gen_spec_commit")?,
                        gen_spec_join_adapt: load_sym(handle, "tk_qwen4exp_gen_spec_join_adapt")?,
                        gen_spec_draft: load_sym(handle, "tk_qwen4exp_gen_spec_draft")?,
                        gen_spec_finish: load_sym(handle, "tk_qwen4exp_gen_spec_finish")?,
                        gen_finish: load_sym(handle, "tk_qwen4exp_gen_finish")?,
                        dump_tensor: load_sym(handle, "tk_qwen4exp_dump_tensor")?,
                        p1_query_weights_info: load_sym(
                            handle,
                            "tk_qwen4exp_p1_query_weights_info",
                        )?,
                        p1_expert_cache_info: load_sym(handle, "tk_qwen4exp_p1_expert_cache_info")?,
                        p1_borrow_slots: load_sym(handle, "tk_qwen4exp_p1_borrow_slots")?,
                        p1_refill_slots: load_sym(handle, "tk_qwen4exp_p1_refill_slots")?,
                        p2_prefill_chunk: load_sym(handle, "tk_qwen4exp_p2_prefill_chunk")?,
                        p2_tp_moe_gemm: load_sym(handle, "tk_qwen4exp_p2_tp_moe_gemm")?,
                        p2_draft_kv: load_sym(handle, "tk_qwen4exp_p2_draft_kv")?,
                        p3_ple_step: load_sym(handle, "tk_qwen4exp_p3_ple_step")?,
                        p3_embed_token: load_sym(handle, "tk_qwen4exp_p3_embed_token")?,
                        p3_run_layer: load_sym(handle, "tk_qwen4exp_p3_run_layer")?,
                        p3_run_head: load_sym(handle, "tk_qwen4exp_p3_run_head")?,
                        p4_verify_run: load_sym(handle, "tk_qwen4exp_p4_verify_run")?,
                        p4_verify_commit: load_sym(handle, "tk_qwen4exp_p4_verify_commit")?,
                        p4_mtp_draft: load_sym(handle, "tk_qwen4exp_p4_mtp_draft")?,
                        p5_state_sizes: load_sym(handle, "tk_qwen4exp_p5_state_sizes")?,
                        p5_snapshot_capture: load_sym(handle, "tk_qwen4exp_p5_snapshot_capture")?,
                        p5_snapshot_restore: load_sym(handle, "tk_qwen4exp_p5_snapshot_restore")?,
                        p5_session_file_op: load_sym(handle, "tk_qwen4exp_p5_session_file_op")?,
                        p6_kld_eval: load_sym(handle, "tk_qwen4exp_p6_kld_eval")?,
                    })
                })();
                match res {
                    Ok(k) => return Ok(k),
                    Err(e) => {
                        last_err = format!("{} in {}", e, cand.display());
                        continue;
                    }
                }
            }
        }
        Err(format!(
            "failed to load libtalus_kernels.so for qwen4exp: {last_err}"
        ))
    }
}

// ---- qwen4exp-p3 begin ----
// Piece 3: the decode window's launch-only wrappers (`tk_qwen4exp_p3_*`, src/program/qwen4exp_p3.cpp). Resolved with
// `dlsym(RTLD_DEFAULT, ..)` because `Qwen4ExpKernels::load` opens libtalus_kernels.so with RTLD_GLOBAL, which keeps
// this table out of the shared loader above (no merge contention with the other pieces).

/// Mirror of `tk_qwen4exp_p3_window_view_t` (field order is the C struct's).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP3WindowView {
    pub ver: *mut c_void,
    pub ss: *mut c_void,
    pub wt: *const c_void,
    pub g: *const c_void,
    pub n_embd: i64,
    pub hc: i64,
    pub hc_lr: i64,
    pub k: i64,
    pub conv_channels: i64,
    pub value_dim: i64,
    pub v_heads: i64,
    pub k_heads: i64,
    pub state_size: i64,
    pub d_conv: i64,
    pub n_head: i64,
    pub head_dim: i64,
    pub n_head_kv: i64,
    pub idx_q_heads: i64,
    pub idx_key_dim: i64,
    pub n_expert: i64,
    pub n_ff: i64,
    pub n_layers: i64,
    pub n_vocab: i64,
    pub idx_block: i64,
    pub n_rot: i64,
    pub step_count: i64,
    pub step_pos: i64,
    pub ng_hist: i64,
    pub ng_hc_dim: i64,
    pub ng_n_embd: i64,
    pub max_t: i64,
    pub plan_i32: i64,
    pub cap: i64,
    pub max_blocks: i64,
    pub attn_scratch_floats: i64,
    pub lb: i64,
    pub le: i64,
    pub gdn_ord0: i64,
    pub n_qsa_layers: i64,
    pub staging_blobs: i64,
    pub pcie_group_rows: i64,
    pub gdn_floats: i64,
    pub max_blob: i64,
    pub ple_block_scratch_bytes: i64,
    pub eps: c_float,
    pub batch_rec: i32,
    pub split: i32,
    pub prof_on: i32,
    pub trace_on: i32,
    pub ar_on: i32,
    pub device_plan: i32,
    pub remote_opt: i32,
    pub always_publish: i32,
    pub df_branch: i32,
    pub head_loaded: i32,
    pub head_type: i32,
    pub ple_ready: i32,
    pub ple_stage: i32,
    pub native_layout: i32,
    pub pcie_mode: i32,
    pub sh_fork_ready: i32,
    pub ple_key_bf16: i32,
    pub ple_value_bf16: i32,
    pub ple_key_native: i32,
    pub has_hand_in: i32,
    pub head_weights: *const c_void,
    pub cs: *mut c_void,
    pub sh_cs: *mut c_void,
    pub ev_fork: *mut c_void,
    pub ev_join: *mut c_void,
    pub tok: *mut i32,
    pub step: *mut i32,
    pub pos: *mut i32,
    pub m_tok: *const i32,
    pub m_step: *const i32,
    pub m_pos: *const i32,
    pub ple: *mut f32,
    pub emb: *mut f32,
    pub r: *mut f32,
    pub mixed: *mut f32,
    pub bo: *mut f32,
    pub inj: *mut f32,
    pub inj2: *mut f32,
    pub lo: *mut f32,
    pub rs: *mut f32,
    pub xn: *mut f32,
    pub m_ple: *const f32,
    pub xq: *mut u8,
    pub xil: *mut u8,
    pub sh_xq: *mut u8,
    pub qkv_l: *mut f32,
    pub h_l: *mut f32,
    pub gate_l: *mut f32,
    pub beta_l: *mut f32,
    pub z: *mut f32,
    pub y: *mut f32,
    pub qfull: *mut f32,
    pub qcur: *mut f32,
    pub kcur: *mut f32,
    pub vcur: *mut f32,
    pub idx_raw_l: *mut f32,
    pub qidx: *mut f32,
    pub scores: *mut f32,
    pub attn: *mut f32,
    pub attn32: *mut f32,
    pub attn_scratch: *mut f32,
    pub tail_snap: *mut f32,
    pub sel: *mut i32,
    pub logits: *mut f32,
    pub w: *mut f32,
    pub shared: *mut f32,
    pub parts: *mut f32,
    pub hit_out: *mut f32,
    pub ids: *mut i32,
    pub plan: *mut i32,
    pub staging: *mut u8,
    pub hit_xq: *mut u8,
    pub nat_xq: *mut u8,
    pub hit_xs: *mut f32,
    pub hit_scratch: *mut c_void,
    pub head_mixed: *mut f32,
    pub head_inj: *mut f32,
    pub head_logits: *mut f32,
    pub sh_bf16: *mut u16,
    pub arg_scratch: *mut u8,
    pub one: *mut i32,
    pub sh_gate: *mut f32,
    pub sh_up: *mut f32,
    pub sh_g: *mut f32,
    pub hist_snap: *mut f32,
    pub qcnt: *mut u32,
    pub skip: *mut u32,
    pub slot_off_d: *mut u64,
    pub m_plan_err: *mut u32,
    pub m_x: *mut f32,
    pub m_ids: *mut i32,
    pub m_w: *mut f32,
    pub m_seq: *mut u32,
    pub m_flag: *mut u32,
    pub m_flag_a: *mut u32,
    pub m_flag_b: *mut u32,
    pub m_plan: *mut i32,
    pub m_ymiss: *mut f32,
    pub m_out: *mut i32,
    pub m_nf: *mut u32,
    pub gdn_state: *mut f32,
    pub ple_hist: *mut f32,
    pub ple_scratch: *mut c_void,
    pub d_res: *const i32,
    pub cache_base: *const u8,
    pub blob: i64,
    pub ple_key: *mut f32,
    pub ple_val: *mut f32,
    pub ple_w_key_bf16: *const u16,
    pub ple_w_value_bf16: *const u16,
    pub ple_w_key_native: *const c_void,
    pub ple_w_key_native_type: i32,
}

/// Mirror of `tk_qwen4exp_p3_qsa_view_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP3QsaView {
    pub idx_tail: *mut f32,
    pub idx_dead: *mut f32,
    pub idx_pooled: *mut f32,
    pub idx_block_pos: *mut i32,
    pub cos_tab: *const f32,
    pub sin_tab: *const f32,
    pub max_cells: i64,
    pub kv_rot: i32,
    pub kv_hybrid: i32,
    pub kv_q4: i32,
    pub kv_int8: i32,
}

/// Mirror of `tk_qwen4exp_p3_layer_weights_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP3LayerWeights {
    pub is_qsa: i32,
    pub hc_norm: [*const f32; 2],
    pub hc_down: [*const u16; 2],
    pub hc_up: [*const u16; 2],
    pub hc_inject: [*const u16; 2],
    pub hc_q8_down: [*const u8; 2],
    pub hc_q8_up: [*const u8; 2],
    pub hc_q8_inject: [*const u8; 2],
    pub qkv_type: i32,
    pub gate_type: i32,
    pub out_type: i32,
    pub qkv: *const c_void,
    pub gate: *const c_void,
    pub out: *const c_void,
    pub alpha: *const u16,
    pub beta: *const u16,
    pub conv1d: *const f32,
    pub ssm_norm: *const f32,
    pub dt: *const f32,
    pub ssm_a: *const f32,
    pub idx_k: *const u16,
    pub idx_q: *const u16,
    pub q_type: i32,
    pub k_type: i32,
    pub v_type: i32,
    pub o_type: i32,
    pub q: *const c_void,
    pub k: *const c_void,
    pub v: *const c_void,
    pub o: *const c_void,
    pub q_norm: *const f32,
    pub k_norm: *const f32,
    pub iq_norm: *const f32,
    pub ik_norm: *const f32,
    pub router: *const u16,
    pub shexp_gate_inp: *const u16,
    pub sg_type: i32,
    pub su_type: i32,
    pub sd_type: i32,
    pub sg: *const c_void,
    pub su: *const c_void,
    pub sd: *const c_void,
}

/// Mirror of `tk_qwen4exp_p3_head_weights_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP3HeadWeights {
    pub norm: *const f32,
    pub down: *const u16,
    pub up: *const u16,
    pub q8_down: *const u8,
    pub q8_up: *const u8,
    pub forms_ok: i32,
}

/// Mirror of `tk_qwen4exp_p3_gr_args_t` (one row of a fused hyper-connection read).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP3GrArgs {
    pub r: *const f32,
    pub r_out: *mut f32,
    pub apply: i32,
    pub bo_prev: *const f32,
    pub inj_prev: *const f32,
    pub w_norm: *const f32,
    pub w_down: *const u16,
    pub w_up: *const u16,
    pub w_inject: *const u16,
    pub eps: c_float,
    pub lo: *mut f32,
    pub rs: *mut f32,
    pub inject_out: *mut f32,
    pub mixed: *mut f32,
    pub q8_down: *const u8,
    pub q8_up: *const u8,
    pub q8_inject: *const u8,
    pub q8_mixed: *mut u8,
    pub q8_cnt: *mut u32,
}

impl Default for TkP3GrArgs {
    fn default() -> Self {
        // FusedGrArgs' defaults: every pointer null, eps 1e-6
        let mut a: Self = unsafe { std::mem::zeroed() };
        a.eps = 1e-6;
        a
    }
}

pub type TkP3RecorderFn =
    unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, *mut c_void, *mut c_char, usize) -> c_int;

type S = *mut c_void; // a stream

macro_rules! p3_fns {
    ($($name:ident : fn($($arg:ty),*) -> $ret:ty;)*) => {
        /// The `tk_qwen4exp_p3_*` entry points.
        #[derive(Clone, Copy)]
        pub struct Fnp3Fns { $(pub $name: unsafe extern "C" fn($($arg),*) -> $ret,)* }
        unsafe impl Send for Fnp3Fns {}
        unsafe impl Sync for Fnp3Fns {}
        impl Fnp3Fns {
            /// Resolves every entry point from the already loaded (RTLD_GLOBAL) libtalus_kernels.so.
            pub fn load() -> Result<Self, String> {
                unsafe {
                    Ok(Self { $($name: {
                        let n = concat!("tk_qwen4exp_p3_", stringify!($name), "\0");
                        let p = dlsym(std::ptr::null_mut(), n.as_ptr().cast());
                        if p.is_null() {
                            return Err(format!("missing symbol {}", &n[..n.len() - 1]));
                        }
                        std::mem::transmute_copy(&p)
                    },)* })
                }
            }
        }
    };
}

p3_fns! {
    set_window_recorder: fn(Option<TkP3RecorderFn>, *mut c_void) -> ();
    last_error: fn(*mut c_char, usize) -> c_int;
    runtime_flags: fn() -> u64;
    window_view: fn(*mut c_void, *mut TkP3WindowView) -> c_int;
    set_groups: fn(*mut c_void, c_int, c_int) -> c_int;
    qsa_view: fn(*mut c_void, i64, *mut TkP3QsaView) -> c_int;
    layer_weights: fn(*const c_void, i32, i32, *mut TkP3LayerWeights) -> c_int;
    head_weights: fn(*const c_void, *mut TkP3HeadWeights) -> c_int;
    pdl_scope: fn(c_int) -> c_int;
    cvec_covers: fn(i64) -> c_int;
    norm_rope_usable: fn(c_int, c_int) -> c_int;
    mmvq_il_supported: fn(c_int, c_int, c_int) -> c_int;
    combine_hits_supported: fn(*const f32, *const f32, *const f32, i64, i64, c_int) -> c_int;
    expert_blob_bytes: fn(i64, *mut i64) -> c_int;
    event_fork: fn(*mut c_void, S, S) -> c_int;
    is_qsa_layer: fn(*const c_void, i64) -> c_int;
    copy_i32_from_mapped: fn(*mut i32, *const i32, i64, S) -> c_int;
    copy_from_mapped: fn(*mut f32, *const f32, i64, S) -> c_int;
    embed_rows: fn(*const c_void, *const i32, c_int, *mut f32, S) -> c_int;
    broadcast_streams: fn(*const f32, *mut f32, i64, c_int, c_int, S) -> c_int;
    wait_flag_ge: fn(*const u32, u32, S) -> c_int;
    wait_flag_ge_or: fn(*const u32, u32, *const u32, S) -> c_int;
    bf16_gemv: fn(*const f32, *const u16, *mut f32, i64, i64, S) -> c_int;
    bf16_gemv_multi: fn(*const f32, i64, *const u16, *mut f32, i64, i64, i64, c_int, S) -> c_int;
    bf16_gemv_multi_aux: fn(*const f32, i64, *const u16, *mut f32, i64, i64, i64, c_int, *const u16, *mut f32, i64, S) -> c_int;
    quantize_q8_1: fn(*const f32, *mut c_void, c_int, c_int, S) -> c_int;
    mmvq: fn(c_int, *const c_void, *const c_void, *mut f32, c_int, c_int, c_int, S) -> c_int;
    mmvq_il: fn(c_int, *const c_void, *const c_void, *const c_void, *mut f32, c_int, c_int, c_int, S) -> c_int;
    q8_1_interleave: fn(*const c_void, *mut c_void, c_int, c_int, S) -> c_int;
    gr_write: fn(*const f32, *const f32, *const f32, i64, i64, i64, *mut f32, c_int, c_int, S) -> c_int;
    gr_post_multi: fn(*const f32, *const f32, *const f32, *mut f32, c_int, c_int, c_int, i64, i64, i64, S) -> c_int;
    fused_gr_read_multi: fn(*const TkP3GrArgs, c_int, *mut f32, S) -> c_int;
    ple_postops_batch_snap: fn(*mut c_void, *mut f32, *mut f32, *const f32, *mut f32, *mut f32, *mut f32, c_int, *mut f32, S) -> c_int;
    ple_block: fn(*mut c_void, *const f32, *mut f32, *mut f32, *mut f32, *const f32, *const f32, c_int, S) -> c_int;
    ple_block_projected: fn(*mut c_void, *const f32, *const f32, *mut f32, *mut f32, *mut f32, S) -> c_int;
    ple_history_advance: fn(*mut f32, *const f32, S) -> c_int;
    gdn_ab_multi: fn(*const f32, *const u16, *const u16, *const f32, *const f32, *mut f32, *mut f32, c_int, c_int, c_int, S) -> c_int;
    gdn_conv_ab_multi: fn(*const f32, *const f32, *const f32, *mut f32, c_int, c_int, c_float, c_int, c_int, c_int, *const f32, *const u16, *const u16, *const f32, *const f32, *mut f32, *mut f32, c_int, c_int, S) -> c_int;
    gdn_conv_l2_multi: fn(*const f32, *const f32, *const f32, *mut f32, c_int, c_int, c_float, c_int, c_int, c_int, S) -> c_int;
    gdn_step_norm_multi: fn(*mut f32, *const f32, c_int, *const f32, *const f32, *const f32, *const f32, c_float, *mut f32, c_int, c_int, c_int, *const i32, c_int, *mut c_void, S) -> c_int;
    rms_norm: fn(*mut f32, *const f32, c_int, c_int, c_float, c_int, S) -> c_int;
    rope: fn(*mut f32, c_int, c_int, c_int, *const i32, *const f32, *const f32, c_int, S) -> c_int;
    rms_norm_rope: fn(*const f32, c_int, *const f32, *mut f32, c_int, c_int, c_int, c_float, *const i32, S) -> c_int;
    copy_rows_strided: fn(*mut f32, *const f32, i64, i64, i64, S) -> c_int;
    copy_2d: fn(*mut f32, *const f32, i64, i64, S) -> c_int;
    fwht256: fn(*mut f32, i64, S) -> c_int;
    kv_append: fn(*const c_void, *mut c_void, i64, *const i32, c_int, *const f32, *const f32, c_int, c_int, c_int, S) -> c_int;
    indexer_append: fn(*const c_void, *mut c_void, i64, *const f32, i64, *const i32, c_int, c_int, c_int, *const f32, c_float, S) -> c_int;
    qsa_block_scores: fn(*const c_void, *mut c_void, i64, *const f32, *const i32, i64, i64, *mut f32, S) -> c_int;
    qsa_block_topk: fn(*const c_void, *const f32, *const i32, i64, i64, i64, *mut i32, S) -> c_int;
    qsa_kv_resolve: fn(*const c_void, *mut c_void, i64, *const i32, *const i32, i64, i64, S) -> c_int;
    qsa_attn: fn(*const c_void, *mut c_void, i64, *const f32, *const i32, *const i32, i64, *mut f32, *mut f32, i64, S) -> c_int;
    qsa_gate: fn(*const f32, *const f32, *mut f32, c_int, c_int, c_int, *const c_void, S) -> c_int;
    router_top10_multi: fn(*const f32, *mut i32, *mut f32, c_int, S) -> c_int;
    moe_route_row: fn(*const c_void, *const c_void, *mut c_void, i64, i64, *const f32, *mut f32, *mut i32, *mut f32, S) -> c_int;
    resident_plan: fn(*const i32, c_int, c_int, *const i32, c_int, *const u8, *const u64, i64, *mut i32, i64, *mut u32, u32, *mut u32, S) -> c_int;
    doorbell_publish: fn(c_int, *const f32, *const i32, *const f32, *const i32, c_int, i64, i64, *mut f32, *mut i32, *mut f32, *mut u32, u32, S) -> c_int;
    f32_to_bf16: fn(*const f32, *mut u16, i64, S) -> c_int;
    shared_expert_multi: fn(c_int, *const f32, *const u16, i32, *const c_void, i32, *const c_void, i32, *const c_void, *mut c_void, *const u16, *mut f32, *mut f32, *mut f32, *mut f32, i64, i64, *const c_void, c_int, S) -> c_int;
    quantize_q8_1_rows: fn(*const f32, i64, i64, *mut c_void, S) -> c_int;
    quantize_q8_0_scaled: fn(*const f32, *mut u8, *mut f32, i64, S) -> c_int;
    expert_grouped: fn(i64, c_int, *const u64, *const i32, *const i32, *const i32, *const i32, i64, *const c_void, *const u8, *const f32, *mut c_void, *mut f32, i64, S) -> c_int;
    copy_i32_from_mapped_unless: fn(*mut i32, *const i32, i64, *const u32, u32, S) -> c_int;
    wait_copy_i32_from_mapped: fn(*const u32, u32, *mut i32, *const i32, i64, S) -> c_int;
    fetch_blobs: fn(*const u64, *const i32, *mut u8, i64, c_int, S) -> c_int;
    rebase_ptrs: fn(*mut u64, *const i32, *mut u8, i64, S) -> c_int;
    copy_or_zero_from_mapped: fn(*mut f32, *const f32, i64, *const u32, u32, S) -> c_int;
    copy_rows_from_mapped: fn(*const u32, u32, *mut f32, *const f32, i64, i64, *const i32, *const i32, S) -> c_int;
    moe_hit_add: fn(*mut f32, *const f32, *const i32, *const i32, i64, i64, S) -> c_int;
    moe_combine: fn(c_int, *const f32, *const f32, *const i32, *const i32, *const f32, *const f32, *const f32, *mut f32, i64, i64, c_int, S) -> c_int;
    moe_combine_row: fn(*const c_void, *mut c_void, i64, i64, *const f32, *const f32, *const f32, *mut f32, S) -> c_int;
    cvec_apply: fn(*mut f32, i64, i64, i64, *const f32, i64, *const f32, i64, c_int, S) -> c_int;
    head_mix_row: fn(*const c_void, *const c_void, *mut c_void, *mut f32, *mut f32, *mut f32, c_int, S) -> c_int;
    argmax_rows: fn(*const f32, c_int, c_int, *mut c_void, *mut i32, S) -> c_int;
    sample_greedy: fn(*const f32, c_int, c_int, *mut i32, S) -> c_int;
    logits_nonfinite_rows: fn(*const f32, c_int, c_int, *mut u32, S) -> c_int;
    gpu_noop: fn(S) -> c_int;
}
// ---- qwen4exp-p3 end ----

// ---- qwen4exp-p5 begin ----
/// `tk_qwen4exp_p6_take_logits_dump`: takes the open `--dump-logits` file from the C++ step loop.
pub type FnP6TakeLogitsDump = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpGenCtx,
    out_path: *mut c_char,
    path_cap: usize,
    out_stride: *mut i64,
    out_positions: *mut i64,
) -> c_int;
/// `tk_qwen4exp_p6_step_logits`: copies the last single-token step's logits row to host memory.
pub type FnP6StepLogits =
    unsafe extern "C" fn(ctx: *mut TkQwen4ExpGenCtx, out_logits: *mut c_float) -> c_int;

/// Pieces 5 + 6 (`qwen4exp-p5`) entry points, resolved from the already loaded (`RTLD_GLOBAL`)
/// `libtalus_kernels.so` so the shared [`Qwen4ExpKernels`] table stays untouched.
#[derive(Clone, Copy)]
pub struct P5Kernels {
    pub p6_take_logits_dump: FnP6TakeLogitsDump,
    pub p6_step_logits: FnP6StepLogits,
}

unsafe impl Send for P5Kernels {}
unsafe impl Sync for P5Kernels {}

impl P5Kernels {
    /// Resolves the `qwen4exp-p5` symbols (`dlsym(RTLD_DEFAULT, ..)`); call after [`Qwen4ExpKernels::load`].
    pub fn resolve() -> Result<Self, String> {
        unsafe {
            let h = std::ptr::null_mut();
            Ok(Self {
                p6_take_logits_dump: load_sym(h, "tk_qwen4exp_p6_take_logits_dump")?,
                p6_step_logits: load_sym(h, "tk_qwen4exp_p6_step_logits")?,
            })
        }
    }
}
// ---- qwen4exp-p5 end ----

// ---- qwen4exp-p2 begin ----
// Piece 2: the batched prompt path's C ABI (include/talus_qwen4exp_kernels.h, its qwen4exp-p2 block). Resolved from
// the already loaded libtalus_kernels.so (RTLD_GLOBAL) through dlsym(RTLD_DEFAULT): `P2Ffi::load`.

pub type TkP2RunFn = unsafe extern "C" fn(
    user: *mut c_void,
    prefill: *mut c_void,
    tokens: *const i64,
    n: i64,
    pos0: i64,
    err: *mut c_char,
    err_cap: usize,
) -> c_int;
pub type TkP2DraftFn = unsafe extern "C" fn(
    user: *mut c_void,
    prefill: *mut c_void,
    mtp: *mut c_void,
    r_rows: *const f32,
    next_tokens: *const i32,
    n: i64,
    cell0: i64,
    err: *mut c_char,
    err_cap: usize,
    why: *mut *const c_char,
) -> c_int;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkP2QsaShapes {
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub n_rot: i64,
    pub idx_n_head: i64,
    pub idx_dim: i64,
    pub idx_block: i64,
    pub idx_top_k: i64,
    pub page_size: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2KvPools {
    pub k_pool: *mut u16,
    pub v_pool: *mut u16,
    pub k_q: *mut i8,
    pub v_q: *mut i8,
    pub k_scale: *mut u16,
    pub v_scale: *mut u16,
    pub k_q4: *mut u8,
    pub v_q4: *mut u8,
}
impl Default for TkP2KvPools {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}
impl TkP2KvPools {
    pub fn present(&self) -> bool {
        !self.k_pool.is_null() || !self.k_q.is_null() || !self.k_q4.is_null()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2AttnPools {
    pub k_pool: *const u16,
    pub v_pool: *const u16,
    pub k_q: *const i8,
    pub v_q: *const i8,
    pub k_scale: *const u16,
    pub v_scale: *const u16,
    pub k_q4: *const u8,
    pub v_q4: *const u8,
    pub page_table: *const i32,
}
impl Default for TkP2AttnPools {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2IdxBufs {
    pub tail: *mut f32,
    pub dead: *mut f32,
    pub pooled: *mut f32,
    pub block_pos: *mut i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2QsaInfo {
    pub kv_mode: i32,
    pub kv_int8: i32,
    pub kv_q4: i32,
    pub kv_hybrid: i32,
    pub kv_rot: i32,
    pub host_present: i32,
    pub fmt: i32,
    pub pad0: i32,
    pub n_pages: i64,
    pub max_cells: i64,
    pub idx_pooled_rows: i64,
    pub n_slots: i64,
    pub k_pool: *mut u16,
    pub v_pool: *mut u16,
    pub k_q: *mut i8,
    pub v_q: *mut i8,
    pub k_scale: *mut u16,
    pub v_scale: *mut u16,
    pub k_q4: *mut u8,
    pub v_q4: *mut u8,
    pub page_table: *mut i32,
    pub host: TkP2KvPools,
    pub attn: TkP2AttnPools,
    pub idx: TkP2IdxBufs,
}
impl Default for TkP2QsaInfo {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2Weight {
    pub wref: *const c_void,
    pub data: *const c_void,
    pub bf16: i32,
    pub hc_q8: i32,
    pub native_type: i32,
    pub pad0: i32,
    pub native_data: *const c_void,
    pub ne0: i64,
    pub ne1: i64,
}
impl Default for TkP2Weight {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkP2ExpertFmt {
    pub native: i32,
    pub gu_type: i32,
    pub d_type: i32,
    pub mmq_layer: i32,
    pub mmq_fo: i32,
    pub native_supported: i32,
    pub pad0: i32,
    pub pad1: i32,
    pub n_embd: i64,
    pub n_ff: i64,
    pub gu_row: u64,
    pub d_row: u64,
    pub up_off: u64,
    pub down_off: u64,
    pub tail_off: u64,
    pub blob_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TkP2Stats {
    pub tokens: i64,
    pub chunks: i64,
    pub ms_total: f64,
    pub ms_experts_host: f64,
    pub experts_streamed: i64,
    pub experts_dma: i64,
    pub experts_resident: i64,
    pub experts_cpu: i64,
    pub cpu_share: f64,
    pub ms_ple: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2Layout {
    pub stage_lb: i64,
    pub stage_le: i64,
    pub has_next: i32,
    pub has_hand_in: i32,
    pub has_peer: i32,
    pub has_cpu_pool: i32,
    pub has_embd_rows: i32,
    pub has_on_chunk: i32,
    pub has_on_stage_chunk: i32,
    pub has_should_stop: i32,
    pub metal: i32,
    pub borrowed: i32,
    pub device: i32,
    pub f16_io: i32,
    pub fused_bufs: i32,
    pub ring: i32,
    pub fused_ring: i32,
    pub mmq_any: i32,
    pub mmq_fallback: i32,
    pub mmq_built: i32,
    pub n_layers: i32,
    pub n_expert: i32,
    pub cpu_share_on: i32,
    pub peer_portable: i32,
    pub t: i64,
    pub t_max: i64,
    pub cap: i64,
    pub max_blocks: i64,
    pub sel_batch: i64,
    pub attn_batch: i64,
    pub stream_all_min: i64,
    pub max_blob: i64,
    pub n_embd: i64,
    pub hc: i64,
    pub hc_lr: i64,
    pub n_used: i64,
    pub n_ff: i64,
    pub conv_ch: i64,
    pub value_dim: i64,
    pub v_heads: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub idx_heads: i64,
    pub idx_dim: i64,
    pub gdn_floats: i64,
    pub gdn_state_floats: i64,
    pub ple_n_heads: i64,
    pub ng_n_embd: i64,
    pub ng_hc_dim: i64,
    pub eps: f32,
    pub stage_ring: i32,
    pub ring_max: i32,
    pub pad3: i32,
    pub mmq_layer: *const c_char,
    pub mmq_fo: *const c_char,
    pub is_qsa: *const c_char,
    pub shapes: TkP2QsaShapes,
    pub cs: *mut c_void,
    pub copy: *mut c_void,
    pub wt: *mut c_void,
    pub g: *mut c_void,
    pub ss: *mut c_void,
    pub src: *mut c_void,
    pub cache: *mut c_void,
    pub gemm: *mut c_void,
    pub mmq_ctx: *mut c_void,
    pub ple_weights: *mut c_void,
    pub ple_table: *mut c_void,
    pub ple_consts: *mut c_void,
    pub ple_scratch: *mut c_void,
    pub host_res: *const i32,
    pub stats: *mut TkP2Stats,
    pub gdn_state: *mut f32,
    pub gdn_ord0: i64,
    pub gdn_alloc: i64,
    pub qsa_ord0: i64,
    pub n_qsa_states: i64,
    pub ple_prev: *mut i32,
    pub ple_ready: i32,
    pub ple_batch_ok: i32,
    pub ple_direct: i32,
    pub pad1: i32,
    pub ple_hist: *mut f32,
    pub ple_key_bf16: *const u16,
    pub ple_key_native_data: *const c_void,
    pub ple_key_native_type: i32,
    pub pad2: i32,
    pub ple_value_bf16: *const u16,
    pub region: *mut u8,
    pub region_bytes: u64,
    pub emb: *mut f32,
    pub r: *mut f32,
    pub xn: *mut f32,
    pub grs: *mut f32,
    pub lo: *mut f32,
    pub gated: *mut f32,
    pub inj: *mut f32,
    pub mixed: *mut f32,
    pub bo: *mut f32,
    pub xn16: *mut u16,
    pub lo16: *mut u16,
    pub mixed_bf: *mut u16,
    pub mixed_h: *mut u16,
    pub xn16_lo: *mut u16,
    pub lo16_lo: *mut u16,
    pub mixed_bf_lo: *mut u16,
    pub qkv: *mut f32,
    pub z: *mut f32,
    pub ab: *mut f32,
    pub gate: *mut f32,
    pub beta: *mut f32,
    pub hbuf: *mut f32,
    pub y: *mut f32,
    pub y_h: *mut u16,
    pub kc: *mut f32,
    pub vc: *mut f32,
    pub qf: *mut f32,
    pub q: *mut f32,
    pub idx_raw: *mut f32,
    pub q_idx: *mut f32,
    pub attn: *mut f32,
    pub attn_h: *mut u16,
    pub steps_dev: *mut i32,
    pub sel_ids: *mut i32,
    pub sel_scores: *mut f32,
    pub attn_scratch: *mut f32,
    pub logits: *mut f32,
    pub w: *mut f32,
    pub gu: *mut f32,
    pub dm: *mut f32,
    pub sgate: *mut f32,
    pub sup: *mut f32,
    pub shared: *mut f32,
    pub sg: *mut f32,
    pub ids: *mut i32,
    pub slot_dev: *mut i32,
    pub src_dev: *mut i32,
    pub xs: *mut u16,
    pub hh: *mut u16,
    pub sh_h: *mut u16,
    pub xq: *mut c_void,
    pub hq: *mut c_void,
    pub h: *mut f32,
    pub ids_identity: *mut i32,
    pub bounds_dev: *mut i32,
    pub grp_gu: *mut u8,
    pub grp_d: *mut u8,
    pub xscale: *mut f32,
    pub hscale: *mut f32,
    pub grp_tail: *mut f32,
    pub row_sd: *mut f32,
    pub grp_host: *mut i32,
    pub grp_dev: *mut i32,
    pub grp_tk: u64,
    pub dq_gu: [*mut u16; 2],
    pub dq_d: [*mut u16; 2],
    pub stage_dev: *mut *mut u8,
    pub copied: *mut *mut c_void,
    pub used: *mut *mut c_void,
    pub stage_live: *mut u8,
    pub used_of: *mut i32,
    pub ple_emb: *mut f32,
    pub ple_norm: *mut f32,
    pub ple_emb_host: [*mut f32; 2],
    pub ple_pageable: [i32; 2],
    pub ple_copied: [*mut c_void; 2],
    pub ple_rows: [*mut u32; 2],
    pub tok_dev: *mut i32,
    pub stage: TkP2KvPools,
    pub ident_table: *mut i32,
}
impl Default for TkP2Layout {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2StagerJob {
    pub src: *const u8,
    pub bytes: u64,
    pub from_source: i32,
    pub l: i32,
    pub e: i32,
    pub pad0: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2Cvec {
    pub covers: i32,
    pub mode: i32,
    pub tables_ok: i32,
    pub pad0: i32,
    pub n_embd: i64,
    pub hc: i64,
    pub dir: *const f32,
    pub s: *const f32,
    pub on: *const c_int,
}
impl Default for TkP2Cvec {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkP2MtpInfo {
    pub w_ne: *const f32,
    pub w_fe: *const c_void,
    pub w_nh: *const f32,
    pub w_fh: *const c_void,
    pub w_hn: *const f32,
    pub w_dn: *const u16,
    pub w_up: *const u16,
    pub w_k: *const c_void,
    pub w_v: *const c_void,
    pub w_kn: *const f32,
    pub first_needed: i64,
    pub device: i32,
    pub hnorm_per_stream: i32,
    pub kv: TkP2QsaInfo,
}
impl Default for TkP2MtpInfo {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

type Vp = *mut c_void;
type Cvp = *const c_void;
type Ef = *const f32;
type Mf = *mut f32;
type Eh = *const u16;
type Mh = *mut u16;
type Ei = *const i32;
type Mi = *mut i32;
type Shp = *const TkP2QsaShapes;
type Kvp = *const TkP2KvPools;
type Atp = *const TkP2AttnPools;
type Ibp = *const TkP2IdxBufs;

const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

macro_rules! p2_ffi_table {
    ($( $field:ident : fn($($ty:ty),*) $(-> $ret:ty)? ; )*) => {
        /// The qwen4exp-p2 C ABI: one entry per `tk_qwen4exp_p2_<name>`.
        #[derive(Clone, Copy)]
        pub struct P2Ffi {
            pub dump_tensor: FnDumpTensor,
            $( pub $field: unsafe extern "C" fn($($ty),*) $(-> $ret)?, )*
        }
        unsafe impl Send for P2Ffi {}
        unsafe impl Sync for P2Ffi {}
        impl P2Ffi {
            /// Resolves every entry from the process's global symbols (the library was dlopen'ed RTLD_GLOBAL).
            pub fn load() -> Result<Self, String> {
                unsafe {
                    Ok(Self {
                        dump_tensor: load_sym(RTLD_DEFAULT, "tk_qwen4exp_dump_tensor")?,
                        $( $field: load_sym(RTLD_DEFAULT, concat!("tk_qwen4exp_p2_", stringify!($field)))?, )*
                    })
                }
            }
        }
    };
}

p2_ffi_table! {
    install: fn(Option<TkP2RunFn>, Option<TkP2DraftFn>, Vp);
    installed: fn() -> c_int;
    abi_sizes: fn(*mut i64);
    bind: fn(Vp, *mut TkP2Layout) -> c_int;
    stager_start: fn(Vp, *const TkP2StagerJob, i64);
    stager_wait: fn(Vp, i64) -> *const u8;
    stager_issued_one: fn(Vp, i64, Vp);
    stager_finish: fn(Vp);
    steps_upload: fn(Vp, Mi, Ei, u64, Vp) -> c_int;
    tok_upload: fn(Vp, Mi, Ei, u64, Vp) -> c_int;
    upring_h2d: fn(Vp, Vp, Cvp, u64, Vp) -> c_int;
    upring_d2h_sync: fn(Vp, Vp, Cvp, u64, Vp) -> c_int;
    should_stop: fn(Vp) -> c_int;
    on_chunk: fn(Vp, Ef, i64, i64, *mut c_char, usize) -> c_int;
    on_stage_chunk: fn(Vp, i64, *mut c_char, usize) -> c_int;
    ensure_mmq_ctx: fn(Vp) -> Vp;
    set_device: fn(i32) -> c_int;
    memcpy: fn(Vp, Cvp, u64, i32, Vp) -> c_int;
    memset: fn(Vp, i32, u64, Vp) -> c_int;
    event_create: fn(*mut Vp, i32) -> c_int;
    event_destroy: fn(Vp);
    event_record: fn(Vp, Vp) -> c_int;
    event_sync: fn(Vp) -> c_int;
    event_elapsed: fn(*mut f32, Vp, Vp) -> c_int;
    stream_wait_event: fn(Vp, Vp) -> c_int;
    stream_sync: fn(Vp) -> c_int;
    peek_error: fn(*mut c_char, usize) -> c_int;
    progress_at: fn(*const c_char, i64, i64);
    progress_beat: fn();
    pin_thread: fn();
    weight: fn(Vp, i64, *const c_char, *mut TkP2Weight) -> c_int;
    hc_bf16_view: fn(Cvp, Vp, i32, i32) -> Eh;
    hc_bf16_release: fn(Vp, i32, i32);
    embed_kind: fn(Vp, Vp, *mut i64) -> c_int;
    embed_gather: fn(Vp, Ei, i64, Mf, Vp);
    embed_row: fn(Vp, Vp, i64, Mf, Vp, *mut c_char, usize) -> c_int;
    gemm_bf16: fn(Vp, Eh, Eh, Mf, i64, i64, i64, i64, f32, i64);
    gemm_f16: fn(Vp, Eh, Eh, Mf, i64, i64, i64);
    gemm_native: fn(Vp, Eh, i32, Cvp, Mf, i64, i64, i64, i64, f32, i64);
    gemm_bf16_hcd_exact: fn(Vp, Eh, i64, Eh, Mf, i64, i64, i64) -> c_int;
    gr_broadcast: fn(Ef, Mf, i64, Vp);
    gr_norm: fn(Ef, Ef, f32, Mf, Mh, i64, Vp, Mh);
    gr_norm_rs: fn(Ef, Ef, f32, Mf, Mh, i64, Vp, Mh, i64);
    gr_mix_r: fn(Ef, Ef, Ef, Ef, Mf, Mh, i64, Vp, Mh, Mh);
    gr_mix: fn(Ef, Ef, Mf, Mh, i64, Vp, Mh, Mh);
    gr_upmix: fn(Eh, Eh, Ef, Ef, Ef, Mf, Mh, Mh, i64, Vp) -> c_int;
    gr_silu: fn(Ef, Mh, i64, Vp, Mh);
    gr_write: fn(Mf, Ef, Ef, i64, i64, Vp);
    gr_write_norm_rs: fn(Mf, Ef, Ef, i64, Ef, f32, Mf, Mh, i64, Vp, Mh, i64);
    gr_write_cvec_norm_rs: fn(Mf, Ef, Ef, i64, Ef, Ef, *const c_int, i32, Ef, f32, Mf, Mh, i64, Vp, Mh, i64);
    gr16_broadcast: fn(Ef, Mh, i64, i32, Vp);
    gr16_norm_rs: fn(Eh, Ef, f32, Mf, Mh, i64, i32, Vp, Mh, i64);
    gr16_mix_r: fn(Eh, Ef, Ef, Ef, Mf, Mh, i64, i32, Vp, Mh, Mh);
    gr16_upmix: fn(Eh, Eh, Eh, Ef, Ef, Mf, Mh, Mh, i64, i32, Vp) -> c_int;
    gr16_write_norm_rs: fn(Mh, Ef, Ef, i64, Ef, f32, Mf, Mh, i64, i32, Vp, Mh, i64);
    gr16_write: fn(Mh, Ef, Ef, i64, i64, i32, Vp);
    gr16_widen: fn(Eh, Mf, i64, i32, Vp);
    gr16_narrow: fn(Ef, Mh, i64, i32, Vp);
    pf_hcdown_bf16: fn(Eh, i64, Eh, Eh, i64, i64, Mf, Mf, i64, i64, Vp) -> c_int;
    cvec_info: fn(i64, *mut TkP2Cvec);
    cvec_apply: fn(Mf, i64, i64, i64, Vp);
    ple_ngram_rows: fn(Vp, Ei, Ei, i32, *mut u32);
    ple_gather_batch: fn(Vp, *const u32, u64, Mf, *mut c_char, usize) -> c_int;
    ple_postops_batch: fn(Vp, Mf, Mf, Ef, Mf, Mf, Mf, Mf, i32, Vp, *mut c_char, usize) -> c_int;
    ple_block: fn(Vp, Ef, Ef, Ef, Mf, Mf, Vp, Vp, *mut c_char, usize) -> c_int;
    ple_history_advance: fn(Mf, Ef, Vp);
    to_f16: fn(Ef, Mh, i64, Vp);
    to_bf16: fn(Ef, Mh, i64, Vp, Mh);
    gdn_gates: fn(Ef, Ef, Ef, Mf, Mf, i64, Vp);
    gdn_conv: fn(Mf, Ef, Ef, Mf, i64, f32, Vp);
    gdn_recurrence: fn(Mf, Ef, Ef, Ef, Ef, Ef, f32, Mf, Mh, i64, Vp, i64);
    qsa_info: fn(Vp, i64, *mut TkP2QsaInfo) -> c_int;
    rms_rows: fn(Mf, Ef, i64, i64, i64, f32, Vp);
    rope: fn(Mf, i64, i64, i64, i64, i64, Vp);
    split_q: fn(Ef, Mf, i64, Vp);
    gate_attn: fn(Ef, Ef, Mh, i64, Vp, i64);
    fwht256: fn(Mf, i64, Vp);
    kv_stage_from_host: fn(Atp, Kvp, i32, i64, Shp, Vp);
    kv_unstage_to_host: fn(Atp, Kvp, i32, i64, i64, Shp, Vp);
    kv_append: fn(Ef, Ef, i64, i64, Ei, i64, Mh, Mh, *mut i8, *mut i8, Mh, Mh, Vp, Kvp, Kvp);
    kv_append_q4: fn(*mut u8, *mut u8, Ei, i64, i64, Ef, Ef, Shp, Vp, Kvp, Kvp);
    indexer_append_batch: fn(Ef, i64, i64, Ef, f32, Ibp, Shp, i64, Vp, *mut c_char, usize) -> c_int;
    indexer_append: fn(Ef, Ei, Ef, f32, Ibp, Shp, i64, Vp, *mut c_char, usize) -> c_int;
    block_scores_tc: fn(Ef, Ef, Ef, Ei, i64, i64, Shp, Mf, Vp, i64) -> c_int;
    block_scores: fn(Ef, Ef, Ef, Ei, i64, i64, Shp, Mf, Vp, i64);
    block_topk: fn(Ef, Ei, i64, i64, i64, Shp, Mi, Vp, i64);
    fnp_tile_attn: fn(Ef, Atp, Ei, Ei, i64, Shp, Mf, i64, Vp) -> c_int;
    prompt_attn_batch: fn(Ef, Atp, Ei, Ei, i64, Shp, Mf, i64, Vp) -> c_int;
    decode_attn_batch: fn(Ef, Atp, Ei, Ei, i64, Shp, Mf, Mf, i64, Vp);
    route: fn(Ef, Mi, Mf, i64, i64, Vp);
    swiglu_pair: fn(Ef, Ef, Mh, i64, Vp);
    swiglu_interleaved: fn(Ef, Mh, i64, Vp);
    copy_i32: fn(Mi, Ei, i64, Vp);
    gather_rows16: fn(Eh, Ei, Mh, i64, i64, Vp);
    moe_combine: fn(Ef, Ei, Ef, Ef, Ef, Mf, i64, Vp, Ef);
    expert_fmt: fn(i64, *mut TkP2ExpertFmt) -> c_int;
    fused_native_supported: fn(i32, i32) -> c_int;
    fused_enabled: fn() -> c_int;
    fused_quantize_act: fn(Ef, i64, i64, Vp, Vp, i32);
    fused_group: fn(Ei, i64, i32, i32, Vp, Mi, Mi, Vp);
    fused_experts: fn(i32, *const TkP2ExpertFmt, i32, i32, *const *const u8, i32, i64, Cvp, Cvp, Ei, Vp, Mf, Vp);
    fused_max_batch: fn() -> i32;
    mmq_matrix_bytes: fn(i32, i64, i64) -> u64;
    mmq_fp4_activations: fn(i32) -> c_int;
    mmq_fits: fn(i32, i64) -> c_int;
    mmq_quantize: fn(Ef, Ei, Vp, i32, i64, i64, i64, Vp, Mf);
    mmq_gather_native: fn(Cvp, Cvp, u64, Cvp, u64, Vp, Vp, Vp, Cvp, Vp, u64);
    mmq_gather_native_group: fn(*const *const u8, i32, i32, u64, u64, u64, u64, Vp, u64, Vp, u64, Vp, u64, Vp, u64) -> c_int;
    mmq_gather_strata_q2: fn(*const u8, Vp, Vp, Vp);
    mmq_product: fn(Vp, Cvp, i32, i64, i64, u64, i32, Cvp, Ei, Ei, i64, i64, Mf, i64, Ef, i32, Vp);
    mmq_swiglu: fn(Ef, Mf, i64, i64, i32, Vp);
    mmq_swiglu_scaled: fn(Ef, Mf, i64, i64, Ei, i32, Ef, i64, Vp);
    mmq_down_row_scales: fn(Mf, Ei, i32, Ef, i64, Vp);
    mmq_iota: fn(Mi, i64, Vp);
    mmq_q8_bytes: fn(i64, i64) -> u64;
    iq_dequant_gu_f16: fn(i32, Cvp, Cvp, i64, i64, Mh, Vp, Ef);
    iq_dequant_f16: fn(i32, Cvp, i64, Mh, Vp, Ef);
    blob_dequant_f16: fn(*const u8, Mh, Mh, Vp);
    src_transient: fn(Vp, i64, i64) -> c_int;
    src_pinned: fn(Vp, i64, i64) -> c_int;
    src_blob_stable: fn(Vp, i64, i64) -> *const u8;
    cache_device_slot: fn(Vp, i32) -> *const u8;
    mtp_info: fn(Vp, *mut TkP2MtpInfo) -> c_int;
    mtp_idle: fn(Vp, *mut c_char, usize) -> c_int;
    mtp_add_ms_prefill: fn(Vp, f64);
    rms_norm_grouped: fn(Ef, Ef, Mf, i32, i32, i32, f32, Vp);
    add_streams_broadcast: fn(Ef, Ef, Mf, i64, i32, i32, Vp);
    mmq_run: fn(Vp, Cvp, i32, i64, i64, u64, i32, Cvp, Ei, Ei, i64, i64, Mf, i64, Vp);
}
// ---- qwen4exp-p2 end ----
