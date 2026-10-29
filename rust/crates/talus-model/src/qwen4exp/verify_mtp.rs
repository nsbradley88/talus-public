//! Piece 4 (`qwen4exp-p4`): Flash-Next (`qwen4exp`) multi-token speculative `Verifier` (`verify.cpp`)
//! and `MtpDrafter` (`mtp.cpp`) host plane, coupled / probabilistic draft acceptance (`spec_prob.hpp`,
//! `coupled_draft.hpp`), and the serve / generate verify-commit-draft window bodies (`generate.cpp`).
//!
//! ## What runs here
//! - `RustVerifier::run` = `Verifier::run`: input staging (positions, QSA step records), the PLE n-gram rows,
//!   the window graph's launch, the per-layer doorbell service (the CPU expert pool, the GPU plan slot, flags
//!   A/B and the ring, the adaptive overlap gate), the head's host-side sampling (exact match or rejection
//!   sampling against the drafter's q lists), the picks and the non-finite guard.
//! - `RustVerifier::commit` / `wait_commit` / `copy_logits` / `window_logprobs` / `set_sampling`.
//! - `RustMtp::draft` / `draft_first` = `MtpDrafter::draft` / `draft_first`: the round's step records, the
//!   round and step graph launches with the `--spec-min-p` early stop, the PLE prefetch of the drafts;
//!   `set_draft_sampling` / `set_draft_history` (coupled drafting's chain and penalty ring).
//! - `Piece4VerifyMtp::{serve,gen}_*`: the `tk_qwen4exp_serve_{verify,commit,draft}_window` and
//!   `tk_qwen4exp_gen_spec_*` bodies (penalty rows, the q-list hand-off, the TALUS_FN_DUMP tensors,
//!   the non-finite verdict, the error protocol), with the generate-local parts (adaptive tier, K/V growth,
//!   timing) behind `TK_P4_HOOK_*`.
//!
//! ## What stays C++ (narrow wrappers, `src/program/qwen4exp_p4.cpp`)
//! - The captured graphs: `Verifier::capture` / `record_window` (the T-token 48-layer sequence) and
//!   `capture_commit`, `MtpDrafter::capture_round` / `capture_step` (`record_front` / `record_rest`).  They
//!   move once Piece 3's per-layer launch API exists.  The objects' storage (mapped staging, device arena,
//!   streams) stays in the C++ `Verifier` / `MtpDrafter`; Rust works on views of it.
//! - `Verifier::init` / `MtpDrafter::load` / `bind`, the PLE table (`ngram_rows`, `prefetch_rows`,
//!   `gather_batch`: Piece 1's `PleTable`), the sampler kernels (`sample_tokens`, `sample_tokens_spec`).
//! - Layer splits, remote expert opt, pipelined windows, batch windows, and the debug traces
//!   (`STRATA_VERIFY_TRACE`, `STRATA_VERIFY_PROFILE`, `STRATA_MTP_PROFILE`, `STRATA_DRAFT_TIMING`): those
//!   runs take the C++ entry points (`eligible == 0`).
//!
//! ## Bitwise verification (`TALUS_FN_DUMP`)
//! Compare `verify_p{pos}_T{t}_final_R.bin`, `verify_p{pos}_T{t}_logits0.bin`, and `first_logits.bin`
//! between `TALUS_FN_RUST=0` and `TALUS_FN_RUST=1` (`talus --model qwen4exp --fn-dump-compare A B`).
//! `TALUS_P4_CPP=1` keeps the C++ window bodies under `TALUS_FN_RUST=1` (A/B switch).

#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::Write;
use std::sync::atomic::{fence, AtomicBool, AtomicI32, AtomicI64, AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use hip_sys::{
    hipErrorNotReady, hipError_t, hipEventRecord, hipEventSynchronize, hipEvent_t,
    hipGetErrorString, hipGetLastError, hipGraphExec_t, hipGraphLaunch, hipMalloc, hipMemcpy,
    hipMemcpyAsync, hipMemcpyDeviceToDevice, hipMemcpyDeviceToHost, hipMemcpyHostToDevice,
    hipStreamQuery, hipStreamSynchronize, hipStream_t, hipSuccess,
};

use crate::qwen4exp::ffi::{load_sym_from, Qwen4ExpKernels, TkQwen4ExpGenCtx, TkQwen4ExpServeCtx};

// ---- qwen4exp-r2v: Verifier::init / set_logit_bias and MtpDrafter::load / bind in Rust (verify_mtp/setup.rs)
mod setup;
use crate::qwen4exp::skeleton::Qwen4ExpModelConfig;

// ============================================================================
// Constants shared with the C++ side
// ============================================================================

/// `strata::kernels::kVerifyMaxT`: the widest window (verify_kernels.hpp).
pub const K_VERIFY_MAX_T: usize = 8;
/// `strata::core::kSpecQStride`: one q row = 64 ids then 64 probabilities as float bits (spec_prob.hpp).
pub const K_SPEC_Q_STRIDE: usize = 128;
/// `strata::core::kCoupledHistCap`: the coupled drafter's penalty window cap (coupled_draft.hpp).
pub const K_COUPLED_HIST_CAP: i32 = 4096;
/// `strata::kernels::PLE_N_HEADS`: n-gram rows per token (ngram.hpp).
pub const PLE_N_HEADS: usize = 16;
/// `strata::kernels::kStepCount` (qsa.hpp).
pub const K_STEP_COUNT: usize = 4;

const TK_P4_HOOK_PRE_WINDOW: c_int = 0;
const TK_P4_HOOK_ENSURE_KV: c_int = 1;
const TK_P4_HOOK_WINDOW_STARTED: c_int = 2;
const TK_P4_HOOK_POST_WINDOW: c_int = 3;
const TK_P4_HOOK_FIRST_VERIFIED: c_int = 4;
const TK_P4_HOOK_COMMIT_BEGIN: c_int = 5;
const TK_P4_HOOK_JOIN_ADAPT: c_int = 6;
const TK_P4_HOOK_COMMIT_DONE: c_int = 7;
const TK_P4_HOOK_MARK_NONFINITE: c_int = 8;
const TK_P4_HOOK_DRAFT_DONE: c_int = 9;

// ============================================================================
// C mirrors (include/talus_qwen4exp_kernels.h qwen4exp-p4 block, and two strata structs)
// ============================================================================

#[repr(C)]
pub struct TkP4Hooks {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4HooksInfo {
    pub serve: i32,
    pub use_mtp: i32,
    pub spec_corrupt: i32,
    pub commit_async: i32,
    pub hist_n: *mut i32,
    pub d_hist: *mut i32,
    pub hist_cap: i32,
    pub pool: *mut c_void,
    pub pool_user: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4VerView {
    pub ver: *mut c_void,
    pub eligible: i32,
    pub max_t: i32,
    pub n_layers: i32,
    pub lb: i32,
    pub le: i32,
    pub n_embd: i32,
    pub hc: i32,
    pub k: i32,
    pub n_head: i32,
    pub n_head_kv: i32,
    pub idx_q_heads: i32,
    pub step_count: i32,
    pub idx_block: i64,
    pub idx_top_k: i64,
    pub n_vocab: i64,
    pub h_tok: *mut i32,
    pub h_step: *mut i32,
    pub h_pos: *mut i32,
    pub h_commit: *mut i32,
    pub h_out: *mut i32,
    pub m_out: *mut i32,
    pub h_ple: *mut f32,
    pub h_nf: *mut u32,
    pub h_x: *mut f32,
    pub h_ids: *mut i32,
    pub h_ymiss: *mut f32,
    pub h_seq: *mut u32,
    pub h_flag: *mut u32,
    pub h_flag_a: *mut u32,
    pub h_flag_b: *mut u32,
    pub h_plan_err: *mut u32,
    pub h_plan: *mut i32,
    pub plan_i32: i64,
    pub sink: *mut c_void,
    pub staging: *mut u8,
    pub staging_blobs: i64,
    pub max_blob: i64,
    pub cur_layer: *mut u32,
    pub groups: *mut i32,
    pub nonfinite_blocks: i32,
    pub verify_max_t: i32,
    pub head_logits: *mut f32,
    pub r: *mut f32,
    pub stream: *mut c_void,
    pub copy_stream: *mut c_void,
    pub commit_done: *mut c_void,
    pub commit_exec: *mut *mut c_void,
    pub last_t: *mut i32,
    pub last_pos0: *mut i64,
    pub last_tokens: *mut i32,
    pub staged: *mut u8,
    pub last_batch: *mut u8,
    pub copy_used: *mut u8,
    pub commit_pending: *mut u8,
    pub always_publish: *mut u8,
    pub head_sampling: *mut u8,
    pub nf_row: *mut i32,
    pub nf_pos: *mut i64,
    pub nf_mask: *mut u32,
    pub ms_wait: *mut f64,
    pub ms_pool: *mut f64,
    pub ms_host: *mut f64,
    pub ms_commit: *mut f64,
    pub ms_gate: *mut f64,
    pub windows: *mut i64,
    pub spec_windows: *mut i64,
    pub sampling: *mut c_void,
    pub hist_d: *mut *const i32,
    pub hist_len: *mut i32,
    pub released: *const u8,
    pub adapt_gate: *mut c_void,
    pub d_spec_slot: *mut *mut i32,
    pub ple_prev: *mut i32,
    pub ple_on: i32,
    pub ple_ss: *mut c_void,
    pub max_cells: *const i64,
    pub sizeof_sink: i32,
    pub sizeof_sampler: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4GateView {
    pub landed: *const i32,
    pub active: *const u8,
    pub hland: *const i32,
    pub has: *const u8,
    pub n_has: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4MtpView {
    pub mtp: *mut c_void,
    pub eligible: i32,
    pub max_t: i32,
    pub n_head: i32,
    pub max_drafts: *const i32,
    pub hcn: i64,
    pub max_cells: *const i64,
    pub h_tok: *mut i32,
    pub h_step: *mut i32,
    pub h_pos: *mut i32,
    pub h_row: *mut i32,
    pub h_out: *mut i32,
    pub h_prob: *mut f32,
    pub h_force: *mut i32,
    pub stream: *mut c_void,
    pub window_r: *mut f32,
    pub src_r: *mut *const f32,
    pub coupled_active: *mut u8,
    pub coupled_ok: u8,
    pub chain_live: *mut u8,
    pub cparams: *mut c_void,
    pub h_chist: *mut i32,
    pub h_q: *const i32,
    pub spec_q_stride: i32,
    pub coupled_hist_cap: i32,
    pub ms_draft: *mut f64,
    pub ms_src: *mut f64,
    pub rounds: *mut i64,
    pub n_src: *mut i64,
    pub ple_prev: *mut i32,
    pub ple_on: i32,
    pub ple_ss: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4CaptureView {
    pub exec: *mut *mut c_void,
    pub exec_nr: *mut *mut c_void,
    pub all_resident: *mut u8,
    pub ar_off: *mut u8,
    pub h_res: *const i32,
    pub n_expert: i64,
    pub commit: *mut i32,
    pub m_commit: *const i32,
    pub qkv_l: *mut f32,
    pub h_l: *mut f32,
    pub gate_l: *mut f32,
    pub beta_l: *mut f32,
    pub z: *mut f32,
    pub y_dummy: *mut f32,
    pub idx_raw_l: *mut f32,
    pub tail_snap: *mut f32,
    pub hist_snap: *mut f32,
    pub ple_hist: *mut f32,
    pub ple_copy: i32,
    pub gdn_state: *mut f32,
    pub gdn_ord0: i64,
    pub ssm_state_size: i64,
    pub ssm_k_heads: i64,
    pub ssm_v_heads: i64,
    pub ssm_d_conv: i64,
    pub ssm_conv_channels: i64,
    pub idx_key_dim: i64,
    pub ng_hist_hc: i64,
    pub layer_is_qsa: *const u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4QsaState {
    pub idx_tail: *mut f32,
    pub idx_dead: *mut f32,
    pub idx_pooled: *mut f32,
    pub idx_block_pos: *mut i32,
    pub max_cells: i64,
}

/// `strata::kernels::QsaIndexerLayers::kMax`.
const K_INDEXER_LAYERS_MAX: usize = 16;
/// verify.cpp EPS.
const VERIFY_EPS: f32 = 1e-6;

/// `strata::kernels::SamplerParams` (sampler.hpp), field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerParams {
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub temperature: f32,
    pub min_keep: i32,
    pub penalty_last_n: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub seed: u64,
    pub counter: u64,
    pub greedy: bool,
    pub logit_bias: *const f32,
    pub gumbel: bool,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            temperature: 1.0,
            min_keep: 1,
            penalty_last_n: 0,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            seed: 0,
            counter: 0,
            greedy: false,
            logit_bias: std::ptr::null(),
            gumbel: false,
        }
    }
}

/// `strata::core::GpuPlanSink` (expert_source.hpp), field for field: the plan the CPU pool writes per layer.
#[repr(C)]
pub struct GpuPlanSink {
    pub counts: *mut i32,
    pub start: *mut i32,
    pub dst: *mut i32,
    pub tok: *mut i32,
    pub ptr: *mut u64,
    pub ptr2: *mut u64,
    pub start2: *mut i32,
    pub staging: u64,
    pub staging_cap: i64,
    pub cap: i64,
    pub publish: *mut c_void,
    pub fetch: *mut c_void,
    pub ctx: *mut c_void,
    pub pcie_mode: c_int,
}

/// `strata::core::PoolMultiFn`: the CPU pool for a window's layer.
pub type PoolMultiFn = unsafe extern "C" fn(
    user: *mut c_void,
    x_f: *const f32,
    ids: *const i32,
    n_tok: i64,
    k: i64,
    out: *mut f32,
    layer: i64,
);

// ============================================================================
// The tk_qwen4exp_p4_* table
// ============================================================================

type FnHooks = unsafe extern "C" fn(*mut c_void) -> *mut TkP4Hooks;
type FnHooksInfo = unsafe extern "C" fn(*mut TkP4Hooks, *mut TkP4HooksInfo) -> c_int;
type FnHook = unsafe extern "C" fn(*mut TkP4Hooks, c_int, i64, i64) -> c_int;
type FnDriveFailed = unsafe extern "C" fn(*mut TkP4Hooks) -> *const c_char;
type FnVerView = unsafe extern "C" fn(*mut TkP4Hooks, *mut TkP4VerView) -> c_int;
type FnMtpView = unsafe extern "C" fn(*mut TkP4Hooks, *mut TkP4MtpView) -> c_int;
type FnGateView = unsafe extern "C" fn(*mut c_void, *mut TkP4GateView) -> c_int;
type FnVerPrepare = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    *mut *mut c_void,
    *mut c_int,
    *mut c_char,
    usize,
) -> c_int;
type FnVerRelease = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type FnMtpPrepare = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    *mut *mut c_void,
    *mut *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
type FnPleRows = unsafe extern "C" fn(*mut c_void, *const i32, c_int, *mut i32, *mut u32) -> c_int;
type FnPleGather =
    unsafe extern "C" fn(*mut c_void, *const u32, usize, *mut f32, *mut c_char, usize) -> c_int;
type FnSample = unsafe extern "C" fn(
    *const f32,
    c_int,
    c_int,
    *const i32,
    c_int,
    *const c_void,
    *mut i32,
    *mut c_void,
) -> c_int;
type FnSampleSpec = unsafe extern "C" fn(
    *const f32,
    c_int,
    c_int,
    *const i32,
    c_int,
    *const c_void,
    *const i32,
    *const i32,
    c_int,
    *mut i32,
    *mut c_void,
) -> c_int;
type FnProgress = unsafe extern "C" fn(*const c_char, i64, c_int);
type FnCapView = unsafe extern "C" fn(*mut c_void, *mut TkP4CaptureView) -> c_int;
type FnVerViewOf = unsafe extern "C" fn(*mut c_void, *mut TkP4VerView) -> c_int;
type FnMtpViewOf = unsafe extern "C" fn(*mut c_void, *mut TkP4MtpView) -> c_int;
type FnMtpPfView = unsafe extern "C" fn(*mut c_void, *mut TkP4MtpPfView) -> c_int;
type FnMtpPrefillGraph =
    unsafe extern "C" fn(*mut c_void, c_int, c_int, *mut *mut c_void, *mut c_char, usize) -> c_int;
type FnMtpPfUpload =
    unsafe extern "C" fn(*mut c_void, *const i32, i64, *mut *mut i32, *mut c_char, usize) -> c_int;
type FnInstall = unsafe extern "C" fn(*const TkP4Ops);
type FnMtpRecView = unsafe extern "C" fn(*mut c_void, *mut TkP4MtpRecView) -> c_int;
type FnMtpTensor =
    unsafe extern "C" fn(*mut c_void, *const c_char, c_int, *mut c_int) -> *const c_void;
type FnMOp = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    *const i32,
    *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
type FnMtpInstantiate =
    unsafe extern "C" fn(*mut c_void, c_int, c_int, *mut c_void, *mut c_char, usize) -> c_int;
type FnGRmsNorm =
    unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, f32, *mut c_void) -> c_int;
type FnGRmsNormGrouped = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *mut f32,
    c_int,
    c_int,
    c_int,
    f32,
    *mut c_void,
) -> c_int;
type FnGQuant = unsafe extern "C" fn(*const f32, *mut c_void, c_int, c_int, *mut c_void) -> c_int;
type FnGMmvq = unsafe extern "C" fn(
    c_int,
    *const c_void,
    *const c_void,
    *mut f32,
    c_int,
    c_int,
    c_int,
    *mut c_void,
) -> c_int;
type FnGAddStreams =
    unsafe extern "C" fn(*const f32, *const f32, *mut f32, i64, c_int, c_int, *mut c_void) -> c_int;
type FnGRmsNormRope = unsafe extern "C" fn(
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
    c_int,
    c_int,
    f32,
    *const i32,
    *mut c_void,
) -> c_int;
type FnGRope = unsafe extern "C" fn(
    *const f32,
    *mut f32,
    c_int,
    c_int,
    c_int,
    *const i32,
    *mut c_void,
) -> c_int;
type FnGRopeNeox = unsafe extern "C" fn(
    *const f32,
    *mut f32,
    i64,
    c_int,
    c_int,
    *const f32,
    *const f32,
    *const i32,
    *mut c_void,
) -> c_int;
type FnGFwht = unsafe extern "C" fn(*mut f32, i64, *mut c_void) -> c_int;
type FnGCopy2d = unsafe extern "C" fn(
    *mut c_void,
    usize,
    *const c_void,
    usize,
    usize,
    usize,
    *mut c_void,
) -> c_int;
type FnGGate =
    unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, *mut c_void) -> c_int;
type FnEnvMtp = unsafe extern "C" fn() -> c_int;
type FnQsaState = unsafe extern "C" fn(*mut c_void, i64, *mut TkP4QsaState) -> c_int;
type FnLayerWeight =
    unsafe extern "C" fn(*mut c_void, i64, *const c_char, *mut c_char, usize) -> *const f32;
type FnRecordWindow = unsafe extern "C" fn(*mut c_void, c_int, *mut c_char, usize) -> c_int;
type FnWindowInstantiate =
    unsafe extern "C" fn(*mut c_void, c_int, *mut c_void, *mut c_char, usize) -> c_int;
type FnKCopyI32 = unsafe extern "C" fn(*mut i32, *const i32, i64, *mut c_void) -> c_int;
type FnKCopyF32 = unsafe extern "C" fn(*mut f32, *const f32, i64, *mut c_void) -> c_int;
type FnKCopyIndexed =
    unsafe extern "C" fn(*mut f32, *const f32, i64, *const i32, i64, *mut c_void) -> c_int;
type FnKGdnConvCommit =
    unsafe extern "C" fn(*mut f32, *const f32, c_int, *const i32, *mut c_void) -> c_int;
type FnKGdnStepNormMulti = unsafe extern "C" fn(
    *mut f32,
    *const f32,
    c_int,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    f32,
    *mut f32,
    c_int,
    c_int,
    c_int,
    *const i32,
    *mut c_void,
    c_int,
) -> c_int;
type FnKGdnCommitLayers = unsafe extern "C" fn(
    *mut f32,
    i64,
    i64,
    *const f32,
    i64,
    *const f32,
    i64,
    *const f32,
    *const f32,
    i64,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const i32,
    *mut c_void,
) -> c_int;
type FnKIdxStepsLayers = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    i64,
    *const i32,
    c_int,
    c_int,
    i32,
    f32,
    *const *const f32,
    *const *const f32,
    *const *mut f32,
    *const *mut f32,
    *const *mut f32,
    *const *mut i32,
    c_int,
    i64,
    *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
type FnKIdxSteps = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const i32,
    c_int,
    c_int,
    i32,
    *const f32,
    f32,
    *const TkP4QsaState,
    *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
type FnKIdx = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const i32,
    i32,
    *const f32,
    f32,
    *const TkP4QsaState,
    *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
type FnEnvFlags = unsafe extern "C" fn() -> c_int;

#[derive(Clone, Copy)]
struct P4Ffi {
    serve_hooks: FnHooks,
    gen_hooks: FnHooks,
    hooks_info: FnHooksInfo,
    hook: FnHook,
    drive_failed: FnDriveFailed,
    ver_view: FnVerView,
    mtp_view: FnMtpView,
    gate_view: FnGateView,
    ver_prepare: FnVerPrepare,
    ver_release: FnVerRelease,
    mtp_prepare: FnMtpPrepare,
    ple_rows: FnPleRows,
    ple_gather: FnPleGather,
    sample_tokens: FnSample,
    sample_tokens_spec: FnSampleSpec,
    progress: FnProgress,
    env_flags: FnEnvFlags,
    cap_view: FnCapView,
    qsa_state: FnQsaState,
    layer_weight: FnLayerWeight,
    record_window: FnRecordWindow,
    window_instantiate: FnWindowInstantiate,
    k_copy_i32_from_mapped: FnKCopyI32,
    k_copy_from_mapped: FnKCopyF32,
    k_copy_indexed: FnKCopyIndexed,
    k_gdn_conv_commit: FnKGdnConvCommit,
    k_gdn_step_norm_multi: FnKGdnStepNormMulti,
    k_gdn_commit_layers: FnKGdnCommitLayers,
    k_idx_steps_layers: FnKIdxStepsLayers,
    k_idx_steps: FnKIdxSteps,
    k_idx: FnKIdx,
    ver_view_of: FnVerViewOf,
    mtp_view_of: FnMtpViewOf,
    mtp_pf_view: FnMtpPfView,
    mtp_prefill_graph: FnMtpPrefillGraph,
    mtp_pf_upload: FnMtpPfUpload,
    install: FnInstall,
    mtp_rec_view: FnMtpRecView,
    mtp_tensor: FnMtpTensor,
    m_op: FnMOp,
    mtp_instantiate: FnMtpInstantiate,
    g_rms_norm: FnGRmsNorm,
    g_rms_norm_grouped: FnGRmsNormGrouped,
    g_quantize_q8_1: FnGQuant,
    g_mmvq: FnGMmvq,
    g_add_streams_broadcast: FnGAddStreams,
    g_rms_norm_rope: FnGRmsNormRope,
    g_rope: FnGRope,
    g_rope_neox: FnGRopeNeox,
    g_fwht256: FnGFwht,
    g_copy_2d: FnGCopy2d,
    g_gate_apply: FnGGate,
    env_mtp: FnEnvMtp,
}

unsafe impl Send for P4Ffi {}
unsafe impl Sync for P4Ffi {}

static P4_FFI: OnceLock<Option<P4Ffi>> = OnceLock::new();

fn p4_ffi(kernels: &Qwen4ExpKernels) -> Option<&'static P4Ffi> {
    P4_FFI
        .get_or_init(|| unsafe {
            let h = kernels.raw_handle();
            let r = (|| -> Result<P4Ffi, String> {
                Ok(P4Ffi {
                    serve_hooks: load_sym_from(h, "tk_qwen4exp_p4_serve_hooks")?,
                    gen_hooks: load_sym_from(h, "tk_qwen4exp_p4_gen_hooks")?,
                    hooks_info: load_sym_from(h, "tk_qwen4exp_p4_hooks_info")?,
                    hook: load_sym_from(h, "tk_qwen4exp_p4_hook")?,
                    drive_failed: load_sym_from(h, "tk_qwen4exp_p4_drive_failed")?,
                    ver_view: load_sym_from(h, "tk_qwen4exp_p4_ver_view")?,
                    mtp_view: load_sym_from(h, "tk_qwen4exp_p4_mtp_view")?,
                    gate_view: load_sym_from(h, "tk_qwen4exp_p4_gate_view")?,
                    ver_prepare: load_sym_from(h, "tk_qwen4exp_p4_ver_prepare")?,
                    ver_release: load_sym_from(h, "tk_qwen4exp_p4_ver_release")?,
                    mtp_prepare: load_sym_from(h, "tk_qwen4exp_p4_mtp_prepare")?,
                    ple_rows: load_sym_from(h, "tk_qwen4exp_p4_ple_rows")?,
                    ple_gather: load_sym_from(h, "tk_qwen4exp_p4_ple_gather")?,
                    sample_tokens: load_sym_from(h, "tk_qwen4exp_p4_sample_tokens")?,
                    sample_tokens_spec: load_sym_from(h, "tk_qwen4exp_p4_sample_tokens_spec")?,
                    progress: load_sym_from(h, "tk_qwen4exp_p4_progress")?,
                    env_flags: load_sym_from(h, "tk_qwen4exp_p4_env_flags")?,
                    cap_view: load_sym_from(h, "tk_qwen4exp_p4_capture_view")?,
                    qsa_state: load_sym_from(h, "tk_qwen4exp_p4_qsa_state")?,
                    layer_weight: load_sym_from(h, "tk_qwen4exp_p4_layer_weight")?,
                    record_window: load_sym_from(h, "tk_qwen4exp_p4_record_window")?,
                    window_instantiate: load_sym_from(h, "tk_qwen4exp_p4_window_instantiate")?,
                    k_copy_i32_from_mapped: load_sym_from(h, "tk_qwen4exp_p4_k_copy_i32_from_mapped")?,
                    k_copy_from_mapped: load_sym_from(h, "tk_qwen4exp_p4_k_copy_from_mapped")?,
                    k_copy_indexed: load_sym_from(h, "tk_qwen4exp_p4_k_copy_indexed")?,
                    k_gdn_conv_commit: load_sym_from(h, "tk_qwen4exp_p4_k_gdn_conv_commit")?,
                    k_gdn_step_norm_multi: load_sym_from(h, "tk_qwen4exp_p4_k_gdn_step_norm_multi")?,
                    k_gdn_commit_layers: load_sym_from(h, "tk_qwen4exp_p4_k_gdn_commit_layers")?,
                    k_idx_steps_layers: load_sym_from(h, "tk_qwen4exp_p4_k_indexer_append_steps_layers")?,
                    k_idx_steps: load_sym_from(h, "tk_qwen4exp_p4_k_indexer_append_steps")?,
                    k_idx: load_sym_from(h, "tk_qwen4exp_p4_k_indexer_append")?,
                    ver_view_of: load_sym_from(h, "tk_qwen4exp_p4_ver_view_of")?,
                    mtp_view_of: load_sym_from(h, "tk_qwen4exp_p4_mtp_view_of")?,
                    mtp_pf_view: load_sym_from(h, "tk_qwen4exp_p4_mtp_pf_view")?,
                    mtp_prefill_graph: load_sym_from(h, "tk_qwen4exp_p4_mtp_prefill_graph")?,
                    mtp_pf_upload: load_sym_from(h, "tk_qwen4exp_p4_mtp_pf_upload")?,
                    install: load_sym_from(h, "tk_qwen4exp_p4_install")?,
                    mtp_rec_view: load_sym_from(h, "tk_qwen4exp_p4_mtp_rec_view")?,
                    mtp_tensor: load_sym_from(h, "tk_qwen4exp_p4_mtp_tensor")?,
                    m_op: load_sym_from(h, "tk_qwen4exp_p4_m_op")?,
                    mtp_instantiate: load_sym_from(h, "tk_qwen4exp_p4_mtp_instantiate")?,
                    g_rms_norm: load_sym_from(h, "tk_qwen4exp_p4_g_rms_norm")?,
                    g_rms_norm_grouped: load_sym_from(h, "tk_qwen4exp_p4_g_rms_norm_grouped")?,
                    g_quantize_q8_1: load_sym_from(h, "tk_qwen4exp_p4_g_quantize_q8_1")?,
                    g_mmvq: load_sym_from(h, "tk_qwen4exp_p4_g_mmvq")?,
                    g_add_streams_broadcast: load_sym_from(h, "tk_qwen4exp_p4_g_add_streams_broadcast")?,
                    g_rms_norm_rope: load_sym_from(h, "tk_qwen4exp_p4_g_rms_norm_rope")?,
                    g_rope: load_sym_from(h, "tk_qwen4exp_p4_g_rope")?,
                    g_rope_neox: load_sym_from(h, "tk_qwen4exp_p4_g_rope_neox")?,
                    g_fwht256: load_sym_from(h, "tk_qwen4exp_p4_g_fwht256")?,
                    g_copy_2d: load_sym_from(h, "tk_qwen4exp_p4_g_copy_2d")?,
                    g_gate_apply: load_sym_from(h, "tk_qwen4exp_p4_g_gate_apply")?,
                    env_mtp: load_sym_from(h, "tk_qwen4exp_p4_env_mtp")?,
                })
            })();
            match r {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!("talus-qwen4exp: qwen4exp-p4: the Rust verify plane is off ({e}); the C++ window bodies run");
                    None
                }
            }
        })
        .as_ref()
}

// ============================================================================
// Pure host helpers (unit-tested)
// ============================================================================

/// `qsa_step_fill` (qsa.cu): the four step entries every QSA kernel reads for position `pos`.
pub fn qsa_step_fill(step: &mut [i32], pos: i64, idx_block: i64, idx_top_k: i64) {
    let n_kv = pos + 1;
    let w = idx_top_k + idx_block - 1;
    step[0] = pos as i32;
    step[1] = n_kv as i32;
    step[2] = (n_kv / idx_block) as i32;
    step[3] = (if n_kv < w { n_kv } else { w }) as i32;
}

/// `strata::kernels::penalty_rows` (sampler.hpp): row t of `out` (T rows of `h` slots) is the last `h` tokens
/// of `tail` followed by `window[0..=t]`, most recent last, -1 in the unused front slots.
pub fn penalty_rows(tail: &[i32], window: &[i32], t_rows: usize, h: usize, out: &mut [i32]) {
    let n_tail = tail.len() as i64;
    for t in 0..t_rows {
        let row = &mut out[t * h..(t + 1) * h];
        let avail = n_tail + t as i64 + 1;
        let take = (h as i64).min(avail) as usize;
        for v in row[..h - take].iter_mut() {
            *v = -1;
        }
        for j in 0..take {
            let i = avail - take as i64 + j as i64;
            row[h - take + j] = if i < n_tail {
                tail[i as usize]
            } else {
                window[(i - n_tail) as usize]
            };
        }
    }
}

/// `coupled_hist_len`: min(penalty_last_n, cap), 0 when penalties are off.
pub fn coupled_hist_len(penalty_last_n: i32, cap: i32) -> i32 {
    if penalty_last_n <= 0 {
        0
    } else if penalty_last_n < cap {
        penalty_last_n
    } else {
        cap
    }
}

/// The verify window's accepted prefix: drafts `window[1..]` kept while they equal the picks.
pub fn accepted_prefix(window: &[i32], out: &[i32], t: usize) -> usize {
    let mut a = 0;
    while a + 1 < t && window[a + 1] == out[a] {
        a += 1;
    }
    a
}

/// The plan block's offsets inside one group's slot: (start, dst, tok, ptr (in i32 words), start2).
pub fn plan_slot_offsets(cap: i64) -> (i64, i64, i64, i64, i64) {
    let i32n = 4 + (cap + 1) + cap + cap;
    let ptr_off = (i32n + 1) & !1;
    let start = 4;
    let dst = start + cap + 1;
    let tok = dst + cap;
    let start2 = ptr_off + 4 * cap;
    (start, dst, tok, ptr_off, start2)
}

/// `nonfinite` rows of a window from the guard's flags: bit t set when any block of row t flagged.
pub fn nonfinite_mask(flags: &[u32], t_rows: usize, blocks: usize, max_t: usize) -> u32 {
    let mut mask = 0u32;
    for t in 0..t_rows.min(max_t) {
        if flags[t * blocks..(t + 1) * blocks].iter().any(|&f| f != 0) {
            mask |= 1 << t;
        }
    }
    mask
}

/// `STRATA_NAN_INJECT=<n>[,<n>...]`: the windows (from 1) the guard reports as non-finite.
pub fn parse_nan_inject(s: &str) -> Vec<i64> {
    let mut v = Vec::new();
    for part in s.split(',') {
        let t = part.trim();
        let digits: String = t
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        match digits.parse::<i64>() {
            Ok(n) if n > 0 => v.push(n),
            Ok(_) => {}
            Err(_) => break,
        }
    }
    v
}

/// C's `%.9f` (Rust prints NaN as "NaN" and infinities as "inf"; the STRATA_LOGPOS format is C's).
fn c_fmt(x: f64, prec: usize) -> String {
    if x.is_nan() {
        if x.is_sign_negative() {
            "-nan".into()
        } else {
            "nan".into()
        }
    } else if x.is_infinite() {
        if x < 0.0 {
            "-inf".into()
        } else {
            "inf".into()
        }
    } else {
        format!("{x:.prec$}")
    }
}

/// One STRATA_LOGPOS line of `Verifier::window_logprobs` for a head row.
pub fn window_logprob_line(
    row: &[f32],
    pos: i64,
    tgt: i32,
    extra_id: i32,
    topk: usize,
) -> Option<String> {
    let n_vocab = row.len() as i64;
    if tgt < 0 || (tgt as i64) >= n_vocab {
        return None;
    }
    let mut top = 0usize;
    for v in 1..row.len() {
        if row[v] > row[top] {
            top = v;
        }
    }
    let maxv = row[top] as f64;
    let has_extra = extra_id >= 0 && (extra_id as i64) < n_vocab;
    let mut sum = 0.0f64;
    let mut sum_without = 0.0f64;
    for (v, &x) in row.iter().enumerate() {
        let e = ((x as f64) - maxv).exp();
        sum += e;
        if v as i64 != extra_id as i64 {
            sum_without += e;
        }
    }
    let lse = maxv + sum.ln();
    let extra = if has_extra {
        row[extra_id as usize] as f64 - lse
    } else {
        f64::NAN
    };
    let without = if has_extra && tgt != extra_id && sum_without > 0.0 {
        row[tgt as usize] as f64 - (maxv + sum_without.ln())
    } else {
        f64::NAN
    };
    let mut line = format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        pos,
        tgt,
        c_fmt(row[tgt as usize] as f64 - lse, 9),
        top,
        c_fmt(maxv - lse, 9),
        (top as i64 == tgt as i64) as i32,
        c_fmt(extra, 9),
        c_fmt(without, 9)
    );
    if topk > 0 {
        let mut order: Vec<usize> = (0..row.len()).collect();
        let k = topk.min(row.len());
        order.select_nth_unstable_by(k.saturating_sub(1), |&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        order[..k].sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &id in &order[..k] {
            line.push_str(&format!("\t{}:{}", id, c_fmt(row[id] as f64 - lse, 6)));
        }
    }
    Some(line)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| !v.is_empty() && !v.starts_with('0'))
        .unwrap_or(false)
}

fn env_atoi(name: &str) -> Option<i64> {
    std::env::var(name).ok().map(|v| {
        let t: String = v
            .trim()
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        t.parse::<i64>().unwrap_or(0)
    })
}

// ============================================================================
// HIP helpers
// ============================================================================

fn hip_err_str(e: hipError_t) -> String {
    unsafe {
        let p = hipGetErrorString(e);
        if p.is_null() {
            format!("hip error {}", e.0)
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

#[inline(always)]
fn cpu_pause() {
    std::hint::spin_loop();
}

#[inline(always)]
fn store_fence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
    #[cfg(not(target_arch = "x86_64"))]
    fence(Ordering::SeqCst);
}

#[inline(always)]
unsafe fn vload_u32(p: *const u32) -> u32 {
    std::ptr::read_volatile(p)
}

#[inline(always)]
unsafe fn vstore_u32(p: *mut u32, v: u32) {
    std::ptr::write_volatile(p, v)
}

/// `Verifier::raise_flag`: flag B only rises (a host function of an earlier layer may run after a later layer
/// already raised it directly).
unsafe fn raise_flag(flag: *mut u32, value: u32) {
    let a = &*(flag as *const AtomicU32);
    let mut cur = a.load(Ordering::SeqCst);
    while cur < value {
        match a.compare_exchange(cur, value, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => break,
            Err(c) => cur = c,
        }
    }
}

fn err_buf() -> Vec<c_char> {
    vec![0 as c_char; 1024]
}

fn err_from(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() }
}

fn released_note(drained: bool) -> &'static str {
    if drained {
        "; its GPU waits were released and the GPU finished (#267)"
    } else {
        "; its GPU waits were released but the GPU did not finish within 5 s (#267)"
    }
}

/// The verify.cpp process-wide switches the Rust plane mirrors.
struct VerifyEnv {
    nan_guard: bool,
    nan_inject: Vec<i64>,
    test_stall: i64,
    dbg_nan: bool,
    logpos_topk: usize,
    commit_timing: bool,
    catchup_all: bool,
    spec_min_temp: f32,
    capture_cpp: bool,
    mtp_prefill_sync: bool,
    commit_batch: bool,
    no_batch_kv: bool,
    draft_ahead: bool,
}

fn verify_env() -> &'static VerifyEnv {
    static E: OnceLock<VerifyEnv> = OnceLock::new();
    E.get_or_init(|| VerifyEnv {
        nan_guard: env_atoi("STRATA_NAN_GUARD").map(|v| v != 0).unwrap_or(true),
        nan_inject: std::env::var("STRATA_NAN_INJECT")
            .map(|s| parse_nan_inject(&s))
            .unwrap_or_default(),
        test_stall: env_atoi("STRATA_TEST_VERIFY_STALL").unwrap_or(0),
        dbg_nan: std::env::var_os("STRATA_DBG_NAN").is_some(),
        logpos_topk: env_atoi("STRATA_LOGPOS_TOPK")
            .map(|v| v.clamp(0, 256) as usize)
            .unwrap_or(0),
        commit_timing: std::env::var("STRATA_COMMIT_TIMING")
            .map(|v| v.starts_with('1'))
            .unwrap_or(false),
        catchup_all: env_flag("STRATA_MTP_CATCHUP_ALL"),
        capture_cpp: env_flag("TALUS_P4_CAPTURE_CPP")
            || std::env::var_os("STRATA_VERIFY_NODES").is_some(),
        mtp_prefill_sync: env_atoi("STRATA_MTP_PREFILL_SYNC")
            .map(|v| v != 0)
            .unwrap_or(true), // HIP default
        commit_batch: std::env::var("STRATA_COMMIT_BATCH")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true),
        no_batch_kv: env_flag("STRATA_NO_BATCH_KV_STEP"),
        // with --spec-min-p, launch draft step j + 1 before the host has read step j's probability (one step
        // queued ahead: no GPU idle gap between steps; a step past the early stop is computed and dropped, like a
        // rejected draft).  The drafts kept are the same ones; greedy output unchanged.
        draft_ahead: env_atoi("TALUS_FN_DRAFT_AHEAD")
            .map(|v| v != 0)
            .unwrap_or(false),
        spec_min_temp: std::env::var("STRATA_SPEC_MIN_TEMP")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.0),
    })
}

/// The windows the non-finite guard has checked (STRATA_NAN_INJECT counts from 1 over the Rust-run windows).
static NF_WINDOWS: AtomicI64 = AtomicI64::new(0);

// ============================================================================
// RustVerifier: Verifier::run / commit / wait_commit / copy_logits / window_logprobs / set_sampling
// ============================================================================

pub struct RustVerifier {
    f: &'static P4Ffi,
    v: TkP4VerView,
    commit_async: bool,
    one_token_self_commit: bool,
    ple_rows: Vec<u32>,
    commit_timing: CommitTiming,
    cap: Option<TkP4CaptureView>,
    layer_is_qsa: Vec<u8>,
}

#[derive(Default)]
struct CommitTiming {
    ev: Option<(hipEvent_t, hipEvent_t)>,
    pending: bool,
    ms: f64,
    n: i64,
}

impl RustVerifier {
    unsafe fn new(f: &'static P4Ffi, hooks: *mut TkP4Hooks, commit_async: bool) -> Option<Self> {
        let mut v: TkP4VerView = std::mem::zeroed();
        if (f.ver_view)(hooks, &mut v) != 0 {
            return None;
        }
        Self::from_view(f, v, commit_async)
    }

    /// The Rust plane over a `strata::core::Verifier*` a C++ caller handed over (phase 2b).
    unsafe fn of_ptr(f: &'static P4Ffi, ver: *mut c_void) -> Option<Self> {
        let mut v: TkP4VerView = std::mem::zeroed();
        if (f.ver_view_of)(ver, &mut v) != 0 {
            return None;
        }
        Self::from_view(f, v, true)
    }

    unsafe fn from_view(f: &'static P4Ffi, v: TkP4VerView, commit_async: bool) -> Option<Self> {
        if v.eligible == 0 {
            return None;
        }
        if v.sizeof_sink as usize != std::mem::size_of::<GpuPlanSink>()
            || v.sizeof_sampler as usize != std::mem::size_of::<SamplerParams>()
        {
            eprintln!(
                "talus-qwen4exp: qwen4exp-p4: struct layout mismatch (GpuPlanSink {} vs {}, SamplerParams {} vs {}): the C++ window bodies run",
                v.sizeof_sink,
                std::mem::size_of::<GpuPlanSink>(),
                v.sizeof_sampler,
                std::mem::size_of::<SamplerParams>()
            );
            return None;
        }
        let flags = (f.env_flags)();
        Some(Self {
            f,
            v,
            commit_async: commit_async && (flags & 4) != 0,
            one_token_self_commit: (flags & 2) != 0,
            ple_rows: vec![0u32; K_VERIFY_MAX_T * PLE_N_HEADS],
            commit_timing: CommitTiming::default(),
            cap: None,
            layer_is_qsa: Vec::new(),
        })
    }

    fn progress(&self, what: &str, detail: i64) {
        // the watchdog keeps the pointer: static strings only
        let c: &'static CStr = match what {
            "reach" => c"verify window: waiting for the GPU to reach layer",
            "experts" => c"verify window: the CPU experts of layer",
            "finish" => {
                c"verify window: waiting for the GPU to finish the window (flags A/B/M raised)"
            }
            "copies" => c"verify window: waiting for the expert copies",
            "decode" => c"decode",
            "mtp" => c"mtp draft: waiting for the GPU",
            _ => c"verify window",
        };
        unsafe { (self.f.progress)(c.as_ptr(), detail, 0) };
    }

    pub fn vocab(&self) -> i64 {
        self.v.n_vocab
    }

    pub fn final_r_all(&self) -> *const f32 {
        self.v.r
    }

    fn row_floats(&self) -> usize {
        (self.v.hc as usize) * (self.v.n_embd as usize)
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn sampling(&self) -> &mut SamplerParams {
        &mut *(self.v.sampling as *mut SamplerParams)
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn sink(&self) -> &mut GpuPlanSink {
        &mut *(self.v.sink as *mut GpuPlanSink)
    }

    /// `Verifier::set_sampling`: the request's head sampling (greedy by default); the logit bias stays the one
    /// `set_logit_bias` uploaded.
    pub unsafe fn set_sampling(&mut self, sp: &SamplerParams) {
        let cur = self.sampling();
        let bias = cur.logit_bias;
        *cur = *sp;
        cur.logit_bias = bias;
    }

    pub fn nonfinite_row(&self) -> i32 {
        unsafe { *self.v.nf_row }
    }

    pub fn nonfinite_pos(&self) -> i64 {
        unsafe { *self.v.nf_pos }
    }

    /// `Verifier::stage_inputs`: tokens, step records, per-head positions; the doorbell words reset.
    unsafe fn stage_inputs(&mut self, t: usize, tokens: &[i32], pos0: i64) {
        let v = &self.v;
        let nh = v.n_head as usize;
        let nkv = v.n_head_kv as usize;
        let iq = v.idx_q_heads as usize;
        let max_t = v.max_t as usize;
        let h_pos = std::slice::from_raw_parts_mut(v.h_pos, max_t * (nh + nkv + iq));
        let (hp, rest) = h_pos.split_at_mut(max_t * nh);
        let (pk, pi) = rest.split_at_mut(max_t * nkv);
        let h_step = std::slice::from_raw_parts_mut(v.h_step, max_t * K_STEP_COUNT);
        for ti in 0..t {
            *v.h_tok.add(ti) = tokens[ti];
            qsa_step_fill(
                &mut h_step[ti * K_STEP_COUNT..(ti + 1) * K_STEP_COUNT],
                pos0 + ti as i64,
                v.idx_block,
                v.idx_top_k,
            );
            let p = (pos0 + ti as i64) as i32;
            hp[ti * nh..(ti + 1) * nh].fill(p);
            pk[ti * nkv..(ti + 1) * nkv].fill(p);
            pi[ti * iq..(ti + 1) * iq].fill(p);
        }
        vstore_u32(v.h_seq, 0);
        vstore_u32(v.h_flag, 0);
        vstore_u32(v.h_flag_a, 0);
        vstore_u32(v.h_flag_b, 0);
        fence(Ordering::SeqCst);
        *v.last_t = t as i32;
        *v.last_pos0 = pos0;
        for ti in 0..t {
            *v.last_tokens.add(ti) = tokens[ti];
        }
        *v.staged = 1;
    }

    /// `Verifier::set_plan_slot`: point the pool's plan sink at group `grp`'s plan block and staging half.
    unsafe fn set_plan_slot(&mut self, grp: usize) {
        let v = &self.v;
        let sink = self.sink();
        let cap = sink.cap;
        let base = v.h_plan.add(grp * v.plan_i32 as usize);
        let (start, dst, tok, ptr_off, start2) = plan_slot_offsets(cap);
        sink.counts = base;
        sink.start = base.add(start as usize);
        sink.dst = base.add(dst as usize);
        sink.tok = base.add(tok as usize);
        sink.ptr = base.add(ptr_off as usize) as *mut u64;
        sink.ptr2 = sink.ptr.add(cap as usize);
        sink.start2 = base.add(start2 as usize);
        let last_t = *v.last_t as usize;
        let g = if *v.last_batch != 0 {
            1
        } else {
            let gg = *v.groups.add(last_t);
            if gg > 0 {
                gg
            } else {
                1
            }
        };
        let per = if g == 2 {
            v.staging_blobs / 2
        } else {
            v.staging_blobs
        };
        sink.staging = v.staging.add((grp as i64 * per * v.max_blob) as usize) as u64;
        sink.staging_cap = per;
    }

    /// `Verifier::adapt_gate_wait`: hold the host until the adaptive tier's copies up to `layer` have landed.
    unsafe fn adapt_gate_wait(&mut self, gate: &TkP4GateView, layer: i32) -> Result<(), String> {
        let t0 = Instant::now();
        let landed = &*(gate.landed as *const AtomicI32);
        let reached = || {
            let a = landed.load(Ordering::Acquire);
            if gate.hland.is_null() {
                a
            } else {
                let b = std::ptr::read_volatile(gate.hland);
                a.max(b)
            }
        };
        let mut spins = 0u32;
        while reached() < layer {
            cpu_pause();
            spins = spins.wrapping_add(1);
            if spins & 1023 != 0 {
                continue;
            }
            if t0.elapsed() > Duration::from_secs(20) {
                let drained = (self.f.ver_release)(self.v.ver, 5000) != 0;
                return Err(format!(
                    "verify: the adaptive tier's copies for layer {layer} never landed{}",
                    released_note(drained)
                ));
            }
        }
        *self.v.ms_gate += ms_since(t0);
        Ok(())
    }

    unsafe fn gate(&self) -> Option<TkP4GateView> {
        if self.v.adapt_gate.is_null() {
            return None;
        }
        let mut g: TkP4GateView = std::mem::zeroed();
        if (self.f.gate_view)(self.v.adapt_gate, &mut g) != 0 {
            return None;
        }
        if (*(g.active as *const AtomicBool)).load(Ordering::Acquire) {
            Some(g)
        } else {
            None
        }
    }

    /// refresh_ar + capture(T) + capture_commit: the window graph `run` launches and whether it is the
    /// zero-doorbell (all-resident) one.  TALUS_P4_CAPTURE_CPP=1 (or STRATA_VERIFY_NODES) keeps the C++ capture.
    unsafe fn prepare(&mut self, t: i32) -> Result<(*mut c_void, bool), String> {
        if verify_env().capture_cpp {
            let mut exec: *mut c_void = std::ptr::null_mut();
            let mut ar_on: c_int = 0;
            let mut eb = err_buf();
            if (self.f.ver_prepare)(
                self.v.ver,
                t,
                &mut exec,
                &mut ar_on,
                eb.as_mut_ptr(),
                eb.len(),
            ) != 0
            {
                return Err(err_from(&eb));
            }
            return Ok((exec, ar_on != 0));
        }
        if self.cap.is_none() {
            let mut cv: TkP4CaptureView = std::mem::zeroed();
            if (self.f.cap_view)(self.v.ver, &mut cv) != 0 {
                return Err("verify: the capture view is not available".into());
            }
            self.layer_is_qsa =
                std::slice::from_raw_parts(cv.layer_is_qsa, self.v.n_layers as usize).to_vec();
            self.cap = Some(cv);
        }
        let cv = self.cap.unwrap();
        self.refresh_ar(&cv);
        self.capture(t, &cv)?;
        self.capture_commit(&cv)?;
        let ar_off = *cv.ar_off != 0;
        let tu = t as usize;
        let exec = if ar_off {
            *cv.exec_nr.add(tu)
        } else {
            *cv.exec.add(tu)
        };
        Ok((exec, *cv.all_resident != 0 && !ar_off))
    }

    /// `Verifier::refresh_ar` (#871): the zero-doorbell graph only while every expert of the stage is in VRAM.
    unsafe fn refresh_ar(&mut self, cv: &TkP4CaptureView) {
        if *cv.all_resident == 0 || cv.h_res.is_null() {
            return;
        }
        let ne = cv.n_expert as usize;
        let lb = self.v.lb as usize;
        let le = self.v.le as usize;
        let r = std::slice::from_raw_parts(cv.h_res.add(lb * ne), (le - lb) * ne);
        let off = r.iter().any(|&x| x < 0);
        if off != (*cv.ar_off != 0) {
            *cv.ar_off = off as u8;
            eprintln!(
                "talus verify: layers [{lb}, {le}) {}",
                if off {
                    "have experts out of VRAM (a prompt loan, a shrink or a swap): the doorbell graph runs the windows"
                } else {
                    "are 100% VRAM resident again: the zero-doorbell graph"
                }
            );
        }
    }

    /// `Verifier::capture`: the T-token window graph (exec_ or, #871, exec_nr_), captured on first use.  The window's
    /// kernels are recorded by `tk_qwen4exp_p4_record_window` (Verifier::record_window; Piece 3's Rust recorder
    /// plugs in there).
    unsafe fn capture(&mut self, t: i32, cv: &TkP4CaptureView) -> Result<(), String> {
        let tu = t as usize;
        let slot = if *cv.ar_off != 0 {
            cv.exec_nr.add(tu)
        } else {
            cv.exec.add(tu)
        };
        if !(*slot).is_null() {
            return Ok(());
        }
        let mut free_b = 0usize;
        let mut total_b = 0usize;
        if hip_sys::hipMemGetInfo(&mut free_b, &mut total_b) == hipSuccess {
            eprintln!(
                "talus verify: capturing the {t}-token window ({} MiB of VRAM free)",
                free_b >> 20
            );
        } else {
            let _ = hipGetLastError();
        }
        let cs = self.v.stream as hipStream_t;
        if hip_sys::hipStreamBeginCapture(cs, hip_sys::hipStreamCaptureModeThreadLocal)
            != hipSuccess
        {
            return Err("verify: begin capture failed".into());
        }
        let mut eb = err_buf();
        // Piece 3's recorder (decode.rs, through its hook), else Verifier::record_window
        let rec = (self.f.record_window)(self.v.ver, t, eb.as_mut_ptr(), eb.len());
        let mut graph: hip_sys::hipGraph_t = std::ptr::null_mut();
        let ce = hip_sys::hipStreamEndCapture(cs, &mut graph);
        if rec != 0 {
            if !graph.is_null() {
                hip_sys::hipGraphDestroy(graph);
            }
            return Err(err_from(&eb));
        }
        if ce != hipSuccess {
            return Err(format!("verify: end capture: {}", hip_err_str(ce)));
        }
        let mut eb2 = err_buf();
        if (self.f.window_instantiate)(
            self.v.ver,
            t,
            graph as *mut c_void,
            eb2.as_mut_ptr(),
            eb2.len(),
        ) != 0
        {
            return Err(err_from(&eb2));
        }
        Ok(())
    }

    fn need_weight(&self, l: usize, suffix: &CStr) -> Result<*const f32, String> {
        let mut eb = err_buf();
        let p = unsafe {
            (self.f.layer_weight)(
                self.v.ver,
                l as i64,
                suffix.as_ptr(),
                eb.as_mut_ptr(),
                eb.len(),
            )
        };
        if p.is_null() {
            Err(err_from(&eb))
        } else {
            Ok(p)
        }
    }

    /// `Verifier::capture_commit`: the graph that keeps the first `n_keep` rows of a window - every GDN layer's
    /// conv history and recurrence replayed from the rows the window stored, every QSA layer's indexer tail
    /// restored from its snapshot and the kept keys appended, the PLE history set to its snapshot after row
    /// n_keep - 1.  `n_keep` and the positions come from the mapped commit record at replay.
    unsafe fn capture_commit(&mut self, cv: &TkP4CaptureView) -> Result<(), String> {
        if !(*self.v.commit_exec).is_null() {
            return Ok(());
        }
        let cs = self.v.stream as hipStream_t;
        if hip_sys::hipStreamBeginCapture(cs, hip_sys::hipStreamCaptureModeThreadLocal)
            != hipSuccess
        {
            return Err("verify: begin commit capture failed".into());
        }
        let r = self.record_commit(cv);
        let mut graph: hip_sys::hipGraph_t = std::ptr::null_mut();
        let ce = hip_sys::hipStreamEndCapture(cs, &mut graph);
        if let Err(e) = r {
            if !graph.is_null() {
                hip_sys::hipGraphDestroy(graph);
            }
            return Err(e);
        }
        let mut exec: hipGraphExec_t = std::ptr::null_mut();
        if ce != hipSuccess
            || hip_sys::hipGraphInstantiateWithFlags(&mut exec, graph, 0) != hipSuccess
        {
            if !graph.is_null() {
                hip_sys::hipGraphDestroy(graph);
            }
            return Err(format!("verify: commit capture: {}", hip_err_str(ce)));
        }
        hip_sys::hipGraphDestroy(graph);
        *self.v.commit_exec = exec as *mut c_void;
        Ok(())
    }

    unsafe fn record_commit(&mut self, cv: &TkP4CaptureView) -> Result<(), String> {
        let f = self.f;
        let st = self.v.stream;
        let fail = |what: &str| -> String { format!("verify commit: {what} failed") };
        let mt = self.v.max_t as i64;
        let c = cv.ssm_conv_channels;
        let hv = cv.ssm_v_heads;
        let id = cv.idx_key_dim;
        let ss = cv.ssm_state_size;
        let state_floats = ss * hv * ss;
        let gdn_floats = state_floats + c * (cv.ssm_d_conv - 1);
        let ts = (self.v.idx_block - 1) * id;
        let commit = cv.commit as *const i32;
        if (f.k_copy_i32_from_mapped)(cv.commit, cv.m_commit, 2 + mt, st) != 0 {
            return Err(fail("the commit record copy"));
        }
        let (lb, le) = (self.v.lb as usize, self.v.le as usize);
        let is_qsa = |l: usize| self.layer_is_qsa[l] != 0;
        let mut qsa_index = 0i64;
        let mut gdn_index = 0i64;
        for l in 0..lb {
            if is_qsa(l) {
                qsa_index += 1
            } else {
                gdn_index += 1
            }
        }
        let env = verify_env();
        // STRATA_COMMIT_BATCH (default on): every GDN layer in two launches, every QSA layer's append in one
        let mut gdn_batched = false;
        if env.commit_batch {
            let mut n_gdn = 0;
            for l in lb..le {
                if !is_qsa(l) {
                    n_gdn += 1;
                    self.need_weight(l, c"ssm_norm.weight")?;
                }
            }
            if n_gdn > 0 {
                let rc = (f.k_gdn_commit_layers)(
                    cv.gdn_state
                        .add(((gdn_index - cv.gdn_ord0) * gdn_floats) as usize),
                    gdn_floats,
                    state_floats,
                    cv.qkv_l.add((gdn_index * mt * c) as usize),
                    mt * c,
                    cv.h_l.add((gdn_index * mt * c) as usize),
                    mt * c,
                    cv.gate_l.add((gdn_index * mt * hv) as usize),
                    cv.beta_l.add((gdn_index * mt * hv) as usize),
                    mt * hv,
                    c as c_int,
                    cv.ssm_k_heads as c_int,
                    hv as c_int,
                    mt as c_int,
                    n_gdn,
                    commit,
                    st,
                );
                if rc < 0 {
                    return Err(fail("the GDN commit"));
                }
                gdn_batched = rc > 0;
            }
        }
        let mut qsa_batched = false;
        if env.commit_batch && !env.no_batch_kv {
            let mut gamma = [std::ptr::null::<f32>(); K_INDEXER_LAYERS_MAX];
            let mut tail_src = [std::ptr::null::<f32>(); K_INDEXER_LAYERS_MAX];
            let mut tail = [std::ptr::null_mut::<f32>(); K_INDEXER_LAYERS_MAX];
            let mut dead = [std::ptr::null_mut::<f32>(); K_INDEXER_LAYERS_MAX];
            let mut pooled = [std::ptr::null_mut::<f32>(); K_INDEXER_LAYERS_MAX];
            let mut block_pos = [std::ptr::null_mut::<i32>(); K_INDEXER_LAYERS_MAX];
            let mut nq = 0usize;
            let mut mc = -1i64;
            let mut fits = true;
            let mut qi = qsa_index;
            for l in lb..le {
                if !is_qsa(l) {
                    continue;
                }
                let g = self.need_weight(l, c"indexer.k_norm.weight")?;
                let mut q: TkP4QsaState = std::mem::zeroed();
                (f.qsa_state)(self.v.ver, qi, &mut q);
                if nq >= K_INDEXER_LAYERS_MAX || (mc >= 0 && q.max_cells != mc) {
                    fits = false;
                    break;
                }
                mc = q.max_cells;
                gamma[nq] = g;
                tail_src[nq] = cv.tail_snap.add((qi * ts) as usize);
                tail[nq] = q.idx_tail;
                dead[nq] = q.idx_dead;
                pooled[nq] = q.idx_pooled;
                block_pos[nq] = q.idx_block_pos;
                nq += 1;
                qi += 1;
            }
            if fits && nq > 0 {
                let mut eb = err_buf();
                if (f.k_idx_steps_layers)(
                    self.v.ver,
                    cv.idx_raw_l.add((qsa_index * mt * id) as usize),
                    mt * id,
                    commit.add(2),
                    1,
                    mt as c_int,
                    0,
                    VERIFY_EPS,
                    gamma.as_ptr(),
                    tail_src.as_ptr(),
                    tail.as_ptr(),
                    dead.as_ptr(),
                    pooled.as_ptr(),
                    block_pos.as_ptr(),
                    nq as c_int,
                    mc,
                    st,
                    eb.as_mut_ptr(),
                    eb.len(),
                ) != 0
                {
                    return Err(err_from(&eb));
                }
                qsa_batched = true;
            }
        }
        for l in lb..le {
            if is_qsa(l) && qsa_batched {
                qsa_index += 1;
            } else if !is_qsa(l) && gdn_batched {
                gdn_index += 1;
            } else if !is_qsa(l) {
                let wnm = self.need_weight(l, c"ssm_norm.weight")?;
                let state = cv
                    .gdn_state
                    .add(((gdn_index - cv.gdn_ord0) * gdn_floats) as usize);
                let conv = state.add(state_floats as usize);
                let qkv = cv.qkv_l.add((gdn_index * mt * c) as usize);
                if (f.k_gdn_conv_commit)(conv, qkv, c as c_int, commit, st) != 0 {
                    return Err(fail("gdn_conv_commit"));
                }
                if (f.k_gdn_step_norm_multi)(
                    state,
                    cv.h_l.add((gdn_index * mt * c) as usize),
                    c as c_int,
                    cv.gate_l.add((gdn_index * mt * hv) as usize),
                    cv.beta_l.add((gdn_index * mt * hv) as usize),
                    cv.z,
                    wnm,
                    VERIFY_EPS,
                    cv.y_dummy,
                    cv.ssm_k_heads as c_int,
                    hv as c_int,
                    mt as c_int,
                    commit,
                    st,
                    mt as c_int, // S26: t_out_begin = MT - the replay's outputs are never read
                ) != 0
                {
                    return Err(fail("gdn_step_norm_multi"));
                }
                gdn_index += 1;
            } else {
                let mut q: TkP4QsaState = std::mem::zeroed();
                (f.qsa_state)(self.v.ver, qsa_index, &mut q);
                let wikn = self.need_weight(l, c"indexer.k_norm.weight")?;
                if (f.k_copy_from_mapped)(
                    q.idx_tail,
                    cv.tail_snap.add((qsa_index * ts) as usize),
                    ts,
                    st,
                ) != 0
                {
                    return Err(fail("the indexer tail restore"));
                }
                let mut eb = err_buf();
                if !env.no_batch_kv {
                    if (f.k_idx_steps)(
                        self.v.ver,
                        cv.idx_raw_l.add((qsa_index * mt * id) as usize),
                        commit.add(2),
                        1,
                        mt as c_int,
                        0,
                        wikn,
                        VERIFY_EPS,
                        &q,
                        st,
                        eb.as_mut_ptr(),
                        eb.len(),
                    ) != 0
                    {
                        return Err(err_from(&eb));
                    }
                } else {
                    for ti in 0..mt {
                        if (f.k_idx)(
                            self.v.ver,
                            cv.idx_raw_l.add(((qsa_index * mt + ti) * id) as usize),
                            commit.add(2 + ti as usize),
                            0,
                            wikn,
                            VERIFY_EPS,
                            &q,
                            st,
                            eb.as_mut_ptr(),
                            eb.len(),
                        ) != 0
                        {
                            return Err(err_from(&eb));
                        }
                    }
                }
                qsa_index += 1;
            }
        }
        if cv.ple_copy != 0
            && (f.k_copy_indexed)(
                cv.ple_hist,
                cv.hist_snap,
                cv.ng_hist_hc,
                commit.add(1),
                cv.ng_hist_hc,
                st,
            ) != 0
        {
            return Err(fail("the PLE history restore"));
        }
        Ok(())
    }

    /// `Verifier::note_nonfinite`: the window's guard flags (+ STRATA_NAN_INJECT) -> nf_row / nf_pos / nf_mask.
    unsafe fn note_nonfinite(&mut self, t: usize) {
        let v = &self.v;
        *v.nf_row = -1;
        *v.nf_pos = -1;
        *v.nf_mask = 0;
        let env = verify_env();
        if !env.nan_guard || v.h_nf.is_null() {
            return;
        }
        let nb = v.nonfinite_blocks as usize;
        let max_t = v.verify_max_t as usize;
        let mut flags = vec![0u32; max_t * nb];
        for (i, f) in flags.iter_mut().enumerate() {
            *f = vload_u32(v.h_nf.add(i));
        }
        let mut mask = nonfinite_mask(&flags, t, nb, max_t);
        if !env.nan_inject.is_empty() {
            let k = NF_WINDOWS.fetch_add(1, Ordering::SeqCst) + 1;
            if mask == 0 && env.nan_inject.contains(&k) {
                mask = 1;
                eprintln!("talus verify: STRATA_NAN_INJECT: window {k} reported as non-finite");
            }
        }
        *v.nf_mask = mask;
        for ti in 0..t {
            if (mask >> ti) & 1 != 0 {
                *v.nf_row = ti as i32;
                *v.nf_pos = *v.last_pos0 + ti as i64;
                break;
            }
        }
    }

    /// `Verifier::run`: one window of `t` tokens at `pos0`; `out[i]` = the pick after token i.  `spec_q`: the
    /// drafter's q rows for the window's first drafts (probabilistic acceptance; `set_spec_q` for this run).
    pub unsafe fn run(
        &mut self,
        t: i32,
        tokens: &[i32],
        pos0: i64,
        pool: Option<PoolMultiFn>,
        user: *mut c_void,
        out: &mut [i32],
        spec_q: Option<(*const i32, i32)>,
    ) -> Result<(), String> {
        let v = self.v;
        *v.last_batch = 0;
        if t < 1 || t > v.max_t {
            return Err("verify: window size out of range".into());
        }
        if (*(v.released as *const AtomicBool)).load(Ordering::SeqCst) {
            return Err(
                "verify: an earlier window never finished on the GPU (#267); restart the engine"
                    .into(),
            );
        }
        if pos0 + t as i64 > *v.max_cells {
            return Err("verify: the window runs past the context".into());
        }
        let tu = t as usize;
        let (exec, ar_on) = self.prepare(t)?;
        let t0 = Instant::now();
        if *v.staged == 0 {
            self.stage_inputs(tu, tokens, pos0);
        }
        *v.staged = 0;
        let do_ple = v.ple_on != 0;
        if do_ple {
            let mut prev = [*v.ple_prev, *v.ple_prev.add(1)];
            if (self.f.ple_rows)(
                v.ple_ss,
                tokens.as_ptr(),
                t,
                prev.as_mut_ptr(),
                self.ple_rows.as_mut_ptr(),
            ) != 0
            {
                return Err("verify: the PLE rows could not be hashed".into());
            }
        }
        fence(Ordering::SeqCst);
        *v.ms_host += ms_since(t0);
        if ar_on && !v.h_plan_err.is_null() {
            vstore_u32(v.h_plan_err, 0);
        }
        let gate = self.gate();
        if let Some(ref g) = gate {
            if ar_on {
                self.adapt_gate_wait(g, v.n_layers)?;
            }
        }
        let cs = v.stream as hipStream_t;
        let le = hipGraphLaunch(exec as hipGraphExec_t, cs);
        if le != hipSuccess {
            return Err(format!("verify: launch: {}", hip_err_str(le)));
        }
        let _ = hipStreamQuery(cs);
        let gg = *v.groups.add(tu);
        let g = if gg > 0 { gg as usize } else { 1 };
        let half = (tu + 1) / 2;
        let gtb = [0usize, half];
        let gte = [if g == 2 { half } else { tu }, tu];
        let steps = (v.le - v.lb) as i64 * g as i64;
        let env = verify_env();
        let test_stall = env.test_stall > 0 && *v.windows + 1 == env.test_stall;
        let n_embd = v.n_embd as usize;
        let k = v.k as usize;
        if ar_on && !test_stall {
            if do_ple {
                let tp = Instant::now();
                self.gather_ple(tu)?;
                fence(Ordering::SeqCst);
                store_fence();
                vstore_u32(v.h_flag, 1);
                *v.ms_host += ms_since(tp);
            } else {
                vstore_u32(v.h_flag, 1);
            }
        } else {
            for kk in 0..steps {
                let l = v.lb as i64 + kk / g as i64;
                let grp = (kk % g as i64) as usize;
                let want = (kk + 1) as u32;
                let a = Instant::now();
                let mut last_flush = a;
                let mut spins = 0u32;
                self.progress("reach", l);
                while vload_u32(v.h_seq) < want {
                    cpu_pause();
                    spins = spins.wrapping_add(1);
                    if spins & 1023 != 0 {
                        continue;
                    }
                    let now = Instant::now();
                    if now.duration_since(last_flush) > Duration::from_micros(2000) {
                        last_flush = now;
                        let q = hipStreamQuery(cs);
                        if q != hipErrorNotReady && vload_u32(v.h_seq) < want {
                            return Err(format!(
                                "verify: layer {l} never rang ({})",
                                if q == hipSuccess {
                                    "graph finished".to_string()
                                } else {
                                    hip_err_str(q)
                                }
                            ));
                        }
                    }
                    if now.duration_since(a) > Duration::from_secs(20) {
                        let drained = (self.f.ver_release)(v.ver, 5000) != 0;
                        return Err(format!(
                            "verify: timed out at layer {l}{}",
                            released_note(drained)
                        ));
                    }
                }
                let b = Instant::now();
                if let Some(ref gv) = gate {
                    if l >= 0 && (l as usize) < gv.n_has && *gv.has.add(l as usize) != 0 {
                        self.adapt_gate_wait(gv, l as i32)?;
                    }
                }
                *v.cur_layer = want - 1;
                self.set_plan_slot(grp);
                let tb = gtb[grp];
                let n = gte[grp] - gtb[grp];
                self.progress("experts", l);
                if let Some(pf) = pool {
                    pf(
                        user,
                        v.h_x.add(tb * n_embd),
                        v.h_ids.add(tb * k),
                        n as i64,
                        k as i64,
                        v.h_ymiss.add(tb * k * n_embd),
                        l,
                    );
                }
                (self.f.progress)(std::ptr::null(), 0, 1);
                fence(Ordering::SeqCst);
                store_fence();
                if vload_u32(v.h_flag_a) != want {
                    // the pool did not publish a plan: an empty one
                    let sink = self.sink();
                    *sink.counts = 0;
                    *sink.counts.add(1) = 0;
                    *sink.counts.add(2) = 0;
                    *sink.start = 0;
                    *sink.start2 = 0;
                    fence(Ordering::SeqCst);
                    vstore_u32(v.h_flag_a, want);
                    raise_flag(v.h_flag_b, want);
                }
                // the PLE rows are gathered while layer 0 is served (after its plan is out, before flag 1)
                if kk == 0 && do_ple {
                    let tp = Instant::now();
                    self.gather_ple(tu)?;
                    fence(Ordering::SeqCst);
                    store_fence();
                    *v.ms_host += ms_since(tp);
                }
                if !(test_stall && kk + 1 == steps) {
                    vstore_u32(v.h_flag, want);
                }
                *v.ms_wait += b.duration_since(a).as_secs_f64() * 1000.0;
                *v.ms_pool += ms_since(b);
            }
        }
        self.progress("finish", t as i64);
        let se = hipStreamSynchronize(cs);
        if se != hipSuccess {
            return Err(format!("verify: {}", hip_err_str(se)));
        }
        if ar_on && !v.h_plan_err.is_null() && vload_u32(v.h_plan_err) != 0 {
            vstore_u32(v.h_plan_err, 0);
            return Err("verify: the all-resident plan met an expert that is not in VRAM (the residency table changed during the window)".into());
        }
        *v.commit_pending = 0;
        self.progress("copies", t as i64);
        if *v.copy_used != 0 {
            let _ = hipStreamSynchronize(v.copy_stream as hipStream_t);
            *v.copy_used = 0;
        }
        // the head's sampling again, host-side so its parameters are this call's own; row t draws
        // Philox(seed, pos0 + t)
        let sp_now = *self.sampling();
        let sampled = !sp_now.greedy && sp_now.temperature > 0.0;
        let hist_d = *v.hist_d;
        let hist_len = *v.hist_len;
        if *v.head_sampling != 0 && (sampled || !hist_d.is_null() || !sp_now.logit_bias.is_null()) {
            let mut sp = sp_now;
            sp.counter = pos0 as u64;
            let mut judged = false;
            if let Some((q, nq)) = spec_q {
                if sampled && !q.is_null() && nq > 0 && t > 1 && tu <= K_VERIFY_MAX_T {
                    if (*v.d_spec_slot).is_null() {
                        let mut p: *mut c_void = std::ptr::null_mut();
                        if hipMalloc(&mut p, K_VERIFY_MAX_T * (1 + K_SPEC_Q_STRIDE) * 4)
                            != hipSuccess
                        {
                            let _ = hipGetLastError();
                        } else {
                            *v.d_spec_slot = p as *mut i32;
                        }
                    }
                    let d_spec = *v.d_spec_slot;
                    if !d_spec.is_null() {
                        let nq = nq.min(t - 1);
                        let d_q = d_spec.add(K_VERIFY_MAX_T);
                        if hipMemcpyAsync(
                            d_spec as *mut c_void,
                            tokens.as_ptr().add(1) as *const c_void,
                            (tu - 1) * 4,
                            hipMemcpyHostToDevice,
                            cs,
                        ) == hipSuccess
                            && hipMemcpyAsync(
                                d_q as *mut c_void,
                                q as *const c_void,
                                nq as usize * K_SPEC_Q_STRIDE * 4,
                                hipMemcpyHostToDevice,
                                cs,
                            ) == hipSuccess
                        {
                            judged = (self.f.sample_tokens_spec)(
                                v.head_logits,
                                t,
                                v.n_vocab as c_int,
                                hist_d,
                                hist_len,
                                &sp as *const SamplerParams as *const c_void,
                                d_spec,
                                d_q,
                                nq,
                                v.m_out,
                                v.stream,
                            ) != 0;
                        }
                    }
                }
            }
            if judged {
                *v.spec_windows += 1;
            } else {
                (self.f.sample_tokens)(
                    v.head_logits,
                    t,
                    v.n_vocab as c_int,
                    hist_d,
                    hist_len,
                    &sp as *const SamplerParams as *const c_void,
                    v.m_out,
                    v.stream,
                );
            }
            if hipStreamSynchronize(cs) != hipSuccess {
                return Err("verify: the head sampling failed".into());
            }
        }
        for ti in 0..tu {
            out[ti] = std::ptr::read_volatile(v.h_out.add(ti));
        }
        self.note_nonfinite(tu);
        if env.dbg_nan {
            self.debug_nan(tu, pos0, out);
        }
        *v.windows += 1;
        self.progress("decode", -1);
        (self.f.progress)(std::ptr::null(), 0, 2);
        Ok(())
    }

    unsafe fn gather_ple(&mut self, t: usize) -> Result<(), String> {
        let mut eb = err_buf();
        if (self.f.ple_gather)(
            self.v.ple_ss,
            self.ple_rows.as_ptr(),
            t,
            self.v.h_ple,
            eb.as_mut_ptr(),
            eb.len(),
        ) != 0
        {
            return Err(err_from(&eb));
        }
        Ok(())
    }

    /// STRATA_DBG_NAN (debug): the first non-finite head row of the process, once.
    unsafe fn debug_nan(&self, t: usize, pos0: i64, out: &[i32]) {
        static REPORTED: AtomicBool = AtomicBool::new(false);
        if REPORTED.load(Ordering::Relaxed) {
            return;
        }
        let nv = self.v.n_vocab as usize;
        let mut h = vec![0f32; t * nv];
        if hipMemcpy(
            h.as_mut_ptr() as *mut c_void,
            self.v.head_logits as *const c_void,
            h.len() * 4,
            hipMemcpyDeviceToHost,
        ) != hipSuccess
        {
            return;
        }
        for ti in 0..t {
            let bad = h[ti * nv..(ti + 1) * nv]
                .iter()
                .filter(|x| !x.is_finite())
                .count();
            if bad > 0 {
                REPORTED.store(true, Ordering::Relaxed);
                eprintln!(
                    "talus dbg: verify window at position {}, row {}: {} of {} logits non-finite (token out {})",
                    pos0, ti, bad, nv, out[ti]
                );
                break;
            }
        }
    }

    /// `Verifier::commit`: keep the first `n_keep` tokens of the last window (the commit graph replays them into
    /// the GDN / indexer state); advances the session's PLE history.
    pub unsafe fn commit(&mut self, n_keep: i32) -> Result<(), String> {
        self.commit_as(n_keep, self.commit_async)
    }

    /// `commit` with the caller's `Verifier::set_commit_async` state (phase 2b: the C++ entry point passes it).
    pub unsafe fn commit_as(&mut self, n_keep: i32, commit_async: bool) -> Result<(), String> {
        let v = self.v;
        let last_t = *v.last_t;
        if n_keep < 1 || n_keep > last_t {
            return Err("verify: commit count out of range".into());
        }
        let t0 = Instant::now();
        *v.h_commit = n_keep;
        *v.h_commit.add(1) = n_keep - 1;
        for ti in 0..v.max_t {
            *v.h_commit.add(2 + ti as usize) = if ti < n_keep {
                (*v.last_pos0 + ti as i64) as i32
            } else {
                -1
            };
        }
        if last_t == 1 && *v.always_publish == 0 && self.one_token_self_commit {
            // a one-token window has advanced the state itself: no commit graph
        } else {
            fence(Ordering::SeqCst);
            let cs = v.stream as hipStream_t;
            let ct_on = verify_env().commit_timing;
            if ct_on {
                self.commit_timing_mark(cs);
            }
            let le = hipGraphLaunch(*v.commit_exec as hipGraphExec_t, cs);
            if ct_on {
                if let Some((_, e)) = self.commit_timing.ev {
                    hipEventRecord(e, cs);
                    self.commit_timing.pending = true;
                }
            }
            if le != hipSuccess {
                return Err(format!("verify: commit launch: {}", hip_err_str(le)));
            }
            if !commit_async {
                let se = hipStreamSynchronize(cs);
                if se != hipSuccess {
                    return Err(format!("verify: commit: {}", hip_err_str(se)));
                }
            } else {
                let re = hipEventRecord(v.commit_done as hipEvent_t, cs);
                if re != hipSuccess {
                    return Err(format!("verify: commit event: {}", hip_err_str(re)));
                }
                let _ = hipStreamQuery(cs);
                *v.commit_pending = 1;
            }
        }
        // the stage holds layer 1 (the whole model): the PLE history advances by the kept tokens
        if v.lb <= 1 && 1 < v.le {
            for ti in 0..n_keep as usize {
                *v.ple_prev = *v.ple_prev.add(1);
                *v.ple_prev.add(1) = *v.last_tokens.add(ti);
            }
        }
        *v.ms_commit += ms_since(t0);
        Ok(())
    }

    /// STRATA_COMMIT_TIMING=1 (measurement only): the commit graph's GPU time, read at the next commit.
    unsafe fn commit_timing_mark(&mut self, cs: hipStream_t) {
        let ct = &mut self.commit_timing;
        if ct.ev.is_none() {
            let mut b: hipEvent_t = std::ptr::null_mut();
            let mut e: hipEvent_t = std::ptr::null_mut();
            if hip_sys::hipEventCreate(&mut b) != hipSuccess
                || hip_sys::hipEventCreate(&mut e) != hipSuccess
            {
                let _ = hipGetLastError();
                return;
            }
            ct.ev = Some((b, e));
        }
        let (b, e) = ct.ev.unwrap();
        if ct.pending {
            let mut ms = 0f32;
            if hipEventSynchronize(e) == hipSuccess
                && hip_sys::hipEventElapsedTime(&mut ms, b, e) == hipSuccess
            {
                ct.ms += ms as f64;
                ct.n += 1;
            }
            ct.pending = false;
            if ct.n > 0 && ct.n % 300 == 0 {
                eprintln!(
                    "talus verify: commit graph {:.3} ms GPU on average over {} commits",
                    ct.ms / ct.n as f64,
                    ct.n
                );
            }
        }
        hipEventRecord(b, cs);
    }

    /// `Verifier::wait_commit`: the last asynchronous commit graph has completed.
    pub unsafe fn wait_commit(&mut self) -> Result<(), String> {
        let v = self.v;
        if *v.commit_pending != 0 {
            *v.commit_pending = 0;
            let se = hipEventSynchronize(v.commit_done as hipEvent_t);
            if se != hipSuccess {
                return Err(format!("verify: commit: {}", hip_err_str(se)));
            }
        }
        Ok(())
    }

    /// `Verifier::copy_logits`: row `t` of the last window's head logits to the host.
    pub unsafe fn copy_logits(&self, t: i32, host: &mut [f32]) -> bool {
        let nv = self.v.n_vocab;
        if self.v.head_logits.is_null() || t < 0 || nv <= 0 || host.len() < nv as usize {
            return false;
        }
        hipMemcpy(
            host.as_mut_ptr() as *mut c_void,
            self.v.head_logits.add(t as usize * nv as usize) as *const c_void,
            nv as usize * 4,
            hipMemcpyDeviceToHost,
        ) == hipSuccess
    }

    /// `Verifier::window_logprobs` (STRATA_LOGPOS): one line per row of the last window's head.
    pub unsafe fn window_logprobs(
        &self,
        targets: &[i32],
        t: usize,
        pos0: i64,
        extra_id: i32,
        out: &mut dyn Write,
    ) -> Result<(), String> {
        let nv = self.v.n_vocab as usize;
        if self.v.head_logits.is_null() || nv == 0 || t == 0 || targets.len() < t {
            return Err("window_logprobs: no head logits for this window".into());
        }
        let mut h = vec![0f32; t * nv];
        if hipMemcpy(
            h.as_mut_ptr() as *mut c_void,
            self.v.head_logits as *const c_void,
            h.len() * 4,
            hipMemcpyDeviceToHost,
        ) != hipSuccess
        {
            return Err("window_logprobs: the head logits copy failed".into());
        }
        let topk = verify_env().logpos_topk;
        for ti in 0..t {
            if let Some(line) = window_logprob_line(
                &h[ti * nv..(ti + 1) * nv],
                pos0 + ti as i64,
                targets[ti],
                extra_id,
                topk,
            ) {
                let _ = writeln!(out, "{line}");
            }
        }
        let _ = out.flush();
        Ok(())
    }
}

// ============================================================================
// RustMtp: MtpDrafter::draft / draft_first / set_draft_sampling / set_draft_history
// ============================================================================

pub struct RustMtp {
    f: &'static P4Ffi,
    v: TkP4MtpView,
    rows16: [u32; PLE_N_HEADS],
    pf: Option<TkP4MtpPfView>,
    rec: Option<MtpRec>,
    /// when > 0, the next `draft` makes at most this many drafts (the catch-up round still runs in full).
    /// Set by `Piece4VerifyMtp::serve_draft` from its `draft_step_cap`.
    pub step_cap: i32,
}

/// The drafter's buffers, weights and kernel choices, for recording its layer (phase 3).
#[derive(Clone)]
struct MtpRec {
    r: TkP4MtpRecView,
    env: i32,
    w: std::collections::HashMap<&'static str, (*const c_void, i32)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4MtpRecView {
    pub tok: *mut i32,
    pub step: *mut i32,
    pub pos: *mut i32,
    pub row: *mut i32,
    pub ident: *mut i32,
    pub ids: *mut i32,
    pub hit_slot: *mut i32,
    pub hit_dst: *mut i32,
    pub hit_count: *mut i32,
    pub out_ids: *mut i32,
    pub grp_start: *mut i32,
    pub grp_counts: *mut i32,
    pub grp_ptr: *mut u64,
    pub rin: *mut f32,
    pub r: *mut f32,
    pub emb: *mut f32,
    pub en: *mut f32,
    pub e2: *mut f32,
    pub hn: *mut f32,
    pub h2: *mut f32,
    pub mixed: *mut f32,
    pub inj: *mut f32,
    pub inj2: *mut f32,
    pub lo: *mut f32,
    pub rs: *mut f32,
    pub bo: *mut f32,
    pub xn: *mut f32,
    pub qfull: *mut f32,
    pub qcur: *mut f32,
    pub kcur: *mut f32,
    pub vcur: *mut f32,
    pub attn: *mut f32,
    pub attn32: *mut f32,
    pub logits: *mut f32,
    pub w: *mut f32,
    pub shared: *mut f32,
    pub parts: *mut f32,
    pub y: *mut f32,
    pub sample: *mut f32,
    pub head_logits: *mut f32,
    pub dummy_inj: *mut f32,
    pub hit_xs: *mut f32,
    pub probs: *mut f32,
    pub xq: *mut u8,
    pub hit_xq: *mut u8,
    pub experts: *mut u8,
    pub hit_scratch: *mut c_void,
    pub arg_scratch: *mut c_void,
    pub top_scratch: *mut c_void,
    pub x_bf16: *mut u16,
    pub m_tok: *const i32,
    pub m_step: *const i32,
    pub m_pos: *const i32,
    pub m_row: *const i32,
    pub m_force: *const i32,
    pub m_out: *mut i32,
    pub m_prob: *mut f32,
    pub n: i32,
    pub hc: i32,
    pub nh: i32,
    pub hd: i32,
    pub nkv: i32,
    pub k: i32,
    pub hc_lr: i32,
    pub n_rot: i32,
    pub n_expert: i32,
    pub max_t: i32,
    pub window: i64,
    pub n_vocab: i64,
    pub n_dvocab: i64,
    pub hnorm_stream: i32,
    pub force_on: i32,
    pub kv_rot: i32,
    pub coupled_ok: i32,
    pub spec_q: i32,
    pub cos_tab: *mut f32,
    pub sin_tab: *mut f32,
    pub stream: *mut c_void,
    pub round_exec: *mut *mut c_void,
    pub round_exec_c: *mut *mut c_void,
    pub step_exec: *mut *mut c_void,
    pub step_exec_c: *mut *mut c_void,
    pub prefill_exec: *mut *mut c_void,
    pub prefill_dev_exec: *mut *mut c_void,
    pub native_rope: i32,
    pub norm_rope_usable: i32,
    pub head_type: i32,
    pub sub_head: i32,
    pub dhead_type: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkP4MtpPfView {
    pub tok: *mut i32,
    pub step: *mut i32,
    pub pos: *mut i32,
    pub rin: *mut f32,
    pub window: i64,
    pub prompt_len: *const i64,
    pub ms_prefill: *mut f64,
}

impl RustMtp {
    unsafe fn new(f: &'static P4Ffi, hooks: *mut TkP4Hooks) -> Option<Self> {
        let mut v: TkP4MtpView = std::mem::zeroed();
        if (f.mtp_view)(hooks, &mut v) != 0 || v.eligible == 0 {
            return None;
        }
        Some(Self {
            f,
            v,
            rows16: [0; PLE_N_HEADS],
            pf: None,
            rec: None,
            step_cap: 0,
        })
    }

    /// Re-read the view: `bind` (after a prompt's prefill in the generate loop) sets the window rows, the coupled
    /// buffers and the q lists.
    unsafe fn refresh(&mut self) {
        let mut v: TkP4MtpView = std::mem::zeroed();
        if (self.f.mtp_view_of)(self.v.mtp, &mut v) == 0 {
            self.v = v;
        }
    }

    /// The Rust plane over a `strata::core::MtpDrafter*` a C++ caller handed over (phase 2b).
    unsafe fn of_ptr(f: &'static P4Ffi, mtp: *mut c_void) -> Option<Self> {
        let mut v: TkP4MtpView = std::mem::zeroed();
        if (f.mtp_view_of)(mtp, &mut v) != 0 || v.eligible == 0 {
            return None;
        }
        Some(Self {
            f,
            v,
            rows16: [0; PLE_N_HEADS],
            pf: None,
            rec: None,
            step_cap: 0,
        })
    }

    /// `MtpDrafter::prefill`: the draft layer's K/V for prompt cells [cell0, cell0 + n) from the residual rows
    /// `r_rows` (device) and `next_tokens` (the token at cell + 1), in groups of up to max_t rows.  E-4 (one upload of
    /// every group's records, one sync) unless STRATA_MTP_PREFILL_SYNC (HIP default: a sync per group).
    pub unsafe fn prefill(
        &mut self,
        r_rows: *const f32,
        next_tokens: &[i32],
        n: i64,
        cell0: i64,
        sync: bool,
    ) -> Result<(), String> {
        let t0 = Instant::now();
        if self.pf.is_none() {
            let mut pv: TkP4MtpPfView = std::mem::zeroed();
            if (self.f.mtp_pf_view)(self.v.mtp, &mut pv) != 0 {
                return Err("mtp prefill: the drafter view is not available".into());
            }
            self.pf = Some(pv);
        }
        let pf = self.pf.unwrap();
        let v = self.v;
        let hcn = v.hcn as usize;
        let prompt_len = *pf.prompt_len;
        let first_needed = if pf.window > 0 && prompt_len > 0 {
            prompt_len - pf.window - 64
        } else {
            0
        };
        let nh = v.n_head as usize;
        let mt = v.max_t as i64;
        let cs = v.stream as hipStream_t;
        let last_err = || format!("mtp prefill: {}", hip_err_str(hipGetLastError()));
        if !verify_env().mtp_prefill_sync && n > 0 {
            let nu = n as usize;
            let per_row = 1 + 4 + nh;
            let mut rec = vec![0i32; nu * per_row];
            {
                let (tk, rest) = rec.split_at_mut(nu);
                let (stp, ps) = rest.split_at_mut(4 * nu);
                for i in 0..nu {
                    let cell = cell0 + i as i64;
                    tk[i] = next_tokens[i];
                    stp[i * 4] = cell as i32;
                    stp[i * 4 + 1] = (cell + 1) as i32;
                    stp[i * 4 + 2] = ((cell + 1) / 4) as i32;
                    stp[i * 4 + 3] = (cell + 1) as i32;
                    ps[i * nh..(i + 1) * nh].fill(cell as i32);
                }
            }
            let mut d: *mut i32 = std::ptr::null_mut();
            let mut eb = err_buf();
            if (self.f.mtp_pf_upload)(
                v.mtp,
                rec.as_ptr(),
                rec.len() as i64,
                &mut d,
                eb.as_mut_ptr(),
                eb.len(),
            ) != 0
            {
                return Err(err_from(&eb));
            }
            let d_tk = d;
            let d_stp = d.add(nu);
            let d_ps = d_stp.add(4 * nu);
            let mut c = 0i64;
            while c < n {
                let t = mt.min(n - c);
                if cell0 + c + t <= first_needed {
                    c += mt;
                    continue;
                }
                let exec = self.prefill_graph(t as i32, true)?;
                let cu = c as usize;
                let tu = t as usize;
                let ok = hipMemcpyAsync(
                    pf.tok as *mut c_void,
                    d_tk.add(cu) as *const c_void,
                    tu * 4,
                    hipMemcpyDeviceToDevice,
                    cs,
                ) == hipSuccess
                    && hipMemcpyAsync(
                        pf.step as *mut c_void,
                        d_stp.add(cu * 4) as *const c_void,
                        tu * 16,
                        hipMemcpyDeviceToDevice,
                        cs,
                    ) == hipSuccess
                    && hipMemcpyAsync(
                        pf.pos as *mut c_void,
                        d_ps.add(cu * nh) as *const c_void,
                        tu * nh * 4,
                        hipMemcpyDeviceToDevice,
                        cs,
                    ) == hipSuccess
                    && hipMemcpyAsync(
                        pf.rin as *mut c_void,
                        r_rows.add(cu * hcn) as *const c_void,
                        tu * hcn * 4,
                        hipMemcpyDeviceToDevice,
                        cs,
                    ) == hipSuccess
                    && hipGraphLaunch(exec as hipGraphExec_t, cs) == hipSuccess;
                if !ok {
                    return Err(last_err());
                }
                c += mt;
            }
            if !sync {
                let _ = hipStreamQuery(cs);
                *pf.ms_prefill += ms_since(t0);
                return Ok(());
            }
            if hipStreamSynchronize(cs) != hipSuccess {
                return Err(last_err());
            }
            *pf.ms_prefill += ms_since(t0);
            return Ok(());
        }
        let mut c = 0i64;
        while c < n {
            let t = mt.min(n - c);
            if cell0 + c + t <= first_needed {
                c += mt;
                continue;
            }
            let exec = self.prefill_graph(t as i32, false)?;
            for ti in 0..t as usize {
                let cell = cell0 + c + ti as i64;
                *v.h_tok.add(ti) = next_tokens[c as usize + ti];
                let st = v.h_step.add(ti * 4);
                *st = cell as i32;
                *st.add(1) = (cell + 1) as i32;
                *st.add(2) = ((cell + 1) / 4) as i32;
                *st.add(3) = (cell + 1) as i32;
                for h in 0..nh {
                    *v.h_pos.add(ti * nh + h) = cell as i32;
                }
            }
            let ok = hipMemcpyAsync(
                pf.rin as *mut c_void,
                r_rows.add(c as usize * hcn) as *const c_void,
                t as usize * hcn * 4,
                hipMemcpyDeviceToDevice,
                cs,
            ) == hipSuccess
                && hipGraphLaunch(exec as hipGraphExec_t, cs) == hipSuccess
                && hipStreamSynchronize(cs) == hipSuccess;
            if !ok {
                return Err(last_err());
            }
            c += mt;
        }
        *pf.ms_prefill += ms_since(t0);
        Ok(())
    }

    /// `MtpDrafter::coupled`: coupled draft sampling is on for this request.
    pub fn coupled(&self) -> bool {
        unsafe { *self.v.coupled_active != 0 }
    }

    /// `MtpDrafter::prob`: the drafts are sampled from q and their q lists are in `spec_q`.
    pub fn prob(&self) -> bool {
        self.coupled() && !self.v.h_q.is_null()
    }

    pub fn spec_q(&self) -> *const i32 {
        self.v.h_q
    }

    /// `MtpDrafter::set_draft_sampling`: a sampled request drafts with the target's chain (coupled mode).
    pub unsafe fn set_draft_sampling(&mut self, sp: &SamplerParams) {
        let on = self.v.coupled_ok != 0
            && !sp.greedy
            && sp.temperature > 0.0
            && sp.temperature >= verify_env().spec_min_temp;
        *self.v.coupled_active = on as u8;
        if !on || self.v.cparams.is_null() {
            return;
        }
        *(self.v.cparams as *mut SamplerParams) = *sp;
        std::slice::from_raw_parts_mut(self.v.h_chist, self.v.coupled_hist_cap as usize).fill(-1);
    }

    /// `MtpDrafter::set_draft_history`: the penalty ring's base for the next chain - the last h of
    /// (`tail`, `next`).
    pub unsafe fn set_draft_history(&mut self, tail: &[i32], next: i32) {
        if !self.coupled() || self.v.cparams.is_null() {
            return;
        }
        let cap = self.v.coupled_hist_cap;
        let h = coupled_hist_len(
            (*(self.v.cparams as *const SamplerParams)).penalty_last_n,
            cap,
        );
        if h <= 0 {
            return;
        }
        let out =
            std::slice::from_raw_parts_mut(self.v.h_chist.add((cap - h) as usize), h as usize);
        penalty_rows(tail, &[next], 1, h as usize, out);
    }

    unsafe fn put(&self, row: usize, cell: i64) {
        let v = &self.v;
        let s = v.h_step.add(row * 4);
        *s = cell as i32;
        *s.add(1) = (cell + 1) as i32;
        *s.add(2) = ((cell + 1) / 4) as i32;
        *s.add(3) = (cell + 1) as i32;
        let nh = v.n_head as usize;
        for h in 0..nh {
            *v.h_pos.add(row * nh + h) = cell as i32;
        }
    }

    unsafe fn prefetch_ple(&mut self, do_ple: bool, prev: &mut [i32; 2], tok: i32) {
        if !do_ple || tok < 0 {
            return;
        }
        let one = [tok];
        (self.f.ple_rows)(
            self.v.ple_ss,
            one.as_ptr(),
            1,
            prev.as_mut_ptr(),
            self.rows16.as_mut_ptr(),
        );
    }

    unsafe fn wait_out(&self, j: usize) {
        let cs = self.v.stream as hipStream_t;
        let mut spins = 0u32;
        while std::ptr::read_volatile(self.v.h_out.add(j)) < 0 {
            cpu_pause();
            spins = spins.wrapping_add(1);
            if spins & 1023 == 0 && hipStreamQuery(cs) != hipErrorNotReady {
                break;
            }
        }
    }

    /// `MtpDrafter::draft`: catch-up over the window's cells from `p` (rows = the window's final residuals,
    /// `tokens` = its picks), then the chain from row `a` for up to max_t - 1 drafts, stopped early once a draft's
    /// probability is below `min_p`.  Returns the drafts made.
    pub unsafe fn draft(
        &mut self,
        t_in: i32,
        tokens: &[i32],
        p: i64,
        a: i32,
        drafts: &mut [i32],
        probs: Option<&mut [f32]>,
        min_p: f32,
    ) -> Result<i32, String> {
        let v = self.v;
        if t_in < 1 || t_in > v.max_t || a < 0 || a >= t_in {
            return Err("mtp: draft arguments out of range".into());
        }
        if *v.chain_live != 0 {
            return Err("mtp: a pipelined chain is still in flight".into());
        }
        // #783 PR-i: the round runs for the cells up to the accepted row a
        let t = if verify_env().catchup_all {
            t_in
        } else {
            a + 1
        };
        let cp = *v.coupled_active != 0;
        let mut max_steps =
            ((v.max_t - 1).min(*v.max_drafts) as i64).min(*v.max_cells - (p + a as i64)) as i32;
        if self.step_cap > 0 {
            max_steps = max_steps.min(self.step_cap);
        }
        if max_steps > 0
            && (max_steps as usize > drafts.len()
                || probs
                    .as_deref()
                    .is_some_and(|p| (max_steps as usize) > p.len()))
        {
            return Err("mtp: the drafts buffer is shorter than the drafter's window".into());
        }
        let mut round: *mut c_void = std::ptr::null_mut();
        let mut steps = [std::ptr::null_mut::<c_void>(); 9];
        let mut eb = err_buf();
        self.prepare_round(t, cp, max_steps, &mut round, &mut steps, &mut eb)?;
        let t0 = Instant::now();
        let mt = v.max_t as usize;
        for ti in 0..t as usize {
            *v.h_tok.add(ti) = tokens[ti];
            self.put(ti, p + ti as i64);
        }
        self.put(2 * mt - 1, p + a as i64);
        for j in 1..max_steps.max(0) as usize {
            self.put(mt + j - 1, p + a as i64 + j as i64);
        }
        for j in 0..max_steps.max(0) as usize {
            std::ptr::write_volatile(v.h_out.add(j), -1);
        }
        *v.h_row = a;
        *v.h_row.add(1) = 0;
        if !v.h_force.is_null() {
            for j in 0..16 {
                *v.h_force.add(j) = -1;
            }
        }
        fence(Ordering::SeqCst);
        self.stage_source_r(t)?;
        let do_ple = v.ple_on != 0;
        let mut prev = if do_ple {
            [*v.ple_prev, *v.ple_prev.add(1)]
        } else {
            [0, 0]
        };
        let cs = v.stream as hipStream_t;
        let mut probs = probs;
        let mut n = 0i32;
        let launch_err =
            |what: &str| -> String { format!("{what}: {}", hip_err_str(hipGetLastError())) };
        if min_p <= 0.0 && max_steps > 0 {
            if hipGraphLaunch(round as hipGraphExec_t, cs) != hipSuccess {
                return Err(launch_err("mtp draft"));
            }
            for j in 1..max_steps as usize {
                if hipGraphLaunch(steps[j] as hipGraphExec_t, cs) != hipSuccess {
                    return Err(launch_err("mtp draft step"));
                }
            }
            let _ = hipStreamQuery(cs);
            self.prefetch_ple(do_ple, &mut prev, tokens[a as usize]);
            (self.f.progress)(c"mtp draft: waiting for the GPU".as_ptr(), -1, 0);
            for j in 0..(max_steps - 1) as usize {
                self.wait_out(j);
                drafts[j] = std::ptr::read_volatile(v.h_out.add(j));
                if let Some(pr) = probs.as_deref_mut() {
                    pr[j] = std::ptr::read_volatile(v.h_prob.add(j));
                }
                self.prefetch_ple(do_ple, &mut prev, drafts[j]);
            }
            if hipStreamSynchronize(cs) != hipSuccess {
                return Err(launch_err("mtp draft"));
            }
            let last = (max_steps - 1) as usize;
            drafts[last] = std::ptr::read_volatile(v.h_out.add(last));
            if let Some(pr) = probs.as_deref_mut() {
                pr[last] = std::ptr::read_volatile(v.h_prob.add(last));
            }
            self.prefetch_ple(do_ple, &mut prev, drafts[last]);
            n = max_steps;
        } else if max_steps > 0 {
            if hipGraphLaunch(round as hipGraphExec_t, cs) != hipSuccess {
                return Err(launch_err("mtp draft"));
            }
            let ahead = verify_env().draft_ahead;
            let mut launched = 1usize; // steps[1..launched) are queued
            if ahead && max_steps > 1 {
                if hipGraphLaunch(steps[1] as hipGraphExec_t, cs) != hipSuccess {
                    return Err(launch_err("mtp draft step"));
                }
                launched = 2;
            }
            let _ = hipStreamQuery(cs);
            self.prefetch_ple(do_ple, &mut prev, tokens[a as usize]);
            (self.f.progress)(c"mtp draft: waiting for the GPU".as_ptr(), -1, 0);
            self.wait_out(0);
            drafts[0] = std::ptr::read_volatile(v.h_out);
            let mut pj = std::ptr::read_volatile(v.h_prob);
            if let Some(pr) = probs.as_deref_mut() {
                pr[0] = pj;
            }
            n = 1;
            let mut j = 1usize;
            while (j as i32) < max_steps && pj >= min_p {
                while launched <= j || (ahead && launched <= j + 1 && (launched as i32) < max_steps)
                {
                    if hipGraphLaunch(steps[launched] as hipGraphExec_t, cs) != hipSuccess {
                        return Err(launch_err("mtp draft step"));
                    }
                    launched += 1;
                }
                let _ = hipStreamQuery(cs);
                self.prefetch_ple(do_ple, &mut prev, drafts[j - 1]);
                self.wait_out(j);
                drafts[j] = std::ptr::read_volatile(v.h_out.add(j));
                pj = std::ptr::read_volatile(v.h_prob.add(j));
                if let Some(pr) = probs.as_deref_mut() {
                    pr[j] = pj;
                }
                n += 1;
                j += 1;
            }
            self.prefetch_ple(do_ple, &mut prev, drafts[(n - 1) as usize]);
            if hipStreamSynchronize(cs) != hipSuccess {
                return Err(launch_err("mtp draft"));
            }
        }
        for j in n.max(0) as usize..(mt - 1).min(drafts.len()) {
            drafts[j] = 0;
            if let Some(pr) = probs.as_deref_mut() {
                if j < pr.len() {
                    pr[j] = 0.0;
                }
            }
        }
        *v.ms_draft += ms_since(t0);
        *v.rounds += 1;
        Ok(n)
    }

    /// `MtpDrafter::stage_source_R`: `set_source_R`'s rows into the bound buffer (null or the bound one: nothing).
    unsafe fn stage_source_r(&mut self, t: i32) -> Result<(), String> {
        let src = *self.v.src_r;
        if src.is_null() || src == self.v.window_r as *const f32 {
            return Ok(());
        }
        let ts = Instant::now();
        let rc = hipMemcpyAsync(
            self.v.window_r as *mut c_void,
            src as *const c_void,
            t as usize * self.v.hcn as usize * 4,
            hipMemcpyDeviceToDevice,
            self.v.stream as hipStream_t,
        );
        *self.v.ms_src += ms_since(ts);
        *self.v.n_src += 1;
        if rc != hipSuccess {
            return Err("mtp: staging the window's residual rows failed".into());
        }
        Ok(())
    }

    /// `MtpDrafter::draft_first`: the first round from one residual row (device) and its token; rows 1.. repeat it.
    pub unsafe fn draft_first(
        &mut self,
        t: i32,
        r_row: *const f32,
        token: i32,
        cell: i64,
        drafts: &mut [i32],
        probs: Option<&mut [f32]>,
        min_p: f32,
    ) -> Result<i32, String> {
        let hcn = self.v.hcn as usize;
        for ti in 0..t.max(0) as usize {
            if hipMemcpy(
                self.v.window_r.add(ti * hcn) as *mut c_void,
                r_row as *const c_void,
                hcn * 4,
                hipMemcpyDeviceToDevice,
            ) != hipSuccess
            {
                return Err("mtp: staging the first residual failed".into());
            }
        }
        let toks = vec![token; t.max(0) as usize];
        self.draft(t, &toks, cell, 0, drafts, probs, min_p)
    }
}

// ============================================================================
// Piece4VerifyMtp: the serve / generate window bodies
// ============================================================================

/// The Rust plane bound to one loop's hooks (serve or generate).
struct P4Bound {
    hooks: *mut TkP4Hooks,
    info: TkP4HooksInfo,
    ver: RustVerifier,
    mtp: Option<RustMtp>,
    hist_stage: Vec<i32>,
}

/// Piece 4 speculative verifier + MTP drafter state owned by `Qwen4ExpRustEngine`.
pub struct Piece4VerifyMtp {
    pub max_spec_window: i32,
    pub mtp_max_t: i32,
    pub pcie_mode: i32,
    pub windows_verified: u64,
    pub tokens_committed: u64,
    /// when > 0, `serve_draft` makes at most this many MTP drafts (the next window is a copy).
    pub draft_step_cap: i32,
    /// `TALUS_P4_CPP=1`: the C++ window bodies under TALUS_FN_RUST=1 (A/B).
    pub force_cpp: bool,
    bound: Option<P4Bound>,
    bound_ctx: *mut c_void,
    dump_dir: Option<String>,
}

impl Clone for Piece4VerifyMtp {
    fn clone(&self) -> Self {
        Self {
            max_spec_window: self.max_spec_window,
            mtp_max_t: self.mtp_max_t,
            pcie_mode: self.pcie_mode,
            windows_verified: self.windows_verified,
            tokens_committed: self.tokens_committed,
            draft_step_cap: self.draft_step_cap,
            force_cpp: self.force_cpp,
            bound: None,
            bound_ctx: std::ptr::null_mut(),
            dump_dir: self.dump_dir.clone(),
        }
    }
}

impl std::fmt::Debug for Piece4VerifyMtp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Piece4VerifyMtp")
            .field("windows_verified", &self.windows_verified)
            .field("tokens_committed", &self.tokens_committed)
            .field("force_cpp", &self.force_cpp)
            .field("bound", &self.bound.is_some())
            .finish()
    }
}

unsafe impl Send for Piece4VerifyMtp {}

fn print_serve_err(msg: &str) {
    println!("ERR {msg}");
    let _ = std::io::stdout().flush();
}

impl Piece4VerifyMtp {
    pub fn new(_cfg: &Qwen4ExpModelConfig) -> Self {
        Self {
            max_spec_window: 16,
            mtp_max_t: 5,
            pcie_mode: 2, // copy kernel default (avoids host DMA driver lock in verify window)
            windows_verified: 0,
            tokens_committed: 0,
            draft_step_cap: 0,
            force_cpp: env_flag("TALUS_P4_CPP")
                || std::env::var("TALUS_FN_P4")
                    .map(|v| v.trim() == "0")
                    .unwrap_or(false),
            bound: None,
            bound_ctx: std::ptr::null_mut(),
            dump_dir: std::env::var("TALUS_FN_DUMP")
                .ok()
                .filter(|d| !d.is_empty()),
        }
    }

    /// Bind the Rust plane to this loop's hooks (once per ctx); false: the C++ bodies run.
    unsafe fn bind(&mut self, kernels: &Qwen4ExpKernels, ctx: *mut c_void, serve: bool) -> bool {
        if self.force_cpp {
            return false;
        }
        if !self.bound_ctx.is_null() && self.bound_ctx == ctx {
            return self.bound.is_some();
        }
        self.bound_ctx = ctx;
        self.bound = None;
        let Some(f) = p4_ffi(kernels) else {
            return false;
        };
        let hooks = if serve {
            (f.serve_hooks)(ctx)
        } else {
            (f.gen_hooks)(ctx)
        };
        if hooks.is_null() {
            return false;
        }
        let mut info: TkP4HooksInfo = std::mem::zeroed();
        if (f.hooks_info)(hooks, &mut info) != 0 {
            return false;
        }
        let Some(ver) = RustVerifier::new(f, hooks, info.commit_async != 0) else {
            eprintln!("talus-qwen4exp: qwen4exp-p4: this verifier is not one the Rust plane runs (layer split / remote / trace): C++ window bodies");
            return false;
        };
        // the drafter (the generate loop has bound it in spec_init, before its first window)
        let mtp = if info.use_mtp != 0 {
            match RustMtp::new(f, hooks) {
                Some(m) => Some(m),
                None => {
                    eprintln!("talus-qwen4exp: qwen4exp-p4: the drafter is not one the Rust plane runs (draft diagnostics on): C++ window bodies");
                    return false;
                }
            }
        } else {
            None
        };
        let cap = if info.hist_cap > 0 {
            info.hist_cap as usize
        } else {
            0
        };
        eprintln!(
            "talus-qwen4exp: qwen4exp-p4: the Rust verify/MTP plane runs the {} windows{}",
            if serve { "serve" } else { "generate" },
            if verify_env().capture_cpp {
                " (C++ graph capture)"
            } else {
                ""
            }
        );
        self.bound = Some(P4Bound {
            hooks,
            info,
            ver,
            mtp,
            hist_stage: vec![-1; cap],
        });
        true
    }

    fn mtp_of(b: &mut P4Bound) -> Option<&mut RustMtp> {
        b.mtp.as_mut()
    }

    unsafe fn hook(b: &P4Bound, which: c_int, a: i64, bb: i64) -> c_int {
        (b.ver.f.hook)(b.hooks, which, a, bb)
    }

    unsafe fn drive_failed(b: &P4Bound) -> Option<String> {
        let p = (b.ver.f.drive_failed)(b.hooks);
        if p.is_null() {
            None
        } else {
            Some(CStr::from_ptr(p).to_string_lossy().into_owned())
        }
    }

    unsafe fn dump_window(
        &self,
        kernels: &Qwen4ExpKernels,
        b: &P4Bound,
        pos: i64,
        t: i32,
        first: bool,
    ) {
        if self.dump_dir.is_none() {
            return;
        }
        let nv = b.ver.vocab().max(0) as usize;
        let mut lrow = vec![0f32; nv];
        if first && b.ver.copy_logits(0, &mut lrow) {
            let tag = CString::new("first_logits").unwrap();
            (kernels.dump_tensor)(
                tag.as_ptr(),
                -1,
                pos,
                lrow.as_ptr(),
                lrow.len(),
                0,
                std::ptr::null_mut(),
            );
        }
        let tag = CString::new(format!("verify_p{pos}_T{t}_final_R")).unwrap();
        (kernels.dump_tensor)(
            tag.as_ptr(),
            47,
            pos,
            b.ver.final_r_all(),
            t as usize * b.ver.row_floats(),
            1,
            std::ptr::null_mut(),
        );
        if b.ver.copy_logits(0, &mut lrow) {
            let tag = CString::new(format!("verify_p{pos}_T{t}_logits0")).unwrap();
            (kernels.dump_tensor)(
                tag.as_ptr(),
                -1,
                pos,
                lrow.as_ptr(),
                lrow.len(),
                0,
                std::ptr::null_mut(),
            );
        }
    }

    /// The q lists of the window's MTP drafts when the drafter samples (probabilistic acceptance).
    unsafe fn spec_q_of(
        b: &mut P4Bound,
        from_sfx: bool,
        t_mtp: i32,
        gen_corrupt: bool,
    ) -> Option<(*const i32, i32)> {
        if b.info.use_mtp == 0 || from_sfx || t_mtp <= 1 || gen_corrupt {
            return None;
        }
        let m = Self::mtp_of(b)?;
        if m.prob() {
            Some((m.spec_q(), t_mtp - 1))
        } else {
            None
        }
    }

    // ------------------------------------------------------------------ serve

    /// `tk_qwen4exp_serve_verify_window`: one verify window of the serve loop.  1 ok, -1 error (ERR printed).
    pub unsafe fn serve_verify(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        t: i32,
        t_mtp: i32,
        from_sfx: bool,
        pos: i64,
        window: &[i32],
        consumed: &[i32],
        out_tokens: &mut [i32],
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, true) {
            return (kernels.serve_verify_window)(
                ctx,
                t,
                t_mtp,
                from_sfx as c_int,
                pos,
                window.as_ptr(),
                consumed.as_ptr(),
                consumed.len(),
                out_tokens.as_mut_ptr(),
            );
        }
        let mut b = self.bound.take().unwrap();
        let rc = self.serve_verify_bound(
            kernels, &mut b, t, t_mtp, from_sfx, pos, window, consumed, out_tokens,
        );
        self.bound = Some(b);
        if rc > 0 {
            self.windows_verified += 1;
        }
        rc
    }

    unsafe fn serve_verify_bound(
        &mut self,
        kernels: &Qwen4ExpKernels,
        b: &mut P4Bound,
        t: i32,
        t_mtp: i32,
        from_sfx: bool,
        pos: i64,
        window: &[i32],
        consumed: &[i32],
        out_tokens: &mut [i32],
    ) -> i32 {
        if Self::hook(b, TK_P4_HOOK_PRE_WINDOW, 0, 0) < 0 {
            return -1;
        }
        let hist_n = if b.info.hist_n.is_null() {
            0
        } else {
            *b.info.hist_n
        };
        if hist_n > 0 {
            let h = hist_n as usize;
            let need = t as usize * h;
            if b.hist_stage.len() < need {
                b.hist_stage.resize(need, -1);
            }
            penalty_rows(consumed, window, t as usize, h, &mut b.hist_stage[..need]);
            hipMemcpy(
                b.info.d_hist as *mut c_void,
                b.hist_stage.as_ptr() as *const c_void,
                need * 4,
                hipMemcpyHostToDevice,
            );
        }
        if Self::hook(b, TK_P4_HOOK_ENSURE_KV, pos, t as i64) < 0 {
            return -1;
        }
        Self::hook(b, TK_P4_HOOK_WINDOW_STARTED, 0, 0);
        let spec_q = Self::spec_q_of(b, from_sfx, t_mtp, false);
        let pool: Option<PoolMultiFn> = if b.info.pool.is_null() {
            None
        } else {
            Some(std::mem::transmute::<*mut c_void, PoolMultiFn>(b.info.pool))
        };
        let r = b
            .ver
            .run(t, window, pos, pool, b.info.pool_user, out_tokens, spec_q);
        let drive = Self::drive_failed(b);
        if r.is_err() || drive.is_some() {
            let msg = match (&drive, &r) {
                (Some(d), _) if !d.is_empty() => d.clone(),
                (_, Err(e)) => e.clone(),
                _ => String::new(),
            };
            print_serve_err(&msg);
            return -1;
        }
        self.dump_window(kernels, b, pos, t, false);
        Self::hook(b, TK_P4_HOOK_POST_WINDOW, 0, 0);
        1
    }

    /// `tk_qwen4exp_serve_copy_logits`: row `row` of the last window's head.
    pub unsafe fn serve_copy_logits(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        row: i32,
        out: &mut [f32],
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, true) {
            return (kernels.serve_copy_logits)(ctx, row, out.as_mut_ptr());
        }
        self.bound.as_ref().unwrap().ver.copy_logits(row, out) as i32
    }

    /// `tk_qwen4exp_serve_commit_window`: keep `keep` tokens.  1 ok, 0 non-finite logits (out_nf_hit = 1),
    /// -1 error (ERR printed).
    pub unsafe fn serve_commit(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        keep: i32,
        rounds: i64,
        first_window: bool,
        out_nf_hit: &mut i32,
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, true) {
            return (kernels.serve_commit_window)(
                ctx,
                keep,
                rounds,
                first_window as c_int,
                out_nf_hit,
            );
        }
        let b = self.bound.as_mut().unwrap();
        Self::hook(b, TK_P4_HOOK_COMMIT_BEGIN, rounds, 0);
        if let Err(e) = b.ver.commit(keep) {
            Self::hook(b, TK_P4_HOOK_JOIN_ADAPT, 0, 0);
            print_serve_err(&e);
            return -1;
        }
        let row = b.ver.nonfinite_row();
        if row >= 0 {
            eprintln!(
                "talus serve: non-finite logits (#879) in a {} window at position {} (row {}): its tokens are not emitted, the request ends (DONE nonfinite) and the next request reads its prompt from token 0",
                if first_window { "first decode" } else { "decode" },
                b.ver.nonfinite_pos(),
                row
            );
            let _ = std::io::stderr().flush();
            Self::hook(b, TK_P4_HOOK_MARK_NONFINITE, 0, 0);
            if Self::hook(b, TK_P4_HOOK_JOIN_ADAPT, 0, 0) < 0 {
                print_serve_err("an adaptive refill failed");
                return -1;
            }
            *out_nf_hit = 1;
            return 0;
        }
        *out_nf_hit = 0;
        Self::hook(b, TK_P4_HOOK_COMMIT_DONE, 0, 0);
        self.tokens_committed += keep.max(0) as u64;
        1
    }

    /// `tk_qwen4exp_serve_draft_window`: the drafts for the next window.  1 ok, -1 error (ERR printed).
    pub unsafe fn serve_draft(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        t: i32,
        accepted: i32,
        pos: i64,
        need_draft: bool,
        req_spec_min_p: f64,
        out_tokens: &[i32],
        consumed: &[i32],
        out_drafts: &mut [i32],
        out_dprob: &mut [f32],
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, true) {
            return (kernels.serve_draft_window)(
                ctx,
                t,
                accepted,
                pos,
                need_draft as c_int,
                req_spec_min_p,
                out_tokens.as_ptr(),
                consumed.as_ptr(),
                consumed.len(),
                out_drafts.as_mut_ptr(),
                out_dprob.as_mut_ptr(),
            );
        }
        let b = self.bound.as_mut().unwrap();
        let use_mtp = b.info.use_mtp != 0;
        let hist_n = if b.info.hist_n.is_null() {
            0
        } else {
            *b.info.hist_n
        };
        let mut drafted: Result<i32, String> = Ok(0);
        if use_mtp && need_draft {
            match Self::mtp_of(b) {
                Some(m) => {
                    if hist_n > 0 && m.coupled() {
                        m.set_draft_history(consumed, out_tokens[accepted as usize]);
                    }
                    m.step_cap = self.draft_step_cap;
                    drafted = m.draft(
                        t,
                        out_tokens,
                        pos,
                        accepted,
                        out_drafts,
                        Some(out_dprob),
                        req_spec_min_p as f32,
                    );
                }
                None => drafted = Err("mtp: the drafter is not available to the Rust plane".into()),
            }
        }
        Self::hook(b, TK_P4_HOOK_DRAFT_DONE, t as i64, 0);
        if Self::hook(b, TK_P4_HOOK_JOIN_ADAPT, 0, 0) < 0 {
            print_serve_err("an adaptive refill failed");
            return -1;
        }
        if let Err(e) = drafted {
            print_serve_err(&e);
            return -1;
        }
        1
    }

    // ------------------------------------------------------------------ generate

    /// `tk_qwen4exp_gen_spec_verify`: one verify window of the generate loop.  1 ok, -1 error (printed).
    pub unsafe fn gen_verify(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpGenCtx,
        t: i32,
        t_mtp: i32,
        from_sfx: bool,
        pos: i64,
        window: &[i32],
        out_tokens: &mut [i32],
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, false) {
            return (kernels.gen_spec_verify)(
                ctx,
                t,
                t_mtp,
                from_sfx as c_int,
                pos,
                window.as_ptr(),
                out_tokens.as_mut_ptr(),
            );
        }
        let mut b = self.bound.take().unwrap();
        let rc = (|| {
            if Self::hook(&b, TK_P4_HOOK_PRE_WINDOW, 0, 0) < 0 {
                return -1;
            }
            let corrupt = b.info.spec_corrupt > 0;
            let spec_q = Self::spec_q_of(&mut b, from_sfx, t_mtp, corrupt);
            let pool: Option<PoolMultiFn> = if b.info.pool.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, PoolMultiFn>(b.info.pool))
            };
            if let Err(e) = b
                .ver
                .run(t, window, pos, pool, b.info.pool_user, out_tokens, spec_q)
            {
                eprintln!("talus generate: {e}");
                return -1;
            }
            if let Some(d) = Self::drive_failed(&b) {
                eprintln!("talus generate: {d}");
                return -1;
            }
            let first = Self::hook(&b, TK_P4_HOOK_FIRST_VERIFIED, 0, 0) == 1;
            self.dump_window(kernels, &b, pos, t, first);
            1
        })();
        self.bound = Some(b);
        if rc > 0 {
            self.windows_verified += 1;
        }
        rc
    }

    /// `tk_qwen4exp_gen_spec_copy_logits`.
    pub unsafe fn gen_copy_logits(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpGenCtx,
        row: i32,
        out: &mut [f32],
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, false) {
            return (kernels.gen_spec_copy_logits)(ctx, row, out.as_mut_ptr());
        }
        self.bound.as_ref().unwrap().ver.copy_logits(row, out) as i32
    }

    /// `tk_qwen4exp_gen_spec_commit`.  1 ok, -1 error (printed).
    pub unsafe fn gen_commit(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpGenCtx,
        keep: i32,
        rounds: i64,
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, false) {
            return (kernels.gen_spec_commit)(ctx, keep, rounds);
        }
        let b = self.bound.as_mut().unwrap();
        Self::hook(b, TK_P4_HOOK_COMMIT_BEGIN, rounds, 0);
        if let Err(e) = b.ver.commit(keep) {
            Self::hook(b, TK_P4_HOOK_JOIN_ADAPT, 0, 0);
            eprintln!("talus generate: {e}");
            return -1;
        }
        self.tokens_committed += keep.max(0) as u64;
        1
    }

    /// `tk_qwen4exp_gen_spec_join_adapt`.
    pub unsafe fn gen_join_adapt(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpGenCtx,
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, false) {
            return (kernels.gen_spec_join_adapt)(ctx);
        }
        Self::hook(self.bound.as_ref().unwrap(), TK_P4_HOOK_JOIN_ADAPT, 0, 0)
    }

    /// `tk_qwen4exp_gen_spec_draft`.  1 ok, -1 error (printed).
    pub unsafe fn gen_draft(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpGenCtx,
        t: i32,
        accepted: i32,
        pos: i64,
        need_draft: bool,
        spec_min_p: f64,
        out_tokens: &[i32],
        out_drafts: &mut [i32],
        out_dprob: &mut [f32],
    ) -> i32 {
        if !self.bind(kernels, ctx as *mut c_void, false) {
            return (kernels.gen_spec_draft)(
                ctx,
                t,
                accepted,
                pos,
                need_draft as c_int,
                out_tokens.as_ptr(),
                out_drafts.as_mut_ptr(),
                out_dprob.as_mut_ptr(),
            );
        }
        let b = self.bound.as_mut().unwrap();
        let mut drafted: Result<i32, String> = Ok(0);
        if b.info.use_mtp != 0 && need_draft {
            drafted = match Self::mtp_of(b) {
                Some(m) => m.draft(
                    t,
                    out_tokens,
                    pos,
                    accepted,
                    out_drafts,
                    Some(out_dprob),
                    spec_min_p as f32,
                ),
                None => Err("mtp: the drafter is not available to the Rust plane".into()),
            };
        }
        if Self::hook(b, TK_P4_HOOK_JOIN_ADAPT, 0, 0) < 0 {
            return -1;
        }
        if let Err(e) = drafted {
            eprintln!("talus generate: {e}");
            return -1;
        }
        1
    }

    /// The bound verifier (null before the first window of a loop, or when the C++ bodies run).
    pub fn verifier(&mut self) -> Option<&mut RustVerifier> {
        self.bound.as_mut().map(|b| &mut b.ver)
    }

    // ------------------------------------------------------------------ the coarse skeleton entry points

    /// Runs a speculative verify window of `t` tokens starting at `pos` (serve loop).
    pub unsafe fn verify_window(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        t: i32,
        t_mtp: i32,
        from_sfx: bool,
        pos: i64,
        window: &[i32],
        consumed: &[i32],
        out_tokens: &mut [i32],
    ) -> Result<(), String> {
        if self.serve_verify(
            kernels, ctx, t, t_mtp, from_sfx, pos, window, consumed, out_tokens,
        ) < 0
        {
            return Err(format!("verify_window failed at pos={pos}, t={t}"));
        }
        Ok(())
    }

    /// Commits `keep` accepted tokens (`a + 1`) from the current verify window into GDN/QSA/PLE state.
    /// Ok(true): the window met non-finite logits.
    pub unsafe fn commit_window(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        keep: i32,
        rounds: i64,
        first_window: bool,
    ) -> Result<bool, String> {
        let mut nf_hit = 0i32;
        let rc = self.serve_commit(kernels, ctx, keep, rounds, first_window, &mut nf_hit);
        if rc < 0 {
            return Err(format!(
                "commit_window failed at round={rounds}, keep={keep}"
            ));
        }
        Ok(nf_hit != 0 || rc == 0)
    }
}

// ============================================================================
// Phase 3: the MTP layer's graphs recorded from Rust (MtpDrafter::capture_round / capture_step / capture_prefill,
// record_front / record_rest), over the drafter's buffers and launch-only wrappers.
// ============================================================================

const TK_P4_M_EMBED: c_int = 0;
const TK_P4_M_GR_READ: c_int = 1;
const TK_P4_M_GR_READ_ROW: c_int = 2;
const TK_P4_M_KV_APPEND: c_int = 3;
const TK_P4_M_ATTN: c_int = 4;
const TK_P4_M_SHARED: c_int = 5;
const TK_P4_M_ROUTE: c_int = 6;
const TK_P4_M_EXPERTS: c_int = 7;
const TK_P4_M_COMBINE: c_int = 8;
const TK_P4_M_GR_WRITE: c_int = 9;
const TK_P4_M_HEAD: c_int = 10;
const TK_P4_M_ROUND_TAIL: c_int = 11;
const MTP_EPS: f32 = 1e-6;

impl RustMtp {
    /// TALUS_P4_CAPTURE_CPP / STRATA_VERIFY_NODES, a CUDA build (one-launch input copies) or the shared expert's
    /// side branch keep mtp.cpp's own capture.
    fn capture_in_rust(&mut self) -> bool {
        if verify_env().capture_cpp {
            return false;
        }
        let hip = unsafe { (self.f.env_flags)() } & 1 != 0;
        if !hip {
            return false;
        }
        if self.rec.is_none() {
            unsafe {
                let mut r: TkP4MtpRecView = std::mem::zeroed();
                if (self.f.mtp_rec_view)(self.v.mtp, &mut r) != 0 {
                    return false;
                }
                let env = (self.f.env_mtp)();
                self.rec = Some(MtpRec {
                    r,
                    env,
                    w: Default::default(),
                });
            }
        }
        self.rec.as_ref().map(|r| r.env & 32 == 0).unwrap_or(false)
    }

    fn tensor(&mut self, name: &'static str, kind: c_int) -> Result<(*const c_void, i32), String> {
        let rec = self.rec.as_mut().unwrap();
        if let Some(&p) = rec.w.get(name) {
            return Ok(p);
        }
        let c = CString::new(name).unwrap();
        let mut ty: c_int = -1;
        let p = unsafe { (self.f.mtp_tensor)(self.v.mtp, c.as_ptr(), kind, &mut ty) };
        rec.w.insert(name, (p, ty));
        Ok((p, ty))
    }

    fn f32w(&mut self, name: &'static str) -> Result<*const f32, String> {
        Ok(self.tensor(name, 0)?.0 as *const f32)
    }

    fn op(&self, op: c_int, t: i32, n: i32, j: i32, step: *const i32) -> Result<(), String> {
        let r = &self.rec.as_ref().unwrap().r;
        let mut eb = err_buf();
        if unsafe {
            (self.f.m_op)(
                self.v.mtp,
                op,
                t,
                n,
                j,
                step,
                r.stream,
                eb.as_mut_ptr(),
                eb.len(),
            )
        } < 0
        {
            return Err(err_from(&eb));
        }
        Ok(())
    }

    fn ck(rc: c_int, what: &str) -> Result<(), String> {
        if rc != 0 {
            Err(format!("mtp: {what} failed"))
        } else {
            Ok(())
        }
    }

    /// `record_front`: the layer's front for T rows at step rows [row0, +T) - the embedding and hidden branches, the
    /// attention hyper-connection read, the K/V appended in the state's own format.
    unsafe fn record_front(&mut self, t: i32, row0: i32) -> Result<(), String> {
        let (fe, t_fe) = self.tensor("fc_embedding.weight", 2)?;
        let (fh, t_fh) = self.tensor("fc_hidden.weight", 2)?;
        let (kp, t_kp) = self.tensor("self_attn.k_proj.weight", 2)?;
        let (vp, t_vp) = self.tensor("self_attn.v_proj.weight", 2)?;
        let w_emb = self.f32w("pre_fc_norm_embedding.weight")?;
        let w_hid = self.f32w("pre_fc_norm_hidden.weight")?;
        let w_kn = self.f32w("self_attn.k_norm.weight")?;
        let rec = self.rec.clone().unwrap();
        let r = &rec.r;
        let f = self.f;
        let cs = r.stream;
        let (n, hc, nh, hd, nkv) = (r.n, r.hc, r.nh, r.hd, r.nkv);
        let step = r.step.add(row0 as usize * 4) as *const i32;
        let pos = r.pos.add(row0 as usize * nh as usize) as *const i32;
        self.op(TK_P4_M_EMBED, 0, t, 0, std::ptr::null())?;
        Self::ck(
            (f.g_rms_norm)(r.emb, w_emb, r.en, n, t, MTP_EPS, cs),
            "rms_norm",
        )?;
        Self::ck(
            (f.g_quantize_q8_1)(r.en, r.xq as *mut c_void, n, t, cs),
            "quantize",
        )?;
        Self::ck(
            (f.g_mmvq)(t_fe, fe, r.xq as *const c_void, r.e2, n, n, t, cs),
            "mmvq",
        )?;
        if r.hnorm_stream != 0 {
            Self::ck(
                (f.g_rms_norm_grouped)(r.rin, w_hid, r.hn, n, hc, t * hc, MTP_EPS, cs),
                "rms_norm_grouped",
            )?;
        } else {
            Self::ck(
                (f.g_rms_norm)(r.rin, w_hid, r.hn, hc * n, t, MTP_EPS, cs),
                "rms_norm",
            )?;
        }
        let mut c0 = 0;
        while c0 < t * hc {
            let nc = 8.min(t * hc - c0);
            Self::ck(
                (f.g_quantize_q8_1)(r.hn.add((c0 * n) as usize), r.xq as *mut c_void, n, nc, cs),
                "quantize",
            )?;
            Self::ck(
                (f.g_mmvq)(
                    t_fh,
                    fh,
                    r.xq as *const c_void,
                    r.h2.add((c0 * n) as usize),
                    n,
                    n,
                    nc,
                    cs,
                ),
                "mmvq",
            )?;
            c0 += 8;
        }
        Self::ck(
            (f.g_add_streams_broadcast)(r.h2, r.e2, r.r, n as i64, hc, t, cs),
            "add_streams_broadcast",
        )?;
        self.op(TK_P4_M_GR_READ, 0, t, 0, std::ptr::null())?;
        Self::ck(
            (f.g_quantize_q8_1)(r.mixed, r.xq as *mut c_void, n, t, cs),
            "quantize",
        )?;
        Self::ck(
            (f.g_mmvq)(t_kp, kp, r.xq as *const c_void, r.kcur, n, nkv * hd, t, cs),
            "mmvq",
        )?;
        Self::ck(
            (f.g_mmvq)(t_vp, vp, r.xq as *const c_void, r.vcur, n, nkv * hd, t, cs),
            "mmvq",
        )?;
        let fuse_nr = r.native_rope != 0 && r.norm_rope_usable != 0;
        if t == 1 && fuse_nr {
            Self::ck(
                (f.g_rms_norm_rope)(r.kcur, hd, w_kn, r.kcur, nkv, hd, r.n_rot, MTP_EPS, pos, cs),
                "rms_norm_rope",
            )?;
        } else {
            Self::ck(
                (f.g_rms_norm)(r.kcur, w_kn, r.kcur, hd, t * nkv, MTP_EPS, cs),
                "rms_norm",
            )?;
            for ti in 0..t as usize {
                let kc = r.kcur.add(ti * (nkv * hd) as usize);
                let p = pos.add(ti * nh as usize);
                if r.native_rope != 0 {
                    Self::ck((f.g_rope)(kc, kc, nkv, hd, r.n_rot, p, cs), "rope")?;
                } else {
                    Self::ck(
                        (f.g_rope_neox)(
                            kc, kc, nkv as i64, hd, r.n_rot, r.cos_tab, r.sin_tab, p, cs,
                        ),
                        "rope",
                    )?;
                }
            }
        }
        if r.kv_rot != 0 {
            Self::ck((f.g_fwht256)(r.kcur, (t * nkv) as i64, cs), "fwht")?;
            Self::ck((f.g_fwht256)(r.vcur, (t * nkv) as i64, cs), "fwht")?;
        }
        self.op(TK_P4_M_KV_APPEND, (rec.env & 16 != 0) as i32, t, 0, step)?;
        Ok(())
    }

    /// `record_rest`: row 0 at step row `step_row` - the attention from the q projection, the MLP hyper-connection,
    /// the MoE (router, the 512 resident experts, the shared expert, the combine, the write), the final mixer, the
    /// head and the draft with its probability.
    unsafe fn record_rest(&mut self, step_row: i32, coupled: bool, j: i32) -> Result<(), String> {
        let (qp, t_qp) = self.tensor("self_attn.q_proj.weight", 2)?;
        let (op_, t_op) = self.tensor("self_attn.o_proj.weight", 2)?;
        let w_qn = self.f32w("self_attn.q_norm.weight")?;
        let rec = self.rec.clone().unwrap();
        let r = &rec.r;
        let f = self.f;
        let cs = r.stream;
        let t = 1i32;
        let (n, nh, hd) = (r.n, r.nh, r.hd);
        let step = r.step.add(step_row as usize * 4) as *const i32;
        let pos = r.pos.add(step_row as usize * nh as usize) as *const i32;
        let fuse_nr = r.native_rope != 0 && r.norm_rope_usable != 0;
        Self::ck(
            (f.g_mmvq)(
                t_qp,
                qp,
                r.xq as *const c_void,
                r.qfull,
                n,
                nh * 2 * hd,
                t,
                cs,
            ),
            "mmvq",
        )?;
        if fuse_nr {
            Self::ck(
                (f.g_rms_norm_rope)(
                    r.qfull,
                    2 * hd,
                    w_qn,
                    r.qcur,
                    t * nh,
                    hd,
                    r.n_rot,
                    MTP_EPS,
                    pos,
                    cs,
                ),
                "rms_norm_rope",
            )?;
        } else {
            if (f.g_copy_2d)(
                r.qcur as *mut c_void,
                hd as usize * 4,
                r.qfull as *const c_void,
                hd as usize * 2 * 4,
                hd as usize * 4,
                (t * nh) as usize,
                cs,
            ) != 0
            {
                return Err("mtp: q split failed".into());
            }
            // norm_rope
            Self::ck(
                (f.g_rms_norm)(r.qcur, w_qn, r.qcur, hd, t * nh, MTP_EPS, cs),
                "rms_norm",
            )?;
            if r.native_rope != 0 {
                Self::ck(
                    (f.g_rope)(r.qcur, r.qcur, t * nh, hd, r.n_rot, pos, cs),
                    "rope",
                )?;
            } else {
                Self::ck(
                    (f.g_rope_neox)(
                        r.qcur,
                        r.qcur,
                        (t * nh) as i64,
                        hd,
                        r.n_rot,
                        r.cos_tab,
                        r.sin_tab,
                        pos,
                        cs,
                    ),
                    "rope",
                )?;
            }
        }
        if r.kv_rot != 0 {
            Self::ck((f.g_fwht256)(r.qcur, (t * nh) as i64, cs), "fwht")?;
        }
        self.op(TK_P4_M_ATTN, 0, t, 0, step)?;
        if r.kv_rot != 0 {
            Self::ck((f.g_fwht256)(r.attn, (t * nh) as i64, cs), "fwht")?;
        }
        Self::ck(
            (f.g_gate_apply)(r.attn, r.qfull, r.attn32, t * nh, hd, cs),
            "gate",
        )?;
        Self::ck(
            (f.g_quantize_q8_1)(r.attn32, r.xq as *mut c_void, nh * hd, t, cs),
            "quantize",
        )?;
        Self::ck(
            (f.g_mmvq)(t_op, op_, r.xq as *const c_void, r.bo, nh * hd, n, t, cs),
            "mmvq",
        )?;
        self.op(TK_P4_M_GR_READ, 1, t, 0, std::ptr::null())?;
        // (HIP: the shared expert on the one stream, after the routed experts)
        self.op(TK_P4_M_ROUTE, 0, t, 0, std::ptr::null())?;
        self.op(TK_P4_M_EXPERTS, 0, t, 0, std::ptr::null())?;
        self.op(TK_P4_M_SHARED, 0, t, 0, std::ptr::null())?;
        let native_combine = rec.env & 1 != 0;
        let fuse_head_gr = rec.env & 2 != 0;
        let head_mix_multi = rec.env & 4 != 0;
        let no_multi_gr = rec.env & 8 != 0;
        for ti in 0..t {
            if !(t > 1 && native_combine) {
                self.op(TK_P4_M_COMBINE, ti, t, 0, std::ptr::null())?;
            }
            if !fuse_head_gr && no_multi_gr {
                self.op(TK_P4_M_GR_WRITE, ti, t, 0, std::ptr::null())?;
            }
        }
        if t > 1 && native_combine {
            self.op(TK_P4_M_COMBINE, -1, t, 0, std::ptr::null())?;
        }
        if !fuse_head_gr && !no_multi_gr {
            self.op(TK_P4_M_GR_WRITE, -1, t, 0, std::ptr::null())?;
        }
        if fuse_head_gr {
            self.op(TK_P4_M_GR_READ, 2, t, 0, std::ptr::null())?;
        } else if head_mix_multi {
            self.op(TK_P4_M_GR_READ, 3, t, 0, std::ptr::null())?;
        } else {
            for ti in 0..t {
                self.op(TK_P4_M_GR_READ_ROW, ti, t, 0, std::ptr::null())?;
            }
        }
        self.op(TK_P4_M_HEAD, coupled as i32, t, j, step)?;
        Ok(())
    }

    unsafe fn begin_capture(&self) -> Result<(), String> {
        let cs = self.rec.as_ref().unwrap().r.stream as hipStream_t;
        if hip_sys::hipStreamBeginCapture(cs, hip_sys::hipStreamCaptureModeThreadLocal)
            != hipSuccess
        {
            return Err("mtp: begin capture".into());
        }
        Ok(())
    }

    /// End the capture and instantiate into slot (which, idx); a failed recording discards the graph.
    unsafe fn finish_capture(
        &self,
        recorded: Result<(), String>,
        which: c_int,
        idx: i32,
    ) -> Result<(), String> {
        let cs = self.rec.as_ref().unwrap().r.stream as hipStream_t;
        let mut graph: hip_sys::hipGraph_t = std::ptr::null_mut();
        let ce = hip_sys::hipStreamEndCapture(cs, &mut graph);
        if let Err(e) = recorded {
            if !graph.is_null() {
                hip_sys::hipGraphDestroy(graph);
            }
            return Err(e);
        }
        if ce != hipSuccess || graph.is_null() {
            if !graph.is_null() {
                hip_sys::hipGraphDestroy(graph);
            }
            return Err(format!("mtp: capture: {}", hip_err_str(ce)));
        }
        let mut eb = err_buf();
        if (self.f.mtp_instantiate)(
            self.v.mtp,
            which,
            idx,
            graph as *mut c_void,
            eb.as_mut_ptr(),
            eb.len(),
        ) != 0
        {
            return Err(err_from(&eb));
        }
        Ok(())
    }

    /// `capture_round`: the catch-up over the window's T cells (their K/V), then the rest for row a on row a's
    /// intermediates copied to row 0, at the cell in step row 2*max_t - 1, and draft 0 selected.
    unsafe fn capture_round(&mut self, t: i32, coupled: bool) -> Result<*mut c_void, String> {
        let r = self.rec.as_ref().unwrap().r;
        let slot = if coupled {
            r.round_exec_c.add(t as usize)
        } else {
            r.round_exec.add(t as usize)
        };
        if !(*slot).is_null() {
            return Ok(*slot);
        }
        let f = self.f;
        let cs = r.stream;
        self.begin_capture()?;
        let res = (|| -> Result<(), String> {
            if coupled {
                self.op(TK_P4_M_ROUND_TAIL, 0, 0, 0, std::ptr::null())?;
            }
            let ra = 2 * r.max_t - 1;
            let nh = r.nh as usize;
            let hcn = (r.hc * r.n) as i64;
            Self::ck(
                (f.k_copy_i32_from_mapped)(r.tok, r.m_tok, t as i64, cs),
                "copy",
            )?;
            Self::ck(
                (f.k_copy_i32_from_mapped)(r.step, r.m_step, 2 * t as i64 * 4, cs),
                "copy",
            )?;
            Self::ck(
                (f.k_copy_i32_from_mapped)(r.pos, r.m_pos, 2 * t as i64 * nh as i64, cs),
                "copy",
            )?;
            Self::ck((f.k_copy_i32_from_mapped)(r.row, r.m_row, 2, cs), "copy")?;
            Self::ck(
                (f.k_copy_from_mapped)(r.rin, self.v.window_r, t as i64 * hcn, cs),
                "copy",
            )?;
            Self::ck(
                (f.k_copy_i32_from_mapped)(
                    r.step.add(ra as usize * 4),
                    r.m_step.add(ra as usize * 4),
                    4,
                    cs,
                ),
                "copy",
            )?;
            Self::ck(
                (f.k_copy_i32_from_mapped)(
                    r.pos.add(ra as usize * nh),
                    r.m_pos.add(ra as usize * nh),
                    nh as i64,
                    cs,
                ),
                "copy",
            )?;
            self.record_front(t, 0)?;
            if t > 1 {
                self.op(TK_P4_M_ROUND_TAIL, 1, 0, 0, std::ptr::null())?;
            }
            self.record_rest(ra, coupled, 0)?;
            self.op(TK_P4_M_ROUND_TAIL, 2, 0, 0, std::ptr::null())?;
            if r.force_on != 0 && !coupled {
                self.op(TK_P4_M_ROUND_TAIL, 3, 0, 0, std::ptr::null())?;
            }
            Ok(())
        })();
        self.finish_capture(res, if coupled { 1 } else { 0 }, t)?;
        Ok(*slot)
    }

    /// `capture_step`: chain step j (1..max_t-2), one row at the cell in step row max_t + j - 1, from the previous
    /// step's residual and token; draft j and its probability to the mapped outputs.
    unsafe fn capture_step(&mut self, j: i32, coupled: bool) -> Result<*mut c_void, String> {
        let r = self.rec.as_ref().unwrap().r;
        let slot = if coupled {
            r.step_exec_c.add(j as usize)
        } else {
            r.step_exec.add(j as usize)
        };
        if !(*slot).is_null() {
            return Ok(*slot);
        }
        let f = self.f;
        let cs = r.stream;
        let row = r.max_t + j - 1;
        let nh = r.nh as usize;
        self.begin_capture()?;
        let res = (|| -> Result<(), String> {
            Self::ck(
                (f.k_copy_i32_from_mapped)(
                    r.step.add(row as usize * 4),
                    r.m_step.add(row as usize * 4),
                    4,
                    cs,
                ),
                "copy",
            )?;
            Self::ck(
                (f.k_copy_i32_from_mapped)(
                    r.pos.add(row as usize * nh),
                    r.m_pos.add(row as usize * nh),
                    nh as i64,
                    cs,
                ),
                "copy",
            )?;
            self.record_front(1, row)?;
            self.record_rest(row, coupled, j)?;
            self.op(TK_P4_M_ROUND_TAIL, 2, 0, j, std::ptr::null())?;
            if r.force_on != 0 && !coupled {
                self.op(TK_P4_M_ROUND_TAIL, 3, 0, j, std::ptr::null())?;
            }
            Ok(())
        })();
        self.finish_capture(res, if coupled { 3 } else { 2 }, j)?;
        Ok(*slot)
    }

    /// The round graph for T rows and the step graphs 1..max_steps-1 (each captured on first use).
    unsafe fn prepare_round(
        &mut self,
        t: i32,
        coupled: bool,
        max_steps: i32,
        round: &mut *mut c_void,
        steps: &mut [*mut c_void; 9],
        eb: &mut [c_char],
    ) -> Result<(), String> {
        if !self.capture_in_rust() {
            if (self.f.mtp_prepare)(
                self.v.mtp,
                t,
                coupled as c_int,
                max_steps,
                round,
                steps.as_mut_ptr(),
                eb.as_mut_ptr(),
                eb.len(),
            ) != 0
            {
                return Err(err_from(eb));
            }
            return Ok(());
        }
        *round = self.capture_round(t, coupled)?;
        for j in 1..max_steps.max(0) {
            steps[j as usize] = self.capture_step(j, coupled)?;
        }
        Ok(())
    }

    /// `capture_prefill` / `capture_prefill_dev`: the K/V-only layer for T prompt rows (the E-4 variant reads inputs
    /// the host copied on the device; the other copies them from the mapped staging in the graph).
    unsafe fn prefill_graph(&mut self, t: i32, dev: bool) -> Result<*mut c_void, String> {
        if !self.capture_in_rust() {
            let mut exec: *mut c_void = std::ptr::null_mut();
            let mut eb = err_buf();
            if (self.f.mtp_prefill_graph)(
                self.v.mtp,
                t,
                dev as c_int,
                &mut exec,
                eb.as_mut_ptr(),
                eb.len(),
            ) != 0
            {
                return Err(err_from(&eb));
            }
            return Ok(exec);
        }
        let r = self.rec.as_ref().unwrap().r;
        let slot = if dev {
            r.prefill_dev_exec.add(t as usize)
        } else {
            r.prefill_exec.add(t as usize)
        };
        if !(*slot).is_null() {
            return Ok(*slot);
        }
        let f = self.f;
        let cs = r.stream;
        self.begin_capture()?;
        let res = (|| -> Result<(), String> {
            if !dev {
                Self::ck(
                    (f.k_copy_i32_from_mapped)(r.tok, r.m_tok, t as i64, cs),
                    "copy",
                )?;
                Self::ck(
                    (f.k_copy_i32_from_mapped)(r.step, r.m_step, t as i64 * 4, cs),
                    "copy",
                )?;
                Self::ck(
                    (f.k_copy_i32_from_mapped)(r.pos, r.m_pos, t as i64 * r.nh as i64, cs),
                    "copy",
                )?;
            }
            self.record_front(t, 0)
        })();
        self.finish_capture(res, if dev { 5 } else { 4 }, t)?;
        Ok(*slot)
    }
}
// ============================================================================
// Phase 2b: every C++ Verifier::run / commit and MtpDrafter::draft / prefill to the Rust plane
// ============================================================================

#[repr(C)]
pub struct TkP4Ops {
    pub user: *mut c_void,
    pub ver_run: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            c_int,
            *const i32,
            i64,
            *mut c_void,
            *mut c_void,
            *mut i32,
            *const i32,
            c_int,
            *mut c_char,
            usize,
        ) -> c_int,
    >,
    pub ver_commit: Option<
        unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, c_int, *mut c_char, usize) -> c_int,
    >,
    pub mtp_draft: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            c_int,
            *const i32,
            i64,
            c_int,
            *mut i32,
            *mut f32,
            f32,
            *mut c_int,
            *mut c_char,
            usize,
        ) -> c_int,
    >,
    pub mtp_prefill: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *mut c_void,
            *const f32,
            *const i32,
            i64,
            i64,
            c_int,
            *mut c_char,
            usize,
        ) -> c_int,
    >,
    pub ver_wait_commit:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_char, usize) -> c_int>,
    pub ver_copy_logits:
        Option<unsafe extern "C" fn(*mut c_void, *const c_void, c_int, *mut f32) -> c_int>,
    pub mtp_set_draft_sampling:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void) -> c_int>,
    pub mtp_set_draft_history:
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void, *const i32, i64, i32) -> c_int>,
}

/// The Rust planes of the Verifier / MtpDrafter objects C++ code hands over, by object address (None: declined).
struct P4Registry {
    f: &'static P4Ffi,
    vers: std::collections::HashMap<usize, Option<RustVerifier>>,
    mtps: std::collections::HashMap<usize, Option<RustMtp>>,
}

unsafe impl Send for P4Registry {}

static P4_REG: OnceLock<std::sync::Mutex<P4Registry>> = OnceLock::new();

fn put_c_err(msg: &str, err: *mut c_char, cap: usize) {
    if err.is_null() || cap == 0 {
        return;
    }
    let b = msg.as_bytes();
    let n = b.len().min(cap - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr() as *const c_char, err, n);
        *err.add(n) = 0;
    }
}

fn with_reg<R>(f: impl FnOnce(&mut P4Registry) -> R) -> Option<R> {
    let m = P4_REG.get()?;
    let mut g = m.lock().unwrap_or_else(|e| e.into_inner());
    Some(f(&mut g))
}

unsafe extern "C" fn cb_ver_run(
    _user: *mut c_void,
    ver: *mut c_void,
    t: c_int,
    tokens: *const i32,
    pos0: i64,
    pool: *mut c_void,
    pool_user: *mut c_void,
    out: *mut i32,
    spec_q: *const i32,
    spec_nq: c_int,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let slot = reg
                .vers
                .entry(ver as usize)
                .or_insert_with(|| RustVerifier::of_ptr(f, ver));
            let Some(rv) = slot.as_mut() else { return 0 };
            if t < 1 || t as usize > K_VERIFY_MAX_T {
                put_c_err("verify: window size out of range", err, cap);
                return -1;
            }
            let toks = std::slice::from_raw_parts(tokens, t as usize);
            let outs = std::slice::from_raw_parts_mut(out, t as usize);
            let poolf: Option<PoolMultiFn> = if pool.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, PoolMultiFn>(pool))
            };
            let q = if spec_q.is_null() || spec_nq <= 0 {
                None
            } else {
                Some((spec_q, spec_nq))
            };
            match rv.run(t, toks, pos0, poolf, pool_user, outs, q) {
                Ok(()) => 1,
                Err(e) => {
                    put_c_err(&e, err, cap);
                    -1
                }
            }
        })
        .unwrap_or(0)
    }));
    r.unwrap_or_else(|_| {
        put_c_err("verify: the Rust plane panicked", err, cap);
        -1
    })
}

unsafe extern "C" fn cb_ver_commit(
    _user: *mut c_void,
    ver: *mut c_void,
    n_keep: c_int,
    commit_async: c_int,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let slot = reg
                .vers
                .entry(ver as usize)
                .or_insert_with(|| RustVerifier::of_ptr(f, ver));
            let Some(rv) = slot.as_mut() else { return 0 };
            match rv.commit_as(n_keep, commit_async != 0) {
                Ok(()) => 1,
                Err(e) => {
                    put_c_err(&e, err, cap);
                    -1
                }
            }
        })
        .unwrap_or(0)
    }));
    r.unwrap_or_else(|_| {
        put_c_err("verify: the Rust plane panicked", err, cap);
        -1
    })
}

unsafe extern "C" fn cb_mtp_draft(
    _user: *mut c_void,
    mtp: *mut c_void,
    t: c_int,
    tokens: *const i32,
    p: i64,
    a: c_int,
    drafts: *mut i32,
    probs: *mut f32,
    min_p: f32,
    n_drafts: *mut c_int,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let slot = reg
                .mtps
                .entry(mtp as usize)
                .or_insert_with(|| RustMtp::of_ptr(f, mtp));
            let Some(rm) = slot.as_mut() else { return 0 };
            rm.refresh();
            let mt = rm.v.max_t.max(1) as usize;
            if t < 1 || t as usize > mt {
                put_c_err("mtp: draft arguments out of range", err, cap);
                return -1;
            }
            let toks = std::slice::from_raw_parts(tokens, t as usize);
            let d = std::slice::from_raw_parts_mut(drafts, mt - 1);
            let pr = if probs.is_null() {
                None
            } else {
                Some(std::slice::from_raw_parts_mut(probs, mt - 1))
            };
            match rm.draft(t, toks, p, a, d, pr, min_p) {
                Ok(n) => {
                    if !n_drafts.is_null() {
                        *n_drafts = n;
                    }
                    1
                }
                Err(e) => {
                    put_c_err(&e, err, cap);
                    -1
                }
            }
        })
        .unwrap_or(0)
    }));
    r.unwrap_or_else(|_| {
        put_c_err("mtp: the Rust plane panicked", err, cap);
        -1
    })
}

unsafe extern "C" fn cb_mtp_prefill(
    _user: *mut c_void,
    mtp: *mut c_void,
    r_rows: *const f32,
    next_tokens: *const i32,
    n: i64,
    cell0: i64,
    sync: c_int,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let slot = reg
                .mtps
                .entry(mtp as usize)
                .or_insert_with(|| RustMtp::of_ptr(f, mtp));
            let Some(rm) = slot.as_mut() else { return 0 };
            rm.refresh();
            let nt = if n > 0 {
                std::slice::from_raw_parts(next_tokens, n as usize)
            } else {
                &[][..]
            };
            match rm.prefill(r_rows, nt, n, cell0, sync != 0) {
                Ok(()) => 1,
                Err(e) => {
                    put_c_err(&e, err, cap);
                    -1
                }
            }
        })
        .unwrap_or(0)
    }));
    r.unwrap_or_else(|_| {
        put_c_err("mtp: the Rust plane panicked", err, cap);
        -1
    })
}

unsafe extern "C" fn cb_ver_wait_commit(
    _user: *mut c_void,
    ver: *mut c_void,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let slot = reg
                .vers
                .entry(ver as usize)
                .or_insert_with(|| RustVerifier::of_ptr(f, ver));
            let Some(rv) = slot.as_mut() else { return 0 };
            match rv.wait_commit() {
                Ok(()) => 1,
                Err(e) => {
                    put_c_err(&e, err, cap);
                    -1
                }
            }
        })
        .unwrap_or(0)
    }))
    .unwrap_or(-1)
}

unsafe extern "C" fn cb_ver_copy_logits(
    _user: *mut c_void,
    ver: *const c_void,
    t: c_int,
    host: *mut f32,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let ver = ver as *mut c_void;
            let slot = reg
                .vers
                .entry(ver as usize)
                .or_insert_with(|| RustVerifier::of_ptr(f, ver));
            let Some(rv) = slot.as_mut() else { return -1 };
            let nv = rv.vocab();
            if host.is_null() || nv <= 0 {
                return 0;
            }
            let out = std::slice::from_raw_parts_mut(host, nv as usize);
            rv.copy_logits(t, out) as c_int
        })
        .unwrap_or(-1)
    }))
    .unwrap_or(-1)
}

unsafe extern "C" fn cb_mtp_set_draft_sampling(
    _user: *mut c_void,
    mtp: *mut c_void,
    sp: *const c_void,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let slot = reg
                .mtps
                .entry(mtp as usize)
                .or_insert_with(|| RustMtp::of_ptr(f, mtp));
            let Some(rm) = slot.as_mut() else { return 0 };
            rm.refresh();
            if sp.is_null() {
                return 0;
            }
            rm.set_draft_sampling(&*(sp as *const SamplerParams));
            1
        })
        .unwrap_or(0)
    }))
    .unwrap_or(0)
}

unsafe extern "C" fn cb_mtp_set_draft_history(
    _user: *mut c_void,
    mtp: *mut c_void,
    tail: *const i32,
    n_tail: i64,
    next: i32,
) -> c_int {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_reg(|reg| {
            let f = reg.f;
            let slot = reg
                .mtps
                .entry(mtp as usize)
                .or_insert_with(|| RustMtp::of_ptr(f, mtp));
            let Some(rm) = slot.as_mut() else { return 0 };
            rm.refresh();
            let t = if tail.is_null() || n_tail <= 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(tail, n_tail as usize)
            };
            rm.set_draft_history(t, next);
            1
        })
        .unwrap_or(0)
    }))
    .unwrap_or(0)
}

impl Piece4VerifyMtp {
    /// Hand every C++ `Verifier::run` / `commit` and `MtpDrafter::draft` / `prefill` (prompt windows, the drafter's
    /// prompt K/V, `draft_first`, the C++ fallbacks) to the Rust plane.  Called once with TALUS_FN_RUST=1
    /// (engine.rs qwen4exp-p4 hook); TALUS_FN_P4=0 / TALUS_P4_CPP=1 keeps every C++ body.
    pub fn install_plane(kernels: &Qwen4ExpKernels) -> Result<(), String> {
        // qwen4exp-r2v: the objects' setup (init / load / bind / logit bias) first; TALUS_FN_R2V=0 keeps the C++ bodies
        if let Err(e) = setup::install(kernels) {
            eprintln!("talus-qwen4exp: {e}");
        }
        if env_flag("TALUS_P4_CPP")
            || std::env::var("TALUS_FN_P4")
                .map(|v| v.trim() == "0")
                .unwrap_or(false)
        {
            return Ok(());
        }
        let f =
            p4_ffi(kernels).ok_or("qwen4exp-p4: the tk_qwen4exp_p4_* table is not available")?;
        let _ = P4_REG.set(std::sync::Mutex::new(P4Registry {
            f,
            vers: Default::default(),
            mtps: Default::default(),
        }));
        let ops = TkP4Ops {
            user: std::ptr::null_mut(),
            ver_run: Some(cb_ver_run),
            ver_commit: Some(cb_ver_commit),
            mtp_draft: Some(cb_mtp_draft),
            mtp_prefill: Some(cb_mtp_prefill),
            ver_wait_commit: Some(cb_ver_wait_commit),
            ver_copy_logits: Some(cb_ver_copy_logits),
            mtp_set_draft_sampling: Some(cb_mtp_set_draft_sampling),
            mtp_set_draft_history: Some(cb_mtp_set_draft_history),
        };
        unsafe { (f.install)(&ops) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_mtp_defaults_use_copy_kernel_pcie_mode() {
        let cfg = Qwen4ExpModelConfig::default();
        let vm = Piece4VerifyMtp::new(&cfg);
        assert_eq!(vm.pcie_mode, 2);
        assert_eq!(vm.windows_verified, 0);
    }

    #[test]
    fn struct_layouts_match_cpp() {
        // x86-64 SysV sizes of strata::kernels::SamplerParams and strata::core::GpuPlanSink
        assert_eq!(std::mem::size_of::<SamplerParams>(), 80);
        assert_eq!(std::mem::size_of::<GpuPlanSink>(), 112);
        assert_eq!(std::mem::offset_of!(SamplerParams, seed), 40);
        assert_eq!(std::mem::offset_of!(SamplerParams, greedy), 56);
        assert_eq!(std::mem::offset_of!(SamplerParams, logit_bias), 64);
        assert_eq!(std::mem::offset_of!(GpuPlanSink, staging), 56);
        assert_eq!(std::mem::offset_of!(GpuPlanSink, pcie_mode), 104);
    }

    #[test]
    fn step_fill_matches_qsa_cu() {
        let mut s = [0i32; 4];
        qsa_step_fill(&mut s, 0, 4, 2048);
        assert_eq!(s, [0, 1, 0, 1]);
        qsa_step_fill(&mut s, 2049, 4, 2048);
        assert_eq!(s, [2049, 2050, 512, 2050]);
        qsa_step_fill(&mut s, 2050, 4, 2048);
        assert_eq!(s, [2050, 2051, 512, 2051]);
        qsa_step_fill(&mut s, 99999, 4, 2048);
        assert_eq!(s, [99999, 100000, 25000, 2051]);
    }

    #[test]
    fn penalty_rows_matches_sampler_hpp() {
        // tail [1 2 3], window [10 11 12], h = 4
        let mut out = vec![0i32; 3 * 4];
        penalty_rows(&[1, 2, 3], &[10, 11, 12], 3, 4, &mut out);
        assert_eq!(out, vec![1, 2, 3, 10, 2, 3, 10, 11, 3, 10, 11, 12]);
        // short history: -1 padded in front
        let mut out2 = vec![0i32; 2 * 4];
        penalty_rows(&[7], &[8, 9], 2, 4, &mut out2);
        assert_eq!(out2, vec![-1, -1, 7, 8, -1, 7, 8, 9]);
        // coupled_hist_base = row 0 of the one-token window [next]
        let mut base = vec![0i32; 3];
        penalty_rows(&[4, 5, 6, 7], &[9], 1, 3, &mut base);
        assert_eq!(base, vec![6, 7, 9]);
    }

    #[test]
    fn coupled_hist_len_caps() {
        assert_eq!(coupled_hist_len(0, 4096), 0);
        assert_eq!(coupled_hist_len(-3, 4096), 0);
        assert_eq!(coupled_hist_len(64, 4096), 64);
        assert_eq!(coupled_hist_len(10000, 4096), 4096);
    }

    #[test]
    fn plan_slot_offsets_match_verify_init() {
        // T = 7, k = 10: cap 70
        let cap = 70i64;
        let (start, dst, tok, ptr_off, start2) = plan_slot_offsets(cap);
        assert_eq!(start, 4);
        assert_eq!(dst, 4 + 71);
        assert_eq!(tok, 4 + 71 + 70);
        let i32n = 4 + 71 + 70 + 70;
        assert_eq!(ptr_off, (i32n + 1) & !1);
        assert_eq!(ptr_off % 2, 0);
        assert_eq!(start2, ptr_off + 4 * cap);
    }

    #[test]
    fn accepted_prefix_counts_matching_drafts() {
        assert_eq!(accepted_prefix(&[5, 6, 7, 8], &[6, 7, 9, 1], 4), 2);
        assert_eq!(accepted_prefix(&[5, 6], &[1, 2], 2), 0);
        assert_eq!(accepted_prefix(&[5], &[1], 1), 0);
        assert_eq!(accepted_prefix(&[5, 6, 7], &[6, 7, 3], 3), 2);
    }

    #[test]
    fn nonfinite_mask_and_inject_list() {
        let mut f = vec![0u32; 8 * 16];
        f[16 + 3] = 1;
        f[4 * 16] = 7;
        assert_eq!(nonfinite_mask(&f, 3, 16, 8), 0b010);
        assert_eq!(nonfinite_mask(&f, 5, 16, 8), 0b10010);
        assert_eq!(parse_nan_inject("3,7"), vec![3, 7]);
        assert_eq!(parse_nan_inject("0,2"), vec![2]);
        assert_eq!(parse_nan_inject("x"), Vec::<i64>::new());
    }

    #[test]
    fn logprob_line_matches_c_format() {
        let row = [0.0f32, 1.0, 2.0, -1.0];
        let line = window_logprob_line(&row, 10, 1, -1, 0).unwrap();
        let parts: Vec<&str> = line.split('\t').collect();
        assert_eq!(parts[0], "10");
        assert_eq!(parts[1], "1");
        assert_eq!(parts[3], "2");
        assert_eq!(parts[5], "0");
        assert_eq!(parts[6], "nan");
        assert_eq!(parts[7], "nan");
        let lse = 2.0f64 + ((-2.0f64).exp() + (-1.0f64).exp() + 1.0 + (-3.0f64).exp()).ln();
        assert_eq!(parts[2], format!("{:.9}", 1.0 - lse));
        let with_extra = window_logprob_line(&row, 0, 1, 2, 2).unwrap();
        let p: Vec<&str> = with_extra.split('\t').collect();
        assert_eq!(p.len(), 10);
        assert!(p[8].starts_with("2:"));
        assert!(p[9].starts_with("1:"));
        assert!(window_logprob_line(&row, 0, 9, -1, 0).is_none());
    }
}
