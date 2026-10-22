//! qwen4exp-r2: the prompt path's buffer setup in Rust - `src/prefill/prefill.cpp`'s sizing statics
//! (`Prefill::bytes_needed` / `_owned` / `_no_ring`, `ring_slots_for`, `ring_default_slots`, `ring_max_slots`,
//! `ring_cap_for`, `stream_all_min_tokens`, `ring_bytes_enabled`, `arm_cpu_share` and the state their setters keep),
//! `Prefill::carve`, `Prefill::relayout` and `Prefill::init`'s device half (the checks, the token buffer, the copy
//! stream, the ring events, the mapped grouping table, the identity page table, the GEMM scratch and the carve).
//!
//! The C++ statics and `Prefill::init` / `relayout` dispatch here once [`install`] registered the hooks
//! (`tk_qwen4exp_r2_pf_install`, `TALUS_FN_RUST=1`; `TALUS_FN_R2=0` keeps the C++ code). The C++ object adopts what
//! this file laid out (`tk_qwen4exp_r2_pf_adopt_init` / `_adopt_bufs`) and keeps its private host machinery: the
//! Stager threads, the pinned uploads, the host vectors and the PLE host buffers (`tk_qwen4exp_r2_pf_init_host`).
//! A layer-split stage, a stage helper or a peer GPU are declined (the C++ init runs).
//!
//! `TALUS_FN_R2_CHECK=1`: every sizing call is compared with the C++ statics (hooks bypassed) and every carve with
//! the C++ carve on the same (fake) base, field for field; mismatches are printed (`qwen4exp-r2 CHECK`).

use core::ffi::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::{c_atoi, c_atoll, env_first_is, env_str, write_err};
use crate::qwen4exp::ffi::{load_sym_from, TkP2ExpertFmt, TkP2KvPools, TkP2Layout};

// prefill.cpp's constants (strata::arch::qwen4exp caps); init checks the model against them
const N: u64 = 2560;
const HC: u64 = 4;
const D: u64 = N * HC;
const LR: u64 = 320;
const K: u64 = 10;
const NE: i64 = 512;
const C: u64 = 10240;
const ZV: u64 = 6144;
const HV: u64 = 48;
const FF: u64 = 640;
const QH: u64 = 24;
const QKV: u64 = 2;
const HD: u64 = 256;
const IDXH: u64 = 4;
const IDXD: u64 = 128;
const EPS: f32 = 1e-6;
const STAGE: i64 = 8;
const RING_MAX: i64 = 1024;
const DQ: usize = 2;
const MMQ_GROUP: u64 = 16;
const MMQ_TAIL: u64 = 4096;
const XN_PAD: u64 = 64;
const ZV_PAD: u64 = 64;
const Q2_0_BLOB: u64 = 1_382_400;
const SEL_BATCH: i64 = 256;
const ATTN_BATCH: i64 = 32;
const GGML_Q2_0: i32 = 42;

// ============================================================================================ C ABI mirrors

type Vp = *mut c_void;
type Cvp = *const c_void;

#[repr(C)]
pub struct TkR2PfHooks {
    pub user: Vp,
    pub bytes_needed: unsafe extern "C" fn(Vp, Cvp, Cvp, i64, i32, i32) -> u64,
    pub bytes_needed_no_ring: unsafe extern "C" fn(Vp, Cvp, Cvp, i64, i32) -> u64,
    pub ring_slots_for: unsafe extern "C" fn(Vp, i64) -> i64,
    pub ring_default_slots: unsafe extern "C" fn(Vp) -> i64,
    pub ring_max_slots: unsafe extern "C" fn(Vp) -> i64,
    pub ring_cap_for: unsafe extern "C" fn(Vp, i64) -> i64,
    pub stream_all_min: unsafe extern "C" fn(Vp) -> i64,
    pub ring_bytes_enabled: unsafe extern "C" fn(Vp) -> i32,
    pub pinned_share: unsafe extern "C" fn(Vp) -> f64,
    pub set_pinned_share: unsafe extern "C" fn(Vp, f64),
    pub set_ring_override: unsafe extern "C" fn(Vp, i32),
    pub set_ring_budget: unsafe extern "C" fn(Vp, i32, i64),
    pub arm_cpu_share: unsafe extern "C" fn(Vp, i32, i32),
    #[allow(clippy::type_complexity)]
    pub init: unsafe extern "C" fn(
        Vp,
        Vp,
        Cvp,
        Cvp,
        Vp,
        Vp,
        Cvp,
        *const i32,
        i64,
        Vp,
        Vp,
        u64,
        i32,
        *mut c_char,
        usize,
    ) -> c_int,
    pub relayout: unsafe extern "C" fn(Vp, Vp, i64, Vp, u64, *mut c_char, usize) -> c_int,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct TkR2PfFacts {
    pub is_hip: i32,
    pub layout_native: i32,
    pub fused_enabled: i32,
    pub peer_portable: i32,
    pub mmq_built: i32,
    pub prompt_f16: i32,
    pub pad0: i32,
    pub pad1: i32,
    pub lay_n_layers: i64,
    pub n_fmt: i64,
    pub max_blob: i64,
    pub ng_hc_dim: i64,
    pub step_count: i64,
    pub ring_max: i64,
    pub gemm_scratch_elems: i64,
    pub gemm_ws_bytes: i64,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct TkR2PfSession {
    pub n_layers: i64,
    pub n_expert: i64,
    pub n_embd: i64,
    pub hc: i64,
    pub hc_lr: i64,
    pub n_ff: i64,
    pub n_ff_shexp: i64,
    pub conv_ch: i64,
    pub value_dim: i64,
    pub v_heads: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub head_dim_v: i64,
    pub idx_heads: i64,
    pub idx_dim: i64,
    pub k: i64,
    pub rms_eps: f64,
    pub cap: i64,
    pub max_cells: i64,
    pub idx_block: i64,
    pub attn_scratch_floats: i64,
    pub n_pages: i64,
    pub page_size: i64,
    pub kv_q4_bytes_per_head: i64,
    pub kv_mode: i32,
    pub kv_hybrid: i32,
    pub kv_q4: i32,
    pub kv_int8: i32,
}

#[repr(C)]
pub struct TkR2PfBufs {
    pub t: i64,
    pub cap: i64,
    pub max_blocks: i64,
    pub ring: i32,
    pub fused_bufs: i32,
    pub region: *mut u8,
    pub region_bytes: u64,
    pub emb: *mut u8,
    pub r: *mut u8,
    pub xn: *mut u8,
    pub grs: *mut u8,
    pub xn16: *mut u8,
    pub lo: *mut u8,
    pub lo16: *mut u8,
    pub gated: *mut u8,
    pub inj: *mut u8,
    pub mixed: *mut u8,
    pub mixed_bf: *mut u8,
    pub mixed_h: *mut u8,
    pub bo: *mut u8,
    pub xn16_lo: *mut u8,
    pub lo16_lo: *mut u8,
    pub mixed_bf_lo: *mut u8,
    pub steps_dev: *mut u8,
    pub qkv: *mut u8,
    pub z: *mut u8,
    pub ab: *mut u8,
    pub gate: *mut u8,
    pub beta: *mut u8,
    pub hbuf: *mut u8,
    pub y: *mut u8,
    pub y_h: *mut u8,
    pub kc: *mut u8,
    pub vc: *mut u8,
    pub qf: *mut u8,
    pub q: *mut u8,
    pub idx_raw: *mut u8,
    pub q_idx: *mut u8,
    pub attn: *mut u8,
    pub attn_h: *mut u8,
    pub sel_ids: *mut u8,
    pub sel_scores: *mut u8,
    pub attn_scratch: *mut u8,
    pub logits: *mut u8,
    pub w: *mut u8,
    pub ids: *mut u8,
    pub slot_dev: *mut u8,
    pub src_dev: *mut u8,
    pub xs: *mut u8,
    pub gu: *mut u8,
    pub hh: *mut u8,
    pub dm: *mut u8,
    pub sgate: *mut u8,
    pub sup: *mut u8,
    pub sh_h: *mut u8,
    pub shared: *mut u8,
    pub sg: *mut u8,
    pub xq: *mut u8,
    pub h: *mut u8,
    pub hq: *mut u8,
    pub xscale: *mut u8,
    pub hscale: *mut u8,
    pub grp_tail: *mut u8,
    pub row_sd: *mut u8,
    pub dq_gu: [*mut u8; 2],
    pub dq_d: [*mut u8; 2],
    pub ids_identity: *mut u8,
    pub bounds_dev: *mut u8,
    pub grp_gu: *mut u8,
    pub grp_d: *mut u8,
    pub stage_dev: *const *mut u8,
    pub ple_emb: *mut u8,
    pub ple_norm: *mut u8,
    pub stage: TkP2KvPools,
    pub owned: *const Vp,
    pub n_owned: i64,
}

#[repr(C)]
pub struct TkR2PfInit {
    pub t_max: i64,
    pub borrowed: i32,
    pub borrow_spill: i32,
    pub device: i32,
    pub n_events: i32,
    pub wt: Cvp,
    pub g: Cvp,
    pub ss: Vp,
    pub src: Vp,
    pub cache: Cvp,
    pub host_res: *const i32,
    pub cs: Vp,
    pub copy: Vp,
    pub copied: *const Vp,
    pub used: *const Vp,
    pub tok_dev: *mut i32,
    pub ident_table: *mut i32,
    pub grp_host: *mut i32,
    pub grp_dev: *mut i32,
    pub grp_n: u64,
    pub grp_tk: u64,
    pub owned: *const Vp,
    pub n_owned: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TkR2PfState {
    pub borrowed: i32,
    pub borrow_spill: i32,
    pub has_src: i32,
    pub f16_io: i32,
    pub t: i64,
    pub t_max: i64,
    pub cs: Vp,
    pub copy: Vp,
    pub kv_copy: Vp,
    pub g: Cvp,
    pub ss: Cvp,
}

macro_rules! r2_ffi {
    ($( $field:ident = $sym:literal : fn($($ty:ty),*) $(-> $ret:ty)?; )*) => {
        struct R2Ffi { $( $field: unsafe extern "C" fn($($ty),*) $(-> $ret)?, )* }
        unsafe impl Send for R2Ffi {}
        unsafe impl Sync for R2Ffi {}
        impl R2Ffi {
            fn load() -> Result<Self, String> {
                unsafe { Ok(Self { $( $field: load_sym_from(std::ptr::null_mut(), $sym)?, )* }) }
            }
        }
    };
}

r2_ffi! {
    install = "tk_qwen4exp_r2_pf_install": fn(*const TkR2PfHooks);
    facts = "tk_qwen4exp_r2_pf_facts": fn(*mut TkR2PfFacts);
    session = "tk_qwen4exp_r2_pf_session": fn(Cvp, Cvp, *mut TkR2PfSession);
    group_bytes = "tk_qwen4exp_r2_fused_group_bytes": fn(i64, i32) -> u64;
    act_bytes = "tk_qwen4exp_r2_fused_act_bytes": fn(i64, i64) -> u64;
    malloc = "tk_qwen4exp_r2_malloc": fn(*mut Vp, u64) -> c_int;
    free = "tk_qwen4exp_r2_free": fn(Vp);
    memcpy_h2d_sync = "tk_qwen4exp_r2_memcpy_h2d_sync": fn(Vp, Cvp, u64) -> c_int;
    host_alloc_mapped = "tk_qwen4exp_r2_host_alloc_mapped": fn(*mut Vp, *mut Vp, u64, i32) -> c_int;
    stream_create = "tk_qwen4exp_r2_stream_create": fn(*mut Vp) -> c_int;
    get_device = "tk_qwen4exp_r2_get_device": fn(*mut i32) -> c_int;
    mem_info = "tk_qwen4exp_r2_mem_info": fn(*mut u64, *mut u64);
    clear_error = "tk_qwen4exp_r2_clear_error": fn();
    stage_ok = "tk_qwen4exp_r2_pf_stage_ok": fn(Vp, i64) -> c_int;
    adopt_init = "tk_qwen4exp_r2_pf_adopt_init": fn(Vp, *const TkR2PfInit) -> c_int;
    init_host = "tk_qwen4exp_r2_pf_init_host": fn(Vp, i64) -> c_int;
    gemm_init = "tk_qwen4exp_r2_pf_gemm_init": fn(Vp, Vp, *mut u16, i64, Vp, u64, *mut i32, *mut c_char, usize) -> c_int;
    gemm_rebind = "tk_qwen4exp_r2_pf_gemm_rebind": fn(Vp, *mut u16, i64, Vp, u64);
    adopt_bufs = "tk_qwen4exp_r2_pf_adopt_bufs": fn(Vp, *const TkR2PfBufs) -> c_int;
    state = "tk_qwen4exp_r2_pf_state": fn(Vp, *mut TkR2PfState) -> c_int;
    cpp_statics = "tk_qwen4exp_r2_pf_cpp_statics": fn(Cvp, Cvp, i64, i32, *mut u64);
    cpp_carve = "tk_qwen4exp_r2_pf_cpp_carve": fn(Vp, i64, *mut u8, u64, *mut TkP2Layout) -> c_int;
    // qwen4exp-p2's entries (the runtime calls and the layer formats it already exports)
    event_create = "tk_qwen4exp_p2_event_create": fn(*mut Vp, i32) -> c_int;
    stream_sync = "tk_qwen4exp_p2_stream_sync": fn(Vp) -> c_int;
    expert_fmt = "tk_qwen4exp_p2_expert_fmt": fn(i64, *mut TkP2ExpertFmt) -> c_int;
    fused_native_supported = "tk_qwen4exp_p2_fused_native_supported": fn(i32, i32) -> c_int;
    mmq_fits = "tk_qwen4exp_p2_mmq_fits": fn(i32, i64) -> c_int;
    mmq_matrix_bytes = "tk_qwen4exp_p2_mmq_matrix_bytes": fn(i32, i64, i64) -> u64;
    mmq_q8_bytes = "tk_qwen4exp_p2_mmq_q8_bytes": fn(i64, i64) -> u64;
}

static FFI: OnceLock<Option<R2Ffi>> = OnceLock::new();
fn ffi() -> &'static R2Ffi {
    FFI.get()
        .and_then(|o| o.as_ref())
        .expect("qwen4exp-r2: used before install")
}

// ============================================================================================ environment

/// prefill.cpp's function-local statics (read once, on the first sizing / carve call - never at install: generate.cpp
/// applies the arch defaults (src/core/arch_defaults.cpp: STRATA_PF_PAD, STRATA_HCD_EXACT on gfx1151) and
/// STRATA_PF_HCDOWN=auto with setenv after the Rust engine is built) and the per-call getenv for STRATA_PREFILL_RING.
struct Env {
    stream_min: i64,         // STRATA_PREFILL_STREAM_MIN (-1: unset)
    cpu_share_explicit: f64, // -2 unset, -1 auto, else [0, 1]
    cpu_share_max: i64,
    cpu_share_max_explicit: bool,
    hip_wmma: bool,
    ring_bytes_on: bool,
    emb_reuse_account: bool,
    gr_unfused: bool,
    pf_pad: bool,
    bf16x2_env: i32,
    stage_own: bool,
    mmq_env_on: bool,
    group_copy: bool,
    force_pageable: bool,
    check: bool,
}

fn env() -> &'static Env {
    static E: OnceLock<Env> = OnceLock::new();
    E.get_or_init(|| {
        let cpu_share_explicit = match env_str("STRATA_PREFILL_CPU_SHARE") {
            None => -2.0,
            Some(v) if v == "auto" => -1.0,
            Some(v) => c_atof(&v).clamp(0.0, 1.0),
        };
        Env {
            stream_min: env_str("STRATA_PREFILL_STREAM_MIN")
                .map(|v| c_atoll(&v))
                .unwrap_or(-1),
            cpu_share_explicit,
            cpu_share_max: env_str("STRATA_PREFILL_CPU_SHARE_MAX")
                .map(|v| c_atoll(&v).max(1024))
                .unwrap_or(3072),
            cpu_share_max_explicit: env_str("STRATA_PREFILL_CPU_SHARE_MAX").is_some(),
            hip_wmma: env_first_is("STRATA_HIP_WMMA", b'1'),
            ring_bytes_on: env_str("STRATA_RING_BYTES")
                .map(|v| v.as_bytes().first() != Some(&b'0'))
                .unwrap_or(true),
            emb_reuse_account: env_first_is("STRATA_EMB_REUSE_ACCOUNT", b'1'),
            gr_unfused: env_first_is("STRATA_GR_UNFUSED", b'1'),
            pf_pad: env_first_is("STRATA_PF_PAD", b'1') && env_first_is("STRATA_PF_GEMM", b'1'),
            bf16x2_env: env_str("STRATA_PREFILL_BF16X2")
                .map(|v| c_atoi(&v))
                .unwrap_or(0),
            stage_own: env_str("STRATA_KV_STAGE_OWN").is_some(),
            mmq_env_on: env_str("STRATA_PREFILL_MMQ")
                .map(|v| c_atoi(&v) != 0)
                .unwrap_or(true),
            group_copy: env_first_is("STRATA_GROUP_COPY", b'1'),
            force_pageable: env_first_is("STRATA_TEST_PAGEABLE", b'1'),
            check: env_first_is("TALUS_FN_R2_CHECK", b'1'),
        }
    })
}

/// rb-num (opt-in): `TALUS_FN_HC_BF16` = "1" (BF16) / "2" (FP16) keeps the prompt path's 4-stream residual R in 16 bits,
/// in the `xn` slot (unused by the fused path; T x 10240 x 2 bytes); 0 = off (unset, any other value). prefill.cpp's
/// `fn_hc16()` parses the same way (its own carve, the TALUS_FN_RUST=0 / r2-off arms).
pub(crate) fn hc16_fmt() -> i32 {
    static V: OnceLock<i32> = OnceLock::new();
    *V.get_or_init(|| match env_str("TALUS_FN_HC_BF16").as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

/// `hc_pad_alloc()`: STRATA_PF_HCDOWN (0 off, 1 forced, 2 auto) or STRATA_HCD_EXACT. Its own once-cell, first read on
/// the first sizing / carve (as prefill.cpp's static): generate.cpp sets STRATA_PF_HCDOWN=auto on gfx1151 at startup,
/// after this file's other switches were read at install.
fn hc_pad_alloc() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        let mode = match env_str("STRATA_PF_HCDOWN") {
            None => 0,
            Some(v) if v.is_empty() || v.as_bytes()[0] == b'0' => 0,
            Some(v) if v.as_bytes()[0] == b'1' => 1,
            Some(_) => 2,
        };
        mode != 0 || env_first_is("STRATA_HCD_EXACT", b'1')
    })
}

/// C's atof for the share values ("0.5", "1", "auto" handled by the caller): the leading number, 0 for none.
fn c_atof(s: &str) -> f64 {
    let t = s.trim_start();
    let mut end = 0;
    let b = t.as_bytes();
    let mut seen_dot = false;
    let mut seen_e = false;
    while end < b.len() {
        let c = b[end];
        let ok = c.is_ascii_digit()
            || ((c == b'+' || c == b'-') && (end == 0 || matches!(b[end - 1], b'e' | b'E')))
            || (c == b'.' && !seen_dot && !seen_e)
            || ((c == b'e' || c == b'E') && !seen_e && end > 0);
        if !ok {
            break;
        }
        seen_dot |= c == b'.';
        seen_e |= c == b'e' || c == b'E';
        end += 1;
    }
    let mut cand = &t[..end];
    loop {
        if cand.is_empty() {
            return 0.0;
        }
        if let Ok(v) = cand.parse::<f64>() {
            return v;
        }
        cand = &cand[..cand.len() - 1];
    }
}

// ============================================================================================ state and rules

/// What prefill.cpp's setters keep (g_pinned_share, g_ring_override, g_ring_budget, g_ring_small_max,
/// g_stream_min_share, g_share_default).
struct Rules {
    pinned_share: f64,
    ring_override: i32,
    ring_budget: i32,
    ring_small_max: i64,
    stream_min_share: i64,
    share_default: bool,
}
static RULES: Mutex<Rules> = Mutex::new(Rules {
    pinned_share: 1.0,
    ring_override: 0,
    ring_budget: 0,
    ring_small_max: 0,
    stream_min_share: 0,
    share_default: false,
});
fn rules() -> std::sync::MutexGuard<'static, Rules> {
    RULES.lock().unwrap_or_else(|p| p.into_inner())
}

fn facts() -> TkR2PfFacts {
    let mut f = TkR2PfFacts::default();
    unsafe { (ffi().facts)(&mut f) };
    f
}

fn session(g: Cvp, ss: Cvp) -> TkR2PfSession {
    let mut s = TkR2PfSession::default();
    unsafe { (ffi().session)(g, ss, &mut s) };
    s
}

fn stream_all_min() -> i64 {
    let e = env();
    if e.stream_min >= 0 {
        return e.stream_min;
    }
    let share = rules().stream_min_share;
    if share > 0 {
        share
    } else {
        1024
    }
}

fn cpu_share_env() -> f64 {
    let v = env().cpu_share_explicit;
    if v == -2.0 {
        if rules().share_default {
            -1.0
        } else {
            0.0
        }
    } else {
        v
    }
}
fn cpu_share_on() -> bool {
    cpu_share_env() != 0.0
}
fn cpu_share_max() -> i64 {
    let e = env();
    if rules().share_default && e.cpu_share_explicit == -2.0 && !e.cpu_share_max_explicit {
        return 1024;
    }
    e.cpu_share_max
}

/// `fused_ring()`'s "every layer's formats are covered" - a function-local static in C++ (computed on first use).
fn all_layers_native_fused() -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    *ALL.get_or_init(|| {
        let f = facts();
        if f.n_fmt <= 0 {
            return false;
        }
        (0..f.n_fmt).all(|l| {
            let mut fm = TkP2ExpertFmt::default();
            unsafe { (ffi().expert_fmt)(l, &mut fm) };
            unsafe { (ffi().fused_native_supported)(fm.gu_type, fm.d_type) != 0 }
        })
    })
}

fn fused_ring(f: &TkR2PfFacts) -> bool {
    if f.peer_portable != 0 {
        return false;
    }
    if f.layout_native == 0 {
        return f.fused_enabled != 0;
    }
    all_layers_native_fused()
}

fn ring_cap(f: &TkR2PfFacts) -> i64 {
    if fused_ring(f) && f.layout_native == 0 {
        RING_MAX
    } else {
        512
    }
}

fn ring_bytes(f: &TkR2PfFacts, pinned_share: f64) -> u64 {
    let slots: u64 = if fused_ring(f) { 1024 } else { 384 };
    (if pinned_share >= 0.9 {
        slots
    } else {
        slots / 4
    }) * Q2_0_BLOB
}

fn ring_budget_slots(f: &TkR2PfFacts, pinned_share: f64) -> i64 {
    let per = f.max_blob;
    let n = if per > 0 {
        (ring_bytes(f, pinned_share) / per as u64) as i64
    } else {
        0
    };
    let cap = ring_cap(f);
    if n <= 0 {
        0
    } else if n > cap {
        cap
    } else {
        n
    }
}

fn ring_slots(f: &TkR2PfFacts, t: i64) -> i64 {
    let v = env_str("STRATA_PREFILL_RING");
    let (pinned, override_, budget_slots, small_max) = {
        let r = rules();
        (
            r.pinned_share,
            r.ring_override,
            r.ring_budget,
            r.ring_small_max,
        )
    };
    if f.is_hip != 0 && v.is_none() && override_ <= 0 && env().hip_wmma {
        return if t >= stream_all_min() { 96 } else { STAGE };
    }
    let pinned_ring: i64 = if pinned >= 0.9 {
        if fused_ring(f) {
            1024
        } else {
            384
        }
    } else {
        96
    };
    let budget = env().ring_bytes_on && budget_slots > 0 && t > small_max;
    let r: i64 = match &v {
        Some(s) => c_atoi(s) as i64,
        None if override_ > 0 => override_ as i64,
        None if budget => budget_slots as i64,
        None => pinned_ring,
    };
    if v.is_some() && r == STAGE {
        return STAGE;
    }
    let cap = ring_cap(f);
    let big = if r < 16 {
        16
    } else if r > cap {
        cap
    } else {
        r
    };
    if t >= stream_all_min() {
        big
    } else {
        STAGE
    }
}

fn ring_default_slots(f: &TkR2PfFacts) -> i64 {
    let r: i64 = if rules().pinned_share >= 0.9 {
        if fused_ring(f) {
            1024
        } else {
            384
        }
    } else {
        96
    };
    r.min(ring_cap(f))
}
fn ring_max_slots(f: &TkR2PfFacts) -> i64 {
    let pinned = rules().pinned_share;
    if pinned >= 0.9 {
        ring_budget_slots(f, pinned)
    } else {
        96.min(ring_cap(f))
    }
}
fn ring_cap_for(f: &TkR2PfFacts, old_chunk: i64) -> i64 {
    const KEEP_RING_CHUNK: i64 = 6144;
    if old_chunk >= KEEP_RING_CHUNK {
        ring_default_slots(f)
    } else {
        ring_max_slots(f)
    }
}

/// `mmq_plan()`: which layers' experts go through MMQ, whether any keeps the FP16 path, the group slot sizes
/// (a function-local static in C++: computed on first use).
struct MmqPlan {
    any: bool,
    fallback: bool,
    gu_max: u64,
    d_max: u64,
}
fn mmq_plan() -> &'static MmqPlan {
    static PLAN: OnceLock<MmqPlan> = OnceLock::new();
    PLAN.get_or_init(|| {
        let f = facts();
        let x = ffi();
        let on = f.mmq_built != 0 && env().mmq_env_on;
        let layers = if f.layout_native != 0 {
            f.n_fmt
        } else {
            f.lay_n_layers
        };
        let mut p = MmqPlan {
            any: false,
            fallback: !on || layers <= 0,
            gu_max: 0,
            d_max: 0,
        };
        let mut l = 0;
        while on && l < layers {
            let (gt, dt) = if f.layout_native != 0 {
                let mut fm = TkP2ExpertFmt::default();
                unsafe { (x.expert_fmt)(l, &mut fm) };
                (fm.gu_type, fm.d_type)
            } else {
                (GGML_Q2_0, GGML_Q2_0)
            };
            let fits = unsafe {
                (x.mmq_fits)(gt, (2 * FF) as i64) != 0 && (x.mmq_fits)(dt, N as i64) != 0
            };
            if !fits {
                p.fallback = true;
                if f.layout_native != 0 && unsafe { (x.fused_native_supported)(gt, dt) } != 0 {
                    p.any = true;
                }
                l += 1;
                continue;
            }
            p.any = true;
            p.gu_max = p
                .gu_max
                .max(unsafe { (x.mmq_matrix_bytes)(gt, (2 * FF) as i64, N as i64) });
            p.d_max = p
                .d_max
                .max(unsafe { (x.mmq_matrix_bytes)(dt, N as i64, FF as i64) });
            l += 1;
        }
        p
    })
}

fn fused_layout(f: &TkR2PfFacts, t: i64, src: bool) -> bool {
    src && fused_ring(f) && mmq_plan().any && ring_slots(f, t) > STAGE && t >= stream_all_min()
}

/// `moe_bufs`: GU and H in floats, Xq and Hq in bytes
struct MoeBufs {
    gu: u64,
    h: u64,
    xq: u64,
    hq: u64,
}
fn moe_bufs(t: u64, n_expert: i64, fused: bool) -> MoeBufs {
    let x = ffi();
    let q8 = |rows: u64, cols: u64| unsafe { (x.mmq_q8_bytes)(rows as i64, cols as i64) };
    if !fused {
        return MoeBufs {
            gu: t * K * 2 * FF,
            h: t * K * FF,
            xq: q8(t * K, N),
            hq: q8(t * K, FF),
        };
    }
    let ts = (t as i64).min(stream_all_min() - 1) as u64;
    let gb = unsafe { (x.group_bytes)((t * K) as i64, n_expert as i32) };
    let ab_h = unsafe { (x.act_bytes)((t * K) as i64, FF as i64) };
    let ab_x = unsafe { (x.act_bytes)(t as i64, N as i64) };
    MoeBufs {
        gu: (ts * K * 2 * FF).max((gb + 3) / 4),
        h: (ts * K * FF).max((ab_h + 3) / 4),
        xq: q8(ts * K, N).max(ab_x),
        hq: q8(ts * K, FF),
    }
}

// ============================================================================================ the allocator

/// prefill.cpp's `Alloc`: a bump allocation from a borrowed region, cudaMalloc per take without one (owned
/// buffers, or a spilled region), or a count (`count_only`, each take rounded to `granule`).
struct Alloc {
    base: *mut u8,
    cap: u64,
    used: u64,
    failed_bytes: u64,
    granule: u64,
    count_only: bool,
    spill: bool,
    spilled: bool,
    ok: bool,
    owned: Vec<Vp>,
}
impl Alloc {
    fn count(granule: u64) -> Self {
        Self::region(std::ptr::null_mut(), 0, false, true, granule)
    }
    fn region(base: *mut u8, cap: u64, spill: bool, count_only: bool, granule: u64) -> Self {
        Alloc {
            base,
            cap,
            used: 0,
            failed_bytes: 0,
            granule,
            count_only,
            spill,
            spilled: false,
            ok: true,
            owned: Vec::new(),
        }
    }
    /// `take<T>(n)`: `bytes` = n * sizeof(T)
    fn take(&mut self, raw: u64) -> *mut u8 {
        let mut bytes = (raw + 256 + 255) & !255u64;
        if self.count_only {
            if self.granule > 0 {
                bytes = bytes.div_ceil(self.granule) * self.granule;
            }
            self.used += bytes;
            return std::ptr::null_mut();
        }
        if !self.base.is_null() && !self.spilled && self.used + bytes <= self.cap {
            let p = unsafe { self.base.add(self.used as usize) };
            self.used += bytes;
            return p;
        }
        if !self.base.is_null() && !self.spill {
            self.ok = false;
            self.failed_bytes = bytes;
            return std::ptr::null_mut();
        }
        let mut p: Vp = std::ptr::null_mut();
        if unsafe { (ffi().malloc)(&mut p, bytes) } == 0 {
            self.ok = false;
            self.failed_bytes = bytes;
            return std::ptr::null_mut();
        }
        self.spilled = true;
        self.owned.push(p);
        self.used += bytes;
        p.cast()
    }
    fn f32(&mut self, n: u64) -> *mut u8 {
        self.take(n * 4)
    }
    fn u16(&mut self, n: u64) -> *mut u8 {
        self.take(n * 2)
    }
    fn i32(&mut self, n: u64) -> *mut u8 {
        self.take(n * 4)
    }
    fn u8(&mut self, n: u64) -> *mut u8 {
        self.take(n)
    }
}

/// The inputs of a carve / count that come from the object or the session
struct CarveCtx {
    f: TkR2PfFacts,
    s: TkR2PfSession,
    src: bool,
    f16_io: bool,
}
impl CarveCtx {
    fn bf16x2_mode(&self) -> i32 {
        if self.f16_io {
            0
        } else {
            env().bf16x2_env
        }
    }
    fn max_blocks(&self) -> i64 {
        self.s.max_cells / self.s.idx_block + 2
    }
}

fn gdn_set_bytes(t: u64) -> u64 {
    let mut a = Alloc::count(0);
    a.f32(t * C);
    a.f32(t * ZV);
    a.f32(t * 2 * HV);
    a.f32(t * HV);
    a.f32(t * HV);
    a.f32(t * C);
    a.f32(t * ZV);
    a.u16(t * (ZV + if env().pf_pad { ZV_PAD } else { 0 }));
    a.used
}
fn qsa_set_bytes(t: u64, cap: i64, max_blocks: i64, attn_scratch_floats: i64) -> u64 {
    let mut a = Alloc::count(0);
    a.f32(t * QKV * HD);
    a.f32(t * QKV * HD);
    a.f32(t * 2 * QH * HD);
    a.f32(t * QH * HD);
    a.f32(t * IDXD);
    a.f32(t * IDXH * IDXD);
    a.f32(t * QH * HD);
    a.u16(t * (QH * HD + if env().pf_pad { ZV_PAD } else { 0 }));
    a.i32(t * cap as u64);
    a.f32(SEL_BATCH as u64 * max_blocks as u64);
    a.f32(ATTN_BATCH as u64 * attn_scratch_floats as u64);
    a.used
}
fn moe_set_bytes(t: u64, n_expert: i64, fused: bool) -> u64 {
    let mp = mmq_plan();
    let mb = moe_bufs(t, n_expert, fused);
    let mut a = Alloc::count(0);
    a.f32(t * n_expert as u64);
    a.f32(t * K);
    a.i32(t * K);
    a.i32(t * K);
    a.i32(t * K);
    if mp.fallback {
        a.u16(t * K * N);
    }
    a.f32(mb.gu);
    if mp.fallback {
        a.u16(t * K * FF);
    }
    a.f32(t * K * N);
    a.f32(t * FF);
    a.f32(t * FF);
    a.u16(t * FF);
    a.f32(t * N);
    a.f32(t);
    if mp.any {
        a.u8(mb.xq);
        a.f32(mb.h);
        a.u8(mb.hq);
        a.f32(t * K);
        a.f32(t * K);
        a.f32(MMQ_GROUP * 4);
        a.f32(t * K);
    }
    a.used
}

/// `take_stage`: the KV streaming staging pool (every page of one layer)
fn take_stage(o: &mut Alloc, c: &CarveCtx, st: &mut TkP2KvPools) {
    let s = &c.s;
    if s.kv_mode != 1 {
        return;
    }
    let stage_own = env().stage_own;
    if stage_own && o.count_only {
        return;
    }
    let mut own = Alloc::region(std::ptr::null_mut(), 0, false, false, 0);
    let a: &mut Alloc = if stage_own { &mut own } else { o };
    let rows = (s.n_pages * s.n_head_kv * s.page_size) as u64;
    let hd = s.head_dim as u64;
    let q4 = s.kv_q4_bytes_per_head as u64;
    if s.kv_hybrid != 0 {
        st.k_q = a.u8(rows * hd).cast();
        st.k_scale = a.u16(rows * (hd / 64)).cast();
        st.v_q4 = a.u8(rows * q4);
    } else if s.kv_q4 != 0 {
        st.k_q4 = a.u8(rows * q4);
        st.v_q4 = a.u8(rows * q4);
    } else if s.kv_int8 != 0 {
        st.k_q = a.u8(rows * hd).cast();
        st.v_q = a.u8(rows * hd).cast();
        st.k_scale = a.u16(rows * (hd / 64)).cast();
        st.v_scale = a.u16(rows * (hd / 64)).cast();
    } else {
        st.k_pool = a.u16(rows * hd).cast();
        st.v_pool = a.u16(rows * hd).cast();
    }
    if stage_own {
        let ok = own.ok;
        let owned = std::mem::take(&mut own.owned);
        o.owned.extend(owned);
        o.ok &= ok;
    }
}

/// `Prefill::bytes_needed_impl`: the same allocation sequence as `carve`, counted.
fn bytes_needed_impl(c: &CarveCtx, chunk: i64, owned_pages: bool) -> u64 {
    let e = env();
    let t = chunk as u64;
    let mut o = Alloc::count(if owned_pages { 2 << 20 } else { 0 });
    o.u16(c.f.gemm_scratch_elems as u64);
    o.u8(c.f.gemm_ws_bytes as u64);
    if e.ring_bytes_on {
        o.f32(t * N);
        o.f32(t * D);
        if e.gr_unfused {
            o.f32(t * D);
        } else if hc16_fmt() != 0 {
            o.u16(t * D); // the 16-bit R
        }
        o.f32(t * HC);
    } else {
        o.f32(t * N);
        o.f32(t * D);
        o.f32(t * D);
    }
    o.u16(t * (D + if hc_pad_alloc() { XN_PAD } else { 0 }));
    o.f32(t * LR);
    o.u16(t * LR);
    o.f32(t * D);
    o.f32(t * HC);
    o.f32(t * N);
    o.u16(t * N);
    o.u16(t * N);
    if !e.emb_reuse_account {
        o.f32(t * N);
    }
    let mode = c.bf16x2_mode();
    if mode == 1 {
        o.u16(t * D);
        o.u16(t * LR);
    }
    if mode != 0 {
        o.u16(t * N);
    }
    o.i32(t * c.f.step_count as u64);
    let max_blocks = c.max_blocks();
    let region = gdn_set_bytes(t)
        .max(qsa_set_bytes(
            t,
            c.s.cap,
            max_blocks,
            c.s.attn_scratch_floats,
        ))
        .max(moe_set_bytes(
            t,
            c.s.n_expert,
            fused_layout(&c.f, chunk, c.src),
        ));
    o.u8(region);
    for _ in 0..DQ {
        o.u16(2 * FF * N);
        o.u16(N * FF);
    }
    let mp = mmq_plan();
    if mp.any {
        let ne = c.s.n_expert as u64;
        o.i32(t * K);
        o.i32(2 * (ne + ne / MMQ_GROUP + 2));
        o.u8(MMQ_GROUP * mp.gu_max + MMQ_TAIL);
        o.u8(MMQ_GROUP * mp.d_max + MMQ_TAIL);
    }
    let ring = ring_slots(&c.f, chunk);
    if owned_pages {
        if ring > 0 {
            o.u8(ring as u64 * c.f.max_blob as u64);
        }
    } else {
        for _ in 0..ring {
            o.u8(c.f.max_blob as u64);
        }
    }
    o.f32(t * N);
    o.f32(c.f.ng_hc_dim as u64);
    let mut stage = TkP2KvPools::default();
    take_stage(&mut o, c, &mut stage);
    o.used + (8u64 << 20)
}

/// A carve's result: what the object adopts (`ring_ptrs` backs `bufs.stage_dev`).
struct Carved {
    bufs: TkR2PfBufs,
    ring_ptrs: Vec<*mut u8>,
}

/// `Prefill::carve`: every device buffer of a chunk of `t` tokens from `o` (after the GEMM scratch and workspace).
fn carve(c: &CarveCtx, t_i: i64, o: &mut Alloc) -> Carved {
    let e = env();
    let t = t_i as u64;
    let mut b: TkR2PfBufs = unsafe { std::mem::zeroed() };
    b.emb = o.f32(t * N);
    b.r = o.f32(t * D);
    b.xn = if e.gr_unfused {
        o.f32(t * D)
    } else if hc16_fmt() != 0 {
        o.u16(t * D) // the 16-bit R (TALUS_FN_HC_BF16)
    } else {
        std::ptr::null_mut()
    };
    b.grs = o.f32(t * HC);
    b.xn16 = o.u16(t * (D + if hc_pad_alloc() { XN_PAD } else { 0 }));
    b.lo = o.f32(t * LR);
    b.lo16 = o.u16(t * LR);
    b.gated = o.f32(t * D);
    b.inj = o.f32(t * HC);
    b.mixed = o.f32(t * N);
    b.mixed_bf = o.u16(t * N);
    b.mixed_h = o.u16(t * N);
    b.bo = b.emb; // the embedding rows are dead after gr_broadcast
    let mode = c.bf16x2_mode();
    if mode == 1 {
        b.xn16_lo = o.u16(t * D);
        b.lo16_lo = o.u16(t * LR);
    }
    if mode != 0 {
        b.mixed_bf_lo = o.u16(t * N);
    }
    b.steps_dev = o.i32(t * c.f.step_count as u64);
    b.cap = c.s.cap;
    b.max_blocks = c.max_blocks();
    {
        let fz = fused_layout(&c.f, t_i, c.src);
        b.fused_bufs = fz as i32;
        let mb = moe_bufs(t, c.s.n_expert, fz);
        let region = gdn_set_bytes(t)
            .max(qsa_set_bytes(
                t,
                b.cap,
                b.max_blocks,
                c.s.attn_scratch_floats,
            ))
            .max(moe_set_bytes(t, c.s.n_expert, fz));
        let base = o.u8(region);
        b.region = base;
        b.region_bytes = region;
        if base.is_null() {
            o.ok = false;
        } else {
            let mut a = Alloc::region(base, region, false, false, 0);
            b.qkv = a.f32(t * C);
            b.z = a.f32(t * ZV);
            b.ab = a.f32(t * 2 * HV);
            b.gate = a.f32(t * HV);
            b.beta = a.f32(t * HV);
            b.hbuf = a.f32(t * C);
            b.y = a.f32(t * ZV);
            b.y_h = a.u16(t * (ZV + if e.pf_pad { ZV_PAD } else { 0 }));
            let mut q = Alloc::region(base, region, false, false, 0);
            b.kc = q.f32(t * QKV * HD);
            b.vc = q.f32(t * QKV * HD);
            b.qf = q.f32(t * 2 * QH * HD);
            b.q = q.f32(t * QH * HD);
            b.idx_raw = q.f32(t * IDXD);
            b.q_idx = q.f32(t * IDXH * IDXD);
            b.attn = q.f32(t * QH * HD);
            b.attn_h = q.u16(t * (QH * HD + if e.pf_pad { ZV_PAD } else { 0 }));
            b.sel_ids = q.i32(t * b.cap as u64);
            b.sel_scores = q.f32(SEL_BATCH as u64 * b.max_blocks as u64);
            b.attn_scratch = q.f32(ATTN_BATCH as u64 * c.s.attn_scratch_floats as u64);
            let mut m = Alloc::region(base, region, false, false, 0);
            let mp = mmq_plan();
            b.logits = m.f32(t * c.s.n_expert as u64);
            b.w = m.f32(t * K);
            b.ids = m.i32(t * K);
            b.slot_dev = m.i32(t * K);
            b.src_dev = m.i32(t * K);
            b.xs = if mp.fallback {
                m.u16(t * K * N)
            } else {
                std::ptr::null_mut()
            };
            b.gu = m.f32(mb.gu);
            b.hh = if mp.fallback {
                m.u16(t * K * FF)
            } else {
                std::ptr::null_mut()
            };
            b.dm = m.f32(t * K * N);
            b.sgate = m.f32(t * FF);
            b.sup = m.f32(t * FF);
            b.sh_h = m.u16(t * FF);
            b.shared = m.f32(t * N);
            b.sg = m.f32(t);
            if mp.any {
                b.xq = m.u8(mb.xq);
                b.h = m.f32(mb.h);
                b.hq = m.u8(mb.hq);
                b.xscale = m.f32(t * K);
                b.hscale = m.f32(t * K);
                b.grp_tail = m.f32(MMQ_GROUP * 4);
                b.row_sd = m.f32(t * K);
            }
            o.ok &= a.ok && q.ok && m.ok;
        }
    }
    for i in 0..DQ {
        b.dq_gu[i] = o.u16(2 * FF * N);
        b.dq_d[i] = o.u16(N * FF);
    }
    let mp = mmq_plan();
    if mp.any {
        let ne = c.s.n_expert as u64;
        b.ids_identity = o.i32(t * K);
        b.bounds_dev = o.i32(2 * (ne + ne / MMQ_GROUP + 2));
        b.grp_gu = o.u8(MMQ_GROUP * mp.gu_max + MMQ_TAIL);
        b.grp_d = o.u8(MMQ_GROUP * mp.d_max + MMQ_TAIL);
    }
    let ring = ring_slots(&c.f, t_i);
    b.ring = ring as i32;
    let blob = c.f.max_blob as u64;
    let mut ring_ptrs = Vec::with_capacity(ring.max(0) as usize);
    if o.base.is_null() && ring > 0 {
        // OWNED buffers: the ring in one allocation
        let ring_base = o.u8(ring as u64 * blob);
        if o.ok {
            for i in 0..ring as u64 {
                ring_ptrs.push(unsafe { ring_base.add((i * blob) as usize) });
            }
        }
    } else {
        for _ in 0..ring {
            ring_ptrs.push(o.u8(blob));
        }
    }
    b.ple_emb = o.f32(t * N);
    b.ple_norm = o.f32(c.f.ng_hc_dim as u64);
    let mut stage = TkP2KvPools::default();
    take_stage(o, c, &mut stage);
    b.stage = stage;
    b.t = t_i;
    Carved { bufs: b, ring_ptrs }
}

/// Hands the object a carve (and every allocation `o` made, which it frees with its own).
unsafe fn adopt(prefill: Vp, carved: &mut Carved, o: &mut Alloc) -> bool {
    carved.bufs.stage_dev = carved.ring_ptrs.as_ptr();
    carved.bufs.owned = o.owned.as_ptr();
    carved.bufs.n_owned = o.owned.len() as i64;
    let ok = (ffi().adopt_bufs)(prefill, &carved.bufs) == 1;
    o.owned.clear();
    ok
}

/// The allocations of a failed setup still belong to the object (C++ frees them with it).
unsafe fn adopt_owned_only(prefill: Vp, o: &mut Alloc) {
    if o.owned.is_empty() {
        return;
    }
    let mut c = Carved {
        bufs: std::mem::zeroed(),
        ring_ptrs: Vec::new(),
    };
    let _ = adopt(prefill, &mut c, o);
}

fn fit_error(chunk: i64, o: &Alloc) -> String {
    let (mut fb, mut tb) = (0u64, 0u64);
    unsafe { (ffi().mem_info)(&mut fb, &mut tb) };
    format!(
        "prefill: device buffers for a chunk of {chunk} tokens do not fit ({} of {} MiB free at the failure; {} MiB \
         taken, the failing buffer wanted {} MiB)",
        fb >> 20,
        tb >> 20,
        o.used >> 20,
        o.failed_bytes >> 20
    )
}

// ============================================================================================ the check mode

static CHECKS: AtomicU64 = AtomicU64::new(0);
static MISMATCHES: AtomicU64 = AtomicU64::new(0);

fn check_statics(g: Cvp, ss: Cvp, chunk: i64, src: bool) {
    let mut cpp = [0u64; 9];
    unsafe { (ffi().cpp_statics)(g, ss, chunk, src as i32, cpp.as_mut_ptr()) };
    let c = CarveCtx {
        f: facts(),
        s: session(g, ss),
        src,
        f16_io: facts().prompt_f16 != 0,
    };
    let f = &c.f;
    let need = bytes_needed_impl(&c, chunk, false);
    let rust = [
        need,
        bytes_needed_impl(&c, chunk, true),
        need - ring_slots(f, chunk) as u64 * f.max_blob as u64,
        ring_slots(f, chunk) as u64,
        ring_default_slots(f) as u64,
        ring_max_slots(f) as u64,
        ring_cap_for(f, chunk) as u64,
        stream_all_min() as u64,
        env().ring_bytes_on as u64,
    ];
    const NAMES: [&str; 9] = [
        "bytes_needed",
        "bytes_needed_owned",
        "bytes_needed_no_ring",
        "ring_slots_for",
        "ring_default_slots",
        "ring_max_slots",
        "ring_cap_for",
        "stream_all_min",
        "ring_bytes_enabled",
    ];
    CHECKS.fetch_add(1, Ordering::Relaxed);
    for i in 0..9 {
        if cpp[i] != rust[i] {
            MISMATCHES.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "qwen4exp-r2 CHECK MISMATCH {} chunk={chunk} src={src}: C++ {} Rust {}",
                NAMES[i], cpp[i], rust[i]
            );
        }
    }
}

/// The C++ carve and this file's on the same fake base, field for field (before the real carve is adopted: the C++
/// carve writes the object's pointers).
unsafe fn check_carve(prefill: Vp, c: &CarveCtx, chunk: i64) {
    if env().stage_own {
        return; // STRATA_KV_STAGE_OWN allocates the staging pool for real: not on a fake base
    }
    let fake = 0x7f00_0000_0000usize as *mut u8;
    let cap = 1u64 << 44;
    let mut lay: TkP2Layout = std::mem::zeroed();
    let rc = (ffi().cpp_carve)(prefill, chunk, fake, cap, &mut lay);
    let mut o = Alloc::region(fake, cap, false, false, 0);
    let r = carve(c, chunk, &mut o);
    let b = &r.bufs;
    let p = |x: *mut u8| x as usize;
    let mut fields: Vec<(&str, usize, usize)> = vec![
        ("emb", lay.emb as usize, p(b.emb)),
        ("R", lay.r as usize, p(b.r)),
        ("xn", lay.xn as usize, p(b.xn)),
        ("grs", lay.grs as usize, p(b.grs)),
        ("xn16", lay.xn16 as usize, p(b.xn16)),
        ("lo", lay.lo as usize, p(b.lo)),
        ("lo16", lay.lo16 as usize, p(b.lo16)),
        ("gated", lay.gated as usize, p(b.gated)),
        ("inj", lay.inj as usize, p(b.inj)),
        ("mixed", lay.mixed as usize, p(b.mixed)),
        ("mixed_bf", lay.mixed_bf as usize, p(b.mixed_bf)),
        ("mixed_h", lay.mixed_h as usize, p(b.mixed_h)),
        ("bo", lay.bo as usize, p(b.bo)),
        ("xn16_lo", lay.xn16_lo as usize, p(b.xn16_lo)),
        ("lo16_lo", lay.lo16_lo as usize, p(b.lo16_lo)),
        ("mixed_bf_lo", lay.mixed_bf_lo as usize, p(b.mixed_bf_lo)),
        ("steps_dev", lay.steps_dev as usize, p(b.steps_dev)),
        ("region", lay.region as usize, p(b.region)),
        (
            "region_bytes",
            lay.region_bytes as usize,
            b.region_bytes as usize,
        ),
        ("qkv", lay.qkv as usize, p(b.qkv)),
        ("z", lay.z as usize, p(b.z)),
        ("ab", lay.ab as usize, p(b.ab)),
        ("gate", lay.gate as usize, p(b.gate)),
        ("beta", lay.beta as usize, p(b.beta)),
        ("hbuf", lay.hbuf as usize, p(b.hbuf)),
        ("y", lay.y as usize, p(b.y)),
        ("y_h", lay.y_h as usize, p(b.y_h)),
        ("Kc", lay.kc as usize, p(b.kc)),
        ("Vc", lay.vc as usize, p(b.vc)),
        ("Qf", lay.qf as usize, p(b.qf)),
        ("q", lay.q as usize, p(b.q)),
        ("idx_raw", lay.idx_raw as usize, p(b.idx_raw)),
        ("q_idx", lay.q_idx as usize, p(b.q_idx)),
        ("attn", lay.attn as usize, p(b.attn)),
        ("attn_h", lay.attn_h as usize, p(b.attn_h)),
        ("sel_ids", lay.sel_ids as usize, p(b.sel_ids)),
        ("sel_scores", lay.sel_scores as usize, p(b.sel_scores)),
        ("attn_scratch", lay.attn_scratch as usize, p(b.attn_scratch)),
        ("logits", lay.logits as usize, p(b.logits)),
        ("w", lay.w as usize, p(b.w)),
        ("ids", lay.ids as usize, p(b.ids)),
        ("slot_dev", lay.slot_dev as usize, p(b.slot_dev)),
        ("src_dev", lay.src_dev as usize, p(b.src_dev)),
        ("Xs", lay.xs as usize, p(b.xs)),
        ("GU", lay.gu as usize, p(b.gu)),
        ("Hh", lay.hh as usize, p(b.hh)),
        ("Dm", lay.dm as usize, p(b.dm)),
        ("sgate", lay.sgate as usize, p(b.sgate)),
        ("sup", lay.sup as usize, p(b.sup)),
        ("sh_h", lay.sh_h as usize, p(b.sh_h)),
        ("shared", lay.shared as usize, p(b.shared)),
        ("sg", lay.sg as usize, p(b.sg)),
        ("Xq", lay.xq as usize, p(b.xq)),
        ("H", lay.h as usize, p(b.h)),
        ("Hq", lay.hq as usize, p(b.hq)),
        ("Xscale", lay.xscale as usize, p(b.xscale)),
        ("Hscale", lay.hscale as usize, p(b.hscale)),
        ("grp_tail", lay.grp_tail as usize, p(b.grp_tail)),
        ("row_sd", lay.row_sd as usize, p(b.row_sd)),
        ("dq_gu0", lay.dq_gu[0] as usize, p(b.dq_gu[0])),
        ("dq_gu1", lay.dq_gu[1] as usize, p(b.dq_gu[1])),
        ("dq_d0", lay.dq_d[0] as usize, p(b.dq_d[0])),
        ("dq_d1", lay.dq_d[1] as usize, p(b.dq_d[1])),
        ("ids_identity", lay.ids_identity as usize, p(b.ids_identity)),
        ("bounds_dev", lay.bounds_dev as usize, p(b.bounds_dev)),
        ("grp_gu", lay.grp_gu as usize, p(b.grp_gu)),
        ("grp_d", lay.grp_d as usize, p(b.grp_d)),
        ("ple_emb", lay.ple_emb as usize, p(b.ple_emb)),
        ("ple_norm", lay.ple_norm as usize, p(b.ple_norm)),
        (
            "stage.k_pool",
            lay.stage.k_pool as usize,
            b.stage.k_pool as usize,
        ),
        (
            "stage.v_pool",
            lay.stage.v_pool as usize,
            b.stage.v_pool as usize,
        ),
        ("stage.k_q", lay.stage.k_q as usize, b.stage.k_q as usize),
        ("stage.v_q", lay.stage.v_q as usize, b.stage.v_q as usize),
        (
            "stage.k_scale",
            lay.stage.k_scale as usize,
            b.stage.k_scale as usize,
        ),
        (
            "stage.v_scale",
            lay.stage.v_scale as usize,
            b.stage.v_scale as usize,
        ),
        ("stage.k_q4", lay.stage.k_q4 as usize, b.stage.k_q4 as usize),
        ("stage.v_q4", lay.stage.v_q4 as usize, b.stage.v_q4 as usize),
        ("T", lay.t as usize, b.t as usize),
        ("cap", lay.cap as usize, b.cap as usize),
        ("max_blocks", lay.max_blocks as usize, b.max_blocks as usize),
        ("ring", lay.ring as usize, b.ring as usize),
        ("fused_bufs", lay.fused_bufs as usize, b.fused_bufs as usize),
    ];
    for (i, &rp) in r.ring_ptrs.iter().enumerate() {
        let cp = if lay.stage_dev.is_null() {
            0
        } else {
            *lay.stage_dev.add(i) as usize
        };
        fields.push(("stage_dev[i]", cp, rp as usize));
    }
    let mut bad = 0;
    for (name, cpp, rust) in &fields {
        if cpp != rust {
            bad += 1;
            if bad <= 20 {
                eprintln!(
                    "qwen4exp-r2 CHECK MISMATCH carve T={chunk} {name}: C++ {:#x} Rust {:#x}",
                    cpp, rust
                );
            }
        }
    }
    MISMATCHES.fetch_add(bad, Ordering::Relaxed);
    CHECKS.fetch_add(1, Ordering::Relaxed);
    eprintln!(
        "qwen4exp-r2 check: carve T={chunk} (C++ rc {rc}, used {} MiB): {} fields, {bad} mismatches; statics checked {} \
         times, {} mismatches in all",
        o.used >> 20,
        fields.len(),
        CHECKS.load(Ordering::Relaxed),
        MISMATCHES.load(Ordering::Relaxed)
    );
}

// ============================================================================================ the hooks

unsafe extern "C" fn h_bytes_needed(
    _: Vp,
    g: Cvp,
    ss: Cvp,
    chunk: i64,
    owned: i32,
    src: i32,
) -> u64 {
    if env().check {
        check_statics(g, ss, chunk, src != 0);
    }
    let f = facts();
    let c = CarveCtx {
        f,
        s: session(g, ss),
        src: src != 0,
        f16_io: f.prompt_f16 != 0,
    };
    bytes_needed_impl(&c, chunk, owned != 0)
}
unsafe extern "C" fn h_bytes_needed_no_ring(u: Vp, g: Cvp, ss: Cvp, chunk: i64, src: i32) -> u64 {
    let need = h_bytes_needed(u, g, ss, chunk, 0, src);
    let f = facts();
    need - ring_slots(&f, chunk) as u64 * f.max_blob as u64
}
unsafe extern "C" fn h_ring_slots_for(_: Vp, chunk: i64) -> i64 {
    ring_slots(&facts(), chunk)
}
unsafe extern "C" fn h_ring_default_slots(_: Vp) -> i64 {
    ring_default_slots(&facts())
}
unsafe extern "C" fn h_ring_max_slots(_: Vp) -> i64 {
    ring_max_slots(&facts())
}
unsafe extern "C" fn h_ring_cap_for(_: Vp, old_chunk: i64) -> i64 {
    ring_cap_for(&facts(), old_chunk)
}
unsafe extern "C" fn h_stream_all_min(_: Vp) -> i64 {
    stream_all_min()
}
unsafe extern "C" fn h_ring_bytes_enabled(_: Vp) -> i32 {
    env().ring_bytes_on as i32
}
unsafe extern "C" fn h_pinned_share(_: Vp) -> f64 {
    rules().pinned_share
}
unsafe extern "C" fn h_set_pinned_share(_: Vp, share: f64) {
    rules().pinned_share = share;
}
unsafe extern "C" fn h_set_ring_override(_: Vp, slots: i32) {
    rules().ring_override = slots.max(0);
}
unsafe extern "C" fn h_set_ring_budget(_: Vp, slots: i32, small_max: i64) {
    let mut r = rules();
    r.ring_budget = slots.max(0);
    r.ring_small_max = small_max.max(0);
}
unsafe extern "C" fn h_arm_cpu_share(_: Vp, applies: i32, by_default: i32) {
    // the by-default arm is CUDA-only (#if !STRATA_USE_HIP in prefill.cpp)
    if facts().is_hip == 0
        && applies != 0
        && by_default != 0
        && env().cpu_share_explicit == -2.0
        && !rules().share_default
    {
        rules().share_default = true;
    }
    if applies != 0 && cpu_share_on() {
        let m = cpu_share_max();
        rules().stream_min_share = m;
    }
}

/// `Prefill::init` (one stage): 1 done, 0 failed, -1 declined.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn h_init(
    _: Vp,
    prefill: Vp,
    wt: Cvp,
    g: Cvp,
    ss: Vp,
    src: Vp,
    cache: Cvp,
    host_res: *const i32,
    chunk: i64,
    stream: Vp,
    borrow: Vp,
    borrow_bytes: u64,
    borrow_spill: i32,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        init(
            prefill,
            wt,
            g,
            ss,
            src,
            cache,
            host_res,
            chunk,
            stream,
            borrow,
            borrow_bytes,
            borrow_spill != 0,
        )
    }));
    match res {
        Ok(Ok(true)) => 1,
        Ok(Ok(false)) => -1,
        Ok(Err(e)) => {
            write_err(err, err_cap, &e);
            0
        }
        Err(_) => {
            write_err(err, err_cap, "prefill (rust): panic in init");
            0
        }
    }
}

unsafe extern "C" fn h_relayout(
    _: Vp,
    prefill: Vp,
    chunk: i64,
    borrow: Vp,
    borrow_bytes: u64,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        relayout(prefill, chunk, borrow, borrow_bytes)
    }));
    match res {
        Ok(Ok(true)) => 1,
        Ok(Ok(false)) => -1,
        Ok(Err(e)) => {
            write_err(err, err_cap, &e);
            0
        }
        Err(_) => {
            write_err(err, err_cap, "prefill (rust): panic in relayout");
            0
        }
    }
}

/// The geometry the prompt path is compiled for, against the model (init's first check)
fn check_geometry(s: &TkR2PfSession) -> Result<(), String> {
    let req: [(&str, i64, i64, bool); 15] = [
        ("n_embd", s.n_embd, N as i64, s.n_embd == N as i64),
        ("hyper-connection count", s.hc, HC as i64, s.hc == HC as i64),
        (
            "hyper-connection rank",
            s.hc_lr,
            LR as i64,
            s.hc_lr == LR as i64,
        ),
        ("experts used (top-k)", s.k, K as i64, s.k == K as i64),
        (
            "experts (at most)",
            s.n_expert,
            NE,
            s.n_expert >= 1 && s.n_expert <= NE,
        ),
        ("expert width", s.n_ff, FF as i64, s.n_ff == FF as i64),
        (
            "shared expert width",
            s.n_ff_shexp,
            FF as i64,
            s.n_ff_shexp == FF as i64,
        ),
        (
            "GDN conv channels",
            s.conv_ch,
            C as i64,
            s.conv_ch == C as i64,
        ),
        (
            "GDN value width",
            s.value_dim,
            ZV as i64,
            s.value_dim == ZV as i64,
        ),
        (
            "GDN value heads",
            s.v_heads,
            HV as i64,
            s.v_heads == HV as i64,
        ),
        (
            "attention heads",
            s.n_head,
            QH as i64,
            s.n_head == QH as i64,
        ),
        (
            "KV heads",
            s.n_head_kv,
            QKV as i64,
            s.n_head_kv == QKV as i64,
        ),
        (
            "head dim",
            s.head_dim,
            HD as i64,
            s.head_dim == HD as i64 && s.head_dim_v == HD as i64,
        ),
        (
            "indexer heads",
            s.idx_heads,
            IDXH as i64,
            s.idx_heads == IDXH as i64,
        ),
        (
            "indexer key width",
            s.idx_dim,
            IDXD as i64,
            s.idx_dim == IDXD as i64,
        ),
    ];
    for (what, have, need, ok) in req {
        if !ok {
            return Err(format!(
                "prefill: the model's {what} is {have}, the prompt path requires {need}"
            ));
        }
    }
    if s.rms_eps as f32 != EPS {
        return Err("prefill: the model's RMS epsilon differs from the prompt path's 1e-6".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn init(
    prefill: Vp,
    wt: Cvp,
    g: Cvp,
    ss: Vp,
    src: Vp,
    cache: Cvp,
    host_res: *const i32,
    chunk: i64,
    stream: Vp,
    borrow: Vp,
    borrow_bytes: u64,
    borrow_spill: bool,
) -> Result<bool, String> {
    let x = ffi();
    let s = session(g, ss);
    check_geometry(&s)?;
    if (x.stage_ok)(prefill, s.n_layers) == 0 {
        return Ok(false); // a layer split stage, a helper / peer, or a re-init: C++
    }
    let mut device = -1;
    (x.get_device)(&mut device);
    let f = facts();
    let t = chunk as u64;
    let borrowed = !borrow.is_null();
    let spill = borrowed && borrow_spill;
    let mut ok = true;
    let mut owned: Vec<Vp> = Vec::new();
    // the token ids on the device (one batched embedding gather)
    let mut tok: Vp = std::ptr::null_mut();
    if (x.malloc)(&mut tok, t * 4) == 0 {
        (x.clear_error)();
        return Err("prefill: the token id buffer (out of memory)".into());
    }
    owned.push(tok);
    let mut copy: Vp = std::ptr::null_mut();
    if (x.stream_create)(&mut copy) == 0 {
        adopt_list(prefill, &owned);
        return Err("prefill: copy stream".into());
    }
    // the ring's events, for the largest ring this pack can lay out
    let n_ev = ring_cap(&f);
    let mut copied: Vec<Vp> = vec![std::ptr::null_mut(); n_ev as usize];
    let mut used: Vec<Vp> = vec![std::ptr::null_mut(); n_ev as usize];
    for i in 0..n_ev as usize {
        if (x.event_create)(&mut copied[i], 0) == 0 {
            ok = false;
        }
        if (x.event_create)(&mut used[i], 0) == 0 {
            ok = false;
        }
    }
    // the grouping tables in mapped pinned memory (kernels read and write them in place; STRATA_GROUP_COPY=1: copies)
    let need = 3 * t * K + 2 * (s.n_expert as u64 + s.n_expert as u64 / MMQ_GROUP + 2);
    let (mut grp_host, mut grp_dev, mut grp_n, mut grp_tk) = (
        std::ptr::null_mut::<c_void>(),
        std::ptr::null_mut::<c_void>(),
        0u64,
        0u64,
    );
    if !env().group_copy && !env().force_pageable {
        let (mut h, mut d) = (std::ptr::null_mut(), std::ptr::null_mut());
        if (x.host_alloc_mapped)(&mut h, &mut d, need * 4, f.peer_portable) == 1 {
            grp_host = h;
            grp_dev = d;
            grp_n = need;
            grp_tk = t * K;
        }
    }
    // KV streaming: the staging pool's identity page table
    let mut ident: Vp = std::ptr::null_mut();
    if s.kv_mode == 1 {
        let pages = s.n_pages;
        let id: Vec<i32> = (0..pages as i32).collect();
        if (x.malloc)(&mut ident, id.len() as u64 * 4) == 0 {
            ok = false;
            ident = std::ptr::null_mut();
        } else if (x.memcpy_h2d_sync)(ident, id.as_ptr().cast(), id.len() as u64 * 4) == 0 {
            ok = false;
            (x.free)(ident);
            ident = std::ptr::null_mut();
        } else {
            owned.push(ident);
        }
    }
    let init_in = TkR2PfInit {
        t_max: chunk,
        borrowed: borrowed as i32,
        borrow_spill: spill as i32,
        device,
        n_events: n_ev as i32,
        wt,
        g,
        ss,
        src,
        cache,
        host_res,
        cs: stream,
        copy,
        copied: copied.as_ptr(),
        used: used.as_ptr(),
        tok_dev: tok.cast(),
        ident_table: ident.cast(),
        grp_host: grp_host.cast(),
        grp_dev: grp_dev.cast(),
        grp_n,
        grp_tk,
        owned: owned.as_ptr(),
        n_owned: owned.len() as i64,
    };
    if (x.adopt_init)(prefill, &init_in) != 1 {
        return Err("prefill (rust): adopt_init".into());
    }
    // the host side (Stager threads, pinned uploads, host vectors, PLE host buffers)
    if (x.init_host)(prefill, chunk) != 1 {
        ok = false;
    }
    if !ok {
        return Err(format!(
            "prefill: host buffers or events for a chunk of {chunk} tokens"
        ));
    }
    let mut o = Alloc::region(borrow.cast(), borrow_bytes, spill, false, 0);
    let gs = o.u16(f.gemm_scratch_elems as u64);
    let ws = o.u8(f.gemm_ws_bytes as u64);
    if !o.ok {
        adopt_owned_only(prefill, &mut o);
        return Err("prefill: GEMM scratch does not fit".into());
    }
    let mut f16 = 0i32;
    let mut eb = [0 as c_char; 512];
    if (x.gemm_init)(
        prefill,
        stream,
        gs.cast(),
        f.gemm_scratch_elems,
        ws.cast(),
        f.gemm_ws_bytes as u64,
        &mut f16,
        eb.as_mut_ptr(),
        eb.len(),
    ) != 1
    {
        adopt_owned_only(prefill, &mut o);
        return Err(std::ffi::CStr::from_ptr(eb.as_ptr())
            .to_string_lossy()
            .into_owned());
    }
    let c = CarveCtx {
        f,
        s,
        src: !src.is_null(),
        f16_io: f16 != 0,
    };
    let mut carved = carve(&c, chunk, &mut o);
    if !o.ok {
        let e = fit_error(chunk, &o);
        adopt_owned_only(prefill, &mut o);
        return Err(e);
    }
    if env().check {
        check_carve(prefill, &c, chunk);
    }
    if !adopt(prefill, &mut carved, &mut o) {
        return Err("prefill (rust): adopt_bufs".into());
    }
    Ok(true)
}

unsafe fn adopt_list(prefill: Vp, owned: &[Vp]) {
    let mut o = Alloc::region(std::ptr::null_mut(), 0, false, false, 0);
    o.owned.extend_from_slice(owned);
    adopt_owned_only(prefill, &mut o);
}

unsafe fn relayout(prefill: Vp, chunk: i64, borrow: Vp, borrow_bytes: u64) -> Result<bool, String> {
    let x = ffi();
    let mut st: TkR2PfState = std::mem::zeroed();
    if (x.state)(prefill, &mut st) != 1 {
        return Ok(false);
    }
    if st.borrow_spill != 0 {
        return Err(
            "prefill: a hybrid borrowed/owned layout cannot be changed between requests".into(),
        );
    }
    if st.borrowed == 0 || borrow.is_null() || chunk <= 0 || chunk > st.t_max {
        return Err(format!(
            "prefill: relayout needs borrowed buffers and a chunk of at most {}",
            st.t_max
        ));
    }
    if (x.stream_sync)(st.cs) == 0
        || (x.stream_sync)(st.copy) == 0
        || (!st.kv_copy.is_null() && (x.stream_sync)(st.kv_copy) == 0)
    {
        return Err("prefill: relayout: the stream failed".into());
    }
    let f = facts();
    let mut o = Alloc::region(borrow.cast(), borrow_bytes, false, false, 0);
    let gs = o.u16(f.gemm_scratch_elems as u64);
    let ws = o.u8(f.gemm_ws_bytes as u64);
    if o.ok {
        (x.gemm_rebind)(
            prefill,
            gs.cast(),
            f.gemm_scratch_elems,
            ws.cast(),
            f.gemm_ws_bytes as u64,
        );
    }
    let c = CarveCtx {
        f,
        s: session(st.g, st.ss),
        src: st.has_src != 0,
        f16_io: st.f16_io != 0,
    };
    let ok_before = o.ok;
    let mut carved = if ok_before {
        Some(carve(&c, chunk, &mut o))
    } else {
        None
    };
    if !ok_before || !o.ok {
        let e = fit_error(chunk, &o);
        adopt_owned_only(prefill, &mut o);
        return Err(e);
    }
    if env().check {
        check_carve(prefill, &c, chunk);
    }
    if !adopt(prefill, carved.as_mut().unwrap(), &mut o) {
        return Err("prefill (rust): adopt_bufs".into());
    }
    Ok(true)
}

/// Registers the hooks (idempotent). False: `TALUS_FN_R2=0`, or the library has no qwen4exp-r2 entries.
pub(super) fn install() -> bool {
    if env_str("TALUS_FN_R2")
        .map(|v| v.trim() == "0")
        .unwrap_or(false)
    {
        return false;
    }
    let f = FFI.get_or_init(|| match R2Ffi::load() {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("talus-qwen4exp: qwen4exp-r2 entries not found ({e}); the C++ prompt-path setup stays");
            None
        }
    });
    let Some(f) = f.as_ref() else { return false };
    let hooks = TkR2PfHooks {
        user: std::ptr::null_mut(),
        bytes_needed: h_bytes_needed,
        bytes_needed_no_ring: h_bytes_needed_no_ring,
        ring_slots_for: h_ring_slots_for,
        ring_default_slots: h_ring_default_slots,
        ring_max_slots: h_ring_max_slots,
        ring_cap_for: h_ring_cap_for,
        stream_all_min: h_stream_all_min,
        ring_bytes_enabled: h_ring_bytes_enabled,
        pinned_share: h_pinned_share,
        set_pinned_share: h_set_pinned_share,
        set_ring_override: h_set_ring_override,
        set_ring_budget: h_set_ring_budget,
        arm_cpu_share: h_arm_cpu_share,
        init: h_init,
        relayout: h_relayout,
    };
    unsafe { (f.install)(&hooks) };
    // (env() is not read here: see Env)
    let check = env_first_is("TALUS_FN_R2_CHECK", b'1');
    if env_str("TALUS_FN_P2_SAY").is_some() || check {
        eprintln!(
            "talus-qwen4exp: qwen4exp-r2 prompt-path setup installed (sizing, carve, relayout, init){}",
            if check { "; TALUS_FN_R2_CHECK on" } else { "" }
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_count_rounds_like_cpp() {
        let mut a = Alloc::count(0);
        a.take(1);
        assert_eq!(a.used, 512);
        a.take(256);
        assert_eq!(a.used, 512 + 512);
        a.take(257);
        assert_eq!(a.used, 512 + 512 + 768);
        let mut b = Alloc::count(2 << 20);
        b.take(1);
        assert_eq!(b.used, 2 << 20);
    }

    #[test]
    fn atof_matches_c() {
        assert_eq!(c_atof("0.5"), 0.5);
        assert_eq!(c_atof(" 1"), 1.0);
        assert_eq!(c_atof("x"), 0.0);
        assert_eq!(c_atof("0.25abc"), 0.25);
        assert_eq!(c_atof("1e"), 1.0);
    }
}
