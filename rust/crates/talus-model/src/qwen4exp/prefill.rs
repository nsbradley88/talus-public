//! Piece 2 (`qwen4exp-p2`): Flash-Next (`qwen4exp`) batched prompt processing in Rust.
//!
//! This is `src/prefill/prefill.cpp`'s `Prefill::run_impl` and `Prefill::draft_kv` as the Rust host's data plane:
//! the chunk loop, the per-chunk host preparation (token ids, QSA step records, the PLE rows read on a thread one
//! chunk ahead), the expert stream plan (every non-resident expert of every layer through the ring, in id order)
//! and its issuer thread, the layer / half sequencing (hyper-connection read, GDN or QSA mixer, MoE, the fused
//! hyper-connection write), the routed rows' host grouping and MMQ group bounds, the fused-expert batches, the
//! `TALUS_FN_DUMP` layer dumps and the MTP draft layer's batched K/V (E-9).  Every GPU step is one launch through the
//! `tk_qwen4exp_p2_*` C ABI (`src/program/qwen4exp_p2.cpp`); routed experts go through the same entries as the C++
//! path - `fused::experts_native` (fp16 kernels, `STRATA_PF_W16`, default on gfx1151) and
//! `run_mmq_product` (tp_moe's `moe_gemm`, then MMQ) - so the bits are the C++ path's.
//!
//! The C++ `strata::prefill::Prefill` object still owns the buffers (`init` / `carve` / `relayout`, which the serve
//! lend in generate.cpp calls) and hands `run` / `draft_kv` to the callbacks installed here when the Rust data plane
//! is selected (`TALUS_FN_RUST=1` or `--rust-engine`; `TALUS_FN_P2=0` keeps the C++ prompt path for an A/B).
//! Modes this file does not run are declined and keep the C++ path: a layer split stage, `--peer-device`, the CPU
//! expert share, picture rows, KV prefetch, and the debug dumps listed in `debug_env_set`.
//!
//! ## Bitwise verification (`TALUS_FN_DUMP`)
//! `prefill_p{p0}_input_R`, `prefill_p{p0}_l{00..47}_mixer_R`, `prefill_p{p0}_l{00..47}_R` are written by both
//! paths; compare `TALUS_FN_RUST=0` against `TALUS_FN_RUST=1` with `--fn-dump-compare`.

use core::ffi::{c_char, c_int, c_void};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::qwen4exp::ffi::{
    P2Ffi, Qwen4ExpKernels, TkP2AttnPools, TkP2Cvec, TkP2ExpertFmt, TkP2KvPools, TkP2Layout,
    TkP2MtpInfo, TkP2QsaInfo, TkP2StagerJob, TkP2Weight, TkQwen4ExpPrefillStats,
    TkQwen4ExpServeCtx,
};
use crate::qwen4exp::skeleton::Qwen4ExpModelConfig;

// qwen4exp-r2: the prompt path's buffer setup (Prefill's sizing statics, carve, relayout, init's device half)
mod setup;

/// Equal-chunk divider matching `equal_chunk(n, max_c)` in `src/program/generate.cpp`
/// (delegates to `weights::serve_equal_chunk` / `weights::request_chunk`).
#[inline]
pub fn equal_chunk(tokens: i64, max_chunk: i64) -> i64 {
    let equal_chunks = std::env::var("STRATA_PREFILL_EQUAL")
        .ok()
        .map(|v| v.starts_with('1'))
        .unwrap_or(false);
    let stream_all_min = std::env::var("STRATA_PREFILL_STREAM_ALL")
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .unwrap_or(1024);
    crate::qwen4exp::weights::serve_equal_chunk(tokens, max_chunk, equal_chunks, stream_all_min)
}

/// Piece 2 prefill orchestrator state owned by `Qwen4ExpRustEngine`.
#[derive(Clone, Debug)]
pub struct Piece2Prefill {
    pub max_chunk: i64,
    pub short_read_threshold: i64,
    pub pf_switch_min_t: i64,
    pub tp_moe_enabled: bool,
    pub w16_enabled: bool,
    pub last_stats: TkQwen4ExpPrefillStats,
    /// The Rust prompt path is installed into the C++ `Prefill` objects (`TALUS_FN_RUST=1`, `TALUS_FN_P2` not 0).
    pub installed: bool,
}

impl Piece2Prefill {
    pub fn new(_cfg: &Qwen4ExpModelConfig) -> Self {
        let pf_switch_min_t = std::env::var("STRATA_PF_SWITCH_MIN_T")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(0);
        let tp_moe_enabled = std::env::var("TALUS_PF_MOE")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        let w16_enabled = std::env::var("STRATA_PF_W16")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        let argv: Vec<String> = std::env::args().collect();
        let rust_plane = crate::qwen4exp::cli::use_rust_engine(&argv);
        let installed = rust_plane && install();
        // qwen4exp-r2: the setup is its own switch (TALUS_FN_R2=0), so TALUS_FN_P2=0 still A/Bs the prompt path alone
        let _ = rust_plane && setup::install();
        Self {
            max_chunk: 4096,
            short_read_threshold: 32,
            pf_switch_min_t,
            tp_moe_enabled,
            w16_enabled,
            last_stats: TkQwen4ExpPrefillStats::default(),
            installed,
        }
    }

    /// Runs a prefill segment `[at .. to)` through the serve segment entry (verify windows or batched chunks); the
    /// batched chunks themselves run in this file's `run_prompt` (installed into the C++ `Prefill::run`).
    pub unsafe fn run_segment(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        at: i64,
        to: i64,
        use_windows: bool,
    ) -> Result<(bool, bool), String> {
        let mut nf_hit = 0i32;
        let mut cancelled = 0i32;
        let rc = (kernels.serve_prefill_segment)(
            ctx,
            at,
            to,
            if use_windows { 1 } else { 0 },
            &mut nf_hit,
            &mut cancelled,
        );
        if rc < 0 {
            return Err(format!("prefill segment [{at}..{to}) failed (rc={rc})"));
        }
        let _ = (kernels.p2_prefill_chunk)(ctx, at, to, &mut self.last_stats);
        Ok((nf_hit != 0, cancelled != 0))
    }
}

// ============================================================================================ installation

static FFI: OnceLock<Option<P2Ffi>> = OnceLock::new();
/// Per C++ Prefill object (its address): the host-side scratch of the Rust path.
static HOSTS: OnceLock<Mutex<HashMap<usize, P2Host>>> = OnceLock::new();

fn ffi() -> Option<&'static P2Ffi> {
    FFI.get().and_then(|o| o.as_ref())
}

/// The ABI struct sizes the C++ side was compiled with, against this file's mirrors.
fn abi_ok(f: &P2Ffi) -> bool {
    let mut got = [0i64; 8];
    unsafe { (f.abi_sizes)(got.as_mut_ptr()) };
    let want = [
        std::mem::size_of::<TkP2Layout>() as i64,
        std::mem::size_of::<TkP2QsaInfo>() as i64,
        std::mem::size_of::<TkP2Weight>() as i64,
        std::mem::size_of::<TkP2ExpertFmt>() as i64,
        std::mem::size_of::<crate::qwen4exp::ffi::TkP2Stats>() as i64,
        std::mem::size_of::<TkP2MtpInfo>() as i64,
        std::mem::size_of::<TkP2Cvec>() as i64,
        std::mem::size_of::<TkP2StagerJob>() as i64,
    ];
    if got != want {
        eprintln!("talus-qwen4exp: qwen4exp-p2 ABI mismatch (C++ {got:?}, Rust {want:?}); the C++ prompt path stays");
        return false;
    }
    true
}

/// Installs the Rust prompt path into the loaded library (idempotent). False: `TALUS_FN_P2=0`, or the library has
/// no qwen4exp-p2 entries (an older build) - the C++ path then runs.
fn install() -> bool {
    if env_str("TALUS_FN_P2")
        .map(|v| v.trim() == "0")
        .unwrap_or(false)
    {
        return false;
    }
    let f = FFI.get_or_init(|| match P2Ffi::load() {
        Ok(f) if abi_ok(&f) => Some(f),
        Ok(_) => None,
        Err(e) => {
            eprintln!(
                "talus-qwen4exp: qwen4exp-p2 entries not found ({e}); the C++ prompt path stays"
            );
            None
        }
    });
    let Some(f) = f.as_ref() else { return false };
    HOSTS.get_or_init(|| Mutex::new(HashMap::new()));
    let draft = env_str("TALUS_FN_P2_DRAFT")
        .map(|v| v.trim() != "0")
        .unwrap_or(true);
    unsafe {
        (f.install)(
            Some(p2_run_cb),
            if draft { Some(p2_draft_cb) } else { None },
            std::ptr::null_mut(),
        )
    };
    if env_str("TALUS_FN_P2_SAY").is_some() {
        eprintln!(
            "talus-qwen4exp: qwen4exp-p2 prompt path installed (draft_kv {})",
            if draft { "rust" } else { "c++" }
        );
    }
    true
}

unsafe fn write_err(buf: *mut c_char, cap: usize, msg: &str) {
    if buf.is_null() || cap == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let n = bytes.len().min(cap - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.cast::<u8>(), n);
    *buf.add(n) = 0;
}

unsafe extern "C" fn p2_run_cb(
    _user: *mut c_void,
    prefill: *mut c_void,
    tokens: *const i64,
    n: i64,
    pos0: i64,
    err: *mut c_char,
    err_cap: usize,
) -> c_int {
    let Some(f) = ffi() else { return -1 };
    let Some(hosts) = HOSTS.get() else { return -1 };
    let mut guard = match hosts.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let host = guard.entry(prefill as usize).or_default();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_prompt(f, prefill, host, tokens, n, pos0)
    }));
    match res {
        Ok(Ok(true)) => 1,
        Ok(Ok(false)) => -1,
        Ok(Err(e)) => {
            write_err(err, err_cap, &e);
            0
        }
        Err(_) => {
            write_err(err, err_cap, "prefill (rust): panic in the prompt path");
            0
        }
    }
}

unsafe extern "C" fn p2_draft_cb(
    _user: *mut c_void,
    prefill: *mut c_void,
    mtp: *mut c_void,
    r_rows: *const f32,
    next_tokens: *const i32,
    n: i64,
    cell0: i64,
    err: *mut c_char,
    err_cap: usize,
    why: *mut *const c_char,
) -> c_int {
    let Some(f) = ffi() else { return -1 };
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        draft_kv(f, prefill, mtp, r_rows, next_tokens, n, cell0)
    }));
    match res {
        Ok(Ok(Draft::Done)) => 1,
        Ok(Ok(Draft::NotHandled)) => -1,
        Ok(Ok(Draft::Decline(reason))) => {
            if !why.is_null() {
                *why = reason.as_ptr();
            }
            0
        }
        Ok(Err(e)) => {
            write_err(err, err_cap, &e);
            0
        }
        Err(_) => {
            write_err(err, err_cap, "prefill (rust): panic in draft_kv");
            0
        }
    }
}

// ============================================================================================ environment

fn env_str(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}
/// C's atoll: leading blanks, a sign, the digits (0 for none).
fn c_atoll(s: &str) -> i64 {
    let b = s.trim_start().as_bytes();
    let mut i = 0;
    let mut neg = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((b[i] - b'0') as i64);
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}
fn c_atoi(s: &str) -> i32 {
    c_atoll(s) as i32
}
/// `e && e[0] == c`
fn env_first_is(name: &str, c: u8) -> bool {
    env_str(name)
        .map(|v| v.as_bytes().first() == Some(&c))
        .unwrap_or(false)
}
/// `v == nullptr || atoi(v) != 0`
fn env_default_on(name: &str) -> bool {
    env_str(name).map(|v| c_atoi(&v) != 0).unwrap_or(true)
}

/// prefill.cpp's environment switches, read once (as its function-local statics are).
struct Env {
    pf_hcdown_mode: i32,
    pf_hcdown_min: i64,
    hcd_exact: bool,
    gr_unfused: bool,
    hc_upmix: bool,
    pf_switch_min_t: i64,
    pf_pad: bool,
    cvec_fuse: bool,
    fnp_tile_attn: bool,
    bf16x2_env: i32,
    stream_ahead: bool,
    issuer_on: bool,
    ple_batch_env: bool,
    ple_threads: i64,
    group_gather_env: bool,
    step_sync: bool,
    kv_host_dma: bool,
    indexer_per_token: bool,
    select_old: bool,
    prompt_attn_old: bool,
    timing: bool,
    trace: bool,
    kv_prefetch: bool,
    mtp_batch_off: bool,
    mtp_batch_ring_ok: bool,
    mtp_batch_f16: bool,
    draft_timing: bool,
    /// a debug / research switch of the C++ path is set: the run is declined (named for the log)
    debug: Option<&'static str>,
}

fn env() -> &'static Env {
    static E: OnceLock<Env> = OnceLock::new();
    E.get_or_init(|| {
        let pf_hcdown_mode = match env_str("STRATA_PF_HCDOWN") {
            None => 0,
            Some(v) => match v.as_bytes().first() {
                None | Some(b'0') => 0,
                Some(b'1') => 1,
                _ => 2,
            },
        };
        const DEBUG: [&str; 15] = [
            "STRATA_SEL_OVERLAP",
            "STRATA_IDX_FP16_CHECK",
            "STRATA_QSA_DUMP",
            "STRATA_DBG_FEAT",
            "STRATA_DBG_ROUTE_OVERRIDE",
            "STRATA_DUMP_MOE_INPUT",
            "STRATA_DBG_NAN",
            "STRATA_PREFILL_DUMP_R",
            "STRATA_PREFILL_DUMP_R_ALL",
            "STRATA_HC_UPMIX_CHECK",
            "STRATA_CVEC_FUSE_CHECK",
            "STRATA_STATE_HASH_GDN",
            "STRATA_MMQ_RESIDENT_SORT_NE",
            "STRATA_DBG_FORCE_SLOW_LAYER",
            "STRATA_DBG_CPU_GATE",
        ];
        let debug = DEBUG.iter().copied().find(|n| match *n {
            // the checks are off at 0 / unset
            "STRATA_HC_UPMIX_CHECK" | "STRATA_CVEC_FUSE_CHECK" => {
                env_str(n).map(|v| c_atoi(&v) > 0).unwrap_or(false)
            }
            "STRATA_DBG_FORCE_SLOW_LAYER" => env_first_is(n, b'1'),
            "STRATA_MMQ_RESIDENT_SORT_NE" => env_str(n).is_some(),
            _ => env_str(n).is_some(),
        });
        Env {
            pf_hcdown_mode,
            pf_hcdown_min: env_str("STRATA_PF_HCDOWN_MIN")
                .map(|v| c_atoll(&v))
                .unwrap_or(16384),
            hcd_exact: env_first_is("STRATA_HCD_EXACT", b'1'),
            gr_unfused: env_first_is("STRATA_GR_UNFUSED", b'1'),
            hc_upmix: env_first_is("STRATA_HC_UPMIX", b'1'),
            pf_switch_min_t: env_str("STRATA_PF_SWITCH_MIN_T")
                .map(|v| c_atoll(&v))
                .unwrap_or(0),
            pf_pad: env_first_is("STRATA_PF_PAD", b'1') && env_first_is("STRATA_PF_GEMM", b'1'),
            cvec_fuse: env_first_is("STRATA_CVEC_FUSE", b'1'),
            fnp_tile_attn: env_first_is("STRATA_FNP_TILE_ATTN", b'1'),
            bf16x2_env: env_str("STRATA_PREFILL_BF16X2")
                .map(|v| c_atoi(&v))
                .unwrap_or(0),
            stream_ahead: env_default_on("STRATA_PREFILL_STREAM_AHEAD"),
            issuer_on: env_default_on("STRATA_PREFILL_ISSUER"),
            ple_batch_env: env_default_on("STRATA_PLE_BATCH"),
            ple_threads: env_str("STRATA_PLE_GATHER_THREADS")
                .map(|v| (c_atoi(&v) as i64).clamp(1, 32))
                .unwrap_or(8),
            group_gather_env: env_default_on("STRATA_PREFILL_GROUP_GATHER"),
            step_sync: env_first_is("STRATA_PF_STEP_SYNC", b'1'),
            kv_host_dma: env_first_is("STRATA_KV_HOST_DMA", b'1'),
            indexer_per_token: env_str("STRATA_INDEXER_PER_TOKEN").is_some(),
            select_old: env_str("STRATA_SELECT_OLD").is_some(),
            prompt_attn_old: env_str("STRATA_PROMPT_ATTN_OLD").is_some(),
            timing: env_str("STRATA_PREFILL_TIMING").is_some(),
            trace: env_str("STRATA_TRACE").is_some(),
            kv_prefetch: env_str("STRATA_KV_PREFETCH")
                .map(|v| c_atoi(&v) != 0)
                .unwrap_or(false),
            mtp_batch_off: env_first_is("STRATA_MTP_BATCH", b'0'),
            mtp_batch_ring_ok: env_str("STRATA_MTP_BATCH_RING")
                .map(|v| v.as_bytes().first() != Some(&b'0'))
                .unwrap_or(true),
            mtp_batch_f16: env_first_is("STRATA_MTP_BATCH_F16", b'1'),
            draft_timing: env_str("STRATA_DRAFT_TIMING").is_some(),
            debug,
        }
    })
}

// ============================================================================================ small helpers

const H2D: i32 = 1;
const D2D: i32 = 3;
const MMQ_GROUP: usize = 16;
const MMQ_TAIL: u64 = 4096;
const XN_PAD: i64 = 64;
const ZV_PAD: i64 = 64;
const DQ: usize = 2;
const STEP_COUNT: usize = 4;
const STEP_POS: usize = 0;
const STEP_NBID: usize = 2;
const KQ8_0: i32 = 8; // GGML_TYPE_Q8_0

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    /// The pointer (a method, so a closure captures the Send wrapper and not its raw field).
    fn get(&self) -> *mut T {
        self.0
    }
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap_or_default()
}

#[inline]
unsafe fn atomic_i32<'a>(p: *mut i32) -> &'a AtomicI32 {
    &*(p as *const AtomicI32)
}
#[inline]
unsafe fn atomic_u8<'a>(p: *mut u8) -> &'a AtomicU8 {
    &*(p as *const AtomicU8)
}

// STRATA_PREFILL_TIMING: the prompt path's GPU time by phase (prefill.cpp's PfTimer)
const PF_START: usize = 0;
const PF_HC: usize = 1;
const PF_GDN: usize = 2;
const PF_QSA: usize = 3;
const PF_QSA_IDX: usize = 4;
const PF_QSA_SEL: usize = 5;
const PF_QSA_ATTN: usize = 6;
const PF_ROUTER: usize = 7;
const PF_HOST_GROUP: usize = 8;
const PF_GATHER: usize = 9;
const PF_WAIT_COPY: usize = 10;
const PF_DEQUANT: usize = 11;
const PF_GEMM_GU: usize = 12;
const PF_GEMM_D: usize = 13;
const PF_COMBINE: usize = 14;
const PF_PLE: usize = 15;
const PF_KV_STAGE: usize = 16;
const PF_GDN_CONV: usize = 17;
const PF_GDN_REC: usize = 18;
const PF_GDN_OUT: usize = 19;
const PF_COUNT: usize = 20;
const PF_NAMES: [&str; PF_COUNT] = [
    "embed+steps",
    "hc read",
    "gdn",
    "qsa proj",
    "qsa indexer",
    "qsa select",
    "qsa attn",
    "router+shared",
    "host grouping",
    "gather",
    "wait copy",
    "dequant",
    "gemm gate/up",
    "gemm down",
    "combine",
    "ple",
    "kv stage",
    "gdn conv+gates",
    "gdn recurrence",
    "gdn out proj",
];

struct PfTimer {
    f: P2Ffi,
    on: bool,
    ev: Vec<*mut c_void>,
    ph: Vec<usize>,
    used: usize,
    ms: [f64; PF_COUNT],
}
impl PfTimer {
    fn new(f: P2Ffi, on: bool) -> Self {
        Self {
            f,
            on,
            ev: Vec::new(),
            ph: Vec::new(),
            used: 0,
            ms: [0.0; PF_COUNT],
        }
    }
    fn mark(&mut self, phase: usize, s: *mut c_void) {
        if !self.on {
            return;
        }
        if self.used == self.ev.len() {
            let mut e: *mut c_void = std::ptr::null_mut();
            unsafe { (self.f.event_create)(&mut e, 1) };
            self.ev.push(e);
            self.ph.push(0);
        }
        self.ph[self.used] = phase;
        unsafe { (self.f.event_record)(self.ev[self.used], s) };
        self.used += 1;
    }
    fn fold(&mut self) {
        if !self.on || self.used < 2 {
            return;
        }
        for i in 0..self.used - 1 {
            let mut t = 0.0f32;
            if unsafe { (self.f.event_elapsed)(&mut t, self.ev[i], self.ev[i + 1]) } != 0 {
                self.ms[self.ph[i]] += t as f64;
            }
        }
        self.ev.swap(0, self.used - 1);
        self.ph.swap(0, self.used - 1);
        self.used = 1;
    }
}
impl Drop for PfTimer {
    fn drop(&mut self) {
        for &e in &self.ev {
            unsafe { (self.f.event_destroy)(e) };
        }
    }
}

// ============================================================================================ per-object host state

/// The host side of one C++ Prefill object's prompt path: the routing scratch and the weights resolved once.
#[derive(Default)]
struct P2Host {
    wt: usize,
    weights: HashMap<(i64, &'static str), TkP2Weight>,
    fmts: Vec<TkP2ExpertFmt>,
    tok_host: Vec<i32>,
    steps_host: Vec<i32>,
    ids_host: Vec<i32>,
    slot_host: Vec<i32>,
    src_host: Vec<i32>,
    cnt: Vec<i32>,
    off: Vec<i32>,
    bounds_host: Vec<i32>,
}

// the raw pointers in it are the weights' device addresses, used only by the prompt stream that owns the object
unsafe impl Send for P2Host {}

impl P2Host {
    fn weight(
        &mut self,
        f: &P2Ffi,
        wt: *mut c_void,
        l: i64,
        suffix: &'static str,
    ) -> Result<TkP2Weight, String> {
        if self.wt != wt as usize {
            self.weights.clear();
            self.wt = wt as usize;
        }
        if let Some(w) = self.weights.get(&(l, suffix)) {
            return Ok(*w);
        }
        let c = cstr(suffix);
        let mut w = TkP2Weight::default();
        if unsafe { (f.weight)(wt, l, c.as_ptr(), &mut w) } == 0 {
            return Err(format!("{} is missing", wname(l, suffix)));
        }
        self.weights.insert((l, suffix), w);
        Ok(w)
    }
}

fn wname(l: i64, suffix: &str) -> String {
    format!("blk.{l}.{suffix}")
}

// ============================================================================================ the run

#[derive(Clone, Copy)]
struct StreamEntry {
    e: i32,
    blob: *const u8,
    job: i32,
    bytes: u64,
}
unsafe impl Send for StreamEntry {}
unsafe impl Sync for StreamEntry {}

/// One issue of stream entry `idx` (by the issuer thread, or inline): the copy into ring slot idx % ring, after the
/// slot's last reader recorded its `used` event.
#[derive(Clone, Copy)]
struct IssueCtx {
    f: P2Ffi,
    m: TkP2Layout,
    pf: *mut c_void,
    ring: usize,
}
unsafe impl Send for IssueCtx {}
impl IssueCtx {
    unsafe fn issue(&self, idx: usize, en: &StreamEntry) -> (f64, bool) {
        let (f, m) = (&self.f, &self.m);
        let sl = idx % self.ring;
        let th = Instant::now();
        if atomic_u8(m.stage_live.add(sl)).load(Ordering::Relaxed) != 0 {
            let uo = atomic_i32(m.used_of.add(sl)).load(Ordering::Relaxed) as usize;
            (f.stream_wait_event)(m.copy, *m.used.add(uo));
        }
        let dma = en.job < 0;
        if dma {
            (f.memcpy)(
                (*m.stage_dev.add(sl)).cast(),
                en.blob.cast(),
                en.bytes,
                H2D,
                m.copy,
            );
        } else {
            let hb = (f.stager_wait)(self.pf, en.job as i64);
            (f.memcpy)(
                (*m.stage_dev.add(sl)).cast(),
                hb.cast(),
                en.bytes,
                H2D,
                m.copy,
            );
            (f.stager_issued_one)(self.pf, en.job as i64, m.copy);
        }
        (f.event_record)(*m.copied.add(sl), m.copy);
        atomic_u8(m.stage_live.add(sl)).store(1, Ordering::Relaxed);
        (ms_since(th), dma)
    }
}

struct IssuerShared {
    issued: AtomicUsize,
    consumed: AtomicUsize,
    stop: AtomicBool,
}

/// What an early return must settle before the caller refills the borrowed slots (prefill.cpp's destructors).
struct Pending {
    f: P2Ffi,
    pf: SendPtr<c_void>,
    issuer: Option<(JoinHandle<(f64, i64, i64)>, Arc<IssuerShared>)>,
    stager_layer: bool,
    stager_chunk: bool,
    ple_next: Option<JoinHandle<Result<(), String>>>,
}
impl Drop for Pending {
    fn drop(&mut self) {
        if let Some((h, sh)) = self.issuer.take() {
            sh.stop.store(true, Ordering::Release);
            let _ = h.join();
        }
        if self.stager_layer || self.stager_chunk {
            unsafe { (self.f.stager_finish)(self.pf.0) };
        }
        if let Some(h) = self.ple_next.take() {
            let _ = h.join();
        }
    }
}

/// The device pointers of a run (the layout), with the kernels' C ABI, and the run's own state.
struct Ctx<'a> {
    f: &'a P2Ffi,
    pf: *mut c_void,
    m: TkP2Layout,
    e: &'static Env,
    pt: PfTimer,
    stats: *mut crate::qwen4exp::ffi::TkP2Stats,
    hcd_now: bool,
}

impl<'a> Ctx<'a> {
    #[allow(clippy::mut_from_ref)]
    fn st(&self) -> &mut crate::qwen4exp::ffi::TkP2Stats {
        unsafe { &mut *self.stats }
    }
    fn hc_pad(&self) -> bool {
        self.hcd_now || self.e.hcd_exact
    }
    fn bf16x2(&self) -> bool {
        self.m.f16_io == 0 && self.e.bf16x2_env != 0
    }
    fn progress(&self, what: &str, detail: i64, chunk: i64) {
        let c = cstr(what);
        unsafe { (self.f.progress_at)(c.as_ptr(), detail, chunk) };
    }
    /// bf16_proj: Y = X . W^T for a resident BF16 weight (+ the activations' BF16 remainder when given).
    #[allow(clippy::too_many_arguments)]
    fn bf16_proj(
        &self,
        w: &TkP2Weight,
        x: *const u16,
        y: *mut f32,
        t: i64,
        name: impl FnOnce() -> String,
        ldy: i64,
        x_lo: *const u16,
        ldx: i64,
    ) -> Result<(), String> {
        if w.bf16 == 0 || w.data.is_null() {
            return Err(format!("prefill: {} is not a resident BF16 tensor", name()));
        }
        let n = if w.ne1 > 0 { w.ne1 } else { 1 };
        unsafe {
            (self.f.gemm_bf16)(self.m.gemm, x, w.data.cast(), y, t, n, w.ne0, ldy, 0.0, ldx);
            if !x_lo.is_null() {
                (self.f.gemm_bf16)(
                    self.m.gemm,
                    x_lo,
                    w.data.cast(),
                    y,
                    t,
                    n,
                    w.ne0,
                    ldy,
                    1.0,
                    0,
                );
            }
        }
        Ok(())
    }
    /// native_proj: Y = X . W^T from the weight's GGUF blocks.
    fn native_proj(
        &self,
        w: &TkP2Weight,
        x: *const u16,
        y: *mut f32,
        t: i64,
        name: impl FnOnce() -> String,
        ldy: i64,
        ldx: i64,
    ) -> Result<(), String> {
        if w.native_data.is_null() {
            return Err(format!(
                "prefill: {} has no native GGUF blocks (run with --native)",
                name()
            ));
        }
        unsafe {
            (self.f.gemm_native)(
                self.m.gemm,
                x,
                w.native_type,
                w.native_data,
                y,
                t,
                w.ne1,
                w.ne0,
                ldy,
                0.0,
                ldx,
            )
        };
        Ok(())
    }
    fn cuda_err(&self) -> String {
        let mut b = [0 as c_char; 256];
        unsafe {
            (self.f.peek_error)(b.as_mut_ptr(), b.len());
            CStr::from_ptr(b.as_ptr()).to_string_lossy().into_owned()
        }
    }
}

/// PLE rows for chunk `c0` (prefill.cpp's ple_gather): the n-gram rows of every token, then one batched table read
/// (several threads on disjoint token ranges for a mapped table).  Runs on its own thread, a chunk ahead.
#[allow(clippy::too_many_arguments)]
fn ple_gather(
    f: P2Ffi,
    consts: SendPtr<c_void>,
    table: SendPtr<c_void>,
    direct: bool,
    threads: i64,
    tokens: SendPtr<i64>,
    n: i64,
    chunk: i64,
    prev0: [i32; 2],
    c0: i64,
    rows: SendPtr<u32>,
    emb_host: SendPtr<f32>,
    ple_n_heads: i64,
    ng_n_embd: i64,
) -> Result<(), String> {
    let t_n = chunk.min(n - c0);
    let at = |i: i64| -> i32 {
        if i < 2 {
            prev0[i as usize]
        } else {
            unsafe { *tokens.get().add((i - 2) as usize) as i32 }
        }
    };
    let mut pv = [at(c0), at(c0 + 1)];
    for t in 0..t_n {
        let tok = unsafe { *tokens.get().add((c0 + t) as usize) as i32 };
        unsafe {
            (f.ple_ngram_rows)(
                consts.get(),
                &tok,
                pv.as_ptr(),
                1,
                rows.get().add((t * ple_n_heads) as usize),
            );
        }
        pv[0] = pv[1];
        pv[1] = tok;
    }
    let gather = |a: i64, b: i64| -> Result<(), String> {
        let mut eb = [0 as c_char; 512];
        let ok = unsafe {
            (f.ple_gather_batch)(
                table.get(),
                rows.get().add((a * ple_n_heads) as usize),
                (b - a) as u64,
                emb_host.get().add((a * ng_n_embd) as usize),
                eb.as_mut_ptr(),
                eb.len(),
            )
        };
        if ok == 0 {
            return Err(unsafe { CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned() });
        }
        Ok(())
    };
    if !direct && threads > 1 && t_n >= 512 {
        let nth = threads.min(t_n / 256);
        let results: Vec<Result<(), String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nth)
                .map(|k| {
                    let (a, b) = (t_n * k / nth, t_n * (k + 1) / nth);
                    let g = &gather;
                    sc.spawn(move || g(a, b))
                })
                .collect();
            hs.into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("prefill: PLE gather thread panicked".into()))
                })
                .collect()
        });
        for r in results {
            r?;
        }
        return Ok(());
    }
    gather(0, t_n)
}

/// Why a mode is declined (the C++ path runs it), or None.
fn decline_reason(m: &TkP2Layout, e: &Env) -> Option<&'static str> {
    if m.metal != 0 {
        return Some("metal");
    }
    if m.stage_lb != 0 || m.stage_le != m.n_layers as i64 || m.has_next != 0 || m.has_hand_in != 0 {
        return Some("a layer split stage");
    }
    if m.has_peer != 0 || m.peer_portable != 0 {
        return Some("--peer-device / a stage helper");
    }
    if m.has_cpu_pool != 0 && m.cpu_share_on != 0 {
        return Some("the CPU expert share");
    }
    if m.has_embd_rows != 0 {
        return Some("picture rows");
    }
    if m.has_on_stage_chunk != 0 {
        return Some("on_stage_chunk");
    }
    if e.kv_prefetch {
        return Some("STRATA_KV_PREFETCH");
    }
    e.debug
}

/// The prompt: positions [pos0, pos0 + n) through all layers in chunks of the layout's T (prefill.cpp's run_impl).
/// Ok(false): declined before any work (the C++ path runs it).
unsafe fn run_prompt(
    f: &P2Ffi,
    pf: *mut c_void,
    host: &mut P2Host,
    tokens: *const i64,
    n: i64,
    pos0: i64,
) -> Result<bool, String> {
    let e = env();
    let mut m = TkP2Layout::default();
    if (f.bind)(pf, &mut m) <= 0 {
        return Ok(false);
    }
    if let Some(why) = decline_reason(&m, e) {
        if env_str("TALUS_FN_P2_SAY").is_some() {
            eprintln!(
                "talus-qwen4exp: qwen4exp-p2 declines this prompt ({why}); the C++ path runs it"
            );
        }
        return Ok(false);
    }
    let mut c = Ctx {
        f,
        pf,
        m,
        e,
        pt: PfTimer::new(*f, e.timing),
        stats: m.stats,
        hcd_now: e.pf_hcdown_mode == 1
            || (e.pf_hcdown_mode == 2 && pos0 + n + 1 >= e.pf_hcdown_min),
    };
    run_chunks(&mut c, host, tokens, n, pos0)?;
    Ok(true)
}

/// One stderr line per process the first time a prompt runs with TALUS_FN_HC_BF16 set: active, or why not.
fn hc16_say(fmt: i32, active: bool) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if fmt == 0 || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    if active {
        eprintln!(
            "talus-qwen4exp: TALUS_FN_HC_BF16={fmt} active (prompt-path residual streams in {}, opt-in numerics)",
            if fmt == 1 { "BF16" } else { "FP16" }
        );
    } else {
        eprintln!("talus-qwen4exp: TALUS_FN_HC_BF16={fmt} set but inactive (STRATA_GR_UNFUSED or no 16-bit R buffer)");
    }
}

#[allow(clippy::cognitive_complexity)]
unsafe fn run_chunks(
    c: &mut Ctx,
    host: &mut P2Host,
    tokens: *const i64,
    n: i64,
    pos0: i64,
) -> Result<(), String> {
    let f = *c.f;
    let m = c.m;
    let e = c.e;
    let pf = c.pf;
    let cs = m.cs;
    let t_start = Instant::now();
    let (lb, le) = (0i64, m.n_layers as i64);
    let n_layers = m.n_layers as i64;
    let ne = m.n_expert as i64;
    let (nn, hc, d, lr, k_used) = (m.n_embd, m.hc, m.n_embd * m.hc, m.hc_lr, m.n_used);
    let (zv, hv, qh, qkv, hd, idxh, idxd) = (
        m.value_dim,
        m.v_heads,
        m.n_head,
        m.n_head_kv,
        m.head_dim,
        m.idx_heads,
        m.idx_dim,
    );
    let eps = m.eps;
    // rb-num (opt-in TALUS_FN_HC_BF16=1 BF16 / =2 FP16): the 4-stream residual R in 16 bits, in the `xn` slot that
    // setup.rs / prefill.cpp carve for it; m.r (FP32) only backs the consumers that read R as floats (the PLE block,
    // control vectors, dumps, on_chunk): widened before them, narrowed after the ones that write it.
    let h16 = setup::hc16_fmt();
    let hc16 = h16 != 0 && !e.gr_unfused && !m.xn.is_null() && nn == 2560 && hc == 4;
    hc16_say(h16, hc16);
    let r16 = m.xn.cast::<u16>();
    let r16_widen = |t_n: i64| {
        if hc16 {
            (f.gr16_widen)(r16, m.r, t_n, h16, cs);
        }
    };
    let r16_narrow = |t_n: i64| {
        if hc16 {
            (f.gr16_narrow)(m.r, r16, t_n, h16, cs);
        }
    };
    let s = m.shapes;
    let stage = m.stage_ring as usize;
    let ring = m.ring as usize;
    let mut host_sync_ms = 0.0f64;
    let mut host_chunk_ms = 0.0f64;
    let mut host_setup_ms = 0.0f64;
    let mut prev: [i32; 2] = [*m.ple_prev, *m.ple_prev.add(1)];
    if host.fmts.len() != n_layers as usize {
        host.fmts = (0..n_layers)
            .map(|l| {
                let mut x = TkP2ExpertFmt::default();
                (f.expert_fmt)(l, &mut x);
                x.mmq_layer = if m.mmq_layer.is_null() {
                    0
                } else {
                    *m.mmq_layer.add(l as usize) as i32
                };
                x.mmq_fo = if m.mmq_fo.is_null() {
                    0
                } else {
                    *m.mmq_fo.add(l as usize) as i32
                };
                x
            })
            .collect();
    }
    let qsa_infos: Vec<TkP2QsaInfo> = (0..m.n_qsa_states)
        .map(|i| {
            let mut x = TkP2QsaInfo::default();
            (f.qsa_info)(m.ss, m.qsa_ord0 + i, &mut x);
            x
        })
        .collect();
    let qsa_of = |ord: i64| -> &TkP2QsaInfo { &qsa_infos[(ord - m.qsa_ord0) as usize] };
    let mut emb_rows = 0i64;
    let emb_kind = (f.embed_kind)(m.wt, m.g, &mut emb_rows);
    let tt = m.t as usize;
    host.tok_host.resize(tt.max(1), 0);
    host.steps_host.resize(tt * STEP_COUNT, 0);
    host.ids_host.resize(tt * k_used as usize, 0);
    host.slot_host.resize(tt * k_used as usize, 0);
    host.src_host.resize(tt * k_used as usize, 0);
    host.cnt.resize(ne as usize, 0);
    host.off.resize(ne as usize + 1, 0);
    let is_qsa = |l: i64| -> bool { *m.is_qsa.add(l as usize) != 0 };
    // the MMQ row table lives in the borrowed cache slots, which the refill after a prompt overwrites: every prompt
    if !m.ids_identity.is_null() {
        (f.mmq_iota)(m.ids_identity, m.t * k_used, cs);
    }
    let ple_on = m.ple_ready != 0 && lb <= 1 && 1 < le;
    let ple_batch = ple_on && e.ple_batch_env && m.ple_batch_ok != 0;
    let prev0 = prev;
    let mut pending = Pending {
        f,
        pf: SendPtr(pf),
        issuer: None,
        stager_layer: false,
        stager_chunk: false,
        ple_next: None,
    };
    let spawn_gather = |c0: i64, buf: usize| -> JoinHandle<Result<(), String>> {
        let (consts, table) = (SendPtr(m.ple_consts), SendPtr(m.ple_table));
        let (tok, rows, emb_h) = (
            SendPtr(tokens as *mut i64),
            SendPtr(m.ple_rows[buf]),
            SendPtr(m.ple_emb_host[buf]),
        );
        let (direct, threads, mt, pnh, nge) = (
            m.ple_direct != 0,
            e.ple_threads,
            m.t,
            m.ple_n_heads,
            m.ng_n_embd,
        );
        std::thread::spawn(move || {
            (f.pin_thread)();
            ple_gather(
                f, consts, table, direct, threads, tok, n, mt, prev0, c0, rows, emb_h, pnh, nge,
            )
        })
    };
    let mut ple_buf = 0usize;
    let dump_dir = env_str("TALUS_FN_DUMP").filter(|v| !v.is_empty());

    let mut c0 = 0i64;
    while c0 < n {
        if m.has_should_stop != 0 && (f.should_stop)(pf) != 0 {
            return Err("cancelled".into());
        }
        if e.trace {
            eprintln!("strata trace: prompt chunk {c0} of {n}");
        }
        let t_n = m.t.min(n - c0);
        let p0 = pos0 + c0;
        c.progress(
            "reading the prompt (batched): preparing the chunk from token",
            p0,
            -1,
        );
        c.st().chunks += 1;
        c.pt.mark(PF_START, cs);
        let tsetup = Instant::now();
        // ---- embeddings, broadcast to the four streams (C-4: the whole chunk in one gather where it can)
        let mut batched = !m.tok_dev.is_null() && emb_kind != 0;
        for t in 0..t_n {
            if !batched {
                break;
            }
            let tok = *tokens.add((c0 + t) as usize);
            if tok < 0 || (emb_kind == 2 && tok >= emb_rows) {
                batched = false;
            } else {
                host.tok_host[t as usize] = tok as i32;
            }
        }
        if batched {
            if (f.tok_upload)(pf, m.tok_dev, host.tok_host.as_ptr(), t_n as u64, cs) == 0 {
                return Err("prefill: the token id upload failed".into());
            }
            (f.embed_gather)(m.wt, m.tok_dev, t_n, m.emb, cs);
        } else {
            for t in 0..t_n {
                let mut eb = [0 as c_char; 512];
                if (f.embed_row)(
                    m.wt,
                    m.g,
                    *tokens.add((c0 + t) as usize),
                    m.emb.add((t * nn) as usize),
                    cs,
                    eb.as_mut_ptr(),
                    eb.len(),
                ) == 0
                {
                    return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
                }
            }
        }
        if hc16 {
            (f.gr16_broadcast)(m.emb, r16, t_n, h16, cs);
        } else {
            (f.gr_broadcast)(m.emb, m.r, t_n, cs);
        }
        // ---- the PLE rows of the whole chunk on a thread (a later chunk's were read ahead during the previous one)
        if ple_on && pending.ple_next.is_none() {
            pending.ple_next = Some(spawn_gather(c0, ple_buf));
        }
        let mut ple_pending = ple_on;
        for t in 0..t_n {
            prev[0] = prev[1];
            prev[1] = *tokens.add((c0 + t) as usize) as i32;
        }
        // ---- the QSA step records of every position in the chunk (qsa_step_fill)
        for t in 0..t_n as usize {
            let pos = p0 + t as i64;
            let n_kv = pos + 1;
            let w = (s.idx_top_k + s.idx_block - 1).min(n_kv);
            let st = &mut host.steps_host[t * STEP_COUNT..(t + 1) * STEP_COUNT];
            st[0] = pos as i32;
            st[1] = n_kv as i32;
            st[2] = (n_kv / s.idx_block) as i32;
            st[3] = w as i32;
        }
        if (f.steps_upload)(
            pf,
            m.steps_dev,
            host.steps_host.as_ptr(),
            (t_n as u64) * STEP_COUNT as u64,
            cs,
        ) == 0
        {
            return Err("prefill: the QSA step records' upload failed".into());
        }
        let mut qsa_index = 0i64; // global ordinals (LB = 0)
        let mut gdn_index = 0i64;
        // ---- step 3: this chunk's stream - every non-resident expert of every layer in id order through the ring
        let stream_all = ring > stage && t_n >= m.stream_all_min && !m.src.is_null();
        let mut seq: Vec<StreamEntry> = Vec::new();
        let mut seq_start: Vec<usize> = Vec::new();
        let resident = |l: i64, ex: i64| -> bool {
            !m.host_res.is_null()
                && !m.cache.is_null()
                && *m.host_res.add((l * ne + ex) as usize) >= 0
        };
        if stream_all {
            seq_start.resize(n_layers as usize + 1, 0);
            let mut js: Vec<TkP2StagerJob> = Vec::new();
            for l in lb..le {
                seq_start[l as usize] = seq.len();
                let bytes = host.fmts[l as usize].blob_bytes;
                for ex in 0..ne {
                    if resident(l, ex) {
                        continue;
                    }
                    let mut job = -1i32;
                    let mut b: *const u8 = std::ptr::null();
                    if (f.src_transient)(m.src, l, ex) != 0 {
                        job = js.len() as i32;
                        js.push(TkP2StagerJob {
                            src: std::ptr::null(),
                            bytes,
                            from_source: 1,
                            l: l as i32,
                            e: ex as i32,
                            pad0: 0,
                        });
                    } else {
                        b = (f.src_blob_stable)(m.src, l, ex);
                        if b.is_null() {
                            return Err("prefill: expert source has no blob".into());
                        }
                        if (f.src_pinned)(m.src, l, ex) == 0 {
                            job = js.len() as i32;
                            js.push(TkP2StagerJob {
                                src: b,
                                bytes,
                                from_source: 0,
                                l: 0,
                                e: 0,
                                pad0: 0,
                            });
                        }
                    }
                    seq.push(StreamEntry {
                        e: ex as i32,
                        blob: b,
                        job,
                        bytes,
                    });
                }
            }
            for l in le..=n_layers {
                seq_start[l as usize] = seq.len();
            }
            if !js.is_empty() {
                (f.stager_start)(pf, js.as_ptr(), js.len() as i64);
            }
            pending.stager_chunk = true;
        }
        let seq = Arc::new(seq);
        let mut issued = 0usize;
        let mut consumed = 0usize;
        // D-5: the stream is issued by its own host thread (STRATA_PREFILL_ISSUER=0: inline)
        let threaded_issue = stream_all && e.issuer_on;
        let shared = Arc::new(IssuerShared {
            issued: AtomicUsize::new(0),
            consumed: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
        });
        let issuer_ctx = IssueCtx { f, m, pf, ring };
        if threaded_issue {
            let (sq, sh) = (seq.clone(), shared.clone());
            let dev = m.device;
            let issue = issuer_ctx;
            let h = std::thread::spawn(move || {
                (f.pin_thread)();
                (f.set_device)(dev);
                let (mut ms, mut streamed, mut dma) = (0.0f64, 0i64, 0i64);
                for idx in 0..sq.len() {
                    while idx >= sh.consumed.load(Ordering::Acquire) + ring {
                        if sh.stop.load(Ordering::Acquire) {
                            return (ms, streamed, dma);
                        }
                        std::thread::yield_now();
                    }
                    let (t, was_dma) = issue.issue(idx, &sq[idx]);
                    ms += t;
                    streamed += 1;
                    if was_dma {
                        dma += 1;
                    }
                    sh.issued.store(idx + 1, Ordering::Release);
                }
                (ms, streamed, dma)
            });
            pending.issuer = Some((h, shared.clone()));
        }
        // the inline issuer: entries up to `limit`
        let issue_until = |limit: usize, c: &mut Ctx, issued: &mut usize| {
            let limit = limit.min(seq.len());
            while *issued < limit {
                let (t, was_dma) = issuer_ctx.issue(*issued, &seq[*issued]);
                let st = c.st();
                if was_dma {
                    st.experts_dma += 1;
                }
                st.ms_experts_host += t;
                st.experts_streamed += 1;
                *issued += 1;
            }
        };
        if !threaded_issue && stream_all {
            issue_until(ring, c, &mut issued); // layer 0's first experts, behind the embedding and the PLE
        }
        // the consumer's side: entry k's copy is on the copy stream, then k is given back
        let wait_issued = |k: usize| {
            if threaded_issue {
                while shared.issued.load(Ordering::Acquire) <= k {
                    std::thread::yield_now();
                }
            }
        };
        macro_rules! give_back {
            ($upto:expr) => {{
                let upto: usize = $upto;
                if threaded_issue {
                    shared.consumed.store(upto, Ordering::Release);
                } else {
                    issue_until(upto + ring, c, &mut issued);
                }
            }};
        }
        host_setup_ms += ms_since(tsetup);
        let mut normed = false; // F-2: the previous half's write already normed R for this half
        let pf_step = |c: &Ctx, what: &str, layer: i64| {
            if !e.step_sync {
                return;
            }
            c.progress(what, layer, p0);
            let ts = Instant::now();
            let a = (f.stream_sync)(cs);
            let b = (f.stream_sync)(m.copy);
            let ms = ms_since(ts);
            if ms > 250.0 || a == 0 || b == 0 {
                eprintln!(
                    "strata pf-step: chunk from token {p0}, layer {layer}: {what} took {ms:.0} ms ({} / {})",
                    if a != 0 { "no error" } else { "error" },
                    if b != 0 { "no error" } else { "error" }
                );
            }
        };
        let dump_floats = (t_n.min(16) * d) as usize;
        let dump = |tag: &str, layer: i32| {
            if dump_dir.is_some() {
                r16_widen(t_n);
                let ct = cstr(tag);
                (f.dump_tensor)(ct.as_ptr(), layer, p0, m.r, dump_floats, 1, cs);
            }
        };
        dump(&format!("prefill_p{p0}_input_R"), -1);
        // the chunk's rows onto the device just before layer 1 reads them, and the next chunk's gather started
        let ple_land = |c: &mut Ctx,
                        pending: &mut Pending,
                        ple_pending: &mut bool,
                        ple_buf: &mut usize|
         -> Result<(), String> {
            if !*ple_pending {
                return Ok(());
            }
            *ple_pending = false;
            let tp = Instant::now();
            let h = pending
                .ple_next
                .take()
                .ok_or_else(|| "prefill: the PLE gather was not started".to_string())?;
            h.join()
                .unwrap_or_else(|_| Err("prefill: the PLE gather thread panicked".into()))?;
            let b = *ple_buf;
            let bytes = (t_n * nn * 4) as u64;
            let up = if m.ple_pageable[b] == 0 {
                (f.memcpy)(m.ple_emb.cast(), m.ple_emb_host[b].cast(), bytes, H2D, cs)
            } else {
                (f.upring_h2d)(pf, m.ple_emb.cast(), m.ple_emb_host[b].cast(), bytes, cs)
            };
            if up == 0 || (f.event_record)(m.ple_copied[b], cs) == 0 {
                return Err(format!(
                    "prefill: the PLE rows' upload failed: {}",
                    c.cuda_err()
                ));
            }
            if c0 + m.t < n {
                // the other buffer's upload (a chunk ago) is done before the SSD thread refills it
                if (f.event_sync)(m.ple_copied[b ^ 1]) == 0 {
                    return Err(format!(
                        "prefill: the PLE rows' upload failed: {}",
                        c.cuda_err()
                    ));
                }
                pending.ple_next = Some(spawn_gather(c0 + m.t, b ^ 1));
            }
            *ple_buf ^= 1;
            c.st().ms_ple += ms_since(tp);
            Ok(())
        };

        for l in lb..le {
            (f.progress_beat)();
            if l > lb {
                pf_step(
                    c,
                    "reading the prompt (batched, step sync): the experts and the rest of layer",
                    l - 1,
                );
            }
            c.progress("reading the prompt (batched): layer", l, p0);
            if l == lb.max(1) {
                ple_land(c, &mut pending, &mut ple_pending, &mut ple_buf)?;
            }
            // ---- the PLE block at layer 1 (on FP32 R: widened / narrowed around it under TALUS_FN_HC_BF16)
            if l == 1 && ple_on {
                r16_widen(t_n);
            }
            if l == 1 && ple_on && ple_batch {
                c.pt.mark(PF_PLE, cs);
                let tp = Instant::now();
                let hdim = m.ng_hc_dim;
                let per_token =
                    ((3 * hdim + nn + 4) * 4 + nn * if c.bf16x2() { 4 } else { 2 } + 4096) as u64;
                let sb = t_n.min((m.region_bytes / per_token) as i64);
                let mut s0 = 0i64;
                while s0 < t_n {
                    let nb = sb.min(t_n - s0);
                    let mut q = m.region;
                    let mut carve_f = |cnt: i64| -> *mut f32 {
                        let p = q as *mut f32;
                        q = q.add(((cnt as usize) * 4 + 255) & !255usize);
                        p
                    };
                    let key = carve_f(nb * hdim);
                    let qn = carve_f(nb * hdim);
                    let gated = carve_f(nb * hdim);
                    let val = carve_f(nb * nn);
                    let gate = carve_f(nb * 4);
                    let e16 = carve_f(nb * nn / 2) as *mut u16;
                    let e16_lo = if c.bf16x2() {
                        carve_f(nb * nn / 2) as *mut u16
                    } else {
                        std::ptr::null_mut()
                    };
                    let emb = m.ple_emb.add((s0 * nn) as usize);
                    if !m.ple_key_bf16.is_null() {
                        (f.to_bf16)(emb, e16, nb * nn, cs, e16_lo);
                        (f.gemm_bf16)(m.gemm, e16, m.ple_key_bf16, key, nb, hdim, nn, 0, 0.0, 0);
                        if !e16_lo.is_null() {
                            (f.gemm_bf16)(
                                m.gemm,
                                e16_lo,
                                m.ple_key_bf16,
                                key,
                                nb,
                                hdim,
                                nn,
                                0,
                                1.0,
                                0,
                            );
                        }
                    } else {
                        (f.to_f16)(emb, e16, nb * nn, cs);
                        (f.gemm_native)(
                            m.gemm,
                            e16,
                            m.ple_key_native_type,
                            m.ple_key_native_data,
                            key,
                            nb,
                            hdim,
                            nn,
                            0,
                            0.0,
                            0,
                        );
                        (f.to_bf16)(emb, e16, nb * nn, cs, e16_lo);
                    }
                    (f.gemm_bf16)(m.gemm, e16, m.ple_value_bf16, val, nb, nn, nn, 0, 0.0, 0);
                    if !e16_lo.is_null() {
                        (f.gemm_bf16)(m.gemm, e16_lo, m.ple_value_bf16, val, nb, nn, nn, 0, 1.0, 0);
                    }
                    let mut eb = [0 as c_char; 512];
                    if (f.ple_postops_batch)(
                        m.ple_weights,
                        key,
                        m.r.add((s0 * d) as usize),
                        val,
                        m.ple_hist,
                        qn,
                        gated,
                        gate,
                        nb as i32,
                        cs,
                        eb.as_mut_ptr(),
                        eb.len(),
                    ) == 0
                    {
                        return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
                    }
                    s0 += sb;
                }
                c.st().ms_ple += ms_since(tp);
            } else if l == 1 && ple_on {
                c.pt.mark(PF_PLE, cs);
                let tp = Instant::now();
                for t in 0..t_n {
                    let mut eb = [0 as c_char; 512];
                    let row = m.r.add((t * d) as usize);
                    if (f.ple_block)(
                        m.ple_weights,
                        m.ple_emb.add((t * nn) as usize),
                        row,
                        m.ple_hist,
                        m.ple_norm,
                        row,
                        m.ple_scratch,
                        cs,
                        eb.as_mut_ptr(),
                        eb.len(),
                    ) == 0
                    {
                        return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
                    }
                    (f.ple_history_advance)(m.ple_hist, m.ple_norm, cs);
                }
                c.st().ms_ple += ms_since(tp);
            }
            if l == 1 && ple_on {
                r16_narrow(t_n);
            }
            for half in 0..2 {
                // ---- the hyper-connection read of this half
                let (sn, sd, su, si): (&'static str, &'static str, &'static str, &'static str) =
                    if half == 0 {
                        (
                            "hc_attn_norm.weight",
                            "hc_attn_down.weight",
                            "hc_attn_up.weight",
                            "hc_attn_inject.weight",
                        )
                    } else {
                        (
                            "hc_ffn_norm.weight",
                            "hc_ffn_down.weight",
                            "hc_ffn_up.weight",
                            "hc_ffn_inject.weight",
                        )
                    };
                let wn = host.weight(&f, m.wt, l, sn)?;
                let mut wd = host.weight(&f, m.wt, l, sd)?;
                let mut wu = host.weight(&f, m.wt, l, su)?;
                let wi = host.weight(&f, m.wt, l, si)?;
                // STRATA_HC_Q8 without a bf16 copy: the matrices are dequantized per half
                let mut hc_q8 = false;
                if wd.data.is_null() && wd.hc_q8 != 0 && wu.data.is_null() && wu.hc_q8 != 0 {
                    wd.data = (f.hc_bf16_view)(wd.wref, cs, 0, 0).cast();
                    wu.data = (f.hc_bf16_view)(wu.wref, cs, 1, 0).cast();
                    if wd.data.is_null() || wu.data.is_null() {
                        return Err("prefill: the Q8_0 hyper-connection projections could not be dequantized".into());
                    }
                    hc_q8 = true;
                }
                c.pt.mark(PF_HC, cs);
                let ldx = if c.hc_pad()
                    && t_n >= e.pf_switch_min_t.max(64)
                    && !e.gr_unfused
                    && m.xn16_lo.is_null()
                {
                    d + XN_PAD
                } else {
                    d
                };
                if e.gr_unfused {
                    (f.gr_norm)(m.r, wn.data.cast(), eps, m.xn, m.xn16, t_n, cs, m.xn16_lo);
                } else if !normed && hc16 {
                    (f.gr16_norm_rs)(
                        r16,
                        wn.data.cast(),
                        eps,
                        m.grs,
                        m.xn16,
                        t_n,
                        h16,
                        cs,
                        m.xn16_lo,
                        ldx,
                    );
                } else if !normed {
                    (f.gr_norm_rs)(
                        m.r,
                        wn.data.cast(),
                        eps,
                        m.grs,
                        m.xn16,
                        t_n,
                        cs,
                        m.xn16_lo,
                        ldx,
                    );
                }
                normed = false;
                let mut hcd = false;
                let mut hdown = false;
                if ldx != d && !c.hcd_now {
                    hdown = wd.bf16 != 0
                        && !wd.data.is_null()
                        && wd.ne0 == d
                        && wd.ne1 == lr
                        && (f.gemm_bf16_hcd_exact)(
                            m.gemm,
                            m.xn16,
                            ldx,
                            wd.data.cast(),
                            m.lo,
                            t_n,
                            lr,
                            d,
                        ) != 0;
                } else if ldx != d {
                    if !(wd.bf16 != 0
                        && !wd.data.is_null()
                        && wd.ne0 == d
                        && wd.ne1 == lr
                        && wi.bf16 != 0
                        && !wi.data.is_null()
                        && wi.ne0 == d
                        && wi.ne1 == hc
                        && (f.pf_hcdown_bf16)(
                            m.xn16,
                            ldx,
                            wd.data.cast(),
                            wi.data.cast(),
                            lr,
                            hc,
                            m.lo,
                            m.inj,
                            t_n,
                            d,
                            cs,
                        ) != 0)
                    {
                        return Err("prefill: STRATA_PF_HCDOWN could not run the hyper-connection projections".into());
                    }
                    hcd = true;
                }
                let ldx_arg = if ldx != d { ldx } else { 0 };
                if !hcd && !hdown {
                    c.bf16_proj(
                        &wd,
                        m.xn16,
                        m.lo,
                        t_n,
                        || wname(l, sd),
                        0,
                        m.xn16_lo,
                        ldx_arg,
                    )?;
                }
                (f.gr_silu)(m.lo, m.lo16, t_n, cs, m.lo16_lo);
                let mut upmixed = false;
                if e.hc_upmix
                    && t_n >= e.pf_switch_min_t
                    && !e.gr_unfused
                    && m.lo16_lo.is_null()
                    && m.mixed_bf_lo.is_null()
                    && wu.bf16 != 0
                    && !wu.data.is_null()
                    && wu.ne0 == lr
                    && wu.ne1 == d
                {
                    upmixed = if hc16 {
                        (f.gr16_upmix)(
                            m.lo16,
                            wu.data.cast(),
                            r16,
                            m.grs,
                            wn.data.cast(),
                            m.mixed,
                            m.mixed_bf,
                            m.mixed_h,
                            t_n,
                            h16,
                            cs,
                        ) != 0
                    } else {
                        (f.gr_upmix)(
                            m.lo16,
                            wu.data.cast(),
                            m.r,
                            m.grs,
                            wn.data.cast(),
                            m.mixed,
                            m.mixed_bf,
                            m.mixed_h,
                            t_n,
                            cs,
                        ) != 0
                    };
                }
                if !upmixed {
                    c.bf16_proj(&wu, m.lo16, m.gated, t_n, || wname(l, su), 0, m.lo16_lo, 0)?;
                }
                if !hcd {
                    c.bf16_proj(
                        &wi,
                        m.xn16,
                        m.inj,
                        t_n,
                        || wname(l, si),
                        0,
                        m.xn16_lo,
                        ldx_arg,
                    )?;
                }
                if upmixed {
                } else if e.gr_unfused {
                    (f.gr_mix)(
                        m.xn,
                        m.gated,
                        m.mixed,
                        m.mixed_bf,
                        t_n,
                        cs,
                        m.mixed_h,
                        m.mixed_bf_lo,
                    );
                } else if hc16 {
                    (f.gr16_mix_r)(
                        r16,
                        m.grs,
                        wn.data.cast(),
                        m.gated,
                        m.mixed,
                        m.mixed_bf,
                        t_n,
                        h16,
                        cs,
                        m.mixed_h,
                        m.mixed_bf_lo,
                    );
                } else {
                    (f.gr_mix_r)(
                        m.r,
                        m.grs,
                        wn.data.cast(),
                        m.gated,
                        m.mixed,
                        m.mixed_bf,
                        t_n,
                        cs,
                        m.mixed_h,
                        m.mixed_bf_lo,
                    );
                }
                if hc_q8 {
                    (f.hc_bf16_release)(cs, 0, 0);
                    (f.hc_bf16_release)(cs, 1, 0);
                }
                if half == 0 && !is_qsa(l) {
                    // ======================= GDN =======================
                    let wqkv = host.weight(&f, m.wt, l, "attn_qkv.weight")?;
                    let wg = host.weight(&f, m.wt, l, "attn_gate.weight")?;
                    let wo = host.weight(&f, m.wt, l, "ssm_out.weight")?;
                    let wa = host.weight(&f, m.wt, l, "ssm_alpha.weight")?;
                    let wb = host.weight(&f, m.wt, l, "ssm_beta.weight")?;
                    let wc = host.weight(&f, m.wt, l, "ssm_conv1d.weight")?;
                    let wnm = host.weight(&f, m.wt, l, "ssm_norm.weight")?;
                    let wdt = host.weight(&f, m.wt, l, "ssm_dt.bias")?;
                    let wsa = host.weight(&f, m.wt, l, "ssm_a")?;
                    c.pt.mark(PF_GDN, cs);
                    let state = m
                        .gdn_state
                        .add(((gdn_index - m.gdn_ord0) * m.gdn_floats) as usize);
                    let conv = state.add(m.gdn_state_floats as usize);
                    c.native_proj(
                        &wqkv,
                        m.mixed_h,
                        m.qkv,
                        t_n,
                        || wname(l, "attn_qkv.weight"),
                        0,
                        0,
                    )?;
                    c.native_proj(
                        &wg,
                        m.mixed_h,
                        m.z,
                        t_n,
                        || wname(l, "attn_gate.weight"),
                        0,
                        0,
                    )?;
                    c.bf16_proj(
                        &wa,
                        m.mixed_bf,
                        m.ab,
                        t_n,
                        || wname(l, "ssm_alpha.weight"),
                        2 * hv,
                        m.mixed_bf_lo,
                        0,
                    )?;
                    c.bf16_proj(
                        &wb,
                        m.mixed_bf,
                        m.ab.add(hv as usize),
                        t_n,
                        || wname(l, "ssm_beta.weight"),
                        2 * hv,
                        m.mixed_bf_lo,
                        0,
                    )?;
                    c.pt.mark(PF_GDN_CONV, cs);
                    (f.gdn_gates)(
                        m.ab,
                        wdt.data.cast(),
                        wsa.data.cast(),
                        m.gate,
                        m.beta,
                        t_n,
                        cs,
                    );
                    (f.gdn_conv)(conv, m.qkv, wc.data.cast(), m.hbuf, t_n, eps, cs);
                    c.pt.mark(PF_GDN_REC, cs);
                    let ld_y = if e.pf_pad && t_n >= e.pf_switch_min_t.max(64) {
                        zv + ZV_PAD
                    } else {
                        0
                    };
                    (f.gdn_recurrence)(
                        state,
                        m.hbuf,
                        m.gate,
                        m.beta,
                        m.z,
                        wnm.data.cast(),
                        eps,
                        m.y,
                        m.y_h,
                        t_n,
                        cs,
                        ld_y,
                    );
                    c.pt.mark(PF_GDN_OUT, cs);
                    c.native_proj(
                        &wo,
                        m.y_h,
                        m.bo,
                        t_n,
                        || wname(l, "ssm_out.weight"),
                        0,
                        ld_y,
                    )?;
                    gdn_index += 1;
                } else if half == 0 {
                    // ======================= QSA =======================
                    let st = *qsa_of(qsa_index);
                    let wq = host.weight(&f, m.wt, l, "attn_q.weight")?;
                    let wk = host.weight(&f, m.wt, l, "attn_k.weight")?;
                    let wv = host.weight(&f, m.wt, l, "attn_v.weight")?;
                    let wo = host.weight(&f, m.wt, l, "attn_output.weight")?;
                    let wik = host.weight(&f, m.wt, l, "indexer.k_proj.weight")?;
                    let wiq = host.weight(&f, m.wt, l, "indexer.q_proj.weight")?;
                    let wqn = host.weight(&f, m.wt, l, "attn_q_norm.weight")?;
                    let wkn = host.weight(&f, m.wt, l, "attn_k_norm.weight")?;
                    let wiqn = host.weight(&f, m.wt, l, "indexer.q_norm.weight")?;
                    let wikn = host.weight(&f, m.wt, l, "indexer.k_norm.weight")?;
                    c.pt.mark(PF_QSA, cs);
                    c.native_proj(
                        &wk,
                        m.mixed_h,
                        m.kc,
                        t_n,
                        || wname(l, "attn_k.weight"),
                        0,
                        0,
                    )?;
                    c.native_proj(
                        &wv,
                        m.mixed_h,
                        m.vc,
                        t_n,
                        || wname(l, "attn_v.weight"),
                        0,
                        0,
                    )?;
                    c.native_proj(
                        &wq,
                        m.mixed_h,
                        m.qf,
                        t_n,
                        || wname(l, "attn_q.weight"),
                        0,
                        0,
                    )?;
                    c.bf16_proj(
                        &wik,
                        m.mixed_bf,
                        m.idx_raw,
                        t_n,
                        || wname(l, "indexer.k_proj.weight"),
                        0,
                        m.mixed_bf_lo,
                        0,
                    )?;
                    c.bf16_proj(
                        &wiq,
                        m.mixed_bf,
                        m.q_idx,
                        t_n,
                        || wname(l, "indexer.q_proj.weight"),
                        0,
                        m.mixed_bf_lo,
                        0,
                    )?;
                    (f.rms_rows)(m.kc, wkn.data.cast(), t_n * qkv, hd, hd, eps, cs);
                    (f.rope)(m.kc, t_n, qkv, hd, qkv * hd, p0, cs);
                    let staged = st.kv_mode == 1;
                    let stage_pools = TkP2AttnPools {
                        k_pool: m.stage.k_pool,
                        v_pool: m.stage.v_pool,
                        k_q: m.stage.k_q,
                        v_q: m.stage.v_q,
                        k_scale: m.stage.k_scale,
                        v_scale: m.stage.v_scale,
                        k_q4: m.stage.k_q4,
                        v_q4: m.stage.v_q4,
                        page_table: m.ident_table,
                    };
                    if staged {
                        c.pt.mark(PF_KV_STAGE, cs);
                        (f.kv_stage_from_host)(
                            &stage_pools,
                            &st.host,
                            st.fmt,
                            (p0 + s.page_size - 1) / s.page_size,
                            &s,
                            cs,
                        );
                        c.pt.mark(PF_QSA, cs);
                        pf_step(c, "reading the prompt (batched, step sync): the K/V staged from RAM at layer", l);
                    }
                    let host_by_dma = staged && e.kv_host_dma;
                    let host_w: *const TkP2KvPools = if host_by_dma {
                        std::ptr::null()
                    } else {
                        &st.host
                    };
                    let stage_w: *const TkP2KvPools =
                        if staged { &m.stage } else { std::ptr::null() };
                    if st.kv_hybrid != 0 {
                        // K8V4: K INT8 unrotated, V rotated Q4_0
                        (f.fwht256)(m.vc, t_n * 2, cs);
                        let half_k = |h: &TkP2KvPools| TkP2KvPools {
                            k_q: h.k_q,
                            v_q: h.k_q,
                            k_scale: h.k_scale,
                            v_scale: h.k_scale,
                            ..TkP2KvPools::default()
                        };
                        let half_v = |h: &TkP2KvPools| TkP2KvPools {
                            k_q4: h.v_q4,
                            v_q4: h.v_q4,
                            ..TkP2KvPools::default()
                        };
                        let (hk, hvv, sk, sv) = (
                            half_k(&st.host),
                            half_v(&st.host),
                            half_k(&m.stage),
                            half_v(&m.stage),
                        );
                        let mirror = st.host_present != 0;
                        (f.kv_append)(
                            m.kc,
                            m.kc,
                            t_n,
                            p0,
                            st.page_table,
                            s.page_size,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            st.k_q,
                            st.k_q,
                            st.k_scale,
                            st.k_scale,
                            cs,
                            if mirror { &hk } else { std::ptr::null() },
                            if staged { &sk } else { std::ptr::null() },
                        );
                        (f.kv_append_q4)(
                            st.v_q4,
                            st.v_q4,
                            st.page_table,
                            p0,
                            t_n,
                            m.vc,
                            m.vc,
                            &s,
                            cs,
                            if mirror { &hvv } else { std::ptr::null() },
                            if staged { &sv } else { std::ptr::null() },
                        );
                    } else {
                        if st.kv_rot != 0 {
                            (f.fwht256)(m.kc, t_n * 2, cs);
                            (f.fwht256)(m.vc, t_n * 2, cs);
                        }
                        if st.kv_q4 != 0 {
                            (f.kv_append_q4)(
                                st.k_q4,
                                st.v_q4,
                                st.page_table,
                                p0,
                                t_n,
                                m.kc,
                                m.vc,
                                &s,
                                cs,
                                host_w,
                                stage_w,
                            );
                        } else {
                            let i8m = st.kv_int8 != 0;
                            (f.kv_append)(
                                m.kc,
                                m.vc,
                                t_n,
                                p0,
                                st.page_table,
                                s.page_size,
                                if i8m { std::ptr::null_mut() } else { st.k_pool },
                                if i8m { std::ptr::null_mut() } else { st.v_pool },
                                st.k_q,
                                st.v_q,
                                st.k_scale,
                                st.v_scale,
                                cs,
                                host_w,
                                stage_w,
                            );
                        }
                        if host_by_dma {
                            (f.kv_unstage_to_host)(
                                &stage_pools,
                                &st.host,
                                st.fmt,
                                p0 / s.page_size,
                                (p0 + t_n + s.page_size - 1) / s.page_size,
                                &s,
                                cs,
                            );
                        }
                    }
                    if staged {
                        pf_step(
                            c,
                            "reading the prompt (batched, step sync): the K/V append at layer",
                            l,
                        );
                    }
                    (f.split_q)(m.qf, m.q, t_n, cs);
                    (f.rms_rows)(m.q, wqn.data.cast(), t_n * qh, hd, hd, eps, cs);
                    (f.rope)(m.q, t_n, qh, hd, qh * hd, p0, cs);
                    if st.kv_rot != 0 {
                        (f.fwht256)(m.q, t_n * qh, cs);
                    }
                    (f.rms_rows)(m.q_idx, wiqn.data.cast(), t_n * idxh, idxd, idxd, eps, cs);
                    (f.rope)(m.q_idx, t_n, idxh, idxd, idxh * idxd, p0, cs);
                    c.pt.mark(PF_QSA_IDX, cs);
                    let mut eb = [0 as c_char; 512];
                    if !e.indexer_per_token {
                        if (f.indexer_append_batch)(
                            m.idx_raw,
                            t_n,
                            p0,
                            wikn.data.cast(),
                            eps,
                            &st.idx,
                            &s,
                            st.max_cells,
                            cs,
                            eb.as_mut_ptr(),
                            eb.len(),
                        ) == 0
                        {
                            return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
                        }
                    } else {
                        for t in 0..t_n {
                            let step_t = m.steps_dev.add(t as usize * STEP_COUNT);
                            if (f.indexer_append)(
                                m.idx_raw.add((t * idxd) as usize),
                                step_t.add(STEP_POS),
                                wikn.data.cast(),
                                eps,
                                &st.idx,
                                &s,
                                st.max_cells,
                                cs,
                                eb.as_mut_ptr(),
                                eb.len(),
                            ) == 0
                            {
                                return Err(CStr::from_ptr(eb.as_ptr())
                                    .to_string_lossy()
                                    .into_owned());
                            }
                        }
                    }
                    c.pt.mark(PF_QSA_SEL, cs);
                    let mut t0 = 0i64;
                    while t0 < t_n {
                        let nb = m.sel_batch.min(t_n - t0);
                        let steps0 = m.steps_dev.add(t0 as usize * STEP_COUNT);
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        let active = host.steps_host
                            [((t0 + nb - 1) as usize) * STEP_COUNT + STEP_NBID]
                            as i64
                            + 1;
                        let qi = m.q_idx.add((t0 * idxh * idxd) as usize);
                        if e.select_old
                            || (f.block_scores_tc)(
                                st.idx.pooled,
                                st.idx.dead,
                                qi,
                                steps0,
                                nb,
                                m.max_blocks,
                                &s,
                                m.sel_scores,
                                cs,
                                active,
                            ) == 0
                        {
                            (f.block_scores)(
                                st.idx.pooled,
                                st.idx.dead,
                                qi,
                                steps0,
                                nb,
                                m.max_blocks,
                                &s,
                                m.sel_scores,
                                cs,
                                active,
                            );
                        }
                        (f.block_topk)(
                            m.sel_scores,
                            steps0,
                            nb,
                            m.max_blocks,
                            m.cap,
                            &s,
                            m.sel_ids.add((t0 * m.cap) as usize),
                            cs,
                            active,
                        );
                        t0 += m.sel_batch;
                    }
                    let pools: TkP2AttnPools = if staged { stage_pools } else { st.attn };
                    c.pt.mark(PF_QSA_ATTN, cs);
                    let mut fnp_done = false;
                    if !e.prompt_attn_old && e.fnp_tile_attn && t_n >= e.pf_switch_min_t {
                        fnp_done = (f.fnp_tile_attn)(
                            m.q,
                            &pools,
                            m.sel_ids,
                            m.steps_dev,
                            m.cap,
                            &s,
                            m.attn,
                            t_n,
                            cs,
                        ) != 0;
                    }
                    if !fnp_done
                        && (e.prompt_attn_old
                            || (f.prompt_attn_batch)(
                                m.q,
                                &pools,
                                m.sel_ids,
                                m.steps_dev,
                                m.cap,
                                &s,
                                m.attn,
                                t_n,
                                cs,
                            ) == 0)
                    {
                        let mut t0 = 0i64;
                        while t0 < t_n {
                            let nb = m.attn_batch.min(t_n - t0);
                            (f.decode_attn_batch)(
                                m.q.add((t0 * qh * hd) as usize),
                                &pools,
                                m.sel_ids.add((t0 * m.cap) as usize),
                                m.steps_dev.add(t0 as usize * STEP_COUNT),
                                m.cap,
                                &s,
                                m.attn_scratch,
                                m.attn.add((t0 * qh * hd) as usize),
                                nb,
                                cs,
                            );
                            t0 += m.attn_batch;
                        }
                    }
                    if st.kv_rot != 0 || st.kv_hybrid != 0 {
                        (f.fwht256)(m.attn, t_n * qh, cs);
                    }
                    c.pt.mark(PF_QSA, cs);
                    let ld_a = if e.pf_pad && t_n >= e.pf_switch_min_t.max(64) {
                        zv + ZV_PAD
                    } else {
                        0
                    };
                    (f.gate_attn)(m.attn, m.qf, m.attn_h, t_n, cs, ld_a);
                    c.native_proj(
                        &wo,
                        m.attn_h,
                        m.bo,
                        t_n,
                        || wname(l, "attn_output.weight"),
                        0,
                        ld_a,
                    )?;
                    qsa_index += 1;
                } else {
                    // ======================= MoE =======================
                    let wr = host.weight(&f, m.wt, l, "ffn_gate_inp.weight")?;
                    let wgi = host.weight(&f, m.wt, l, "ffn_gate_inp_shexp.weight")?;
                    let wsg = host.weight(&f, m.wt, l, "ffn_gate_shexp.weight")?;
                    let wsu = host.weight(&f, m.wt, l, "ffn_up_shexp.weight")?;
                    let wsd = host.weight(&f, m.wt, l, "ffn_down_shexp.weight")?;
                    c.pt.mark(PF_ROUTER, cs);
                    c.bf16_proj(
                        &wr,
                        m.mixed_bf,
                        m.logits,
                        t_n,
                        || wname(l, "ffn_gate_inp.weight"),
                        0,
                        m.mixed_bf_lo,
                        0,
                    )?;
                    (f.route)(m.logits, m.ids, m.w, t_n, ne, cs);
                    // the shared expert and its scalar gate
                    let shared_expert = |c: &Ctx| -> Result<(), String> {
                        c.native_proj(
                            &wsg,
                            m.mixed_h,
                            m.sgate,
                            t_n,
                            || wname(l, "ffn_gate_shexp.weight"),
                            0,
                            0,
                        )?;
                        c.native_proj(
                            &wsu,
                            m.mixed_h,
                            m.sup,
                            t_n,
                            || wname(l, "ffn_up_shexp.weight"),
                            0,
                            0,
                        )?;
                        (f.swiglu_pair)(m.sgate, m.sup, m.sh_h, t_n, cs);
                        c.native_proj(
                            &wsd,
                            m.sh_h,
                            m.shared,
                            t_n,
                            || wname(l, "ffn_down_shexp.weight"),
                            0,
                            0,
                        )?;
                        if wgi.bf16 == 0 {
                            return Err("prefill: shared gate is not BF16".into());
                        }
                        (f.gemm_bf16)(
                            m.gemm,
                            m.mixed_bf,
                            wgi.data.cast(),
                            m.sg,
                            t_n,
                            1,
                            nn,
                            0,
                            0.0,
                            0,
                        );
                        if !m.mixed_bf_lo.is_null() {
                            (f.gemm_bf16)(
                                m.gemm,
                                m.mixed_bf_lo,
                                wgi.data.cast(),
                                m.sg,
                                t_n,
                                1,
                                nn,
                                0,
                                1.0,
                                0,
                            );
                        }
                        Ok(())
                    };
                    // Only the router needs to finish before readback: the shared expert after it where it can
                    // overlap the host grouping and the routed-only path's first uploads
                    let defer_shared = !stream_all && e.stream_ahead;
                    if !defer_shared {
                        shared_expert(c)?;
                    }
                    let fm = host.fmts[l as usize];
                    let native = fm.native != 0;
                    let use_mmq = m.mmq_any != 0 && fm.mmq_layer != 0;
                    let (gt, dt) = if native {
                        (fm.gu_type, fm.d_type)
                    } else {
                        (42, 42)
                    };
                    let fused_only = m.mmq_any != 0 && fm.mmq_fo != 0;
                    let fused_nat =
                        (use_mmq || fused_only) && stream_all && native && fm.native_supported != 0;
                    let fused_l =
                        (use_mmq && stream_all && !native && (f.fused_enabled)() != 0) || fused_nat;
                    if !fused_l && m.fused_bufs != 0 && t_n >= m.stream_all_min {
                        return Err(format!(
                            "prefill: the fused layout's MoE buffers are too small for layer {l}'s expert path at a chunk of {t_n} tokens"
                        ));
                    }
                    if fused_l {
                        static SAID: AtomicBool = AtomicBool::new(false);
                        if !SAID.swap(true, Ordering::Relaxed) {
                            eprintln!("strata: prompt experts on the fused int8 kernels (STRATA_PF_FUSED=1, #136)");
                        }
                        c.pt.mark(PF_GATHER, cs);
                        (f.fused_quantize_act)(m.mixed, t_n, nn, m.xq, cs, fused_nat as i32);
                        (f.fused_group)(
                            m.ids,
                            t_n * k_used,
                            k_used as i32,
                            ne as i32,
                            m.gu.cast(),
                            m.slot_dev,
                            m.src_dev,
                            cs,
                        );
                        // launches over the experts in id order, each at most kMaxBatch experts of which at most a
                        // third of the ring streamed: the next batch's blobs arrive while one computes
                        let per = (ring / 3).max(1);
                        let max_batch = (f.fused_max_batch)() as i64;
                        let mut k = seq_start[l as usize];
                        let kend = seq_start[l as usize + 1];
                        let mut ex = 0i64;
                        let mut blobs: Vec<*const u8> = Vec::with_capacity(max_batch as usize);
                        while ex < ne {
                            let e0 = ex;
                            let k0 = k;
                            blobs.clear();
                            while ex < ne && ex - e0 < max_batch {
                                if k < kend && seq[k].e as i64 == ex {
                                    if k - k0 == per {
                                        break;
                                    }
                                    blobs.push(*m.stage_dev.add(k % ring) as *const u8);
                                    k += 1;
                                } else {
                                    // not streamed: resident (the walk streams all others)
                                    blobs.push((f.cache_device_slot)(
                                        m.cache,
                                        *m.host_res.add((l * ne + ex) as usize),
                                    ));
                                    c.st().experts_resident += 1;
                                }
                                ex += 1;
                            }
                            if k > k0 {
                                // one copy stream, in order: the batch's last blob covers the others
                                c.pt.mark(PF_WAIT_COPY, cs);
                                wait_issued(k - 1);
                                (f.stream_wait_event)(cs, *m.copied.add((k - 1) % ring));
                            }
                            c.pt.mark(PF_GEMM_GU, cs);
                            (f.fused_experts)(
                                fused_nat as i32,
                                &fm,
                                e0 as i32,
                                ex as i32,
                                blobs.as_ptr(),
                                ne as i32,
                                t_n * k_used,
                                m.gu.cast(),
                                m.xq,
                                m.src_dev,
                                m.h.cast(),
                                m.dm,
                                cs,
                            );
                            for kk in k0..k {
                                (f.event_record)(*m.used.add(kk % ring), cs);
                                atomic_i32(m.used_of.add(kk % ring))
                                    .store((kk % ring) as i32, Ordering::Relaxed);
                            }
                            if k > k0 {
                                consumed = k;
                                give_back!(consumed);
                            }
                        }
                    } else {
                        moe_grouped(
                            c,
                            host,
                            &mut MoeCtx {
                                l,
                                t_n,
                                p0,
                                stream_all,
                                use_mmq,
                                gt,
                                dt,
                                native,
                                fm,
                                defer_shared,
                                seq: &seq,
                                seq_start: &seq_start,
                                consumed: &mut consumed,
                                wait_issued: &wait_issued,
                                give_back: &mut |upto: usize, c: &mut Ctx| {
                                    if threaded_issue {
                                        shared.consumed.store(upto, Ordering::Release);
                                    } else {
                                        issue_until(upto + ring, c, &mut issued);
                                    }
                                },
                                shared_expert: &shared_expert,
                                pending: &mut pending,
                            },
                        )?;
                    }
                    c.pt.mark(PF_COMBINE, cs);
                    let row_sd = if use_mmq && !fused_l && native && fm.tail_off != 0 {
                        m.row_sd
                    } else {
                        std::ptr::null_mut()
                    };
                    (f.moe_combine)(m.dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, t_n, cs, row_sd);
                }
                // ---- the hyper-connection write, fused with the next half's norm where it can be
                let nl = if half == 0 { l } else { l + 1 };
                let mut cv = TkP2Cvec::default();
                if half == 1 {
                    (f.cvec_info)(l, &mut cv);
                }
                let steered = half == 1 && cv.covers != 0;
                let cv_fused = steered
                    && !hc16 // a steered layer under TALUS_FN_HC_BF16 takes the plain write + cvec_apply
                    && e.cvec_fuse
                    && !e.gr_unfused
                    && nl < le
                    && !(nl == 1 && ple_on)
                    && cv.n_embd == nn
                    && cv.hc == hc
                    && cv.tables_ok != 0;
                let fuse = !e.gr_unfused
                    && nl < le
                    && !(half == 1 && nl == 1 && ple_on)
                    && (!steered || cv_fused);
                let wnn = if fuse {
                    Some(host.weight(
                        &f,
                        m.wt,
                        nl,
                        if half == 0 {
                            "hc_ffn_norm.weight"
                        } else {
                            "hc_attn_norm.weight"
                        },
                    )?)
                } else {
                    None
                };
                match wnn {
                    Some(wnn) if cv_fused => {
                        (f.gr_write_cvec_norm_rs)(
                            m.r,
                            m.bo,
                            m.inj,
                            hc,
                            cv.dir.add((l * nn) as usize),
                            cv.s.add(l as usize),
                            cv.on,
                            cv.mode,
                            wnn.data.cast(),
                            eps,
                            m.grs,
                            m.xn16,
                            t_n,
                            cs,
                            m.xn16_lo,
                            ldx,
                        );
                        normed = true;
                    }
                    Some(wnn) if hc16 => {
                        (f.gr16_write_norm_rs)(
                            r16,
                            m.bo,
                            m.inj,
                            hc,
                            wnn.data.cast(),
                            eps,
                            m.grs,
                            m.xn16,
                            t_n,
                            h16,
                            cs,
                            m.xn16_lo,
                            ldx,
                        );
                        normed = true;
                    }
                    Some(wnn) => {
                        (f.gr_write_norm_rs)(
                            m.r,
                            m.bo,
                            m.inj,
                            hc,
                            wnn.data.cast(),
                            eps,
                            m.grs,
                            m.xn16,
                            t_n,
                            cs,
                            m.xn16_lo,
                            ldx,
                        );
                        normed = true;
                    }
                    None if hc16 => (f.gr16_write)(r16, m.bo, m.inj, hc, t_n, h16, cs),
                    None => (f.gr_write)(m.r, m.bo, m.inj, hc, t_n, cs),
                }
                if steered && !cv_fused {
                    r16_widen(t_n);
                    (f.cvec_apply)(m.r, l, t_n, d, cs);
                    r16_narrow(t_n);
                }
                if dump_dir.is_some() {
                    let tag = if half == 0 {
                        format!("prefill_p{p0}_l{l:02}_mixer_R")
                    } else {
                        format!("prefill_p{p0}_l{l:02}_R")
                    };
                    dump(&tag, l as i32);
                }
            }
        }
        ple_land(c, &mut pending, &mut ple_pending, &mut ple_buf)?;
        if let Some((h, _sh)) = pending.issuer.take() {
            let (ms, streamed, dma) = h.join().unwrap_or((0.0, 0, 0));
            let st = c.st();
            st.ms_experts_host += ms;
            st.experts_streamed += streamed;
            st.experts_dma += dma;
        }
        c.st().tokens += t_n;
        c.progress(
            "reading the prompt (batched): finishing the chunk from token",
            p0,
            -1,
        );
        c.pt.mark(PF_START, cs);
        // TALUS_FN_HC_BF16: the chunk's final R as FP32 for whatever reads it after the layers (on_chunk, the C++ side)
        r16_widen(t_n);
        if m.has_on_chunk != 0 {
            let toc = Instant::now();
            if (f.stream_sync)(cs) == 0 {
                return Err(format!("prefill: {}", c.cuda_err()));
            }
            let toc2 = Instant::now();
            let mut eb = [0 as c_char; 1024];
            if (f.on_chunk)(pf, m.r, t_n, p0, eb.as_mut_ptr(), eb.len()) == 0 {
                return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
            }
            host_sync_ms += (toc2 - toc).as_secs_f64() * 1000.0;
            host_chunk_ms += ms_since(toc2);
        }
        // the chunk's stager jobs are done (prefill.cpp's StagerDone at the end of the iteration)
        if pending.stager_chunk {
            (f.stager_finish)(pf);
            pending.stager_chunk = false;
        }
        c0 += m.t;
    }
    *m.ple_prev = prev[0];
    *m.ple_prev.add(1) = prev[1];
    if (f.stream_sync)(cs) == 0 {
        return Err(format!("prefill: {}", c.cuda_err()));
    }
    if (f.stream_sync)(m.copy) == 0 {
        return Err(format!("prefill: expert copy stream: {}", c.cuda_err()));
    }
    c.st().ms_total += ms_since(t_start);
    if c.pt.on {
        c.pt.fold();
        let total: f64 = c.pt.ms.iter().sum();
        let mut line = String::new();
        for i in 0..PF_COUNT {
            if c.pt.ms[i] <= 0.0 {
                continue;
            }
            line += &format!(
                " {} {:.0} ({:.1}%)",
                PF_NAMES[i],
                c.pt.ms[i],
                if total > 0.0 {
                    100.0 * c.pt.ms[i] / total
                } else {
                    0.0
                }
            );
        }
        eprintln!(
            "strata prefill timing: {n} tokens, GPU timeline {total:.0} ms, wall {:.0} ms, host staging {:.0} ms:{line}",
            ms_since(t_start),
            c.st().ms_experts_host
        );
        eprintln!(
            "strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) {host_setup_ms:.0} ms, waiting for each chunk {host_sync_ms:.0} ms, after each chunk (the draft layer, progress) {host_chunk_ms:.0} ms, PLE {:.0} ms",
            c.st().ms_ple
        );
    }
    drop(pending);
    Ok(())
}

/// A non-fused MoE layer's state (prefill.cpp's host-grouped path: MMQ groups, or the FP16 path).
struct MoeCtx<'a, 'b> {
    l: i64,
    t_n: i64,
    p0: i64,
    stream_all: bool,
    use_mmq: bool,
    gt: i32,
    dt: i32,
    native: bool,
    fm: TkP2ExpertFmt,
    defer_shared: bool,
    seq: &'a [StreamEntry],
    seq_start: &'a [usize],
    consumed: &'a mut usize,
    wait_issued: &'a dyn Fn(usize),
    give_back: &'a mut dyn FnMut(usize, &mut Ctx<'b>),
    shared_expert: &'a dyn Fn(&Ctx<'b>) -> Result<(), String>,
    pending: &'a mut Pending,
}

/// The routed rows grouped by expert on the host, then the experts in id order: resident ones from VRAM, the others
/// through the staging ring (the whole-layer stream of a streamed chunk, or STAGE slots after the routing).
#[allow(unused_assignments)]
unsafe fn moe_grouped<'b>(
    c: &mut Ctx<'b>,
    host: &mut P2Host,
    x: &mut MoeCtx<'_, 'b>,
) -> Result<(), String> {
    let f = *c.f;
    let m = c.m;
    let e = c.e;
    let pf = c.pf;
    let cs = m.cs;
    let (l, t_n, p0) = (x.l, x.t_n, x.p0);
    let ne = m.n_expert as i64;
    let (nn, k_used, ff) = (m.n_embd, m.n_used, m.n_ff);
    let tk = (t_n * k_used) as usize;
    let ring = m.ring as usize;
    let stage = m.stage_ring as usize;
    let fm = x.fm;
    c.pt.mark(PF_HOST_GROUP, cs);
    let grp_mapped = !m.grp_host.is_null();
    let (ids_h, slot_h, src_h): (*mut i32, *mut i32, *mut i32) = if grp_mapped {
        (
            m.grp_host,
            m.grp_host.add(m.grp_tk as usize),
            m.grp_host.add(2 * m.grp_tk as usize),
        )
    } else {
        (
            host.ids_host.as_mut_ptr(),
            host.slot_host.as_mut_ptr(),
            host.src_host.as_mut_ptr(),
        )
    };
    if grp_mapped {
        (f.copy_i32)(m.grp_dev, m.ids, tk as i64, cs);
    } else if (f.upring_d2h_sync)(pf, ids_h.cast(), m.ids.cast(), (tk * 4) as u64, cs) == 0 {
        return Err("prefill: the routed ids' download failed".into());
    }
    c.progress(
        "reading the prompt (batched): waiting for the GPU (attention, router) at layer",
        l,
        p0,
    );
    (f.stream_sync)(cs);
    c.progress("reading the prompt (batched): layer", l, p0);
    c.pt.fold();
    if x.defer_shared {
        c.pt.mark(PF_ROUTER, cs);
        (x.shared_expert)(c)?;
        c.pt.mark(PF_HOST_GROUP, cs);
    }
    let ids_s = std::slice::from_raw_parts(ids_h as *const i32, tk);
    host.cnt.iter_mut().for_each(|v| *v = 0);
    for &ex in ids_s {
        if ex < 0 || ex as i64 >= ne {
            return Err("prefill: routed id out of range".into());
        }
        host.cnt[ex as usize] += 1;
    }
    let rows_local = tk as i64;
    {
        let mut r = 0i32;
        for ex in 0..ne as usize {
            host.off[ex] = r;
            r += host.cnt[ex];
        }
        host.off[ne as usize] = r;
    }
    {
        let mut fill: Vec<i32> = host.off[..ne as usize].to_vec();
        let slot_s = std::slice::from_raw_parts_mut(slot_h, tk);
        let src_s = std::slice::from_raw_parts_mut(src_h, tk);
        for i in 0..tk {
            let ex = ids_s[i] as usize;
            let p = fill[ex];
            fill[ex] += 1;
            slot_s[i] = p;
            src_s[p as usize] = (i as i64 / k_used) as i32;
        }
    }
    let order: Vec<i32> = (0..ne as i32)
        .filter(|&ex| host.cnt[ex as usize] > 0)
        .collect();
    if grp_mapped {
        (f.copy_i32)(m.slot_dev, m.grp_dev.add(m.grp_tk as usize), tk as i64, cs);
        (f.copy_i32)(
            m.src_dev,
            m.grp_dev.add(2 * m.grp_tk as usize),
            tk as i64,
            cs,
        );
    } else if (f.upring_h2d)(
        pf,
        m.slot_dev.cast(),
        host.slot_host.as_ptr().cast(),
        (tk * 4) as u64,
        cs,
    ) == 0
        || (f.upring_h2d)(
            pf,
            m.src_dev.cast(),
            host.src_host.as_ptr().cast(),
            (tk * 4) as u64,
            cs,
        ) == 0
    {
        return Err("prefill: the routing tables' upload failed".into());
    }
    let gub = if x.use_mmq {
        (f.mmq_matrix_bytes)(x.gt, 2 * ff, nn)
    } else {
        0
    };
    let db = if x.use_mmq {
        (f.mmq_matrix_bytes)(x.dt, nn, ff)
    } else {
        0
    };
    c.pt.mark(PF_GATHER, cs);
    let n_ord = order.len();
    if x.use_mmq {
        // step 2b: the layer's activations as q8_1 rows in expert order, straight from `mixed`
        (f.mmq_quantize)(
            m.mixed, m.src_dev, m.xq, x.gt, nn, nn, tk as i64, cs, m.xscale,
        );
        // each group's rows: absolute bounds (gate/up), relative ones (down reads the group's own quantized H)
        let ng = n_ord.div_ceil(MMQ_GROUP);
        host.bounds_host.resize(n_ord + 1 + ng * (MMQ_GROUP + 1), 0);
        for (j, &ex) in order.iter().enumerate() {
            host.bounds_host[j] = host.off[ex as usize];
        }
        host.bounds_host[n_ord] = rows_local as i32;
        for g in 0..ng {
            for i in 0..=MMQ_GROUP {
                host.bounds_host[n_ord + 1 + g * (MMQ_GROUP + 1) + i] = host.bounds_host
                    [n_ord.min(g * MMQ_GROUP + i)]
                    - host.bounds_host[g * MMQ_GROUP];
            }
        }
        let nb = host.bounds_host.len();
        if grp_mapped {
            let bh = m.grp_host.add(3 * m.grp_tk as usize);
            std::ptr::copy_nonoverlapping(host.bounds_host.as_ptr(), bh, nb);
            (f.copy_i32)(
                m.bounds_dev,
                m.grp_dev.add(3 * m.grp_tk as usize),
                nb as i64,
                cs,
            );
        } else if (f.upring_h2d)(
            pf,
            m.bounds_dev.cast(),
            host.bounds_host.as_ptr().cast(),
            (nb * 4) as u64,
            cs,
        ) == 0
        {
            return Err("prefill: the group bounds' upload failed".into());
        }
    } else {
        (f.gather_rows16)(m.mixed_h, m.src_dev, m.xs, tk as i64, nn, cs);
    }
    let resident = |ex: i64| -> bool {
        !m.host_res.is_null() && !m.cache.is_null() && *m.host_res.add((l * ne + ex) as usize) >= 0
    };
    let mut stage_next = 0usize;
    let mut stage_of: Vec<i32> = vec![-1; n_ord];
    let mut job_of: Vec<i64> = vec![-1; n_ord];
    if !x.stream_all {
        let mut js: Vec<TkP2StagerJob> = Vec::new();
        for (j, &ex) in order.iter().enumerate() {
            let ex = ex as i64;
            if resident(ex) {
                continue;
            }
            if m.src.is_null() {
                return Err("prefill: an expert is neither resident nor has a source".into());
            }
            if (f.src_pinned)(m.src, l, ex) != 0 {
                continue;
            }
            job_of[j] = js.len() as i64;
            if (f.src_transient)(m.src, l, ex) != 0 {
                js.push(TkP2StagerJob {
                    src: std::ptr::null(),
                    bytes: fm.blob_bytes,
                    from_source: 1,
                    l: l as i32,
                    e: ex as i32,
                    pad0: 0,
                });
                continue;
            }
            let b = (f.src_blob_stable)(m.src, l, ex);
            if b.is_null() {
                return Err("prefill: expert source has no blob".into());
            }
            js.push(TkP2StagerJob {
                src: b,
                bytes: fm.blob_bytes,
                from_source: 0,
                l: 0,
                e: 0,
                pad0: 0,
            });
        }
        if !js.is_empty() {
            (f.stager_start)(pf, js.as_ptr(), js.len() as i64);
        }
        x.pending.stager_layer = true;
    }
    let group_gather = e.group_gather_env && x.stream_all && x.use_mmq && x.native;
    let mut gg_blob: [*const u8; MMQ_GROUP] = [std::ptr::null(); MMQ_GROUP];
    let mut gg_first = 0i32;
    let mut gg_n = 0i32;
    let mut gg_slots = [0usize; MMQ_GROUP];
    let mut gg_nslots = 0usize;
    let tail = fm.tail_off != 0;

    // the open group's experts gathered in one launch (or one at a time where the group launch declines)
    macro_rules! flush {
        ($zero:expr) => {{
            let zero: u64 = $zero;
            if gg_n > gg_first {
                if gg_nslots > 0 {
                    // the copies land in order: the last one covers the others
                    c.pt.mark(PF_WAIT_COPY, cs);
                    (f.stream_wait_event)(cs, *m.copied.add(gg_slots[gg_nslots - 1]));
                    c.pt.mark(PF_DEQUANT, cs);
                }
                if (f.mmq_gather_native_group)(
                    gg_blob.as_ptr(),
                    gg_first,
                    gg_n,
                    fm.up_off,
                    gub / 2,
                    fm.down_off,
                    db,
                    m.grp_gu.cast(),
                    gub,
                    m.grp_d.cast(),
                    db,
                    cs,
                    fm.tail_off,
                    if tail {
                        m.grp_tail.cast()
                    } else {
                        std::ptr::null_mut()
                    },
                    zero,
                ) == 0
                {
                    for i in gg_first..gg_n {
                        let b = gg_blob[i as usize];
                        (f.mmq_gather_native)(
                            b.cast(),
                            b.add(fm.up_off as usize).cast(),
                            gub / 2,
                            b.add(fm.down_off as usize).cast(),
                            db,
                            m.grp_gu.add(i as usize * gub as usize).cast(),
                            m.grp_d.add(i as usize * db as usize).cast(),
                            cs,
                            if tail {
                                b.add(fm.tail_off as usize).cast()
                            } else {
                                std::ptr::null()
                            },
                            if tail {
                                m.grp_tail.add(i as usize * 4).cast()
                            } else {
                                std::ptr::null_mut()
                            },
                            if i + 1 == gg_n { zero } else { 0 },
                        );
                    }
                }
                if gg_nslots > 0 {
                    let rel = gg_slots[gg_nslots - 1];
                    (f.event_record)(*m.used.add(rel), cs);
                    for &sl in &gg_slots[..gg_nslots] {
                        atomic_i32(m.used_of.add(sl)).store(rel as i32, Ordering::Relaxed);
                    }
                }
                gg_nslots = 0;
                gg_first = gg_n;
            }
        }};
    }
    // expert order[j] from `blob_dev` (ring slot `slot`, or -1: resident)
    macro_rules! compute {
        ($j:expr, $blob:expr, $slot:expr) => {{
            let j: usize = $j;
            let blob_dev: *const u8 = $blob;
            let slot: i64 = $slot;
            let ex = order[j] as usize;
            c.pt.mark(PF_DEQUANT, cs);
            if x.use_mmq {
                let q = j % MMQ_GROUP;
                let last_of_group = q + 1 == MMQ_GROUP || j + 1 == n_ord;
                let mut gathered = true;
                if group_gather {
                    gg_blob[q] = blob_dev;
                    gg_n = q as i32 + 1;
                    if slot >= 0 {
                        gg_slots[gg_nslots] = slot as usize;
                        gg_nslots += 1;
                    }
                    if !last_of_group {
                        gathered = false;
                    } else {
                        flush!(MMQ_TAIL);
                        gg_blob = [std::ptr::null(); MMQ_GROUP];
                        gg_first = 0;
                        gg_n = 0;
                    }
                } else if x.native {
                    (f.mmq_gather_native)(
                        blob_dev.cast(),
                        blob_dev.add(fm.up_off as usize).cast(),
                        gub / 2,
                        blob_dev.add(fm.down_off as usize).cast(),
                        db,
                        m.grp_gu.add(q * gub as usize).cast(),
                        m.grp_d.add(q * db as usize).cast(),
                        cs,
                        if tail {
                            blob_dev.add(fm.tail_off as usize).cast()
                        } else {
                            std::ptr::null()
                        },
                        if tail {
                            m.grp_tail.add(q * 4).cast()
                        } else {
                            std::ptr::null_mut()
                        },
                        if last_of_group { MMQ_TAIL } else { 0 },
                    );
                } else {
                    (f.mmq_gather_strata_q2)(
                        blob_dev,
                        m.grp_gu.add(q * gub as usize).cast(),
                        m.grp_d.add(q * db as usize).cast(),
                        cs,
                    );
                    if last_of_group {
                        (f.memset)(m.grp_gu.add((q + 1) * gub as usize).cast(), 0, MMQ_TAIL, cs);
                        (f.memset)(m.grp_d.add((q + 1) * db as usize).cast(), 0, MMQ_TAIL, cs);
                    }
                }
                if gathered {
                    if slot >= 0 && !group_gather {
                        (f.event_record)(*m.used.add(slot as usize), cs);
                        atomic_i32(m.used_of.add(slot as usize))
                            .store(slot as i32, Ordering::Relaxed);
                    }
                    if q + 1 >= MMQ_GROUP || j + 1 >= n_ord {
                        let j0 = j - q;
                        let g = j0 / MMQ_GROUP;
                        let ngx = (q + 1) as i32;
                        let r0 = host.bounds_host[j0] as i64;
                        let nr = host.bounds_host[j + 1] as i64 - r0;
                        let maxr = (j0..=j)
                            .map(|i| host.cnt[order[i] as usize] as i64)
                            .max()
                            .unwrap_or(0);
                        c.pt.mark(PF_GEMM_GU, cs);
                        (f.mmq_product)(
                            m.mmq_ctx,
                            m.grp_gu.cast(),
                            x.gt,
                            2 * ff,
                            nn,
                            gub,
                            ngx,
                            m.xq,
                            m.bounds_dev.add(j0),
                            m.ids_identity,
                            tk as i64,
                            maxr,
                            m.gu,
                            2 * ff,
                            if (f.mmq_fp4_activations)(x.gt) != 0 {
                                m.xscale
                            } else {
                                std::ptr::null()
                            },
                            1,
                            cs,
                        );
                        if x.native && tail {
                            (f.mmq_swiglu_scaled)(
                                m.gu.add((r0 * 2 * ff) as usize),
                                m.h.add((r0 * ff) as usize),
                                nr,
                                ff,
                                m.bounds_dev.add(j0),
                                ngx,
                                m.grp_tail,
                                r0,
                                cs,
                            );
                        } else {
                            (f.mmq_swiglu)(
                                m.gu.add((r0 * 2 * ff) as usize),
                                m.h.add((r0 * ff) as usize),
                                nr,
                                ff,
                                (!x.native) as i32,
                                cs,
                            );
                        }
                        c.pt.mark(PF_GEMM_D, cs);
                        (f.mmq_quantize)(
                            m.h.add((r0 * ff) as usize),
                            std::ptr::null(),
                            m.hq,
                            x.dt,
                            ff,
                            ff,
                            nr,
                            cs,
                            m.hscale,
                        );
                        let dbounds = m.bounds_dev.add(n_ord + 1 + g * (MMQ_GROUP + 1));
                        (f.mmq_product)(
                            m.mmq_ctx,
                            m.grp_d.cast(),
                            x.dt,
                            nn,
                            ff,
                            db,
                            ngx,
                            m.hq,
                            dbounds,
                            m.ids_identity,
                            nr,
                            maxr,
                            m.dm.add((r0 * nn) as usize),
                            nn,
                            if (f.mmq_fp4_activations)(x.dt) != 0 {
                                m.hscale
                            } else {
                                std::ptr::null()
                            },
                            1,
                            cs,
                        );
                        if x.native && tail {
                            // s_down: applied by the combine
                            (f.mmq_down_row_scales)(
                                m.row_sd.add(r0 as usize),
                                dbounds,
                                ngx,
                                m.grp_tail,
                                nr,
                                cs,
                            );
                        }
                    }
                }
            } else {
                let q = j % DQ;
                if x.native {
                    let tp = if tail {
                        blob_dev.add(fm.tail_off as usize) as *const f32
                    } else {
                        std::ptr::null()
                    };
                    (f.iq_dequant_gu_f16)(
                        fm.gu_type,
                        blob_dev.cast(),
                        blob_dev.add(fm.up_off as usize).cast(),
                        fm.n_ff,
                        fm.n_embd,
                        m.dq_gu[q],
                        cs,
                        tp,
                    );
                    (f.iq_dequant_f16)(
                        fm.d_type,
                        blob_dev.add(fm.down_off as usize).cast(),
                        fm.n_embd * fm.n_ff,
                        m.dq_d[q],
                        cs,
                        if tail { tp.add(2) } else { std::ptr::null() },
                    );
                } else {
                    (f.blob_dequant_f16)(blob_dev, m.dq_gu[q], m.dq_d[q], cs);
                }
                if slot >= 0 {
                    (f.event_record)(*m.used.add(slot as usize), cs);
                    atomic_i32(m.used_of.add(slot as usize)).store(slot as i32, Ordering::Relaxed);
                }
                let o0 = host.off[ex] as i64;
                let nex = host.cnt[ex] as i64;
                c.pt.mark(PF_GEMM_GU, cs);
                (f.gemm_f16)(
                    m.gemm,
                    m.xs.add((o0 * nn) as usize),
                    m.dq_gu[q],
                    m.gu.add((o0 * 2 * ff) as usize),
                    nex,
                    2 * ff,
                    nn,
                );
                (f.swiglu_interleaved)(
                    m.gu.add((o0 * 2 * ff) as usize),
                    m.hh.add((o0 * ff) as usize),
                    nex,
                    cs,
                );
                c.pt.mark(PF_GEMM_D, cs);
                (f.gemm_f16)(
                    m.gemm,
                    m.hh.add((o0 * ff) as usize),
                    m.dq_d[q],
                    m.dm.add((o0 * nn) as usize),
                    nex,
                    nn,
                    ff,
                );
            }
        }};
    }

    if !x.stream_all {
        let mut staged = 0usize;
        let mut pend = 0usize;
        let lookahead = stage - 1;
        for j in 0..n_ord {
            while staged < n_ord
                && (if e.stream_ahead {
                    pend < stage
                } else {
                    staged <= j + lookahead
                })
            {
                // stage_one(staged)
                let jj = staged;
                let ex = order[jj] as i64;
                if !resident(ex) {
                    let sl = stage_next;
                    stage_next = (stage_next + 1) % stage;
                    let th = Instant::now();
                    let pinned = (f.src_pinned)(m.src, l, ex) != 0;
                    let b = if pinned {
                        (f.src_blob_stable)(m.src, l, ex)
                    } else {
                        std::ptr::null()
                    };
                    if pinned && b.is_null() {
                        return Err("prefill: expert source has no blob".into());
                    }
                    let live = atomic_u8(m.stage_live.add(sl)).load(Ordering::Relaxed) != 0;
                    if pinned {
                        if live {
                            let uo = atomic_i32(m.used_of.add(sl)).load(Ordering::Relaxed) as usize;
                            (f.stream_wait_event)(m.copy, *m.used.add(uo));
                        }
                        (f.memcpy)(
                            (*m.stage_dev.add(sl)).cast(),
                            b.cast(),
                            fm.blob_bytes,
                            H2D,
                            m.copy,
                        );
                        c.st().experts_dma += 1;
                    } else {
                        let hb = (f.stager_wait)(pf, job_of[jj]);
                        if live {
                            let uo = atomic_i32(m.used_of.add(sl)).load(Ordering::Relaxed) as usize;
                            (f.stream_wait_event)(m.copy, *m.used.add(uo));
                        }
                        (f.memcpy)(
                            (*m.stage_dev.add(sl)).cast(),
                            hb.cast(),
                            fm.blob_bytes,
                            H2D,
                            m.copy,
                        );
                        (f.stager_issued_one)(pf, job_of[jj], m.copy);
                    }
                    (f.event_record)(*m.copied.add(sl), m.copy);
                    atomic_u8(m.stage_live.add(sl)).store(1, Ordering::Relaxed);
                    stage_of[jj] = sl as i32;
                    let st = c.st();
                    st.ms_experts_host += ms_since(th);
                    st.experts_streamed += 1;
                }
                if stage_of[staged] >= 0 {
                    pend += 1;
                }
                staged += 1;
            }
            let ex = order[j] as i64;
            if stage_of[j] < 0 {
                c.st().experts_resident += 1;
                let bp = (f.cache_device_slot)(m.cache, *m.host_res.add((l * ne + ex) as usize));
                compute!(j, bp, -1);
            } else {
                let sl = stage_of[j] as usize;
                c.pt.mark(PF_WAIT_COPY, cs);
                (f.stream_wait_event)(cs, *m.copied.add(sl));
                compute!(j, *m.stage_dev.add(sl) as *const u8, sl as i64);
                pend -= 1; // compute recorded the slot's release event before any reuse
            }
        }
    } else {
        let mut k = x.seq_start[l as usize];
        let kend = x.seq_start[l as usize + 1];
        // this layer's ring entries the routing did not pick give their slots back
        macro_rules! release_to {
            ($e_stop:expr) => {{
                let e_stop: i64 = $e_stop;
                while k < kend && (x.seq[k].e as i64) < e_stop {
                    if gg_nslots > 0 {
                        flush!(0); // the open group's slots get their event first
                    }
                    let sl = k % ring;
                    (f.event_record)(*m.used.add(sl), cs);
                    atomic_i32(m.used_of.add(sl)).store(sl as i32, Ordering::Relaxed);
                    k += 1;
                    *x.consumed = k;
                    (x.give_back)(*x.consumed, c);
                }
            }};
        }
        for j in 0..n_ord {
            let ex = order[j] as i64;
            release_to!(ex);
            if k < kend && x.seq[k].e as i64 == ex {
                let sl = k % ring;
                (x.wait_issued)(k);
                if !group_gather {
                    c.pt.mark(PF_WAIT_COPY, cs);
                    (f.stream_wait_event)(cs, *m.copied.add(sl));
                }
                compute!(j, *m.stage_dev.add(sl) as *const u8, sl as i64);
                k += 1;
                *x.consumed = k;
                if !group_gather || gg_nslots == 0 {
                    (x.give_back)(*x.consumed, c); // its group was gathered
                }
            } else {
                if !resident(ex) {
                    return Err(
                        "prefill: an expert is neither resident, streamed nor on the peer".into(),
                    );
                }
                let bp = (f.cache_device_slot)(m.cache, *m.host_res.add((l * ne + ex) as usize));
                if bp.is_null() {
                    return Err(
                        "prefill: an expert is neither resident, streamed nor on the peer".into(),
                    );
                }
                c.st().experts_resident += 1;
                compute!(j, bp, -1);
                if group_gather && gg_nslots == 0 {
                    (x.give_back)(*x.consumed, c);
                }
            }
        }
        release_to!(ne);
    }
    if x.pending.stager_layer {
        (f.stager_finish)(pf);
        x.pending.stager_layer = false;
    }
    Ok(())
}

// ============================================================================================ draft_kv (E-9)

enum Draft {
    Done,
    Decline(&'static CStr),
    NotHandled,
}

/// The draft layer's K/V for prompt cells [cell0, cell0 + n) from their final residual rows and the next tokens, in
/// batches through the prompt path's scratch region (prefill.cpp's Prefill::draft_kv).
unsafe fn draft_kv(
    f: &P2Ffi,
    pf: *mut c_void,
    mtp: *mut c_void,
    r_rows: *const f32,
    next_tokens: *const i32,
    n: i64,
    cell0: i64,
) -> Result<Draft, String> {
    let e = env();
    let mut m = TkP2Layout::default();
    if (f.bind)(pf, &mut m) <= 0
        || m.metal != 0
        || m.stage_lb != 0
        || m.stage_le != m.n_layers as i64
    {
        return Ok(Draft::NotHandled);
    }
    if e.mtp_batch_off {
        return Ok(Draft::Decline(c"STRATA_MTP_BATCH=0"));
    }
    let mut info = TkP2MtpInfo::default();
    (f.mtp_info)(mtp, &mut info);
    let st = info.kv;
    if n <= 0 || m.g.is_null() || m.region.is_null() {
        return Ok(Draft::Decline(c"no prompt scratch region on this path"));
    }
    if st.kv_mode != 0 && !(st.kv_mode == 2 && e.mtp_batch_ring_ok) {
        return Ok(Draft::Decline(if st.kv_mode == 2 {
            c"the drafter's K/V is a ring and STRATA_MTP_BATCH_RING=0"
        } else {
            c"the drafter's K/V mode is not paged/ring"
        }));
    }
    if st.kv_hybrid != 0 {
        return Ok(Draft::Decline(c"the drafter's K/V is hybrid (kv_hybrid)"));
    }
    if info.device != m.device {
        return Ok(Draft::Decline(
            c"the drafter is on another device than this prefill path",
        ));
    }
    let t0 = Instant::now();
    let cs = m.cs;
    let (nn, hcn, kv, hc, lr) = (
        m.n_embd,
        m.hc * m.n_embd,
        m.n_head_kv * m.head_dim,
        m.hc,
        m.hc_lr,
    );
    let eps = m.eps;
    if info.w_ne.is_null()
        || info.w_fe.is_null()
        || info.w_nh.is_null()
        || info.w_fh.is_null()
        || info.w_hn.is_null()
        || info.w_dn.is_null()
        || info.w_up.is_null()
        || info.w_k.is_null()
        || info.w_v.is_null()
        || info.w_kn.is_null()
    {
        return Ok(Draft::Decline(
            c"a drafter tensor is missing (Q8_0 projections / norms)",
        ));
    }
    let mut emb_rows = 0i64;
    if (f.embed_kind)(m.wt, m.g, &mut emb_rows) == 0 {
        return Ok(Draft::Decline(
            c"the token embedding is not gatherable on this path",
        ));
    }
    // the cells the drafter's window can still reach
    let r0 = (info.first_needed - cell0).max(0);
    if r0 >= n {
        return Ok(Draft::Done);
    }
    // E-9: the drafter's Q8_0 matrices through Q8_1 x Q8_0 MMQ; STRATA_MTP_BATCH_F16=1: FP16 GEMMs (the A/B)
    let q8 = !e.mtp_batch_f16
        && m.mmq_built != 0
        && (f.mmq_fits)(KQ8_0, nn) != 0
        && (f.mmq_fits)(KQ8_0, kv) != 0;
    let per_row: u64 = (4 * (2 * nn + 4 * hcn + lr + hc + nn + 2 * kv + 1)
        + 2 * (nn + 2 * hcn + lr + nn)
        + 64) as u64
        + if q8 {
            (f.mmq_q8_bytes)(hc, nn) + 4 * hc as u64
        } else {
            0
        };
    let mut b_rows = (n - r0).min(((m.region_bytes / per_row) as i64) & !63i64);
    if st.kv_mode == 2 {
        // a ring: one batch's cells must not share a slot (a batch can straddle one page more)
        b_rows = b_rows.min(((st.n_slots - 1) * m.shapes.page_size) & !63i64);
    }
    if b_rows < 64 {
        return Ok(Draft::Decline(c"too little scratch for a 64-row batch"));
    }
    let bu = b_rows as usize;
    let mut q = m.region;
    let mut carve = |bytes: usize| -> *mut u8 {
        let p = q;
        q = q.add((bytes + 255) & !255usize);
        p
    };
    let emb = carve(bu * nn as usize * 4) as *mut f32;
    let e2 = carve(bu * nn as usize * 4) as *mut f32;
    let en16 = carve(bu * nn as usize * 2) as *mut u16;
    let hn = carve(bu * hcn as usize * 4) as *mut f32;
    let h2 = carve(bu * hcn as usize * 4) as *mut f32;
    let rm = carve(bu * hcn as usize * 4) as *mut f32;
    let gated = carve(bu * hcn as usize * 4) as *mut f32;
    let hn16 = carve(bu * hcn as usize * 2) as *mut u16;
    let xn16 = carve(bu * hcn as usize * 2) as *mut u16;
    let lo = carve(bu * lr as usize * 4) as *mut f32;
    let lo16 = carve(bu * lr as usize * 2) as *mut u16;
    let grs = carve(bu * hc as usize * 4) as *mut f32;
    let mixed = carve(bu * nn as usize * 4) as *mut f32;
    let mixed_h = carve(bu * nn as usize * 2) as *mut u16;
    let kc = carve(bu * kv as usize * 4) as *mut f32;
    let vc = carve(bu * kv as usize * 4) as *mut f32;
    let tok = carve(bu * 4) as *mut i32;
    let xq = if q8 {
        carve((f.mmq_q8_bytes)(b_rows * hc, nn) as usize) as *mut c_void
    } else {
        std::ptr::null_mut()
    };
    let ident = if q8 {
        carve(bu * hc as usize * 4) as *mut i32
    } else {
        std::ptr::null_mut()
    };
    let bnd = if q8 {
        carve(16) as *mut i32
    } else {
        std::ptr::null_mut()
    };
    if (q as u64 - m.region as u64) > m.region_bytes {
        return Ok(Draft::Decline(
            c"the batch buffers do not fit the scratch region",
        ));
    }
    if e.draft_timing {
        (f.stream_sync)(cs);
    }
    let ti0 = Instant::now();
    let mut eb = [0 as c_char; 512];
    if (f.mtp_idle)(mtp, eb.as_mut_ptr(), eb.len()) == 0 {
        return Err(CStr::from_ptr(eb.as_ptr()).to_string_lossy().into_owned());
    }
    let ms_idle = ms_since(ti0);
    let tl0 = Instant::now();
    let s = m.shapes;
    let mut tk = vec![0i32; bu];
    let mmq_ctx = if q8 {
        let ctx = (f.ensure_mmq_ctx)(pf);
        (f.mmq_iota)(ident, b_rows * hc, cs);
        ctx
    } else {
        std::ptr::null_mut()
    };
    // y[rows, n_out] = x[rows, k] . w^T for a Q8_0 matrix (bounds slot 0: rows, 1: rows * hc)
    let proj = |x: *const f32,
                x16: *const u16,
                w: *const c_void,
                y: *mut f32,
                rows: i64,
                n_out: i64,
                k: i64,
                slot: usize| {
        if !q8 {
            (f.gemm_native)(m.gemm, x16, KQ8_0, w, y, rows, n_out, k, 0, 0.0, 0);
            return;
        }
        (f.mmq_quantize)(
            x,
            std::ptr::null(),
            xq,
            KQ8_0,
            k,
            k,
            rows,
            cs,
            std::ptr::null_mut(),
        );
        (f.mmq_run)(
            mmq_ctx,
            w,
            KQ8_0,
            n_out,
            k,
            (f.mmq_matrix_bytes)(KQ8_0, n_out, k),
            1,
            xq,
            bnd.add(2 * slot),
            ident,
            rows,
            rows,
            y,
            n_out,
            cs,
        );
    };
    let mut b0 = r0;
    while b0 < n {
        let nb = b_rows.min(n - b0);
        let c0 = cell0 + b0;
        for i in 0..nb as usize {
            tk[i] = *next_tokens.add(b0 as usize + i);
        }
        let bh: [i32; 4] = [0, nb as i32, 0, (nb * hc) as i32];
        if q8 && (f.upring_h2d)(pf, bnd.cast(), bh.as_ptr().cast(), 16, cs) == 0 {
            return Err("prefill: the draft bounds' upload failed".into());
        }
        if (f.upring_h2d)(pf, tok.cast(), tk.as_ptr().cast(), (nb * 4) as u64, cs) == 0 {
            return Err("prefill: the draft tokens' upload failed".into());
        }
        // the input branches: the next token's embedding, and this cell's final residual rows
        (f.embed_gather)(m.wt, tok, nb, emb, cs);
        (f.rms_rows)(emb, info.w_ne, nb, nn, nn, eps, cs);
        if !q8 {
            (f.to_f16)(emb, en16, nb * nn, cs);
        }
        proj(emb, en16, info.w_fe, e2, nb, nn, nn, 0);
        (f.memcpy)(
            hn.cast(),
            r_rows.add((b0 * hcn) as usize).cast(),
            (nb * hcn * 4) as u64,
            D2D,
            cs,
        );
        if info.hnorm_per_stream != 0 {
            (f.rms_norm_grouped)(
                hn,
                info.w_nh,
                hn,
                nn as i32,
                hc as i32,
                (nb * hc) as i32,
                eps,
                cs,
            );
        } else {
            (f.rms_rows)(hn, info.w_nh, nb, hcn, hcn, eps, cs);
        }
        if !q8 {
            (f.to_f16)(hn, hn16, nb * hcn, cs);
        }
        proj(hn, hn16, info.w_fh, h2, nb * hc, nn, nn, 1); // every stream through fc_hidden
        (f.add_streams_broadcast)(h2, e2, rm, nn, hc as i32, nb as i32, cs);
        // the attention hyper-connection's read (its mixed input only)
        (f.gr_norm_rs)(
            rm,
            info.w_hn,
            eps,
            grs,
            xn16,
            nb,
            cs,
            std::ptr::null_mut(),
            0,
        );
        (f.gemm_bf16)(m.gemm, xn16, info.w_dn, lo, nb, lr, hcn, 0, 0.0, 0);
        (f.gr_silu)(lo, lo16, nb, cs, std::ptr::null_mut());
        (f.gemm_bf16)(m.gemm, lo16, info.w_up, gated, nb, hcn, lr, 0, 0.0, 0);
        (f.gr_mix_r)(
            rm,
            grs,
            info.w_hn,
            gated,
            mixed,
            std::ptr::null_mut(),
            nb,
            cs,
            mixed_h,
            std::ptr::null_mut(),
        );
        // K and V into the drafter's cache, as the prompt path's QSA layers append theirs
        proj(mixed, mixed_h, info.w_k, kc, nb, kv, nn, 0);
        proj(mixed, mixed_h, info.w_v, vc, nb, kv, nn, 0);
        (f.rms_rows)(
            kc,
            info.w_kn,
            nb * m.n_head_kv,
            m.head_dim,
            m.head_dim,
            eps,
            cs,
        );
        (f.rope)(kc, nb, m.n_head_kv, m.head_dim, kv, c0, cs);
        if st.kv_rot != 0 {
            (f.fwht256)(kc, nb * m.n_head_kv, cs);
            (f.fwht256)(vc, nb * m.n_head_kv, cs);
        }
        if st.kv_q4 != 0 {
            (f.kv_append_q4)(
                st.k_q4,
                st.v_q4,
                st.page_table,
                c0,
                nb,
                kc,
                vc,
                &s,
                cs,
                &st.host,
                std::ptr::null(),
            );
        } else {
            let i8m = st.kv_int8 != 0;
            (f.kv_append)(
                kc,
                vc,
                nb,
                c0,
                st.page_table,
                s.page_size,
                if i8m { std::ptr::null_mut() } else { st.k_pool },
                if i8m { std::ptr::null_mut() } else { st.v_pool },
                st.k_q,
                st.v_q,
                st.k_scale,
                st.v_scale,
                cs,
                &st.host,
                std::ptr::null(),
            );
        }
        b0 += b_rows;
    }
    if (f.stream_sync)(cs) == 0 {
        let mut b = [0 as c_char; 256];
        (f.peek_error)(b.as_mut_ptr(), b.len());
        return Err(format!(
            "prefill: the draft layer's K/V: {}",
            CStr::from_ptr(b.as_ptr()).to_string_lossy()
        ));
    }
    (f.mtp_add_ms_prefill)(mtp, ms_since(t0));
    if e.draft_timing {
        eprintln!(
            "strata draft kv: {n} cells from {cell0} (first needed {}), batch {b_rows}: drafter idle {ms_idle:.1} ms, the batches {:.1} ms, all {:.1} ms",
            cell0 + r0,
            ms_since(tl0),
            ms_since(t0)
        );
    }
    Ok(Draft::Done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_chunk_rounds_to_256_multiples() {
        assert_eq!(equal_chunk(512, 4096), 512);
        assert_eq!(equal_chunk(4096, 4096), 4096);
        assert_eq!(equal_chunk(5000, 4096), 4096);
        assert_eq!(
            crate::qwen4exp::weights::serve_equal_chunk(5000, 4096, true, 900),
            2560
        );
    }

    #[test]
    fn c_atoll_matches_c() {
        assert_eq!(c_atoll("16384"), 16384);
        assert_eq!(c_atoll("  -12abc"), -12);
        assert_eq!(c_atoll("auto"), 0);
        assert_eq!(c_atoll(""), 0);
        assert_eq!(c_atoi("+7"), 7);
    }

    #[test]
    fn abi_struct_sizes_are_stable() {
        // the C side checks these at install (tk_qwen4exp_p2_abi_sizes); pinned here so a field edit is noticed
        assert_eq!(std::mem::size_of::<TkP2Weight>(), 56);
        assert_eq!(std::mem::size_of::<TkP2ExpertFmt>(), 96);
        assert_eq!(std::mem::size_of::<crate::qwen4exp::ffi::TkP2Stats>(), 80);
        assert_eq!(std::mem::size_of::<TkP2StagerJob>(), 32);
    }
}
