//! Pure-Rust model & engine orchestrator for Flash-Next (`qwen4exp`), driving `libtalus_kernels.so`
//! via `tk_qwen4exp_rust_driver_t` and the fine-grained `tk_qwen4exp_serve_*` / `tk_qwen4exp_gen_*` C-ABI:
//! - Background stdin reader thread with immediate `STOP` interception (`tk_qwen4exp_request_stop`)
//!   and arbitrary-length line storage (supporting 60K–128K token `GEN` lines without truncation).
//! - Pure-Rust `--serve` protocol loop (`cb_run_serve_loop`) owning command dispatch, request validation,
//!   prompt cut & checkpoint boundary planning (`compute_prompt_cuts`), segmented prefill orchestration
//!   (short verify-window vs batched prefill), and speculative decode (`SuffixDrafter` + `DraftPolicy`).
//! - Pure-Rust single-request `generate` loop (`cb_run_generate_loop`) owning batched prefill, per-token
//!   conditioning/decode, and speculative verify/MTP decode.
//! - Keeps the legacy C++ loop selectable via `--cpp-engine` or `TALUS_FN_CPP=1`.

use core::ffi::{c_char, c_double, c_float, c_int, c_void};
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use crate::qwen4exp::cli::{
    classify_serve_line, parse_fn_dump_compare_args, parse_serve_gen_line, use_cpp_engine,
    ServeCommandKind,
};
use crate::qwen4exp::ffi::P5Kernels; // qwen4exp-p5
use crate::qwen4exp::ffi::{
    FnRequestStop, Qwen4ExpKernels, TkQwen4ExpGenCtx, TkQwen4ExpGenInfo, TkQwen4ExpReqSampling,
    TkQwen4ExpRustDriver, TkQwen4ExpServeCtx, TkQwen4ExpServeInfo,
};
use crate::qwen4exp::kld_flags::{split_rust_only_flags, KldCliOpts}; // qwen4exp-p5
use crate::qwen4exp::skeleton::{LayerDumpHarness, Qwen4ExpModelConfig, Qwen4ExpRustEngine};
use crate::qwen4exp::spec::{parse_logit_bias, DraftPolicy, SuffixDrafter};
use crate::qwen4exp::state::compute_prompt_cuts;
use talus_spec::{copy_cfg_from_env, copy_enabled, copy_env_f64, CopyMatch, CopyProposer};

struct StdinShared {
    queue: Mutex<(VecDeque<String>, bool)>,
    cv: Condvar,
    stop: AtomicBool,
    request_stop_fn: FnRequestStop,
}

struct StdinReader {
    shared: Option<Arc<StdinShared>>,
    current_line: CString,
}

impl StdinReader {
    fn new(_request_stop_fn: FnRequestStop) -> Self {
        Self {
            shared: None,
            current_line: CString::new("").unwrap(),
        }
    }

    fn ensure_started(&mut self, request_stop_fn: FnRequestStop) -> Arc<StdinShared> {
        if let Some(ref s) = self.shared {
            return Arc::clone(s);
        }
        let shared = Arc::new(StdinShared {
            queue: Mutex::new((VecDeque::new(), false)),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
            request_stop_fn,
        });
        let bg_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("talus-qwen4exp-stdin".to_string())
            .spawn(move || {
                let stdin = std::io::stdin();
                let mut reader = std::io::BufReader::new(stdin.lock());
                let mut line = String::new();
                loop {
                    line.clear();
                    match reader.read_line(&mut line) {
                        Ok(0) => {
                            let mut g = bg_shared.queue.lock().unwrap();
                            g.1 = true;
                            bg_shared.cv.notify_all();
                            break;
                        }
                        Ok(_) => {
                            while line.ends_with('\n') || line.ends_with('\r') {
                                line.pop();
                            }
                            if matches!(classify_serve_line(&line), ServeCommandKind::Stop) {
                                bg_shared.stop.store(true, Ordering::Release);
                                unsafe {
                                    (bg_shared.request_stop_fn)();
                                }
                                continue;
                            }
                            let is_quit =
                                matches!(classify_serve_line(&line), ServeCommandKind::Quit);
                            let mut g = bg_shared.queue.lock().unwrap();
                            g.0.push_back(std::mem::take(&mut line));
                            bg_shared.cv.notify_one();
                            if is_quit {
                                break;
                            }
                        }
                        Err(_) => {
                            let mut g = bg_shared.queue.lock().unwrap();
                            g.1 = true;
                            bg_shared.cv.notify_all();
                            break;
                        }
                    }
                }
            })
            .expect("failed to spawn talus-qwen4exp-stdin thread");
        self.shared = Some(Arc::clone(&shared));
        shared
    }

    fn next_line(
        &mut self,
        block: bool,
        request_stop_fn: FnRequestStop,
    ) -> Option<(*const c_char, usize)> {
        let shared = self.ensure_started(request_stop_fn);
        let mut guard = shared.queue.lock().unwrap();
        if block {
            while guard.0.is_empty() && !guard.1 {
                guard = shared.cv.wait(guard).unwrap();
            }
        }
        if let Some(s) = guard.0.pop_front() {
            drop(guard);
            let clean: Vec<u8> = s.into_bytes().into_iter().filter(|&b| b != 0).collect();
            let len = clean.len();
            self.current_line = unsafe { CString::from_vec_unchecked(clean) };
            Some((self.current_line.as_ptr(), len))
        } else if guard.1 {
            None
        } else {
            Some((std::ptr::null(), 0))
        }
    }

    fn next_line_string(&mut self, block: bool, request_stop_fn: FnRequestStop) -> Option<String> {
        let shared = self.ensure_started(request_stop_fn);
        let mut guard = shared.queue.lock().unwrap();
        if block {
            while guard.0.is_empty() && !guard.1 {
                guard = shared.cv.wait(guard).unwrap();
            }
        }
        guard.0.pop_front()
    }

    fn clear_stop(&self) {
        if let Some(ref s) = self.shared {
            s.stop.store(false, Ordering::Release);
        }
    }

    fn is_stopped(&self) -> bool {
        if let Some(ref s) = self.shared {
            s.stop.load(Ordering::Acquire)
        } else {
            false
        }
    }
}

/// the long-span copy proposer (`talus_spec::CopyProposer`, idea from gufo, MIT) in the Rust serve loop.
/// A copy of a long earlier stretch of the context either fills the next verify window on its own (the MTP then only
/// catches up and makes one draft: `Piece4VerifyMtp::draft_step_cap`) or extends the MTP's drafts in the same
/// window.  Drafting only: greedy output is unchanged, sampled output stays exact (a copy is a point-mass proposal).
/// `TALUS_COPY=0` turns it off; `TALUS_COPY_BYPASS=0` / `TALUS_COPY_CHAIN=0` turn off either use.
struct FnCopy {
    cp: CopyProposer,
    bypass: bool,
    chain: bool,
    /// a verify row's cost relative to a one-token step (`TALUS_COPY_ROWCOST`)
    row_cost: f64,
    /// the expected speedup a copy must promise (`TALUS_COPY_GAIN`)
    min_gain: f64,
    /// the next window's copy (decided before the MTP drafts for it)
    next: Option<CopyMatch>,
    next_toks: Vec<i32>,
    /// this window's copy: (match, rows, chained after the MTP's drafts)
    cur: Option<(CopyMatch, usize, bool)>,
    out: Vec<i32>,
    /// `TALUS_COPY_LOG=1`: round time sums / counts by window size and kind (0 MTP, 1 copy, 2 MTP + copy)
    log: bool,
    tsum: [[f64; 9]; 3],
    tn: [[u32; 9]; 3],
}

impl FnCopy {
    fn from_env() -> Option<Self> {
        if !copy_enabled() {
            return None;
        }
        let flag = |n: &str| copy_env_f64(n).is_none_or(|v| v != 0.0);
        Some(Self {
            cp: CopyProposer::new(copy_cfg_from_env()),
            bypass: flag("BYPASS"),
            chain: flag("CHAIN"),
            row_cost: copy_env_f64("ROWCOST").unwrap_or(0.10).max(0.0),
            min_gain: copy_env_f64("GAIN").unwrap_or(0.03),
            next: None,
            next_toks: Vec::new(),
            cur: None,
            out: Vec::new(),
            log: copy_env_f64("LOG").is_some_and(|v| v != 0.0),
            tsum: [[0.0; 9]; 3],
            tn: [[0; 9]; 3],
        })
    }

    fn time_round(&mut self, kind: usize, t: i32, ms: f64) {
        if self.log && (1..=8).contains(&t) {
            self.tsum[kind][t as usize] += ms;
            self.tn[kind][t as usize] += 1;
        }
    }

    /// A request starts with `ids` as its whole context.
    fn begin(&mut self, ids: &[i32]) {
        self.cp.sync(ids);
        self.cp.begin_request();
        self.next = None;
        self.cur = None;
        let _ = self.cp.take_stats();
    }

    /// Decides the next window's copy (before its drafts are made); true when it has one.
    fn plan_next(&mut self, cap: i32, sampled: bool, eos: &[i64]) -> bool {
        self.next = None;
        if !self.bypass || cap < 1 {
            return false;
        }
        let m = self.cp.propose(
            &[],
            cap as usize,
            sampled,
            sampled,
            |t| eos.contains(&(t as i64)),
            &mut self.out,
        );
        if let Some(m) = m {
            let k = self
                .cp
                .choose_len(&m, self.out.len(), self.row_cost, self.min_gain);
            if k > 0 {
                self.next_toks.clear();
                self.next_toks.extend_from_slice(&self.out[..k]);
                self.next = Some(m);
                return true;
            }
        }
        false
    }

    /// Extends the MTP's `drafts` with a copy of up to `cap` tokens into `cbuf`; returns how many.
    fn chain_after(
        &mut self,
        drafts: &[i32],
        cap: i32,
        sampled: bool,
        eos: &[i64],
        cbuf: &mut [i32],
    ) -> i32 {
        if !self.chain || cap < 1 || drafts.is_empty() {
            return 0;
        }
        if eos.iter().any(|&e| drafts.contains(&(e as i32))) {
            return 0;
        }
        let cap = (cap as usize).min(cbuf.len());
        let Some(m) = self.cp.propose(
            drafts,
            cap,
            sampled,
            false,
            |t| eos.contains(&(t as i64)),
            &mut self.out,
        ) else {
            return 0;
        };
        // the MTP rows come first: their drafts are paid for already, so only the copy's own rows are priced
        let k = self
            .cp
            .choose_len(&m, self.out.len(), self.row_cost, self.min_gain);
        if k == 0 {
            return 0;
        }
        cbuf[..k].copy_from_slice(&self.out[..k]);
        self.cur = Some((m, k, true));
        k as i32
    }

    /// Records the verdict on this window's copy (`a` drafts accepted, `t_mtp - 1` of them the MTP's).
    fn observe(&mut self, a: i32, t_mtp: i32) {
        if let Some((m, k, chained)) = self.cur.take() {
            if chained {
                let mtp_rows = (t_mtp - 1).max(0);
                if a >= mtp_rows {
                    self.cp.observe(&m, k, (a - mtp_rows) as usize, true);
                }
            } else {
                self.cp.observe(&m, k, a.max(0) as usize, false);
            }
        }
    }

    fn report(&mut self) {
        if self.log {
            for (kind, name) in ["mtp", "copy", "mtp+copy"].iter().enumerate() {
                let cells: Vec<String> = (1..=8)
                    .filter(|&t| self.tn[kind][t] > 0)
                    .map(|t| {
                        format!(
                            "t{t} {:.1}ms x{}",
                            self.tsum[kind][t] / self.tn[kind][t] as f64,
                            self.tn[kind][t]
                        )
                    })
                    .collect();
                if !cells.is_empty() {
                    eprintln!("talus copy: rounds {name}: {}", cells.join(", "));
                }
            }
            self.tsum = [[0.0; 9]; 3];
            self.tn = [[0; 9]; 3];
        }
        let s = self.cp.take_stats();
        if s.windows + s.chain_windows > 0 {
            eprintln!(
                "talus copy: {} windows, {} of {} copied drafts accepted; chained after the MTP: {} windows, {} of {}",
                s.windows, s.accepted, s.proposed, s.chain_windows, s.chain_accepted, s.chain_proposed
            );
        }
    }
}

/// copies fill verify windows of up to `TALUS_COPY_VERIFY` (default 8, the verify kernels' kVerifyMaxT)
/// tokens.  The C++ option parser widens `--spec S` to S + 2 for prompt lookup; with the copy proposer on and no
/// `--mtp-max-t`, the window becomes `--spec 8 --mtp-max-t S` instead (the MTP keeps its S-token windows).
fn widen_verify_for_copy(argv: Vec<String>) -> Vec<String> {
    if !copy_enabled() || argv.iter().any(|a| a == "--mtp-max-t") {
        return argv;
    }
    let w = copy_env_f64("VERIFY")
        .map(|v| v as i64)
        .unwrap_or(8)
        .clamp(2, 8);
    let Some(i) = argv.iter().rposition(|a| a == "--spec") else {
        return argv;
    };
    let Some(s) = argv.get(i + 1).and_then(|v| v.trim().parse::<i64>().ok()) else {
        return argv;
    };
    if s < 2 || s >= w {
        return argv;
    }
    let mut out = argv;
    out[i + 1] = w.to_string();
    out.push("--mtp-max-t".into());
    out.push(s.to_string());
    out
}

struct SpecState {
    suffix_draft: i32,
    lookup_chain: i32,
    lookup_chain_min: i32,
    sfx: SuffixDrafter,
    policy: DraftPolicy,
}

impl SpecState {
    fn new(
        max_t: i32,
        suffix_draft: i32,
        lookup_chain: i32,
        lookup_chain_min: i32,
        capacity_tokens: usize,
    ) -> Self {
        let min_match = if suffix_draft > 0 {
            suffix_draft
        } else {
            lookup_chain_min
        };
        Self {
            suffix_draft,
            lookup_chain,
            lookup_chain_min,
            sfx: SuffixDrafter::new(min_match, 64, capacity_tokens.max(1024)),
            policy: DraftPolicy::new(max_t, 0.05),
        }
    }
}

pub struct Qwen4ExpDriverState {
    stdin_reader: StdinReader,
    request_stop_fn: FnRequestStop,
    kernels: Qwen4ExpKernels,
    cpp_engine: bool,
    spec: Option<SpecState>,
    pub rust_engine: Qwen4ExpRustEngine,
}

impl Qwen4ExpDriverState {
    pub fn new(kernels: Qwen4ExpKernels, cpp_engine: bool) -> Self {
        let mut rust_engine =
            Qwen4ExpRustEngine::new(Qwen4ExpModelConfig::default(), Some(kernels));
        rust_engine.seam.rust_data_plane = !cpp_engine;
        if !cpp_engine {
            // qwen4exp-p3 hook: the decode window graphs are recorded by Rust (decode.rs)
            if let Err(e) = rust_engine.piece3_decode.install_window_recorder() {
                eprintln!("talus-qwen4exp: qwen4exp-p3 window recorder not installed: {e}");
            }
            // qwen4exp-p4 hook: every Verifier::run / commit and MtpDrafter::draft / prefill to the Rust plane
            if let Err(e) = crate::qwen4exp::verify_mtp::Piece4VerifyMtp::install_plane(&kernels) {
                eprintln!("talus-qwen4exp: {e}");
            }
        }
        Self {
            stdin_reader: StdinReader::new(kernels.request_stop),
            request_stop_fn: kernels.request_stop,
            kernels,
            cpp_engine,
            spec: None,
            rust_engine,
        }
    }

    pub fn build_c_driver(&mut self) -> TkQwen4ExpRustDriver {
        TkQwen4ExpRustDriver {
            user_data: (self as *mut Self).cast::<c_void>(),
            next_line: Some(cb_next_line),
            clear_stop: Some(cb_clear_stop),
            is_stopped: Some(cb_is_stopped),
            parse_logit_bias: Some(cb_parse_logit_bias),
            plan_prompt_cuts: Some(cb_plan_prompt_cuts),
            init_spec: Some(cb_init_spec),
            reset_request_spec: Some(cb_reset_request_spec),
            plan_decode_window: Some(cb_plan_decode_window),
            append_emitted_token: Some(cb_append_emitted_token),
            observe_decode_round: Some(cb_observe_decode_round),
            run_serve_loop: if self.cpp_engine {
                None
            } else {
                Some(cb_run_serve_loop)
            },
            run_generate_loop: if self.cpp_engine {
                None
            } else {
                Some(cb_run_generate_loop)
            },
        }
    }
}

unsafe extern "C" fn cb_next_line(
    user_data: *mut c_void,
    block: c_int,
    out_ptr: *mut *const c_char,
    out_len: *mut usize,
) -> c_int {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    let req_stop = state.request_stop_fn;
    match state.stdin_reader.next_line(block != 0, req_stop) {
        Some((ptr, len)) => {
            if ptr.is_null() {
                0
            } else {
                if !out_ptr.is_null() {
                    *out_ptr = ptr;
                }
                if !out_len.is_null() {
                    *out_len = len;
                }
                1
            }
        }
        None => -1,
    }
}

unsafe extern "C" fn cb_clear_stop(user_data: *mut c_void) {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    state.stdin_reader.clear_stop();
}

unsafe extern "C" fn cb_is_stopped(user_data: *mut c_void) -> c_int {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    if state.stdin_reader.is_stopped() {
        1
    } else {
        0
    }
}

unsafe extern "C" fn cb_parse_logit_bias(
    _user_data: *mut c_void,
    text: *const c_char,
    vocab: c_int,
    out_bias: *mut c_float,
    out_err: *mut c_char,
    err_cap: usize,
) -> c_int {
    let s = if text.is_null() {
        ""
    } else {
        match CStr::from_ptr(text).to_str() {
            Ok(v) => v,
            Err(_) => {
                write_err_buf(
                    out_err,
                    err_cap,
                    "bad logit_bias (expected id:bias,id:bias)",
                );
                return 0;
            }
        }
    };
    let vocab_usize = vocab.max(0) as usize;
    match parse_logit_bias(s, vocab_usize) {
        Ok(vec) => {
            if !out_bias.is_null() && vocab_usize > 0 {
                let dst = std::slice::from_raw_parts_mut(out_bias, vocab_usize);
                if vec.is_empty() {
                    dst.fill(0.0);
                } else {
                    dst.copy_from_slice(&vec[..vocab_usize]);
                }
            }
            1
        }
        Err(msg) => {
            write_err_buf(out_err, err_cap, &msg);
            0
        }
    }
}

unsafe fn write_err_buf(out_err: *mut c_char, err_cap: usize, msg: &str) {
    if out_err.is_null() || err_cap == 0 {
        return;
    }
    let bytes = msg.as_bytes();
    let copy_len = bytes.len().min(err_cap - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_err.cast::<u8>(), copy_len);
    *out_err.add(copy_len) = 0;
}

unsafe extern "C" fn cb_plan_prompt_cuts(
    _user_data: *mut c_void,
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
) -> c_int {
    let slice = if ids.is_null() || n_ids == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ids, n_ids)
    };
    let plan = compute_prompt_cuts(
        slice,
        resume,
        read_from,
        resumed_from0 != 0,
        reread_to,
        prompt_cache,
        turn_token,
        tail_role_token,
        prompt_cache_root,
        req_ckpt != 0,
        multi_gpu != 0,
        message_checkpoint_env != 0,
        pin_at,
    );
    if !out_turn_at.is_null() {
        *out_turn_at = plan.turn_at;
    }
    if !out_root_at.is_null() {
        *out_root_at = plan.root_at;
    }
    if !out_message_at.is_null() {
        *out_message_at = plan.message_at;
    }
    let n_cuts = plan.cuts.len().min(6);
    if !out_cuts.is_null() && n_cuts > 0 {
        std::ptr::copy_nonoverlapping(plan.cuts.as_ptr(), out_cuts, n_cuts);
    }
    if !out_n_cuts.is_null() {
        *out_n_cuts = n_cuts;
    }
    1
}

unsafe extern "C" fn cb_init_spec(
    user_data: *mut c_void,
    max_t: c_int,
    suffix_draft: c_int,
    lookup_chain: c_int,
    lookup_chain_min: c_int,
    capacity_tokens: usize,
) {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    state.spec = Some(SpecState::new(
        max_t,
        suffix_draft,
        lookup_chain,
        lookup_chain_min,
        capacity_tokens,
    ));
}

unsafe extern "C" fn cb_reset_request_spec(
    user_data: *mut c_void,
    prompt_ids: *const i64,
    n_prompt: usize,
    produced_ids: *const i64,
    n_produced: usize,
) {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    let Some(ref mut spec) = state.spec else {
        return;
    };
    spec.sfx.reset();
    if !prompt_ids.is_null() && n_prompt > 0 {
        let p_slice = std::slice::from_raw_parts(prompt_ids, n_prompt);
        for &tok in p_slice {
            spec.sfx.append(tok as i32);
        }
    }
    if !produced_ids.is_null() && n_produced > 0 {
        let prod_slice = std::slice::from_raw_parts(produced_ids, n_produced);
        for &tok in prod_slice {
            spec.sfx.append(tok as i32);
        }
    }
}

unsafe extern "C" fn cb_plan_decode_window(
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
) -> c_int {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    let Some(ref mut spec) = state.spec else {
        return 0;
    };

    let is_first = first_window != 0;
    let mtp_on = use_mtp != 0;
    let mut t = initial_t;

    let drafts_len = (s_mtp.max(max_s).max(1) as usize).saturating_sub(1);
    let drafts_slice: &[i32] = if drafts.is_null() || drafts_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(drafts, drafts_len)
    };
    let dprob_slice: &[f32] = if dprob.is_null() || drafts_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(dprob, drafts_len)
    };

    if mtp_on && spec_min_p > 0.0 {
        let cap = s_mtp.min(room);
        t = 1;
        while t < cap
            && ((t - 1) as usize) < dprob_slice.len()
            && (dprob_slice[(t - 1) as usize] as f64) >= spec_min_p
        {
            t += 1;
        }
    }
    if is_first {
        t = 1;
    }

    let sbuf_cap = (max_s.max(1) as usize).saturating_sub(1);
    let sbuf_slice: &mut [i32] = if out_sbuf.is_null() || sbuf_cap == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_sbuf, sbuf_cap)
    };
    let cbuf_slice: &mut [i32] = if out_cbuf.is_null() || sbuf_cap == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut(out_cbuf, sbuf_cap)
    };

    let mut from_sfx = false;
    let mut sfx_match = 0i32;
    let max_lookup = max_s.min(room) - 1;
    if spec.suffix_draft > 0 && !is_first && max_lookup > 0 {
        let k = spec.sfx.propose(max_lookup, sbuf_slice);
        sfx_match = spec.sfx.last_match();
        let mtp_agrees = !mtp_on
            || (!sbuf_slice.is_empty()
                && !drafts_slice.is_empty()
                && sbuf_slice[0] == drafts_slice[0]);
        if k > 0 && mtp_agrees {
            let pk = spec.policy.choose(t, k, sfx_match);
            if pk.lookup {
                t = pk.t;
                from_sfx = true;
            }
        }
    }

    let mut chain_n = 0i32;
    let mut cm = 0i32;
    let chain_cap = max_s.min(room) - t;
    if allow_chain != 0 && spec.lookup_chain > 0 && !is_first && !from_sfx && chain_cap > 0 {
        let ask_k = spec.lookup_chain.min(chain_cap);
        let (cn, m) = spec.sfx.propose_chain(
            drafts_slice,
            t - 1,
            ask_k,
            spec.lookup_chain_min,
            cbuf_slice,
        );
        chain_n = cn;
        cm = m;
        if chain_n > 0 && chain_fixed == 0 {
            let mut p_mtp = 1.0f64;
            for i in 0..((t - 1).max(0) as usize).min(dprob_slice.len()) {
                p_mtp *= dprob_slice[i] as f64;
            }
            chain_n = spec.policy.chain(t, p_mtp, chain_n, cm);
        }
    }

    let t_mtp = t;
    t += chain_n;

    if !out_t.is_null() {
        *out_t = t;
    }
    if !out_t_mtp.is_null() {
        *out_t_mtp = t_mtp;
    }
    if !out_from_sfx.is_null() {
        *out_from_sfx = if from_sfx { 1 } else { 0 };
    }
    if !out_sfx_match.is_null() {
        *out_sfx_match = sfx_match;
    }
    if !out_chain_n.is_null() {
        *out_chain_n = chain_n;
    }
    if !out_chain_match.is_null() {
        *out_chain_match = cm;
    }
    1
}

unsafe extern "C" fn cb_append_emitted_token(user_data: *mut c_void, token: i32) {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    if let Some(ref mut spec) = state.spec {
        spec.sfx.append(token);
    }
}

unsafe extern "C" fn cb_observe_decode_round(
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
) {
    if timed_round == 0 || eos != 0 {
        return;
    }
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    let Some(ref mut spec) = state.spec else {
        return;
    };
    if chain_n == 0 {
        spec.policy
            .observe(from_sfx != 0, t, accepted, sfx_match, round_ms);
    } else {
        spec.policy
            .observe_chain(t_mtp, chain_n, accepted, chain_match, round_ms);
    }
}

struct BusyGuard {
    set_busy: unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx, busy: c_int),
    ctx: *mut TkQwen4ExpServeCtx,
}

impl BusyGuard {
    unsafe fn new(
        set_busy: unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx, busy: c_int),
        ctx: *mut TkQwen4ExpServeCtx,
    ) -> Self {
        set_busy(ctx, 1);
        Self { set_busy, ctx }
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        unsafe {
            (self.set_busy)(self.ctx, 0);
        }
    }
}

fn plan_window_rust(
    spec: &mut SpecState,
    first_window: bool,
    use_mtp: bool,
    initial_t: i32,
    s_mtp: i32,
    max_s: i32,
    room: i32,
    allow_chain: bool,
    spec_min_p: f64,
    chain_fixed: bool,
    drafts: &[i32],
    dprob: &[f32],
    sbuf: &mut [i32],
    cbuf: &mut [i32],
) -> (i32, i32, bool, i32, i32, i32) {
    let mut t = initial_t;
    if use_mtp && spec_min_p > 0.0 {
        let cap = s_mtp.min(room);
        t = 1;
        while t < cap
            && ((t - 1) as usize) < dprob.len()
            && (dprob[(t - 1) as usize] as f64) >= spec_min_p
        {
            t += 1;
        }
    }
    if first_window {
        t = 1;
    }

    let mut from_sfx = false;
    let mut sfx_match = 0i32;
    let max_lookup = max_s.min(room) - 1;
    if spec.suffix_draft > 0 && !first_window && max_lookup > 0 {
        let k = spec.sfx.propose(max_lookup, sbuf);
        sfx_match = spec.sfx.last_match();
        let mtp_agrees =
            !use_mtp || (!sbuf.is_empty() && !drafts.is_empty() && sbuf[0] == drafts[0]);
        if k > 0 && mtp_agrees {
            let pk = spec.policy.choose(t, k, sfx_match);
            if pk.lookup {
                t = pk.t;
                from_sfx = true;
            }
        }
    }

    let mut chain_n = 0i32;
    let mut cm = 0i32;
    let chain_cap = max_s.min(room) - t;
    if allow_chain && spec.lookup_chain > 0 && !first_window && !from_sfx && chain_cap > 0 {
        let ask_k = spec.lookup_chain.min(chain_cap);
        let (cn, m) = spec
            .sfx
            .propose_chain(drafts, t - 1, ask_k, spec.lookup_chain_min, cbuf);
        chain_n = cn;
        cm = m;
        if chain_n > 0 && !chain_fixed {
            let mut p_mtp = 1.0f64;
            for &pr in dprob.iter().take((t - 1).max(0) as usize) {
                p_mtp *= pr as f64;
            }
            chain_n = spec.policy.chain(t, p_mtp, chain_n, cm);
        }
    }

    let t_mtp = t;
    t += chain_n;
    (t, t_mtp, from_sfx, sfx_match, chain_n, cm)
}

unsafe extern "C" fn cb_run_serve_loop(
    user_data: *mut c_void,
    ctx: *mut TkQwen4ExpServeCtx,
) -> c_int {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    let k = state.kernels;
    let mut info = TkQwen4ExpServeInfo::default();
    (k.serve_get_info)(ctx, &mut info);
    let _ = state.rust_engine.piece1_weights.sync_from_ctx(&k, ctx);
    let _ = state
        .rust_engine
        .piece5_conv_state
        .query_state_sizes(&k, ctx);
    // ---- qwen4exp-p5: the conversation state (checkpoints, parking, SAVE/RESTORE) is Rust-owned (conv_state.rs) ----
    if std::env::var("TALUS_FN_CONV_CPP").ok().as_deref() != Some("1")
        && std::env::var("TALUS_FN_P5").ok().as_deref() != Some("0")
    {
        if let Err(e) = state.rust_engine.piece5_conv_state.attach_serve(ctx) {
            eprintln!("talus-qwen4exp: conversation state stays in C++: {e}");
        }
    }

    // with copies on, the verify window is 8 (widen_verify_for_copy) but the suffix drafter / draft policy keep
    // their release width (the MTP's windows + 2): only copies fill the extra rows
    let copy_on_here = copy_enabled();
    let sfx_spec = if copy_on_here && info.lookup_chain == 0 && info.mtp_max_t >= 2 {
        info.spec.min(info.mtp_max_t + 2)
    } else {
        info.spec
    };
    if state.spec.is_none() {
        state.spec = Some(SpecState::new(
            sfx_spec,
            info.suffix_draft,
            info.lookup_chain,
            info.lookup_chain_min,
            (info.max_context.max(1024) as usize) + 4096,
        ));
    }

    let eos_slice = &info.eos_ids[..info.n_eos.min(16)];
    let chain_fixed = std::env::var_os("STRATA_LOOKUP_CHAIN_FIXED").is_some();
    let mut copy = FnCopy::from_env();
    if copy.is_some() {
        eprintln!(
            "talus copy: long-span copy drafts on (verify window {}, MTP windows {}; TALUS_COPY=0 turns them off)",
            info.spec, info.mtp_max_t
        );
    }

    loop {
        let req_stop = state.request_stop_fn;
        let Some(line) = state.stdin_reader.next_line_string(true, req_stop) else {
            break;
        };
        let is_quit = line == "QUIT";
        let c_line = CString::new(
            line.as_bytes()
                .iter()
                .copied()
                .filter(|&b| b != 0)
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let pre_rc = (k.serve_pre_command)(ctx, c_line.as_ptr(), if is_quit { 1 } else { 0 });
        if pre_rc < 0 {
            return 1;
        }
        if is_quit {
            break;
        }
        if pre_rc == 0 {
            continue;
        }

        let _busy = BusyGuard::new(k.serve_set_busy, ctx);

        let parsed = match parse_serve_gen_line(&line, info.pcie_frac, info.spec_min_p) {
            Ok(p) => p,
            Err(err_line) => {
                println!("{err_line}");
                let _ = std::io::stdout().flush();
                continue;
            }
        };

        let mut req_bias: Vec<f32> = Vec::new();
        if let Some(ref btxt) = parsed.bias_text {
            match parse_logit_bias(btxt, info.n_vocab.max(0) as usize) {
                Ok(v) => req_bias = v,
                Err(e) => {
                    println!("ERR logit_bias: {e}");
                    let _ = std::io::stdout().flush();
                    continue;
                }
            }
        }

        let n_prompt = parsed.ids.len() as i64;
        if n_prompt + parsed.max_new + 8 > info.max_context {
            println!(
                "ERR prompt ({} tokens) + max_new ({}) exceeds the context ({})",
                n_prompt, parsed.max_new, info.max_context
            );
            let _ = std::io::stdout().flush();
            continue;
        }
        if parsed.ids.iter().any(|&t| t < 0 || t >= info.n_vocab) {
            println!("ERR a token id is outside the vocabulary");
            let _ = std::io::stdout().flush();
            continue;
        }

        let r0 = Instant::now();
        let sampling = TkQwen4ExpReqSampling {
            temperature: parsed.temperature,
            top_p: parsed.top_p,
            top_k: parsed.top_k,
            min_p: parsed.min_p,
            penalty_last_n: parsed.penalty_last_n,
            penalty_repeat: parsed.penalty_repeat,
            penalty_freq: parsed.penalty_freq,
            penalty_present: parsed.penalty_present,
            seed: parsed.seed,
            pcie_frac: parsed.pcie_frac,
            spec_min_p: parsed.spec_min_p,
            aux_cpus: parsed.aux_cpus,
        };

        let mut resume = 0i64;
        let mut read_from = 0i64;
        let mut resumed_from0 = 0i32;
        let mut reread_to = -1i64;
        let mut pin_at = -1i64;

        let begin_rc = (k.serve_begin_request)(
            ctx,
            parsed.ids.as_ptr(),
            parsed.ids.len(),
            if parsed.req_cache { 1 } else { 0 },
            parsed.req_ckpt,
            parsed.req_pin,
            parsed.req_cvec,
            &sampling,
            if req_bias.is_empty() {
                std::ptr::null()
            } else {
                req_bias.as_ptr()
            },
            req_bias.len(),
            &mut resume,
            &mut read_from,
            &mut resumed_from0,
            &mut reread_to,
            &mut pin_at,
        );
        if begin_rc < 0 {
            return 1;
        }
        if begin_rc == 0 {
            continue;
        }

        let cut_plan = compute_prompt_cuts(
            &parsed.ids,
            resume,
            read_from,
            resumed_from0 != 0,
            reread_to,
            info.prompt_cache,
            info.turn_token,
            info.tail_role_token,
            info.prompt_cache_root,
            parsed.req_ckpt != 0,
            false,
            info.message_checkpoint_env != 0,
            pin_at,
        );

        let mut at = read_from;
        let mut cancelled = false;
        let mut nf_hit = false;

        for &to in &cut_plan.cuts {
            if to <= at {
                continue;
            }
            let use_win = if to - at <= info.short_read { 1 } else { 0 };
            let mut seg_nf = 0i32;
            let mut seg_cancel = 0i32;
            let seg_rc =
                (k.serve_prefill_segment)(ctx, at, to, use_win, &mut seg_nf, &mut seg_cancel);
            if seg_rc < 0 {
                return 1;
            }
            if seg_nf != 0 {
                nf_hit = true;
            }
            if seg_rc == 0 || seg_cancel != 0 {
                cancelled = true;
                break;
            }
            at = to;
            if to == cut_plan.turn_at
                || to == cut_plan.root_at
                || to == cut_plan.message_at
                || to == pin_at
            {
                let as_tail = if to == cut_plan.message_at
                    && to != cut_plan.turn_at
                    && to != cut_plan.root_at
                    && to != pin_at
                {
                    1
                } else {
                    0
                };
                let is_pin = if to == pin_at { 1 } else { 0 };
                if (k.serve_checkpoint_at)(ctx, to, as_tail, is_pin, cut_plan.turn_at) < 0 {
                    return 1;
                }
            }
        }

        if (k.serve_prefill_finish)(ctx) < 0 {
            return 1;
        }

        let prompt_ms = r0.elapsed().as_secs_f64() * 1000.0;
        println!("REUSED {resume}");
        let _ = std::io::stdout().flush();

        let s_cap = info.spec.max(info.mtp_max_t).max(1) as usize;
        let mut drafts = vec![0i32; s_cap];
        let mut dprob = vec![0.0f32; s_cap];
        let mut window = vec![0i32; s_cap];
        let mut outv = vec![0i32; s_cap];
        let mut sbuf = vec![0i32; s_cap];
        let mut cbuf = vec![0i32; s_cap];

        let sfx_on = info.suffix_draft > 0 || info.lookup_chain > 0;
        if let Some(ref mut spec) = state.spec {
            if sfx_on {
                spec.sfx.reset();
                for &id in &parsed.ids {
                    spec.sfx.append(id as i32);
                }
            }
        }
        let sampled = parsed.temperature > 0.0;
        if let Some(fc) = copy.as_mut() {
            let ids: Vec<i32> = parsed.ids.iter().map(|&t| t as i32).collect();
            fc.begin(&ids);
        }

        let mut p = n_prompt - 1;
        let mut x = parsed.ids[(n_prompt - 1) as usize] as i32;
        let mut first_window = true;
        let mut produced_n = 0i64;
        let mut rounds = 0i64;
        let mut sfx_windows = 0i64;
        let mut sfx_drafts = 0i64;
        let mut sfx_ok = 0i64;
        let mut chain_windows = 0i64;
        let mut chain_drafts = 0i64;
        let mut chain_ok = 0i64;
        let mut draft_offered = 0i64;
        let mut draft_accepted = 0i64;
        let mut dec_windows = 0i64;

        let mut consumed: Vec<i32> =
            Vec::with_capacity((n_prompt + parsed.max_new + info.spec as i64).max(16) as usize);
        for &id in &parsed.ids[..(n_prompt - 1) as usize] {
            consumed.push(id as i32);
        }

        let mut finish = if cancelled {
            if nf_hit {
                "nonfinite"
            } else {
                "cancel"
            }
        } else {
            "length"
        };

        let d0 = Instant::now();
        let use_mtp = info.use_mtp != 0;

        while !cancelled && produced_n < parsed.max_new {
            let initial_t = if use_mtp { info.mtp_max_t } else { 1 };
            // a copy decided at the end of the last window fills this one
            let copy_next = copy.as_mut().and_then(|fc| fc.next.take().map(|m| (m, fc)));
            let (mut t, t_mtp, from_sfx, sfx_match, mut chain_n, mut cm) =
                if let Some((m, fc)) = copy_next {
                    let n = fc.next_toks.len().min(sbuf.len());
                    sbuf[..n].copy_from_slice(&fc.next_toks[..n]);
                    fc.cur = Some((m, n, false));
                    let t = 1 + n as i32;
                    (t, t, true, m.len as i32, 0, 0)
                } else if let Some(ref mut spec) = state.spec {
                    plan_window_rust(
                        spec,
                        first_window,
                        use_mtp,
                        initial_t,
                        info.mtp_max_t,
                        sfx_spec,
                        sfx_spec,
                        info.lookup_chain > 0,
                        parsed.spec_min_p,
                        chain_fixed,
                        &drafts,
                        &dprob,
                        &mut sbuf,
                        &mut cbuf,
                    )
                } else {
                    (1, 1, false, 0, 0, 0)
                };
            // or a copy extends the MTP's drafts in this window
            if let Some(fc) = copy.as_mut() {
                if use_mtp && !first_window && !from_sfx && chain_n == 0 && t >= 2 {
                    let room_c = (info.max_context - p) as i32 - t;
                    let cap = (info.spec - t).min(room_c);
                    let n = fc.chain_after(
                        &drafts[..(t - 1) as usize],
                        cap,
                        sampled,
                        eos_slice,
                        &mut cbuf,
                    );
                    if n > 0 {
                        chain_n = n;
                        cm = fc.cur.map_or(0, |c| c.0.len as i32);
                        t += n;
                    }
                }
            }
            let copy_round = copy.as_ref().is_some_and(|fc| fc.cur.is_some());
            let copy_kind = if !copy_round {
                0
            } else if from_sfx {
                1
            } else {
                2
            };

            let timed_round = !first_window;
            let round0 = Instant::now();
            if p + (t as i64) > info.max_context {
                break;
            }
            window[0] = x;
            for i in 1..(t_mtp as usize) {
                window[i] = if from_sfx { sbuf[i - 1] } else { drafts[i - 1] };
            }
            for i in 0..(chain_n as usize) {
                window[(t_mtp as usize) + i] = cbuf[i];
            }

            if state.rust_engine.piece4_verify_mtp.serve_verify(
                &k, ctx, t, t_mtp, from_sfx, p, &window, &consumed, &mut outv,
            ) < 0
            // qwen4exp-p4 hook
            {
                return 1;
            }

            if first_window {
                // ---- qwen4exp-p5: STRATA_DUMP_FIRST_LOGITS (kld_flags.rs) ----
                let n_vocab = info.n_vocab.max(0) as usize;
                state
                    .rust_engine
                    .piece6_kld_flags
                    .first_logits
                    .write(true, n_vocab, |row| {
                        (k.serve_copy_logits)(ctx, 0, row.as_mut_ptr()) != 0
                    });
            }

            let mut a = 0i32;
            while a < t - 1 && window[(a + 1) as usize] == outv[a as usize] {
                a += 1;
            }
            if from_sfx {
                sfx_windows += 1;
                sfx_drafts += (t - 1) as i64;
                sfx_ok += a as i64;
            }
            if chain_n > 0 {
                chain_windows += 1;
                chain_drafts += chain_n as i64;
                chain_ok += (a - (t_mtp - 1)).max(0) as i64;
            }
            if let Some(fc) = copy.as_mut() {
                fc.observe(a, t_mtp);
            }

            let mut win_nf = 0i32;
            let commit_rc = state.rust_engine.piece4_verify_mtp.serve_commit(
                &k,
                ctx,
                a + 1,
                rounds,
                first_window,
                &mut win_nf,
            ); // qwen4exp-p4 hook
            if commit_rc < 0 {
                return 1;
            }
            if win_nf != 0 || commit_rc == 0 {
                nf_hit = true;
                finish = "nonfinite";
                break;
            }

            for i in 0..=(a as usize) {
                consumed.push(window[i]);
            }
            draft_offered += (t - 1) as i64;
            draft_accepted += a as i64;
            first_window = false;

            let mut eos = false;
            for i in 0..=(a as usize) {
                if produced_n >= parsed.max_new || eos {
                    break;
                }
                let tok = outv[i];
                println!("T {tok}");
                produced_n += 1;
                if sfx_on {
                    if let Some(ref mut spec) = state.spec {
                        spec.sfx.append(tok);
                    }
                }
                if let Some(fc) = copy.as_mut() {
                    fc.cp.push(tok);
                }
                if !parsed.req_ignore_eos && eos_slice.contains(&(tok as i64)) {
                    eos = true;
                }
            }
            let _ = std::io::stdout().flush();
            rounds += 1;

            let need_draft = if !eos && produced_n < parsed.max_new {
                1
            } else {
                0
            };
            // decide the next window's copy now; with one, the MTP only catches up (and makes one draft)
            let mut step_cap = 0;
            if let Some(fc) = copy.as_mut() {
                let p_next = p + (a + 1) as i64;
                let cap = (info.spec - 1).min((info.max_context - p_next) as i32 - 1);
                if need_draft != 0 && fc.plan_next(cap, sampled, eos_slice) {
                    step_cap = 1;
                }
            }
            state.rust_engine.piece4_verify_mtp.draft_step_cap = step_cap;
            if state.rust_engine.piece4_verify_mtp.serve_draft(
                &k,
                ctx,
                t,
                a,
                p,
                need_draft != 0,
                parsed.spec_min_p,
                &outv,
                &consumed,
                &mut drafts,
                &mut dprob,
            ) < 0
            // qwen4exp-p4 hook
            {
                return 1;
            }
            dec_windows += 1;

            let round_ms = round0.elapsed().as_secs_f64() * 1000.0;
            if let Some(fc) = copy.as_mut() {
                if timed_round && !eos {
                    fc.time_round(copy_kind, t, round_ms);
                }
            }
            if timed_round && !eos && !copy_round {
                if let Some(ref mut spec) = state.spec {
                    if chain_n == 0 {
                        spec.policy.observe(from_sfx, t, a, sfx_match, round_ms);
                    } else {
                        spec.policy.observe_chain(t_mtp, chain_n, a, cm, round_ms);
                    }
                }
            }

            if eos {
                finish = "stop";
                break;
            }
            if state.stdin_reader.is_stopped() {
                finish = "cancel";
                break;
            }
            x = outv[a as usize];
            p += (a + 1) as i64;
        }

        let decode_ms = d0.elapsed().as_secs_f64() * 1000.0;
        if let Some(fc) = copy.as_mut() {
            fc.report();
        }
        let finish_c = CString::new(finish).unwrap();
        if (k.serve_finish_request)(
            ctx,
            if cancelled { 1 } else { 0 },
            if nf_hit { 1 } else { 0 },
            parsed.req_ckpt,
            produced_n,
            n_prompt,
            resume,
            prompt_ms,
            decode_ms,
            finish_c.as_ptr(),
            draft_accepted,
            draft_offered,
            sfx_windows,
            sfx_ok,
            sfx_drafts,
            chain_windows,
            chain_ok,
            chain_drafts,
            dec_windows,
            consumed.as_ptr(),
            consumed.len(),
        ) < 0
        {
            return 1;
        }
    }

    (k.serve_on_exit)(ctx);
    0
}

unsafe extern "C" fn cb_run_generate_loop(
    user_data: *mut c_void,
    ctx: *mut TkQwen4ExpGenCtx,
) -> c_int {
    let state = &mut *user_data.cast::<Qwen4ExpDriverState>();
    let k = state.kernels;
    let mut info = TkQwen4ExpGenInfo::default();
    (k.gen_get_info)(ctx, &mut info);

    let prompt_slice: &[i64] = if info.prompt_tokens.is_null() || info.n_prompt == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(info.prompt_tokens, info.n_prompt)
    };
    let n_prompt = prompt_slice.len() as i64;
    let eos_slice = &info.eos_ids[..info.n_eos.min(16)];

    let mut pos_start = 0i64;
    let mut tok = if !prompt_slice.is_empty() {
        prompt_slice[0]
    } else {
        0
    };
    let mut prefill_ms = 0.0f64;
    let mut total_ms = 0.0f64;
    let mut ttft_ms = 0.0f64;

    // ---- qwen4exp-p5: --dump-logits / --kld-in are owned by kld_flags.rs ----
    let p5k = P5Kernels::resolve().ok();
    if let Err(e) = state.rust_engine.piece6_kld_flags.gen_begin(
        p5k.as_ref(),
        ctx,
        info.n_vocab,
        n_prompt,
        info.max_new,
    ) {
        eprintln!("{e}");
        return 1;
    }

    if (k.gen_prefill_batched)(ctx, &mut pos_start, &mut tok, &mut prefill_ms) < 0 {
        return 1;
    }

    let mut produced: Vec<i64> = Vec::with_capacity(info.max_new.max(0) as usize);
    let mut spec_pos = 0i64;
    let mut pos = pos_start;

    loop {
        if info.native_pack != 0 {
            spec_pos = pos;
            break;
        }
        if pos >= info.max_context {
            eprintln!("strata generate: ran out of context at position {pos}");
            return 2;
        }
        let mut next = 0i32;
        let mut step_ms = 0.0f64;
        if (k.gen_step_token)(ctx, pos, tok, &mut next, &mut step_ms, &mut ttft_ms) < 0 {
            return 1;
        }
        // ---- qwen4exp-p5 ----
        if let Err(e) = state
            .rust_engine
            .piece6_kld_flags
            .gen_after_step(p5k.as_ref(), ctx, pos)
        {
            eprintln!("{e}");
            return 1;
        }
        if pos >= n_prompt - 1 {
            total_ms += step_ms;
            produced.push(next as i64);
        } else {
            prefill_ms += step_ms;
        }
        if (produced.len() as i64) >= info.max_new {
            break;
        }
        if info.stop_eos != 0 && pos >= n_prompt - 1 && eos_slice.contains(&(next as i64)) {
            break;
        }
        tok = if pos + 1 < n_prompt {
            prompt_slice[(pos + 1) as usize]
        } else {
            next as i64
        };
        if info.spec > 0 && pos >= n_prompt - 1 {
            spec_pos = pos + 1;
            break;
        }
        pos += 1;
    }

    let ended =
        info.stop_eos != 0 && !produced.is_empty() && eos_slice.contains(produced.last().unwrap());

    if spec_pos > 0 && (produced.len() as i64) < info.max_new && !ended {
        let s_cap = info.spec.max(1) as usize;
        let mut drafts = vec![0i32; s_cap];
        let mut dprob = vec![1.0f32; s_cap];
        let mut window = vec![0i32; s_cap];
        let mut outv = vec![0i32; s_cap];
        let mut sbuf = vec![0i32; s_cap];
        let mut cbuf = vec![0i32; s_cap];
        let mut window_hist = vec![0i64; s_cap + 1];
        let mut accepted_hist = vec![0i64; s_cap];

        let mut first_window_i = 0i32;
        let mut max_new = info.max_new;
        let mut p = spec_pos;
        let mut x = tok as i32;

        let init_rc = (k.gen_spec_init)(
            ctx,
            spec_pos,
            x,
            &mut first_window_i,
            &mut max_new,
            drafts.as_mut_ptr(),
            dprob.as_mut_ptr(),
        );
        if init_rc == -2 {
            return 2;
        }
        if init_rc < 0 {
            return 1;
        }
        let mut first_window = first_window_i != 0;

        let mut spec_state = SpecState::new(
            info.spec,
            info.suffix_draft,
            info.lookup_chain,
            info.lookup_chain_min,
            (info.max_context.max(1024) as usize) + 4096,
        );
        let sfx_on = info.suffix_draft > 0 || info.lookup_chain > 0;
        if sfx_on {
            spec_state.sfx.reset();
            for &id in prompt_slice {
                spec_state.sfx.append(id as i32);
            }
            for &id in &produced {
                spec_state.sfx.append(id as i32);
            }
        }

        let use_mtp = info.use_mtp != 0;
        let s_mtp = info.mtp_max_t;
        let chain_fixed = std::env::var_os("STRATA_LOOKUP_CHAIN_FIXED").is_some();

        let mut rounds = 0i64;
        let mut drafts_total = 0i64;
        let mut drafts_ok = 0i64;
        let mut corrupt_counter = 0i64;
        let mut sfx_windows = 0i64;
        let mut sfx_drafts = 0i64;
        let mut sfx_ok = 0i64;
        let mut chain_windows = 0i64;
        let mut chain_drafts = 0i64;
        let mut chain_ok = 0i64;

        while (produced.len() as i64) < max_new {
            let t0 = Instant::now();
            let room = (info.max_context - p).min(info.spec as i64) as i32;
            let (t, t_mtp, from_sfx, sfx_match, chain_n, cm) = plan_window_rust(
                &mut spec_state,
                first_window,
                use_mtp,
                s_mtp,
                s_mtp,
                info.spec,
                room,
                info.lookup_chain > 0 && use_mtp,
                info.spec_min_p,
                chain_fixed,
                &drafts,
                &dprob,
                &mut sbuf,
                &mut cbuf,
            );

            let timed_round = !first_window;
            window_hist[t as usize] += 1;
            if p + (t as i64) > info.max_context {
                eprintln!("strata generate: ran out of context at position {p}");
                return 2;
            }
            window[0] = x;
            for i in 1..(t as usize) {
                let mut d = if from_sfx {
                    sbuf[i - 1]
                } else if use_mtp {
                    drafts[i - 1]
                } else {
                    0
                };
                if (i as i32) >= t_mtp {
                    d = cbuf[i - (t_mtp as usize)];
                }
                if info.spec_corrupt > 0 {
                    corrupt_counter += 1;
                    if corrupt_counter % (info.spec_corrupt as i64) == 0 {
                        d = (d + 1) % (info.n_vocab as i32);
                    }
                }
                window[i] = d;
            }

            if state
                .rust_engine
                .piece4_verify_mtp
                .gen_verify(&k, ctx, t, t_mtp, from_sfx, p, &window, &mut outv)
                < 0
            // qwen4exp-p4 hook
            {
                return 1;
            }

            let mut a = 0i32;
            while a < t - 1 && window[(a + 1) as usize] == outv[a as usize] {
                a += 1;
            }
            if first_window {
                first_window = false;
                // ---- qwen4exp-p5: STRATA_DUMP_FIRST_LOGITS (kld_flags.rs) ----
                let n_vocab = info.n_vocab.max(0) as usize;
                state
                    .rust_engine
                    .piece6_kld_flags
                    .first_logits
                    .write(false, n_vocab, |row| {
                        (k.gen_spec_copy_logits)(ctx, 0, row.as_mut_ptr()) != 0
                    });
            }

            if state
                .rust_engine
                .piece4_verify_mtp
                .gen_commit(&k, ctx, a + 1, rounds)
                < 0
            {
                // qwen4exp-p4 hook
                return 1;
            }
            rounds += 1;
            drafts_total += (t - 1) as i64;
            drafts_ok += a as i64;
            accepted_hist[a as usize] += 1;
            if from_sfx {
                sfx_windows += 1;
                sfx_drafts += (t - 1) as i64;
                sfx_ok += a as i64;
            }
            if chain_n > 0 {
                chain_windows += 1;
                chain_drafts += chain_n as i64;
                chain_ok += (a - (t_mtp - 1)).max(0) as i64;
            }

            let mut eos = false;
            for i in 0..=(a as usize) {
                if (produced.len() as i64) >= max_new || eos {
                    break;
                }
                let tok_out = outv[i];
                produced.push(tok_out as i64);
                if sfx_on {
                    spec_state.sfx.append(tok_out);
                }
                if info.stop_eos != 0 && eos_slice.contains(&(tok_out as i64)) {
                    eos = true;
                }
            }
            if eos {
                if state.rust_engine.piece4_verify_mtp.gen_join_adapt(&k, ctx) < 0 {
                    // qwen4exp-p4 hook
                    return 1;
                }
                total_ms += t0.elapsed().as_secs_f64() * 1000.0;
                break;
            }

            let need_draft = if (produced.len() as i64) < max_new {
                1
            } else {
                0
            };
            if state.rust_engine.piece4_verify_mtp.gen_draft(
                &k,
                ctx,
                t,
                a,
                p,
                need_draft != 0,
                info.spec_min_p,
                &outv,
                &mut drafts,
                &mut dprob,
            ) < 0
            // qwen4exp-p4 hook
            {
                return 1;
            }
            x = outv[a as usize];
            p += (a + 1) as i64;
            let round_ms = t0.elapsed().as_secs_f64() * 1000.0;
            total_ms += round_ms;
            if timed_round {
                if chain_n == 0 {
                    spec_state
                        .policy
                        .observe(from_sfx, t, a, sfx_match, round_ms);
                } else {
                    spec_state
                        .policy
                        .observe_chain(t_mtp, chain_n, a, cm, round_ms);
                }
            }
            if rounds % 64 == 0 {
                eprintln!(
                    "strata generate: position {}, {} tokens, {} rounds",
                    p,
                    produced.len(),
                    rounds
                );
            }
        }

        if (k.gen_spec_finish)(
            ctx,
            rounds,
            drafts_ok,
            drafts_total,
            sfx_windows,
            sfx_ok,
            sfx_drafts,
            chain_windows,
            chain_ok,
            chain_drafts,
            window_hist.as_ptr(),
            window_hist.len(),
            accepted_hist.as_ptr(),
            accepted_hist.len(),
        ) < 0
        {
            return 1;
        }
    }

    // ---- qwen4exp-p5 ----
    if let Err(e) = state.rust_engine.piece6_kld_flags.gen_finish() {
        eprintln!("{e}");
        return 1;
    }
    if (k.gen_finish)(
        ctx,
        produced.as_ptr(),
        produced.len(),
        total_ms,
        prefill_ms,
        ttft_ms,
    ) < 0
    {
        return 1;
    }
    state.rust_engine.piece6_kld_flags.gen_report();
    0
}

/// Entry point invoked by `talus-cli` when a Flash-Next (`qwen4exp`) model or `--pack` is specified.
pub fn run_qwen4exp_cli<S: AsRef<str>>(argv: &[S]) -> i32 {
    if let Some((lhs, rhs)) = parse_fn_dump_compare_args(argv) {
        match LayerDumpHarness::compare_dirs(std::path::Path::new(&lhs), std::path::Path::new(&rhs))
        {
            Ok(cmp) => {
                print!("{}", cmp.format_report());
                return if cmp.is_bitwise_identical() { 0 } else { 1 };
            }
            Err(e) => {
                eprintln!("talus-qwen4exp: --fn-dump-compare failed: {e}");
                return 2;
            }
        }
    }

    let kernels = match Qwen4ExpKernels::load(None) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("talus-qwen4exp: {e}");
            return 1;
        }
    };

    let cpp_engine = use_cpp_engine(argv);
    if !cpp_engine {
        eprintln!(
            "talus-qwen4exp: TALUS_FN_RUST=1 active (pure-Rust data plane default unless TALUS_FN_RUST=0, w16={}, tp_moe={})",
            if std::env::var("STRATA_PF_W16").ok().as_deref() == Some("0") {
                0
            } else {
                1
            },
            if std::env::var("TALUS_PF_MOE").ok().as_deref() == Some("0") {
                0
            } else {
                1
            }
        );
    }

    // ---- qwen4exp-p5: Rust-only --logits-out / --kld-in / --kld-top (kld_flags.rs) ----
    let (argv, kld_opts) = if cpp_engine {
        (
            argv.iter()
                .map(|a| a.as_ref().to_string())
                .collect::<Vec<_>>(),
            KldCliOpts::default(),
        )
    } else {
        match split_rust_only_flags(argv) {
            Ok((a, o)) => (widen_verify_for_copy(a), o),
            Err(e) => {
                eprintln!("talus-qwen4exp: {e}");
                return 2;
            }
        }
    };

    let mut c_strings = Vec::with_capacity(argv.len());
    for arg in &argv {
        let s = arg.as_str();
        let bytes: Vec<u8> = s.as_bytes().iter().copied().filter(|&b| b != 0).collect();
        c_strings.push(unsafe { CString::from_vec_unchecked(bytes) });
    }
    let mut c_ptrs: Vec<*const c_char> = c_strings.iter().map(|cs| cs.as_ptr()).collect();
    let argc = c_ptrs.len() as c_int;
    c_ptrs.push(std::ptr::null());

    let mut driver_state = Qwen4ExpDriverState::new(kernels, cpp_engine);
    driver_state.rust_engine.piece6_kld_flags.set_opts(kld_opts);
    let c_driver = driver_state.build_c_driver();

    unsafe { (kernels.run_with_rust_driver)(argc, c_ptrs.as_ptr(), &c_driver) }
}
