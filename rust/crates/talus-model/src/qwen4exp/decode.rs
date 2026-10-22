//! Piece 3 (`qwen4exp-p3`): Flash-Next (`qwen4exp`) single-token decode, owned by Rust.
//!
//! ## What decode is on this engine (and why the port lives where it does)
//! Every Flash-Next pack the engine serves is a NATIVE pack (`native_experts.txt`, `expert_layout().native`), and a
//! native pack never runs the per-layer token graphs of `session.cpp`/`layer.cpp` (`block_layer_pre/post`,
//! `session_loop`, `session_run_token`): `generate.cpp` says "a native pack runs verify windows only", and both the
//! generate loop and the serve loop decode through `Verifier` windows - a plain decode step is the `T = 1` window,
//! with `--spec 0` too. So the production single-token decode is `Verifier::record_window`, and that is what this
//! piece ports: the window's whole layer sequence (embedding, PLE at layer 1, the hyper-connection reads/writes, the
//! GDN recurrent mixer, the QSA sparse-attention mixer, the router + doorbell + routed experts (VRAM hits, PCIe share,
//! CPU pool rows) + shared expert + combine, the LM head and the greedy pick), for `T` rows, recorded from Rust over
//! launch-only wrappers (`tk_qwen4exp_p3_*`, `src/program/qwen4exp_p3.cpp`).
//!
//! The graph is captured ONCE per window size and replayed every step, so the Rust recorder is not on the decode
//! hot path at all; what matters is that it records the same kernels with the same arguments in the same order.
//! Each wrapper enqueues exactly the kernel `record_window` enqueues at that point, so the captured graph is the
//! C++ one node for node and the window's arithmetic is bitwise the C++ path's.
//!
//! ## Per-layer API (Piece 4 builds on this)
//! [`WindowRecorder`] binds one `Verifier` (via [`TkP3WindowView`]) and exposes
//! - [`WindowRecorder::record_inputs`]: the window's token/step/position uploads + embedding broadcast;
//! - [`WindowRecorder::record_pre`]`(l, grp)`: layer `l`'s mixer half and FFN front up to the router/doorbell + shared expert;
//! - [`WindowRecorder::record_post`]`(l, grp)`: layer `l`'s routed experts (hits, PCIe share, CPU rows) + combine;
//! - [`WindowRecorder::record_head`]: the final mixer, LM head, greedy pick, non-finite guard;
//! - [`WindowRecorder::record_window`]: the whole sequence (`pre(lb) | post(l) pre(l+1) | .. | head`).
//!
//! `Piece3Decode::install_window_recorder` registers the recorder with `Verifier::capture` (TALUS_FN_RUST=1); a
//! configuration it does not cover (batch windows, a layer-split stage, the stage profiler / trace, a remote expert
//! helper, STRATA_DF_BRANCH, route prior / route resident, gfx906) is declined before anything is enqueued and the
//! C++ recorder records it instead.
//!
//! ## Residual (not ported, listed for Group B)
//! The S-form (non-native) token-graph decode (`session_capture[_token]`, `session_loop`, `session_run_token`,
//! `block_layer_pre/post`, `generate.cpp` `put_input`/`run_head`/`drive_hits`/`gen_ctx.step_token`) is unreachable
//! with native packs; proposed to drop with Group B.

use core::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use crate::qwen4exp::ffi::{
    Fnp3Fns, Qwen4ExpKernels, TkP3GrArgs, TkP3HeadWeights, TkP3LayerWeights, TkP3QsaView,
    TkP3WindowView, TkQwen4ExpGenCtx,
};
use crate::qwen4exp::skeleton::Qwen4ExpModelConfig;

/// Piece 3 single-token decode state owned by `Qwen4ExpRustEngine`.
#[derive(Clone, Debug)]
pub struct Piece3Decode {
    pub n_layers: i64,
    pub token_graph_enabled: bool,
    pub session_graphs_enabled: bool,
    pub steps_executed: u64,
    /// The Rust window recorder is registered with `Verifier::capture`.
    pub recorder_installed: bool,
}

impl Piece3Decode {
    pub fn new(cfg: &Qwen4ExpModelConfig) -> Self {
        Self {
            n_layers: cfg.n_layers,
            token_graph_enabled: true,
            session_graphs_enabled: true,
            steps_executed: 0,
            recorder_installed: false,
        }
    }

    /// Registers the Rust window recorder with the C++ `Verifier::capture` (every decode window graph the engine
    /// captures from now on is recorded by [`WindowRecorder`]). `TALUS_FN_P3=0` / `TALUS_FN_P3_RECORD=0` keeps the C++ recorder.
    pub fn install_window_recorder(&mut self) -> Result<(), String> {
        if std::env::var("TALUS_FN_P3")
            .map(|v| v.trim() == "0")
            .unwrap_or(false)
            || std::env::var("TALUS_FN_P3_RECORD")
                .map(|v| v.trim() == "0")
                .unwrap_or(false)
        {
            return Ok(());
        }
        let st = global_recorder()?;
        unsafe { (st.fns.set_window_recorder)(Some(recorder_cb), std::ptr::null_mut()) };
        self.recorder_installed = true;
        Ok(())
    }

    /// Executes a single-token decode step at position `pos` for input token `tok`, returning `(next_tok, step_ms, ttft_ms)`.
    ///
    /// The S-form token-graph path behind `gen_step_token` runs only for non-native packs (see the module notes);
    /// native packs decode through the verify window this piece records.
    pub unsafe fn step_token(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpGenCtx,
        pos: i64,
        tok: i64,
    ) -> Result<(i32, f64, f64), String> {
        let mut next = 0i32;
        let mut step_ms = 0.0f64;
        let mut ttft_ms = 0.0f64;
        if (kernels.gen_step_token)(ctx, pos, tok, &mut next, &mut step_ms, &mut ttft_ms) < 0 {
            return Err(format!("decode step_token failed at pos={pos}"));
        }
        self.steps_executed += 1;
        Ok((next, step_ms, ttft_ms))
    }
}

// ================================ host-side switches (verify.cpp's, read the same way) ================================

fn c_atoi(s: &str) -> i64 {
    let b = s.trim_start().as_bytes();
    let (neg, mut i) = match b.first() {
        Some(b'-') => (true, 1),
        Some(b'+') => (false, 1),
        _ => (false, 0),
    };
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v.saturating_mul(10).saturating_add((b[i] - b'0') as i64);
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

fn c_atof(s: &str) -> f64 {
    let t = s.trim_start();
    let mut end = 0;
    let mut prev = ' ';
    for (i, c) in t.char_indices() {
        let sign_ok = (c == '-' || c == '+') && (i == 0 || prev == 'e' || prev == 'E');
        if c.is_ascii_digit() || c == '.' || sign_ok || c == 'e' || c == 'E' {
            end = i + c.len_utf8();
            prev = c;
        } else {
            break;
        }
    }
    let mut cut = &t[..end];
    while !cut.is_empty() {
        if let Ok(v) = cut.parse::<f64>() {
            return v;
        }
        cut = &cut[..cut.len() - 1];
    }
    0.0
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
fn first_byte(v: &str) -> u8 {
    v.as_bytes().first().copied().unwrap_or(0)
}
/// `v != nullptr && v[0] == '1'`
fn env_eq1(name: &str) -> bool {
    env(name).map(|v| first_byte(&v) == b'1').unwrap_or(false)
}
/// `v == nullptr || v[0] != '0'`
fn env_not0_default_on(name: &str) -> bool {
    env(name).map(|v| first_byte(&v) != b'0').unwrap_or(true)
}
/// `v != nullptr && atoi(v) != 0`
fn env_atoi_on(name: &str) -> bool {
    env(name).map(|v| c_atoi(&v) != 0).unwrap_or(false)
}
/// `v == nullptr || atoi(v) != 0`
fn env_atoi_default_on(name: &str) -> bool {
    env(name).map(|v| c_atoi(&v) != 0).unwrap_or(true)
}
/// verify.cpp `env_on`: set, nonempty, and not starting with '0'
fn env_on(name: &str) -> bool {
    env(name)
        .map(|v| {
            let b = first_byte(&v);
            b != 0 && b != b'0'
        })
        .unwrap_or(false)
}

/// The kernel library's own switches (`tk_qwen4exp_p3_runtime_flags`).
#[derive(Clone, Copy, Debug, Default)]
pub struct RtFlags {
    pub native_router: bool,
    pub native_moe_combine: bool,
    pub shared_native_bf16: bool,
    pub native_qsa: bool,
    pub native_rope: bool,
    pub ple_native_bf16: bool,
    pub ple_native_postops: bool,
    pub gr_native_mmvf: bool,
    pub argmax_rows: bool,
    pub nan_guard: bool,
    pub pdl_supported: bool,
    pub gfx12: bool,
    pub hip: bool,
    pub gfx906: bool,
}

impl RtFlags {
    fn from_bits(b: u64) -> Self {
        let on = |i: u32| (b >> i) & 1 != 0;
        Self {
            native_router: on(0),
            native_moe_combine: on(1),
            shared_native_bf16: on(2),
            native_qsa: on(3),
            native_rope: on(4),
            ple_native_bf16: on(5),
            ple_native_postops: on(6),
            gr_native_mmvf: on(7),
            argmax_rows: on(8),
            nan_guard: on(9),
            pdl_supported: on(10),
            gfx12: on(11),
            hip: on(12),
            gfx906: on(14),
        }
    }
}

/// The recorder's environment switches - the same names, defaults and parsing as `verify.cpp`'s.
#[derive(Clone, Copy, Debug, Default)]
pub struct WindowEnv {
    pub dec_batch: bool,
    pub lfuse: bool,
    pub lfuse_gate: bool,
    pub lfuse_pair: bool,
    pub qdedup: bool,
    pub mmvq_il: bool,
    pub qfuse: bool,
    pub doorbell_store: bool,
    pub no_multi_gr: bool,
    pub no_batch_kv: bool,
    pub sh_stream: bool,
    pub head_mix_multi: bool,
    pub fuse_wait: bool,
    pub fuse_convab: bool,
    pub fuse_hits: bool,
    pub one_token_commit: bool,
    pub ple_batch_env: bool,
    pub ple_batch: bool,
    pub ple_post_env: bool,
    pub qb_q4: bool,
    pub sh_fork_late_env: bool,
    pub fuse_head_gr_env: bool,
    pub dummy_nodes: i64,
    pub route_prior: f64,
    pub route_resident: f64,
}

impl WindowEnv {
    pub fn from_env(rt: &RtFlags) -> Self {
        let hip = rt.hip;
        // fuse_wait_on / fuse_hits_on: HIP: the env decides when set, else on for gfx12 (R9700); CUDA: off
        let gfx12_or = |name: &str| -> bool {
            if !hip {
                return false;
            }
            match env(name) {
                Some(v) => c_atoi(&v) != 0,
                None => rt.gfx12,
            }
        };
        Self {
            dec_batch: env_atoi_default_on("STRATA_DEC_BATCH"),
            lfuse: env_eq1("STRATA_LFUSE"),
            lfuse_gate: env_not0_default_on("STRATA_LFUSE_GATE"),
            lfuse_pair: env_not0_default_on("STRATA_LFUSE_PAIR"),
            qdedup: env_atoi_on("STRATA_VERIFY_QDEDUP"),
            mmvq_il: env_not0_default_on("STRATA_MMVQ_IL"),
            qfuse: env_atoi_on("STRATA_QFUSE"),
            doorbell_store: hip && env_on("STRATA_DOORBELL_STORE"),
            no_multi_gr: env_on("STRATA_NO_MULTI_GR"),
            no_batch_kv: env_on("STRATA_NO_BATCH_KV_STEP"),
            sh_stream: match env("STRATA_SH_STREAM") {
                Some(v) => first_byte(&v) != b'0',
                None => !hip,
            },
            head_mix_multi: if hip {
                env_atoi_on("STRATA_HEAD_MIX_MULTI")
            } else {
                env_atoi_default_on("STRATA_HEAD_MIX_MULTI")
            },
            fuse_wait: gfx12_or("STRATA_FUSE_WAIT"),
            fuse_convab: hip && env_atoi_on("STRATA_FUSE_CONVAB"),
            fuse_hits: gfx12_or("STRATA_FUSE_HITS"),
            one_token_commit: if hip {
                env_atoi_on("STRATA_ONE_TOKEN_COMMIT")
            } else {
                env_atoi_default_on("STRATA_ONE_TOKEN_COMMIT")
            },
            ple_batch_env: env_not0_default_on("STRATA_PLE_BATCH"),
            ple_batch: env_eq1("STRATA_PLE_BATCH"),
            ple_post_env: env_atoi_on("STRATA_DF_PLE"),
            qb_q4: env_atoi_on("STRATA_DF_QB_Q4"),
            sh_fork_late_env: env_not0_default_on("STRATA_SH_FORK_LATE"),
            fuse_head_gr_env: env_eq1("STRATA_FUSE_HEAD_GR"),
            dummy_nodes: env("STRATA_DUMMY_NODES").map(|v| c_atoi(&v)).unwrap_or(0),
            route_prior: env("STRATA_ROUTE_PRIOR").map(|v| c_atof(&v)).unwrap_or(0.0),
            route_resident: env("STRATA_ROUTE_RESIDENT")
                .map(|v| c_atof(&v))
                .unwrap_or(0.0),
        }
    }
}

// ================================ the registered recorder ================================

struct GlobalRecorder {
    fns: Fnp3Fns,
    flags: OnceLock<(RtFlags, WindowEnv)>,
}

static RECORDER: OnceLock<Result<GlobalRecorder, String>> = OnceLock::new();

fn global_recorder() -> Result<&'static GlobalRecorder, String> {
    RECORDER
        .get_or_init(|| {
            Fnp3Fns::load().map(|fns| GlobalRecorder {
                fns,
                flags: OnceLock::new(),
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

unsafe extern "C" fn recorder_cb(
    _user: *mut c_void,
    ver: *mut c_void,
    t: c_int,
    stream: *mut c_void,
    err: *mut c_char,
    cap: usize,
) -> c_int {
    let write_err = |msg: &str| {
        if !err.is_null() && cap > 0 {
            let b = msg.as_bytes();
            let n = b.len().min(cap - 1);
            std::ptr::copy_nonoverlapping(b.as_ptr(), err.cast::<u8>(), n);
            *err.add(n) = 0;
        }
    };
    let res = std::panic::catch_unwind(|| -> Result<bool, String> {
        let st = global_recorder()?;
        let (rt, wenv) = *st.flags.get_or_init(|| {
            let rt = RtFlags::from_bits((st.fns.runtime_flags)());
            (rt, WindowEnv::from_env(&rt))
        });
        let mut rec = WindowRecorder::bind(&st.fns, ver, rt, wenv)?;
        if let Some(why) = rec.decline_reason(t) {
            if std::env::var_os("TALUS_FN_P3_TRACE").is_some() {
                eprintln!("qwen4exp-p3: the {t}-token window is recorded by C++ ({why})");
            }
            return Ok(false);
        }
        rec.record_window(t, stream)?;
        if std::env::var_os("TALUS_FN_P3_TRACE").is_some() {
            eprintln!("qwen4exp-p3: recorded the {t}-token window from Rust");
        }
        Ok(true)
    });
    match res {
        Ok(Ok(true)) => 1,
        Ok(Ok(false)) => 0,
        Ok(Err(e)) => {
            write_err(&format!("qwen4exp-p3 recorder: {e}"));
            -1
        }
        Err(_) => {
            write_err("qwen4exp-p3 recorder: panicked");
            -1
        }
    }
}

// ================================ the window recorder ================================

#[inline]
fn pf<T>(p: *mut T, n: i64) -> *mut T {
    p.wrapping_add(n as usize)
}
#[inline]
fn pc<T>(p: *const T, n: i64) -> *const T {
    p.wrapping_add(n as usize)
}

/// Records one `Verifier`'s decode window (T rows) from Rust: the per-layer API Piece 4 builds on.
pub struct WindowRecorder<'a> {
    f: &'a Fnp3Fns,
    pub v: TkP3WindowView,
    pub rt: RtFlags,
    pub env: WindowEnv,
    layers: Vec<TkP3LayerWeights>,
    qsa: Vec<TkP3QsaView>,
    gdn_idx: Vec<i64>,
    qsa_idx: Vec<i64>,
    // per window (set by `begin`)
    t: i32,
    groups: i32,
    tb: [i32; 2],
    te: [i32; 2],
    self_commit: bool,
    ple_on: bool,
    fuse_head_gr: bool,
    sg_gated: [bool; 2],
    cs: *mut c_void,
}

macro_rules! ck {
    ($self:ident, $e:expr) => {{
        let rc = unsafe { $e };
        if rc < 0 {
            return Err($self.last_error(stringify!($e)));
        }
        rc
    }};
}

impl<'a> WindowRecorder<'a> {
    /// Binds a `Verifier` (`ver` is `strata::core::Verifier*`) and reads its layer weights and QSA states.
    pub fn bind(
        f: &'a Fnp3Fns,
        ver: *mut c_void,
        rt: RtFlags,
        env: WindowEnv,
    ) -> Result<Self, String> {
        let mut v: TkP3WindowView = unsafe { std::mem::zeroed() };
        if unsafe { (f.window_view)(ver, &mut v) } < 0 {
            return Err("window view failed".into());
        }
        let nl = v.n_layers;
        let mut gdn_idx = vec![-1i64; nl as usize];
        let mut qsa_idx = vec![-1i64; nl as usize];
        let (mut qi, mut gi) = (0i64, 0i64);
        for l in 0..nl {
            if unsafe { (f.is_qsa_layer)(v.g, l) } != 0 {
                qsa_idx[l as usize] = qi;
                qi += 1;
            } else {
                gdn_idx[l as usize] = gi;
                gi += 1;
            }
        }
        let mut qsa = Vec::with_capacity(qi as usize);
        for j in 0..qi {
            let mut q: TkP3QsaView = unsafe { std::mem::zeroed() };
            if unsafe { (f.qsa_view)(v.ss, j, &mut q) } < 0 {
                return Err(format!("QSA state {j}: view failed"));
            }
            qsa.push(q);
        }
        Ok(Self {
            f,
            v,
            rt,
            env,
            layers: Vec::new(),
            qsa,
            gdn_idx,
            qsa_idx,
            t: 0,
            groups: 1,
            tb: [0, 0],
            te: [0, 0],
            self_commit: false,
            ple_on: false,
            fuse_head_gr: false,
            sg_gated: [false; 2],
            cs: std::ptr::null_mut(),
        })
    }

    fn last_error(&self, what: &str) -> String {
        let mut buf = vec![0u8; 1024];
        unsafe { (self.f.last_error)(buf.as_mut_ptr().cast(), buf.len()) };
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        let msg = String::from_utf8_lossy(&buf[..end]).into_owned();
        let call = what
            .split('(')
            .next()
            .unwrap_or(what)
            .trim_start_matches("(self.f.")
            .trim();
        if msg.is_empty() {
            format!("{call} failed")
        } else {
            msg
        }
    }

    /// Why this window has to stay on the C++ recorder, or `None` when Rust records it. Decided before anything is
    /// enqueued, so a declined window is recorded by C++ into the same (still empty) capture.
    pub fn decline_reason(&self, t: i32) -> Option<&'static str> {
        let v = &self.v;
        let e = &self.env;
        if t < 1 || (t as i64) > v.max_t || t > 8 {
            return Some("window size outside 1..8");
        }
        if v.batch_rec != 0 {
            return Some("batch window");
        }
        if v.lb != 0 || v.le != v.n_layers || v.has_hand_in != 0 {
            return Some("layer-split stage");
        }
        if v.prof_on != 0 || v.trace_on != 0 {
            return Some("stage profiler / trace stamps");
        }
        if v.remote_opt != 0 {
            return Some("remote expert helper");
        }
        if v.df_branch != 0 {
            return Some("STRATA_DF_BRANCH side streams");
        }
        if e.route_prior > 0.0 || e.route_resident > 0.0 {
            return Some("STRATA_ROUTE_PRIOR / STRATA_ROUTE_RESIDENT");
        }
        if self.rt.gfx906 {
            return Some("gfx906 router path");
        }
        None
    }

    /// Records the whole window on `stream` (inside the caller's capture): the C++ `record_window`'s sequence.
    pub fn record_window(&mut self, t: i32, stream: *mut c_void) -> Result<(), String> {
        if let Some(why) = self.decline_reason(t) {
            return Err(format!("window not supported by the Rust recorder: {why}"));
        }
        self.load_weights()?;
        self.begin(t, stream)?;
        // PdlScope: the window's launches may get programmatic edges (sm_90+ only; off on HIP)
        let pdl = self.groups == 1 && self.rt.pdl_supported;
        let prev = unsafe { (self.f.pdl_scope)(pdl as c_int) };
        let r = self.record_body();
        unsafe { (self.f.pdl_scope)(prev) };
        r
    }

    fn record_body(&mut self) -> Result<(), String> {
        self.record_inputs()?;
        let (lb, le, g) = (self.v.lb, self.v.le, self.groups);
        for grp in 0..g {
            self.record_pre(lb, grp)?;
        }
        for l in lb..le {
            for grp in 0..g {
                self.record_post(l, grp)?;
                if l + 1 < le {
                    self.record_pre(l + 1, grp)?;
                }
            }
        }
        self.record_head()
    }

    /// Reads every layer's weight pointers (once per bind).
    pub fn load_weights(&mut self) -> Result<(), String> {
        if !self.layers.is_empty() {
            return Ok(());
        }
        for l in 0..self.v.n_layers {
            let mut w: TkP3LayerWeights = unsafe { std::mem::zeroed() };
            let is_qsa = self.qsa_idx[l as usize] >= 0;
            if unsafe { (self.f.layer_weights)(self.v.wt, l as i32, is_qsa as i32, &mut w) } < 0 {
                return Err(self.last_error("layer_weights"));
            }
            self.layers.push(w);
        }
        Ok(())
    }

    /// Per-window setup: groups, row ranges, the self-commit and PLE switches (`record_window`'s preamble).
    pub fn begin(&mut self, t: i32, stream: *mut c_void) -> Result<(), String> {
        let v = &self.v;
        let e = &self.env;
        self.t = t;
        self.cs = stream;
        self.ple_on = v.ple_ready != 0 && v.ple_stage != 0;
        self.groups = if v.split != 0 && t >= 2 && v.batch_rec == 0 {
            2
        } else {
            1
        };
        self.self_commit =
            t == 1 && v.always_publish == 0 && v.batch_rec == 0 && !e.qfuse && e.one_token_commit;
        self.tb = [0, (t + 1) / 2];
        self.te = [if self.groups == 2 { (t + 1) / 2 } else { t }, t];
        self.sg_gated = [false; 2];
        self.fuse_head_gr = e.fuse_head_gr_env
            && v.le == v.n_layers
            && v.head_loaded != 0
            && unsafe { (self.f.cvec_covers)(v.n_layers - 1) } == 0;
        if v.batch_rec == 0 {
            ck!(self, (self.f.set_groups)(v.ver, t, self.groups));
        }
        Ok(())
    }

    #[inline]
    fn rt_row(&self, t: i64) -> *mut f32 {
        pf(self.v.r, t * self.v.hc * self.v.n_embd)
    }

    /// The window's inputs from mapped staging, and the embeddings broadcast to the hc streams.
    pub fn record_inputs(&mut self) -> Result<(), String> {
        let v = self.v;
        let (f, cs, t) = (self.f, self.cs, self.t as i64);
        ck!(self, (f.copy_i32_from_mapped)(v.tok, v.m_tok, t, cs));
        ck!(
            self,
            (f.copy_i32_from_mapped)(v.step, v.m_step, t * v.step_count, cs)
        );
        ck!(
            self,
            (f.copy_i32_from_mapped)(
                v.pos,
                v.m_pos,
                v.max_t * (v.n_head + v.n_head_kv + v.idx_q_heads),
                cs
            )
        );
        ck!(self, (f.embed_rows)(v.wt, v.tok, t as c_int, v.emb, cs));
        ck!(
            self,
            (f.broadcast_streams)(v.emb, v.r, v.n_embd, v.hc as c_int, t as c_int, cs)
        );
        Ok(())
    }

    fn lfuse_on(&self, n: i32) -> bool {
        let e = &self.env;
        e.lfuse
            && self.v.ar_on != 0
            && self.rt.native_moe_combine
            && e.dec_batch
            && n > 1
            && n <= 8
            && self.rt.shared_native_bf16
    }

    /// One group's hyper-connection read (`gr_read_group`): returns whether the read also wrote the consumer's q8_1.
    fn gr_read_group(
        &self,
        l: i64,
        grp: i32,
        half: usize,
        apply: bool,
        inj_prev: *mut f32,
        inj_out: *mut f32,
    ) -> Result<bool, String> {
        let v = &self.v;
        let w = &self.layers[l as usize];
        let (tb, te) = (self.tb[grp as usize] as i64, self.te[grp as usize] as i64);
        let n = te - tb;
        let n_q8 = (v.n_embd / 32) * 36;
        let mut fa = [TkP3GrArgs::default(); 8];
        for t in tb..te {
            let a = &mut fa[(t - tb) as usize];
            a.r = self.rt_row(t);
            a.r_out = self.rt_row(t);
            a.apply = apply as i32;
            a.bo_prev = pf(v.bo, t * v.n_embd);
            a.inj_prev = pf(inj_prev, t * v.hc);
            a.w_norm = w.hc_norm[half];
            a.w_down = w.hc_down[half];
            a.w_up = w.hc_up[half];
            a.w_inject = w.hc_inject[half];
            a.q8_down = w.hc_q8_down[half];
            a.q8_up = w.hc_q8_up[half];
            a.q8_inject = w.hc_q8_inject[half];
            a.eps = v.eps;
            a.lo = pf(v.lo, t * v.hc_lr);
            a.rs = pf(v.rs, t * v.hc);
            a.inject_out = pf(inj_out, t * v.hc);
            a.mixed = pf(v.mixed, t * v.n_embd);
            if !v.qcnt.is_null() {
                // S26 STRATA_QFUSE: the consumer's q8_1 image written by the read itself
                a.q8_cnt = v.qcnt;
                if half == 0 {
                    a.q8_mixed = pf(v.xq, (t - tb) * n_q8);
                } else if v.native_layout != 0 {
                    a.q8_mixed = pf(v.nat_xq, t * n_q8);
                }
            }
        }
        let rc = ck!(
            self,
            (self.f.fused_gr_read_multi)(
                fa.as_ptr(),
                n as c_int,
                pf(v.xn, tb * v.hc * v.n_embd),
                self.cs
            )
        );
        Ok(rc > 0)
    }

    /// A native projection of the group's `xq_` columns (`mm`): the interleaved form where it applies.
    fn mm(
        &self,
        il_ready: &mut bool,
        ty: i32,
        w: *const c_void,
        out: *mut f32,
        n_in: i64,
        n_out: i64,
        n: i32,
    ) -> Result<(), String> {
        let (f, v, cs) = (self.f, &self.v, self.cs);
        if self.env.mmvq_il && unsafe { (f.mmvq_il_supported)(ty, n, n_out as c_int) } != 0 {
            if !*il_ready && n >= 2 {
                ck!(
                    self,
                    (f.q8_1_interleave)(v.xq.cast(), v.xil.cast(), n_in as c_int, n, cs)
                );
                *il_ready = true;
            }
            ck!(
                self,
                (f.mmvq_il)(
                    ty,
                    w,
                    v.xq.cast(),
                    v.xil.cast(),
                    out,
                    n_in as c_int,
                    n_out as c_int,
                    n,
                    cs
                )
            );
        } else {
            ck!(
                self,
                (f.mmvq)(
                    ty,
                    w,
                    v.xq.cast(),
                    out,
                    n_in as c_int,
                    n_out as c_int,
                    n,
                    cs
                )
            );
        }
        Ok(())
    }

    /// The PLE block at layer 1 (with the previous layer's FFN write folded in), row by row or as one pass.
    fn record_ple(&mut self, grp: i32) -> Result<(), String> {
        let v = self.v;
        let (f, cs, rt, e) = (self.f, self.cs, self.rt, self.env);
        let (tb, te) = (self.tb[grp as usize] as i64, self.te[grp as usize] as i64);
        let n = (te - tb) as i32;
        let (nn, hc, hcd) = (v.n_embd, v.hc, v.ng_hc_dim);
        let hs = v.ng_hist * v.ng_hc_dim;
        if grp == 0 {
            if v.ar_on != 0 {
                ck!(self, (f.wait_flag_ge)(v.m_flag, 1, cs));
            }
            ck!(
                self,
                (f.copy_from_mapped)(v.ple, v.m_ple, self.t as i64 * nn, cs)
            );
        }
        let normalized: *mut f32 = v
            .ple_scratch
            .cast::<u8>()
            .wrapping_add(v.ple_block_scratch_bytes as usize)
            .cast();
        let ple_batch_kv = e.ple_batch_env
            && e.dec_batch
            && n > 1
            && v.ple_key_bf16 != 0
            && v.ple_value_bf16 != 0
            && rt.ple_native_bf16
            && rt.ple_native_postops;
        let nkey = v.ple_key_native != 0 && v.ple_key_bf16 == 0;
        let batched = !ple_batch_kv
            && e.ple_batch
            && n > 1
            && (v.ple_key_bf16 != 0 || nkey)
            && rt.ple_native_bf16;
        if ple_batch_kv {
            ck!(
                self,
                (f.bf16_gemv_multi)(
                    pf(v.ple, tb * nn),
                    nn,
                    v.ple_w_key_bf16,
                    pf(v.xn, tb * hc * nn),
                    hc * nn,
                    nn,
                    hc * nn,
                    n,
                    cs
                )
            );
            ck!(
                self,
                (f.bf16_gemv_multi)(
                    pf(v.ple, tb * nn),
                    nn,
                    v.ple_w_value_bf16,
                    pf(v.z, tb * nn),
                    nn,
                    nn,
                    nn,
                    n,
                    cs
                )
            );
        } else if batched {
            if !v.ple_w_key_bf16.is_null() {
                ck!(
                    self,
                    (f.bf16_gemv_multi)(
                        pf(v.ple, tb * nn),
                        nn,
                        v.ple_w_key_bf16,
                        pf(v.ple_key, tb * hcd),
                        hcd,
                        nn,
                        hcd,
                        n,
                        cs
                    )
                );
            } else {
                ck!(
                    self,
                    (f.quantize_q8_1)(pf(v.ple, tb * nn), v.xq.cast(), nn as c_int, n, cs)
                );
                ck!(
                    self,
                    (f.mmvq)(
                        v.ple_w_key_native_type,
                        v.ple_w_key_native,
                        v.xq.cast(),
                        pf(v.ple_key, tb * hcd),
                        nn as c_int,
                        hcd as c_int,
                        n,
                        cs
                    )
                );
            }
            ck!(
                self,
                (f.bf16_gemv_multi)(
                    pf(v.ple, tb * nn),
                    nn,
                    v.ple_w_value_bf16,
                    pf(v.ple_val, tb * nn),
                    nn,
                    nn,
                    nn,
                    n,
                    cs
                )
            );
        }
        let ple_post = ple_batch_kv && e.ple_post_env && v.batch_rec == 0;
        let multi_write = !ple_post && e.dec_batch && !e.no_multi_gr && n > 1;
        if multi_write {
            ck!(
                self,
                (f.gr_write)(
                    self.rt_row(tb),
                    pf(v.bo, tb * nn),
                    pf(v.inj2, tb * hc),
                    nn,
                    hc,
                    v.hc_lr,
                    self.rt_row(tb),
                    n,
                    1,
                    cs
                )
            );
        }
        if ple_post {
            let kn = v.k * nn;
            let qn = pf(v.parts, tb * kn);
            let gated = pf(v.hit_out, tb * kn);
            let pgate = pf(v.lo, tb * v.hc_lr);
            if rt.gr_native_mmvf {
                ck!(
                    self,
                    (f.gr_post_multi)(
                        self.rt_row(tb),
                        pf(v.bo, tb * nn),
                        pf(v.inj2, tb * hc),
                        self.rt_row(tb),
                        nn as c_int,
                        hc as c_int,
                        n,
                        hc * nn,
                        nn,
                        hc,
                        cs
                    )
                );
            } else {
                for t in tb..te {
                    ck!(
                        self,
                        (f.gr_write)(
                            self.rt_row(t),
                            pf(v.bo, t * nn),
                            pf(v.inj2, t * hc),
                            nn,
                            hc,
                            v.hc_lr,
                            self.rt_row(t),
                            1,
                            0,
                            cs
                        )
                    );
                }
            }
            ck!(
                self,
                (f.ple_postops_batch_snap)(
                    v.ss,
                    pf(v.xn, tb * hc * nn),
                    self.rt_row(tb),
                    pf(v.z, tb * nn),
                    qn,
                    gated,
                    pgate,
                    n,
                    pf(v.hist_snap, tb * hs),
                    cs
                )
            );
        } else {
            for t in tb..te {
                if !multi_write {
                    ck!(
                        self,
                        (f.gr_write)(
                            self.rt_row(t),
                            pf(v.bo, t * nn),
                            pf(v.inj2, t * hc),
                            nn,
                            hc,
                            v.hc_lr,
                            self.rt_row(t),
                            1,
                            0,
                            cs
                        )
                    );
                }
                let hist = v.ple_hist;
                if ple_batch_kv {
                    ck!(
                        self,
                        (f.ple_block_projected)(
                            v.ss,
                            pf(v.xn, t * hc * nn),
                            pf(v.z, t * nn),
                            self.rt_row(t),
                            hist,
                            normalized,
                            cs
                        )
                    );
                } else {
                    ck!(
                        self,
                        (f.ple_block)(
                            v.ss,
                            pf(v.ple, t * nn),
                            self.rt_row(t),
                            hist,
                            normalized,
                            pf(v.ple_key, t * hcd),
                            pf(v.ple_val, t * nn),
                            batched as c_int,
                            cs
                        )
                    );
                }
                ck!(self, (f.ple_history_advance)(hist, normalized, cs));
                ck!(
                    self,
                    (f.copy_from_mapped)(pf(v.hist_snap, t * hs), hist, hs, cs)
                );
            }
        }
        Ok(())
    }

    /// The GDN recurrent mixer for the group's rows: q/k/v + conv + l2, a/b, z, the recurrence, the out projection.
    fn record_gdn(
        &self,
        l: i64,
        grp: i32,
        q8_attn: bool,
        il_ready: &mut bool,
    ) -> Result<(), String> {
        let v = &self.v;
        let w = &self.layers[l as usize];
        let (f, cs, e) = (self.f, self.cs, &self.env);
        let (tb, te) = (self.tb[grp as usize] as i64, self.te[grp as usize] as i64);
        let n = (te - tb) as i32;
        let (nn, c, zv, hv, hk, mt) = (
            v.n_embd,
            v.conv_channels,
            v.value_dim,
            v.v_heads,
            v.k_heads,
            v.max_t,
        );
        let xm = pf(v.mixed, tb * nn);
        let gi = self.gdn_idx[l as usize];
        let state = pf(v.gdn_state, (gi - v.gdn_ord0) * v.gdn_floats);
        let conv = pf(state, v.state_size * v.v_heads * v.state_size);
        let qkv = pf(v.qkv_l, gi * mt * c);
        let hb = pf(v.h_l, gi * mt * c);
        let gate = pf(v.gate_l, gi * mt * hv);
        let beta = pf(v.beta_l, gi * mt * hv);
        if !q8_attn {
            ck!(self, (f.quantize_q8_1)(xm, v.xq.cast(), nn as c_int, n, cs));
        }
        let fuse_ab = e.fuse_convab && v.batch_rec == 0 && v.prof_on == 0;
        self.mm(il_ready, w.qkv_type, w.qkv, pf(qkv, tb * c), nn, c, n)?;
        if fuse_ab {
            ck!(
                self,
                (f.gdn_conv_ab_multi)(
                    conv,
                    qkv,
                    w.conv1d,
                    hb,
                    c as c_int,
                    (2 * hk) as c_int,
                    v.eps,
                    n,
                    tb as c_int,
                    self.self_commit as c_int,
                    xm,
                    w.alpha,
                    w.beta,
                    w.dt,
                    w.ssm_a,
                    pf(gate, tb * hv),
                    pf(beta, tb * hv),
                    nn as c_int,
                    hv as c_int,
                    cs
                )
            );
        } else {
            ck!(
                self,
                (f.gdn_conv_l2_multi)(
                    conv,
                    qkv,
                    w.conv1d,
                    hb,
                    c as c_int,
                    (2 * hk) as c_int,
                    v.eps,
                    n,
                    tb as c_int,
                    self.self_commit as c_int,
                    cs
                )
            );
            ck!(
                self,
                (f.gdn_ab_multi)(
                    xm,
                    w.alpha,
                    w.beta,
                    w.dt,
                    w.ssm_a,
                    pf(gate, tb * hv),
                    pf(beta, tb * hv),
                    nn as c_int,
                    hv as c_int,
                    n,
                    cs
                )
            );
        }
        self.mm(il_ready, w.gate_type, w.gate, pf(v.z, tb * zv), nn, zv, n)?;
        // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
        let n_keep = if self.self_commit {
            v.one as *const i32
        } else {
            std::ptr::null()
        };
        let xq_out = if e.qfuse {
            v.xq.cast::<c_void>()
        } else {
            std::ptr::null_mut()
        };
        ck!(
            self,
            (f.gdn_step_norm_multi)(
                state,
                hb,
                c as c_int,
                gate,
                beta,
                v.z,
                w.ssm_norm,
                v.eps,
                v.y,
                hk as c_int,
                hv as c_int,
                te as c_int,
                n_keep,
                tb as c_int,
                xq_out,
                cs
            )
        );
        if !e.qfuse || v.batch_rec != 0 {
            ck!(
                self,
                (f.quantize_q8_1)(pf(v.y, tb * zv), v.xq.cast(), zv as c_int, n, cs)
            );
        }
        *il_ready = false;
        self.mm(il_ready, w.out_type, w.out, pf(v.bo, tb * nn), zv, nn, n)
    }

    /// The QSA sparse-attention mixer for the group's rows: K/V + indexer key appends, queries, block selection,
    /// the sparse attention, the output gate and projection.
    fn record_qsa(
        &self,
        l: i64,
        grp: i32,
        q8_attn: bool,
        il_ready: &mut bool,
    ) -> Result<(), String> {
        let v = &self.v;
        let w = &self.layers[l as usize];
        let (f, cs, e, rt) = (self.f, self.cs, &self.env, &self.rt);
        let (tb, te) = (self.tb[grp as usize] as i64, self.te[grp as usize] as i64);
        let n = (te - tb) as i32;
        let (nn, nh, hd, nkv, iq, id, mt) = (
            v.n_embd,
            v.n_head,
            v.head_dim,
            v.n_head_kv,
            v.idx_q_heads,
            v.idx_key_dim,
            v.max_t,
        );
        let (sc, sp, n_rot) = (v.step_count, v.step_pos, v.n_rot as c_int);
        let qi = self.qsa_idx[l as usize];
        let st = self.qsa[qi as usize];
        let pos = v.pos as *const i32;
        let pos_k = pc(pos, mt * nh);
        let pos_i = pc(pos, mt * (nh + nkv));
        let fuse_nr = rt.native_qsa
            && rt.native_rope
            && unsafe { (f.norm_rope_usable)(hd as c_int, n_rot) } != 0;
        let norm_rope_on = |data: *mut f32,
                            norm: *const f32,
                            rows: i64,
                            cols: i64,
                            p: *const i32|
         -> Result<(), String> {
            ck!(
                self,
                (f.rms_norm)(
                    data,
                    norm,
                    rows as c_int,
                    cols as c_int,
                    v.eps,
                    rt.native_qsa as c_int,
                    cs
                )
            );
            ck!(
                self,
                (f.rope)(
                    data,
                    rows as c_int,
                    cols as c_int,
                    n_rot,
                    p,
                    st.cos_tab,
                    st.sin_tab,
                    rt.native_rope as c_int,
                    cs
                )
            );
            Ok(())
        };
        let norm_rope = |data: *mut f32,
                         norm: *const f32,
                         rows: i64,
                         cols: i64,
                         p: *const i32|
         -> Result<(), String> {
            if fuse_nr && unsafe { (f.norm_rope_usable)(cols as c_int, n_rot) } != 0 {
                ck!(
                    self,
                    (f.rms_norm_rope)(
                        data,
                        cols as c_int,
                        norm,
                        data,
                        rows as c_int,
                        cols as c_int,
                        n_rot,
                        v.eps,
                        p,
                        cs
                    )
                );
                return Ok(());
            }
            norm_rope_on(data, norm, rows, cols, p)
        };
        let idx_raw = pf(v.idx_raw_l, qi * mt * id);
        let qb =
            e.dec_batch && n > 1 && rt.native_qsa && rt.native_rope && (st.kv_q4 == 0 || e.qb_q4);
        let xm = pf(v.mixed, tb * nn);
        if !q8_attn {
            ck!(self, (f.quantize_q8_1)(xm, v.xq.cast(), nn as c_int, n, cs));
        }
        if qb {
            ck!(
                self,
                (f.bf16_gemv_multi)(
                    pf(v.mixed, tb * nn),
                    nn,
                    w.idx_k,
                    pf(idx_raw, tb * id),
                    id,
                    nn,
                    id,
                    n,
                    cs
                )
            );
        } else {
            for t in tb..te {
                ck!(
                    self,
                    (f.bf16_gemv)(
                        pf(v.mixed, t * nn),
                        w.idx_k,
                        pf(idx_raw, t * id),
                        nn,
                        id,
                        cs
                    )
                );
            }
        }
        self.mm(
            il_ready,
            w.k_type,
            w.k,
            pf(v.kcur, tb * nkv * hd),
            nn,
            nkv * hd,
            n,
        )?;
        self.mm(
            il_ready,
            w.v_type,
            w.v,
            pf(v.vcur, tb * nkv * hd),
            nn,
            nkv * hd,
            n,
        )?;
        if qb {
            norm_rope(
                pf(v.kcur, tb * nkv * hd),
                w.k_norm,
                n as i64 * nkv,
                hd,
                pc(pos_k, tb * nkv),
            )?;
        } else {
            for t in tb..te {
                norm_rope(pf(v.kcur, t * nkv * hd), w.k_norm, nkv, hd, pc(pos, t * nh))?;
            }
        }
        if st.kv_rot != 0 {
            ck!(
                self,
                (f.fwht256)(pf(v.kcur, tb * nkv * hd), n as i64 * nkv, cs)
            );
            ck!(
                self,
                (f.fwht256)(pf(v.vcur, tb * nkv * hd), n as i64 * nkv, cs)
            );
        } else if st.kv_hybrid != 0 {
            ck!(
                self,
                (f.fwht256)(pf(v.vcur, tb * nkv * hd), n as i64 * nkv, cs)
            );
        }
        let ts = (v.idx_block - 1) * id;
        if grp == 0 {
            ck!(
                self,
                (f.copy_from_mapped)(pf(v.tail_snap, qi * ts), st.idx_tail, ts, cs)
            );
        }
        let batch_steps = e.dec_batch && !e.no_batch_kv && v.batch_rec == 0;
        ck!(
            self,
            (f.kv_append)(
                v.g,
                v.ss,
                qi,
                pf(v.step, tb * sc),
                sc as c_int,
                pf(v.kcur, tb * nkv * hd),
                pf(v.vcur, tb * nkv * hd),
                (nkv * hd) as c_int,
                n,
                batch_steps as c_int,
                cs
            )
        );
        ck!(
            self,
            (f.indexer_append)(
                v.g,
                v.ss,
                qi,
                pf(idx_raw, tb * id),
                id,
                pf(v.step, tb * sc + sp),
                sc as c_int,
                n,
                batch_steps as c_int,
                w.ik_norm,
                v.eps,
                cs
            )
        );
        self.mm(
            il_ready,
            w.q_type,
            w.q,
            pf(v.qfull, tb * nh * 2 * hd),
            nn,
            nh * 2 * hd,
            n,
        )?;
        if qb {
            if fuse_nr {
                ck!(
                    self,
                    (f.rms_norm_rope)(
                        pf(v.qfull, tb * nh * 2 * hd),
                        (2 * hd) as c_int,
                        w.q_norm,
                        pf(v.qcur, tb * nh * hd),
                        (n as i64 * nh) as c_int,
                        hd as c_int,
                        n_rot,
                        v.eps,
                        pc(pos, tb * nh),
                        cs
                    )
                );
            } else {
                ck!(
                    self,
                    (f.copy_rows_strided)(
                        pf(v.qcur, tb * nh * hd),
                        pf(v.qfull, tb * nh * 2 * hd),
                        n as i64 * nh,
                        hd,
                        2 * hd,
                        cs
                    )
                );
                norm_rope(
                    pf(v.qcur, tb * nh * hd),
                    w.q_norm,
                    n as i64 * nh,
                    hd,
                    pc(pos, tb * nh),
                )?;
            }
            if st.kv_rot != 0 {
                ck!(
                    self,
                    (f.fwht256)(pf(v.qcur, tb * nh * hd), n as i64 * nh, cs)
                );
            }
            ck!(
                self,
                (f.bf16_gemv_multi)(
                    pf(v.mixed, tb * nn),
                    nn,
                    w.idx_q,
                    pf(v.qidx, tb * iq * id),
                    iq * id,
                    nn,
                    iq * id,
                    n,
                    cs
                )
            );
            norm_rope(
                pf(v.qidx, tb * iq * id),
                w.iq_norm,
                n as i64 * iq,
                id,
                pc(pos_i, tb * iq),
            )?;
        } else {
            for t in tb..te {
                let qc = pf(v.qcur, t * nh * hd);
                if fuse_nr {
                    ck!(
                        self,
                        (f.rms_norm_rope)(
                            pf(v.qfull, t * nh * 2 * hd),
                            (2 * hd) as c_int,
                            w.q_norm,
                            qc,
                            nh as c_int,
                            hd as c_int,
                            n_rot,
                            v.eps,
                            pc(pos, t * nh),
                            cs
                        )
                    );
                } else {
                    ck!(
                        self,
                        (f.copy_2d)(qc, pf(v.qfull, t * nh * 2 * hd), hd, nh, cs)
                    );
                    norm_rope(qc, w.q_norm, nh, hd, pc(pos, t * nh))?;
                }
                if st.kv_rot != 0 {
                    ck!(self, (f.fwht256)(qc, nh, cs));
                }
            }
            for t in tb..te {
                let qx = pf(v.qidx, t * iq * id);
                ck!(
                    self,
                    (f.bf16_gemv)(pf(v.mixed, t * nn), w.idx_q, qx, nn, iq * id, cs)
                );
                norm_rope(qx, w.iq_norm, iq, id, pc(pos, t * nh))?;
            }
        }
        let (mb, cap) = (v.max_blocks, v.cap);
        let steps = pf(v.step, tb * sc);
        ck!(
            self,
            (f.qsa_block_scores)(
                v.g,
                v.ss,
                qi,
                pf(v.qidx, tb * iq * id),
                steps,
                n as i64,
                mb,
                pf(v.scores, tb * mb),
                cs
            )
        );
        ck!(
            self,
            (f.qsa_block_topk)(
                v.g,
                pf(v.scores, tb * mb),
                steps,
                n as i64,
                mb,
                cap,
                pf(v.sel, tb * cap),
                cs
            )
        );
        // KV streaming: the n selections' blocks resident (device-side, inside the graph)
        ck!(
            self,
            (f.qsa_kv_resolve)(v.g, v.ss, qi, pf(v.sel, tb * cap), steps, n as i64, cap, cs)
        );
        ck!(
            self,
            (f.qsa_attn)(
                v.g,
                v.ss,
                qi,
                pf(v.qcur, tb * nh * hd),
                pf(v.sel, tb * cap),
                steps,
                cap,
                pf(v.attn_scratch, tb * v.attn_scratch_floats),
                pf(v.attn, tb * nh * hd),
                n as i64,
                cs
            )
        );
        if st.kv_rot != 0 || st.kv_hybrid != 0 {
            ck!(
                self,
                (f.fwht256)(pf(v.attn, tb * nh * hd), n as i64 * nh, cs)
            );
        }
        if qb {
            ck!(
                self,
                (f.qsa_gate)(
                    pf(v.attn, tb * nh * hd),
                    pf(v.qfull, tb * nh * 2 * hd),
                    pf(v.attn32, tb * nh * hd),
                    (n as i64 * nh) as c_int,
                    hd as c_int,
                    1,
                    v.g,
                    cs
                )
            );
        } else {
            for t in tb..te {
                ck!(
                    self,
                    (f.qsa_gate)(
                        pf(v.attn, t * nh * hd),
                        pf(v.qfull, t * nh * 2 * hd),
                        pf(v.attn32, t * nh * hd),
                        nh as c_int,
                        hd as c_int,
                        rt.native_qsa as c_int,
                        v.g,
                        cs
                    )
                );
            }
        }
        ck!(
            self,
            (f.quantize_q8_1)(
                pf(v.attn32, tb * nh * hd),
                v.xq.cast(),
                (nh * hd) as c_int,
                n,
                cs
            )
        );
        *il_ready = false;
        self.mm(il_ready, w.o_type, w.o, pf(v.bo, tb * nn), nh * hd, nn, n)
    }

    /// Layer `l`, group `grp`, up to the ring: PLE (layer 1), the mixer's read, GDN or QSA, the FFN-side read, the
    /// router, the doorbell / device plan, the shared expert and the routed experts' activation image.
    pub fn record_pre(&mut self, l: i64, grp: i32) -> Result<(), String> {
        let mut pending = l > 0 && unsafe { (self.f.cvec_covers)(l - 1) } == 0;
        if l == 1 && self.ple_on {
            self.record_ple(grp)?;
            pending = false;
        }
        let v = self.v;
        let (f, cs, e, rt) = (self.f, self.cs, self.env, self.rt);
        let (tb, te) = (self.tb[grp as usize] as i64, self.te[grp as usize] as i64);
        let n = (te - tb) as i32;
        let (nn, k, ne, gsz) = (v.n_embd, v.k, v.n_expert, self.groups as i64);
        let q8_attn = self.gr_read_group(l, grp, 0, pending, v.inj2, v.inj)?;
        for _ in 0..e.dummy_nodes.max(0) {
            ck!(self, (f.gpu_noop)(cs));
        }
        let mut il_ready = false;
        if self.qsa_idx[l as usize] < 0 {
            self.record_gdn(l, grp, q8_attn, &mut il_ready)?;
        } else {
            self.record_qsa(l, grp, q8_attn, &mut il_ready)?;
        }
        let q8_ffn = self.gr_read_group(l, grp, 1, true, v.inj, v.inj2)?;
        let w = self.layers[l as usize];
        let xm = pf(v.mixed, tb * nn);
        let sh_fork = e.sh_stream && v.prof_on == 0 && v.sh_fork_ready != 0;
        let sh_fork_late = sh_fork && e.sh_fork_late_env && v.ar_on == 0;
        let sh_stream = if sh_fork { v.sh_cs } else { cs };
        if sh_fork && !sh_fork_late {
            ck!(self, (f.event_fork)(v.ev_fork, cs, v.sh_cs));
        }
        let w_sgi = if self.lfuse_on(n) && e.lfuse_gate {
            w.shexp_gate_inp
        } else {
            std::ptr::null()
        };
        let mut sg_ready = false;
        if e.dec_batch && n > 1 && !w.router.is_null() && rt.native_router && ne == 512 && k == 10 {
            if !w_sgi.is_null() {
                sg_ready = ck!(
                    self,
                    (f.bf16_gemv_multi_aux)(
                        xm,
                        nn,
                        w.router,
                        pf(v.logits, tb * ne),
                        ne,
                        nn,
                        ne,
                        n,
                        w_sgi,
                        pf(v.sh_g, tb),
                        1,
                        cs
                    )
                ) > 0;
            }
            if !sg_ready {
                ck!(
                    self,
                    (f.bf16_gemv_multi)(xm, nn, w.router, pf(v.logits, tb * ne), ne, nn, ne, n, cs)
                );
            }
            ck!(
                self,
                (f.router_top10_multi)(
                    pf(v.logits, tb * ne),
                    pf(v.ids, tb * k),
                    pf(v.w, tb * k),
                    n,
                    cs
                )
            );
        } else {
            for t in tb..te {
                ck!(
                    self,
                    (f.moe_route_row)(
                        v.wt,
                        v.g,
                        v.ss,
                        l,
                        k,
                        pf(v.mixed, t * nn),
                        pf(v.logits, t * ne),
                        pf(v.ids, t * k),
                        pf(v.w, t * k),
                        cs
                    )
                );
            }
        }
        let plan_g = pf(v.plan, grp as i64 * (v.plan_i32 + 16));
        let res_l = if v.d_res.is_null() {
            std::ptr::null()
        } else {
            pc(v.d_res, l * ne)
        };
        let ring = ((l - v.lb) * gsz + grp as i64 + 1) as u32;
        if v.ar_on != 0 {
            ck!(
                self,
                (f.resident_plan)(
                    pf(v.ids, tb * k),
                    (n as i64 * k) as c_int,
                    k as c_int,
                    res_l,
                    ne as c_int,
                    v.cache_base,
                    v.slot_off_d,
                    v.blob,
                    plan_g,
                    v.max_t * k,
                    std::ptr::null_mut(),
                    0,
                    v.m_plan_err,
                    cs
                )
            );
        } else {
            if v.device_plan != 0 {
                // E-6: every routed expert resident: this group's plan without the host
                ck!(
                    self,
                    (f.resident_plan)(
                        pf(v.ids, tb * k),
                        (n as i64 * k) as c_int,
                        k as c_int,
                        res_l,
                        ne as c_int,
                        v.cache_base,
                        v.slot_off_d,
                        v.blob,
                        plan_g,
                        v.max_t * k,
                        pf(v.skip, grp as i64),
                        ring,
                        std::ptr::null_mut(),
                        cs
                    )
                );
            }
            let (x_out, ids_out, w_out) =
                (pf(v.m_x, tb * nn), pf(v.m_ids, tb * k), pf(v.m_w, tb * k));
            let (nx, nk) = (n as i64 * nn, n as i64 * k);
            if e.doorbell_store {
                ck!(
                    self,
                    (f.doorbell_publish)(
                        2,
                        xm,
                        pf(v.ids, tb * k),
                        pf(v.w, tb * k),
                        std::ptr::null(),
                        ne as c_int,
                        nx,
                        nk,
                        x_out,
                        ids_out,
                        w_out,
                        v.m_seq,
                        ring,
                        cs
                    )
                );
            } else if v.remote_opt != 0 || v.always_publish != 0 {
                ck!(
                    self,
                    (f.doorbell_publish)(
                        0,
                        xm,
                        pf(v.ids, tb * k),
                        pf(v.w, tb * k),
                        std::ptr::null(),
                        ne as c_int,
                        nx,
                        nk,
                        x_out,
                        ids_out,
                        w_out,
                        v.m_seq,
                        0,
                        cs
                    )
                );
            } else {
                ck!(
                    self,
                    (f.doorbell_publish)(
                        1,
                        xm,
                        pf(v.ids, tb * k),
                        std::ptr::null(),
                        res_l,
                        ne as c_int,
                        nx,
                        nk,
                        x_out,
                        ids_out,
                        std::ptr::null_mut(),
                        v.m_seq,
                        0,
                        cs
                    )
                );
            }
            if sh_fork_late {
                ck!(self, (f.event_fork)(v.ev_fork, cs, v.sh_cs));
            }
        }
        // the shared expert
        let n_q8 = (nn / 32) * 36;
        let q8_1 = if sh_fork { v.sh_xq } else { v.xq };
        let qdedup = e.qdedup && !sh_fork && v.native_layout != 0;
        if qdedup && !q8_ffn {
            ck!(
                self,
                (f.quantize_q8_1_rows)(xm, n as i64, nn, pf(v.nat_xq, tb * n_q8).cast(), cs)
            );
        }
        if !rt.shared_native_bf16 {
            if e.dec_batch {
                ck!(
                    self,
                    (f.f32_to_bf16)(
                        pf(v.mixed, tb * nn),
                        pf(v.sh_bf16, tb * nn),
                        n as i64 * nn,
                        sh_stream
                    )
                );
            } else {
                for t in tb..te {
                    ck!(
                        self,
                        (f.f32_to_bf16)(pf(v.mixed, t * nn), pf(v.sh_bf16, t * nn), nn, sh_stream)
                    );
                }
            }
        }
        let xq_ready: *const c_void = if qdedup {
            pf(v.nat_xq, tb * n_q8).cast()
        } else {
            std::ptr::null()
        };
        let lfuse_bits = (sg_ready as c_int)
            | if self.lfuse_on(n) && e.lfuse_pair {
                2
            } else {
                0
            };
        ck!(
            self,
            (f.shared_expert_multi)(
                n,
                xm,
                pf(v.sh_bf16, tb * nn),
                w.sg_type,
                w.sg,
                w.su_type,
                w.su,
                w.sd_type,
                w.sd,
                q8_1.cast(),
                w.shexp_gate_inp,
                pf(v.sh_gate, tb * v.n_ff),
                pf(v.sh_up, tb * v.n_ff),
                pf(v.sh_g, tb),
                pf(v.shared, tb * nn),
                nn,
                v.n_ff,
                xq_ready,
                lfuse_bits,
                sh_stream
            )
        );
        self.sg_gated[grp as usize] = sg_ready;
        if sh_fork {
            ck!(
                self,
                (f.event_fork)(v.ev_join, v.sh_cs, std::ptr::null_mut())
            );
        }
        if v.native_layout != 0 {
            if !qdedup && !q8_ffn {
                ck!(
                    self,
                    (f.quantize_q8_1_rows)(xm, n as i64, nn, pf(v.nat_xq, tb * n_q8).cast(), cs)
                );
            }
        } else {
            let nb = nn / 32;
            ck!(
                self,
                (f.quantize_q8_0_scaled)(
                    xm,
                    pf(v.hit_xq, tb * nb * 34),
                    pf(v.hit_xs, tb * nb),
                    n as i64 * nn,
                    cs
                )
            );
        }
        Ok(())
    }

    /// Layer `l`, group `grp`, after the ring: the VRAM-resident experts, the PCIe share, the CPU pool's rows, the
    /// combine with the shared expert, and the last layer's residual write.
    pub fn record_post(&mut self, l: i64, grp: i32) -> Result<(), String> {
        let v = self.v;
        let (f, cs, e, rt) = (self.f, self.cs, self.env, self.rt);
        let (tb, te) = (self.tb[grp as usize] as i64, self.te[grp as usize] as i64);
        let n = (te - tb) as i32;
        let (nn, k, hc, gsz) = (v.n_embd, v.k, v.hc, self.groups as i64);
        let ring = ((l - v.lb) * gsz + grp as i64 + 1) as u32;
        let cap = n as i64 * k;
        let capx = v.max_t * k;
        let pl = pf(v.plan, grp as i64 * (v.plan_i32 + 16));
        let p_counts = pl as *const i32;
        let p_start = pc(p_counts, 4);
        let p_dst = pc(p_start, capx + 1);
        let p_tok = pc(p_dst, capx);
        let ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & !1i64;
        let p_ptr = pf(pl, ptr_off) as *mut u64;
        let p_ptr2 = pf(p_ptr, capx);
        let p_start2 = pc(p_counts, ptr_off + 4 * capx);
        let hit_out = pf(v.hit_out, tb * k * nn);
        let parts_out = pf(v.parts, tb * k * nn);
        let nb = nn / 32;
        let grouped = |gp: *const u64,
                       gs: *const i32,
                       gn: *const i32,
                       gy: i64,
                       dst_buf: *mut f32|
         -> Result<(), String> {
            ck!(
                self,
                (f.expert_grouped)(
                    l,
                    v.native_layout,
                    gp,
                    gs,
                    gn,
                    p_dst,
                    p_tok,
                    cap,
                    pf(v.nat_xq, tb * nb * 36).cast(),
                    pf(v.hit_xq, tb * nb * 34),
                    pf(v.hit_xs, tb * nb),
                    v.hit_scratch,
                    dst_buf,
                    gy,
                    cs
                )
            );
            Ok(())
        };
        let mut hits_in_combine = false;
        if v.ar_on != 0 {
            grouped(p_ptr, p_start, p_counts, 0, parts_out)?;
        } else {
            let skip_g = pf(v.skip, grp as i64) as *const u32;
            let m_plan_g = pf(v.m_plan, grp as i64 * v.plan_i32) as *const i32;
            if v.device_plan != 0 {
                ck!(self, (f.wait_flag_ge_or)(v.m_flag_a, ring, skip_g, cs));
                ck!(
                    self,
                    (f.copy_i32_from_mapped_unless)(pl, m_plan_g, v.plan_i32, skip_g, ring, cs)
                );
            } else if e.fuse_wait && v.prof_on == 0 {
                ck!(
                    self,
                    (f.wait_copy_i32_from_mapped)(v.m_flag_a, ring, pl, m_plan_g, v.plan_i32, cs)
                );
            } else {
                ck!(self, (f.wait_flag_ge)(v.m_flag_a, ring, cs));
                ck!(self, (f.copy_i32_from_mapped)(pl, m_plan_g, v.plan_i32, cs));
            }
            grouped(p_ptr, p_start, p_counts, 0, hit_out)?;
            if v.device_plan != 0 {
                ck!(self, (f.wait_flag_ge_or)(v.m_flag_b, ring, skip_g, cs));
            } else {
                ck!(self, (f.wait_flag_ge)(v.m_flag_b, ring, cs));
            }
            if v.pcie_mode == 2 {
                // stage the PCIe share with a copy kernel, then point at staging
                let per = if self.groups == 2 {
                    v.staging_blobs / 2
                } else {
                    v.staging_blobs
                };
                let stage = pf(v.staging, grp as i64 * per * v.max_blob);
                let mut bb = 0i64;
                ck!(self, (f.expert_blob_bytes)(l, &mut bb));
                ck!(
                    self,
                    (f.fetch_blobs)(p_ptr2, pc(p_counts, 2), stage, bb, per as c_int, cs)
                );
                ck!(
                    self,
                    (f.rebase_ptrs)(p_ptr2, pc(p_counts, 2), stage, bb, cs)
                );
            }
            grouped(
                p_ptr2,
                p_start2,
                pc(p_counts, 2),
                v.pcie_group_rows,
                hit_out,
            )?;
            let ymiss = pf(v.m_ymiss, tb * k * nn) as *const f32;
            if v.device_plan != 0 {
                ck!(self, (f.wait_flag_ge_or)(v.m_flag, ring, skip_g, cs));
                ck!(
                    self,
                    (f.copy_or_zero_from_mapped)(
                        parts_out,
                        ymiss,
                        n as i64 * k * nn,
                        skip_g,
                        ring,
                        cs
                    )
                );
            } else {
                let fuse_rows = e.fuse_wait && v.prof_on == 0 && v.remote_opt == 0 && e.dec_batch;
                if !fuse_rows {
                    ck!(self, (f.wait_flag_ge)(v.m_flag, ring, cs));
                }
                if fuse_rows {
                    ck!(
                        self,
                        (f.copy_rows_from_mapped)(
                            v.m_flag,
                            ring,
                            parts_out,
                            ymiss,
                            n as i64 * k,
                            nn,
                            p_dst,
                            pc(p_counts, 1),
                            cs
                        )
                    );
                } else if e.dec_batch {
                    ck!(
                        self,
                        (f.copy_rows_from_mapped)(
                            std::ptr::null(),
                            0,
                            parts_out,
                            ymiss,
                            n as i64 * k,
                            nn,
                            p_dst,
                            pc(p_counts, 1),
                            cs
                        )
                    );
                } else {
                    ck!(
                        self,
                        (f.copy_from_mapped)(parts_out, ymiss, n as i64 * k * nn, cs)
                    );
                }
            }
            hits_in_combine = e.fuse_hits
                && v.prof_on == 0
                && v.remote_opt == 0
                && !self.sg_gated[grp as usize]
                && e.dec_batch
                && n > 1
                && rt.native_moe_combine
                && n <= 8
                && unsafe {
                    (f.combine_hits_supported)(
                        pf(v.parts, tb * k * nn),
                        pf(v.shared, tb * nn),
                        pf(v.bo, tb * nn),
                        nn,
                        k,
                        n,
                    )
                } != 0;
            if !hits_in_combine {
                ck!(
                    self,
                    (f.moe_hit_add)(parts_out, hit_out, p_dst, pc(p_counts, 1), cap, nn, cs)
                );
            }
        }
        if e.sh_stream && v.prof_on == 0 && v.sh_fork_ready != 0 {
            ck!(self, (f.event_fork)(v.ev_join, std::ptr::null_mut(), cs));
        }
        let (wg, sh, bo) = (pf(v.w, tb * k), pf(v.shared, tb * nn), pf(v.bo, tb * nn));
        if self.sg_gated[grp as usize] {
            ck!(
                self,
                (f.moe_combine)(
                    2,
                    parts_out,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    wg,
                    sh,
                    pf(v.sh_g, tb),
                    bo,
                    nn,
                    k,
                    n,
                    cs
                )
            );
        } else if hits_in_combine {
            ck!(
                self,
                (f.moe_combine)(
                    1,
                    pf(v.parts, tb * k * nn),
                    hit_out,
                    p_dst,
                    pc(p_counts, 1),
                    wg,
                    sh,
                    std::ptr::null(),
                    bo,
                    nn,
                    k,
                    n,
                    cs
                )
            );
        } else if e.dec_batch && n > 1 && rt.native_moe_combine {
            ck!(
                self,
                (f.moe_combine)(
                    0,
                    pf(v.parts, tb * k * nn),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    wg,
                    sh,
                    std::ptr::null(),
                    bo,
                    nn,
                    k,
                    n,
                    cs
                )
            );
        } else {
            for t in tb..te {
                ck!(
                    self,
                    (f.moe_combine_row)(
                        v.g,
                        v.ss,
                        l,
                        k,
                        pf(v.w, t * k),
                        pf(v.shared, t * nn),
                        pf(v.parts, t * k * nn),
                        pf(v.bo, t * nn),
                        cs
                    )
                );
            }
        }
        if l == v.n_layers - 1 {
            if !self.fuse_head_gr {
                if e.dec_batch && !e.no_multi_gr {
                    ck!(
                        self,
                        (f.gr_write)(
                            self.rt_row(tb),
                            bo,
                            pf(v.inj2, tb * hc),
                            nn,
                            hc,
                            v.hc_lr,
                            self.rt_row(tb),
                            n,
                            1,
                            cs
                        )
                    );
                } else {
                    for t in tb..te {
                        ck!(
                            self,
                            (f.gr_write)(
                                self.rt_row(t),
                                pf(v.bo, t * nn),
                                pf(v.inj2, t * hc),
                                nn,
                                hc,
                                v.hc_lr,
                                self.rt_row(t),
                                1,
                                0,
                                cs
                            )
                        );
                    }
                }
                if unsafe { (f.cvec_covers)(l) } != 0 {
                    ck!(
                        self,
                        (f.cvec_apply)(
                            self.rt_row(tb),
                            l,
                            n as i64,
                            hc * nn,
                            std::ptr::null(),
                            0,
                            std::ptr::null(),
                            0,
                            0,
                            cs
                        )
                    );
                }
            }
        } else if unsafe { (f.cvec_covers)(l) } != 0 {
            ck!(
                self,
                (f.cvec_apply)(
                    self.rt_row(tb),
                    l,
                    n as i64,
                    hc * nn,
                    bo,
                    nn,
                    pf(v.inj2, tb * hc),
                    hc,
                    1,
                    cs
                )
            );
        }
        Ok(())
    }

    /// The head over the window's T rows: the final hyper-connection mixer, the LM head GEMV, the greedy pick and
    /// the non-finite guard.
    pub fn record_head(&mut self) -> Result<(), String> {
        let v = self.v;
        let (f, cs, e, rt) = (self.f, self.cs, self.env, self.rt);
        let t_rows = self.t as i64;
        let nn = v.n_embd;
        let mut hw: TkP3HeadWeights = unsafe { std::mem::zeroed() };
        ck!(self, (f.head_weights)(v.wt, &mut hw));
        let head = v.head_loaded != 0;
        let mix_q8 = !hw.q8_down.is_null() && !hw.q8_up.is_null() && head;
        let mut fa = [TkP3GrArgs::default(); 8];
        let mut fill = |apply: bool, q8: bool, inject_out: *mut f32| {
            for t in 0..t_rows {
                let a = &mut fa[t as usize];
                a.r = self.rt_row(t);
                a.r_out = self.rt_row(t);
                a.apply = apply as i32;
                if apply {
                    a.bo_prev = pf(v.bo, t * nn);
                    a.inj_prev = pf(v.inj2, t * v.hc);
                    a.inject_out = inject_out;
                }
                a.w_norm = hw.norm;
                a.w_down = hw.down;
                a.w_up = hw.up;
                if q8 {
                    a.q8_down = hw.q8_down;
                    a.q8_up = hw.q8_up;
                }
                a.eps = v.eps;
                a.lo = pf(v.lo, t * v.hc_lr);
                a.rs = pf(v.rs, t * v.hc);
                a.mixed = pf(v.head_mixed, t * nn);
            }
        };
        if mix_q8 {
            fill(false, true, std::ptr::null_mut());
            ck!(
                self,
                (f.fused_gr_read_multi)(fa.as_ptr(), t_rows as c_int, v.xn, cs)
            );
        } else if self.fuse_head_gr {
            fill(true, false, v.head_inj);
            ck!(
                self,
                (f.fused_gr_read_multi)(fa.as_ptr(), t_rows as c_int, v.xn, cs)
            );
        } else if e.head_mix_multi && head {
            if hw.forms_ok == 0 {
                return Err("verify: the output_hc_* weights have the wrong engine forms".into());
            }
            fill(false, false, std::ptr::null_mut());
            ck!(
                self,
                (f.fused_gr_read_multi)(fa.as_ptr(), t_rows as c_int, v.xn, cs)
            );
        } else {
            for t in 0..t_rows {
                ck!(
                    self,
                    (f.head_mix_row)(
                        v.wt,
                        v.g,
                        v.ss,
                        self.rt_row(t),
                        pf(v.head_mixed, t * nn),
                        pf(v.head_logits, t * v.n_vocab),
                        (!head) as c_int,
                        cs
                    )
                );
            }
        }
        let tr = t_rows as c_int;
        if head {
            ck!(
                self,
                (f.quantize_q8_1)(v.head_mixed, v.xq.cast(), nn as c_int, tr, cs)
            );
            if e.mmvq_il
                && unsafe { (f.mmvq_il_supported)(v.head_type, tr, v.n_vocab as c_int) } != 0
            {
                if tr >= 2 {
                    ck!(
                        self,
                        (f.q8_1_interleave)(v.xq.cast(), v.xil.cast(), nn as c_int, tr, cs)
                    );
                }
                ck!(
                    self,
                    (f.mmvq_il)(
                        v.head_type,
                        v.head_weights,
                        v.xq.cast(),
                        v.xil.cast(),
                        v.head_logits,
                        nn as c_int,
                        v.n_vocab as c_int,
                        tr,
                        cs
                    )
                );
            } else {
                ck!(
                    self,
                    (f.mmvq)(
                        v.head_type,
                        v.head_weights,
                        v.xq.cast(),
                        v.head_logits,
                        nn as c_int,
                        v.n_vocab as c_int,
                        tr,
                        cs
                    )
                );
            }
        }
        // greedy, recorded (a sampling request re-samples host-side after the replay, see Verifier::run)
        if rt.argmax_rows {
            ck!(
                self,
                (f.argmax_rows)(
                    v.head_logits,
                    tr,
                    v.n_vocab as c_int,
                    v.arg_scratch.cast(),
                    v.m_out,
                    cs
                )
            );
        } else {
            ck!(
                self,
                (f.sample_greedy)(v.head_logits, tr, v.n_vocab as c_int, v.m_out, cs)
            );
        }
        if rt.nan_guard {
            ck!(
                self,
                (f.logits_nonfinite_rows)(v.head_logits, tr, v.n_vocab as c_int, v.m_nf, cs)
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_piece_initializes_for_48_layers() {
        let cfg = Qwen4ExpModelConfig::default();
        let dec = Piece3Decode::new(&cfg);
        assert_eq!(dec.n_layers, 48);
        assert_eq!(dec.steps_executed, 0);
        assert!(!dec.recorder_installed);
    }

    #[test]
    fn c_parsers_match_libc() {
        assert_eq!(c_atoi("12abc"), 12);
        assert_eq!(c_atoi("  -3"), -3);
        assert_eq!(c_atoi("x"), 0);
        assert_eq!(c_atoi(""), 0);
        assert!((c_atof("0.5x") - 0.5).abs() < 1e-12);
        assert_eq!(c_atof("abc"), 0.0);
        assert!((c_atof("1e-2") - 0.01).abs() < 1e-12);
    }

    #[test]
    fn plan_pointer_layout_matches_verify() {
        // verify.cpp: ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1 (u64 pointers 8-byte aligned in the i32 block)
        for capx in [10i64, 30, 50, 80] {
            let ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & !1i64;
            assert_eq!(ptr_off % 2, 0);
            assert!(ptr_off >= 4 + (capx + 1) + 2 * capx);
        }
    }

    #[test]
    fn rt_flag_bits() {
        let f = RtFlags::from_bits((1 << 0) | (1 << 9) | (1 << 12));
        assert!(f.native_router && f.nan_guard && f.hip);
        assert!(!f.native_qsa && !f.gfx906);
    }
}
