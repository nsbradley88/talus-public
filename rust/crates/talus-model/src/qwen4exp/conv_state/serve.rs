//! The `--serve` conversation state in Rust: the hooks the C++ serve lambdas call (`tk_qwen4exp_p5_hooks_t`) and
//! the logic they replace (`generate.cpp`: the conversation part of a request's begin, `checkpoint_at`,
//! `pin_checkpoint`, `park_current`, the live update at a request's end, `SAVE` / `RESTORE`).
//!
//! Messages and protocol lines are the C++ engine's, byte for byte.

use std::ffi::{CStr, CString};
use std::io::Write;
use std::os::raw::{c_char, c_int};
use std::time::Instant;

use super::file::{
    self, SessionConfig, SessionError, SessionFileIdentity, SessionModelFile, SessionRope,
    SessionSaveLive, SessionStatus, SessionWriteOptions,
};
use super::memory::{ram_note, sample_host_memory};
use super::{
    checkpoint_restore, checkpoint_save, eviction_victim, memory_admit, parked_draft,
    session_read_limits, session_save_reclaim, session_sizes, snapshot_bytes,
    snapshot_capture_bytes, snapshot_restore, snapshot_save, snapshot_sources, snapshot_validate,
    Checkpoint, ConvView, ConversationCache, FnP5Copy, FnP5KvResidency, FnP5Layout, FnP5SetPlePrev,
    FnP5Sync, ImageKey, KvReuse, P5Device, P5Layout, RestoreOutcome,
};
use crate::qwen4exp::ffi::TkQwen4ExpServeCtx;

// ---------------------------------------------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------------------------------------------

/// `tk_qwen4exp_p5_options_t`: the engine settings the conversation code reads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct P5Options {
    pub prompt_cache: i32,
    pub prompt_cache_tail: i32,
    pub prompt_cache_every: i64,
    pub conversation_cache_mib: i64,
    pub conversation_cache_slots: i64,
    pub conversation_cache_min_tokens: i64,
    pub conversation_cache_min_free_mib: i64,
    pub session_min_free_mib: i64,
    pub session_save_reclaim: i32,
    pub trace: i32,
    pub max_context: i64,
    pub use_mtp: i32,
    pub reserved: i32,
}

/// `tk_qwen4exp_p5_hooks_t`: where the C++ serve lambdas hand the conversation state to Rust.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct P5Hooks {
    pub user: *mut std::ffi::c_void,
    pub begin: unsafe extern "C" fn(
        user: *mut std::ffi::c_void,
        ids: *const i64,
        n: usize,
        req_cache: c_int,
        req_ckpt: c_int,
        req_pin: i64,
        want_cvec: c_int,
        out_resume: *mut i64,
        out_read_from: *mut i64,
        out_reread_to: *mut i64,
    ) -> c_int,
    pub pin: unsafe extern "C" fn(user: *mut std::ffi::c_void, len: i64) -> c_int,
    pub checkpoint: unsafe extern "C" fn(
        user: *mut std::ffi::c_void,
        ids: *const i64,
        n_ids: usize,
        len: i64,
        as_tail: c_int,
        why: *mut c_char,
        why_cap: usize,
    ) -> c_int,
    pub finish: unsafe extern "C" fn(
        user: *mut std::ffi::c_void,
        consumed: *const i32,
        n_consumed: usize,
        cur: *const i64,
        n_cur: usize,
        cancelled: c_int,
        nf_hit: c_int,
        req_ckpt: c_int,
        pp_exact: c_int,
        pp_reached: i64,
    ),
    pub session: unsafe extern "C" fn(user: *mut std::ffi::c_void, line: *const c_char) -> c_int,
    pub check_count: unsafe extern "C" fn(user: *mut std::ffi::c_void) -> usize,
}

type Ctx = *mut TkQwen4ExpServeCtx;

/// The `qwen4exp-p5` C primitives, resolved from the loaded kernels library.
#[derive(Clone, Copy)]
pub struct P5Api {
    pub layout: FnP5Layout,
    pub copy: FnP5Copy,
    pub sync: FnP5Sync,
    pub kv_residency: FnP5KvResidency,
    pub set_ple_prev: FnP5SetPlePrev,
    pub options: unsafe extern "C" fn(ctx: Ctx, out: *mut P5Options) -> c_int,
    pub register: unsafe extern "C" fn(ctx: Ctx, hooks: *const P5Hooks) -> c_int,
    pub kvg_ensure: unsafe extern "C" fn(ctx: Ctx, cells: i64) -> c_int,
    pub kvg_trim: unsafe extern "C" fn(ctx: Ctx, cells: i64, hold: i64) -> c_int,
    pub session_zero: unsafe extern "C" fn(ctx: Ctx),
    pub mtp_kv_restore: unsafe extern "C" fn(ctx: Ctx, resume: i64),
    pub take_nan_flush: unsafe extern "C" fn(ctx: Ctx) -> c_int,
    pub wait_commit: unsafe extern "C" fn(ctx: Ctx, err: *mut c_char, cap: usize) -> c_int,
    pub session_inputs: unsafe extern "C" fn(ctx: Ctx, buf: *mut c_char, cap: usize) -> usize,
    pub set_busy: unsafe extern "C" fn(ctx: Ctx, busy: c_int),
    pub progress_at: unsafe extern "C" fn(text: *const c_char, detail: i64),
    pub progress_allow: unsafe extern "C" fn(seconds: i64),
    pub progress_beat: unsafe extern "C" fn(),
}

extern "C" {
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const c_char) -> *mut std::ffi::c_void;
}

unsafe fn sym<T>(name: &str) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    let p = dlsym(std::ptr::null_mut(), c.as_ptr());
    if p.is_null() {
        return Err(format!("missing symbol {name}"));
    }
    Ok(std::mem::transmute_copy(&p))
}

impl P5Api {
    pub fn resolve() -> Result<Self, String> {
        unsafe {
            Ok(Self {
                layout: sym("tk_qwen4exp_p5_layout")?,
                copy: sym("tk_qwen4exp_p5_copy")?,
                sync: sym("tk_qwen4exp_p5_sync")?,
                kv_residency: sym("tk_qwen4exp_p5_kv_residency")?,
                set_ple_prev: sym("tk_qwen4exp_p5_set_ple_prev")?,
                options: sym("tk_qwen4exp_p5_options")?,
                register: sym("tk_qwen4exp_p5_register_hooks")?,
                kvg_ensure: sym("tk_qwen4exp_p5_kvg_ensure")?,
                kvg_trim: sym("tk_qwen4exp_p5_kvg_trim")?,
                session_zero: sym("tk_qwen4exp_p5_session_zero")?,
                mtp_kv_restore: sym("tk_qwen4exp_p5_mtp_kv_restore")?,
                take_nan_flush: sym("tk_qwen4exp_p5_take_nan_flush")?,
                wait_commit: sym("tk_qwen4exp_p5_wait_commit")?,
                session_inputs: sym("tk_qwen4exp_p5_session_inputs")?,
                set_busy: sym("tk_qwen4exp_serve_set_busy")?,
                progress_at: sym("tk_qwen4exp_p5_progress_at")?,
                progress_allow: sym("tk_qwen4exp_p5_progress_allow")?,
                progress_beat: sym("tk_qwen4exp_p5_progress_beat")?,
            })
        }
    }
}

/// The watchdog keeps the stage text's pointer: each distinct text is interned for the process lifetime.
fn intern(text: &str) -> *const c_char {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static TEXTS: Mutex<Option<HashMap<String, &'static CStr>>> = Mutex::new(None);
    let mut g = TEXTS.lock().unwrap_or_else(|e| e.into_inner());
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(c) = m.get(text) {
        return c.as_ptr();
    }
    let c: &'static CStr = Box::leak(CString::new(text).unwrap_or_default().into_boxed_c_str());
    m.insert(text.to_string(), c);
    c.as_ptr()
}

fn progress_at(api: &P5Api, text: &str, detail: i64) {
    unsafe { (api.progress_at)(intern(text), detail) }
}

fn out_line(s: &str) {
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{s}");
    let _ = o.flush();
}

// ---------------------------------------------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------------------------------------------

/// The serve-side conversation state (`live`, `checks`, `conversations`, ...), Rust-owned.
pub struct ConvServe {
    pub ctx: Ctx,
    pub api: P5Api,
    pub dev: P5Device,
    pub opts: P5Options,
    /// What the session holds right now: the tokens it has consumed.
    pub live: Vec<i32>,
    pub live_imgs: Vec<ImageKey>,
    pub live_ok: bool,
    /// The saved points; every one is a prefix of `live` (a chain).
    pub checks: Vec<Checkpoint>,
    pub check_clock: u64,
    /// `--prompt-cache-tail`: the length of the one tail checkpoint alive (-1 = none).
    pub tail_ckpt_len: i64,
    pub cvec_cached: bool,
    pub conversations: ConversationCache,
    model_fp: Option<u64>,
    config_fp: Option<u64>,
}

impl std::fmt::Debug for ConvServe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvServe")
            .field("live", &self.live.len())
            .field("live_ok", &self.live_ok)
            .field(
                "checks",
                &self.checks.iter().map(|c| c.ids.len()).collect::<Vec<_>>(),
            )
            .field("parked", &self.conversations.size())
            .finish()
    }
}

impl ConvServe {
    pub unsafe fn resolve(ctx: Ctx) -> Result<Box<Self>, String> {
        let api = P5Api::resolve()?;
        Self::with_api(ctx, api)
    }

    pub unsafe fn with_api(ctx: Ctx, api: P5Api) -> Result<Box<Self>, String> {
        let mut opts = P5Options::default();
        if (api.options)(ctx, &mut opts) < 0 {
            return Err("tk_qwen4exp_p5_options failed".to_string());
        }
        let dev = P5Device {
            ctx,
            layout: api.layout,
            copy: api.copy,
            sync: api.sync,
            kv_residency: api.kv_residency,
            set_ple_prev: api.set_ple_prev,
        };
        let budget = if opts.prompt_cache > 0 {
            (opts.conversation_cache_mib.max(0) as usize) * 1024 * 1024
        } else {
            0
        };
        Ok(Box::new(Self {
            ctx,
            api,
            dev,
            opts,
            live: Vec::new(),
            live_imgs: Vec::new(),
            live_ok: false,
            checks: Vec::new(),
            check_clock: 0,
            tail_ckpt_len: -1,
            cvec_cached: true,
            conversations: ConversationCache::new(
                budget,
                opts.conversation_cache_slots.max(0) as usize,
                opts.conversation_cache_min_tokens,
            ),
            model_fp: None,
            config_fp: None,
        }))
    }

    fn layout(&self) -> Result<P5Layout, String> {
        self.dev.query_layout()
    }

    fn memory_limit(&self) -> u64 {
        self.layout().map(|l| l.memory_limit_bytes).unwrap_or(0)
    }

    // ---- checkpoints ------------------------------------------------------------------------------------------

    /// `checkpoint_at(L, nullptr, as_tail)`: a checkpoint of the state after `cur[0, L)`. Err = the reason suffix
    /// for "saving a conversation checkpoint failed".
    pub fn checkpoint_at(&mut self, cur: &[i64], len: i64, as_tail: bool) -> Result<(), String> {
        if self.opts.prompt_cache <= 0 || len < 1 {
            return Ok(());
        }
        let l = len as usize;
        for c in &mut self.checks {
            if c.ids.len() == l {
                self.check_clock += 1;
                c.used = self.check_clock;
                if !as_tail && len == self.tail_ckpt_len {
                    self.tail_ckpt_len = -1;
                }
                return Ok(());
            }
        }
        let mut c = Checkpoint {
            ids: cur[..l.min(cur.len())].iter().map(|&t| t as i32).collect(),
            ..Default::default()
        };
        if let Err(e) = self.dev.sync() {
            return Err(format!(
                ": the GPU stopped responding before it ({e}) - a GPU hang; on Windows the driver is then reset"
            ));
        }
        let lay = self.layout().map_err(|e| {
            eprintln!("strata serve: checkpoint save: {e}");
            String::new()
        })?;
        if let Err(e) = checkpoint_save(&mut c, &self.dev, &lay) {
            eprintln!("strata serve: checkpoint save: {e}");
            return Err(String::new());
        }
        self.check_clock += 1;
        c.used = self.check_clock;
        if as_tail {
            let old = self.tail_ckpt_len;
            self.checks
                .retain(|k| !(old >= 0 && k.ids.len() as i64 == old));
            self.tail_ckpt_len = len;
        } else if len == self.tail_ckpt_len {
            self.tail_ckpt_len = -1;
        }
        self.checks.push(c);
        while self.checks.len() as i64 > self.opts.prompt_cache as i64 {
            let stamps: Vec<u64> = self.checks.iter().map(|k| k.used).collect();
            let tail: Vec<bool> = self
                .checks
                .iter()
                .map(|k| self.tail_ckpt_len >= 0 && k.ids.len() as i64 == self.tail_ckpt_len)
                .collect();
            let pins: Option<Vec<bool>> = self
                .checks
                .iter()
                .any(|k| k.pinned)
                .then(|| self.checks.iter().map(|k| k.pinned).collect());
            let victim = eviction_victim(
                &stamps,
                self.opts.prompt_cache as i64,
                Some(&tail),
                pins.as_deref(),
            );
            super::recycle_gdn(self.checks.remove(victim).gdn);
        }
        Ok(())
    }

    /// `pin_checkpoint(L)`: marks the checkpoint of length L as the shared prefix (and only it).
    pub fn pin(&mut self, len: i64) -> bool {
        let mut found = false;
        for c in &mut self.checks {
            c.pinned = c.ids.len() as i64 == len;
            found = found || c.pinned;
        }
        found
    }

    // ---- parking ----------------------------------------------------------------------------------------------

    /// `park_current(held)`: the outgoing conversation into the parked cache. Err only when the capture itself
    /// failed (every refusal is a logged, non-fatal skip).
    fn park_current(&mut self, held: usize) -> Result<(), String> {
        if !self.conversations.enabled() || !self.live_ok || self.live.is_empty() {
            return Ok(());
        }
        progress_at(&self.api, "request: parking the previous conversation", -1);
        let r = self.park_body(held);
        progress_at(&self.api, "request", -1);
        unsafe { (self.api.progress_beat)() };
        r
    }

    fn park_body(&mut self, held: usize) -> Result<(), String> {
        if !self.conversations.wants(self.live.len() as i64) {
            eprintln!(
                "strata serve: conversation cache: skip parking ({} tokens, below the {} minimum; parked={})",
                self.live.len(),
                self.opts.conversation_cache_min_tokens,
                self.conversations.size()
            );
            return Ok(());
        }
        let dropped = self.conversations.drop_superseded(
            &self.live,
            &self.live_imgs,
            &self.checks,
            self.cvec_cached,
        );
        if dropped > 0 {
            eprintln!(
                "strata serve: conversation cache: dropped {dropped} superseded cop{} of this conversation; parked={}",
                if dropped == 1 { "y" } else { "ies" },
                self.conversations.size()
            );
        }
        let lay = self.layout()?;
        let draft = parked_draft(&lay);
        let view = ConvView {
            ids: &self.live,
            images: &self.live_imgs,
            checkpoints: &self.checks,
            cvec: self.cvec_cached,
        };
        let mut reuse = self.conversations.take_reuse();
        let mut estimate = match snapshot_bytes(&view, &lay, draft) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("strata serve: conversation cache: skip parking ({e})");
                return Ok(());
            }
        };
        let fresh = estimate;
        if !reuse.kv.is_empty() {
            match snapshot_capture_bytes(&reuse, &view, &lay, draft) {
                Ok(b) => estimate = b,
                Err(_) => {
                    reuse = KvReuse::default();
                    estimate = fresh;
                }
            }
        }
        if !self.conversations.make_room(estimate, held) {
            eprintln!(
                "strata serve: conversation cache: skip parking (snapshot {} MiB exceeds available budget)",
                estimate >> 20
            );
            return Ok(());
        }
        let t0 = Instant::now();
        let floor = (self.opts.conversation_cache_min_free_mib.max(0) as u64) << 20;
        let retained = reuse.bytes();
        let additional = estimate.saturating_sub(retained) as u64;
        let limit = lay.memory_limit_bytes;
        let mut mem = None;
        let admit = |mem: &mut Option<_>| {
            *mem = sample_host_memory(limit);
            memory_admit(
                mem.map(|m: super::memory::HostMemoryReading| m.available),
                additional,
                floor,
            )
        };
        let mut evicted = 0usize;
        while !admit(&mut mem)
            && self.conversations.size() > 0
            && evicted < self.conversations.slots()
            && self.conversations.evict_oldest()
        {
            evicted += 1;
        }
        if !admit(&mut mem) {
            eprintln!(
                "strata serve: conversation cache: skip parking (physical RAM admission; need {} MiB plus {} MiB floor, {}; evicted {}, {} still parked)",
                additional >> 20,
                self.opts.conversation_cache_min_free_mib,
                ram_note(&mem),
                evicted,
                self.conversations.size()
            );
            return Ok(());
        }
        if evicted > 0 {
            eprintln!(
                "strata serve: conversation cache: evicted {evicted} parked conversation{} to admit this snapshot ({} MiB plus {} MiB floor)",
                if evicted == 1 { "" } else { "s" },
                additional >> 20,
                self.opts.conversation_cache_min_free_mib
            );
        }
        let mut reused_bytes = 0usize;
        let view = ConvView {
            ids: &self.live,
            images: &self.live_imgs,
            checkpoints: &self.checks,
            cvec: self.cvec_cached,
        };
        let image = snapshot_save(&view, &lay, &self.dev, draft, reuse, &mut reused_bytes)?;
        let mem_after = sample_host_memory(limit);
        if !memory_admit(mem_after.map(|m| m.available), 0, floor) {
            eprintln!(
                "strata serve: conversation cache: skip parking (RAM floor after capture: {} MiB needed, {})",
                self.opts.conversation_cache_min_free_mib,
                ram_note(&mem_after)
            );
            return Ok(());
        }
        let snapshot_bytes_n = image.bytes();
        let stored = self.conversations.put(image, held);
        eprintln!(
            "strata serve: conversation cache: {} {} tokens in {:.1} ms; parked={} bytes={} evictions={} snapshot_bytes={} reused_kv_bytes={}",
            if stored { "parked" } else { "skipped" },
            self.live.len(),
            t0.elapsed().as_secs_f64() * 1000.0,
            self.conversations.size(),
            self.conversations.bytes(),
            self.conversations.evictions(),
            snapshot_bytes_n,
            reused_bytes
        );
        Ok(())
    }

    // ---- a request's begin ------------------------------------------------------------------------------------

    /// The conversation part of a request's begin: picks the resume point (live session, a checkpoint, a parked
    /// conversation), parks the outgoing conversation, restores what the request continues from.
    /// Returns (resume, read_from, reread_to); Err = an `ERR` line was printed and the engine stops.
    #[allow(clippy::too_many_arguments, clippy::result_unit_err)]
    pub fn begin(
        &mut self,
        cur: &[i64],
        req_cache: bool,
        req_ckpt: bool,
        req_pin: i64,
        want_cvec: bool,
    ) -> Result<(i64, i64, i64), ()> {
        let n = cur.len() as i64;
        let starts_with = |pre: &[i32], pre_imgs: &[ImageKey]| -> bool {
            let len = pre.len() as i64;
            if len < 1 || len > n - 1 {
                return false;
            }
            pre.iter().zip(cur).all(|(&a, &b)| a as i64 == b) && pre_imgs.is_empty()
        };
        if unsafe { (self.api.take_nan_flush)(self.ctx) } != 0 {
            let n_checks = self.checks.len();
            self.live_ok = false;
            for c in self.checks.drain(..) {
                super::recycle_gdn(c.gdn);
            }
            self.tail_ckpt_len = -1;
            let n_parked = self.conversations.clear();
            eprintln!(
                "strata serve: after non-finite logits (#879): dropped the live session, {n_checks} checkpoint{}, {n_parked} parked conversation{} and 0 idle slots (0 running slots not kept after their reply); reading this prompt from token 0",
                if n_checks == 1 { "" } else { "s" },
                if n_parked == 1 { "" } else { "s" }
            );
        }
        let mut resume = 0i64;
        let mut from_live = false;
        if self.opts.prompt_cache > 0 && req_cache && want_cvec == self.cvec_cached {
            if self.live_ok && starts_with(&self.live, &self.live_imgs) {
                resume = self.live.len() as i64;
                from_live = true;
            }
            for c in &self.checks {
                if c.ids.len() as i64 > resume && starts_with(&c.ids, &c.imgs) {
                    resume = c.ids.len() as i64;
                    from_live = false;
                }
            }
        }
        let parked = self.conversations.best(cur, &[], want_cvec);
        let mut incoming = None;
        if req_cache && parked.tokens > resume {
            incoming = Some(self.conversations.take(parked.index));
        }
        if let Some(inc) = &incoming {
            let v = self
                .layout()
                .and_then(|l| snapshot_validate(inc, &l, parked_draft(&l)));
            if let Err(e) = v {
                eprintln!("strata serve: conversation cache: discard invalid snapshot ({e})");
                incoming = None;
            }
        }
        let mut pin_sibling = false;
        if req_pin > 0
            && req_ckpt
            && self.opts.prompt_cache >= 3
            && !from_live
            && incoming.is_none()
            && resume == req_pin
            && self.live_ok
        {
            pin_sibling = self
                .checks
                .iter()
                .any(|c| c.ids.len() as i64 == resume && c.pinned);
        }
        if (!from_live || incoming.is_some()) && !pin_sibling {
            let held = incoming.as_ref().map_or(0, |i| i.bytes());
            if let Err(e) = self.park_current(held) {
                out_line(&format!("ERR {e}"));
                return Err(());
            }
        }
        if let Some(inc) = &incoming {
            if unsafe { (self.api.kvg_ensure)(self.ctx, inc.live.ids.len() as i64 + 256) } == 0 {
                out_line("ERR the K/V cannot grow to the parked conversation: no VRAM is left");
                return Err(());
            }
        }
        if let Some(inc) = incoming.take() {
            let t0 = Instant::now();
            let lay = match self.layout() {
                Ok(l) => l,
                Err(e) => {
                    out_line(&format!("ERR restoring parked conversation: {e}"));
                    return Err(());
                }
            };
            let (o, e) = snapshot_restore(&inc, &lay, &self.dev, parked_draft(&lay));
            if o != RestoreOutcome::Restored {
                out_line(&format!("ERR restoring parked conversation: {e}"));
                return Err(());
            }
            let super::SavedConversation {
                live,
                checkpoints,
                kv,
                cvec,
                ..
            } = inc;
            self.live = live.ids;
            self.live_imgs = live.imgs;
            self.checks = checkpoints;
            self.cvec_cached = cvec;
            resume = parked.tokens;
            from_live = parked.live;
            if std::env::var_os("STRATA_SNAPSHOT_FULL_CAPTURE").is_none() {
                self.conversations.retain(kv, self.live.len() as i64);
            }
            eprintln!(
                "strata serve: conversation cache: restored {} tokens ({}) in {:.1} ms; parked={} bytes={}",
                resume,
                if from_live { "live" } else { "checkpoint" },
                t0.elapsed().as_secs_f64() * 1000.0,
                self.conversations.size(),
                self.conversations.bytes()
            );
        }
        if want_cvec != self.cvec_cached {
            self.live_ok = false;
            for c in self.checks.drain(..) {
                super::recycle_gdn(c.gdn);
            }
            self.cvec_cached = want_cvec;
        }
        for c in std::mem::take(&mut self.checks) {
            if c.ids.len() as i64 <= resume && starts_with(&c.ids, &c.imgs) {
                self.checks.push(c);
            } else {
                super::recycle_gdn(c.gdn);
            }
        }
        self.live_ok = false;
        let hold_on = std::env::var("STRATA_KV_GROW_HOLD")
            .map(|v| atoi(&v) != 0)
            .unwrap_or(false);
        let hold = if hold_on {
            self.conversations.longest_tokens() + 256
        } else {
            0
        };
        if unsafe { (self.api.kvg_trim)(self.ctx, n + 256, hold) } == 0 {
            out_line("ERR the K/V could not give its VRAM back to the expert cache");
            return Err(());
        }
        let mut reread_to = -1i64;
        if resume == 0 {
            unsafe { (self.api.session_zero)(self.ctx) };
            for c in self.checks.drain(..) {
                super::recycle_gdn(c.gdn);
            }
        } else if !from_live {
            let ci = self
                .checks
                .iter()
                .rposition(|k| k.ids.len() as i64 == resume);
            if let Some(i) = ci {
                self.check_clock += 1;
                self.checks[i].used = self.check_clock;
            }
            let reread = std::env::var_os("STRATA_CKPT_REREAD").is_some();
            if reread && ci.is_some() {
                unsafe { (self.api.session_zero)(self.ctx) };
                reread_to = resume;
                eprintln!(
                    "strata serve: STRATA_CKPT_REREAD: reading {resume} tokens again instead of restoring"
                );
            } else {
                let ok = match (ci, self.layout()) {
                    (Some(i), Ok(lay)) => {
                        match checkpoint_restore(&self.checks[i], &self.dev, &lay) {
                            Ok(()) => true,
                            Err(e) => {
                                eprintln!("strata serve: checkpoint restore: {e}");
                                false
                            }
                        }
                    }
                    (Some(_), Err(e)) => {
                        eprintln!("strata serve: checkpoint restore: {e}");
                        false
                    }
                    (None, _) => false,
                };
                if !ok {
                    out_line("ERR restoring a conversation checkpoint failed");
                    return Err(());
                }
            }
        }
        if resume > 0 && reread_to <= 0 {
            unsafe { (self.api.mtp_kv_restore)(self.ctx, resume) };
        }
        let read_from = if reread_to > 0 { 0 } else { resume };
        self.conversations.limit_reuse(read_from);
        Ok((resume, read_from, reread_to))
    }

    /// The live update at a request's end.
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        &mut self,
        consumed: &[i32],
        cur: &[i64],
        cancelled: bool,
        nf_hit: bool,
        req_ckpt: bool,
        pp_exact: bool,
        pp_reached: i64,
    ) {
        if !cancelled {
            self.live = consumed.to_vec();
            self.live_imgs.clear();
            self.live_ok = self.opts.prompt_cache > 0 && req_ckpt;
            if nf_hit {
                self.live_ok = false;
            }
        } else if !nf_hit
            && pp_exact
            && pp_reached > 0
            && self.opts.prompt_cache > 0
            && req_ckpt
            && pp_reached <= cur.len() as i64
        {
            self.live = cur[..pp_reached as usize]
                .iter()
                .map(|&t| t as i32)
                .collect();
            self.live_imgs.clear();
            self.live_ok = true;
            eprintln!(
                "strata serve: cancelled request keeps its conversation: {pp_reached} tokens live"
            );
        }
    }

    // ---- SAVE / RESTORE ---------------------------------------------------------------------------------------

    fn identity(&mut self, beat: &mut dyn FnMut()) -> Result<SessionFileIdentity, String> {
        if self.model_fp.is_none() || self.config_fp.is_none() {
            let text = unsafe {
                let need = (self.api.session_inputs)(self.ctx, std::ptr::null_mut(), 0);
                let mut buf = vec![0u8; need + 1];
                (self.api.session_inputs)(self.ctx, buf.as_mut_ptr().cast(), buf.len());
                buf.truncate(need);
                String::from_utf8_lossy(&buf).into_owned()
            };
            let (files, cfg) = parse_session_inputs(&text)?;
            if self.model_fp.is_none() {
                beat();
                self.model_fp = Some(file::session_model_fingerprint(&files, &mut *beat)?);
            }
            if self.config_fp.is_none() {
                self.config_fp = Some(file::session_config_fingerprint(&cfg));
            }
        }
        Ok(SessionFileIdentity {
            model: self.model_fp.unwrap(),
            config: self.config_fp.unwrap(),
        })
    }

    /// A `SAVE <path>` / `RESTORE <path>` line. 0 = answered (SAVED / RESTORED / SERR), -1 = the engine stops.
    pub fn session_command(&mut self, line: &str) -> i32 {
        unsafe { (self.api.set_busy)(self.ctx, 1) };
        let r = self.session_body(line);
        unsafe { (self.api.set_busy)(self.ctx, 0) };
        let _ = std::io::stdout().flush();
        r
    }

    fn session_body(&mut self, line: &str) -> i32 {
        let save = line.starts_with('S');
        let path = line
            .get(if save { 5 } else { 8 }..)
            .unwrap_or("")
            .to_string();
        let t0 = Instant::now();
        let ms = || t0.elapsed().as_secs_f64() * 1000.0;
        let refuse = |why: &str, kind: SessionError, published: bool| {
            eprintln!(
                "strata serve: session {} {}: {}",
                if save { "save" } else { "restore" },
                path,
                why
            );
            let one: String = why
                .chars()
                .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                .collect();
            out_line(&format!(
                "SERR {} {} {}",
                kind.name(),
                published as i32,
                one
            ));
        };
        let api = self.api;
        let mut moving = |done: u64, total: u64| {
            progress_at(
                &api,
                if save {
                    "writing a session file, MiB"
                } else {
                    "reading a session file, MiB"
                },
                (done >> 20) as i64,
            );
            unsafe { (api.progress_allow)(0) };
            out_line(&format!("SESSION {done} {total}"));
        };
        let blocking = |phase: &str, bytes: u64| {
            let s = file::session_phase_limit_s(bytes);
            progress_at(&api, phase, -1);
            unsafe { (api.progress_allow)(s) };
            out_line(&format!("SWAIT {phase} {s}"));
        };
        progress_at(
            &api,
            if save {
                "saving a session"
            } else {
                "restoring a session"
            },
            -1,
        );
        if path.is_empty() {
            refuse("missing path", SessionError::Invalid, false);
            return 0;
        }
        if self.opts.prompt_cache <= 0 {
            refuse(
                "session files need --prompt-cache > 0",
                SessionError::Invalid,
                false,
            );
            return 0;
        }
        let mut e = [0u8; 512];
        if unsafe { (api.wait_commit)(self.ctx, e.as_mut_ptr().cast(), e.len()) } == 0 {
            out_line(&format!("ERR {}", super::err_text(&e)));
            return -1;
        }
        let id = match self.identity(&mut || blocking("fingerprint", 2u64 << 20)) {
            Ok(id) => id,
            Err(e) => {
                refuse(&e, SessionError::Io, false);
                return 0;
            }
        };
        let limit = self.memory_limit();
        let floor = (self.opts.conversation_cache_min_free_mib.max(0) as u64) << 20;
        let floor_mib = self.opts.conversation_cache_min_free_mib;
        if save {
            if !self.live_ok || self.live.is_empty() {
                refuse("no complete session to save", SessionError::Invalid, false);
                return 0;
            }
            let lay = match self.layout() {
                Ok(l) => l,
                Err(e) => {
                    refuse(&e, SessionError::Io, false);
                    return 0;
                }
            };
            let mut sl = SessionSaveLive::default();
            let mut state = u64::MAX;
            if let Ok(z) = session_sizes(&lay) {
                let q = lay.qsa_alloc.max(0) as u64;
                let per = (z.tail + z.dead + z.block_pos) as u64;
                let (g, p) = (z.gdn as u64, z.ple as u64);
                if q == 0 || per <= (u64::MAX - g - p) / q {
                    state = g + p + q * per;
                }
            }
            sl.state_bytes = state;
            sl.tokens = self.live.len() as u64;
            sl.images = self.live_imgs.len() as u64;
            sl.kv_layers = lay.qsa_alloc.max(0) as u64 + 1;
            if self.opts.session_save_reclaim != 0 {
                let freed = session_save_reclaim(
                    &mut self.checks,
                    &mut self.conversations,
                    &sl,
                    floor,
                    || sample_host_memory(limit).map(|m| m.available),
                );
                if freed.bytes > 0 || freed.checkpoints > 0 || freed.parked > 0 {
                    if self.tail_ckpt_len >= 0
                        && !self
                            .checks
                            .iter()
                            .any(|c| c.ids.len() as i64 == self.tail_ckpt_len)
                    {
                        self.tail_ckpt_len = -1;
                    }
                    eprintln!(
                        "strata serve: save reclaimed {} checkpoints, {} parked, {} host bytes; checking RAM again",
                        freed.checkpoints, freed.parked, freed.bytes
                    );
                }
            }
            let admit =
                |need: u64| -> Result<(), String> {
                    let mem = sample_host_memory(limit);
                    if memory_admit(mem.map(|m| m.available), need, floor) {
                        return Ok(());
                    }
                    Err(format!(
                    "not enough RAM to save the session ({} MiB plus a floor of {} MiB needed, {})",
                    if need == u64::MAX { "unknown".to_string() } else { (need >> 20).to_string() },
                    floor_mib,
                    ram_note(&mem)
                ))
                };
            let deepest = file::session_deepest_checkpoint(&self.checks);
            if let Err(why) = admit(file::session_save_peak_bytes(deepest, &sl)) {
                refuse(&why, SessionError::Memory, false);
                return 0;
            }
            let disk_checks: Vec<Checkpoint> = deepest.cloned().into_iter().collect();
            let kept = disk_checks.len();
            let view = ConvView {
                ids: &self.live,
                images: &self.live_imgs,
                checkpoints: &disk_checks,
                cvec: self.cvec_cached,
            };
            blocking(
                "capture",
                if sl.state_bytes == u64::MAX {
                    0
                } else {
                    sl.state_bytes
                },
            );
            let (meta, mut sources) = match snapshot_sources(&view, &lay, &self.dev) {
                Ok(v) => v,
                Err(e) => {
                    refuse(&e, SessionError::Io, false);
                    return 0;
                }
            };
            let capture_ms = ms();
            let mut st = SessionStatus::default();
            let mut phase = |p: &str, b: u64| blocking(p, b);
            let opt = SessionWriteOptions {
                min_free_bytes: (self.opts.session_min_free_mib.max(0) as u64) << 20,
                durable: true,
                progress: Some(&mut moving),
                phase: Some(&mut phase),
            };
            let bytes =
                match file::session_file_write(&path, &meta, Some(&mut sources), &id, opt, &mut st)
                {
                    Ok(b) => b,
                    Err(e) => {
                        refuse(&e, st.error, st.published);
                        return 0;
                    }
                };
            if st.dir_flush_unsupported {
                eprintln!(
                    "strata serve: session saved to {path}; this filesystem cannot flush a folder, so the new name is not flushed"
                );
            }
            eprintln!(
                "strata serve: session saved {} tokens, {} of {} checkpoints, {} bytes to {} in {:.1} ms (state {:.1} ms)",
                self.live.len(),
                kept,
                self.checks.len(),
                bytes,
                path,
                ms(),
                capture_ms
            );
            out_line(&format!("SAVED {} {} {:.1}", self.live.len(), bytes, ms()));
        } else {
            let lay = match self.layout() {
                Ok(l) => l,
                Err(e) => {
                    refuse(&e, SessionError::Io, false);
                    return 0;
                }
            };
            let mut limits = match session_read_limits(
                &lay,
                self.opts.max_context.max(0) as u64,
                self.opts.prompt_cache.max(1) as u64,
            ) {
                Ok(l) => l,
                Err(e) => {
                    refuse(&e, SessionError::Io, false);
                    return 0;
                }
            };
            let mut admit_read = |need: u64| -> Result<(), String> {
                let mem = sample_host_memory(limit);
                if memory_admit(mem.map(|m| m.available), need, floor) {
                    return Ok(());
                }
                Err(format!(
                    "not enough RAM to read it ({} MiB plus a floor of {} MiB needed, {})",
                    need >> 20,
                    floor_mib,
                    ram_note(&mem)
                ))
            };
            limits.progress = Some(&mut moving);
            limits.admit = Some(&mut admit_read);
            let mut st = SessionStatus::default();
            let (image, bytes) = match file::session_file_read(&path, &id, &mut limits, &mut st) {
                Ok(v) => v,
                Err(e) => {
                    refuse(&e, st.error, false);
                    return 0;
                }
            };
            drop(limits);
            let read_ms = ms();
            blocking("validate", bytes);
            if let Err(e) = snapshot_validate(&image, &lay, Some(&lay.draft)) {
                refuse(&e, SessionError::Invalid, false);
                return 0;
            }
            if unsafe { (api.kvg_ensure)(self.ctx, image.live.ids.len() as i64 + 256) } == 0 {
                refuse(
                    "the K/V cannot grow to the saved conversation: no VRAM is left",
                    SessionError::Memory,
                    false,
                );
                return 0;
            }
            drop(self.conversations.take_reuse());
            self.live_ok = false;
            blocking("transfer", bytes);
            let lay = self.layout().unwrap_or(lay);
            let (o, e) = snapshot_restore(&image, &lay, &self.dev, Some(&lay.draft));
            if o != RestoreOutcome::Restored {
                eprintln!("strata serve: session restore {path}: transfer failed: {e}");
                out_line(&format!(
                    "FATAL restoring the session file failed after the device state was changed: {e}"
                ));
                return -1;
            }
            let super::SavedConversation {
                live,
                checkpoints,
                cvec,
                ..
            } = image;
            self.live = live.ids;
            self.live_imgs = live.imgs;
            self.checks = checkpoints;
            for c in &self.checks {
                self.check_clock = self.check_clock.max(c.used);
            }
            self.cvec_cached = cvec;
            self.live_ok = true;
            if unsafe { (api.take_nan_flush)(self.ctx) } != 0 {
                let n_parked = self.conversations.clear();
                eprintln!(
                    "strata serve: after non-finite logits (#879): the restored session replaces the live one; dropped {n_parked} parked conversation{}",
                    if n_parked == 1 { "" } else { "s" }
                );
            }
            eprintln!(
                "strata serve: session restored {} tokens, {} checkpoints, {} bytes from {} in {:.1} ms (read+check {:.1} ms)",
                self.live.len(),
                self.checks.len(),
                bytes,
                path,
                ms(),
                read_ms
            );
            out_line(&format!(
                "RESTORED {} {} {:.1}",
                self.live.len(),
                bytes,
                ms()
            ));
        }
        0
    }
}

fn atoi(s: &str) -> i64 {
    let t = s.trim_start();
    let (neg, d) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let digits: String = d.chars().take_while(|c| c.is_ascii_digit()).collect();
    let v: i64 = digits.parse().unwrap_or(0);
    if neg {
        -v
    } else {
        v
    }
}

/// Parses the `tk_qwen4exp_p5_session_inputs` text: `F\t<optional>\t<role>\t<path>` lines (the model inputs in
/// order), then the config: `S|I|D|U\t<key>\t<value>` (D: a double's bits in hex) and `W\t<name>\t<value>`
/// (the arithmetic switches in order).
pub fn parse_session_inputs(text: &str) -> Result<(Vec<SessionModelFile>, SessionConfig), String> {
    let mut files = Vec::new();
    let mut c = SessionConfig::default();
    let bad = |l: &str| format!("session identity: malformed input line '{l}'");
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.splitn(4, '\t').collect();
        match f[0] {
            "F" if f.len() == 4 => files.push(SessionModelFile {
                role: f[2].to_string(),
                path: f[3].to_string(),
                optional: f[1] == "1",
            }),
            "S" if f.len() >= 3 => {
                let v = line.splitn(3, '\t').nth(2).unwrap_or("").to_string();
                match f[1] {
                    "engine" => c.engine_version = v,
                    "backend" => c.backend = v,
                    "kv" => c.kv = v,
                    _ => return Err(bad(line)),
                }
            }
            "I" if f.len() >= 3 => {
                let v: i64 = f[2].parse().map_err(|_| bad(line))?;
                match f[1] {
                    "max_context" => c.max_context = v,
                    "kv_resident" => c.kv_resident = v,
                    "mtp_window" => c.mtp_window = v,
                    "kv_rot" => c.kv_rot = v != 0,
                    "rope.type" => c.rope.ty = v,
                    _ => return Err(bad(line)),
                }
            }
            "D" if f.len() >= 3 => {
                let v = f64::from_bits(u64::from_str_radix(f[2], 16).map_err(|_| bad(line))?);
                let r: &mut SessionRope = &mut c.rope;
                match f[1] {
                    "rope.freq_base" => r.freq_base = v,
                    "rope.factor" => r.factor = v,
                    "rope.freq_scale" => r.freq_scale = v,
                    "rope.orig_ctx" => r.orig_ctx = v,
                    "rope.ext_factor" => r.ext_factor = v,
                    "rope.attn_factor" => r.attn_factor = v,
                    "rope.beta_fast" => r.beta_fast = v,
                    "rope.beta_slow" => r.beta_slow = v,
                    _ => return Err(bad(line)),
                }
            }
            "U" if f.len() >= 3 && f[1] == "cvec" => {
                c.cvec = f[2].parse().map_err(|_| bad(line))?
            }
            "W" if f.len() >= 3 => c
                .switches
                .push((f[1].to_string(), f[2].parse().map_err(|_| bad(line))?)),
            _ => return Err(bad(line)),
        }
    }
    Ok((files, c))
}

// ---------------------------------------------------------------------------------------------------------------
// hooks
// ---------------------------------------------------------------------------------------------------------------

unsafe fn me<'a>(user: *mut std::ffi::c_void) -> &'a mut ConvServe {
    &mut *user.cast::<ConvServe>()
}

unsafe fn slice<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if p.is_null() || n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(p, n)
    }
}

unsafe extern "C" fn hook_begin(
    user: *mut std::ffi::c_void,
    ids: *const i64,
    n: usize,
    req_cache: c_int,
    req_ckpt: c_int,
    req_pin: i64,
    want_cvec: c_int,
    out_resume: *mut i64,
    out_read_from: *mut i64,
    out_reread_to: *mut i64,
) -> c_int {
    let s = me(user);
    match s.begin(
        slice(ids, n),
        req_cache != 0,
        req_ckpt != 0,
        req_pin,
        want_cvec != 0,
    ) {
        Ok((resume, read_from, reread_to)) => {
            *out_resume = resume;
            *out_read_from = read_from;
            *out_reread_to = reread_to;
            1
        }
        Err(()) => -1,
    }
}

unsafe extern "C" fn hook_pin(user: *mut std::ffi::c_void, len: i64) -> c_int {
    me(user).pin(len) as c_int
}

unsafe extern "C" fn hook_checkpoint(
    user: *mut std::ffi::c_void,
    ids: *const i64,
    n_ids: usize,
    len: i64,
    as_tail: c_int,
    why: *mut c_char,
    why_cap: usize,
) -> c_int {
    match me(user).checkpoint_at(slice(ids, n_ids), len, as_tail != 0) {
        Ok(()) => {
            if !why.is_null() && why_cap > 0 {
                *why = 0;
            }
            1
        }
        Err(w) => {
            if !why.is_null() && why_cap > 0 {
                let b = w.as_bytes();
                let n = b.len().min(why_cap - 1);
                std::ptr::copy_nonoverlapping(b.as_ptr(), why.cast::<u8>(), n);
                *why.add(n) = 0;
            }
            0
        }
    }
}

unsafe extern "C" fn hook_finish(
    user: *mut std::ffi::c_void,
    consumed: *const i32,
    n_consumed: usize,
    cur: *const i64,
    n_cur: usize,
    cancelled: c_int,
    nf_hit: c_int,
    req_ckpt: c_int,
    pp_exact: c_int,
    pp_reached: i64,
) {
    me(user).finish(
        slice(consumed, n_consumed),
        slice(cur, n_cur),
        cancelled != 0,
        nf_hit != 0,
        req_ckpt != 0,
        pp_exact != 0,
        pp_reached,
    );
}

unsafe extern "C" fn hook_session(user: *mut std::ffi::c_void, line: *const c_char) -> c_int {
    let l = if line.is_null() {
        String::new()
    } else {
        CStr::from_ptr(line).to_string_lossy().into_owned()
    };
    me(user).session_command(&l)
}

unsafe extern "C" fn hook_check_count(user: *mut std::ffi::c_void) -> usize {
    me(user).checks.len()
}

/// The hook table (its `user` is set by the caller to the boxed [`ConvServe`]).
pub fn hooks_for(_s: &mut ConvServe) -> P5Hooks {
    P5Hooks {
        user: std::ptr::null_mut(),
        begin: hook_begin,
        pin: hook_pin,
        checkpoint: hook_checkpoint,
        finish: hook_finish,
        session: hook_session,
        check_count: hook_check_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_inputs_parse() {
        let t = "F\t0\tnative shard 0\t/m/a.gguf\nF\t1\tpack native_experts.txt\t/p/native_experts.txt\n\
                 S\tengine\t0.1.50\nS\tbackend\thip\nS\tkv\tint8\nI\tmax_context\t40000\nI\tkv_resident\t0\n\
                 I\tmtp_window\t8192\nI\tkv_rot\t0\nI\trope.type\t2\nD\trope.freq_base\t40c3880000000000\n\
                 U\tcvec\t0\nW\tno_ple\t0\nW\tvision\t0\n";
        let (f, c) = parse_session_inputs(t).unwrap();
        assert_eq!(f.len(), 2);
        assert!(f[1].optional);
        assert_eq!(c.backend, "hip");
        assert_eq!(c.rope.freq_base, 10000.0);
        assert_eq!(
            c.switches,
            vec![("no_ple".to_string(), 0), ("vision".to_string(), 0)]
        );
        assert!(parse_session_inputs("X\tfoo\n").is_err());
    }

    #[test]
    fn atoi_matches_c() {
        assert_eq!(atoi("1"), 1);
        assert_eq!(atoi(" 12x"), 12);
        assert_eq!(atoi("x"), 0);
        assert_eq!(atoi("-3"), -3);
    }
}
