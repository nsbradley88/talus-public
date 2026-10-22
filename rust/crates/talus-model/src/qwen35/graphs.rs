//! HIP Graph capture, speculative round execution (`gA`/`gB`/`gB0`/`gC`/`gS`/`gD`/`gD0`/`gDs`),
//! adaptive draft length (`DraftLen`), DFlash2 drafter (`DFlash`), N-gram proposer (`NgramMod`),
//! and sampling (`Sampling`) from `q36_engine.cu` lines 1549–1993 & 3102–3345.

#![allow(non_snake_case, unused_unsafe)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::ptr;

use super::engine::{hip_alloc_zeroed, hip_host_alloc_zeroed, Q36Engine};
use super::ffi::*;
use super::weights::{
    hip_check, hip_malloc_bytes, meta_f32_or, meta_i64_or, qtype_row_bytes, Mat, GGML_TYPE_Q8_0,
};
use talus_gguf::{GgufFile, GgufValue};

// ============================================================================
// Adaptive Draft Length (`q36_adraft_policy.{hpp,cpp}`)
// ============================================================================

pub const K_MAX_DRAFT: usize = 15;
pub const K_MAX_STEPS: usize = 8;

#[derive(Debug)]
pub struct DraftLen {
    pub mode: i32,
    pub dfix: i32,
    pub dmin: i32,
    pub dmax: i32,
    pub chain: i32,
    pub pmin: [f32; 2],
    pub cum: [f32; 2],
    pub dfl_stop: bool,
    pub dfl_sample: bool,
    pub samp: bool,
    pub dfl_dmin: i32,
    pub dfl_cum: [f32; 2],
    pub tries: [[f64; K_MAX_DRAFT + 1]; 2],
    pub keeps: [[f64; K_MAX_DRAFT + 1]; 2],
    pub cost: [f64; K_MAX_DRAFT + 1],

    pub tau_d: *mut f32,
    pub lmin_d: *mut f32,
    pub top_p_d: *mut f32,
    pub top_k_d: *mut i32,
    pub conf_d: *mut f32,
    pub conf_h: *mut f32,
    pub top_d: *mut f32,
    pub top_h: *mut f32,
    pub lse_d: *mut f32,
    pub thr_d: *mut f32,

    pub tau_last: f32,
    pub lmin_last: f32,
    pub top_p_last: f32,
    pub top_k_last: i32,

    pub last_conf: [f32; 16],
    pub last_top: [f32; 16],
    pub hist_acc: [i32; K_MAX_STEPS],
    pub hist_n: [i32; K_MAX_STEPS],
    pub hist_acc_sum: i32,
    pub hist_n_sum: i32,
    pub ema_alpha: f32,
    pub c_stop: f32,
    pub c_cont: f32,
    pub k_cap: i32,
    pub enabled: bool,
    pub fixed_k: i32,
    pub steps: i32,

    pub total_rounds: i64,
    pub total_drafted: i64,
    pub total_accepted: i64,
    pub early_stops: i64,

    allocs: Vec<*mut c_void>,
    hallocs: Vec<*mut c_void>,
}

unsafe impl Send for DraftLen {}
unsafe impl Sync for DraftLen {}

impl Default for DraftLen {
    fn default() -> Self {
        Self {
            mode: 0,
            dfix: 0,
            dmin: 2,
            dmax: 7,
            chain: 3,
            pmin: [0.5, 0.0],
            cum: [0.40, 0.45],
            dfl_stop: false,
            dfl_sample: false,
            samp: false,
            dfl_dmin: 2,
            dfl_cum: [0.40, 0.30],
            tries: [[8.0; K_MAX_DRAFT + 1]; 2],
            keeps: [[8.0 * 0.80; K_MAX_DRAFT + 1], [8.0 * 0.70; K_MAX_DRAFT + 1]],
            cost: [0.0; K_MAX_DRAFT + 1],
            tau_d: ptr::null_mut(),
            lmin_d: ptr::null_mut(),
            top_p_d: ptr::null_mut(),
            top_k_d: ptr::null_mut(),
            conf_d: ptr::null_mut(),
            conf_h: ptr::null_mut(),
            top_d: ptr::null_mut(),
            top_h: ptr::null_mut(),
            lse_d: ptr::null_mut(),
            thr_d: ptr::null_mut(),
            tau_last: -1.0,
            lmin_last: -1.0,
            top_p_last: -1.0,
            top_k_last: -1,
            last_conf: [0.0; 16],
            last_top: [0.0; 16],
            hist_acc: [0; K_MAX_STEPS],
            hist_n: [0; K_MAX_STEPS],
            hist_acc_sum: 0,
            hist_n_sum: 0,
            ema_alpha: 0.5,
            c_stop: 0.35,
            c_cont: 0.60,
            k_cap: 4,
            enabled: false,
            fixed_k: 0,
            steps: 0,
            total_rounds: 0,
            total_drafted: 0,
            total_accepted: 0,
            early_stops: 0,
            allocs: Vec::new(),
            hallocs: Vec::new(),
        }
    }
}

impl DraftLen {
    pub fn init(&mut self, _tk: &KernelsContext) -> Result<()> {
        self.tau_d = hip_alloc_zeroed::<f32>(1, &mut self.allocs)?;
        self.lmin_d = hip_alloc_zeroed::<f32>(1, &mut self.allocs)?;
        self.top_p_d = hip_alloc_zeroed::<f32>(1, &mut self.allocs)?;
        self.top_k_d = hip_alloc_zeroed::<i32>(1, &mut self.allocs)?;
        self.conf_d = hip_alloc_zeroed::<f32>(16, &mut self.allocs)?;
        self.conf_h = hip_host_alloc_zeroed::<f32>(16, &mut self.hallocs)?;
        self.top_d = hip_alloc_zeroed::<f32>(16, &mut self.allocs)?;
        self.top_h = hip_host_alloc_zeroed::<f32>(16, &mut self.hallocs)?;
        self.lse_d = hip_alloc_zeroed::<f32>(1, &mut self.allocs)?;
        self.thr_d = hip_alloc_zeroed::<f32>(1, &mut self.allocs)?;
        self.init_arch("gfx1151", 0);
        self.reset();
        Ok(())
    }

    /// Initialize policy matching `DraftLen::init(const char* gcn, int d_fixed)` (`q36_adraft_policy.cpp:43-90`).
    pub fn init_arch(&mut self, gcn: &str, d_fixed: i32) {
        let env_int = |k: &str, d: i32| -> i32 {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<i32>().ok())
                .unwrap_or(d)
        };
        let env_float = |k: &str, d: f32| -> f32 {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(d)
        };

        self.dfix = d_fixed;
        let is_gfx1151 = gcn.contains("gfx1151");
        if is_gfx1151 {
            self.mode = 2;
            self.dmin = 2;
            self.dmax = 7;
            self.chain = 3;
            self.pmin = [0.5, 0.0];
            self.cum = [0.40, 0.45];
            let c8 = [72.1, 76.9, 82.1, 85.2, 90.6, 95.7, 101.0, 106.5];
            for d in 1..=8 {
                self.cost[d] = c8[d - 1];
            }
        } else {
            self.mode = 0;
            for d in 1..=8 {
                self.cost[d] = 68.0 + 4.0 * (d as f64);
            }
        }
        for d in 9..=K_MAX_DRAFT {
            self.cost[d] = self.cost[8] + 6.0 * ((d - 8) as f64);
        }

        self.mode = env_int("STRATA_ADRAFT", self.mode);
        self.dmin = env_int("STRATA_ADRAFT_DMIN", self.dmin).clamp(1, K_MAX_DRAFT as i32);
        self.dmax = env_int("STRATA_ADRAFT_DMAX", self.dmax).clamp(self.dmin, K_MAX_DRAFT as i32);
        self.chain = env_int("STRATA_ADRAFT_CHAIN", self.chain).clamp(1, self.dmax);
        self.pmin[0] = env_float("STRATA_ADRAFT_PMIN", self.pmin[0]);
        self.pmin[1] = env_float("STRATA_ADRAFT_PMIN_S", self.pmin[1]);
        self.cum[0] = env_float("STRATA_ADRAFT_CUM", self.cum[0]);
        self.cum[1] = env_float("STRATA_ADRAFT_CUM_S", self.cum[1]);
        if d_fixed <= 0 {
            self.mode = 0;
        }
        if self.mode < 0 || self.mode > 2 {
            self.mode = 0;
        }
        for k in 1..=K_MAX_DRAFT {
            self.tries[0][k] = 8.0;
            self.tries[1][k] = 8.0;
            self.keeps[0][k] = 8.0 * 0.80;
            self.keeps[1][k] = 8.0 * 0.70;
        }
        self.dfl_stop = env_int("STRATA_DFL_STOP", if is_gfx1151 { 1 } else { 0 }) != 0;
        self.dfl_sample = env_int("STRATA_DFL_SAMPLE", if is_gfx1151 { 1 } else { 0 }) != 0;
        self.samp = env_int("STRATA_ADRAFT_S", 0) != 0;
        self.dfl_dmin = env_int("STRATA_DFL_DMIN", 2).clamp(0, K_MAX_DRAFT as i32);
        self.dfl_cum[0] = env_float("STRATA_DFL_CUM", 0.40);
        self.dfl_cum[1] = env_float("STRATA_DFL_CUM_S", 0.30);

        self.enabled = self.mode != 0;
        self.fixed_k = if self.mode == 0 { d_fixed } else { 0 };
        self.c_stop = self.cum[0];
        self.c_cont = self.pmin[0];
    }

    #[inline]
    pub fn mode(&self) -> i32 {
        self.mode
    }

    #[inline]
    pub fn want_conf(&self) -> bool {
        self.mode == 2 || std::env::var("STRATA_ADRAFT_TRACE").is_ok()
    }

    #[inline]
    pub fn stop_mode(&self) -> bool {
        self.mode == 2
    }

    #[inline]
    pub fn dmax(&self) -> i32 {
        if self.mode != 0 {
            self.dmax
        } else {
            self.dfix
        }
    }

    #[inline]
    pub fn dmax_for(&self, sampled: bool) -> i32 {
        if self.mode != 0 && (!sampled || self.samp) {
            self.dmax
        } else {
            self.dfix
        }
    }

    #[inline]
    pub fn dfl_sample(&self) -> bool {
        self.dfl_sample
    }

    #[inline]
    pub fn dfl_stop(&self) -> bool {
        self.dfl_stop
    }

    /// Initial chain length before verification (`DraftLen::chain_len` in `q36_adraft_policy.cpp:96-110`).
    pub fn chain_len(&self, sampled: bool) -> i32 {
        if self.mode == 0 || (self.mode == 2 && sampled && !self.samp) {
            return self.dfix;
        }
        if self.mode == 2 {
            return self.chain;
        }
        let s = if sampled { 1 } else { 0 };
        let mut best = -1.0f64;
        let mut e = 1.0f64;
        let mut p = 1.0f64;
        let mut bd = self.dmin;
        for d in 1..=(self.dmax as usize) {
            p *= self.keeps[s][d] / self.tries[s][d].max(1e-9);
            e += p;
            if (d as i32) >= self.dmin && e / self.cost[d] > best {
                best = e / self.cost[d];
                bd = d as i32;
            }
        }
        bd
    }

    /// Stop-mode prefix length to keep (`DraftLen::keep` in `q36_adraft_policy.cpp:112-121`).
    pub fn keep(&self, top: &[f32], n: i32, sampled: bool) -> i32 {
        if sampled && !self.samp {
            return n;
        }
        let s = if sampled { 1 } else { 0 };
        let mut P = 1.0f32;
        let lim = (n.max(0) as usize).min(top.len());
        for i in 0..lim {
            P *= top[i];
            if (i as i32) >= self.dmin && !(top[i] >= self.pmin[s] && P >= self.cum[s]) {
                return i as i32;
            }
        }
        n
    }

    /// DFlash2 stop-mode prefix length (`DraftLen::keep_dfl` in `q36_adraft_policy.cpp:123-130`).
    pub fn keep_dfl(&self, w: &[f32], n: i32, sampled: bool) -> i32 {
        let mut P = 1.0f32;
        let lim = (n.max(0) as usize).min(w.len());
        let thr = self.dfl_cum[if sampled { 1 } else { 0 }];
        for i in 0..lim {
            P *= w[i];
            if (i as i32) >= self.dfl_dmin && !(P >= thr) {
                return i as i32;
            }
        }
        n
    }

    /// Observe round outcome (`DraftLen::observe` in `q36_adraft_policy.cpp:132-151`).
    pub fn observe(
        &mut self,
        k: i32,
        _chained: i32,
        a: i32,
        _conf: &[f32],
        _top: &[f32],
        sampled: bool,
        dflash: bool,
    ) {
        self.end_round(k, a);
        if self.mode != 1 || dflash {
            return;
        }
        let s = if sampled { 1 } else { 0 };
        const K_DECAY: f64 = 0.99;
        let hi = (a + 1).min(k).clamp(0, K_MAX_DRAFT as i32) as usize;
        for i in 1..=hi {
            self.tries[s][i] = self.tries[s][i] * K_DECAY + 1.0;
            self.keeps[s][i] = self.keeps[s][i] * K_DECAY + if (i as i32) <= a { 1.0 } else { 0.0 };
        }
    }

    pub fn describe(&self) -> String {
        if self.mode == 0 {
            format!("draft length fixed ({})", self.dfix)
        } else if self.mode == 1 {
            format!(
                "draft length adaptive (est: {} .. {} by the round-cost curve)",
                self.dmin, self.dmax
            )
        } else if self.samp {
            format!(
                "draft length adaptive (stop: chain {}, {} .. {} drafts, keep while top >= {:.2} / {:.2} and prod top >= {:.2} / {:.2}, greedy / sampled)",
                self.chain,
                self.dmin,
                self.dmax,
                self.pmin[0],
                self.pmin[1],
                self.cum[0],
                self.cum[1]
            )
        } else {
            format!(
                "draft length adaptive for greedy requests (stop: chain {}, {} .. {} drafts, keep while top >= {:.2} and prod top >= {:.2}; sampled: {})",
                self.chain, self.dmin, self.dmax, self.pmin[0], self.cum[0], self.dfix
            )
        }
    }

    pub fn free_buffers(&mut self) {
        for &p in &self.allocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipFree(p);
                }
            }
        }
        self.allocs.clear();
        for &p in &self.hallocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipHostFree(p);
                }
            }
        }
        self.hallocs.clear();
    }

    pub fn configure_from_env(&mut self) {
        let dfix = self.dfix.max(4);
        self.init_arch("gfx1151", dfix);
    }

    pub fn reset(&mut self) {
        self.last_conf = [0.0; 16];
        self.last_top = [0.0; 16];
        self.hist_acc = [0; K_MAX_STEPS];
        self.hist_n = [0; K_MAX_STEPS];
        self.hist_acc_sum = 0;
        self.hist_n_sum = 0;
        self.steps = 0;
    }

    pub fn begin_round(&mut self, max_k: i32, tau: f32, min_p: f32, top_p: f32, top_k: i32) {
        self.k_cap = max_k.clamp(1, K_MAX_DRAFT as i32);
        let t = if tau > 1e-5 { tau } else { 1.0 };
        let lmin = if min_p > 0.0 && min_p < 1.0 {
            min_p.ln()
        } else {
            -1e30
        };
        let tp = if top_p > 0.0 && top_p < 1.0 {
            top_p
        } else {
            1.0
        };
        let tk = if top_k > 0 { top_k } else { 0 };
        if t != self.tau_last && !self.tau_d.is_null() {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.tau_d.cast::<c_void>(),
                    (&t as *const f32).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    ptr::null_mut(),
                );
            }
            self.tau_last = t;
        }
        if lmin != self.lmin_last && !self.lmin_d.is_null() {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.lmin_d.cast::<c_void>(),
                    (&lmin as *const f32).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    ptr::null_mut(),
                );
            }
            self.lmin_last = lmin;
        }
        if tp != self.top_p_last && !self.top_p_d.is_null() {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.top_p_d.cast::<c_void>(),
                    (&tp as *const f32).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    ptr::null_mut(),
                );
            }
            self.top_p_last = tp;
        }
        if tk != self.top_k_last && !self.top_k_d.is_null() {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.top_k_d.cast::<c_void>(),
                    (&tk as *const i32).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    ptr::null_mut(),
                );
            }
            self.top_k_last = tk;
        }
    }

    pub fn step_conf(
        &mut self,
        tk: &KernelsContext,
        logits: *const f32,
        n_vocab: i32,
        pick_d: *const i32,
        map_d: *const i32,
        step: i32,
    ) {
        if step < 0 || step >= 16 {
            return;
        }
        let filtered = self.top_p_last < 1.0 || self.top_k_last > 0;
        if filtered {
            let _ = tk_q36_draft_support(
                tk.as_ptr(),
                logits,
                n_vocab,
                self.tau_d,
                self.lmin_d,
                self.top_p_d,
                self.top_k_d,
                self.lse_d,
                self.thr_d,
                ptr::null_mut(),
            );
            let _ = tk_q36_draft_conf(
                tk.as_ptr(),
                logits,
                n_vocab,
                self.tau_d,
                self.lmin_d,
                pick_d,
                map_d,
                self.conf_d,
                self.top_d,
                step,
                self.lse_d,
                self.thr_d,
                ptr::null_mut(),
            );
        } else {
            let _ = tk_q36_draft_conf(
                tk.as_ptr(),
                logits,
                n_vocab,
                self.tau_d,
                self.lmin_d,
                pick_d,
                map_d,
                self.conf_d,
                self.top_d,
                step,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
            );
        }
    }

    pub fn should_continue(&mut self, step: i32) -> bool {
        if step < 0 || step >= (self.k_cap - 1) {
            return false;
        }
        if !self.enabled {
            return (step + 1) < self.k_cap;
        }
        unsafe {
            let _ = hipMemcpy(
                self.conf_h.add(step as usize).cast::<c_void>(),
                self.conf_d.add(step as usize).cast::<c_void>(),
                4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = hipMemcpy(
                self.top_h.add(step as usize).cast::<c_void>(),
                self.top_d.add(step as usize).cast::<c_void>(),
                4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
        }
        let c = unsafe { *self.conf_h.add(step as usize) };
        let tp = unsafe { *self.top_h.add(step as usize) };
        self.last_conf[step as usize] = c;
        self.last_top[step as usize] = tp;
        let k = self.keep(&self.last_top[..(step as usize + 1)], step + 1, false);
        let cont = k == (step + 1);
        if !cont {
            self.early_stops += 1;
        }
        cont
    }

    pub fn end_round(&mut self, drafted: i32, accepted: i32) {
        self.total_rounds += 1;
        self.total_drafted += drafted as i64;
        self.total_accepted += accepted as i64;
        let idx = (self.steps as usize) % K_MAX_STEPS;
        if self.steps >= K_MAX_STEPS as i32 {
            self.hist_acc_sum -= self.hist_acc[idx];
            self.hist_n_sum -= self.hist_n[idx];
        } else {
            self.steps += 1;
        }
        self.hist_acc[idx] = accepted;
        self.hist_n[idx] = drafted;
        self.hist_acc_sum += accepted;
        self.hist_n_sum += drafted;
    }

    pub fn choose_next_k(&self, max_k: i32) -> i32 {
        let cap = max_k.clamp(1, K_MAX_DRAFT as i32);
        if self.mode != 0 {
            return self.chain_len(false).clamp(1, cap);
        }
        if self.fixed_k > 0 {
            return self.fixed_k.min(cap);
        }
        cap
    }
}

// ============================================================================
// Context-Depth Drafter Policy (`q36_depth_policy.{hpp,cpp}` -> `talus_spec`)
// ============================================================================

pub use talus_spec::{CrossoverTable, DfCtx, DrafterPick, ModelSpecDecl, SpecEngineService};

// ============================================================================
// DFlash2 Block-Diffusion Drafter (`q36_dflash_model.{hpp,cpp}` & `q36_dflash.{hpp,cu}`)
// ============================================================================

pub const K_DFLASH_MAGIC: u32 = 0x4C46_4454; // "TDFL"
pub const K_DFLASH_VERSION: u32 = 1;
pub const K_DFLASH_TAPS: usize = 5;

#[derive(Debug, Default)]
pub struct DFlashLayer {
    pub norm_q: *const f32,
    pub norm_kv: *const f32,
    pub q_norm: *const f32,
    pub k_norm: *const f32,
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
    pub wo: Mat,
    pub conv_base: *const f32,
    pub w_dyn: Mat,
    pub ffn_conv_base: *const f32,
    pub w_ffn_dyn: Mat,
    pub norm_ffn: *const f32,
    pub w_gate: Mat,
    pub w_up: Mat,
    pub w_down: Mat,
}

unsafe impl Send for DFlashLayer {}
unsafe impl Sync for DFlashLayer {}

#[derive(Debug)]
pub struct DFlashModel {
    pub n_layers: i32,
    pub d_model: i32,
    pub n_heads: i32,
    pub n_kv: i32,
    pub head_dim: i32,
    pub d_ff: i32,
    pub window: i32,
    pub block_size: i32,
    pub conv_group: i32,
    pub mask_token_id: i32,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub tap_layers: [i32; K_DFLASH_TAPS],
    pub emb_norm: *const f32,
    pub tap_proj: [Mat; K_DFLASH_TAPS],
    pub tap_norm: *const f32,
    pub layers: Vec<DFlashLayer>,
    pub out_norm: *const f32,
    pub head: Mat,
    pub sel_hidden: Mat,
    pub has_own_head: bool,
    pub has_trans: bool,
    pub is_gguf: bool,
    pub trans_rank: i32,
    pub trans_gate: *const f32,
    pub trans_pred: *const u16,
    pub trans_succ: *const u16,
    allocs: Vec<*mut c_void>,
}

unsafe impl Send for DFlashModel {}
unsafe impl Sync for DFlashModel {}

impl Default for DFlashModel {
    fn default() -> Self {
        Self {
            n_layers: 0,
            d_model: 0,
            n_heads: 0,
            n_kv: 0,
            head_dim: 0,
            d_ff: 0,
            window: 512,
            block_size: 4,
            conv_group: 16,
            mask_token_id: 0,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            tap_layers: [0; K_DFLASH_TAPS],
            emb_norm: ptr::null(),
            tap_proj: [Mat::null(); K_DFLASH_TAPS],
            tap_norm: ptr::null(),
            layers: Vec::new(),
            out_norm: ptr::null(),
            head: Mat::null(),
            sel_hidden: Mat::null(),
            has_own_head: false,
            has_trans: false,
            is_gguf: false,
            trans_rank: 0,
            trans_gate: ptr::null(),
            trans_pred: ptr::null(),
            trans_succ: ptr::null(),
            allocs: Vec::new(),
        }
    }
}

impl DFlashModel {
    pub fn free_buffers(&mut self) {
        for &p in &self.allocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipFree(p);
                }
            }
        }
        self.allocs.clear();
    }

    /// Load a GGUF DFlash2 drafter (`src/qwen35/q36_dflash_model.cpp:74-224`).
    pub fn load_gguf(
        &mut self,
        path: &Path,
        target_n_layer: i32,
        target_d_model: i32,
    ) -> Result<bool> {
        let f = match GgufFile::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(false),
        };
        if f.arch() != Some("dflash") {
            return Ok(false);
        }
        let nl = meta_i64_or(&f, "dflash.block_count", 0) as i32;
        self.n_layers = nl;
        self.d_model = meta_i64_or(&f, "dflash.embedding_length", 0) as i32;
        self.d_ff = meta_i64_or(&f, "dflash.feed_forward_length", 0) as i32;
        self.n_heads = meta_i64_or(&f, "dflash.attention.head_count", 0) as i32;
        self.n_kv = meta_i64_or(&f, "dflash.attention.head_count_kv", 0) as i32;
        self.head_dim = meta_i64_or(&f, "dflash.attention.key_length", 128) as i32;
        self.block_size = meta_i64_or(&f, "dflash.block_size", 8) as i32;
        self.conv_group = meta_i64_or(&f, "dflash.conv_group_size", 16) as i32;
        self.window = meta_i64_or(&f, "dflash.attention.sliding_window", 2048) as i32;
        self.rope_theta = meta_f32_or(&f, "dflash.rope.freq_base", 1e7);
        self.rms_eps = meta_f32_or(&f, "dflash.attention.layer_norm_rms_epsilon", 1e-6);
        self.mask_token_id = meta_i64_or(&f, "tokenizer.ggml.mask_token_id", 248070) as i32;

        let tl = match f.metadata().get("dflash.target_layers") {
            Some(GgufValue::Array(items)) if items.len() == K_DFLASH_TAPS => items,
            _ => return Ok(false),
        };
        if nl < 1
            || self.d_model <= 0
            || (target_d_model > 0 && self.d_model != target_d_model)
            || self.n_heads != 4 * self.n_kv
            || self.head_dim != 128
            || self.block_size != 8
            || meta_i64_or(&f, "dflash.selector_rank", 0) != 256
            || meta_i64_or(&f, "dflash.selector_top_k", 0) != 16
            || meta_i64_or(&f, "dflash.conv_kernel_size", 2) != 2
            || (self.d_model % self.conv_group) != 0
        {
            return Ok(false);
        }
        for (j, item) in tl.iter().enumerate() {
            let t = match item {
                GgufValue::U8(v) => *v as i32,
                GgufValue::I8(v) => *v as i32,
                GgufValue::U16(v) => *v as i32,
                GgufValue::I16(v) => *v as i32,
                GgufValue::U32(v) => *v as i32,
                GgufValue::I32(v) => *v,
                GgufValue::U64(v) => *v as i32,
                GgufValue::I64(v) => *v as i32,
                _ => -1,
            };
            if t < 1 || (target_n_layer > 0 && t > target_n_layer) {
                return Ok(false);
            }
            self.tap_layers[j] = t;
        }

        let load_f32 =
            |name: &str, want: usize, allocs: &mut Vec<*mut c_void>| -> Result<*const f32> {
                let ti = f
                    .tensor(name)
                    .ok_or_else(|| anyhow!("talus-dflash: missing {name}"))?;
                ensure!(
                    ti.ggml_type_u32() == 0 && (ti.n_elements() as usize) == want,
                    "talus-dflash: {name}: want F32 x {want}"
                );
                let bytes = want * 4;
                let src = f.tensor_data(ti)?;
                let d = hip_malloc_bytes(bytes)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            src.as_ptr().cast::<c_void>(),
                            bytes,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(dflash gguf f32)",
                )?;
                allocs.push(d.cast::<c_void>());
                Ok(d.cast::<f32>())
            };

        let load_mat =
            |name: &str, n_in: i32, n_out: i32, allocs: &mut Vec<*mut c_void>| -> Result<Mat> {
                let ti = f
                    .tensor(name)
                    .ok_or_else(|| anyhow!("talus-dflash: missing {name}"))?;
                let dims = ti.dimensions();
                ensure!(
                    dims.len() >= 2 && (dims[0] as i32) == n_in && (dims[1] as i32) == n_out,
                    "talus-dflash: {name}: shape mismatch"
                );
                let src = f.tensor_data(ti)?;
                let nb = src.len();
                let d = hip_malloc_bytes(nb)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            src.as_ptr().cast::<c_void>(),
                            nb,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(dflash gguf mat)",
                )?;
                allocs.push(d.cast::<c_void>());
                Ok(Mat {
                    d,
                    qtype: ti.ggml_type_u32() as i32,
                    n_in,
                    n_out,
                    w_scale: 1.0,
                    ne2: 1,
                    bytes: nb,
                })
            };

        let n = self.d_model;
        {
            let ti = f
                .tensor("fc.weight")
                .ok_or_else(|| anyhow!("talus-dflash: missing fc.weight"))?;
            let dims = ti.dimensions();
            ensure!(
                dims.len() >= 2
                    && (dims[0] as usize) == K_DFLASH_TAPS * (n as usize)
                    && (dims[1] as i32) == n,
                "talus-dflash: fc.weight shape mismatch"
            );
            let qtype = ti.ggml_type_u32();
            let rb = qtype_row_bytes(qtype, (K_DFLASH_TAPS * (n as usize)) as i64).unwrap_or(0);
            let sb = qtype_row_bytes(qtype, n as i64).unwrap_or(0);
            ensure!(
                rb > 0 && sb * K_DFLASH_TAPS == rb,
                "talus-dflash: fc.weight slices not block-aligned"
            );
            let src = f.tensor_data(ti)?;
            let mut sl = vec![0u8; sb * (n as usize)];
            for j in 0..K_DFLASH_TAPS {
                for r in 0..(n as usize) {
                    let src_off = r * rb + j * sb;
                    sl[r * sb..(r + 1) * sb].copy_from_slice(&src[src_off..src_off + sb]);
                }
                let d = hip_malloc_bytes(sl.len())?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            sl.as_ptr().cast::<c_void>(),
                            sl.len(),
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(dflash fc slice)",
                )?;
                self.allocs.push(d.cast::<c_void>());
                self.tap_proj[j] = Mat {
                    d,
                    qtype: qtype as i32,
                    n_in: n,
                    n_out: n,
                    w_scale: 1.0,
                    ne2: 1,
                    bytes: sl.len(),
                };
            }
        }

        self.tap_norm = load_f32("enc.output_norm.weight", n as usize, &mut self.allocs)?;
        self.out_norm = load_f32("output_norm.weight", n as usize, &mut self.allocs)?;
        self.sel_hidden = load_mat("selector_hidden.weight", n, 256, &mut self.allocs)?;
        for &nm in &["selector_predecessor.weight", "selector_successor.weight"] {
            let ti = f
                .tensor(nm)
                .ok_or_else(|| anyhow!("talus-dflash: missing {nm}"))?;
            ensure!(
                ti.ggml_type_u32() == 12 && ti.dimensions().first().copied() == Some(256),
                "talus-dflash: {nm}: want Q4_K [256 x vocab]"
            );
            let src = f.tensor_data(ti)?;
            let d = hip_malloc_bytes(src.len())?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        d.cast::<c_void>(),
                        src.as_ptr().cast::<c_void>(),
                        src.len(),
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(dflash selector)",
            )?;
            self.allocs.push(d.cast::<c_void>());
            if nm == "selector_predecessor.weight" {
                self.trans_pred = d.cast::<u16>();
            } else {
                self.trans_succ = d.cast::<u16>();
            }
        }
        self.has_trans = true;
        self.trans_rank = 256;
        self.is_gguf = true;

        let qd = self.n_heads * self.head_dim;
        let kvd = self.n_kv * self.head_dim;
        let ng = n / self.conv_group;
        self.layers.resize_with(nl as usize, DFlashLayer::default);
        for l in 0..(nl as usize) {
            let p = format!("blk.{l}.");
            let L = &mut self.layers[l];
            L.norm_q = load_f32(
                &format!("{p}attn_norm.weight"),
                n as usize,
                &mut self.allocs,
            )?;
            L.norm_ffn = load_f32(&format!("{p}ffn_norm.weight"), n as usize, &mut self.allocs)?;
            L.q_norm = load_f32(
                &format!("{p}attn_q_norm.weight"),
                self.head_dim as usize,
                &mut self.allocs,
            )?;
            L.k_norm = load_f32(
                &format!("{p}attn_k_norm.weight"),
                self.head_dim as usize,
                &mut self.allocs,
            )?;
            L.conv_base = load_f32(
                &format!("{p}attn_conv_base"),
                (n as usize) * 4,
                &mut self.allocs,
            )?;
            L.ffn_conv_base = load_f32(
                &format!("{p}ffn_conv_base"),
                (n as usize) * 4,
                &mut self.allocs,
            )?;
            L.wq = load_mat(&format!("{p}attn_q.weight"), n, qd, &mut self.allocs)?;
            L.wk = load_mat(&format!("{p}attn_k.weight"), n, kvd, &mut self.allocs)?;
            L.wv = load_mat(&format!("{p}attn_v.weight"), n, kvd, &mut self.allocs)?;
            L.wo = load_mat(&format!("{p}attn_output.weight"), qd, n, &mut self.allocs)?;
            L.w_gate = load_mat(
                &format!("{p}ffn_gate.weight"),
                n,
                self.d_ff,
                &mut self.allocs,
            )?;
            L.w_up = load_mat(&format!("{p}ffn_up.weight"), n, self.d_ff, &mut self.allocs)?;
            L.w_down = load_mat(
                &format!("{p}ffn_down.weight"),
                self.d_ff,
                n,
                &mut self.allocs,
            )?;
            L.w_dyn = load_mat(
                &format!("{p}attn_conv_proj.weight"),
                n,
                4 * ng,
                &mut self.allocs,
            )?;
            L.w_ffn_dyn = load_mat(
                &format!("{p}ffn_conv_proj.weight"),
                n,
                4 * ng,
                &mut self.allocs,
            )?;
        }
        Ok(true)
    }

    pub fn load(&mut self, path: &str, target_n_layer: i32, target_d_model: i32) -> Result<bool> {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(false),
        };
        let mut hdr = [0u8; 64];
        if f.read_exact(&mut hdr).is_err() {
            return Ok(false);
        }
        if &hdr[0..4] == b"GGUF" {
            drop(f);
            return self.load_gguf(Path::new(path), target_n_layer, target_d_model);
        }
        let r_u32 = |off: usize| -> u32 {
            u32::from_le_bytes([hdr[off], hdr[off + 1], hdr[off + 2], hdr[off + 3]])
        };
        let r_i32 = |off: usize| -> i32 { r_u32(off) as i32 };
        let r_f32 = |off: usize| -> f32 { f32::from_bits(r_u32(off)) };

        let magic = r_u32(0);
        let version = r_u32(4);
        if magic != K_DFLASH_MAGIC || version != K_DFLASH_VERSION {
            return Ok(false);
        }

        self.n_layers = r_i32(8);
        self.d_model = r_i32(12);
        self.n_heads = r_i32(16);
        self.n_kv = r_i32(20);
        self.head_dim = r_i32(24);
        self.d_ff = r_i32(28);
        self.window = r_i32(32);
        self.block_size = r_i32(36);
        self.mask_token_id = r_i32(40);
        self.rope_theta = r_f32(44);
        self.rms_eps = r_f32(48);
        let flags = r_u32(52);
        let n_vocab_head = r_i32(56);

        if self.d_model != target_d_model
            || self.n_layers < 1
            || self.n_layers > 8
            || (self.n_heads * self.head_dim) != self.d_model
            || (self.n_heads % self.n_kv) != 0
        {
            return Ok(false);
        }

        let mut taps_buf = [0u8; K_DFLASH_TAPS * 4];
        f.read_exact(&mut taps_buf).map_err(io_err)?;
        for i in 0..K_DFLASH_TAPS {
            let t = i32::from_le_bytes([
                taps_buf[i * 4],
                taps_buf[i * 4 + 1],
                taps_buf[i * 4 + 2],
                taps_buf[i * 4 + 3],
            ]);
            if t < 0 || t >= target_n_layer {
                return Ok(false);
            }
            self.tap_layers[i] = t;
        }

        let read_f32_vec =
            |f: &mut File, n: usize, allocs: &mut Vec<*mut c_void>| -> Result<*const f32> {
                let bytes = n * 4;
                let mut host = vec![0u8; bytes];
                f.read_exact(&mut host).map_err(io_err)?;
                let d = hip_malloc_bytes(bytes)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            host.as_ptr().cast::<c_void>(),
                            bytes,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(dflash f32)",
                )?;
                allocs.push(d.cast::<c_void>());
                Ok(d.cast::<f32>())
            };

        let read_q8_mat =
            |f: &mut File, n_in: i32, n_out: i32, allocs: &mut Vec<*mut c_void>| -> Result<Mat> {
                ensure!(n_in % 32 == 0, "DFlash Q8_0 n_in must be multiple of 32");
                let nb = (n_out as usize) * ((n_in as usize) / 32) * 34;
                let mut host = vec![0u8; nb];
                f.read_exact(&mut host).map_err(io_err)?;
                let d = hip_malloc_bytes(nb)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            host.as_ptr().cast::<c_void>(),
                            nb,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(dflash q8)",
                )?;
                allocs.push(d.cast::<c_void>());
                Ok(Mat {
                    d,
                    qtype: GGML_TYPE_Q8_0 as i32,
                    n_in,
                    n_out,
                    w_scale: 1.0,
                    ne2: 1,
                    bytes: nb,
                })
            };

        let E = self.d_model;
        let KV = self.n_kv * self.head_dim;
        let F = self.d_ff;

        self.emb_norm = read_f32_vec(&mut f, E as usize, &mut self.allocs)?;
        for i in 0..K_DFLASH_TAPS {
            self.tap_proj[i] = read_q8_mat(&mut f, E, E, &mut self.allocs)?;
        }
        self.tap_norm = read_f32_vec(&mut f, E as usize, &mut self.allocs)?;

        self.layers
            .resize_with(self.n_layers as usize, DFlashLayer::default);
        for l in 0..(self.n_layers as usize) {
            let L = &mut self.layers[l];
            L.norm_q = read_f32_vec(&mut f, E as usize, &mut self.allocs)?;
            L.norm_kv = read_f32_vec(&mut f, E as usize, &mut self.allocs)?;
            L.q_norm = read_f32_vec(&mut f, self.head_dim as usize, &mut self.allocs)?;
            L.k_norm = read_f32_vec(&mut f, self.head_dim as usize, &mut self.allocs)?;
            L.wq = read_q8_mat(&mut f, E, E, &mut self.allocs)?;
            L.wk = read_q8_mat(&mut f, E, KV, &mut self.allocs)?;
            L.wv = read_q8_mat(&mut f, E, KV, &mut self.allocs)?;
            L.wo = read_q8_mat(&mut f, E, E, &mut self.allocs)?;
            L.conv_base = read_f32_vec(&mut f, (E as usize) * 5, &mut self.allocs)?;
            L.w_dyn = read_q8_mat(&mut f, E, (E / 64) * 5, &mut self.allocs)?;
            L.norm_ffn = read_f32_vec(&mut f, E as usize, &mut self.allocs)?;
            L.w_gate = read_q8_mat(&mut f, E, F, &mut self.allocs)?;
            L.w_up = read_q8_mat(&mut f, E, F, &mut self.allocs)?;
            L.w_down = read_q8_mat(&mut f, F, E, &mut self.allocs)?;
        }

        self.out_norm = read_f32_vec(&mut f, E as usize, &mut self.allocs)?;
        self.has_own_head = (flags & 1) != 0 && n_vocab_head > 0;
        if self.has_own_head {
            self.head = read_q8_mat(&mut f, E, n_vocab_head, &mut self.allocs)?;
        }
        self.has_trans = (flags & 2) != 0 && n_vocab_head > 0;
        if self.has_trans {
            self.trans_rank = 64;
            let nv = n_vocab_head as usize;
            let r = self.trans_rank as usize;
            self.trans_gate = read_f32_vec(&mut f, nv, &mut self.allocs)?;
            let bytes = nv * r * 2;
            let mut host = vec![0u8; bytes];
            f.read_exact(&mut host).map_err(io_err)?;
            let dp = hip_malloc_bytes(bytes)?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        dp.cast::<c_void>(),
                        host.as_ptr().cast::<c_void>(),
                        bytes,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(trans_pred)",
            )?;
            self.allocs.push(dp.cast::<c_void>());
            self.trans_pred = dp.cast::<u16>();

            f.read_exact(&mut host).map_err(io_err)?;
            let ds = hip_malloc_bytes(bytes)?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        ds.cast::<c_void>(),
                        host.as_ptr().cast::<c_void>(),
                        bytes,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(trans_succ)",
            )?;
            self.allocs.push(ds.cast::<c_void>());
            self.trans_succ = ds.cast::<u16>();
        }
        Ok(true)
    }
}

pub const K_DFLASH_MAX_B: usize = 8;

#[derive(Debug, Default)]
pub struct DFlash {
    pub mdl: DFlashModel,
    pub ready: bool,
    pub max_rows: i32,
    pub k_ring: Vec<*mut f32>,
    pub v_ring: Vec<*mut f32>,
    pub ring_len: i32,
    pub tap_buf: [*mut f32; K_DFLASH_TAPS],
    pub tap_slot: Vec<i32>,
    pub tap_rows: i32,
    pub ctx_vec: *mut f32,
    pub ctx_part: [*mut f32; K_DFLASH_TAPS],
    pub x: *mut f32,
    pub h_q: *mut f32,
    pub h_kv: *mut f32,
    pub q: *mut f32,
    pub k: *mut f32,
    pub v: *mut f32,
    pub att: *mut f32,
    pub att_part: *mut f32,
    pub dyn_w: *mut f32,
    pub dyn_f: *mut f32,
    pub conv_out: *mut f32,
    pub g: *mut f32,
    pub u: *mut f32,
    pub s_buf: *mut f32,
    pub ffn_out: *mut f32,
    pub x_q8: *mut c_void,
    pub logits: *mut f32,
    pub block_tok: *mut i32,
    pub pos_d: *mut i32,
    pub cand_id: *mut i32,
    pub cand_u: *mut f32,
    pub chain_sc: *mut f32,
    pub q_prop: *mut f32,
    pub q_prop_h: *mut f32,
    pub hcand: *mut i32,
    pub tau_d: *mut f32,
    pub seed_d: *mut u32,
    pub ctr_d: *mut u32,
    pub tau_last: f32,
    pub seed_last: u32,
    pub ctr_host: u32,
    pub n_vocab: i32,
    pub shared_head: Mat,
    pub draft_map: *const i32,
    tk_ptr: *mut talus_kernels_sys::tk_ctx,
    allocs: Vec<*mut c_void>,
    hallocs: Vec<*mut c_void>,
}

unsafe impl Send for DFlash {}
unsafe impl Sync for DFlash {}

impl DFlash {
    #[inline]
    pub fn block(&self) -> i32 {
        self.mdl.block_size
    }

    #[inline]
    pub fn n_draft(&self) -> i32 {
        self.mdl.block_size - 1
    }

    #[inline]
    pub fn keep(&self) -> i32 {
        self.mdl.window + 2 * self.mdl.block_size
    }

    #[inline]
    pub fn window(&self) -> i32 {
        self.mdl.window
    }

    #[inline]
    pub fn ring(&self) -> i32 {
        self.ring_len
    }

    /// Bytes to stash `rows` context rows across all DFlash2 layers (K and V).
    #[inline]
    pub fn ctx_bytes(&self, rows: i32) -> usize {
        2 * self.mdl.layers.len()
            * (rows.max(0) as usize)
            * (self.mdl.n_kv as usize)
            * (self.mdl.head_dim as usize)
            * 4
    }

    /// Copy `rows` context cells starting at `p0` between the ring and `buf` (`DFlash::ctx_copy` in `q36_dflash.cu:505-522`).
    pub fn ctx_copy(
        &self,
        buf: *mut u8,
        p0: i64,
        rows: i32,
        stride: i32,
        save: bool,
        stream: hipStream_t,
    ) {
        if !self.ready || rows <= 0 || stride < rows || self.ring_len <= 0 {
            return;
        }
        let kv_dim = (self.mdl.n_kv as usize) * (self.mdl.head_dim as usize);
        let rb = kv_dim * 4;
        let ring = self.ring_len as i64;
        let c0 = ((p0 % ring) + ring) % ring;
        let n1 = (rows as i64).min(ring - c0) as usize;
        let n2 = (rows as usize) - n1;
        let mut o = 0usize;
        for l in 0..self.mdl.layers.len() {
            for r in [&self.k_ring[l], &self.v_ring[l]] {
                let ring_ptr = (*r).cast::<u8>();
                unsafe {
                    let a = ring_ptr.add((c0 as usize) * rb);
                    let b = buf.add(o);
                    let _ = hipMemcpyAsync(
                        if save { b } else { a }.cast::<c_void>(),
                        if save { a } else { b }.cast::<c_void>(),
                        n1 * rb,
                        HIP_MEMCPY_DEVICE_TO_DEVICE,
                        stream,
                    );
                    if n2 > 0 {
                        let b2 = buf.add(o + n1 * rb);
                        let _ = hipMemcpyAsync(
                            if save { b2 } else { ring_ptr }.cast::<c_void>(),
                            if save { ring_ptr } else { b2 }.cast::<c_void>(),
                            n2 * rb,
                            HIP_MEMCPY_DEVICE_TO_DEVICE,
                            stream,
                        );
                    }
                }
                o += (stride as usize) * rb;
            }
        }
    }

    #[inline]
    pub fn scores(&self) -> *const f32 {
        self.chain_sc
    }

    pub fn init(
        &mut self,
        tk: &KernelsContext,
        path: &str,
        target_n_layer: i32,
        target_d_model: i32,
        target_n_vocab: i32,
        fallback_head: &Mat,
        dmap: *const i32,
        max_batch: i32,
    ) -> Result<bool> {
        self.tk_ptr = tk.as_ptr();
        if !self.mdl.load(path, target_n_layer, target_d_model)? {
            return Ok(false);
        }
        let E = self.mdl.d_model as usize;
        let KV = (self.mdl.n_kv * self.mdl.head_dim) as usize;
        let F = self.mdl.d_ff as usize;
        let B = (self.mdl.block_size as usize).max(K_DFLASH_MAX_B);

        // Ring length: power of 2 >= window + 2 * block_size + 64 (`q36_dflash_model.cpp:76-77`)
        let mut ring = 1i32;
        let want_ring = self.mdl.window + 2 * self.mdl.block_size + 64;
        while ring < want_ring {
            ring *= 2;
        }
        self.ring_len = ring;
        self.max_rows = max_batch.max(512).max(self.mdl.block_size);
        let R = self.max_rows as usize;

        self.k_ring
            .resize(self.mdl.n_layers as usize, ptr::null_mut());
        self.v_ring
            .resize(self.mdl.n_layers as usize, ptr::null_mut());
        for l in 0..(self.mdl.n_layers as usize) {
            self.k_ring[l] =
                hip_alloc_zeroed::<f32>((self.ring_len as usize) * KV, &mut self.allocs)?;
            self.v_ring[l] =
                hip_alloc_zeroed::<f32>((self.ring_len as usize) * KV, &mut self.allocs)?;
        }

        self.tap_slot = vec![-1; (target_n_layer as usize) + 2];
        for i in 0..K_DFLASH_TAPS {
            self.tap_buf[i] = hip_alloc_zeroed::<f32>(R * E, &mut self.allocs)?;
            let tl = self.mdl.tap_layers[i] as usize;
            if tl < self.tap_slot.len() {
                self.tap_slot[tl] = i as i32;
            }
            self.ctx_part[i] = hip_alloc_zeroed::<f32>(R * E, &mut self.allocs)?;
        }
        self.ctx_vec = hip_alloc_zeroed::<f32>(R * E, &mut self.allocs)?;
        self.x = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.h_q = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.h_kv = hip_alloc_zeroed::<f32>((B + 16) * E, &mut self.allocs)?;
        self.q = hip_alloc_zeroed::<f32>(
            B * (self.mdl.n_heads as usize) * (self.mdl.head_dim as usize),
            &mut self.allocs,
        )?;
        self.k = hip_alloc_zeroed::<f32>(R * KV, &mut self.allocs)?;
        self.v = hip_alloc_zeroed::<f32>(R * KV, &mut self.allocs)?;
        self.att = hip_alloc_zeroed::<f32>(
            B * (self.mdl.n_heads as usize) * (self.mdl.head_dim as usize),
            &mut self.allocs,
        )?;
        let scr_floats =
            tk_q36_dflash_attn_scratch_floats(self.mdl.n_heads, self.mdl.head_dim, self.mdl.window);
        self.att_part = hip_alloc_zeroed::<f32>(scr_floats.max(16), &mut self.allocs)?;
        let ng = (E / (self.mdl.conv_group.max(1) as usize)).max(E / 64);
        self.dyn_w = hip_alloc_zeroed::<f32>(B * ng * 5, &mut self.allocs)?;
        self.dyn_f = hip_alloc_zeroed::<f32>(B * ng * 5, &mut self.allocs)?;
        self.conv_out = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.g = hip_alloc_zeroed::<f32>(B * F, &mut self.allocs)?;
        self.u = hip_alloc_zeroed::<f32>(B * F, &mut self.allocs)?;
        self.s_buf = hip_alloc_zeroed::<f32>(B * F, &mut self.allocs)?;
        self.ffn_out = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;

        let q8_b = unsafe { tk_q8_1_bytes(E.max(F) as i32, R.max(B + 16) as i32) };
        self.x_q8 = hip_alloc_zeroed::<u8>(q8_b.max(16), &mut self.allocs)?.cast::<c_void>();

        self.shared_head = if self.mdl.has_own_head {
            self.mdl.head
        } else {
            *fallback_head
        };
        self.n_vocab = if self.mdl.has_own_head {
            self.mdl.head.n_out
        } else if !fallback_head.is_null() {
            fallback_head.n_out
        } else {
            target_n_vocab
        };
        if self.mdl.has_own_head {
            self.draft_map = ptr::null();
        } else if !dmap.is_null() {
            self.draft_map = dmap;
        } else {
            let ids: Vec<i32> = (0..self.n_vocab).collect();
            let dm = hip_alloc_zeroed::<i32>(ids.len().max(1), &mut self.allocs)?;
            if !ids.is_empty() {
                hip_check(
                    unsafe {
                        hipMemcpy(
                            dm.cast::<c_void>(),
                            ids.as_ptr().cast::<c_void>(),
                            ids.len() * 4,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(dflash identity dv_map)",
                )?;
            }
            self.draft_map = dm;
        }
        self.logits = hip_alloc_zeroed::<f32>(B * (self.n_vocab as usize), &mut self.allocs)?;
        self.block_tok = hip_alloc_zeroed::<i32>(B, &mut self.allocs)?;
        self.pos_d = hip_alloc_zeroed::<i32>(1, &mut self.allocs)?;
        self.cand_id = hip_alloc_zeroed::<i32>(B * 16, &mut self.allocs)?;
        self.cand_u = hip_alloc_zeroed::<f32>(B * 16, &mut self.allocs)?;
        self.chain_sc = hip_alloc_zeroed::<f32>(16, &mut self.allocs)?;
        self.q_prop = hip_alloc_zeroed::<f32>(B * 16, &mut self.allocs)?;
        self.q_prop_h = hip_host_alloc_zeroed::<f32>(B * 16, &mut self.hallocs)?;
        self.hcand = hip_host_alloc_zeroed::<i32>(B * 16, &mut self.hallocs)?;
        if self.mdl.is_gguf && self.mdl.trans_gate.is_null() {
            let gate_buf = hip_alloc_zeroed::<f32>(B * 256, &mut self.allocs)?;
            self.mdl.trans_gate = gate_buf;
        }
        self.tau_d = hip_alloc_zeroed::<f32>(1, &mut self.allocs)?;
        self.seed_d = hip_alloc_zeroed::<u32>(1, &mut self.allocs)?;
        self.ctr_d = hip_alloc_zeroed::<u32>(1, &mut self.allocs)?;

        self.ready = true;
        Ok(true)
    }

    /// Slice `target_head` (`output.weight` from the target model) using `dv_ids` when non-empty,
    /// matching `DFlash::load` (`q36_dflash_model.cpp:190-209`).
    pub fn configure_target_head(&mut self, target_head: &Mat, dv_ids: &[i32]) -> Result<()> {
        if !self.ready || self.mdl.has_own_head || target_head.is_null() {
            return Ok(());
        }
        let B = (self.mdl.block_size as usize).max(K_DFLASH_MAX_B);
        if !dv_ids.is_empty() {
            let rb = target_head.row_bytes();
            ensure!(rb > 0, "target_head row_bytes must be > 0");
            let total_bytes = rb * dv_ids.len();
            let d = hip_malloc_bytes(total_bytes + 4096)?;
            self.allocs.push(d.cast::<c_void>());
            for (i, &id) in dv_ids.iter().enumerate() {
                if id >= 0 && (id as i32) < target_head.n_out {
                    hip_check(
                        unsafe {
                            hipMemcpy(
                                d.add(i * rb).cast::<c_void>(),
                                target_head.d.add((id as usize) * rb).cast::<c_void>(),
                                rb,
                                HIP_MEMCPY_DEVICE_TO_DEVICE,
                            )
                        },
                        "hipMemcpy(dflash sliced target head row)",
                    )?;
                }
            }
            self.shared_head = Mat {
                d,
                qtype: target_head.qtype,
                n_in: target_head.n_in,
                n_out: dv_ids.len() as i32,
                w_scale: target_head.w_scale,
                ne2: 1,
                bytes: total_bytes,
            };
            let dm = hip_alloc_zeroed::<i32>(dv_ids.len(), &mut self.allocs)?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        dm.cast::<c_void>(),
                        dv_ids.as_ptr().cast::<c_void>(),
                        dv_ids.len() * 4,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(dflash dv_map)",
            )?;
            self.draft_map = dm;
            if self.n_vocab != dv_ids.len() as i32 {
                self.n_vocab = dv_ids.len() as i32;
                self.logits =
                    hip_alloc_zeroed::<f32>(B * (self.n_vocab as usize), &mut self.allocs)?;
            }
        } else {
            self.shared_head = *target_head;
            let ids: Vec<i32> = (0..target_head.n_out).collect();
            let dm = hip_alloc_zeroed::<i32>(ids.len().max(1), &mut self.allocs)?;
            if !ids.is_empty() {
                hip_check(
                    unsafe {
                        hipMemcpy(
                            dm.cast::<c_void>(),
                            ids.as_ptr().cast::<c_void>(),
                            ids.len() * 4,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(dflash identity dv_map)",
                )?;
            }
            self.draft_map = dm;
            if self.n_vocab != target_head.n_out {
                self.n_vocab = target_head.n_out;
                self.logits =
                    hip_alloc_zeroed::<f32>(B * (self.n_vocab as usize), &mut self.allocs)?;
            }
        }
        Ok(())
    }

    pub fn free_buffers(&mut self) {
        self.mdl.free_buffers();
        for &p in &self.allocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipFree(p);
                }
            }
        }
        self.allocs.clear();
        for &p in &self.hallocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipHostFree(p);
                }
            }
        }
        self.hallocs.clear();
        self.ready = false;
    }

    pub fn reset(&mut self) {
        if !self.ready {
            return;
        }
        let KV = (self.mdl.n_kv * self.mdl.head_dim) as usize;
        for l in 0..(self.mdl.n_layers as usize) {
            unsafe {
                let _ = hipMemset(
                    self.k_ring[l].cast::<c_void>(),
                    0,
                    (self.ring_len as usize) * KV * 4,
                );
                let _ = hipMemset(
                    self.v_ring[l].cast::<c_void>(),
                    0,
                    (self.ring_len as usize) * KV * 4,
                );
            }
        }
        self.tap_rows = 0;
    }

    #[inline]
    pub fn reset_taps(&mut self) {
        self.tap_rows = 0;
    }

    /// Record target layer `next` (`1..=n_layer`) hidden state (`DFlash::tap` in `q36_dflash_model.cpp:226-230`).
    pub fn tap(&mut self, next: i32, x_layer: *const f32, T: i32, stream: hipStream_t) {
        if !self.ready || T <= 0 {
            return;
        }
        let E = self.mdl.d_model as usize;
        let rows = (T as usize).min(self.max_rows as usize);
        for j in 0..K_DFLASH_TAPS {
            let match_tap = if self.mdl.is_gguf {
                self.mdl.tap_layers[j] == next
            } else {
                self.mdl.tap_layers[j] == next || (self.mdl.tap_layers[j] + 1) == next
            };
            if match_tap {
                unsafe {
                    let _ = hipMemcpyAsync(
                        self.tap_buf[j].cast::<c_void>(),
                        x_layer.cast::<c_void>(),
                        rows * E * 4,
                        HIP_MEMCPY_DEVICE_TO_DEVICE,
                        stream,
                    );
                }
                self.tap_rows = rows as i32;
            }
        }
    }

    pub fn capture_tap(&mut self, li: i32, x_layer: *const f32, n_tok: i32) {
        let next = if self.mdl.is_gguf { li + 1 } else { li };
        self.tap(next, x_layer, n_tok, ptr::null_mut());
    }

    fn apply_mm_preq8(&self, w: &Mat, out: *mut f32, rows: i32, stream: hipStream_t) {
        if rows <= 0 || w.is_null() {
            return;
        }
        let s = stream.cast::<c_void>();
        unsafe {
            let col_bytes = tk_q8_1_bytes(w.n_in, 1);
            let cmax = if tk_mmvq_w32_active() != 0 { 16 } else { 8 };
            let mut t0 = 0i32;
            while t0 < rows {
                let chunk = (rows - t0).min(cmax);
                let chunk_x = self
                    .x_q8
                    .cast::<u8>()
                    .add((t0 as usize) * col_bytes)
                    .cast::<c_void>();
                let chunk_y = out.add((t0 as usize) * (w.n_out as usize));
                let _ = tk_mmvq(
                    self.tk_ptr,
                    w.tk_qtype(),
                    w.d.cast::<c_void>(),
                    chunk_x,
                    chunk_y,
                    w.n_in,
                    w.n_out,
                    chunk,
                    s,
                );
                t0 += chunk;
            }
        }
    }

    fn apply_mm(&self, w: &Mat, inp: *const f32, out: *mut f32, rows: i32, stream: hipStream_t) {
        if rows <= 0 || w.is_null() {
            return;
        }
        let s = stream.cast::<c_void>();
        unsafe {
            let _ = tk_quantize_q8_1(self.tk_ptr, inp, self.x_q8, w.n_in, rows, s);
        }
        self.apply_mm_preq8(w, out, rows, stream);
    }

    #[inline]
    fn apply_q8(&self, w: &Mat, inp: *const f32, out: *mut f32, rows: i32) {
        self.apply_mm(w, inp, out, rows, ptr::null_mut());
    }

    #[inline]
    fn apply_q8_pre(&self, w: &Mat, out: *mut f32, rows: i32) {
        self.apply_mm_preq8(w, out, rows, ptr::null_mut());
    }

    /// Project tapped target rows `r0..T` and store into the DFlash KV ring (`DFlash::inject_rows` in `q36_dflash_model.cpp:232-246`).
    pub fn inject_rows(&mut self, T: i32, pos_d: *const i32, r0: i32, stream: hipStream_t) {
        if !self.ready {
            return;
        }
        let n = T - r0;
        if n <= 0 {
            return;
        }
        let E = self.mdl.d_model;
        let HKV = self.mdl.n_kv;
        let HD = self.mdl.head_dim;
        let KV = HKV * HD;
        let eps = self.mdl.rms_eps;
        let s = stream.cast::<c_void>();

        for j in 0..K_DFLASH_TAPS {
            let src = unsafe { self.tap_buf[j].add((r0 as usize) * (E as usize)) };
            self.apply_mm(&self.mdl.tap_proj[j], src, self.ctx_part[j], n, stream);
        }
        let ys: [*const f32; K_DFLASH_TAPS] = [
            self.ctx_part[0],
            self.ctx_part[1],
            self.ctx_part[2],
            self.ctx_part[3],
            self.ctx_part[4],
        ];
        let _ = tk_q36_dflash_sum5_rms_q8(
            self.tk_ptr,
            ys.as_ptr(),
            self.mdl.tap_norm,
            self.ctx_vec,
            self.x_q8,
            E,
            n,
            eps,
            s,
        );
        for l in 0..(self.mdl.n_layers as usize) {
            let L = &self.mdl.layers[l];
            self.apply_mm_preq8(&L.wk, self.k, n, stream);
            self.apply_mm_preq8(&L.wv, self.v, n, stream);
            let _ = tk_q36_dflash_qk_norm_rope(
                self.tk_ptr,
                self.k,
                L.k_norm,
                HKV,
                HD,
                n,
                pos_d,
                self.mdl.rope_theta,
                eps,
                s,
            );
            let _ = tk_q36_dflash_kv_store(
                self.tk_ptr,
                self.k,
                self.v,
                self.k_ring[l],
                self.v_ring[l],
                KV,
                n,
                pos_d,
                self.ring_len,
                s,
            );
        }
    }

    /// Inject `T` tapped rows starting at `*pos_d` (`DFlash::inject` in `q36_dflash_model.cpp:248`).
    pub fn inject(&mut self, T: i32, pos_d: *const i32, stream: hipStream_t) {
        self.inject_rows(T, pos_d, 0, stream);
    }

    /// Inject tapped rows `r0..T` starting at host position `pos0 + r0` (`DFlash::inject_at` in `q36_dflash_model.cpp:250-256`).
    pub fn inject_at(&mut self, T: i32, pos0: i64, r0: i32, stream: hipStream_t) {
        if !self.ready {
            return;
        }
        let step = 1.max(self.max_rows.min(self.ring_len - 2 * self.mdl.block_size));
        let mut r = r0;
        while r < T {
            let e = T.min(r + step);
            let _ = tk_q36_dflash_set_i32(
                self.tk_ptr,
                self.pos_d,
                (pos0 + (r as i64)) as i32,
                stream.cast::<c_void>(),
            );
            self.inject_rows(e, self.pos_d, r, stream);
            r += step;
        }
    }

    /// Run DFlash2 block-diffusion drafter (`DFlash::draft` in `q36_dflash_model.cpp:258-291`).
    #[allow(clippy::too_many_arguments)]
    pub fn draft(
        &mut self,
        embd_table: *const u8,
        embd_type: i32,
        embd_row: usize,
        anchor_d: *const i32,
        pos_d: *const i32,
        out_d: *mut i32,
        samp_tau_d: *const f32,
        samp_seed_d: *const u32,
        samp_ctr_d: *mut u32,
        stream: hipStream_t,
    ) {
        if !self.ready {
            return;
        }
        let B = self.mdl.block_size;
        let E = self.mdl.d_model;
        let H = self.mdl.n_heads;
        let HKV = self.mdl.n_kv;
        let HD = self.mdl.head_dim;
        let KV = HKV * HD;
        let F = self.mdl.d_ff;
        let eps = self.mdl.rms_eps;
        let sc = 1.0f32 / (HD as f32).sqrt();
        let s = stream.cast::<c_void>();

        let _ = tk_q36_dflash_block_tokens(
            self.tk_ptr,
            anchor_d,
            self.block_tok,
            B,
            self.mdl.mask_token_id,
            s,
        );
        let _ = tk_q36_embed_rows(
            self.tk_ptr,
            embd_type,
            embd_table.cast::<c_void>(),
            embd_row,
            self.block_tok,
            B as i64,
            E as i64,
            self.x,
            s,
        );

        for l in 0..(self.mdl.n_layers as usize) {
            let L = &self.mdl.layers[l];
            let _ = tk_q36_dflash_rms_rows_q8(
                self.tk_ptr,
                self.x,
                L.norm_q,
                self.h_q,
                self.x_q8,
                E,
                B,
                eps,
                s,
            );
            self.apply_mm_preq8(&L.w_dyn, self.dyn_w, B, stream);
            let _ = tk_q36_dflash_dyn_conv_q8(
                self.tk_ptr,
                self.h_q,
                self.dyn_w,
                L.conv_base,
                self.conv_out,
                self.x_q8,
                E,
                B,
                self.mdl.conv_group,
                0,
                s,
            );
            self.apply_mm_preq8(&L.wq, self.q, B, stream);
            self.apply_mm_preq8(&L.wk, self.k, B, stream);
            self.apply_mm_preq8(&L.wv, self.v, B, stream);
            let _ = tk_q36_dflash_qk_norm_rope(
                self.tk_ptr,
                self.q,
                L.q_norm,
                H,
                HD,
                B,
                pos_d,
                self.mdl.rope_theta,
                eps,
                s,
            );
            let _ = tk_q36_dflash_qk_norm_rope(
                self.tk_ptr,
                self.k,
                L.k_norm,
                HKV,
                HD,
                B,
                pos_d,
                self.mdl.rope_theta,
                eps,
                s,
            );
            let _ = tk_q36_dflash_kv_store(
                self.tk_ptr,
                self.k,
                self.v,
                self.k_ring[l],
                self.v_ring[l],
                KV,
                B,
                pos_d,
                self.ring_len,
                s,
            );
            let _ = tk_q36_dflash_block_attn(
                self.tk_ptr,
                self.q,
                self.k_ring[l],
                self.v_ring[l],
                pos_d,
                H,
                HKV,
                HD,
                self.ring_len,
                self.mdl.window,
                sc,
                self.att_part,
                self.att,
                s,
            );
            self.apply_mm(&L.wo, self.att, self.ffn_out, B, stream);
            let _ = tk_q36_dflash_dyn_conv(
                self.tk_ptr,
                self.ffn_out,
                self.dyn_w,
                L.conv_base,
                self.conv_out,
                E,
                B,
                self.mdl.conv_group,
                1,
                s,
            );
            let _ = tk_q36_dflash_add_inplace(
                self.tk_ptr,
                self.x,
                self.conv_out,
                (B as i64) * (E as i64),
                s,
            );

            let _ = tk_q36_dflash_rms_rows_q8(
                self.tk_ptr,
                self.x,
                L.norm_ffn,
                self.h_q,
                self.x_q8,
                E,
                B,
                eps,
                s,
            );
            self.apply_mm_preq8(&L.w_ffn_dyn, self.dyn_f, B, stream);
            let _ = tk_q36_dflash_dyn_conv_q8(
                self.tk_ptr,
                self.h_q,
                self.dyn_f,
                L.ffn_conv_base,
                self.conv_out,
                self.x_q8,
                E,
                B,
                self.mdl.conv_group,
                0,
                s,
            );
            self.apply_mm_preq8(&L.w_gate, self.g, B, stream);
            self.apply_mm_preq8(&L.w_up, self.u, B, stream);
            let _ = tk_q36_dflash_swiglu_q8(
                self.tk_ptr,
                self.g,
                self.u,
                self.s_buf,
                self.x_q8,
                (B as i64) * (F as i64),
                s,
            );
            self.apply_mm_preq8(&L.w_down, self.ffn_out, B, stream);
            let _ = tk_q36_dflash_dyn_conv(
                self.tk_ptr,
                self.ffn_out,
                self.dyn_f,
                L.ffn_conv_base,
                self.conv_out,
                E,
                B,
                self.mdl.conv_group,
                1,
                s,
            );
            let _ = tk_q36_dflash_add_inplace(
                self.tk_ptr,
                self.x,
                self.conv_out,
                (B as i64) * (E as i64),
                s,
            );
        }

        let _ = tk_q36_dflash_rms_rows_q8(
            self.tk_ptr,
            self.x,
            self.mdl.out_norm,
            self.h_q,
            self.x_q8,
            E,
            B,
            eps,
            s,
        );
        self.apply_mm_preq8(&self.shared_head, self.logits, B, stream);
        self.apply_mm_preq8(
            &self.mdl.sel_hidden,
            self.mdl.trans_gate.cast_mut(),
            B,
            stream,
        );
        let _ = tk_q36_dflash_top16(
            self.tk_ptr,
            self.logits,
            self.n_vocab,
            B,
            self.draft_map,
            self.cand_id,
            self.cand_u,
            s,
        );
        if !samp_tau_d.is_null() && !samp_seed_d.is_null() && !samp_ctr_d.is_null() {
            let _ = tk_q36_dflash_select_chain_q(
                self.tk_ptr,
                self.cand_id,
                self.cand_u,
                self.mdl.trans_gate,
                self.mdl.trans_pred.cast::<c_void>(),
                self.mdl.trans_succ.cast::<c_void>(),
                anchor_d,
                out_d,
                self.chain_sc,
                self.q_prop,
                samp_tau_d,
                samp_seed_d,
                samp_ctr_d,
                B - 1,
                s,
            );
        } else {
            let _ = tk_q36_dflash_select_chain(
                self.tk_ptr,
                self.cand_id,
                self.cand_u,
                self.mdl.trans_gate,
                self.mdl.trans_pred.cast::<c_void>(),
                self.mdl.trans_succ.cast::<c_void>(),
                anchor_d,
                out_d,
                self.chain_sc,
                B - 1,
                s,
            );
        }
    }

    /// Copy `cand_id_` and `q_prop` to pinned host (`DFlash::q_to_host` in `q36_dflash_model.cpp:293-296`).
    pub fn q_to_host(&mut self, stream: hipStream_t) {
        if !self.ready {
            return;
        }
        let n = (self.mdl.block_size as usize) * 16 * 4;
        unsafe {
            let _ = hipMemcpyAsync(
                self.hcand.cast::<c_void>(),
                self.cand_id.cast::<c_void>(),
                n,
                HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            );
            let _ = hipMemcpyAsync(
                self.q_prop_h.cast::<c_void>(),
                self.q_prop.cast::<c_void>(),
                n,
                HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            );
        }
    }

    /// Lookup proposal probability `q(t)` at draft position `i` (`0..B-1`) (`DFlash::q_host` in `q36_dflash_model.cpp:298-305`).
    pub fn q_host(&self, i: usize, t: i32) -> f64 {
        let r = i + 1;
        for c in 0..16 {
            let id = unsafe { *self.hcand.add(r * 16 + c) };
            if id == t {
                return (unsafe { *self.q_prop_h.add(r * 16 + c) }) as f64;
            }
        }
        0.0
    }

    pub fn sync_sampling(&mut self, tau: f32, seed: u32) {
        if !self.ready {
            return;
        }
        if tau != self.tau_last {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.tau_d.cast::<c_void>(),
                    (&tau as *const f32).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    ptr::null_mut(),
                );
            }
            self.tau_last = tau;
        }
        if seed != self.seed_last {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.seed_d.cast::<c_void>(),
                    (&seed as *const u32).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    ptr::null_mut(),
                );
            }
            self.seed_last = seed;
        }
        self.ctr_host = self.ctr_host.wrapping_add(1);
        unsafe {
            let _ = hipMemcpyAsync(
                self.ctr_d.cast::<c_void>(),
                (&self.ctr_host as *const u32).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                ptr::null_mut(),
            );
        }
    }

    pub fn draft_block(
        &mut self,
        embd_table: *const u8,
        embd_type: i32,
        embd_row: usize,
        anchor_tok_d: *const i32,
        tap_row_idx_d: *const i32,
        pos0_d: *const i32,
        n_spec: i32,
        out_picks_d: *mut i32,
        arg_pv: *mut f32,
        arg_pi: *mut i32,
        sampled: bool,
    ) {
        if !self.ready || n_spec <= 0 {
            return;
        }
        let B = (n_spec + 1).min(self.mdl.block_size.min(K_DFLASH_MAX_B as i32));
        let E = self.mdl.d_model;
        let H = self.mdl.n_heads;
        let HKV = self.mdl.n_kv;
        let HD = self.mdl.head_dim;
        let KV = HKV * HD;
        let F = self.mdl.d_ff;
        let eps = self.mdl.rms_eps;
        let stream = ptr::null_mut();

        for i in 0..K_DFLASH_TAPS {
            if !tap_row_idx_d.is_null() {
                let _ = tk_q36_row_copy(
                    self.tk_ptr,
                    self.h_q,
                    self.tap_buf[i],
                    tap_row_idx_d,
                    E,
                    stream,
                );
                self.apply_q8(&self.mdl.tap_proj[i], self.h_q, self.ctx_part[i], 1);
            } else {
                self.apply_q8(&self.mdl.tap_proj[i], self.tap_buf[i], self.ctx_part[i], 1);
            }
        }
        let ys: [*const f32; K_DFLASH_TAPS] = [
            self.ctx_part[0],
            self.ctx_part[1],
            self.ctx_part[2],
            self.ctx_part[3],
            self.ctx_part[4],
        ];
        let _ = tk_q36_dflash_sum5_rms_q8(
            self.tk_ptr,
            ys.as_ptr(),
            self.mdl.tap_norm,
            self.ctx_vec,
            if self.mdl.is_gguf {
                self.x_q8
            } else {
                ptr::null_mut()
            },
            E,
            1,
            eps,
            stream,
        );

        let _ = tk_q36_dflash_block_tokens(
            self.tk_ptr,
            anchor_tok_d,
            self.block_tok,
            B,
            self.mdl.mask_token_id,
            stream,
        );
        let _ = tk_q36_embed_rows(
            self.tk_ptr,
            embd_type,
            embd_table.cast::<c_void>(),
            embd_row,
            self.block_tok,
            B as i64,
            E as i64,
            self.x,
            stream,
        );
        if !self.mdl.emb_norm.is_null() {
            let _ = tk_q36_dflash_rms_rows(
                self.tk_ptr,
                self.x,
                self.mdl.emb_norm,
                self.x,
                E,
                B,
                eps,
                stream,
            );
        }

        let sc = 1.0f32 / (HD as f32).sqrt();
        if self.mdl.is_gguf {
            for l in 0..(self.mdl.n_layers as usize) {
                let L = &self.mdl.layers[l];
                self.apply_q8_pre(&L.wk, self.k, 1);
                self.apply_q8_pre(&L.wv, self.v, 1);
                let _ = tk_q36_dflash_qk_norm_rope(
                    self.tk_ptr,
                    self.k,
                    L.k_norm,
                    HKV,
                    HD,
                    1,
                    pos0_d,
                    self.mdl.rope_theta,
                    eps,
                    stream,
                );
                let _ = tk_q36_dflash_kv_store(
                    self.tk_ptr,
                    self.k,
                    self.v,
                    self.k_ring[l],
                    self.v_ring[l],
                    KV,
                    1,
                    pos0_d,
                    self.ring_len,
                    stream,
                );
            }
            for l in 0..(self.mdl.n_layers as usize) {
                let L = &self.mdl.layers[l];
                let _ = tk_q36_dflash_rms_rows_q8(
                    self.tk_ptr,
                    self.x,
                    L.norm_q,
                    self.h_q,
                    self.x_q8,
                    E,
                    B,
                    eps,
                    stream,
                );
                self.apply_q8_pre(&L.w_dyn, self.dyn_w, B);
                let _ = tk_q36_dflash_dyn_conv_q8(
                    self.tk_ptr,
                    self.h_q,
                    self.dyn_w,
                    L.conv_base,
                    self.conv_out,
                    self.x_q8,
                    E,
                    B,
                    self.mdl.conv_group,
                    0,
                    stream,
                );
                self.apply_q8_pre(&L.wq, self.q, B);
                self.apply_q8_pre(&L.wk, self.k, B);
                self.apply_q8_pre(&L.wv, self.v, B);
                let _ = tk_q36_dflash_qk_norm_rope(
                    self.tk_ptr,
                    self.q,
                    L.q_norm,
                    H,
                    HD,
                    B,
                    pos0_d,
                    self.mdl.rope_theta,
                    eps,
                    stream,
                );
                let _ = tk_q36_dflash_qk_norm_rope(
                    self.tk_ptr,
                    self.k,
                    L.k_norm,
                    HKV,
                    HD,
                    B,
                    pos0_d,
                    self.mdl.rope_theta,
                    eps,
                    stream,
                );
                let _ = tk_q36_dflash_kv_store(
                    self.tk_ptr,
                    self.k,
                    self.v,
                    self.k_ring[l],
                    self.v_ring[l],
                    KV,
                    B,
                    pos0_d,
                    self.ring_len,
                    stream,
                );
                let _ = tk_q36_dflash_block_attn(
                    self.tk_ptr,
                    self.q,
                    self.k_ring[l],
                    self.v_ring[l],
                    pos0_d,
                    H,
                    HKV,
                    HD,
                    self.ring_len,
                    self.mdl.window,
                    sc,
                    self.att_part,
                    self.att,
                    stream,
                );
                self.apply_q8(&L.wo, self.att, self.ffn_out, B);
                let _ = tk_q36_dflash_dyn_conv(
                    self.tk_ptr,
                    self.ffn_out,
                    self.dyn_w,
                    L.conv_base,
                    self.conv_out,
                    E,
                    B,
                    self.mdl.conv_group,
                    1,
                    stream,
                );
                let _ = tk_q36_dflash_add_inplace(
                    self.tk_ptr,
                    self.x,
                    self.conv_out,
                    (B as i64) * (E as i64),
                    stream,
                );
                let _ = tk_q36_dflash_rms_rows_q8(
                    self.tk_ptr,
                    self.x,
                    L.norm_ffn,
                    self.h_q,
                    self.x_q8,
                    E,
                    B,
                    eps,
                    stream,
                );
                self.apply_q8_pre(&L.w_ffn_dyn, self.dyn_w, B);
                let _ = tk_q36_dflash_dyn_conv_q8(
                    self.tk_ptr,
                    self.h_q,
                    self.dyn_w,
                    L.ffn_conv_base,
                    self.conv_out,
                    self.x_q8,
                    E,
                    B,
                    self.mdl.conv_group,
                    0,
                    stream,
                );
                self.apply_q8_pre(&L.w_gate, self.g, B);
                self.apply_q8_pre(&L.w_up, self.u, B);
                let _ = tk_q36_dflash_swiglu_q8(
                    self.tk_ptr,
                    self.g,
                    self.u,
                    self.g,
                    self.x_q8,
                    (B as i64) * (F as i64),
                    stream,
                );
                self.apply_q8_pre(&L.w_down, self.ffn_out, B);
                let _ = tk_q36_dflash_dyn_conv(
                    self.tk_ptr,
                    self.ffn_out,
                    self.dyn_w,
                    L.ffn_conv_base,
                    self.conv_out,
                    E,
                    B,
                    self.mdl.conv_group,
                    1,
                    stream,
                );
                let _ = tk_q36_dflash_add_inplace(
                    self.tk_ptr,
                    self.x,
                    self.conv_out,
                    (B as i64) * (E as i64),
                    stream,
                );
            }
            let n_out = (B - 1).min(n_spec);
            let _ = tk_q36_dflash_rms_rows_q8(
                self.tk_ptr,
                self.x,
                self.mdl.out_norm,
                self.h_q,
                self.x_q8,
                E,
                B,
                eps,
                stream,
            );
            self.apply_q8_pre(&self.shared_head, self.logits, B);
            self.apply_q8_pre(&self.mdl.sel_hidden, self.mdl.trans_gate.cast_mut(), B);
            let _ = tk_q36_dflash_top16(
                self.tk_ptr,
                self.logits,
                self.n_vocab,
                B,
                self.draft_map,
                self.cand_id,
                self.cand_u,
                stream,
            );
            if sampled {
                let _ = tk_q36_dflash_select_chain_q(
                    self.tk_ptr,
                    self.cand_id,
                    self.cand_u,
                    self.mdl.trans_gate,
                    self.mdl.trans_pred.cast::<c_void>(),
                    self.mdl.trans_succ.cast::<c_void>(),
                    anchor_tok_d,
                    out_picks_d,
                    self.chain_sc,
                    self.q_prop,
                    self.tau_d,
                    self.seed_d,
                    self.ctr_d,
                    n_out,
                    stream,
                );
            } else {
                let _ = tk_q36_dflash_select_chain(
                    self.tk_ptr,
                    self.cand_id,
                    self.cand_u,
                    self.mdl.trans_gate,
                    self.mdl.trans_pred.cast::<c_void>(),
                    self.mdl.trans_succ.cast::<c_void>(),
                    anchor_tok_d,
                    out_picks_d,
                    self.chain_sc,
                    n_out,
                    stream,
                );
            }
            return;
        }

        for l in 0..(self.mdl.n_layers as usize) {
            let L = &self.mdl.layers[l];
            let _ =
                tk_q36_dflash_rms_rows(self.tk_ptr, self.x, L.norm_q, self.h_q, E, B, eps, stream);
            let _ = tk_q36_dflash_rms_rows(
                self.tk_ptr,
                self.ctx_vec,
                L.norm_kv,
                self.h_kv,
                E,
                1,
                eps,
                stream,
            );
            unsafe {
                let _ = hipMemcpyAsync(
                    self.h_kv.add(E as usize).cast::<c_void>(),
                    self.h_q.add(E as usize).cast::<c_void>(),
                    ((B - 1) as usize) * (E as usize) * 4,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    ptr::null_mut(),
                );
            }
            self.apply_q8(&L.wq, self.h_q, self.q, B);
            self.apply_q8(&L.wk, self.h_kv, self.k, B);
            self.apply_q8(&L.wv, self.h_kv, self.v, B);

            let _ = tk_q36_dflash_qk_norm_rope(
                self.tk_ptr,
                self.q,
                L.q_norm,
                H,
                HD,
                B,
                pos0_d,
                self.mdl.rope_theta,
                eps,
                stream,
            );
            let _ = tk_q36_dflash_qk_norm_rope(
                self.tk_ptr,
                self.k,
                L.k_norm,
                HKV,
                HD,
                B,
                pos0_d,
                self.mdl.rope_theta,
                eps,
                stream,
            );
            let _ = tk_q36_dflash_kv_store(
                self.tk_ptr,
                self.k,
                self.v,
                self.k_ring[l],
                self.v_ring[l],
                KV,
                B,
                pos0_d,
                self.ring_len,
                stream,
            );
            let _ = tk_q36_dflash_block_attn(
                self.tk_ptr,
                self.q,
                self.k_ring[l],
                self.v_ring[l],
                pos0_d,
                H,
                HKV,
                HD,
                self.ring_len,
                self.mdl.window,
                sc,
                self.att_part,
                self.att,
                stream,
            );
            self.apply_q8(&L.wo, self.att, self.ffn_out, B);
            let _ = tk_q36_dflash_add_inplace(
                self.tk_ptr,
                self.x,
                self.ffn_out,
                (B as i64) * (E as i64),
                stream,
            );

            self.apply_q8(&L.w_dyn, self.x, self.dyn_w, B);
            let _ = tk_q36_dflash_dyn_conv(
                self.tk_ptr,
                self.x,
                self.dyn_w,
                L.conv_base,
                self.conv_out,
                E,
                B,
                64,
                5,
                stream,
            );
            let _ = tk_q36_dflash_add_inplace(
                self.tk_ptr,
                self.x,
                self.conv_out,
                (B as i64) * (E as i64),
                stream,
            );

            let _ = tk_q36_dflash_rms_rows(
                self.tk_ptr,
                self.x,
                L.norm_ffn,
                self.h_q,
                E,
                B,
                eps,
                stream,
            );
            self.apply_q8(&L.w_gate, self.h_q, self.g, B);
            self.apply_q8(&L.w_up, self.h_q, self.u, B);
            let _ = tk_q36_dflash_swiglu(
                self.tk_ptr,
                self.g,
                self.u,
                self.g,
                (B as i64) * (F as i64),
                stream,
            );
            self.apply_q8(&L.w_down, self.g, self.ffn_out, B);
            let _ = tk_q36_dflash_add_inplace(
                self.tk_ptr,
                self.x,
                self.ffn_out,
                (B as i64) * (E as i64),
                stream,
            );
        }

        let n_out = (B - 1).min(n_spec);
        let _ = tk_q36_dflash_rms_rows(
            self.tk_ptr,
            unsafe { self.x.add(E as usize) },
            self.mdl.out_norm,
            self.h_q,
            E,
            n_out,
            eps,
            stream,
        );
        self.apply_q8(&self.shared_head, self.h_q, self.logits, n_out);

        if self.mdl.has_trans && n_out >= 1 {
            let _ = tk_q36_dflash_top16(
                self.tk_ptr,
                self.logits,
                self.n_vocab,
                n_out,
                self.draft_map,
                self.cand_id,
                self.cand_u,
                stream,
            );
            if sampled {
                let _ = tk_q36_dflash_select_chain_q(
                    self.tk_ptr,
                    self.cand_id,
                    self.cand_u,
                    self.mdl.trans_gate,
                    self.mdl.trans_pred.cast::<c_void>(),
                    self.mdl.trans_succ.cast::<c_void>(),
                    anchor_tok_d,
                    out_picks_d,
                    self.chain_sc,
                    self.q_prop,
                    self.tau_d,
                    self.seed_d,
                    self.ctr_d,
                    n_out,
                    stream,
                );
            } else {
                let _ = tk_q36_dflash_select_chain(
                    self.tk_ptr,
                    self.cand_id,
                    self.cand_u,
                    self.mdl.trans_gate,
                    self.mdl.trans_pred.cast::<c_void>(),
                    self.mdl.trans_succ.cast::<c_void>(),
                    anchor_tok_d,
                    out_picks_d,
                    self.chain_sc,
                    n_out,
                    stream,
                );
            }
        } else {
            let nb = ((self.n_vocab + 1023) / 1024).clamp(1, 256);
            let _ = tk_q36_argmax(
                self.tk_ptr,
                nb,
                n_out,
                self.logits,
                self.n_vocab,
                out_picks_d,
                arg_pv,
                arg_pi,
                false,
                ptr::null(),
                stream,
            );
            let _ = tk_q36_argmax(
                self.tk_ptr,
                1,
                n_out,
                self.logits,
                nb,
                out_picks_d,
                arg_pv,
                arg_pi,
                true,
                self.draft_map,
                stream,
            );
        }
    }

    pub fn draft_step_bonus(
        &mut self,
        embd_table: *const u8,
        embd_type: i32,
        embd_row: usize,
        bonus_tok_d: *const i32,
        n_spec: i32,
        pos0_d: *const i32,
        out_picks_d: *mut i32,
        arg_pv: *mut f32,
        arg_pi: *mut i32,
        sampled: bool,
    ) {
        if !self.ready {
            return;
        }
        let _ = tk_q36_dflash_set_i32(self.tk_ptr, self.pos_d, n_spec, ptr::null_mut());
        self.draft_block(
            embd_table,
            embd_type,
            embd_row,
            bonus_tok_d,
            self.pos_d,
            pos0_d,
            n_spec,
            out_picks_d,
            arg_pv,
            arg_pi,
            sampled,
        );
    }
}

// ============================================================================
// N-gram Suffix Proposer (`NgramMod` in `q36_engine.cu:446-478`)
// ============================================================================

pub struct NgramMod {
    pub n: usize,
    pub e: Vec<i64>,
    pub used: usize,
    pub i_last: usize,
    pub hist: Vec<i32>,
    pub head_3: [i32; 1 << 15],
    pub head_2: [i32; 1 << 14],
    pub prev_3: Vec<i32>,
    pub prev_2: Vec<i32>,
}

impl Default for NgramMod {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramMod {
    pub const N_ENTRIES: usize = 4 << 20;

    pub fn new() -> Self {
        Self::with_n(24)
    }

    pub fn with_n(n: usize) -> Self {
        Self {
            n: n.max(2),
            e: vec![-1i64; Self::N_ENTRIES],
            used: 0,
            i_last: 0,
            hist: Vec::new(),
            head_3: [-1; 1 << 15],
            head_2: [-1; 1 << 14],
            prev_3: Vec::new(),
            prev_2: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.e.fill(-1);
        self.used = 0;
        self.i_last = 0;
        self.hist.clear();
        self.head_3.fill(-1);
        self.head_2.fill(-1);
        self.prev_3.clear();
        self.prev_2.clear();
    }

    #[inline]
    fn h2(a: i32, b: i32) -> usize {
        (((a as u32).wrapping_mul(2654435761)) ^ ((b as u32).wrapping_mul(40503))) as usize
            & ((1 << 14) - 1)
    }

    #[inline]
    fn h3(a: i32, b: i32, c: i32) -> usize {
        (((a as u32).wrapping_mul(2654435761))
            ^ ((b as u32).wrapping_mul(40503))
            ^ ((c as u32).wrapping_mul(65537))) as usize
            & ((1 << 15) - 1)
    }

    pub fn feed(&mut self, t: i32) {
        let pos = self.hist.len() as i32;
        self.hist.push(t);
        self.prev_2.push(-1);
        self.prev_3.push(-1);
        if pos >= 1 {
            let b = Self::h2(self.hist[(pos - 1) as usize], t);
            self.prev_2[pos as usize] = self.head_2[b];
            self.head_2[b] = pos;
        }
        if pos >= 2 {
            let b = Self::h3(
                self.hist[(pos - 2) as usize],
                self.hist[(pos - 1) as usize],
                t,
            );
            self.prev_3[pos as usize] = self.head_3[b];
            self.head_3[b] = pos;
        }
    }

    #[inline]
    fn idx_window(&self, t: &[i32]) -> usize {
        let mut r = 0u64;
        for i in 0..self.n {
            r = r
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add((t[i] as u32) as u64);
        }
        (r as usize) % self.e.len()
    }

    /// Update the n-gram hash table with any unseen n-grams in `ctx` (`NgramMod::update` in `q36_engine.cu:458-466`).
    pub fn update_ctx(&mut self, ctx: &[i32]) {
        let n = self.n;
        if ctx.len() <= n {
            return;
        }
        let mut i = self.i_last;
        while i + n < ctx.len() {
            let s = self.idx_window(&ctx[i..i + n]);
            if self.e[s] < 0 {
                self.used += 1;
            }
            self.e[s] = ctx[i + n] as i64;
            i += 1;
        }
        self.i_last = ctx.len() - n;
        if (self.used as f64) / (self.e.len() as f64) > 0.25 {
            self.e.fill(-1);
            self.used = 0;
            self.i_last = 0;
        }
    }

    /// Rolling n-gram open-addressed lookup (`NgramMod::propose` in `q36_engine.cu:468-477`).
    pub fn propose_ctx(&mut self, ctx: &[i32], k_max: i32, k_min: i32) -> Vec<i32> {
        let n = self.n;
        if ctx.len() < n {
            return Vec::new();
        }
        self.update_ctx(ctx);
        let mut w: Vec<i32> = ctx[ctx.len() - n..].to_vec();
        let mut out = Vec::new();
        for i in 0..(k_max.max(0) as usize) {
            let t = self.e[self.idx_window(&w[i..i + n])];
            if t < 0 {
                break;
            }
            out.push(t as i32);
            w.push(t as i32);
        }
        if (out.len() as i32) < k_min {
            out.clear();
        }
        out
    }

    pub fn propose(&mut self, k: i32, out: &mut [i32]) -> i32 {
        if self.hist.len() >= self.n {
            let v = self.propose_ctx(&self.hist.clone(), k, 1);
            let m = v.len().min(out.len()).min(k.max(0) as usize);
            out[..m].copy_from_slice(&v[..m]);
            return m as i32;
        }
        let n = self.hist.len() as i32;
        if n < 3 || k <= 0 {
            return 0;
        }
        let a = self.hist[(n - 3) as usize];
        let b = self.hist[(n - 2) as usize];
        let c = self.hist[(n - 1) as usize];

        let mut p = self.head_3[Self::h3(a, b, c)];
        let mut steps = 0;
        while p >= 2 && steps < 64 {
            if p < n - 1
                && self.hist[(p - 2) as usize] == a
                && self.hist[(p - 1) as usize] == b
                && self.hist[p as usize] == c
            {
                let mut m = 0i32;
                while m < k && (p + 1 + m) < n && (m as usize) < out.len() {
                    out[m as usize] = self.hist[(p + 1 + m) as usize];
                    m += 1;
                }
                if m > 0 {
                    return m;
                }
            }
            p = self.prev_3[p as usize];
            steps += 1;
        }

        let mut p2 = self.head_2[Self::h2(b, c)];
        let mut steps2 = 0;
        while p2 >= 1 && steps2 < 64 {
            if p2 < n - 1 && self.hist[(p2 - 1) as usize] == b && self.hist[p2 as usize] == c {
                let mut m = 0i32;
                while m < k && (p2 + 1 + m) < n && (m as usize) < out.len() {
                    out[m as usize] = self.hist[(p2 + 1 + m) as usize];
                    m += 1;
                }
                if m > 0 {
                    return m;
                }
            }
            p2 = self.prev_2[p2 as usize];
            steps2 += 1;
        }
        0
    }
}

// ============================================================================
// Sampling (`Sampling`, `SplitMix64`, `make_dist`, `draw_dist`, `sample` in `q36_engine.cu:3102-3298`)
// ============================================================================

#[derive(Debug, Clone)]
pub struct Sampling {
    pub temp: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: i32,
    pub rep_pen: f32,
    pub pres_pen: f32,
    pub freq_pen: f32,
    pub rep_last_n: i32,
    pub seed: u64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temp: 0.6,
            top_p: 0.95,
            min_p: 0.0,
            top_k: 20,
            rep_pen: 1.0,
            pres_pen: 0.0,
            freq_pen: 0.0,
            rep_last_n: 64,
            seed: 1234,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    pub s: u64,
    mt: [u64; 312],
    mti: usize,
}

impl SplitMix64 {
    const NN: usize = 312;
    const MM: usize = 156;
    const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
    const UM: u64 = 0xFFFF_FFFF_8000_0000;
    const LM: u64 = 0x0000_0000_7FFF_FFFF;
    const INIT_MULT: u64 = 6_364_136_223_846_793_005;

    pub fn new(seed: u64) -> Self {
        let mut mt = [0u64; Self::NN];
        mt[0] = seed;
        for i in 1..Self::NN {
            let prev = mt[i - 1];
            mt[i] = Self::INIT_MULT
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        Self {
            s: seed,
            mt,
            mti: Self::NN,
        }
    }

    fn twist(&mut self) {
        const MAG01: [u64; 2] = [0, SplitMix64::MATRIX_A];
        for i in 0..(Self::NN - Self::MM) {
            let x = (self.mt[i] & Self::UM) | (self.mt[i + 1] & Self::LM);
            self.mt[i] = self.mt[i + Self::MM] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
        }
        for i in (Self::NN - Self::MM)..(Self::NN - 1) {
            let x = (self.mt[i] & Self::UM) | (self.mt[i + 1] & Self::LM);
            self.mt[i] = self.mt[i + Self::MM - Self::NN] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
        }
        let x = (self.mt[Self::NN - 1] & Self::UM) | (self.mt[0] & Self::LM);
        self.mt[Self::NN - 1] = self.mt[Self::MM - 1] ^ (x >> 1) ^ MAG01[(x & 1) as usize];
        self.mti = 0;
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.mti >= Self::NN {
            self.twist();
        }
        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^= x >> 43;
        self.s = x;
        x
    }

    #[inline]
    pub fn uniform_01(&mut self) -> f64 {
        const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0_f64;
        const MAX_LT_ONE: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF);
        let ret = (self.next_u64() as f64) / TWO_POW_64;
        if ret >= 1.0 {
            MAX_LT_ONE
        } else {
            ret
        }
    }

    #[inline]
    pub fn uniform_range(&mut self, high: f64) -> f64 {
        self.uniform_01() * high
    }

    pub fn next_f32(&mut self) -> f32 {
        self.uniform_01() as f32
    }
}

pub fn penalize(logits: &mut [f32], s: &Sampling, recent: &[i32]) {
    if s.rep_last_n == 0
        || (s.rep_pen == 1.0 && s.pres_pen == 0.0 && s.freq_pen == 0.0)
        || recent.is_empty()
    {
        return;
    }
    let n = logits.len() as i32;
    let start = if s.rep_last_n > 0 && (recent.len() as i32) > s.rep_last_n {
        recent.len() - (s.rep_last_n as usize)
    } else {
        0
    };
    let mut counts: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    for &t in &recent[start..] {
        if t >= 0 && t < n {
            *counts.entry(t).or_insert(0) += 1;
        }
    }
    for (id, c) in counts {
        let mut v = logits[id as usize];
        if s.rep_pen != 1.0 {
            v = if v <= 0.0 {
                v * s.rep_pen
            } else {
                v / s.rep_pen
            };
        }
        v -= s.pres_pen + (c as f32) * s.freq_pen;
        logits[id as usize] = v;
    }
}

pub fn make_dist(logits_in: &[f32], s: &Sampling, recent: &[i32]) -> Vec<(i32, f32)> {
    let has_pen = s.rep_last_n != 0
        && (s.rep_pen != 1.0 || s.pres_pen != 0.0 || s.freq_pen != 0.0)
        && !recent.is_empty();
    let mut pen_buf;
    let lg: &[f32] = if has_pen {
        pen_buf = logits_in.to_vec();
        penalize(&mut pen_buf, s, recent);
        &pen_buf
    } else {
        logits_in
    };

    let n = lg.len();
    if n == 0 {
        return Vec::new();
    }
    if s.temp <= 0.0 || s.top_k == 1 {
        let mut best = 0usize;
        for i in 1..n {
            if lg[i] > lg[best] {
                best = i;
            }
        }
        return vec![(best as i32, 1.0)];
    }

    let kk = (if s.top_k > 0 { s.top_k as usize } else { 64 })
        .min(n)
        .max(1);

    // Min-heap of size `kk` where `hp[0]` is the smallest of the top `kk` candidates
    // (`q36_engine.cu:2648-2659`).
    #[inline]
    fn worse(a: (f32, i32), b: (f32, i32)) -> bool {
        a.0 < b.0 || (a.0 == b.0 && a.1 > b.1)
    }

    let mut hp: Vec<(f32, i32)> = Vec::with_capacity(kk + 1);
    for (i, &v) in lg.iter().enumerate() {
        if hp.len() == kk && !(v > hp[0].0) {
            continue;
        }
        let item = (v, i as i32);
        if hp.len() < kk {
            hp.push(item);
            let mut cur = hp.len() - 1;
            while cur > 0 {
                let p = (cur - 1) / 2;
                if worse(hp[cur], hp[p]) {
                    hp.swap(cur, p);
                    cur = p;
                } else {
                    break;
                }
            }
        } else {
            hp[0] = item;
            let mut cur = 0usize;
            loop {
                let l = 2 * cur + 1;
                if l >= kk {
                    break;
                }
                let r = l + 1;
                let mut smallest = if worse(hp[l], hp[cur]) { l } else { cur };
                if r < kk && worse(hp[r], hp[smallest]) {
                    smallest = r;
                }
                if smallest != cur {
                    hp.swap(cur, smallest);
                    cur = smallest;
                } else {
                    break;
                }
            }
        }
    }

    hp.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let mx = hp[0].0 as f64;
    let temp = s.temp as f64;
    let mut e = Vec::with_capacity(hp.len());
    let mut sum = 0.0f64;
    for &(val, _) in &hp {
        let ev = ((val as f64 - mx) / temp).exp();
        sum += ev;
        e.push(ev);
    }

    let mut keep = hp.len();
    let mut cum = 0.0f64;
    let min_p = s.min_p as f64;
    let top_p = s.top_p as f64;
    let p0 = e[0] / sum;
    for (i, &ev) in e.iter().enumerate() {
        let pi = ev / sum;
        if min_p > 0.0 && pi < min_p * p0 {
            keep = i;
            break;
        }
        cum += pi;
        if top_p > 0.0 && top_p < 1.0 && cum >= top_p {
            keep = i + 1;
            break;
        }
    }
    keep = keep.max(1);

    let mut tot = 0.0f64;
    for &ev in &e[..keep] {
        tot += ev;
    }
    let mut cand = Vec::with_capacity(keep);
    for i in 0..keep {
        cand.push((hp[i].1, (e[i] / tot) as f32));
    }
    cand
}

/// One verify row's decision in a sampled speculative round (`spec_run_with_callback`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowDecision {
    /// The draft is kept.
    Accept,
    /// The row emits this token (residual sample) and the round ends.
    End(i32),
}

/// The sampled verify row's decision for draft `d` against the target list `p` (`make_dist`, modified in place):
/// - `qs == false` (n-gram / point-mass drafts): accept with `p(d)`, else draw from `p` without `d`;
/// - `qs == true` (drafts sampled from the drafter's q, `qof(t)` = q(t)): accept with `min(1, p(d) / q(d))`, else
///   draw from `norm(max(0, p - q))` (`refetch()` = `p` again, for the measure-zero `p <= q` case).
///
/// Extracted verbatim (same arithmetic and RNG order) from the spec loop so it can be tested on CPU.
pub fn spec_decide_row(
    p: &mut [(i32, f32)],
    d: i32,
    qs: bool,
    qof: impl Fn(i32) -> f64,
    rng: &mut SplitMix64,
    refetch: impl FnOnce() -> Vec<(i32, f32)>,
) -> RowDecision {
    let mut pd = 0.0f64;
    let mut at = p.len();
    for (k, &(id, pr)) in p.iter().enumerate() {
        if id == d {
            pd = pr as f64;
            at = k;
        }
    }
    if !qs {
        if rng.uniform_01() < pd {
            return RowDecision::Accept;
        }
        if at < p.len() {
            p[at].1 = 0.0;
        }
        let rsum: f64 = p.iter().map(|&(_, pr)| pr as f64).sum();
        if rsum > 0.0 {
            let inv_r = (1.0 / rsum) as f32;
            for (_, pr) in p.iter_mut() {
                *pr *= inv_r;
            }
        }
        return RowDecision::End(draw_dist(p, rng));
    }
    let qd_val = qof(d);
    if qd_val > 0.0 && rng.uniform_01() * qd_val < pd {
        return RowDecision::Accept;
    }
    let mut rsum = 0.0f64;
    for (id, pr) in p.iter_mut() {
        let diff = ((*pr as f64) - qof(*id)).max(0.0);
        *pr = diff as f32;
        rsum += diff;
    }
    if rsum > 0.0 {
        let inv_r = (1.0 / rsum) as f32;
        for (_, pr) in p.iter_mut() {
            *pr *= inv_r;
        }
        RowDecision::End(draw_dist(p, rng))
    } else {
        let fallback = refetch();
        RowDecision::End(draw_dist(&fallback, rng))
    }
}

pub fn draw_dist(dist: &[(i32, f32)], rng: &mut SplitMix64) -> i32 {
    if dist.is_empty() {
        return 0;
    }
    let tot: f64 = dist.iter().map(|&(_, p)| p as f64).sum();
    let mut r = rng.uniform_range(tot);
    for &(id, p) in dist {
        let pf = p as f64;
        r -= pf;
        if r <= 0.0 && pf > 0.0 {
            return id;
        }
    }
    for &(id, p) in dist.iter().rev() {
        if p > 0.0 {
            return id;
        }
    }
    dist[0].0
}

pub fn sample(logits_in: &[f32], s: &Sampling, recent: &[i32], rng: &mut SplitMix64) -> i32 {
    let dist = make_dist(logits_in, s, recent);
    draw_dist(&dist, rng)
}

pub fn sample_selftest() -> bool {
    let v = [1.0f32, 5.0, 3.0, 2.0, -1.0];
    let mut rng = SplitMix64::new(42);
    let recent: Vec<i32> = Vec::new();
    let s = Sampling {
        temp: 0.0,
        ..Sampling::default()
    };
    if sample(&v, &s, &recent, &mut rng) != 1 {
        return false;
    }
    let s_rep = Sampling {
        temp: 0.0,
        rep_pen: 10.0,
        ..Sampling::default()
    };
    let rec = [1i32];
    if sample(&v, &s_rep, &rec, &mut rng) != 2 {
        return false;
    }
    true
}

// ============================================================================
// Prompt Cache (`PromptCache` in `q36_engine.cu:3301-3345`)
// ============================================================================

#[derive(Default)]
pub struct PromptCache {
    pub tokens: Vec<i32>,
    pub state: Vec<u8>,
    pub kv: Vec<u8>,
}

impl PromptCache {
    pub fn lookup(&self, prompt: &[i32]) -> usize {
        if self.tokens.is_empty() || self.tokens.len() >= prompt.len() {
            return 0;
        }
        for (i, &t) in self.tokens.iter().enumerate() {
            if prompt[i] != t {
                return 0;
            }
        }
        self.tokens.len()
    }

    pub fn save(&mut self, eng: &mut Q36Engine, pfx: &[i32]) {
        self.tokens = pfx.to_vec();
        self.state.resize(eng.state_bytes(), 0);
        self.kv.resize(eng.kv_bytes(pfx.len() as i32), 0);
        eng.state_copy(self.state.as_mut_ptr().cast::<c_void>(), true);
        eng.kv_copy(
            self.kv.as_mut_ptr().cast::<c_void>(),
            pfx.len() as i32,
            true,
        );
    }

    pub fn save_disk(&self, path: &Path, arch: &str, n_layer: i64, n_embd: i64) -> Result<bool> {
        let mut f = File::create(path).map_err(io_err)?;
        let magic: u32 = 0x5343_3336;
        let n_tok = self.tokens.len() as u64;
        let sb = self.state.len() as u64;
        let kb = self.kv.len() as u64;
        let mut a = [0u8; 16];
        let ab = arch.as_bytes();
        let copy_len = ab.len().min(15);
        a[..copy_len].copy_from_slice(&ab[..copy_len]);
        f.write_all(&magic.to_le_bytes()).map_err(io_err)?;
        f.write_all(&a).map_err(io_err)?;
        f.write_all(&n_layer.to_le_bytes()).map_err(io_err)?;
        f.write_all(&n_embd.to_le_bytes()).map_err(io_err)?;
        f.write_all(&n_tok.to_le_bytes()).map_err(io_err)?;
        f.write_all(&sb.to_le_bytes()).map_err(io_err)?;
        f.write_all(&kb.to_le_bytes()).map_err(io_err)?;
        let tok_bytes = unsafe {
            std::slice::from_raw_parts(self.tokens.as_ptr().cast::<u8>(), self.tokens.len() * 4)
        };
        f.write_all(tok_bytes).map_err(io_err)?;
        f.write_all(&self.state).map_err(io_err)?;
        f.write_all(&self.kv).map_err(io_err)?;
        Ok(true)
    }

    pub fn load_disk(
        &mut self,
        path: &Path,
        arch: &str,
        n_layer: i64,
        n_embd: i64,
        eng: &Q36Engine,
    ) -> Result<bool> {
        let mut f = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(false),
        };
        let mut buf4 = [0u8; 4];
        let mut a = [0u8; 16];
        let mut buf8 = [0u8; 8];
        if f.read_exact(&mut buf4).is_err() {
            return Ok(false);
        }
        let magic = u32::from_le_bytes(buf4);
        f.read_exact(&mut a).map_err(io_err)?;
        f.read_exact(&mut buf8).map_err(io_err)?;
        let nl = i64::from_le_bytes(buf8);
        f.read_exact(&mut buf8).map_err(io_err)?;
        let ne = i64::from_le_bytes(buf8);
        f.read_exact(&mut buf8).map_err(io_err)?;
        let n_tok = u64::from_le_bytes(buf8) as usize;
        f.read_exact(&mut buf8).map_err(io_err)?;
        let sb = u64::from_le_bytes(buf8) as usize;
        f.read_exact(&mut buf8).map_err(io_err)?;
        let kb = u64::from_le_bytes(buf8) as usize;

        let mut expected_a = [0u8; 16];
        let ab = arch.as_bytes();
        let copy_len = ab.len().min(15);
        expected_a[..copy_len].copy_from_slice(&ab[..copy_len]);
        if magic != 0x5343_3336
            || a != expected_a
            || nl != n_layer
            || ne != n_embd
            || n_tok == 0
            || n_tok >= eng.ctx_len as usize
            || sb != eng.state_bytes()
            || kb != eng.kv_bytes(n_tok as i32)
        {
            return Ok(false);
        }
        self.tokens.resize(n_tok, 0);
        let tok_bytes = unsafe {
            std::slice::from_raw_parts_mut(self.tokens.as_mut_ptr().cast::<u8>(), n_tok * 4)
        };
        f.read_exact(tok_bytes).map_err(io_err)?;
        self.state.resize(sb, 0);
        f.read_exact(&mut self.state).map_err(io_err)?;
        self.kv.resize(kb, 0);
        f.read_exact(&mut self.kv).map_err(io_err)?;
        Ok(true)
    }
}

// ============================================================================
// HIP Graph Capture & Speculative Decoding (`Q36Engine`)
// ============================================================================

impl Q36Engine {
    /// Destroy all captured `hipGraphExec_t` handles (`Q36Engine::destroy_graphs`).
    pub fn destroy_graphs(&mut self) {
        unsafe {
            if !self.g1.is_null() {
                let _ = hipGraphExecDestroy(self.g1);
                self.g1 = ptr::null_mut();
            }
            if !self.gm1.is_null() {
                let _ = hipGraphExecDestroy(self.gm1);
                self.gm1 = ptr::null_mut();
            }
            if !self.gDs.is_null() {
                let _ = hipGraphExecDestroy(self.gDs);
                self.gDs = ptr::null_mut();
            }
            for n in 0..=16 {
                if !self.gv[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gv[n]);
                    self.gv[n] = ptr::null_mut();
                }
                if !self.gA[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gA[n]);
                    self.gA[n] = ptr::null_mut();
                }
                if !self.gB0[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gB0[n]);
                    self.gB0[n] = ptr::null_mut();
                }
                if !self.gB0i[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gB0i[n]);
                    self.gB0i[n] = ptr::null_mut();
                }
                if !self.gC[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gC[n]);
                    self.gC[n] = ptr::null_mut();
                }
                if !self.gD[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gD[n]);
                    self.gD[n] = ptr::null_mut();
                }
                if !self.gDm[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gDm[n]);
                    self.gDm[n] = ptr::null_mut();
                }
                if !self.gD0[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gD0[n]);
                    self.gD0[n] = ptr::null_mut();
                }
                if !self.gD0m[n].is_null() {
                    let _ = hipGraphExecDestroy(self.gD0m[n]);
                    self.gD0m[n] = ptr::null_mut();
                }
                for k in 0..16 {
                    if !self.gB[n][k].is_null() {
                        let _ = hipGraphExecDestroy(self.gB[n][k]);
                        self.gB[n][k] = ptr::null_mut();
                    }
                    if !self.gBi[n][k].is_null() {
                        let _ = hipGraphExecDestroy(self.gBi[n][k]);
                        self.gBi[n][k] = ptr::null_mut();
                    }
                }
            }
            for k in 0..16 {
                if !self.gS[k].is_null() {
                    let _ = hipGraphExecDestroy(self.gS[k]);
                    self.gS[k] = ptr::null_mut();
                }
            }
        }
    }

    pub(crate) fn capture_graph(&mut self, body: impl FnOnce(&mut Self)) -> Result<hipGraphExec_t> {
        hip_check(
            unsafe { hipStreamBeginCapture(self.s_main, HIP_STREAM_CAPTURE_MODE_RELAXED) },
            "hipStreamBeginCapture",
        )?;
        body(self);
        let mut gr: hipGraph_t = ptr::null_mut();
        hip_check(
            unsafe { hipStreamEndCapture(self.s_main, &mut gr) },
            "hipStreamEndCapture",
        )?;
        let mut exec: hipGraphExec_t = ptr::null_mut();
        hip_check(
            unsafe { hipGraphInstantiate(&mut exec, gr, ptr::null_mut(), ptr::null_mut(), 0) },
            "hipGraphInstantiate",
        )?;
        let _ = unsafe { hipGraphDestroy(gr) };
        Ok(exec)
    }

    pub fn ensure_single_decode_graph(&mut self) -> Result<hipGraphExec_t> {
        if !self.g1.is_null() {
            return Ok(self.g1);
        }
        let exec = self.capture_graph(|eng| {
            eng.forward_full(eng.tok_d, 1, eng.pos0_d, 1, false);
        })?;
        self.g1 = exec;
        Ok(exec)
    }

    pub fn ensure_verify_graph(&mut self, n: usize) -> Result<hipGraphExec_t> {
        ensure!(n >= 1 && n <= 16, "verify graph n must be in 1..=16");
        if !self.gv[n].is_null() {
            return Ok(self.gv[n]);
        }
        let exec = self.capture_graph(|eng| {
            eng.forward_full(eng.tok_d, n as i32, eng.pos0_d, n as i32, true);
        })?;
        self.gv[n] = exec;
        Ok(exec)
    }

    pub fn ensure_mtp_graph(&mut self) -> Result<hipGraphExec_t> {
        if !self.gm1.is_null() {
            return Ok(self.gm1);
        }
        let exec = self.capture_graph(|eng| {
            eng.forward_mtp_batch(1, 0, eng.h_mtp, true, true, eng.pick_m, -1);
        })?;
        self.gm1 = exec;
        Ok(exec)
    }

    #[inline]
    pub fn argmax_draft_dev(&mut self, logits: *const f32, rows: i32, out: *mut i32) {
        self.argmax_dev(logits, rows, out, self.draft_n_vocab, self.draft_map);
    }

    /// One chained MTP draft step `k >= 2` (`Engine::chain_step` in `q36_engine.cu:1716-1720`).
    pub fn chain_step(&mut self, k: i32) {
        let E = self.d_model() as usize;
        let stream = self.s_main.cast::<c_void>();
        let _ = tk_q36_chain_prep(
            self.tk.as_ptr(),
            self.tok_d,
            self.tok_m_d,
            self.pos0_d,
            self.pos0_m_d,
            k,
            stream,
        );
        let seed = if k > 2 {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.y.cast::<c_void>(),
                    self.h_mtp_b.cast::<c_void>(),
                    E * 4,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.s_main,
                );
            }
            self.y
        } else {
            self.h_mtp_prev
        };
        let dst = unsafe { self.tok_d.add(k as usize) };
        self.forward_mtp_batch(
            1,
            (self.n_past + k - 1) as i64,
            seed,
            true,
            true,
            dst,
            k - 1,
        );
    }

    /// Run chained MTP draft steps `2..=D` (`Engine::draft_chain` in `q36_engine.cu:1722-1724`).
    pub fn draft_chain(&mut self, D: i32) {
        for k in 2..=D {
            self.chain_step(k);
        }
    }

    /// Copy `conf_md` (`32` floats: `16` conf + `16` top) to pinned `hconf` (`Engine::conf_to_host` in `q36_engine.cu:1725-1727`).
    pub fn conf_to_host(&mut self) {
        if self.dl.want_conf() {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.hconf.cast::<c_void>(),
                    self.conf_md.cast::<c_void>(),
                    32 * 4,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    self.s_main,
                );
            }
        }
    }

    /// Stop-mode prefix extension (`Engine::stop_extend` in `q36_engine.cu:1728-1735`).
    pub fn stop_extend(&mut self, mut n: i32) -> i32 {
        let s = self.draft_samp_on;
        loop {
            let _ = unsafe { hipStreamSynchronize(self.s_main) };
            let top_slice = unsafe { std::slice::from_raw_parts(self.hconf.add(16), 16) };
            let k = self.dl.keep(top_slice, n, s);
            if k < n || n >= self.dl.dmax_for(s) {
                self.win_chain = n;
                return k;
            }
            n += 1;
            let idx = n as usize;
            if idx < self.gS.len() && !self.gS[idx].is_null() {
                unsafe {
                    let _ = hipGraphLaunch(self.gS[idx], self.s_main);
                }
            } else {
                self.chain_step(n);
            }
            self.conf_to_host();
        }
    }

    /// Run DFlash2 draft into `out_d` and copy `scores()` to `hconf + 16` (`Engine::dfl_draft_into` in `q36_engine.cu:1838-1842`).
    pub fn dfl_draft_into(&mut self, anchor_d: *const i32, pos_d: *const i32, out_d: *mut i32) {
        let embd = self.model().embd;
        let embd_type = self.model().embd_type;
        let embd_row = self.model().embd_row;
        let ds_ptr = self.ds_d;
        let (tau_d, seed_d, ctr_d) = if self.dfl_sample {
            unsafe {
                (
                    ptr::addr_of!((*ds_ptr).tau),
                    ptr::addr_of!((*ds_ptr).seed),
                    ptr::addr_of_mut!((*ds_ptr).ctr),
                )
            }
        } else {
            (ptr::null(), ptr::null(), ptr::null_mut())
        };
        self.df.draft(
            embd,
            embd_type,
            embd_row,
            anchor_d,
            pos_d,
            out_d,
            tau_d,
            seed_d,
            ctr_d,
            self.s_main,
        );
    }

    /// Copy DFlash2 selector weights and sampled `q` to host (`Engine::dfl_scores` in `q36_engine.cu:1855-1858`).
    pub fn dfl_scores(&mut self) {
        if self.dl.want_conf() || self.dl.dfl_stop() {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.hconf.add(16).cast::<c_void>(),
                    self.df.scores().cast::<c_void>(),
                    (self.df.n_draft().max(0) as usize) * 4,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    self.s_main,
                );
            }
        }
        if self.dfl_sample && unsafe { (*self.ds_h).tau > 0.0 } {
            self.df.q_to_host(self.s_main);
        }
    }

    /// Advance DFlash2 sampling noise counter (`Engine::dfl_samp_prep` in `q36_engine.cu:1866-1871`).
    pub fn dfl_samp_prep(&mut self) {
        if !self.dfl_sample {
            return;
        }
        unsafe {
            (*self.ds_h).ctr = (*self.ds_h).ctr.wrapping_add(1);
            let _ = hipMemcpyAsync(
                self.ds_d.cast::<c_void>(),
                self.ds_h.cast::<c_void>(),
                std::mem::size_of::<tk_q36_draft_samp>(),
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }
    }

    /// Capture device speculative round graphs (`gA`, `gB`, `gB0`, `gC`, `gS`, `gDs`, `gD`, `gD0`)
    /// matching `Engine` in `q36_engine.cu:1737-1900`.
    pub fn ensure_round_graphs(
        &mut self,
        N: usize,
        _samp_active: bool,
        use_dflash: bool,
    ) -> Result<()> {
        let dmax = N.max(self.dl.dmax() as usize).clamp(1, 15);
        let max_t = if use_dflash && self.df.ready {
            self.df.block().clamp(1, 16) as usize
        } else {
            (dmax + 1).clamp(1, 16)
        };

        for t in 1..=max_t {
            if self.gA[t].is_null() {
                let exec = self.capture_graph(|eng| {
                    eng.forward_full(eng.tok_d, t as i32, eng.pos0_d, t as i32, true);
                })?;
                self.gA[t] = exec;
            }
        }

        if use_dflash && self.df.ready {
            let E = self.d_model() as usize;
            if self.gDs.is_null() {
                let exec = self.capture_graph(|eng| {
                    let out_d = unsafe { eng.tok_d.add(1) };
                    eng.dfl_draft_into(eng.tok_d, eng.pos0_d, out_d);
                })?;
                self.gDs = exec;
            }
            for t in 1..=max_t {
                if self.gD[t].is_null() {
                    let exec = self.capture_graph(|eng| {
                        let stream = eng.s_main.cast::<c_void>();
                        eng.df.inject(t as i32, eng.pos0_d, eng.s_main);
                        let _ = tk_q36_round_prep(
                            eng.tk.as_ptr(),
                            eng.rctl,
                            eng.tok_d,
                            eng.tok_m_d,
                            eng.pos0_d,
                            eng.pos0_m_d,
                            eng.gctl,
                            t as i32,
                            stream,
                        );
                        let _ = tk_q36_row_copy(
                            eng.tk.as_ptr(),
                            eng.h_mtp,
                            eng.h_last,
                            eng.rctl,
                            E as i32,
                            stream,
                        );
                        let out_d = unsafe { eng.tok_d.add(1) };
                        eng.dfl_draft_into(eng.tok_d, eng.pos0_d, out_d);
                    })?;
                    self.gD[t] = exec;
                }
                if self.gD0[t].is_null() {
                    let exec = self.capture_graph(|eng| {
                        let stream = eng.s_main.cast::<c_void>();
                        eng.df.inject(t as i32, eng.pos0_d, eng.s_main);
                        let _ = tk_q36_round_prep(
                            eng.tk.as_ptr(),
                            eng.rctl,
                            eng.tok_d,
                            eng.tok_m_d,
                            eng.pos0_d,
                            eng.pos0_m_d,
                            eng.gctl,
                            t as i32,
                            stream,
                        );
                        let _ = tk_q36_row_copy(
                            eng.tk.as_ptr(),
                            eng.h_mtp,
                            eng.h_last,
                            eng.rctl,
                            E as i32,
                            stream,
                        );
                    })?;
                    self.gD0[t] = exec;
                }
            }
        } else if self.model().has_mtp {
            let E = self.d_model() as usize;
            for t in 1..=max_t {
                if self.gB0[t].is_null() {
                    let exec = self.capture_graph(|eng| {
                        let stream = eng.s_main.cast::<c_void>();
                        let _ = tk_q36_round_prep(
                            eng.tk.as_ptr(),
                            eng.rctl,
                            eng.tok_d,
                            eng.tok_m_d,
                            eng.pos0_d,
                            eng.pos0_m_d,
                            eng.gctl,
                            t as i32,
                            stream,
                        );
                        let _ = tk_q36_row_copy(
                            eng.tk.as_ptr(),
                            eng.h_mtp,
                            eng.h_last,
                            eng.rctl,
                            E as i32,
                            stream,
                        );
                        eng.forward_mtp_batch(
                            t as i32,
                            0,
                            eng.h_last,
                            false,
                            true,
                            ptr::null_mut(),
                            -1,
                        );
                    })?;
                    self.gB0[t] = exec;
                }
                for d in 1..=dmax {
                    if self.gB[t][d].is_null() {
                        let exec = self.capture_graph(|eng| {
                            let stream = eng.s_main.cast::<c_void>();
                            let _ = tk_q36_round_prep(
                                eng.tk.as_ptr(),
                                eng.rctl,
                                eng.tok_d,
                                eng.tok_m_d,
                                eng.pos0_d,
                                eng.pos0_m_d,
                                eng.gctl,
                                t as i32,
                                stream,
                            );
                            let _ = tk_q36_row_copy(
                                eng.tk.as_ptr(),
                                eng.h_mtp,
                                eng.h_last,
                                eng.rctl,
                                E as i32,
                                stream,
                            );
                            eng.forward_mtp_batch(
                                t as i32,
                                0,
                                eng.h_last,
                                false,
                                true,
                                ptr::null_mut(),
                                -1,
                            );
                            let _ = tk_q36_row_copy(
                                eng.tk.as_ptr(),
                                eng.h_mtp_prev,
                                eng.h_mtp_b,
                                eng.rctl,
                                E as i32,
                                stream,
                            );
                            let dst1 = unsafe { eng.tok_d.add(1) };
                            eng.draft_head_step(eng.h_mtp_prev, 0, dst1);
                            eng.draft_chain(d as i32);
                        })?;
                        self.gB[t][d] = exec;
                    }
                }
            }
            for d in 2..=dmax {
                if self.gC[d].is_null() {
                    let exec = self.capture_graph(|eng| {
                        eng.draft_chain(d as i32);
                    })?;
                    self.gC[d] = exec;
                }
            }
            for n in 2..=dmax {
                if self.gS[n].is_null() {
                    let exec = self.capture_graph(|eng| {
                        eng.chain_step(n as i32);
                    })?;
                    self.gS[n] = exec;
                }
            }
        }
        Ok(())
    }

    // ========================================================================
    // Speculative Round Primitives (`spec_start`, `spec_verify`, `spec_advance`, `spec_finish`, `spec_step`)
    // ========================================================================

    /// Start a DFlash2 speculative window (`Engine::dfl_start` in `q36_engine.cu:1843-1860`).
    pub fn dfl_start(&mut self, x_tok: i32) {
        self.flush_gdn();
        let p0 = self.n_past;
        let samp_on = unsafe { (*self.ds_h).tau > 0.0 };
        unsafe {
            let _ = hipStreamSynchronize(self.s_main);
            *self.hpin.add(0) = x_tok;
            *self.hpin.add(1) = p0;
            let _ = hipMemcpyAsync(
                self.tok_d.cast::<c_void>(),
                self.hpin.cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
            let _ = hipMemcpyAsync(
                self.pos0_d.cast::<c_void>(),
                self.hpin.add(1).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }
        if self.model().has_mtp && self.dp.mtp_rows_in_dfl(p0 as i64, samp_on) {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.tok_m_d.cast::<c_void>(),
                    self.hpin.cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
                let _ = hipMemcpyAsync(
                    self.pos0_m_d.cast::<c_void>(),
                    self.hpin.add(1).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
            }
            self.forward_mtp_batch(1, p0 as i64, self.h_mtp, false, false, ptr::null_mut(), -1);
        }
        self.dfl_samp_prep();
        if self.opts.use_graph && !self.gDs.is_null() {
            unsafe {
                let _ = hipGraphLaunch(self.gDs, self.s_main);
            }
        } else {
            let out_d = unsafe { self.tok_d.add(1) };
            self.dfl_draft_into(self.tok_d, self.pos0_d, out_d);
        }
        self.dfl_scores();
        self.win_t = self.df.block();
        self.dfl_win = true;
        self.last_mtp = false;
        self.win_chain = self.df.n_draft();
    }

    /// Start an MTP or DFlash2 speculative window from `x_tok` (`Engine::spec_start` in `q36_engine.cu:1737-1768`).
    pub fn spec_start_window(&mut self, x_tok: i32, mut D: i32) {
        let samp_on = unsafe { (*self.ds_h).tau > 0.0 };
        if self.use_dflash && self.df.ready {
            let p0 = self.n_past as i64;
            self.dctx.at(p0);
            self.dp.request_start();
            if self.dp.want_dfl(p0, samp_on) {
                if self.dctx.valid(p0) || !self.model().has_mtp {
                    self.dfl_start(x_tok);
                    return;
                }
                self.dp.fallback(p0);
            }
            self.dfl_win = false;
            if !self.model().has_mtp {
                self.dfl_start(x_tok);
                return;
            }
        }
        if self.dl.mode() != 0 {
            D = self.dl.chain_len(samp_on);
        }
        let D = D.clamp(1, 15);
        let E = self.d_model() as usize;
        self.flush_gdn();
        let p0 = self.n_past;
        unsafe {
            let _ = hipStreamSynchronize(self.s_main);
            *self.hpin.add(0) = x_tok;
            *self.hpin.add(1) = p0;
            let _ = hipMemcpyAsync(
                self.tok_d.cast::<c_void>(),
                self.hpin.cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
            let _ = hipMemcpyAsync(
                self.tok_m_d.cast::<c_void>(),
                self.hpin.cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
            let _ = hipMemcpyAsync(
                self.pos0_d.cast::<c_void>(),
                self.hpin.add(1).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
            let _ = hipMemcpyAsync(
                self.pos0_m_d.cast::<c_void>(),
                self.hpin.add(1).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
            (*self.ds_h).ctr = (*self.ds_h).ctr.wrapping_add(1);
            let _ = hipMemcpyAsync(
                self.ds_d.cast::<c_void>(),
                self.ds_h.cast::<c_void>(),
                std::mem::size_of::<tk_q36_draft_samp>(),
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }
        let dst1 = unsafe { self.tok_d.add(1) };
        self.forward_mtp_batch(1, p0 as i64, self.h_mtp, D > 0, false, dst1, 0);
        unsafe {
            let _ = hipMemcpyAsync(
                self.h_mtp_prev.cast::<c_void>(),
                self.h_mtp_b.cast::<c_void>(),
                E * 4,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
                self.s_main,
            );
        }
        if D > 1 {
            let idx = D as usize;
            if self.opts.use_graph && !self.model().hybrid && self.gC[idx].is_null() {
                if let Ok(exec) = self.capture_graph(|eng| {
                    eng.draft_chain(D);
                }) {
                    self.gC[idx] = exec;
                }
            }
            if self.opts.use_graph && !self.gC[idx].is_null() {
                unsafe {
                    let _ = hipGraphLaunch(self.gC[idx], self.s_main);
                }
            } else {
                self.draft_chain(D);
            }
        }
        self.conf_to_host();
        self.win_chain = D;
        self.win_t = D + 1;
        self.last_mtp = true;
        self.dfl_win = false;
    }

    /// Verify current device speculative window (`Engine::spec_verify` in `q36_engine.cu:1762-1774`).
    pub fn spec_verify_window(
        &mut self,
        D_out: &mut i32,
        win: &mut [i32; 16],
        picks: &mut [i32; 16],
    ) {
        let samp_on = unsafe { (*self.ds_h).tau > 0.0 };
        if self.dl.stop_mode() && self.last_mtp && self.win_t > 1 {
            self.win_t = 1 + self.stop_extend(self.win_t - 1);
        }
        if self.use_dflash && self.df.ready && self.dfl_win && self.dl.dfl_stop() {
            let _ = unsafe { hipStreamSynchronize(self.s_main) };
            let top_slice = unsafe { std::slice::from_raw_parts(self.hconf.add(16), 16) };
            self.win_t = 1 + self.dl.keep_dfl(top_slice, self.win_t - 1, samp_on);
        }
        let T = self.win_t.clamp(1, 16);
        *D_out = T - 1;
        let idx = T as usize;
        if self.opts.use_graph && !self.model().hybrid && self.gA[idx].is_null() {
            if let Ok(exec) = self.capture_graph(|eng| {
                eng.forward_full(eng.tok_d, T, eng.pos0_d, T, true);
            }) {
                self.gA[idx] = exec;
            }
        }
        if self.opts.use_graph && !self.gA[idx].is_null() {
            unsafe {
                let _ = hipGraphLaunch(self.gA[idx], self.s_main);
            }
        } else {
            self.forward_full(self.tok_d, T, self.pos0_d, T, true);
        }
        unsafe {
            let _ = hipMemcpyAsync(
                self.hpin.add(16).cast::<c_void>(),
                self.tok_d.cast::<c_void>(),
                (T as usize) * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
                self.s_main,
            );
            let _ = hipMemcpyAsync(
                self.hpin.add(32).cast::<c_void>(),
                self.pick_v.cast::<c_void>(),
                (T as usize) * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
                self.s_main,
            );
            let _ = hipStreamSynchronize(self.s_main);
            for i in 0..(T as usize) {
                win[i] = *self.hpin.add(16 + i);
                picks[i] = *self.hpin.add(32 + i);
            }
        }
    }

    /// Advance a DFlash2 speculative round (`Engine::dfl_advance` in `q36_engine.cu:1874-1906`).
    pub fn dfl_advance(&mut self, a: i32, xn: i32, ng_opt: Option<&[i32]>, mrows: bool) {
        let E = self.d_model() as usize;
        let T = self.win_t.clamp(1, 16);
        let t_idx = T as usize;
        let use_ng = ng_opt.is_some_and(|ng| !ng.is_empty());
        let samp_on = unsafe { (*self.ds_h).tau > 0.0 };
        let run_mrows = mrows && self.model().has_mtp;
        if self.dfl_win && self.dl.want_conf() {
            let top_slice = unsafe { std::slice::from_raw_parts(self.hconf.add(16), 16) };
            self.dl.observe(
                T - 1,
                self.df.n_draft(),
                a,
                top_slice,
                top_slice,
                samp_on,
                true,
            );
        }
        unsafe {
            *self.rctl_h.add(0) = a;
            *self.rctl_h.add(1) = xn;
            let _ = hipMemcpyAsync(
                self.rctl.cast::<c_void>(),
                self.rctl_h.cast::<c_void>(),
                8,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }
        if !use_ng {
            self.dfl_samp_prep();
        }
        if use_ng {
            let ng = ng_opt.unwrap();
            let ng_len = ng.len().min(15);
            let slot_null = if run_mrows {
                self.gD0m[t_idx].is_null()
            } else {
                self.gD0[t_idx].is_null()
            };
            if self.opts.use_graph && !self.model().hybrid && slot_null {
                if let Ok(exec) = self.capture_graph(|eng| {
                    let stream = eng.s_main.cast::<c_void>();
                    eng.df.inject(T, eng.pos0_d, eng.s_main);
                    let _ = tk_q36_round_prep(
                        eng.tk.as_ptr(),
                        eng.rctl,
                        eng.tok_d,
                        eng.tok_m_d,
                        eng.pos0_d,
                        eng.pos0_m_d,
                        eng.gctl,
                        T,
                        stream,
                    );
                    let _ = tk_q36_row_copy(
                        eng.tk.as_ptr(),
                        eng.h_mtp,
                        eng.h_last,
                        eng.rctl,
                        E as i32,
                        stream,
                    );
                    if run_mrows {
                        eng.forward_mtp_batch(T, 0, eng.h_last, false, true, ptr::null_mut(), -1);
                    }
                }) {
                    if run_mrows {
                        self.gD0m[t_idx] = exec;
                    } else {
                        self.gD0[t_idx] = exec;
                    }
                }
            }
            let gx = if run_mrows {
                self.gD0m[t_idx]
            } else {
                self.gD0[t_idx]
            };
            if self.opts.use_graph && !gx.is_null() {
                unsafe {
                    let _ = hipGraphLaunch(gx, self.s_main);
                }
            } else {
                let stream = self.s_main.cast::<c_void>();
                self.df.inject(T, self.pos0_d, self.s_main);
                let _ = tk_q36_round_prep(
                    self.tk.as_ptr(),
                    self.rctl,
                    self.tok_d,
                    self.tok_m_d,
                    self.pos0_d,
                    self.pos0_m_d,
                    self.gctl,
                    T,
                    stream,
                );
                let _ = tk_q36_row_copy(
                    self.tk.as_ptr(),
                    self.h_mtp,
                    self.h_last,
                    self.rctl,
                    E as i32,
                    stream,
                );
                if run_mrows {
                    self.forward_mtp_batch(
                        T,
                        (self.n_past + 1) as i64,
                        self.h_last,
                        false,
                        true,
                        ptr::null_mut(),
                        -1,
                    );
                }
            }
            unsafe {
                for i in 0..ng_len {
                    *self.hpin.add(64 + i) = ng[i];
                }
                let _ = hipMemcpyAsync(
                    self.tok_d.add(1).cast::<c_void>(),
                    self.hpin.add(64).cast::<c_void>(),
                    ng_len * 4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
            }
            self.win_t = (ng_len as i32) + 1;
            self.win_chain = 0;
        } else {
            let slot_null = if run_mrows {
                self.gDm[t_idx].is_null()
            } else {
                self.gD[t_idx].is_null()
            };
            if self.opts.use_graph && !self.model().hybrid && slot_null {
                if let Ok(exec) = self.capture_graph(|eng| {
                    let stream = eng.s_main.cast::<c_void>();
                    eng.df.inject(T, eng.pos0_d, eng.s_main);
                    let _ = tk_q36_round_prep(
                        eng.tk.as_ptr(),
                        eng.rctl,
                        eng.tok_d,
                        eng.tok_m_d,
                        eng.pos0_d,
                        eng.pos0_m_d,
                        eng.gctl,
                        T,
                        stream,
                    );
                    let _ = tk_q36_row_copy(
                        eng.tk.as_ptr(),
                        eng.h_mtp,
                        eng.h_last,
                        eng.rctl,
                        E as i32,
                        stream,
                    );
                    if run_mrows {
                        eng.forward_mtp_batch(T, 0, eng.h_last, false, true, ptr::null_mut(), -1);
                    }
                    let out_d = unsafe { eng.tok_d.add(1) };
                    eng.dfl_draft_into(eng.tok_d, eng.pos0_d, out_d);
                }) {
                    if run_mrows {
                        self.gDm[t_idx] = exec;
                    } else {
                        self.gD[t_idx] = exec;
                    }
                }
            }
            let gx = if run_mrows {
                self.gDm[t_idx]
            } else {
                self.gD[t_idx]
            };
            if self.opts.use_graph && !gx.is_null() {
                unsafe {
                    let _ = hipGraphLaunch(gx, self.s_main);
                }
            } else {
                let stream = self.s_main.cast::<c_void>();
                self.df.inject(T, self.pos0_d, self.s_main);
                let _ = tk_q36_round_prep(
                    self.tk.as_ptr(),
                    self.rctl,
                    self.tok_d,
                    self.tok_m_d,
                    self.pos0_d,
                    self.pos0_m_d,
                    self.gctl,
                    T,
                    stream,
                );
                let _ = tk_q36_row_copy(
                    self.tk.as_ptr(),
                    self.h_mtp,
                    self.h_last,
                    self.rctl,
                    E as i32,
                    stream,
                );
                if run_mrows {
                    self.forward_mtp_batch(
                        T,
                        (self.n_past + 1) as i64,
                        self.h_last,
                        false,
                        true,
                        ptr::null_mut(),
                        -1,
                    );
                }
                let out_d = unsafe { self.tok_d.add(1) };
                self.dfl_draft_into(self.tok_d, self.pos0_d, out_d);
            }
            self.dfl_scores();
            self.win_t = self.df.block();
            self.win_chain = self.df.n_draft();
        }
        self.dfl_win = !use_ng;
        self.last_mtp = false;
        self.n_past += a + 1;
        self.gdn_pending = a + 1;
        self.gdn_valid = false;
    }

    /// Advance a speculative round after accepting `a` drafts and bonus `xn` (`Engine::spec_advance` in `q36_engine.cu:1797-1848`).
    pub fn spec_advance_window(&mut self, a: i32, xn: i32, mut D: i32, ng_opt: Option<&[i32]>) {
        let E = self.d_model() as usize;
        let T = self.win_t.clamp(1, 16);
        let t_idx = T as usize;
        let samp_on = unsafe { (*self.ds_h).tau > 0.0 };
        if self.last_mtp && self.dl.want_conf() {
            let conf_slice = unsafe { std::slice::from_raw_parts(self.hconf, 16) };
            let top_slice = unsafe { std::slice::from_raw_parts(self.hconf.add(16), 16) };
            self.dl.observe(
                T - 1,
                self.win_chain,
                a,
                conf_slice,
                top_slice,
                samp_on,
                false,
            );
        }
        let mut inj = false;
        if self.use_dflash && self.df.ready {
            let p_next = (self.n_past + a + 1) as i64;
            let want = self.dp.want_dfl(p_next, samp_on);
            if self.dfl_win || self.last_mtp {
                self.dp
                    .observe(self.dfl_win, T - 1, a, self.n_past as i64, samp_on);
            }
            inj = want || self.dp.inject_in_mtp(p_next, samp_on);
            if inj {
                self.dctx.inject(self.n_past as i64, a + 1, T);
            }
            if want && (self.dctx.valid(p_next) || !self.model().has_mtp) {
                let mrows = self.dp.mtp_rows_in_dfl(p_next, samp_on);
                self.dfl_advance(a, xn, ng_opt, mrows);
                return;
            }
            if want {
                self.dp.fallback(p_next);
            }
            self.dfl_win = false;
            if !self.model().has_mtp {
                self.dfl_advance(a, xn, ng_opt, false);
                return;
            }
        }
        unsafe {
            *self.rctl_h.add(0) = a;
            *self.rctl_h.add(1) = xn;
            let _ = hipMemcpyAsync(
                self.rctl.cast::<c_void>(),
                self.rctl_h.cast::<c_void>(),
                8,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
            (*self.ds_h).ctr = (*self.ds_h).ctr.wrapping_add(1);
            let _ = hipMemcpyAsync(
                self.ds_d.cast::<c_void>(),
                self.ds_h.cast::<c_void>(),
                std::mem::size_of::<tk_q36_draft_samp>(),
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }
        if let Some(ng) = ng_opt.filter(|s| !s.is_empty()) {
            let ng_len = ng.len().min(15);
            let slot_null = if inj {
                self.gB0i[t_idx].is_null()
            } else {
                self.gB0[t_idx].is_null()
            };
            if self.opts.use_graph && !self.model().hybrid && slot_null {
                if let Ok(exec) = self.capture_graph(|eng| {
                    let stream = eng.s_main.cast::<c_void>();
                    if inj {
                        eng.df.inject(T, eng.pos0_d, eng.s_main);
                    }
                    let _ = tk_q36_round_prep(
                        eng.tk.as_ptr(),
                        eng.rctl,
                        eng.tok_d,
                        eng.tok_m_d,
                        eng.pos0_d,
                        eng.pos0_m_d,
                        eng.gctl,
                        T,
                        stream,
                    );
                    let _ = tk_q36_row_copy(
                        eng.tk.as_ptr(),
                        eng.h_mtp,
                        eng.h_last,
                        eng.rctl,
                        E as i32,
                        stream,
                    );
                    eng.forward_mtp_batch(T, 0, eng.h_last, false, true, ptr::null_mut(), -1);
                }) {
                    if inj {
                        self.gB0i[t_idx] = exec;
                    } else {
                        self.gB0[t_idx] = exec;
                    }
                }
            }
            let gx = if inj {
                self.gB0i[t_idx]
            } else {
                self.gB0[t_idx]
            };
            if self.opts.use_graph && !gx.is_null() {
                unsafe {
                    let _ = hipGraphLaunch(gx, self.s_main);
                }
            } else {
                let stream = self.s_main.cast::<c_void>();
                if inj {
                    self.df.inject(T, self.pos0_d, self.s_main);
                }
                let _ = tk_q36_round_prep(
                    self.tk.as_ptr(),
                    self.rctl,
                    self.tok_d,
                    self.tok_m_d,
                    self.pos0_d,
                    self.pos0_m_d,
                    self.gctl,
                    T,
                    stream,
                );
                let _ = tk_q36_row_copy(
                    self.tk.as_ptr(),
                    self.h_mtp,
                    self.h_last,
                    self.rctl,
                    E as i32,
                    stream,
                );
                self.forward_mtp_batch(
                    T,
                    (self.n_past + 1) as i64,
                    self.h_last,
                    false,
                    true,
                    ptr::null_mut(),
                    -1,
                );
            }
            unsafe {
                for i in 0..ng_len {
                    *self.hpin.add(64 + i) = ng[i];
                }
                let _ = hipMemcpyAsync(
                    self.tok_d.add(1).cast::<c_void>(),
                    self.hpin.add(64).cast::<c_void>(),
                    ng_len * 4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
            }
            self.win_t = (ng_len as i32) + 1;
            self.last_mtp = false;
            self.dfl_win = false;
            self.win_chain = 0;
            self.n_past += a + 1;
            self.gdn_pending = a + 1;
            self.gdn_valid = false;
            return;
        }
        if self.dl.mode() != 0 {
            D = self.dl.chain_len(samp_on);
        }
        let D = D.clamp(1, 15);
        self.win_t = D + 1;
        self.win_chain = D;
        self.last_mtp = true;
        self.dfl_win = false;
        let d_idx = D as usize;
        let slot_null = if inj {
            self.gBi[t_idx][d_idx].is_null()
        } else {
            self.gB[t_idx][d_idx].is_null()
        };
        if self.opts.use_graph && !self.model().hybrid && slot_null {
            if let Ok(exec) = self.capture_graph(|eng| {
                let stream = eng.s_main.cast::<c_void>();
                if inj {
                    eng.df.inject(T, eng.pos0_d, eng.s_main);
                }
                let _ = tk_q36_round_prep(
                    eng.tk.as_ptr(),
                    eng.rctl,
                    eng.tok_d,
                    eng.tok_m_d,
                    eng.pos0_d,
                    eng.pos0_m_d,
                    eng.gctl,
                    T,
                    stream,
                );
                let _ = tk_q36_row_copy(
                    eng.tk.as_ptr(),
                    eng.h_mtp,
                    eng.h_last,
                    eng.rctl,
                    E as i32,
                    stream,
                );
                eng.forward_mtp_batch(T, 0, eng.h_last, false, true, ptr::null_mut(), -1);
                let _ = tk_q36_row_copy(
                    eng.tk.as_ptr(),
                    eng.h_mtp_prev,
                    eng.h_mtp_b,
                    eng.rctl,
                    E as i32,
                    stream,
                );
                let dst1 = unsafe { eng.tok_d.add(1) };
                eng.draft_head_step(eng.h_mtp_prev, 0, dst1);
                eng.draft_chain(D);
            }) {
                if inj {
                    self.gBi[t_idx][d_idx] = exec;
                } else {
                    self.gB[t_idx][d_idx] = exec;
                }
            }
        }
        let gx = if inj {
            self.gBi[t_idx][d_idx]
        } else {
            self.gB[t_idx][d_idx]
        };
        if self.opts.use_graph && !gx.is_null() {
            unsafe {
                let _ = hipGraphLaunch(gx, self.s_main);
            }
        } else {
            let stream = self.s_main.cast::<c_void>();
            if inj {
                self.df.inject(T, self.pos0_d, self.s_main);
            }
            let _ = tk_q36_round_prep(
                self.tk.as_ptr(),
                self.rctl,
                self.tok_d,
                self.tok_m_d,
                self.pos0_d,
                self.pos0_m_d,
                self.gctl,
                T,
                stream,
            );
            let _ = tk_q36_row_copy(
                self.tk.as_ptr(),
                self.h_mtp,
                self.h_last,
                self.rctl,
                E as i32,
                stream,
            );
            self.forward_mtp_batch(
                T,
                (self.n_past + 1) as i64,
                self.h_last,
                false,
                true,
                ptr::null_mut(),
                -1,
            );
            let _ = tk_q36_row_copy(
                self.tk.as_ptr(),
                self.h_mtp_prev,
                self.h_mtp_b,
                self.rctl,
                E as i32,
                stream,
            );
            let dst1 = unsafe { self.tok_d.add(1) };
            self.draft_head_step(self.h_mtp_prev, 0, dst1);
            self.draft_chain(D);
        }
        self.conf_to_host();
        self.n_past += a + 1;
        self.gdn_pending = a + 1;
        self.gdn_valid = false;
    }

    /// Finish a speculative run cleanly at stop/EOS (`Engine::spec_finish` in `q36_engine.cu:1908-1920`).
    pub fn spec_finish_window(&mut self, a: i32, xn: i32) {
        let E = self.d_model() as usize;
        if self.use_dflash && self.df.ready {
            self.df.inject(self.win_t, self.pos0_d, self.s_main);
            self.dctx.inject(self.n_past as i64, a + 1, self.win_t);
        }
        unsafe {
            *self.rctl_h.add(0) = a;
            *self.rctl_h.add(1) = xn;
            let _ = hipMemcpyAsync(
                self.rctl.cast::<c_void>(),
                self.rctl_h.cast::<c_void>(),
                8,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }
        let stream = self.s_main.cast::<c_void>();
        let _ = tk_q36_round_prep(
            self.tk.as_ptr(),
            self.rctl,
            self.tok_d,
            self.tok_m_d,
            self.pos0_d,
            self.pos0_m_d,
            self.gctl,
            0,
            stream,
        );
        let _ = tk_q36_row_copy(
            self.tk.as_ptr(),
            self.h_mtp,
            self.h_last,
            self.rctl,
            E as i32,
            stream,
        );
        self.gdn_pending = a + 1;
        self.flush_gdn();
        self.n_past += a + 1;
        let _ = unsafe { hipStreamSynchronize(self.s_main) };
    }

    /// Legacy host-driven `spec_start` helper.
    pub fn spec_start(&mut self, first_tok: i32) -> i32 {
        self.flush_gdn();
        let p0 = self.n_past;
        unsafe {
            let _ = hipMemcpy(
                self.tok_d.cast::<c_void>(),
                (&first_tok as *const i32).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
            let _ = hipMemcpy(
                self.pos0_d.cast::<c_void>(),
                (&p0 as *const i32).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
        }
        self.forward_full(self.tok_d, 1, self.pos0_d, 1, false);
        unsafe {
            let _ = hipMemcpy(
                self.pick_v_h.cast::<c_void>(),
                self.pick_v.cast::<c_void>(),
                4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = hipMemcpy(
                self.logits_h.cast::<c_void>(),
                self.logits.cast::<c_void>(),
                (self.n_vocab() as usize) * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = hipMemcpy(
                self.h_mtp.cast::<c_void>(),
                self.h_last.cast::<c_void>(),
                (self.d_model() as usize) * 4,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            );
            let _ = hipDeviceSynchronize();
        }
        self.gdn_valid = true;
        self.n_past += 1;
        unsafe { *self.pick_v_h }
    }

    /// Run target verification on `toks[0..n]` starting at `n_past`, returning `&logits_v_h[0..n*n_vocab]`.
    pub fn spec_verify(&mut self, toks: &[i32], n: i32) -> &[f32] {
        let n_clamped = n.clamp(1, 16) as usize;
        assert!(toks.len() >= n_clamped, "spec_verify: toks shorter than n");
        self.flush_gdn();

        let p0 = self.n_past;
        unsafe {
            let _ = hipMemcpy(
                self.tok_d.cast::<c_void>(),
                toks.as_ptr().cast::<c_void>(),
                n_clamped * 4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
            let _ = hipMemcpy(
                self.pos0_d.cast::<c_void>(),
                (&p0 as *const i32).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
            let _ = hipMemset(self.gctl.cast::<c_void>(), 0, 16);
        }
        self.forward_full(
            self.tok_d,
            n_clamped as i32,
            self.pos0_d,
            n_clamped as i32,
            true,
        );

        let total_floats = n_clamped * (self.n_vocab() as usize);
        unsafe {
            let _ = hipMemcpy(
                self.logits_v_h.cast::<c_void>(),
                self.logits_v.cast::<c_void>(),
                total_floats * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = hipMemcpy(
                self.pick_v_h.cast::<c_void>(),
                self.pick_v.cast::<c_void>(),
                n_clamped * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = hipDeviceSynchronize();
            std::slice::from_raw_parts(self.logits_v_h, total_floats)
        }
    }

    /// Commit the accepted prefix (`n_accept` accepted draft tokens out of `n_tok` verified tokens).
    pub fn spec_advance(&mut self, _toks: &[i32], n_tok: i32, n_accept: i32) {
        let commit_n = (n_accept + 1).clamp(1, n_tok.max(1));
        self.commit(commit_n);
        let E = self.d_model() as usize;
        let row = (n_accept.max(0) as usize).min((n_tok.max(1) - 1) as usize);
        unsafe {
            let _ = hipMemcpyAsync(
                self.h_mtp.cast::<c_void>(),
                self.h_last.add(row * E).cast::<c_void>(),
                E * 4,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
                self.s_main,
            );
        }
        self.n_past += commit_n;
    }

    /// Finish a speculative round with `bonus_tok` and run MTP to produce the next draft token.
    pub fn spec_finish(
        &mut self,
        bonus_tok: i32,
        _bonus_from_verify: bool,
        _n_drafted: i32,
        _n_accept: i32,
    ) -> i32 {
        if !self.model().has_mtp {
            return bonus_tok;
        }
        let pm = (self.n_past - 1).max(0);
        unsafe {
            let _ = hipMemcpyAsync(
                self.tok_m_d.cast::<c_void>(),
                (&bonus_tok as *const i32).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
            let _ = hipMemcpyAsync(
                self.pos0_m_d.cast::<c_void>(),
                (&pm as *const i32).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }
        self.forward_mtp_batch(1, pm as i64, self.h_mtp, true, false, self.pick_m, -1);
        unsafe {
            let _ = hipMemcpy(
                self.pick_m_h.cast::<c_void>(),
                self.pick_m.cast::<c_void>(),
                4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = hipDeviceSynchronize();
            *self.pick_m_h
        }
    }

    /// Execute one non-graph MTP speculative round (`Engine::spec_step` in `q36_engine.cu:1636-1680`).
    pub fn spec_step(
        &mut self,
        draft_Out: &mut Vec<i32>,
        bonus_Out: &mut i32,
    ) -> (i32, Vec<i32>, i32) {
        let N = if draft_Out.len() > 1 {
            (draft_Out.len() - 1).min(15)
        } else {
            1
        };
        let anchor = draft_Out.first().copied().unwrap_or(*bonus_Out);
        if draft_Out.len() <= 1 && self.model().has_mtp {
            let d0 = self.spec_finish(anchor, false, 0, 0);
            draft_Out.clear();
            draft_Out.push(anchor);
            draft_Out.push(d0);
        }
        let nt = draft_Out.len().min(N + 1);
        let _ = self.spec_verify(&draft_Out[..nt], nt as i32);
        let mut acc = 0usize;
        let mut accepted = Vec::with_capacity(N);
        for i in 0..(nt - 1) {
            let v_pick = unsafe { *self.pick_v_h.add(i) };
            if v_pick == draft_Out[i + 1] {
                acc += 1;
                accepted.push(v_pick);
            } else {
                break;
            }
        }
        let bonus = unsafe { *self.pick_v_h.add(acc) };
        self.spec_advance(&draft_Out[..nt], nt as i32, acc as i32);
        *bonus_Out = bonus;
        *draft_Out = accepted.clone();
        (acc as i32, accepted, bonus)
    }

    /// Greedy speculative generation loop (`Engine::spec_greedy` in `q36_engine.cu:1966-1985`).
    pub fn spec_greedy(
        &mut self,
        n_spec: i32,
        max_new: i32,
        eos: &[i32],
        first_tok: i32,
        out: &mut Vec<i32>,
        n_drafted: &mut i64,
        n_accepted: &mut i64,
    ) {
        let greedy = Sampling {
            temp: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 1,
            rep_pen: 1.0,
            pres_pen: 0.0,
            freq_pen: 0.0,
            rep_last_n: 0,
            seed: 1,
        };
        let mut recent = Vec::new();
        self.spec_run(
            n_spec,
            max_new,
            eos,
            first_tok,
            &greedy,
            &mut recent,
            out,
            n_drafted,
            n_accepted,
            None,
        );
    }

    /// Full speculative decoding loop with HIP graphs `gA`/`gB`/`gB0`/`gC`/`gS`/`gD`/`gD0`/`gDs`
    /// (`Engine::spec_run` in `q36_engine.cu:1918-1965`).
    pub fn spec_run(
        &mut self,
        n_spec: i32,
        max_new: i32,
        eos: &[i32],
        first_tok: i32,
        samp: &Sampling,
        recent: &mut Vec<i32>,
        out: &mut Vec<i32>,
        n_drafted: &mut i64,
        n_accepted: &mut i64,
        ngram: Option<&mut NgramMod>,
    ) {
        self.spec_run_with_callback(
            n_spec,
            max_new,
            eos,
            first_tok,
            samp,
            recent,
            out,
            n_drafted,
            n_accepted,
            ngram,
            |_| true,
        );
    }

    /// Speculative decoding loop with a per-token emission callback (`on_token(tok) -> keep_going`),
    /// matching `Engine::spec_run` + `decide_sampled` in `q36_engine.cu:1918-1985` & `2788-2860`.
    #[allow(clippy::too_many_arguments)]
    pub fn spec_run_with_callback<F>(
        &mut self,
        n_spec: i32,
        max_new: i32,
        eos: &[i32],
        first_tok: i32,
        samp: &Sampling,
        recent: &mut Vec<i32>,
        out: &mut Vec<i32>,
        n_drafted: &mut i64,
        n_accepted: &mut i64,
        mut ngram: Option<&mut NgramMod>,
        mut on_token: F,
    ) where
        F: FnMut(i32) -> bool,
    {
        if max_new <= 0 {
            return;
        }
        let is_eos = |t: i32| eos.contains(&t);
        let D_init = if self.use_dflash && self.df.ready && !self.dp.uses_mtp() {
            self.df.n_draft().clamp(1, 15)
        } else {
            n_spec.clamp(1, 15)
        };
        let d_cap = D_init
            .max(self.dl.dmax())
            .max(if self.use_dflash && self.df.ready {
                self.df.n_draft()
            } else {
                0
            });

        // the copy index holds the request's context (prompt + output) as `recent` does
        if let Some(c) = self.copy.as_mut() {
            c.sync(recent);
            c.begin_request();
            let _ = c.take_stats();
            c.push(first_tok);
        }
        // `first_tok` (`x` in `q36_engine.cu:1945`) is already decided by `process(prompt)`!
        out.push(first_tok);
        recent.push(first_tok);
        if let Some(ref mut ng) = ngram {
            ng.feed(first_tok);
        }
        if !on_token(first_tok) || is_eos(first_tok) || (out.len() as i32) >= max_new {
            return;
        }
        if self.n_past + d_cap + 1 > self.ctx_len {
            return;
        }

        let mut rng = SplitMix64::new(samp.seed);
        let greedy = samp.temp <= 0.0 || samp.top_k == 1;
        if !greedy {
            // `first_tok` consumed the first draw from `mt19937_64(samp.seed)` in `sample()`.
            let _ = rng.next_u64();
        }
        let draft_greedy = std::env::var("STRATA_DRAFT_GREEDY").is_ok();
        let draft_minp = std::env::var("STRATA_DRAFT_MINP")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(-1.0);
        let eff_minp = if draft_minp >= 0.0 {
            draft_minp
        } else {
            samp.min_p
        };
        // The drafters' proposal temperature = the request's x TALUS_Q27_QTEMP (default 1.0; 0.5 is
        // opt-in). The MTP device draw, its support / lse (k_draft_support), the DFlash2 chain draw and the
        // host q (qof) all read this same tau, so q stays the distribution the drafts are drawn from: exact for any
        // scale; a sharper q (< 1) is kept more often when the drafter is less confident than the target.
        let tau = if !greedy && !draft_greedy {
            samp.temp * talus_spec::ssamp::q_temp_scale_or("TALUS_Q27_QTEMP", 1.0)
        } else {
            0.0f32
        };
        let lmin = if eff_minp > 0.0 && eff_minp < 1.0 {
            eff_minp.ln()
        } else {
            f32::NEG_INFINITY
        };
        let topp = if samp.top_p > 0.0 && samp.top_p < 1.0 {
            samp.top_p
        } else {
            1.0f32
        };
        let topk = if samp.top_k > 0 { samp.top_k } else { 0i32 };
        let seed = samp.seed as u32;
        self.draft_samp_on = tau > 0.0;
        unsafe {
            (*self.ds_h).tau = tau;
            (*self.ds_h).seed = seed;
            (*self.ds_h).ctr = 0;
            (*self.ds_h).lmin = lmin;
            (*self.ds_h).topp = topp;
            (*self.ds_h).topk = topk;
            let _ = hipMemcpyAsync(
                self.ds_d.cast::<c_void>(),
                self.ds_h.cast::<c_void>(),
                std::mem::size_of::<tk_q36_draft_samp>(),
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.s_main,
            );
        }

        if self.opts.use_graph && !self.model().hybrid {
            let _ = self.ensure_round_graphs(D_init as usize, !greedy, self.use_dflash);
        }

        self.spec_start_window(first_tok, D_init);

        let V = self.n_vocab() as usize;
        let vd = self.draft_n_vocab.max(0) as usize;
        let pen_on = samp.rep_last_n != 0
            && (samp.rep_pen != 1.0 || samp.freq_pen != 0.0 || samp.pres_pen != 0.0);
        let par_dist = !pen_on && std::env::var("STRATA_SAMPLE_SERIAL").is_err();
        let mut copy_buf: Vec<i32> = Vec::new();
        let mut copy_cur: Option<(talus_spec::CopyMatch, usize)> = None;
        let copy_row_cost = talus_spec::copy_env_f64("ROWCOST").unwrap_or(0.10).max(0.0);
        // a copy replaces the MTP / DFlash2 drafts: only buckets that keep >= this many copied tokens
        let copy_qmin = talus_spec::copy_env_f64("QMIN").unwrap_or(0.85);

        loop {
            let mut d_cur = D_init;
            let mut win = [0i32; 16];
            let mut picks = [0i32; 16];
            self.spec_verify_window(&mut d_cur, &mut win, &mut picks);

            let mut a = 0i32;
            let mut xn = 0i32;
            let td = std::time::Instant::now();

            if greedy {
                while a < d_cur && picks[a as usize] == win[(a + 1) as usize] {
                    a += 1;
                }
                xn = picks[a as usize];
            } else {
                let n_rows = (d_cur + 1) as usize;
                let qd = self.use_dflash
                    && self.df.ready
                    && self.dfl_win
                    && self.dfl_sample
                    && tau > 0.0;
                let qs = (self.last_mtp && tau > 0.0) || qd;
                unsafe {
                    let _ = hipMemcpyAsync(
                        self.logits_v_h.cast::<c_void>(),
                        self.logits.cast::<c_void>(),
                        n_rows * V * 4,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                        self.s_main,
                    );
                    if qs && !qd && d_cur > 0 {
                        let _ = hipMemcpyAsync(
                            self.logits_m_h.cast::<c_void>(),
                            self.logits_md.cast::<c_void>(),
                            (d_cur as usize) * vd * 4,
                            HIP_MEMCPY_DEVICE_TO_HOST,
                            self.s_main,
                        );
                        let _ = hipMemcpyAsync(
                            self.lse_v_h.cast::<c_void>(),
                            self.lse_md.cast::<c_void>(),
                            32 * 4,
                            HIP_MEMCPY_DEVICE_TO_HOST,
                            self.s_main,
                        );
                    }
                    let _ = hipStreamSynchronize(self.s_main);
                }

                let logits_all = unsafe { std::slice::from_raw_parts(self.logits_v_h, n_rows * V) };
                let q_all = if qs && !qd && d_cur > 0 {
                    unsafe { std::slice::from_raw_parts(self.logits_m_h, (d_cur as usize) * vd) }
                } else {
                    &[][..]
                };
                let hlse = unsafe { std::slice::from_raw_parts(self.lse_v_h, 32) };

                let pre_dists: Vec<Vec<(i32, f32)>> = if par_dist && n_rows > 1 {
                    let hist_ref: &[i32] = recent.as_slice();
                    std::thread::scope(|s| {
                        let mut handles = Vec::with_capacity(n_rows - 1);
                        for i in 1..n_rows {
                            let row_slice = &logits_all[i * V..(i + 1) * V];
                            handles.push(s.spawn(move || make_dist(row_slice, samp, hist_ref)));
                        }
                        let mut out_dists = Vec::with_capacity(n_rows);
                        out_dists.push(make_dist(&logits_all[0..V], samp, hist_ref));
                        for h in handles {
                            out_dists.push(h.join().unwrap_or_default());
                        }
                        out_dists
                    })
                } else {
                    Vec::new()
                };

                let qof = |i: usize, tokid: i32| -> f64 {
                    if qd {
                        return self.df.q_host(i, tokid);
                    }
                    let r = if self.dv_inv.is_empty() {
                        tokid
                    } else if tokid >= 0 && (tokid as usize) < self.dv_inv.len() {
                        self.dv_inv[tokid as usize]
                    } else {
                        -1
                    };
                    if r < 0 || (r as usize) >= vd {
                        return 0.0;
                    }
                    let lv = q_all[i * vd + (r as usize)];
                    let xs = lv / tau;
                    if !(xs >= hlse[16 + i]) {
                        return 0.0;
                    }
                    ((lv as f64) / (tau as f64) - (hlse[i] as f64)).exp()
                };

                let mut h2 = if par_dist { Vec::new() } else { recent.clone() };
                for i in 0..=d_cur as usize {
                    let mut p = if par_dist && i < pre_dists.len() {
                        pre_dists[i].clone()
                    } else {
                        make_dist(&logits_all[i * V..(i + 1) * V], samp, &h2)
                    };
                    if i == d_cur as usize {
                        a = d_cur;
                        xn = draw_dist(&p, &mut rng);
                        break;
                    }
                    let d = win[i + 1];
                    // the row's exact speculative-sampling decision (spec_decide_row; CPU-tested in tests.rs)
                    let decision = spec_decide_row(
                        &mut p,
                        d,
                        qs,
                        |t| qof(i, t),
                        &mut rng,
                        || {
                            if par_dist && i < pre_dists.len() {
                                pre_dists[i].clone()
                            } else {
                                make_dist(&logits_all[i * V..(i + 1) * V], samp, &h2)
                            }
                        },
                    );
                    match decision {
                        RowDecision::Accept => {
                            if !par_dist {
                                h2.push(d);
                            }
                            continue;
                        }
                        RowDecision::End(x) => {
                            a = i as i32;
                            xn = x;
                            break;
                        }
                    }
                }
            }
            self.st_decide_ms += td.elapsed().as_secs_f64() * 1000.0;
            self.st_rounds += 1;

            *n_drafted += d_cur as i64;
            *n_accepted += a as i64;
            if let (Some((m, k)), Some(c)) = (copy_cur.take(), self.copy.as_mut()) {
                c.observe(&m, k, (a.max(0) as usize).min(k), false);
            }

            for i in 1..=(a as usize) {
                let tok = win[i];
                out.push(tok);
                recent.push(tok);
                if let Some(c) = self.copy.as_mut() {
                    c.push(tok);
                }
                if let Some(ref mut ng) = ngram {
                    ng.feed(tok);
                }
                if !on_token(tok) || is_eos(tok) || (out.len() as i32) >= max_new {
                    self.spec_finish_window(i as i32, picks[i]);
                    return;
                }
            }

            out.push(xn);
            recent.push(xn);
            if let Some(c) = self.copy.as_mut() {
                c.push(xn);
            }
            if let Some(ref mut ng) = ngram {
                ng.feed(xn);
            }
            if !on_token(xn) || is_eos(xn) || (out.len() as i32) >= max_new {
                self.spec_finish_window(a, xn);
                return;
            }

            if self.n_past + (a + 1) + 16 > self.ctx_len {
                self.spec_finish_window(a, xn);
                return;
            }

            let k_max = (self.gemm_t - 1).min(self.ng_kmax).clamp(1, 15);
            let mut ng_vec = if let Some(ref mut ng) = ngram {
                ng.propose_ctx(recent, k_max, self.ng_kmin)
            } else {
                Vec::new()
            };
            // the n-gram lookup needs 24 matching tokens; a long-span copy fills in below that
            if ng_vec.is_empty() {
                if let Some(c) = self.copy.as_ref() {
                    if let Some(m) = c.propose(
                        &[],
                        k_max as usize,
                        !greedy,
                        !greedy,
                        |t| eos.contains(&t),
                        &mut copy_buf,
                    ) {
                        let k =
                            c.choose_len_min(&m, copy_buf.len(), copy_row_cost, 0.03, copy_qmin);
                        if k as i32 >= self.ng_kmin {
                            ng_vec.extend_from_slice(&copy_buf[..k]);
                            copy_cur = Some((m, k));
                        }
                    }
                }
            }
            let ng_slice = if ng_vec.is_empty() {
                None
            } else {
                Some(ng_vec.as_slice())
            };
            self.spec_advance_window(a, xn, D_init, ng_slice);
        }
    }
}
