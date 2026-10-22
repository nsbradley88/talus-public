//! Adaptive draft count controller (`DraftLen`).
//!
//! Ports `DraftLen` from `src/qwen35/q36_adraft_policy.{hpp,cpp}`, supporting both:
//! - The EMA acceptance-rate step controller (`enabled`, `ema_alpha`, `n_cur`, `n_max`,
//!   `ema_rate`, `rounds`, `downshifts`, `upshifts`, `sum_n`, `hist[9]`, `init`,
//!   `reset_request`, `choose`, `observe` with exact `f32` thresholds `>= 0.72f`, `< 0.50f`,
//!   `< 0.30f`, `< 0.45f`).
//! - The `q36_adraft_policy.cpp` confidence-gated / round-cost policy (`kMax = 15`, `init_arch`,
//!   `mode`, `want_conf`, `stop_mode`, `dmax`, `dmax_for`, `dfl_sample`, `chain_len`, `keep`,
//!   `keep_dfl`, `observe_window`, `dfl_stop`, `describe`).
//! - The [`talus_core_api::SpecController`] trait.

use talus_core_api::{RoundPlan, RoundStats, SpecController};

/// Maximum draft chain length supported by `hist[9]` (`0..=8`).
pub const MAX_HIST_DRAFTS: usize = 8;

/// Maximum draft window length for the `q36_adraft_policy` confidence/cost controller.
pub const K_MAX: usize = 15;

/// Default EMA smoothing factor (`ema_alpha`).
pub const DEFAULT_EMA_ALPHA: f32 = 0.25_f32;

/// Initial neutral EMA acceptance rate on `init` / `reset_request`.
pub const INITIAL_EMA_RATE: f32 = 0.60_f32;

/// Threshold above or equal to which `DraftLen::observe` steps up `n_cur` (`>= 0.72f`).
pub const THRESH_UPSHIFT: f32 = 0.72_f32;

/// Threshold below which `DraftLen::observe` steps down when acceptance is weak (`< 0.50f`).
pub const THRESH_DOWNSHIFT_MILD: f32 = 0.50_f32;

/// Threshold below which `DraftLen::observe` steps down `n_cur` by 1 (`< 0.45f`).
pub const THRESH_DOWNSHIFT: f32 = 0.45_f32;

/// Threshold below which `DraftLen::observe` steps down `n_cur` aggressively (`< 0.30f`).
pub const THRESH_DOWNSHIFT_FAST: f32 = 0.30_f32;

/// Strix Halo (`gfx1151`) measured MTP round-cost table (ms) for `D = 1..=8`.
const GFX1151_COST_MS: [f64; 8] = [72.1, 76.9, 82.1, 85.2, 90.6, 95.7, 101.0, 106.5];

/// Adaptive draft count controller (`DraftLen`).
#[derive(Debug, Clone)]
pub struct DraftLen {
    /// Whether adaptive draft length stepping is enabled (when `false`, `choose()` returns `n_max`).
    pub enabled: bool,
    /// Exponential moving average weight for each round's acceptance rate (`0.0..=1.0`).
    pub ema_alpha: f32,
    /// Current draft length (`1..=n_max`, or `0` when `n_max == 0`).
    pub n_cur: i32,
    /// Maximum configured draft length (`--mtp N`).
    pub n_max: i32,
    /// Running exponential moving average of acceptance rate (`accepted / drafted`).
    pub ema_rate: f32,
    /// Total speculative rounds observed.
    pub rounds: i64,
    /// Number of times `n_cur` was reduced.
    pub downshifts: i64,
    /// Number of times `n_cur` was increased.
    pub upshifts: i64,
    /// Cumulative sum of draft lengths proposed (`sum_n`).
    pub sum_n: i64,
    /// Histogram of draft lengths `0..=8`.
    pub hist: [i64; 9],

    // ---- q36_adraft_policy.hpp confidence & cost-curve fields ----
    mode_: i32,
    dfix_: i32,
    dmin_: i32,
    dmax_: i32,
    chain_: i32,
    pmin_: [f32; 2],
    cum_: [f32; 2],
    tries_: [[f64; K_MAX + 1]; 2],
    keeps_: [[f64; K_MAX + 1]; 2],
    cost_: [f64; K_MAX + 2],
    dfl_stop_: bool,
    dfl_sample_: bool,
    samp_: bool,
    dfl_dmin_: i32,
    dfl_cum_: [f32; 2],
}

impl Default for DraftLen {
    fn default() -> Self {
        Self::new()
    }
}

impl DraftLen {
    /// Create a default `DraftLen` controller before [`Self::init`].
    pub fn new() -> Self {
        let mut s = Self {
            enabled: false,
            ema_alpha: DEFAULT_EMA_ALPHA,
            n_cur: 0,
            n_max: 0,
            ema_rate: INITIAL_EMA_RATE,
            rounds: 0,
            downshifts: 0,
            upshifts: 0,
            sum_n: 0,
            hist: [0; 9],
            mode_: 0,
            dfix_: 0,
            dmin_: 1,
            dmax_: 7,
            chain_: 4,
            pmin_: [0.5, 0.0],
            cum_: [0.40, 0.45],
            tries_: [[0.0; K_MAX + 1]; 2],
            keeps_: [[0.0; K_MAX + 1]; 2],
            cost_: [0.0; K_MAX + 2],
            dfl_stop_: false,
            dfl_sample_: false,
            samp_: false,
            dfl_dmin_: 2,
            dfl_cum_: [0.40, 0.30],
        };
        s.init(0, false);
        s
    }

    /// Initialize the adaptive draft controller with `n_max` and `enabled`.
    pub fn init(&mut self, n_max: i32, enabled: bool) {
        let clamped_max = n_max.clamp(0, MAX_HIST_DRAFTS as i32);
        self.enabled = enabled && clamped_max > 1;
        if self.ema_alpha <= 0.0 || self.ema_alpha > 1.0 {
            self.ema_alpha = DEFAULT_EMA_ALPHA;
        }
        self.n_max = clamped_max;
        self.n_cur = clamped_max;
        self.ema_rate = INITIAL_EMA_RATE;
        self.rounds = 0;
        self.downshifts = 0;
        self.upshifts = 0;
        self.sum_n = 0;
        self.hist = [0; 9];
        self.dfix_ = clamped_max;
    }

    /// Initialize with explicit `ema_alpha`.
    pub fn init_with_alpha(&mut self, n_max: i32, enabled: bool, ema_alpha: f32) {
        self.ema_alpha = ema_alpha.clamp(0.01, 1.0);
        self.init(n_max, enabled);
    }

    /// Reset per-request adaptive state (`n_cur = n_max`, `ema_rate = INITIAL_EMA_RATE`).
    pub fn reset_request(&mut self) {
        self.n_cur = self.n_max;
        self.ema_rate = INITIAL_EMA_RATE;
    }

    /// Choose the draft count `n` (`0..=n_max`) for the upcoming speculative round.
    #[inline]
    pub fn choose(&mut self) -> i32 {
        if self.n_max <= 0 {
            return 0;
        }
        if !self.enabled {
            return self.n_max;
        }
        self.n_cur.clamp(1, self.n_max)
    }

    /// Observe the outcome of a speculative round that proposed `n_draft` tokens and accepted
    /// `n_accept` tokens (`0..=n_draft`), updating `ema_rate`, `n_cur`, and telemetry counters.
    ///
    /// Threshold rules (exact `f32` comparisons):
    /// - Instantaneous round rate `r = (n_accept as f32) / (n_draft as f32)`.
    /// - `ema_rate = (1.0 - ema_alpha) * ema_rate + ema_alpha * r` (seeded with `r` on round 0
    ///   when `rounds == 0`, or blended from `ema_rate`).
    /// - Step-up when `ema_rate >= 0.72f` (and `r >= 0.50f`): `n_cur = (n_cur + 1).min(n_max)`.
    /// - Fast step-down when `ema_rate < 0.30f`: `n_cur = (n_cur - 2).max(1)`.
    /// - Step-down by 1 when `ema_rate < 0.45f` or (`ema_rate < 0.50f` and `r < 0.30f`):
    ///   `n_cur = (n_cur - 1).max(1)`.
    pub fn observe(&mut self, n_draft: i32, n_accept: i32) {
        if n_draft <= 0 {
            return;
        }
        let d = n_draft.clamp(1, MAX_HIST_DRAFTS as i32);
        let a = n_accept.clamp(0, d);
        let r = (a as f32) / (d as f32);

        self.rounds += 1;
        self.sum_n += d as i64;
        self.hist[d as usize] += 1;

        self.ema_rate = (1.0_f32 - self.ema_alpha) * self.ema_rate + self.ema_alpha * r;

        if !self.enabled || self.n_max <= 1 {
            return;
        }

        if self.ema_rate >= THRESH_UPSHIFT && r >= THRESH_DOWNSHIFT_MILD {
            if self.n_cur < self.n_max {
                self.n_cur = (self.n_cur + 1).min(self.n_max);
                self.upshifts += 1;
            }
        } else if self.ema_rate < THRESH_DOWNSHIFT_FAST {
            if self.n_cur > 1 {
                let next = if self.n_cur > 2 && r < THRESH_DOWNSHIFT_FAST {
                    (self.n_cur - 2).max(1)
                } else {
                    (self.n_cur - 1).max(1)
                };
                self.n_cur = next;
                self.downshifts += 1;
            }
        } else if self.ema_rate < THRESH_DOWNSHIFT
            || (self.ema_rate < THRESH_DOWNSHIFT_MILD && r < THRESH_DOWNSHIFT_FAST)
        {
            if self.n_cur > 1 {
                self.n_cur -= 1;
                self.downshifts += 1;
            }
        }
    }

    /// Mean draft length across all observed rounds (`sum_n / rounds`).
    pub fn mean_draft_len(&self) -> f32 {
        if self.rounds > 0 {
            (self.sum_n as f32) / (self.rounds as f32)
        } else {
            self.n_cur as f32
        }
    }

    // ========================================================================
    // q36_adraft_policy.{hpp,cpp} architecture & confidence-stop methods
    // ========================================================================

    /// Initialize the `q36_adraft_policy` confidence/cost controller for GPU architecture `gcn`
    /// and fixed MTP length `d_fixed`.
    pub fn init_arch(&mut self, gcn: &str, d_fixed: i32) {
        let clamped = d_fixed.clamp(0, K_MAX as i32);
        self.init(clamped.min(MAX_HIST_DRAFTS as i32), clamped > 1);
        self.dfix_ = clamped;
        let base_arch = gcn.split(':').next().unwrap_or("").trim();
        let is_gfx1151 = base_arch == "gfx1151";

        if is_gfx1151 {
            self.mode_ = 2;
            self.dmin_ = 2;
            self.dmax_ = 7;
            self.chain_ = 3;
            self.pmin_ = [0.5, 0.0];
            self.cum_ = [0.40, 0.45];
            for (d, &c) in GFX1151_COST_MS.iter().enumerate() {
                self.cost_[d + 1] = c;
            }
            self.dfl_stop_ = true;
            self.dfl_sample_ = true;
            self.samp_ = false;
        } else {
            self.mode_ = 0;
            self.dmin_ = 1;
            self.dmax_ = 7;
            self.chain_ = 4;
            self.pmin_ = [0.5, 0.0];
            self.cum_ = [0.40, 0.45];
            for d in 1..=8 {
                self.cost_[d] = 68.0 + 4.0 * (d as f64);
            }
            self.dfl_stop_ = false;
            self.dfl_sample_ = false;
            self.samp_ = false;
        }
        for d in 9..=K_MAX {
            self.cost_[d] = self.cost_[8] + 6.0 * ((d - 8) as f64);
        }
        if d_fixed <= 0 {
            self.mode_ = 0;
        }
        for k in 1..=K_MAX {
            self.tries_[0][k] = 8.0;
            self.tries_[1][k] = 8.0;
            self.keeps_[0][k] = 8.0 * 0.80;
            self.keeps_[1][k] = 8.0 * 0.70;
        }
    }

    /// Policy mode (`0` = fixed, `1` = estimated cost curve, `2` = confidence stop).
    #[inline]
    pub const fn mode(&self) -> i32 {
        self.mode_
    }

    /// True if the engine should record per-step `conf` / `top`.
    #[inline]
    pub const fn want_conf(&self) -> bool {
        self.mode_ != 0
    }

    /// True if running in confidence-stop mode (`mode == 2`).
    #[inline]
    pub const fn stop_mode(&self) -> bool {
        self.mode_ == 2
    }

    /// Maximum drafts a window may chain.
    #[inline]
    pub fn dmax(&self) -> i32 {
        if self.mode_ == 0 {
            self.dfix_
        } else {
            self.dmax_.max(self.dfix_)
        }
    }

    /// Maximum drafts a window may chain for a greedy or sampled request.
    #[inline]
    pub const fn dmax_for(&self, sampled: bool) -> i32 {
        if self.mode_ == 0 || (sampled && !self.samp_) {
            self.dfix_
        } else {
            self.dmax_
        }
    }

    /// True if DFlash2 sampled requests draw from the selector's `q`.
    #[inline]
    pub const fn dfl_sample(&self) -> bool {
        self.dfl_sample_
    }

    /// True if DFlash2 windows apply the cumulative selector-weight stop rule.
    #[inline]
    pub const fn dfl_stop(&self) -> bool {
        self.dfl_stop_
    }

    /// Drafts chained by graph B / C for the next window (`q36_adraft_policy.cpp::chain_len`).
    pub fn chain_len(&self, sampled: bool) -> i32 {
        if self.mode_ == 0 || (self.mode_ == 2 && sampled && !self.samp_) {
            return self.dfix_;
        }
        if self.mode_ == 2 {
            return self.chain_;
        }
        let s = usize::from(sampled);
        let mut best = -1.0_f64;
        let mut e = 1.0_f64;
        let mut p = 1.0_f64;
        let mut bd = self.dmin_;
        for d in 1..=(self.dmax_ as usize).min(K_MAX) {
            p *= self.keeps_[s][d] / self.tries_[s][d].max(1e-9);
            e += p;
            if (d as i32) >= self.dmin_ && e / self.cost_[d] > best {
                best = e / self.cost_[d];
                bd = d as i32;
            }
        }
        bd
    }

    /// Stop mode: of the `n` drafts chained so far (`top[0..n]`), how many the window keeps.
    pub fn keep(&self, top: &[f32], n: i32, sampled: bool) -> i32 {
        let count = (n.max(0) as usize).min(top.len());
        if sampled && !self.samp_ {
            return count as i32;
        }
        let s = usize::from(sampled);
        let mut prod = 1.0_f32;
        for (i, &t) in top[..count].iter().enumerate() {
            prod *= t;
            if (i as i32) >= self.dmin_ && !(t >= self.pmin_[s] && prod >= self.cum_[s]) {
                return i as i32;
            }
        }
        count as i32
    }

    /// DFlash2 cumulative selector-weight stop rule (`q36_adraft_policy.cpp::keep_dfl`).
    pub fn keep_dfl(&self, w: &[f32], n: i32, sampled: bool) -> i32 {
        let count = (n.max(0) as usize).min(w.len());
        let s = usize::from(sampled);
        let mut prod = 1.0_f32;
        for (i, &wi) in w[..count].iter().enumerate() {
            prod *= wi;
            if (i as i32) >= self.dfl_dmin_ && !(prod >= self.dfl_cum_[s]) {
                return i as i32;
            }
        }
        count as i32
    }

    /// Observe a verified MTP or DFlash2 window (`q36_adraft_policy.cpp::observe`).
    pub fn observe_window(
        &mut self,
        k: i32,
        _chained: i32,
        a: i32,
        _conf: &[f32],
        _top: &[f32],
        sampled: bool,
        dflash: bool,
    ) {
        self.observe(k, a);
        if self.mode_ != 1 || dflash {
            return;
        }
        let s = usize::from(sampled);
        const DECAY: f64 = 0.99;
        let limit = (a + 1).min(k).clamp(0, K_MAX as i32) as usize;
        for i in 1..=limit {
            self.tries_[s][i] = self.tries_[s][i] * DECAY + 1.0;
            self.keeps_[s][i] = self.keeps_[s][i] * DECAY + if (i as i32) <= a { 1.0 } else { 0.0 };
        }
    }

    /// Human-readable one-line description matching `DraftLen::describe()`.
    pub fn describe(&self) -> String {
        if self.mode_ == 0 {
            format!("draft length fixed ({})", self.dfix_)
        } else if self.mode_ == 1 {
            format!(
                "draft length adaptive (est: {} .. {} by the round-cost curve)",
                self.dmin_, self.dmax_
            )
        } else if self.samp_ {
            format!(
                "draft length adaptive (stop: chain {}, {} .. {} drafts, keep while top >= {:.2} / {:.2} and prod top >= {:.2} / {:.2}, greedy / sampled)",
                self.chain_,
                self.dmin_,
                self.dmax_,
                self.pmin_[0],
                self.pmin_[1],
                self.cum_[0],
                self.cum_[1]
            )
        } else {
            format!(
                "draft length adaptive for greedy requests (stop: chain {}, {} .. {} drafts, keep while top >= {:.2} and prod top >= {:.2}; sampled: {})",
                self.chain_,
                self.dmin_,
                self.dmax_,
                self.pmin_[0],
                self.cum_[0],
                self.dfix_
            )
        }
    }
}

impl SpecController for DraftLen {
    fn plan(&mut self, stats: &RoundStats) -> RoundPlan {
        if stats.proposed > 0 {
            self.observe(stats.proposed as i32, stats.accepted as i32);
        }
        let pause_device_drafter =
            stats.consecutive_zero_accepts >= 3 && self.ema_rate < THRESH_DOWNSHIFT_FAST;
        let prefer_host_draft = stats.host_draft_available;
        let draft_depth = if pause_device_drafter && !prefer_host_draft {
            0
        } else {
            self.choose().max(0) as u32
        };
        RoundPlan {
            draft_depth,
            pause_device_drafter,
            prefer_host_draft,
            window_len: draft_depth + 1,
        }
    }
}

/// Operating mode for [`DraftLen`] adaptive draft controller (`AdraftMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdraftMode {
    /// Fixed draft length (`mode == 0`).
    Fixed = 0,
    /// Round-cost curve expected-value estimation (`mode == 1`).
    CostCurve = 1,
    /// Confidence-stop mode (`mode == 2`).
    ConfidenceStop = 2,
}

/// Stopping rule mode for DFlash / DFlash2 chains (`DFlashStopMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashStopMode {
    /// Always emit the full `dmax` draft chain.
    Fixed = 0,
    /// Stop when top-1 probability or cumulative confidence drops below threshold.
    Confidence = 1,
    /// Stop when cumulative DFlash2 selector weight drops below `dfl_cum`.
    SelectorWeight = 2,
}

/// Stopping policy parameters for DFlash / DFlash2 chains (`DFlashStoppingPolicy`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DFlashStoppingPolicy {
    /// Stopping rule mode.
    pub mode: DFlashStopMode,
    /// Minimum draft steps before stopping rule applies.
    pub dmin: i32,
    /// Maximum draft steps in a block/chain.
    pub dmax: i32,
    /// Minimum per-step top-1 probability threshold.
    pub pmin: f32,
    /// Minimum cumulative confidence product threshold.
    pub cum_min: f32,
}

impl Default for DFlashStoppingPolicy {
    fn default() -> Self {
        Self {
            mode: DFlashStopMode::SelectorWeight,
            dmin: 1,
            dmax: 15,
            pmin: 0.50,
            cum_min: 0.25,
        }
    }
}

impl DFlashStoppingPolicy {
    /// Evaluates how many of the `weights` (`0..=weights.len()`) should be kept.
    pub fn keep_count(&self, weights: &[f32]) -> usize {
        if self.mode == DFlashStopMode::Fixed {
            return weights.len().min(self.dmax.max(0) as usize);
        }
        let limit = weights.len().min(self.dmax.max(0) as usize);
        let mut prod = 1.0f32;
        for (i, &w) in weights.iter().enumerate().take(limit) {
            prod *= w;
            if (i as i32) >= self.dmin {
                let min_step = if self.mode == DFlashStopMode::Confidence {
                    self.pmin
                } else {
                    0.0
                };
                if w < min_step || prod < self.cum_min {
                    return i;
                }
            }
        }
        limit
    }
}
