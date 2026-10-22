//! Phase C2 adaptive pause policy (`AdaptivePausePolicy`, default threshold `1.2`)
//! and `ngram-mod` workload/edit profiles (`NgramEditProfile`, `SpecWorkloadProfile`).
//!
//! Ported from `halo-tuning` patches `0017`/`0019` (`LLAMA_SPEC_ADAPTIVE=1.2`) and
//! the Laguna / MiMo speculation configurations.

use crate::ngram::{NgramMod, Q36Ngram};
use crate::prompt_lookup::PromptLookupDrafter;
use talus_core_api::{HostDrafter, RoundPlan, RoundStats, SpecController};

/// Default adaptive pause threshold (`1.2` in `LLAMA_SPEC_ADAPTIVE=1.2`).
pub const DEFAULT_PAUSE_RATIO: f32 = 1.2;

/// Default relative cost per drafted token (`0.40`).
pub const DEFAULT_COST_PER_DRAFT: f32 = 0.40;

/// Default EMA retention factor (`0.70`, weighting new round at `0.30`).
pub const DEFAULT_PAUSE_EMA_DECAY: f32 = 0.70;

/// Default initial/minimum cooldown backoff steps (`8` decode steps).
pub const DEFAULT_MIN_BACKOFF: u32 = 8;

/// Default maximum exponential cooldown backoff steps (`256` decode steps).
pub const DEFAULT_MAX_BACKOFF: u32 = 256;

/// Evaluation metric used by [`AdaptivePausePolicy`] to decide when to pause device drafting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptivePauseMetric {
    /// Rolling EMA of accepted tokens per round vs `pause_ratio` (`LLAMA_SPEC_ADAPTIVE=1.2`).
    AcceptedEma,
    /// Rolling EMA of net speedup `(1 + accepted) / (1 + cost_per_draft * proposed)` vs `pause_ratio`.
    SpeedupRatio,
    /// Pauses when either `ema_accepted < pause_ratio` or `ema_speedup < pause_ratio`.
    Combined,
}

/// Computes the single-round net token/cost speedup ratio:
/// `(1 + accepted) / (1 + cost_per_draft * proposed)`.
#[inline]
pub fn round_speedup_ratio(proposed: u32, accepted: u32, cost_per_draft: f32) -> f32 {
    if proposed == 0 {
        return 1.0;
    }
    (1.0 + accepted as f32) / (1.0 + cost_per_draft * (proposed as f32))
}

/// Phase C2 adaptive pause controller (`AdaptivePausePolicy`, threshold `1.2`).
///
/// Tracks rolling EMA acceptance and cost-adjusted speedup `(1 + accepted) / (1 + cost * offered)`.
/// When performance drops below `pause_ratio` (default `1.2`), device drafting is paused for a
/// cooldown window (`backoff` steps, doubling from `min_backoff = 8` up to `max_backoff = 256`),
/// optionally falling back to host `ngram-mod` / `PromptLookupDrafter` or single-token decode,
/// and resumes when the cooldown expires or host copy acceptance recovers.
#[derive(Debug, Clone)]
pub struct AdaptivePausePolicy {
    /// Pause threshold (`1.2` by default).
    pub pause_ratio: f32,
    /// Relative verification cost per proposed draft token (default `0.40`).
    pub cost_per_draft: f32,
    /// EMA retention weight `alpha` in `alpha * ema + (1 - alpha) * sample` (default `0.70`).
    pub ema_decay: f32,
    /// Which metric triggers pause (`AcceptedEma`, `SpeedupRatio`, or `Combined`).
    pub metric: AdaptivePauseMetric,
    /// Rolling EMA of accepted draft tokens per round.
    pub ema_accepted: f32,
    /// Rolling EMA of `(1 + accepted) / (1 + cost_per_draft * proposed)`.
    pub ema_speedup: f32,
    /// Most recent round's instantaneous speedup ratio.
    pub last_speedup: f32,
    /// Remaining decode steps in the current pause cooldown (`0` = active).
    pub cooldown_remaining: u32,
    /// Next pause duration in decode steps (`min_backoff..=max_backoff`).
    pub backoff: u32,
    /// Minimum backoff window (default `8`).
    pub min_backoff: u32,
    /// Maximum backoff window (default `256`).
    pub max_backoff: u32,
    /// Target device draft depth when active (default `4`).
    pub max_draft: u32,
    /// Whether to allow host drafter fallback while the device drafter is paused.
    pub fallback_to_host: bool,
    /// Number of times device drafting was paused.
    pub pause_count: u64,
    /// Number of times device drafting resumed after pause or recovery.
    pub resume_count: u64,
    /// Total rounds observed.
    pub total_rounds: u64,
    /// Total rounds spent with device drafter paused.
    pub paused_rounds: u64,
}

impl Default for AdaptivePausePolicy {
    fn default() -> Self {
        Self::new(DEFAULT_PAUSE_RATIO)
    }
}

impl AdaptivePausePolicy {
    /// Creates a new `AdaptivePausePolicy` with the given `pause_ratio` (e.g. `1.2`).
    pub fn new(pause_ratio: f32) -> Self {
        let init = if pause_ratio > 0.0 {
            pause_ratio
        } else {
            DEFAULT_PAUSE_RATIO
        };
        Self {
            pause_ratio,
            cost_per_draft: DEFAULT_COST_PER_DRAFT,
            ema_decay: DEFAULT_PAUSE_EMA_DECAY,
            metric: AdaptivePauseMetric::Combined,
            ema_accepted: init,
            ema_speedup: init,
            last_speedup: init,
            cooldown_remaining: 0,
            backoff: DEFAULT_MIN_BACKOFF,
            min_backoff: DEFAULT_MIN_BACKOFF,
            max_backoff: DEFAULT_MAX_BACKOFF,
            max_draft: 4,
            fallback_to_host: true,
            pause_count: 0,
            resume_count: 0,
            total_rounds: 0,
            paused_rounds: 0,
        }
    }

    /// Builder: sets `max_draft` depth when active.
    pub fn with_max_draft(mut self, max_draft: u32) -> Self {
        self.max_draft = max_draft;
        self
    }

    /// Builder: sets `cost_per_draft`.
    pub fn with_cost_per_draft(mut self, cost_per_draft: f32) -> Self {
        self.cost_per_draft = cost_per_draft;
        self
    }

    /// Builder: sets backoff bounds (`min_backoff`, `max_backoff`).
    pub fn with_backoff_bounds(mut self, min_backoff: u32, max_backoff: u32) -> Self {
        let min_b = min_backoff.max(1);
        let max_b = max_backoff.max(min_b);
        self.min_backoff = min_b;
        self.max_backoff = max_b;
        self.backoff = min_b;
        self
    }

    /// Builder: selects the pause evaluation metric.
    pub fn with_metric(mut self, metric: AdaptivePauseMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Builder: configures whether host n-gram / prompt-lookup drafting is allowed during device pause.
    pub fn with_fallback_to_host(mut self, fallback: bool) -> Self {
        self.fallback_to_host = fallback;
        self
    }

    /// Returns `true` if device drafting is currently paused in a cooldown window.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.cooldown_remaining > 0
    }

    /// Resets the policy state for a new request/sequence.
    pub fn reset(&mut self) {
        let init = if self.pause_ratio > 0.0 {
            self.pause_ratio
        } else {
            DEFAULT_PAUSE_RATIO
        };
        self.ema_accepted = init;
        self.ema_speedup = init;
        self.last_speedup = init;
        self.cooldown_remaining = 0;
        self.backoff = self.min_backoff;
    }

    /// Evaluates whether the current EMA values fall below `self.pause_ratio`.
    #[inline]
    pub fn is_below_threshold(&self) -> bool {
        if self.pause_ratio <= 0.0 {
            return false;
        }
        match self.metric {
            AdaptivePauseMetric::AcceptedEma => self.ema_accepted < self.pause_ratio,
            AdaptivePauseMetric::SpeedupRatio => self.ema_speedup < self.pause_ratio,
            AdaptivePauseMetric::Combined => {
                self.ema_accepted < self.pause_ratio || self.ema_speedup < self.pause_ratio
            }
        }
    }

    /// Observes the outcome of a decode/speculation round (`proposed`, `accepted`) and updates
    /// EMA metrics, cooldown state, and exponential backoff.
    ///
    /// Returns `true` if device drafting is paused after this update.
    pub fn observe_round(&mut self, proposed: u32, accepted: u32) -> bool {
        self.total_rounds += 1;
        if self.pause_ratio <= 0.0 {
            return false;
        }

        // If currently in a pause cooldown window:
        if self.cooldown_remaining > 0 {
            self.paused_rounds += 1;
            if proposed > 0 {
                let speedup = round_speedup_ratio(proposed, accepted, self.cost_per_draft);
                self.last_speedup = speedup;
                // Strong host copy/ngram acceptance while paused recovers immediately,
                // resetting EMA to break-even (`pause_ratio`) for the next device probe round.
                if (accepted as f32) >= self.pause_ratio && speedup >= self.pause_ratio {
                    self.cooldown_remaining = 0;
                    self.ema_accepted = self.pause_ratio;
                    self.ema_speedup = self.pause_ratio;
                    self.backoff = (self.backoff / 2).max(self.min_backoff);
                    self.resume_count += 1;
                    return false;
                }
            }
            self.cooldown_remaining -= 1;
            if self.cooldown_remaining == 0 {
                // Cooldown expired: reset EMA to break-even threshold for probe round.
                self.ema_accepted = self.pause_ratio;
                self.ema_speedup = self.pause_ratio;
                self.resume_count += 1;
            }
            return self.cooldown_remaining > 0;
        }

        // Active round where tokens were proposed:
        if proposed > 0 {
            let speedup = round_speedup_ratio(proposed, accepted, self.cost_per_draft);
            self.last_speedup = speedup;
            let w_old = self.ema_decay;
            let w_new = 1.0 - w_old;
            self.ema_accepted = w_old * self.ema_accepted + w_new * (accepted as f32);
            self.ema_speedup = w_old * self.ema_speedup + w_new * speedup;

            if self.is_below_threshold() {
                self.cooldown_remaining = self.backoff;
                self.backoff = (self.backoff.saturating_mul(2)).min(self.max_backoff);
                // Reset EMA to break-even (`pause_ratio`) so the post-cooldown probe starts fresh.
                self.ema_accepted = self.pause_ratio;
                self.ema_speedup = self.pause_ratio;
                self.pause_count += 1;
            } else {
                self.backoff = (self.backoff / 2).max(self.min_backoff);
            }
        }

        self.cooldown_remaining > 0
    }
}

impl SpecController for AdaptivePausePolicy {
    fn plan(&mut self, last: &RoundStats) -> RoundPlan {
        // Observe the previous round if it executed (or if we are ticking through a pause cooldown).
        if last.proposed > 0 || self.cooldown_remaining > 0 {
            self.observe_round(last.proposed, last.accepted);
        }

        if self.is_paused() {
            let use_host = self.fallback_to_host && last.host_draft_available;
            let draft_depth = if use_host { self.max_draft } else { 0 };
            RoundPlan {
                draft_depth,
                prefer_host_draft: use_host,
                pause_device_drafter: true,
                window_len: draft_depth + 1,
            }
        } else {
            RoundPlan {
                draft_depth: self.max_draft,
                prefer_host_draft: false,
                pause_device_drafter: false,
                window_len: self.max_draft + 1,
            }
        }
    }
}

/// Workload preset kind for [`NgramEditProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecWorkloadProfile {
    /// General conversational workload: device drafter primary, host n-gram fallback on pause.
    Chat,
    /// Code generation workload: device drafter + 3-gram host drafter (`k_min = 1..2`).
    Code,
    /// Code-edit / diff / refactor workload: aggressive copy/n-gram drafting (`k_min = 1`, `k_max = 7`).
    Edit,
    /// Device-only DFlash workload (e.g. MiMo-V2-Flash where DFlash is the primary drafter; see docs/RESULTS.md).
    DeviceOnly,
}

/// Alias for [`SpecWorkloadProfile`].
pub type NgramProfileKind = SpecWorkloadProfile;

/// Configurable `ngram-mod` & prompt-lookup edit profile (`NgramEditProfile`).
///
/// Coordinates `NgramQ36`, `NgramMod`, `PromptLookupDrafter`, and device drafters
/// (`DFlashDrafterSpec`) across `chat`, `code`, `edit`, and `device_only` workloads.
#[derive(Debug, Clone, PartialEq)]
pub struct NgramEditProfile {
    /// Workload preset category.
    pub kind: SpecWorkloadProfile,
    /// N-gram key length (default `3`).
    pub ngram_n: usize,
    /// Minimum draft match length (`kmin`) required before proposing host tokens (`0` = disabled).
    pub k_min: usize,
    /// Maximum draft tokens (`kmax`) proposed per round.
    pub k_max: usize,
    /// Sliding history window size in tokens inspected for copy/n-gram matching.
    pub window: usize,
    /// Minimum verbatim suffix match length for `PromptLookupDrafter`.
    pub min_match: usize,
    /// Whether a host drafter match of length `>= k_min` is preferred over device drafting.
    pub prefer_host_on_match: bool,
    /// Whether host drafting remains active when `AdaptivePausePolicy` pauses the device drafter.
    pub enable_on_device_pause: bool,
    /// Adaptive pause threshold (`1.2` for Phase C2 / Phase H).
    pub pause_ratio: f32,
}

impl Default for NgramEditProfile {
    fn default() -> Self {
        Self::edit()
    }
}

impl NgramEditProfile {
    /// Creates a custom `NgramEditProfile` with explicit `k_min`, `k_max`, and `window`.
    pub fn new(k_min: usize, k_max: usize, window: usize) -> Self {
        Self {
            kind: SpecWorkloadProfile::Edit,
            ngram_n: 3,
            k_min,
            k_max: k_max.max(k_min),
            window: window.max(16),
            min_match: 4,
            prefer_host_on_match: true,
            enable_on_device_pause: true,
            pause_ratio: DEFAULT_PAUSE_RATIO,
        }
    }

    /// Preset for general `chat` workloads (`k_min = 2`, `k_max = 4`, `window = 1024`).
    pub fn chat() -> Self {
        Self {
            kind: SpecWorkloadProfile::Chat,
            ngram_n: 3,
            k_min: 2,
            k_max: 4,
            window: 1024,
            min_match: 8,
            prefer_host_on_match: false,
            enable_on_device_pause: true,
            pause_ratio: DEFAULT_PAUSE_RATIO,
        }
    }

    /// Preset for `code` generation workloads (`k_min = 1`, `k_max = 4`, `window = 2048`).
    pub fn code() -> Self {
        Self {
            kind: SpecWorkloadProfile::Code,
            ngram_n: 3,
            k_min: 1,
            k_max: 4,
            window: 2048,
            min_match: 6,
            prefer_host_on_match: true,
            enable_on_device_pause: true,
            pause_ratio: DEFAULT_PAUSE_RATIO,
        }
    }

    /// Preset for code-`edit` workloads (`k_min = 1`, `k_max = 7`, `window = 4096`, `min_match = 4`).
    pub fn edit() -> Self {
        Self {
            kind: SpecWorkloadProfile::Edit,
            ngram_n: 3,
            k_min: 1,
            k_max: 7,
            window: 4096,
            min_match: 4,
            prefer_host_on_match: true,
            enable_on_device_pause: true,
            pause_ratio: DEFAULT_PAUSE_RATIO,
        }
    }

    /// Preset for MiMo-V2-Flash DFlash (`--spec-type none`, `n4`, adaptive pause `1.2`).
    pub fn mimo_dflash() -> Self {
        Self {
            kind: SpecWorkloadProfile::DeviceOnly,
            ngram_n: 3,
            k_min: 0,
            k_max: 4,
            window: 1024,
            min_match: 8,
            prefer_host_on_match: false,
            enable_on_device_pause: false,
            pause_ratio: DEFAULT_PAUSE_RATIO,
        }
    }

    /// Returns `true` if host n-gram / copy drafting is enabled in this profile.
    #[inline]
    pub fn host_enabled(&self) -> bool {
        self.kind != SpecWorkloadProfile::DeviceOnly && self.k_min > 0 && self.k_max >= self.k_min
    }

    /// Builds an [`AdaptivePausePolicy`] configured for this profile.
    pub fn build_pause_policy(&self) -> AdaptivePausePolicy {
        AdaptivePausePolicy::new(self.pause_ratio)
            .with_max_draft(self.k_max.max(1) as u32)
            .with_fallback_to_host(self.enable_on_device_pause && self.host_enabled())
    }

    /// Builds a [`PromptLookupDrafter`] configured with this profile's `min_match` and `k_max`.
    pub fn build_prompt_lookup(&self) -> PromptLookupDrafter {
        PromptLookupDrafter::with_params(self.min_match, self.k_max)
    }

    /// Builds an initialized [`NgramMod`] configured with this profile's `k_min`/`k_max`/`window`.
    pub fn build_ngram_mod(&self, n_vocab: usize) -> NgramMod {
        let mut ng = NgramMod::new();
        ng.init();
        ng.set_vocab_limit(n_vocab);
        ng.set_draft_bounds(self.k_min, self.k_max, self.window);
        ng
    }

    /// Builds a [`Q36Ngram`] (`NgramQ36`) rolling-hash drafter.
    pub fn build_ngram_q36(&self) -> Q36Ngram {
        Q36Ngram::new()
    }

    /// Slices `hist` to the trailing `self.window` tokens.
    #[inline]
    pub fn windowed_history<'a>(&self, hist: &'a [u32]) -> &'a [u32] {
        if self.window > 0 && hist.len() > self.window {
            &hist[hist.len() - self.window..]
        } else {
            hist
        }
    }

    /// Coordinates host draft proposal across [`PromptLookupDrafter`], [`Q36Ngram`], and [`NgramMod`],
    /// enforcing `window`, `k_min`, and `k_max`.
    ///
    /// Priority order:
    /// 1. `PromptLookupDrafter` (suffix-verified verbatim copy)
    /// 2. `Q36Ngram` (4096-entry rolling hash table)
    /// 3. `NgramMod` (65536-entry 3-gram modulo table)
    pub fn propose_coordinated(
        &self,
        hist: &[u32],
        max_tokens: usize,
        lookup: Option<&mut PromptLookupDrafter>,
        q36: Option<&mut Q36Ngram>,
        ngram: Option<&mut NgramMod>,
        out: &mut Vec<u32>,
    ) -> usize {
        out.clear();
        if !self.host_enabled() || max_tokens == 0 {
            return 0;
        }
        let cap = max_tokens.min(self.k_max);
        if cap < self.k_min {
            return 0;
        }
        let win = self.windowed_history(hist);

        if let Some(pl) = lookup {
            pl.propose(win, cap, out);
            if out.len() >= self.k_min {
                out.truncate(cap);
                return out.len();
            }
            out.clear();
        }

        if let Some(q) = q36 {
            let win_i32: Vec<i32> = win.iter().map(|&t| t as i32).collect();
            if win_i32.len() >= 4 {
                q.ingest(&win_i32);
            }
            let drafted = q.draft(&win_i32, cap);
            if drafted.len() >= self.k_min {
                out.extend(drafted.into_iter().take(cap).map(|t| t as u32));
                return out.len();
            }
        }

        if let Some(ng) = ngram {
            ng.propose(win, cap, out);
            if out.len() >= self.k_min {
                out.truncate(cap);
                return out.len();
            }
            out.clear();
        }

        0
    }

    /// Produces a [`RoundPlan`] by coordinating this edit profile with an [`AdaptivePausePolicy`].
    pub fn plan_with_pause(&self, pause: &mut AdaptivePausePolicy, last: &RoundStats) -> RoundPlan {
        pause.fallback_to_host = self.enable_on_device_pause && self.host_enabled();
        let mut plan = pause.plan(last);
        if self.kind == SpecWorkloadProfile::DeviceOnly {
            plan.prefer_host_draft = false;
            if plan.pause_device_drafter {
                plan.draft_depth = 0;
                plan.window_len = 1;
            }
            return plan;
        }
        if !plan.pause_device_drafter && self.prefer_host_on_match && last.host_draft_available {
            plan.prefer_host_draft = true;
            plan.draft_depth = (self.k_max as u32).max(plan.draft_depth);
            plan.window_len = plan.draft_depth + 1;
        } else if plan.pause_device_drafter && plan.prefer_host_draft {
            plan.draft_depth = self.k_max as u32;
            plan.window_len = plan.draft_depth + 1;
        }
        plan
    }
}
