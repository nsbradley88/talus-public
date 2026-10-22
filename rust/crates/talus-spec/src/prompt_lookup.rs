//! Exact FNV-1a 3-gram / 2-gram suffix-verified prompt-lookup copy drafter
//! (`PromptLookupDrafter`), ported from `src/laguna/laguna_serve.cpp`.
//!
//! Uses a hash index over all historical 3-grams and 2-grams with an 8-token
//! suffix match check (`min_match = 8`) before proposing up to `k` tokens, paired
//! with an expected-value draft length chooser (`choose_draft_len`) and EMA
//! acceptance tracker (`acc_est`).

use std::collections::HashMap;
use talus_core_api::HostDrafter;

/// FNV-1a 64-bit offset basis (`1469598103934665603ULL` in `laguna_serve.cpp`).
pub const FNV1A_OFFSET_BASIS: u64 = 1_469_598_103_934_665_603u64;

/// FNV-1a 64-bit prime (`1099511628211ULL` in `laguna_serve.cpp`).
pub const FNV1A_PRIME: u64 = 1_099_511_628_211u64;

/// Default minimum suffix match length (`min_match = 8` in `laguna_serve.cpp`).
pub const DEFAULT_MIN_MATCH: usize = 8;

/// Default maximum draft horizon (`max_k = 7` in `laguna_serve.cpp`).
pub const DEFAULT_MAX_K: usize = 7;

/// Default per-draft-token relative cost (`kDraftCost = 0.40` in `laguna_serve.cpp`).
pub const DEFAULT_DRAFT_COST: f64 = 0.40;

/// Default minimum expected net speedup required to propose a draft (`1.03` in `laguna_serve.cpp`).
pub const DEFAULT_MIN_GAIN: f64 = 1.03;

/// Default initial EMA token acceptance estimate (`0.9` in `laguna_serve.cpp`).
pub const INITIAL_ACC_EST: f64 = 0.90;

/// Computes the exact FNV-1a n-gram hash key from `src/laguna/laguna_serve.cpp`:
///
/// ```cpp
/// static uint64_t key(const int32_t * t, int n) {
///     uint64_t h = 1469598103934665603ULL;
///     for (int i = 0; i < n; i++) h = (h ^ (uint32_t)t[i]) * 1099511628211ULL;
///     return h ^ (uint64_t)n;
/// }
/// ```
#[inline]
pub fn fnv1a_ngram_key(tokens: &[i32]) -> u64 {
    let mut h = FNV1A_OFFSET_BASIS;
    for &t in tokens {
        h = (h ^ ((t as u32) as u64)).wrapping_mul(FNV1A_PRIME);
    }
    h ^ (tokens.len() as u64)
}

/// Computes the exact FNV-1a n-gram hash key over `u32` token IDs.
#[inline]
pub fn fnv1a_ngram_key_u32(tokens: &[u32]) -> u64 {
    let mut h = FNV1A_OFFSET_BASIS;
    for &t in tokens {
        h = (h ^ (t as u64)).wrapping_mul(FNV1A_PRIME);
    }
    h ^ (tokens.len() as u64)
}

/// Chooses the expected-value optimal draft length `k in 0..=min(max_k, room)`
/// matching `src/laguna/laguna_serve.cpp`:
///
/// For each candidate `kk in 1..=max_k`:
/// - `ex = (kk + 1)` if `acc_est >= 0.999`, else `(1 - acc_est^(kk + 1)) / (1 - acc_est)`
/// - `r = ex / (1 + draft_cost * kk)`
///
/// Picks the `kk` maximizing `r` strictly above `min_gain` (default `1.03`), then clamps to `room`.
#[inline]
pub fn choose_draft_len(
    max_k: usize,
    room: usize,
    acc_est: f64,
    draft_cost: f64,
    min_gain: f64,
) -> usize {
    if max_k == 0 || room == 0 {
        return 0;
    }
    let mut best_k = 0usize;
    let mut best = min_gain;
    for kk in 1..=max_k {
        let ex = if acc_est >= 0.999 {
            (kk + 1) as f64
        } else {
            (1.0 - acc_est.powi((kk + 1) as i32)) / (1.0 - acc_est)
        };
        let r = ex / (1.0 + draft_cost * (kk as f64));
        if r > best {
            best = r;
            best_k = kk;
        }
    }
    best_k.min(room)
}

/// Updates the EMA acceptance estimate `acc_est` after a speculative round,
/// matching `src/laguna/laguna_serve.cpp` (lines 290 and 315):
///
/// - When `drafted == 0` (no draft proposed): drifts toward `0.9`:
///   `acc_est += 0.05 * (0.9 - acc_est)`
/// - When `drafted > 0`: for each drafted position `i in 0..drafted`:
///   `acc_est = 0.9 * acc_est + 0.1 * (if i < accepted { 1.0 } else { 0.0 })`
#[inline]
pub fn update_acc_est_ema(acc_est: f64, drafted: usize, accepted: usize) -> f64 {
    let mut est = acc_est;
    if drafted == 0 {
        est += 0.05 * (INITIAL_ACC_EST - est);
    } else {
        for i in 0..drafted {
            let bit = if i < accepted { 1.0 } else { 0.0 };
            est = 0.9 * est + 0.1 * bit;
        }
    }
    est
}

/// Prompt-lookup copy drafter (`Drafter` in `src/laguna/laguna_serve.cpp`).
///
/// Indexes 3-grams and 2-grams over the token history using 64-bit FNV-1a hashing.
/// Before proposing tokens from a historical match position `at`, verifies that at
/// least `min_match` (default 8) preceding tokens match verbatim between the suffix
/// ending at `at` and the current sequence suffix.
#[derive(Debug, Clone)]
pub struct PromptLookupDrafter {
    /// Complete token history indexed so far.
    pub hist: Vec<i32>,
    /// Maps 3-gram FNV-1a key `key(hist[i-3..i])` -> continuation index `i`.
    pub last3: HashMap<u64, usize>,
    /// Maps 2-gram FNV-1a key `key(hist[i-2..i])` -> continuation index `i`.
    pub last2: HashMap<u64, usize>,
    /// Minimum verbatim suffix match length required before proposing (default `8`).
    pub min_match: usize,
    /// Maximum draft tokens to propose when using expected-value chooser (default `7`).
    pub max_k: usize,
    /// Remaining generation token budget (`room`, default `usize::MAX`).
    pub room: usize,
    /// Rolling EMA acceptance estimate (`acc_est`, initialized to `0.90`).
    pub acc_est: f64,
    /// Per-draft-token cost ratio for `choose_draft_len` (default `0.40`).
    pub draft_cost: f64,
    /// Minimum expected gain threshold for `choose_draft_len` (default `1.03`).
    pub min_gain: f64,
    /// Whether `HostDrafter::propose` uses expected-value `choose_draft_len` (default `true`).
    pub use_expected_value_len: bool,
    /// Last draft tokens proposed via `HostDrafter::propose`, used by `HostDrafter::observe`.
    last_proposed: Vec<u32>,
}

impl Default for PromptLookupDrafter {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptLookupDrafter {
    /// Creates a new `PromptLookupDrafter` with `laguna_serve.cpp` defaults:
    /// `min_match = 8`, `max_k = 7`, `acc_est = 0.90`, `draft_cost = 0.40`, `min_gain = 1.03`.
    pub fn new() -> Self {
        Self {
            hist: Vec::new(),
            last3: HashMap::new(),
            last2: HashMap::new(),
            min_match: DEFAULT_MIN_MATCH,
            max_k: DEFAULT_MAX_K,
            room: usize::MAX,
            acc_est: INITIAL_ACC_EST,
            draft_cost: DEFAULT_DRAFT_COST,
            min_gain: DEFAULT_MIN_GAIN,
            use_expected_value_len: true,
            last_proposed: Vec::new(),
        }
    }

    /// Creates a `PromptLookupDrafter` with custom `min_match` and `max_k`.
    pub fn with_params(min_match: usize, max_k: usize) -> Self {
        Self {
            min_match,
            max_k,
            ..Self::new()
        }
    }

    /// Configures expected-value draft length parameters (`draft_cost`, `min_gain`).
    pub fn with_cost_model(mut self, draft_cost: f64, min_gain: f64) -> Self {
        self.draft_cost = draft_cost;
        self.min_gain = min_gain;
        self
    }

    /// Appends a single token `t` to history after recording the trailing 3-gram and 2-gram
    /// pointing to continuation index `n = hist.len()`, matching `Drafter::push` in `laguna_serve.cpp`.
    #[inline]
    pub fn push(&mut self, t: i32) {
        let n = self.hist.len();
        if n >= 3 {
            let k3 = fnv1a_ngram_key(&self.hist[n - 3..n]);
            self.last3.insert(k3, n);
        }
        if n >= 2 {
            let k2 = fnv1a_ngram_key(&self.hist[n - 2..n]);
            self.last2.insert(k2, n);
        }
        self.hist.push(t);
    }

    /// Clears and rebuilds the n-gram index from `tokens`, matching `Drafter::reset` in `laguna_serve.cpp`.
    pub fn reset(&mut self, tokens: &[i32]) {
        self.hist.clear();
        self.last3.clear();
        self.last2.clear();
        self.last_proposed.clear();
        self.hist.reserve(tokens.len());
        for &t in tokens {
            self.push(t);
        }
    }

    /// Clears and rebuilds the n-gram index from `u32` token IDs (`reset_u32`).
    pub fn reset_u32(&mut self, tokens: &[u32]) {
        self.hist.clear();
        self.last3.clear();
        self.last2.clear();
        self.last_proposed.clear();
        self.hist.reserve(tokens.len());
        for &t in tokens {
            self.push(t as i32);
        }
    }

    /// Resets the EMA acceptance estimate back to `INITIAL_ACC_EST` (`0.90`).
    #[inline]
    pub fn reset_acc_est(&mut self) {
        self.acc_est = INITIAL_ACC_EST;
    }

    /// Chooses the draft length `k` using the current `acc_est`, `draft_cost`, and `min_gain`.
    #[inline]
    pub fn choose_k(&self, max_k: usize, room: usize) -> usize {
        choose_draft_len(max_k, room, self.acc_est, self.draft_cost, self.min_gain)
    }

    /// Updates `self.acc_est` given `drafted` proposed tokens and `accepted` verified tokens.
    #[inline]
    pub fn update_acc_est(&mut self, drafted: usize, accepted: usize) -> f64 {
        self.acc_est = update_acc_est_ema(self.acc_est, drafted, accepted);
        self.acc_est
    }

    /// Attempts to draft up to `k` continuation tokens into `out` after verifying a
    /// suffix match of at least `min_match` tokens, matching `Drafter::draft` in `laguna_serve.cpp`.
    ///
    /// Returns the number of tokens written to `out[..m]`.
    pub fn draft(&self, k: usize, out: &mut [i32], min_match: usize) -> usize {
        let cap = k.min(out.len());
        if cap == 0 {
            return 0;
        }
        let n = self.hist.len();
        let mut at: Option<usize> = None;
        if n >= 3 {
            let k3 = fnv1a_ngram_key(&self.hist[n - 3..n]);
            if let Some(&idx) = self.last3.get(&k3) {
                at = Some(idx);
            }
        }
        if at.is_none() && n >= 2 {
            let k2 = fnv1a_ngram_key(&self.hist[n - 2..n]);
            if let Some(&idx) = self.last2.get(&k2) {
                at = Some(idx);
            }
        }
        let Some(at_idx) = at else {
            return 0;
        };

        let mut match_len = 0usize;
        while match_len < min_match
            && at_idx > match_len
            && n > match_len
            && self.hist[at_idx - 1 - match_len] == self.hist[n - 1 - match_len]
        {
            match_len += 1;
        }
        if match_len < min_match {
            return 0;
        }

        let mut m = 0usize;
        while m < cap && at_idx + m < n {
            out[m] = self.hist[at_idx + m];
            m += 1;
        }
        m
    }

    /// Synchronizes `self.hist` with an external `hist: &[u32]` slice, incrementally pushing
    /// only new suffix tokens when `hist` extends `self.hist`.
    pub fn sync_history(&mut self, hist: &[u32]) {
        if hist.is_empty() {
            return;
        }
        let cur_len = self.hist.len();
        if cur_len <= hist.len()
            && self
                .hist
                .iter()
                .zip(hist.iter())
                .all(|(&a, &b)| (a as u32) == b)
        {
            for &t in &hist[cur_len..] {
                self.push(t as i32);
            }
        } else {
            self.reset_u32(hist);
        }
    }
}

impl HostDrafter for PromptLookupDrafter {
    fn propose(&mut self, hist: &[u32], max: usize, out: &mut Vec<u32>) {
        out.clear();
        self.last_proposed.clear();
        self.sync_history(hist);

        let max_req = max.min(self.max_k);
        let k = if self.use_expected_value_len {
            self.choose_k(max_req, self.room)
        } else {
            max_req.min(self.room)
        };
        if k == 0 {
            return;
        }

        let mut buf = vec![0i32; k];
        let nd = self.draft(k, &mut buf, self.min_match);
        out.reserve(nd);
        for &tok in &buf[..nd] {
            if tok >= 0 {
                let u = tok as u32;
                out.push(u);
                self.last_proposed.push(u);
            } else {
                break;
            }
        }
    }

    fn observe(&mut self, accepted: &[u32]) {
        let drafted = self.last_proposed.len();
        if drafted == 0 {
            self.update_acc_est(0, 0);
        } else {
            let mut n_acc = 0usize;
            for (&prop, &act) in self.last_proposed.iter().zip(accepted.iter()) {
                if prop == act {
                    n_acc += 1;
                } else {
                    break;
                }
            }
            self.update_acc_est(drafted, n_acc);
        }
        self.last_proposed.clear();
        for &t in accepted {
            self.push(t as i32);
        }
    }
}
