//! Host-side n-gram speculative drafter (`NgramMod`).
//!
//! Ports `NgramMod` from `src/qwen35/q36_engine.cu`, supporting both:
//! - The 3-gram hash table (`TABLE_SZ = 1 << 18`, `EMPTY = -1`, `N_NGRAM = 3`, `HASH_SEED = 55353`,
//!   `hash_step = (h * 65599) ^ (u32(t) * 2654435761u32)`, `init`, `reset`, `add`, `draft`).
//! - The `q36_engine.cu` / `llama.cpp` `ngram-mod` rolling context hash (`n = 24`, LCG multiplier
//!   `6364136223846793005ULL`, 25% occupancy reset rule, `update`, `propose_q36`).
//! - The [`talus_core_api::HostDrafter`] trait.

use talus_core_api::HostDrafter;

/// Default 3-gram table size (`262,144` entries = 1 MiB).
pub const TABLE_SZ: usize = 1 << 18;

/// Sentinel value indicating an empty slot in the n-gram hash table.
pub const EMPTY: i32 = -1;

/// Trigram context width (`3` tokens -> `1` continuation token).
pub const N_NGRAM: usize = 3;

/// Initial seed for the 3-gram hash function.
pub const HASH_SEED: u32 = 55353;

/// Default `q36_engine.cu` `ngram-mod` match length.
pub const DEFAULT_Q36_N: usize = 24;

/// Default `q36_engine.cu` `ngram-mod` table size (`4 << 20` entries).
pub const DEFAULT_Q36_TABLE_SZ: usize = 4 << 20;

/// Single step of the 3-gram hash function:
/// `(h * 65599) ^ (u32(t) * 2654435761u32)`.
#[inline]
pub const fn hash_step(h: u32, t: i32) -> u32 {
    h.wrapping_mul(65599) ^ (t as u32).wrapping_mul(2_654_435_761u32)
}

/// Compute the table index in `0..TABLE_SZ` for a slice of at least [`N_NGRAM`] tokens.
#[inline]
pub fn ngram_key(toks: &[i32]) -> usize {
    debug_assert!(toks.len() >= N_NGRAM);
    let mut h = HASH_SEED;
    let mut i = 0;
    while i < N_NGRAM {
        h = hash_step(h, toks[i]);
        i += 1;
    }
    (h as usize) & (TABLE_SZ - 1)
}

/// Configuration for [`NgramMod`] and [`NgramQ36`] host n-gram drafters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramConfig {
    /// N-gram context order (`3` for 3-gram hash table, `24` for `q36` rolling context).
    pub n_ngram: usize,
    /// Minimum chain length required before proposing (`kmin`).
    pub k_min: usize,
    /// Maximum tokens drafted per round (`kmax`).
    pub k_max: usize,
    /// Sliding history window size (`0` = full history).
    pub window: usize,
    /// Target vocabulary limit for valid token IDs.
    pub n_vocab: usize,
}

impl Default for NgramConfig {
    fn default() -> Self {
        Self {
            n_ngram: N_NGRAM,
            k_min: 1,
            k_max: 4,
            window: 4096,
            n_vocab: i32::MAX as usize,
        }
    }
}

/// Rolling-hash `q36` n-gram drafter wrapper (`NgramQ36`) implementing [`HostDrafter`].
#[derive(Debug, Clone)]
pub struct NgramQ36 {
    inner: NgramMod,
    pub k_min: usize,
    pub k_max: usize,
}

impl Default for NgramQ36 {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramQ36 {
    /// Creates a new `NgramQ36` with default `n = 4` (for code/edit matching) and 64K slots.
    pub fn new() -> Self {
        Self::with_params(4, 1 << 16, 1, 7)
    }

    /// Creates a `NgramQ36` with custom `n`, `table_size`, `k_min`, and `k_max`.
    pub fn with_params(n: usize, table_size: usize, k_min: usize, k_max: usize) -> Self {
        Self {
            inner: NgramMod::with_q36(n, table_size),
            k_min: k_min.max(1),
            k_max: k_max.max(k_min.max(1)),
        }
    }

    /// Resets the underlying rolling n-gram table.
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Ingests context tokens into both the 3-gram and rolling `q36` tables.
    pub fn ingest(&mut self, ctx: &[i32]) {
        self.inner.add(ctx);
        self.inner.update(ctx);
    }

    /// Proposes up to `max_tokens` continuation tokens from `ctx`.
    pub fn draft(&mut self, ctx: &[i32], max_tokens: usize) -> Vec<i32> {
        let cap = max_tokens.min(self.k_max);
        if cap < self.k_min {
            return Vec::new();
        }
        let mut r = self.inner.propose_q36(ctx, cap, self.k_min);
        if r.len() < self.k_min && ctx.len() >= N_NGRAM {
            let mut tmp = vec![0i32; cap];
            let nd = self.inner.draft(ctx, &mut tmp, cap, i32::MAX);
            if nd >= self.k_min {
                r.clear();
                r.extend_from_slice(&tmp[..nd]);
            }
        }
        r
    }
}

impl HostDrafter for NgramQ36 {
    fn propose(&mut self, hist: &[u32], max: usize, out: &mut Vec<u32>) {
        out.clear();
        let hist_i32: Vec<i32> = hist.iter().map(|&t| t as i32).collect();
        let drafted = self.draft(&hist_i32, max);
        out.extend(drafted.into_iter().map(|t| t as u32));
    }

    fn observe(&mut self, accepted: &[u32]) {
        self.inner.observe(accepted);
    }
}

/// Alias for [`NgramQ36`].
pub type Q36Ngram = NgramQ36;

/// Host n-gram speculative drafter (`NgramMod`).
#[derive(Debug, Clone)]
pub struct NgramMod {
    /// Fixed 3-gram lookup table of length [`TABLE_SZ`], initialized to [`EMPTY`].
    pub table: Vec<i32>,
    /// Number of occupied entries in `table`.
    pub table_used: usize,
    /// Rolling `q36` `ngram-mod` n-gram order (`n`).
    pub n: usize,
    /// Rolling `q36` `ngram-mod` table (`e`), lazily or explicitly sized.
    pub e: Vec<i32>,
    /// Number of occupied slots in `e`.
    pub used: usize,
    /// Last processed index in rolling context for `update`.
    pub i_last: usize,
    /// Minimum chain length required in `HostDrafter::propose` (`default = 1`).
    pub k_min: usize,
    /// Maximum chain length proposed in `HostDrafter::propose` (`default = usize::MAX`).
    pub k_max: usize,
    /// Sliding window size (`0` = unbounded).
    pub window: usize,
    /// Vocabulary upper bound (`default = i32::MAX`).
    pub n_vocab_limit: i32,
    /// Accumulated token history for [`HostDrafter::observe`].
    history: Vec<i32>,
}

impl Default for NgramMod {
    fn default() -> Self {
        Self::new()
    }
}

impl NgramMod {
    /// Create and initialize a new `NgramMod` with a `TABLE_SZ` (`1 << 18`) table filled with `EMPTY`.
    pub fn new() -> Self {
        let mut ng = Self {
            table: Vec::new(),
            table_used: 0,
            n: DEFAULT_Q36_N,
            e: Vec::new(),
            used: 0,
            i_last: 0,
            k_min: 1,
            k_max: usize::MAX,
            window: 0,
            n_vocab_limit: i32::MAX,
            history: Vec::new(),
        };
        ng.init();
        ng
    }

    /// Construct an `NgramMod` from [`NgramConfig`].
    pub fn with_config(cfg: NgramConfig) -> Self {
        let mut ng = Self::new();
        ng.n = cfg.n_ngram.max(1);
        ng.k_min = cfg.k_min;
        ng.k_max = cfg.k_max.max(cfg.k_min);
        ng.window = cfg.window;
        ng.n_vocab_limit = cfg.n_vocab.min(i32::MAX as usize) as i32;
        ng
    }

    /// Set the vocabulary upper bound enforced during [`HostDrafter::propose`].
    pub fn set_vocab_limit(&mut self, n_vocab: usize) {
        self.n_vocab_limit = n_vocab.min(i32::MAX as usize) as i32;
    }

    /// Configure `k_min`, `k_max`, and sliding `window` for [`HostDrafter::propose`].
    pub fn set_draft_bounds(&mut self, k_min: usize, k_max: usize, window: usize) {
        self.k_min = k_min;
        self.k_max = k_max;
        self.window = window;
    }

    /// Construct a `q36_engine.cu`-style `NgramMod(n, size)`.
    pub fn with_q36(n: usize, size: usize) -> Self {
        let n_clamped = n.max(1);
        let sz = size.max(1);
        Self {
            table: vec![EMPTY; TABLE_SZ],
            table_used: 0,
            n: n_clamped,
            e: vec![EMPTY; sz],
            used: 0,
            i_last: 0,
            k_min: 1,
            k_max: usize::MAX,
            window: 0,
            n_vocab_limit: i32::MAX,
            history: Vec::new(),
        }
    }

    /// Initialize (or re-initialize) the 3-gram hash table to `TABLE_SZ` entries of `EMPTY`.
    pub fn init(&mut self) {
        if self.table.len() != TABLE_SZ {
            self.table.resize(TABLE_SZ, EMPTY);
        }
        self.reset();
    }

    /// Reset all n-gram table entries to `EMPTY` (`-1`) and clear counters.
    pub fn reset(&mut self) {
        if self.table.len() != TABLE_SZ {
            self.table.resize(TABLE_SZ, EMPTY);
        } else {
            self.table.fill(EMPTY);
        }
        self.table_used = 0;
        if !self.e.is_empty() {
            self.e.fill(EMPTY);
        }
        self.used = 0;
        self.i_last = 0;
        self.history.clear();
    }

    /// Compute the 3-gram key for the first `N_NGRAM` tokens of `toks`.
    #[inline]
    pub fn key(toks: &[i32]) -> usize {
        ngram_key(toks)
    }

    /// Insert all `(N_NGRAM + 1)`-grams from `toks` into `self.table`.
    ///
    /// When `toks.len() == N_NGRAM + 1` (4 tokens, as in `ngram.add(all_tokens.data() + i, 4)`),
    /// records `table[key(&toks[0..3])] = toks[3]`. For longer slices, slides a window of
    /// length `N_NGRAM + 1` across `toks`.
    pub fn add(&mut self, toks: &[i32]) {
        if toks.len() <= N_NGRAM {
            return;
        }
        if self.table.len() != TABLE_SZ {
            self.init();
        }
        let limit = toks.len() - N_NGRAM;
        for i in 0..limit {
            let k = ngram_key(&toks[i..i + N_NGRAM]);
            if self.table[k] == EMPTY {
                self.table_used += 1;
            }
            self.table[k] = toks[i + N_NGRAM];
        }
    }

    /// C-style helper matching `ngram.add(ptr_slice, count)`: inserts the first `count` tokens of `toks`.
    pub fn add_n(&mut self, toks: &[i32], count: usize) {
        let n = count.min(toks.len());
        self.add(&toks[..n]);
    }

    /// Draft up to `max_d` continuation tokens from `hist` into `out`.
    ///
    /// Exact port of `NgramMod::draft`:
    /// - When `hist.len() < N_NGRAM`, returns `0`.
    /// - Copies the last `N_NGRAM` tokens of `hist` into a local rolling buffer `buf`.
    /// - Loops `d in 0..max_d`, looks up `table[key(&buf[d..])]`, and stops immediately if
    ///   `nxt == EMPTY || nxt < 0 || nxt >= n_vocab`.
    pub fn draft(&self, hist: &[i32], out: &mut [i32], max_d: usize, n_vocab: i32) -> usize {
        if hist.len() < N_NGRAM || max_d == 0 || out.is_empty() || self.table.len() < TABLE_SZ {
            return 0;
        }
        let limit = max_d.min(out.len());
        let mut buf = Vec::with_capacity(N_NGRAM + limit);
        buf.extend_from_slice(&hist[hist.len() - N_NGRAM..]);

        let mut drafted = 0usize;
        for d in 0..limit {
            let k = ngram_key(&buf[d..d + N_NGRAM]);
            let nxt = self.table[k];
            if nxt == EMPTY || nxt < 0 || nxt >= n_vocab {
                break;
            }
            out[d] = nxt;
            buf.push(nxt);
            drafted += 1;
        }
        drafted
    }

    /// Compute the `q36_engine.cu` LCG hash index for `n` tokens:
    /// `for i in 0..n { r = r * 6364136223846793005ULL + (uint32_t)t[i]; } r % e.size()`.
    #[inline]
    pub fn idx(&self, t: &[i32]) -> usize {
        debug_assert!(t.len() >= self.n && !self.e.is_empty());
        let mut r: usize = 0;
        for &tok in &t[..self.n] {
            r = r
                .wrapping_mul(6_364_136_223_846_793_005usize)
                .wrapping_add((tok as u32) as usize);
        }
        r % self.e.len()
    }

    /// Update the `q36` rolling n-gram table with every n-gram of `ctx` not added yet.
    ///
    /// Resets the table if occupancy exceeds 25% (`(used as f64) / (e.len() as f64) > 0.25`).
    pub fn update(&mut self, ctx: &[i32]) {
        if ctx.len() <= self.n {
            return;
        }
        if self.e.is_empty() {
            self.e.resize(DEFAULT_Q36_TABLE_SZ, EMPTY);
        }
        if self.i_last > ctx.len() - self.n {
            self.i_last = 0;
        }
        let mut i = self.i_last;
        while i + self.n < ctx.len() {
            let j = self.idx(&ctx[i..]);
            if self.e[j] < 0 {
                self.used += 1;
            }
            self.e[j] = ctx[i + self.n];
            i += 1;
        }
        self.i_last = ctx.len() - self.n;
        if (self.used as f64) / (self.e.len() as f64) > 0.25 {
            self.e.fill(EMPTY);
            self.used = 0;
            self.i_last = 0;
        }
    }

    /// Propose up to `k_max` tokens continuing `ctx` using the `q36` rolling table;
    /// returns an empty `Vec` unless at least `k_min` tokens chain.
    pub fn propose_q36(&mut self, ctx: &[i32], k_max: usize, k_min: usize) -> Vec<i32> {
        let mut r = Vec::new();
        if ctx.len() < self.n || k_max == 0 {
            return r;
        }
        self.update(ctx);
        if self.e.is_empty() {
            return r;
        }
        let mut w: Vec<i32> = ctx[ctx.len() - self.n..].to_vec();
        for i in 0..k_max {
            let t = self.e[self.idx(&w[i..])];
            if t < 0 {
                break;
            }
            r.push(t);
            w.push(t);
        }
        if r.len() < k_min {
            r.clear();
        }
        r
    }
}

impl HostDrafter for NgramMod {
    fn propose(&mut self, hist: &[u32], max: usize, out: &mut Vec<u32>) {
        out.clear();
        let cap = max.min(self.k_max);
        if cap == 0 || cap < self.k_min || hist.len() < N_NGRAM {
            return;
        }
        let slice = if self.window > 0 && hist.len() > self.window {
            &hist[hist.len() - self.window..]
        } else {
            hist
        };
        if slice.len() < N_NGRAM {
            return;
        }
        let hist_i32: Vec<i32> = slice.iter().map(|&t| t as i32).collect();
        let mut tmp = vec![0i32; cap];
        let n_drafted = self.draft(&hist_i32, &mut tmp, cap, self.n_vocab_limit);
        if n_drafted >= self.k_min.max(1) {
            out.extend(tmp[..n_drafted].iter().map(|&t| t as u32));
            return;
        }
        if !self.e.is_empty() && hist_i32.len() >= self.n {
            let q36_drafts = self.propose_q36(&hist_i32, cap, self.k_min.max(1));
            out.extend(q36_drafts.into_iter().map(|t| t as u32));
        }
    }

    fn observe(&mut self, accepted: &[u32]) {
        if accepted.is_empty() {
            return;
        }
        let start_len = self.history.len();
        self.history.extend(accepted.iter().map(|&t| t as i32));
        let window_start = start_len.saturating_sub(N_NGRAM);
        let window = self.history[window_start..].to_vec();
        self.add(&window);
        if !self.e.is_empty() && self.history.len() > self.n {
            let hist_clone = self.history.clone();
            self.update(&hist_clone);
        }
    }
}
