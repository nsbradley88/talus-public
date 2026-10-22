//! Exact CPU sampling, 64-bit Mersenne Twister (`std::mt19937_64`), and speculative
//! rejection sampling (`decide_sampled`) ported bit-for-bit from `src/qwen35/q36_engine.cu`.

use std::cmp::Ordering;

/// Exact 64-bit Mersenne Twister (`MT19937-64`) matching C++11 `std::mt19937_64`
/// (`std::mersenne_twister_engine<uint64_t, 64, 312, 156, 31, 0xb5026f5aa96619e9ULL,
/// 29, 0x5555555555555555ULL, 17, 0x71d67fffeda60000ULL, 37, 0xfff7eee000000000ULL,
/// 43, 6364136223846793005ULL>`).
#[derive(Debug, Clone)]
pub struct Mt19937_64 {
    mt: [u64; Self::NN],
    mti: usize,
}

impl Mt19937_64 {
    /// State vector length (`n = 312`).
    pub const NN: usize = 312;
    /// Middle word offset (`m = 156`).
    pub const MM: usize = 156;
    /// Constant vector `a`.
    pub const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
    /// Most significant 33 bits (`r = 31`).
    pub const UM: u64 = 0xFFFF_FFFF_8000_0000;
    /// Least significant 31 bits (`r = 31`).
    pub const LM: u64 = 0x0000_0000_7FFF_FFFF;
    /// Initialization multiplier `f = 6364136223846793005ULL`.
    pub const INIT_MULT: u64 = 6_364_136_223_846_793_005;
    /// C++ standard default seed (`default_seed = 5489u`).
    pub const DEFAULT_SEED: u64 = 5489;

    /// Construct a new `Mt19937_64` seeded with `seed`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            mt: [0u64; Self::NN],
            mti: Self::NN + 1,
        };
        rng.seed(seed);
        rng
    }

    /// Re-seed the generator state matching `std::mt19937_64::seed(uint64_t)`.
    #[inline]
    pub fn seed(&mut self, seed: u64) {
        self.mt[0] = seed;
        for i in 1..Self::NN {
            let prev = self.mt[i - 1];
            self.mt[i] = Self::INIT_MULT
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        self.mti = Self::NN;
    }

    /// Advance state by twisting all 312 words.
    #[inline]
    fn twist(&mut self) {
        const MAG01: [u64; 2] = [0, Mt19937_64::MATRIX_A];

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

    /// Generate the next 64-bit unsigned integer (`operator()`).
    #[inline]
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
        x
    }

    /// Advance the generator by `z` steps (`std::mt19937_64::discard`).
    #[inline]
    pub fn discard(&mut self, z: u64) {
        for _ in 0..z {
            let _ = self.next_u64();
        }
    }

    /// Draw a uniform `f64` in `[0.0, 1.0)` matching GCC/libstdc++
    /// `std::uniform_real_distribution<double>(0.0, 1.0)(rng)` /
    /// `std::generate_canonical<double, 53, std::mt19937_64>`.
    ///
    /// For `std::mt19937_64`, `R = 2^64` and `b = 53`, so `k = 1`:
    /// `(rng() as f64) / 18446744073709551616.0_f64`, clamped to `nextafter(1.0, 0.0)`
    /// (`0x3fefffffffffffff`) if rounding reaches `1.0`.
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

    /// Draw from `std::uniform_real_distribution<double>(0.0, high)(rng)`.
    #[inline]
    pub fn uniform_range(&mut self, high: f64) -> f64 {
        self.uniform_01() * high
    }
}

impl Default for Mt19937_64 {
    #[inline]
    fn default() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }
}

/// Free-function wrapper for C++11 `std::generate_canonical<double, 53>(rng)`.
#[inline]
pub fn generate_canonical_f64(rng: &mut Mt19937_64) -> f64 {
    rng.uniform_01()
}

/// Free-function wrapper returning a uniform `f64` in `[0.0, 1.0)` from `rng`.
#[inline]
pub fn uniform_01(rng: &mut Mt19937_64) -> f64 {
    rng.uniform_01()
}

/// Candidate token and logit/probability (`Cand` in `q36_engine.cu`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cand {
    /// Vocabulary token ID.
    pub id: i32,
    /// Logit or probability value.
    pub p: f32,
}

impl Cand {
    /// Create a new [`Cand`].
    #[inline]
    pub const fn new(id: i32, p: f32) -> Self {
        Self { id, p }
    }

    /// Alias matching `Cand.v` in `q36_engine.cu`.
    #[inline]
    pub const fn v(&self) -> f32 {
        self.p
    }
}

/// Sampling configuration (`SampCfg` / `Sampling` in `q36_engine.cu`).
#[derive(Debug, Clone)]
pub struct SampCfg {
    /// Temperature (`<= 0.0` selects greedy argmax).
    pub temp: f32,
    /// Nucleus sampling threshold (`1.0` disables; `<= 0.0` selects greedy argmax).
    pub top_p: f32,
    /// Top-K candidate cutoff (`<= 0` disables; `1` selects greedy argmax).
    pub top_k: i32,
    /// Minimum probability relative to the top candidate (`0.0` disables).
    pub min_p: f32,
    /// Multiplicative repetition penalty (`1.0` disables).
    pub repeat_penalty: f32,
    /// Additive frequency penalty per token occurrence (`0.0` disables).
    pub frequency_penalty: f32,
    /// Additive presence penalty if token occurred at least once (`0.0` disables).
    pub presence_penalty: f32,
    /// Window size for repetition/frequency/presence penalties (`0` disables).
    pub repeat_last_n: i32,
    /// RNG seed.
    pub seed: u64,
    /// Stop token IDs.
    pub stop_tokens: Vec<i32>,
    /// Stop strings.
    pub stop_strings: Vec<String>,
}

/// Alias matching `struct Sampling` in `q36_engine.cu`.
pub type Sampling = SampCfg;

impl Default for SampCfg {
    fn default() -> Self {
        Self {
            temp: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: 64,
            seed: 1,
            stop_tokens: Vec::new(),
            stop_strings: Vec::new(),
        }
    }
}

impl SampCfg {
    /// Whether this configuration collapses deterministically to greedy argmax.
    #[inline]
    pub fn is_greedy(&self) -> bool {
        self.temp <= 0.0 || self.top_k == 1 || self.top_p <= 0.0
    }

    /// Whether repetition/frequency/presence penalties are active for `recent`.
    #[inline]
    pub fn has_penalties(&self, recent: &[i32]) -> bool {
        !recent.is_empty()
            && self.repeat_last_n != 0
            && (self.repeat_penalty != 1.0
                || self.frequency_penalty != 0.0
                || self.presence_penalty != 0.0)
    }
}

/// Compare two candidates in descending order of `p`, breaking ties by smaller `id`.
///
/// Matches deterministic top-K / sorting behavior when logits are distinct and provides
/// a strict total order by ascending token ID when equal or NaN.
#[inline]
fn cmp_cand_desc(a: &Cand, b: &Cand) -> Ordering {
    match (a.p.is_nan(), b.p.is_nan()) {
        (false, false) => match b.p.partial_cmp(&a.p) {
            Some(Ordering::Equal) | None => a.id.cmp(&b.id),
            Some(ord) => ord,
        },
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (true, true) => a.id.cmp(&b.id),
    }
}

/// Select the top `k` (`1 <= k <= n`) candidates from `x[..n]` in exact `cmp_cand_desc` order.
///
/// - When `k <= 256` and `k < n`, performs a single `O(n)` pass over `x[..n]` maintaining the
///   sorted top-`k` slice of size `k` in L1 cache (zero `O(n)` heap allocation), producing a
///   result bit-for-bit identical to sorting all `n` candidates and truncating to `k`.
/// - When `256 < k < n`, uses `select_nth_unstable_by(k - 1, cmp_cand_desc)` (`O(n + k log k)`).
/// - When `k == n`, sorts all `n` candidates.
pub fn select_top_k_cands(x: &[f32], n: usize, k: usize) -> Vec<Cand> {
    let n = n.min(x.len());
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    if k <= 256 && k < n {
        let mut cands: Vec<Cand> = x[..k]
            .iter()
            .enumerate()
            .map(|(i, &v)| Cand::new(i as i32, v))
            .collect();
        cands.sort_by(cmp_cand_desc);
        let mut worst_p = cands[k - 1].p;
        for (i, &v) in x.iter().enumerate().take(n).skip(k) {
            // Since i > all existing candidate IDs, a candidate at i with v == worst_p
            // always loses the ascending-ID tie-break against cands[k - 1].
            if v > worst_p || (worst_p.is_nan() && !v.is_nan()) {
                let cand = Cand::new(i as i32, v);
                let pos = cands.partition_point(|c| cmp_cand_desc(c, &cand) == Ordering::Less);
                if pos < k {
                    cands.copy_within(pos..(k - 1), pos + 1);
                    cands[pos] = cand;
                    worst_p = cands[k - 1].p;
                }
            }
        }
        return cands;
    }

    let mut cands: Vec<Cand> = x
        .iter()
        .enumerate()
        .take(n)
        .map(|(i, &v)| Cand::new(i as i32, v))
        .collect();
    if k < n {
        cands.select_nth_unstable_by(k - 1, cmp_cand_desc);
        cands.truncate(k);
    }
    cands.sort_by(cmp_cand_desc);
    cands
}

/// Apply repetition, frequency, and presence penalties to a copy of `logits[..n]` if active.
fn apply_penalties<'a>(
    logits: &'a [f32],
    n: usize,
    cfg: &SampCfg,
    recent: &[i32],
    penalized_buf: &'a mut Vec<f32>,
) -> &'a [f32] {
    if !cfg.has_penalties(recent) {
        return &logits[..n];
    }

    penalized_buf.clear();
    penalized_buf.extend_from_slice(&logits[..n]);

    let w = if cfg.repeat_last_n < 0 {
        recent.len()
    } else {
        (cfg.repeat_last_n as usize).min(recent.len())
    };
    let tail = &recent[recent.len() - w..];

    let mut counts: Vec<(i32, i32)> = Vec::with_capacity(tail.len());
    for &t in tail {
        if t < 0 || (t as usize) >= n {
            continue;
        }
        if let Some(entry) = counts.iter_mut().find(|(id, _)| *id == t) {
            entry.1 += 1;
        } else {
            counts.push((t, 1));
        }
    }

    for (t, cnt) in counts {
        let l = &mut penalized_buf[t as usize];
        if cfg.repeat_penalty != 1.0 {
            if *l <= 0.0 {
                *l *= cfg.repeat_penalty;
            } else {
                *l /= cfg.repeat_penalty;
            }
        }
        *l -= (cnt as f32) * cfg.frequency_penalty + cfg.presence_penalty;
    }

    penalized_buf.as_slice()
}

/// Greedy argmax over `x[..n]`, matching C++ strict `>` scan from index 0.
#[inline]
fn argmax_slice(x: &[f32], n: usize) -> i32 {
    if n == 0 {
        return 0;
    }
    let mut best_id = 0usize;
    let mut best_val = x[0];
    for (i, &val) in x.iter().enumerate().take(n).skip(1) {
        if val > best_val {
            best_val = val;
            best_id = i;
        }
    }
    best_id as i32
}

/// Filter candidates `(id, logit)` by `top_k`, temperature softmax, `top_p`, and `min_p`.
///
/// Returns `(candidates, total_mass)` where each candidate's `p` holds its unnormalized
/// probability mass and `total_mass` is the sum of `p` over the retained candidates.
fn build_filtered_candidates(x: &[f32], n: usize, cfg: &SampCfg) -> (Vec<Cand>, f64) {
    let k = if cfg.top_k > 0 {
        (cfg.top_k as usize).min(n)
    } else {
        n
    };

    let mut cands = select_top_k_cands(x, n, k);

    if cands.is_empty() {
        return (cands, 0.0);
    }

    let inv_t = 1.0f64 / (cfg.temp as f64);
    let max_l = (cands[0].p as f64) * inv_t;

    let mut sum = 0.0f64;
    let mut probs: Vec<f64> = Vec::with_capacity(cands.len());
    for c in &cands {
        let p = (((c.p as f64) * inv_t) - max_l).exp();
        probs.push(p);
        sum += p;
    }

    if sum <= 0.0 || !sum.is_finite() {
        cands.truncate(1);
        cands[0].p = 1.0;
        return (cands, 1.0);
    }

    let inv_sum = 1.0 / sum;
    for p in &mut probs {
        *p *= inv_sum;
    }

    let mut m = cands.len();
    if cfg.top_p < 1.0 {
        let mut cum = 0.0f64;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= cfg.top_p as f64 {
                m = i + 1;
                break;
            }
        }
    }
    if cfg.min_p > 0.0 {
        let thr = probs[0] * (cfg.min_p as f64);
        let mut i = 1usize;
        while i < m && probs[i] >= thr {
            i += 1;
        }
        m = i;
    }
    m = m.max(1);
    cands.truncate(m);
    probs.truncate(m);

    let mut tot = 0.0f64;
    for (c, &p) in cands.iter_mut().zip(probs.iter()) {
        c.p = p as f32;
        tot += p;
    }

    (cands, tot)
}

/// Sample a single token from `logits[..n_vocab]` using `cfg`, `recent` history, and `rng`.
pub fn sample(
    logits: &[f32],
    n_vocab: i32,
    cfg: &SampCfg,
    recent: &[i32],
    rng: &mut Mt19937_64,
) -> i32 {
    let n = (n_vocab.max(0) as usize).min(logits.len());
    if n == 0 {
        return 0;
    }

    let mut penalized = Vec::new();
    let x = apply_penalties(logits, n, cfg, recent, &mut penalized);

    if cfg.is_greedy() {
        return argmax_slice(x, n);
    }

    let (cands, tot) = build_filtered_candidates(x, n, cfg);
    if cands.len() == 1 || tot <= 0.0 {
        return cands.first().map_or(0, |c| c.id);
    }

    let r = rng.uniform_range(tot);
    let mut acc = 0.0f64;
    for c in &cands {
        acc += c.p as f64;
        if r <= acc {
            return c.id;
        }
    }
    cands.last().map_or(0, |c| c.id)
}

/// Build the normalized target sampling distribution over `logits[..n_vocab]`.
///
/// Return code convention (matching `q36_engine.cu`):
/// - `-1`: deterministic collapse (`cfg.temp <= 0.0 || cfg.top_k == 1 || cfg.top_p <= 0.0`).
/// - `top >= 0`: degenerate single-token distribution after filtering (`dist_out` is zeroed
///   with `dist_out[top as usize] = 1.0`).
/// - `-2`: general stochastic distribution (`dist_out[..n_vocab]` populated with normalized
///   probabilities summing to `1.0`).
pub fn make_dist(
    logits: &[f32],
    n_vocab: i32,
    cfg: &SampCfg,
    recent: &[i32],
    dist_out: &mut [f32],
) -> i32 {
    let n = (n_vocab.max(0) as usize)
        .min(logits.len())
        .min(dist_out.len());
    if n == 0 {
        return -1;
    }

    let mut penalized = Vec::new();
    let x = apply_penalties(logits, n, cfg, recent, &mut penalized);

    if cfg.is_greedy() {
        let top = argmax_slice(x, n);
        dist_out[..n].fill(0.0);
        dist_out[top as usize] = 1.0;
        return -1;
    }

    let (cands, tot) = build_filtered_candidates(x, n, cfg);
    dist_out[..n].fill(0.0);

    if cands.is_empty() {
        return -1;
    }
    if cands.len() == 1 || tot <= 0.0 {
        let top = cands[0].id;
        dist_out[top as usize] = 1.0;
        return top;
    }

    let inv_tot = (1.0 / tot) as f32;
    for c in &cands {
        dist_out[c.id as usize] = c.p * inv_tot;
    }
    -2
}

/// Draw a token from a dense distribution `dist[..n_vocab]` using `rng`.
pub fn draw_dist(dist: &[f32], n_vocab: i32, rng: &mut Mt19937_64) -> i32 {
    let n = (n_vocab.max(0) as usize).min(dist.len());
    if n == 0 {
        return 0;
    }

    let mut tot = 0.0f64;
    for &p in &dist[..n] {
        if p > 0.0 {
            tot += p as f64;
        }
    }
    if tot <= 0.0 {
        return 0;
    }

    let r = rng.uniform_range(tot);
    let mut acc = 0.0f64;
    let mut last_nz = 0i32;
    for (i, &p) in dist.iter().enumerate().take(n) {
        if p > 0.0 {
            last_nz = i as i32;
            acc += p as f64;
            if r <= acc {
                return i as i32;
            }
        }
    }
    last_nz
}

/// Build the sparse representation `(idx, p)` of the target sampling distribution
/// matching `make_dist(logits, n_vocab, s, hist, idx, p)` in `q36_engine.cu`.
pub fn make_dist_sparse(
    logits: &[f32],
    n_vocab: i32,
    cfg: &SampCfg,
    recent: &[i32],
    idx_out: &mut Vec<i32>,
    p_out: &mut Vec<f64>,
) {
    idx_out.clear();
    p_out.clear();

    let n = (n_vocab.max(0) as usize).min(logits.len());
    if n == 0 {
        return;
    }

    let mut penalized = Vec::new();
    let x = apply_penalties(logits, n, cfg, recent, &mut penalized);

    if cfg.is_greedy() {
        let top = argmax_slice(x, n);
        idx_out.push(top);
        p_out.push(1.0);
        return;
    }

    let (cands, tot) = build_filtered_candidates(x, n, cfg);
    if cands.is_empty() {
        return;
    }
    if cands.len() == 1 || tot <= 0.0 {
        idx_out.push(cands[0].id);
        p_out.push(1.0);
        return;
    }

    let inv_tot = 1.0 / tot;
    idx_out.reserve(cands.len());
    p_out.reserve(cands.len());
    for c in &cands {
        idx_out.push(c.id);
        p_out.push((c.p as f64) * inv_tot);
    }
}

/// Draw a token from a sparse distribution `(idx, p)` with total mass `tot`.
pub fn draw_dist_sparse(idx: &[i32], p: &[f64], tot: f64, rng: &mut Mt19937_64) -> i32 {
    if idx.is_empty() || p.is_empty() {
        return 0;
    }
    let r = rng.uniform_range(tot);
    let mut acc = 0.0f64;
    for (&id, &prob) in idx.iter().zip(p.iter()) {
        acc += prob;
        if r <= acc {
            return id;
        }
    }
    *idx.last().unwrap_or(&0)
}

/// Speculative decoding rejection sampler (`decide_sampled` in `q36_engine.cu`).
///
/// Verifies up to `n_draft` draft tokens in `draft_ids` against target logits rows
/// (`target_logits` of shape `(n_draft + 1) * n_vocab`), using optional draft proposal
/// probabilities `q_draft` (`q_draft(step, token_id)` if stochastic drafter, or `None` /
/// `<= 0.0` for deterministic point-mass drafter `q(draft) = 1`).
///
/// Appends accepted tokens to `recent`, sets `*accepted_out` to the number of accepted
/// draft tokens (`0..=n_draft`), and returns the resampled or bonus token ID (also appended
/// to `recent`).
#[allow(clippy::too_many_arguments)]
pub fn decide_sampled(
    target_logits: &[f32],
    n_vocab: i32,
    draft_ids: &[i32],
    n_draft: usize,
    cfg: &SampCfg,
    recent: &mut Vec<i32>,
    rng: &mut Mt19937_64,
    accepted_out: &mut usize,
    q_draft: Option<&dyn Fn(usize, i32) -> f64>,
) -> i32 {
    let nv = n_vocab.max(0) as usize;
    let nd = n_draft.min(draft_ids.len());
    *accepted_out = 0;

    if nv == 0 || target_logits.len() < (nd + 1) * nv {
        return 0;
    }

    let mut idx = Vec::new();
    let mut p = Vec::new();

    for i in 0..nd {
        let row = &target_logits[i * nv..(i + 1) * nv];
        let draft_tok = draft_ids[i];

        make_dist_sparse(row, n_vocab, cfg, recent, &mut idx, &mut p);

        if idx.len() == 1 {
            if idx[0] == draft_tok {
                *accepted_out += 1;
                recent.push(draft_tok);
                continue;
            }
            let sampled = idx[0];
            recent.push(sampled);
            return sampled;
        }

        let mut p_d = 0.0f64;
        let mut pos_d: Option<usize> = None;
        for (k, &id) in idx.iter().enumerate() {
            if id == draft_tok {
                p_d = p[k];
                pos_d = Some(k);
                break;
            }
        }

        let q_d = q_draft.map_or(0.0, |f| f(i, draft_tok));

        if q_d > 0.0 {
            // Stochastic draft with proposal probability q_d: accept with min(1, p_d / q_d).
            if p_d >= q_d || rng.uniform_01() < (p_d / q_d) {
                *accepted_out += 1;
                recent.push(draft_tok);
                continue;
            }
            // Residual distribution max(0, p - q), where for top-16 chain sampling only the
            // drafted token's marginal q_d is known at rejection time.
            if let Some(k) = pos_d {
                p[k] = (p_d - q_d).max(0.0);
            }
            let tot: f64 = p.iter().sum();
            let sampled = if tot > 0.0 {
                draw_dist_sparse(&idx, &p, tot, rng)
            } else {
                idx[0]
            };
            recent.push(sampled);
            return sampled;
        } else {
            // Deterministic / point-mass draft (q(draft) = 1, q(other) = 0):
            // Accept with probability p(draft); on reject, sample from p \ {draft}.
            if p_d > 0.0 && rng.uniform_01() < p_d {
                *accepted_out += 1;
                recent.push(draft_tok);
                continue;
            }
            if let Some(k) = pos_d {
                p[k] = 0.0;
            }
            let tot = (1.0 - p_d).max(0.0);
            let sampled = if tot > 0.0 {
                draw_dist_sparse(&idx, &p, tot, rng)
            } else {
                idx[0]
            };
            recent.push(sampled);
            return sampled;
        }
    }

    // All nd draft tokens accepted: sample bonus token from row nd.
    let bonus_row = &target_logits[nd * nv..(nd + 1) * nv];
    let bonus = sample(bonus_row, n_vocab, cfg, recent, rng);
    recent.push(bonus);
    bonus
}

/// Alias for [`SampCfg`] (`SamplingParams`).
pub type SamplingParams = SampCfg;

/// Normalized draft or target probability distribution over vocabulary logits (`DraftDist`).
#[derive(Debug, Clone)]
pub struct DraftDist {
    /// Dense normalized probabilities of length `n_vocab`.
    pub probs: Vec<f32>,
    /// Return code from [`make_dist`] (`-1` greedy, `>= 0` single token, `-2` stochastic).
    pub rc: i32,
}

impl DraftDist {
    /// Builds a normalized [`DraftDist`] from `logits` using `cfg` and `recent` history.
    pub fn from_logits(logits: &[f32], cfg: &SampCfg, recent: &[i32]) -> Self {
        let mut probs = vec![0.0f32; logits.len()];
        let rc = make_dist(logits, logits.len() as i32, cfg, recent, &mut probs);
        Self { probs, rc }
    }

    /// Queries the probability of `token_id`.
    #[inline]
    pub fn prob(&self, token_id: i32) -> f32 {
        if token_id < 0 {
            return 0.0;
        }
        self.probs.get(token_id as usize).copied().unwrap_or(0.0)
    }

    /// Returns the dense probability slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.probs
    }
}

/// Convenience wrapper for [`sample`] over the full `logits` slice.
#[inline]
pub fn sample_from_logits(
    logits: &[f32],
    cfg: &SampCfg,
    recent: &[i32],
    rng: &mut Mt19937_64,
) -> i32 {
    sample(logits, logits.len() as i32, cfg, recent, rng)
}

/// Computes the exact greedy argmax over `logits` (ties broken to lowest token ID).
#[inline]
pub fn greedy_argmax(logits: &[f32]) -> i32 {
    argmax_slice(logits, logits.len())
}

/// Verifies `n_draft` draft tokens in `draft_ids` greedily against `(n_draft + 1) * n_vocab`
/// target logits, setting `*accepted_out` to the number of accepted draft tokens (`0..=n_draft`)
/// and returning the correction or bonus token ID.
pub fn decide_greedy(
    target_logits: &[f32],
    n_vocab: usize,
    draft_ids: &[i32],
    n_draft: usize,
    accepted_out: &mut usize,
) -> i32 {
    let nd = n_draft.min(draft_ids.len());
    *accepted_out = 0;
    if n_vocab == 0 || target_logits.len() < (nd + 1) * n_vocab {
        return 0;
    }
    for (i, &draft_tok) in draft_ids.iter().enumerate().take(nd) {
        let row = &target_logits[i * n_vocab..(i + 1) * n_vocab];
        let winner = greedy_argmax(row);
        if winner == draft_tok {
            *accepted_out += 1;
        } else {
            return winner;
        }
    }
    let bonus_row = &target_logits[nd * n_vocab..(nd + 1) * n_vocab];
    greedy_argmax(bonus_row)
}

/// Self-test verifying all 4 core sampling behaviors AND speculative rejection sampling
/// distribution invariance (`sample_selftest` in `q36_engine.cu`).
pub fn sample_selftest() -> bool {
    // 1. Greedy argmax (temp = 0.0)
    {
        let logits = [0.1f32, 2.5, 1.9, -0.4, 2.49];
        let cfg = SampCfg {
            temp: 0.0,
            ..SampCfg::default()
        };
        let mut rng = Mt19937_64::new(42);
        let tok = sample(&logits, logits.len() as i32, &cfg, &[], &mut rng);
        if tok != 1 {
            return false;
        }
        let mut dist = [0.0f32; 5];
        let rc = make_dist(&logits, 5, &cfg, &[], &mut dist);
        if rc != -1 || (dist[1] - 1.0).abs() > 1e-6 {
            return false;
        }
    }

    // 2. Repetition, frequency, and presence penalties
    {
        // Token 1 has highest raw logit (2.5), token 2 is close (2.4).
        // Penalizing token 1 via recent history [1, 1] must flip greedy winner to token 2.
        let logits = [0.0f32, 2.5, 2.4, -1.0];
        let cfg = SampCfg {
            temp: 0.0,
            repeat_penalty: 1.2,
            frequency_penalty: 0.2,
            presence_penalty: 0.1,
            repeat_last_n: 64,
            ..SampCfg::default()
        };
        let recent = [1i32, 1];
        let mut rng = Mt19937_64::new(1);
        let tok = sample(&logits, 4, &cfg, &recent, &mut rng);
        if tok != 2 {
            return false;
        }
    }

    // 3. Top-K = 2 filtering
    {
        let logits = [5.0f32, 1.0, 4.8, 2.0, -3.0];
        let cfg = SampCfg {
            temp: 1.0,
            top_k: 2,
            top_p: 1.0,
            min_p: 0.0,
            ..SampCfg::default()
        };
        let mut dist = [0.0f32; 5];
        let rc = make_dist(&logits, 5, &cfg, &[], &mut dist);
        if rc != -2 {
            return false;
        }
        // Only tokens 0 and 2 may have non-zero probability.
        if dist[0] <= 0.0 || dist[2] <= 0.0 || dist[1] != 0.0 || dist[3] != 0.0 || dist[4] != 0.0 {
            return false;
        }
        let mut rng = Mt19937_64::new(123);
        for _ in 0..64 {
            let t = sample(&logits, 5, &cfg, &[], &mut rng);
            if t != 0 && t != 2 {
                return false;
            }
        }
    }

    // 4. Top-P + Min-P filtering and speculative rejection sampling equivalence
    {
        let logits = [3.2f32, 2.8, 2.4, 1.9, 1.2, 0.5, -0.3, -1.5];
        let n_vocab = 8i32;
        let cfg = SampCfg {
            temp: 0.8,
            top_k: 6,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            frequency_penalty: 0.1,
            presence_penalty: 0.05,
            repeat_last_n: 8,
            ..SampCfg::default()
        };
        let recent_base = vec![0i32, 2, 0, 5];

        let mut idx = Vec::new();
        let mut p = Vec::new();
        make_dist_sparse(&logits, n_vocab, &cfg, &recent_base, &mut idx, &mut p);
        if idx.is_empty() {
            return false;
        }

        let mut target_p = [0.0f64; 8];
        for (&id, &prob) in idx.iter().zip(p.iter()) {
            target_p[id as usize] = prob;
        }

        // Verify point-mass speculative rejection and q-proposal speculative rejection match
        // direct `sample` across Monte Carlo trials (`q36_engine.cu:2520-2603`).
        let n_trials = 25_000usize;
        let mut f_sample = [0usize; 8];
        let mut f_spec = [0usize; 8];
        let mut rng = Mt19937_64::new(12345);

        // Two-row target logits buffer (row 0 = step 0, row 1 = bonus step).
        let mut two_rows = [0.0f32; 16];
        two_rows[..8].copy_from_slice(&logits);
        two_rows[8..].copy_from_slice(&logits);

        for t in 0..n_trials {
            let tok = sample(&logits, n_vocab, &cfg, &recent_base, &mut rng);
            f_sample[tok as usize] += 1;

            let draft = [(t & 7) as i32];
            let mut recent = recent_base.clone();
            let mut acc = 0usize;
            let _ = decide_sampled(
                &two_rows,
                n_vocab,
                &draft,
                1,
                &cfg,
                &mut recent,
                &mut rng,
                &mut acc,
                None,
            );
            // The token produced for position 0 is `recent[recent_base.len()]`.
            let pos0_tok = recent[recent_base.len()];
            f_spec[pos0_tok as usize] += 1;
        }

        for v in 0..8 {
            let d1 = (f_sample[v] as f64 / n_trials as f64) - target_p[v];
            let d2 = (f_spec[v] as f64 / n_trials as f64) - target_p[v];
            if d1.abs() > 0.015 || d2.abs() > 0.015 {
                return false;
            }
        }
    }

    true
}
