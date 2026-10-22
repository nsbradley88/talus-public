//! Laguna-S-2.1 (`laguna`) stdio server, sampler, prompt-lookup speculative drafter,
//! and CLI execution modes (`src/laguna/laguna_serve.cpp` & `src/laguna/laguna_main.cpp`).

#![allow(non_upper_case_globals)]

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use talus_core_api::{CoreError, Result};

use crate::laguna::engine::dflash::{LagunaDFlash, DF_MAX_BLOCK, DF_WINDOW};
use crate::laguna::engine::{Engine, K_MAX_VERIFY};
use crate::laguna::weights::{K_EMBD, K_SWA, K_VOCAB};

/// Relative wall-time overhead per drafted token in `verify(1 + k)` vs `decode(1)`
/// (`src/laguna/laguna_serve.cpp:128`).
pub const K_DRAFT_COST: f64 = 0.40;
pub const kDraftCost: f64 = K_DRAFT_COST;

/// Kill switch for exact speculative sampling of sampled requests: default ON; `0`
/// restores the exact-match rule. The per-seed sampled bytes depend on it, the output distribution does not.
pub const SSAMP_ENV: &str = "TALUS_LAGUNA_SSAMP";

/// Default EOS token IDs for Laguna (`<|EOS|>` = 2, `</assistant>` = 24).
pub const DEFAULT_EOS_IDS: [i32; 2] = [2, 24];

/// Sampling parameters (`src/laguna/laguna_serve.cpp:Sampling`, `lines 40-50`).
#[derive(Debug, Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: i32,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub penalty_last_n: i32,
    pub seed: i64,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            min_p: 0.0,
            top_k: 0,
            penalty_repeat: 1.0,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            penalty_last_n: 64,
            seed: -1,
        }
    }
}

impl Sampling {
    /// Returns true when sampling is greedy without any repetition/frequency/presence penalties
    /// (`src/laguna/laguna_serve.cpp:47-49`).
    #[inline]
    pub fn greedy_plain(&self) -> bool {
        self.temperature <= 0.0
            && self.penalty_repeat == 1.0
            && self.penalty_freq == 0.0
            && self.penalty_present == 0.0
    }
}

/// Bit-exact 64-bit Mersenne Twister (`std::mt19937_64` from ISO C++11 `[rand.eng.mers]`).
#[derive(Clone, Debug)]
pub struct Mt19937_64 {
    mt: [u64; Self::NN],
    index: usize,
}

impl Mt19937_64 {
    const NN: usize = 312;
    const MM: usize = 156;
    const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
    const UM: u64 = 0xFFFF_FFFF_8000_0000;
    const LM: u64 = 0x0000_0000_7FFF_FFFF;
    const F: u64 = 6_364_136_223_846_793_005;

    pub fn new(seed: u64) -> Self {
        let mut s = Self {
            mt: [0u64; Self::NN],
            index: Self::NN,
        };
        s.seed(seed);
        s
    }

    pub fn seed(&mut self, seed: u64) {
        self.mt[0] = seed;
        for i in 1..Self::NN {
            let prev = self.mt[i - 1];
            self.mt[i] = Self::F
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        self.index = Self::NN;
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        if self.index >= Self::NN {
            self.twist();
        }
        let mut x = self.mt[self.index];
        self.index += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^ (x >> 43)
    }

    /// Matches `std::uniform_real_distribution<double>(0.0, tot)(rng)`.
    #[inline]
    pub fn uniform_f64(&mut self, tot: f64) -> f64 {
        let r = self.next_u64();
        const B_F64: f64 = 18_446_744_073_709_551_616.0_f64; // 2^64
        (r as f64 / B_F64) * tot
    }

    fn twist(&mut self) {
        for i in 0..Self::NN {
            let x = (self.mt[i] & Self::UM) | (self.mt[(i + 1) % Self::NN] & Self::LM);
            let mut xa = x >> 1;
            if (x & 1) != 0 {
                xa ^= Self::MATRIX_A;
            }
            self.mt[i] = self.mt[(i + Self::MM) % Self::NN] ^ xa;
        }
        self.index = 0;
    }
}

/// Apply repetition/frequency/presence penalties and sample a token ID from `logits`,
/// matching `src/laguna/laguna_serve.cpp:63-112` bit-for-bit.
pub fn sample(logits: &mut [f32], sp: &Sampling, recent: &[i32], rng: &mut Mt19937_64) -> i32 {
    let v_len = logits.len();
    assert!(v_len > 0);

    if sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0 {
        let n = recent.len().min(sp.penalty_last_n.max(0) as usize);
        let mut count: HashMap<i32, i32> = HashMap::with_capacity(n * 2);
        for &t in &recent[recent.len() - n..] {
            *count.entry(t).or_insert(0) += 1;
        }
        for (t, c) in count {
            if t >= 0 && (t as usize) < v_len {
                let l = &mut logits[t as usize];
                if sp.penalty_repeat != 1.0 {
                    *l = if *l > 0.0 {
                        *l / sp.penalty_repeat
                    } else {
                        *l * sp.penalty_repeat
                    };
                }
                *l -= (c as f32) * sp.penalty_freq + if c > 0 { sp.penalty_present } else { 0.0 };
            }
        }
    }

    if sp.temperature <= 0.0 {
        let mut best = 0usize;
        for i in 1..v_len {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return best as i32;
    }

    let k = if sp.top_k > 0 {
        (sp.top_k as usize).min(v_len)
    } else {
        v_len
    };
    let idx: Vec<i32> = talus_spec::select_top_k_cands(logits, v_len, k)
        .into_iter()
        .map(|c| c.id)
        .collect();

    let mx = logits[idx[0] as usize] as f64;
    let temp = sp.temperature as f64;
    let mut p = Vec::with_capacity(idx.len());
    let mut sum = 0.0_f64;
    for &id in &idx {
        let val = ((logits[id as usize] as f64 - mx) / temp).exp();
        p.push(val);
        sum += val;
    }
    for v in &mut p {
        *v /= sum;
    }

    let mut keep = p.len();
    if sp.min_p > 0.0 {
        let floor_p = p[0] * (sp.min_p as f64);
        keep = 1;
        while keep < p.len() && p[keep] >= floor_p {
            keep += 1;
        }
    }
    if sp.top_p < 1.0 {
        let mut c = 0.0_f64;
        let mut n = 0usize;
        while n < keep {
            c += p[n];
            n += 1;
            if c >= (sp.top_p as f64) {
                break;
            }
        }
        keep = n.max(1);
    }

    let tot: f64 = p[..keep].iter().sum();
    let mut r = rng.uniform_f64(tot);
    for i in 0..keep {
        r -= p[i];
        if r <= 0.0 {
            return idx[i];
        }
    }
    idx[keep - 1]
}

impl talus_spec::ssamp::Uniform01 for Mt19937_64 {
    #[inline]
    fn u01(&mut self) -> f64 {
        self.uniform_f64(1.0)
    }
}

/// The distribution [`sample`] draws from (temperature > 0): penalties -> top-k -> softmax at the temperature ->
/// min-p -> top-p, renormalised, as a sparse list in [`sample`]'s candidate order. `ids` maps a position of `logits`
/// to its token id (a drafter's top-16 candidate list); `None` = position is the token id (a full-vocab row).
/// Penalties look tokens up through `ids`. Mutates `logits` (penalties), as [`sample`] does.
///
/// Used for both the target row (`p`) and the drafter's candidates (`q`) by exact speculative sampling
/// (`talus_spec::ssamp`). Greedy (`temperature <= 0`) is the point mass at the argmax.
pub fn sample_dist(
    logits: &mut [f32],
    ids: Option<&[i32]>,
    sp: &Sampling,
    recent: &[i32],
) -> talus_spec::ssamp::SparseDist {
    use talus_spec::ssamp::SparseDist;
    let v_len = logits.len();
    if v_len == 0 {
        return SparseDist::new();
    }
    let tok = |i: usize| -> i32 { ids.map_or(i as i32, |m| m[i]) };
    if sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0 {
        let n = recent.len().min(sp.penalty_last_n.max(0) as usize);
        let mut count: HashMap<i32, i32> = HashMap::with_capacity(n * 2);
        for &t in &recent[recent.len() - n..] {
            *count.entry(t).or_insert(0) += 1;
        }
        for (t, c) in count {
            let pos = match ids {
                None => (t >= 0 && (t as usize) < v_len).then_some(t as usize),
                Some(m) => m.iter().position(|&x| x == t),
            };
            if let Some(i) = pos {
                let l = &mut logits[i];
                if sp.penalty_repeat != 1.0 {
                    *l = if *l > 0.0 {
                        *l / sp.penalty_repeat
                    } else {
                        *l * sp.penalty_repeat
                    };
                }
                *l -= (c as f32) * sp.penalty_freq + if c > 0 { sp.penalty_present } else { 0.0 };
            }
        }
    }
    if sp.temperature <= 0.0 {
        let mut best = 0usize;
        for i in 1..v_len {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return SparseDist::one_hot(tok(best));
    }
    let k = if sp.top_k > 0 {
        (sp.top_k as usize).min(v_len)
    } else {
        v_len
    };
    let idx: Vec<i32> = talus_spec::select_top_k_cands(logits, v_len, k)
        .into_iter()
        .map(|c| c.id)
        .collect();
    let mx = logits[idx[0] as usize] as f64;
    let temp = sp.temperature as f64;
    let mut p = Vec::with_capacity(idx.len());
    let mut sum = 0.0_f64;
    for &i in &idx {
        let val = ((logits[i as usize] as f64 - mx) / temp).exp();
        p.push(val);
        sum += val;
    }
    for v in &mut p {
        *v /= sum;
    }
    let mut keep = p.len();
    if sp.min_p > 0.0 {
        let floor_p = p[0] * (sp.min_p as f64);
        keep = 1;
        while keep < p.len() && p[keep] >= floor_p {
            keep += 1;
        }
    }
    if sp.top_p < 1.0 {
        let mut c = 0.0_f64;
        let mut n = 0usize;
        while n < keep {
            c += p[n];
            n += 1;
            if c >= (sp.top_p as f64) {
                break;
            }
        }
        keep = n.max(1);
    }
    let pairs = idx[..keep]
        .iter()
        .zip(p[..keep].iter())
        .map(|(&i, &v)| (tok(i as usize), v));
    if ids.is_none() {
        // full-vocab row: ids are unique, skip the duplicate merge (O(n^2) on a 150K-entry list)
        let mut d = SparseDist {
            ids: Vec::with_capacity(keep),
            p: Vec::with_capacity(keep),
        };
        for (id, v) in pairs {
            if v > 0.0 {
                d.ids.push(id);
                d.p.push(v);
            }
        }
        d.normalize();
        d
    } else {
        SparseDist::from_weights(pairs)
    }
}

/// A drafter's proposal distribution `q` over its top candidates `(target id, logit)` for a sampled request:
/// [`sample_dist`] with the request's transforms at `TALUS_SSAMP_QTEMP` x the temperature.
pub fn draft_q(
    cands: &[(i32, f32)],
    sp: &Sampling,
    recent: &[i32],
    q_temp_scale: f32,
) -> talus_spec::ssamp::SparseDist {
    let ids: Vec<i32> = cands.iter().map(|c| c.0).collect();
    let mut lg: Vec<f32> = cands.iter().map(|c| c.1).collect();
    let mut spq = *sp;
    spq.temperature *= q_temp_scale;
    sample_dist(&mut lg, Some(&ids), &spq, recent)
}

/// Prompt-lookup speculative drafter (`src/laguna/laguna_serve.cpp:Drafter`, `lines 117-146`).
#[derive(Debug, Clone, Default)]
pub struct Drafter {
    pub last3: HashMap<u64, usize>,
    pub last2: HashMap<u64, usize>,
    pub hist: Vec<i32>,
}

impl Drafter {
    pub fn new() -> Self {
        Self::default()
    }

    /// 64-bit FNV-1a n-gram hash (`src/laguna/laguna_serve.cpp:120-124`).
    #[inline]
    pub fn key(t: &[i32]) -> u64 {
        let mut h: u64 = 1_469_598_103_934_665_603_u64;
        for &tok in t {
            h = (h ^ (tok as u32 as u64)).wrapping_mul(1_099_511_628_211_u64);
        }
        h ^ (t.len() as u64)
    }

    /// Reset history and n-gram tables from `h` (`src/laguna/laguna_serve.cpp:125`).
    pub fn reset(&mut self, h: &[i32]) {
        self.hist.clear();
        self.last3.clear();
        self.last2.clear();
        for &t in h {
            self.push(t);
        }
    }

    /// Index the 3-gram and 2-gram ending immediately before `t`, mapping them to `t`'s position,
    /// then append `t` to `hist` (`src/laguna/laguna_serve.cpp:126-132`).
    pub fn push(&mut self, t: i32) {
        let n = self.hist.len();
        if n >= 3 {
            self.last3.insert(Self::key(&self.hist[n - 3..n]), n);
        }
        if n >= 2 {
            self.last2.insert(Self::key(&self.hist[n - 2..n]), n);
        }
        self.hist.push(t);
    }

    /// Look up up to `k` draft tokens following the current suffix of `hist` when the suffix
    /// matches earlier text for at least `min_match` tokens (`src/laguna/laguna_serve.cpp:135-146`).
    pub fn draft(&self, k: usize, out: &mut [i32], min_match: usize) -> usize {
        let n = self.hist.len();
        let mut at: Option<usize> = None;
        if n >= 3 {
            if let Some(&pos) = self.last3.get(&Self::key(&self.hist[n - 3..n])) {
                at = Some(pos);
            }
        }
        if at.is_none() && n >= 2 {
            if let Some(&pos) = self.last2.get(&Self::key(&self.hist[n - 2..n])) {
                at = Some(pos);
            }
        }
        let Some(at) = at else {
            return 0;
        };

        let mut matched = 0usize;
        while matched < min_match
            && at > matched
            && n > matched
            && self.hist[at - 1 - matched] == self.hist[n - 1 - matched]
        {
            matched += 1;
        }
        if matched < min_match {
            return 0;
        }

        let limit = k.min(out.len());
        let mut m = 0usize;
        while m < limit && at + m < n {
            out[m] = self.hist[at + m];
            m += 1;
        }
        m
    }

    /// Like `draft`, and also returns how many tokens of the suffix match the earlier text (checked up to 16).
    pub fn draft_m(&self, k: usize, out: &mut [i32], min_match: usize) -> (usize, usize) {
        let n = self.hist.len();
        let mut at: Option<usize> = None;
        if n >= 3 {
            if let Some(&pos) = self.last3.get(&Self::key(&self.hist[n - 3..n])) {
                at = Some(pos);
            }
        }
        if at.is_none() && n >= 2 {
            if let Some(&pos) = self.last2.get(&Self::key(&self.hist[n - 2..n])) {
                at = Some(pos);
            }
        }
        let Some(at) = at else {
            return (0, 0);
        };
        let mut matched = 0usize;
        while matched < 16
            && at > matched
            && n > matched
            && self.hist[at - 1 - matched] == self.hist[n - 1 - matched]
        {
            matched += 1;
        }
        if matched < min_match {
            return (0, matched);
        }
        let limit = k.min(out.len());
        let mut m = 0usize;
        while m < limit && at + m < n {
            out[m] = self.hist[at + m];
            m += 1;
        }
        (m, matched)
    }

    /// Select the optimal draft length `best_k` (`0..=min(spec, K_MAX_VERIFY - 1)`) by expected
    /// value given running acceptance estimate `acc_est` (`src/laguna/laguna_serve.cpp:286-294`).
    #[inline]
    pub fn select_best_k(spec: usize, acc_est: f64) -> usize {
        let max_kk = spec.min(K_MAX_VERIFY - 1);
        let mut best_k = 0usize;
        let mut best = 1.03_f64;
        for kk in 1..=max_kk {
            let ex = if acc_est >= 0.999 {
                (kk as f64) + 1.0
            } else {
                (1.0 - acc_est.powi((kk + 1) as i32)) / (1.0 - acc_est)
            };
            let r = ex / (1.0 + K_DRAFT_COST * (kk as f64));
            if r > best {
                best = r;
                best_k = kk;
            }
        }
        best_k
    }
}

/// Calibrated verify(n) round times (ms, n = 1 ..= 64) at depth 2K and 32K on gfx1151.
const VMS_2K: [f64; K_MAX_VERIFY] = [
    32.9, 46.1, 56.6, 66.3, 76.6, 85.0, 94.3, 104.9, 116.4, 132.0, 140.6, 150.7, 160.3, 172.1,
    180.6, 188.0, 206.2, 214.9, 225.3, 231.8, 242.2, 250.6, 262.6, 275.1, 282.6, 291.8, 304.5,
    314.0, 326.3, 336.5, 346.6, 359.9, 370.6, 382.7, 397.2, 408.0, 414.8, 420.2, 436.2, 438.9,
    448.1, 458.6, 470.2, 479.9, 494.7, 498.1, 507.7, 519.3, 533.3, 547.1, 551.5, 561.4, 572.5,
    581.1, 597.7, 600.1, 609.9, 623.9, 629.9, 648.4, 650.4, 664.6, 672.2, 690.8,
];
const VMS_32K: [f64; K_MAX_VERIFY] = [
    37.2, 52.2, 66.6, 80.1, 93.1, 108.2, 117.5, 132.1, 147.8, 164.2, 179.1, 194.5, 207.1, 220.4,
    233.8, 242.1, 263.7, 274.4, 287.6, 299.6, 309.3, 319.0, 331.7, 343.5, 357.1, 366.4, 380.2,
    395.9, 402.3, 417.1, 428.6, 441.9, 457.8, 482.5, 496.2, 511.3, 521.6, 528.2, 544.0, 547.9,
    564.0, 576.5, 588.0, 606.6, 612.5, 629.0, 643.0, 654.9, 668.8, 682.7, 696.2, 712.1, 719.4,
    732.9, 750.5, 765.3, 774.1, 786.1, 799.8, 816.3, 824.8, 841.5, 860.3, 862.7,
];

/// Verify(n) round times (ms, n = 1 ..= 64) with the grouped routed experts from 6 tokens
/// (gfx1151, 2K and 32K).
const VMS_LWV_2K: [f64; K_MAX_VERIFY] = [
    33.0, 46.1, 56.9, 66.8, 76.5, 83.3, 90.9, 99.9, 109.0, 119.4, 129.7, 137.0, 144.9, 154.5,
    161.4, 166.0, 180.5, 187.9, 195.1, 203.7, 210.8, 218.7, 227.7, 235.9, 245.3, 255.1, 264.7,
    273.3, 281.3, 291.0, 300.3, 305.0, 318.8, 326.8, 334.1, 340.7, 348.0, 355.4, 361.6, 370.1,
    376.3, 384.0, 390.7, 396.2, 404.2, 412.7, 419.9, 427.2, 439.3, 445.9, 453.0, 460.0, 466.4,
    473.0, 480.5, 487.9, 494.9, 501.9, 510.2, 517.1, 524.9, 535.6, 543.2, 547.7,
];
const VMS_LWV_32K: [f64; K_MAX_VERIFY] = [
    37.4, 52.3, 66.7, 80.4, 92.5, 102.8, 114.2, 127.5, 140.7, 153.2, 166.1, 177.9, 188.8, 199.7,
    210.4, 218.7, 235.2, 245.7, 255.4, 265.6, 276.7, 285.7, 295.2, 306.2, 315.7, 324.4, 336.0,
    344.2, 355.9, 368.0, 376.2, 386.4, 400.8, 411.2, 419.3, 431.0, 438.8, 449.4, 460.3, 469.5,
    480.5, 493.3, 502.0, 511.7, 522.6, 534.1, 544.6, 553.6, 570.9, 580.3, 590.6, 600.4, 610.1,
    620.9, 634.9, 646.2, 652.7, 665.7, 673.7, 683.0, 693.7, 705.3, 714.6, 722.7,
];
/// Verify(n) round times with the exact wide dense kernels (native_mmvq_xt, STRATA_MMVQ_XT) and the grouped
/// routed experts (gfx1151, 2K and 32K; every row bitwise verify(1)).
const VMS_XT_2K: [f64; K_MAX_VERIFY] = [
    33.0, 46.3, 56.8, 66.9, 77.3, 83.2, 90.6, 99.3, 109.7, 119.5, 129.5, 135.9, 143.9, 151.6,
    159.5, 164.6, 182.8, 190.3, 195.6, 204.8, 213.3, 220.1, 229.1, 236.4, 247.1, 254.6, 264.7,
    270.0, 279.7, 285.5, 293.5, 298.0, 315.8, 321.0, 328.8, 333.4, 341.5, 348.8, 352.9, 358.5,
    377.0, 384.4, 388.7, 392.0, 403.0, 408.5, 416.9, 421.0, 441.1, 447.0, 453.2, 459.8, 466.8,
    470.6, 476.4, 483.7, 496.3, 500.2, 510.0, 513.4, 521.5, 529.5, 533.9, 539.5,
];
const VMS_XT_32K: [f64; K_MAX_VERIFY] = [
    37.4, 52.4, 66.3, 80.0, 93.4, 101.0, 113.6, 124.8, 138.7, 150.0, 163.3, 174.2, 184.7, 195.2,
    204.4, 214.6, 234.1, 243.7, 254.8, 263.5, 275.8, 284.0, 294.0, 303.3, 312.8, 324.1, 333.2,
    343.5, 354.6, 364.0, 373.3, 380.9, 401.4, 411.3, 419.2, 428.7, 440.3, 450.8, 457.9, 467.8,
    487.0, 496.7, 505.7, 519.4, 524.7, 537.0, 545.9, 555.5, 577.5, 584.8, 596.8, 604.4, 614.9,
    626.0, 636.6, 646.4, 659.9, 668.5, 676.6, 690.3, 697.5, 709.2, 714.4, 728.2,
];
/// Verify(n) round times with the opt-in GEMM verify from 9 tokens (TALUS_LAGUNA_VERIFY_GEMM=9; calibrated on
/// the earlier grouped kernel, below 9 tokens only approximate).
const VMS_GEMM_2K: [f64; K_MAX_VERIFY] = [
    32.9, 46.0, 56.9, 66.6, 76.3, 85.1, 94.3, 104.7, 112.5, 119.9, 124.6, 128.5, 131.4, 134.1,
    136.1, 135.0, 164.0, 167.3, 168.9, 172.9, 176.3, 178.4, 182.5, 186.9, 192.7, 197.1, 201.7,
    206.0, 209.5, 213.3, 216.2, 215.1, 237.1, 239.8, 241.2, 242.7, 246.0, 248.8, 250.7, 254.2,
    256.7, 258.5, 259.6, 260.2, 262.4, 263.6, 264.9, 263.9, 280.6, 281.9, 283.7, 284.2, 285.2,
    286.9, 287.4, 289.8, 291.1, 291.9, 292.3, 293.3, 295.6, 296.8, 297.3, 296.4,
];
const VMS_GEMM_32K: [f64; K_MAX_VERIFY] = [
    37.2, 52.1, 67.0, 80.2, 92.6, 104.9, 117.9, 132.9, 151.0, 158.0, 165.1, 170.9, 175.0, 178.1,
    180.8, 179.9, 210.9, 213.8, 218.8, 221.4, 225.0, 223.6, 225.4, 227.3, 230.4, 231.3, 232.3,
    233.9, 236.9, 238.9, 240.5, 238.7, 259.3, 260.2, 262.4, 263.6, 265.5, 271.3, 274.3, 276.5,
    280.0, 284.0, 294.7, 296.0, 298.6, 301.4, 303.6, 302.2, 320.4, 321.7, 323.7, 325.2, 327.0,
    327.9, 331.5, 334.5, 338.0, 339.4, 341.4, 342.6, 343.8, 347.0, 348.4, 346.2,
];

/// Per-width round costs (ms) and the draft-length choice. Verify rows are exact, so the choice only moves
/// time, never the greedy output. `fixed` (the default, TALUS_LAGUNA_SPEC_STATIC=0: off): the calibrated verify times
/// above, interpolated by depth, so the drafting decisions depend on token counts and acceptance only (never on
/// wall-clock time); otherwise an EMA of the decode / verify wall times.
struct SpecCost {
    ms: [f64; K_MAX_VERIFY + 1],
    seen: [bool; K_MAX_VERIFY + 1],
    df_ms: f64,
    fixed: bool,
    depth: usize,
    // the fixed tables (2K, 32K) of the verify kernels in use
    tab: (&'static [f64; K_MAX_VERIFY], &'static [f64; K_MAX_VERIFY]),
}

/// the fixed verify-time tables for the kernels this engine runs: the wide-verify tables with the grouped routed
/// experts on (the default), else the earlier tables (TALUS_LAGUNA_EGROUP_MIN=0 keeps the earlier drafting decisions too).
/// When the wide dense verify kernels are on (STRATA_MMVQ_XT > 0; the Laguna CLI sets 9 in served mode), uses VMS_XT.
fn xt_active() -> bool {
    // TALUS_LAGUNA_XT_TABLES=0: keep the earlier wide-verify tables with the xt kernels on
    if std::env::var("TALUS_LAGUNA_XT_TABLES")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return false;
    }
    std::env::var("STRATA_MMVQ_XT")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .map(|v| v > 0)
        .unwrap_or(false)
}
fn verify_tables(eng: &Engine) -> (&'static [f64; K_MAX_VERIFY], &'static [f64; K_MAX_VERIFY]) {
    let (egroup, vgemm) = eng.verify_modes();
    if vgemm > 0 {
        // opt-in GEMM verify: measured with TALUS_LAGUNA_VERIFY_GEMM=9 (and the grouped experts below it)
        (&VMS_GEMM_2K, &VMS_GEMM_32K)
    } else if egroup > 0 && xt_active() {
        (&VMS_XT_2K, &VMS_XT_32K)
    } else if egroup > 0 {
        (&VMS_LWV_2K, &VMS_LWV_32K)
    } else {
        (&VMS_2K, &VMS_32K)
    }
}

impl SpecCost {
    fn new(fixed: bool, tab: (&'static [f64; K_MAX_VERIFY], &'static [f64; K_MAX_VERIFY])) -> Self {
        let mut ms = [0.0; K_MAX_VERIFY + 1];
        for (w, m) in ms.iter_mut().enumerate() {
            // Initial prior before depth interpolation: linear slope up to 8 tokens, shallower slope past 8 where
            // the window's tokens start sharing routed experts in the cache.
            let w = w.max(1);
            *m = if w <= 8 {
                32.0 + 11.0 * (w - 1) as f64
            } else {
                109.0 + 10.0 * (w - 8) as f64
            };
        }
        let mut c = Self {
            ms,
            seen: [false; K_MAX_VERIFY + 1],
            df_ms: if fixed { 7.3 } else { 10.0 },
            fixed,
            depth: usize::MAX,
            tab,
        };
        c.at_depth(0);
        c
    }

    /// The fixed table at context depth `d` (no-op for the measured EMA).
    fn at_depth(&mut self, d: usize) {
        if !self.fixed {
            return;
        }
        // linear in depth between 2K and 32K, held outside (block-sparse attention keeps the cost flat past 32K);
        // quantised to 256 positions so the table is not rebuilt every token
        let d = d.clamp(2048, 32768) & !255;
        if d == self.depth {
            return;
        }
        self.depth = d;
        let t = (d - 2048) as f64 / (32768 - 2048) as f64;
        for w in 1..=K_MAX_VERIFY {
            self.ms[w] = self.tab.0[w - 1] + t * (self.tab.1[w - 1] - self.tab.0[w - 1]);
        }
        self.ms[0] = self.ms[1];
    }

    fn record_df(&mut self, ms: f64) {
        if !self.fixed {
            self.df_ms = 0.8 * self.df_ms + 0.2 * ms;
        }
    }

    fn record(&mut self, w: usize, ms: f64) {
        if self.fixed || !(1..=K_MAX_VERIFY).contains(&w) || ms <= 0.0 {
            return;
        }
        // the first call of a width captures its graph: not a round cost; a stall (another process, a page fault
        // storm) is clipped so one sample cannot lock a width out
        if !self.seen[w] {
            self.seen[w] = true;
            return;
        }
        let ms = ms.min(2.0 * self.ms[w]);
        self.ms[w] = 0.8 * self.ms[w] + 0.2 * ms;
    }

    /// The draft length in `0..=kmax` with the best expected tokens per ms, given the acceptance chance of each
    /// successive draft (`p[i]`: draft i + 1 accepted when draft i was) and a fixed extra cost per drafted round.
    /// Returns the length and its expected tokens per ms (0 when no draft exceeds a plain decode step).
    fn best_k_rate(&self, p: &[f64], kmax: usize, extra_ms: f64) -> (usize, f64) {
        let mut best = 1.0 / self.ms[1];
        let mut best_k = 0usize;
        let mut ex = 1.0;
        let mut chain = 1.0;
        for k in 1..=kmax.min(K_MAX_VERIFY - 1).min(p.len()) {
            chain *= p[k - 1];
            ex += chain;
            let r = ex / (self.ms[k + 1] + extra_ms);
            if r > best * 1.02 {
                best = r;
                best_k = k;
            }
        }
        (best_k, if best_k > 0 { best } else { 0.0 })
    }
}

struct InputQueue {
    mu: Mutex<(VecDeque<String>, bool)>,
    cv: Condvar,
    stop: AtomicBool,
}

impl InputQueue {
    fn new() -> Self {
        Self {
            mu: Mutex::new((VecDeque::new(), false)),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
        }
    }

    fn next_line(&self) -> Option<String> {
        let mut guard = self.mu.lock().unwrap();
        while guard.0.is_empty() && !guard.1 {
            guard = self.cv.wait(guard).unwrap();
        }
        guard.0.pop_front()
    }
}

#[inline]
fn emit(line: &str) {
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

#[inline]
fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Run the Laguna stdio server loop (`src/laguna/laguna_serve.cpp:serve_main`).
pub fn serve_main(
    eng: &mut Engine,
    eos_ids: &[i32],
    spec: usize,
    spec_min_match: usize,
    load_seconds: f64,
    mut df: Option<&mut LagunaDFlash>,
    fast_serve: bool,
) -> Result<()> {
    // DFlash block size (drafts per round) and the prompt-lookup length that wins over a DFlash draft
    let df_k = std::env::var("TALUS_LAGUNA_DF_K")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(7)
        .clamp(1, DF_MAX_BLOCK - 1);
    // TALUS_LAGUNA_SPEC=N overrides --spec (prompt-lookup drafts per round, up to K_MAX_VERIFY - 1)
    // TALUS_LAGUNA_LSRV=0 (master switch): legacy serve behaviour
    let master_off = std::env::var("TALUS_LAGUNA_LSRV")
        .map(|v| v == "0")
        .unwrap_or(false);
    let spec_env = std::env::var("TALUS_LAGUNA_SPEC")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|_| !master_off);
    // Fast serve mode: prompt lookup up to 63 drafts by default (--spec 0 still turns it off)
    let spec = spec_env
        .unwrap_or(if fast_serve && spec > 0 {
            K_MAX_VERIFY - 1
        } else {
            spec
        })
        .min(K_MAX_VERIFY - 1);
    // Fast serve mode (the default; or the drafter / TALUS_LAGUNA_SPEC): calibrated-cost draft lengths, decode through
    // verify(1) (greedy output independent of the drafts). Without either, legacy policy and decode kernels.
    // TALUS_LAGUNA_SPEC_COST / TALUS_LAGUNA_DEC_VERIFY = 0 / 1 force either part.
    let fast_mode = !master_off && (fast_serve || df.is_some() || spec_env.is_some());
    let cost_on = std::env::var("TALUS_LAGUNA_SPEC_COST")
        .map(|v| v != "0")
        .unwrap_or(fast_mode);
    eng.set_dec_verify(fast_mode);
    eprintln!(
        "talus-laguna: spec {} (cost model {}), dflash {}, decode via verify {}, sparse exact {}",
        spec,
        cost_on,
        df.is_some(),
        eng.dec_verify(),
        eng.sparse_exact()
    );
    // Every verify / DFlash graph a request can reach is captured here, before READY (TALUS_LAGUNA_GRAPH_WARM=0:
    // on first use).
    if fast_mode
        && std::env::var("TALUS_LAGUNA_GRAPH_WARM")
            .map(|v| v != "0")
            .unwrap_or(true)
    {
        let tw = Instant::now();
        let wmax =
            (if cost_on { spec } else { spec.min(7) }).max(if df.is_some() { df_k } else { 0 }) + 1;
        let mut n_graphs = eng.capture_verify_graphs(wmax)?;
        if let Some(d) = df.as_deref_mut() {
            n_graphs += d.capture_graphs(eng, df_k + 1, wmax.min(K_MAX_VERIFY))?;
        }
        eprintln!(
            "talus-laguna: captured {n_graphs} graphs (verify widths 1..={wmax}) in {:.0} ms",
            ms_since(tw)
        );
    }
    // TALUS_LAGUNA_SPEC_STATIC=0: wall-clock round costs
    let spec_static = std::env::var("TALUS_LAGUNA_SPEC_STATIC")
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut cost = SpecCost::new(spec_static, verify_tables(eng));
    // DFlash: chance that draft i + 1 is accepted when draft i was (EMA over rounds and requests)
    // [greedy, sampled]: a sampled request accepts a draft only when its draw equals it
    // TALUS_LAGUNA_DF_P0 = the prior of a DFlash position (0.85; was 0.6). Positions the policy never drafts stay
    // at their prior, so under 0.6 DFlash rounds never went past 3 drafts although later positions are accepted at
    // high rates once drafted (DFlash vs lookup / copy is decided under the 0.6-prior estimate, TALUS_LAGUNA_DF_P0_CMP).
    // Drafting only: greedy output never depends on it.
    let p_df0: f64 = std::env::var("TALUS_LAGUNA_DF_P0")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.85)
        .clamp(0.05, 0.99);
    // TALUS_LAGUNA_DF_EXPLORE=N: every N-th DFlash round drafts the full block (df_k), so the later positions'
    // acceptance is measured instead of staying at the prior (0 = off). Only time moves: greedy output never depends on
    // the drafts, sampled requests keep the exact rule.
    let df_explore: usize = std::env::var("TALUS_LAGUNA_DF_EXPLORE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let mut df_round_ctr: usize = 0;
    let p_df0_init: f64 = p_df0;
    let mut p_df_all = [[p_df0_init; DF_MAX_BLOCK - 1]; 2];
    // the same estimates under the 0.6 prior (TALUS_LAGUNA_DF_P0_CMP), used only to compare a DFlash round
    // against a prompt-lookup / copy draft, so an optimistic prior sizes DFlash rounds without outbidding copies
    let p_df_cmp0: f64 = std::env::var("TALUS_LAGUNA_DF_P0_CMP")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.6)
        .clamp(0.05, 0.99);
    let mut p_df_cmp = [[p_df_cmp0; DF_MAX_BLOCK - 1]; 2];
    // every request starts from these priors (TALUS_LAGUNA_SPEC_RESET=0: carried across requests), and an
    // estimate a round did not draft relaxes toward its prior (TALUS_LAGUNA_SPEC_DECAY=0: off), so a length the policy
    // stopped drafting cannot stay frozen at a low value
    let spec_reset = std::env::var("TALUS_LAGUNA_SPEC_RESET")
        .map(|v| v != "0")
        .unwrap_or(true);
    let spec_decay = std::env::var("TALUS_LAGUNA_SPEC_DECAY")
        .map(|v| v != "0")
        .unwrap_or(true);
    let mut df_pause = 0usize;
    let kv_clear = std::env::var("TALUS_LAGUNA_KV_CLEAR")
        .map(|v| v != "0")
        .unwrap_or(true);
    // the sliding-window rings too (TALUS_LAGUNA_SWA_CLEAR=0: full layers only)
    let swa_clear = kv_clear
        && std::env::var("TALUS_LAGUNA_SWA_CLEAR")
            .map(|v| v != "0")
            .unwrap_or(true);
    // prompt lookup: acceptance per drafted token by suffix-match bucket (< 8, >= 8 tokens), and the shortest match
    // that may draft (TALUS_LAGUNA_PL_MIN, default --spec-min-match)
    // bucket 2 = a long-span copy (talus_spec::CopyProposer, idea from gufo, MIT): the longest of the recent
    // matches, >= 12 tokens fresh (sampled: 24, from the request's starting context), >= 5 where the last copy stopped
    const PL_ACC0: [f64; 3] = [0.5, 0.9, 0.95];
    let mut pl_acc_all = [PL_ACC0; 2]; // [greedy, sampled][bucket]
                                       // default ON (TALUS_LAGUNA_COPY=0: off).
    let mut lcopy = (talus_spec::copy_enabled()
        && std::env::var("TALUS_LAGUNA_COPY")
            .map(|v| v != "0")
            .unwrap_or(true))
    .then(|| talus_spec::CopyProposer::new(talus_spec::copy_cfg_from_env()));
    let mut lcopy_out: Vec<i32> = Vec::new();
    let pl_min = std::env::var("TALUS_LAGUNA_PL_MIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(spec_min_match)
        .max(1);
    let v_len = K_VOCAB;
    let ctx = eng.max_ctx();
    let kv_str = if eng.is_kv_q8() { "q8" } else { "f16" };
    emit(&format!(
        "INFO engine=talus model=laguna vocab={} ubatch={} kv={} load_s={:.1}",
        v_len,
        eng.ubatch(),
        kv_str,
        load_seconds
    ));
    emit(&format!("READY {ctx} stop"));

    let input = Arc::new(InputQueue::new());
    {
        let input_bg = Arc::clone(&input);
        std::thread::spawn(move || {
            let stdin = io::stdin();
            for line_res in stdin.lock().lines() {
                match line_res {
                    Ok(mut line) => {
                        if line.ends_with('\r') {
                            line.pop();
                        }
                        if line == "STOP" {
                            input_bg.stop.store(true, Ordering::SeqCst);
                            continue;
                        }
                        let mut guard = input_bg.mu.lock().unwrap();
                        guard.0.push_back(line);
                        input_bg.cv.notify_one();
                    }
                    Err(_) => break,
                }
            }
            let mut guard = input_bg.mu.lock().unwrap();
            guard.1 = true;
            input_bg.cv.notify_one();
        });
    }

    let mut cached: Vec<i32> = Vec::new();
    let mut logits = vec![0.0_f32; v_len];
    let mut rng = Mt19937_64::new(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42),
    );

    while let Some(line) = input.next_line() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.split_whitespace();
        let cmd = parts.next().unwrap_or("");
        if cmd == "QUIT" {
            break;
        }
        if cmd == "SAVE" || cmd == "RESTORE" {
            let rest = trimmed[cmd.len()..].trim();
            if rest.is_empty() {
                emit("SERR invalid 0 no path");
                continue;
            }
            let ts = Instant::now();
            let path = Path::new(rest);
            let res = if cmd == "SAVE" {
                if cached.is_empty() {
                    Err("invalid nothing to save".to_string())
                } else {
                    eng.save_session(path, &cached, |done, total| {
                        emit(&format!("SESSION {done} {total}"));
                    })
                }
            } else {
                match eng.load_session(path, |done, total| {
                    emit(&format!("SESSION {done} {total}"));
                }) {
                    Ok((toks, bytes)) => {
                        cached = toks;
                        Ok(bytes)
                    }
                    Err(err) => {
                        if err.starts_with("io read failed part way") {
                            cached.clear();
                        }
                        Err(err)
                    }
                }
            };
            match res {
                Ok(bytes) => {
                    let tag = if cmd == "SAVE" { "SAVED" } else { "RESTORED" };
                    emit(&format!(
                        "{tag} {} {bytes} {:.1}",
                        cached.size_tokens(),
                        ms_since(ts)
                    ));
                }
                Err(err) => {
                    let (kind, why) = match err.split_once(' ') {
                        Some((k, w)) => (k, w),
                        None => (err.as_str(), err.as_str()),
                    };
                    let valid_kind = matches!(kind, "io" | "invalid" | "memory" | "storage");
                    emit(&format!(
                        "SERR {} 0 {}",
                        if valid_kind { kind } else { "io" },
                        why
                    ));
                }
            }
            continue;
        }
        // Diagnostic: FORCE <out.f32> <n_force> <id,id,...> = prefill all but the last n_force ids from position 0
        // (no prefix reuse), decode the last n_force one by one (teacher-forced) and append their logit rows to out.f32
        // probe: KVDUMP <path> <from> <to>: the stored K / V cells of positions from .. to (kv_dump), then the block
        // bounds, as raw bytes
        if cmd == "KVDUMP" {
            let path = parts.next().unwrap_or("").to_string();
            let a: usize = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let b: usize = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let (kv, bnd) = eng.kv_dump(a, b)?;
            let ok = std::fs::write(&path, [kv.as_slice(), bnd.as_slice()].concat()).is_ok();
            emit(&format!("KVDUMPED {} {} {}", kv.len(), bnd.len(), ok));
            continue;
        }
        if cmd == "FORCE" {
            let out = parts.next().unwrap_or("");
            let nf: usize = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            let ids: Vec<i32> = parts
                .next()
                .unwrap_or("")
                .split(',')
                .filter_map(|t| t.parse().ok())
                .collect();
            if ids.len() <= nf || nf == 0 || ids.len() + 8 > ctx {
                emit("ERR FORCE <out> <n_force> <ids>");
                continue;
            }
            let np = ids.len() - nf;
            eng.clear_kv_from(0, swa_clear)?;
            let mut done = 0usize;
            while done < np {
                let n = eng.ubatch().min(np - done);
                let last = done + n == np;
                eng.prefill(
                    &ids[done..done + n],
                    done,
                    if last { Some(&mut logits) } else { None },
                    None,
                )?;
                if let Some(d) = df.as_deref_mut() {
                    let from = np.saturating_sub(DF_WINDOW).max(done);
                    if from < done + n {
                        d.inject(eng, from - done, from, done + n - from)?;
                    }
                }
                done += n;
            }
            let mut rows: Vec<u8> = Vec::with_capacity((nf + 1) * v_len * 4);
            let push = |rows: &mut Vec<u8>, l: &[f32]| {
                for v in l {
                    rows.extend_from_slice(&v.to_le_bytes());
                }
            };
            push(&mut rows, &logits);
            for i in 0..nf {
                let pos = np + i;
                eng.decode(ids[pos], pos as i32, Some(&mut logits))?;
                if let Some(d) = df.as_deref_mut() {
                    d.inject(eng, 0, pos, 1)?;
                }
                if i + 1 < nf {
                    push(&mut rows, &logits);
                }
            }
            cached = ids.clone();
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(out)
                .and_then(|mut f| f.write_all(&rows));
            emit(&format!("FORCED {} {}", np, nf));
            continue;
        }
        if cmd != "GEN" {
            emit("ERR unsupported command (the Laguna engine takes GEN, STOP, QUIT)");
            continue;
        }

        // GEN <max_new> [key=val ...] <id,id,...>
        let Some(max_new_str) = parts.next() else {
            emit("ERR expected: GEN <max_new> <id,id,...>");
            continue;
        };
        let Ok(max_new) = max_new_str.parse::<i64>() else {
            emit("ERR expected: GEN <max_new> <id,id,...>");
            continue;
        };
        if max_new < 0 {
            emit("ERR expected: GEN <max_new> <id,id,...>");
            continue;
        }

        let mut sp = Sampling::default();
        let mut ignore_eos = false;
        let mut use_cache = true;
        let mut ids_text = "";
        for tok in parts {
            if let Some((k, v)) = tok.split_once('=') {
                match k {
                    "temperature" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.temperature = x;
                        }
                    }
                    "top_p" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.top_p = x;
                        }
                    }
                    "top_k" => {
                        if let Ok(x) = v.parse::<i32>() {
                            sp.top_k = x;
                        }
                    }
                    "min_p" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.min_p = x;
                        }
                    }
                    "seed" => {
                        if let Ok(x) = v.parse::<i64>() {
                            sp.seed = x;
                        }
                    }
                    "penalty_repeat" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.penalty_repeat = x;
                        }
                    }
                    "penalty_freq" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.penalty_freq = x;
                        }
                    }
                    "penalty_present" => {
                        if let Ok(x) = v.parse::<f32>() {
                            sp.penalty_present = x;
                        }
                    }
                    "penalty_last_n" => {
                        if let Ok(x) = v.parse::<i32>() {
                            sp.penalty_last_n = x;
                        }
                    }
                    "ignore_eos" => {
                        ignore_eos = matches!(v, "1" | "true" | "yes");
                    }
                    "cache" => {
                        use_cache = !matches!(v, "0" | "false" | "no");
                    }
                    _ => {}
                }
            } else {
                ids_text = tok;
                break;
            }
        }

        let mut ids: Vec<i32> = Vec::new();
        let mut bad = false;
        for t in ids_text.split(',') {
            if t.is_empty() {
                continue;
            }
            match t.parse::<i32>() {
                Ok(id) if id >= 0 && (id as usize) < v_len => ids.push(id),
                _ => {
                    bad = true;
                    break;
                }
            }
        }
        if bad || ids.is_empty() {
            emit("ERR bad prompt ids");
            continue;
        }
        if (ids.len() as i64) + max_new + 8 > (ctx as i64) {
            emit(&format!(
                "ERR prompt {} + max_new {} exceeds the context {}",
                ids.len(),
                max_new,
                ctx
            ));
            continue;
        }
        if sp.seed >= 0 {
            rng.seed(sp.seed as u64);
        }
        input.stop.store(false, Ordering::SeqCst);

        // Reuse longest common prefix with cached tokens (at least 1 token must be prefilled)
        let tp = Instant::now();
        let mut lcp = 0usize;
        if use_cache {
            while lcp < cached.len() && lcp < ids.len() && cached[lcp] == ids[lcp] {
                lcp += 1;
            }
            if lcp == ids.len() && lcp > 0 {
                lcp -= 1;
            }
            if lcp > 0 && eng.ring() < eng.ctx16() {
                // the ring slots a resumed request keeps must still hold their positions
                let first_swa = if swa_clear {
                    Engine::swa_keep_from(lcp)
                } else {
                    lcp.saturating_sub(K_SWA - 1)
                };
                let ring_end = cached.len() + K_MAX_VERIFY;
                if ring_end.saturating_sub(first_swa) > eng.ring() {
                    lcp = 0;
                    cached.clear();
                }
            }
        } else {
            cached.clear();
        }
        // cells past the reused prefix look as on a fresh engine (TALUS_LAGUNA_KV_CLEAR=0: off)
        if kv_clear {
            eng.clear_kv_from(lcp, swa_clear)?;
        }
        if lcp > 0 {
            emit(&format!("RESUME {lcp}"));
        }
        let n_read = ids.len() - lcp;
        let ubatch = eng.ubatch();
        let mut done = 0usize;
        while done < n_read {
            let n = ubatch.min(n_read - done);
            let is_last = done + n == n_read;
            eng.prefill(
                &ids[lcp + done..lcp + done + n],
                lcp + done,
                if is_last { Some(&mut logits) } else { None },
                None,
            )?;
            if let Some(d) = df.as_deref_mut() {
                // the drafter only attends to the last DF_WINDOW positions of the context
                let p0 = lcp + done;
                let from = ids.len().saturating_sub(DF_WINDOW).max(p0);
                if from < p0 + n {
                    d.inject(eng, from - p0, from, p0 + n - from)?;
                }
            }
            done += n;
            let ms = ms_since(tp);
            let tps = if ms > 0.0 {
                (done as f64) / (ms / 1000.0)
            } else {
                0.0
            };
            emit(&format!("PP {done} {n_read} {:.0} {:.1}", ms, tps));
        }
        cached.clear();
        cached.extend_from_slice(&ids);
        let prompt_ms = ms_since(tp);

        // Generation loop (with prompt-lookup speculation when spec > 0)
        let td = Instant::now();
        let recent_start = ids.len().saturating_sub(256);
        let mut recent: Vec<i32> = ids[recent_start..].to_vec();
        let mut gen: i64 = 0;
        let mut out_log: Vec<i32> = Vec::new();
        let mut drafted: i64 = 0;
        let mut accepted_total: i64 = 0;
        let mut finish = "length";
        let mut next = sample(&mut logits, &sp, &recent, &mut rng);
        let fast = sp.greedy_plain();
        if spec_reset {
            p_df_all = [[p_df0_init; DF_MAX_BLOCK - 1]; 2];
            p_df_cmp = [[p_df_cmp0; DF_MAX_BLOCK - 1]; 2];
            pl_acc_all = [PL_ACC0; 2];
            df_pause = 0;
        }
        let mut dr = Drafter::new();
        let mut spec_svc = talus_spec::SpecEngineService::from_decl(
            talus_spec::ModelSpecDecl::laguna(spec, spec_min_match),
        );
        let use_suffix_fallback = std::env::var("TALUS_LAGUNA_SUFFIX")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if spec > 0 {
            dr.reset(&cached);
            spec_svc.reset(&cached);
            if let Some(c) = lcopy.as_mut() {
                c.sync(&cached);
                c.begin_request();
                let _ = c.take_stats();
            }
        }
        let mut acc_est = 0.9_f64;
        let mut df_rounds: i64 = 0;
        let mut df_acc: i64 = 0;
        let mut df_last = false;
        let mut vlog: Vec<f32> = Vec::new();
        let mut ended = false;
        // exact speculative sampling of the DFlash drafts for sampled requests (default on;
        // TALUS_LAGUNA_SSAMP=0 = kill switch). Prompt-lookup drafts keep the exact-match rule (a point-mass q). Greedy never takes it.
        let ssamp_on = !fast && sp.temperature > 0.0 && talus_spec::ssamp::ssamp_enabled(SSAMP_ENV);
        let q_scale = talus_spec::ssamp::q_temp_scale();
        let pen_on =
            sp.penalty_repeat != 1.0 || sp.penalty_freq != 0.0 || sp.penalty_present != 0.0;
        let mut qs: Vec<talus_spec::ssamp::SparseDist> = Vec::new();

        while !ended {
            let mut drafts = [0i32; K_MAX_VERIFY];
            let mut nd = 0usize;
            let room = (max_new - gen - 1).max(0) as usize;
            // without per-token sparse bounds (TALUS_LAGUNA_SPARSE_EXACT=0) verify windows stay <= 8 once block-sparse
            // attention is on (a window's later keys enter its block bounds)
            let pos_now = cached.len();
            let wmax = if eng.is_sparse()
                && !eng.sparse_exact()
                && pos_now + K_MAX_VERIFY >= eng.sparse_min_pos().max(0) as usize
            {
                8
            } else {
                K_MAX_VERIFY
            };
            cost.at_depth(pos_now);
            let mut pl_bucket = usize::MAX;
            let mut pl_rate = 0.0;
            let mut copied: Option<talus_spec::CopyMatch> = None;
            if spec > 0 && cost_on {
                // Look up the whole continuation, then size it by the cost model with the acceptance seen at
                // this match length (bucket: suffix match < 8 or >= 8 tokens; 2 = a long-span copy)
                dr.push(next);
                spec_svc.ingest_token(next);
                if let Some(c) = lcopy.as_mut() {
                    c.push(next);
                }
                let cap = spec.min(wmax - 1).min(room);
                if let (true, Some(c)) = (cap > 0, lcopy.as_ref()) {
                    let cm = c.propose(
                        &[],
                        cap,
                        !fast,
                        false,
                        |t| eos_ids.contains(&t),
                        &mut lcopy_out,
                    );
                    if let Some(cm) = cm {
                        let n = lcopy_out.len();
                        drafts[..n].copy_from_slice(&lcopy_out);
                        let (k, r) = cost.best_k_rate(
                            &[pl_acc_all[usize::from(!fast)][2]; K_MAX_VERIFY],
                            n,
                            0.0,
                        );
                        if k > 0 {
                            nd = k;
                            pl_bucket = 2;
                            pl_rate = r;
                            copied = Some(cm);
                        }
                    }
                }
                if cap > 0 && copied.is_none() {
                    let (m, matched) = dr.draft_m(cap, &mut drafts, pl_min);
                    if m > 0 {
                        let b = usize::from(matched >= 8);
                        let (k, r) = cost.best_k_rate(
                            &[pl_acc_all[usize::from(!fast)][b]; K_MAX_VERIFY],
                            m,
                            0.0,
                        );
                        if k > 0 {
                            nd = k;
                            pl_bucket = b;
                            pl_rate = r;
                        }
                    }
                }
            } else if spec > 0 {
                dr.push(next);
                spec_svc.ingest_token(next);
                if let Some(c) = lcopy.as_mut() {
                    c.push(next);
                }
                let best_k = Drafter::select_best_k(spec.min(7), acc_est);
                if best_k > 0 && room > 0 {
                    let cap = best_k.min(room);
                    nd = dr.draft(cap, &mut drafts, spec_min_match);
                    if nd == 0 && use_suffix_fallback {
                        nd = spec_svc.draft_host(cap, &mut drafts);
                    }
                }
                if nd == 0 {
                    acc_est += 0.05 * (0.9 - acc_est);
                }
            }
            if let Some(d) = df.as_deref_mut() {
                let kmax = df_k.min(room).min(wmax - 1);
                let (mut k, _) = cost.best_k_rate(&p_df_all[usize::from(!fast)], kmax, cost.df_ms);
                // whether to draft DFlash at all, and DFlash against a lookup / copy draft, under the conservative
                // prior; the optimistic prior only sizes a round the conservative estimate already takes
                let (k_cmp, r) = cost.best_k_rate(&p_df_cmp[usize::from(!fast)], kmax, cost.df_ms);
                if k_cmp == 0 {
                    k = 0;
                }
                if nd > 0 && r <= pl_rate {
                    k = 0; // the prompt-lookup draft is the better bet
                } else if nd > 0 {
                    nd = 0; // DFlash expects more tokens per ms than this lookup
                    pl_bucket = usize::MAX;
                    copied = None;
                }
                if nd == 0 {
                    if k == 0 && kmax > 0 {
                        // paused: probe every 16th round so the acceptance estimate can recover
                        df_pause += 1;
                        if df_pause >= 16 {
                            df_pause = 0;
                            k = kmax.min(2);
                        }
                    }
                    if k > 0 && df_explore > 0 {
                        df_round_ctr += 1;
                        if df_round_ctr % df_explore == 0 {
                            k = kmax;
                        }
                    }
                    if k > 0 {
                        let t = Instant::now();
                        nd = d.draft(eng, next, pos_now, k, &mut drafts)?;
                        if ssamp_on && nd > 0 {
                            // draw draft i from q_i (the head's top-16 under the request's transforms); the block's
                            // rows do not condition on earlier drafts, so q_i is fixed for the round
                            let cands = d.draft_cands(eng, nd)?;
                            let mut hist = if pen_on { recent.clone() } else { Vec::new() };
                            if pen_on {
                                hist.push(next);
                            }
                            for i in 0..nd.min(cands.len()) {
                                let h: &[i32] = if pen_on { &hist } else { &recent };
                                let q = draft_q(&cands[i], &sp, h, q_scale);
                                if q.is_empty() {
                                    qs.push(talus_spec::ssamp::SparseDist::one_hot(drafts[i]));
                                } else {
                                    drafts[i] = q.draw(&mut rng);
                                    qs.push(q);
                                }
                                if pen_on {
                                    hist.push(drafts[i]);
                                }
                            }
                        }
                        cost.record_df(ms_since(t));
                        df_rounds += 1;
                        df_last = true;
                    }
                }
            }

            if nd == 0 {
                emit(&format!("T {next}"));
                out_log.push(next);
                gen += 1;
                recent.push(next);
                if !ignore_eos && eos_ids.contains(&next) {
                    finish = "stop";
                    break;
                }
                if input.stop.load(Ordering::SeqCst) {
                    finish = "cancel";
                    break;
                }
                if gen >= max_new {
                    break;
                }
                let pos = cached.len() as i32;
                let t = Instant::now();
                if fast {
                    next = eng.decode(next, pos, None)?;
                } else {
                    eng.decode(next, pos, Some(&mut logits))?;
                    next = sample(&mut logits, &sp, &recent, &mut rng);
                }
                cost.record(1, ms_since(t));
                if let Some(d) = df.as_deref_mut() {
                    d.inject(eng, 0, pos as usize, 1)?;
                }
                cached.push(*recent.last().unwrap());
                continue;
            }

            // Speculative verify [next, drafts[0..nd]]
            let nv = nd + 1;
            let mut toks = [0i32; K_MAX_VERIFY];
            let mut am = [0i32; K_MAX_VERIFY];
            toks[0] = next;
            toks[1..nv].copy_from_slice(&drafts[..nd]);
            let pos = cached.len() as i32;
            let t = Instant::now();
            if !fast {
                vlog.resize(nv * v_len, 0.0);
                eng.verify(&toks[..nv], pos, &mut am[..nv], Some(&mut vlog))?;
            } else {
                eng.verify(&toks[..nv], pos, &mut am[..nv], None)?;
            }
            cost.record(nv, ms_since(t));
            if let Some(d) = df.as_deref_mut() {
                // every verified row's context (rejected rows are overwritten by the next round)
                d.inject(eng, 0, pos as usize, nv)?;
            }
            drafted += nd as i64;

            let mut acc_here = 0usize;
            for i in 0..nv {
                if i > 0 {
                    acc_here += 1;
                }
                emit(&format!("T {}", toks[i]));
                out_log.push(toks[i]);
                gen += 1;
                recent.push(toks[i]);
                if !ignore_eos && eos_ids.contains(&toks[i]) {
                    finish = "stop";
                    ended = true;
                    break;
                }
                if input.stop.load(Ordering::SeqCst) {
                    finish = "cancel";
                    ended = true;
                    break;
                }
                if gen >= max_new {
                    ended = true;
                    break;
                }
                cached.push(toks[i]);
                if i > 0 {
                    dr.push(toks[i]);
                    spec_svc.ingest_token(toks[i]);
                    if let Some(c) = lcopy.as_mut() {
                        c.push(toks[i]);
                    }
                }
                // `None` = draft i + 1 kept; `Some(t)` = the row emits t and the round ends
                let ends: Option<i32> = if !fast && i + 1 < nv && i < qs.len() {
                    let row = &mut vlog[i * v_len..(i + 1) * v_len];
                    let p = sample_dist(row, None, &sp, &recent);
                    match talus_spec::ssamp::judge(&p, &qs[i], toks[i + 1], &mut rng) {
                        talus_spec::ssamp::Verdict::Accept => None,
                        talus_spec::ssamp::Verdict::Reject(y) => Some(y),
                    }
                } else {
                    let choice = if fast {
                        am[i]
                    } else {
                        let row = &mut vlog[i * v_len..(i + 1) * v_len];
                        sample(row, &sp, &recent, &mut rng)
                    };
                    if i + 1 < nv && choice == toks[i + 1] {
                        None
                    } else {
                        Some(choice)
                    }
                };
                match ends {
                    None => {
                        accepted_total += 1;
                        continue;
                    }
                    Some(c) => {
                        next = c;
                        break;
                    }
                }
            }
            qs.clear();
            spec_svc.record_verify(nd, acc_here);
            if let (Some(cm), Some(c)) = (copied, lcopy.as_mut()) {
                c.observe(&cm, nd, acc_here, false);
            }
            if spec_decay {
                // the DFlash lengths this round did not draft (all of them after a prompt-lookup round) and the
                // prompt-lookup bucket it did not use relax toward their priors
                let from = if df_last { nd } else { 0 };
                for p in p_df_all[usize::from(!fast)].iter_mut().skip(from) {
                    *p += 0.02 * (p_df0_init - *p);
                }
                for p in p_df_cmp[usize::from(!fast)].iter_mut().skip(from) {
                    *p += 0.02 * (p_df_cmp0 - *p);
                }
                for (b, p) in pl_acc_all[usize::from(!fast)].iter_mut().enumerate() {
                    if b != pl_bucket {
                        *p += 0.02 * (PL_ACC0[b] - *p);
                    }
                }
            }
            if df_last {
                df_acc += acc_here as i64;
                df_last = false;
                for i in 0..nd.min(acc_here + 1) {
                    let hit = if i < acc_here { 1.0 } else { 0.0 };
                    let p = &mut p_df_all[usize::from(!fast)][i];
                    *p = 0.9 * *p + 0.1 * hit;
                    let pc = &mut p_df_cmp[usize::from(!fast)][i];
                    *pc = 0.9 * *pc + 0.1 * hit;
                }
            } else if pl_bucket != usize::MAX {
                for i in 0..nd.min(acc_here + 1) {
                    let hit = if i < acc_here { 1.0 } else { 0.0 };
                    let p = &mut pl_acc_all[usize::from(!fast)][pl_bucket];
                    *p = 0.9 * *p + 0.1 * hit;
                }
            } else {
                for i in 0..nd {
                    let hit = if i < acc_here { 1.0 } else { 0.0 };
                    acc_est = 0.9 * acc_est + 0.1 * hit;
                }
            }
        }

        let decode_ms = ms_since(td);
        if let Some(c) = lcopy.as_mut() {
            let st = c.take_stats();
            if st.windows > 0 {
                eprintln!(
                    "talus copy: {} windows, {} of {} copied drafts accepted",
                    st.windows, st.accepted, st.proposed
                );
            }
        }
        eprintln!(
            "gen: {} tokens in {:.0} ms at depth {}, {} drafted, {} accepted (dflash {} rounds, {} accepted, {:.1} ms; \
             cost ms w1 {:.1} w2 {:.1} w5 {:.1} w9 {:.1} w17 {:.1} w32 {:.1} w64 {:.1}; pl acc {:.2}/{:.2}; \
             p_df {:.2} {:.2} {:.2} {:.2} {:.2} {:.2})",
            gen,
            decode_ms,
            cached.len(),
            drafted,
            accepted_total,
            df_rounds,
            df_acc,
            cost.df_ms,
            cost.ms[1],
            cost.ms[2],
            cost.ms[5],
            cost.ms[9],
            cost.ms[17],
            cost.ms[32],
            cost.ms[K_MAX_VERIFY],
            pl_acc_all[usize::from(!fast)][0],
            pl_acc_all[usize::from(!fast)][1],
            p_df_all[usize::from(!fast)][0],
            p_df_all[usize::from(!fast)][1],
            p_df_all[usize::from(!fast)][2],
            p_df_all[usize::from(!fast)][3],
            p_df_all[usize::from(!fast)][4],
            p_df_all[usize::from(!fast)][5]
        );
        emit(&format!(
            "DONE {} {} {:.1} {:.1} {} {} {} {} 0 0 0 0 0 {}",
            gen,
            ids.len(),
            prompt_ms,
            decode_ms,
            finish,
            accepted_total,
            drafted,
            lcp,
            n_read
        ));
        // Diagnostic: TALUS_LAGUNA_GENLOG=FILE appends the GEN line, the reused prefix and the emitted tokens
        if let Some(path) = std::env::var_os("TALUS_LAGUNA_GENLOG") {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let toks: Vec<String> = out_log.iter().map(|t| t.to_string()).collect();
                let _ = writeln!(f, "{}\nLCP {lcp}\nOUT {}", trimmed, toks.join(","));
            }
        }
    }

    Ok(())
}

trait VecSizeTokens {
    fn size_tokens(&self) -> usize;
}

impl VecSizeTokens for Vec<i32> {
    #[inline]
    fn size_tokens(&self) -> usize {
        self.len()
    }
}

/// Configuration for Laguna CLI batch/ubatch/generate/force-decode/verify-test modes
/// (`src/laguna/laguna_main.cpp`).
#[derive(Debug, Clone)]
pub struct LagunaCliConfig {
    pub tokens: Vec<i32>,
    pub top: usize,
    pub generate: usize,
    pub force_decode: usize,
    pub verify_test: usize,
    pub dump_layers: bool,
    pub dump_logits: Option<std::path::PathBuf>,
    pub dump_full_logits: Option<std::path::PathBuf>,
    pub dump_decode_only: bool,
    pub use_graph: bool,
}

/// Run Laguna CLI batch/ubatch/generate/force-decode/verify-test modes matching
/// `src/laguna/laguna_main.cpp:139-221`.
pub fn run_cli_ubatch(eng: &mut Engine, cfg: &LagunaCliConfig) -> Result<()> {
    let n_all = cfg.tokens.len();
    if n_all == 0 {
        return Err(CoreError::InvalidArgument {
            reason: "no tokens".to_string(),
        });
    }
    if cfg.force_decode >= n_all {
        return Err(CoreError::InvalidArgument {
            reason: "--force-decode must leave at least one prompt token".to_string(),
        });
    }
    if n_all + cfg.generate + cfg.verify_test > eng.max_ctx() {
        return Err(CoreError::InvalidArgument {
            reason: "--ctx too small for prompt + --generate".to_string(),
        });
    }
    for &t in &cfg.tokens {
        if t < 0 || (t as usize) >= K_VOCAB {
            return Err(CoreError::InvalidArgument {
                reason: format!("token id {t} out of range"),
            });
        }
    }

    let mut lf: Option<File> = match &cfg.dump_logits {
        Some(p) => Some(File::create(p).map_err(|e| CoreError::IoError {
            message: format!("cannot write {}: {e}", p.display()),
        })?),
        None => None,
    };
    let mut full_lf: Option<File> = match &cfg.dump_full_logits {
        Some(p) if cfg.dump_logits.is_none() => {
            Some(File::create(p).map_err(|e| CoreError::IoError {
                message: format!("cannot write {}: {e}", p.display()),
            })?)
        }
        _ => None,
    };

    let n_tok = n_all - cfg.force_decode;
    let mut last = vec![0.0_f32; K_VOCAB];
    let t1 = Instant::now();

    let mut dump_cb = |name: &str, h: &[f32], rows: usize, cols: usize| {
        if let Some(il_str) = name.strip_prefix("l_out-") {
            if let Ok(il) = il_str.parse::<usize>() {
                if let Some(out_path) = std::env::var_os("DUMP_RESID_OUT") {
                    if !out_path.is_empty() {
                        let max_layers = std::env::var("DUMP_RESID_LAYERS")
                            .ok()
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(crate::laguna::K_LAYERS);
                        if il < max_layers {
                            let bytes = unsafe {
                                std::slice::from_raw_parts(h.as_ptr().cast::<u8>(), h.len() * 4)
                            };
                            if let Ok(mut f) = std::fs::OpenOptions::new()
                                .create(true)
                                .write(true)
                                .truncate(il == 0)
                                .append(il > 0)
                                .open(out_path)
                            {
                                let _ = f.write_all(bytes);
                            }
                        }
                    }
                }
                for t in 0..rows {
                    let row = &h[t * cols..(t + 1) * cols];
                    let mut sum = 0.0_f64;
                    let mut ss = 0.0_f64;
                    for &v in row {
                        sum += v as f64;
                        ss += (v as f64) * (v as f64);
                    }
                    let rms = (ss / (cols as f64)).sqrt();
                    println!(
                        "l_out-{} t{} sum {:.6} rms {:.6} [{:.5} {:.5} {:.5} ... {:.5} {:.5} {:.5}]",
                        il,
                        t,
                        sum,
                        rms,
                        row[0],
                        row[1],
                        row[2],
                        row[cols - 3],
                        row[cols - 2],
                        row[cols - 1]
                    );
                }
            }
        }
    };

    eng.prefill_with_file(
        &cfg.tokens[..n_tok],
        0,
        Some(&mut last),
        if cfg.dump_decode_only {
            None
        } else {
            lf.as_mut()
        },
        if cfg.dump_layers {
            Some(&mut dump_cb)
        } else {
            None
        },
    )?;

    if !cfg.dump_decode_only {
        if let Some(f) = full_lf.as_mut() {
            let bytes =
                unsafe { std::slice::from_raw_parts(last.as_ptr() as *const u8, K_VOCAB * 4) };
            f.write_all(bytes).map_err(|e| CoreError::IoError {
                message: format!("write full logits: {e}"),
            })?;
        }
    }

    let run_s = t1.elapsed().as_secs_f64();
    eprintln!(
        "{} tokens (ubatch {}), {:.2} s, {:.1} tok/s",
        n_tok,
        eng.ubatch(),
        run_s,
        if run_s > 0.0 {
            (n_tok as f64) / run_s
        } else {
            0.0
        }
    );

    if cfg.top > 0 {
        let k = cfg.top.min(K_VOCAB);
        let mut id: Vec<usize> = (0..K_VOCAB).collect();
        id.sort_by(|&a, &b| {
            last[b]
                .partial_cmp(&last[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        print!("pos {} top:", n_tok - 1);
        for &idx in &id[..k] {
            print!(" {}({:.3})", idx, last[idx]);
        }
        println!();
    }

    // LAGUNA_FORCE_VERIFY=w: the forced tokens through teacher-forced verify windows of w tokens (the rows a
    // w-token verify emits, e.g. the opt-in GEMM verify's) instead of one decode step each
    let force_w: usize = std::env::var("LAGUNA_FORCE_VERIFY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
        .min(K_MAX_VERIFY);
    if cfg.force_decode > 0 && force_w > 0 {
        let tf = Instant::now();
        let mut pos = n_tok;
        while pos < n_all {
            let w = force_w.min(n_all - pos);
            let mut am = vec![0i32; w];
            let mut rows = vec![0.0_f32; w * K_VOCAB];
            // window = the tokens at pos .. pos + w: row i predicts pos + i + 1 given the forced tokens up to pos + i
            eng.verify(
                &cfg.tokens[pos..pos + w],
                pos as i32,
                &mut am,
                Some(&mut rows),
            )?;
            if rows.iter().any(|v| !v.is_finite()) {
                return Err(CoreError::KernelError {
                    code: -1,
                    message: format!("non-finite verify logits at {pos}"),
                });
            }
            if let Some(f) = lf.as_mut().or(full_lf.as_mut()) {
                let bytes = unsafe {
                    std::slice::from_raw_parts(rows.as_ptr() as *const u8, w * K_VOCAB * 4)
                };
                f.write_all(bytes).map_err(|e| CoreError::IoError {
                    message: format!("write forced verify logits: {e}"),
                })?;
            }
            pos += w;
        }
        let fs = tf.elapsed().as_secs_f64();
        eprintln!(
            "forced verify (windows of {force_w}): {} tokens, {:.2} s",
            cfg.force_decode, fs
        );
    } else if cfg.force_decode > 0 {
        let mut row = vec![0.0_f32; K_VOCAB];
        let tf = Instant::now();
        for pos in n_tok..n_all {
            eng.decode(cfg.tokens[pos], pos as i32, Some(&mut row))?;
            for &v in &row {
                if !v.is_finite() {
                    return Err(CoreError::KernelError {
                        code: -1,
                        message: format!("non-finite decode logits at {pos}"),
                    });
                }
            }
            if let Some(f) = lf.as_mut().or(full_lf.as_mut()) {
                let bytes =
                    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, K_VOCAB * 4) };
                f.write_all(bytes).map_err(|e| CoreError::IoError {
                    message: format!("write forced decode logits: {e}"),
                })?;
            }
        }
        let fs = tf.elapsed().as_secs_f64();
        eprintln!(
            "forced decode: {} tokens, {:.2} s, {:.2} tok/s",
            cfg.force_decode,
            fs,
            if fs > 0.0 {
                (cfg.force_decode as f64) / fs
            } else {
                0.0
            }
        );
    }

    if cfg.verify_test > 0 {
        let first_tok = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        // Keep decode's logits rows too, to compare verify rows with them bit for bit
        let mut seq = vec![first_tok];
        let mut dec_rows: Vec<f32> = Vec::with_capacity(cfg.verify_test * K_VOCAB);
        let mut row = vec![0.0_f32; K_VOCAB];
        for i in 1..=cfg.verify_test {
            let next = eng.decode(*seq.last().unwrap(), (n_all + i - 1) as i32, Some(&mut row))?;
            dec_rows.extend_from_slice(&row);
            seq.push(next);
        }
        // Reference rows from sequential verify(1) calls (the verify kernels at one column)
        let mut v1_rows: Vec<f32> = Vec::with_capacity(cfg.verify_test * K_VOCAB);
        for i in 0..cfg.verify_test {
            let mut am1 = [0i32; 1];
            let mut r1 = vec![0.0_f32; K_VOCAB];
            eng.verify(&seq[i..i + 1], (n_all + i) as i32, &mut am1, Some(&mut r1))?;
            v1_rows.extend_from_slice(&r1);
        }
        // LAGUNA_VT_N=a,b,..: only these widths (profiling)
        let only: Vec<usize> = std::env::var("LAGUNA_VT_N")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
            .unwrap_or_default();
        // LAGUNA_KV_TEST=1: the K / V bytes stored for the same tokens written by sequential verify(1), by verify(3)
        // windows, and by a mix of widths with rejected drafts, must be identical
        if std::env::var("LAGUNA_KV_TEST")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let t_n = cfg.verify_test;
            let rec = eng.kv_record_bytes();
            let mut dumps: Vec<(&str, Vec<u8>, Vec<u8>)> = Vec::new();
            for mode in ["v1", "v3", "mix"] {
                let mut i = 0usize;
                let mut step = 0usize;
                const W: [usize; 8] = [32, 3, 1, 5, 17, 2, 64, 9];
                while i < t_n {
                    let (w, acc) = match mode {
                        "v1" => (1, 1),
                        "v3" => (3.min(t_n - i), 3.min(t_n - i)),
                        _ => {
                            let w = W[step % W.len()].min(t_n - i).max(1);
                            // accept about two thirds, the rest junk (rejected drafts)
                            (w, (w * 2 / 3).max(1))
                        }
                    };
                    let mut toks: Vec<i32> = seq[i..i + w].to_vec();
                    for (j, t) in toks.iter_mut().enumerate().skip(acc) {
                        *t = ((seq[i + j] as i64 * 7919 + 104729 * (i + j) as i64) % 90000 + 1000)
                            as i32;
                    }
                    let mut am = vec![0i32; w];
                    eng.verify(&toks, (n_all + i) as i32, &mut am, None)?;
                    i += acc;
                    step += 1;
                }
                let (kv, bnd) = eng.kv_dump(n_all, n_all + t_n)?;
                println!(
                    "kv-test {mode}: {} verify calls, {} KV bytes, {} bound bytes",
                    step,
                    kv.len(),
                    bnd.len()
                );
                dumps.push((mode, kv, bnd));
            }
            let nl = dumps[0].1.len() / (rec * t_n).max(1);
            for m in 1..dumps.len() {
                let (a, b) = (&dumps[0].1, &dumps[m].1);
                let nd = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                let first = a.iter().zip(b.iter()).position(|(x, y)| x != y);
                let where_ = first.map(|o| {
                    let (l, r) = (o / (rec * t_n), o % (rec * t_n));
                    format!("layer {l} pos +{} byte {}", r / rec, r % rec)
                });
                let nb = dumps[0]
                    .2
                    .iter()
                    .zip(dumps[m].2.iter())
                    .filter(|(x, y)| x != y)
                    .count();
                println!(
                    "kv-test v1 vs {}: {nd} KV bytes differ{} ({} layers); block bounds: {nb} bytes differ",
                    dumps[m].0,
                    where_.map(|w| format!(", first at {w}")).unwrap_or_default(),
                    nl
                );
            }
        }
        // LAGUNA_KV_TEST=1 (cont.): a request resuming at `from` clears the cells past it; the cells before it (the
        // reused prefix) must be untouched whatever the high-water mark the last verify window left
        if std::env::var("LAGUNA_KV_TEST")
            .map(|v| v == "1")
            .unwrap_or(false)
            && cfg.verify_test >= 40
        {
            let mut bad = 0usize;
            for (from_off, junk) in [(37usize, 3usize), (37, 9), (35, 1), (32, 5), (33, 30)] {
                let from = n_all + from_off;
                // the prefix cells: rewrite them in order, then a rejected window past `from`
                for i in 0..from_off {
                    let mut am1 = [0i32; 1];
                    eng.verify(&seq[i..i + 1], (n_all + i) as i32, &mut am1, None)?;
                }
                let before = eng.kv_dump(from - 32, from)?.0;
                let toks: Vec<i32> = (0..junk).map(|j| 1000 + 37 * j as i32).collect();
                let mut am = vec![0i32; junk];
                eng.verify(&toks, from as i32, &mut am, None)?;
                eng.clear_kv_from(from, true)?;
                let after = eng.kv_dump(from - 32, from)?.0;
                let nd = before
                    .iter()
                    .zip(after.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                bad += nd;
                println!("kv-test clear from +{from_off} after a {junk}-token window: {nd} prefix bytes changed");
            }
            println!(
                "kv-test clear: {}",
                if bad == 0 {
                    "prefix intact"
                } else {
                    "PREFIX CHANGED"
                }
            );
        }
        // LAGUNA_VT_OFF=o: the windows start o tokens later (block-unaligned / boundary-crossing windows)
        let vo: usize = std::env::var("LAGUNA_VT_OFF")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
            .min(cfg.verify_test - 1);
        let dec_rows = dec_rows[vo * K_VOCAB..].to_vec();
        let v1_rows = v1_rows[vo * K_VOCAB..].to_vec();
        let seq = seq[vo..].to_vec();
        let n_all = n_all + vo;
        let vt = cfg.verify_test - vo;
        for n in 1..=vt.min(K_MAX_VERIFY) {
            if !only.is_empty() && !only.contains(&n) {
                continue;
            }
            let mut am = vec![0i32; n];
            let mut vl = vec![0.0_f32; n * K_VOCAB];
            eng.verify(&seq[..n], n_all as i32, &mut am, Some(&mut vl))?;
            let same = (0..n).all(|i| am[i] == seq[i + 1]);
            let mut nbits = 0usize;
            let mut maxd = 0.0_f32;
            for (a, b) in vl.iter().zip(dec_rows[..n * K_VOCAB].iter()) {
                if a.to_bits() != b.to_bits() {
                    nbits += 1;
                    maxd = maxd.max((a - b).abs());
                }
            }
            let mut nb1 = 0usize;
            let mut maxd1 = 0.0_f32;
            for (a, b) in vl.iter().zip(v1_rows[..n * K_VOCAB].iter()) {
                if a.to_bits() != b.to_bits() {
                    nb1 += 1;
                    maxd1 = maxd1.max((a - b).abs());
                }
            }
            println!(
                "verify n={n} logits vs decode: {nbits} values differ (max |d| {maxd:.3e}); vs verify(1): {nb1} differ (max |d| {maxd1:.3e})"
            );
            let tv = Instant::now();
            let reps = 10usize;
            for _ in 0..reps {
                eng.verify(&seq[..n], n_all as i32, &mut am, None)?;
            }
            let ms = tv.elapsed().as_secs_f64() * 1000.0 / (reps as f64);
            println!(
                "verify n={}: {}, {:.1} ms ({:.1} ms/token)",
                n,
                if same {
                    "matches sequential decode"
                } else {
                    "DIFFERS"
                },
                ms,
                ms / (n as f64)
            );
            if !same {
                print!("   seq:");
                for i in 0..n {
                    print!(" {}", seq[i + 1]);
                }
                print!("\n   ver:");
                for &v in &am[..n] {
                    print!(" {v}");
                }
                println!();
            }
        }
        // LAGUNA_VT_JUNK=1: leave a rejected window behind (verify(n) of junk drafts after seq[0]), then redo the
        // sequential verify(1) rows: they must equal the reference rows bitwise (the junk cells / block bounds past
        // the accepted position never change a later row)
        if std::env::var("LAGUNA_VT_JUNK")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            for n in 2..=vt.min(K_MAX_VERIFY) {
                if !only.is_empty() && !only.contains(&n) {
                    continue;
                }
                let mut junk: Vec<i32> = seq[..n].to_vec();
                for (i, t) in junk.iter_mut().enumerate().skip(1) {
                    *t = ((seq[i] as i64 * 7919 + 104729 * i as i64) % 90000 + 1000) as i32;
                }
                let mut am = vec![0i32; n];
                eng.verify(&junk, n_all as i32, &mut am, None)?;
                let (mut nb, mut first) = (0usize, usize::MAX);
                for i in 0..vt {
                    let mut am1 = [0i32; 1];
                    let mut r1 = vec![0.0_f32; K_VOCAB];
                    eng.verify(&seq[i..i + 1], (n_all + i) as i32, &mut am1, Some(&mut r1))?;
                    let d = r1
                        .iter()
                        .zip(&v1_rows[i * K_VOCAB..(i + 1) * K_VOCAB])
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    if d > 0 && first == usize::MAX {
                        first = i;
                    }
                    nb += d;
                }
                println!(
                    "junk n={n}: verify(1) rows after it: {nb} values differ (first row {})",
                    if first == usize::MAX {
                        -1
                    } else {
                        first as i64
                    }
                );
            }
        }
        let td = Instant::now();
        for _ in 0..10 {
            eng.decode(seq[0], n_all as i32, None)?;
        }
        println!(
            "decode (1 token): {:.1} ms",
            td.elapsed().as_secs_f64() * 1000.0 / 10.0
        );
    }

    if cfg.generate > 0 {
        let mut tok = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        let mut out = vec![tok];
        let t2 = Instant::now();
        for i in 1..cfg.generate {
            tok = eng.decode(tok, (n_all + i - 1) as i32, None)?;
            out.push(tok);
        }
        let gen_s = t2.elapsed().as_secs_f64();
        print!("generated:");
        for t in &out {
            print!(" {t}");
        }
        println!();
        eprintln!(
            "decode: {} tokens in {:.2} s, {:.2} tok/s ({})",
            cfg.generate - 1,
            gen_s,
            if gen_s > 0.0 {
                ((cfg.generate - 1) as f64) / gen_s
            } else {
                0.0
            },
            if cfg.use_graph { "graph" } else { "eager" }
        );
    }

    let _ = K_EMBD;
    Ok(())
}
