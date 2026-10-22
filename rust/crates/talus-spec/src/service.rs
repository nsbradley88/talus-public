//! Unified Speculative Decoding Engine Service (`WP-5`: `SpecEngineService`, `ModelSpecDecl`,
//! `DfCtx`, `CrossoverTable`, `DrafterPick`, `SuffixDrafter`, `DraftPick`, `DraftPolicy`,
//! `parse_logit_bias`).
//!
//! Consolidates the model-independent speculative decoding state machines across Qwen3.8-27B,
//! Qwen3.8-Flash-Next (`qwen4exp`), Laguna-S-2.1, MiMo-V2.6-Flash, and GLM-5.3-Flash into
//! `talus-spec`:
//! - [`DfCtx`]: DFlash KV ring / sliding-window coverage tracker (`[lo, hi)`, `at`, `rewind`,
//!   `inject`, `valid`).
//! - [`CrossoverTable`] & [`DrafterPick`]: context-depth crossover controller (`mtp`, `dflash`,
//!   `auto`, `lookup`, `ngram`) with per-model greedy/sampled crossover depths, MTP row margin,
//!   and probe/hold hysteresis.
//! - [`SuffixDrafter`]: 8-way set-associative power-of-2 hash table suffix drafter (`WAYS = 8`),
//!   implementing [`HostDrafter`].
//! - [`DraftPick`] & [`DraftPolicy`]: multi-bucket (`match_len < 6`, `6..12`, `12..24`, `>= 24`)
//!   lookup-vs-device cost/acceptance controller and hybrid MTP+lookup chain planner.
//! - [`ModelSpecDecl`] & [`SpecEngineService`]: declarative per-model speculation specification
//!   and runtime engine service owning `DraftLen`, `DfCtx`, `DrafterPick`, `SuffixDrafter`,
//!   `DraftPolicy`, `PromptLookupDrafter`, `NgramMod`, `AdaptivePausePolicy`, and `Mt19937_64`.

use std::fs::File;
use std::io::Write as _;

use talus_core_api::HostDrafter;

use crate::adraft::DraftLen;
use crate::dflash::DFlashModelProfile;
use crate::draft_vocab::DraftVocab;
use crate::ngram::NgramMod;
use crate::pause::AdaptivePausePolicy;
use crate::prompt_lookup::PromptLookupDrafter;
use crate::sampling::{decide_greedy, decide_sampled, sample, Mt19937_64, SampCfg};

// ============================================================================
// 1. DfCtx — DFlash KV ring / window coverage tracker
// ============================================================================

/// Tracks which sequence positions `[lo, hi)` are currently resident as one contiguous run
/// in the DFlash drafter's KV ring buffer (`talus::q36::DfCtx` in `q36_depth_policy.hpp:30-66`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfCtx {
    window: i32,
    ring: i32,
    block: i32,
    lo: i64,
    hi: i64,
    last: i64,
}

impl Default for DfCtx {
    fn default() -> Self {
        Self {
            window: 2048,
            ring: 4096,
            block: 8,
            lo: 0,
            hi: 0,
            last: 0,
        }
    }
}

impl DfCtx {
    /// Initialize DFlash KV ring geometry (`window`, `ring` capacity, `block` size) and clear bounds.
    pub fn init(&mut self, window: i32, ring: i32, block: i32) {
        self.window = window;
        self.ring = ring;
        self.block = block;
        self.clear(0);
        self.last = 0;
    }

    #[inline]
    fn clear(&mut self, p: i64) {
        self.lo = p;
        self.hi = p;
    }

    /// Notify that the engine is at sequence position `p` (a step backward triggers [`Self::rewind`]).
    pub fn at(&mut self, p: i64) {
        if p < self.last {
            self.rewind(p);
        }
        self.last = p;
    }

    /// Invalidate all sequence positions `>= p` (on KV rollback or session restore).
    pub fn rewind(&mut self, p: i64) {
        if self.hi > p {
            self.hi = p;
        }
        if self.lo > self.hi {
            self.clear(p);
        }
        if self.last > p {
            self.last = p;
        }
    }

    /// Record that sequence positions `p0 .. p0 + n` have been written into the DFlash KV ring
    /// (`wrote` is the number of ring cells written starting at `p0`, accounting for draft padding).
    pub fn inject(&mut self, p0: i64, n: i32, wrote: i32) {
        if n <= 0 {
            return;
        }
        let n64 = n as i64;
        if p0 > self.hi || p0 + n64 < self.lo {
            self.lo = p0;
            self.hi = p0 + n64;
        } else {
            self.lo = self.lo.min(p0);
            self.hi = self.hi.max(p0 + n64);
        }
        let w64 = wrote.max(n) as i64;
        let top = p0 + w64 + (self.block as i64);
        if self.lo < top - (self.ring as i64) {
            self.lo = top - (self.ring as i64);
        }
        if self.lo > self.hi {
            let h = self.hi;
            self.clear(h);
        }
    }

    /// Returns `true` if the entire attention window ending at anchor `p` (`max(0, p - window + 1) ..= p`)
    /// is valid and resident in the DFlash KV ring.
    #[inline]
    pub fn valid(&self, p: i64) -> bool {
        let need = (p - (self.window as i64) + 1).max(0);
        self.lo <= need && self.hi >= p
    }

    /// Lower bound (inclusive) of the valid sequence range in the ring.
    #[inline]
    pub fn lo(&self) -> i64 {
        self.lo
    }

    /// Upper bound (exclusive) of the valid sequence range in the ring.
    #[inline]
    pub fn hi(&self) -> i64 {
        self.hi
    }

    /// Configured attention window length.
    #[inline]
    pub fn window(&self) -> i32 {
        self.window
    }

    /// Configured KV ring capacity.
    #[inline]
    pub fn ring(&self) -> i32 {
        self.ring
    }

    /// Explicitly set `[lo, hi)` bounds (used when restoring a saved session).
    #[inline]
    pub fn set_bounds(&mut self, lo: i64, hi: i64) {
        self.lo = lo;
        self.hi = hi;
        if self.last > hi {
            self.last = hi;
        }
    }
}

// ============================================================================
// 2. CrossoverTable & DrafterPick — Context-depth drafter selection
// ============================================================================

/// Per-model context-depth crossover parameters for [`DrafterPick`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossoverTable {
    /// Greedy context-depth threshold below which DFlash is preferred (`n[0]`).
    pub greedy_depth: i64,
    /// Sampled context-depth threshold below which DFlash is preferred (`n[1]`).
    pub sampled_depth: i64,
    /// Margin before `n` where DFlash rounds also compute MTP hidden rows (`mtp_rows_margin`).
    pub mtp_rows_margin: i64,
    /// Minimum accepted+bonus tokens/round before hysteresis holds MTP (`tmin`).
    pub tmin: f64,
    /// Number of probe rounds before evaluating `tmin` (`probe`).
    pub probe: i32,
    /// Number of rounds to hold on fallback drafter when `probe_tok / probe_n < tmin` (`hold`).
    pub hold: i32,
}

impl Default for CrossoverTable {
    fn default() -> Self {
        Self::qwen35_27b()
    }
}

impl CrossoverTable {
    /// Qwen3.5-27B (`gfx1151`): DFlash2 below 16K greedy / 32K sampled, MTP above, 2K margin.
    pub const fn qwen35_27b() -> Self {
        Self {
            greedy_depth: 16384,
            sampled_depth: 32768,
            mtp_rows_margin: 2048,
            tmin: 0.0,
            probe: 8,
            hold: 64,
        }
    }

    /// Qwen3.8-Flash-Next (`qwen4exp`): MTP + SuffixDrafter across full context.
    pub const fn qwen35_35b_a3b() -> Self {
        Self {
            greedy_depth: 0,
            sampled_depth: 0,
            mtp_rows_margin: 1 << 30,
            tmin: 0.0,
            probe: 8,
            hold: 64,
        }
    }

    /// Laguna-S-2.1: DFlash (`window = 1024`) + Suffix/PromptLookup fallback across 32K context.
    pub const fn laguna_s_2_1() -> Self {
        Self {
            greedy_depth: 32768,
            sampled_depth: 32768,
            mtp_rows_margin: 1 << 30,
            tmin: 1.2,
            probe: 8,
            hold: 16,
        }
    }

    /// MiMo-V2.6-Flash: DFlash (`window = 1024`, `B = 5`) + Suffix/PromptLookup across 32K context.
    pub const fn mimo_v2_flash() -> Self {
        Self {
            greedy_depth: 32768,
            sampled_depth: 32768,
            mtp_rows_margin: 1 << 30,
            tmin: 1.2,
            probe: 8,
            hold: 16,
        }
    }

    /// GLM-4.7-Flash: MTP + Suffix/Ngram across 32K context.
    pub const fn glm_4_7_flash() -> Self {
        Self {
            greedy_depth: 0,
            sampled_depth: 0,
            mtp_rows_margin: 1 << 30,
            tmin: 0.0,
            probe: 8,
            hold: 64,
        }
    }
}

/// Context-depth drafter selector (`talus::q36::DrafterPick` in `q36_depth_policy.{hpp,cpp}`).
#[derive(Debug, Clone)]
pub struct DrafterPick {
    mode: i32,
    n: [i64; 2],
    mtp_rows_margin: i64,
    tmin: f64,
    probe: i32,
    hold: i32,
    hold_left: i32,
    probe_n: i32,
    probe_tok: f64,
    pub n_dfl: i64,
    pub n_mtp: i64,
    pub n_fallback: i64,
    trace_path: Option<String>,
}

impl Default for DrafterPick {
    fn default() -> Self {
        Self {
            mode: Self::K_DFLASH,
            n: [16384, 16384],
            mtp_rows_margin: 1 << 30,
            tmin: 0.0,
            probe: 8,
            hold: 64,
            hold_left: 0,
            probe_n: 0,
            probe_tok: 0.0,
            n_dfl: 0,
            n_mtp: 0,
            n_fallback: 0,
            trace_path: None,
        }
    }
}

impl DrafterPick {
    pub const K_MTP: i32 = 0;
    pub const K_DFLASH: i32 = 1;
    pub const K_AUTO: i32 = 2;
    pub const K_LOOKUP: i32 = 3;
    pub const K_NGRAM: i32 = 4;

    /// Initialize drafter selection policy for GPU architecture `gcn`, whether `both` MTP and
    /// DFlash are loaded, and CLI/env override `cli`.
    pub fn init(&mut self, gcn: &str, both: bool, cli: &str) {
        self.init_with_table(CrossoverTable::qwen35_27b(), gcn, both, cli);
    }

    /// Initialize drafter selection policy using a model-specific [`CrossoverTable`].
    pub fn init_with_table(&mut self, table: CrossoverTable, gcn: &str, both: bool, cli: &str) {
        let env_i64 = |k: &str, d: i64| -> i64 {
            std::env::var(k)
                .ok()
                .and_then(|v| v.trim().parse::<i64>().ok())
                .unwrap_or(d)
        };
        if gcn.contains("gfx1151") {
            self.mode = Self::K_AUTO;
            self.n = [table.greedy_depth, table.sampled_depth];
            self.mtp_rows_margin = table.mtp_rows_margin;
            self.tmin = table.tmin;
            self.probe = table.probe;
            self.hold = table.hold;
        } else {
            self.mode = Self::K_DFLASH;
        }
        let m = std::env::var("STRATA_DRAFTER")
            .ok()
            .unwrap_or_else(|| cli.to_string());
        if m == "mtp" {
            self.mode = Self::K_MTP;
        } else if m == "dflash" {
            self.mode = Self::K_DFLASH;
        } else if m == "auto" {
            self.mode = Self::K_AUTO;
        } else if m == "lookup" {
            self.mode = Self::K_LOOKUP;
        } else if m == "ngram" {
            self.mode = Self::K_NGRAM;
        } else if !m.is_empty() {
            eprintln!("talus-drafter: unknown mode '{m}' (mtp, dflash, auto)");
        }
        if !both && self.mode != Self::K_LOOKUP && self.mode != Self::K_NGRAM {
            self.mode = Self::K_DFLASH;
        }
        self.n[0] = env_i64("STRATA_DRAFTER_N", self.n[0]);
        self.n[1] = env_i64("STRATA_DRAFTER_N_S", self.n[1]);
        self.mtp_rows_margin = env_i64("STRATA_DRAFTER_MTPROWS", self.mtp_rows_margin);
        if let Ok(v) = std::env::var("STRATA_DRAFTER_TMIN") {
            if let Ok(f) = v.trim().parse::<f64>() {
                self.tmin = f;
            }
        }
        self.probe = (env_i64("STRATA_DRAFTER_PROBE", self.probe as i64) as i32).max(1);
        self.hold = env_i64("STRATA_DRAFTER_HOLD", self.hold as i64) as i32;
        if let Ok(f) = std::env::var("STRATA_DRAFTER_TRACE") {
            if !f.is_empty() {
                let _ = File::create(&f);
                self.trace_path = Some(f);
            }
        }
    }

    #[inline]
    pub fn mode(&self) -> i32 {
        self.mode
    }

    #[inline]
    pub fn uses_mtp(&self) -> bool {
        self.mode != Self::K_DFLASH
    }

    #[inline]
    pub fn uses_dfl(&self) -> bool {
        self.mode != Self::K_MTP
    }

    #[inline]
    pub fn want_dfl(&self, p: i64, sampled: bool) -> bool {
        if self.mode != Self::K_AUTO {
            return self.mode == Self::K_DFLASH;
        }
        if self.hold_left > 0 {
            return false;
        }
        p < self.n[if sampled { 1 } else { 0 }]
    }

    #[inline]
    pub fn inject_in_mtp(&self, p: i64, sampled: bool) -> bool {
        self.mode == Self::K_AUTO && p < self.n[if sampled { 1 } else { 0 }]
    }

    #[inline]
    pub fn mtp_rows_in_dfl(&self, p: i64, sampled: bool) -> bool {
        self.mode == Self::K_AUTO && p >= self.n[if sampled { 1 } else { 0 }] - self.mtp_rows_margin
    }

    pub fn fallback(&mut self, p: i64) {
        self.n_fallback += 1;
        if let Some(ref path) = self.trace_path {
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path) {
                let _ = writeln!(f, "{p} F");
            }
        }
    }

    pub fn observe(&mut self, dfl: bool, drafted: i32, kept: i32, p: i64, sampled: bool) {
        if dfl {
            self.n_dfl += 1;
        } else {
            self.n_mtp += 1;
        }
        if let Some(ref path) = self.trace_path {
            if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path) {
                let _ = writeln!(
                    f,
                    "{p} {} {drafted} {kept} {}",
                    if dfl { 'D' } else { 'M' },
                    if sampled { 1 } else { 0 }
                );
            }
        }
        if self.mode != Self::K_AUTO || self.tmin <= 0.0 {
            return;
        }
        if !dfl {
            if self.hold_left > 0 {
                self.hold_left -= 1;
            }
            return;
        }
        self.probe_tok += (kept + 1) as f64;
        self.probe_n += 1;
        if self.probe_n >= self.probe {
            if self.probe_tok / (self.probe_n as f64) < self.tmin {
                self.hold_left = self.hold;
            }
            self.probe_n = 0;
            self.probe_tok = 0.0;
        }
    }

    #[inline]
    pub fn request_start(&mut self) {
        self.probe_n = 0;
        self.probe_tok = 0.0;
        self.hold_left = 0;
    }

    pub fn describe(&self) -> String {
        let nm = ["mtp", "dflash", "auto", "lookup", "ngram"];
        if self.mode == Self::K_AUTO {
            format!(
                "drafter auto: DFlash2 below {} (greedy) / {} (sampled) tokens of context, else the MTP; MTP rows in DFlash2 rounds from {} below it; hysteresis {}",
                self.n[0],
                self.n[1],
                self.mtp_rows_margin,
                if self.tmin > 0.0 { "on" } else { "off" }
            )
        } else {
            let idx = (self.mode.clamp(0, 4)) as usize;
            format!("drafter {}", nm[idx])
        }
    }
}

// ============================================================================
// 3. SuffixDrafter — 8-way set-associative suffix hash drafter
// ============================================================================

const WAYS: usize = 8;

#[derive(Debug, Clone, Copy, Default)]
struct Slot {
    key: u64,
    pos: [u32; WAYS],
    n: u8,
}

#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

/// 1:1 port of `strata::spec::SuffixDrafter` (`src/spec/suffix_drafter.cpp`).
///
/// Indexes 3-gram suffixes in an 8-way set-associative power-of-2 hash table and selects
/// the candidate occurrence with the longest backward suffix match (`min_match..=max_match`).
#[derive(Debug, Clone)]
pub struct SuffixDrafter {
    min_match: i32,
    max_match: i32,
    last_match: i32,
    hist: Vec<i32>,
    table: Vec<Slot>,
    mask: usize,
}

impl Default for SuffixDrafter {
    fn default() -> Self {
        Self::new(4, 32, 32768)
    }
}

impl SuffixDrafter {
    pub fn new(min_match: i32, max_match: i32, capacity_tokens: usize) -> Self {
        let min_m = min_match.max(3);
        let max_m = max_match.max(min_match);
        let mut cap = 1usize;
        let target = capacity_tokens.saturating_mul(2).max(16);
        while cap < target {
            cap <<= 1;
        }
        let hist = Vec::with_capacity(capacity_tokens);
        Self {
            min_match: min_m,
            max_match: max_m,
            last_match: 0,
            hist,
            table: vec![Slot::default(); cap],
            mask: cap - 1,
        }
    }

    pub fn reset(&mut self) {
        self.hist.clear();
        self.table.fill(Slot::default());
        self.last_match = 0;
    }

    #[inline]
    pub fn history(&self) -> &[i32] {
        &self.hist
    }

    #[inline]
    pub fn last_match(&self) -> i32 {
        self.last_match
    }

    #[inline]
    pub fn min_match(&self) -> i32 {
        self.min_match
    }

    #[inline]
    pub fn set_min_match(&mut self, min_match: i32) {
        self.min_match = min_match.max(3);
    }

    #[inline]
    fn key_at(&self, end: usize) -> u64 {
        let a = (self.hist[end - 2] as u32) as u64;
        let b = (self.hist[end - 1] as u32) as u64;
        let c = (self.hist[end] as u32) as u64;
        let inner =
            a.wrapping_mul(0x9E3779B97F4A7C15) ^ mix(b.wrapping_add(0x632BE59BD9B4E019)) ^ (c << 1);
        mix(inner) | 1u64
    }

    fn find_slot_index(&mut self, key: u64, insert: bool) -> Option<usize> {
        let mut i = (key as usize) & self.mask;
        for _probes in 0..=self.mask {
            let s_key = self.table[i].key;
            if s_key == key {
                return Some(i);
            }
            if s_key == 0 {
                if !insert {
                    return None;
                }
                self.table[i].key = key;
                return Some(i);
            }
            i = (i + 1) & self.mask;
        }
        None
    }

    fn lookup_slot(&self, key: u64) -> Option<&Slot> {
        let mut i = (key as usize) & self.mask;
        for _probes in 0..=self.mask {
            let s = &self.table[i];
            if s.key == key {
                return Some(s);
            }
            if s.key == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
        None
    }

    pub fn append_slice(&mut self, tokens: &[i32]) {
        for &tok in tokens {
            self.hist.push(tok);
            let end = self.hist.len() - 1;
            if end < 2 {
                continue;
            }
            let key = self.key_at(end);
            if let Some(idx) = self.find_slot_index(key, true) {
                let s = &mut self.table[idx];
                for w in (1..WAYS).rev() {
                    s.pos[w] = s.pos[w - 1];
                }
                s.pos[0] = end as u32;
                if (s.n as usize) < WAYS {
                    s.n += 1;
                }
            }
        }
    }

    #[inline]
    pub fn append(&mut self, token: i32) {
        self.append_slice(&[token]);
    }

    pub fn propose(&mut self, max_k: i32, out: &mut [i32]) -> i32 {
        self.last_match = 0;
        let n = self.hist.len();
        if n < 4 || max_k <= 0 {
            return 0;
        }
        let max_k_usize = (max_k as usize).min(out.len());
        if max_k_usize == 0 {
            return 0;
        }
        let cur = n - 1;
        let key = self.key_at(cur);
        let Some(s) = self.lookup_slot(key) else {
            return 0;
        };
        let s_n = s.n as usize;
        let s_pos = s.pos;
        let mut best_end = 0usize;
        let mut best_len = 0i32;
        for w in 0..s_n {
            let p = s_pos[w] as usize;
            if p >= cur {
                continue;
            }
            let mut len = 0i32;
            while len < self.max_match
                && (len as usize) <= p
                && self.hist[p - (len as usize)] == self.hist[cur - (len as usize)]
            {
                len += 1;
            }
            if len > best_len {
                best_len = len;
                best_end = p;
            }
        }
        if best_len < self.min_match {
            return 0;
        }
        self.last_match = best_len;
        let mut k = 0usize;
        let mut q = best_end + 1;
        while q <= cur && k < max_k_usize {
            out[k] = self.hist[q];
            k += 1;
            q += 1;
        }
        k as i32
    }

    pub fn propose_after(&mut self, extra: &[i32], max_k: i32, out: &mut [i32]) -> i32 {
        if extra.is_empty() {
            return self.propose(max_k, out);
        }
        let n0 = self.hist.len();
        self.hist.extend_from_slice(extra);
        let k = self.propose(max_k, out);
        self.hist.truncate(n0);
        k
    }

    /// 1:1 port of `chain_tail` + `PromptLookupSource::propose` + `propose_from_sources`
    /// (`generate.cpp:74-83`, `include/strata/spec/draft_source.hpp`, `src/spec/draft_source.cpp`).
    pub fn propose_chain(
        &mut self,
        drafts: &[i32],
        n_drafts: i32,
        max_k: i32,
        min_match: i32,
        out: &mut [i32],
    ) -> (i32, i32) {
        if max_k <= 0 {
            return (0, 0);
        }
        let n_d = n_drafts.max(0) as usize;
        let keep = (64i32 - (n_d as i32)).max(0) as usize;
        let h_len = self.hist.len();
        let h0 = h_len.saturating_sub(keep);
        let mut ctail = [0i32; 64];
        let mut nt = 0usize;
        for i in h0..h_len {
            if nt < 64 {
                ctail[nt] = self.hist[i];
                nt += 1;
            }
        }
        for i in 0..n_d.min(drafts.len()) {
            if nt < 64 {
                ctail[nt] = drafts[i];
                nt += 1;
            }
        }
        let n_pending = n_d.min(nt);
        let max_k_usize = (max_k as usize).min(out.len());
        let mut buf = vec![0i32; max_k_usize];
        let k = self.propose_after(&ctail[(nt - n_pending)..nt], max_k, &mut buf);
        let m = if k > 0 { self.last_match() } else { 0 };
        if k <= 0 || m < min_match {
            return (0, 0);
        }
        let best = (k as usize).min(max_k_usize);
        out[..best].copy_from_slice(&buf[..best]);
        (best as i32, m)
    }
}

impl HostDrafter for SuffixDrafter {
    fn observe(&mut self, committed: &[u32]) {
        let toks: Vec<i32> = committed.iter().map(|&t| t as i32).collect();
        self.append_slice(&toks);
    }

    fn propose(&mut self, context_suffix: &[u32], max_draft: usize, out: &mut Vec<u32>) {
        out.clear();
        if self.hist.is_empty() && !context_suffix.is_empty() {
            let toks: Vec<i32> = context_suffix.iter().map(|&t| t as i32).collect();
            self.append_slice(&toks);
        }
        let mut buf = vec![0i32; max_draft];
        let n = SuffixDrafter::propose(self, max_draft as i32, &mut buf);
        for &t in &buf[..(n.max(0) as usize)] {
            if t >= 0 {
                out.push(t as u32);
            }
        }
    }
}

// ============================================================================
// 4. DraftPick & DraftPolicy — Lookup vs Device Drafter Cost/Acceptance Policy
// ============================================================================

/// 1:1 port of `strata::spec::DraftPolicy::Pick`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DraftPick {
    pub lookup: bool,
    pub t: i32,
}

/// 1:1 port of `strata::spec::DraftPolicy` (`src/spec/draft_policy.cpp`).
#[derive(Debug, Clone)]
pub struct DraftPolicy {
    max_t: i32,
    margin: f64,
    cost: [f64; Self::K_MAX_T + 1],
    cost_n: [f64; Self::K_MAX_T + 1],
    last: [f64; Self::K_MAX_T + 1],
    rounds: f64,
    mtp_tok: [f64; Self::K_MAX_T + 1],
    mtp_n: [f64; Self::K_MAX_T + 1],
    ok: [f64; Self::K_BUCKETS],
    bad: [f64; Self::K_BUCKETS],
    cok: [f64; Self::K_BUCKETS],
    cbad: [f64; Self::K_BUCKETS],
}

impl Default for DraftPolicy {
    fn default() -> Self {
        Self::new(8, 0.05)
    }
}

impl DraftPolicy {
    pub const K_MAX_T: usize = 8;
    pub const K_BUCKETS: usize = 4;

    const K_SHAPE: [f64; Self::K_MAX_T + 1] = [0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6];
    const K_COST_ALPHA: f64 = 0.1;
    const K_TOK_ALPHA: f64 = 0.05;
    const K_DECAY: f64 = 0.97;
    const K_PRIOR_Q: [f64; Self::K_BUCKETS] = [0.75, 0.88, 0.93, 0.96];
    const K_PRIOR_N: f64 = 4.0;
    const K_PROBES: f64 = 3.0;
    const K_STALE_ROUNDS: f64 = 300.0;
    const K_CHAIN_PRIOR: [f64; Self::K_BUCKETS] = [0.45, 0.65, 0.8, 0.9];

    pub fn new(max_t: i32, margin: f64) -> Self {
        Self {
            max_t: max_t.clamp(1, Self::K_MAX_T as i32),
            margin,
            cost: [0.0; Self::K_MAX_T + 1],
            cost_n: [0.0; Self::K_MAX_T + 1],
            last: [0.0; Self::K_MAX_T + 1],
            rounds: 0.0,
            mtp_tok: [0.0; Self::K_MAX_T + 1],
            mtp_n: [0.0; Self::K_MAX_T + 1],
            ok: [0.0; Self::K_BUCKETS],
            bad: [0.0; Self::K_BUCKETS],
            cok: [0.0; Self::K_BUCKETS],
            cbad: [0.0; Self::K_BUCKETS],
        }
    }

    #[inline]
    pub fn bucket(match_len: i32) -> usize {
        if match_len < 6 {
            0
        } else if match_len < 12 {
            1
        } else if match_len < 24 {
            2
        } else {
            3
        }
    }

    pub fn lookup_rate(&self, match_len: i32) -> f64 {
        let b = Self::bucket(match_len);
        (self.ok[b] + Self::K_PRIOR_N * Self::K_PRIOR_Q[b])
            / (self.ok[b] + self.bad[b] + Self::K_PRIOR_N)
    }

    pub fn cost_ms(&self, t: i32) -> f64 {
        let t_idx = t.clamp(1, Self::K_MAX_T as i32) as usize;
        if self.cost_n[t_idx] > 0.0 {
            return self.cost[t_idx];
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for u in 1..=Self::K_MAX_T {
            if self.cost_n[u] > 0.0 {
                let w = self.cost_n[u].min(20.0);
                num += w * self.cost[u] * Self::K_SHAPE[t_idx] / Self::K_SHAPE[u];
                den += w;
            }
        }
        if den > 0.0 {
            num / den
        } else {
            Self::K_SHAPE[t_idx]
        }
    }

    #[inline]
    fn stale(&self, t: usize) -> bool {
        self.cost_n[t] > 0.0 && (self.rounds - self.last[t]) > Self::K_STALE_ROUNDS
    }

    #[inline]
    fn mtp_tokens(&self, t: i32) -> f64 {
        let t_idx = t.clamp(1, Self::K_MAX_T as i32) as usize;
        if self.mtp_n[t_idx] > 0.0 {
            self.mtp_tok[t_idx]
        } else {
            1.0 + 0.7 * ((t - 1) as f64)
        }
    }

    pub fn choose(&self, t_mtp: i32, lookup_k: i32, match_len: i32) -> DraftPick {
        let mut p = DraftPick {
            lookup: false,
            t: t_mtp.clamp(1, self.max_t),
        };
        if lookup_k <= 0 {
            return p;
        }
        let base = self.mtp_tokens(p.t) / self.cost_ms(p.t);
        let q = self.lookup_rate(match_len);
        let mut e = 1.0;
        let mut qi = 1.0;
        let mut best = 0.0;
        let mut best_t = 0i32;
        let k_limit = lookup_k.min(self.max_t - 1);
        for k in 1..=k_limit {
            qi *= q;
            e += qi;
            let r = e / self.cost_ms(k + 1);
            if r > best {
                best = r;
                best_t = k + 1;
            }
        }
        let t_full = lookup_k.min(self.max_t - 1) + 1;
        if best_t > 0 && best > base * (1.0 + self.margin) {
            p.lookup = true;
            p.t = best_t;
            if best_t < t_full && q >= 0.85 && self.stale(t_full as usize) {
                p.t = t_full;
            }
            return p;
        }
        if t_full > p.t
            && (self.cost_n[t_full as usize] < Self::K_PROBES || self.stale(t_full as usize))
            && q >= 0.85
        {
            p.lookup = true;
            p.t = t_full;
        }
        p
    }

    pub fn observe(&mut self, lookup: bool, t: i32, accepted: i32, match_len: i32, round_ms: f64) {
        let t_clamped = t.clamp(1, Self::K_MAX_T as i32);
        let t_idx = t_clamped as usize;
        if round_ms > 0.0 {
            self.cost[t_idx] = if self.cost_n[t_idx] > 0.0 {
                (1.0 - Self::K_COST_ALPHA) * self.cost[t_idx] + Self::K_COST_ALPHA * round_ms
            } else {
                round_ms
            };
            self.cost_n[t_idx] += 1.0;
            self.last[t_idx] = self.rounds;
        }
        self.rounds += 1.0;
        if lookup {
            let b = Self::bucket(match_len);
            self.ok[b] = Self::K_DECAY * self.ok[b] + (accepted as f64);
            self.bad[b] =
                Self::K_DECAY * self.bad[b] + if accepted < t_clamped - 1 { 1.0 } else { 0.0 };
        } else {
            let got = (accepted as f64) + 1.0;
            self.mtp_tok[t_idx] = if self.mtp_n[t_idx] > 0.0 {
                (1.0 - Self::K_TOK_ALPHA) * self.mtp_tok[t_idx] + Self::K_TOK_ALPHA * got
            } else {
                got
            };
            self.mtp_n[t_idx] += 1.0;
        }
    }

    pub fn chain_rate(&self, match_len: i32) -> f64 {
        let b = Self::bucket(match_len);
        (self.cok[b] + Self::K_PRIOR_N * Self::K_CHAIN_PRIOR[b])
            / (self.cok[b] + self.cbad[b] + Self::K_PRIOR_N)
    }

    pub fn chain(&self, t_mtp: i32, p_mtp: f64, k_avail: i32, match_len: i32) -> i32 {
        let t_m = t_mtp.clamp(1, self.max_t);
        let kmax = k_avail.min(self.max_t - t_m);
        if kmax <= 0 {
            return 0;
        }
        let e0 = self.mtp_tokens(t_m);
        let base = e0 / self.cost_ms(t_m);
        let c = self.chain_rate(match_len);
        let p_m = p_mtp.clamp(0.0, 1.0);
        let mut gain = 0.0;
        let mut ci = 1.0;
        let mut best = 0.0;
        let mut best_k = 0i32;
        for k in 1..=kmax {
            ci *= c;
            gain += ci;
            let r = (e0 + p_m * gain) / self.cost_ms(t_m + k);
            if r > best {
                best = r;
                best_k = k;
            }
        }
        let t_full = (t_m + kmax) as usize;
        if best_k > 0 && best > base * (1.0 + self.margin) {
            if best_k < kmax && p_m * c >= 0.6 && self.stale(t_full) {
                return kmax;
            }
            return best_k;
        }
        if (self.cost_n[t_full] < Self::K_PROBES || self.stale(t_full)) && p_m * c >= 0.6 {
            return kmax;
        }
        0
    }

    pub fn observe_chain(
        &mut self,
        t_mtp: i32,
        k: i32,
        accepted: i32,
        match_len: i32,
        round_ms: f64,
    ) {
        let t_idx = (t_mtp + k).clamp(1, Self::K_MAX_T as i32) as usize;
        if round_ms > 0.0 {
            self.cost[t_idx] = if self.cost_n[t_idx] > 0.0 {
                (1.0 - Self::K_COST_ALPHA) * self.cost[t_idx] + Self::K_COST_ALPHA * round_ms
            } else {
                round_ms
            };
            self.cost_n[t_idx] += 1.0;
            self.last[t_idx] = self.rounds;
        }
        self.rounds += 1.0;
        let mtp_acc = accepted.min(t_mtp - 1);
        let got = (mtp_acc as f64) + 1.0;
        let tm_idx = t_mtp.clamp(1, Self::K_MAX_T as i32) as usize;
        self.mtp_tok[tm_idx] = if self.mtp_n[tm_idx] > 0.0 {
            (1.0 - Self::K_TOK_ALPHA) * self.mtp_tok[tm_idx] + Self::K_TOK_ALPHA * got
        } else {
            got
        };
        self.mtp_n[tm_idx] += 1.0;
        if accepted < t_mtp - 1 {
            return;
        }
        let b = Self::bucket(match_len);
        let ok = accepted - (t_mtp - 1);
        self.cok[b] = Self::K_DECAY * self.cok[b] + (ok as f64);
        self.cbad[b] = Self::K_DECAY * self.cbad[b] + if ok < k { 1.0 } else { 0.0 };
    }
}

/// 1:1 port of `strata::core::parse_logit_bias` (`include/strata/core/logit_bias.hpp`).
pub fn parse_logit_bias(text: &str, vocab: usize) -> Result<Vec<f32>, String> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let mut bias = vec![0.0f32; vocab];
    let bytes = text.as_bytes();
    let mut p = 0usize;
    while p < bytes.len() {
        if bytes[p] == b',' {
            p += 1;
            continue;
        }
        let item_start = p;
        let mut item_end = p;
        while item_end < bytes.len() && bytes[item_end] != b',' {
            item_end += 1;
        }
        let item = &text[item_start..item_end];
        let Some(colon_rel) = item.find(':') else {
            return Err("bad logit_bias (expected id:bias,id:bias)".to_string());
        };
        let id_str = &item[..colon_rel];
        let val_str = &item[colon_rel + 1..];
        if id_str.is_empty() || val_str.is_empty() {
            return Err("bad logit_bias (expected id:bias,id:bias)".to_string());
        }
        let id: i64 = id_str
            .trim_start()
            .parse::<i64>()
            .map_err(|_| "bad logit_bias (expected id:bias,id:bias)".to_string())?;
        let val_f64: f64 = val_str
            .trim_start()
            .parse::<f64>()
            .map_err(|_| "bad logit_bias (expected id:bias,id:bias)".to_string())?;
        if val_f64.is_nan() {
            return Err("bad logit_bias (expected id:bias,id:bias)".to_string());
        }
        let val = val_f64 as f32;
        if id < 0 || (id as usize) >= vocab {
            return Err(format!("logit_bias token id {id} out of [0, {vocab})"));
        }
        let clamped = if val < -1e9f32 || val == f32::NEG_INFINITY {
            -1e9f32
        } else if val > 100.0f32 || val == f32::INFINITY {
            100.0f32
        } else if !val.is_finite() {
            return Err("bad logit_bias (expected id:bias,id:bias)".to_string());
        } else {
            val
        };
        bias[id as usize] = clamped;
        p = item_end;
    }
    if bias.iter().all(|&v| v == 0.0) {
        bias.clear();
    }
    Ok(bias)
}

// ============================================================================
// 5. ModelSpecDecl & SpecEngineService — Declarative Model Spec + Engine Service
// ============================================================================

/// Declarative per-model speculation specification (`WP-5`).
///
/// Each model declares its vocabulary size, supported device/host drafters, crossover table,
/// DFlash profile (if any), and adaptive pause thresholds, while [`SpecEngineService`] executes
/// the drafter selection, host suffix/lookup/ngram drafting, adaptive draft length, and
/// rejection/greedy verification uniformly.
#[derive(Debug, Clone)]
pub struct ModelSpecDecl {
    /// Short model family identifier (`"qwen35-27b"`, `"qwen4exp"`, `"laguna"`, `"mimo"`, `"glm"`).
    pub model_name: &'static str,
    /// Target model vocabulary size.
    pub n_vocab: usize,
    /// Whether the model has a native Multi-Token Prediction (MTP) head.
    pub has_mtp: bool,
    /// Whether the model supports a DFlash / DFlash2 block-diffusion drafter.
    pub has_dflash: bool,
    /// Whether the 8-way set-associative [`SuffixDrafter`] is enabled.
    pub has_suffix: bool,
    /// Whether the FNV-1a [`PromptLookupDrafter`] is enabled.
    pub has_prompt_lookup: bool,
    /// Whether the 4-gram [`NgramMod`] table drafter is enabled.
    pub has_ngram: bool,
    /// Minimum suffix match length for [`SuffixDrafter`] / [`PromptLookupDrafter`].
    pub min_match: i32,
    /// Maximum draft tokens per round.
    pub max_draft: i32,
    /// Per-model context-depth crossover table.
    pub crossover: CrossoverTable,
    /// Optional [`DFlashModelProfile`] when `has_dflash` is true.
    pub dflash_profile: Option<DFlashModelProfile>,
    /// Minimum speedup ratio before [`AdaptivePausePolicy`] pauses the device drafter (`1.2`).
    pub pause_ratio: f32,
}

impl ModelSpecDecl {
    /// Qwen3.5-27B (`qwen35`): DFlash2 (`B=16`) + MTP + `NgramMod` + `DraftLen`.
    pub fn qwen35_27b() -> Self {
        Self {
            model_name: "qwen35-27b",
            n_vocab: 248_320,
            has_mtp: true,
            has_dflash: true,
            has_suffix: true,
            has_prompt_lookup: false,
            has_ngram: true,
            min_match: 4,
            max_draft: 15,
            crossover: CrossoverTable::qwen35_27b(),
            dflash_profile: Some(DFlashModelProfile::qwen38_27b_dflash2()),
            pause_ratio: 1.2,
        }
    }

    /// Qwen3.8-Flash-Next (`qwen4exp` / Flash-Next): MTP + `SuffixDrafter` + `DraftPolicy`.
    pub fn qwen35_35b_a3b() -> Self {
        Self {
            model_name: "qwen4exp",
            n_vocab: 248_320,
            has_mtp: true,
            has_dflash: false,
            has_suffix: true,
            has_prompt_lookup: false,
            has_ngram: true,
            min_match: 4,
            max_draft: 7,
            crossover: CrossoverTable::qwen35_35b_a3b(),
            dflash_profile: None,
            pause_ratio: 1.2,
        }
    }

    /// Laguna-S-2.1 (`laguna`): `PromptLookupDrafter` + `SuffixDrafter` + `NgramMod` + DFlash (`B=8`).
    pub fn laguna_s_2_1() -> Self {
        Self {
            model_name: "laguna",
            n_vocab: 100_352,
            has_mtp: false,
            has_dflash: true,
            has_suffix: true,
            has_prompt_lookup: true,
            has_ngram: true,
            min_match: 5,
            max_draft: 7,
            crossover: CrossoverTable::laguna_s_2_1(),
            dflash_profile: Some(DFlashModelProfile::laguna_s_2_1_dflash()),
            pause_ratio: 1.2,
        }
    }

    /// MiMo-V2.6-Flash (`mimo`): DFlash (`B=5`, `k_dim=128`, `Q8_0` head) + `SuffixDrafter` + `PromptLookupDrafter`.
    pub fn mimo_v2_flash() -> Self {
        Self {
            model_name: "mimo",
            n_vocab: 152_064,
            has_mtp: false,
            has_dflash: true,
            has_suffix: true,
            has_prompt_lookup: true,
            has_ngram: true,
            min_match: 5,
            max_draft: 4,
            crossover: CrossoverTable::mimo_v2_flash(),
            dflash_profile: Some(DFlashModelProfile::mimo_v2_flash_dflash()),
            pause_ratio: 1.2,
        }
    }

    /// GLM-4.7-Flash (`glm`): MTP + `SuffixDrafter` + `NgramMod`.
    pub fn glm_4_7_flash() -> Self {
        Self {
            model_name: "glm",
            n_vocab: 154_880,
            has_mtp: true,
            has_dflash: false,
            has_suffix: true,
            has_prompt_lookup: true,
            has_ngram: true,
            min_match: 5,
            max_draft: 7,
            crossover: CrossoverTable::glm_4_7_flash(),
            dflash_profile: None,
            pause_ratio: 1.2,
        }
    }

    /// Construct a parameterized Laguna-S-2.1 declaration with custom `spec` and `min_match`.
    pub fn laguna(spec: usize, min_match: usize) -> Self {
        let mut d = Self::laguna_s_2_1();
        d.max_draft = spec.max(1) as i32;
        d.min_match = min_match.max(2) as i32;
        d
    }

    /// Construct a parameterized MiMo-V2.6-Flash declaration with custom `draft_max` and `has_dflash`.
    pub fn mimo(draft_max: usize, has_dflash: bool) -> Self {
        let mut d = Self::mimo_v2_flash();
        d.max_draft = draft_max.max(1) as i32;
        d.has_dflash = has_dflash;
        d
    }
}

/// Unified Speculative Decoding Engine Service (`WP-5`).
///
/// Centralizes drafter selection (`DrafterPick`), DFlash KV window tracking (`DfCtx`),
/// adaptive draft length (`DraftLen`), host suffix/lookup/n-gram drafters (`SuffixDrafter`,
/// `PromptLookupDrafter`, `NgramMod`), cost/bucket policy (`DraftPolicy`), device-drafter
/// adaptive pause (`AdaptivePausePolicy`), and fast `O(V + K log K)` sampling/verification.
#[derive(Debug, Clone)]
pub struct SpecEngineService {
    /// Declarative per-model speculation configuration.
    pub decl: ModelSpecDecl,
    /// Adaptive draft length controller (`DraftLen`).
    pub adraft: DraftLen,
    /// DFlash KV ring / sliding-window coverage tracker (`DfCtx`).
    pub df_ctx: DfCtx,
    /// Context-depth drafter selector (`DrafterPick`).
    pub drafter_pick: DrafterPick,
    /// 8-way set-associative suffix hash drafter (`SuffixDrafter`).
    pub suffix: SuffixDrafter,
    /// Cost/bucket lookup-vs-device controller (`DraftPolicy`).
    pub draft_policy: DraftPolicy,
    /// FNV-1a prompt-lookup drafter (`PromptLookupDrafter`).
    pub prompt_lookup: PromptLookupDrafter,
    /// 4-gram table drafter (`NgramMod`).
    pub ngram: NgramMod,
    /// Adaptive device-drafter pause policy (`AdaptivePausePolicy`).
    pub pause_policy: AdaptivePausePolicy,
    /// Optional reduced draft vocabulary (`DraftVocab`).
    pub draft_vocab: Option<DraftVocab>,
    /// Mersenne Twister RNG (`Mt19937_64`).
    pub rng: Mt19937_64,
}

impl SpecEngineService {
    /// Construct a new [`SpecEngineService`] from a model's [`ModelSpecDecl`].
    pub fn new(decl: ModelSpecDecl) -> Self {
        let mut adraft = DraftLen::new();
        adraft.init(decl.max_draft, decl.max_draft > 1);

        let mut df_ctx = DfCtx::default();
        if let Some(ref prof) = decl.dflash_profile {
            df_ctx.init(
                prof.swa_window as i32,
                (prof.swa_window * 2) as i32,
                prof.block_size as i32,
            );
        }

        let mut drafter_pick = DrafterPick::default();
        drafter_pick.init_with_table(
            decl.crossover,
            "gfx1151",
            decl.has_mtp && decl.has_dflash,
            if decl.has_mtp && decl.has_dflash {
                "auto"
            } else if decl.has_dflash {
                "dflash"
            } else if decl.has_mtp {
                "mtp"
            } else {
                "lookup"
            },
        );

        let suffix = SuffixDrafter::new(decl.min_match, 32, 32768);
        let draft_policy = DraftPolicy::new((decl.max_draft + 1).clamp(2, 8), 0.05);
        let mut prompt_lookup = PromptLookupDrafter::new();
        prompt_lookup.min_match = decl.min_match.max(2) as usize;
        let mut ngram = NgramMod::new();
        ngram.init();
        let mut pause_policy =
            AdaptivePausePolicy::new(decl.pause_ratio).with_max_draft(decl.max_draft.max(1) as u32);
        if decl.has_dflash && !decl.has_mtp {
            if decl.model_name == "mimo" {
                pause_policy = pause_policy
                    .with_metric(crate::pause::AdaptivePauseMetric::SpeedupRatio)
                    .with_cost_per_draft(0.20)
                    .with_backoff_bounds(8, 16);
                pause_policy.pause_ratio = 1.02;
                pause_policy.ema_decay = 0.85;
                pause_policy.ema_speedup = 1.40;
            } else {
                pause_policy = pause_policy
                    .with_metric(crate::pause::AdaptivePauseMetric::SpeedupRatio)
                    .with_cost_per_draft(0.20);
                pause_policy.pause_ratio = 1.02;
                pause_policy.ema_speedup = 1.35;
            }
        }

        Self {
            decl,
            adraft,
            df_ctx,
            drafter_pick,
            suffix,
            draft_policy,
            prompt_lookup,
            ngram,
            pause_policy,
            draft_vocab: None,
            rng: Mt19937_64::default(),
        }
    }

    /// Reset all per-request drafters and seed them with `prompt` tokens.
    pub fn reset_request(&mut self, prompt: &[i32]) {
        self.adraft.reset_request();
        self.drafter_pick.request_start();
        self.pause_policy.reset();
        if self.decl.has_dflash && !self.decl.has_mtp {
            self.pause_policy.ema_speedup = if self.decl.model_name == "mimo" {
                1.40
            } else {
                1.35
            };
        }
        if self.decl.has_suffix {
            self.suffix.reset();
            self.suffix.append_slice(prompt);
        }
        if self.decl.has_prompt_lookup {
            self.prompt_lookup.reset(prompt);
        }
        if self.decl.has_ngram {
            self.ngram.reset();
            self.ngram.add_n(prompt, prompt.len());
        }
    }

    /// Append newly committed tokens to all active host drafters (`SuffixDrafter`,
    /// `PromptLookupDrafter`, `NgramMod`).
    pub fn commit_tokens(&mut self, toks: &[i32]) {
        if toks.is_empty() {
            return;
        }
        if self.decl.has_suffix {
            self.suffix.append_slice(toks);
        }
        if self.decl.has_prompt_lookup {
            for &t in toks {
                self.prompt_lookup.push(t);
            }
        }
        if self.decl.has_ngram {
            for &t in toks {
                let hist = self.suffix.history();
                if hist.len() >= 4 {
                    self.ngram.add(&hist[hist.len() - 4..]);
                } else {
                    let _ = t;
                }
            }
        }
    }

    /// Alias for [`SpecEngineService::new`].
    #[inline]
    pub fn from_decl(decl: ModelSpecDecl) -> Self {
        Self::new(decl)
    }

    /// Alias for [`SpecEngineService::reset_request`].
    #[inline]
    pub fn reset(&mut self, prompt: &[i32]) {
        self.reset_request(prompt);
    }

    /// Ingest a single committed token into all active host drafters.
    #[inline]
    pub fn ingest_token(&mut self, tok: i32) {
        self.commit_tokens(&[tok]);
    }

    /// Propose up to `max_k` host draft tokens into `out` and return the count drafted.
    #[inline]
    pub fn draft_host(&mut self, max_k: usize, out: &mut [i32]) -> usize {
        let min_m = self.decl.min_match.max(2) as usize;
        let (nd, _) = self.propose_host(max_k, max_k, min_m, out);
        nd
    }

    /// Select the active device draft length `0..=decl.max_draft` for the current position and
    /// sampling mode, returning `0` when the adaptive pause policy is backed off or the depth
    /// crossover disables the device drafter.
    #[inline]
    pub fn select_draft_len(&mut self, pos: usize, greedy: bool) -> usize {
        if !self.decl.has_dflash && !self.decl.has_mtp {
            return 0;
        }
        if self.pause_policy.is_paused() {
            return 0;
        }
        let _ = self.drafter_pick.want_dfl(pos as i64, !greedy);
        if self.decl.has_dflash && !self.decl.has_mtp {
            return self.decl.max_draft.max(0) as usize;
        }
        self.adraft.choose().clamp(0, self.decl.max_draft) as usize
    }

    /// Record verification outcome `(proposed, accepted)` in `adraft`, `draft_policy`, and `pause_policy`.
    #[inline]
    pub fn record_verify(&mut self, proposed: usize, accepted: usize) {
        let used_host = !self.decl.has_dflash && !self.decl.has_mtp;
        self.observe_round(used_host, proposed, accepted, self.decl.min_match, 0.0);
    }

    /// Propose up to `max_k.min(room)` host draft tokens using [`SuffixDrafter`] (8-way longest
    /// suffix match) and [`PromptLookupDrafter`], returning `(n_drafted, match_len)`.
    pub fn propose_host(
        &mut self,
        max_k: usize,
        room: usize,
        min_match: usize,
        out: &mut [i32],
    ) -> (usize, i32) {
        let limit = max_k.min(room).min(out.len());
        if limit == 0 {
            if self.decl.has_prompt_lookup {
                self.prompt_lookup.update_acc_est(0, 0);
            }
            return (0, 0);
        }

        // 1. Prefer SuffixDrafter (8-way longest match up to 32 tokens) when match_len >= min_match
        if self.decl.has_suffix {
            let prev_min = self.suffix.min_match();
            self.suffix.set_min_match(min_match.max(3) as i32);
            let k = self.suffix.propose(limit as i32, out);
            let m = self.suffix.last_match();
            self.suffix.set_min_match(prev_min);
            if k > 0 && (m as usize) >= min_match {
                return (k as usize, m);
            }
        }

        // 2. Fall back to PromptLookupDrafter (3-gram / 2-gram FNV-1a index)
        if self.decl.has_prompt_lookup {
            let best_k = self.prompt_lookup.choose_k(limit, room);
            if best_k > 0 {
                let nd = self.prompt_lookup.draft(best_k, out, min_match);
                if nd > 0 {
                    return (nd, min_match as i32);
                }
            }
            self.prompt_lookup.update_acc_est(0, 0);
        }

        (0, 0)
    }

    /// Observe the outcome of a speculative round and update `adraft`, `prompt_lookup`,
    /// `draft_policy`, and `pause_policy`.
    pub fn observe_round(
        &mut self,
        used_host: bool,
        proposed: usize,
        accepted: usize,
        match_len: i32,
        round_ms: f64,
    ) {
        if proposed > 0 {
            self.adraft.observe(proposed as i32, accepted as i32);
        }
        if used_host {
            if self.decl.has_prompt_lookup {
                self.prompt_lookup.update_acc_est(proposed, accepted);
            }
            self.draft_policy.observe(
                true,
                (proposed + 1) as i32,
                accepted as i32,
                match_len,
                round_ms,
            );
            if accepted > 0 {
                let _ = self
                    .pause_policy
                    .observe_round(proposed as u32, accepted as u32);
            }
        } else if proposed > 0 {
            self.draft_policy
                .observe(false, (proposed + 1) as i32, accepted as i32, 0, round_ms);
            let _ = self
                .pause_policy
                .observe_round(proposed as u32, accepted as u32);
        } else {
            let was_paused = self.pause_policy.is_paused();
            let _ = self.pause_policy.observe_round(0, 0);
            if was_paused
                && !self.pause_policy.is_paused()
                && self.decl.has_dflash
                && !self.decl.has_mtp
            {
                self.pause_policy.ema_speedup = 1.25;
            }
        }
    }

    /// Sample a single token from `logits` using `cfg` and `recent` history via `O(V + K log K)`
    /// partial selection.
    #[inline]
    pub fn sample_token(&mut self, logits: &[f32], cfg: &SampCfg, recent: &[i32]) -> i32 {
        sample(logits, self.decl.n_vocab as i32, cfg, recent, &mut self.rng)
    }

    /// Verify `drafts` greedily against `(drafts.len() + 1) * n_vocab` target logits, returning
    /// `(accepted_count, bonus_token)`.
    #[inline]
    pub fn verify_greedy(&self, target_logits: &[f32], drafts: &[i32]) -> (usize, i32) {
        let mut accepted = 0usize;
        let bonus = decide_greedy(
            target_logits,
            self.decl.n_vocab,
            drafts,
            drafts.len(),
            &mut accepted,
        );
        (accepted, bonus)
    }

    /// Verify `drafts` under stochastic rejection sampling against `(drafts.len() + 1) * n_vocab`
    /// target logits, appending accepted + bonus tokens to `recent` and returning
    /// `(accepted_count, bonus_token)`.
    #[inline]
    pub fn verify_sampled(
        &mut self,
        target_logits: &[f32],
        drafts: &[i32],
        cfg: &SampCfg,
        recent: &mut Vec<i32>,
        q_draft: Option<&dyn Fn(usize, i32) -> f64>,
    ) -> (usize, i32) {
        let mut accepted = 0usize;
        let bonus = decide_sampled(
            target_logits,
            self.decl.n_vocab as i32,
            drafts,
            drafts.len(),
            cfg,
            recent,
            &mut self.rng,
            &mut accepted,
            q_draft,
        );
        (accepted, bonus)
    }
}
