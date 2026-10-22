//! Long-span copy proposer (`CopyProposer`): exact drafting by copying a long earlier stretch of the context.
//!
//! When the last tokens of the context (committed tokens plus any uncommitted drafts of this round) repeat an
//! earlier stretch of the prompt or of the generated text by many tokens, the tokens that followed that stretch are
//! a very good guess for what comes next: code edits re-emit most of the file they edit, quoted input is quoted
//! again, tool calls repeat their arguments.  This proposer finds the longest such match and proposes a long copy
//! of what followed it, so a verify window can be filled without the neural drafter (or extend the neural
//! drafter's own proposal).
//!
//! The idea (a greedy copy lookup that proposes long copies of 12+ matching tokens and resumes after a short edit)
//! comes from gufo (github.com/gufo-org/gufo, MIT, `src/core/speculative/prompt_lookup.hpp`); this is an
//! independent implementation, see THIRD_PARTY_NOTICES.md.
//!
//! Exactness: a proposal depends only on tokens that are already in the context, never on the target's logits, so
//! greedy verification keeps the output byte-identical and rejection sampling (a copy is a point-mass proposal)
//! keeps the sampled distribution exact.  The proposer only changes which tokens are verified.
//!
//! Index: every position `e >= KEY` is chained by the hash of the `KEY` tokens that end just before it
//! (`toks[e-KEY..e]`), newest first, so a lookup walks the most recent occurrences of the context's last `KEY`
//! tokens and extends each match backwards against the context.

/// Tokens hashed into one index key (the shortest match that is looked at).
pub const COPY_KEY: usize = 3;
const NONE: u32 = u32::MAX;
const EMPTY_KEY: u64 = 0;

/// Tuning of [`CopyProposer`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopyCfg {
    /// A fresh copy needs at least this many matching context tokens (greedy requests).
    pub min_match: usize,
    /// A fresh copy needs at least this many matching context tokens (sampled requests).
    pub sampled_min_match: usize,
    /// Near the point where the last mostly-accepted copy stopped, this many matching tokens are enough (the
    /// output edited a few tokens of the text it copies and rejoins it).
    pub resume_min_match: usize,
    /// The backward match is not extended beyond this many tokens.
    pub max_match: usize,
    /// At most this many earlier occurrences of the key are compared.
    pub max_cands: usize,
    /// A resumed copy may start up to this many tokens after the last copy's stop point ...
    pub resume_window: usize,
    /// ... or this many before it.
    pub resume_back: usize,
    /// Copies shorter than this (the source runs out) are not proposed.
    pub min_copy: usize,
    /// Sampled requests copy only from the context they started with (not from their own fresh output).
    pub sampled_prompt_only: bool,
}

impl Default for CopyCfg {
    fn default() -> Self {
        Self {
            min_match: 12,
            sampled_min_match: 24,
            resume_min_match: 5,
            max_match: 64,
            max_cands: 16,
            resume_window: 512,
            resume_back: 8,
            min_copy: 3,
            sampled_prompt_only: true,
        }
    }
}

/// A match found by [`CopyProposer::find`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CopyMatch {
    /// Committed index of the first token to copy.
    pub start: usize,
    /// Matching context tokens (`>= COPY_KEY`).
    pub len: usize,
    /// The match qualified only through the resume rule.
    pub resumed: bool,
}

/// Running counts (for logs and tuning).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CopyStats {
    pub windows: u64,
    pub proposed: u64,
    pub accepted: u64,
    pub chain_windows: u64,
    pub chain_proposed: u64,
    pub chain_accepted: u64,
}

/// Long-span exact copy proposer (see the module docs).
#[derive(Debug, Clone)]
pub struct CopyProposer {
    pub cfg: CopyCfg,
    toks: Vec<i32>,
    /// `prev[e]`: the previous position whose key equals the key ending at `e` (NONE: none).
    prev: Vec<u32>,
    keys: Vec<u64>,
    vals: Vec<u32>,
    used: usize,
    /// Where the last mostly-accepted copy stopped (NONE: nothing to resume).
    resume: usize,
    /// Committed tokens when the current request started (NONE: not started).
    req_start: usize,
    /// Per match-length bucket, decayed accepted / rejected token counts of copies.
    ok: [f64; 4],
    bad: [f64; 4],
    pub stats: CopyStats,
}

impl Default for CopyProposer {
    fn default() -> Self {
        Self::new(CopyCfg::default())
    }
}

#[inline]
fn key_of(t: &[i32]) -> u64 {
    // splitmix-style mix of the KEY tokens; never EMPTY_KEY
    let mut h: u64 = 0x2545_F491_4F6C_DD1D;
    for &x in t {
        h ^= (x as u32) as u64;
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
    }
    h ^= h >> 32;
    h | 1
}

impl CopyProposer {
    pub fn new(cfg: CopyCfg) -> Self {
        Self {
            cfg,
            toks: Vec::new(),
            prev: Vec::new(),
            keys: vec![EMPTY_KEY; 1024],
            vals: vec![NONE; 1024],
            used: 0,
            resume: usize::MAX,
            req_start: usize::MAX,
            ok: [0.0; 4],
            bad: [0.0; 4],
            stats: CopyStats::default(),
        }
    }

    /// Forgets the index and the request state (keeps the learned acceptance and the capacity).
    pub fn clear(&mut self) {
        self.toks.clear();
        self.prev.clear();
        self.keys.fill(EMPTY_KEY);
        self.vals.fill(NONE);
        self.used = 0;
        self.resume = usize::MAX;
        self.req_start = usize::MAX;
    }

    /// Committed tokens indexed so far.
    #[inline]
    pub fn len(&self) -> usize {
        self.toks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.toks.is_empty()
    }

    #[inline]
    pub fn tokens(&self) -> &[i32] {
        &self.toks
    }

    /// A request starts now, with the tokens indexed so far as its context: nothing to resume, and matches may
    /// not reach back over this point (a shared chat header must not count as a long matching answer).
    pub fn begin_request(&mut self) {
        self.resume = usize::MAX;
        self.req_start = self.toks.len();
    }

    /// Rebuilds the index to hold exactly `toks` when it does not already (a restored or edited conversation);
    /// appends the new tail otherwise.
    pub fn sync(&mut self, toks: &[i32]) {
        let n = self.toks.len();
        if toks.len() < n || toks[..n] != self.toks[..] {
            let rs = self.req_start;
            self.clear();
            self.req_start = if rs == usize::MAX {
                rs
            } else {
                rs.min(toks.len())
            };
            self.extend(toks);
        } else {
            self.extend(&toks[n..]);
        }
    }

    fn grow(&mut self) {
        let cap = self.keys.len() * 2;
        let old_k = std::mem::replace(&mut self.keys, vec![EMPTY_KEY; cap]);
        let old_v = std::mem::replace(&mut self.vals, vec![NONE; cap]);
        for (k, v) in old_k.into_iter().zip(old_v) {
            if k != EMPTY_KEY {
                let mut i = (k as usize) & (cap - 1);
                while self.keys[i] != EMPTY_KEY {
                    i = (i + 1) & (cap - 1);
                }
                self.keys[i] = k;
                self.vals[i] = v;
            }
        }
    }

    /// Sets the head of `key` to `pos`; returns the previous head.
    fn swap_head(&mut self, key: u64, pos: u32) -> u32 {
        if 2 * (self.used + 1) > self.keys.len() {
            self.grow();
        }
        let m = self.keys.len() - 1;
        let mut i = (key as usize) & m;
        loop {
            let k = self.keys[i];
            if k == key {
                return std::mem::replace(&mut self.vals[i], pos);
            }
            if k == EMPTY_KEY {
                self.keys[i] = key;
                self.vals[i] = pos;
                self.used += 1;
                return NONE;
            }
            i = (i + 1) & m;
        }
    }

    fn head(&self, key: u64) -> u32 {
        let m = self.keys.len() - 1;
        let mut i = (key as usize) & m;
        loop {
            let k = self.keys[i];
            if k == key {
                return self.vals[i];
            }
            if k == EMPTY_KEY {
                return NONE;
            }
            i = (i + 1) & m;
        }
    }

    /// Appends one committed token.
    pub fn push(&mut self, tok: i32) {
        let e = self.toks.len();
        self.toks.push(tok);
        // the key ending just before e is now followed by toks[e]: index position e
        let p = if e >= COPY_KEY && e < NONE as usize {
            let k = key_of(&self.toks[e - COPY_KEY..e]);
            self.swap_head(k, e as u32)
        } else {
            NONE
        };
        self.prev.push(p);
    }

    /// Appends committed tokens.
    pub fn extend(&mut self, toks: &[i32]) {
        self.toks.reserve(toks.len());
        self.prev.reserve(toks.len());
        for &t in toks {
            self.push(t);
        }
    }

    #[inline]
    fn resumes(&self, e: usize) -> bool {
        self.resume != usize::MAX
            && e + self.cfg.resume_back >= self.resume
            && e <= self.resume + self.cfg.resume_window
    }

    /// Whether a mostly-accepted copy left a resume point.
    #[inline]
    pub fn has_resume(&self) -> bool {
        self.resume != usize::MAX
    }

    /// The best (longest) match of the context `committed ++ tail` against the indexed tokens, among the matches
    /// that qualify: `len >= min_match`, or near the resume point with `len >= resume_min_match` (with
    /// `require_resume`: near the resume point and `len >= min_match`).  `src_end` bounds the copy source (the first token to copy lies below
    /// it).  None when nothing qualifies.
    pub fn find(
        &self,
        tail: &[i32],
        min_match: usize,
        require_resume: bool,
        src_end: usize,
    ) -> Option<CopyMatch> {
        let nc = self.toks.len();
        let ctx = nc + tail.len();
        let at = |i: usize| -> i32 {
            if i < nc {
                self.toks[i]
            } else {
                tail[i - nc]
            }
        };
        // a match may not reach back over the request's start
        let avail = if self.req_start == usize::MAX {
            ctx
        } else {
            ctx.saturating_sub(self.req_start)
        };
        let max_match = self.cfg.max_match.min(avail);
        if max_match < COPY_KEY || (require_resume && self.resume == usize::MAX) {
            return None;
        }
        let mut kt = [0i32; COPY_KEY];
        for (i, k) in kt.iter_mut().enumerate() {
            *k = at(ctx - COPY_KEY + i);
        }
        // the hash may collide: every candidate is checked token by token below
        let mut e = self.head(key_of(&kt));
        let mut best: Option<CopyMatch> = None;
        let mut n = 0usize;
        let src_end = src_end.min(nc);
        while e != NONE && n < self.cfg.max_cands {
            n += 1;
            let eu = e as usize;
            let next = self.prev[eu];
            if eu >= src_end {
                e = next;
                continue;
            }
            let mut len = 0usize;
            while len < max_match && len < eu && self.toks[eu - len - 1] == at(ctx - len - 1) {
                len += 1;
            }
            if len >= COPY_KEY {
                let near = self.resumes(eu);
                let ok = if require_resume {
                    // replacing a sampled neural proposal needs an established copy and a full match
                    near && len >= min_match
                } else {
                    len >= min_match || (near && len >= self.cfg.resume_min_match)
                };
                if ok && best.is_none_or(|b| len > b.len) {
                    best = Some(CopyMatch {
                        start: eu,
                        len,
                        resumed: len < min_match,
                    });
                }
            }
            e = next;
        }
        best
    }

    /// The copy source bound for a request (`sampled`: its starting context only, when so configured).
    #[inline]
    pub fn src_end(&self, sampled: bool) -> usize {
        if sampled && self.cfg.sampled_prompt_only && self.req_start != usize::MAX {
            self.req_start
        } else {
            self.toks.len()
        }
    }

    /// Proposes up to `cap` tokens continuing `committed ++ tail` into `out` (cleared first).  Stops after a token
    /// for which `is_stop` holds (it is still proposed).  Returns the match used, or None (and `out` empty).
    pub fn propose(
        &self,
        tail: &[i32],
        cap: usize,
        sampled: bool,
        require_resume: bool,
        is_stop: impl Fn(i32) -> bool,
        out: &mut Vec<i32>,
    ) -> Option<CopyMatch> {
        out.clear();
        if cap == 0 {
            return None;
        }
        let min_match = if sampled {
            self.cfg.sampled_min_match
        } else {
            self.cfg.min_match
        };
        let src_end = self.src_end(sampled);
        let m = self.find(tail, min_match, require_resume, src_end)?;
        let avail = src_end - m.start;
        if avail < self.cfg.min_copy.min(cap) {
            return None;
        }
        for &t in &self.toks[m.start..m.start + avail.min(cap)] {
            out.push(t);
            if is_stop(t) {
                break;
            }
        }
        Some(m)
    }

    #[inline]
    fn bucket(m: &CopyMatch) -> usize {
        if m.resumed {
            0
        } else if m.len < 24 {
            1
        } else if m.len < 48 {
            2
        } else {
            3
        }
    }

    /// Estimated per-token acceptance of a copy from a match like `m` (decayed counts with a prior).
    pub fn accept_rate(&self, m: &CopyMatch) -> f64 {
        const PRIOR: [f64; 4] = [0.85, 0.9, 0.95, 0.97];
        const N: f64 = 8.0;
        let b = Self::bucket(m);
        (self.ok[b] + N * PRIOR[b]) / (self.ok[b] + self.bad[b] + N)
    }

    /// The copy length (`<= cap`) that maximises the expected tokens per verify cost, where a window of `k` drafts
    /// costs `1 + row_cost * k` (relative to a single-token step); 0 when no length beats `1 + min_gain`.
    pub fn choose_len(&self, m: &CopyMatch, cap: usize, row_cost: f64, min_gain: f64) -> usize {
        self.choose_len_min(m, cap, row_cost, min_gain, 0.0)
    }

    /// [`Self::choose_len`], but 0 while the estimated acceptance of copies like `m` is below `q_min` (where a copy
    /// replaces a neural drafter that is better on this kind of text).
    pub fn choose_len_min(
        &self,
        m: &CopyMatch,
        cap: usize,
        row_cost: f64,
        min_gain: f64,
        q_min: f64,
    ) -> usize {
        let q = self.accept_rate(m).clamp(0.0, 0.9999);
        if q < q_min {
            return 0;
        }
        let mut best = 1.0 + min_gain;
        let mut best_k = 0usize;
        let mut e = 1.0;
        let mut qi = 1.0;
        for k in 1..=cap {
            qi *= q;
            e += qi;
            let r = e / (1.0 + row_cost * k as f64);
            if r > best {
                best = r;
                best_k = k;
            }
        }
        best_k
    }

    /// Records a verified copy from `m` of `proposed` tokens of which `accepted` were kept (`chain`: it followed
    /// the neural drafter's proposals in the same window).  Only a mostly-accepted copy leaves a resume point.
    pub fn observe(&mut self, m: &CopyMatch, proposed: usize, accepted: usize, chain: bool) {
        if proposed == 0 {
            return;
        }
        let accepted = accepted.min(proposed);
        let b = Self::bucket(m);
        const DECAY: f64 = 0.98;
        self.ok[b] = DECAY * self.ok[b] + accepted as f64;
        self.bad[b] = DECAY * self.bad[b] + if accepted < proposed { 1.0 } else { 0.0 };
        self.resume = if 2 * accepted < proposed {
            usize::MAX
        } else {
            m.start + accepted
        };
        let s = &mut self.stats;
        if chain {
            s.chain_windows += 1;
            s.chain_proposed += proposed as u64;
            s.chain_accepted += accepted as u64;
        } else {
            s.windows += 1;
            s.proposed += proposed as u64;
            s.accepted += accepted as u64;
        }
    }

    /// Takes the running counts (for a per-request log line).
    pub fn take_stats(&mut self) -> CopyStats {
        std::mem::take(&mut self.stats)
    }
}

/// `TALUS_COPY` (default on): `0` turns the copy proposer off everywhere.
pub fn copy_enabled() -> bool {
    !matches!(
        std::env::var("TALUS_COPY").ok().as_deref().map(str::trim),
        Some("0") | Some("off") | Some("false")
    )
}

/// `TALUS_COPY_<name>` as a number, when set.
pub fn copy_env_f64(name: &str) -> Option<f64> {
    std::env::var(format!("TALUS_COPY_{name}"))
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
}

/// [`CopyCfg::default`] with any `TALUS_COPY_MIN`, `TALUS_COPY_SMIN`, `TALUS_COPY_RESUME`, `TALUS_COPY_MAXMATCH`
/// overrides.
pub fn copy_cfg_from_env() -> CopyCfg {
    let mut c = CopyCfg::default();
    let get = |n: &str| copy_env_f64(n).map(|v| v.max(0.0) as usize);
    if let Some(v) = get("MIN") {
        c.min_match = v.max(COPY_KEY);
    }
    if let Some(v) = get("SMIN") {
        c.sampled_min_match = v.max(COPY_KEY);
    }
    if let Some(v) = get("RESUME") {
        c.resume_min_match = v.max(COPY_KEY);
    }
    if let Some(v) = get("MAXMATCH") {
        c.max_match = v.max(COPY_KEY);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(seq: &[i32]) -> CopyProposer {
        let mut c = CopyProposer::default();
        c.extend(seq);
        c
    }

    #[test]
    fn copies_after_a_long_match() {
        // a 20-token "file", then the output re-emits its first 14 tokens
        let file: Vec<i32> = (100..120).collect();
        let mut c = CopyProposer::default();
        c.extend(&[1, 2, 3]);
        c.extend(&file);
        c.extend(&[7, 8]);
        c.begin_request();
        let mut out = Vec::new();
        // too short a match: nothing
        c.extend(&file[..8]);
        assert!(c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .is_none());
        c.extend(&file[8..14]);
        let m = c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .unwrap();
        assert_eq!(m.len, 14);
        assert_eq!(
            out,
            (114..120)
                .chain(std::iter::once(7))
                .chain(std::iter::once(8))
                .collect::<Vec<_>>()
        );
        // with uncommitted drafts in the tail, the copy continues after them
        let m2 = c
            .propose(&[114, 115], 3, false, false, |_| false, &mut out)
            .unwrap();
        assert_eq!(m2.len, 16);
        assert_eq!(out, vec![116, 117, 118]);
    }

    #[test]
    fn request_start_bounds_the_match() {
        let file: Vec<i32> = (100..140).collect();
        let mut c = mk(&file);
        c.extend(&file[..20]); // the prompt repeats it: still the prompt
        c.begin_request();
        let mut out = Vec::new();
        // nothing generated yet: no match may count the prompt's own tokens
        assert!(c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .is_none());
        c.extend(&file[20..32]);
        assert!(c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .is_some());
        assert_eq!(out[0], 132);
    }

    #[test]
    fn resume_after_an_edit() {
        let file: Vec<i32> = (100..200).collect();
        let mut c = mk(&file);
        c.begin_request();
        let mut out = Vec::new();
        c.extend(&file[..30]);
        let m = c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .unwrap();
        assert_eq!(out[0], 130);
        // the copy is accepted, then the output edits two tokens and rejoins the file 5 tokens later
        c.extend(&out.clone());
        c.observe(&m, 8, 8, false);
        c.extend(&[1, 2]);
        c.extend(&file[40..45]);
        // 5 matching tokens: only the resume rule lets it copy
        let m2 = c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .unwrap();
        assert!(m2.resumed);
        assert_eq!(out[0], 145);
        // a mostly rejected copy drops the resume point
        c.observe(&m2, 8, 1, false);
        assert!(!c.has_resume());
        assert!(c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .is_none());
    }

    #[test]
    fn sampled_copies_only_from_the_prompt_and_need_longer_matches() {
        let file: Vec<i32> = (100..200).collect();
        let mut c = mk(&file);
        c.begin_request();
        let mut out = Vec::new();
        c.extend(&file[..20]);
        assert!(c
            .propose(&[], 8, true, false, |_| false, &mut out)
            .is_none());
        c.extend(&file[20..26]);
        let m = c.propose(&[], 8, true, false, |_| false, &mut out).unwrap();
        assert_eq!(m.len, 26);
        assert_eq!(out[0], 126);
        // generated text repeating itself is not a source for sampled requests
        let rep: Vec<i32> = (500..530).collect();
        c.extend(&rep);
        c.extend(&rep);
        assert!(c
            .propose(&[], 8, true, false, |_| false, &mut out)
            .is_none());
        assert!(c
            .propose(&[], 8, false, false, |_| false, &mut out)
            .is_some());
    }

    #[test]
    fn stops_after_a_stop_token_and_sync_rebuilds() {
        let mut seq: Vec<i32> = (100..120).collect();
        seq.push(2);
        seq.extend(300..310);
        let mut c = mk(&seq);
        c.begin_request();
        c.extend(&seq[..15]);
        let mut out = Vec::new();
        c.propose(&[], 16, false, false, |t| t == 2, &mut out)
            .unwrap();
        assert_eq!(out, vec![115, 116, 117, 118, 119, 2]);
        // sync with a diverged history rebuilds the index
        let mut other = seq.clone();
        other[5] = 9999;
        c.sync(&other);
        assert_eq!(c.tokens(), &other[..]);
        let mut d = mk(&other);
        d.req_start = c.req_start;
        assert_eq!(
            c.find(&[], 3, false, usize::MAX),
            d.find(&[], 3, false, usize::MAX)
        );
    }

    #[test]
    fn choose_len_prefers_full_windows_when_acceptance_is_high() {
        let c = CopyProposer::default();
        let m = CopyMatch {
            start: 0,
            len: 40,
            resumed: false,
        };
        assert_eq!(c.choose_len(&m, 7, 0.08, 0.03), 7);
        // expensive rows and a weak estimate: shorter
        let mut d = CopyProposer::default();
        for _ in 0..50 {
            d.observe(&m, 7, 1, false);
        }
        assert!(d.choose_len(&m, 7, 0.3, 0.03) < 3);
    }
}
