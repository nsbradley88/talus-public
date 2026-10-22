//! Piece 1 (`qwen4exp-p1`): the loan planning (plan_lend, lend_slots, the serve chunk scan).

#![allow(clippy::missing_safety_doc)]

use super::*;

// =====================================================================================================================
// The prompt path's loan planning: plan_lend (generate / the resident RAM lend region) and the serve chunk scan
// =====================================================================================================================

/// `tk_qwen4exp_p1_planner_t`: the cache, the knobs, and the prompt path's pricing (Prefill statics) as callbacks.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TkQwen4ExpP1Planner {
    pub user: *mut c_void,
    pub cache_slots: i64,
    pub cache_bytes: i64,
    pub slot_offsets: *const u64,
    pub max_blob: u64,
    pub prefill_auto: i32,
    pub prefill_chunk: i64,
    pub prefill_auto_max: i64,
    pub prefill_auto_cap: i64,
    pub max_context: i64,
    pub auto_ceiling: i64,
    pub lend_pct: i64,
    pub bytes_needed: Option<unsafe extern "C" fn(user: *mut c_void, chunk: i64) -> u64>,
    pub bytes_needed_no_ring: Option<unsafe extern "C" fn(user: *mut c_void, chunk: i64) -> u64>,
    pub set_ring_budget: Option<unsafe extern "C" fn(user: *mut c_void, room: i64, small: i64)>,
    pub ring_cap_for: Option<unsafe extern "C" fn(user: *mut c_void, chunk: i64) -> i64>,
    pub ring_default_slots: Option<unsafe extern "C" fn(user: *mut c_void) -> i64>,
    pub ring_bytes_enabled: Option<unsafe extern "C" fn(user: *mut c_void) -> c_int>,
}

/// The prompt path's pricing, as the planner sees it (a trait so the arithmetic is unit-testable).
pub trait PrefillPricing {
    fn bytes_needed(&mut self, chunk: i64) -> u64;
    fn bytes_needed_no_ring(&mut self, chunk: i64) -> u64;
    fn set_ring_budget(&mut self, room: i64, small: i64);
    fn ring_cap_for(&mut self, chunk: i64) -> i64;
    fn ring_default_slots(&mut self) -> i64;
    fn ring_bytes_enabled(&mut self) -> bool;
}

pub(super) struct CPricing(pub(super) TkQwen4ExpP1Planner);

impl PrefillPricing for CPricing {
    fn bytes_needed(&mut self, c: i64) -> u64 {
        self.0
            .bytes_needed
            .map(|f| unsafe { f(self.0.user, c) })
            .unwrap_or(0)
    }
    fn bytes_needed_no_ring(&mut self, c: i64) -> u64 {
        self.0
            .bytes_needed_no_ring
            .map(|f| unsafe { f(self.0.user, c) })
            .unwrap_or(0)
    }
    fn set_ring_budget(&mut self, room: i64, small: i64) {
        if let Some(f) = self.0.set_ring_budget {
            unsafe { f(self.0.user, room, small) }
        }
    }
    fn ring_cap_for(&mut self, c: i64) -> i64 {
        self.0
            .ring_cap_for
            .map(|f| unsafe { f(self.0.user, c) })
            .unwrap_or(0)
    }
    fn ring_default_slots(&mut self) -> i64 {
        self.0
            .ring_default_slots
            .map(|f| unsafe { f(self.0.user) })
            .unwrap_or(0)
    }
    fn ring_bytes_enabled(&mut self) -> bool {
        self.0
            .ring_bytes_enabled
            .map(|f| unsafe { f(self.0.user) } != 0)
            .unwrap_or(false)
    }
}

/// The planner's numbers (the cache and the knobs).
#[derive(Clone, Debug)]
pub struct LendPlanner {
    pub slots: i64,
    pub bytes: i64,
    pub offsets: Option<Vec<u64>>,
    pub max_blob: u64,
    pub prefill_auto: bool,
    pub prefill_chunk: i64,
    pub prefill_auto_max: i64,
    pub prefill_auto_cap: i64,
    pub max_context: i64,
    pub auto_ceiling: i64,
    pub lend_pct: i64,
}

/// 0.1.39's chunk list (`kAutoChunks`).
pub(super) const AUTO_CHUNKS: [i64; 10] =
    [32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256];
/// `kRingMin`: at or below it the ring is STAGE (streaming off).
pub(super) const RING_MIN: i64 = 16;

/// `biggest_chunk`: the largest chunk on the 256-token grid up to `ceiling` that `ok` accepts (`ok` monotone).
pub fn biggest_chunk(ceiling: i64, mut ok: impl FnMut(i64) -> bool) -> i64 {
    if ceiling < 256 {
        return 0;
    }
    let (mut lo, mut hi, mut best) = (1i64, ceiling / 256, 0i64);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if ok(mid * 256) {
            best = mid * 256;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    best
}

impl LendPlanner {
    pub(super) fn from_c(c: &TkQwen4ExpP1Planner) -> Self {
        Self {
            slots: c.cache_slots,
            bytes: c.cache_bytes,
            offsets: if c.slot_offsets.is_null() || c.cache_slots <= 0 {
                None
            } else {
                Some(
                    unsafe {
                        std::slice::from_raw_parts(c.slot_offsets, c.cache_slots as usize + 1)
                    }
                    .to_vec(),
                )
            },
            max_blob: c.max_blob,
            prefill_auto: c.prefill_auto != 0,
            prefill_chunk: c.prefill_chunk,
            prefill_auto_max: c.prefill_auto_max,
            prefill_auto_cap: c.prefill_auto_cap,
            max_context: c.max_context,
            auto_ceiling: c.auto_ceiling,
            lend_pct: c.lend_pct,
        }
    }

    /// `slots_from_bytes` / `part_slots`: the tail slots `need` bytes take.
    pub fn slots_for_bytes(&self, need: u64) -> i64 {
        split_lend_slots(
            self.slots,
            self.bytes,
            self.offsets.as_deref(),
            self.max_blob,
            need,
        )
    }

    /// `bytes_from_slots` (k <= 0: nothing) == `part_bytes(slots - k)`.
    pub fn bytes_from_slots(&self, k: i64) -> u64 {
        if k <= 0 {
            return 0;
        }
        match &self.offsets {
            Some(off) => (self.bytes - off[(self.slots - k) as usize] as i64) as u64,
            None => k as u64 * self.max_blob,
        }
    }

    /// `lend_slots`: the tail slots a `c`-token chunk's buffers take.
    pub fn lend_slots(&self, p: &mut dyn PrefillPricing, c: i64) -> i64 {
        let need = p.bytes_needed(c);
        self.slots_for_bytes(need)
    }

    fn skip_auto(&self, c: i64) -> bool {
        (c > 8192 && (c > self.prefill_auto_max || c > self.max_context))
            || (self.prefill_auto_cap > 0 && c > self.prefill_auto_cap)
    }

    /// the fit test with the auto cap (`k + 128 <= slots && k * 100 <= pct * slots`)
    fn fits_capped(&self, k: i64) -> bool {
        k + 128 <= self.slots && k * 100 <= self.lend_pct * self.slots
    }

    /// 0.1.39's list (`old_rule` / `old_pick` for one part): the first listed chunk that fits under the cap.
    fn old_rule(&self, p: &mut dyn PrefillPricing, serve: bool) -> i64 {
        p.set_ring_budget(0, 0);
        for &c in AUTO_CHUNKS.iter() {
            if self.skip_auto(c) {
                continue;
            }
            let k = self.lend_slots(p, c);
            // the serve form (fits_one) also refuses a loan of no slots
            if serve && k <= 0 {
                continue;
            }
            if self.fits_capped(k) {
                return c;
            }
        }
        0
    }

    /// The ring room a `t`-token chunk leaves (capped at `cap`), from the lendable budget's bytes.
    fn room_of(&self, p: &mut dyn PrefillPricing, t: i64, cap: i64, clamp_budget: bool) -> i64 {
        let mut budget = (self.slots - 128).min(self.lend_pct * self.slots / 100);
        if clamp_budget {
            budget = budget.max(0);
        }
        let avail = self.bytes_from_slots(budget);
        let nonring = p.bytes_needed_no_ring(t);
        let blob = self.max_blob.max(1);
        (((avail - avail.min(nonring)) / blob) as i64).min(cap)
    }

    /// generate.cpp `plan_lend`: the chunk (in/out) and the slots it borrows (0: no loan).
    pub fn plan_lend(&self, p: &mut dyn PrefillPricing, chunk: &mut i64) -> i64 {
        if self.prefill_auto && !p.ring_bytes_enabled() {
            let c = self.old_rule(p, false);
            if c > 0 {
                *chunk = c;
                return self.lend_slots(p, c);
            }
            return 0;
        }
        if self.prefill_auto {
            let small = self.old_rule(p, false);
            let ring_max = p.ring_cap_for(small);
            let scan = |p: &mut dyn PrefillPricing, floor: i64| -> i64 {
                biggest_chunk(self.auto_ceiling, |t| {
                    let room = self.room_of(p, t, ring_max, false);
                    if room < floor {
                        return false;
                    }
                    p.set_ring_budget(room, 0);
                    let k = self.lend_slots(p, t);
                    self.fits_capped(k)
                })
            };
            let mut c = scan(p, ring_max);
            if c == 0 && ring_max < p.ring_default_slots() {
                c = scan(p, RING_MIN);
            }
            if small >= c {
                p.set_ring_budget(0, 0);
                if small > 0 {
                    *chunk = small;
                    return self.lend_slots(p, small);
                }
                return 0;
            }
            let room = self.room_of(p, c, ring_max, false);
            p.set_ring_budget(room, small);
            *chunk = c;
            return self.lend_slots(p, c);
        }
        let mut c = *chunk;
        while c >= 256 {
            let k = self.lend_slots(p, c);
            if k + 128 <= self.slots {
                *chunk = c;
                return k;
            }
            c /= 2;
        }
        0
    }

    /// `fits_one` for the one serve part.
    fn fits_one(&self, p: &mut dyn PrefillPricing, c: i64, cap: bool) -> bool {
        let k = self.lend_slots(p, c);
        if k <= 0 || k + 128 > self.slots {
            return false;
        }
        !(cap && k * 100 > self.lend_pct * self.slots)
    }

    /// The serve start's chunk scan for the one prompt path (generate.cpp `scan(nullptr)` and the ring budget it
    /// leaves): returns `(chunk, first)`, `first` = the first lendable slot (-1 when no chunk).
    pub fn serve_scan(&self, p: &mut dyn PrefillPricing) -> (i64, i64) {
        let old_pick = |p: &mut dyn PrefillPricing| self.old_rule(p, true);
        let chunk = if !self.prefill_auto {
            let mut c = self.prefill_chunk;
            let mut got = 0;
            while c >= 256 {
                if self.fits_one(p, c, false) {
                    got = c;
                    break;
                }
                c /= 2;
            }
            got
        } else if !p.ring_bytes_enabled() {
            old_pick(p)
        } else {
            let small = old_pick(p);
            let cap = p.ring_cap_for(small);
            let probe = |p: &mut dyn PrefillPricing, floor: i64| -> i64 {
                biggest_chunk(self.auto_ceiling, |t| {
                    let room = self.room_of(p, t, cap, true);
                    if room < floor {
                        return false;
                    }
                    p.set_ring_budget(room, 0);
                    self.fits_one(p, t, true)
                })
            };
            let mut c = probe(p, cap);
            if c == 0 && cap < p.ring_default_slots() {
                c = probe(p, RING_MIN);
            }
            c.max(small)
        };
        if chunk > 0 && self.prefill_auto && p.ring_bytes_enabled() {
            let small = old_pick(p);
            let room = if small >= chunk {
                0
            } else {
                let cap = p.ring_cap_for(small);
                self.room_of(p, chunk, cap, true)
            };
            p.set_ring_budget(room, small);
        }
        if chunk > 0 {
            let first = self.slots - self.lend_slots(p, chunk);
            (chunk, first)
        } else {
            (0, -1)
        }
    }
}
