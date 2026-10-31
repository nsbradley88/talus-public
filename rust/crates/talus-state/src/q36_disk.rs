//! Qwen3.5 / Qwen3.6 (`qwen35`, 27B) adapter for the disk tier.
//!
//! An entry holds what [`Q36PromptCache::restore_with_pf`] needs to resume a conversation: the KV stash of the
//! conversation (`ids`, its prefill-produced length `pf`, the stash bytes exactly as the RAM tier keeps them,
//! DFlash drafter context included) and up to [`MAX_EXPORT_SNAPS`] recurrent-state snapshots at prefixes of it.
//! Importing puts those bytes back into the RAM tier as the same `Q36Stash` / `Q36Snap` entries, and the regular
//! restore path runs unchanged. A restore from disk is therefore the same operation, on the same bytes, as a hit
//! on the RAM tier of a process that never restarted.

use crate::disk::{lcp, tag, DiskEntry, Loaded, MetaReader, MetaWriter, WriteJob};
use hip_sys::hipStream_t;

use crate::q36_cache::{d2h, q36_kv_bytes, q36_state_bytes, Q36LayerStateView, Q36PromptCache};

/// Adapter format (bumped when the meta layout or the meaning of a section changes).
pub const FORMAT: u64 = 1;
/// KV stash section.
pub const TAG_KV: u64 = tag(b"Q36KV");
/// Most snapshots written per entry: the prompt-end snapshot and the message-boundary one before it.
pub const MAX_EXPORT_SNAPS: usize = 2;

/// Section tag of the `i`-th snapshot (`Q36ST0`, `Q36ST1`, ...).
pub const fn tag_state(i: usize) -> u64 {
    tag(b"Q36ST") | (((b'0' + i as u8) as u64) << 40)
}

/// Parsed entry meta.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Q36DiskMeta {
    pub state_bytes: u64,
    pub kv_bytes: u64,
    pub pf: u64,
    /// `(prefix length, message-boundary flag)` per snapshot, deepest first.
    pub snaps: Vec<(u64, bool)>,
}

impl Q36DiskMeta {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = MetaWriter::default();
        w.u64(FORMAT)
            .u64(self.state_bytes)
            .u64(self.kv_bytes)
            .u64(self.pf)
            .u64(self.snaps.len() as u64);
        for (n, m) in &self.snaps {
            w.u64(*n).u64(*m as u64);
        }
        w.finish()
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        let mut r = MetaReader(b);
        if r.u64()? != FORMAT {
            return None;
        }
        let state_bytes = r.u64()?;
        let kv_bytes = r.u64()?;
        let pf = r.u64()?;
        let ns = r.u64()?;
        if ns == 0 || ns as usize > MAX_EXPORT_SNAPS {
            return None;
        }
        let mut snaps = Vec::with_capacity(ns as usize);
        for _ in 0..ns {
            snaps.push((r.u64()?, r.u64()? != 0));
        }
        if !r.0.is_empty() {
            return None;
        }
        Some(Self {
            state_bytes,
            kv_bytes,
            pf,
            snaps,
        })
    }
}

/// What [`export`] would write for `live`: the stash index, the snapshot indices (deepest first) and the bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportPlan {
    pub stash: usize,
    pub snaps: Vec<usize>,
    pub bytes: u64,
    pub depth: usize,
}

/// Pick the RAM-tier stash that covers `live` and the deepest snapshots usable with it (prefill-produced, i.e.
/// within the stash's `pf`). `None` when there is nothing worth writing (deepest snapshot below `min_tokens`).
pub fn plan_export(pc: &Q36PromptCache, live: &[i32], min_tokens: usize) -> Option<ExportPlan> {
    if live.is_empty() || pc.snap_bytes == 0 {
        return None;
    }
    // Prefer the stash whose ids are exactly `live`, else the shortest one that covers it.
    let stash = pc
        .stashes
        .iter()
        .enumerate()
        .filter(|(_, s)| s.ids.len() >= live.len() && lcp(&s.ids, live) == live.len())
        .min_by_key(|(_, s)| s.ids.len())
        .map(|(i, _)| i)?;
    let st = &pc.stashes[stash];
    let mut snaps: Vec<usize> = pc
        .snaps
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            let n = s.ids.len();
            n > 0 && n <= st.pf && n <= st.ids.len() && lcp(&s.ids, &st.ids) == n
        })
        .map(|(i, _)| i)
        .collect();
    snaps.sort_by_key(|&i| std::cmp::Reverse(pc.snaps[i].ids.len()));
    snaps.dedup_by_key(|i| pc.snaps[*i].ids.len());
    snaps.truncate(MAX_EXPORT_SNAPS);
    let depth = pc.snaps[*snaps.first()?].ids.len();
    if depth < min_tokens.max(1) {
        return None;
    }
    let bytes = st.bytes as u64 + (pc.snap_bytes * snaps.len()) as u64;
    Some(ExportPlan {
        stash,
        snaps,
        bytes,
        depth,
    })
}

fn staging(len: usize) -> Option<Vec<u8>> {
    let mut v: Vec<u8> = Vec::new();
    v.try_reserve_exact(len).ok()?;
    // SAFETY: the caller fills every byte (d2h) before the buffer is read, and drops it on failure.
    #[allow(clippy::uninit_vec)]
    unsafe {
        v.set_len(len)
    };
    Some(v)
}

/// Device-to-host copy in 64 MiB chunks; false when a copy fails or `abort()` turns true between chunks.
fn d2h_chunked(host: &mut [u8], src: *const u8, st: hipStream_t, abort: &dyn Fn() -> bool) -> bool {
    const CHUNK: usize = 64 << 20;
    for (k, c) in host.chunks_mut(CHUNK).enumerate() {
        if abort() {
            return false;
        }
        // SAFETY: `src` points at a device buffer of at least `host.len()` bytes; the offset stays inside it.
        if !d2h(c, unsafe { src.add(k * CHUNK) }, st) {
            return false;
        }
    }
    true
}

/// Copy the planned stash and snapshots off the GPU into a [`WriteJob`]. `None` when a copy fails or `abort()`
/// turns true (checked between 64 MiB chunks: a request that arrives meanwhile is not held up by the staging).
pub fn export(
    pc: &Q36PromptCache,
    eng: &Q36LayerStateView,
    plan: &ExportPlan,
    abort: &dyn Fn() -> bool,
) -> Option<WriteJob> {
    let st = pc.stashes.get(plan.stash)?;
    let mut kv = staging(st.bytes)?;
    if !d2h_chunked(&mut kv, st.buf, eng.st, abort) {
        return None;
    }
    let mut sections = vec![(TAG_KV, kv)];
    let mut snaps = Vec::new();
    for (k, &i) in plan.snaps.iter().enumerate() {
        let s = pc.snaps.get(i)?;
        let mut b = staging(pc.snap_bytes)?;
        if !d2h_chunked(&mut b, s.buf, eng.st, abort) {
            return None;
        }
        sections.push((tag_state(k), b));
        snaps.push((s.ids.len() as u64, s.msg));
    }
    let meta = Q36DiskMeta {
        state_bytes: pc.snap_bytes as u64,
        kv_bytes: st.bytes as u64,
        pf: st.pf as u64,
        snaps,
    };
    Some(WriteJob {
        tokens: st.ids.clone(),
        meta: meta.encode(),
        sections,
        external: None,
        depth: plan.depth as u64,
        supersede: true,
    })
}

/// How many tokens of `ids` the entry can restore on this engine (0 = unusable): the deepest snapshot `n` with
/// `n < ids.len()`, a prefix of `ids`, and (`pf_only`) within the stash's prefill-produced part. Entries whose
/// sizes do not match this engine's geometry, or whose stash exceeds the RAM stash budget, are unusable.
pub fn usable(
    e: &DiskEntry,
    ids: &[i32],
    pf_only: bool,
    pc: &Q36PromptCache,
    eng: &Q36LayerStateView,
) -> usize {
    let Some(m) = Q36DiskMeta::decode(&e.meta) else {
        return 0;
    };
    if pc.max_snaps == 0
        || m.state_bytes != q36_state_bytes(eng) as u64
        || m.kv_bytes != q36_kv_bytes(eng, e.tokens.len() as i32, 0) as u64
        || m.kv_bytes > pc.stash_budget as u64
    {
        return 0;
    }
    let common = lcp(&e.tokens, ids);
    m.snaps
        .iter()
        .map(|&(n, _)| n as usize)
        .filter(|&n| n < ids.len() && n <= common && (!pf_only || n as u64 <= m.pf))
        .max()
        .unwrap_or(0)
}

/// Put a loaded entry back into the RAM tier. Returns false (and leaves no partial snapshot set behind that
/// could be restored without its KV) if any size check or copy fails.
pub fn import(pc: &mut Q36PromptCache, eng: &Q36LayerStateView, l: &Loaded) -> bool {
    let Some(m) = Q36DiskMeta::decode(&l.meta) else {
        return false;
    };
    let Some(kv) = l.section(TAG_KV) else {
        return false;
    };
    if kv.len() as u64 != m.kv_bytes {
        return false;
    }
    if !pc.insert_stash_host(eng, &l.tokens, m.pf as usize, kv) {
        return false;
    }
    let mut any = false;
    for (k, &(n, msg)) in m.snaps.iter().enumerate() {
        let n = n as usize;
        let Some(b) = l.section(tag_state(k)) else {
            continue;
        };
        if n == 0 || n > l.tokens.len() || b.len() as u64 != m.state_bytes {
            continue;
        }
        any |= pc.insert_snap_host(eng, &l.tokens[..n], msg, b);
    }
    any
}
