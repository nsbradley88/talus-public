//! Qwen3.5 / Qwen3.6 (`qwen35`, 27B) adapter for the disk tier.
//!
//! An entry holds what [`Q36PromptCache::restore_with_pf`] needs to resume a conversation: the KV stash of the
//! conversation (`ids`, its prefill-produced length `pf`, the stash bytes exactly as the RAM tier keeps them,
//! DFlash drafter context included) and up to [`MAX_EXPORT_SNAPS`] recurrent-state snapshots at prefixes of it.
//! Importing puts those bytes back into the RAM tier as the same `Q36Stash` / `Q36Snap` entries, and the regular
//! restore path runs unchanged. A restore from disk is therefore the same operation, on the same bytes, as a hit
//! on the RAM tier of a process that never restarted.

use crate::disk::{
    lcp, tag, DiskCache, DiskEntry, DiskError, Loaded, MetaReader, MetaWriter, WriteJob,
};
use hip_sys::hipStream_t;

use crate::q36_cache::{
    d2h, h2d, q36_kv_bytes, q36_kv_layered_begin, q36_kv_piece_load, q36_kv_piece_save,
    q36_kv_pieces, q36_state_bytes, Q36LayerStateView, Q36PromptCache,
};
use core::ffi::c_void;
use hip_sys::{hipFree, hipMalloc};

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

/// : [`export`] streamed straight to disk ([`DiskCache::write_streamed`]): the stash and snapshots are copied off
/// the GPU one [`crate::disk::STREAM_CHUNK`] at a time, so host memory stays at one chunk instead of the whole entry
/// (on a 64 GB machine at 128K context the 6.8 GiB copy could not keep the 8 GiB floor). Same entry (tokens, meta,
/// sections, depth) as [`export`] + a background write. Runs on the engine thread; `abort()` stops it between
/// chunks.
pub fn export_streamed(
    pc: &Q36PromptCache,
    eng: &Q36LayerStateView,
    plan: &ExportPlan,
    disk: &DiskCache,
    abort: &dyn Fn() -> bool,
) -> Result<DiskEntry, DiskError> {
    let st = pc
        .stashes
        .get(plan.stash)
        .ok_or_else(|| DiskError::Io("stash gone".into()))?;
    let mut srcs: Vec<*const u8> = vec![st.buf];
    let mut sections = vec![(TAG_KV, st.bytes as u64)];
    let mut snaps = Vec::new();
    for (k, &i) in plan.snaps.iter().enumerate() {
        let s = pc
            .snaps
            .get(i)
            .ok_or_else(|| DiskError::Io("snapshot gone".into()))?;
        srcs.push(s.buf);
        sections.push((tag_state(k), pc.snap_bytes as u64));
        snaps.push((s.ids.len() as u64, s.msg));
    }
    let meta = Q36DiskMeta {
        state_bytes: pc.snap_bytes as u64,
        kv_bytes: st.bytes as u64,
        pf: st.pf as u64,
        snaps,
    };
    let mut fill = |si: usize, off: u64, buf: &mut [u8]| -> bool {
        // SAFETY: `off + buf.len()` stays inside section `si`, whose device buffer is `srcs[si]` of that size.
        d2h(buf, unsafe { srcs[si].add(off as usize) }, eng.st)
    };
    disk.write_streamed(
        &st.ids,
        &meta.encode(),
        plan.depth as u64,
        true,
        &sections,
        &mut fill,
        abort,
    )
}

/// : [`DiskCache::load`] + [`import`] with the KV stash streamed from the file into its device buffer
/// ([`DiskCache::load_streamed`]): no host copy of the stash. The stash is inserted only after its hash checked
/// out, then the snapshots (small, read whole) as [`import`] inserts them. Returns `(restored, bytes, read ms)`.
pub fn load_import_streamed(
    pc: &mut Q36PromptCache,
    eng: &Q36LayerStateView,
    disk: &DiskCache,
    e: &DiskEntry,
) -> Result<(bool, u64, f64), DiskError> {
    let m = Q36DiskMeta::decode(&e.meta).ok_or_else(|| DiskError::Mismatch("meta".into()))?;
    if m.kv_bytes != q36_kv_bytes(eng, e.tokens.len() as i32, 0) as u64 {
        return Err(DiskError::Mismatch("KV size".into()));
    }
    let mut loaded: Option<Result<Loaded, DiskError>> = None;
    let ok = pc.insert_stash_with(eng, &e.tokens, m.pf as usize, |buf| {
        let mut sink = |off: u64, b: &[u8]| -> bool {
            // SAFETY: the stash buffer holds `kv_bytes`; the streamed section has that length (checked by
            // `usable` and the meta) and `off + b.len()` stays inside it.
            h2d(unsafe { buf.add(off as usize) }, b, eng.st)
        };
        let r = disk.load_streamed(e, TAG_KV, &mut sink);
        let good = r.is_ok();
        loaded = Some(r);
        good
    });
    let l = match loaded {
        Some(Ok(l)) => l,
        Some(Err(err)) => return Err(err),
        None => return Ok((false, 0, 0.0)),
    };
    if !ok {
        return Ok((false, l.bytes, l.ms));
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
    Ok((any, l.bytes, l.ms))
}

/// Device staging buffer freed on drop.
struct DevBuf(*mut u8);

impl DevBuf {
    fn new(bytes: usize) -> Option<Self> {
        let mut raw: *mut c_void = std::ptr::null_mut();
        // SAFETY: plain allocation; freed in Drop.
        unsafe { hipMalloc(&mut raw, bytes.max(1)) }
            .is_success()
            .then(|| Self(raw.cast()))
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // SAFETY: allocated by hipMalloc above.
        unsafe {
            let _ = hipFree(self.0.cast());
        }
    }
}

/// : the snapshots a live-KV entry can carry: prefill-produced prefixes of `live` (within `live_pf`), deepest
/// first, at most [`MAX_EXPORT_SNAPS`]; `None` when the deepest is below `min_tokens`. As [`plan_export`] picks them
/// for a stash that is exactly `live`.
pub fn plan_live(
    pc: &Q36PromptCache,
    live: &[i32],
    live_pf: usize,
    min_tokens: usize,
) -> Option<ExportPlan> {
    if live.is_empty() || pc.snap_bytes == 0 {
        return None;
    }
    let pf = live_pf.min(live.len());
    let mut snaps: Vec<usize> = pc
        .snaps
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            let n = s.ids.len();
            n > 0 && n <= pf && lcp(&s.ids, live) == n
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
    Some(ExportPlan {
        stash: usize::MAX,
        snaps,
        bytes: 0,
        depth,
    })
}

/// : when no RAM-tier stash fits (64 GB at 128K with the int8 KV copy on), write the entry straight from the
/// live KV: the stash layout is gathered one attention layer at a time into a device staging buffer of one piece
/// (~0.4 GiB at 90K tokens) and streamed to disk ([`DiskCache::write_streamed`]). The entry is the one a stash of
/// `live` would have produced (same tokens, meta, sections), so it restores through either path.
pub fn export_streamed_live(
    pc: &Q36PromptCache,
    eng: &Q36LayerStateView,
    live: &[i32],
    live_pf: usize,
    plan: &ExportPlan,
    disk: &DiskCache,
    abort: &dyn Fn() -> bool,
) -> Result<DiskEntry, DiskError> {
    let n = live.len() as i32;
    let kv_bytes = q36_kv_bytes(eng, n, 0);
    let pieces = q36_kv_pieces(eng, n);
    let mut starts = Vec::with_capacity(pieces.len());
    let mut acc = 0usize;
    for (_, b) in &pieces {
        starts.push(acc);
        acc += *b;
    }
    if acc != kv_bytes || pieces.is_empty() {
        return Err(DiskError::Mismatch(
            "KV pieces do not add up to the stash layout".into(),
        ));
    }
    let max_piece = pieces.iter().map(|(_, b)| *b).max().unwrap_or(0);
    if let Some(avail) = crate::disk::mem_available_effective() {
        if avail < (max_piece + crate::disk::STREAM_CHUNK) as u64 + disk.config().mem_floor_bytes {
            return Err(DiskError::Budget(format!(
                "a {} MiB staging piece would cross the memory floor",
                max_piece >> 20
            )));
        }
    }
    let staging = DevBuf::new(max_piece)
        .ok_or_else(|| DiskError::Budget("staging allocation failed".into()))?;
    let mut srcs: Vec<*const u8> = Vec::new();
    let mut sections = vec![(TAG_KV, kv_bytes as u64)];
    let mut snaps = Vec::new();
    for (k, &i) in plan.snaps.iter().enumerate() {
        let s = pc
            .snaps
            .get(i)
            .ok_or_else(|| DiskError::Io("snapshot gone".into()))?;
        srcs.push(s.buf);
        sections.push((tag_state(k), pc.snap_bytes as u64));
        snaps.push((s.ids.len() as u64, s.msg));
    }
    let meta = Q36DiskMeta {
        state_bytes: pc.snap_bytes as u64,
        kv_bytes: kv_bytes as u64,
        pf: live_pf.min(live.len()) as u64,
        snaps,
    };
    let mut cur = usize::MAX;
    let mut fill = |si: usize, off: u64, buf: &mut [u8]| -> bool {
        if si > 0 {
            // SAFETY: inside snapshot si - 1's device buffer of snap_bytes.
            return d2h(buf, unsafe { srcs[si - 1].add(off as usize) }, eng.st);
        }
        let mut done = 0usize;
        while done < buf.len() {
            let pos = off as usize + done;
            let i = starts.partition_point(|&st| st <= pos) - 1;
            if cur != i {
                q36_kv_piece_save(&pieces[i].0, staging.0, n);
                cur = i;
            }
            let within = pos - starts[i];
            let take = (pieces[i].1 - within).min(buf.len() - done);
            // SAFETY: `within + take` stays inside piece i, which fits the staging buffer.
            if !d2h(
                &mut buf[done..done + take],
                unsafe { staging.0.add(within) },
                eng.st,
            ) {
                return false;
            }
            done += take;
        }
        true
    };
    disk.write_streamed(
        live,
        &meta.encode(),
        plan.depth as u64,
        true,
        &sections,
        &mut fill,
        abort,
    )
}

/// : restore an entry straight into the live KV (no RAM-tier stash): the KV section streams from the file through
/// one host chunk into a one-piece device staging buffer and is scattered layer by layer into the KV cells; the
/// snapshots go into the snapshot pool. On success the caller makes the entry's tokens the live sequence (with
/// prefill-produced length `pf`), so the regular restore resumes from a snapshot with no KV copy. Returns
/// `(snapshots inserted, pf, bytes, read ms)`; on an error the KV cells are partly overwritten and the caller must
/// treat the engine as cold.
pub fn load_direct(
    pc: &mut Q36PromptCache,
    eng: &mut Q36LayerStateView,
    disk: &DiskCache,
    e: &DiskEntry,
) -> Result<(bool, usize, u64, f64), DiskError> {
    let m = Q36DiskMeta::decode(&e.meta).ok_or_else(|| DiskError::Mismatch("meta".into()))?;
    let n = e.tokens.len() as i32;
    if m.kv_bytes != q36_kv_bytes(eng, n, 0) as u64 || n > eng.ctx {
        return Err(DiskError::Mismatch("KV size".into()));
    }
    let pieces = q36_kv_pieces(eng, n);
    let max_piece = pieces.iter().map(|(_, b)| *b).max().unwrap_or(0);
    let staging = DevBuf::new(max_piece)
        .ok_or_else(|| DiskError::Budget("staging allocation failed".into()))?;
    q36_kv_layered_begin(eng);
    let mut i = 0usize; // piece being filled
    let mut start = 0u64; // its offset in the section
    let mut ok = true;
    let l = {
        let eng_ref = &mut *eng;
        let mut sink = |off: u64, b: &[u8]| -> bool {
            let mut done = 0usize;
            while done < b.len() {
                if i >= pieces.len() {
                    ok = false;
                    return false;
                }
                let pos = off + done as u64;
                let within = (pos - start) as usize;
                let take = (pieces[i].1 - within).min(b.len() - done);
                // SAFETY: inside piece i, which fits the staging buffer.
                if !h2d(
                    unsafe { staging.0.add(within) },
                    &b[done..done + take],
                    eng_ref.st,
                ) {
                    ok = false;
                    return false;
                }
                done += take;
                if within + take == pieces[i].1 {
                    q36_kv_piece_load(eng_ref, &pieces[i].0, staging.0, n);
                    start += pieces[i].1 as u64;
                    i += 1;
                }
            }
            true
        };
        disk.load_streamed(e, TAG_KV, &mut sink)?
    };
    if !ok || i != pieces.len() {
        return Err(DiskError::Io("KV scatter incomplete".into()));
    }
    let mut any = false;
    for (k, &(sn, msg)) in m.snaps.iter().enumerate() {
        let sn = sn as usize;
        let Some(b) = l.section(tag_state(k)) else {
            continue;
        };
        if sn == 0 || sn > l.tokens.len() || b.len() as u64 != m.state_bytes {
            continue;
        }
        any |= pc.insert_snap_host(eng, &l.tokens[..sn], msg, b);
    }
    Ok((any, m.pf as usize, l.bytes, l.ms))
}
