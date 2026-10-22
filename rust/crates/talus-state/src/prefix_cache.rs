//! Implementation of [`talus_core_api::PrefixCache`] for [`PromptCache`] and [`TalusPrefixCache`].

use talus_core_api::{CacheHandle, PrefixCache, SnapReason, Snapshot};

use crate::pcache::{PromptCache, DEFAULT_MAX_MSG, DEFAULT_MAX_SNAPS};

/// Compute a deterministic 64-bit FNV-1a hash of a token slice.
#[inline]
pub fn compute_tokens_hash(tokens: &[u32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &tok in tokens {
        for b in tok.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Compute a deterministic 64-bit FNV-1a hash of an `i32` token slice.
#[inline]
pub fn compute_i32_tokens_hash(tokens: &[i32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &tok in tokens {
        for b in (tok as u32).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Stored metadata entry for [`talus_core_api::PrefixCache`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMetaEntry {
    pub handle: CacheHandle,
    pub at: u32,
    pub snapshot: Snapshot,
    pub reason: SnapReason,
    pub tokens: Option<Vec<u32>>,
}

#[inline]
fn i32_slice_matches_u32(a: &[i32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(&x, &y)| (x as u32) == y)
}

impl PrefixCache for PromptCache {
    fn longest(&self, ids: &[u32]) -> Option<(u32, CacheHandle)> {
        if ids.is_empty() {
            return None;
        }

        let mut best_len: usize = 0;
        let mut best_handle: Option<CacheHandle> = None;

        // 1. Check `snaps` pool (tier tag 1 << 60).
        for (idx, s) in self.snaps.iter().enumerate() {
            let n = s.tokens.len();
            if n <= ids.len() && n > best_len && i32_slice_matches_u32(&s.tokens, &ids[..n]) {
                best_len = n;
                best_handle = Some(CacheHandle((1u64 << 60) | (idx as u64)));
            }
        }

        // 2. Check `msg` pool (tier tag 2 << 60).
        for (idx, s) in self.msg.iter().enumerate() {
            let n = s.tokens.len();
            if n <= ids.len() && n > best_len && i32_slice_matches_u32(&s.tokens, &ids[..n]) {
                best_len = n;
                best_handle = Some(CacheHandle((2u64 << 60) | (idx as u64)));
            }
        }

        // 3. Check `stashes` pool (tier tag 3 << 60) if no longer snap/msg was found.
        if best_handle.is_none() {
            for (idx, st) in self.stashes.iter().enumerate() {
                let n = st.tokens.len();
                if n <= ids.len() && n > best_len && i32_slice_matches_u32(&st.tokens, &ids[..n]) {
                    best_len = n;
                    best_handle = Some(CacheHandle((3u64 << 60) | (idx as u64)));
                }
            }
        }

        // 4. Check metadata entries registered via `PrefixCache::put`.
        for entry in &self.meta_entries {
            let n = entry.at as usize;
            if n == 0 || n > ids.len() || n <= best_len {
                continue;
            }
            let matches = if let Some(ref toks) = entry.tokens {
                toks.len() == n && toks.as_slice() == &ids[..n]
            } else {
                compute_tokens_hash(&ids[..n]) == entry.snapshot.tokens_hash
                    || entry.snapshot.tokens_hash == 0
            };
            if matches {
                best_len = n;
                best_handle = Some(entry.handle);
            }
        }

        best_handle.map(|h| (best_len as u32, h))
    }

    fn put(&mut self, at: u32, snap: Snapshot, why: SnapReason) {
        if at == 0 {
            return;
        }
        let is_msg = matches!(why, SnapReason::MessageStart);
        let cap = if is_msg { self.max_msg } else { self.max_snaps };
        if cap == 0 {
            return;
        }

        if let Some(existing) = self.meta_entries.iter_mut().find(|e| {
            matches!(e.reason, SnapReason::MessageStart) == is_msg
                && e.at == at
                && e.snapshot.tokens_hash == snap.tokens_hash
        }) {
            existing.snapshot = snap;
            existing.reason = why;
            return;
        }

        let tier_count = self
            .meta_entries
            .iter()
            .filter(|e| matches!(e.reason, SnapReason::MessageStart) == is_msg)
            .count();
        if tier_count >= cap {
            if let Some(pos) = self
                .meta_entries
                .iter()
                .position(|e| matches!(e.reason, SnapReason::MessageStart) == is_msg)
            {
                self.meta_entries.remove(pos);
            }
        }

        let handle = CacheHandle(self.next_handle_id);
        self.next_handle_id += 1;
        self.meta_entries.push(PrefixMetaEntry {
            handle,
            at,
            snapshot: snap,
            reason: why,
            tokens: None,
        });
    }
}

/// Standalone metadata + token prefix index implementing [`talus_core_api::PrefixCache`].
#[derive(Debug, Clone)]
pub struct TalusPrefixCache {
    msg_entries: Vec<PrefixMetaEntry>,
    snap_entries: Vec<PrefixMetaEntry>,
    max_msg: usize,
    max_snaps: usize,
    next_id: u64,
}

impl Default for TalusPrefixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl TalusPrefixCache {
    /// Create a new prefix cache with default slot limits (`max_snaps = 8`, `max_msg = 4`).
    pub fn new() -> Self {
        Self::with_slots(DEFAULT_MAX_SNAPS, DEFAULT_MAX_MSG)
    }

    /// Create a new prefix cache with custom slot limits.
    pub fn with_slots(max_snaps: usize, max_msg: usize) -> Self {
        Self {
            msg_entries: Vec::new(),
            snap_entries: Vec::new(),
            max_msg,
            max_snaps,
            next_id: 1,
        }
    }

    /// Insert a snapshot with explicit token IDs and return its [`CacheHandle`].
    pub fn put_with_tokens(
        &mut self,
        tokens: &[u32],
        mut snap: Snapshot,
        why: SnapReason,
    ) -> Option<CacheHandle> {
        if tokens.is_empty() {
            return None;
        }
        let at = tokens.len() as u32;
        snap.pos = at;
        if snap.tokens_hash == 0 {
            snap.tokens_hash = compute_tokens_hash(tokens);
        }
        let is_msg = matches!(why, SnapReason::MessageStart);
        let cap = if is_msg { self.max_msg } else { self.max_snaps };
        if cap == 0 {
            return None;
        }
        let pool = if is_msg {
            &mut self.msg_entries
        } else {
            &mut self.snap_entries
        };

        if let Some(existing) = pool
            .iter_mut()
            .find(|e| e.tokens.as_deref() == Some(tokens))
        {
            existing.at = at;
            existing.snapshot = snap;
            existing.reason = why;
            return Some(existing.handle);
        }

        if pool.len() >= cap {
            pool.remove(0);
        }
        let handle = CacheHandle(self.next_id);
        self.next_id += 1;
        pool.push(PrefixMetaEntry {
            handle,
            at,
            snapshot: snap,
            reason: why,
            tokens: Some(tokens.to_vec()),
        });
        Some(handle)
    }

    /// Look up a stored entry by its [`CacheHandle`].
    pub fn get(&self, handle: CacheHandle) -> Option<&PrefixMetaEntry> {
        self.snap_entries
            .iter()
            .chain(self.msg_entries.iter())
            .find(|e| e.handle == handle)
    }

    /// Total number of cached entries across both pools.
    pub fn len(&self) -> usize {
        self.snap_entries.len() + self.msg_entries.len()
    }

    /// True when both pools are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PrefixCache for TalusPrefixCache {
    fn longest(&self, ids: &[u32]) -> Option<(u32, CacheHandle)> {
        if ids.is_empty() {
            return None;
        }
        let mut best_len: usize = 0;
        let mut best_handle: Option<CacheHandle> = None;

        for entry in self.snap_entries.iter().chain(self.msg_entries.iter()) {
            let n = entry.at as usize;
            if n == 0 || n > ids.len() || n <= best_len {
                continue;
            }
            let matched = if let Some(ref toks) = entry.tokens {
                toks.len() == n && toks.as_slice() == &ids[..n]
            } else {
                compute_tokens_hash(&ids[..n]) == entry.snapshot.tokens_hash
            };
            if matched {
                best_len = n;
                best_handle = Some(entry.handle);
            }
        }

        best_handle.map(|h| (best_len as u32, h))
    }

    fn put(&mut self, at: u32, snap: Snapshot, why: SnapReason) {
        if at == 0 {
            return;
        }
        let is_msg = matches!(why, SnapReason::MessageStart);
        let cap = if is_msg { self.max_msg } else { self.max_snaps };
        if cap == 0 {
            return;
        }
        let pool = if is_msg {
            &mut self.msg_entries
        } else {
            &mut self.snap_entries
        };

        if let Some(existing) = pool
            .iter_mut()
            .find(|e| e.at == at && e.snapshot.tokens_hash == snap.tokens_hash)
        {
            existing.snapshot = snap;
            existing.reason = why;
            return;
        }

        if pool.len() >= cap {
            pool.remove(0);
        }
        let handle = CacheHandle(self.next_id);
        self.next_id += 1;
        pool.push(PrefixMetaEntry {
            handle,
            at,
            snapshot: snap,
            reason: why,
            tokens: None,
        });
    }
}
