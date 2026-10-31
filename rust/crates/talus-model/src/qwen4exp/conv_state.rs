//! Piece 5 (`qwen4exp-p5`): Flash-Next (`qwen4exp`) multi-turn conversation state, owned by the Rust host.
//!
//! Rust port of the `--serve` conversation cache (`generate.cpp`: `ConvCheckpoint` chain, `checkpoint_at`,
//! `park_current`, the conversation part of a request's begin, `SAVE` / `RESTORE`) and of the host code behind it
//! (`src/core/conversation_{state,snapshot,memory,file}.cpp`, `conversation_{cache,buffer}.hpp`,
//! `session_save_reclaim.hpp`, `program/conv_cache.hpp`).
//!
//! Rust owns every host structure (the live token list, the checkpoint chain and its LRU clock, the parked
//! conversations with their K/V images, session files) and every decision (resume point, retention, eviction, RAM
//! admission, file format). The C++ side keeps launch-only primitives (`src/program/qwen4exp_p5.cpp` and the marked
//! `qwen4exp-p5` lambdas in `generate.cpp`): a layout descriptor of the session's device buffers, a device/host copy,
//! a device sync, the K/V residency refresh after a restore, `session_zero`, the drafter's K/V rewind, the elastic
//! K/V grow/trim and the control-vector switch. The C++ serve lambdas call the Rust hooks
//! ([`serve`]) where they used to run the C++ conversation code.
//!
//! Single GPU, text only (the Rust serve path refuses layer splits, peers and vision): no stage parts, no images.
//!
//! ## Verification
//! - `cargo test -p talus-model qwen4exp::conv_state` (formats, policies, host round trips);
//! - GPU: multi-turn greedy parity (warm follow-up turns resume at the same `RESUMED`/`cache_n` and produce the
//!   same tokens as `TALUS_FN_RUST=0`), `SAVE` file bytes equal to the C++ engine's, `RESTORE` round trips both ways.

pub mod file;
pub mod serve;

use std::os::raw::c_char;

use crate::qwen4exp::ffi::{Qwen4ExpKernels, TkQwen4ExpConvStateSizes, TkQwen4ExpServeCtx};
use crate::qwen4exp::skeleton::Qwen4ExpModelConfig;

pub use file::{SessionError, SessionFileIdentity, SessionStatus};

// ---------------------------------------------------------------------------------------------------------------
// C ABI: the session layout descriptor and the primitives (`qwen4exp-p5` block of talus_qwen4exp_kernels.h)
// ---------------------------------------------------------------------------------------------------------------

pub const P5_MAX_QSA: usize = 32;

/// `tk_qwen4exp_p5_qsa_t`: one QSA state's buffers and flags.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct P5Qsa {
    /// The authoritative K/V parts: K, V, K scale, V scale, indexer pooled rows (host pools when streamed).
    pub pools: [*mut u8; 5],
    pub idx_tail: *mut u8,
    pub idx_dead: *mut u8,
    pub idx_block_pos: *mut u8,
    pub idx_pooled: *mut u8,
    pub max_cells: i64,
    pub n_pages: i64,
    pub n_slots: i64,
    pub idx_pooled_rows: i64,
    pub kv_mode: i32,
    pub kv_format: i32,
    pub kv_int8: i32,
    pub kv_q4: i32,
    pub kv_hybrid: i32,
    pub kv_rot: i32,
    pub host_present: i32,
    pub map_ok: i32,
    pub ring_ok: i32,
    pub reserved: i32,
}

impl Default for P5Qsa {
    fn default() -> Self {
        Self {
            pools: [std::ptr::null_mut(); 5],
            idx_tail: std::ptr::null_mut(),
            idx_dead: std::ptr::null_mut(),
            idx_block_pos: std::ptr::null_mut(),
            idx_pooled: std::ptr::null_mut(),
            max_cells: 0,
            n_pages: 0,
            n_slots: 0,
            idx_pooled_rows: 0,
            kv_mode: 0,
            kv_format: 0,
            kv_int8: 0,
            kv_q4: 0,
            kv_hybrid: 0,
            kv_rot: 0,
            host_present: 0,
            map_ok: 0,
            ring_ok: 0,
            reserved: 0,
        }
    }
}

/// `tk_qwen4exp_p5_layout_t`: the session's running-state and K/V buffers, geometry and constants.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct P5Layout {
    pub geometry: [i64; 18],
    pub layer_lo: i64,
    pub layer_hi: i64,
    pub n_gdn_layers: i64,
    pub n_qsa_layers: i64,
    pub gdn_alloc: i64,
    pub qsa_ord0: i64,
    pub qsa_alloc: i64,
    pub max_cells: i64,
    pub ssm_state_size: i64,
    pub ssm_v_heads: i64,
    pub ssm_conv_channels: i64,
    pub ssm_d_conv: i64,
    pub idx_key_dim: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub ng_hist: i64,
    pub ng_hc_dim: i64,
    pub idx_block: i64,
    pub page_size: i64,
    pub kv_q4_bytes_per_head: i64,
    pub memory_limit_bytes: u64,
    pub gdn_state: *mut u8,
    pub ple_hist: *mut u8,
    pub has_draft: i32,
    pub n_qsa: i32,
    pub qsa: [P5Qsa; P5_MAX_QSA],
    pub draft: P5Qsa,
}

impl Default for P5Layout {
    fn default() -> Self {
        Self {
            geometry: [0; 18],
            layer_lo: 0,
            layer_hi: 0,
            n_gdn_layers: 0,
            n_qsa_layers: 0,
            gdn_alloc: 0,
            qsa_ord0: 0,
            qsa_alloc: 0,
            max_cells: 0,
            ssm_state_size: 0,
            ssm_v_heads: 0,
            ssm_conv_channels: 0,
            ssm_d_conv: 0,
            idx_key_dim: 0,
            n_head_kv: 0,
            head_dim: 0,
            ng_hist: 0,
            ng_hc_dim: 0,
            idx_block: 0,
            page_size: 0,
            kv_q4_bytes_per_head: 0,
            memory_limit_bytes: 0,
            gdn_state: std::ptr::null_mut(),
            ple_hist: std::ptr::null_mut(),
            has_draft: 0,
            n_qsa: 0,
            qsa: [P5Qsa::default(); P5_MAX_QSA],
            draft: P5Qsa::default(),
        }
    }
}

pub type FnP5Layout = unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx, out: *mut P5Layout) -> i32;
pub type FnP5Copy = unsafe extern "C" fn(
    dst: *mut u8,
    src: *const u8,
    n: usize,
    err: *mut c_char,
    err_cap: usize,
) -> i32;
pub type FnP5Sync = unsafe extern "C" fn(err: *mut c_char, err_cap: usize) -> i32;
pub type FnP5KvResidency = unsafe extern "C" fn(
    ctx: *mut TkQwen4ExpServeCtx,
    which: i32,
    upto: i64,
    err: *mut c_char,
    err_cap: usize,
) -> i32;
pub type FnP5SetPlePrev = unsafe extern "C" fn(ctx: *mut TkQwen4ExpServeCtx, a: i32, b: i32);

/// The device primitives the conversation code needs (`qwen4exp-p5`); tests substitute host implementations.
#[derive(Clone, Copy)]
pub struct P5Device {
    pub ctx: *mut TkQwen4ExpServeCtx,
    pub layout: FnP5Layout,
    pub copy: FnP5Copy,
    pub sync: FnP5Sync,
    pub kv_residency: FnP5KvResidency,
    pub set_ple_prev: FnP5SetPlePrev,
}

fn err_text(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

impl P5Device {
    pub fn query_layout(&self) -> Result<P5Layout, String> {
        let mut l = P5Layout::default();
        if unsafe { (self.layout)(self.ctx, &mut l) } < 0 {
            return Err("conversation snapshot: no session layout".to_string());
        }
        Ok(l)
    }
    /// `conversation_detail::copy_nonblocking` (cudaMemcpyDefault on a non-blocking stream, then synchronized).
    pub fn copy(&self, dst: *mut u8, src: *const u8, n: usize) -> Result<(), String> {
        let mut e = [0u8; 256];
        if unsafe { (self.copy)(dst, src, n, e.as_mut_ptr().cast(), e.len()) } != 0 {
            return Err(err_text(&e));
        }
        Ok(())
    }
    pub fn sync(&self) -> Result<(), String> {
        let mut e = [0u8; 256];
        if unsafe { (self.sync)(e.as_mut_ptr().cast(), e.len()) } != 0 {
            return Err(err_text(&e));
        }
        Ok(())
    }
    pub fn kv_residency(&self, which: i32, upto: i64) -> Result<(), String> {
        let mut e = [0u8; 256];
        if unsafe { (self.kv_residency)(self.ctx, which, upto, e.as_mut_ptr().cast(), e.len()) }
            != 0
        {
            return Err(err_text(&e));
        }
        Ok(())
    }
    pub fn set_ple_prev(&self, a: i32, b: i32) {
        unsafe { (self.set_ple_prev)(self.ctx, a, b) }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// host structures
// ---------------------------------------------------------------------------------------------------------------

/// `ConversationImageKey` (always empty on the text-only Rust path; kept for the file format).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageKey {
    pub start: i64,
    pub hash: u64,
}

/// `sizeof` of the C++ structures, for the byte accounting the cache budget and RAM preflights use.
pub const SIZEOF_CHECKPOINT: usize = 208;
pub const SIZEOF_CONV_KV: usize = 216;
pub const SIZEOF_KV_REUSE: usize = 64;
const SIZEOF_SEGMENT: usize = 24;

/// `ConversationCheckpoint`: the running state after `ids` (GDN recurrence, PLE history, QSA indexer tails).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Checkpoint {
    pub ids: Vec<i32>,
    pub imgs: Vec<ImageKey>,
    pub gdn: Vec<u8>,
    pub ple: Vec<u8>,
    pub tails: Vec<u8>,
    pub dead: Vec<u8>,
    pub block_pos: Vec<u8>,
    /// LRU stamp (root-pinned / LRU retention).
    pub used: u64,
    /// A shared-prefix pin (`pin=N`): never evicted; run-time only, not in the session file.
    pub pinned: bool,
}

// retired checkpoints' GDN running-state buffers (~118 MB on Flash-Next) kept for the next checkpoint, so a
// turn checkpoint copies into memory that is already allocated and faulted in instead of a fresh zero-filled Vec (the
// copy overwrites every byte: the same checkpoint).  At most GDN_POOL_MAX buffers; small ones are not kept.  The serve
// state hands discarded checkpoints back with recycle_gdn.
const GDN_POOL_MAX: usize = 3;
const GDN_POOL_MIN_BYTES: usize = 1 << 20;
static GDN_POOL: std::sync::Mutex<Vec<Vec<u8>>> = std::sync::Mutex::new(Vec::new());

/// A buffer of `n` bytes for a checkpoint's GDN state, every byte of which the caller overwrites.
fn gdn_buffer(mut v: Vec<u8>, n: usize) -> Vec<u8> {
    if v.capacity() < n && n >= GDN_POOL_MIN_BYTES {
        if let Ok(mut pool) = GDN_POOL.lock() {
            if let Some(i) = pool.iter().position(|b| b.capacity() >= n) {
                v = pool.swap_remove(i);
            }
        }
    }
    if v.capacity() >= n {
        // SAFETY: u8 has no invalid bit patterns; the caller writes all n bytes before anything reads them
        unsafe { v.set_len(n) };
        v
    } else {
        v.resize(n, 0);
        v
    }
}

/// Gives a discarded checkpoint's GDN buffer to the pool (see gdn_buffer).
pub fn recycle_gdn(v: Vec<u8>) {
    if v.capacity() < GDN_POOL_MIN_BYTES {
        return;
    }
    if let Ok(mut pool) = GDN_POOL.lock() {
        if pool.len() < GDN_POOL_MAX {
            pool.push(v);
        }
    }
}

impl Checkpoint {
    pub fn bytes(&self) -> usize {
        self.ids.capacity() * 4
            + self.imgs.capacity() * 16
            + self.gdn.capacity()
            + self.ple.capacity()
            + self.tails.capacity()
            + self.dead.capacity()
            + self.block_pos.capacity()
    }
}

/// `ConversationBuffer`: a byte array kept as 16 MiB segments, each its own allocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConvBuffer {
    segments: Vec<Vec<u8>>,
    size: usize,
}

impl ConvBuffer {
    pub const SEGMENT: usize = 16 * 1024 * 1024;

    pub fn size(&self) -> usize {
        self.size
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    pub fn bytes(&self) -> usize {
        self.segments.capacity() * SIZEOF_SEGMENT
            + self.segments.iter().map(|s| s.capacity()).sum::<usize>()
    }
    fn segment_capacity(count: usize, previous: usize, growing: bool) -> usize {
        if !growing {
            return count;
        }
        count.max(Self::SEGMENT.min(65536usize.max(previous.min(Self::SEGMENT / 2) * 2)))
    }
    /// The allocation if the buffer were resized to `n` (`allocation_peak`).
    pub fn allocation_peak(&self, n: usize) -> usize {
        let mut total = self.bytes();
        if n <= self.size {
            return total;
        }
        let spare = self.segments.last().map_or(0, |s| s.capacity() - s.len());
        let extra = n - self.size - (n - self.size).min(spare);
        let (full, tail) = (extra / Self::SEGMENT, extra % Self::SEGMENT);
        total = total.saturating_add(full * Self::SEGMENT);
        if tail > 0 {
            let prev = if full > 0 {
                Self::SEGMENT
            } else {
                self.segments.last().map_or(0, |s| s.capacity())
            };
            total = total.saturating_add(Self::segment_capacity(tail, prev, self.size != 0));
        }
        let count = self.segments.len() + full + (tail != 0) as usize;
        if count > self.segments.capacity() {
            total = total.saturating_add(count * SIZEOF_SEGMENT);
        }
        total
    }
    pub fn resize(&mut self, n: usize) {
        if n > self.size {
            let growing = self.size != 0;
            let spare = self.segments.last().map_or(0, |s| s.capacity() - s.len());
            let extend = (n - self.size).min(spare);
            let extra = n - self.size - extend;
            self.segments
                .reserve_exact(extra / Self::SEGMENT + (extra % Self::SEGMENT != 0) as usize);
            if extend > 0 {
                let last = self.segments.last_mut().unwrap();
                last.resize(last.len() + extend, 0);
                self.size += extend;
            }
            while self.size < n {
                let count = Self::SEGMENT.min(n - self.size);
                let previous = self.segments.last().map_or(0, |s| s.capacity());
                let mut seg = Vec::with_capacity(Self::segment_capacity(count, previous, growing));
                seg.resize(count, 0);
                self.segments.push(seg);
                self.size += count;
            }
        } else {
            while let Some(last) = self.segments.last() {
                if self.size - last.len() >= n {
                    self.size -= last.len();
                    self.segments.pop();
                } else {
                    break;
                }
            }
            if self.size > n {
                let cut = self.size - n;
                let last = self.segments.last_mut().unwrap();
                let l = last.len();
                last.truncate(l - cut);
            }
            self.size = n;
        }
    }
    /// Calls `f(chunk, offset)` over `[offset, offset + count)`; false when out of range or `f` refused.
    pub fn visit(
        &self,
        mut offset: usize,
        mut count: usize,
        mut f: impl FnMut(&[u8], usize) -> bool,
    ) -> bool {
        if offset > self.size || count > self.size - offset {
            return false;
        }
        let mut begin = 0usize;
        for seg in &self.segments {
            if count == 0 {
                break;
            }
            let end = begin + seg.len();
            if offset < end {
                let n = count.min(end - offset);
                if !f(&seg[offset - begin..offset - begin + n], offset) {
                    return false;
                }
                offset += n;
                count -= n;
            }
            begin = end;
        }
        count == 0
    }
    pub fn visit_mut(
        &mut self,
        mut offset: usize,
        mut count: usize,
        mut f: impl FnMut(&mut [u8], usize) -> bool,
    ) -> bool {
        if offset > self.size || count > self.size - offset {
            return false;
        }
        let mut begin = 0usize;
        for seg in &mut self.segments {
            if count == 0 {
                break;
            }
            let end = begin + seg.len();
            if offset < end {
                let n = count.min(end - offset);
                if !f(&mut seg[offset - begin..offset - begin + n], offset) {
                    return false;
                }
                offset += n;
                count -= n;
            }
            begin = end;
        }
        count == 0
    }
}

/// `ConversationKv`: identity-layout K/V pages and completed indexer rows of one QSA layer (or the draft layer).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConvKv {
    pub format: i32,
    pub cells: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub page_size: i64,
    pub pooled_rows: i64,
    pub idx_dim: i64,
    pub k: ConvBuffer,
    pub v: ConvBuffer,
    pub k_scale: ConvBuffer,
    pub v_scale: ConvBuffer,
    pub pooled: ConvBuffer,
}

impl ConvKv {
    pub fn bytes(&self) -> usize {
        self.k.bytes()
            + self.v.bytes()
            + self.k_scale.bytes()
            + self.v_scale.bytes()
            + self.pooled.bytes()
    }
    fn parts(&self) -> [&ConvBuffer; 5] {
        [&self.k, &self.v, &self.k_scale, &self.v_scale, &self.pooled]
    }
    fn parts_mut(&mut self) -> [&mut ConvBuffer; 5] {
        [
            &mut self.k,
            &mut self.v,
            &mut self.k_scale,
            &mut self.v_scale,
            &mut self.pooled,
        ]
    }
}

/// `ConversationKvReuse`: a restored image's K/V kept to make the next park cheaper.
#[derive(Clone, Debug, Default)]
pub struct KvReuse {
    pub kv: Vec<ConvKv>,
    pub captured_tokens: i64,
    pub unchanged_tokens: i64,
}

impl KvReuse {
    pub fn bytes(&self) -> usize {
        self.kv.capacity() * SIZEOF_CONV_KV + self.kv.iter().map(|k| k.bytes()).sum::<usize>()
    }
}

/// `SavedConversation`: a parked conversation, or a session file's content.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SavedConversation {
    pub geometry: [i64; 18],
    pub layer_lo: i64,
    pub layer_hi: i64,
    pub live: Checkpoint,
    pub checkpoints: Vec<Checkpoint>,
    /// Main QSA layers followed by the draft layer.
    pub kv: Vec<ConvKv>,
    pub cvec: bool,
}

impl SavedConversation {
    pub fn pinned(&self) -> bool {
        self.checkpoints.iter().any(|c| c.pinned)
    }
    pub fn bytes(&self) -> usize {
        self.live.bytes()
            + self.checkpoints.capacity() * SIZEOF_CHECKPOINT
            + self.kv.capacity() * SIZEOF_CONV_KV
            + self.checkpoints.iter().map(|c| c.bytes()).sum::<usize>()
            + self.kv.iter().map(|k| k.bytes()).sum::<usize>()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// running-state sizes and checkpoint save / restore (conversation_state.cpp)
// ---------------------------------------------------------------------------------------------------------------

/// `ConversationStateSizes`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateSizes {
    pub gdn: usize,
    pub ple: usize,
    pub tail: usize,
    pub dead: usize,
    pub block_pos: usize,
}

fn snap_fail(m: &str) -> String {
    format!("conversation snapshot: {m}")
}

fn product(f: &[u64]) -> Option<usize> {
    let mut v: usize = 1;
    for &x in f {
        v = v.checked_mul(usize::try_from(x).ok()?)?;
    }
    Some(v)
}

/// `conversation_state_sizes` + `conversation_session_sizes`.
pub fn session_sizes(l: &P5Layout) -> Result<StateSizes, String> {
    for (i, &k) in l.geometry.iter().enumerate() {
        if k < 0 || (i != 1 && k == 0) {
            return Err(snap_fail("invalid model geometry"));
        }
    }
    let ovf = || snap_fail("running-state byte count overflow");
    let rec = product(&[
        l.ssm_state_size as u64,
        l.ssm_v_heads as u64,
        l.ssm_state_size as u64,
    ])
    .ok_or_else(ovf)?;
    let conv = product(&[l.ssm_conv_channels as u64, (l.ssm_d_conv - 1) as u64]).ok_or_else(ovf)?;
    let recurrence = rec.checked_add(conv).ok_or_else(ovf)?;
    let mut z = StateSizes {
        gdn: product(&[l.n_gdn_layers as u64, recurrence as u64, 4]).ok_or_else(ovf)?,
        ple: product(&[l.ng_hist as u64, l.ng_hc_dim as u64, 4]).ok_or_else(ovf)?,
        tail: product(&[(l.idx_block - 1) as u64, l.idx_key_dim as u64, 4]).ok_or_else(ovf)?,
        dead: product(&[l.idx_key_dim as u64, 4]).ok_or_else(ovf)?,
        block_pos: 4,
    };
    let (n_gdn, n_qsa) = (l.n_gdn_layers, l.n_qsa_layers);
    if l.gdn_alloc < 0
        || l.gdn_alloc > n_gdn
        || l.qsa_ord0 < 0
        || l.qsa_alloc < 0
        || l.qsa_ord0 > n_qsa
        || l.qsa_alloc > n_qsa - l.qsa_ord0
    {
        return Err(snap_fail("invalid session layer carve"));
    }
    z.gdn = if n_gdn > 0 {
        z.gdn / n_gdn as usize * l.gdn_alloc as usize
    } else {
        0
    };
    Ok(z)
}

fn owned_qsa(l: &P5Layout) -> usize {
    (l.qsa_alloc.max(0) as usize).min(l.n_qsa.max(0) as usize)
}

fn checkpoint_targets(l: &P5Layout, tokens: usize) -> Result<StateSizes, String> {
    let z = session_sizes(l)?;
    if l.max_cells < 0
        || tokens as u64 > l.max_cells as u64
        || (z.gdn > 0 && l.gdn_state.is_null())
        || (l.qsa_alloc as usize) > l.n_qsa.max(0) as usize
    {
        return Err(snap_fail("invalid session running-state targets"));
    }
    for st in &l.qsa[..owned_qsa(l)] {
        let block = l.idx_block as u64;
        if st.idx_tail.is_null()
            || st.idx_dead.is_null()
            || st.idx_block_pos.is_null()
            || st.idx_pooled.is_null()
            || st.max_cells < 0
            || tokens as u64 > st.max_cells as u64
            || (tokens > 0 && tokens as u64 / block >= st.idx_pooled_rows.max(0) as u64)
            || product(&[tokens as u64 / block + 1, l.idx_key_dim as u64, 4]).is_none()
        {
            return Err(snap_fail("invalid indexer running-state target"));
        }
    }
    Ok(z)
}

/// `conversation_checkpoint_validate`.
pub fn checkpoint_validate(c: &Checkpoint, l: &P5Layout) -> Result<(), String> {
    let z = checkpoint_targets(l, c.ids.len())?;
    let layers = owned_qsa(l);
    if c.gdn.len() != z.gdn
        || c.ple.len() != if l.ple_hist.is_null() { 0 } else { z.ple }
        || c.tails.len() != layers * z.tail
        || c.dead.len() != layers * z.dead
        || c.block_pos.len() != layers * z.block_pos
        || !image_keys(&c.imgs, c.ids.len())
    {
        return Err(snap_fail("invalid checkpoint running-state payload"));
    }
    Ok(())
}

fn image_keys(images: &[ImageKey], tokens: usize) -> bool {
    let mut previous = -1i64;
    for k in images {
        if k.start <= previous || k.start < 0 || k.start as u64 >= tokens as u64 {
            return false;
        }
        previous = k.start;
    }
    true
}

fn dev_copy(
    dev: &P5Device,
    dst: *mut u8,
    src: *const u8,
    n: usize,
    what: &str,
) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    if dst.is_null() || src.is_null() {
        return Err(snap_fail("missing running-state buffer"));
    }
    dev.copy(dst, src, n)
        .map_err(|e| format!("conversation snapshot {what}: {e}"))
}

/// `conversation_checkpoint_save`: copies the running state out (this session's carve).
pub fn checkpoint_save(c: &mut Checkpoint, dev: &P5Device, l: &P5Layout) -> Result<(), String> {
    let z = checkpoint_targets(l, c.ids.len())?;
    let layers = owned_qsa(l);
    c.gdn = gdn_buffer(std::mem::take(&mut c.gdn), z.gdn);
    c.ple
        .resize(if l.ple_hist.is_null() { 0 } else { z.ple }, 0);
    c.tails.resize(layers * z.tail, 0);
    c.dead.resize(layers * z.dead, 0);
    c.block_pos.resize(layers * z.block_pos, 0);
    let w = "running-state copy";
    dev_copy(dev, c.gdn.as_mut_ptr(), l.gdn_state, c.gdn.len(), w)?;
    dev_copy(dev, c.ple.as_mut_ptr(), l.ple_hist, c.ple.len(), w)?;
    for (j, st) in l.qsa[..layers].iter().enumerate() {
        dev_copy(
            dev,
            c.tails[j * z.tail..].as_mut_ptr(),
            st.idx_tail,
            z.tail,
            w,
        )?;
        dev_copy(
            dev,
            c.dead[j * z.dead..].as_mut_ptr(),
            st.idx_dead,
            z.dead,
            w,
        )?;
        dev_copy(
            dev,
            c.block_pos[j * z.block_pos..].as_mut_ptr(),
            st.idx_block_pos,
            z.block_pos,
            w,
        )?;
    }
    Ok(())
}

/// `conversation_checkpoint_restore`: puts a checkpoint's running state back.
pub fn checkpoint_restore(c: &Checkpoint, dev: &P5Device, l: &P5Layout) -> Result<(), String> {
    checkpoint_validate(c, l)?;
    let z = session_sizes(l)?;
    dev.sync()
        .map_err(|e| format!("conversation snapshot synchronize: {e}"))?;
    let w = "running-state copy";
    dev_copy(dev, l.gdn_state, c.gdn.as_ptr(), c.gdn.len(), w)?;
    dev_copy(dev, l.ple_hist, c.ple.as_ptr(), c.ple.len(), w)?;
    for (j, st) in l.qsa[..owned_qsa(l)].iter().enumerate() {
        dev_copy(dev, st.idx_tail, c.tails[j * z.tail..].as_ptr(), z.tail, w)?;
        dev_copy(dev, st.idx_dead, c.dead[j * z.dead..].as_ptr(), z.dead, w)?;
        dev_copy(
            dev,
            st.idx_block_pos,
            c.block_pos[j * z.block_pos..].as_ptr(),
            z.block_pos,
            w,
        )?;
        if !c.ids.is_empty() {
            let row = c.ids.len() / l.idx_block as usize;
            let dst = unsafe { st.idx_pooled.add(row * l.idx_key_dim as usize * 4) };
            dev_copy(dev, dst, c.dead[j * z.dead..].as_ptr(), z.dead, w)?;
        }
    }
    let t = c.ids.len();
    dev.set_ple_prev(
        if t >= 2 { c.ids[t - 2] } else { -1 },
        if t >= 1 { c.ids[t - 1] } else { -1 },
    );
    dev.sync()
        .map_err(|e| format!("conversation snapshot synchronize: {e}"))
}

// ---------------------------------------------------------------------------------------------------------------
// K/V images (conversation_snapshot.cpp)
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct KvLayout {
    format: i32,
    cells: i64,
    pooled_rows: i64,
    page_size: i64,
    sizes: [usize; 5], // data, value_data, scales, value_scales, pooled
}

fn kv_layout(st: &P5Qsa, l: &P5Layout, upto: i64, index: bool) -> Result<KvLayout, String> {
    let page = l.page_size;
    if upto < 0 || upto > st.max_cells || upto > i64::MAX - (page - 1) {
        return Err(snap_fail("invalid K/V extent"));
    }
    let (q4, int8, hybrid) = (st.kv_q4 != 0, st.kv_int8 != 0, st.kv_hybrid != 0);
    if hybrid && (st.kv_mode == 2 || q4 || int8) {
        return Err(snap_fail(
            "hybrid K8V4 cannot be a ring and needs distinct format flags",
        ));
    }
    let int8_keys = int8 || hybrid;
    if l.n_head_kv <= 0
        || l.head_dim <= 0
        || l.head_dim > i32::MAX as i64
        || l.idx_key_dim <= 0
        || (q4 && l.head_dim % 32 != 0)
        || (int8_keys && !q4 && l.head_dim % 64 != 0)
    {
        return Err(snap_fail("invalid K/V geometry"));
    }
    let cells = (upto + page - 1) / page * page;
    let per = if q4 {
        l.kv_q4_bytes_per_head as u64
    } else {
        l.head_dim as u64 * if int8_keys { 1 } else { 2 }
    };
    let pooled = if index && upto > 0 {
        upto / l.idx_block + 1
    } else {
        0
    };
    let rotated = if st.kv_rot != 0 && !q4 && !hybrid {
        16
    } else {
        0
    };
    let ovf = || snap_fail("K/V byte count overflow");
    let data = product(&[cells as u64, l.n_head_kv as u64, per]).ok_or_else(ovf)?;
    let scales = product(&[
        cells as u64,
        l.n_head_kv as u64,
        if int8_keys && !q4 {
            (l.head_dim / 64) as u64 * 2
        } else {
            0
        },
    ])
    .ok_or_else(ovf)?;
    let pooled_b = product(&[pooled as u64, l.idx_key_dim as u64, 4]).ok_or_else(ovf)?;
    let (mut value_data, mut value_scales) = (data, scales);
    if hybrid {
        value_data = product(&[
            cells as u64,
            l.n_head_kv as u64,
            l.kv_q4_bytes_per_head as u64,
        ])
        .ok_or_else(|| snap_fail("hybrid V byte count overflow"))?;
        value_scales = 0;
    }
    Ok(KvLayout {
        format: (if hybrid { 3 } else { st.kv_format }) + rotated,
        cells,
        pooled_rows: pooled,
        page_size: page,
        sizes: [data, value_data, scales, value_scales, pooled_b],
    })
}

fn kv_valid(st: &P5Qsa, kl: &KvLayout, upto: i64) -> Result<(), String> {
    if upto < 0
        || upto > st.max_cells
        || st.n_pages < 0
        || kl.cells / kl.page_size > st.n_pages
        || st.kv_mode < 0
        || st.kv_mode > 2
        || st.n_slots <= 0
        || (st.kv_mode == 0 && st.n_slots < st.n_pages)
        || kl.pooled_rows > st.idx_pooled_rows
        || (st.kv_mode != 0 && st.host_present == 0)
    {
        return Err(snap_fail(
            "invalid K/V extent or missing authoritative host pool",
        ));
    }
    if st.kv_mode == 1 && st.map_ok == 0 {
        return Err(snap_fail("invalid streaming map"));
    }
    if st.kv_mode == 2 && st.ring_ok == 0 {
        return Err(snap_fail("invalid draft ring"));
    }
    Ok(())
}

fn kv_transfer(dev: &P5Device, dst: *mut u8, src: *const u8, n: usize) -> Result<(), String> {
    if n == 0 {
        return Ok(());
    }
    if src.is_null() || dst.is_null() {
        return Err(snap_fail("missing state buffer"));
    }
    dev.copy(dst, src, n)
        .map_err(|e| format!("conversation snapshot copy: {e}"))
}

/// `conversation_kv_capture_bytes`: what a capture into `image` allocates in all.
fn kv_capture_bytes(
    image: &ConvKv,
    st: &P5Qsa,
    l: &P5Layout,
    upto: i64,
    index: bool,
) -> Result<usize, String> {
    let kl = kv_layout(st, l, upto, index)?;
    kv_valid(st, &kl, upto)?;
    let mut bytes = 0usize;
    for (b, &s) in image.parts().iter().zip(kl.sizes.iter()) {
        bytes = bytes
            .checked_add(b.allocation_peak(s))
            .ok_or_else(|| snap_fail("segmented K/V allocation overflow"))?;
    }
    Ok(bytes)
}

/// `conversation_kv_bytes` (0 = invalid).
pub fn kv_bytes(st: &P5Qsa, l: &P5Layout, upto: i64, index: bool) -> usize {
    kv_capture_bytes(&ConvKv::default(), st, l, upto, index).unwrap_or(0)
}

/// `conversation_kv_save`: recopies from the first page a rewrite touched (`unchanged_tokens`).
fn kv_save(
    image: &mut ConvKv,
    st: &P5Qsa,
    l: &P5Layout,
    dev: &P5Device,
    upto: i64,
    index: bool,
    unchanged: i64,
    reused: &mut usize,
) -> Result<(), String> {
    let kl = kv_layout(st, l, upto, index)?;
    kv_valid(st, &kl, upto)?;
    if unchanged < 0 || unchanged > upto || unchanged > image.cells {
        return Err(snap_fail("invalid unchanged prefix"));
    }
    let whole_cells = unchanged / kl.page_size * kl.page_size;
    image.format = kl.format;
    image.cells = kl.cells;
    image.heads = l.n_head_kv;
    image.head_dim = l.head_dim;
    image.page_size = kl.page_size;
    image.pooled_rows = kl.pooled_rows;
    image.idx_dim = l.idx_key_dim;
    for (i, dst) in image.parts_mut().into_iter().enumerate() {
        let size = kl.sizes[i];
        let keep = if i == 4 {
            if index {
                (unchanged / l.idx_block) as usize * l.idx_key_dim as usize * 4
            } else {
                0
            }
        } else if kl.cells != 0 {
            (size / kl.cells as usize) * whole_cells as usize
        } else {
            0
        };
        if keep > dst.size() {
            return Err(snap_fail("missing reusable prefix"));
        }
        dst.resize(size);
        let src = st.pools[i];
        let mut err = None;
        let ok = dst.visit_mut(keep, size - keep, |p, at| {
            let s = if src.is_null() {
                std::ptr::null()
            } else {
                unsafe { src.add(at) as *const u8 }
            };
            match kv_transfer(dev, p.as_mut_ptr(), s, p.len()) {
                Ok(()) => true,
                Err(e) => {
                    err = Some(e);
                    false
                }
            }
        });
        if !ok {
            return Err(err.unwrap_or_else(|| snap_fail("K/V capture out of range")));
        }
        *reused += keep;
    }
    Ok(())
}

/// `conversation_kv_validate`.
fn kv_validate(
    image: &ConvKv,
    st: &P5Qsa,
    l: &P5Layout,
    upto: i64,
    index: bool,
) -> Result<(), String> {
    let kl = kv_layout(st, l, upto, index)?;
    kv_valid(st, &kl, upto)?;
    if image.format != kl.format
        || image.cells != kl.cells
        || image.heads != l.n_head_kv
        || image.head_dim != l.head_dim
        || image.page_size != kl.page_size
        || image.pooled_rows != kl.pooled_rows
        || image.idx_dim != l.idx_key_dim
    {
        return Err(snap_fail("incompatible K/V geometry"));
    }
    for (b, &s) in image.parts().iter().zip(kl.sizes.iter()) {
        if b.size() != s {
            return Err(snap_fail("invalid K/V payload size"));
        }
    }
    for (i, &s) in kl.sizes.iter().enumerate() {
        if s > 0 && st.pools[i].is_null() {
            return Err(snap_fail("missing target state buffer"));
        }
    }
    Ok(())
}

/// `conversation_kv_restore` (`which`: owned QSA ordinal, or -1 for the draft layer).
fn kv_restore(
    image: &ConvKv,
    st: &P5Qsa,
    l: &P5Layout,
    dev: &P5Device,
    upto: i64,
    index: bool,
    which: i32,
) -> Result<(), String> {
    kv_validate(image, st, l, upto, index)?;
    for (i, b) in image.parts().iter().enumerate() {
        let dst = st.pools[i];
        let mut err = None;
        let ok = b.visit(0, b.size(), |p, at| {
            match kv_transfer(dev, unsafe { dst.add(at) }, p.as_ptr(), p.len()) {
                Ok(()) => true,
                Err(e) => {
                    err = Some(e);
                    false
                }
            }
        });
        if !ok {
            return Err(err.unwrap_or_else(|| snap_fail("K/V restore out of range")));
        }
    }
    // VRAM slots / the draft ring still hold the outgoing conversation: refill them from the restored pools
    dev.kv_residency(which, upto)
        .map_err(|e| format!("conversation snapshot residency restore: {e}"))
}

/// `conversation_kv_part_sizes`.
fn kv_part_sizes(st: &P5Qsa, l: &P5Layout, upto: i64, index: bool) -> Result<[u64; 5], String> {
    let kl = kv_layout(st, l, upto, index)?;
    Ok(kl.sizes.map(|s| s as u64))
}

// ---------------------------------------------------------------------------------------------------------------
// snapshots (conversation_state.cpp: view / snapshot)
// ---------------------------------------------------------------------------------------------------------------

/// `ConversationView`: what a park or a SAVE captures.
pub struct ConvView<'a> {
    pub ids: &'a [i32],
    pub images: &'a [ImageKey],
    pub checkpoints: &'a [Checkpoint],
    pub cvec: bool,
}

/// The draft layer's K/V state when parking includes it (`--mtp`); session files always carry it.
pub fn parked_draft(l: &P5Layout) -> Option<&P5Qsa> {
    (l.has_draft != 0).then_some(&l.draft)
}

fn view_validate(v: &ConvView, l: &P5Layout) -> Result<(), String> {
    checkpoint_targets(l, v.ids.len())?;
    if v.ids.is_empty() || !image_keys(v.images, v.ids.len()) || v.ids.iter().any(|&i| i < 0) {
        return Err(snap_fail("invalid live token/image metadata"));
    }
    for c in v.checkpoints {
        if c.ids.is_empty() || c.ids.len() > v.ids.len() || c.ids[..] != v.ids[..c.ids.len()] {
            return Err(snap_fail("checkpoint is not a live token prefix"));
        }
        checkpoint_validate(c, l)?;
        let mut image = 0usize;
        for key in v.images {
            if key.start as u64 >= c.ids.len() as u64 {
                break;
            }
            if image >= c.imgs.len() || c.imgs[image] != *key {
                return Err(snap_fail("checkpoint image identity differs"));
            }
            image += 1;
        }
        if image != c.imgs.len() {
            return Err(snap_fail("checkpoint image prefix differs"));
        }
    }
    Ok(())
}

fn metadata_bytes(c: &Checkpoint) -> usize {
    c.ids.len() * 4
        + c.imgs.len() * 16
        + c.gdn.len()
        + c.ple.len()
        + c.tails.len()
        + c.dead.len()
        + c.block_pos.len()
}

/// `conversation_snapshot_bytes`: a fresh capture's size.
pub fn snapshot_bytes(v: &ConvView, l: &P5Layout, draft: Option<&P5Qsa>) -> Result<usize, String> {
    view_validate(v, l)?;
    let z = session_sizes(l)?;
    let qsa = owned_qsa(l);
    let mut bytes = v.ids.len() * 4
        + v.images.len() * 16
        + v.checkpoints.len() * SIZEOF_CHECKPOINT
        + (qsa + draft.is_some() as usize) * SIZEOF_CONV_KV
        + qsa * z.tail
        + qsa * z.dead
        + qsa * z.block_pos
        + z.gdn
        + if l.ple_hist.is_null() { 0 } else { z.ple };
    for c in v.checkpoints {
        bytes += metadata_bytes(c);
    }
    let upto = v.ids.len() as i64;
    for i in 0..qsa + draft.is_some() as usize {
        let (st, index) = if i == qsa {
            (draft.unwrap(), false)
        } else {
            (&l.qsa[i], true)
        };
        let n = kv_bytes(st, l, upto, index);
        if n == 0 {
            return Err(snap_fail("invalid or overflowing K/V byte estimate"));
        }
        bytes += n;
    }
    Ok(bytes)
}

/// `conversation_snapshot_capture_bytes`: with a retained K/V, what the capture allocates instead.
pub fn snapshot_capture_bytes(
    reuse: &KvReuse,
    v: &ConvView,
    l: &P5Layout,
    draft: Option<&P5Qsa>,
) -> Result<usize, String> {
    let mut bytes = snapshot_bytes(v, l, draft)?;
    if reuse.kv.is_empty() {
        return Ok(bytes);
    }
    let qsa = owned_qsa(l);
    let layers = qsa + draft.is_some() as usize;
    if reuse.kv.len() != layers
        || reuse.unchanged_tokens < 0
        || reuse.unchanged_tokens > reuse.captured_tokens
        || reuse.unchanged_tokens > v.ids.len() as i64
    {
        return Err(snap_fail("invalid retained K/V prefix"));
    }
    for i in 0..layers {
        let index = i < qsa;
        let st = if index { &l.qsa[i] } else { draft.unwrap() };
        kv_validate(&reuse.kv[i], st, l, reuse.captured_tokens, index)?;
        let fresh = kv_bytes(st, l, v.ids.len() as i64, index);
        let retained = kv_capture_bytes(&reuse.kv[i], st, l, v.ids.len() as i64, index)?;
        bytes = bytes - fresh + retained;
    }
    bytes += (reuse.kv.capacity() - layers) * SIZEOF_CONV_KV;
    Ok(bytes)
}

/// `conversation_snapshot_save`: the whole conversation into host memory (`reuse`: a retained K/V prefix).
pub fn snapshot_save(
    v: &ConvView,
    l: &P5Layout,
    dev: &P5Device,
    draft: Option<&P5Qsa>,
    reuse: KvReuse,
    reused_bytes: &mut usize,
) -> Result<SavedConversation, String> {
    snapshot_capture_bytes(&reuse, v, l, draft)?;
    dev.sync()
        .map_err(|e| format!("conversation snapshot synchronize: {e}"))?;
    let mut cap = SavedConversation {
        geometry: l.geometry,
        layer_lo: l.layer_lo,
        layer_hi: l.layer_hi,
        live: Checkpoint {
            ids: v.ids.to_vec(),
            imgs: v.images.to_vec(),
            ..Default::default()
        },
        checkpoints: v.checkpoints.to_vec(),
        kv: Vec::new(),
        cvec: v.cvec,
    };
    let unchanged = if reuse.kv.is_empty() {
        0
    } else {
        reuse.unchanged_tokens
    };
    cap.kv = reuse.kv;
    let layers = owned_qsa(l);
    cap.kv
        .resize_with(layers + draft.is_some() as usize, ConvKv::default);
    checkpoint_save(&mut cap.live, dev, l)?;
    let upto = v.ids.len() as i64;
    for j in 0..layers {
        kv_save(
            &mut cap.kv[j],
            &l.qsa[j],
            l,
            dev,
            upto,
            true,
            unchanged,
            reused_bytes,
        )?;
    }
    if let Some(d) = draft {
        // the draft's final cell may not have been computed when the output cap was reached: its page is recopied
        let last = cap.kv.len() - 1;
        kv_save(
            &mut cap.kv[last],
            d,
            l,
            dev,
            upto,
            false,
            (unchanged - 1).max(0),
            reused_bytes,
        )?;
    }
    Ok(cap)
}

/// `conversation_snapshot_validate`.
pub fn snapshot_validate(
    image: &SavedConversation,
    l: &P5Layout,
    draft: Option<&P5Qsa>,
) -> Result<(), String> {
    if image.geometry != l.geometry {
        return Err(snap_fail("incompatible runtime geometry"));
    }
    if image.layer_lo != l.layer_lo || image.layer_hi != l.layer_hi {
        return Err(snap_fail("snapshot from another session layer range"));
    }
    let v = ConvView {
        ids: &image.live.ids,
        images: &image.live.imgs,
        checkpoints: &image.checkpoints,
        cvec: image.cvec,
    };
    view_validate(&v, l)?;
    checkpoint_validate(&image.live, l)?;
    let layers = owned_qsa(l);
    if image.kv.len() != layers + draft.is_some() as usize {
        return Err(snap_fail("invalid K/V layer count"));
    }
    let upto = image.live.ids.len() as i64;
    for j in 0..layers {
        kv_validate(&image.kv[j], &l.qsa[j], l, upto, true)?;
    }
    if let Some(d) = draft {
        kv_validate(image.kv.last().unwrap(), d, l, upto, false)?;
    }
    Ok(())
}

/// `ConversationRestore`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestoreOutcome {
    Restored,
    Invalid,
    TransferFailed,
}

/// `conversation_snapshot_restore`.
pub fn snapshot_restore(
    image: &SavedConversation,
    l: &P5Layout,
    dev: &P5Device,
    draft: Option<&P5Qsa>,
) -> (RestoreOutcome, String) {
    if let Err(e) = snapshot_validate(image, l, draft) {
        return (RestoreOutcome::Invalid, e);
    }
    if let Err(e) = dev.sync() {
        return (
            RestoreOutcome::TransferFailed,
            format!("conversation snapshot synchronize: {e}"),
        );
    }
    let upto = image.live.ids.len() as i64;
    for j in 0..owned_qsa(l) {
        if let Err(e) = kv_restore(&image.kv[j], &l.qsa[j], l, dev, upto, true, j as i32) {
            return (RestoreOutcome::TransferFailed, e);
        }
    }
    if let Some(d) = draft {
        if let Err(e) = kv_restore(image.kv.last().unwrap(), d, l, dev, upto, false, -1) {
            return (RestoreOutcome::TransferFailed, e);
        }
    }
    if let Err(e) = checkpoint_restore(&image.live, dev, l) {
        return (RestoreOutcome::TransferFailed, e);
    }
    (RestoreOutcome::Restored, String::new())
}

/// `conversation_session_read_limits`.
pub fn session_read_limits<'a>(
    l: &P5Layout,
    max_tokens: u64,
    max_checkpoints: u64,
) -> Result<file::SessionReadLimits<'a>, String> {
    let z = session_sizes(l)?;
    if l.max_cells < 0 {
        return Err(snap_fail("invalid session cells"));
    }
    let tokens = max_tokens.min(l.max_cells as u64);
    let layers = owned_qsa(l);
    let mut lim = file::SessionReadLimits {
        max_tokens: tokens,
        max_checkpoints,
        max_kv_layers: layers as u64 + 1,
        geometry: Some(l.geometry),
        layer_range: Some((l.layer_lo, l.layer_hi)),
        max_state_bytes: [
            z.gdn as u64,
            if l.ple_hist.is_null() {
                0
            } else {
                z.ple as u64
            },
            (layers * z.tail) as u64,
            (layers * z.dead) as u64,
            (layers * z.block_pos) as u64,
        ],
        ..Default::default()
    };
    for st in &l.qsa[..layers] {
        let upto = tokens.min(st.max_cells.max(0) as u64) as i64;
        lim.max_kv_bytes.push(kv_part_sizes(st, l, upto, true)?);
    }
    let d = &l.draft;
    let dupto = tokens.min(d.max_cells.max(0) as u64) as i64;
    lim.max_kv_bytes.push(kv_part_sizes(d, l, dupto, false)?);
    Ok(lim)
}

/// `conversation_snapshot_sources`: the SAVE's capture: the live running state copied, the K/V read straight from
/// the device into the file block.
pub fn snapshot_sources<'a>(
    v: &ConvView,
    l: &'a P5Layout,
    dev: &'a P5Device,
) -> Result<(SavedConversation, Vec<file::KvSource<'a>>), String> {
    view_validate(v, l)?;
    dev.sync()
        .map_err(|e| format!("conversation snapshot synchronize: {e}"))?;
    let mut meta = SavedConversation {
        geometry: l.geometry,
        layer_lo: l.layer_lo,
        layer_hi: l.layer_hi,
        live: Checkpoint {
            ids: v.ids.to_vec(),
            imgs: v.images.to_vec(),
            ..Default::default()
        },
        checkpoints: v.checkpoints.to_vec(),
        kv: Vec::new(),
        cvec: v.cvec,
    };
    checkpoint_save(&mut meta.live, dev, l)?;
    let upto = v.ids.len() as i64;
    let layers = owned_qsa(l);
    let mut out = Vec::with_capacity(layers + 1);
    for i in 0..=layers {
        let (st, index) = if i == layers {
            (&l.draft, false)
        } else {
            (&l.qsa[i], true)
        };
        let kl = kv_layout(st, l, upto, index)?;
        kv_valid(st, &kl, upto)?;
        let pools = st.pools;
        let sizes = kl.sizes;
        out.push(file::KvSource {
            format: kl.format,
            cells: kl.cells,
            heads: l.n_head_kv,
            head_dim: l.head_dim,
            page_size: kl.page_size,
            pooled_rows: kl.pooled_rows,
            idx_dim: l.idx_key_dim,
            sizes,
            read: Box::new(move |part, offset, dst: &mut [u8]| {
                if part >= 5 || offset > sizes[part] || dst.len() > sizes[part] - offset {
                    return Err("K/V read out of range".to_string());
                }
                let src = pools[part];
                let s = if src.is_null() {
                    std::ptr::null()
                } else {
                    unsafe { src.add(offset) as *const u8 }
                };
                kv_transfer(dev, dst.as_mut_ptr(), s, dst.len())
            }),
        });
    }
    Ok((meta, out))
}

// ---------------------------------------------------------------------------------------------------------------
// retention policies
// ---------------------------------------------------------------------------------------------------------------

/// `conv_cache::eviction_victim` (with the tail and pin flags): which checkpoint leaves when the chain is full.
/// The root (index 0) is pinned; a stale tail checkpoint leaves first; pinned prefixes stay.
pub fn eviction_victim(
    stamps: &[u64],
    cap: i64,
    tail: Option<&[bool]>,
    pinned: Option<&[bool]>,
) -> usize {
    let n = stamps.len();
    let lru_from1 = |filter: &dyn Fn(usize) -> bool| -> Option<usize> {
        let mut v: Option<usize> = None;
        for i in 1..n {
            if filter(i) && v.is_none_or(|x| stamps[i] < stamps[x]) {
                v = Some(i);
            }
        }
        v
    };
    let newest = || {
        let mut newest = 0usize;
        for i in 1..n {
            if stamps[i] > stamps[newest] {
                newest = i;
            }
        }
        newest
    };
    match pinned {
        None => {
            if let Some(t) = tail {
                if n >= 2 && cap >= 2 {
                    let nw = newest();
                    for i in 1..n {
                        if t[i] && i != nw {
                            return i;
                        }
                    }
                }
            }
            if cap < 2 || n < 2 {
                return 0;
            }
            lru_from1(&|_| true).unwrap_or(1)
        }
        Some(p) => {
            if cap < 2 || n < 2 {
                return 0;
            }
            let nw = newest();
            if let Some(t) = tail {
                for i in 1..n {
                    if t[i] && !p[i] && i != nw {
                        return i;
                    }
                }
            }
            if let Some(v) = lru_from1(&|i| !p[i]) {
                return v;
            }
            lru_from1(&|_| true).unwrap_or(1)
        }
    }
}

/// `conversation_prefix`: the tokens a checkpoint offers to `prompt` (it must leave the last prompt token).
pub fn conversation_prefix(c: &Checkpoint, prompt: &[i64], images: &[ImageKey]) -> i64 {
    let n = c.ids.len();
    if n == 0 || n >= prompt.len() || c.ids.iter().zip(prompt).any(|(&a, &b)| a as i64 != b) {
        return 0;
    }
    let mut j = 0usize;
    for image in images {
        if image.start >= n as i64 {
            continue;
        }
        if j == c.imgs.len() || c.imgs[j] != *image {
            return 0;
        }
        j += 1;
    }
    if j != c.imgs.len() {
        return 0;
    }
    n as i64
}

/// `ConversationCache::Match`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheMatch {
    pub index: usize,
    pub tokens: i64,
    pub live: bool,
}

/// `ConversationCache`: parked conversations, least recently active first, under a byte budget and a slot count.
#[derive(Debug, Default)]
pub struct ConversationCache {
    budget: usize,
    slots: usize,
    min_tokens: i64,
    bytes: usize,
    evictions: usize,
    superseded: usize,
    entries: std::collections::VecDeque<SavedConversation>,
    reuse: KvReuse,
}

impl ConversationCache {
    pub fn new(budget: usize, slots: usize, min_tokens: i64) -> Self {
        Self {
            budget,
            slots,
            min_tokens,
            ..Default::default()
        }
    }
    pub fn enabled(&self) -> bool {
        self.budget != 0 && self.slots != 0
    }
    pub fn wants(&self, tokens: i64) -> bool {
        self.min_tokens <= 0 || tokens >= self.min_tokens
    }
    pub fn bytes(&self) -> usize {
        self.bytes + self.reuse.bytes()
    }
    pub fn size(&self) -> usize {
        self.entries.len()
    }
    pub fn evictions(&self) -> usize {
        self.evictions
    }
    pub fn slots(&self) -> usize {
        self.slots
    }
    pub fn superseded(&self) -> usize {
        self.superseded
    }
    pub fn longest_tokens(&self) -> i64 {
        self.entries
            .iter()
            .map(|e| e.live.ids.len() as i64)
            .max()
            .unwrap_or(0)
    }
    pub fn retain(&mut self, kv: Vec<ConvKv>, tokens: i64) {
        self.reuse = KvReuse::default();
        let cand = KvReuse {
            kv,
            captured_tokens: tokens,
            unchanged_tokens: tokens,
        };
        if self.enabled() && cand.bytes() <= self.budget.saturating_sub(self.bytes) {
            self.reuse = cand;
        }
    }
    pub fn limit_reuse(&mut self, first_dirty: i64) {
        self.reuse.unchanged_tokens = self.reuse.unchanged_tokens.min(first_dirty);
        if self.reuse.unchanged_tokens <= 0 {
            self.reuse = KvReuse::default();
        }
    }
    pub fn take_reuse(&mut self) -> KvReuse {
        std::mem::take(&mut self.reuse)
    }
    pub fn retained_bytes(&self) -> usize {
        self.reuse.bytes()
    }
    pub fn best(&self, prompt: &[i64], images: &[ImageKey], cvec: bool) -> CacheMatch {
        let mut best = CacheMatch::default();
        for i in (0..self.entries.len()).rev() {
            let e = &self.entries[i];
            if e.cvec != cvec {
                continue;
            }
            let mut consider = |c: &Checkpoint, live: bool| {
                let n = conversation_prefix(c, prompt, images);
                if n > best.tokens {
                    best = CacheMatch {
                        index: i,
                        tokens: n,
                        live,
                    };
                }
            };
            consider(&e.live, true);
            for c in &e.checkpoints {
                consider(c, false);
            }
        }
        best
    }
    pub fn take(&mut self, index: usize) -> SavedConversation {
        let out = self
            .entries
            .remove(index)
            .expect("parked conversation index");
        self.bytes -= out.bytes();
        out
    }
    pub fn make_room(&mut self, incoming: usize, held: usize) -> bool {
        if !self.enabled() || held > self.budget || incoming > self.budget - held {
            return false;
        }
        if self.bytes() > self.budget - held - incoming {
            self.reuse = KvReuse::default();
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= self.slots || self.bytes > self.budget - held - incoming)
        {
            if !self.evict_oldest() {
                return false;
            }
        }
        true
    }
    pub fn evict_oldest(&mut self) -> bool {
        let Some(i) = self.entries.iter().position(|e| !e.pinned()) else {
            return false;
        };
        let e = self.entries.remove(i).unwrap();
        self.bytes -= e.bytes();
        self.evictions += 1;
        true
    }
    pub fn clear(&mut self) -> usize {
        let n = self.entries.len();
        self.entries.clear();
        self.bytes = 0;
        self.reuse = KvReuse::default();
        n
    }
    pub fn drop_superseded(
        &mut self,
        ids: &[i32],
        images: &[ImageKey],
        checkpoints: &[Checkpoint],
        cvec: bool,
    ) -> usize {
        let held = |c: &Checkpoint| {
            (c.ids == ids && c.imgs == images)
                || checkpoints
                    .iter()
                    .any(|k| k.ids == c.ids && k.imgs == c.imgs)
        };
        let mut dropped = 0usize;
        let mut i = 0usize;
        while i < self.entries.len() {
            let e = &self.entries[i];
            let mut deepest: Option<&Checkpoint> = None;
            for c in &e.checkpoints {
                if deepest.is_none_or(|d| c.ids.len() > d.ids.len()) {
                    deepest = Some(c);
                }
            }
            if e.cvec == cvec && deepest.is_some_and(|d| !d.ids.is_empty() && held(d)) {
                let e = self.entries.remove(i).unwrap();
                self.bytes -= e.bytes();
                dropped += 1;
                continue;
            }
            i += 1;
        }
        self.superseded += dropped;
        dropped
    }
    pub fn put(&mut self, image: SavedConversation, held: usize) -> bool {
        let n = image.bytes();
        if !self.enabled() || held > self.budget || n > self.budget - held {
            return false;
        }
        self.drop_superseded(
            &image.live.ids,
            &image.live.imgs,
            &image.checkpoints,
            image.cvec,
        );
        if !self.make_room(n, held) {
            return false;
        }
        self.entries.push_back(image);
        self.bytes += n;
        true
    }
}

/// `conversation_memory_admit`.
pub fn memory_admit(available: Option<u64>, allocation: u64, floor: u64) -> bool {
    matches!(available, Some(a) if a >= floor && allocation <= a - floor)
}

/// `SessionSaveReclaimed`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaveReclaimed {
    pub checkpoints: usize,
    pub parked: usize,
    pub bytes: usize,
}

/// `session_save_reclaim`: frees reconstructible host caches (retained K/V, parked conversations, then checkpoints
/// other than the one the file keeps and the pinned ones) until the SAVE's peak fits beside `floor`.
pub fn session_save_reclaim(
    chain: &mut Vec<Checkpoint>,
    cache: &mut ConversationCache,
    live: &file::SessionSaveLive,
    floor: u64,
    mut available: impl FnMut() -> Option<u64>,
) -> SaveReclaimed {
    let mut out = SaveReclaimed::default();
    let selected = file::session_deepest_checkpoint(chain).map(|c| c as *const Checkpoint);
    let need = file::session_save_peak_bytes(selected.map(|p| unsafe { &*p }), live);
    let mut room = available();
    let done =
        |room: Option<u64>| room.is_none() || need == u64::MAX || memory_admit(room, need, floor);
    if done(room) {
        return out;
    }
    if cache.retained_bytes() > 0 {
        out.bytes += cache.retained_bytes();
        drop(cache.take_reuse());
        room = available();
        if done(room) {
            return out;
        }
    }
    loop {
        let before = cache.bytes();
        if !cache.evict_oldest() {
            break;
        }
        out.parked += 1;
        out.bytes += before - cache.bytes();
        room = available();
        if done(room) {
            return out;
        }
    }
    let mut keep = selected
        .and_then(|p| chain.iter().position(|c| std::ptr::eq(c, p)))
        .unwrap_or(chain.len());
    let mut i = chain.len();
    while i > 0 {
        i -= 1;
        if i == keep || chain[i].pinned {
            continue;
        }
        out.bytes += chain[i].bytes();
        chain.remove(i);
        if i < keep {
            keep -= 1;
        }
        out.checkpoints += 1;
        room = available();
        if done(room) {
            break;
        }
    }
    out
}

pub mod memory;

// ---------------------------------------------------------------------------------------------------------------
// Piece 5 state owned by the engine
// ---------------------------------------------------------------------------------------------------------------

/// Host-side checkpoint metadata (for reports).
#[derive(Clone, Debug)]
pub struct CheckpointMeta {
    pub token_len: usize,
    pub pinned: bool,
    pub is_tail: bool,
    pub host_bytes: u64,
}

/// Piece 5 multi-turn conversation state manager owned by `Qwen4ExpRustEngine`.
#[derive(Debug)]
pub struct Piece5ConvState {
    pub prompt_cache_cap: usize,
    pub min_free_ram_mib: u64,
    pub state_sizes: TkQwen4ExpConvStateSizes,
    pub checkpoints: Vec<CheckpointMeta>,
    /// The serve-side conversation state once the C++ serve lambdas hand it over ([`serve::ConvServe`]).
    pub serve: Option<Box<serve::ConvServe>>,
}

impl Clone for Piece5ConvState {
    fn clone(&self) -> Self {
        Self {
            prompt_cache_cap: self.prompt_cache_cap,
            min_free_ram_mib: self.min_free_ram_mib,
            state_sizes: self.state_sizes,
            checkpoints: self.checkpoints.clone(),
            serve: None,
        }
    }
}

impl Piece5ConvState {
    pub fn new(_cfg: &Qwen4ExpModelConfig) -> Self {
        Self {
            prompt_cache_cap: 8,
            min_free_ram_mib: 4096,
            state_sizes: TkQwen4ExpConvStateSizes::default(),
            checkpoints: Vec::new(),
            serve: None,
        }
    }

    /// Queries the per-checkpoint GDN, PLE and QSA tail byte sizes of the active session.
    pub unsafe fn query_state_sizes(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
    ) -> Result<TkQwen4ExpConvStateSizes, String> {
        let mut sizes = TkQwen4ExpConvStateSizes::default();
        if (kernels.p5_state_sizes)(ctx, &mut sizes) < 0 {
            return Err("tk_qwen4exp_p5_state_sizes failed".to_string());
        }
        self.state_sizes = sizes;
        Ok(sizes)
    }

    /// Records a checkpoint at `pos` through the serve context (routes to [`serve`] once attached).
    pub unsafe fn checkpoint_at(
        &mut self,
        kernels: &Qwen4ExpKernels,
        ctx: *mut TkQwen4ExpServeCtx,
        pos: i64,
        as_tail: bool,
        is_pin: bool,
        turn_at: i64,
    ) -> Result<(), String> {
        if (kernels.serve_checkpoint_at)(ctx, pos, as_tail as i32, is_pin as i32, turn_at) < 0 {
            return Err(format!("checkpoint_at failed at pos={pos}"));
        }
        Ok(())
    }

    /// Hands the `--serve` conversation state to Rust: registers the hooks the C++ serve lambdas call.
    pub unsafe fn attach_serve(&mut self, ctx: *mut TkQwen4ExpServeCtx) -> Result<(), String> {
        let mut s = serve::ConvServe::resolve(ctx)?;
        let hooks = serve::hooks_for(&mut s);
        let reg = s.api.register;
        self.serve = Some(s);
        let p = self.serve.as_mut().unwrap();
        let hooks = serve::P5Hooks {
            user: (&mut **p as *mut serve::ConvServe).cast(),
            ..hooks
        };
        if reg(ctx, &hooks) < 0 {
            self.serve = None;
            return Err("tk_qwen4exp_p5_register_hooks failed".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
