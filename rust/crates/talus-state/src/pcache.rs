//! Multi-tier recurrent + KV prompt cache (`PromptCache`, `Snap`, `Stash`, `EngineStateView`).
//!
//! Bit-for-bit Rust port of the Qwen3.5 / Qwen3.6 (`q36_engine.cu`) prompt cache:
//! - Device recurrent state snapshots (`Snap` in `snaps` and message-boundary `msg` pools).
//! - Host (or unified memory) recurrent + KV stashes (`Stash` in `stashes` pool with byte budget).
//! - Exact 1D and 2D pitched asynchronous HIP copies (`state_copy`, `kv_copy`) synchronized on
//!   the engine's main stream (`s_main`).
//! - Message-boundary cutting helper for `<|im_start|>` (`248045`) not followed by `tool` (`74455`).

use core::ffi::{c_char, c_void};
use std::ptr;
use std::sync::OnceLock;

use hip_sys::{
    hipErrorInvalidPitchValue, hipErrorInvalidValue, hipError_t, hipFree, hipMalloc,
    hipMemcpyAsync, hipMemcpyDefault, hipMemcpyKind, hipStreamSynchronize, hipStream_t, hipSuccess,
    HipRuntime,
};

/// Token ID for `<|im_start|>` in the Qwen3.5 / Qwen3.6 vocabulary (`248045`).
pub const IM_START_TOKEN_ID: i32 = 248045;

/// Token ID for `tool` role following `<|im_start|>` in the Qwen3.5 / Qwen3.6 vocabulary (`74455`).
pub const TOOL_ROLE_TOKEN_ID: i32 = 74455;

/// Default capacity of the message-boundary snapshot pool (`max_msg = 4`).
pub const DEFAULT_MAX_MSG: usize = 4;

/// Default capacity of the turn-end snapshot pool (`max_snaps = 8`).
pub const DEFAULT_MAX_SNAPS: usize = 8;

/// Default host stash byte budget (`2 GiB`).
pub const DEFAULT_STASH_BUDGET: usize = 2usize << 30;

/// Default minimum fresh-token distance for taking a message-boundary snapshot (`1024`).
pub const DEFAULT_SNAP_MSG_MIN: usize = 1024;

// ============================================================================
// Dynamic `hipMemcpy2DAsync` Resolution + Simulated 2D Pitched Fallback
// ============================================================================

type FnHipMemcpy2DAsync = unsafe extern "C" fn(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: hipMemcpyKind,
    stream: hipStream_t,
) -> hipError_t;

#[cfg(unix)]
extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

#[cfg(unix)]
const RTLD_DEFAULT: *mut c_void = ptr::null_mut();

fn resolve_hip_memcpy_2d_async() -> Option<FnHipMemcpy2DAsync> {
    static SYM: OnceLock<Option<FnHipMemcpy2DAsync>> = OnceLock::new();
    *SYM.get_or_init(|| {
        #[cfg(unix)]
        unsafe {
            let sym = dlsym(RTLD_DEFAULT, c"hipMemcpy2DAsync".as_ptr());
            if sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, FnHipMemcpy2DAsync>(sym))
            }
        }
        #[cfg(not(unix))]
        {
            None
        }
    })
}

/// Asynchronous 2D pitched memory copy.
///
/// Uses the native `hipMemcpy2DAsync` symbol from `libamdhip64.so` when running on the dynamic
/// ROCm backend, and falls back to exact row-by-row `hipMemcpyAsync` on the simulated backend
/// (or if `hipMemcpy2DAsync` is unavailable).
///
/// # Safety
/// `dst` and `src` must point to valid allocations of at least `dpitch * height` and
/// `spitch * height` bytes respectively (or tightly packed for the last row).
#[allow(clippy::too_many_arguments)]
pub unsafe fn hip_memcpy_2d_async(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: hipMemcpyKind,
    stream: hipStream_t,
) -> hipError_t {
    if width == 0 || height == 0 {
        return hipSuccess;
    }
    if dst.is_null() || src.is_null() {
        return hipErrorInvalidValue;
    }
    if width > dpitch || width > spitch {
        return hipErrorInvalidPitchValue;
    }

    if HipRuntime::global().is_dynamic() {
        if let Some(f) = resolve_hip_memcpy_2d_async() {
            let status = f(dst, dpitch, src, spitch, width, height, kind, stream);
            if status.is_success() {
                return status;
            }
        }
    }

    if dpitch == width && spitch == width {
        return hipMemcpyAsync(dst, src, width * height, kind, stream);
    }

    let dst_u8 = dst.cast::<u8>();
    let src_u8 = src.cast::<u8>();
    for r in 0..height {
        let status = hipMemcpyAsync(
            dst_u8.add(r * dpitch).cast::<c_void>(),
            src_u8.add(r * spitch).cast::<c_void>(),
            width,
            kind,
            stream,
        );
        if !status.is_success() {
            return status;
        }
    }
    hipSuccess
}

#[inline]
fn check_hip(status: hipError_t, op: &str) {
    assert!(
        status.is_success(),
        "talus-state: {op} failed with HIP error {} ({})",
        status.0,
        hip_sys::error_string(status)
    );
}

// ============================================================================
// `EngineStateView` & Byte-Size / Copy Helpers (`q36_engine.cu:2257-2377`)
// ============================================================================

/// Lightweight engine state descriptor allowing [`PromptCache`] to snapshot and restore
/// `Q36Engine` recurrent + KV state without a circular dependency on `talus-model`.
#[derive(Debug, Clone, Copy)]
pub struct EngineStateView {
    pub n_gdn: usize,
    pub n_attn: usize,
    pub gdn_h: i32,
    pub gdn_d: i32,
    pub gdn_s: i32,
    pub gdn_q: i32,
    pub gdn_c: i32,
    pub n_kv: i32,
    pub d_head: i32,
    pub max_seq: i32,
    pub d_model: i32,
    pub use_bf16_kv: bool,
    pub gdn_state: *mut f32,
    pub conv_state: *mut f32,
    pub k_cache: *mut c_void,
    pub v_cache: *mut c_void,
    pub h_mtp_prev: *mut f32,
    pub x: *mut f32,
    pub s_main: hipStream_t,
    pub n_past: i32,
    pub gdn_valid: bool,
}

// SAFETY: Device and stream pointers in `EngineStateView` are synchronized through `s_main`.
unsafe impl Send for EngineStateView {}
unsafe impl Sync for EngineStateView {}

impl Default for EngineStateView {
    fn default() -> Self {
        Self {
            n_gdn: 0,
            n_attn: 0,
            gdn_h: 0,
            gdn_d: 0,
            gdn_s: 0,
            gdn_q: 0,
            gdn_c: 0,
            n_kv: 0,
            d_head: 0,
            max_seq: 0,
            d_model: 0,
            use_bf16_kv: false,
            gdn_state: ptr::null_mut(),
            conv_state: ptr::null_mut(),
            k_cache: ptr::null_mut(),
            v_cache: ptr::null_mut(),
            h_mtp_prev: ptr::null_mut(),
            x: ptr::null_mut(),
            s_main: ptr::null_mut(),
            n_past: 0,
            gdn_valid: false,
        }
    }
}

impl EngineStateView {
    /// Byte size of the GDN recurrent state tensor (`n_gdn * gdn_h * gdn_d * gdn_s * 4`).
    #[inline]
    pub const fn gdn_bytes(&self) -> usize {
        let h = if self.gdn_h > 0 {
            self.gdn_h as usize
        } else {
            0
        };
        let d = if self.gdn_d > 0 {
            self.gdn_d as usize
        } else {
            0
        };
        let s = if self.gdn_s > 0 {
            self.gdn_s as usize
        } else {
            0
        };
        self.n_gdn * h * d * s * 4
    }

    /// Byte size of the causal conv state tensor (`n_gdn * (gdn_c - 1) * gdn_q * 4`).
    #[inline]
    pub const fn conv_bytes(&self) -> usize {
        let c_minus_1 = if self.gdn_c > 1 {
            (self.gdn_c - 1) as usize
        } else {
            0
        };
        let q = if self.gdn_q > 0 {
            self.gdn_q as usize
        } else {
            0
        };
        self.n_gdn * c_minus_1 * q * 4
    }

    /// Byte size of the MTP previous hidden state (`d_model * 4`).
    #[inline]
    pub const fn mtp_bytes(&self) -> usize {
        let dm = if self.d_model > 0 {
            self.d_model as usize
        } else {
            0
        };
        dm * 4
    }

    /// Byte size of the last residual / hidden activation `x` (`d_model * 4`).
    #[inline]
    pub const fn x_bytes(&self) -> usize {
        let dm = if self.d_model > 0 {
            self.d_model as usize
        } else {
            0
        };
        dm * 4
    }
}

/// Round sequence length `l` up to the next multiple of 16 for transposed FP16 V tiles.
///
/// Exact port of `vt_round(int l)` (`(l + 15) / 16 * 16`).
#[inline]
pub const fn vt_round(l: i32) -> i32 {
    (l + 15) / 16 * 16
}

/// Total recurrent + activation state snapshot size in bytes:
/// `gdn_bytes + conv_bytes + mtp_bytes + x_bytes`.
#[inline]
pub const fn state_bytes(eng: &EngineStateView) -> usize {
    eng.gdn_bytes() + eng.conv_bytes() + eng.mtp_bytes() + eng.x_bytes()
}

/// Copy recurrent and hidden state buffers between `eng` and `buf` on `eng.s_main`.
///
/// Copies in exact order:
/// 1. `gdn_state` (`gdn_bytes`)
/// 2. `conv_state` (`conv_bytes`)
/// 3. `h_mtp_prev` (`mtp_bytes`)
/// 4. `x` (`x_bytes`)
///
/// followed by `hipStreamSynchronize(eng.s_main)`.
pub fn state_copy(eng: &EngineStateView, buf: *mut u8, to_cache: bool) {
    if buf.is_null() {
        return;
    }

    let gdn_b = eng.gdn_bytes();
    let conv_b = eng.conv_bytes();
    let mtp_b = eng.mtp_bytes();
    let x_b = eng.x_bytes();

    let segments: [(*mut u8, usize); 4] = [
        (eng.gdn_state.cast::<u8>(), gdn_b),
        (eng.conv_state.cast::<u8>(), conv_b),
        (eng.h_mtp_prev.cast::<u8>(), mtp_b),
        (eng.x.cast::<u8>(), x_b),
    ];

    unsafe {
        let mut p = buf;
        for (eng_ptr, nbytes) in segments {
            if nbytes > 0 && !eng_ptr.is_null() {
                let (dst, src) = if to_cache {
                    (p.cast::<c_void>(), eng_ptr.cast_const().cast::<c_void>())
                } else {
                    (eng_ptr.cast::<c_void>(), p.cast_const().cast::<c_void>())
                };
                let st = hipMemcpyAsync(dst, src, nbytes, hipMemcpyDefault, eng.s_main);
                check_hip(st, "state_copy hipMemcpyAsync");
            }
            p = p.add(nbytes);
        }
        let st = hipStreamSynchronize(eng.s_main);
        check_hip(st, "state_copy hipStreamSynchronize");
    }
}

/// Total KV cache byte size for prefix length `l`:
/// - `row_bytes = n_kv * d_head * 2`
/// - `k_bytes = n_attn * l * row_bytes`
/// - `v_bytes = if eng.use_bf16_kv { k_bytes } else { n_attn * d_head * n_kv * vt_round(l) * 2 }`
/// - Total = `k_bytes + v_bytes`.
#[inline]
pub const fn kv_bytes(eng: &EngineStateView, l: i32) -> usize {
    if l <= 0 || eng.n_attn == 0 || eng.n_kv <= 0 || eng.d_head <= 0 {
        return 0;
    }
    let n_kv = eng.n_kv as usize;
    let d_head = eng.d_head as usize;
    let l_usize = l as usize;
    let row_bytes = n_kv * d_head * 2;
    let k_bytes = eng.n_attn * l_usize * row_bytes;
    let v_bytes = if eng.use_bf16_kv {
        k_bytes
    } else {
        eng.n_attn * d_head * n_kv * (vt_round(l) as usize) * 2
    };
    k_bytes + v_bytes
}

/// Copy `l` positions of K and V cache between `eng` and `buf` on `eng.s_main`, then synchronize.
///
/// - For `ai in 0..n_attn`, copies `l * row_bytes` between
///   `(eng.k_cache as *mut u8).add(ai * max_seq * row_bytes)` and `p`.
/// - If `eng.use_bf16_kv`, for `ai in 0..n_attn`, copies `l * row_bytes` between
///   `(eng.v_cache as *mut u8).add(ai * max_seq * row_bytes)` and `p`.
/// - Otherwise (transposed FP16 V cache `[n_attn, d_head, n_kv, max_seq]`):
///   `pitch_eng = max_seq * 2`, `pitch_buf = vt_round(l) * 2`, `rows = n_attn * d_head * n_kv`,
///   and copies via `hipMemcpy2DAsync` on `eng.s_main`.
/// - Synchronizes `eng.s_main`.
pub fn kv_copy(eng: &EngineStateView, buf: *mut u8, l: i32, to_cache: bool) {
    if buf.is_null() || l <= 0 || eng.n_attn == 0 || eng.n_kv <= 0 || eng.d_head <= 0 {
        return;
    }

    let n_kv = eng.n_kv as usize;
    let d_head = eng.d_head as usize;
    let l_usize = l as usize;
    let max_seq = eng.max_seq.max(l) as usize;
    let row_bytes = n_kv * d_head * 2;
    let layer_k_bytes = l_usize * row_bytes;
    let layer_stride_bytes = max_seq * row_bytes;

    unsafe {
        let mut p = buf;

        // K cache: per attention layer linear slice of `l * row_bytes`.
        let k_base = eng.k_cache.cast::<u8>();
        for ai in 0..eng.n_attn {
            if !k_base.is_null() && layer_k_bytes > 0 {
                let eng_k = k_base.add(ai * layer_stride_bytes);
                let (dst, src) = if to_cache {
                    (p.cast::<c_void>(), eng_k.cast_const().cast::<c_void>())
                } else {
                    (eng_k.cast::<c_void>(), p.cast_const().cast::<c_void>())
                };
                let st = hipMemcpyAsync(dst, src, layer_k_bytes, hipMemcpyDefault, eng.s_main);
                check_hip(st, "kv_copy K hipMemcpyAsync");
            }
            p = p.add(layer_k_bytes);
        }

        // V cache: either BF16 non-transposed rows or FP16 transposed 2D pitched layout.
        let v_base = eng.v_cache.cast::<u8>();
        if eng.use_bf16_kv {
            for ai in 0..eng.n_attn {
                if !v_base.is_null() && layer_k_bytes > 0 {
                    let eng_v = v_base.add(ai * layer_stride_bytes);
                    let (dst, src) = if to_cache {
                        (p.cast::<c_void>(), eng_v.cast_const().cast::<c_void>())
                    } else {
                        (eng_v.cast::<c_void>(), p.cast_const().cast::<c_void>())
                    };
                    let st = hipMemcpyAsync(dst, src, layer_k_bytes, hipMemcpyDefault, eng.s_main);
                    check_hip(st, "kv_copy V (bf16) hipMemcpyAsync");
                }
                p = p.add(layer_k_bytes);
            }
        } else {
            let pitch_eng = (eng.max_seq as usize) * 2;
            let pitch_buf = (vt_round(l) as usize) * 2;
            let rows = eng.n_attn * d_head * n_kv;
            let width = pitch_buf.min(pitch_eng);
            if !v_base.is_null() && rows > 0 && width > 0 {
                let (dst, dpitch, src, spitch) = if to_cache {
                    (
                        p.cast::<c_void>(),
                        pitch_buf,
                        v_base.cast_const().cast::<c_void>(),
                        pitch_eng,
                    )
                } else {
                    (
                        v_base.cast::<c_void>(),
                        pitch_eng,
                        p.cast_const().cast::<c_void>(),
                        pitch_buf,
                    )
                };
                let st = hip_memcpy_2d_async(
                    dst,
                    dpitch,
                    src,
                    spitch,
                    width,
                    rows,
                    hipMemcpyDefault,
                    eng.s_main,
                );
                check_hip(st, "kv_copy V (transposed 2D) hipMemcpy2DAsync");
            }
        }

        let st = hipStreamSynchronize(eng.s_main);
        check_hip(st, "kv_copy hipStreamSynchronize");
    }
}

// ============================================================================
// Message-Boundary Cutting Helper (`q36_engine.cu:2425-2443`)
// ============================================================================

/// Find the latest message-start boundary `<|im_start|>` (`248045`) not followed by `tool`
/// (`74455`) at or beyond `reuse + snap_msg_min`.
///
/// Exact port of `q36_engine.cu` message-boundary chunking logic:
/// - Scans `i` from `reuse + 1` to `prompt.len() - 1`:
///   if `prompt[i] == 248045 && (i + 1 >= prompt.len() || prompt[i + 1] != 74455)`, sets `cut = i`.
/// - Returns `Some(cut)` if `cut > reuse && cut - reuse >= snap_msg_min`, else `None`.
pub fn find_message_boundary(prompt: &[i32], reuse: usize, snap_msg_min: usize) -> Option<usize> {
    if prompt.len() <= reuse + 1 {
        return None;
    }
    let mut cut: usize = 0;
    for i in (reuse + 1)..prompt.len() {
        if prompt[i] == IM_START_TOKEN_ID
            && (i + 1 >= prompt.len() || prompt[i + 1] != TOOL_ROLE_TOKEN_ID)
        {
            cut = i;
        }
    }
    if cut > reuse && (cut - reuse) >= snap_msg_min {
        Some(cut)
    } else {
        None
    }
}

/// Alias for [`find_message_boundary`].
#[inline]
pub fn find_msg_cut(prompt: &[i32], reuse: usize, snap_msg_min: usize) -> Option<usize> {
    find_message_boundary(prompt, reuse, snap_msg_min)
}

// ============================================================================
// `Snap`, `Stash`, and `PromptCache` (`q36_engine.cu:2303-2377`)
// ============================================================================

/// Device-resident recurrent + hidden state snapshot for a token prefix.
#[derive(Debug)]
pub struct Snap {
    pub tokens: Vec<i32>,
    pub gdn_valid: bool,
    pub buf: *mut u8,
}

// SAFETY: `buf` is a device allocation owned by `PromptCache` and accessed only via synchronized HIP copies.
unsafe impl Send for Snap {}
unsafe impl Sync for Snap {}

/// Host-resident recurrent + KV cache stash entry for conversation switching.
#[derive(Debug, Clone)]
pub struct Stash {
    pub tokens: Vec<i32>,
    pub gdn_valid: bool,
    pub state_buf: Vec<u8>,
    pub kv_buf: Vec<u8>,
}

impl Stash {
    /// Total host memory bytes held by this stash (`state_buf.len() + kv_buf.len()`).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.state_buf.len() + self.kv_buf.len()
    }
}

/// Structured outcome returned by [`PromptCache::restore_outcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RestoreOutcome {
    /// Number of prefix tokens restored (`0` on miss).
    pub reused_tokens: i32,
    /// Restored `gdn_valid` flag from the matched snapshot or stash.
    pub gdn_valid: bool,
    /// `true` if restored from the host `stashes` pool (KV + state), `false` if from `snaps`/`msg`.
    pub from_stash: bool,
}

/// Trait implemented by `&EngineStateView`, `&mut EngineStateView`, and `EngineStateView` so
/// [`PromptCache::restore`] accepts shared references, mutable references (updating `n_past` and
/// `gdn_valid` in place), or values seamlessly.
pub trait EngineStateTarget {
    fn as_view(&self) -> &EngineStateView;
    fn update_restored(&mut self, n_past: i32, gdn_valid: bool);
}

impl EngineStateTarget for EngineStateView {
    #[inline]
    fn as_view(&self) -> &EngineStateView {
        self
    }

    #[inline]
    fn update_restored(&mut self, n_past: i32, gdn_valid: bool) {
        self.n_past = n_past;
        self.gdn_valid = gdn_valid;
    }
}

impl EngineStateTarget for &EngineStateView {
    #[inline]
    fn as_view(&self) -> &EngineStateView {
        self
    }

    #[inline]
    fn update_restored(&mut self, _n_past: i32, _gdn_valid: bool) {}
}

impl EngineStateTarget for &mut EngineStateView {
    #[inline]
    fn as_view(&self) -> &EngineStateView {
        self
    }

    #[inline]
    fn update_restored(&mut self, n_past: i32, gdn_valid: bool) {
        self.n_past = n_past;
        self.gdn_valid = gdn_valid;
    }
}

/// Multi-tier recurrent + KV prompt cache (`msg` boundary snaps, turn-end `snaps`, and `stashes`).
#[derive(Debug)]
pub struct PromptCache {
    /// Message-boundary snapshots (`<|im_start|>` before final message).
    pub msg: Vec<Snap>,
    /// Turn-end / prompt-end snapshots.
    pub snaps: Vec<Snap>,
    /// Host-resident state + KV stashes for switching between distinct conversations.
    pub stashes: Vec<Stash>,
    /// Maximum number of message-boundary snapshots (`DEFAULT_MAX_MSG = 4`).
    pub max_msg: usize,
    /// Maximum number of turn-end snapshots (`DEFAULT_MAX_SNAPS = 8`).
    pub max_snaps: usize,
    /// Maximum total bytes across all entries in `stashes` (`DEFAULT_STASH_BUDGET = 2 GiB`).
    pub stash_budget: usize,
    /// Metadata recorded via [`talus_core_api::PrefixCache::put`].
    pub(crate) meta_entries: Vec<crate::prefix_cache::PrefixMetaEntry>,
    pub(crate) next_handle_id: u64,
}

impl Default for PromptCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptCache {
    /// Create a `PromptCache` with default capacities (`max_msg = 4`, `max_snaps = 8`, `stash_budget = 2 GiB`).
    pub fn new() -> Self {
        Self {
            msg: Vec::new(),
            snaps: Vec::new(),
            stashes: Vec::new(),
            max_msg: DEFAULT_MAX_MSG,
            max_snaps: DEFAULT_MAX_SNAPS,
            stash_budget: DEFAULT_STASH_BUDGET,
            meta_entries: Vec::new(),
            next_handle_id: 1,
        }
    }

    /// Create a `PromptCache` with custom `snap_slots`, `snap_msg_slots`, and `stash_budget` (in bytes).
    pub fn with_config(snap_slots: usize, snap_msg_slots: usize, stash_budget: usize) -> Self {
        Self {
            msg: Vec::new(),
            snaps: Vec::new(),
            stashes: Vec::new(),
            max_msg: snap_msg_slots,
            max_snaps: snap_slots,
            stash_budget,
            meta_entries: Vec::new(),
            next_handle_id: 1,
        }
    }

    /// Create a `PromptCache` with custom `snap_slots` and `snap_msg_slots` and the default 2 GiB stash budget.
    pub fn with_slots(snap_slots: usize, snap_msg_slots: usize) -> Self {
        Self::with_config(snap_slots, snap_msg_slots, DEFAULT_STASH_BUDGET)
    }

    /// Allocate a GPU buffer of `state_bytes(eng)` bytes via `hipMalloc`.
    pub fn alloc_buf(eng: &EngineStateView) -> *mut u8 {
        let bytes = state_bytes(eng);
        if bytes == 0 {
            return ptr::null_mut();
        }
        let mut raw: *mut c_void = ptr::null_mut();
        let st = unsafe { hipMalloc(&mut raw, bytes) };
        check_hip(st, "PromptCache::alloc_buf hipMalloc");
        raw.cast::<u8>()
    }

    /// Total bytes currently used in the `stashes` pool.
    pub fn stash_used_bytes(&self) -> usize {
        self.stashes.iter().map(Stash::byte_len).sum()
    }

    /// Save a recurrent state snapshot into `msg` (`is_msg == true`) or `snaps` (`is_msg == false`).
    ///
    /// - If `pool` already contains an entry with `s.tokens == toks`, overwrites that slot in place.
    /// - Else if `pool.len() < cap`, allocates a new device buffer via [`Self::alloc_buf`] and appends.
    /// - Else evicts `pool[0]` (FIFO rotation reusing `pool[0].buf` at the back) and overwrites it.
    pub fn save(&mut self, eng: &EngineStateView, toks: &[i32], is_msg: bool) {
        let cap = if is_msg { self.max_msg } else { self.max_snaps };
        if cap == 0 || toks.is_empty() {
            return;
        }
        let pool = if is_msg {
            &mut self.msg
        } else {
            &mut self.snaps
        };

        // 1. Deduplicate if an entry already matches `toks` exactly.
        for s in pool.iter_mut() {
            if s.tokens.len() == toks.len() && s.tokens.as_slice() == toks {
                s.gdn_valid = eng.gdn_valid;
                state_copy(eng, s.buf, true);
                return;
            }
        }

        // 2. Allocate a fresh slot if under capacity.
        if pool.len() < cap {
            let buf = Self::alloc_buf(eng);
            state_copy(eng, buf, true);
            pool.push(Snap {
                tokens: toks.to_vec(),
                gdn_valid: eng.gdn_valid,
                buf,
            });
            return;
        }

        // 3. FIFO eviction: move `pool[0]` to the back and reuse its GPU buffer.
        pool.rotate_left(1);
        if let Some(evicted) = pool.last_mut() {
            evicted.tokens.clear();
            evicted.tokens.extend_from_slice(toks);
            evicted.gdn_valid = eng.gdn_valid;
            if evicted.buf.is_null() && state_bytes(eng) > 0 {
                evicted.buf = Self::alloc_buf(eng);
            }
            state_copy(eng, evicted.buf, true);
        }
    }

    /// Save both recurrent state and KV cache for `toks` into the host `stashes` pool.
    ///
    /// - Returns immediately if `stash_budget == 0 || toks.is_empty()` or `need > stash_budget`.
    /// - Reuses an existing stash slot if `st.tokens == toks`.
    /// - Otherwise evicts from the front (`stashes.remove(0)`) until `used + need <= stash_budget`,
    ///   then pushes a new `Stash` and copies state + KV.
    pub fn stash(&mut self, eng: &EngineStateView, toks: &[i32]) {
        if self.stash_budget == 0 || toks.is_empty() {
            return;
        }
        let s_bytes = state_bytes(eng);
        let k_bytes = kv_bytes(eng, toks.len() as i32);
        let need = s_bytes + k_bytes;
        if need > self.stash_budget {
            return;
        }

        // Check if an entry in `stashes` already matches `toks`.
        if let Some(st) = self
            .stashes
            .iter_mut()
            .find(|st| st.tokens.len() == toks.len() && st.tokens.as_slice() == toks)
        {
            st.gdn_valid = eng.gdn_valid;
            st.state_buf.resize(s_bytes, 0);
            st.kv_buf.resize(k_bytes, 0);
            state_copy(eng, st.state_buf.as_mut_ptr(), true);
            kv_copy(eng, st.kv_buf.as_mut_ptr(), toks.len() as i32, true);
            return;
        }

        let mut used: usize = self.stash_used_bytes();
        while used + need > self.stash_budget && !self.stashes.is_empty() {
            let evicted = self.stashes.remove(0);
            used = used.saturating_sub(evicted.byte_len());
        }

        let mut st = Stash {
            tokens: toks.to_vec(),
            gdn_valid: eng.gdn_valid,
            state_buf: vec![0u8; s_bytes],
            kv_buf: vec![0u8; k_bytes],
        };
        state_copy(eng, st.state_buf.as_mut_ptr(), true);
        kv_copy(eng, st.kv_buf.as_mut_ptr(), toks.len() as i32, true);
        self.stashes.push(st);
    }

    /// Restore the longest matching prefix of `prompt` from `snaps`/`msg` or `stashes`.
    ///
    /// Returns `(reused_len, gdn_valid, from_stash)` and updates `eng.n_past` and `eng.gdn_valid`
    /// if `eng` is passed as `&mut EngineStateView`.
    ///
    /// - Searches both `snaps` and `msg` pools for the longest prefix `n <= prompt.len()`.
    ///   If found, copies device state (`state_copy(eng, best.buf, false)`) and returns
    ///   `(best_len, best.gdn_valid, false)`.
    /// - Otherwise searches `stashes` for the longest prefix `n <= prompt.len()`.
    ///   If found, copies state and KV (`state_copy` + `kv_copy`) and returns
    ///   `(best_s_len, best_s.gdn_valid, true)`.
    /// - Returns `(0, false, false)` when no prefix matches.
    pub fn restore<E: EngineStateTarget>(&self, mut eng: E, prompt: &[i32]) -> (i32, bool, bool) {
        let view = *eng.as_view();
        if prompt.is_empty() {
            return (0, false, false);
        }

        let mut best: Option<&Snap> = None;
        let mut best_len: usize = 0;

        for pool in [&self.snaps, &self.msg] {
            for s in pool {
                let n = s.tokens.len();
                if n > prompt.len() || n <= best_len {
                    continue;
                }
                if s.tokens.as_slice() == &prompt[..n] {
                    best = Some(s);
                    best_len = n;
                }
            }
        }

        if let Some(best_snap) = best {
            state_copy(&view, best_snap.buf, false);
            let len_i32 = best_len as i32;
            eng.update_restored(len_i32, best_snap.gdn_valid);
            return (len_i32, best_snap.gdn_valid, false);
        }

        let mut best_s: Option<&Stash> = None;
        let mut best_s_len: usize = 0;

        for st in &self.stashes {
            let n = st.tokens.len();
            if n > prompt.len() || n <= best_s_len {
                continue;
            }
            if st.tokens.as_slice() == &prompt[..n] {
                best_s = Some(st);
                best_s_len = n;
            }
        }

        if let Some(best_stash) = best_s {
            state_copy(&view, best_stash.state_buf.as_ptr() as *mut u8, false);
            kv_copy(
                &view,
                best_stash.kv_buf.as_ptr() as *mut u8,
                best_s_len as i32,
                false,
            );
            let len_i32 = best_s_len as i32;
            eng.update_restored(len_i32, best_stash.gdn_valid);
            return (len_i32, best_stash.gdn_valid, true);
        }

        (0, false, false)
    }

    /// Convenience wrapper around [`Self::restore`] returning a structured [`RestoreOutcome`].
    pub fn restore_outcome<E: EngineStateTarget>(&self, eng: E, prompt: &[i32]) -> RestoreOutcome {
        let (reused_tokens, gdn_valid, from_stash) = self.restore(eng, prompt);
        RestoreOutcome {
            reused_tokens,
            gdn_valid,
            from_stash,
        }
    }

    /// Convenience wrapper around [`Self::restore`] returning only the number of restored tokens.
    pub fn restore_len<E: EngineStateTarget>(&self, eng: E, prompt: &[i32]) -> i32 {
        self.restore(eng, prompt).0
    }

    /// Find a message-boundary cut position in `prompt` given `reuse` and `snap_msg_min`.
    /// Returns `None` if `self.max_msg == 0`.
    pub fn find_message_boundary(
        &self,
        prompt: &[i32],
        reuse: usize,
        snap_msg_min: usize,
    ) -> Option<usize> {
        if self.max_msg == 0 {
            return None;
        }
        find_message_boundary(prompt, reuse, snap_msg_min)
    }

    /// Free all device buffers and clear all snapshot and stash pools.
    pub fn clear(&mut self) {
        for s in self.msg.iter_mut().chain(self.snaps.iter_mut()) {
            if !s.buf.is_null() {
                unsafe {
                    let _ = hipFree(s.buf.cast::<c_void>());
                }
                s.buf = ptr::null_mut();
            }
        }
        self.msg.clear();
        self.snaps.clear();
        self.stashes.clear();
        self.meta_entries.clear();
    }
}

impl Drop for PromptCache {
    fn drop(&mut self) {
        for s in self.msg.iter_mut().chain(self.snaps.iter_mut()) {
            if !s.buf.is_null() {
                unsafe {
                    let _ = hipFree(s.buf.cast::<c_void>());
                }
                s.buf = ptr::null_mut();
            }
        }
    }
}
