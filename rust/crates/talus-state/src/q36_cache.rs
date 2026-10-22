//! Per-layer Q36 engine state view and LRU/FIFO prompt cache (`q36_engine.cu:2032-2218`).
//!
//! Complements [`crate::pcache::EngineStateView`] for engines that hold per-layer device pointers
//! (`conv[l]`, `S[l]`, `kc[l]`, `vc[l]`, `vtc[l]`, `hlast`, `gctl`).

use core::ffi::c_void;
use std::ptr;

use hip_sys::{
    hipFree, hipMalloc, hipMemcpyAsync, hipMemcpyDeviceToDevice, hipMemcpyDeviceToHost,
    hipMemcpyHostToDevice, hipStreamSynchronize, hipStream_t,
};

use crate::pcache::{hip_memcpy_2d_async, vt_round, DEFAULT_MAX_MSG, DEFAULT_MAX_SNAPS};

/// Default VRAM KV stash budget for `Q36PromptCache` (`8 GiB`, matching `g_stash_mb = 8192`).
pub const DEFAULT_Q36_STASH_BUDGET: usize = 8usize << 30;

/// Per-layer state view of `Q36Engine` matching `q36_engine.cu:2032-2090`.
#[derive(Debug, Clone)]
pub struct Q36LayerStateView {
    pub ssm_d_conv: i32,
    pub ssm_conv_channels: i32,
    pub ssm_v_heads: i32,
    pub ssm_state_size: i32,
    pub n_head_kv: i32,
    pub head_dim: i32,
    pub n_embd: i32,
    pub ctx: i32,
    pub vt_ctx: i32,
    pub has_mtp: bool,
    pub conv: Vec<*mut f32>,
    pub s: Vec<*mut f32>,
    pub kc: Vec<*mut c_void>,
    pub vc: Vec<*mut c_void>,
    pub vtc: Vec<*mut c_void>,
    pub hlast: *mut f32,
    pub gctl: *mut i32,
    pub gdn_pending: i32,
    pub pos: i64,
    pub st: hipStream_t,
    pub dfl_k_ring: Vec<*mut f32>,
    pub dfl_v_ring: Vec<*mut f32>,
    pub dfl_ring: i32,
    pub dfl_keep: i32,
    pub dfl_kv_dim: i32,
    pub dfl_block: i32,
    pub dctx_lo: i64,
    pub dctx_hi: i64,
}

// SAFETY: Device pointers in `Q36LayerStateView` are accessed only via synchronized `st` copies.
unsafe impl Send for Q36LayerStateView {}
unsafe impl Sync for Q36LayerStateView {}

/// Total recurrent state bytes for [`Q36LayerStateView`] (`q36_engine.cu:2034-2042`).
pub fn q36_state_bytes(eng: &Q36LayerStateView) -> usize {
    let cb = ((eng.ssm_d_conv - 1).max(0) as usize) * (eng.ssm_conv_channels.max(0) as usize) * 4;
    let sb = (eng.ssm_v_heads.max(0) as usize)
        * (eng.ssm_state_size.max(0) as usize)
        * (eng.ssm_state_size.max(0) as usize)
        * 4;
    let mut n = 0usize;
    let n_layers = eng.conv.len().max(eng.s.len());
    for l in 0..n_layers {
        if !eng
            .conv
            .get(l)
            .copied()
            .unwrap_or(ptr::null_mut())
            .is_null()
        {
            n += cb;
        }
        if !eng.s.get(l).copied().unwrap_or(ptr::null_mut()).is_null() {
            n += sb;
        }
    }
    if eng.has_mtp && !eng.hlast.is_null() {
        n += (eng.n_embd.max(0) as usize) * 4;
    }
    n + 16
}

/// Copy recurrent state between [`Q36LayerStateView`] and `buf` (`q36_engine.cu:2044-2062`).
pub fn q36_state_copy(eng: &mut Q36LayerStateView, buf: *mut u8, to_buf: bool) {
    if buf.is_null() {
        return;
    }
    let cb = ((eng.ssm_d_conv - 1).max(0) as usize) * (eng.ssm_conv_channels.max(0) as usize) * 4;
    let sb = (eng.ssm_v_heads.max(0) as usize)
        * (eng.ssm_state_size.max(0) as usize)
        * (eng.ssm_state_size.max(0) as usize)
        * 4;

    unsafe {
        let mut p = buf;
        let mut seg = |ptr_dev: *mut u8, sz: usize| {
            if sz > 0 && !ptr_dev.is_null() {
                let (dst, src) = if to_buf {
                    (p.cast::<c_void>(), ptr_dev.cast_const().cast::<c_void>())
                } else {
                    (ptr_dev.cast::<c_void>(), p.cast_const().cast::<c_void>())
                };
                let st = hipMemcpyAsync(dst, src, sz, hipMemcpyDeviceToDevice, eng.st);
                assert!(st.is_success(), "q36_state_copy hipMemcpyAsync failed");
            }
            p = p.add(sz);
        };

        let n_layers = eng.conv.len().max(eng.s.len());
        for l in 0..n_layers {
            if let Some(&c_ptr) = eng.conv.get(l) {
                if !c_ptr.is_null() {
                    seg(c_ptr.cast::<u8>(), cb);
                }
            }
            if let Some(&s_ptr) = eng.s.get(l) {
                if !s_ptr.is_null() {
                    seg(s_ptr.cast::<u8>(), sb);
                }
            }
        }
        if eng.has_mtp && !eng.hlast.is_null() {
            seg(eng.hlast.cast::<u8>(), (eng.n_embd.max(0) as usize) * 4);
        }
        if !eng.gctl.is_null() {
            seg(eng.gctl.cast::<u8>(), 12);
        } else {
            p = p.add(12);
        }

        let pend = p.cast::<i32>();
        let st = if to_buf {
            hipMemcpyAsync(
                pend.cast::<c_void>(),
                (&eng.gdn_pending as *const i32).cast::<c_void>(),
                4,
                hipMemcpyHostToDevice,
                eng.st,
            )
        } else {
            hipMemcpyAsync(
                (&mut eng.gdn_pending as *mut i32).cast::<c_void>(),
                pend.cast_const().cast::<c_void>(),
                4,
                hipMemcpyDeviceToHost,
                eng.st,
            )
        };
        assert!(st.is_success(), "q36_state_copy gdn_pending failed");
        let sync = hipStreamSynchronize(eng.st);
        assert!(
            sync.is_success(),
            "q36_state_copy hipStreamSynchronize failed"
        );
    }
}

/// KV cache byte size for `[n0..n)` across all attention layers (`q36_engine.cu:2280-2288`).
pub fn q36_kv_bytes(eng: &Q36LayerStateView, n: i32, n0: i32) -> usize {
    if n <= n0 {
        return 0;
    }
    let na = eng.kc.iter().filter(|&&p| !p.is_null()).count();
    let row = (eng.n_head_kv.max(0) as usize) * (eng.head_dim.max(0) as usize) * 2;
    let vt0 = if n0 > 0 { (n0 / 16) * 16 } else { 0 };
    let vt_span = (vt_round(n) - vt0).max(0) as usize;
    let mut b = na * (((n - n0) as usize) * 2 + vt_span) * row;
    if n0 <= 0 && !eng.dfl_k_ring.is_empty() {
        let dfl_rows = eng.dfl_keep.max(0) as usize;
        let dfl_rb = (eng.dfl_kv_dim.max(0) as usize) * 4;
        b += 16 + dfl_rows * dfl_rb * 2 * eng.dfl_k_ring.len();
    }
    b
}

unsafe fn q36_dfl_ctx_copy(
    eng: &Q36LayerStateView,
    buf: *mut u8,
    p0: i64,
    rows: i32,
    stride: i32,
    save: bool,
) {
    if rows <= 0 || stride <= 0 || eng.dfl_ring <= 0 || eng.dfl_kv_dim <= 0 {
        return;
    }
    let rb = (eng.dfl_kv_dim as usize) * 4;
    let ring = eng.dfl_ring;
    let c0 = (p0.rem_euclid(ring as i64)) as i32;
    let n1 = rows.min(ring - c0);
    let mut b = buf;
    for l in 0..eng.dfl_k_ring.len() {
        let kr = eng.dfl_k_ring[l];
        let vr = eng.dfl_v_ring.get(l).copied().unwrap_or(ptr::null_mut());
        for r_ptr in [kr, vr] {
            if !r_ptr.is_null() {
                let cell = r_ptr.cast::<u8>();
                let cp = |dev: *mut u8, stash: *mut u8, bytes: usize| {
                    if bytes > 0 {
                        let (dst, src) = if save {
                            (stash.cast::<c_void>(), dev.cast_const().cast::<c_void>())
                        } else {
                            (dev.cast::<c_void>(), stash.cast_const().cast::<c_void>())
                        };
                        let st = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, eng.st);
                        assert!(st.is_success(), "q36_dfl_ctx_copy hipMemcpyAsync failed");
                    }
                };
                cp(cell.add((c0 as usize) * rb), b, (n1 as usize) * rb);
                if rows > n1 {
                    cp(cell, b.add((n1 as usize) * rb), ((rows - n1) as usize) * rb);
                }
            }
            b = b.add((stride as usize) * rb);
        }
    }
}

/// Copy KV positions `[a..b)` between [`Q36LayerStateView`] and a buffer of capacity `l_cap` cells (`q36_engine.cu:2268-2308`).
pub fn q36_kv_copy_range(
    eng: &Q36LayerStateView,
    buf: *mut u8,
    l_cap: i32,
    to_buf: bool,
    a: i32,
    b: i32,
) {
    let _ = q36_kv_copy_range_dfl(eng, buf, l_cap, to_buf, a, b);
}

fn q36_kv_copy_range_dfl(
    eng: &Q36LayerStateView,
    buf: *mut u8,
    l_cap: i32,
    to_buf: bool,
    a: i32,
    b: i32,
) -> Option<(i64, i32)> {
    if buf.is_null() || l_cap <= 0 || b <= a {
        return None;
    }
    let hd = eng.head_dim.max(0) as usize;
    let hd2 = hd * 2;
    let hkv = eng.n_head_kv.max(0) as usize;
    let dev_kv_cells = eng.ctx.max(l_cap) as usize;
    let buf_kv_cells = l_cap.max(0) as usize;
    let dev_vt_cells = eng.vt_ctx.max(vt_round(l_cap)) as usize;
    let buf_vt_cells = vt_round(l_cap).max(0) as usize;

    unsafe {
        let mut p = buf;
        let mut cp2 = |dev: *mut u8, dev_cells: usize, buf_cells: usize, c0: i32, c1: i32| {
            let c0_u = c0.max(0) as usize;
            let c1_u = c1.max(0) as usize;
            if c1_u > c0_u && !dev.is_null() && hkv > 0 && hd2 > 0 {
                let d_ptr = dev.add(c0_u * hd2);
                let s_ptr = p.add(c0_u * hd2);
                let width = (c1_u - c0_u) * hd2;
                let dev_pitch = dev_cells * hd2;
                let host_pitch = buf_cells * hd2;
                let st = if to_buf {
                    hip_memcpy_2d_async(
                        s_ptr.cast::<c_void>(),
                        host_pitch,
                        d_ptr.cast_const().cast::<c_void>(),
                        dev_pitch,
                        width,
                        hkv,
                        hipMemcpyDeviceToDevice,
                        eng.st,
                    )
                } else {
                    hip_memcpy_2d_async(
                        d_ptr.cast::<c_void>(),
                        dev_pitch,
                        s_ptr.cast_const().cast::<c_void>(),
                        host_pitch,
                        width,
                        hkv,
                        hipMemcpyDeviceToDevice,
                        eng.st,
                    )
                };
                assert!(
                    st.is_success(),
                    "q36_kv_copy_range hip_memcpy_2d_async failed"
                );
            }
            p = p.add(hkv * buf_cells * hd2);
        };

        let a_clamped = a.max(0);
        let b_clamped = b.min(l_cap);
        let vt_c0 = (a_clamped / 16) * 16;
        let vt_c1 = vt_round(b).min(vt_round(l_cap));

        for l in 0..eng.kc.len() {
            let kc_ptr = eng.kc[l].cast::<u8>();
            if kc_ptr.is_null() {
                continue;
            }
            let vc_ptr = eng
                .vc
                .get(l)
                .copied()
                .unwrap_or(ptr::null_mut())
                .cast::<u8>();
            let vtc_ptr = eng
                .vtc
                .get(l)
                .copied()
                .unwrap_or(ptr::null_mut())
                .cast::<u8>();

            cp2(kc_ptr, dev_kv_cells, buf_kv_cells, a_clamped, b_clamped);
            cp2(vc_ptr, dev_kv_cells, buf_kv_cells, a_clamped, b_clamped);
            if !vtc_ptr.is_null() {
                cp2(vtc_ptr, dev_vt_cells, buf_vt_cells, vt_c0, vt_c1);
            }
        }

        let mut dfl_injected: Option<(i64, i32)> = None;
        if !eng.dfl_k_ring.is_empty() {
            let l_i64 = l_cap as i64;
            let b_i64 = b as i64;
            if to_buf {
                let k = (eng.dfl_keep.max(0) as i64).min(l_i64);
                let mut vlo: i64 = -1;
                if eng.dctx_hi >= l_i64 && eng.dctx_lo < l_i64 {
                    vlo = eng.dctx_lo.max(l_i64 - k);
                }
                let st = hipMemcpyAsync(
                    p.cast::<c_void>(),
                    (&vlo as *const i64).cast::<c_void>(),
                    8,
                    hipMemcpyHostToDevice,
                    eng.st,
                );
                assert!(st.is_success(), "dfl_stash save vlo failed");
                let sync = hipStreamSynchronize(eng.st);
                assert!(sync.is_success());
                if vlo >= 0 {
                    let rows = (l_i64 - vlo) as i32;
                    q36_dfl_ctx_copy(eng, p.add(16), vlo, rows, rows, true);
                }
            } else {
                let mut vlo: i64 = -1;
                let st = hipMemcpyAsync(
                    (&mut vlo as *mut i64).cast::<c_void>(),
                    p.cast_const().cast::<c_void>(),
                    8,
                    hipMemcpyDeviceToHost,
                    eng.st,
                );
                assert!(st.is_success(), "dfl_stash load vlo failed");
                let sync = hipStreamSynchronize(eng.st);
                assert!(sync.is_success());
                if vlo >= 0 && vlo < b_i64 {
                    let n = (l_i64.min(b_i64) - vlo) as i32;
                    let stride = (l_i64 - vlo) as i32;
                    q36_dfl_ctx_copy(eng, p.add(16), vlo, n, stride, false);
                    dfl_injected = Some((vlo, n));
                }
            }
        }

        let sync = hipStreamSynchronize(eng.st);
        assert!(sync.is_success());
        dfl_injected
    }
}

/// Copy KV positions `[n0..n)` between [`Q36LayerStateView`] and `buf` (`q36_engine.cu:1994-2009`).
pub fn q36_kv_copy(eng: &Q36LayerStateView, buf: *mut u8, n: i32, to_buf: bool, n0: i32) {
    if buf.is_null() || n <= n0 {
        return;
    }
    if n0 <= 0 {
        q36_kv_copy_range(eng, buf, n, to_buf, 0, n);
        return;
    }
    let hd = eng.head_dim.max(0) as usize;
    let hd2 = hd * 2;
    let hkv = eng.n_head_kv.max(0) as usize;
    let w = ((n - n0) as usize) * hd2;
    let sp = (eng.ctx.max(n) as usize) * hd2;
    let off = (n0.max(0) as usize) * hd2;
    let vt0 = (n0.max(0) / 16) * 16;
    let vtw = ((vt_round(n) - vt0).max(0) as usize) * hd2;
    let vtsp = (eng.vt_ctx.max(vt_round(n)) as usize) * hd2;
    let vtoff = (vt0 as usize) * hd2;

    unsafe {
        let mut p = buf;
        for l in 0..eng.kc.len() {
            let kc_ptr = eng.kc[l].cast::<u8>();
            if kc_ptr.is_null() {
                continue;
            }
            let vc_ptr = eng
                .vc
                .get(l)
                .copied()
                .unwrap_or(ptr::null_mut())
                .cast::<u8>();
            let vtc_ptr = eng
                .vtc
                .get(l)
                .copied()
                .unwrap_or(ptr::null_mut())
                .cast::<u8>();

            if to_buf {
                let st1 = hip_memcpy_2d_async(
                    p.cast::<c_void>(),
                    w,
                    kc_ptr.add(off).cast_const().cast::<c_void>(),
                    sp,
                    w,
                    hkv,
                    hipMemcpyDeviceToDevice,
                    eng.st,
                );
                assert!(st1.is_success());
                p = p.add(hkv * w);

                if !vc_ptr.is_null() {
                    let st2 = hip_memcpy_2d_async(
                        p.cast::<c_void>(),
                        w,
                        vc_ptr.add(off).cast_const().cast::<c_void>(),
                        sp,
                        w,
                        hkv,
                        hipMemcpyDeviceToDevice,
                        eng.st,
                    );
                    assert!(st2.is_success());
                }
                p = p.add(hkv * w);

                if !vtc_ptr.is_null() && vtw > 0 {
                    let st3 = hip_memcpy_2d_async(
                        p.cast::<c_void>(),
                        vtw,
                        vtc_ptr.add(vtoff).cast_const().cast::<c_void>(),
                        vtsp,
                        vtw,
                        hkv,
                        hipMemcpyDeviceToDevice,
                        eng.st,
                    );
                    assert!(st3.is_success());
                }
                p = p.add(hkv * vtw);
            } else {
                let st1 = hip_memcpy_2d_async(
                    kc_ptr.add(off).cast::<c_void>(),
                    sp,
                    p.cast_const().cast::<c_void>(),
                    w,
                    w,
                    hkv,
                    hipMemcpyDeviceToDevice,
                    eng.st,
                );
                assert!(st1.is_success());
                p = p.add(hkv * w);

                if !vc_ptr.is_null() {
                    let st2 = hip_memcpy_2d_async(
                        vc_ptr.add(off).cast::<c_void>(),
                        sp,
                        p.cast_const().cast::<c_void>(),
                        w,
                        w,
                        hkv,
                        hipMemcpyDeviceToDevice,
                        eng.st,
                    );
                    assert!(st2.is_success());
                }
                p = p.add(hkv * w);

                if !vtc_ptr.is_null() && vtw > 0 {
                    let st3 = hip_memcpy_2d_async(
                        vtc_ptr.add(vtoff).cast::<c_void>(),
                        vtsp,
                        p.cast_const().cast::<c_void>(),
                        vtw,
                        vtw,
                        hkv,
                        hipMemcpyDeviceToDevice,
                        eng.st,
                    );
                    assert!(st3.is_success());
                }
                p = p.add(hkv * vtw);
            }
        }
        let sync = hipStreamSynchronize(eng.st);
        assert!(sync.is_success());
    }
}

/// Device-allocated recurrent state snapshot wrapper (`Q36StateSnapshot`).
#[derive(Debug)]
pub struct Q36StateSnapshot {
    pub tokens: Vec<i32>,
    pub buf: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for Q36StateSnapshot {}
unsafe impl Sync for Q36StateSnapshot {}

impl Q36StateSnapshot {
    pub fn save_from_device(eng: &mut Q36LayerStateView, tokens: &[i32]) -> Option<Self> {
        let bytes = q36_state_bytes(eng);
        if bytes == 0 {
            return None;
        }
        let mut raw: *mut c_void = ptr::null_mut();
        if !unsafe { hipMalloc(&mut raw, bytes) }.is_success() {
            return None;
        }
        let buf = raw.cast::<u8>();
        q36_state_copy(eng, buf, true);
        Some(Self {
            tokens: tokens.to_vec(),
            buf,
            bytes,
        })
    }

    pub fn restore_to_device(&self, eng: &mut Q36LayerStateView) {
        if !self.buf.is_null() {
            q36_state_copy(eng, self.buf, false);
            eng.pos = self.tokens.len() as i64;
        }
    }
}

impl Drop for Q36StateSnapshot {
    fn drop(&mut self) {
        if !self.buf.is_null() {
            unsafe {
                let _ = hipFree(self.buf.cast::<c_void>());
            }
            self.buf = ptr::null_mut();
        }
    }
}

/// Device-allocated KV cache slice wrapper (`Q36KvCacheSlice`) supporting partial range `[a..b)` copies.
#[derive(Debug)]
pub struct Q36KvCacheSlice {
    pub tokens: Vec<i32>,
    pub max_tokens: usize,
    pub buf: *mut u8,
    pub bytes: usize,
}

unsafe impl Send for Q36KvCacheSlice {}
unsafe impl Sync for Q36KvCacheSlice {}

impl Q36KvCacheSlice {
    pub fn save_from_device(eng: &Q36LayerStateView, tokens: &[i32]) -> Option<Self> {
        let max_tokens = tokens.len();
        let bytes = q36_kv_bytes(eng, max_tokens as i32, 0);
        if bytes == 0 {
            return None;
        }
        let mut raw: *mut c_void = ptr::null_mut();
        if !unsafe { hipMalloc(&mut raw, bytes) }.is_success() {
            return None;
        }
        let buf = raw.cast::<u8>();
        q36_kv_copy_range(eng, buf, max_tokens as i32, true, 0, max_tokens as i32);
        Some(Self {
            tokens: tokens.to_vec(),
            max_tokens,
            buf,
            bytes,
        })
    }

    pub fn restore_to_device(&self, eng: &Q36LayerStateView, a: usize, b: usize) {
        if !self.buf.is_null() && b > a {
            q36_kv_copy_range(
                eng,
                self.buf,
                self.max_tokens as i32,
                false,
                a as i32,
                b.min(self.max_tokens) as i32,
            );
        }
    }
}

impl Drop for Q36KvCacheSlice {
    fn drop(&mut self) {
        if !self.buf.is_null() {
            unsafe {
                let _ = hipFree(self.buf.cast::<c_void>());
            }
            self.buf = ptr::null_mut();
        }
    }
}

/// Snapshot slot in [`Q36PromptCache`].
#[derive(Debug)]
pub struct Q36Snap {
    pub ids: Vec<i32>,
    pub buf: *mut u8,
    pub msg: bool,
    pub used: u64,
}

unsafe impl Send for Q36Snap {}
unsafe impl Sync for Q36Snap {}

/// Device KV stash slot in [`Q36PromptCache`].
#[derive(Debug)]
pub struct Q36Stash {
    pub ids: Vec<i32>,
    pub buf: *mut u8,
    pub bytes: usize,
    pub used: u64,
    pub pf: usize,
}

unsafe impl Send for Q36Stash {}
unsafe impl Sync for Q36Stash {}

/// Direct port of `q36_engine.cu:2332-2430` `PromptCache` with LRU clock, partial KV restore, and `pf` tracking.
#[derive(Debug)]
pub struct Q36PromptCache {
    pub snaps: Vec<Q36Snap>,
    pub stashes: Vec<Q36Stash>,
    pub snap_bytes: usize,
    pub stash_used: usize,
    pub stash_budget: usize,
    pub max_snaps: usize,
    pub max_msg: usize,
    pub clock: u64,
}

impl Default for Q36PromptCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Q36PromptCache {
    pub fn new() -> Self {
        Self {
            snaps: Vec::new(),
            stashes: Vec::new(),
            snap_bytes: 0,
            stash_used: 0,
            stash_budget: DEFAULT_Q36_STASH_BUDGET,
            max_snaps: DEFAULT_MAX_SNAPS,
            max_msg: DEFAULT_MAX_MSG,
            clock: 0,
        }
    }

    fn lcp(a: &[i32], b: &[i32]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    pub fn save(&mut self, eng: &mut Q36LayerStateView, ids: &[i32], msg: bool) {
        let cap = if msg { self.max_msg } else { self.max_snaps };
        if self.max_snaps == 0 || cap == 0 || ids.is_empty() {
            return;
        }
        if self.snap_bytes == 0 {
            self.snap_bytes = q36_state_bytes(eng);
        }
        self.clock += 1;
        let now = self.clock;

        if let Some(idx) = self.snaps.iter().position(|s| s.ids.as_slice() == ids) {
            let buf = self.snaps[idx].buf;
            q36_state_copy(eng, buf, true);
            self.snaps[idx].used = now;
            self.snaps[idx].msg |= msg;
            return;
        }

        let in_pool = self.snaps.iter().filter(|s| s.msg == msg).count();
        let idx = if in_pool < cap {
            let mut raw: *mut c_void = ptr::null_mut();
            let st = unsafe { hipMalloc(&mut raw, self.snap_bytes) };
            assert!(st.is_success(), "Q36PromptCache::save hipMalloc failed");
            self.snaps.push(Q36Snap {
                ids: Vec::new(),
                buf: raw.cast::<u8>(),
                msg,
                used: 0,
            });
            self.snaps.len() - 1
        } else {
            self.snaps
                .iter()
                .enumerate()
                .filter(|(_, s)| s.msg == msg)
                .min_by_key(|(_, s)| s.used)
                .map(|(i, _)| i)
                .unwrap()
        };

        let buf = self.snaps[idx].buf;
        q36_state_copy(eng, buf, true);
        self.snaps[idx].ids.clear();
        self.snaps[idx].ids.extend_from_slice(ids);
        self.snaps[idx].msg = msg;
        self.snaps[idx].used = now;
    }

    pub fn stash_kv(&mut self, eng: &Q36LayerStateView, live: &[i32], keep: usize) {
        self.stash_kv_with_pf(eng, live, keep, usize::MAX);
    }

    pub fn stash_kv_with_pf(
        &mut self,
        eng: &Q36LayerStateView,
        live: &[i32],
        keep: usize,
        pf: usize,
    ) {
        if self.stash_budget == 0 || live.len() < 256 || live.len().saturating_sub(keep) < 64 {
            return;
        }
        for s in &mut self.stashes {
            if s.ids.len() >= live.len() && Self::lcp(&s.ids, live) == live.len() {
                self.clock += 1;
                s.used = self.clock;
                return;
            }
        }
        let need = q36_kv_bytes(eng, live.len() as i32, 0);
        if need == 0 || need > self.stash_budget {
            return;
        }
        while self.stash_used + need > self.stash_budget && !self.stashes.is_empty() {
            let old = self
                .stashes
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.used)
                .map(|(i, _)| i)
                .unwrap();
            let removed = self.stashes.remove(old);
            unsafe {
                let _ = hipFree(removed.buf.cast::<c_void>());
            }
            self.stash_used = self.stash_used.saturating_sub(removed.bytes);
        }
        let mut raw: *mut c_void = ptr::null_mut();
        if unsafe { hipMalloc(&mut raw, need) }.is_success() {
            let buf = raw.cast::<u8>();
            q36_kv_copy(eng, buf, live.len() as i32, true, 0);
            for i in (0..self.stashes.len()).rev() {
                let len = self.stashes[i].ids.len();
                if Self::lcp(&self.stashes[i].ids, live) == len {
                    let removed = self.stashes.remove(i);
                    unsafe {
                        let _ = hipFree(removed.buf.cast::<c_void>());
                    }
                    self.stash_used = self.stash_used.saturating_sub(removed.bytes);
                }
            }
            self.stash_used += need;
            self.clock += 1;
            self.stashes.push(Q36Stash {
                ids: live.to_vec(),
                buf,
                bytes: need,
                used: self.clock,
                pf: pf.min(live.len()),
            });
        }
    }

    pub fn restore(&mut self, eng: &mut Q36LayerStateView, ids: &[i32], live: &[i32]) -> usize {
        self.restore_with_pf(eng, ids, live, usize::MAX, false).0
    }

    pub fn restore_with_pf(
        &mut self,
        eng: &mut Q36LayerStateView,
        ids: &[i32],
        live: &[i32],
        live_pf: usize,
        pf_only: bool,
    ) -> (usize, &'static str) {
        let in_cache = live_pf.min(Self::lcp(live, ids));

        let mut best_idx: Option<usize> = None;
        let mut best_kv_idx: Option<usize> = None;
        let mut best_n = 0usize;

        for (i, s) in self.snaps.iter().enumerate() {
            let n = s.ids.len();
            if n >= ids.len() || n <= best_n || Self::lcp(&s.ids, ids) < n {
                continue;
            }
            let mut kv_i: Option<usize> = None;
            if n > in_cache {
                for (ti, t) in self.stashes.iter().enumerate() {
                    if t.ids.len() >= n
                        && Self::lcp(&t.ids, &ids[..n]) >= n
                        && (!pf_only || t.pf >= n)
                    {
                        kv_i = Some(ti);
                        break;
                    }
                }
                if kv_i.is_none() {
                    continue;
                }
            }
            best_idx = Some(i);
            best_kv_idx = kv_i;
            best_n = n;
        }

        let Some(b_idx) = best_idx else {
            return (0, "");
        };

        let from = if let Some(s_i) = best_kv_idx {
            if !eng.dfl_k_ring.is_empty() {
                let a = in_cache as i64;
                if eng.dctx_hi > a {
                    eng.dctx_hi = a;
                }
                if eng.dctx_lo > eng.dctx_hi {
                    eng.dctx_lo = a;
                    eng.dctx_hi = a;
                }
            }
            let st_buf = self.stashes[s_i].buf;
            let st_len = self.stashes[s_i].ids.len();
            let dfl_inj = q36_kv_copy_range_dfl(
                eng,
                st_buf,
                st_len as i32,
                false,
                in_cache as i32,
                best_n as i32,
            );
            if let Some((vlo, n)) = dfl_inj {
                if n > 0 {
                    let end = vlo + n as i64;
                    if vlo > eng.dctx_hi || end < eng.dctx_lo {
                        eng.dctx_lo = vlo;
                        eng.dctx_hi = end;
                    } else {
                        eng.dctx_lo = eng.dctx_lo.min(vlo);
                        eng.dctx_hi = eng.dctx_hi.max(end);
                    }
                    let top = end + eng.dfl_block as i64;
                    if eng.dctx_lo < top - eng.dfl_ring as i64 {
                        eng.dctx_lo = top - eng.dfl_ring as i64;
                    }
                    if eng.dctx_lo > eng.dctx_hi {
                        eng.dctx_lo = eng.dctx_hi;
                    }
                }
            }
            self.clock += 1;
            self.stashes[s_i].used = self.clock;
            "snapshot + KV stash"
        } else {
            "snapshot"
        };

        let snap_buf = self.snaps[b_idx].buf;
        q36_state_copy(eng, snap_buf, false);
        eng.pos = best_n as i64;
        self.clock += 1;
        self.snaps[b_idx].used = self.clock;
        (best_n, from)
    }
}

impl Drop for Q36PromptCache {
    fn drop(&mut self) {
        for s in &mut self.snaps {
            if !s.buf.is_null() {
                unsafe {
                    let _ = hipFree(s.buf.cast::<c_void>());
                }
                s.buf = ptr::null_mut();
            }
        }
        for s in &mut self.stashes {
            if !s.buf.is_null() {
                unsafe {
                    let _ = hipFree(s.buf.cast::<c_void>());
                }
                s.buf = ptr::null_mut();
            }
        }
    }
}
