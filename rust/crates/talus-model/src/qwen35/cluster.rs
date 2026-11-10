//! talus#47 Phase 1: the 27B as a two-stage pipeline (layer split) across two processes.
//!
//! Stage 0 (the head) runs the embedding, layers `0..cut`, the head, sampler and drafters; the worker runs layers
//! `cut..n_layer`. The output is bitwise identical to one process by construction: every kernel of the one-process
//! sequence runs once with the same launch parameters on equal input bytes.
//!
//! - **The cut.** The head runs layer `cut - 1`'s fused epilogue with layer `cut`'s `attn_norm` (the loader keeps
//!   that one vector), exactly as one process does, and ships everything that epilogue wrote: the residual `x`, the
//!   normed `h`, `h`'s q8_1 copy when the kernel wrote one, plus the device control words the next layers read
//!   (`gctl`, the position). The worker starts from those bytes in the state one process would be in.
//! - **The return.** The worker ships `h_last` (the normed final hidden state), its q8_1 copy when the last layer's
//!   epilogue wrote one (the head GEMM reads it), the DFlash2 tap rows of its layers, and in `--exact-test` the
//!   per-layer residuals.
//! - **Host state.** Everything host-side that a layer reads (rows, verify flag, the host position, the q8 mark) rides
//!   in the step control block. Ops outside the forward pass that touch per-layer state (`reset`, the fused GDN
//!   commit `flush_gdn` with the head's `gctl`, `commit`) are mirrored to the worker in order.
//! - **Graphs.** A captured graph whose body crosses the cut (the decode step `g1`, the verify windows `gA`) is
//!   captured as segments; `launch_graph` replays segment 0, does the exchange, replays segment 1. The worker
//!   captures its own segment graph per window shape, so each stage keeps about one graph per round.
//!
//! Off by default: without `--cluster-role` no `ClState` exists and every single-process path is unchanged.

#![allow(non_snake_case, unused_unsafe)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::time::{Duration, Instant};

use super::engine::{q27_budget, q27_bytes_per_token, Q36Engine};
use super::ffi::*;
use super::weights::Q36Model;
use hip_sys::hipMemcpyKind;
use talus_cluster::{ident, Header, Kv, Link, MsgType, Role};
use talus_state::{q36_shape_view, Q36LayerStateView, Q36PromptCache};

/// Bytes reserved at the start of a bundle for the step control block.
pub const CTL_BYTES: usize = 256;
/// Most DFlash2 tap layers a worker can own.
pub const MAX_TAPS: usize = 8;
/// Protocol revision of the qwen35 stage messages (bumped on any layout change).
pub const STAGE_PROTO: i64 = 1;

/// The step control block at the start of every STEP payload. `gctl` and `pos0` are written by the head's device
/// (captured copies), the rest by the head's host just before the send.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct StepCtl {
    pub gctl: [i32; 4],
    pub pos0: i32,
    pub _pad: [i32; 3],
    pub t: i32,
    pub verify: u32,
    pub cur_p0_h: i32,
    pub q8: u32,
    pub dbg_on: u32,
    pub dbg_row: i32,
    pub graph: u32,
    pub n_taps: u32,
    pub taps: [i32; MAX_TAPS],
    pub q8ret: u32,
}

const _: () = assert!(std::mem::size_of::<StepCtl>() <= CTL_BYTES);
const CTL_HOST_OFF: usize = 32; // first host-written byte (t)

/// What the head knew at a cut (captured once with a graph segment, replayed with it).
#[derive(Debug, Clone, Copy)]
pub struct CutRec {
    pub t: i32,
    pub verify: bool,
    pub q8: bool,
    pub q8ret: bool,
    pub cur_p0_h: i32,
    pub dbg_on: bool,
    pub dbg_row: i32,
    pub graph: bool,
}

/// Per-process cluster state, present only in a cluster stage.
pub struct ClState {
    pub role: Role,
    /// head: the link to the worker (the worker loop owns its link itself)
    pub link: Option<Link>,
    /// pinned host buffers: head tx = bundle, rx = return; worker the other way round
    pub tx: *mut u8,
    pub tx_cap: usize,
    pub rx: *mut u8,
    pub rx_cap: usize,
    /// head: device staging for returned DFlash2 tap rows (MAX_TAPS x B x E floats)
    pub tap_dev: *mut f32,
    /// head: worker-owned layers whose post-layer residual feeds a DFlash2 tap (`next = li + 1`)
    pub wtaps: Vec<i32>,
    /// split capture (head)
    pub cap_on: bool,
    pub cap_parts: Vec<hipGraphExec_t>,
    pub cap_cuts: Vec<CutRec>,
    pub segs: HashMap<usize, Vec<(CutRec, hipGraphExec_t)>>,
    /// worker: segment graphs per (rows, verify, q8, taps)
    pub wgraphs: HashMap<(i32, bool, bool, u64), (hipGraphExec_t, bool)>,
    pub step: u64,
    pub hops: u64,
    pub hop_s: f64,
    /// head: the heartbeat on the control connection, and what a reconnect needs (Phase 2)
    pub beat: Option<talus_cluster::Beat>,
    pub recon: Option<Recon>,
}

unsafe impl Send for ClState {}

fn pinned(bytes: usize) -> super::Result<*mut u8> {
    let mut p: *mut c_void = ptr::null_mut();
    super::weights::hip_check(
        unsafe { hipHostMalloc(&mut p, bytes.max(4096), HIP_HOST_MALLOC_DEFAULT) },
        "hipHostMalloc(cluster)",
    )?;
    Ok(p.cast::<u8>())
}

impl Drop for ClState {
    fn drop(&mut self) {
        unsafe {
            for (_, v) in self.segs.drain() {
                for (_, g) in v {
                    let _ = hipGraphExecDestroy(g);
                }
            }
            for (_, (g, _)) in self.wgraphs.drain() {
                let _ = hipGraphExecDestroy(g);
            }
            if !self.tx.is_null() {
                let _ = hipHostFree(self.tx.cast::<c_void>());
            }
            if !self.rx.is_null() {
                let _ = hipHostFree(self.rx.cast::<c_void>());
            }
            if !self.tap_dev.is_null() {
                let _ = hipFree(self.tap_dev.cast::<c_void>());
            }
        }
    }
}

impl ClState {
    fn new(role: Role, link: Option<Link>) -> Self {
        Self {
            role,
            link,
            tx: ptr::null_mut(),
            tx_cap: 0,
            rx: ptr::null_mut(),
            rx_cap: 0,
            tap_dev: ptr::null_mut(),
            wtaps: Vec::new(),
            cap_on: false,
            cap_parts: Vec::new(),
            cap_cuts: Vec::new(),
            segs: HashMap::new(),
            wgraphs: HashMap::new(),
            step: 0,
            hops: 0,
            hop_s: 0.0,
            beat: None,
            recon: None,
        }
    }
}

/// Bundle layout for `t` rows: (x offset, h offset, q8 offset, q8 bytes, total with q8 / without).
fn bundle_layout(t: i32, e: i32, q8: bool) -> (usize, usize, usize, usize, usize) {
    let row = (t as usize) * (e as usize) * 4;
    let ox = CTL_BYTES;
    let oh = ox + row;
    let oq = oh + row;
    let qb = if q8 { tk_q8_1_bytes(e, t) } else { 0 };
    (ox, oh, oq, qb, oq + qb)
}

/// Return layout for `t` rows: (q8 offset, q8 bytes, taps offset, dbg offset, total).
fn ret_layout(
    t: i32,
    e: i32,
    q8ret: bool,
    n_taps: usize,
    n_dbg_layers: usize,
) -> (usize, usize, usize, usize, usize) {
    let row = (t as usize) * (e as usize) * 4;
    let oq = row;
    let qb = if q8ret { tk_q8_1_bytes(e, t) } else { 0 };
    let ot = oq + qb;
    let od = ot + n_taps * row;
    (oq, qb, ot, od, od + n_dbg_layers * row)
}

fn hcopy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: hipMemcpyKind, s: hipStream_t) {
    if bytes > 0 {
        let _ = unsafe { hipMemcpyAsync(dst, src, bytes, kind, s) };
    }
}

/// The panic payload of a lost or failed cluster link (Phase 2): the head's serve loop catches it, answers the request
/// with `ERR stage_unavailable` (HTTP 503 from talus-server) and reconnects; a worker drops its engine and waits for
/// the next head. Outside a serve loop it ends the process like any panic.
#[derive(Debug, Clone)]
pub struct ClusterLost(pub String);

fn die(what: &str, e: impl std::fmt::Display) -> ! {
    let m = format!("{what}: {e}");
    eprintln!("talus-cluster: link failure: {m}");
    std::panic::panic_any(ClusterLost(m));
}

/// The control (heartbeat) address: the data port + 1 (7701 -> 7702).
pub fn ctl_addr(addr: &str) -> String {
    match addr.rsplit_once(':') {
        Some((h, p)) => match p.parse::<u16>() {
            Ok(n) => format!("{h}:{}", n.wrapping_add(1)),
            Err(_) => addr.to_string(),
        },
        None => addr.to_string(),
    }
}

/// What the head needs to rebuild a lost link with the same settings.
#[derive(Debug, Clone)]
pub struct Recon {
    pub addr: String,
    pub alt: Option<String>,
    pub ctx: i32,
    pub max_batch: i32,
    pub use_graph: bool,
}

/// A handshaken head link: data link, its heartbeat, the agreed context.
pub struct HeadLink {
    pub link: Link,
    pub beat: Option<talus_cluster::Beat>,
    pub ctx: i32,
    pub addr: String,
}

impl Q36Engine {
    #[inline]
    pub fn is_cluster_head(&self) -> bool {
        self.cl.as_ref().is_some_and(|c| c.role == Role::Head)
    }

    /// Head: take over the handshaken link to the worker; allocate the bundle / return buffers.
    pub fn cluster_attach_head(&mut self, hl: HeadLink, recon: Recon) -> super::Result<()> {
        let link = hl.link;
        let m = unsafe { &*self.m };
        let e = m.g.n_embd as i32;
        let b = self.max_batch.max(16);
        let n = m.layers.len();
        let mut cl = ClState::new(Role::Head, Some(link));
        cl.beat = hl.beat;
        cl.recon = Some(recon);
        if self.use_dflash && self.df.ready {
            for li in m.stage_hi..n {
                if self.df.wants_tap((li + 1) as i32) {
                    cl.wtaps.push(li as i32);
                }
            }
        }
        super::ffi::ensure!(
            cl.wtaps.len() <= MAX_TAPS,
            "too many DFlash2 tap layers on the worker"
        );
        let row = (b as usize) * (e as usize) * 4;
        let (_, _, _, _, tx_need) = bundle_layout(b, e, true);
        let n_w = n - m.stage_hi;
        let rx_need = (row + tk_q8_1_bytes(e, b) + cl.wtaps.len() * row)
            .max(ret_layout(16, e, true, cl.wtaps.len(), n_w).4);
        cl.tx = pinned(tx_need)?;
        cl.tx_cap = tx_need;
        cl.rx = pinned(rx_need)?;
        cl.rx_cap = rx_need;
        if !cl.wtaps.is_empty() {
            let bytes = cl.wtaps.len() * row;
            let p = super::weights::hip_malloc_bytes(bytes)?;
            cl.tap_dev = p.cast::<f32>();
        }
        eprintln!(
            "talus-cluster: head stage layers 0..{} (+ embedding, head, drafters); worker owns {}..{} at {}; \
             DFlash2 taps from the worker: {:?}; buffers {:.1} + {:.1} MiB pinned",
            m.stage_hi,
            m.stage_hi,
            n,
            cl.link.as_ref().map(|l| l.peer.clone()).unwrap_or_default(),
            cl.wtaps,
            tx_need as f64 / 1048576.0,
            rx_need as f64 / 1048576.0
        );
        self.cl = Some(Box::new(cl));
        Ok(())
    }

    /// Worker: allocate the receive / send buffers for `max_batch` rows.
    pub fn cluster_attach_worker(&mut self) -> super::Result<()> {
        let m = unsafe { &*self.m };
        let e = m.g.n_embd as i32;
        let b = self.max_batch.max(16);
        let mut cl = ClState::new(Role::Worker, None);
        let (_, _, _, _, rx_need) = bundle_layout(b, e, true);
        let n_w = m.layers.len() - m.stage_lo;
        let n_taps = super::graphs::K_DFLASH_TAPS.min(MAX_TAPS);
        let tx_need = ret_layout(b, e, true, n_taps, 0)
            .4
            .max(ret_layout(16, e, true, MAX_TAPS, n_w).4);
        cl.rx = pinned(rx_need)?;
        cl.rx_cap = rx_need;
        cl.tx = pinned(tx_need)?;
        cl.tx_cap = tx_need;
        self.cl = Some(Box::new(cl));
        Ok(())
    }

    /// Launch a captured qwen35 graph; a graph captured across the cut replays as segments with an exchange between.
    ///
    /// # Safety
    /// `g` is a graph exec instantiated by `capture_graph` on this engine.
    pub(crate) unsafe fn launch_graph(&mut self, g: hipGraphExec_t) -> hipError_t {
        let r = hipGraphLaunch(g, self.s_main);
        if self.cl.is_some() {
            let rest = self
                .cl
                .as_ref()
                .and_then(|c| c.segs.get(&(g as usize)).cloned());
            if let Some(rest) = rest {
                for (cut, gx) in rest {
                    self.cl_exchange(&cut);
                    let _ = hipGraphLaunch(gx, self.s_main);
                }
            }
        }
        r
    }

    /// `capture_graph` hook before the body runs.
    pub(crate) fn cl_capture_begin(&mut self) {
        if let Some(cl) = self.cl.as_mut() {
            if cl.role == Role::Head {
                cl.cap_on = true;
                cl.cap_parts.clear();
                cl.cap_cuts.clear();
            }
        }
    }

    /// `capture_graph` hook after the last segment was instantiated: returns the handle to launch (segment 0) and
    /// records the rest, or `exec` unchanged when the body did not cross the cut.
    pub(crate) fn cl_capture_end(&mut self, exec: hipGraphExec_t) -> hipGraphExec_t {
        let Some(cl) = self.cl.as_mut() else {
            return exec;
        };
        if cl.role != Role::Head || !cl.cap_on {
            return exec;
        }
        cl.cap_on = false;
        if cl.cap_parts.is_empty() {
            // a plain graph: forget any segments recorded under a destroyed exec at the same address
            if let Some(v) = cl.segs.remove(&(exec as usize)) {
                for (_, g) in v {
                    let _ = unsafe { hipGraphExecDestroy(g) };
                }
            }
            return exec;
        }
        let mut parts = std::mem::take(&mut cl.cap_parts);
        parts.push(exec);
        let cuts = std::mem::take(&mut cl.cap_cuts);
        let first = parts[0];
        let rest: Vec<(CutRec, hipGraphExec_t)> =
            cuts.into_iter().zip(parts.into_iter().skip(1)).collect();
        cl.segs.insert(first as usize, rest);
        first
    }

    /// `destroy_graphs` hook: drop the recorded segments (the caller destroys each segment 0 itself).
    pub(crate) fn cl_destroy_graphs(&mut self) {
        if let Some(cl) = self.cl.as_mut() {
            for (_, v) in cl.segs.drain() {
                for (_, g) in v {
                    let _ = unsafe { hipGraphExecDestroy(g) };
                }
            }
            for (_, (g, _)) in cl.wgraphs.drain() {
                let _ = unsafe { hipGraphExecDestroy(g) };
            }
        }
    }

    /// Whether the return carries `h_last`'s q8_1 copy: the last layer's dense epilogue writes it exactly when it
    /// takes the q8 branch (`layer_into`, `wn` = output_norm).
    fn cl_q8ret(&self, t: i32) -> bool {
        let m = unsafe { &*self.m };
        m.dense && t <= self.gemm_t && tk_q36_mmvq_supported(m.head.raw_qtype())
    }

    /// Head, in `forward_full` right after layer `cut - 1`: ship the live set, get the worker's return.
    pub(crate) fn cl_cut(&mut self, pos0: *const i32, n_tok: i32, verify: bool) {
        let e = self.d_model();
        let t = n_tok;
        let q8 = self.q8_src == self.h as *const f32 && self.q8_n == e && self.q8_tok == t;
        let dbg_on = !self.dbg.is_null() && (self.dbg_row + t) as usize <= super::engine::Q27_VMAX;
        let rec = CutRec {
            t,
            verify,
            q8,
            q8ret: self.cl_q8ret(t),
            cur_p0_h: self.cur_p0_h,
            dbg_on,
            dbg_row: self.dbg_row,
            graph: false,
        };
        let s = self.s_main;
        let (tx, capturing) = {
            let cl = self.cl.as_ref().unwrap();
            (cl.tx, cl.cap_on)
        };
        let (ox, oh, oq, qb, blen) = bundle_layout(t, e, q8);
        let row = (t as usize) * (e as usize) * 4;
        {
            let cl = self.cl.as_ref().unwrap();
            let rlen = ret_layout(t, e, rec.q8ret, cl.wtaps.len(), 0).4;
            if blen > cl.tx_cap || rlen > cl.rx_cap {
                die(
                    "cluster cut",
                    format!(
                        "T={t} needs {blen} / {rlen} B, buffers {} / {}",
                        cl.tx_cap, cl.rx_cap
                    ),
                );
            }
        }
        unsafe {
            hcopy(
                tx.cast(),
                self.gctl.cast(),
                16,
                HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            );
            hcopy(
                tx.add(16).cast(),
                pos0.cast(),
                4,
                HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            );
            hcopy(
                tx.add(ox).cast(),
                self.x.cast(),
                row,
                HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            );
            hcopy(
                tx.add(oh).cast(),
                self.h.cast(),
                row,
                HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            );
            if q8 {
                hcopy(
                    tx.add(oq).cast(),
                    self.x_q8.cast_const(),
                    qb,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    s,
                );
            }
        }
        if capturing {
            // end this segment, start the next one: the exchange happens between their launches
            let mut gr: hipGraph_t = ptr::null_mut();
            let mut ex: hipGraphExec_t = ptr::null_mut();
            unsafe {
                if hipStreamEndCapture(s, &mut gr) != hipSuccess
                    || hipGraphInstantiate(&mut ex, gr, ptr::null_mut(), ptr::null_mut(), 0)
                        != hipSuccess
                {
                    die(
                        "graph capture at the cut",
                        "hipStreamEndCapture / hipGraphInstantiate failed",
                    );
                }
                let _ = hipGraphDestroy(gr);
            }
            let cl = self.cl.as_mut().unwrap();
            cl.cap_parts.push(ex);
            cl.cap_cuts.push(CutRec { graph: true, ..rec });
            if unsafe { hipStreamBeginCapture(s, HIP_STREAM_CAPTURE_MODE_RELAXED) } != hipSuccess {
                die("graph capture at the cut", "hipStreamBeginCapture failed");
            }
        } else {
            self.cl_exchange(&rec);
        }
        self.cl_post(&rec);
    }

    /// Head: the host part of a hop. Waits for the bundle, sends STEP, receives RET into the pinned return buffer.
    fn cl_exchange(&mut self, rec: &CutRec) {
        let t0 = Instant::now();
        let e = self.d_model();
        let n = self.model().layers.len();
        let lo_w = self.model().stage_hi;
        let s = self.s_main;
        let _ = unsafe { hipStreamSynchronize(s) };
        let cl = self.cl.as_mut().unwrap();
        let mut c = StepCtl {
            t: rec.t,
            verify: rec.verify as u32,
            cur_p0_h: rec.cur_p0_h,
            q8: rec.q8 as u32,
            dbg_on: rec.dbg_on as u32,
            dbg_row: rec.dbg_row,
            graph: rec.graph as u32,
            n_taps: cl.wtaps.len() as u32,
            q8ret: rec.q8ret as u32,
            ..Default::default()
        };
        for (i, &li) in cl.wtaps.iter().enumerate() {
            c.taps[i] = li;
        }
        unsafe {
            let src = (&c as *const StepCtl).cast::<u8>().add(CTL_HOST_OFF);
            ptr::copy_nonoverlapping(
                src,
                cl.tx.add(CTL_HOST_OFF),
                std::mem::size_of::<StepCtl>() - CTL_HOST_OFF,
            );
        }
        let (_, _, _, _, len) = bundle_layout(rec.t, e, rec.q8);
        cl.step += 1;
        let mut h = Header::new(MsgType::Step);
        h.rows = rec.t as u32;
        h.step = cl.step;
        h.pos0 = rec.cur_p0_h.max(0) as u32;
        let bundle = unsafe { std::slice::from_raw_parts(cl.tx, len) };
        let link = cl.link.as_mut().unwrap();
        // the stepping thread receives every step: start TALUS_CLUSTER_SPIN_US here (once; a no-op afterwards)
        link.steady();
        if let Err(err) = link.send(h, &[bundle]) {
            die("send to the worker", err);
        }
        let rh = match link.recv_header() {
            Ok(rh) => rh,
            Err(err) => die("receive from the worker", err),
        };
        if rh.msg() != Some(MsgType::Ret) || rh.step != cl.step {
            let body = link.recv_vec(&rh).unwrap_or_default();
            die(
                "worker step",
                format!(
                    "expected RET {}, got {:?} {}: {}",
                    cl.step,
                    rh.msg(),
                    rh.step,
                    String::from_utf8_lossy(&body)
                ),
            );
        }
        let rbuf = unsafe { std::slice::from_raw_parts_mut(cl.rx, cl.rx_cap) };
        if let Err(err) = link.recv_into(&rh, rbuf) {
            die("receive the return", err);
        }
        if (rh.layout_id & 1 != 0) != rec.q8ret {
            die(
                "worker step",
                "the worker's final q8 mark differs from the head's prediction",
            );
        }
        cl.hops += 1;
        cl.hop_s += t0.elapsed().as_secs_f64();
        if rec.dbg_on && !self.dbg.is_null() {
            let n_taps = cl.wtaps.len();
            let (_, _, _, od, _) = ret_layout(rec.t, e, rec.q8ret, n_taps, n - lo_w);
            let row = (rec.t as usize) * (e as usize) * 4;
            let rx = cl.rx;
            for (k, li) in (lo_w..n).enumerate() {
                let dst = unsafe {
                    self.dbg
                        .add((li * super::engine::Q27_VMAX + rec.dbg_row as usize) * (e as usize))
                };
                hcopy(
                    dst.cast(),
                    unsafe { rx.add(od + k * row) }.cast(),
                    row,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    s,
                );
            }
        }
    }

    /// Head: device-side part after a hop: `h_last` (+ its q8 copy) and the worker's DFlash2 taps.
    fn cl_post(&mut self, rec: &CutRec) {
        let e = self.d_model();
        let s = self.s_main;
        let (rx, tap_dev, wtaps) = {
            let cl = self.cl.as_ref().unwrap();
            (cl.rx, cl.tap_dev, cl.wtaps.clone())
        };
        let row = (rec.t as usize) * (e as usize) * 4;
        let (oq, qb, ot, _, _) = ret_layout(rec.t, e, rec.q8ret, wtaps.len(), 0);
        hcopy(
            self.h_last.cast(),
            rx.cast_const().cast(),
            row,
            HIP_MEMCPY_HOST_TO_DEVICE,
            s,
        );
        if rec.q8ret {
            hcopy(
                self.x_q8,
                unsafe { rx.add(oq) }.cast(),
                qb,
                HIP_MEMCPY_HOST_TO_DEVICE,
                s,
            );
            self.mark_q8(self.h_last, e, rec.t);
        } else {
            self.mark_q8(ptr::null(), 0, 0);
        }
        if self.use_dflash && self.df.ready {
            for (k, &li) in wtaps.iter().enumerate() {
                let dst = unsafe { tap_dev.cast::<u8>().add(k * row) };
                hcopy(
                    dst.cast(),
                    unsafe { rx.add(ot + k * row) }.cast(),
                    row,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    s,
                );
                self.df
                    .tap(li + 1, dst.cast::<f32>().cast_const(), rec.t, s);
            }
        }
    }

    fn cl_send_ctl(&mut self, ty: MsgType, keep: u32, payload: &[u8]) {
        if let Some(cl) = self.cl.as_mut() {
            if cl.role != Role::Head {
                return;
            }
            if cl.cap_on {
                die("cluster mirror", format!("{ty:?} inside a graph capture"));
            }
            let mut h = Header::new(ty);
            h.commit_keep = keep;
            if let Err(e) = cl.link.as_mut().unwrap().send(h, &[payload]) {
                die("send to the worker", e);
            }
        }
    }

    /// Head: mirror `reset` (after the local one).
    pub(crate) fn cl_mirror_reset(&mut self) {
        if self.is_cluster_head() {
            self.cl_send_ctl(MsgType::Reset, 0, &[]);
        }
    }

    /// Head: mirror the fused GDN commit with the head's current `gctl` (before the local launch).
    pub(crate) fn cl_mirror_flush(&mut self) {
        if !self.is_cluster_head() {
            return;
        }
        let mut g = [0i32; 4];
        unsafe {
            let _ = hipStreamSynchronize(self.s_main);
            let _ = hipMemcpy(
                g.as_mut_ptr().cast::<c_void>(),
                self.gctl.cast::<c_void>().cast_const(),
                16,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
        }
        let b: Vec<u8> = g.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.cl_send_ctl(MsgType::Flush, 0, &b);
    }

    /// Head: mirror `commit(n)` (unfused GDN path).
    pub(crate) fn cl_mirror_commit(&mut self, n: i32) {
        if self.is_cluster_head() {
            self.cl_send_ctl(MsgType::Commit, n as u32, &[]);
        }
    }

    /// Head: whether the link to the worker is up (a link and, with a heartbeat, a live peer).
    pub fn cluster_link_up(&self) -> bool {
        self.cl.as_ref().is_some_and(|c| {
            c.role != Role::Head
                || (c.link.is_some() && !c.beat.as_ref().is_some_and(|b| b.is_dead()))
        })
    }

    /// Head: after a lost link (a caught `ClusterLost`), clean up what the interrupted step left (an open graph
    /// capture, queued work), reconnect with the same settings (the worker rebuilds or reuses its engine) and reset
    /// the sequence on both stages. The caller drops its prompt-cache state (the worker's shard is gone).
    pub fn cluster_reconnect(&mut self, wait_s: u64) -> std::result::Result<(), String> {
        let Some(recon) = self.cl.as_ref().and_then(|c| c.recon.clone()) else {
            return Err("not a cluster head".into());
        };
        let s = self.s_main;
        if let Some(cl) = self.cl.as_mut() {
            if cl.cap_on {
                let mut gr: hipGraph_t = ptr::null_mut();
                unsafe {
                    let _ = hipStreamEndCapture(s, &mut gr);
                    if !gr.is_null() {
                        let _ = hipGraphDestroy(gr);
                    }
                }
                for g in cl.cap_parts.drain(..) {
                    let _ = unsafe { hipGraphExecDestroy(g) };
                }
                cl.cap_cuts.clear();
                cl.cap_on = false;
            }
            cl.beat = None;
            cl.link = None;
            cl.step = 0;
        }
        let _ = unsafe { hipStreamSynchronize(s) };
        let m = unsafe { &*self.m };
        let hl = cluster_connect_head(
            m,
            &recon.addr,
            recon.alt.as_deref(),
            Some(recon.ctx),
            recon.max_batch,
            recon.use_graph,
            wait_s,
        )
        .map_err(|e| e.to_string())?;
        if let Some(cl) = self.cl.as_mut() {
            cl.link = Some(hl.link);
            cl.beat = hl.beat;
        }
        self.reset();
        eprintln!(
            "talus-cluster: link to the worker re-established ({})",
            hl.addr
        );
        Ok(())
    }

    /// Head: link statistics for the log.
    pub fn cluster_stats(&self) -> Option<(u64, f64)> {
        self.cl
            .as_ref()
            .filter(|c| c.role == Role::Head)
            .map(|c| (c.hops, c.hop_s))
    }

    // ------------------------------------------------------------------------------------------------------------
    // Worker
    // ------------------------------------------------------------------------------------------------------------

    /// Worker: the segment body for one step (`rx` holds the bundle; writes the return into `tx`).
    fn cl_worker_body(&mut self, c: &StepCtl) -> bool {
        let m_ptr = self.m;
        let m = unsafe { &*m_ptr };
        let e = m.g.n_embd as i32;
        let t = c.t;
        let s = self.s_main;
        let (rx, tx) = {
            let cl = self.cl.as_ref().unwrap();
            (cl.rx, cl.tx)
        };
        let q8 = c.q8 != 0;
        let verify = c.verify != 0;
        self.prep(t);
        self.cur_p0_h = c.cur_p0_h;
        let (ox, oh, oq, qb, _) = bundle_layout(t, e, q8);
        let row = (t as usize) * (e as usize) * 4;
        unsafe {
            hcopy(
                self.gctl.cast(),
                rx.cast_const().cast(),
                16,
                HIP_MEMCPY_HOST_TO_DEVICE,
                s,
            );
            hcopy(
                self.pos0_d.cast(),
                rx.add(16).cast(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
                s,
            );
            hcopy(
                self.x.cast(),
                rx.add(ox).cast(),
                row,
                HIP_MEMCPY_HOST_TO_DEVICE,
                s,
            );
            hcopy(
                self.h.cast(),
                rx.add(oh).cast(),
                row,
                HIP_MEMCPY_HOST_TO_DEVICE,
                s,
            );
            if q8 {
                hcopy(
                    self.x_q8,
                    rx.add(oq).cast(),
                    qb,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    s,
                );
            }
        }
        if let Some(p) = poison_mode() {
            self.cl_poison(&p, t, q8);
        }
        if q8 {
            self.mark_q8(self.h, e, t);
        } else {
            self.mark_q8(ptr::null(), 0, 0);
        }
        let n = m.layers.len();
        let n_taps = (c.n_taps as usize).min(MAX_TAPS);
        let q8ret = c.q8ret != 0;
        let (oq_r, qb_r, ot, od, _) = ret_layout(t, e, q8ret, n_taps, n - m.stage_lo);
        for (k, li) in (m.stage_lo..n).enumerate() {
            let last = li + 1 == n;
            let wn = if last {
                m.out_norm
            } else {
                m.layers[li + 1].attn_norm_ptr
            };
            let hout = if last { self.h_last } else { self.h };
            let layer_ref = unsafe { &(&(*m_ptr).layers)[li] };
            self.layer_into(li as i32, layer_ref, self.pos0_d, verify, wn, hout);
            for j in 0..n_taps {
                if c.taps[j] == li as i32 {
                    hcopy(
                        unsafe { tx.add(ot + j * row) }.cast(),
                        self.x.cast(),
                        row,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                        s,
                    );
                }
            }
            if c.dbg_on != 0 {
                hcopy(
                    unsafe { tx.add(od + k * row) }.cast(),
                    self.x.cast(),
                    row,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    s,
                );
            }
        }
        if self.fused && t <= self.gemm_t {
            let _ = tk_q36_ctl_end(self.tk.as_ptr(), self.gctl, verify, s.cast::<c_void>());
        }
        hcopy(
            tx.cast(),
            self.h_last.cast(),
            row,
            HIP_MEMCPY_DEVICE_TO_HOST,
            s,
        );
        let mark = self.q8_src == self.h_last as *const f32 && self.q8_n == e && self.q8_tok == t;
        if q8ret && mark {
            hcopy(
                unsafe { tx.add(oq_r) }.cast(),
                self.x_q8.cast_const(),
                qb_r,
                HIP_MEMCPY_DEVICE_TO_HOST,
                s,
            );
        }
        mark
    }

    /// Worker: run one STEP whose payload is in the receive buffer; returns (return bytes, q8 returned).
    pub fn cl_worker_step(&mut self) -> (usize, bool) {
        let c: StepCtl =
            unsafe { ptr::read_unaligned(self.cl.as_ref().unwrap().rx.cast::<StepCtl>()) };
        let m = unsafe { &*self.m };
        let e = m.g.n_embd as i32;
        let n_w = m.layers.len() - m.stage_lo;
        let n_taps = (c.n_taps as usize).min(MAX_TAPS);

        let n_dbg = if c.dbg_on != 0 { n_w } else { 0 };
        let (_, _, _, _, len) = ret_layout(c.t, e, c.q8ret != 0, n_taps, n_dbg);
        let (_, _, _, _, blen) = bundle_layout(c.t, e, c.q8 != 0);
        {
            let cl = self.cl.as_ref().unwrap();
            if len > cl.tx_cap || blen > cl.rx_cap || c.t <= 0 {
                die(
                    "worker step",
                    format!(
                        "T={} needs {len} / {blen} B, buffers {} / {}",
                        c.t, cl.tx_cap, cl.rx_cap
                    ),
                );
            }
        }
        let use_graph = c.graph != 0 && c.dbg_on == 0 && self.opts.use_graph;
        let mut ran: Option<bool> = None;
        if use_graph {
            let mut th = 0u64;
            for j in 0..n_taps {
                th = th.wrapping_mul(131).wrapping_add(c.taps[j] as u64 + 1);
            }
            let key = (c.t, c.verify != 0, c.q8 != 0, th ^ ((c.q8ret as u64) << 63));
            let mut g = self.cl.as_ref().unwrap().wgraphs.get(&key).copied();
            if g.is_none() {
                let cc = c;
                let mut mark = false;
                match self.capture_graph(|eng| mark = eng.cl_worker_body(&cc)) {
                    Ok(x) => {
                        self.cl.as_mut().unwrap().wgraphs.insert(key, (x, mark));
                        g = Some((x, mark));
                    }
                    Err(err) => {
                        eprintln!("talus-cluster: worker graph capture T={} failed ({err}); running eagerly", c.t)
                    }
                }
            }
            if let Some((x, mark)) = g {
                let _ = unsafe { hipGraphLaunch(x, self.s_main) };
                ran = Some(mark);
            }
        }
        let mark = match ran {
            Some(m) => m,
            None => self.cl_worker_body(&c),
        };
        let _ = unsafe { hipStreamSynchronize(self.s_main) };

        (len, c.q8ret != 0 && mark)
    }

    /// Worker: apply the head's fused GDN commit.
    pub fn cl_worker_flush(&mut self, gctl: &[u8]) {
        let mut g = [0i32; 4];
        for (i, v) in g.iter_mut().enumerate() {
            if gctl.len() >= 4 * (i + 1) {
                *v = i32::from_le_bytes([
                    gctl[4 * i],
                    gctl[4 * i + 1],
                    gctl[4 * i + 2],
                    gctl[4 * i + 3],
                ]);
            }
        }
        unsafe {
            let _ = hipStreamSynchronize(self.s_main);
            let _ = hipMemcpy(
                self.gctl.cast::<c_void>(),
                g.as_ptr().cast::<c_void>(),
                16,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
        }
        self.gdn_pending = self.gdn_pending.max(1);
        self.flush_gdn();
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Handshake
// ----------------------------------------------------------------------------------------------------------------

/// Engine-shaping settings the head sends in INIT (one source of truth: the worker adopts them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageInit {
    pub ctx: i32,
    pub max_batch: i32,
    pub no_v16: bool,
    pub attn_w8: bool,
    pub use_graph: bool,
}

/// The 27B's per-stage auto context: the largest of 256K / 128K / 64K whose KV cache fits beside every stage's int8
/// prompt weights and reserve. Stages on the same box share one headroom (Phase 1 loopback), so their needs add up;
/// a stage on its own box checks its own headroom.
pub fn cluster_auto_context(
    head_i8: f64,
    head_pt: f64,
    head_room: f64,
    worker_i8: f64,
    worker_pt: f64,
    worker_room: f64,
    same_box: bool,
) -> (i32, String) {
    let gib = |b: f64| b / f64::from(1u32 << 30);
    let reserve = super::engine::Q27_AUTO_RESERVE_GIB * f64::from(1u32 << 30);
    let fits = |c: i32| {
        let nh = head_i8 + head_pt * f64::from(c) + reserve;
        let nw = worker_i8 + worker_pt * f64::from(c) + reserve;
        if same_box {
            nh + nw <= head_room
        } else {
            nh <= head_room && nw <= worker_room
        }
    };
    let ladder = super::engine::Q27_AUTO_CONTEXTS;
    let pick = ladder
        .iter()
        .copied()
        .find(|&c| fits(c))
        .unwrap_or(ladder[ladder.len() - 1]);
    let why = format!(
        "context {pick} (cluster auto, {}): head int8 {:.1} GiB + KV {:.0} KiB/token, worker int8 {:.1} GiB + KV {:.0} \
         KiB/token, reserve {:.0} GiB per stage; headroom {:.1} GiB{}",
        if same_box { "both stages on one box" } else { "one stage per box" },
        gib(head_i8),
        head_pt / 1024.0,
        gib(worker_i8),
        worker_pt / 1024.0,
        super::engine::Q27_AUTO_RESERVE_GIB,
        gib(head_room),
        if same_box { String::new() } else { format!(" (worker {:.1} GiB)", gib(worker_room)) }
    );
    (pick, why)
}

fn hello_kv(model: &Q36Model) -> Kv {
    let mut kv = Kv::new();
    let env = ident::env_pairs();
    kv.put("proto", STAGE_PROTO)
        .put("exe", ident::exe_sha())
        .put("lib", ident::kernels_sha())
        .put(
            "model",
            model
                .file_path
                .as_ref()
                .map(|p| ident::file_head_sha(&p.to_string_lossy(), 32 << 20))
                .unwrap_or_else(|| "none".into()),
        )
        .put("env", ident::env_hash(&env))
        .put("n_layer", model.layers.len())
        .put("box", ident::box_id());
    for (k, v) in &env {
        kv.put(
            &format!("env.{k}"),
            ident::env_hash(&[(k.clone(), v.clone())]),
        );
    }
    kv
}

/// Head: connect, check identity, agree on the context, have the worker build its engine. Returns the link and the
/// context to build the head engine with.
pub fn cluster_connect_head(
    model: &Q36Model,
    addr: &str,
    alt: Option<&str>,
    ctx_explicit: Option<i32>,
    max_batch: i32,
    use_graph: bool,
    wait_s: u64,
) -> super::Result<HeadLink> {
    eprintln!(
        "talus-cluster: head connecting to the worker at {addr}{} (waiting up to {wait_s} s)",
        alt.map(|a| format!(" (alt {a})")).unwrap_or_default()
    );
    // the primary address first; the alternate (LAN) one when the primary does not answer (link down)
    let mut used = addr.to_string();
    let mut link = match Link::connect(
        addr,
        Duration::from_secs(if alt.is_some() { 5.min(wait_s) } else { wait_s }),
    ) {
        Ok(l) => l,
        Err(e0) => match alt {
            Some(a) => {
                eprintln!("talus-cluster: {addr} unreachable ({e0}); trying the alternate {a}");
                used = a.to_string();
                Link::connect(a, Duration::from_secs(wait_s))
                    .or_else(|_| {
                        Link::connect(addr, Duration::from_secs(5))
                            .inspect(|_| used = addr.to_string())
                    })
                    .map_err(|e| super::ffi::anyhow!("cluster connect {addr} / {a}: {e}"))?
            }
            None => return Err(super::ffi::anyhow!("cluster connect {addr}: {e0}")),
        },
    };
    let addr = used.as_str();
    // the data plane (talus#48): TCP today; HELLO/INIT and the heartbeat always run on TCP
    let dp = talus_cluster::DataPlane::from_env().map_err(|e| super::ffi::anyhow!("{e}"))?;
    let mut kv = hello_kv(model);
    kv.put("split", model.stage_hi).put("dp", dp.as_str());
    let io = |e: std::io::Error| super::ffi::anyhow!("cluster handshake: {e}");
    link.send_msg(MsgType::Hello, &kv.encode()).map_err(io)?;
    let (_, ack) = link.expect(MsgType::HelloAck).map_err(io)?;
    let ack = Kv::decode(&ack);
    if talus_cluster::DataPlane::parse(ack.get("dp")) != Ok(dp) {
        return Err(super::ffi::anyhow!(
            "cluster handshake: the worker answered data plane {:?}, asked for {}",
            ack.get("dp"),
            dp.as_str()
        ));
    }
    let i8h = super::engine::i8x_candidates(model).1 as f64;
    let pth = q27_bytes_per_token(model) as f64;
    let ctx = match ctx_explicit {
        Some(c) => c,
        None => {
            let room = q27_budget().headroom_bytes() as f64;
            let same = ack.get("box") == Some(ident::box_id().as_str());
            let (c, why) = cluster_auto_context(
                i8h,
                pth,
                room,
                ack.get_i64("i8").unwrap_or(0) as f64,
                ack.get_i64("per_tok").unwrap_or(0) as f64,
                ack.get_i64("room").unwrap_or(0) as f64,
                same,
            );
            eprintln!("talus-cli: {why}");
            c
        }
    };
    let mut init = Kv::new();
    init.put("ctx", ctx)
        .put("max_batch", max_batch)
        .put("no_v16", model.opts.no_v16 as i32)
        .put("attn_w8", model.opts.attn_w8 as i32)
        .put("use_graph", use_graph as i32);
    link.send_msg(MsgType::Init, &init.encode()).map_err(io)?;
    let (_, iack) = link.expect(MsgType::InitAck).map_err(io)?;
    let iack = Kv::decode(&iack);
    eprintln!(
        "talus-cluster: worker ready: context {}, {}",
        iack.get("ctx").unwrap_or("?"),
        iack.get("note").unwrap_or("")
    );
    // the heartbeat on the control port (data port + 1)
    let beat = match std::net::TcpStream::connect_timeout(
        &std::net::ToSocketAddrs::to_socket_addrs(&ctl_addr(addr))
            .map_err(io)?
            .next()
            .ok_or_else(|| super::ffi::anyhow!("no control address for {addr}"))?,
        Duration::from_secs(10),
    ) {
        Ok(c) => {
            let data = link.closer().map_err(io)?;
            Some(talus_cluster::Beat::start(c, data, "head").map_err(io)?)
        }
        Err(e) => {
            eprintln!(
                "talus-cluster: WARN: no control connection at {} ({e}): no heartbeat",
                ctl_addr(addr)
            );
            None
        }
    };
    Ok(HeadLink {
        link,
        beat,
        ctx,
        addr: addr.to_string(),
    })
}

/// Worker: check a HELLO against this process.
fn check_hello(model: &Q36Model, h: &Kv) -> std::result::Result<(), String> {
    let mine = hello_kv(model);
    if h.get_i64("proto") != Some(STAGE_PROTO) {
        return Err(format!(
            "mismatch proto: head {:?}, worker {STAGE_PROTO}",
            h.get("proto")
        ));
    }
    for f in ["exe", "lib", "model", "n_layer"] {
        if h.get(f) != mine.get(f) {
            return Err(format!(
                "mismatch {f}: head {:?}, worker {:?}",
                h.get(f),
                mine.get(f)
            ));
        }
    }
    if h.get_i64("split") != Some(model.stage_lo as i64) {
        return Err(format!(
            "mismatch split: head cuts at {:?}, worker owns {}..",
            h.get("split"),
            model.stage_lo
        ));
    }
    talus_cluster::DataPlane::parse(h.get("dp")).map_err(|e| format!("mismatch dp: {e}"))?;
    if h.get("env") != mine.get("env") {
        let names = |kv: &Kv| -> Vec<(String, String)> {
            kv.0.iter()
                .filter_map(|(k, v)| k.strip_prefix("env.").map(|n| (n.to_string(), v.clone())))
                .collect()
        };
        let d = ident::env_diff(&names(h), &names(&mine));
        return Err(format!("mismatch env: variables differ: {}", d.join(" ")));
    }
    Ok(())
}

/// Worker: serve heads forever on `addr` (one at a time). The weights stay loaded across heads; the engine is
/// rebuilt only when a head asks for different engine settings.
pub fn worker_main(model: &mut Q36Model, addr: &str) -> i32 {
    let listener = match Link::listen(addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("talus-cluster: worker cannot listen on {addr}: {e}");
            return 1;
        }
    };
    let ctl_listener = match Link::listen(&ctl_addr(addr)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "talus-cluster: worker cannot listen on {}: {e}",
                ctl_addr(addr)
            );
            return 1;
        }
    };
    eprintln!(
        "talus-cluster: worker stage layers {}..{} of {} listening on {addr}",
        model.stage_lo,
        model.stage_hi,
        model.layers.len()
    );
    let once = std::env::var("TALUS_CLUSTER_ONCE")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut eng: Option<Q36Engine> = None;
    let mut cur: Option<StageInit> = None;
    loop {
        let mut link = match Link::accept(&listener) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("talus-cluster: accept: {e}");
                continue;
            }
        };
        eprintln!("talus-cluster: head connected from {}", link.peer);
        // the head's control connection follows once the handshake is done; it is accepted on a thread so the
        // session can start (heads without one simply run without a heartbeat)
        let data2 = link.closer().ok();
        let cl2 = ctl_listener.try_clone().ok();
        let quit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let quit2 = quit.clone();
        let beat_th = std::thread::spawn(move || -> Option<talus_cluster::Beat> {
            let l = cl2?;
            let _ = l.set_nonblocking(true);
            let t0 = Instant::now();
            loop {
                match l.accept() {
                    Ok((c, _)) => {
                        let _ = c.set_nonblocking(false);
                        return talus_cluster::Beat::start(c, data2?, "worker").ok();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        if quit2.load(std::sync::atomic::Ordering::Relaxed)
                            || t0.elapsed() > Duration::from_secs(900)
                        {
                            return None;
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) => return None,
                }
            }
        });
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            worker_session(model, &mut eng, &mut cur, &mut link)
        }));
        if let Ok(c) = link.closer() {
            c.close();
        }
        quit.store(true, std::sync::atomic::Ordering::Relaxed);
        let beat = beat_th.join().ok().flatten();
        drop(beat);
        match r {
            Ok(Ok(())) => eprintln!("talus-cluster: head said BYE"),
            Ok(Err(e)) => eprintln!("talus-cluster: head session ended: {e}"),
            Err(_) => {
                eprintln!("talus-cluster: head session failed mid-step: rebuilding the engine (weights stay loaded)");
                eng = None;
                cur = None;
            }
        }
        if let Some(e) = eng.as_mut() {
            e.reset();
            if let Some(cl) = e.cl.as_ref() {
                eprintln!("talus-cluster: session served {} steps", cl.step);
            }
            if let Some(cl) = e.cl.as_mut() {
                cl.step = 0;
            }
        }
        if once {
            return 0;
        }
    }
}

fn worker_session(
    model: &mut Q36Model,
    eng: &mut Option<Q36Engine>,
    cur: &mut Option<StageInit>,
    link: &mut Link,
) -> std::result::Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    let (_, hello) = link.expect(MsgType::Hello).map_err(io)?;
    let hello = Kv::decode(&hello);
    if let Err(why) = check_hello(model, &hello) {
        eprintln!("talus-cluster: refusing head: {why}");
        let _ = link.send_msg(MsgType::Err, why.as_bytes());
        return Err(why);
    }
    let mut ack = Kv::new();
    let dp = talus_cluster::DataPlane::parse(hello.get("dp"))?;
    ack.put("ok", 1)
        .put("dp", dp.as_str())
        .put("box", ident::box_id())
        .put("i8", super::engine::i8x_candidates(model).1)
        .put("per_tok", q27_bytes_per_token(model))
        .put("room", q27_budget().headroom_bytes());
    link.send_msg(MsgType::HelloAck, &ack.encode())
        .map_err(io)?;
    let (_, init) = link.expect(MsgType::Init).map_err(io)?;
    let init = Kv::decode(&init);
    let want = StageInit {
        ctx: init.get_i64("ctx").unwrap_or(0) as i32,
        max_batch: init.get_i64("max_batch").unwrap_or(512) as i32,
        no_v16: init.get_i64("no_v16") == Some(1),
        attn_w8: init.get_i64("attn_w8") == Some(1),
        use_graph: init.get_i64("use_graph") != Some(0),
    };
    if want.ctx < 256 {
        let _ = link.send_msg(MsgType::Err, b"bad context");
        return Err("bad context in INIT".into());
    }
    if *cur != Some(want) || eng.is_none() {
        *eng = None; // drop before changing the model's options
        model.opts.no_v16 = want.no_v16;
        model.opts.attn_w8 = want.attn_w8;
        model.opts.use_graph = want.use_graph;
        model.opts.max_seq = want.ctx;
        model.cfg.max_seq = want.ctx;
        let t0 = Instant::now();
        let mut e = Q36Engine::new(model, want.ctx).map_err(|e| format!("engine init: {e}"))?;
        e.max_batch = want.max_batch;
        e.cluster_attach_worker()
            .map_err(|e| format!("cluster buffers: {e}"))?;
        eprintln!(
            "talus-cluster: worker engine context {} max_batch {} built in {:.1} s",
            want.ctx,
            want.max_batch,
            t0.elapsed().as_secs_f64()
        );
        *eng = Some(e);
        *cur = Some(want);
    }
    let e = eng.as_mut().unwrap();
    e.reset();
    let mut ia = Kv::new();
    ia.put("ctx", want.ctx).put(
        "note",
        format!(
            "int8 prompt weights {:.2} GiB, layers {}..{}",
            e.i8c.bytes as f64 / f64::from(1u32 << 30),
            model.stage_lo,
            model.stage_hi
        ),
    );
    link.send_msg(MsgType::InitAck, &ia.encode()).map_err(io)?;
    // the engine is built: this thread receives every step from now on (TALUS_CLUSTER_SPIN_US)
    link.steady();
    // this stage's shard of the prompt cache (created by the head's first cache message; dropped with the session)
    let mut pc: Option<Q36PromptCache> = None;
    loop {
        let h = link.recv_header().map_err(io)?;
        match h.msg() {
            Some(MsgType::Step) => {
                let cl = e.cl.as_ref().unwrap();
                let buf = unsafe { std::slice::from_raw_parts_mut(cl.rx, cl.rx_cap) };
                link.recv_into(&h, buf).map_err(io)?;
                let (len, q8ret) = e.cl_worker_step();
                let cl = e.cl.as_mut().unwrap();
                cl.step = h.step;
                let mut r = Header::new(MsgType::Ret);
                r.step = h.step;
                r.rows = h.rows;
                r.layout_id = q8ret as u32;
                let out = unsafe { std::slice::from_raw_parts(cl.tx, len) };
                link.send(r, &[out]).map_err(io)?;
            }
            Some(MsgType::Reset) => {
                let _ = link.recv_vec(&h).map_err(io)?;
                e.reset();
            }
            Some(MsgType::Flush) => {
                let g = link.recv_vec(&h).map_err(io)?;
                e.cl_worker_flush(&g);
            }
            Some(MsgType::Commit) => {
                let _ = link.recv_vec(&h).map_err(io)?;
                e.commit(h.commit_keep as i32);
            }
            Some(MsgType::Bye) => return Ok(()),
            Some(MsgType::Heartbeat) => {
                let _ = link.recv_vec(&h).map_err(io)?;
            }
            Some(MsgType::Cache) => {
                let body = link.recv_vec(&h).map_err(io)?;
                if let Some(reply) = e.cl_worker_cache(&mut pc, &body)? {
                    link.send_msg(MsgType::CacheAck, &reply).map_err(io)?;
                }
            }
            other => {
                let _ = link.recv_vec(&h);
                return Err(format!("unexpected message {other:?}"));
            }
        }
    }
}

/// Default cut for the 27B: the middle of the trunk, on a 4-layer boundary (3 GDN + 1 attention per group).
pub fn default_split(n_layer: usize) -> usize {
    (n_layer / 2) / 4 * 4
}

/// Head: say goodbye (the worker keeps its weights and waits for the next head).
pub fn cluster_bye(eng: &mut Q36Engine) {
    if let Some(cl) = eng.cl.as_mut() {
        if cl.role == Role::Head {
            if let Some(l) = cl.link.as_mut() {
                let _ = l.send_msg(MsgType::Bye, &[]);
            }
            if cl.hops > 0 {
                eprintln!(
                    "talus-cluster: {} hops, mean {:.3} ms host time per hop (sync + send + worker + receive)",
                    cl.hops,
                    1000.0 * cl.hop_s / cl.hops as f64
                );
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Poison test (DESIGN.md 1.2 / 4.2 items 6-7)
// ----------------------------------------------------------------------------------------------------------------
//
// `TALUS_CLUSTER_POISON` on the worker, after the bundle is copied in:
// - `scratch`: fill every scratch buffer a layer writes before it reads (and x_q8 when no q8 copy was shipped) with
//   NaN. The logits sha must stay equal: nothing outside the bundle is live at the cut.
// - `x`, `h`, `q8`: replace that shipped segment with NaN. The sha must change (the control: the test can see it).

fn poison_mode() -> Option<String> {
    static M: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        std::env::var("TALUS_CLUSTER_POISON")
            .ok()
            .filter(|v| !v.is_empty())
    })
    .clone()
}

impl Q36Engine {
    fn cl_poison(&mut self, mode: &str, t: i32, q8: bool) {
        let m = unsafe { &*self.m };
        let g = &m.g;
        let b = self.max_batch.max(16) as usize;
        let e = g.n_embd as usize;
        let c = g.ssm_qkv_dim() as usize;
        let qg = g.q_gate_dim() as usize;
        let qd = g.q_dim() as usize;
        let kv = g.kv_dim() as usize;
        let v = g.ssm_inner_size as usize;
        let hv = g.ssm_time_step_rank as usize;
        let f = g.n_ff as usize;
        let s = self.s_main;
        let nan = |p: *mut c_void, bytes: usize| {
            if !p.is_null() && bytes > 0 {
                let _ = unsafe { hipMemsetAsync(p, 0xff, bytes, s) };
            }
        };
        let row = (t as usize) * e * 4;
        match mode {
            "x" => nan(self.x.cast(), row),
            "h" => nan(self.h.cast(), row),
            "q8" => {
                if q8 {
                    nan(self.x_q8, tk_q8_1_bytes(e as i32, t))
                }
            }
            "scratch" => {
                let f4 = |n: usize| n * 4;
                nan(self.y.cast(), f4(b * v.max(e)));
                nan(self.qkv.cast(), f4(b * c.max(qg + 2 * kv)));
                nan(self.hc.cast(), f4(b * c));
                nan(self.z.cast(), f4(b * v.max(qd)));
                nan(self.alpha.cast(), f4(b * 2 * hv));
                nan(self.beta.cast(), f4(b * hv));
                nan(self.qg.cast(), f4(b * qg));
                nan(self.q.cast(), f4(b * qd));
                nan(self.k.cast(), f4(b * kv));
                nan(self.v.cast(), f4(b * kv));
                nan(self.gate.cast(), f4(b * qd));
                nan(self.att.cast(), f4(b * qd));
                if m.dense {
                    nan(self.sh_g.cast(), f4(b * f));
                    nan(self.sh_u.cast(), f4(b * f));
                    nan(self.sh_y.cast(), f4(b * e));
                }
                nan(self.h_last.cast(), f4(b * e));
                let max_in = e.max(v).max(qd).max(f) as i32;
                nan(self.y_q8, tk_q8_1_bytes(max_in, b as i32));
                if !q8 {
                    nan(self.x_q8, tk_q8_1_bytes(max_in, b as i32));
                } else {
                    // only the rows past the shipped ones
                    let shipped = tk_q8_1_bytes(e as i32, t);
                    let all = tk_q8_1_bytes(max_in, b as i32);
                    nan(
                        unsafe { self.x_q8.cast::<u8>().add(shipped) }.cast(),
                        all - shipped,
                    );
                }
            }
            _ => {}
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Sharded prompt cache (talus#47 Phase 2)
// ----------------------------------------------------------------------------------------------------------------
//
// Every stage holds a `Q36PromptCache` over its own layers. The head mirrors each cache call (stash, save, restore)
// to the worker in order; both run the same code on the same token arrays, and the stash budget is charged with the
// whole model's bytes on both (`Q36PromptCache::acct_peer`), so both take the same decisions. A restore returns the
// worker's restored length, which the head checks against its own (a mismatch is a bug: the head exits).

const PC_INIT: u8 = 0;
const PC_STASH: u8 = 1;
const PC_SAVE: u8 = 2;
const PC_RESTORE: u8 = 3;

fn put_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put_ids(b: &mut Vec<u8>, ids: &[i32]) {
    put_u64(b, ids.len() as u64);
    for &t in ids {
        b.extend_from_slice(&t.to_le_bytes());
    }
}

struct Rd<'a> {
    b: &'a [u8],
    o: usize,
}

impl Rd<'_> {
    fn u64(&mut self) -> std::result::Result<u64, String> {
        let s = self
            .b
            .get(self.o..self.o + 8)
            .ok_or("short cache message")?;
        self.o += 8;
        let mut a = [0u8; 8];
        a.copy_from_slice(s);
        Ok(u64::from_le_bytes(a))
    }
    fn ids(&mut self) -> std::result::Result<Vec<i32>, String> {
        let n = self.u64()? as usize;
        let s = self
            .b
            .get(self.o..self.o + 4 * n)
            .ok_or("short cache message")?;
        self.o += 4 * n;
        Ok(s.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

/// A stage's KV shape for the cache accounting: attention states, those with a std-layout V, KV heads, head dim,
/// DFlash2 rings / keep rows / kv dim.
type KvShape = [i64; 7];

impl Q36Engine {
    /// The prompt-cache view of this engine's own layers (as talus-cli's `make_layer_state_view`, which the head
    /// uses; flushing the pending GDN commit is the head's job and is mirrored).
    pub fn cl_state_view(&mut self) -> Q36LayerStateView {
        let g = &self.model().g;
        let na = self.n_attn_state as usize;
        let (dk, dv, ring, keep, kvd, block) = if self.use_dflash && self.df.ready {
            (
                self.df.k_ring.clone(),
                self.df.v_ring.clone(),
                self.df.ring(),
                self.df.keep(),
                self.df.mdl.n_kv * self.df.mdl.head_dim,
                self.df.block(),
            )
        } else {
            (Vec::new(), Vec::new(), 0, 0, 0, 0)
        };
        Q36LayerStateView {
            ssm_d_conv: 4,
            ssm_conv_channels: g.ssm_qkv_dim() as i32,
            ssm_v_heads: g.ssm_time_step_rank as i32,
            ssm_state_size: g.ssm_state_size as i32,
            n_head_kv: g.n_head_kv as i32,
            head_dim: g.head_dim as i32,
            n_embd: g.n_embd as i32,
            ctx: self.ctx_len,
            vt_ctx: self.vt_ctx,
            has_mtp: self.model().has_mtp,
            conv: self.conv.clone(),
            s: self.S.clone(),
            kc: self.kc[..na].iter().map(|&p| p.cast::<c_void>()).collect(),
            vc: self.vc[..na].iter().map(|&p| p.cast::<c_void>()).collect(),
            vtc: self.vtc[..na].iter().map(|&p| p.cast::<c_void>()).collect(),
            hlast: self.h_mtp,
            gctl: self.gctl,
            gdn_pending: 0,
            pos: self.n_past as i64,
            st: self.s_main,
            dfl_k_ring: dk,
            dfl_v_ring: dv,
            dfl_ring: ring,
            dfl_keep: keep,
            dfl_kv_dim: kvd,
            dfl_block: block,
            dctx_lo: self.dctx.lo(),
            dctx_hi: self.dctx.hi(),
        }
    }

    fn cl_kv_shape(&self) -> KvShape {
        let g = &self.model().g;
        let na = self.n_attn_state as usize;
        let nv = self.vc.iter().take(na).filter(|p| !p.is_null()).count();
        let (dn, keep, kvd) = if self.use_dflash && self.df.ready {
            (
                self.df.k_ring.len(),
                self.df.keep(),
                self.df.mdl.n_kv * self.df.mdl.head_dim,
            )
        } else {
            (0, 0, 0)
        };
        [
            na as i64,
            nv as i64,
            g.n_head_kv,
            g.head_dim,
            dn as i64,
            keep as i64,
            kvd as i64,
        ]
    }

    fn cl_head_cache(&mut self, payload: &[u8], want_reply: bool) -> Option<Vec<u8>> {
        let cl = self.cl.as_mut()?;
        if cl.role != Role::Head {
            return None;
        }
        let link = cl.link.as_mut().unwrap();
        if let Err(e) = link.send_msg(MsgType::Cache, payload) {
            die("send a cache op to the worker", e);
        }
        if !want_reply {
            return None;
        }
        match link.expect(MsgType::CacheAck) {
            Ok((_, b)) => Some(b),
            Err(e) => die("cache op on the worker", e),
        }
    }

    /// Head: create the worker's cache shard with this cache's settings and make both shards charge the whole
    /// model's KV bytes. No-op outside a cluster head.
    pub fn cl_pc_init(&mut self, pc: &mut Q36PromptCache) {
        if !self.is_cluster_head() {
            return;
        }
        let mut b = vec![PC_INIT];
        put_u64(&mut b, pc.max_snaps as u64);
        put_u64(&mut b, pc.max_msg as u64);
        put_u64(&mut b, pc.stash_budget as u64);
        for v in self.cl_kv_shape() {
            put_u64(&mut b, v as u64);
        }
        let r = self.cl_head_cache(&b, true).unwrap_or_default();
        let mut rd = Rd { b: &r, o: 0 };
        let mut w = [0i64; 7];
        for x in w.iter_mut() {
            *x = rd.u64().unwrap_or(0) as i64;
        }
        pc.acct_peer = Some(shape_view(&w));
        eprintln!("talus-cluster: prompt cache sharded over both stages (worker KV shape {w:?})");
    }

    /// Head: mirror `stash_kv_with_pf(live, keep, pf)` (call right after the local one).
    pub fn cl_pc_stash(&mut self, live: &[i32], keep: usize, pf: usize) {
        if !self.is_cluster_head() {
            return;
        }
        let mut b = vec![PC_STASH];
        put_u64(&mut b, keep as u64);
        put_u64(&mut b, pf as u64);
        put_ids(&mut b, live);
        self.cl_head_cache(&b, false);
    }

    /// Head: mirror `save(ids, msg)`.
    pub fn cl_pc_save(&mut self, ids: &[i32], msg: bool) {
        if !self.is_cluster_head() {
            return;
        }
        let mut b = vec![PC_SAVE];
        put_u64(&mut b, msg as u64);
        put_ids(&mut b, ids);
        self.cl_head_cache(&b, false);
    }

    /// Head: mirror `restore_with_pf(ids, live, live_pf, pf_only)`; `r` is the head's restored length, which the
    /// worker's must equal.
    pub fn cl_pc_restore(
        &mut self,
        ids: &[i32],
        live: &[i32],
        live_pf: usize,
        pf_only: bool,
        r: usize,
    ) {
        if !self.is_cluster_head() {
            return;
        }
        let mut b = vec![PC_RESTORE];
        put_u64(&mut b, live_pf as u64);
        put_u64(&mut b, pf_only as u64);
        put_ids(&mut b, ids);
        put_ids(&mut b, live);
        let rep = self.cl_head_cache(&b, true).unwrap_or_default();
        let rw = Rd { b: &rep, o: 0 }.u64().unwrap_or(u64::MAX) as usize;
        if rw != r {
            die(
                "prompt cache shards diverged",
                format!("head restored {r} tokens, worker {rw}"),
            );
        }
    }

    /// Worker: apply one mirrored cache op on this stage's shard; returns the reply payload if the op has one.
    fn cl_worker_cache(
        &mut self,
        pc: &mut Option<Q36PromptCache>,
        body: &[u8],
    ) -> std::result::Result<Option<Vec<u8>>, String> {
        let op = *body.first().ok_or("empty cache message")?;
        let mut rd = Rd { b: body, o: 1 };
        match op {
            PC_INIT => {
                let mut c = Q36PromptCache::new();
                c.max_snaps = rd.u64()? as usize;
                c.max_msg = rd.u64()? as usize;
                c.stash_budget = rd.u64()? as usize;
                let mut hs = [0i64; 7];
                for x in hs.iter_mut() {
                    *x = rd.u64()? as i64;
                }
                c.acct_peer = Some(shape_view(&hs));
                *pc = Some(c);
                let mut b = Vec::new();
                for v in self.cl_kv_shape() {
                    put_u64(&mut b, v as u64);
                }
                Ok(Some(b))
            }
            PC_STASH => {
                let keep = rd.u64()? as usize;
                let pf = rd.u64()? as usize;
                let live = rd.ids()?;
                let view = self.cl_state_view();
                pc.as_mut()
                    .ok_or("cache op before INIT")?
                    .stash_kv_with_pf(&view, &live, keep, pf);
                Ok(None)
            }
            PC_SAVE => {
                let msg = rd.u64()? != 0;
                let ids = rd.ids()?;
                let mut view = self.cl_state_view();
                pc.as_mut()
                    .ok_or("cache op before INIT")?
                    .save(&mut view, &ids, msg);
                Ok(None)
            }
            PC_RESTORE => {
                let live_pf = rd.u64()? as usize;
                let pf_only = rd.u64()? != 0;
                let ids = rd.ids()?;
                let live = rd.ids()?;
                let mut view = self.cl_state_view();
                let (r, _) = pc
                    .as_mut()
                    .ok_or("cache op before INIT")?
                    .restore_with_pf(&mut view, &ids, &live, live_pf, pf_only);
                if r > 0 {
                    // as sync_layer_state_view_back on the head: the restored state is committed
                    self.gdn_pending = 0;
                    self.gdn_valid = true;
                }
                let mut b = Vec::new();
                put_u64(&mut b, r as u64);
                Ok(Some(b))
            }
            x => Err(format!("unknown cache op {x}")),
        }
    }
}

fn shape_view(s: &KvShape) -> Q36LayerStateView {
    q36_shape_view(
        s[0].max(0) as usize,
        s[1].max(0) as usize,
        s[2] as i32,
        s[3] as i32,
        (s[4].max(0) as usize, s[5] as i32, s[6] as i32),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_are_contiguous() {
        let e = 5120;
        let (ox, oh, oq, qb, total) = bundle_layout(4, e, true);
        assert_eq!(ox, CTL_BYTES);
        assert_eq!(oh, ox + 4 * 5120 * 4);
        assert_eq!(oq, oh + 4 * 5120 * 4);
        assert_eq!(total, oq + qb);
        assert_eq!(qb, tk_q8_1_bytes(e, 4));
        let (_, _, _, _, t2) = bundle_layout(4, e, false);
        assert_eq!(t2, oq);
        let (rq, rqb, rt, rd, rtot) = ret_layout(3, e, true, 2, 5);
        assert_eq!(rq, 3 * 5120 * 4);
        assert_eq!(rt, rq + rqb);
        assert_eq!(rd, rt + 2 * 3 * 5120 * 4);
        assert_eq!(rtot, rd + 5 * 3 * 5120 * 4);
    }

    #[test]
    fn auto_context_shares_one_box() {
        let gib = f64::from(1u32 << 30);
        // 27B-like: 13 GiB int8 + 34 KiB/token per stage
        let (one_box, _) = cluster_auto_context(
            13.0 * gib,
            34816.0,
            50.0 * gib,
            13.0 * gib,
            34816.0,
            50.0 * gib,
            true,
        );
        let (two_box, _) = cluster_auto_context(
            13.0 * gib,
            34816.0,
            50.0 * gib,
            13.0 * gib,
            34816.0,
            50.0 * gib,
            false,
        );
        assert_eq!(two_box, 262_144);
        // on one box the two stages need 2 x (13 + 8.5 + 3) = 49 GiB at 256K: fits in 50
        assert_eq!(one_box, 262_144);
        let (tight, _) = cluster_auto_context(
            13.0 * gib,
            34816.0,
            42.0 * gib,
            13.0 * gib,
            34816.0,
            42.0 * gib,
            true,
        );
        // 2 x (13 + 4.25 + 3) = 40.5 GiB at 128K fits in 42; 256K (49) does not
        assert_eq!(tight, 131_072);
    }

    #[test]
    fn default_split_on_group_boundary() {
        assert_eq!(default_split(64), 32);
        assert_eq!(default_split(48), 24);
        assert_eq!(default_split(46), 20);
    }
}
