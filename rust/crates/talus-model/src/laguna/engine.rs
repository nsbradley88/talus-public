//! Laguna-S-2.1 (`laguna`) inference engine (`src/laguna/laguna_engine.hpp` & `laguna_engine.cpp`).
//!
//! Implements prefill (with grouped MoE MMQ and WMMA flash attention), single-token decode
//! (captured as dense and block-sparse HIP graphs), multi-token speculative verification
//! (`1..=8` tokens, lazily captured as HIP graphs per batch size), eager batch reference path
//! (`--batch 1..8`), and disk KV session save/load (`TALUSKV2`).

use std::ffi::c_void;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

use hip_sys::hipStream_t;
use talus_core_api::{CoreError, Result};
use talus_kernels_sys::{tk_mmvq_item, tk_moe_mmq_product, tk_stream, LibTalusKernels};
use talus_mem::{Budget, BudgetCategory, ReservationGuard};

use crate::laguna::ffi::{
    check_hip, check_tk, tk_add_inplace, tk_build_rope_table, tk_f32_to_f16_bulk, HipGraph,
    HipGraphExec, HipRuntime, RopeScalingParams, TkContext, HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    TK_ROPE_SCALING_NONE, TK_ROPE_SCALING_YARN,
};
use crate::laguna::session::{compute_ctx16_and_ring, session_payload_bytes, SessionHeader};

/// Laguna DFlash drafter (a child module: it drives the engine's stream, kernels and taps).
#[path = "dflash.rs"]
pub mod dflash;
use crate::laguna::weights::{
    dequant_embd_row, is_full, n_head, to_tk_quant_type, vs_index, vt_index, DevTensor,
    LagunaModel, K_EMBD, K_EPS, K_EXPERTS, K_EXP_SCALE, K_FF_DENSE, K_FF_EXP, K_HEAD_DIM,
    K_HEAD_KV, K_SWA, K_TOP_K, K_VOCAB,
};

/// Maximum number of verification tokens in a single `verify` step (`laguna_engine.hpp:kMaxVerify` was 8).
/// Sized to 64 for long prompt-lookup drafts: the routed experts of a wide window share row blocks in the cache,
/// amortizing weight traffic across verified tokens while every row stays bitwise equal to `verify(1)`.
pub const K_MAX_VERIFY: usize = 64;

/// Router mat-vec window (`tk_f32_matvec_rows_n` takes at most 8 tokens; wider verifies run it per 8 tokens, each
/// token's sum is independent of the window).
const K_ROUTER_WIN: usize = 8;
/// default TALUS_LAGUNA_EGROUP_MIN (verify widths from which the routed experts run grouped by expert; the
/// per-type grouped kernel is faster from 6 tokens on gfx1151)
const K_EGROUP_MIN: usize = 6;

/// Decode flash-attention split count (`laguna_engine.cpp:kSplit`).
const K_SPLIT: i32 = 16;

/// Padding past `U * K_TOP_K` on `ident` so MMQ's widest column tile never reads past the buffer
/// (`laguna_engine.cpp:kIdentPad`).
const K_IDENT_PAD: usize = 512;

/// Engine configuration options (`src/laguna/laguna_engine.hpp:EngineOptions`).
#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub ubatch: usize,
    pub use_wmma: bool,
    pub kv_q8: bool,
    pub graph: bool,
    pub sparse: bool,
    pub sparse_bs: i32,
    pub sparse_k: i32,
    pub sparse_min: i32,
    pub sparse_sink: i32,
    pub sparse_recent: i32,
    /// Capture the residual stream at the DFlash drafter's target layers (`TAP_LAYERS`) in prefill, decode
    /// and verify (device copies only: the target's numerics are unchanged).
    pub taps: bool,
    /// Route `step_n`'s dense mat-vecs through `tk_mmvq` (with `STRATA_MMVQ_W32=1`: wave32 kernels, bitwise across
    /// column counts); `TALUS_LAGUNA_DENSE_MMVQ=0/1` overrides.
    pub dense_mmvq: bool,
}

/// Laguna DFlash target layers: the residual entering layers 2, 11, 20, 30, 39 and the last layer's output (48).
pub const TAP_LAYERS: [usize; 6] = [2, 11, 20, 30, 39, 48];

impl Default for Options {
    fn default() -> Self {
        Self {
            ubatch: 2048,
            use_wmma: true,
            kv_q8: true,
            graph: true,
            sparse: false,
            sparse_bs: 64,
            sparse_k: 64,
            sparse_min: 16384,
            sparse_sink: 128,
            sparse_recent: 1024,
            taps: false,
            dense_mmvq: false,
        }
    }
}

/// Optional layer-dump callback invoked during `prefill`: `(name, host_f32_slice, rows, cols)`.
pub type DumpFn<'a> = &'a mut dyn FnMut(&str, &[f32], usize, usize);

/// Laguna-S-2.1 inference engine (`src/laguna/laguna_engine.hpp:Engine`).
#[allow(non_snake_case)]
pub struct Engine {
    pub m: Arc<LagunaModel>,
    ctx: usize,
    U: usize,
    use_wmma: bool,
    kv_q8: bool,
    use_graph: bool,
    emaj: bool,
    s: hipStream_t,
    tk_ctx: TkContext,
    tk: Arc<LibTalusKernels>,
    hip: Arc<HipRuntime>,
    _kv_reservation: Option<ReservationGuard>,
    /// int8 prefill cache of dense projections (default Q5_K / Q6_K / Q8_0: every dense projection;
    /// TALUS_LAGUNA_I8_TYPES selects types), keyed by the GGUF tensor's device pointer: (int8 values, scales, tp_i8gemm
    /// mode). Empty with TALUS_LAGUNA_I8=0.
    i8c: std::collections::HashMap<usize, (usize, usize, i32)>,
    _i8_res: Option<ReservationGuard>,
    allocs: Vec<*mut c_void>,

    // RoPE tables: full layers use dim=64 YaRN (theta=5e5, factor=32, orig_ctx=8192);
    // SWA layers use dim=128 plain (theta=1e4).
    cos_full: *mut f32,
    sin_full: *mut f32,
    cos_swa: *mut f32,
    sin_swa: *mut f32,

    // KV pools
    cell: usize,
    ctx16: usize,
    ring: usize,
    vtl: Vec<usize>,
    kpool: Vec<*mut u16>,
    vpool: Vec<*mut u16>,
    kscale: Vec<*mut u16>,
    vscale: Vec<*mut u16>,
    kv_bytes_total: usize,

    // Block-sparse decode attention on the 12 full-attention layers (requires kv_q8)
    sparse: bool,
    sp_bs: i32,
    sp_k: i32,
    sp_min: i32,
    sp_sink: i32,
    sp_recent: i32,
    sp_per_kv: bool,
    sp_f16: bool,
    nb_max: i32,
    bnd_valid: i32,
    kbnd: Vec<*mut c_void>,
    sp_scr: *mut u32,
    sp_ub: *mut f32,
    sp_sel: *mut i32,
    mmq_kq: bool,

    // Prefill scratch (sized for U tokens)
    x: *mut f32,
    h: *mut f32,
    q: *mut f32,
    kc: *mut f32,
    vc: *mut f32,
    g: *mut f32,
    att: *mut f32,
    a: *mut f32,
    f1: *mut f32,
    f2: *mut f32,
    ysh: *mut f32,
    yexp: *mut f32,
    rl: *mut f32,
    logits_buf: *mut f32,
    pos_q: *mut i32,
    pos_k: *mut i32,
    ident: *mut i32,
    one_b: *mut i32,
    sel_e: *mut i32,
    sel_w: *mut f32,
    cnt: *mut i32,
    bnds: *mut i32,
    gtok: *mut i32,
    smap: *mut i32,
    xq8: *mut c_void,
    xq8g: *mut c_void,
    q8c: *mut c_void,
    ptok_dev: *mut i32,
    one_b_rows: std::sync::atomic::AtomicI32,

    // Single-token decode scratch
    dx: *mut f32,
    dh: *mut f32,
    dq: *mut f32,
    dkc: *mut f32,
    dvc: *mut f32,
    dg: *mut f32,
    da: *mut f32,
    df1: *mut f32,
    df2: *mut f32,
    dysh: *mut f32,
    dyexp: *mut f32,
    drl: *mut f32,
    dlogits: *mut f32,
    dpart: *mut f32,
    dml: *mut f32,
    dids: *mut i32,
    dw: *mut f32,
    route_counter: *mut u32,
    dq8: *mut c_void,
    dq8b: *mut c_void,
    q8c_ff: usize,
    tokpos_dev: *mut i32,
    pos_dev: *mut i32,

    // Multi-token verify scratch (up to K_MAX_VERIFY = 8 tokens)
    vx: *mut f32,
    vh: *mut f32,
    vq: *mut f32,
    vkc: *mut f32,
    vvc: *mut f32,
    vg: *mut f32,
    va: *mut f32,
    vf1: *mut f32,
    vf2: *mut f32,
    vysh: *mut f32,
    vyexp: *mut f32,
    vrl: *mut f32,
    vlogits: *mut f32,
    vpart: *mut f32,
    vml: *mut f32,
    vids: *mut i32,
    vw: *mut f32,
    vtok: *mut i32,
    vout: *mut i32,
    vzero: *mut i32,
    vq8: *mut c_void,
    vq8b: *mut c_void,

    // Captured HIP graphs for decode and verify(1..=8)
    exec: HipGraphExec,
    exec_sp: HipGraphExec,
    vexec: [HipGraphExec; K_MAX_VERIFY + 1],
    vexec_sp: [HipGraphExec; K_MAX_VERIFY + 1],

    // Host staging for embedding lookup during prefill
    hx: Vec<f32>,

    // DFlash taps [TAP_LAYERS.len()][tap_rows][K_EMBD] (null when Options::taps is off)
    taps: *mut f32,
    tap_rows: usize,
    tap_idx: *mut i32,
    // verify with decode's kernels for wo / layer 0 FFN / the head (TALUS_LAGUNA_VERIFY_EXACT)
    vexact: bool,
    // decode through the verify kernels (TALUS_LAGUNA_DEC_VERIFY=0: legacy decode kernels)
    dec_verify: bool,
    // per-token sparse bounds / scores in verify (TALUS_LAGUNA_SPARSE_EXACT=0: one pass over the window)
    sparse_exact: bool,
    vposa: *mut i32,
    wide_min: usize,
    // highest KV position written so far (+1): clear_full_kv_from zeroes the full layers' cells up to it
    kv_hw: usize,
    // a sliding-window ring cell was written since the last clear (clear_kv_from with swa)
    swa_dirty: bool,
    // clear_kv_from: partial V tile up to the tile boundary (TALUS_LAGUNA_CLEAR_TILE=0: legacy bound)
    clear_tile_fix: bool,
    // step_n's dense mat-vecs through tk_mmvq (TALUS_LAGUNA_DENSE_MMVQ)
    dense_mmvq: bool,
    // verify windows of this many tokens and up run the routed experts grouped by expert
    // (expert_major = 3: each expert's rows read once per window; bitwise the per-id kernel). 0: off.
    egroup_min: usize,
    // (opt-in, numerics change) verify windows of this many tokens and up run through the prefill kernels
    // (MMQ dense / grouped experts, WMMA prompt attention, all-row head). 0: off.
    vgemm_min: usize,
    // zero the KV cells past a GEMM verify window first (TALUS_LAGUNA_VERIFY_GEMM_CLEAR=0: off)
    vgemm_clear: bool,
    // columns per dense tk_mmvq call in step_n (TALUS_LAGUNA_DENSE_COLS, 1..=64; 16 = the earlier chunking)
    dense_cols: usize,
}

// SAFETY: Engine owns its HIP stream, graphs, and device allocations.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

#[allow(non_snake_case)]
impl Engine {
    /// Create a new Laguna inference engine, allocating KV pools, RoPE tables, and scratch buffers.
    pub fn new(
        m: Arc<LagunaModel>,
        max_ctx: usize,
        opt: Options,
        budget: Option<Arc<Budget>>,
        hip: Arc<HipRuntime>,
        tk: Arc<LibTalusKernels>,
    ) -> Result<Self> {
        if std::env::var_os("STRATA_MMVQ_W32").is_none() {
            // Match C++ laguna_model.hpp::mmvq_pre, which calls native_q{4,5,6}_k_mmvq /
            // native_q8_0_mmvq directly rather than native_mmvq_w32.
            unsafe {
                std::env::set_var("STRATA_MMVQ_W32", "0");
            }
        }
        // Enable STRATA_MMQ_KQ=1 for dense projections by default (`STRATA_MMQ_KQ=0` disables it).
        // Evaluated BEFORE tk_q36_mmq_built()/tk_q36_mmq_supported_fits() runs ensure_q36_env().
        let mmq_kq = std::env::var("STRATA_MMQ_KQ")
            .map(|v| !v.starts_with('0'))
            .unwrap_or(true);
        unsafe {
            let _ = (tk.tk_q36_mmq_built)();
        }
        let U = opt.ubatch.max(1);
        let use_wmma = opt.use_wmma && std::env::var("LAGUNA_NO_WMMA").is_err();
        let kv_q8 = opt.kv_q8;
        let use_graph = opt.graph;
        let emaj = std::env::var("LAGUNA_EXPERT_MAJOR")
            .map(|v| v.parse::<i32>().unwrap_or(1) != 0)
            .unwrap_or(true);
        let sparse = opt.sparse && opt.kv_q8;
        let sp_bs = if opt.sparse_bs == 16 || opt.sparse_bs == 32 || opt.sparse_bs == 64 {
            opt.sparse_bs
        } else {
            64
        };
        let sp_k = if opt.sparse_k > 0 { opt.sparse_k } else { 64 };
        let sp_min_floor = if std::env::var("LAGUNA_SPARSE_32K")
            .map(|v| v.parse::<i32>().unwrap_or(1) != 0)
            .unwrap_or(true)
        {
            32000.max(sp_bs * sp_k)
        } else {
            4 * sp_bs * sp_k
        };
        let sp_min = std::env::var("LAGUNA_SPARSE_MIN")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .map(|m| m.max(sp_bs * sp_k))
            .unwrap_or_else(|| opt.sparse_min.max(sp_min_floor));
        let sp_sink = opt.sparse_sink;
        let sp_recent = opt.sparse_recent;
        let sp_per_kv = std::env::var("LAGUNA_SPARSE_UNION")
            .map(|v| !v.starts_with('1'))
            .unwrap_or(true);
        let sp_f16 = std::env::var("LAGUNA_SPARSE_F32")
            .map(|v| !v.starts_with('1'))
            .unwrap_or(true);

        let mut s: hipStream_t = std::ptr::null_mut();
        check_hip(unsafe { (hip.hipStreamCreate)(&mut s) }, "hipStreamCreate")?;

        let tk_ctx = match TkContext::new(&tk, 0, s as tk_stream) {
            Ok(c) => c,
            Err(e) => {
                unsafe {
                    let _ = (hip.hipStreamDestroy)(s);
                }
                return Err(e);
            }
        };

        let mut allocs: Vec<*mut c_void> = Vec::new();
        let mut kv_bytes_total: usize = 0;

        let init_res = (|| -> Result<Self> {
            let mut alloc = |bytes: usize| -> Result<*mut c_void> {
                let p = hip.alloc_device_zeroed(bytes)?;
                allocs.push(p);
                Ok(p)
            };

            // Build RoPE tables on host and upload to device
            let cos_full = alloc(max_ctx * 32 * 4)? as *mut f32;
            let sin_full = alloc(max_ctx * 32 * 4)? as *mut f32;
            let cos_swa = alloc(max_ctx * 64 * 4)? as *mut f32;
            let sin_swa = alloc(max_ctx * 64 * 4)? as *mut f32;

            {
                let mut hcos = vec![0.0_f32; max_ctx * 32];
                let mut hsin = vec![0.0_f32; max_ctx * 32];
                let full_p = RopeScalingParams {
                    scaling_type: TK_ROPE_SCALING_YARN,
                    freq_base: 500_000.0,
                    factor: 32.0,
                    freq_scale_in: 0.0,
                    orig_ctx: 8192,
                    ext_factor: 1.0,
                    attn_factor: 1.0,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                };
                tk_build_rope_table(&mut hcos, &mut hsin, max_ctx, 64, &full_p);
                hip.copy_h2d(
                    cos_full as *mut c_void,
                    hcos.as_ptr() as *const c_void,
                    hcos.len() * 4,
                )?;
                hip.copy_h2d(
                    sin_full as *mut c_void,
                    hsin.as_ptr() as *const c_void,
                    hsin.len() * 4,
                )?;
            }
            {
                let mut hcos = vec![0.0_f32; max_ctx * 64];
                let mut hsin = vec![0.0_f32; max_ctx * 64];
                let swa_p = RopeScalingParams {
                    scaling_type: TK_ROPE_SCALING_NONE,
                    freq_base: 10_000.0,
                    ..Default::default()
                };
                tk_build_rope_table(&mut hcos, &mut hsin, max_ctx, 128, &swa_p);
                hip.copy_h2d(
                    cos_swa as *mut c_void,
                    hcos.as_ptr() as *const c_void,
                    hcos.len() * 4,
                )?;
                hip.copy_h2d(
                    sin_swa as *mut c_void,
                    hsin.as_ptr() as *const c_void,
                    hsin.len() * 4,
                )?;
            }

            let cell = K_HEAD_KV * K_HEAD_DIM;
            let (ctx16, ring) = compute_ctx16_and_ring(max_ctx, U);
            let nl = m.layers.len();
            let mut kpool = vec![std::ptr::null_mut(); nl];
            let mut vpool = vec![std::ptr::null_mut(); nl];
            let mut kscale = vec![std::ptr::null_mut(); nl];
            let mut vscale = vec![std::ptr::null_mut(); nl];
            let mut vtl = vec![0usize; nl];
            let kv_elem = if kv_q8 { 1usize } else { 2usize };

            for (il, slot) in vtl.iter_mut().enumerate().take(nl) {
                *slot = if kv_q8 && !is_full(il) { ring } else { ctx16 };
                let bytes = (*slot) * cell * kv_elem;
                kv_bytes_total += 2 * bytes;
                if kv_q8 {
                    let sb = (*slot) * K_HEAD_KV * 4 * 2;
                    kv_bytes_total += 2 * sb;
                }
            }

            let mut nb_max = 0i32;
            let mut kbnd = vec![std::ptr::null_mut(); nl];
            let mut sp_scr: *mut u32 = std::ptr::null_mut();
            let mut sp_ub: *mut f32 = std::ptr::null_mut();
            let mut sp_sel: *mut i32 = std::ptr::null_mut();

            if sparse {
                nb_max = ((ctx16 as i32) + sp_bs - 1) / sp_bs;
                let bsz = if sp_f16 { 2usize } else { 4usize };
                for il in 0..nl {
                    if is_full(il) {
                        let b = (nb_max as usize) * K_HEAD_KV * 2 * K_HEAD_DIM * bsz;
                        kv_bytes_total += b;
                    }
                }
                kv_bytes_total += K_MAX_VERIFY * 72 * (nb_max as usize) * 4;
                kv_bytes_total +=
                    K_MAX_VERIFY * K_HEAD_KV * (nb_max as usize).max(sp_k as usize) * 4;
                kv_bytes_total += K_MAX_VERIFY * K_HEAD_KV * (nb_max as usize) * 4;
            }

            // Reserve KV + sparse budget BEFORE allocating KV/sparse device buffers
            let kv_reservation = if let Some(b) = &budget {
                let guard = b
                    .try_reserve_guard(BudgetCategory::KvCache, "laguna.kv_cache", kv_bytes_total)
                    .map_err(|_| CoreError::OutOfMemory {
                        requested_bytes: kv_bytes_total as u64,
                        available_bytes: b
                            .try_headroom_bytes()
                            .unwrap_or(0)
                            .try_into()
                            .unwrap_or(0),
                    })?;
                Some(guard)
            } else {
                None
            };

            for il in 0..nl {
                let bytes = vtl[il] * cell * kv_elem;
                kpool[il] = alloc(bytes)? as *mut u16;
                vpool[il] = alloc(bytes)? as *mut u16;
                if kv_q8 {
                    let sb = vtl[il] * K_HEAD_KV * 4 * 2;
                    kscale[il] = alloc(sb)? as *mut u16;
                    vscale[il] = alloc(sb)? as *mut u16;
                }
            }

            if sparse {
                let bsz = if sp_f16 { 2usize } else { 4usize };
                for (il, slot) in kbnd.iter_mut().enumerate().take(nl) {
                    if is_full(il) {
                        let b = (nb_max as usize) * K_HEAD_KV * 2 * K_HEAD_DIM * bsz;
                        *slot = alloc(b)?;
                    }
                }
                sp_ub = alloc(K_MAX_VERIFY * 72 * (nb_max as usize) * 4)? as *mut f32;
                sp_sel = alloc(K_MAX_VERIFY * K_HEAD_KV * (nb_max as usize).max(sp_k as usize) * 4)?
                    as *mut i32;
                sp_scr = alloc(K_MAX_VERIFY * K_HEAD_KV * (nb_max as usize) * 4)? as *mut u32;
            }

            if let (Some(b), Some(guard)) = (&budget, &kv_reservation) {
                let _ = b.mark_materialized(guard.id());
            }

            // Prefill scratch (sized for U tokens)
            let x = alloc(U * K_EMBD * 4)? as *mut f32;
            let h = alloc(U * K_EMBD * 4)? as *mut f32;
            let q = alloc(U * 72 * K_HEAD_DIM * 4)? as *mut f32;
            let kc = alloc(U * cell * 4)? as *mut f32;
            let vc = alloc(U * cell * 4)? as *mut f32;
            let g = alloc(U * 72 * 4)? as *mut f32;
            let att = alloc(U * 72 * K_HEAD_DIM * 4)? as *mut f32;
            let a = alloc(U * K_EMBD * 4)? as *mut f32;
            let f1 = alloc(U.max(1) * K_FF_DENSE.max(K_TOP_K * K_FF_EXP) * 4)? as *mut f32;
            let f2 = alloc(U.max(1) * K_FF_DENSE.max(K_TOP_K * K_FF_EXP) * 4)? as *mut f32;
            let ysh = alloc(U * K_EMBD * 4)? as *mut f32;
            let yexp = alloc(U * K_TOP_K * K_EMBD * 4)? as *mut f32;
            let rl = alloc(U * K_EXPERTS * 4)? as *mut f32;
            let logits_buf = alloc(U * K_VOCAB * 4)? as *mut f32;
            let pos_q = alloc(U * 72 * 4)? as *mut i32;
            let pos_k = alloc(U * K_HEAD_KV * 4)? as *mut i32;
            let ident = alloc((U * K_TOP_K + K_IDENT_PAD) * 4)? as *mut i32;
            let one_b = alloc(2 * 4)? as *mut i32;
            let sel_e = alloc(U * K_TOP_K * 4)? as *mut i32;
            let sel_w = alloc(U * K_TOP_K * 4)? as *mut f32;
            let cnt = alloc(K_EXPERTS * 4)? as *mut i32;
            let bnds = alloc((K_EXPERTS + 1) * 4)? as *mut i32;
            let gtok = alloc(U * K_TOP_K * 4)? as *mut i32;
            let smap = alloc(U * K_TOP_K * 4)? as *mut i32;

            let q8_bytes = unsafe {
                (tk.tk_q36_mmq_q8_bytes)(U as i64, K_FF_DENSE as i64)
                    .max((tk.tk_q36_mmq_q8_bytes)(U as i64, (72 * K_HEAD_DIM) as i64))
            };
            let q8g_bytes = unsafe {
                (tk.tk_q36_mmq_q8_bytes)((U * K_TOP_K) as i64, K_EMBD as i64).max((tk
                    .tk_q36_mmq_q8_bytes)(
                    (U * K_TOP_K) as i64,
                    K_FF_EXP as i64,
                ))
            };
            let q8c_bytes = unsafe { (tk.tk_q8_1_bytes)(K_FF_DENSE as i32, 8) };
            let xq8 = alloc(q8_bytes)?;
            let xq8g = alloc(q8g_bytes)?;
            let q8c = alloc(q8c_bytes)?;
            let ptok_dev = alloc(max_ctx * 4)? as *mut i32;

            check_tk(
                tk_ctx.as_ptr(),
                unsafe {
                    (tk.tk_q36_mmq_iota)(
                        tk_ctx.as_ptr(),
                        ident,
                        (U * K_TOP_K + K_IDENT_PAD) as i64,
                        s as tk_stream,
                    )
                },
                "tk_q36_mmq_iota",
            )?;

            // Single-token decode scratch
            let dx = alloc(K_EMBD * 4)? as *mut f32;
            let dh = alloc(K_EMBD * 4)? as *mut f32;
            let dq = alloc(72 * K_HEAD_DIM * 4)? as *mut f32;
            let dkc = alloc(cell * 4)? as *mut f32;
            let dvc = alloc(cell * 4)? as *mut f32;
            let dg = alloc(72 * 4)? as *mut f32;
            let da = alloc(K_EMBD * 4)? as *mut f32;
            let df1 = alloc(K_FF_DENSE.max((K_TOP_K + 1) * K_FF_EXP) * 4)? as *mut f32;
            let df2 = alloc(K_FF_DENSE.max((K_TOP_K + 1) * K_FF_EXP) * 4)? as *mut f32;
            let dysh = alloc(K_EMBD * 4)? as *mut f32;
            let dyexp = alloc(K_TOP_K * K_EMBD * 4)? as *mut f32;
            let drl = alloc(K_EXPERTS * 4)? as *mut f32;
            let dlogits = alloc(K_VOCAB * 4)? as *mut f32;
            let dpart = alloc(72 * (K_SPLIT as usize) * K_HEAD_DIM * 4)? as *mut f32;
            let dml = alloc(72 * (K_SPLIT as usize) * 2 * 4)? as *mut f32;
            let dids = alloc(K_TOP_K * 4)? as *mut i32;
            let dw = alloc(K_TOP_K * 4)? as *mut f32;
            let route_counter = alloc(4)? as *mut u32;
            let q8c_ff = unsafe { (tk.tk_q8_1_bytes)(K_FF_EXP as i32, 1) };
            let dq8 = alloc(unsafe { (tk.tk_q8_1_bytes)(K_FF_DENSE as i32, 1) })?;
            let dq8b = alloc(unsafe {
                (tk.tk_q8_1_bytes)(K_FF_DENSE as i32, 1).max((K_TOP_K + 1) * q8c_ff)
            })?;
            let tokpos_dev = alloc(2 * 4)? as *mut i32;
            let pos_dev = unsafe { tokpos_dev.add(1) };

            // Multi-token verify scratch (up to K_MAX_VERIFY = 8 tokens)
            let vx = alloc(K_MAX_VERIFY * K_EMBD * 4)? as *mut f32;
            let vh = alloc(K_MAX_VERIFY * K_EMBD * 4)? as *mut f32;
            let vq = alloc(K_MAX_VERIFY * 72 * K_HEAD_DIM * 4)? as *mut f32;
            let vkc = alloc(K_MAX_VERIFY * cell * 4)? as *mut f32;
            let vvc = alloc(K_MAX_VERIFY * cell * 4)? as *mut f32;
            let vg = alloc(K_MAX_VERIFY * 72 * 4)? as *mut f32;
            let va = alloc(K_MAX_VERIFY * K_EMBD * 4)? as *mut f32;
            let vf1 =
                alloc(K_MAX_VERIFY * K_FF_DENSE.max((K_TOP_K + 1) * K_FF_EXP) * 4)? as *mut f32;
            let vf2 =
                alloc(K_MAX_VERIFY * K_FF_DENSE.max((K_TOP_K + 1) * K_FF_EXP) * 4)? as *mut f32;
            let vysh = alloc(K_MAX_VERIFY * K_EMBD * 4)? as *mut f32;
            let vyexp = alloc(K_MAX_VERIFY * K_TOP_K * K_EMBD * 4)? as *mut f32;
            let vrl = alloc(K_MAX_VERIFY * K_EXPERTS * 4)? as *mut f32;
            let vlogits = alloc(K_MAX_VERIFY * K_VOCAB * 4)? as *mut f32;
            let vpart = alloc(K_MAX_VERIFY * 72 * (K_SPLIT as usize) * K_HEAD_DIM * 4)? as *mut f32;
            let vml = alloc(K_MAX_VERIFY * 72 * (K_SPLIT as usize) * 2 * 4)? as *mut f32;
            let vids = alloc(K_MAX_VERIFY * K_TOP_K * 4)? as *mut i32;
            let vw = alloc(K_MAX_VERIFY * K_TOP_K * 4)? as *mut f32;
            let vtok = alloc(K_MAX_VERIFY * 4)? as *mut i32;
            let vout = alloc(K_MAX_VERIFY * 4)? as *mut i32;
            let vzero = alloc(K_MAX_VERIFY * K_TOP_K * 4)? as *mut i32;
            let vq8 = alloc(unsafe { (tk.tk_q8_1_bytes)(K_FF_DENSE as i32, K_MAX_VERIFY as i32) })?;
            let vq8b = alloc(unsafe {
                (tk.tk_q8_1_bytes)(K_FF_DENSE as i32, K_MAX_VERIFY as i32)
                    .max(K_MAX_VERIFY * (K_TOP_K + 1) * q8c_ff)
            })?;

            check_hip(
                unsafe { (hip.hipStreamSynchronize)(s) },
                "hipStreamSynchronize(init)",
            )?;

            let vposa = alloc(K_MAX_VERIFY * 4)? as *mut i32;
            let tap_rows = U.max(K_MAX_VERIFY);
            let taps = if opt.taps {
                alloc(TAP_LAYERS.len() * tap_rows * K_EMBD * 4)? as *mut f32
            } else {
                std::ptr::null_mut()
            };
            let tap_idx = if opt.taps {
                let p = alloc(tap_rows * 4)? as *mut i32;
                let ident: Vec<i32> = (0..tap_rows as i32).collect();
                hip.copy_h2d(
                    p as *mut c_void,
                    ident.as_ptr() as *const c_void,
                    tap_rows * 4,
                )?;
                p
            } else {
                std::ptr::null_mut()
            };
            Ok(Self {
                m: Arc::clone(&m),
                ctx: max_ctx,
                U,
                use_wmma,
                kv_q8,
                use_graph,
                emaj,
                s,
                tk_ctx,
                tk: Arc::clone(&tk),
                hip: Arc::clone(&hip),
                i8c: std::collections::HashMap::new(),
                _i8_res: None,
                _kv_reservation: kv_reservation,
                allocs: std::mem::take(&mut allocs),
                cos_full,
                sin_full,
                cos_swa,
                sin_swa,
                cell,
                ctx16,
                ring,
                vtl,
                kpool,
                vpool,
                kscale,
                vscale,
                kv_bytes_total,
                sparse,
                sp_bs,
                sp_k,
                sp_min,
                sp_sink,
                sp_recent,
                sp_per_kv,
                sp_f16,
                nb_max,
                bnd_valid: 0,
                kbnd,
                sp_scr,
                sp_ub,
                sp_sel,
                mmq_kq,
                x,
                h,
                q,
                kc,
                vc,
                g,
                att,
                a,
                f1,
                f2,
                ysh,
                yexp,
                rl,
                logits_buf,
                pos_q,
                pos_k,
                ident,
                one_b,
                sel_e,
                sel_w,
                cnt,
                bnds,
                gtok,
                smap,
                xq8,
                xq8g,
                q8c,
                ptok_dev,
                one_b_rows: std::sync::atomic::AtomicI32::new(-1),
                dx,
                dh,
                dq,
                dkc,
                dvc,
                dg,
                da,
                df1,
                df2,
                dysh,
                dyexp,
                drl,
                dlogits,
                dpart,
                dml,
                dids,
                dw,
                route_counter,
                dq8,
                dq8b,
                q8c_ff,
                tokpos_dev,
                pos_dev,
                vx,
                vh,
                vq,
                vkc,
                vvc,
                vg,
                va,
                vf1,
                vf2,
                vysh,
                vyexp,
                vrl,
                vlogits,
                vpart,
                vml,
                vids,
                vw,
                vtok,
                vout,
                vzero,
                vq8,
                vq8b,
                exec: std::ptr::null_mut(),
                exec_sp: std::ptr::null_mut(),
                vexec: [std::ptr::null_mut(); K_MAX_VERIFY + 1],
                vexec_sp: [std::ptr::null_mut(); K_MAX_VERIFY + 1],
                hx: vec![0.0_f32; U * K_EMBD],
                taps,
                tap_rows,
                tap_idx,
                vexact: std::env::var("TALUS_LAGUNA_VERIFY_EXACT")
                    .map(|v| v == "1")
                    .unwrap_or(false),
                dec_verify: std::env::var("TALUS_LAGUNA_DEC_VERIFY")
                    .map(|v| v != "0")
                    .unwrap_or(false),
                sparse_exact: std::env::var("TALUS_LAGUNA_SPARSE_EXACT")
                    .map(|v| v != "0")
                    .unwrap_or(true),
                vposa,
                kv_hw: 0,
                swa_dirty: false,
                clear_tile_fix: std::env::var("TALUS_LAGUNA_CLEAR_TILE")
                    .map(|v| v != "0")
                    .unwrap_or(true),
                wide_min: std::env::var("TALUS_LAGUNA_WIDE_MIN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(9)
                    .max(2),
                dense_mmvq: std::env::var("TALUS_LAGUNA_DENSE_MMVQ")
                    .map(|v| v == "1")
                    .unwrap_or(opt.dense_mmvq),
                egroup_min: std::env::var("TALUS_LAGUNA_EGROUP_MIN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(K_EGROUP_MIN),
                vgemm_min: std::env::var("TALUS_LAGUNA_VERIFY_GEMM")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0),
                vgemm_clear: std::env::var("TALUS_LAGUNA_VERIFY_GEMM_CLEAR")
                    .map(|v| v != "0")
                    .unwrap_or(true),
                dense_cols: std::env::var("TALUS_LAGUNA_DENSE_COLS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(K_MAX_VERIFY)
                    .clamp(1, K_MAX_VERIFY),
            })
        })();

        if init_res.is_err() {
            for p in allocs {
                hip.free_device(p);
            }
            unsafe {
                let _ = (hip.hipStreamDestroy)(s);
            }
        }
        let mut eng = init_res?;
        eng.build_i8_cache(budget.as_ref());
        if eng.use_graph
            && std::env::var("TALUS_LAGUNA_LAZY_GRAPH")
                .map(|v| v != "1" && !v.eq_ignore_ascii_case("true"))
                .unwrap_or(true)
        {
            eng.ensure_decode_graph(false)?;
            if eng.sparse {
                eng.ensure_decode_graph(true)?;
            }
        }
        Ok(eng)
    }

    /// expand the Q5_K dense prefill projections (attn_q, attn_output) into the int8 prefill cache
    /// (`talus_pf_i8x_expand`, MMQ's tile loaders). Prefill products (nb >= 128) then run `talus_pf_i8gemm` on the
    /// same MMQ-layout activations: the same output bits as `tk_prefill_moe_mmq` (both bitwise MMQ). Budget-guarded;
    /// `TALUS_LAGUNA_I8=0` disables.
    fn build_i8_cache(&mut self, budget: Option<&Arc<Budget>>) {
        if std::env::var("TALUS_LAGUNA_I8")
            .map(|v| v == "0")
            .unwrap_or(false)
        {
            return;
        }
        // TALUS_LAGUNA_I8_TYPES: ggml type ids to cache (`,` or `/` separated; default 13,14,8 = Q5_K attn_q / attn_output,
        // Q6_K attn_k / attn_v / shared expert, Q8_0 layer-0 dense FFN; "13" = the Q5_K-only cache of 751c8a42)
        let types: Vec<i32> = std::env::var("TALUS_LAGUNA_I8_TYPES")
            .unwrap_or_else(|_| "13,14,8".to_string())
            .split([',', '/'])
            .filter_map(|t| t.trim().parse::<i32>().ok())
            .collect();
        // talus_pf_i8x_mode (tp_i8x.cu): 0 per-32 scale, 1 per-16 scales, 2 d + min
        let mode_of = |t: i32| -> i32 {
            match t {
                23 | 20 | 21 | 18 | 16 | 8 | 6 => 0,
                11 | 17 | 22 | 14 => 1,
                12 | 13 => 2,
                _ => -1,
            }
        };
        let mut cands: Vec<(usize, i32, usize, usize, i32)> = Vec::new();
        for l in self.m.layers.iter() {
            for w in [
                &l.wq, &l.wk, &l.wv, &l.wo, &l.d_gate, &l.d_up, &l.d_down, &l.sh_gate, &l.sh_up,
                &l.sh_down,
            ] {
                if w.dev.is_null() {
                    continue;
                }
                let (k, n) = (w.ne[0] as usize, w.ne[1] as usize);
                let qt = to_tk_quant_type(w.qtype) as i32;
                let m = mode_of(qt);
                if m < 0 || !types.contains(&qt) || n % 128 != 0 || k % 256 != 0 {
                    continue;
                }
                cands.push((w.dev as usize, qt, n, k, m));
            }
        }
        let size =
            |n: usize, k: usize, m: i32| (n * k, n * (k / 32) * if m == 0 { 1 } else { 2 } * 4);
        let total: usize = cands
            .iter()
            .map(|&(_, _, n, k, m)| {
                let (a, b) = size(n, k, m);
                a + b
            })
            .sum();
        if total == 0 {
            return;
        }
        let res = match budget {
            Some(b) => {
                match b.try_reserve_guard(BudgetCategory::Weights, "laguna.i8_prefill_cache", total)
                {
                    Ok(g) => Some(g),
                    Err(e) => {
                        eprintln!(
                            "laguna: int8 prefill cache skipped ({:.2} GiB): {e}",
                            total as f64 / (1u64 << 30) as f64
                        );
                        return;
                    }
                }
            }
            None => None,
        };
        for (dev, qt, n, k, m) in cands {
            if self.i8c.contains_key(&dev) {
                continue;
            }
            let (qb, wb) = size(n, k, m);
            let Ok(qs) = self.hip.alloc_device(qb) else {
                break;
            };
            self.allocs.push(qs);
            let Ok(ws) = self.hip.alloc_device(wb) else {
                break;
            };
            self.allocs.push(ws);
            let rc = unsafe {
                (self.tk.talus_pf_i8x_expand)(
                    dev as *const c_void,
                    qt,
                    n as i32,
                    k as i32,
                    qs as *mut i8,
                    ws as *mut f32,
                    self.s as *mut c_void,
                )
            };
            if rc == 0 {
                self.i8c.insert(dev, (qs as usize, ws as usize, m));
            }
        }
        let _ = unsafe { (self.hip.hipStreamSynchronize)(self.s) };
        eprintln!(
            "laguna: int8 prefill cache: {} matrices, {:.2} GiB (TALUS_LAGUNA_I8=0 disables)",
            self.i8c.len(),
            total as f64 / (1u64 << 30) as f64
        );
        self._i8_res = res;
    }

    pub fn ensure_decode_graph(&mut self, sp: bool) -> Result<()> {
        if !self.use_graph {
            return Ok(());
        }
        let ex_ptr = if sp { self.exec_sp } else { self.exec };
        if !ex_ptr.is_null() {
            return Ok(());
        }
        check_hip(
            unsafe { (self.hip.hipStreamSynchronize)(self.s) },
            "hipStreamSynchronize(before capture decode)",
        )?;
        check_hip(
            unsafe {
                (self.hip.hipStreamBeginCapture)(self.s, HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            },
            "hipStreamBeginCapture(decode)",
        )?;
        let step_res = self.step(sp);
        let mut g: HipGraph = std::ptr::null_mut();
        let end_err = unsafe { (self.hip.hipStreamEndCapture)(self.s, &mut g) };
        step_res?;
        check_hip(end_err, "hipStreamEndCapture(decode)")?;

        let mut new_ex: HipGraphExec = std::ptr::null_mut();
        let inst_res = check_hip(
            unsafe {
                (self.hip.hipGraphInstantiate)(
                    &mut new_ex,
                    g,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            },
            "hipGraphInstantiate(decode)",
        );
        unsafe {
            let _ = (self.hip.hipGraphDestroy)(g);
        }
        inst_res?;
        if sp {
            self.exec_sp = new_ex;
        } else {
            self.exec = new_ex;
        }
        Ok(())
    }

    #[inline]
    pub fn kv_bytes(&self) -> usize {
        self.kv_bytes_total
    }

    #[inline]
    pub fn max_ctx(&self) -> usize {
        self.ctx
    }

    #[inline]
    pub fn ubatch(&self) -> usize {
        self.U
    }

    #[inline]
    pub fn is_kv_q8(&self) -> bool {
        self.kv_q8
    }

    /// Zero the full-attention layers' K / V cells of positions `from ..` up to the highest position written so far,
    /// so every request sees cells past its context as a fresh engine does (zeros). The prompt-attention kernels read
    /// whole 16-key V tiles past the context end, and those cells would change the prefill's result when an earlier
    /// request had left values there. A fresh engine is unchanged.
    /// With `swa` the sliding-window layers' rings get the same treatment: every ring slot except the positions
    /// `swa_keep_from(from) .. from` (the window a query at `from` can reach, tiles included) is zeroed, so the cells
    /// a request reads past its context never hold an earlier request's keys / values.
    pub fn clear_full_kv_from(&mut self, from: usize) -> Result<()> {
        self.clear_kv_from(from, false)
    }

    /// The first position a sliding-window layer keeps when a request resumes at `from` (16-aligned; 0 at `from` 0).
    pub fn swa_keep_from(from: usize) -> usize {
        if from == 0 {
            0
        } else {
            from.saturating_sub(K_SWA + 64) & !15
        }
    }

    /// See `clear_full_kv_from`; `swa` also clears the sliding-window rings (skipped while nothing was written since
    /// the last clear to position 0, e.g. on a fresh engine).
    pub fn clear_kv_from(&mut self, from: usize, swa: bool) -> Result<()> {
        if !self.kv_q8 {
            self.kv_hw = self.kv_hw.min(from);
            return Ok(());
        }
        let hw = self.kv_hw.min(self.ctx16);
        if from < hw {
            let hw16 = (hw + 15) & !15;
            for il in 0..self.m.layers.len() {
                if is_full(il) {
                    let hwc = hw16.min(self.vtl[il]);
                    // the partial tile ends at the tile boundary: the whole-tile clear from an unaligned `up` (when the
                    // high-water mark sat inside from's tile) wiped the tile's earlier positions, i.e. the reused
                    // prefix's last cells, on most dims (TALUS_LAGUNA_CLEAR_TILE=0: that old bound, for A/B only)
                    let up = if self.clear_tile_fix {
                        ((from + 15) & !15).min(hwc)
                    } else {
                        ((from + 15) & !15).min(hw.max(from))
                    };
                    self.zero_cells(il, from, up, hwc)?;
                }
            }
        }
        if swa && (self.swa_dirty || from > 0) {
            let keep = Self::swa_keep_from(from);
            for il in 0..self.m.layers.len() {
                let r = self.vtl[il];
                if is_full(il) || r >= self.ctx16 {
                    continue;
                }
                // positions from .. keep + r, as ring slots (r and keep are multiples of 16)
                let (a, e) = (from % r, keep % r);
                let up = ((a + 15) & !15).min(r);
                if from == 0 {
                    self.zero_cells(il, 0, 0, r)?;
                } else if a < e {
                    self.zero_cells(il, a, up.min(e), e)?;
                } else {
                    self.zero_cells(il, a, up, r)?;
                    if e > 0 {
                        self.zero_cells(il, 0, 0, e)?;
                    }
                }
            }
            self.swa_dirty = false;
        }
        self.kv_hw = self.kv_hw.min(from);
        check_hip(
            unsafe { (self.hip.hipStreamSynchronize)(self.s) },
            "sync(clear_kv)",
        )
    }

    /// Zero layer `il`'s K / V cells (and scales) of slots `a .. b`: `a .. up` is the partial V tile (`up` = `a`
    /// rounded up to 16, or `b`), `up .. b` whole tiles (`b` a multiple of 16 or the stride).
    fn zero_cells(&self, il: usize, a: usize, up: usize, b: usize) -> Result<()> {
        use crate::laguna::weights::{ks_index as ksi, vs_index as vsi, vt_index as vti};
        if a >= b {
            return Ok(());
        }
        let st = |e: i32, what: &'static str| check_hip(e, what);
        let stride = self.vtl[il];
        unsafe {
            let k = self.kpool[il] as *mut u8;
            st(
                (self.hip.hipMemsetAsync)(
                    k.add(a * self.cell) as *mut c_void,
                    0,
                    (b - a) * self.cell,
                    self.s,
                ),
                "clear k",
            )?;
            let ks = self.kscale[il] as *mut u16;
            let (x, y) = (ksi(a, 0, 0, K_HEAD_KV), ksi(b, 0, 0, K_HEAD_KV));
            st(
                (self.hip.hipMemsetAsync)(ks.add(x) as *mut c_void, 0, (y - x) * 2, self.s),
                "clear ks",
            )?;
            let v = self.vpool[il] as *mut u8;
            let vs = self.vscale[il] as *mut u16;
            if a == 0 && b == stride {
                // the whole pool: one span per buffer
                st(
                    (self.hip.hipMemsetAsync)(
                        v as *mut c_void,
                        0,
                        vti(K_HEAD_KV, 0, 0, stride),
                        self.s,
                    ),
                    "clear v",
                )?;
                st(
                    (self.hip.hipMemsetAsync)(
                        vs as *mut c_void,
                        0,
                        vsi(K_HEAD_KV, 0, 0, stride) * 2,
                        self.s,
                    ),
                    "clear vs",
                )?;
                return Ok(());
            }
            for kvh in 0..K_HEAD_KV {
                if a < up {
                    // the partial tile: positions a .. up of every dim (rows of 16 cells)
                    let x = vti(kvh, a, 0, stride);
                    st(
                        (self.hip.hipMemset2DAsync)(
                            v.add(x) as *mut c_void,
                            16,
                            0,
                            up - a,
                            K_HEAD_DIM,
                            self.s,
                        ),
                        "clear v tile",
                    )?;
                    let x = vsi(kvh, a, 0, stride);
                    st(
                        (self.hip.hipMemset2DAsync)(
                            vs.add(x) as *mut c_void,
                            32,
                            0,
                            (up - a) * 2,
                            4,
                            self.s,
                        ),
                        "clear vs tile",
                    )?;
                }
                if up < b {
                    let (x, y) = (vti(kvh, up, 0, stride), vti(kvh, b, 0, stride));
                    st(
                        (self.hip.hipMemsetAsync)(v.add(x) as *mut c_void, 0, y - x, self.s),
                        "clear v",
                    )?;
                    let (x, y) = (vsi(kvh, up, 0, stride), vsi(kvh, b, 0, stride));
                    st(
                        (self.hip.hipMemsetAsync)(vs.add(x) as *mut c_void, 0, (y - x) * 2, self.s),
                        "clear vs",
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Decode through verify(1) (`TALUS_LAGUNA_DEC_VERIFY` set wins over `on`): serve turns it on with the
    /// DFlash drafter / long prompt-lookup drafts, so their greedy output does not depend on the drafts.
    pub fn set_dec_verify(&mut self, on: bool) {
        self.dec_verify = match std::env::var("TALUS_LAGUNA_DEC_VERIFY") {
            Ok(v) => v != "0",
            Err(_) => on,
        };
    }

    #[inline]
    pub fn dec_verify(&self) -> bool {
        self.dec_verify
    }

    /// Verify windows take per-token sparse bounds (rows stay verify(1) rows past sparse_min).
    #[inline]
    pub fn sparse_exact(&self) -> bool {
        self.sparse_exact
    }

    #[inline]
    pub fn is_sparse(&self) -> bool {
        self.sparse
    }

    #[inline]
    pub fn sparse_min_pos(&self) -> i32 {
        self.sp_min
    }

    #[inline]
    pub fn ctx16(&self) -> usize {
        self.ctx16
    }

    #[inline]
    pub fn ring(&self) -> usize {
        self.ring
    }

    /// Populate per-block K min/max bounds for `[bnd_valid, upto)` on all full-attention layers,
    /// matching `laguna_engine.cpp:178-184`.
    fn bounds_catch_up(&mut self, upto: i32) -> Result<()> {
        if !self.sparse || self.bnd_valid >= upto {
            return Ok(());
        }
        let nl = self.m.layers.len();
        let stream = self.s as tk_stream;
        let ctx_ptr = self.tk_ctx.as_ptr();
        for il in 0..nl {
            if is_full(il) {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_kbound_update)(
                            ctx_ptr,
                            self.kpool[il] as *const c_void,
                            self.kscale[il],
                            self.kbnd[il],
                            self.nb_max,
                            K_HEAD_KV as i32,
                            self.bnd_valid / self.sp_bs,
                            (upto - 1) / self.sp_bs,
                            upto - 1,
                            std::ptr::null(),
                            0,
                            self.sp_bs,
                            self.sp_f16,
                            stream,
                        )
                    },
                    "tk_kbound_update(bounds_catch_up)",
                )?;
            }
        }
        self.bnd_valid = upto;
        Ok(())
    }

    /// Dense projection (`W` `[K, N]` x `in_ptr` `[nb, K]` -> `out_ptr` `[nb, N]`) using
    /// single-expert MMQ when supported, else 8-column `tk_quantize_q8_1` + `tk_mmvq`,
    /// matching `laguna_engine.cpp:251-278`.
    #[inline]
    fn dense(&self, W: &DevTensor, in_ptr: *const f32, out_ptr: *mut f32, nb: usize) -> Result<()> {
        let mut cached_layout = None;
        self.dense_ex(W, in_ptr, out_ptr, nb, &mut cached_layout)
    }

    fn dense_ex(
        &self,
        W: &DevTensor,
        in_ptr: *const f32,
        out_ptr: *mut f32,
        nb: usize,
        cached_layout: &mut Option<i32>,
    ) -> Result<()> {
        let K = W.ne[0] as usize;
        let N = W.ne[1] as usize;
        let qtype = to_tk_quant_type(W.qtype);
        let ctx_ptr = self.tk_ctx.as_ptr();
        let stream = self.s as tk_stream;

        let is_kquant = matches!(
            W.qtype,
            talus_core_api::QuantType::Q4_K
                | talus_core_api::QuantType::Q5_K
                | talus_core_api::QuantType::Q5_1
                | talus_core_api::QuantType::Q6_K
                | talus_core_api::QuantType::Q2_K
                | talus_core_api::QuantType::Q3_K
        );
        let mmq_ok = (!is_kquant || self.mmq_kq)
            && unsafe { (self.tk.tk_q36_mmq_supported_fits)(qtype as i32, N as i64) };
        if mmq_ok {
            let layout = unsafe { (self.tk.tk_q36_mmq_q8_layout)(qtype as i32) };
            if *cached_layout != Some(layout) {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_q36_mmq_quantize)(
                            ctx_ptr,
                            in_ptr,
                            std::ptr::null(),
                            self.xq8,
                            qtype as i32,
                            K as i64,
                            K as i64,
                            nb as i64,
                            stream,
                        )
                    },
                    "tk_q36_mmq_quantize(dense)",
                )?;
                *cached_layout = Some(layout);
            }
            // prefill products on the int8 cache (same output bits as tk_prefill_moe_mmq)
            if nb >= 128 {
                if let Some(&(qs, ws, mode)) = self.i8c.get(&(W.dev as usize)) {
                    let rc = unsafe {
                        (self.tk.talus_pf_i8gemm)(
                            qs as *const i8,
                            ws as *const f32,
                            N as i32,
                            K as i32,
                            mode,
                            self.xq8 as *const c_void,
                            nb as i32,
                            out_ptr,
                            N as i64,
                            stream as *mut c_void,
                        )
                    };
                    if rc == 0 {
                        return Ok(());
                    }
                }
            }
            if self.one_b_rows.load(std::sync::atomic::Ordering::Relaxed) != nb as i32 {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_q36_mmq_set_bounds)(ctx_ptr, self.one_b, nb as i32, stream)
                    },
                    "tk_q36_mmq_set_bounds(dense)",
                )?;
                self.one_b_rows
                    .store(nb as i32, std::sync::atomic::Ordering::Relaxed);
            }
            let expert_bytes =
                unsafe { (self.tk.tk_q36_mmq_matrix_bytes)(qtype as i32, N as i64, K as i64) };
            let P = tk_moe_mmq_product {
                w: W.dev as *const c_void,
                qtype,
                w_rows: N as i64,
                w_cols: K as i64,
                expert_bytes,
                n_experts: 1,
                x_q8_1: self.xq8 as *const c_void,
                bounds_dev: self.one_b as *const i32,
                ids_dev: self.ident as *const i32,
                total_rows: nb as i64,
                max_rows: nb as i64,
                dst: out_ptr,
                ld_dst: N as i64,
                y_scale_dev: std::ptr::null(),
            };
            check_tk(
                ctx_ptr,
                unsafe { (self.tk.tk_prefill_moe_mmq)(ctx_ptr, &P, stream) },
                "tk_prefill_moe_mmq(dense)",
            )?;
            return Ok(());
        }

        let mut c0 = 0usize;
        while c0 < nb {
            let nc = (nb - c0).min(8);
            let in_col = unsafe { in_ptr.add(c0 * K) };
            let out_col = unsafe { out_ptr.add(c0 * N) };
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_quantize_q8_1)(
                        ctx_ptr, in_col, self.q8c, K as i32, nc as i32, stream,
                    )
                },
                "tk_quantize_q8_1(dense)",
            )?;
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_mmvq)(
                        ctx_ptr,
                        qtype,
                        W.dev as *const c_void,
                        self.q8c as *const c_void,
                        out_col,
                        K as i32,
                        N as i32,
                        nc as i32,
                        stream,
                    )
                },
                "tk_mmvq(dense)",
            )?;
            c0 += 8;
        }
        Ok(())
    }

    /// Grouped 256-expert MMQ projection (`W` `[K, N, 256]` against `xq8g` sorted by expert),
    /// matching `laguna_engine.cpp:280-296`.
    fn grouped(
        &self,
        W: &DevTensor,
        out_ptr: *mut f32,
        total_rows: usize,
        max_rows: usize,
    ) -> Result<()> {
        let K = W.ne[0];
        let N = W.ne[1];
        let P = tk_moe_mmq_product {
            w: W.dev as *const c_void,
            qtype: to_tk_quant_type(W.qtype),
            w_rows: N,
            w_cols: K,
            expert_bytes: (N as usize) * W.row_bytes,
            n_experts: K_EXPERTS as i32,
            x_q8_1: self.xq8g as *const c_void,
            bounds_dev: self.bnds as *const i32,
            ids_dev: self.ident as *const i32,
            total_rows: total_rows as i64,
            max_rows: max_rows as i64,
            dst: out_ptr,
            ld_dst: N,
            y_scale_dev: std::ptr::null(),
        };
        check_tk(
            self.tk_ctx.as_ptr(),
            unsafe { (self.tk.tk_prefill_moe_mmq)(self.tk_ctx.as_ptr(), &P, self.s as tk_stream) },
            "tk_prefill_moe_mmq(grouped)",
        )
    }

    /// Run a single prefill micro-batch of `nb <= U` tokens starting at `p0`,
    /// matching `laguna_engine.cpp:323-411`.
    fn prefill_ubatch(
        &mut self,
        tokens: &[i32],
        tok_off: usize,
        nb: usize,
        p0: usize,
        mut dump: Option<DumpFn<'_>>,
    ) -> Result<()> {
        assert!(nb >= 1 && nb <= self.U && p0 + nb <= self.ctx);

        let stream = self.s as tk_stream;
        let ctx_ptr = self.tk_ctx.as_ptr();
        if self.m.tok_embd.qtype == talus_core_api::QuantType::Q8_0 && !self.m.emb.dev.is_null() {
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_embed_q8_0)(
                        ctx_ptr,
                        self.m.emb.dev as *const c_void,
                        self.ptok_dev.add(tok_off),
                        self.x,
                        K_EMBD as i32,
                        nb as i32,
                        stream,
                    )
                },
                "tk_embed_q8_0(prefill)",
            )?;
        } else {
            let emb_bytes = self.m.tok_embd_host_bytes()?;
            let emb_qtype = self.m.tok_embd.qtype;
            for (t, &tok_id) in tokens.iter().enumerate().take(nb) {
                dequant_embd_row(
                    emb_qtype,
                    emb_bytes,
                    tok_id as usize,
                    &mut self.hx[t * K_EMBD..(t + 1) * K_EMBD],
                )?;
            }
            check_hip(
                unsafe {
                    (self.hip.hipMemcpyAsync)(
                        self.x as *mut c_void,
                        self.hx.as_ptr() as *const c_void,
                        nb * K_EMBD * 4,
                        HipRuntime::MEMCPY_H2D,
                        self.s,
                    )
                },
                "hipMemcpyAsync(hx->x)",
            )?;
            check_hip(
                unsafe { (self.hip.hipStreamSynchronize)(self.s) },
                "hipStreamSynchronize(hx->x)",
            )?;
        }

        let dump_dev = |hip: &HipRuntime,
                        s: hipStream_t,
                        name: &str,
                        d: *const f32,
                        r: usize,
                        c: usize,
                        cb: &mut dyn FnMut(&str, &[f32], usize, usize)|
         -> Result<()> {
            let mut h = vec![0.0_f32; r * c];
            check_hip(
                unsafe { (hip.hipStreamSynchronize)(s) },
                "hipStreamSynchronize(dump)",
            )?;
            hip.copy_d2h(h.as_mut_ptr() as *mut c_void, d as *const c_void, r * c * 4)?;
            cb(name, &h, r, c);
            Ok(())
        };

        if let Some(ref mut cb) = dump {
            dump_dev(&self.hip, self.s, "inp_embd", self.x, nb, K_EMBD, *cb)?;
        }

        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_fill_pos)(
                    ctx_ptr,
                    self.pos_k,
                    nb as i32,
                    K_HEAD_KV as i32,
                    p0 as i32,
                    stream,
                )
            },
            "tk_fill_pos(K)",
        )?;
        let mut pos_q_heads = -1i32;

        let m = Arc::clone(&self.m);
        let nl = m.layers.len();
        for il in 0..nl {
            self.tap(il, self.x, nb)?;
            let L = &m.layers[il];
            let H = n_head(il);
            let full = is_full(il);
            let n_rot = if full { 64 } else { 128 };
            let cos = if full { self.cos_full } else { self.cos_swa };
            let sin = if full { self.sin_full } else { self.sin_swa };

            if pos_q_heads != H as i32 {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_fill_pos)(
                            ctx_ptr, self.pos_q, nb as i32, H as i32, p0 as i32, stream,
                        )
                    },
                    "tk_fill_pos(Q)",
                )?;
                pos_q_heads = H as i32;
            }

            // ---- Attention ----
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_rms_norm_to)(
                        ctx_ptr,
                        self.h,
                        self.x,
                        L.attn_norm.as_f32(),
                        nb as i32,
                        K_EMBD as i32,
                        K_EPS,
                        stream,
                    )
                },
                "tk_rms_norm_to(attn_norm)",
            )?;
            let mut h_attn_q8_layout = None;
            self.dense_ex(&L.wq, self.h, self.q, nb, &mut h_attn_q8_layout)?;
            self.dense_ex(&L.wk, self.h, self.kc, nb, &mut h_attn_q8_layout)?;
            self.dense_ex(&L.wv, self.h, self.vc, nb, &mut h_attn_q8_layout)?;
            self.dense_ex(&L.attn_gate, self.h, self.g, nb, &mut h_attn_q8_layout)?;

            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_head_norm_rope)(
                        ctx_ptr,
                        self.q,
                        L.q_norm.as_f32(),
                        (nb * H) as i32,
                        K_EPS,
                        n_rot,
                        cos,
                        sin,
                        self.pos_q,
                        1,
                        stream,
                    )
                },
                "tk_head_norm_rope(Q)",
            )?;
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_head_norm_rope)(
                        ctx_ptr,
                        self.kc,
                        L.k_norm.as_f32(),
                        (nb * K_HEAD_KV) as i32,
                        K_EPS,
                        n_rot,
                        cos,
                        sin,
                        self.pos_k,
                        1,
                        stream,
                    )
                },
                "tk_head_norm_rope(K)",
            )?;

            if self.kv_q8 {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_kv_store_q8)(
                            ctx_ptr,
                            self.kc,
                            self.vc,
                            self.kpool[il] as *mut c_void,
                            self.vpool[il] as *mut c_void,
                            self.kscale[il],
                            self.vscale[il],
                            nb as i32,
                            K_HEAD_KV as i32,
                            p0 as i32,
                            self.vtl[il] as i32,
                            stream,
                        )
                    },
                    "tk_kv_store_q8",
                )?;
                if self.sparse && full {
                    let b_first = self.bnd_valid.min(p0 as i32) / self.sp_bs;
                    check_tk(
                        ctx_ptr,
                        unsafe {
                            (self.tk.tk_kbound_update)(
                                ctx_ptr,
                                self.kpool[il] as *const c_void,
                                self.kscale[il],
                                self.kbnd[il],
                                self.nb_max,
                                K_HEAD_KV as i32,
                                b_first,
                                ((p0 + nb - 1) as i32) / self.sp_bs,
                                (p0 + nb - 1) as i32,
                                std::ptr::null(),
                                0,
                                self.sp_bs,
                                self.sp_f16,
                                stream,
                            )
                        },
                        "tk_kbound_update(prefill)",
                    )?;
                }
            } else {
                let kp_dst = unsafe { self.kpool[il].add(p0 * self.cell) as *mut c_void };
                check_tk(
                    ctx_ptr,
                    unsafe {
                        tk_f32_to_f16_bulk(
                            &self.tk,
                            ctx_ptr,
                            self.kc,
                            kp_dst,
                            (nb * self.cell) as i64,
                            stream,
                        )
                    },
                    "tk_f32_to_f16_bulk(kc)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_v_store_t)(
                            ctx_ptr,
                            self.vc,
                            self.vpool[il],
                            nb as i32,
                            K_HEAD_KV as i32,
                            p0 as i32,
                            self.vtl[il] as i32,
                            stream,
                        )
                    },
                    "tk_v_store_t",
                )?;
            }

            if self.use_wmma {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_laguna_prompt_attn_wmma)(
                            ctx_ptr,
                            self.q,
                            self.kpool[il],
                            self.vpool[il],
                            self.att,
                            nb as i32,
                            p0 as i32,
                            H as i32,
                            K_HEAD_KV as i32,
                            if full { 0 } else { K_SWA as i32 },
                            self.vtl[il] as i32,
                            std::ptr::null(),
                            1,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            if self.kv_q8 {
                                self.kscale[il]
                            } else {
                                std::ptr::null()
                            },
                            if self.kv_q8 {
                                self.vscale[il]
                            } else {
                                std::ptr::null()
                            },
                            stream,
                        )
                    },
                    "tk_laguna_prompt_attn_wmma",
                )?;
            } else {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_laguna_prompt_attn)(
                            ctx_ptr,
                            self.q,
                            self.kpool[il],
                            self.vpool[il],
                            self.att,
                            nb as i32,
                            p0 as i32,
                            H as i32,
                            K_HEAD_KV as i32,
                            if full { 0 } else { K_SWA as i32 },
                            std::ptr::null(),
                            1,
                            std::ptr::null_mut(),
                            std::ptr::null_mut(),
                            true,
                            self.vtl[il] as i32,
                            if self.kv_q8 {
                                self.kscale[il]
                            } else {
                                std::ptr::null()
                            },
                            if self.kv_q8 {
                                self.vscale[il]
                            } else {
                                std::ptr::null()
                            },
                            stream,
                        )
                    },
                    "tk_laguna_prompt_attn",
                )?;
            }

            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_softplus_head_gate)(
                        ctx_ptr,
                        self.att,
                        self.g,
                        (nb * H) as i32,
                        K_HEAD_DIM as i32,
                        stream,
                    )
                },
                "tk_softplus_head_gate",
            )?;
            self.dense(&L.wo, self.att, self.a, nb)?;
            check_tk(
                ctx_ptr,
                unsafe {
                    tk_add_inplace(
                        &self.tk,
                        ctx_ptr,
                        self.x,
                        self.a,
                        (nb * K_EMBD) as i64,
                        stream,
                    )
                },
                "tk_add_inplace(attn)",
            )?;

            if let Some(ref mut cb) = dump {
                dump_dev(
                    &self.hip,
                    self.s,
                    &format!("attn_out-{il}"),
                    self.x,
                    nb,
                    K_EMBD,
                    *cb,
                )?;
            }

            // ---- FFN (layer 0 dense, layers 1..47 MoE + shared expert) ----
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_rms_norm_to)(
                        ctx_ptr,
                        self.h,
                        self.x,
                        L.ffn_norm.as_f32(),
                        nb as i32,
                        K_EMBD as i32,
                        K_EPS,
                        stream,
                    )
                },
                "tk_rms_norm_to(ffn_norm)",
            )?;

            if il == 0 {
                let mut h_ffn_q8_layout = None;
                self.dense_ex(&L.d_gate, self.h, self.f1, nb, &mut h_ffn_q8_layout)?;
                self.dense_ex(&L.d_up, self.h, self.f2, nb, &mut h_ffn_q8_layout)?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_silu_mul)(
                            ctx_ptr,
                            self.f1,
                            self.f2,
                            (nb * K_FF_DENSE) as i64,
                            stream,
                        )
                    },
                    "tk_silu_mul(layer0)",
                )?;
                self.dense(&L.d_down, self.f1, self.a, nb)?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        tk_add_inplace(
                            &self.tk,
                            ctx_ptr,
                            self.x,
                            self.a,
                            (nb * K_EMBD) as i64,
                            stream,
                        )
                    },
                    "tk_add_inplace(layer0 ffn)",
                )?;
            } else {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_f32_matmul)(
                            ctx_ptr,
                            L.router_inp.as_f32(),
                            self.h,
                            self.rl,
                            K_EMBD as i32,
                            K_EXPERTS as i32,
                            nb as i32,
                            stream,
                        )
                    },
                    "tk_f32_matmul(router)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_route_group)(
                            ctx_ptr,
                            self.rl,
                            L.router_bias.as_f32(),
                            nb as i32,
                            K_TOP_K as i32,
                            K_EXP_SCALE as f32,
                            self.sel_e,
                            self.sel_w,
                            self.cnt,
                            self.bnds,
                            self.gtok,
                            self.smap,
                            stream,
                        )
                    },
                    "tk_route_group",
                )?;

                // Shared expert
                let mut h_sh_q8_layout = None;
                self.dense_ex(&L.sh_gate, self.h, self.f1, nb, &mut h_sh_q8_layout)?;
                self.dense_ex(&L.sh_up, self.h, self.f2, nb, &mut h_sh_q8_layout)?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_silu_mul)(
                            ctx_ptr,
                            self.f1,
                            self.f2,
                            (nb * K_FF_EXP) as i64,
                            stream,
                        )
                    },
                    "tk_silu_mul(shared)",
                )?;
                self.dense(&L.sh_down, self.f1, self.ysh, nb)?;

                // Routed experts: grouped by expert (`max_rows = nb`: an expert appears at most once per token)
                let S = nb * K_TOP_K;
                let max_rows = nb;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_q36_mmq_quantize)(
                            ctx_ptr,
                            self.h,
                            self.gtok,
                            self.xq8g,
                            to_tk_quant_type(L.exp_gate.qtype) as i32,
                            K_EMBD as i64,
                            K_EMBD as i64,
                            S as i64,
                            stream,
                        )
                    },
                    "tk_q36_mmq_quantize(exp_gate)",
                )?;
                self.grouped(&L.exp_gate, self.f1, S, max_rows)?;
                if L.exp_up.qtype != L.exp_gate.qtype {
                    check_tk(
                        ctx_ptr,
                        unsafe {
                            (self.tk.tk_q36_mmq_quantize)(
                                ctx_ptr,
                                self.h,
                                self.gtok,
                                self.xq8g,
                                to_tk_quant_type(L.exp_up.qtype) as i32,
                                K_EMBD as i64,
                                K_EMBD as i64,
                                S as i64,
                                stream,
                            )
                        },
                        "tk_q36_mmq_quantize(exp_up)",
                    )?;
                }
                self.grouped(&L.exp_up, self.f2, S, max_rows)?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_silu_mul)(
                            ctx_ptr,
                            self.f1,
                            self.f2,
                            (S * K_FF_EXP) as i64,
                            stream,
                        )
                    },
                    "tk_silu_mul(routed)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_q36_mmq_quantize)(
                            ctx_ptr,
                            self.f1,
                            std::ptr::null(),
                            self.xq8g,
                            to_tk_quant_type(L.exp_down.qtype) as i32,
                            K_FF_EXP as i64,
                            K_FF_EXP as i64,
                            S as i64,
                            stream,
                        )
                    },
                    "tk_q36_mmq_quantize(exp_down)",
                )?;
                self.grouped(&L.exp_down, self.yexp, S, max_rows)?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_moe_add_slots)(
                            ctx_ptr,
                            self.x,
                            self.ysh,
                            self.yexp,
                            self.smap,
                            self.sel_w,
                            nb as i32,
                            K_EMBD as i32,
                            K_TOP_K as i32,
                            stream,
                        )
                    },
                    "tk_moe_add_slots",
                )?;
            }

            if let Some(ref mut cb) = dump {
                dump_dev(
                    &self.hip,
                    self.s,
                    &format!("l_out-{il}"),
                    self.x,
                    nb,
                    K_EMBD,
                    *cb,
                )?;
            }
        }

        self.tap(nl, self.x, nb)?;
        self.kv_hw = self.kv_hw.max(p0 + nb);
        self.swa_dirty = true;
        if self.kv_q8 && self.sparse {
            self.bnd_valid = (p0 + nb) as i32;
        }
        Ok(())
    }

    /// Run prefill across `tokens` starting at `pos0` in chunks of `U`, optionally copying the
    /// last token's `[K_VOCAB]` logits to `last_logits` and/or writing all rows to `logits_file`,
    /// matching `laguna_engine.cpp:582-624`.
    pub fn prefill(
        &mut self,
        tokens: &[i32],
        pos0: usize,
        last_logits: Option<&mut [f32]>,
        dump: Option<DumpFn<'_>>,
    ) -> Result<()> {
        self.prefill_with_file(tokens, pos0, last_logits, None, dump)
    }

    /// Run prefill with optional raw `.f32` logit dump file (`--dump-logits`).
    pub fn prefill_with_file(
        &mut self,
        tokens: &[i32],
        pos0: usize,
        mut last_logits: Option<&mut [f32]>,
        mut logits_file: Option<&mut File>,
        mut dump: Option<DumpFn<'_>>,
    ) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let n_tok = tokens.len();
        assert!(pos0 + n_tok <= self.ctx);
        let stream = self.s as tk_stream;
        let ctx_ptr = self.tk_ctx.as_ptr();
        if self.m.tok_embd.qtype == talus_core_api::QuantType::Q8_0 && !self.m.emb.dev.is_null() {
            check_hip(
                unsafe {
                    (self.hip.hipMemcpyAsync)(
                        self.ptok_dev as *mut c_void,
                        tokens.as_ptr() as *const c_void,
                        std::mem::size_of_val(tokens),
                        HipRuntime::MEMCPY_H2D,
                        self.s,
                    )
                },
                "hipMemcpyAsync(ptok_dev)",
            )?;
        }
        let mut p = 0usize;
        while p < n_tok {
            let nb = (n_tok - p).min(self.U);
            let is_last = p + nb == n_tok;
            let chunk_dump: Option<DumpFn<'_>> = if is_last {
                match dump.as_mut() {
                    Some(cb) => Some(*cb),
                    None => None,
                }
            } else {
                None
            };
            self.prefill_ubatch(&tokens[p..p + nb], p, nb, pos0 + p, chunk_dump)?;

            if logits_file.is_some() {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_rms_norm_to)(
                            ctx_ptr,
                            self.h,
                            self.x,
                            self.m.output_norm.as_f32(),
                            nb as i32,
                            K_EMBD as i32,
                            K_EPS,
                            stream,
                        )
                    },
                    "tk_rms_norm_to(output_norm all)",
                )?;
                self.dense(&self.m.output, self.h, self.logits_buf, nb)?;
                check_hip(
                    unsafe { (self.hip.hipStreamSynchronize)(self.s) },
                    "hipStreamSynchronize(prefill dump logits)",
                )?;
                let mut host_lv = vec![0.0_f32; nb * K_VOCAB];
                self.hip.copy_d2h(
                    host_lv.as_mut_ptr() as *mut c_void,
                    self.logits_buf as *const c_void,
                    nb * K_VOCAB * 4,
                )?;
                if let Some(lf) = logits_file.as_deref_mut() {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(host_lv.as_ptr() as *const u8, host_lv.len() * 4)
                    };
                    lf.write_all(bytes).map_err(|e| CoreError::IoError {
                        message: format!("write logits file: {e}"),
                    })?;
                }
                if is_last {
                    if let Some(dst) = last_logits.as_deref_mut() {
                        assert!(dst.len() >= K_VOCAB);
                        dst[..K_VOCAB].copy_from_slice(&host_lv[(nb - 1) * K_VOCAB..nb * K_VOCAB]);
                    }
                }
            } else if is_last {
                if let Some(dst) = last_logits.as_deref_mut() {
                    assert!(dst.len() >= K_VOCAB);
                    let last_x = unsafe { self.x.add((nb - 1) * K_EMBD) };
                    check_tk(
                        ctx_ptr,
                        unsafe {
                            (self.tk.tk_rms_norm_to)(
                                ctx_ptr,
                                self.h,
                                last_x,
                                self.m.output_norm.as_f32(),
                                1,
                                K_EMBD as i32,
                                K_EPS,
                                stream,
                            )
                        },
                        "tk_rms_norm_to(output_norm last)",
                    )?;
                    self.dense(&self.m.output, self.h, self.logits_buf, 1)?;
                    check_hip(
                        unsafe { (self.hip.hipStreamSynchronize)(self.s) },
                        "hipStreamSynchronize(prefill last_logits)",
                    )?;
                    self.hip.copy_d2h(
                        dst.as_mut_ptr() as *mut c_void,
                        self.logits_buf as *const c_void,
                        K_VOCAB * 4,
                    )?;
                } else {
                    check_hip(
                        unsafe { (self.hip.hipStreamSynchronize)(self.s) },
                        "hipStreamSynchronize(prefill)",
                    )?;
                }
            }
            p += nb;
        }
        Ok(())
    }

    /// Copy `n` residual rows `x` into the DFlash tap slot of layer `il` (no-op unless taps are on and `il`
    /// is one of `TAP_LAYERS`). Device-to-device on `self.s`, so it records into the decode / verify graphs.
    #[inline]
    fn tap(&self, il: usize, x: *const f32, n: usize) -> Result<()> {
        if self.taps.is_null() {
            return Ok(());
        }
        let Some(k) = TAP_LAYERS.iter().position(|&l| l == il) else {
            return Ok(());
        };
        debug_assert!(n <= self.tap_rows);
        // Use a device gather kernel rather than hipMemcpy nodes inside captured decode / verify graphs.
        check_tk(
            self.tk_ctx.as_ptr(),
            unsafe {
                (self.tk.tk_gather_rows)(
                    self.tk_ctx.as_ptr(),
                    x as *const c_void,
                    self.taps.add(k * self.tap_rows * K_EMBD) as *mut c_void,
                    self.tap_idx,
                    n as i32,
                    K_EMBD * 4,
                    self.s as tk_stream,
                )
            },
            "tk_gather_rows(tap)",
        )
    }

    /// The DFlash tap buffer `[TAP_LAYERS.len()][tap_rows][K_EMBD]` (null when `Options::taps` is off): rows of
    /// the last prefill chunk / decode step / verify.
    #[inline]
    pub fn taps_ptr(&self) -> *const f32 {
        self.taps
    }

    #[inline]
    pub fn tap_rows(&self) -> usize {
        self.tap_rows
    }

    /// Run `n` columns through `tk_mmvq` (decode's kernel for wo / layer-0 down / the head), 8 columns per call,
    /// so a verify row takes decode's arithmetic.
    fn mmvq_cols_dec(&self, t: &DevTensor, xq: *const c_void, y: *mut f32, n: usize) -> Result<()> {
        let ctx_ptr = self.tk_ctx.as_ptr();
        let xs = unsafe { (self.tk.tk_q8_1_bytes)(t.n_in as i32, 1) };
        let mut c0 = 0usize;
        while c0 < n {
            let c = (n - c0).min(8);
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_mmvq)(
                        ctx_ptr,
                        to_tk_quant_type(t.qtype),
                        t.dev as *const c_void,
                        (xq as *const u8).add(c0 * xs) as *const c_void,
                        y.add(c0 * t.n_out as usize),
                        t.n_in as i32,
                        t.n_out as i32,
                        c as i32,
                        self.s as tk_stream,
                    )
                },
                "tk_mmvq(verify, decode kernels)",
            )?;
            c0 += c;
        }
        Ok(())
    }

    /// Dispatch a `step_n` batch of mat-vec items. With `dense_mmvq`, the dense items (no expert stride) go through
    /// `tk_mmvq` (column chunks; wave32 kernels when `STRATA_MMVQ_W32` is on: bitwise across column counts) and only
    /// the routed-expert items go through `tk_mmvq_batch`.
    fn run_items(&self, items: &[tk_mmvq_item], what: &'static str) -> Result<()> {
        let ctx_ptr = self.tk_ctx.as_ptr();
        if !self.dense_mmvq {
            return check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_mmvq_batch)(
                        ctx_ptr,
                        items.as_ptr(),
                        items.len() as i32,
                        self.s as tk_stream,
                    )
                },
                what,
            );
        }
        let mut rest: Vec<tk_mmvq_item> = Vec::with_capacity(items.len());
        for it in items {
            if it.expert_bytes != 0 {
                rest.push(*it);
                continue;
            }
            let n = it.n_ids as usize;
            let xs = unsafe { (self.tk.tk_q8_1_bytes)(it.n_in, 1) };
            let mut c0 = 0usize;
            while c0 < n {
                let c = (n - c0).min(self.dense_cols);
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_mmvq)(
                            ctx_ptr,
                            it.qtype,
                            it.w,
                            (it.x_q8_1 as *const u8).add(c0 * xs) as *const c_void,
                            it.y.add(c0 * it.n_out as usize),
                            it.n_in,
                            it.n_out,
                            c as i32,
                            self.s as tk_stream,
                        )
                    },
                    what,
                )?;
                c0 += c;
            }
        }
        if rest.is_empty() {
            return Ok(());
        }
        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_mmvq_batch)(
                    ctx_ptr,
                    rest.as_ptr(),
                    rest.len() as i32,
                    self.s as tk_stream,
                )
            },
            what,
        )
    }

    /// Run layer 0's gate + up over `n` columns through `tk_mmvq_ids` (decode's kernel), 8 columns per call.
    fn l0_gate_up_dec(&self, n: usize) -> Result<()> {
        let ctx_ptr = self.tk_ctx.as_ptr();
        let L = &self.m.layers[0];
        let xs = unsafe { (self.tk.tk_q8_1_bytes)(L.d_gate.n_in as i32, 1) };
        let mut c0 = 0usize;
        while c0 < n {
            let c = (n - c0).min(8);
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_mmvq_ids)(
                        ctx_ptr,
                        to_tk_quant_type(L.d_gate.qtype),
                        L.d_gate.dev as *const c_void,
                        0,
                        self.vzero as *const i32,
                        c as i32,
                        (self.vq8 as *const u8).add(c0 * xs) as *const c_void,
                        c > 1,
                        self.vf1.add(c0 * K_FF_DENSE),
                        L.d_gate.n_in as i32,
                        L.d_gate.n_out as i32,
                        L.d_up.dev as *const c_void,
                        self.vf2.add(c0 * K_FF_DENSE),
                        self.s as tk_stream,
                    )
                },
                "tk_mmvq_ids(verify layer0 gate+up, decode kernels)",
            )?;
            c0 += c;
        }
        Ok(())
    }

    #[inline]
    fn mmvq_item(
        &self,
        t: &DevTensor,
        xq: *const c_void,
        y: *mut f32,
        ids: *const i32,
        n_ids: i32,
        ebytes: usize,
        x_per: bool,
    ) -> tk_mmvq_item {
        let expert_major = if self.emaj && !ids.is_null() && ebytes > 0 && n_ids <= 10 {
            1
        } else {
            0
        };
        tk_mmvq_item {
            qtype: to_tk_quant_type(t.qtype),
            w: t.dev as *const c_void,
            expert_bytes: ebytes,
            ids_dev: ids,
            n_ids,
            x_q8_1: xq,
            x_per_id: i32::from(x_per),
            x_group: 1,
            y,
            expert_major,
            n_in: t.n_in as i32,
            n_out: t.n_out as i32,
        }
    }

    /// Emit the single-token decode kernel sequence onto `self.s` (either eagerly or inside
    /// HIP stream capture), matching `laguna_engine.cpp:414-490`.
    fn step(&self, sp: bool) -> Result<()> {
        let stream = self.s as tk_stream;
        let ctx_ptr = self.tk_ctx.as_ptr();
        let nl = self.m.layers.len();

        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_embed_q8_0)(
                    ctx_ptr,
                    self.m.emb.dev as *const c_void,
                    self.tokpos_dev,
                    self.dx,
                    K_EMBD as i32,
                    1,
                    stream,
                )
            },
            "tk_embed_q8_0(step)",
        )?;

        if nl > 0 {
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_add_norm_quant)(
                        ctx_ptr,
                        self.dx,
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        self.m.layers[0].attn_norm.as_f32(),
                        std::ptr::null_mut(),
                        self.dq8,
                        K_EMBD as i32,
                        K_EPS,
                        1,
                        stream,
                    )
                },
                "tk_add_norm_quant(layer0 attn)",
            )?;
        }

        for il in 0..nl {
            self.tap(il, self.dx, 1)?;
            let L = &self.m.layers[il];
            let H = n_head(il);
            let full = is_full(il);
            let n_rot = if full { 64 } else { 128 };
            let cos = if full { self.cos_full } else { self.cos_swa };
            let sin = if full { self.sin_full } else { self.sin_swa };
            let next_norm = if il + 1 < nl {
                self.m.layers[il + 1].attn_norm.as_f32()
            } else {
                self.m.output_norm.as_f32()
            };

            let qkv_items = [
                self.mmvq_item(&L.wq, self.dq8, self.dq, std::ptr::null(), 1, 0, false),
                self.mmvq_item(&L.wk, self.dq8, self.dkc, std::ptr::null(), 1, 0, false),
                self.mmvq_item(&L.wv, self.dq8, self.dvc, std::ptr::null(), 1, 0, false),
                self.mmvq_item(
                    &L.attn_gate,
                    self.dq8,
                    self.dg,
                    std::ptr::null(),
                    1,
                    0,
                    false,
                ),
            ];
            check_tk(
                ctx_ptr,
                unsafe { (self.tk.tk_mmvq_batch)(ctx_ptr, qkv_items.as_ptr(), 4, stream) },
                "tk_mmvq_batch(qkv+gate)",
            )?;

            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_qkv_norm_rope_store)(
                        ctx_ptr,
                        self.dq,
                        self.dkc,
                        self.dvc,
                        L.q_norm.as_f32(),
                        L.k_norm.as_f32(),
                        H as i32,
                        K_HEAD_KV as i32,
                        K_EPS,
                        n_rot,
                        cos,
                        sin,
                        self.pos_dev,
                        self.kpool[il] as *mut c_void,
                        self.vpool[il] as *mut c_void,
                        self.vtl[il] as i32,
                        if self.kv_q8 {
                            self.kscale[il]
                        } else {
                            std::ptr::null_mut()
                        },
                        if self.kv_q8 {
                            self.vscale[il]
                        } else {
                            std::ptr::null_mut()
                        },
                        1,
                        stream,
                    )
                },
                "tk_qkv_norm_rope_store(step)",
            )?;

            let spl = sp && full;
            if spl {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_kbound_update)(
                            ctx_ptr,
                            self.kpool[il] as *const c_void,
                            self.kscale[il],
                            self.kbnd[il],
                            self.nb_max,
                            K_HEAD_KV as i32,
                            0,
                            0,
                            0,
                            self.pos_dev,
                            1,
                            self.sp_bs,
                            self.sp_f16,
                            stream,
                        )
                    },
                    "tk_kbound_update(step)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_sparse_scores)(
                            ctx_ptr,
                            self.dq,
                            self.kbnd[il] as *const c_void,
                            self.sp_ub,
                            self.nb_max,
                            H as i32,
                            K_HEAD_KV as i32,
                            self.pos_dev,
                            1,
                            self.sp_bs,
                            self.sp_f16,
                            stream,
                        )
                    },
                    "tk_sparse_scores(step)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_sparse_select)(
                            ctx_ptr,
                            self.sp_ub,
                            self.nb_max,
                            H as i32,
                            self.pos_dev,
                            1,
                            self.sp_k,
                            self.sp_sink,
                            self.sp_recent,
                            self.sp_sel,
                            if self.sp_per_kv { K_HEAD_KV as i32 } else { 1 },
                            self.sp_bs,
                            self.sp_scr,
                            stream,
                        )
                    },
                    "tk_sparse_select(step)",
                )?;
            }

            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_laguna_decode_attn)(
                        ctx_ptr,
                        self.dq,
                        self.kpool[il],
                        self.vpool[il],
                        if self.kv_q8 {
                            self.kscale[il]
                        } else {
                            std::ptr::null()
                        },
                        if self.kv_q8 {
                            self.vscale[il]
                        } else {
                            std::ptr::null()
                        },
                        H as i32,
                        K_HEAD_KV as i32,
                        if full { 0 } else { K_SWA as i32 },
                        self.vtl[il] as i32,
                        self.pos_dev,
                        K_SPLIT,
                        self.dpart,
                        self.dml,
                        1,
                        if spl { self.sp_sel } else { std::ptr::null() },
                        self.sp_k,
                        self.sp_per_kv,
                        self.sp_bs,
                        stream,
                    )
                },
                "tk_laguna_decode_attn(step)",
            )?;

            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_attn_merge_gate_quant)(
                        ctx_ptr, self.dpart, self.dml, self.dg, self.dq8, H as i32, K_SPLIT, 1,
                        stream,
                    )
                },
                "tk_attn_merge_gate_quant(step)",
            )?;
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_mmvq)(
                        ctx_ptr,
                        to_tk_quant_type(L.wo.qtype),
                        L.wo.dev as *const c_void,
                        self.dq8 as *const c_void,
                        self.da,
                        L.wo.n_in as i32,
                        L.wo.n_out as i32,
                        1,
                        stream,
                    )
                },
                "tk_mmvq(wo)",
            )?;
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_add_norm_quant)(
                        ctx_ptr,
                        self.dx,
                        self.da,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        L.ffn_norm.as_f32(),
                        self.dh,
                        self.dq8,
                        K_EMBD as i32,
                        K_EPS,
                        1,
                        stream,
                    )
                },
                "tk_add_norm_quant(ffn)",
            )?;

            if il == 0 {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_mmvq_ids)(
                            ctx_ptr,
                            to_tk_quant_type(L.d_gate.qtype),
                            L.d_gate.dev as *const c_void,
                            0,
                            self.vzero as *const i32,
                            1,
                            self.dq8 as *const c_void,
                            false,
                            self.df1,
                            L.d_gate.n_in as i32,
                            L.d_gate.n_out as i32,
                            L.d_up.dev as *const c_void,
                            self.df2,
                            stream,
                        )
                    },
                    "tk_mmvq_ids(layer0 gate+up)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_swiglu_quantize_q8_1)(
                            ctx_ptr,
                            self.df1,
                            self.df2,
                            self.dq8b,
                            K_FF_DENSE as i32,
                            1,
                            stream,
                        )
                    },
                    "tk_swiglu_quantize_q8_1(layer0)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_mmvq)(
                            ctx_ptr,
                            to_tk_quant_type(L.d_down.qtype),
                            L.d_down.dev as *const c_void,
                            self.dq8b as *const c_void,
                            self.da,
                            L.d_down.n_in as i32,
                            L.d_down.n_out as i32,
                            1,
                            stream,
                        )
                    },
                    "tk_mmvq(layer0 down)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_add_norm_quant)(
                            ctx_ptr,
                            self.dx,
                            self.da,
                            std::ptr::null(),
                            std::ptr::null(),
                            0,
                            next_norm,
                            std::ptr::null_mut(),
                            self.dq8,
                            K_EMBD as i32,
                            K_EPS,
                            1,
                            stream,
                        )
                    },
                    "tk_add_norm_quant(layer0 out)",
                )?;
            } else {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_router_route)(
                            ctx_ptr,
                            L.router_inp.as_f32(),
                            self.dh,
                            self.drl,
                            L.router_bias.as_f32(),
                            self.dids,
                            self.dw,
                            K_EMBD as i32,
                            K_EXPERTS as i32,
                            K_TOP_K as i32,
                            K_EXP_SCALE as f32,
                            self.route_counter,
                            stream,
                        )
                    },
                    "tk_router_route(step)",
                )?;

                let gu = K_FF_EXP * L.exp_gate.row_bytes;
                let dbytes = K_EMBD * L.exp_down.row_bytes;
                let moe_gu_items = [
                    self.mmvq_item(
                        &L.sh_gate,
                        self.dq8,
                        self.df1,
                        std::ptr::null(),
                        1,
                        0,
                        false,
                    ),
                    self.mmvq_item(&L.sh_up, self.dq8, self.df2, std::ptr::null(), 1, 0, false),
                    self.mmvq_item(
                        &L.exp_gate,
                        self.dq8,
                        unsafe { self.df1.add(K_FF_EXP) },
                        self.dids,
                        K_TOP_K as i32,
                        gu,
                        false,
                    ),
                    self.mmvq_item(
                        &L.exp_up,
                        self.dq8,
                        unsafe { self.df2.add(K_FF_EXP) },
                        self.dids,
                        K_TOP_K as i32,
                        gu,
                        false,
                    ),
                ];
                check_tk(
                    ctx_ptr,
                    unsafe { (self.tk.tk_mmvq_batch)(ctx_ptr, moe_gu_items.as_ptr(), 4, stream) },
                    "tk_mmvq_batch(moe gate+up)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_swiglu_quantize_q8_1)(
                            ctx_ptr,
                            self.df1,
                            self.df2,
                            self.dq8b,
                            ((K_TOP_K + 1) * K_FF_EXP) as i32,
                            1,
                            stream,
                        )
                    },
                    "tk_swiglu_quantize_q8_1(moe)",
                )?;

                let moe_down_items = [
                    self.mmvq_item(
                        &L.sh_down,
                        self.dq8b,
                        self.dysh,
                        std::ptr::null(),
                        1,
                        0,
                        false,
                    ),
                    self.mmvq_item(
                        &L.exp_down,
                        unsafe { (self.dq8b as *const u8).add(self.q8c_ff) as *const c_void },
                        self.dyexp,
                        self.dids,
                        K_TOP_K as i32,
                        dbytes,
                        true,
                    ),
                ];
                check_tk(
                    ctx_ptr,
                    unsafe { (self.tk.tk_mmvq_batch)(ctx_ptr, moe_down_items.as_ptr(), 2, stream) },
                    "tk_mmvq_batch(moe down)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_add_norm_quant)(
                            ctx_ptr,
                            self.dx,
                            self.dysh,
                            self.dyexp,
                            self.dw,
                            K_TOP_K as i32,
                            next_norm,
                            std::ptr::null_mut(),
                            self.dq8,
                            K_EMBD as i32,
                            K_EPS,
                            1,
                            stream,
                        )
                    },
                    "tk_add_norm_quant(moe out)",
                )?;
            }
        }
        self.tap(nl, self.dx, 1)?;

        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_mmvq)(
                    ctx_ptr,
                    to_tk_quant_type(self.m.output.qtype),
                    self.m.output.dev as *const c_void,
                    self.dq8 as *const c_void,
                    self.dlogits,
                    self.m.output.n_in as i32,
                    self.m.output.n_out as i32,
                    1,
                    stream,
                )
            },
            "tk_mmvq(output)",
        )?;
        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_argmax_advance)(
                    ctx_ptr,
                    self.dlogits,
                    K_VOCAB as i32,
                    self.tokpos_dev,
                    self.pos_dev,
                    stream,
                )
            },
            "tk_argmax_advance",
        )?;
        Ok(())
    }

    /// Emit the `n`-token (`1..=8`) speculative verification kernel sequence onto `self.s`,
    /// matching `laguna_engine.cpp:494-576`.
    fn step_n(&self, n: usize, sp: bool) -> Result<()> {
        assert!((1..=K_MAX_VERIFY).contains(&n));
        let stream = self.s as tk_stream;
        let ctx_ptr = self.tk_ctx.as_ptr();
        let nl = self.m.layers.len();

        // expert_major = 2 (libtalus_kernels): dense items of TALUS_LAGUNA_WIDE_MIN (9) columns and up go to the
        // wide column kernel (one weight pass per 16 columns, one barrier per pass; up to 8 columns the main batch
        // kernel's column path is used). Every column matches its one-column result bit for bit.
        let wide = n >= self.wide_min;
        // routed experts grouped by expert from egroup_min tokens (0: never)
        let egroup = self.egroup_min > 0 && n >= self.egroup_min.max(2);
        let dense_n = |t: &DevTensor, xq: *const c_void, y: *mut f32| -> tk_mmvq_item {
            let mut it = self.mmvq_item(t, xq, y, self.vzero, n as i32, 0, true);
            if wide {
                it.expert_major = 2;
            }
            it
        };

        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_embed_q8_0)(
                    ctx_ptr,
                    self.m.emb.dev as *const c_void,
                    self.vtok,
                    self.vx,
                    K_EMBD as i32,
                    n as i32,
                    stream,
                )
            },
            "tk_embed_q8_0(step_n)",
        )?;

        if nl > 0 {
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_add_norm_quant)(
                        ctx_ptr,
                        self.vx,
                        std::ptr::null(),
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        self.m.layers[0].attn_norm.as_f32(),
                        std::ptr::null_mut(),
                        self.vq8,
                        K_EMBD as i32,
                        K_EPS,
                        n as i32,
                        stream,
                    )
                },
                "tk_add_norm_quant(step_n layer0)",
            )?;
        }

        for il in 0..nl {
            self.tap(il, self.vx, n)?;
            let L = &self.m.layers[il];
            let H = n_head(il);
            let full = is_full(il);
            let n_rot = if full { 64 } else { 128 };
            let cos = if full { self.cos_full } else { self.cos_swa };
            let sin = if full { self.sin_full } else { self.sin_swa };
            let next_norm = if il + 1 < nl {
                self.m.layers[il + 1].attn_norm.as_f32()
            } else {
                self.m.output_norm.as_f32()
            };

            let qkv_items = [
                dense_n(&L.wq, self.vq8, self.vq),
                dense_n(&L.wk, self.vq8, self.vkc),
                dense_n(&L.wv, self.vq8, self.vvc),
                dense_n(&L.attn_gate, self.vq8, self.vg),
            ];
            self.run_items(&qkv_items, "tk_mmvq_batch(step_n qkv)")?;

            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_qkv_norm_rope_store)(
                        ctx_ptr,
                        self.vq,
                        self.vkc,
                        self.vvc,
                        L.q_norm.as_f32(),
                        L.k_norm.as_f32(),
                        H as i32,
                        K_HEAD_KV as i32,
                        K_EPS,
                        n_rot,
                        cos,
                        sin,
                        self.pos_dev,
                        self.kpool[il] as *mut c_void,
                        self.vpool[il] as *mut c_void,
                        self.vtl[il] as i32,
                        if self.kv_q8 {
                            self.kscale[il]
                        } else {
                            std::ptr::null_mut()
                        },
                        if self.kv_q8 {
                            self.vscale[il]
                        } else {
                            std::ptr::null_mut()
                        },
                        n as i32,
                        stream,
                    )
                },
                "tk_qkv_norm_rope_store(step_n)",
            )?;

            let spl = sp && full;
            // With sparse_exact each token's block bounds and scores are taken as its own decode step takes
            // them (its block's bound over keys up to itself, not the window's later keys): verify rows stay
            // bitwise equal to verify(1) rows past sparse_min too
            let (sp_steps, sp_n) = if spl && self.sparse_exact && n > 1 {
                (n, 1usize)
            } else {
                (1, n)
            };
            for t in 0..if spl { sp_steps } else { 0 } {
                let pos_t = if sp_steps > 1 {
                    unsafe { self.vposa.add(t) }
                } else {
                    self.pos_dev
                };
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_kbound_update)(
                            ctx_ptr,
                            self.kpool[il] as *const c_void,
                            self.kscale[il],
                            self.kbnd[il],
                            self.nb_max,
                            K_HEAD_KV as i32,
                            0,
                            0,
                            0,
                            pos_t,
                            sp_n as i32,
                            self.sp_bs,
                            self.sp_f16,
                            stream,
                        )
                    },
                    "tk_kbound_update(step_n)",
                )?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_sparse_scores)(
                            ctx_ptr,
                            self.vq.add(t * H * K_HEAD_DIM),
                            self.kbnd[il] as *const c_void,
                            self.sp_ub.add(t * H * self.nb_max as usize),
                            self.nb_max,
                            H as i32,
                            K_HEAD_KV as i32,
                            pos_t,
                            sp_n as i32,
                            self.sp_bs,
                            self.sp_f16,
                            stream,
                        )
                    },
                    "tk_sparse_scores(step_n)",
                )?;
            }
            if spl {
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_sparse_select)(
                            ctx_ptr,
                            self.sp_ub,
                            self.nb_max,
                            H as i32,
                            self.pos_dev,
                            n as i32,
                            self.sp_k,
                            self.sp_sink,
                            self.sp_recent,
                            self.sp_sel,
                            if self.sp_per_kv { K_HEAD_KV as i32 } else { 1 },
                            self.sp_bs,
                            self.sp_scr,
                            stream,
                        )
                    },
                    "tk_sparse_select(step_n)",
                )?;
            }

            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_laguna_decode_attn)(
                        ctx_ptr,
                        self.vq,
                        self.kpool[il],
                        self.vpool[il],
                        if self.kv_q8 {
                            self.kscale[il]
                        } else {
                            std::ptr::null()
                        },
                        if self.kv_q8 {
                            self.vscale[il]
                        } else {
                            std::ptr::null()
                        },
                        H as i32,
                        K_HEAD_KV as i32,
                        if full { 0 } else { K_SWA as i32 },
                        self.vtl[il] as i32,
                        self.pos_dev,
                        K_SPLIT,
                        self.vpart,
                        self.vml,
                        n as i32,
                        if spl { self.sp_sel } else { std::ptr::null() },
                        self.sp_k,
                        self.sp_per_kv,
                        self.sp_bs,
                        stream,
                    )
                },
                "tk_laguna_decode_attn(step_n)",
            )?;
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_attn_merge_gate_quant)(
                        ctx_ptr, self.vpart, self.vml, self.vg, self.vq8, H as i32, K_SPLIT,
                        n as i32, stream,
                    )
                },
                "tk_attn_merge_gate_quant(step_n)",
            )?;
            if self.vexact {
                self.mmvq_cols_dec(&L.wo, self.vq8, self.va, n)?;
            } else {
                let wo_items = [dense_n(&L.wo, self.vq8, self.va)];
                self.run_items(&wo_items, "tk_mmvq_batch(step_n wo)")?;
            }
            check_tk(
                ctx_ptr,
                unsafe {
                    (self.tk.tk_add_norm_quant)(
                        ctx_ptr,
                        self.vx,
                        self.va,
                        std::ptr::null(),
                        std::ptr::null(),
                        0,
                        L.ffn_norm.as_f32(),
                        self.vh,
                        self.vq8,
                        K_EMBD as i32,
                        K_EPS,
                        n as i32,
                        stream,
                    )
                },
                "tk_add_norm_quant(step_n ffn)",
            )?;

            if il == 0 {
                if self.vexact {
                    self.l0_gate_up_dec(n)?;
                } else {
                    let d0_items = [
                        dense_n(&L.d_gate, self.vq8, self.vf1),
                        dense_n(&L.d_up, self.vq8, self.vf2),
                    ];
                    self.run_items(&d0_items, "tk_mmvq_batch(step_n layer0 gate+up)")?;
                }
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_swiglu_quantize_q8_1)(
                            ctx_ptr,
                            self.vf1,
                            self.vf2,
                            self.vq8b,
                            K_FF_DENSE as i32,
                            n as i32,
                            stream,
                        )
                    },
                    "tk_swiglu_quantize_q8_1(step_n layer0)",
                )?;
                if self.vexact {
                    self.mmvq_cols_dec(&L.d_down, self.vq8b, self.va, n)?;
                } else {
                    let d0_down_items = [dense_n(&L.d_down, self.vq8b, self.va)];
                    self.run_items(&d0_down_items, "tk_mmvq_batch(step_n layer0 down)")?;
                }
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_add_norm_quant)(
                            ctx_ptr,
                            self.vx,
                            self.va,
                            std::ptr::null(),
                            std::ptr::null(),
                            0,
                            next_norm,
                            std::ptr::null_mut(),
                            self.vq8,
                            K_EMBD as i32,
                            K_EPS,
                            n as i32,
                            stream,
                        )
                    },
                    "tk_add_norm_quant(step_n layer0 out)",
                )?;
            } else {
                let mut t0 = 0usize;
                while t0 < n {
                    let nt = (n - t0).min(K_ROUTER_WIN);
                    check_tk(
                        ctx_ptr,
                        unsafe {
                            (self.tk.tk_f32_matvec_rows_n)(
                                ctx_ptr,
                                L.router_inp.as_f32(),
                                self.vh.add(t0 * K_EMBD),
                                self.vrl.add(t0 * K_EXPERTS),
                                K_EMBD as i32,
                                K_EXPERTS as i32,
                                nt as i32,
                                stream,
                            )
                        },
                        "tk_f32_matvec_rows_n(step_n)",
                    )?;
                    t0 += nt;
                }
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_route_topk)(
                            ctx_ptr,
                            self.vrl,
                            L.router_bias.as_f32(),
                            n as i32,
                            K_TOP_K as i32,
                            K_EXP_SCALE as f32,
                            self.vids,
                            self.vw,
                            stream,
                        )
                    },
                    "tk_route_topk(step_n)",
                )?;

                let gu = K_FF_EXP * L.exp_gate.row_bytes;
                let dbytes = K_EMBD * L.exp_down.row_bytes;
                let mut ge = self.mmvq_item(
                    &L.exp_gate,
                    self.vq8,
                    unsafe { self.vf1.add(n * K_FF_EXP) },
                    self.vids,
                    (n * K_TOP_K) as i32,
                    gu,
                    true,
                );
                ge.x_group = K_TOP_K as i32;
                if egroup {
                    ge.expert_major = 3 | ((K_EXPERTS as i32) << 8); // grouped, pool size for the slot grid
                }
                let mut ue = self.mmvq_item(
                    &L.exp_up,
                    self.vq8,
                    unsafe { self.vf2.add(n * K_FF_EXP) },
                    self.vids,
                    (n * K_TOP_K) as i32,
                    gu,
                    true,
                );
                ue.x_group = K_TOP_K as i32;
                if egroup {
                    ue.expert_major = 3 | ((K_EXPERTS as i32) << 8); // grouped, pool size for the slot grid
                }

                let moe_gu_items = [
                    dense_n(&L.sh_gate, self.vq8, self.vf1),
                    dense_n(&L.sh_up, self.vq8, self.vf2),
                    ge,
                    ue,
                ];
                self.run_items(&moe_gu_items, "tk_mmvq_batch(step_n moe gate+up)")?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_swiglu_quantize_q8_1)(
                            ctx_ptr,
                            self.vf1,
                            self.vf2,
                            self.vq8b,
                            (n * (K_TOP_K + 1) * K_FF_EXP) as i32,
                            1,
                            stream,
                        )
                    },
                    "tk_swiglu_quantize_q8_1(step_n moe)",
                )?;

                let mut de = self.mmvq_item(
                    &L.exp_down,
                    unsafe { (self.vq8b as *const u8).add(n * self.q8c_ff) as *const c_void },
                    self.vyexp,
                    self.vids,
                    (n * K_TOP_K) as i32,
                    dbytes,
                    true,
                );
                if egroup {
                    de.expert_major = 3 | ((K_EXPERTS as i32) << 8); // grouped, pool size for the slot grid
                }
                let moe_down_items = [dense_n(&L.sh_down, self.vq8b, self.vysh), de];
                self.run_items(&moe_down_items, "tk_mmvq_batch(step_n moe down)")?;
                check_tk(
                    ctx_ptr,
                    unsafe {
                        (self.tk.tk_add_norm_quant)(
                            ctx_ptr,
                            self.vx,
                            self.vysh,
                            self.vyexp,
                            self.vw,
                            K_TOP_K as i32,
                            next_norm,
                            std::ptr::null_mut(),
                            self.vq8,
                            K_EMBD as i32,
                            K_EPS,
                            n as i32,
                            stream,
                        )
                    },
                    "tk_add_norm_quant(step_n moe out)",
                )?;
            }
        }

        self.tap(nl, self.vx, n)?;
        if self.vexact {
            self.mmvq_cols_dec(&self.m.output, self.vq8, self.vlogits, n)?;
        } else {
            let out_items = [dense_n(&self.m.output, self.vq8, self.vlogits)];
            self.run_items(&out_items, "tk_mmvq_batch(step_n output)")?;
        }
        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_argmax_rows)(
                    ctx_ptr,
                    self.vlogits,
                    n as i32,
                    K_VOCAB as i32,
                    self.vout,
                    stream,
                )
            },
            "tk_argmax_rows(step_n)",
        )?;
        Ok(())
    }

    /// Run a single-token decode step for `token` at `pos`, returning the greedy argmax token ID
    /// and optionally copying `[K_VOCAB]` f32 logits to `logits`, matching `laguna_engine.cpp:626-656`.
    pub fn decode(&mut self, token: i32, pos: i32, logits: Option<&mut [f32]>) -> Result<i32> {
        if self.dec_verify {
            // A decode step is a one-token verify, so every row a request emits comes from one kernel set and
            // greedy output no longer depends on what was drafted (verify(n) rows are bitwise verify(1) rows).
            let mut am = [0i32; 1];
            self.verify(&[token], pos, &mut am, logits)?;
            return Ok(am[0]);
        }
        let sp = self.sparse && (pos + 1 >= self.sp_min);
        if sp && self.bnd_valid < pos {
            self.bounds_catch_up(pos)?;
        }

        let tp: [i32; 2] = [token, pos];
        check_hip(
            unsafe {
                (self.hip.hipMemcpyAsync)(
                    self.tokpos_dev as *mut c_void,
                    tp.as_ptr() as *const c_void,
                    8,
                    HipRuntime::MEMCPY_H2D,
                    self.s,
                )
            },
            "hipMemcpyAsync(decode tokpos)",
        )?;

        if self.use_graph {
            self.ensure_decode_graph(sp)?;
            let ex = if sp { self.exec_sp } else { self.exec };
            check_hip(
                unsafe { (self.hip.hipGraphLaunch)(ex, self.s) },
                "hipGraphLaunch(decode)",
            )?;
        } else {
            self.step(sp)?;
        }

        let mut next: i32 = 0;
        check_hip(
            unsafe {
                (self.hip.hipMemcpyAsync)(
                    &mut next as *mut i32 as *mut c_void,
                    self.tokpos_dev as *const c_void,
                    4,
                    HipRuntime::MEMCPY_D2H,
                    self.s,
                )
            },
            "hipMemcpyAsync(decode next)",
        )?;
        if let Some(dst) = logits {
            assert!(dst.len() >= K_VOCAB);
            check_hip(
                unsafe {
                    (self.hip.hipMemcpyAsync)(
                        dst.as_mut_ptr() as *mut c_void,
                        self.dlogits as *const c_void,
                        K_VOCAB * 4,
                        HipRuntime::MEMCPY_D2H,
                        self.s,
                    )
                },
                "hipMemcpyAsync(decode logits)",
            )?;
        }
        check_hip(
            unsafe { (self.hip.hipStreamSynchronize)(self.s) },
            "hipStreamSynchronize(decode)",
        )?;
        self.bnd_valid = if sp { pos + 1 } else { self.bnd_valid.min(pos) };
        self.kv_hw = self.kv_hw.max(pos as usize + 1);
        self.swa_dirty = true;
        Ok(next)
    }

    /// (opt-in, TALUS_LAGUNA_VERIFY_GEMM) verify(n) as a prefill micro-batch of the window - MMQ dense
    /// projections, grouped-MMQ routed experts, WMMA prompt attention - plus the head over every row. Other kernels
    /// and reduction orders than verify(1)'s, so its rows are not verify(1)'s (greedy output then depends on what was
    /// drafted); measured with KLD. The KV cells past the window are zeroed first (the prompt-attention V tiles read
    /// past the context end), so a row never depends on an earlier rejected window.
    fn verify_gemm(
        &mut self,
        tokens: &[i32],
        pos: i32,
        argmax: &mut [i32],
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let n = tokens.len();
        let p0 = pos as usize;
        assert!(n <= self.U && p0 + n <= self.ctx);
        if self.vgemm_clear {
            self.clear_kv_from(p0 + n, true)?;
        }
        let stream = self.s as tk_stream;
        let ctx_ptr = self.tk_ctx.as_ptr();
        check_hip(
            unsafe {
                (self.hip.hipMemcpyAsync)(
                    self.ptok_dev as *mut c_void,
                    tokens.as_ptr() as *const c_void,
                    n * 4,
                    HipRuntime::MEMCPY_H2D,
                    self.s,
                )
            },
            "hipMemcpyAsync(verify_gemm tokens)",
        )?;
        self.prefill_ubatch(tokens, 0, n, p0, None)?;
        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_rms_norm_to)(
                    ctx_ptr,
                    self.h,
                    self.x,
                    self.m.output_norm.as_f32(),
                    n as i32,
                    K_EMBD as i32,
                    K_EPS,
                    stream,
                )
            },
            "tk_rms_norm_to(verify_gemm output_norm)",
        )?;
        self.dense(&self.m.output, self.h, self.logits_buf, n)?;
        check_tk(
            ctx_ptr,
            unsafe {
                (self.tk.tk_argmax_rows)(
                    ctx_ptr,
                    self.logits_buf,
                    n as i32,
                    K_VOCAB as i32,
                    self.vout,
                    stream,
                )
            },
            "tk_argmax_rows(verify_gemm)",
        )?;
        check_hip(
            unsafe {
                (self.hip.hipMemcpyAsync)(
                    argmax.as_mut_ptr() as *mut c_void,
                    self.vout as *const c_void,
                    n * 4,
                    HipRuntime::MEMCPY_D2H,
                    self.s,
                )
            },
            "hipMemcpyAsync(verify_gemm argmax)",
        )?;
        if let Some(dst) = logits {
            assert!(dst.len() >= n * K_VOCAB);
            check_hip(
                unsafe {
                    (self.hip.hipMemcpyAsync)(
                        dst.as_mut_ptr() as *mut c_void,
                        self.logits_buf as *const c_void,
                        n * K_VOCAB * 4,
                        HipRuntime::MEMCPY_D2H,
                        self.s,
                    )
                },
                "hipMemcpyAsync(verify_gemm logits)",
            )?;
        }
        check_hip(
            unsafe { (self.hip.hipStreamSynchronize)(self.s) },
            "hipStreamSynchronize(verify_gemm)",
        )
    }

    /// verify widths from which the routed experts run grouped (0: never) and from which verify goes through
    /// the prefill kernels (0: never), for the serve cost table.
    pub fn verify_modes(&self) -> (usize, usize) {
        (self.egroup_min, self.vgemm_min)
    }

    /// Capture verify width `n`'s graph (sparse variant with `sp`) unless it exists.
    pub fn ensure_verify_graph(&mut self, n: usize, sp: bool) -> Result<()> {
        let ex_ptr = if sp { self.vexec_sp[n] } else { self.vexec[n] };
        if !self.use_graph || !ex_ptr.is_null() {
            return Ok(());
        }
        check_hip(
            unsafe { (self.hip.hipStreamSynchronize)(self.s) },
            "hipStreamSynchronize(before capture verify)",
        )?;
        check_hip(
            unsafe {
                (self.hip.hipStreamBeginCapture)(self.s, HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            },
            "hipStreamBeginCapture(verify)",
        )?;
        let step_res = self.step_n(n, sp);
        let mut g: HipGraph = std::ptr::null_mut();
        let end_err = unsafe { (self.hip.hipStreamEndCapture)(self.s, &mut g) };
        step_res?;
        check_hip(end_err, "hipStreamEndCapture(verify)")?;

        let mut new_ex: HipGraphExec = std::ptr::null_mut();
        let inst_res = check_hip(
            unsafe {
                (self.hip.hipGraphInstantiate)(
                    &mut new_ex,
                    g,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            },
            "hipGraphInstantiate(verify)",
        );
        unsafe {
            let _ = (self.hip.hipGraphDestroy)(g);
        }
        inst_res?;
        if sp {
            self.vexec_sp[n] = new_ex;
        } else {
            self.vexec[n] = new_ex;
        }
        Ok(())
    }

    /// Capture the verify graphs of widths `1 ..= wmax` (and their sparse variants when the context reaches
    /// block-sparse attention) up front, so serving never captures one (no first-use stall, no memory growth).
    pub fn capture_verify_graphs(&mut self, wmax: usize) -> Result<usize> {
        let mut n_graphs = 0usize;
        let sp_too = self.sparse && (self.max_ctx() as i32) >= self.sp_min;
        for n in 1..=wmax.min(K_MAX_VERIFY) {
            if self.vgemm_min > 0 && n >= self.vgemm_min.max(2) {
                break; // these widths run the prefill kernels, eagerly
            }
            for sp in [false, true] {
                if sp && !sp_too {
                    continue;
                }
                self.ensure_verify_graph(n, sp)?;
                n_graphs += 1;
            }
        }
        Ok(n_graphs)
    }

    /// Probe: the stored K / V cells (and scales) of positions `from .. to` on every layer, one record per
    /// (layer, position): K row, K scales, V column (all heads / dims), V scales; plus each full-attention layer's
    /// block bounds of the blocks holding `from .. to` (sparse only). For byte comparisons of KV writes.
    pub fn kv_dump(&self, from: usize, to: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        use crate::laguna::weights::{ks_index as ksi, vs_index as vsi, vt_index as vti};
        let d2h = |dst: &mut [u8], src: *const u8| -> Result<()> {
            check_hip(
                unsafe {
                    (self.hip.hipMemcpy)(
                        dst.as_mut_ptr() as *mut c_void,
                        src as *const c_void,
                        dst.len(),
                        HipRuntime::MEMCPY_D2H,
                    )
                },
                "hipMemcpy(kv_dump)",
            )
        };
        check_hip(
            unsafe { (self.hip.hipStreamSynchronize)(self.s) },
            "sync(kv_dump)",
        )?;
        let esz = if self.kv_q8 { 1usize } else { 2usize };
        let mut out = Vec::new();
        for il in 0..self.m.layers.len() {
            let stride = self.vtl[il];
            for pos in from..to {
                let slot = if stride < self.ctx16 {
                    pos % stride
                } else {
                    pos
                };
                let mut k = vec![0u8; self.cell * esz];
                d2h(&mut k, unsafe {
                    (self.kpool[il] as *const u8).add(slot * self.cell * esz)
                })?;
                out.extend_from_slice(&k);
                if self.kv_q8 {
                    let mut ks = vec![0u8; K_HEAD_KV * 4 * 2];
                    d2h(&mut ks, unsafe {
                        (self.kscale[il] as *const u8).add(ksi(slot, 0, 0, K_HEAD_KV) * 2)
                    })?;
                    out.extend_from_slice(&ks);
                }
                for kvh in 0..K_HEAD_KV {
                    // the tile holding the slot: [128 dims][16 positions]
                    let mut t = vec![0u8; 128 * 16 * esz];
                    d2h(&mut t, unsafe {
                        (self.vpool[il] as *const u8).add(vti(kvh, slot & !15, 0, stride) * esz)
                    })?;
                    for d in 0..K_HEAD_DIM {
                        let o = (d * 16 + (slot & 15)) * esz;
                        out.extend_from_slice(&t[o..o + esz]);
                    }
                    if self.kv_q8 {
                        let mut vs = vec![0u8; 4 * 16 * 2];
                        d2h(&mut vs, unsafe {
                            (self.vscale[il] as *const u8).add(vsi(kvh, slot & !15, 0, stride) * 2)
                        })?;
                        for blk in 0..4 {
                            let o = (blk * 16 + (slot & 15)) * 2;
                            out.extend_from_slice(&vs[o..o + 2]);
                        }
                    }
                }
            }
        }
        let mut bnd = Vec::new();
        if self.sparse && to > from {
            let bsz = if self.sp_f16 { 2usize } else { 4usize };
            let (b0, b1) = (from / self.sp_bs as usize, (to - 1) / self.sp_bs as usize);
            for il in 0..self.m.layers.len() {
                if !is_full(il) {
                    continue;
                }
                for kvh in 0..K_HEAD_KV {
                    let mut b = vec![0u8; (b1 + 1 - b0) * 256 * bsz];
                    let off = ((kvh * self.nb_max as usize + b0) * 256) * bsz;
                    d2h(&mut b, unsafe { (self.kbnd[il] as *const u8).add(off) })?;
                    bnd.extend_from_slice(&b);
                }
            }
        }
        Ok((out, bnd))
    }

    /// Bytes per (layer, position) record of `kv_dump`.
    pub fn kv_record_bytes(&self) -> usize {
        let esz = if self.kv_q8 { 1usize } else { 2usize };
        2 * self.cell * esz + if self.kv_q8 { 2 * K_HEAD_KV * 4 * 2 } else { 0 }
    }

    /// Run multi-token speculative verification for `tokens` (`1..=8` tokens) starting at `pos`,
    /// writing greedy argmax IDs into `argmax` and optionally copying `[n * K_VOCAB]` f32 logits
    /// into `logits`, matching `laguna_engine.cpp:658-689`.
    pub fn verify(
        &mut self,
        tokens: &[i32],
        pos: i32,
        argmax: &mut [i32],
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let n = tokens.len();
        assert!((1..=K_MAX_VERIFY).contains(&n) && argmax.len() >= n);
        if self.vgemm_min > 0 && n >= self.vgemm_min.max(2) {
            return self.verify_gemm(tokens, pos, argmax, logits);
        }
        let sp = self.sparse && (pos + (n as i32) >= self.sp_min);
        if sp && self.bnd_valid < pos {
            self.bounds_catch_up(pos)?;
        }

        check_hip(
            unsafe {
                (self.hip.hipMemcpyAsync)(
                    self.vtok as *mut c_void,
                    tokens.as_ptr() as *const c_void,
                    n * 4,
                    HipRuntime::MEMCPY_H2D,
                    self.s,
                )
            },
            "hipMemcpyAsync(verify vtok)",
        )?;
        check_hip(
            unsafe {
                (self.hip.hipMemcpyAsync)(
                    self.pos_dev as *mut c_void,
                    &pos as *const i32 as *const c_void,
                    4,
                    HipRuntime::MEMCPY_H2D,
                    self.s,
                )
            },
            "hipMemcpyAsync(verify pos)",
        )?;
        if sp && self.sparse_exact && n > 1 {
            let pa: [i32; K_MAX_VERIFY] = std::array::from_fn(|i| pos + i as i32);
            check_hip(
                unsafe {
                    (self.hip.hipMemcpyAsync)(
                        self.vposa as *mut c_void,
                        pa.as_ptr() as *const c_void,
                        n * 4,
                        HipRuntime::MEMCPY_H2D,
                        self.s,
                    )
                },
                "hipMemcpyAsync(verify positions)",
            )?;
            check_hip(
                unsafe { (self.hip.hipStreamSynchronize)(self.s) },
                "sync(verify positions)",
            )?;
        }

        if self.use_graph {
            self.ensure_verify_graph(n, sp)?;
            let ex = if sp { self.vexec_sp[n] } else { self.vexec[n] };
            check_hip(
                unsafe { (self.hip.hipGraphLaunch)(ex, self.s) },
                "hipGraphLaunch(verify)",
            )?;
        } else {
            self.step_n(n, sp)?;
        }

        check_hip(
            unsafe {
                (self.hip.hipMemcpyAsync)(
                    argmax.as_mut_ptr() as *mut c_void,
                    self.vout as *const c_void,
                    n * 4,
                    HipRuntime::MEMCPY_D2H,
                    self.s,
                )
            },
            "hipMemcpyAsync(verify argmax)",
        )?;
        if let Some(dst) = logits {
            assert!(dst.len() >= n * K_VOCAB);
            check_hip(
                unsafe {
                    (self.hip.hipMemcpyAsync)(
                        dst.as_mut_ptr() as *mut c_void,
                        self.vlogits as *const c_void,
                        n * K_VOCAB * 4,
                        HipRuntime::MEMCPY_D2H,
                        self.s,
                    )
                },
                "hipMemcpyAsync(verify logits)",
            )?;
        }
        check_hip(
            unsafe { (self.hip.hipStreamSynchronize)(self.s) },
            "hipStreamSynchronize(verify)",
        )?;
        self.bnd_valid = if sp {
            pos + (n as i32)
        } else {
            self.bnd_valid.min(pos)
        };
        self.kv_hw = self.kv_hw.max(pos as usize + n);
        self.swa_dirty = true;
        Ok(())
    }

    /// Save the current KV cache state and `tokens` (`[0..n_tokens]`) to `path` via an atomic
    /// `<path>.tmp` rename, matching `laguna_engine.cpp:702-739`.
    pub fn save_session<F>(
        &self,
        path: &Path,
        tokens: &[i32],
        mut progress: F,
    ) -> std::result::Result<usize, String>
    where
        F: FnMut(usize, usize),
    {
        let n = tokens.len();
        if n == 0 || n > self.ctx {
            return Err("invalid bad token count".to_string());
        }
        let nl = self.m.layers.len();
        let esz = if self.kv_q8 { 1usize } else { 2usize };
        let hdr = SessionHeader::new(
            nl,
            self.kv_q8,
            n as i32,
            self.m.model_id,
            self.ring,
            self.ctx16,
        );
        let payload = session_payload_bytes(nl, self.kv_q8, n, self.ctx16, &self.vtl);
        let total = SessionHeader::SIZE + n * 4 + payload;

        let tmp_path = {
            let mut os = path.as_os_str().to_os_string();
            os.push(".tmp");
            std::path::PathBuf::from(os)
        };
        let mut f = File::create(&tmp_path).map_err(|e| format!("io cannot open {e}"))?;

        if unsafe { (self.hip.hipStreamSynchronize)(self.s) } != 0 {
            let _ = fs::remove_file(&tmp_path);
            return Err("io hipStreamSynchronize failed".to_string());
        }

        let mut buf = vec![0u8; n.max(self.ring) * self.cell * esz + 4096];
        let mut done = SessionHeader::SIZE + n * 4;

        let write_res = (|| -> std::result::Result<(), String> {
            f.write_all(&hdr.to_bytes())
                .map_err(|_| "io write failed (disk full?)".to_string())?;
            let tok_bytes =
                unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, n * 4) };
            f.write_all(tok_bytes)
                .map_err(|_| "io write failed (disk full?)".to_string())?;

            let mut put = |dev: *const c_void, sz: usize| -> std::result::Result<usize, String> {
                if sz == 0 {
                    return Ok(0);
                }
                self.hip
                    .copy_d2h(buf.as_mut_ptr() as *mut c_void, dev, sz)
                    .map_err(|e| format!("io {e}"))?;
                f.write_all(&buf[..sz])
                    .map_err(|_| "io write failed (disk full?)".to_string())?;
                Ok(sz)
            };

            for il in 0..nl {
                let np = if self.vtl[il] == self.ctx16 {
                    n
                } else {
                    self.vtl[il]
                };
                let tiles = (np + 15) / 16;
                done += put(self.kpool[il] as *const c_void, np * self.cell * esz)?;
                if self.kv_q8 {
                    done += put(self.kscale[il] as *const c_void, np * K_HEAD_KV * 4 * 2)?;
                }
                for kvh in 0..K_HEAD_KV {
                    let v_off = vt_index(kvh, 0, 0, self.vtl[il]) * esz;
                    let v_ptr =
                        unsafe { (self.vpool[il] as *const u8).add(v_off) as *const c_void };
                    done += put(v_ptr, tiles * 16 * K_HEAD_DIM * esz)?;
                    if self.kv_q8 {
                        let vs_off = vs_index(kvh, 0, 0, self.vtl[il]);
                        let vs_ptr = unsafe { self.vscale[il].add(vs_off) as *const c_void };
                        done += put(vs_ptr, tiles * 4 * 16 * 2)?;
                    }
                }
                progress(done, total);
            }
            f.flush()
                .map_err(|_| "io write failed (disk full?)".to_string())
        })();

        drop(f);
        if let Err(e) = write_res {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }
        if fs::rename(&tmp_path, path).is_err() {
            let _ = fs::remove_file(&tmp_path);
            return Err("io rename failed".to_string());
        }
        Ok(total)
    }

    /// Restore KV cache state and token history from `path`, matching `laguna_engine.cpp:741-801`.
    pub fn load_session<F>(
        &mut self,
        path: &Path,
        mut progress: F,
    ) -> std::result::Result<(Vec<i32>, usize), String>
    where
        F: FnMut(usize, usize),
    {
        let mut f = File::open(path).map_err(|_| "io session file not found".to_string())?;
        let file_len = f
            .metadata()
            .map_err(|_| "io bad session header".to_string())?
            .len() as usize;

        let mut hdr_bytes = [0u8; SessionHeader::SIZE];
        f.read_exact(&mut hdr_bytes)
            .map_err(|_| "io bad session header".to_string())?;
        let hdr = SessionHeader::from_bytes(&hdr_bytes);
        let nl = self.m.layers.len();
        if !hdr.validate(
            nl,
            self.kv_q8,
            self.m.model_id,
            self.ring,
            self.ctx16,
            self.ctx,
        ) {
            return Err(
                "invalid session was saved by a different model or engine configuration"
                    .to_string(),
            );
        }

        let n = hdr.n_tokens as usize;
        let esz = if self.kv_q8 { 1usize } else { 2usize };
        let payload = session_payload_bytes(nl, self.kv_q8, n, self.ctx16, &self.vtl);
        let total = SessionHeader::SIZE + n * 4 + payload;
        if file_len != total {
            return Err("io the file has the wrong size".to_string());
        }

        let mut toks = vec![0i32; n];
        {
            let tok_bytes =
                unsafe { std::slice::from_raw_parts_mut(toks.as_mut_ptr() as *mut u8, n * 4) };
            f.read_exact(tok_bytes)
                .map_err(|_| "io truncated token list".to_string())?;
        }
        for &t in &toks {
            if t < 0 || (t as usize) >= K_VOCAB {
                return Err("invalid token id out of range".to_string());
            }
        }

        if unsafe { (self.hip.hipStreamSynchronize)(self.s) } != 0 {
            return Err("io hipStreamSynchronize failed".to_string());
        }

        let mut buf = vec![0u8; n.max(self.ring) * self.cell * esz + 4096];
        let mut done = SessionHeader::SIZE + n * 4;

        let mut get = |dev: *mut c_void, sz: usize| -> std::result::Result<usize, String> {
            if sz == 0 {
                return Ok(0);
            }
            f.read_exact(&mut buf[..sz])
                .map_err(|_| "io read failed part way; the cache is no longer valid".to_string())?;
            self.hip
                .copy_h2d(dev, buf.as_ptr() as *const c_void, sz)
                .map_err(|_| "io read failed part way; the cache is no longer valid".to_string())?;
            Ok(sz)
        };

        for il in 0..nl {
            let np = if self.vtl[il] == self.ctx16 {
                n
            } else {
                self.vtl[il]
            };
            let tiles = (np + 15) / 16;
            done += get(self.kpool[il] as *mut c_void, np * self.cell * esz)?;
            if self.kv_q8 {
                done += get(self.kscale[il] as *mut c_void, np * K_HEAD_KV * 4 * 2)?;
            }
            for kvh in 0..K_HEAD_KV {
                let v_off = vt_index(kvh, 0, 0, self.vtl[il]) * esz;
                let v_ptr = unsafe { (self.vpool[il] as *mut u8).add(v_off) as *mut c_void };
                done += get(v_ptr, tiles * 16 * K_HEAD_DIM * esz)?;
                if self.kv_q8 {
                    let vs_off = vs_index(kvh, 0, 0, self.vtl[il]);
                    let vs_ptr = unsafe { self.vscale[il].add(vs_off) as *mut c_void };
                    done += get(vs_ptr, tiles * 4 * 16 * 2)?;
                }
            }
            progress(done, total);
        }

        self.bnd_valid = 0;
        self.kv_hw = self.kv_hw.max(toks.len());
        self.swa_dirty = true;
        Ok((toks, total))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.hip.hipStreamSynchronize)(self.s);
            if !self.exec.is_null() {
                let _ = (self.hip.hipGraphExecDestroy)(self.exec);
            }
            if !self.exec_sp.is_null() {
                let _ = (self.hip.hipGraphExecDestroy)(self.exec_sp);
            }
            for ex in &mut self.vexec {
                if !ex.is_null() {
                    let _ = (self.hip.hipGraphExecDestroy)(*ex);
                    *ex = std::ptr::null_mut();
                }
            }
            for ex in &mut self.vexec_sp {
                if !ex.is_null() {
                    let _ = (self.hip.hipGraphExecDestroy)(*ex);
                    *ex = std::ptr::null_mut();
                }
            }
        }
        for &p in &self.allocs {
            self.hip.free_device(p);
        }
        unsafe {
            let _ = (self.hip.hipStreamDestroy)(self.s);
        }
    }
}
