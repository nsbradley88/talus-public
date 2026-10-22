//! Qwen3.5 / Qwen3.6 / Qwen3.8 inference engine (`q36_engine.cu` lines 566–1599 & 2256–3100).

#![allow(non_snake_case, unused_unsafe)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

use super::ffi::*;
use super::graphs::{DFlash, DfCtx, DraftLen, DrafterPick};
use super::weights::{
    hip_check, hip_malloc_bytes, to_tk_qtype, LayerW, Mat, Q36Model, Q36Options, GGML_TYPE_BF16,
    GGML_TYPE_F32,
};

/// Expanded INT8 trunk matrix (`talus::prefill::I8Mat` in `tp_i8gemm.hpp`).
#[derive(Debug, Clone, Copy)]
pub struct I8Mat {
    pub qs: *mut i8,
    pub ws: *mut f32,
    pub n: i32,
    pub k: i32,
    pub mode: i32,
    pub bytes: usize,
}

unsafe impl Send for I8Mat {}
unsafe impl Sync for I8Mat {}

/// Expanded INT8 weight cache (`talus::prefill::I8Cache` in `tp_i8gemm.hpp`).
#[derive(Default)]
pub struct I8Cache {
    pub mats: HashMap<usize, I8Mat>,
    pub bytes: usize,
    pub budget_guard: Option<talus_mem::ReservationGuard>,
}

unsafe impl Send for I8Cache {}
unsafe impl Sync for I8Cache {}

impl I8Cache {
    #[inline]
    pub fn find(&self, w_ptr: *const u8) -> Option<&I8Mat> {
        self.mats.get(&(w_ptr as usize))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.mats.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mats.is_empty()
    }

    pub fn clear(&mut self) {
        for (_, m) in self.mats.drain() {
            unsafe {
                if !m.qs.is_null() {
                    let _ = hipFree(m.qs.cast::<c_void>());
                }
                if !m.ws.is_null() {
                    let _ = hipFree(m.ws.cast::<c_void>());
                }
            }
        }
        self.bytes = 0;
        self.budget_guard = None;
    }
}

impl Drop for I8Cache {
    fn drop(&mut self) {
        self.clear();
    }
}

pub const K_CHUNK_CELLS: i32 = 256;
pub const K_ATT_CHUNK: i32 = 512;
pub const K_ATT_SMALL: i32 = 128;
pub const K_ATT_SMALL_CELLS: i32 = 4096;
pub const K_ARG_BLOCKS: i32 = 128;

#[inline]
pub const fn att_chunk_size(ck: i32) -> i32 {
    ck & 0xffff
}

#[inline]
pub const fn att_small_cells(ck: i32) -> i32 {
    let s = (ck >> 16) * 128;
    if s > 0 {
        s
    } else {
        4096
    }
}

#[inline]
pub fn att_vt_cells(ctx: i32, ck: i32) -> i32 {
    let sc = att_small_cells(ck);
    let small = ctx.min(sc);
    let rest = (ctx - small).max(0);
    ((small + 127) / 128) * 128 + ((rest + 511) / 512) * 512
}

#[inline]
pub fn att_chunk_count(ctx: i32, ck: i32) -> i32 {
    let cs = att_chunk_size(ck).max(1);
    let sc = att_small_cells(ck);
    let small = ctx.min(sc);
    let rest = (ctx - small).max(0);
    (small + 127) / 128 + (rest + cs - 1) / cs
}

#[inline]
pub(crate) fn hip_alloc_zeroed<T>(count: usize, allocs: &mut Vec<*mut c_void>) -> Result<*mut T> {
    let bytes = (count * std::mem::size_of::<T>()).max(16);
    let p = hip_malloc_bytes(bytes)?;
    hip_check(
        unsafe { hipMemset(p.cast::<c_void>(), 0, bytes) },
        "hipMemset",
    )?;
    allocs.push(p.cast::<c_void>());
    Ok(p.cast::<T>())
}

#[inline]
pub(crate) fn hip_host_alloc_zeroed<T>(
    count: usize,
    hallocs: &mut Vec<*mut c_void>,
) -> Result<*mut T> {
    let bytes = (count * std::mem::size_of::<T>()).max(16);
    let mut ptr: *mut c_void = ptr::null_mut();
    hip_check(
        unsafe { hipHostMalloc(&mut ptr, bytes, HIP_HOST_MALLOC_DEFAULT) },
        "hipHostMalloc",
    )?;
    unsafe {
        ptr::write_bytes(ptr.cast::<u8>(), 0, bytes);
    }
    hallocs.push(ptr);
    Ok(ptr.cast::<T>())
}

/// Qwen3.5 / Qwen3.6 GPU execution engine (`Q36Engine` / `Engine`).
pub struct Q36Engine {
    pub m: *const Q36Model,
    pub tk: KernelsContext,
    pub opts: Q36Options,

    pub ctx_len: i32,
    pub max_seq: i32,
    pub vt_ctx: i32,
    pub chunk_sz: i32,
    pub att_ck: i32,
    pub n_chunk_w: i32,
    pub attn_w8: bool,
    pub v16: bool,
    pub gemm_t: i32,
    pub max_batch: i32,
    pub n_past: i32,
    pub cur_T: i32,
    pub gdn_valid: bool,
    pub use_bf16_kv: bool,
    pub graph_capturing: bool,
    pub fast_argmax: bool,

    pub s_main: hipStream_t,
    pub s_verify: hipStream_t,

    pub attn_slot: Vec<i32>,
    pub gdn_slot: Vec<i32>,
    pub n_attn: i32,
    pub n_gdn: i32,
    pub n_attn_state: i32,

    // KV caches & GDN recurrent states
    pub kc: Vec<*mut u16>,
    pub vc: Vec<*mut u16>,
    pub vtc: Vec<*mut u16>,
    pub k_cache: Vec<*mut c_void>,
    pub v_cache: Vec<*mut c_void>,
    pub conv: Vec<*mut f32>,
    pub S: Vec<*mut f32>,
    pub conv_state: Vec<*mut f32>,
    pub gdn_state: Vec<*mut f32>,

    // Pending verify GDN state buffers (`GdnPend`)
    pub v_qkv: Vec<*mut f32>,
    pub v_hc: Vec<*mut f32>,
    pub v_alpha: Vec<*mut f32>,
    pub v_beta: Vec<*mut f32>,
    pub pq: Vec<[*mut f32; 2]>,
    pub ph: Vec<[*mut f32; 2]>,
    pub pa: Vec<[*mut f32; 2]>,
    pub pb: Vec<[*mut f32; 2]>,
    pub gctl: *mut i32,
    pub pend_on: bool,

    // Scratch & activation buffers
    pub x: *mut f32,
    pub h: *mut f32,
    pub qkv: *mut f32,
    pub hc: *mut f32,
    pub z: *mut f32,
    pub y: *mut f32,
    pub alpha: *mut f32,
    pub beta: *mut f32,

    pub qg: *mut f32,
    pub q: *mut f32,
    pub k: *mut f32,
    pub v: *mut f32,
    pub gate: *mut f32,
    pub att: *mut f32,
    pub pacc: *mut f32,
    pub pm: *mut f32,
    pub pl: *mut f32,
    pub att_s: *mut f32,
    pub att_p16: *mut u16,
    pub fattn_mask: *mut u16,
    pub score_floats: usize,

    pub rlog: *mut f32,
    pub ids: *mut i32,
    pub wts: *mut f32,
    pub parts: *mut f32,
    pub sh_g: *mut f32,
    pub sh_u: *mut f32,
    pub sh_y: *mut f32,
    pub sgate: *mut f32,
    pub x_q8: *mut c_void,
    pub y_q8: *mut c_void,
    pub x_f16: *mut u16,
    pub flush_buf: *mut f32,
    pub q8_src: *const f32,
    pub q8_n: i32,
    pub q8_tok: i32,

    pub grp_ptr: *mut u64,
    pub grp_start: *mut i32,
    pub n_groups: *mut i32,
    pub ent_dst: *mut i32,
    pub ent_tok: *mut i32,
    pub moe_scr: *mut c_void,

    pub mmq_xq: *mut c_void,
    pub mmq_gu: *mut f32,
    pub mmq_h: *mut f32,
    pub mmq_hq: *mut c_void,
    pub mmq_cnt: *mut i32,
    pub mmq_cur: *mut i32,
    pub mmq_bounds: *mut i32,
    pub mmq_max: *mut i32,
    pub mmq_dst: *mut i32,
    pub mmq_src: *mut i32,
    pub mmq_inv: *mut i32,
    pub mmq_ident: *mut i32,

    // Dense MMQ & Split-K buffers (`q36_engine.cu:650, 708, 744-750`)
    pub dq_x: *mut c_void,
    /// rb-q27: the down projection's q8_1 activations written by the fused up + SwiGLU GEMM (dq_x holds its input).
    pub dq_y: *mut c_void,
    pub dq_ident: *mut i32,
    pub dq_bounds: *mut i32,
    pub dq_bounds_t: i32,
    pub sk_part: *mut f32,
    pub sk_part_elems: usize,

    // Talus prefill: expanded int8 trunk cache, budget guard, GDN [alpha; beta] rows, tp_fa / tp_fuse
    pub budget: talus_mem::Budget,
    pub i8c: I8Cache,
    /// opt-in W4A4 FFN prompt GEMMs (`TALUS_Q27_W4I`), empty by default
    pub w4i: super::w4i::W4iCache,
    pub ab_w: Vec<*mut f32>,
    pub ab_tmp: *mut f32,
    pub tp_fa: bool,
    pub tp_fuse: bool,
    /// rb-q27: the prompt's MTP pass runs only what the MTP layer's K / V cache rows need (TALUS_MTP_KV_ONLY=0: the
    /// full layer).  `kv_only_pass` is set for the duration of that one forward_mtp_batch call.
    pub mtp_kv_only: bool,
    pub kv_only_pass: bool,
    pub pf_x: *const f32,
    pub pf_n: i32,
    pub pf_t: i32,
    pub pf_q_layout: i32,
    pub cur_p0_h: i32,

    pub slot_of: Vec<*mut i32>,
    pub slot_host: Vec<Vec<i32>>,
    pub lru_tick: Vec<Vec<u64>>,
    pub slot_exp: Vec<Vec<i32>>,
    pub tick: u64,

    pub ids_h: *mut i32,
    pub cpu_parts: *mut f32,
    pub cpu_parts_d: *mut f32,
    pub miss_idx: *mut i32,
    pub miss_idx_d: *mut i32,
    pub x_h: *mut f32,

    // Logits & speculative decoding buffers
    pub logits: *mut f32,
    pub logits_h: *mut f32,
    pub logits_Pinned: *mut f32,
    pub logits_pinned_len: usize,

    pub tok_d: *mut i32,
    pub pos0_d: *mut i32,
    pub tok_m_d: *mut i32,
    pub pos0_m_d: *mut i32,

    // Aliases matching prompt specification
    pub d_pos: *mut i32,
    pub d_tok_in: *mut i32,
    pub d_verify_toks: *mut i32,
    pub d_accept_idx: *mut i32,
    pub d_draft_out: *mut i32,
    pub d_mtp_tok: *mut i32,

    pub h_last: *mut f32,
    pub h_mtp: *mut f32,
    pub h_mtp_prev: *mut f32,
    pub h_mtp_b: *mut f32,
    pub mtp_en: *mut f32,
    pub mtp_cat: *mut f32,

    pub logits_v: *mut f32,
    pub logits_v_h: *mut f32,
    pub logits_m: *mut f32,
    pub logits_m_h: *mut f32,
    pub pick_v: *mut i32,
    pub pick_v_h: *mut i32,
    pub pick_m: *mut i32,
    pub pick_m_h: *mut i32,
    pub arg_pv: *mut f32,
    pub arg_pi: *mut i32,
    pub rctl: *mut i32,
    pub rctl_h: *mut i32,
    pub ds_d: *mut tk_q36_draft_samp,
    pub ds_h: *mut tk_q36_draft_samp,
    pub lse_v: *mut f32,
    pub lse_v_h: *mut f32,
    pub thr_v: *mut f32,
    pub draft_map: *mut i32,
    pub draft_n_vocab: i32,

    pub dl: DraftLen,
    pub df: DFlash,
    pub dctx: DfCtx,
    pub dp: DrafterPick,
    pub use_dflash: bool,
    pub mtp_d: i32,
    pub dev_rounds: bool,
    pub win_t: i32,
    pub win_chain: i32,
    pub last_mtp: bool,
    pub dfl_win: bool,
    pub dfl_sample: bool,
    pub draft_samp_on: bool,
    pub draft_nucleus: bool,
    pub gdn_pending: i32,
    pub fused: bool,
    pub logits_md: *mut f32,
    pub lse_md: *mut f32,
    pub conf_md: *mut f32,
    pub hconf: *mut f32,
    pub hpin: *mut i32,
    pub dbg: *mut f32,
    pub dbg_row: i32,
    pub dv_inv: Vec<i32>,
    pub ng_kmin: i32,
    pub ng_kmax: i32,
    /// the long-span copy proposer (talus_spec::CopyProposer, idea from gufo, MIT), tried when the n-gram
    /// lookup has nothing (matches of 12-23 tokens, or 5+ where the last accepted copy stopped).  Opt-in
    /// (TALUS_Q27_COPY=1; the n-gram lookup already copies long repeats).
    pub copy: Option<talus_spec::CopyProposer>,

    pub prof: bool,
    pub prof_dt: [f64; 10],
    pub prof_n: i64,
    pub st_decide_ms: f64,
    pub st_rounds: i64,

    // Captured HIP graphs
    pub g1: hipGraphExec_t,
    pub gv: [hipGraphExec_t; 17],
    pub gm1: hipGraphExec_t,
    pub gA: [hipGraphExec_t; 17],
    pub gB: [[hipGraphExec_t; 16]; 17],
    pub gBi: [[hipGraphExec_t; 16]; 17],
    pub gB0: [hipGraphExec_t; 17],
    pub gB0i: [hipGraphExec_t; 17],
    pub gC: [hipGraphExec_t; 17],
    pub gS: [hipGraphExec_t; 16],
    pub gD: [hipGraphExec_t; 17],
    pub gDm: [hipGraphExec_t; 17],
    pub gD0: [hipGraphExec_t; 17],
    pub gD0m: [hipGraphExec_t; 17],
    pub gDs: hipGraphExec_t,

    pub(crate) allocs: Vec<*mut c_void>,
    pub(crate) hallocs: Vec<*mut c_void>,
}

unsafe impl Send for Q36Engine {}
unsafe impl Sync for Q36Engine {}

pub type Engine = Q36Engine;

impl Drop for Q36Engine {
    fn drop(&mut self) {
        self.destroy_graphs();
        self.dl.free_buffers();
        self.df.free_buffers();
        if !self.s_main.is_null() {
            unsafe {
                let _ = hipStreamDestroy(self.s_main);
            }
            self.s_main = ptr::null_mut();
        }
        if !self.s_verify.is_null() {
            unsafe {
                let _ = hipStreamDestroy(self.s_verify);
            }
            self.s_verify = ptr::null_mut();
        }
        for &p in &self.allocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipFree(p);
                }
            }
        }
        self.allocs.clear();
        for &p in &self.hallocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipHostFree(p);
                }
            }
        }
        self.hallocs.clear();
    }
}

impl Q36Engine {
    /// Construct and initialize a `Q36Engine` for `model` with maximum context `ctx_len`.
    pub fn new(model: &Q36Model, ctx_len: i32) -> Result<Self> {
        let tk = KernelsContext::from_env()?;
        let mut eng = Self {
            m: model as *const Q36Model,
            tk,
            opts: model.opts.clone(),
            ctx_len: 0,
            max_seq: 0,
            vt_ctx: 0,
            chunk_sz: K_CHUNK_CELLS,
            att_ck: K_ATT_CHUNK | ((K_ATT_SMALL_CELLS / K_ATT_SMALL) << 16),
            n_chunk_w: 0,
            attn_w8: model.opts.attn_w8,
            v16: !model.opts.no_v16,
            gemm_t: if model.dense && !model.opts.no_v16 {
                16
            } else {
                8
            },
            max_batch: 512,
            n_past: 0,
            cur_T: 0,
            gdn_valid: false,
            use_bf16_kv: model.opts.use_bf16_kv,
            graph_capturing: false,
            fast_argmax: true,
            s_main: ptr::null_mut(),
            s_verify: ptr::null_mut(),
            attn_slot: Vec::new(),
            gdn_slot: Vec::new(),
            n_attn: 0,
            n_gdn: 0,
            n_attn_state: 0,
            kc: Vec::new(),
            vc: Vec::new(),
            vtc: Vec::new(),
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            conv: Vec::new(),
            S: Vec::new(),
            conv_state: Vec::new(),
            gdn_state: Vec::new(),
            v_qkv: Vec::new(),
            v_hc: Vec::new(),
            v_alpha: Vec::new(),
            v_beta: Vec::new(),
            pq: Vec::new(),
            ph: Vec::new(),
            pa: Vec::new(),
            pb: Vec::new(),
            gctl: ptr::null_mut(),
            pend_on: false,
            x: ptr::null_mut(),
            h: ptr::null_mut(),
            qkv: ptr::null_mut(),
            hc: ptr::null_mut(),
            z: ptr::null_mut(),
            y: ptr::null_mut(),
            alpha: ptr::null_mut(),
            beta: ptr::null_mut(),
            qg: ptr::null_mut(),
            q: ptr::null_mut(),
            k: ptr::null_mut(),
            v: ptr::null_mut(),
            gate: ptr::null_mut(),
            att: ptr::null_mut(),
            pacc: ptr::null_mut(),
            pm: ptr::null_mut(),
            pl: ptr::null_mut(),
            att_s: ptr::null_mut(),
            att_p16: ptr::null_mut(),
            fattn_mask: ptr::null_mut(),
            score_floats: 0,
            rlog: ptr::null_mut(),
            ids: ptr::null_mut(),
            wts: ptr::null_mut(),
            parts: ptr::null_mut(),
            sh_g: ptr::null_mut(),
            sh_u: ptr::null_mut(),
            sh_y: ptr::null_mut(),
            sgate: ptr::null_mut(),
            x_q8: ptr::null_mut(),
            y_q8: ptr::null_mut(),
            x_f16: ptr::null_mut(),
            flush_buf: ptr::null_mut(),
            q8_src: ptr::null(),
            q8_n: 0,
            q8_tok: 0,
            grp_ptr: ptr::null_mut(),
            grp_start: ptr::null_mut(),
            n_groups: ptr::null_mut(),
            ent_dst: ptr::null_mut(),
            ent_tok: ptr::null_mut(),
            moe_scr: ptr::null_mut(),
            mmq_xq: ptr::null_mut(),
            mmq_gu: ptr::null_mut(),
            mmq_h: ptr::null_mut(),
            mmq_hq: ptr::null_mut(),
            mmq_cnt: ptr::null_mut(),
            mmq_cur: ptr::null_mut(),
            mmq_bounds: ptr::null_mut(),
            mmq_max: ptr::null_mut(),
            mmq_dst: ptr::null_mut(),
            mmq_src: ptr::null_mut(),
            mmq_inv: ptr::null_mut(),
            mmq_ident: ptr::null_mut(),
            dq_x: ptr::null_mut(),
            dq_y: ptr::null_mut(),
            dq_ident: ptr::null_mut(),
            dq_bounds: ptr::null_mut(),
            dq_bounds_t: 0,
            sk_part: ptr::null_mut(),
            sk_part_elems: 0,
            budget: talus_mem::Budget::strix_default(),
            i8c: I8Cache::default(),
            w4i: super::w4i::W4iCache::default(),
            ab_w: Vec::new(),
            ab_tmp: ptr::null_mut(),
            tp_fa: false,
            tp_fuse: true,
            mtp_kv_only: true,
            kv_only_pass: false,
            pf_x: ptr::null(),
            pf_n: 0,
            pf_t: 0,
            pf_q_layout: -1,
            cur_p0_h: -1,
            slot_of: Vec::new(),
            slot_host: Vec::new(),
            lru_tick: Vec::new(),
            slot_exp: Vec::new(),
            tick: 0,
            ids_h: ptr::null_mut(),
            cpu_parts: ptr::null_mut(),
            cpu_parts_d: ptr::null_mut(),
            miss_idx: ptr::null_mut(),
            miss_idx_d: ptr::null_mut(),
            x_h: ptr::null_mut(),
            logits: ptr::null_mut(),
            logits_h: ptr::null_mut(),
            logits_Pinned: ptr::null_mut(),
            logits_pinned_len: 0,
            tok_d: ptr::null_mut(),
            pos0_d: ptr::null_mut(),
            tok_m_d: ptr::null_mut(),
            pos0_m_d: ptr::null_mut(),
            d_pos: ptr::null_mut(),
            d_tok_in: ptr::null_mut(),
            d_verify_toks: ptr::null_mut(),
            d_accept_idx: ptr::null_mut(),
            d_draft_out: ptr::null_mut(),
            d_mtp_tok: ptr::null_mut(),
            h_last: ptr::null_mut(),
            h_mtp: ptr::null_mut(),
            h_mtp_prev: ptr::null_mut(),
            h_mtp_b: ptr::null_mut(),
            mtp_en: ptr::null_mut(),
            mtp_cat: ptr::null_mut(),
            logits_v: ptr::null_mut(),
            logits_v_h: ptr::null_mut(),
            logits_m: ptr::null_mut(),
            logits_m_h: ptr::null_mut(),
            pick_v: ptr::null_mut(),
            pick_v_h: ptr::null_mut(),
            pick_m: ptr::null_mut(),
            pick_m_h: ptr::null_mut(),
            arg_pv: ptr::null_mut(),
            arg_pi: ptr::null_mut(),
            rctl: ptr::null_mut(),
            rctl_h: ptr::null_mut(),
            ds_d: ptr::null_mut(),
            ds_h: ptr::null_mut(),
            lse_v: ptr::null_mut(),
            lse_v_h: ptr::null_mut(),
            thr_v: ptr::null_mut(),
            draft_map: ptr::null_mut(),
            draft_n_vocab: 0,
            dl: DraftLen::default(),
            df: DFlash::default(),
            dctx: DfCtx::default(),
            dp: DrafterPick::default(),
            use_dflash: false,
            mtp_d: 0,
            dev_rounds: true,
            win_t: 0,
            win_chain: 0,
            last_mtp: true,
            dfl_win: false,
            dfl_sample: false,
            draft_samp_on: false,
            draft_nucleus: false,
            gdn_pending: 0,
            fused: model.dense,
            logits_md: ptr::null_mut(),
            lse_md: ptr::null_mut(),
            conf_md: ptr::null_mut(),
            hconf: ptr::null_mut(),
            hpin: ptr::null_mut(),
            dbg: ptr::null_mut(),
            dbg_row: 0,
            dv_inv: Vec::new(),
            ng_kmin: 2,
            ng_kmax: 4,
            copy: (talus_spec::copy_enabled()
                && std::env::var("TALUS_Q27_COPY")
                    .map(|v| v == "1")
                    .unwrap_or(false))
            .then(|| talus_spec::CopyProposer::new(talus_spec::copy_cfg_from_env())),
            prof: false,
            prof_dt: [0.0; 10],
            prof_n: 0,
            st_decide_ms: 0.0,
            st_rounds: 0,
            g1: ptr::null_mut(),
            gv: [ptr::null_mut(); 17],
            gm1: ptr::null_mut(),
            gA: [ptr::null_mut(); 17],
            gB: [[ptr::null_mut(); 16]; 17],
            gBi: [[ptr::null_mut(); 16]; 17],
            gB0: [ptr::null_mut(); 17],
            gB0i: [ptr::null_mut(); 17],
            gC: [ptr::null_mut(); 17],
            gS: [ptr::null_mut(); 16],
            gD: [ptr::null_mut(); 17],
            gDm: [ptr::null_mut(); 17],
            gD0: [ptr::null_mut(); 17],
            gD0m: [ptr::null_mut(); 17],
            gDs: ptr::null_mut(),
            allocs: Vec::new(),
            hallocs: Vec::new(),
        };
        eng.init(model, ctx_len)?;
        Ok(eng)
    }

    #[inline]
    pub fn model(&self) -> &Q36Model {
        unsafe { &*self.m }
    }

    #[inline]
    pub fn n_vocab(&self) -> i32 {
        self.model().n_vocab
    }

    #[inline]
    pub fn d_model(&self) -> i32 {
        self.model().d_model
    }

    #[inline]
    pub fn n_Layer(&self) -> i32 {
        self.model().n_Layer
    }

    #[inline]
    pub fn n_kv(&self) -> i32 {
        self.model().n_kv
    }

    #[inline]
    pub fn d_head(&self) -> i32 {
        self.model().d_head
    }

    #[inline]
    pub fn max_seq(&self) -> i32 {
        self.max_seq
    }

    #[inline]
    pub fn use_bf16_kv(&self) -> bool {
        self.use_bf16_kv
    }

    /// Initialize all scratch, state, KV, MoE, prefill, and speculative buffers (`Engine::init` in `q36_engine.cu:815-999`).
    pub fn init(&mut self, model: &Q36Model, ctx_len: i32) -> Result<()> {
        self.m = model as *const Q36Model;
        self.opts = model.opts.clone();
        self.ctx_len = ctx_len;
        self.max_seq = ctx_len;
        self.use_bf16_kv = model.opts.use_bf16_kv;
        self.prof = std::env::var("STRATA_Q36_PROF")
            .map(|v| v == "1")
            .unwrap_or(false);

        let g = &model.g;
        let B = self.max_batch as usize;

        if self.s_main.is_null() {
            hip_check(
                unsafe {
                    hip_sys::hipStreamCreateWithFlags(
                        &mut self.s_main,
                        hip_sys::hipStreamNonBlocking,
                    )
                },
                "hipStreamCreateWithFlags(s_main)",
            )?;
        }

        if let Ok(w) = std::env::var("STRATA_MMVQ_W32") {
            let _ = unsafe { tk_mmvq_set_w32(self.tk.as_ptr(), w != "0") };
        } else {
            let _ = unsafe { tk_mmvq_set_w32(self.tk.as_ptr(), self.opts.use_w32) };
        }
        if let Ok(wv) = std::env::var("STRATA_MMVQ_WAVES") {
            if let Ok(n) = wv.parse::<i32>() {
                let _ = unsafe { tk_mmvq_set_w32_waves(self.tk.as_ptr(), n) };
            }
        }

        let max_qkv = g
            .ssm_qkv_dim()
            .max(g.q_gate_dim() + 2 * g.kv_dim())
            .max(2 * g.n_embd);
        let max_w = if model.dense {
            (g.n_ff * g.n_embd).max(max_qkv * g.n_embd)
        } else {
            max_qkv * g.n_embd
        };
        if !self.opts.no_hipblas {
            let _ = tk_q36_prefill_init_scratch(self.tk.as_ptr(), max_w);
        }

        self.attn_slot.assign(g.n_layer as usize, -1);
        self.gdn_slot.assign(g.n_layer as usize, -1);
        self.n_attn = 0;
        self.n_gdn = 0;
        for i in 0..(g.n_layer as usize) {
            if model.layers[i].is_attn {
                self.attn_slot[i] = self.n_attn;
                self.n_attn += 1;
            } else {
                self.gdn_slot[i] = self.n_gdn;
                self.n_gdn += 1;
            }
        }

        self.n_attn_state = self.n_attn + if model.has_mtp { 1 } else { 0 };
        self.v16 = !model.opts.no_v16
            && std::env::var("STRATA_Q36_VERIFY16")
                .map(|v| v != "0")
                .unwrap_or(true);
        self.gemm_t = if model.dense && self.v16 { 16 } else { 8 };
        self.attn_w8 = model.opts.attn_w8
            || std::env::var("STRATA_ATTN_W8")
                .map(|v| v == "1")
                .unwrap_or(false);

        let mut att_chunk = K_ATT_CHUNK;
        for var in ["TALUS_ATT_CHUNK", "STRATA_ATTN_CHUNK"] {
            if let Ok(v) = std::env::var(var) {
                if let Ok(n) = v.parse::<i32>() {
                    att_chunk = (n / 32 * 32).max(32);
                    break;
                }
            }
        }
        let mut small = K_ATT_SMALL_CELLS;
        for var in ["TALUS_ATT_SMALL", "STRATA_ATTN_SMALL"] {
            if let Ok(v) = std::env::var(var) {
                if let Ok(n) = v.parse::<i32>() {
                    small = n.clamp(0, 0x7fff * K_ATT_SMALL);
                    break;
                }
            }
        }
        self.att_ck = att_chunk | ((small / K_ATT_SMALL) << 16);
        self.vt_ctx = att_vt_cells(ctx_len, self.att_ck);
        self.n_chunk_w = att_chunk_count(ctx_len, self.att_ck);

        let kv_elems = (ctx_len as usize) * (g.kv_dim() as usize) + 256 * (g.head_dim as usize);
        let vt_elems = (self.vt_ctx as usize) * (g.kv_dim() as usize);

        self.kc.resize(self.n_attn_state as usize, ptr::null_mut());
        self.vc.resize(self.n_attn_state as usize, ptr::null_mut());
        self.vtc.resize(self.n_attn_state as usize, ptr::null_mut());
        self.k_cache
            .resize(self.n_attn_state as usize, ptr::null_mut());
        self.v_cache
            .resize(self.n_attn_state as usize, ptr::null_mut());
        for i in 0..(self.n_attn_state as usize) {
            self.kc[i] = hip_alloc_zeroed::<u16>(kv_elems, &mut self.allocs)?;
            self.vc[i] = hip_alloc_zeroed::<u16>(kv_elems, &mut self.allocs)?;
            self.vtc[i] = hip_alloc_zeroed::<u16>(vt_elems, &mut self.allocs)?;
            self.k_cache[i] = self.kc[i].cast::<c_void>();
            self.v_cache[i] = self.vc[i].cast::<c_void>();
        }

        let C = g.ssm_qkv_dim() as usize;
        let HV = g.ssm_time_step_rank as usize;
        let S_dim = g.ssm_state_size as usize;

        self.conv.resize(self.n_gdn as usize, ptr::null_mut());
        self.S.resize(self.n_gdn as usize, ptr::null_mut());
        self.v_qkv.resize(self.n_gdn as usize, ptr::null_mut());
        self.v_hc.resize(self.n_gdn as usize, ptr::null_mut());
        self.v_alpha.resize(self.n_gdn as usize, ptr::null_mut());
        self.v_beta.resize(self.n_gdn as usize, ptr::null_mut());
        self.pq
            .resize(self.n_gdn as usize, [ptr::null_mut(), ptr::null_mut()]);
        self.ph
            .resize(self.n_gdn as usize, [ptr::null_mut(), ptr::null_mut()]);
        self.pa
            .resize(self.n_gdn as usize, [ptr::null_mut(), ptr::null_mut()]);
        self.pb
            .resize(self.n_gdn as usize, [ptr::null_mut(), ptr::null_mut()]);
        for i in 0..(self.n_gdn as usize) {
            self.conv[i] = hip_alloc_zeroed::<f32>(3 * C, &mut self.allocs)?;
            self.S[i] = hip_alloc_zeroed::<f32>(HV * S_dim * S_dim, &mut self.allocs)?;
            self.v_qkv[i] = hip_alloc_zeroed::<f32>(16 * C, &mut self.allocs)?;
            self.v_hc[i] = hip_alloc_zeroed::<f32>(16 * C, &mut self.allocs)?;
            self.v_alpha[i] = hip_alloc_zeroed::<f32>(16 * HV, &mut self.allocs)?;
            self.v_beta[i] = hip_alloc_zeroed::<f32>(16 * HV, &mut self.allocs)?;
            for b in 0..2 {
                self.pq[i][b] = hip_alloc_zeroed::<f32>(16 * C, &mut self.allocs)?;
                self.ph[i][b] = hip_alloc_zeroed::<f32>(16 * C, &mut self.allocs)?;
                self.pa[i][b] = hip_alloc_zeroed::<f32>(16 * HV, &mut self.allocs)?;
                self.pb[i][b] = hip_alloc_zeroed::<f32>(16 * HV, &mut self.allocs)?;
            }
        }
        self.conv_state = self.conv.clone();
        self.gdn_state = self.S.clone();
        self.gctl = hip_alloc_zeroed::<i32>(4, &mut self.allocs)?;

        let E = g.n_embd as usize;
        let QG = g.q_gate_dim() as usize;
        let Q_dim = g.q_dim() as usize;
        let KV = g.kv_dim() as usize;
        let H = g.n_head as usize;
        let HD = g.head_dim as usize;
        let V = g.ssm_inner_size as usize;

        self.x = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.h = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.qkv = hip_alloc_zeroed::<f32>(B * C.max(QG + 2 * KV), &mut self.allocs)?;
        self.hc = hip_alloc_zeroed::<f32>(B * C, &mut self.allocs)?;
        self.z = hip_alloc_zeroed::<f32>(B * V.max(Q_dim), &mut self.allocs)?;
        self.y = hip_alloc_zeroed::<f32>(B * V.max(E), &mut self.allocs)?;
        self.alpha = hip_alloc_zeroed::<f32>(B * 2 * HV, &mut self.allocs)?;
        self.beta = hip_alloc_zeroed::<f32>(B * HV, &mut self.allocs)?;

        self.qg = hip_alloc_zeroed::<f32>(B * QG, &mut self.allocs)?;
        self.q = hip_alloc_zeroed::<f32>(B * Q_dim, &mut self.allocs)?;
        self.k = hip_alloc_zeroed::<f32>(B * KV, &mut self.allocs)?;
        self.v = hip_alloc_zeroed::<f32>(B * KV, &mut self.allocs)?;
        self.gate = hip_alloc_zeroed::<f32>(B * Q_dim, &mut self.allocs)?;
        self.att = hip_alloc_zeroed::<f32>(B * Q_dim, &mut self.allocs)?;

        let max_chunks = ((ctx_len + self.chunk_sz - 1) / self.chunk_sz)
            .max(self.n_chunk_w)
            .max(16) as usize;
        let p_rows = 32usize;
        self.pacc = hip_alloc_zeroed::<f32>(p_rows * H * max_chunks * HD, &mut self.allocs)?;
        self.pm = hip_alloc_zeroed::<f32>(p_rows * H * max_chunks, &mut self.allocs)?;
        self.pl = hip_alloc_zeroed::<f32>(p_rows * H * max_chunks, &mut self.allocs)?;
        let ldp_max = ((ctx_len as usize) + 7) / 8 * 8;
        self.score_floats = (16 * H * ldp_max).max(64 * 1024 * 1024);
        self.att_s = hip_alloc_zeroed::<f32>(self.score_floats, &mut self.allocs)?;
        self.att_p16 = hip_alloc_zeroed::<u16>(self.score_floats, &mut self.allocs)?;
        let mask_b = tk_q36_fattn_mask_bytes(B as i64, self.vt_ctx as i64);
        self.fattn_mask = hip_alloc_zeroed::<u16>((mask_b / 2).max(1), &mut self.allocs)?;

        let max_in = E
            .max(V)
            .max(Q_dim)
            .max(g.n_ff as usize)
            .max(g.n_ff_exp as usize)
            .max(g.n_ff_shexp as usize);
        let q8_bytes = unsafe { tk_q8_1_bytes(max_in as i32, B as i32) };
        self.x_q8 = hip_alloc_zeroed::<u8>(q8_bytes.max(16), &mut self.allocs)?.cast::<c_void>();
        self.y_q8 = hip_alloc_zeroed::<u8>(q8_bytes.max(16), &mut self.allocs)?.cast::<c_void>();
        self.x_f16 = hip_alloc_zeroed::<u16>(B * max_in, &mut self.allocs)?;
        self.flush_buf = hip_alloc_zeroed::<f32>(16, &mut self.allocs)?;

        let max_f32_out = (g.n_expert as usize + 16).max(2 * HV).max(64);
        self.sk_part_elems = 16 * B * max_f32_out;
        self.sk_part = hip_alloc_zeroed::<f32>(self.sk_part_elems, &mut self.allocs)?;

        // Talus prefill: each GDN layer's alpha and beta rows in one [2 * HV, E] matrix
        self.ab_tmp = hip_alloc_zeroed::<f32>(B * 2 * HV, &mut self.allocs)?;
        self.ab_w.resize(g.n_layer as usize, ptr::null_mut());
        for li in 0..(g.n_layer as usize) {
            let L = &model.layers[li];
            if !L.is_attn && !L.ssm_alpha_ptr.is_null() && !L.ssm_beta_ptr.is_null() {
                let n = HV * E;
                let w = hip_alloc_zeroed::<f32>(2 * n, &mut self.allocs)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            w.cast::<c_void>(),
                            L.ssm_alpha_ptr.cast::<c_void>(),
                            n * 4,
                            HIP_MEMCPY_DEVICE_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(ab_w alpha)",
                )?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            w.add(n).cast::<c_void>(),
                            L.ssm_beta_ptr.cast::<c_void>(),
                            n * 4,
                            HIP_MEMCPY_DEVICE_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(ab_w beta)",
                )?;
                self.ab_w[li] = w;
            }
        }

        // TALUS_PF_FA=1 is the default on the Rust 27B host; TALUS_PF_FA=0 restores the legacy path.
        self.tp_fa = self.opts.use_fattn
            && talus_pf_fa_prefill_ok(HD as i32, H as i32, g.n_head_kv as i32)
            && std::env::var("TALUS_PF_FA")
                .map(|v| v != "0")
                .unwrap_or(true);
        self.tp_fuse = std::env::var("TALUS_PF_FUSE")
            .map(|v| v != "0")
            .unwrap_or(true);
        self.mtp_kv_only = std::env::var("TALUS_MTP_KV_ONLY")
            .map(|v| v != "0")
            .unwrap_or(true);

        if model.dense {
            let F = g.n_ff as usize;
            self.sh_g = hip_alloc_zeroed::<f32>(B * F, &mut self.allocs)?;
            self.sh_u = hip_alloc_zeroed::<f32>(B * F, &mut self.allocs)?;
            self.sh_y = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
            let use_dense_mmq = tk_q36_mmq_built()
                && std::env::var("STRATA_DENSE_MMQ")
                    .map(|v| v != "0")
                    .unwrap_or(true);
            if use_dense_mmq {
                let dq_bytes = tk_q36_mmq_q8_bytes(B as i64, max_in as i64)
                    .max(B * (E.max(F) / 2))
                    .max(16);
                self.dq_x = hip_alloc_zeroed::<u8>(dq_bytes, &mut self.allocs)?.cast::<c_void>();
                self.dq_y = hip_alloc_zeroed::<u8>(dq_bytes, &mut self.allocs)?.cast::<c_void>();
                self.dq_ident = hip_alloc_zeroed::<i32>(B, &mut self.allocs)?;
                self.dq_bounds = hip_alloc_zeroed::<i32>(2, &mut self.allocs)?;
                self.dq_bounds_t = 0;
                let _ = unsafe { hipDeviceSynchronize() };
                let _ = tk_q36_mmq_iota(
                    self.tk.as_ptr(),
                    self.dq_ident,
                    B as i64,
                    self.s_main.cast::<c_void>(),
                );
                self.tp_levers_init(model)?;
                self.w4i_init(model);
            }
        } else {
            let K_used = g.n_expert_used as usize;
            let NE = g.n_expert as usize;
            let F = g.n_ff_exp as usize;
            let FS = g.n_ff_shexp as usize;
            let max_ent = B * K_used;

            self.rlog = hip_alloc_zeroed::<f32>(B * (NE + 1), &mut self.allocs)?;
            self.ids = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
            self.wts = hip_alloc_zeroed::<f32>(max_ent, &mut self.allocs)?;
            self.parts = hip_alloc_zeroed::<f32>(max_ent * E, &mut self.allocs)?;
            self.sh_g = hip_alloc_zeroed::<f32>(B * FS, &mut self.allocs)?;
            self.sh_u = hip_alloc_zeroed::<f32>(B * FS, &mut self.allocs)?;
            self.sh_y = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
            self.sgate = hip_alloc_zeroed::<f32>(B * 16, &mut self.allocs)?;

            let cap_groups = NE.min(max_ent).max(1);
            self.grp_ptr = hip_alloc_zeroed::<u64>(cap_groups + 1, &mut self.allocs)?;
            self.grp_start = hip_alloc_zeroed::<i32>(cap_groups + 1, &mut self.allocs)?;
            self.n_groups = hip_alloc_zeroed::<i32>(1, &mut self.allocs)?;
            self.ent_dst = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
            self.ent_tok = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
            let scr_b = tk_q36_expert_scratch_bytes(max_ent as i64, F as i64);
            self.moe_scr =
                hip_alloc_zeroed::<u8>(scr_b.max(16), &mut self.allocs)?.cast::<c_void>();

            if tk_q36_mmq_built() {
                let xq_b = tk_q36_mmq_q8_bytes(max_ent as i64, E as i64);
                let hq_b = tk_q36_mmq_q8_bytes(max_ent as i64, F as i64);
                self.mmq_xq =
                    hip_alloc_zeroed::<u8>(xq_b.max(16), &mut self.allocs)?.cast::<c_void>();
                self.mmq_gu = hip_alloc_zeroed::<f32>(max_ent * 2 * F, &mut self.allocs)?;
                self.mmq_h = hip_alloc_zeroed::<f32>(max_ent * F, &mut self.allocs)?;
                self.mmq_hq =
                    hip_alloc_zeroed::<u8>(hq_b.max(16), &mut self.allocs)?.cast::<c_void>();
                self.mmq_cnt = hip_alloc_zeroed::<i32>(NE, &mut self.allocs)?;
                self.mmq_cur = hip_alloc_zeroed::<i32>(NE, &mut self.allocs)?;
                self.mmq_bounds = hip_alloc_zeroed::<i32>(NE + 1, &mut self.allocs)?;
                self.mmq_max = hip_alloc_zeroed::<i32>(1, &mut self.allocs)?;
                self.mmq_dst = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
                self.mmq_src = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
                self.mmq_inv = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
                self.mmq_ident = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
                let _ = unsafe { hipDeviceSynchronize() };
                let _ = tk_q36_mmq_iota(
                    self.tk.as_ptr(),
                    self.mmq_ident,
                    max_ent as i64,
                    self.s_main.cast::<c_void>(),
                );
            }

            if model.hybrid {
                let nl = g.n_layer as usize;
                self.slot_of.resize(nl, ptr::null_mut());
                self.slot_host.resize(nl, Vec::new());
                self.lru_tick.resize(nl, Vec::new());
                self.slot_exp.resize(nl, Vec::new());
                for li in 0..nl {
                    let ns = model.layers[li].n_slots as usize;
                    self.slot_host[li].assign(NE, -1);
                    self.lru_tick[li].assign(ns, 0);
                    self.slot_exp[li].assign(ns, -1);
                    for s in 0..ns {
                        self.slot_host[li][s] = s as i32;
                        self.slot_exp[li][s] = s as i32;
                    }
                    self.slot_of[li] = hip_alloc_zeroed::<i32>(NE, &mut self.allocs)?;
                    hip_check(
                        unsafe {
                            hipMemcpy(
                                self.slot_of[li].cast::<c_void>(),
                                self.slot_host[li].as_ptr().cast::<c_void>(),
                                NE * 4,
                                HIP_MEMCPY_HOST_TO_DEVICE,
                            )
                        },
                        "hipMemcpy(slot_of)",
                    )?;
                }
                self.ids_h = hip_host_alloc_zeroed::<i32>(max_ent, &mut self.hallocs)?;
                self.cpu_parts = hip_host_alloc_zeroed::<f32>(max_ent * E, &mut self.hallocs)?;
                self.cpu_parts_d = hip_alloc_zeroed::<f32>(max_ent * E, &mut self.allocs)?;
                self.miss_idx = hip_host_alloc_zeroed::<i32>(max_ent, &mut self.hallocs)?;
                self.miss_idx_d = hip_alloc_zeroed::<i32>(max_ent, &mut self.allocs)?;
                self.x_h = hip_host_alloc_zeroed::<f32>(B * E, &mut self.hallocs)?;
            }
        }

        let n_vocab = g.n_vocab as usize;
        let max_spec = 16usize;
        self.logits = hip_alloc_zeroed::<f32>(max_spec * n_vocab, &mut self.allocs)?;
        self.logits_h = hip_host_alloc_zeroed::<f32>(n_vocab, &mut self.hallocs)?;
        self.tok_d = hip_alloc_zeroed::<i32>(B, &mut self.allocs)?;
        self.pos0_d = hip_alloc_zeroed::<i32>(1, &mut self.allocs)?;
        self.tok_m_d = hip_alloc_zeroed::<i32>(B, &mut self.allocs)?;
        self.pos0_m_d = hip_alloc_zeroed::<i32>(1, &mut self.allocs)?;

        self.h_last = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.h_mtp = hip_alloc_zeroed::<f32>(E, &mut self.allocs)?;
        self.h_mtp_prev = hip_alloc_zeroed::<f32>(E, &mut self.allocs)?;
        self.h_mtp_b = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.mtp_en = hip_alloc_zeroed::<f32>(B * E, &mut self.allocs)?;
        self.mtp_cat = hip_alloc_zeroed::<f32>(B * 2 * E, &mut self.allocs)?;

        self.logits_v = self.logits;
        self.logits_v_h = hip_host_alloc_zeroed::<f32>(max_spec * n_vocab, &mut self.hallocs)?;
        self.logits_Pinned = self.logits_v_h;
        self.logits_pinned_len = max_spec * n_vocab;

        self.draft_n_vocab = if !model.dv_head.is_null() {
            model.dv_head.n_out
        } else if !model.mtp_head.is_null() {
            model.mtp_head.n_out
        } else {
            g.n_vocab as i32
        };
        let vd = (self.draft_n_vocab.max(1) as usize).max(n_vocab);
        self.logits_md = hip_alloc_zeroed::<f32>(max_spec * vd, &mut self.allocs)?;

        self.logits_m = hip_alloc_zeroed::<f32>(n_vocab, &mut self.allocs)?;
        self.logits_m_h = hip_host_alloc_zeroed::<f32>(max_spec * vd, &mut self.hallocs)?;
        self.pick_v = hip_alloc_zeroed::<i32>(max_spec, &mut self.allocs)?;
        self.pick_v_h = hip_host_alloc_zeroed::<i32>(max_spec, &mut self.hallocs)?;
        self.pick_m = hip_alloc_zeroed::<i32>(max_spec, &mut self.allocs)?;
        self.pick_m_h = hip_host_alloc_zeroed::<i32>(max_spec, &mut self.hallocs)?;
        self.arg_pv = hip_alloc_zeroed::<f32>(max_spec * 256, &mut self.allocs)?;
        self.arg_pi = hip_alloc_zeroed::<i32>(max_spec * 256, &mut self.allocs)?;
        self.rctl = hip_alloc_zeroed::<i32>(8, &mut self.allocs)?;
        self.rctl_h = hip_host_alloc_zeroed::<i32>(8, &mut self.hallocs)?;
        self.ds_d = hip_alloc_zeroed::<tk_q36_draft_samp>(1, &mut self.allocs)?;
        self.ds_h = hip_host_alloc_zeroed::<tk_q36_draft_samp>(1, &mut self.hallocs)?;
        self.lse_v = hip_alloc_zeroed::<f32>(max_spec, &mut self.allocs)?;
        self.lse_v_h = hip_host_alloc_zeroed::<f32>(2 * max_spec, &mut self.hallocs)?;
        self.thr_v = hip_alloc_zeroed::<f32>(max_spec, &mut self.allocs)?;
        self.lse_md = hip_alloc_zeroed::<f32>(2 * max_spec, &mut self.allocs)?;
        self.conf_md = hip_alloc_zeroed::<f32>(2 * max_spec, &mut self.allocs)?;
        self.hconf = hip_host_alloc_zeroed::<f32>(2 * max_spec, &mut self.hallocs)?;
        self.hpin = hip_host_alloc_zeroed::<i32>(128, &mut self.hallocs)?;

        // Sync prompt-alias pointers
        self.d_pos = self.pos0_d;
        self.d_tok_in = self.tok_d;
        self.d_verify_toks = self.tok_d;
        self.d_accept_idx = self.pick_v;
        self.d_draft_out = self.pick_m;
        self.d_mtp_tok = self.tok_m_d;

        self.dv_inv.clear();
        if !model.dv_ids.is_empty() {
            self.draft_map = hip_alloc_zeroed::<i32>(model.dv_ids.len(), &mut self.allocs)?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        self.draft_map.cast::<c_void>(),
                        model.dv_ids.as_ptr().cast::<c_void>(),
                        model.dv_ids.len() * 4,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(draft_map)",
            )?;
            self.dv_inv.resize(n_vocab, -1);
            for (i, &id) in model.dv_ids.iter().enumerate() {
                if id >= 0 && (id as usize) < n_vocab {
                    self.dv_inv[id as usize] = i as i32;
                }
            }
        }

        self.fused = model.dense
            && std::env::var("STRATA_Q36_NO_FUSED_COMMIT").is_err()
            && std::env::var("STRATA_FUSED")
                .map(|v| v != "0")
                .unwrap_or(true);
        self.dev_rounds = self.fused
            && self.opts.use_graph
            && std::env::var("STRATA_NO_DEV_ROUNDS")
                .map(|v| v == "0")
                .unwrap_or(true);
        self.draft_nucleus = std::env::var("STRATA_DRAFT_NUCLEUS")
            .map(|v| v != "0")
            .unwrap_or(false);
        self.ng_kmin = std::env::var("STRATA_NGRAM_MIN")
            .or_else(|_| std::env::var("STRATA_NG_KMIN"))
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .map(|v| v.clamp(1, 15))
            .unwrap_or(4);
        self.ng_kmax = std::env::var("STRATA_NGRAM_MAX")
            .or_else(|_| std::env::var("STRATA_NG_KMAX"))
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .map(|v| v.clamp(self.ng_kmin, 15))
            .unwrap_or(15.max(self.ng_kmin));

        self.dl.init(&self.tk)?;
        self.dl.init_arch("gfx1151", self.mtp_d);
        let df_path = if !self.opts.dflash_path.is_empty() {
            Some(self.opts.dflash_path.clone())
        } else {
            std::env::var("STRATA_DFLASH_MODEL")
                .ok()
                .filter(|s| !s.is_empty())
        };
        if let Some(df_path) = df_path {
            let df_head = if !model.dv_head.is_null() {
                &model.dv_head
            } else {
                &model.head
            };
            self.use_dflash = self.df.init(
                &self.tk,
                &df_path,
                g.n_layer as i32,
                g.n_embd as i32,
                self.draft_n_vocab,
                df_head,
                self.draft_map,
                self.max_batch,
            )?;
            if self.use_dflash {
                self.df.configure_target_head(&model.head, &model.dv_ids)?;
                self.dctx
                    .init(self.df.window(), self.df.ring(), self.df.block());
                self.dp.init("gfx1151", model.has_mtp && self.mtp_d > 0, "");
                if self.mtp_d == 0 {
                    self.mtp_d = self.df.n_draft();
                }
                self.dl.init_arch("gfx1151", 0);
                self.dfl_sample = self.dl.dfl_sample();
            }
        }
        let _ = unsafe { hipDeviceSynchronize() };
        Ok(())
    }

    /// Reset recurrent state and cache position (`Engine::reset` in `q36_engine.cu:883-894`).
    pub fn reset(&mut self) {
        let g = &self.model().g;
        let C = g.ssm_qkv_dim() as usize;
        let HV = g.ssm_time_step_rank as usize;
        let S_dim = g.ssm_state_size as usize;
        for i in 0..(self.n_gdn as usize) {
            unsafe {
                let _ = hipMemsetAsync(self.conv[i].cast::<c_void>(), 0, 3 * C * 4, self.s_main);
                let _ = hipMemsetAsync(
                    self.S[i].cast::<c_void>(),
                    0,
                    HV * S_dim * S_dim * 4,
                    self.s_main,
                );
            }
        }
        unsafe {
            let _ = hipMemsetAsync(
                self.h_mtp.cast::<c_void>(),
                0,
                (g.n_embd as usize) * 4,
                self.s_main,
            );
            if !self.gctl.is_null() {
                let _ = hipMemsetAsync(self.gctl.cast::<c_void>(), 0, 8, self.s_main);
            }
            let _ = hipStreamSynchronize(self.s_main);
        }
        self.pend_on = false;
        self.gdn_pending = 0;
        self.n_past = 0;
        self.gdn_valid = false;
        self.df.reset();
        if self.use_dflash && self.df.ready {
            self.dctx
                .init(self.df.window(), self.df.ring(), self.df.block());
        }
    }

    /// Flush any pending GDN state updates (`Engine::flush_gdn` in `q36_engine.cu:1305-1318`).
    pub fn flush_gdn(&mut self) {
        if self.opts.no_gdn_flush || (!self.fused && self.gdn_pending <= 0) {
            return;
        }
        if self.fused {
            if self.gdn_pending <= 0 {
                return;
            }
            let m = self.model();
            let g = &m.g;
            let C = g.ssm_qkv_dim() as i32;
            let HK = g.ssm_k_heads() as i32;
            let HV = g.ssm_dt_rank() as i32;
            let eps = g.rms_norm_eps;
            let stream = self.s_main.cast::<c_void>();
            for li in 0..(g.n_layer as usize) {
                let L = &m.layers[li];
                if L.is_attn {
                    continue;
                }
                let s = self.gdn_slot[li] as usize;
                let pd = tk_q36_gdn_pend {
                    q: self.pq[s],
                    h: self.ph[s],
                    a: self.pa[s],
                    b: self.pb[s],
                    ctl: self.gctl,
                    store: false,
                };
                let _ = tk_q36_conv_seq(
                    self.tk.as_ptr(),
                    self.conv[s],
                    ptr::null(),
                    L.ssm_conv_ptr,
                    self.hc,
                    C,
                    2 * HK,
                    eps,
                    0,
                    true,
                    &pd as *const _,
                    stream,
                );
                let _ = tk_q36_gdn_seq(
                    self.tk.as_ptr(),
                    self.S[s],
                    self.hc,
                    C,
                    ptr::null(),
                    ptr::null(),
                    self.z,
                    L.ssm_norm_ptr,
                    eps,
                    ptr::null_mut(),
                    HK,
                    HV,
                    0,
                    true,
                    &pd as *const _,
                    ptr::null_mut(),
                    stream,
                );
            }
            let _ = tk_q36_ctl_end(self.tk.as_ptr(), self.gctl, false, stream);
            self.gdn_pending = 0;
            self.gdn_valid = true;
            return;
        }
        if self.gdn_pending > 0 {
            self.commit(self.gdn_pending);
            self.gdn_pending = 0;
        }
    }

    /// Replay `n` accepted tokens through all GDN layers (`Engine::commit` in `q36_engine.cu:1485-1500`).
    pub fn commit(&mut self, n: i32) {
        if n <= 0 {
            return;
        }
        let m = self.model();
        let g = &m.g;
        let C = g.ssm_qkv_dim() as i32;
        let HK = g.ssm_k_heads() as i32;
        let HV = g.ssm_dt_rank() as i32;
        let eps = g.rms_norm_eps;
        let stream = self.s_main.cast::<c_void>();
        for li in 0..(g.n_layer as usize) {
            let L = &m.layers[li];
            if L.is_attn {
                continue;
            }
            let s = self.gdn_slot[li] as usize;
            let _ = tk_q36_conv_seq(
                self.tk.as_ptr(),
                self.conv[s],
                self.v_qkv[s],
                L.ssm_conv_ptr,
                self.v_hc[s],
                C,
                2 * HK,
                eps,
                n,
                true,
                ptr::null(),
                stream,
            );
            let _ = tk_q36_gdn_seq(
                self.tk.as_ptr(),
                self.S[s],
                self.v_hc[s],
                C,
                self.v_alpha[s],
                self.v_beta[s],
                ptr::null(),
                ptr::null(),
                eps,
                ptr::null_mut(),
                HK,
                HV,
                n,
                true,
                ptr::null(),
                ptr::null_mut(),
                stream,
            );
        }
        self.gdn_valid = true;
    }

    #[inline]
    pub fn invalidate_pf(&mut self) {
        self.pf_x = ptr::null();
        self.pf_n = 0;
        self.pf_t = 0;
        self.pf_q_layout = -1;
    }

    #[inline]
    pub fn prep(&mut self, rows: i32) {
        self.q8_src = ptr::null();
        self.q8_n = 0;
        self.q8_tok = 0;
        self.invalidate_pf();
        if self.cur_T != rows {
            self.cur_T = rows;
        }
    }

    /// Expand dense trunk projections into exact INT8 cache (`Engine::tp_levers_init` in `q36_engine.cu:950-1002`),
    /// reserving the ~26 GiB expansion in `talus_mem::Budget` before allocating.
    pub fn tp_levers_init(&mut self, model: &Q36Model) -> Result<()> {
        self.i8c.clear();
        let i8_on = std::env::var("TALUS_PF_I8")
            .map(|v| v != "0")
            .unwrap_or(true);
        if !i8_on {
            return Ok(());
        }

        type I8Candidate = (usize, *const u8, i32, i32, i32, i32, usize, usize);
        let mut candidates: Vec<I8Candidate> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut total_bytes = 0usize;
        for L in &model.layers {
            let mats = [
                &L.wqkv,
                &L.wgate,
                &L.ssm_out_mat,
                &L.wq_mat,
                &L.wk_mat,
                &L.wv_mat,
                &L.wo_mat,
                &L.ffn_gate_mat,
                &L.ffn_up_mat,
                &L.ffn_down_mat,
            ];
            for w in mats {
                if w.is_null() {
                    continue;
                }
                let key = w.d as usize;
                if !seen.insert(key) {
                    continue;
                }
                let raw_qt = w.raw_qtype();
                let mode = talus_pf_i8x_mode(raw_qt);
                if let Some((qb, wb)) = talus_pf_i8x_bytes(w.n_out, w.n_in, mode) {
                    total_bytes += qb + wb;
                    candidates.push((key, w.d, raw_qt, w.n_out, w.n_in, mode, qb, wb));
                }
            }
        }
        if candidates.is_empty() || total_bytes == 0 {
            return Ok(());
        }

        // Account for tp_i8x's ~26 GiB in talus-mem::Budget before expanding.
        let guard = match self.budget.try_reserve_guard(
            talus_mem::BudgetCategory::Weights,
            "tp_i8x_expanded_trunk",
            total_bytes,
        ) {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "strata-q36: skipping tp_i8x expansion ({:.2} GiB): budget check refused ({e})",
                    (total_bytes as f64) / ((1usize << 30) as f64)
                );
                return Ok(());
            }
        };

        let stream = self.s_main.cast::<c_void>();
        for (key, w_ptr, raw_qt, n_out, n_in, mode, qb, wb) in candidates {
            let qs = match hip_malloc_bytes(qb) {
                Ok(p) => p.cast::<i8>(),
                Err(_) => break,
            };
            let ws = match hip_malloc_bytes(wb) {
                Ok(p) => p.cast::<f32>(),
                Err(_) => {
                    unsafe {
                        let _ = hipFree(qs.cast::<c_void>());
                    }
                    break;
                }
            };
            if !talus_pf_i8x_expand(w_ptr.cast::<c_void>(), raw_qt, n_out, n_in, qs, ws, stream) {
                unsafe {
                    let _ = hipFree(qs.cast::<c_void>());
                    let _ = hipFree(ws.cast::<c_void>());
                }
                continue;
            }
            self.i8c.bytes += qb + wb;
            self.i8c.mats.insert(
                key,
                I8Mat {
                    qs,
                    ws,
                    n: n_out,
                    k: n_in,
                    mode,
                    bytes: qb + wb,
                },
            );
        }
        let _ = unsafe { hipStreamSynchronize(self.s_main) };
        let _ = self.budget.mark_materialized(guard.id());
        self.i8c.budget_guard = Some(guard);
        eprintln!(
            "strata-q36: exact int8 prompt GEMM on {} expanded matrices, {:.2} GiB (TALUS_PF_I8=0: MMQ)",
            self.i8c.len(),
            (self.i8c.bytes as f64) / ((1usize << 30) as f64)
        );
        Ok(())
    }

    #[inline]
    pub fn mark_q8(&mut self, p: *const f32, n: i32, n_tok: i32) {
        self.q8_src = p;
        self.q8_n = n;
        self.q8_tok = n_tok;
    }

    #[inline]
    pub fn quant(&mut self, x: *const f32, n_in: i32, n_tok: i32) {
        if self.q8_src == x && self.q8_n == n_in && self.q8_tok == n_tok {
            return;
        }
        let stream = self.s_main.cast::<c_void>();
        let _ = unsafe { tk_quantize_q8_1(self.tk.as_ptr(), x, self.x_q8, n_in, n_tok, stream) };
        self.mark_q8(x, n_in, n_tok);
    }

    /// Matrix-matrix / matrix-vector multiply with optional output leading dimension `ldy` (`Engine::mm`).
    pub fn mm(&mut self, w: &Mat, x: *const f32, y: *mut f32, n_tok: i32, ldy: i32) {
        if n_tok <= 0 || w.is_null() {
            return;
        }
        let stream = self.s_main.cast::<c_void>();
        let raw_qt = w.raw_qtype();

        if raw_qt == GGML_TYPE_F32 as i32 {
            self.f32mm(
                w.d.cast::<f32>(),
                x,
                y,
                w.n_in,
                w.n_out,
                n_tok,
                if ldy > 0 { ldy } else { w.n_out },
            );
            return;
        }

        if n_tok > self.gemm_t
            && (ldy <= 0 || ldy == w.n_out)
            && !self.dq_x.is_null()
            && tk_q36_mmq_supported_fits(raw_qt, w.n_out as i64)
            && (w.n_in % 256) == 0
        {
            if self.dense_mmq(w, x, y, n_tok) {
                return;
            }
        }

        if n_tok > self.gemm_t && !self.opts.no_hipblas {
            if raw_qt == GGML_TYPE_BF16 as i32 {
                // Convert x to BF16 in x_f16 or use F16 path
                let _ = tk_q36_to_f16(
                    self.tk.as_ptr(),
                    x,
                    self.x_f16.cast::<c_void>(),
                    (n_tok as i64) * (w.n_in as i64),
                    stream,
                );
            } else {
                let _ = tk_q36_to_f16(
                    self.tk.as_ptr(),
                    x,
                    self.x_f16.cast::<c_void>(),
                    (n_tok as i64) * (w.n_in as i64),
                    stream,
                );
            }
            let st = unsafe {
                tk_prefill_gemm(
                    self.tk.as_ptr(),
                    w.tk_qtype(),
                    w.d.cast::<c_void>(),
                    self.x_f16.cast::<c_void>(),
                    tk_qtype::TK_QTYPE_F16,
                    y,
                    n_tok as i64,
                    w.n_out as i64,
                    w.n_in as i64,
                    ldy as i64,
                    0,
                    0.0,
                    stream,
                )
            };
            if st == TK_OK {
                return;
            }
        }

        if ldy > 0 && ldy != w.n_out {
            for t in 0..n_tok {
                self.apply(
                    w,
                    unsafe { x.add((t as usize) * (w.n_in as usize)) },
                    unsafe { y.add((t as usize) * (ldy as usize)) },
                    1,
                );
            }
            return;
        }

        if tk_q36_mmvq_supported(raw_qt) {
            self.quant(x, w.n_in, n_tok);
            let col_bytes = unsafe { tk_q8_1_bytes(w.n_in, 1) };
            let cmax = if unsafe { tk_mmvq_w32_active() } != 0 {
                16
            } else {
                8
            };
            let mut t0 = 0i32;
            while t0 < n_tok {
                let chunk = (n_tok - t0).min(cmax);
                let chunk_x = unsafe {
                    self.x_q8
                        .cast::<u8>()
                        .add((t0 as usize) * col_bytes)
                        .cast::<c_void>()
                };
                let chunk_y = unsafe { y.add((t0 as usize) * (w.n_out as usize)) };
                let _ = unsafe {
                    tk_mmvq(
                        self.tk.as_ptr(),
                        w.tk_qtype(),
                        w.d.cast::<c_void>(),
                        chunk_x,
                        chunk_y,
                        w.n_in,
                        w.n_out,
                        chunk,
                        stream,
                    )
                };
                t0 += chunk;
            }
        }
    }

    /// FP32 matrix multiply (`Engine::f32mm` in `q36_engine.cu:902-923`).
    pub fn f32mm(
        &mut self,
        w: *const f32,
        x: *const f32,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        n_tok: i32,
        ldy: i32,
    ) {
        if n_tok <= 0 || w.is_null() {
            return;
        }
        let ldo = if ldy > 0 { ldy } else { n_out };
        let stream = self.s_main.cast::<c_void>();
        if n_tok <= self.gemm_t && ldo == n_out {
            let _ = tk_q36_gemv_f32(self.tk.as_ptr(), n_tok, w, x, y, n_in, n_out, stream);
        } else if (n_in & 3) == 0 {
            let tm = if n_out > 64 { 4 } else { 1 };
            let gx = (n_out + 63) / 64;
            let gy = (n_tok + 16 * tm - 1) / (16 * tm);
            let want = (128 / (gx * gy).max(1)).clamp(1, 16);
            let mut kz = ((n_in / want + 15) / 16 * 16).max(16);
            let mut splits = (n_in + kz - 1) / kz;
            if ldo != n_out
                || self.sk_part.is_null()
                || (splits as usize) * (n_tok as usize) * (n_out as usize) > self.sk_part_elems
            {
                splits = 1;
                kz = n_in;
            }
            if splits > 1 {
                let _ = tk_q36_sgemm_nt(
                    self.tk.as_ptr(),
                    tm,
                    gx,
                    gy,
                    splits,
                    w,
                    x,
                    self.sk_part,
                    n_in,
                    n_out,
                    n_tok,
                    n_out,
                    kz,
                    stream,
                );
                let _ = tk_q36_sum_splits(
                    self.tk.as_ptr(),
                    self.sk_part,
                    y,
                    (n_tok as i64) * (n_out as i64),
                    splits,
                    stream,
                );
            } else {
                let _ = tk_q36_sgemm_nt(
                    self.tk.as_ptr(),
                    tm,
                    gx,
                    gy,
                    1,
                    w,
                    x,
                    y,
                    n_in,
                    n_out,
                    n_tok,
                    ldo,
                    n_in,
                    stream,
                );
            }
        } else {
            for t in 0..n_tok {
                let _ = tk_q36_gemv_f32(
                    self.tk.as_ptr(),
                    1,
                    w,
                    unsafe { x.add((t as usize) * (n_in as usize)) },
                    unsafe { y.add((t as usize) * (ldo as usize)) },
                    n_in,
                    n_out,
                    stream,
                );
            }
        }
    }

    #[inline]
    pub fn apply(&mut self, w: &Mat, x: *const f32, y: *mut f32, n_tok: i32) {
        self.mm(w, x, y, n_tok, 0);
    }

    /// Multi-matrix MMVQ launch sharing quantized `x_q8` (`Engine::apply_multi` in `q36_engine.cu:999-1018`).
    pub fn apply_multi(&mut self, ws: &[&Mat], ys: &[*mut f32], x: *const f32, n_tok: i32) {
        let n_mats = ws.len().min(ys.len());
        if n_tok <= 0 || n_mats == 0 {
            return;
        }
        if n_tok > self.gemm_t || unsafe { tk_mmvq_w32_active() } == 0 {
            for m in 0..n_mats {
                self.apply(ws[m], x, ys[m], n_tok);
            }
            return;
        }
        let stream = self.s_main.cast::<c_void>();
        let mut i = 0usize;
        while i < n_mats {
            let qt = ws[i].raw_qtype();
            let n_in = ws[i].n_in;
            let mut j = i + 1;
            while j < n_mats && (j - i) < 3 && ws[j].raw_qtype() == qt && ws[j].n_in == n_in {
                j += 1;
            }
            if (j - i) >= 2 && tk_q36_mmvq_supported(qt) {
                self.quant(x, n_in, n_tok);
                let mut w_ptrs: [*const c_void; 3] = [ptr::null(); 3];
                let mut y_ptrs: [*mut f32; 3] = [ptr::null_mut(); 3];
                let mut n_outs: [i32; 3] = [0; 3];
                for k in i..j {
                    w_ptrs[k - i] = ws[k].d.cast::<c_void>();
                    y_ptrs[k - i] = ys[k];
                    n_outs[k - i] = ws[k].n_out;
                }
                let st_multi = unsafe {
                    tk_mmvq_w32_multi(
                        self.tk.as_ptr(),
                        ws[i].tk_qtype(),
                        (j - i) as i32,
                        w_ptrs.as_ptr(),
                        y_ptrs.as_ptr(),
                        n_outs.as_ptr(),
                        self.x_q8,
                        n_in,
                        n_tok,
                        stream,
                    )
                };
                if st_multi == TK_OK {
                    i = j;
                    continue;
                }
            }
            for k in i..j {
                self.apply(ws[k], x, ys[k], n_tok);
            }
            i = j;
        }
    }

    /// Quantize activation `x` into `dq_x` and update `dq_bounds` for `dense_mmq` (`q36_engine.cu:972-979`).
    pub fn dense_mmq_quant(&mut self, x: *const f32, qtype: i32, n_in: i32, n_tok: i32) -> bool {
        if self.dq_x.is_null() || !tk_q36_mmq_built() || (n_in % 256) != 0 {
            return false;
        }
        let lay = tk_q36_mmq_q8_layout(qtype);
        if !x.is_null()
            && self.pf_x == x
            && self.pf_n == n_in
            && self.pf_t == n_tok
            && self.pf_q_layout == lay
        {
            return true;
        }
        let stream = self.s_main.cast::<c_void>();
        if tk_q36_mmq_quantize(
            self.tk.as_ptr(),
            x,
            ptr::null(),
            self.dq_x,
            qtype,
            n_in as i64,
            n_in as i64,
            n_tok as i64,
            stream,
        ) != TK_OK
        {
            return false;
        }
        self.pf_x = x;
        self.pf_n = n_in;
        self.pf_t = n_tok;
        self.pf_q_layout = lay;
        if self.dq_bounds_t != n_tok {
            let _ = tk_q36_mmq_set_bounds(self.tk.as_ptr(), self.dq_bounds, n_tok, stream);
            self.dq_bounds_t = n_tok;
        }
        true
    }

    /// load the opt-in W4A4 sidecar (`TALUS_Q27_W4I=<file.w4i>`, planes `TALUS_Q27_W4I_PLANES`) for the FFN
    /// matrices of the selected layers.  Any failure leaves the cache empty (the exact paths run).
    pub fn w4i_init(&mut self, model: &Q36Model) {
        self.w4i.clear();
        let env = std::env::var("TALUS_Q27_W4I").ok();
        let Some((path, def_planes, mode)) =
            super::w4i::resolve(env.as_deref(), model.file_path.as_deref())
        else {
            if env.is_some() {
                eprintln!(
                    "strata-q36: W4A4 prompt GEMMs off (TALUS_Q27_W4I={}): exact int8 prompt path",
                    env.unwrap_or_default()
                );
            }
            return;
        };
        let path = path.to_string_lossy().into_owned();
        if !std::path::Path::new(&path).exists() {
            eprintln!(
                "strata-q36: W4A4 sidecar {path} not found ({mode}): prompts stay on the exact int8 path \
                 (place the matching .w4i next to the GGUF, or TALUS_Q27_W4I=0 to silence this)"
            );
            return;
        }
        if !talus_pf_w4i_available() {
            eprintln!("strata-q36: W4A4 sidecar {path} found but libtalus_kernels has no tp_w4i: prompts stay exact");
            return;
        }
        let planes =
            std::env::var("TALUS_Q27_W4I_PLANES").unwrap_or_else(|_| def_planes.to_string());
        let mut f = match std::fs::File::open(&path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("strata-q36: W4A4 sidecar {path}: {e}: prompts stay exact");
                return;
            }
        };
        let (idx, src) = match super::w4i::read_index(&mut f) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("strata-q36: W4A4 sidecar {path}: {e}: prompts stay exact");
                return;
            }
        };
        if let (Some(src), Some(g)) = (&src, model.file_path.as_deref()) {
            let gsize = std::fs::metadata(g).map(|m| m.len()).unwrap_or(0);
            if src.size != 0 && src.size != gsize {
                eprintln!(
                    "strata-q36: W4A4 sidecar {path} was built for {} ({} bytes), not {} ({gsize} bytes): prompts stay exact",
                    src.name,
                    src.size,
                    g.display()
                );
                return;
            }
        }
        let by_name: HashMap<&str, &super::w4i::W4iEntry> =
            idx.iter().map(|e| (e.name.as_str(), e)).collect();
        let mut want: Vec<(usize, &super::w4i::W4iEntry, i32, i32)> = Vec::new();
        let mut total = 0usize;
        for (li, L) in model.layers.iter().enumerate() {
            for (plane, nm, w) in [
                (0usize, "ffn_gate", &L.ffn_gate_mat),
                (0, "ffn_up", &L.ffn_up_mat),
                (1, "ffn_down", &L.ffn_down_mat),
            ] {
                if w.is_null() || !super::w4i::plane_selected(&planes, li, plane) {
                    continue;
                }
                let name = format!("blk.{li}.{nm}.weight");
                let Some(e) = by_name.get(name.as_str()) else {
                    continue;
                };
                if e.n as i32 != w.n_out || e.k as i32 != w.n_in || e.n % 128 != 0 || e.k % 256 != 0
                {
                    eprintln!("strata-q36: TALUS_Q27_W4I: {name} shape mismatch, skipped");
                    continue;
                }
                let wsb = if e.wg == 0 {
                    e.n as usize * 4
                } else {
                    e.n as usize * (e.k / e.wg) as usize * 4
                };
                total += e.n as usize * e.k as usize / 2 + wsb + e.k as usize;
                want.push((w.d as usize, *e, e.wg as i32, wsb as i32));
            }
        }
        if want.is_empty() {
            eprintln!(
                "strata-q36: TALUS_Q27_W4I: no selected matrix in {path}: prompts stay exact"
            );
            return;
        }
        let rows = self.max_batch.max(1);
        let kmax = want.iter().map(|w| w.1.k as i32).max().unwrap_or(0);
        let act = rows as usize * kmax as usize / 2
            + rows as usize * (kmax as usize / 32)
            + rows as usize * 4;
        let guard = match self.budget.try_reserve_guard(
            talus_mem::BudgetCategory::Weights,
            "tp_w4i_sidecar",
            total + act,
        ) {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "strata-q36: TALUS_Q27_W4I: budget check refused ({e}): prompts stay exact"
                );
                return;
            }
        };
        let mut host = Vec::new();
        let mut ok = true;
        for (key, e, wg, wsb) in &want {
            let n = e.n as usize;
            let k = e.k as usize;
            let wq = super::w4i::upload(&mut f, e.off_wq, n * k / 2, &mut host);
            let ws = super::w4i::upload(&mut f, e.off_ws, *wsb as usize, &mut host);
            let sg = super::w4i::upload(&mut f, e.off_signs, k, &mut host);
            match (wq, ws, sg) {
                (Ok(wq), Ok(ws), Ok(sg)) => {
                    self.w4i.mats.insert(
                        *key,
                        super::w4i::W4iMat {
                            wq,
                            ws: ws.cast::<f32>(),
                            signs: sg.cast::<i8>(),
                            n: e.n as i32,
                            k: e.k as i32,
                            wg: *wg,
                        },
                    );
                    self.w4i.bytes += n * k / 2 + *wsb as usize + k;
                }
                (a, b, c) => {
                    for p in [a.ok(), b.ok(), c.ok()].into_iter().flatten() {
                        unsafe {
                            let _ = hipFree(p.cast::<c_void>());
                        }
                    }
                    ok = false;
                    break;
                }
            }
        }
        let xq = hip_malloc_bytes(rows as usize * kmax as usize / 2);
        let xk = hip_malloc_bytes(rows as usize * (kmax as usize / 32));
        let xs = hip_malloc_bytes(rows as usize * 4);
        match (ok, xq, xk, xs) {
            (true, Ok(a), Ok(b), Ok(c)) => {
                self.w4i.xq = a;
                self.w4i.xk = b;
                self.w4i.xs = c.cast::<f32>();
            }
            (_, a, b, c) => {
                for p in [a.ok(), b.ok(), c.ok()].into_iter().flatten() {
                    unsafe {
                        let _ = hipFree(p.cast::<c_void>());
                    }
                }
                self.w4i.clear();
                eprintln!(
                    "strata-q36: TALUS_Q27_W4I: upload / allocation failed: prompts stay exact"
                );
                return;
            }
        }
        self.w4i.cap_rows = rows;
        self.w4i.cap_k = kmax;
        self.w4i.min_rows = std::env::var("TALUS_Q27_W4I_MIN")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(64)
            .max(1);
        let _ = self.budget.mark_materialized(guard.id());
        self.w4i.budget_guard = Some(guard);
        eprintln!(
            "strata-q36: W4A4 (tp_w4i, quality-gated, not bit-exact) FFN prompt GEMMs on {} matrices ({}, {mode}: {path}), {:.2} GiB sidecar, chunks of >= {} rows",
            self.w4i.mats.len(),
            planes,
            (self.w4i.bytes as f64) / ((1usize << 30) as f64),
            self.w4i.min_rows
        );
    }

    /// the dense FFN of one prompt chunk with W4A4 on the planes the sidecar selects (the rest exact).
    /// false (nothing launched) when no plane of this layer is selected.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_w4i(
        &mut self,
        gate: &Mat,
        up: &Mat,
        down: &Mat,
        out: *mut f32,
        n_tok: i32,
        e: i32,
        f: i32,
    ) -> bool {
        if self.w4i.mats.is_empty() || n_tok < self.w4i.min_rows || n_tok > self.w4i.cap_rows {
            return false;
        }
        let g4 = self.w4i.find(gate.d);
        let u4 = self.w4i.find(up.d);
        let d4 = self.w4i.find(down.d);
        if g4.is_none() && u4.is_none() && d4.is_none() {
            return false;
        }
        let stream = self.s_main.cast::<c_void>();
        let (xq, xk, xs) = (self.w4i.xq, self.w4i.xk, self.w4i.xs);
        let mut packed: Option<(*const i8, i32)> = None;
        for (w, w4, y) in [(gate, g4, self.sh_g), (up, u4, self.sh_u)] {
            match w4 {
                Some(m) => {
                    if packed != Some((m.signs as *const i8, m.wg)) {
                        if !talus_pf_w4i_pack(
                            self.h,
                            ptr::null(),
                            n_tok,
                            e,
                            e as i64,
                            m.signs,
                            m.wg,
                            xq,
                            xk,
                            xs,
                            stream,
                        ) {
                            return false;
                        }
                        packed = Some((m.signs as *const i8, m.wg));
                    }
                    if !talus_pf_w4i_gemm(
                        m.wq, m.ws, m.wg, m.n, m.k, xq, xk, xs, n_tok, y, m.n as i64, stream,
                    ) {
                        return false;
                    }
                }
                None => self.apply(w, self.h, y, n_tok),
            }
        }
        self.invalidate_pf();
        match d4 {
            Some(m) => {
                if !talus_pf_w4i_pack(
                    self.sh_u, self.sh_g, n_tok, f, f as i64, m.signs, m.wg, xq, xk, xs, stream,
                ) || !talus_pf_w4i_gemm(
                    m.wq, m.ws, m.wg, m.n, m.k, xq, xk, xs, n_tok, out, m.n as i64, stream,
                ) {
                    return false;
                }
            }
            None => {
                if !self.tp_down(down, self.sh_g, self.sh_u, out, n_tok, f) {
                    let _ = tk_q36_swiglu(
                        self.tk.as_ptr(),
                        self.sh_g,
                        self.sh_u,
                        self.sh_g,
                        (n_tok as i64) * (f as i64),
                        stream,
                    );
                    self.invalidate_pf();
                    if !self.dense_mmq(down, self.sh_g, out, n_tok) {
                        self.apply(down, self.sh_g, out, n_tok);
                    }
                }
            }
        }
        true
    }

    /// Run `talus_pf_i8gemm` (when expanded in `i8c`) or single-expert `tk_prefill_moe_mmq` product against pre-quantized `dq_x`.
    pub fn dense_mmq_gemm(&mut self, w: &Mat, y: *mut f32, n_tok: i32) -> bool {
        let raw_qt = w.raw_qtype();
        if self.dq_x.is_null()
            || !tk_q36_mmq_built()
            || !tk_q36_mmq_supported_fits(raw_qt, w.n_out as i64)
            || (w.n_in % 256) != 0
        {
            return false;
        }
        let stream = self.s_main.cast::<c_void>();
        if let Some(&x8) = self.i8c.find(w.d) {
            if talus_pf_i8gemm(
                x8.qs,
                x8.ws,
                x8.n,
                x8.k,
                x8.mode,
                self.dq_x,
                n_tok,
                y,
                w.n_out as i64,
                stream,
            ) {
                return true;
            }
        }
        if self.dq_bounds_t != n_tok {
            let _ = tk_q36_mmq_set_bounds(self.tk.as_ptr(), self.dq_bounds, n_tok, stream);
            self.dq_bounds_t = n_tok;
        }
        let p = tk_moe_mmq_product {
            w: w.d.cast::<c_void>(),
            qtype: w.tk_qtype(),
            w_rows: w.n_out as i64,
            w_cols: w.n_in as i64,
            expert_bytes: tk_q36_mmq_matrix_bytes(raw_qt, w.n_out as i64, w.n_in as i64),
            n_experts: 1,
            x_q8_1: self.dq_x,
            bounds_dev: self.dq_bounds,
            ids_dev: self.dq_ident,
            total_rows: n_tok as i64,
            max_rows: n_tok as i64,
            dst: y,
            ld_dst: w.n_out as i64,
            y_scale_dev: ptr::null(),
        };
        unsafe { tk_prefill_moe_mmq(self.tk.as_ptr(), &p, stream) == TK_OK }
    }

    /// Fused SwiGLU + Q8_1 MMQ quantize + down projection (`Engine::tp_down` in `q36_engine.cu:1005-1025`).
    pub fn tp_down(
        &mut self,
        w: &Mat,
        g: *const f32,
        u: *const f32,
        out: *mut f32,
        n_tok: i32,
        fd: i32,
    ) -> bool {
        let raw_qt = w.raw_qtype();
        if !self.tp_fuse
            || (fd % 256) != 0
            || self.dq_x.is_null()
            || !tk_q36_mmq_built()
            || w.n_in != fd
            || !tk_q36_mmq_supported_fits(raw_qt, w.n_out as i64)
        {
            return false;
        }
        let stream = self.s_main.cast::<c_void>();
        if !talus_pf_swiglu_q8_mmq(g, u, self.dq_x, raw_qt, fd as i64, n_tok as i64, stream) {
            return false;
        }
        self.invalidate_pf();
        self.dense_mmq_gemm(w, out, n_tok)
    }

    /// rb-q27: up projection + SwiGLU + q8_1 quantization in one GEMM (the up weights' expanded int8 matrix, epilogue
    /// reading the gate projection `g`), then the down projection on those q8_1 rows.  Bitwise `apply(up)` +
    /// `tp_down`: the up GEMM's per-element arithmetic is talus_pf_i8gemm's on the same quantized input, the epilogue
    /// is k_swiglu_q8's expressions in its order.  false (nothing to undo; the caller runs up + tp_down) when the
    /// shapes / types are not covered (or TALUS_PF_SWQ8=0).
    pub fn tp_gu_down(
        &mut self,
        up: &Mat,
        down: &Mat,
        g: *const f32,
        out: *mut f32,
        n_tok: i32,
        fd: i32,
    ) -> bool {
        let up_qt = up.raw_qtype();
        let down_qt = down.raw_qtype();
        if !self.tp_fuse
            || self.dq_x.is_null()
            || self.dq_y.is_null()
            || !tk_q36_mmq_built()
            || (fd % 512) != 0
            || up.n_out != fd
            || down.n_in != fd
            || !tk_q36_mmq_supported_fits(up_qt, up.n_out as i64)
            || !tk_q36_mmq_supported_fits(down_qt, down.n_out as i64)
            || (up.n_in % 256) != 0
        {
            return false;
        }
        let layout = tk_q36_mmq_q8_layout(down_qt);
        if layout != 0 && layout != 1 {
            return false;
        }
        let x8 = match self.i8c.find(up.d) {
            Some(&x8) => x8,
            None => return false,
        };
        // the up projection's input, quantized for the up weights' layout (as apply(up) -> dense_mmq does; cached when
        // the gate projection already left it in dq_x)
        if !self.dense_mmq_quant(self.h, up_qt, up.n_in, n_tok) {
            return false;
        }
        let stream = self.s_main.cast::<c_void>();
        if !talus_pf_i8gemm_swiglu_q8(
            x8.qs, x8.ws, x8.n, x8.k, x8.mode, self.dq_x, n_tok, g, fd as i64, self.dq_y, layout,
            stream,
        ) {
            return false;
        }
        // the down projection reads dq_y (dense_mmq_gemm takes its activations from dq_x)
        let keep = self.dq_x;
        self.dq_x = self.dq_y;
        let ok = self.dense_mmq_gemm(down, out, n_tok);
        self.dq_x = keep;
        self.invalidate_pf();
        ok
    }

    /// Dense MMQ prefill projection (`Engine::dense_mmq` in `q36_engine.cu:972-992`).
    pub fn dense_mmq(&mut self, w: &Mat, x: *const f32, y: *mut f32, n_tok: i32) -> bool {
        let raw_qt = w.raw_qtype();
        if self.dq_x.is_null()
            || !tk_q36_mmq_built()
            || !tk_q36_mmq_supported_fits(raw_qt, w.n_out as i64)
            || (w.n_in % 256) != 0
        {
            return false;
        }
        self.dense_mmq_quant(x, raw_qt, w.n_in, n_tok) && self.dense_mmq_gemm(w, y, n_tok)
    }

    #[inline]
    pub fn mmq(&mut self, w: &Mat, x: *const f32, y: *mut f32, n_tok: i32) -> bool {
        self.dense_mmq(w, x, y, n_tok)
    }

    /// Execute one transformer layer (`Engine::layer` in `q36_engine.cu:1002-1280`).
    pub fn layer(&mut self, li: i32, L: &LayerW, pos0: *const i32) {
        let m_ptr = self.m;
        let m = unsafe { &*m_ptr };
        let (wn, hout) = if li >= 0 {
            let last = ((li + 1) as usize) == m.layers.len();
            let wn = if last {
                m.out_norm
            } else {
                m.layers[(li + 1) as usize].attn_norm_ptr
            };
            let hout = if last { self.h_last } else { self.h };
            (wn, hout)
        } else {
            let wn = if !m.mtp_shnorm.is_null() {
                m.mtp_shnorm
            } else {
                m.out_norm
            };
            (wn, self.h_mtp_b)
        };
        let verify = self.pend_on;
        self.layer_into(li, L, pos0, verify, wn, hout);
    }

    /// Execute one transformer layer writing normed output (`wn`) into `hout` (`Engine::layer` in `q36_engine.cu:1002-1280`).
    pub fn layer_into(
        &mut self,
        li: i32,
        L: &LayerW,
        pos0: *const i32,
        verify: bool,
        wn: *const f32,
        hout: *mut f32,
    ) {
        let m_ptr = self.m;
        let m = unsafe { &*m_ptr };
        let g = &m.g;
        let T = self.cur_T;
        let E = g.n_embd as i32;
        let eps = g.rms_norm_eps;
        let stream = self.s_main.cast::<c_void>();
        self.invalidate_pf();

        let sync_t = |prof: bool| -> f64 {
            if !prof {
                return 0.0;
            }
            let _ = unsafe { hipDeviceSynchronize() };
            Instant::now().elapsed().as_secs_f64() * 1e6
        };
        let _ = sync_t;

        if L.is_attn {
            let _QG = g.q_gate_dim() as i32;
            let Q_dim = g.q_dim() as i32;
            let _KV = g.kv_dim() as i32;
            let H = g.n_head as i32;
            let HKV = g.n_head_kv as i32;
            let HD = g.head_dim as i32;
            let s = if li >= 0 {
                self.attn_slot[li as usize] as usize
            } else {
                self.n_attn as usize
            };

            let wq = &L.wq_mat;
            let wk = &L.wk_mat;
            let wv = &L.wv_mat;
            let wo = &L.wo_mat;
            // KV-only pass (the prompt's MTP rows): only the K / V cache rows of this layer are ever read, so the
            // q / gate projection, the attention, o_proj and the FFN are skipped.  K and V come from the same
            // kernels on the same quantized input as in the full layer (each projection is its own GEMM), so the
            // cache rows are bitwise the full pass's.  attn_prep still runs over all heads (its q rows read the
            // stale q / gate buffer and are never used; each head is its own block).
            let kv_only = self.kv_only_pass && T > self.gemm_t;
            if T <= self.gemm_t {
                let ws = [wq, wk, wv];
                let ys = [self.qg, self.k, self.v];
                self.apply_multi(&ws, &ys, self.h, T);
            } else if !self.dq_x.is_null()
                && wq.raw_qtype() == wk.raw_qtype()
                && wk.raw_qtype() == wv.raw_qtype()
                && tk_q36_mmq_supported_fits(wq.raw_qtype(), wq.n_out as i64)
                && tk_q36_mmq_supported_fits(wk.raw_qtype(), wk.n_out as i64)
                && (E % 256) == 0
                && self.dense_mmq_quant(self.h, wq.raw_qtype(), E, T)
            {
                if !kv_only {
                    let _ = self.dense_mmq_gemm(wq, self.qg, T);
                }
                let _ = self.dense_mmq_gemm(wk, self.k, T);
                let _ = self.dense_mmq_gemm(wv, self.v, T);
            } else {
                if !kv_only {
                    self.apply(wq, self.h, self.qg, T);
                }
                self.apply(wk, self.h, self.k, T);
                self.apply(wv, self.h, self.v, T);
            }

            let _ = tk_q36_attn_prep(
                self.tk.as_ptr(),
                T,
                self.qg,
                self.k,
                self.v,
                L.q_norm_ptr,
                L.k_norm_ptr,
                self.q,
                self.gate,
                self.kc[s].cast::<c_void>(),
                self.vc[s].cast::<c_void>(),
                H,
                HKV,
                HD,
                g.n_rot as i32,
                pos0,
                self.ctx_len,
                g.rope_freq_base,
                eps,
                self.vtc[s].cast::<c_void>(),
                self.vt_ctx,
                stream,
            );
            if self.kv_only_pass {
                // (also for T <= gemm_t: the projections above ran in full, the rest is skipped)
                self.invalidate_pf();
                self.mark_q8(ptr::null(), 0, 0);
                return;
            }

            let sc = 1.0f32 / (HD as f32).sqrt();
            let q8_out = if m.dense && T <= self.gemm_t && tk_q36_mmvq_supported(wo.raw_qtype()) {
                self.y_q8
            } else {
                ptr::null_mut()
            };

            if T > self.gemm_t {
                let p0_h = if self.cur_p0_h >= 0 {
                    self.cur_p0_h
                } else {
                    let mut v = 0i32;
                    let _ = unsafe {
                        hipMemcpy(
                            (&mut v as *mut i32).cast::<c_void>(),
                            pos0.cast::<c_void>(),
                            4,
                            HIP_MEMCPY_DEVICE_TO_HOST,
                        )
                    };
                    v
                };
                if self.opts.use_fattn
                    && self.tp_fa
                    && !self.vtc[s].is_null()
                    && talus_pf_fa_prefill(
                        self.q,
                        self.kc[s].cast::<c_void>(),
                        self.vtc[s].cast::<c_void>(),
                        self.gate,
                        self.att,
                        T,
                        H,
                        HKV,
                        HD,
                        self.ctx_len as i64,
                        self.vt_ctx as i64,
                        p0_h as i64,
                        sc,
                        stream,
                    )
                {
                    // tp_fa_prefill fused attention + gate
                } else if self.opts.use_fattn && tk_q36_fattn_supported(HD) && ((H / HKV) % 2) == 0
                {
                    let _ = unsafe {
                        tk_attn_prefill_fa(
                            self.tk.as_ptr(),
                            self.q,
                            self.kc[s].cast::<c_void>(),
                            self.vc[s].cast::<c_void>(),
                            self.att,
                            T,
                            H,
                            HKV,
                            HD,
                            self.ctx_len as i64,
                            p0_h as i64,
                            sc,
                            self.fattn_mask.cast::<c_void>(),
                            stream,
                        )
                    };
                    let _ = tk_q36_gate_mul(
                        self.tk.as_ptr(),
                        self.att,
                        self.gate,
                        (T as i64) * (Q_dim as i64),
                        stream,
                    );
                } else {
                    let _ = tk_q36_attn_blas_prefill(
                        self.tk.as_ptr(),
                        self.q,
                        self.x_f16.cast::<c_void>(),
                        self.kc[s].cast::<c_void>(),
                        self.vc[s].cast::<c_void>(),
                        self.att_s,
                        self.att_p16.cast::<c_void>(),
                        self.att,
                        self.gate,
                        T,
                        H,
                        HKV,
                        HD,
                        self.ctx_len,
                        p0_h as i64,
                        self.score_floats,
                        stream,
                    );
                }
            } else {
                let _ = tk_q36_attn_wmma(
                    self.tk.as_ptr(),
                    self.attn_w8,
                    self.n_chunk_w,
                    HKV,
                    self.q,
                    self.kc[s].cast::<c_void>(),
                    self.vtc[s].cast::<c_void>(),
                    pos0,
                    T,
                    H,
                    HKV,
                    self.ctx_len,
                    self.vt_ctx,
                    sc,
                    self.att_ck,
                    self.n_chunk_w,
                    self.pacc,
                    self.pm,
                    self.pl,
                    stream,
                );
                let _ = tk_q36_attn_merge(
                    self.tk.as_ptr(),
                    T * H,
                    HD,
                    self.pacc,
                    self.pm,
                    self.pl,
                    self.gate,
                    pos0,
                    0,
                    H,
                    self.n_chunk_w,
                    self.att,
                    self.att_ck,
                    q8_out,
                    stream,
                );
            }

            if !q8_out.is_null() {
                let _ = unsafe {
                    tk_mmvq(
                        self.tk.as_ptr(),
                        wo.tk_qtype(),
                        wo.d.cast::<c_void>(),
                        q8_out,
                        self.y,
                        wo.n_in,
                        wo.n_out,
                        T,
                        stream,
                    )
                };
            } else {
                self.apply(wo, self.att, self.y, T);
            }
        } else {
            let C = g.ssm_qkv_dim() as i32;
            let HK = g.ssm_k_heads() as i32;
            let HV = g.ssm_dt_rank() as i32;
            let s = self.gdn_slot[li as usize] as usize;
            let wqkv = &L.wqkv;
            let wgate = &L.wgate;
            let ssm_out = &L.ssm_out_mat;
            let qkv_b = if verify { self.v_qkv[s] } else { self.qkv };
            let hc_b = if verify { self.v_hc[s] } else { self.hc };
            let al_b = if verify { self.v_alpha[s] } else { self.alpha };
            let be_b = if verify { self.v_beta[s] } else { self.beta };

            if T <= self.gemm_t {
                let ws = [wqkv, wgate];
                let ys = [qkv_b, self.z];
                self.apply_multi(&ws, &ys, self.h, T);
            } else if !self.dq_x.is_null()
                && wqkv.raw_qtype() == wgate.raw_qtype()
                && tk_q36_mmq_supported_fits(wqkv.raw_qtype(), wqkv.n_out as i64)
                && tk_q36_mmq_supported_fits(wgate.raw_qtype(), wgate.n_out as i64)
                && (E % 256) == 0
                && self.dense_mmq_quant(self.h, wqkv.raw_qtype(), E, T)
            {
                let _ = self.dense_mmq_gemm(wqkv, qkv_b, T);
                let _ = self.dense_mmq_gemm(wgate, self.z, T);
            } else {
                self.apply(wqkv, self.h, qkv_b, T);
                self.apply(wgate, self.h, self.z, T);
            }

            if T <= self.gemm_t {
                let pd = tk_q36_gdn_pend {
                    q: self.pq[s],
                    h: self.ph[s],
                    a: self.pa[s],
                    b: self.pb[s],
                    ctl: if self.fused { self.gctl } else { ptr::null() },
                    store: verify,
                };
                let pd_ptr = if self.fused {
                    &pd as *const _
                } else {
                    ptr::null()
                };

                if m.dense && (E % 1024) == 0 && (C % 256) == 0 {
                    let _ = tk_q36_conv_ab(
                        self.tk.as_ptr(),
                        self.conv[s],
                        qkv_b,
                        L.ssm_conv_ptr,
                        hc_b,
                        C,
                        2 * HK,
                        eps,
                        T,
                        !verify,
                        pd_ptr,
                        L.ssm_alpha_ptr,
                        L.ssm_beta_ptr,
                        self.h,
                        L.ssm_dt_ptr,
                        L.ssm_a_ptr,
                        al_b,
                        be_b,
                        E,
                        HV,
                        stream,
                    );
                } else {
                    let _ = tk_q36_gdn_ab(
                        self.tk.as_ptr(),
                        T,
                        L.ssm_alpha_ptr,
                        L.ssm_beta_ptr,
                        self.h,
                        L.ssm_dt_ptr,
                        L.ssm_a_ptr,
                        al_b,
                        be_b,
                        E,
                        HV,
                        stream,
                    );
                    let _ = tk_q36_conv_seq(
                        self.tk.as_ptr(),
                        self.conv[s],
                        qkv_b,
                        L.ssm_conv_ptr,
                        hc_b,
                        C,
                        2 * HK,
                        eps,
                        T,
                        !verify,
                        pd_ptr,
                        stream,
                    );
                }
                let yq = if m.dense && tk_q36_mmvq_supported(ssm_out.raw_qtype()) {
                    self.y_q8
                } else {
                    ptr::null_mut()
                };
                let _ = tk_q36_gdn_seq(
                    self.tk.as_ptr(),
                    self.S[s],
                    hc_b,
                    C,
                    al_b,
                    be_b,
                    self.z,
                    L.ssm_norm_ptr,
                    eps,
                    self.qkv,
                    HK,
                    HV,
                    T,
                    !verify,
                    pd_ptr,
                    yq,
                    stream,
                );
                if !verify {
                    self.gdn_valid = true;
                }
                if !yq.is_null() {
                    let _ = unsafe {
                        tk_mmvq(
                            self.tk.as_ptr(),
                            ssm_out.tk_qtype(),
                            ssm_out.d.cast::<c_void>(),
                            yq,
                            self.y,
                            ssm_out.n_in,
                            ssm_out.n_out,
                            T,
                            stream,
                        )
                    };
                } else {
                    self.apply(ssm_out, self.qkv, self.y, T);
                }
            } else {
                if li >= 0 && (li as usize) < self.ab_w.len() && !self.ab_w[li as usize].is_null() {
                    self.f32mm(
                        self.ab_w[li as usize],
                        self.h,
                        self.ab_tmp,
                        E,
                        2 * HV,
                        T,
                        2 * HV,
                    );
                    let _ = tk_q36_gdn_gb2(
                        self.tk.as_ptr(),
                        self.ab_tmp,
                        L.ssm_dt_ptr,
                        L.ssm_a_ptr,
                        self.alpha,
                        self.beta,
                        HV,
                        T,
                        stream,
                    );
                } else {
                    self.f32mm(L.ssm_alpha_ptr, self.h, self.alpha, E, HV, T, HV);
                    self.f32mm(L.ssm_beta_ptr, self.h, self.beta, E, HV, T, HV);
                    let _ = tk_q36_gdn_gb(
                        self.tk.as_ptr(),
                        self.alpha,
                        L.ssm_dt_ptr,
                        L.ssm_a_ptr,
                        self.beta,
                        HV,
                        T,
                        stream,
                    );
                }
                let _ = tk_q36_conv_par(
                    self.tk.as_ptr(),
                    self.conv[s],
                    self.qkv,
                    L.ssm_conv_ptr,
                    self.hc,
                    C,
                    2 * HK,
                    eps,
                    T,
                    stream,
                );
                let _ = tk_q36_conv_hist(self.tk.as_ptr(), self.conv[s], self.qkv, C, T, stream);
                let ncol = std::env::var("TALUS_GDN_NCOL")
                    .or_else(|_| std::env::var("STRATA_Q36_GDN_NCOL"))
                    .ok()
                    .and_then(|v| v.parse::<i32>().ok())
                    .filter(|&v| v == 1 || v == 2 || v == 4 || v == 8 || v == 16)
                    .unwrap_or(8);
                let _ = tk_q36_gdn_seq_warp(
                    self.tk.as_ptr(),
                    ncol,
                    self.S[s],
                    self.hc,
                    C,
                    self.alpha,
                    self.beta,
                    self.qkv,
                    HK,
                    HV,
                    T,
                    stream,
                );
                let _ = tk_q36_gdn_norm(
                    self.tk.as_ptr(),
                    T,
                    self.qkv,
                    self.z,
                    L.ssm_norm_ptr,
                    eps,
                    self.qkv,
                    HV,
                    stream,
                );
                self.gdn_valid = true;
                self.apply(ssm_out, self.qkv, self.y, T);
            }
        }

        if m.dense {
            let F = g.n_ff as i32;
            let ffn_gate = &L.ffn_gate_mat;
            let ffn_up = &L.ffn_up_mat;
            let ffn_down = &L.ffn_down_mat;
            if T <= self.gemm_t && tk_q36_mmvq_supported(ffn_gate.raw_qtype()) {
                let _ = tk_q36_add_rms_q8(
                    self.tk.as_ptr(),
                    T,
                    self.x,
                    self.y,
                    L.post_norm_ptr,
                    self.h,
                    self.x_q8,
                    E,
                    eps,
                    stream,
                );
                self.mark_q8(self.h, E, T);
                let ws = [ffn_gate, ffn_up];
                let ys = [self.sh_g, self.sh_u];
                self.apply_multi(&ws, &ys, self.h, T);
                if tk_q36_mmvq_supported(ffn_down.raw_qtype()) {
                    let _ = unsafe {
                        tk_swiglu_quantize_q8_1(
                            self.tk.as_ptr(),
                            self.sh_g,
                            self.sh_u,
                            self.x_q8,
                            F,
                            T,
                            stream,
                        )
                    };
                    self.mark_q8(ptr::null(), 0, 0);
                    let _ = unsafe {
                        tk_mmvq(
                            self.tk.as_ptr(),
                            ffn_down.tk_qtype(),
                            ffn_down.d.cast::<c_void>(),
                            self.x_q8,
                            self.sh_y,
                            ffn_down.n_in,
                            ffn_down.n_out,
                            T,
                            stream,
                        )
                    };
                } else {
                    let _ = tk_q36_swiglu(
                        self.tk.as_ptr(),
                        self.sh_g,
                        self.sh_u,
                        self.sh_g,
                        (T as i64) * (F as i64),
                        stream,
                    );
                    self.apply(ffn_down, self.sh_g, self.sh_y, T);
                }
            } else {
                let _ = tk_q36_add_rms(
                    self.tk.as_ptr(),
                    T,
                    self.x,
                    self.y,
                    L.post_norm_ptr,
                    self.h,
                    E,
                    eps,
                    stream,
                );
                self.invalidate_pf();
                if self.ffn_w4i(ffn_gate, ffn_up, ffn_down, self.sh_y, T, E, F) {
                    // opt-in W4A4 FFN (TALUS_Q27_W4I)
                } else {
                    self.apply(ffn_gate, self.h, self.sh_g, T);
                    if self.tp_gu_down(ffn_up, ffn_down, self.sh_g, self.sh_y, T, F) {
                        // fused up + SwiGLU + q8_1, then down (rb-q27)
                    } else {
                        self.apply(ffn_up, self.h, self.sh_u, T);
                        if !self.tp_down(ffn_down, self.sh_g, self.sh_u, self.sh_y, T, F) {
                            let _ = tk_q36_swiglu(
                                self.tk.as_ptr(),
                                self.sh_g,
                                self.sh_u,
                                self.sh_g,
                                (T as i64) * (F as i64),
                                stream,
                            );
                            self.invalidate_pf();
                            if !self.dense_mmq(ffn_down, self.sh_g, self.sh_y, T) {
                                self.apply(ffn_down, self.sh_g, self.sh_y, T);
                            }
                        }
                    }
                }
            }
            self.invalidate_pf();
            if !wn.is_null() {
                if T <= self.gemm_t && tk_q36_mmvq_supported(m.head.raw_qtype()) {
                    let _ = tk_q36_add_rms_q8(
                        self.tk.as_ptr(),
                        T,
                        self.x,
                        self.sh_y,
                        wn,
                        hout,
                        self.x_q8,
                        E,
                        eps,
                        stream,
                    );
                    self.mark_q8(hout, E, T);
                } else {
                    let _ = tk_q36_add_rms(
                        self.tk.as_ptr(),
                        T,
                        self.x,
                        self.sh_y,
                        wn,
                        hout,
                        E,
                        eps,
                        stream,
                    );
                    self.mark_q8(ptr::null(), 0, 0);
                }
            } else {
                let _ = tk_q36_add(
                    self.tk.as_ptr(),
                    self.x,
                    self.sh_y,
                    (T as i64) * (E as i64),
                    stream,
                );
                self.mark_q8(ptr::null(), 0, 0);
            }
            return;
        }

        // MoE FFN path (`q36_engine.cu:1357-1444`)
        let NE = g.n_expert as i32;
        let K_used = m.top_k;
        let F = g.n_ff_exp as i32;
        let FS = g.n_ff_shexp as i32;
        let n_ent = T * K_used;
        let sh_gate = &L.sh_gate;
        let sh_up = &L.sh_up;
        let sh_down = &L.sh_down;

        if T <= self.gemm_t && (E & 31) == 0 {
            let _ = tk_q36_add_rms_exact_q8(
                self.tk.as_ptr(),
                T,
                self.x,
                self.y,
                L.post_norm_ptr,
                self.h,
                self.x_q8,
                E,
                eps,
                stream,
            );
            self.mark_q8(self.h, E, T);
        } else {
            let _ = tk_q36_add_rms(
                self.tk.as_ptr(),
                T,
                self.x,
                self.y,
                L.post_norm_ptr,
                self.h,
                E,
                eps,
                stream,
            );
            self.mark_q8(ptr::null(), 0, 0);
        }

        let one = T == 1 && !L.router_ext.is_null();
        if one {
            self.f32mm(L.router_ext, self.h, self.rlog, E, NE + 1, 1, NE + 1);
        } else {
            self.f32mm(L.router, self.h, self.rlog, E, NE, T, NE);
        }
        let _ = tk_q36_router_topk(
            self.tk.as_ptr(),
            self.rlog,
            self.ids,
            self.wts,
            NE,
            K_used,
            T,
            stream,
        );

        let use_mmq = !m.hybrid
            && T > self.gemm_t
            && !self.mmq_xq.is_null()
            && tk_q36_mmq_supported_fits(L.gu_type, (2 * F) as i64)
            && tk_q36_mmq_supported_fits(L.d_type, E as i64);

        if !m.hybrid {
            if use_mmq {
                let _ = unsafe {
                    hipMemsetAsync(
                        self.mmq_cnt.cast::<c_void>(),
                        0,
                        (NE as usize) * 4,
                        self.s_main,
                    )
                };
                let _ = tk_q36_route_count(self.tk.as_ptr(), self.ids, n_ent, self.mmq_cnt, stream);
                let _ = tk_q36_route_scan(
                    self.tk.as_ptr(),
                    self.mmq_cnt,
                    NE,
                    self.mmq_bounds,
                    self.mmq_cur,
                    self.mmq_max,
                    stream,
                );
                let _ = tk_q36_route_scatter(
                    self.tk.as_ptr(),
                    self.ids,
                    n_ent,
                    K_used,
                    self.mmq_cur,
                    self.mmq_dst,
                    self.mmq_src,
                    self.mmq_inv,
                    stream,
                );
                let max_r = unsafe {
                    let _ = hipMemcpyAsync(
                        self.hpin.add(10).cast::<c_void>(),
                        self.mmq_max.cast::<c_void>(),
                        4,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                        self.s_main,
                    );
                    let _ = hipStreamSynchronize(self.s_main);
                    *self.hpin.add(10)
                };
                let _ = tk_q36_mmq_quantize(
                    self.tk.as_ptr(),
                    self.h,
                    self.mmq_src,
                    self.mmq_xq,
                    L.gu_type,
                    E as i64,
                    E as i64,
                    n_ent as i64,
                    stream,
                );
                let p_gu = tk_moe_mmq_product {
                    w: L.blobs.cast::<c_void>(),
                    qtype: to_tk_qtype(L.gu_type),
                    w_rows: (2 * F) as i64,
                    w_cols: E as i64,
                    expert_bytes: L.stride,
                    n_experts: NE,
                    x_q8_1: self.mmq_xq,
                    bounds_dev: self.mmq_bounds,
                    ids_dev: self.mmq_ident,
                    total_rows: n_ent as i64,
                    max_rows: max_r as i64,
                    dst: self.mmq_gu,
                    ld_dst: (2 * F) as i64,
                    y_scale_dev: ptr::null(),
                };
                let _ = unsafe { tk_prefill_moe_mmq(self.tk.as_ptr(), &p_gu, stream) };
                let _ = tk_q36_mmq_swiglu(
                    self.tk.as_ptr(),
                    self.mmq_gu,
                    self.mmq_h,
                    n_ent as i64,
                    F as i64,
                    false,
                    stream,
                );
                let _ = tk_q36_mmq_quantize(
                    self.tk.as_ptr(),
                    self.mmq_h,
                    ptr::null(),
                    self.mmq_hq,
                    L.d_type,
                    F as i64,
                    F as i64,
                    n_ent as i64,
                    stream,
                );
                let p_d = tk_moe_mmq_product {
                    w: unsafe { L.blobs.add(L.L.down_off).cast::<c_void>() },
                    qtype: to_tk_qtype(L.d_type),
                    w_rows: E as i64,
                    w_cols: F as i64,
                    expert_bytes: L.stride,
                    n_experts: NE,
                    x_q8_1: self.mmq_hq,
                    bounds_dev: self.mmq_bounds,
                    ids_dev: self.mmq_ident,
                    total_rows: n_ent as i64,
                    max_rows: max_r as i64,
                    dst: self.parts,
                    ld_dst: E as i64,
                    y_scale_dev: ptr::null(),
                };
                let _ = unsafe { tk_prefill_moe_mmq(self.tk.as_ptr(), &p_d, stream) };
            } else {
                let cap_grp = NE.min(n_ent);
                let _ = tk_q36_group(
                    self.tk.as_ptr(),
                    self.ids,
                    n_ent,
                    K_used,
                    NE,
                    L.blobs,
                    L.stride as u64,
                    self.grp_ptr,
                    self.grp_start,
                    self.n_groups,
                    self.ent_dst,
                    self.ent_tok,
                    stream,
                );
                self.quant(self.h, E, T);
                let _ = tk_q36_expert_grouped(
                    self.tk.as_ptr(),
                    L.gu_type,
                    L.d_type,
                    E as i64,
                    F as i64,
                    self.grp_ptr,
                    self.grp_start,
                    self.n_groups,
                    self.ent_dst,
                    self.ent_tok,
                    cap_grp as i64,
                    n_ent as i64,
                    self.x_q8,
                    self.moe_scr,
                    self.parts,
                    cap_grp as i64,
                    stream,
                );
            }
        } else {
            let cap_grp = NE.min(n_ent);
            let li_u = li.max(0) as usize;
            let _ = tk_q36_group_hy(
                self.tk.as_ptr(),
                self.ids,
                n_ent,
                K_used,
                NE,
                self.slot_of[li_u],
                L.blobs,
                L.stride as u64,
                self.grp_ptr,
                self.grp_start,
                self.n_groups,
                self.ent_dst,
                self.ent_tok,
                stream,
            );
            self.quant(self.h, E, T);
            let _ = tk_q36_expert_grouped(
                self.tk.as_ptr(),
                L.gu_type,
                L.d_type,
                E as i64,
                F as i64,
                self.grp_ptr,
                self.grp_start,
                self.n_groups,
                self.ent_dst,
                self.ent_tok,
                cap_grp as i64,
                n_ent as i64,
                self.x_q8,
                self.moe_scr,
                self.parts,
                cap_grp as i64,
                stream,
            );
        }

        self.apply(sh_gate, self.h, self.sh_g, T);
        self.apply(sh_up, self.h, self.sh_u, T);
        if T <= self.gemm_t && tk_q36_mmvq_supported(sh_down.raw_qtype()) {
            let _ = unsafe {
                tk_swiglu_quantize_q8_1(
                    self.tk.as_ptr(),
                    self.sh_g,
                    self.sh_u,
                    self.x_q8,
                    FS,
                    T,
                    stream,
                )
            };
            self.mark_q8(ptr::null(), 0, 0);
            let _ = unsafe {
                tk_mmvq(
                    self.tk.as_ptr(),
                    sh_down.tk_qtype(),
                    sh_down.d.cast::<c_void>(),
                    self.x_q8,
                    self.sh_y,
                    sh_down.n_in,
                    sh_down.n_out,
                    T,
                    stream,
                )
            };
        } else {
            let _ = tk_q36_swiglu(
                self.tk.as_ptr(),
                self.sh_g,
                self.sh_u,
                self.sh_g,
                (T as i64) * (FS as i64),
                stream,
            );
            self.apply(sh_down, self.sh_g, self.sh_y, T);
        }

        if !one {
            self.f32mm(L.sh_inp, self.h, self.sgate, E, 1, T, 1);
        }
        let sgate_ptr = if one {
            unsafe { self.rlog.add(NE as usize) }
        } else {
            self.sgate
        };
        let prow = if use_mmq { self.mmq_inv } else { ptr::null() };
        let use_q8 = !wn.is_null() && T <= self.gemm_t && (E & 31) == 0;
        let _ = tk_q36_moe_out_q8(
            self.tk.as_ptr(),
            T,
            self.x,
            self.parts,
            prow,
            self.wts,
            K_used,
            self.sh_y,
            sgate_ptr,
            1,
            wn,
            hout,
            if use_q8 { self.x_q8 } else { ptr::null_mut() },
            E,
            eps,
            stream,
        );
        if use_q8 {
            self.mark_q8(hout, E, T);
        } else {
            self.mark_q8(ptr::null(), 0, 0);
        }
    }

    /// Launch device argmax (`Engine::argmax_dev` in `q36_engine.cu:1318-1323`).
    pub fn argmax_dev(
        &mut self,
        lg: *const f32,
        rows: i32,
        out: *mut i32,
        V: i32,
        map: *const i32,
    ) {
        let stream = self.s_main.cast::<c_void>();
        let _ = tk_q36_argmax(
            self.tk.as_ptr(),
            128,
            rows,
            lg,
            V,
            out,
            self.arg_pv,
            self.arg_pi,
            false,
            ptr::null(),
            stream,
        );
        let _ = tk_q36_argmax(
            self.tk.as_ptr(),
            1,
            rows,
            lg,
            128,
            out,
            self.arg_pv,
            self.arg_pi,
            true,
            map,
            stream,
        );
    }

    /// Compute vocabulary output logits (and optional argmax picks) for `rows` already-normed hidden state rows (`Engine::head_rows` in `q36_engine.cu:1325-1330`).
    pub fn head_rows_with_pick(
        &mut self,
        hn: *const f32,
        rows: i32,
        dst_logits: *mut f32,
        dst_pick: *mut i32,
        head_opt: Option<&Mat>,
        map: *const i32,
    ) {
        let m_ptr = self.m;
        let head = head_opt.unwrap_or_else(|| unsafe { &(*m_ptr).head });
        let n_out = head.n_out;
        self.invalidate_pf();
        self.apply(head, hn, dst_logits, rows);
        if !dst_pick.is_null() {
            self.argmax_dev(dst_logits, rows, dst_pick, n_out, map);
        }
    }

    /// Compute full-vocabulary output logits for `rows` hidden state rows (`Engine::head_rows`).
    pub fn head_rows(&mut self, hn: *const f32, rows: i32, dst_logits: *mut f32) {
        self.head_rows_with_pick(hn, rows, dst_logits, ptr::null_mut(), None, ptr::null());
    }

    /// Return the active draft head matrix (`Engine::draft_head` in `q36_engine.cu:1312-1315`).
    #[inline]
    pub fn draft_head(&self) -> &Mat {
        let m = self.model();
        if !m.dv_head.is_null() {
            &m.dv_head
        } else if !m.mtp_head.is_null() {
            &m.mtp_head
        } else {
            &m.head
        }
    }

    /// Compute MTP / draft vocabulary logits for `rows` already-normed hidden state rows.
    pub fn draft_head_rows(&mut self, hn: *const f32, rows: i32, dst_logits: *mut f32) {
        let m_ptr = self.m;
        let head = unsafe {
            if !(*m_ptr).dv_head.is_null() {
                &(*m_ptr).dv_head
            } else if !(*m_ptr).mtp_head.is_null() {
                &(*m_ptr).mtp_head
            } else {
                &(*m_ptr).head
            }
        };
        let map = self.draft_map;
        self.head_rows_with_pick(hn, rows, dst_logits, ptr::null_mut(), Some(head), map);
    }

    /// Single draft step head projection + optional sampling + confidence (`Engine::draft_head_rows(hn, step, out)` in `q36_engine.cu:1333-1343`).
    pub fn draft_head_step(&mut self, hn: *const f32, step: i32, out: *mut i32) {
        let m_ptr = self.m;
        let head = unsafe {
            if !(*m_ptr).dv_head.is_null() {
                &(*m_ptr).dv_head
            } else if !(*m_ptr).mtp_head.is_null() {
                &(*m_ptr).mtp_head
            } else {
                &(*m_ptr).head
            }
        };
        let V = head.n_out;
        let stp = step.max(0) as usize;
        let lg = unsafe { self.logits_md.add(stp * (V as usize)) };
        self.invalidate_pf();
        self.apply(head, hn, lg, 1);
        let stream = self.s_main.cast::<c_void>();
        unsafe {
            let lse_p = self.lse_md.add(stp);
            let thr_p = self.lse_md.add(16 + stp);
            if self.draft_nucleus {
                let _ = tk_q36_draft_support(
                    self.tk.as_ptr(),
                    lg,
                    V,
                    ptr::addr_of!((*self.ds_d).tau),
                    ptr::addr_of!((*self.ds_d).lmin),
                    ptr::addr_of!((*self.ds_d).topp),
                    ptr::addr_of!((*self.ds_d).topk),
                    lse_p,
                    thr_p,
                    stream,
                );
            } else {
                let _ = tk_q36_lse(self.tk.as_ptr(), lg, V, self.ds_d, lse_p, thr_p, stream);
            }
            let _ = tk_q36_argmax_samp(
                self.tk.as_ptr(),
                lg,
                V,
                self.arg_pv,
                self.arg_pi,
                self.ds_d,
                step,
                thr_p,
                stream,
            );
            let _ = tk_q36_argmax(
                self.tk.as_ptr(),
                1,
                1,
                lg,
                128,
                out,
                self.arg_pv,
                self.arg_pi,
                true,
                self.draft_map,
                stream,
            );
            if self.dl.want_conf() {
                let _ = tk_q36_draft_conf(
                    self.tk.as_ptr(),
                    lg,
                    V,
                    ptr::addr_of!((*self.ds_d).tau),
                    ptr::addr_of!((*self.ds_d).lmin),
                    out,
                    self.draft_map,
                    self.conf_md,
                    self.conf_md.add(16),
                    step,
                    lse_p,
                    thr_p,
                    stream,
                );
            }
        }
    }

    /// Full forward pass matching `Engine::forward(int T, int64_t p0, int heads, bool graph_mode, bool verify)` (`q36_engine.cu:1283-1310`).
    pub fn forward_full(
        &mut self,
        tokens: *const i32,
        n_tok: i32,
        pos0: *const i32,
        heads: i32,
        verify: bool,
    ) {
        let m_ptr = self.m;
        let m = unsafe { &*m_ptr };
        let g = &m.g;
        let E = g.n_embd as i32;
        let eps = g.rms_norm_eps;
        let stream = self.s_main.cast::<c_void>();
        self.prep(n_tok);
        let _ = tk_q36_embed_rows(
            self.tk.as_ptr(),
            m.embd_type,
            m.embd.cast::<c_void>(),
            m.embd_row,
            tokens,
            n_tok as i64,
            g.n_embd,
            self.x,
            stream,
        );
        if n_tok <= self.gemm_t && (E & 31) == 0 {
            let _ = tk_q36_rms_q8(
                self.tk.as_ptr(),
                n_tok,
                self.x,
                m.layers[0].attn_norm_ptr,
                self.h,
                self.x_q8,
                E,
                eps,
                stream,
            );
            self.mark_q8(self.h, E, n_tok);
        } else {
            let _ = tk_q36_rms(
                self.tk.as_ptr(),
                n_tok,
                self.x,
                m.layers[0].attn_norm_ptr,
                self.h,
                E,
                eps,
                stream,
            );
            self.mark_q8(ptr::null(), 0, 0);
        }
        let n_layer = m.layers.len();
        for li in 0..n_layer {
            let last = (li + 1) == n_layer;
            let wn = if last {
                m.out_norm
            } else {
                m.layers[li + 1].attn_norm_ptr
            };
            let hout = if last { self.h_last } else { self.h };
            let layer_ref = unsafe { &(&(*m_ptr).layers)[li] };
            self.layer_into(li as i32, layer_ref, pos0, verify, wn, hout);
            if self.use_dflash && self.df.ready {
                self.df.tap((li + 1) as i32, self.x, n_tok, self.s_main);
            }
            if !self.dbg.is_null() && self.dbg_row + n_tok <= 16 {
                let dst = unsafe {
                    self.dbg
                        .add((li * 16 + (self.dbg_row as usize)) * (E as usize))
                };
                let _ = unsafe {
                    hipMemcpyAsync(
                        dst.cast::<c_void>(),
                        self.x.cast::<c_void>(),
                        (n_tok as usize) * (E as usize) * 4,
                        HIP_MEMCPY_DEVICE_TO_DEVICE,
                        self.s_main,
                    )
                };
            }
        }
        if self.fused && n_tok <= self.gemm_t {
            let _ = tk_q36_ctl_end(self.tk.as_ptr(), self.gctl, verify, stream);
        }
        if heads > 0 {
            let hn = unsafe { self.h_last.add(((n_tok - heads) as usize) * (E as usize)) };
            self.head_rows_with_pick(hn, heads, self.logits, self.pick_v, None, ptr::null());
        }
    }

    /// Forward pass over `n_tok` tokens starting at `*pos0` (`Engine::forward` in `q36_engine.cu:1283-1310`).
    pub fn forward(
        &mut self,
        tokens: *const i32,
        n_tok: i32,
        pos0: *const i32,
        compute_logits: bool,
    ) {
        let verify = self.pend_on;
        let heads = if compute_logits { 1 } else { 0 };
        self.forward_full(tokens, n_tok, pos0, heads, verify);
    }

    /// Run MTP block for `T` tokens (`Engine::forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode, int32_t* pick_out, int dstep)` in `q36_engine.cu:1346-1363`).
    pub fn forward_mtp_batch(
        &mut self,
        T: i32,
        p0: i64,
        seeds: *const f32,
        draft: bool,
        _graph_mode: bool,
        pick_out: *mut i32,
        dstep: i32,
    ) {
        let m_ptr = self.m;
        let m = unsafe { &*m_ptr };
        if !m.has_mtp || T <= 0 {
            return;
        }
        let g = &m.g;
        let E = g.n_embd as i32;
        let eps = g.rms_norm_eps;
        let stream = self.s_main.cast::<c_void>();
        self.prep(T);
        let _ = tk_q36_embed_rows(
            self.tk.as_ptr(),
            m.embd_type,
            m.embd.cast::<c_void>(),
            m.embd_row,
            self.tok_m_d,
            T as i64,
            g.n_embd,
            self.mtp_en,
            stream,
        );
        let use_q8 = T <= self.gemm_t && (E & 31) == 0;
        let q8_en = if use_q8 { self.x_q8 } else { ptr::null_mut() };
        let q8_hn = if use_q8 {
            unsafe {
                self.x_q8
                    .cast::<u8>()
                    .add(tk_q8_1_bytes(E, 1))
                    .cast::<c_void>()
            }
        } else {
            ptr::null_mut()
        };
        let _ = tk_q36_rms_ld_q8(
            self.tk.as_ptr(),
            T,
            self.mtp_en,
            E,
            m.mtp_enorm,
            self.mtp_cat,
            2 * E,
            q8_en,
            E,
            eps,
            stream,
        );
        let _ = tk_q36_rms_ld_q8(
            self.tk.as_ptr(),
            T,
            seeds,
            E,
            m.mtp_hnorm,
            unsafe { self.mtp_cat.add(E as usize) },
            2 * E,
            q8_hn,
            E,
            eps,
            stream,
        );
        if use_q8 {
            self.mark_q8(self.mtp_cat, 2 * E, T);
        }
        let eh_proj = unsafe { &(*m_ptr).eh_proj };
        self.apply(eh_proj, self.mtp_cat, self.x, T);
        let mtp_layer = unsafe { &(*m_ptr).mtp };
        let _ = tk_q36_rms_q8(
            self.tk.as_ptr(),
            T,
            self.x,
            mtp_layer.attn_norm_ptr,
            self.h,
            if use_q8 { self.x_q8 } else { ptr::null_mut() },
            E,
            eps,
            stream,
        );
        if use_q8 {
            self.mark_q8(self.h, E, T);
        } else {
            self.mark_q8(ptr::null(), 0, 0);
        }
        let wn = if !m.mtp_shnorm.is_null() {
            m.mtp_shnorm
        } else {
            m.out_norm
        };
        let prev_p0_h = self.cur_p0_h;
        self.cur_p0_h = p0 as i32;
        self.cur_T = T;
        self.layer_into(-1, mtp_layer, self.pos0_m_d, false, wn, self.h_mtp_b);
        self.cur_p0_h = prev_p0_h;
        if draft {
            let last_hn = unsafe { self.h_mtp_b.add(((T - 1) as usize) * (E as usize)) };
            let pk = if !pick_out.is_null() {
                pick_out
            } else {
                self.pick_m
            };
            if dstep >= 0 {
                self.draft_head_step(last_hn, dstep, pk);
            } else {
                let head = unsafe {
                    if !(*m_ptr).dv_head.is_null() {
                        &(*m_ptr).dv_head
                    } else if !(*m_ptr).mtp_head.is_null() {
                        &(*m_ptr).mtp_head
                    } else {
                        &(*m_ptr).head
                    }
                };
                self.head_rows_with_pick(last_hn, 1, self.logits_m, pk, Some(head), self.draft_map);
            }
        }
    }

    /// Run single-token MTP draft block (`Engine::forward_mtp` convenience wrapper).
    pub fn forward_mtp(
        &mut self,
        tok_next: *const i32,
        hidden_src: *const f32,
        pos: *const i32,
        compute_logits: bool,
        h_out: *mut f32,
    ) {
        let m_ptr = self.m;
        let m = unsafe { &*m_ptr };
        if !m.has_mtp {
            return;
        }
        let E = m.g.n_embd as usize;
        if tok_next != self.tok_m_d {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.tok_m_d.cast::<c_void>(),
                    tok_next.cast::<c_void>(),
                    4,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.s_main,
                );
            }
        }
        if pos != self.pos0_m_d {
            unsafe {
                let _ = hipMemcpyAsync(
                    self.pos0_m_d.cast::<c_void>(),
                    pos.cast::<c_void>(),
                    4,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.s_main,
                );
            }
        }
        self.forward_mtp_batch(1, 0, hidden_src, compute_logits, false, self.pick_m, -1);
        if !h_out.is_null() && h_out != self.h_mtp_b {
            let _ = unsafe {
                hipMemcpyAsync(
                    h_out.cast::<c_void>(),
                    self.h_mtp_b.cast::<c_void>(),
                    E * 4,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.s_main,
                )
            };
        }
    }

    /// Process a slice of prompt or decode tokens (`Engine::process` in `q36_engine.cu:1595-1635`).
    pub fn process(&mut self, tokens: &[i32], want_logits: bool) {
        self.process_with_callback(tokens, want_logits, |_| {});
    }

    /// Process a slice of prompt or decode tokens with a per-chunk progress callback (`Engine::process` in `q36_engine.cu:1595-1635`).
    pub fn process_with_callback<F: FnMut(usize)>(
        &mut self,
        tokens: &[i32],
        want_logits: bool,
        mut on_chunk: F,
    ) {
        let total = tokens.len() as i32;
        if total <= 0 {
            return;
        }
        let E = self.d_model() as usize;
        if total == 1 && want_logits {
            if !self.fused {
                self.flush_gdn();
            }
            let p0 = self.n_past;
            unsafe {
                let _ = hipStreamSynchronize(self.s_main);
                *self.hpin.add(0) = tokens[0];
                *self.hpin.add(1) = p0;
                let _ = hipMemcpyAsync(
                    self.tok_d.cast::<c_void>(),
                    self.hpin.cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
                let _ = hipMemcpyAsync(
                    self.pos0_d.cast::<c_void>(),
                    self.hpin.add(1).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
            }
            if !self.model().hybrid && self.opts.use_graph {
                if let Ok(exec) = self.ensure_single_decode_graph() {
                    let _ = unsafe { hipGraphLaunch(exec, self.s_main) };
                } else {
                    self.forward_full(self.tok_d, 1, self.pos0_d, 1, false);
                }
            } else {
                self.forward_full(self.tok_d, 1, self.pos0_d, 1, false);
            }
            self.gdn_pending = 0;
            self.gdn_valid = true;
            if self.use_dflash && self.df.ready {
                self.df.inject(1, self.pos0_d, self.s_main);
                self.dctx.at(p0 as i64);
                self.dctx.inject(p0 as i64, 1, 1);
            }
            unsafe {
                let _ = hipMemcpyAsync(
                    self.h_mtp.cast::<c_void>(),
                    self.h_last.cast::<c_void>(),
                    E * 4,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.s_main,
                );
                let _ = hipMemcpyAsync(
                    self.pick_v_h.cast::<c_void>(),
                    self.pick_v.cast::<c_void>(),
                    4,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    self.s_main,
                );
                let _ = hipMemcpyAsync(
                    self.logits_h.cast::<c_void>(),
                    self.logits.cast::<c_void>(),
                    (self.n_vocab() as usize) * 4,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                    self.s_main,
                );
                let _ = hipStreamSynchronize(self.s_main);
            }
            self.n_past += 1;
            on_chunk(1);
            return;
        }

        self.flush_gdn();
        let mut off = 0i32;
        while off < total {
            let T = (total - off).min(self.max_batch);
            let last = (off + T) == total;
            let p0 = self.n_past;
            unsafe {
                let _ = hipStreamSynchronize(self.s_main);
                *self.hpin.add(1) = p0;
                let _ = hipMemcpyAsync(
                    self.tok_d.cast::<c_void>(),
                    tokens.as_ptr().add(off as usize).cast::<c_void>(),
                    (T as usize) * 4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
                let _ = hipMemcpyAsync(
                    self.pos0_d.cast::<c_void>(),
                    self.hpin.add(1).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                    self.s_main,
                );
            }
            let heads = if want_logits && last { 1 } else { 0 };
            self.cur_p0_h = p0;
            self.forward_full(self.tok_d, T, self.pos0_d, heads, false);
            self.cur_p0_h = -1;
            if self.use_dflash && self.df.ready {
                self.dctx.at(p0 as i64);
                if (off + T + self.df.keep()) > total {
                    let r0 = (total - self.df.keep() - off).max(0);
                    self.df.inject_at(T, p0 as i64, r0, self.s_main);
                    self.dctx.inject((p0 + r0) as i64, T - r0, T - r0);
                }
            }
            if self.model().has_mtp && self.mtp_d > 0 {
                // Build MTP seed rows [hlast, hmain[0..T-1]] in self.y (size B * V.max(E) floats)
                let seed_buf = self.y;
                unsafe {
                    let _ = hipMemcpyAsync(
                        seed_buf.cast::<c_void>(),
                        self.h_mtp.cast::<c_void>(),
                        E * 4,
                        HIP_MEMCPY_DEVICE_TO_DEVICE,
                        self.s_main,
                    );
                    if T > 1 {
                        let _ = hipMemcpyAsync(
                            seed_buf.add(E).cast::<c_void>(),
                            self.h_last.cast::<c_void>(),
                            ((T - 1) as usize) * E * 4,
                            HIP_MEMCPY_DEVICE_TO_DEVICE,
                            self.s_main,
                        );
                    }
                    let _ = hipMemcpyAsync(
                        self.tok_m_d.cast::<c_void>(),
                        tokens.as_ptr().add(off as usize).cast::<c_void>(),
                        (T as usize) * 4,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                        self.s_main,
                    );
                    let _ = hipMemcpyAsync(
                        self.pos0_m_d.cast::<c_void>(),
                        self.hpin.add(1).cast::<c_void>(),
                        4,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                        self.s_main,
                    );
                }
                // The MTP rows of the prompt: only their K / V cache rows are read later (the first draft after the
                // prompt reruns the MTP block on its own row and overwrites h_mtp_b), so KV-only by default.
                self.kv_only_pass = self.mtp_kv_only;
                self.forward_mtp_batch(
                    T,
                    self.n_past as i64,
                    seed_buf,
                    false,
                    false,
                    ptr::null_mut(),
                    -1,
                );
                self.kv_only_pass = false;
            }
            unsafe {
                let _ = hipMemcpyAsync(
                    self.h_mtp.cast::<c_void>(),
                    self.h_last.add(((T - 1) as usize) * E).cast::<c_void>(),
                    E * 4,
                    HIP_MEMCPY_DEVICE_TO_DEVICE,
                    self.s_main,
                );
            }
            self.n_past += T;
            off += T;
            let _ = unsafe { hipStreamSynchronize(self.s_main) };
            on_chunk(off as usize);
        }
        if want_logits {
            unsafe {
                let _ = hipMemcpy(
                    self.pick_v_h.cast::<c_void>(),
                    self.pick_v.cast::<c_void>(),
                    4,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                );
                let _ = hipMemcpy(
                    self.logits_h.cast::<c_void>(),
                    self.logits.cast::<c_void>(),
                    (self.n_vocab() as usize) * 4,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                );
            }
        }
    }

    /// Warm up prompt chunk, GEMM chunk, decode step, and speculative step (`Engine::warmup` in `q36_engine.cu:1992-2019`).
    pub fn warmup(&mut self) {
        let t0 = std::time::Instant::now();
        let filler = 198i32;
        let tmax = self.max_batch.max(1) as usize;
        if (tmax as i32) + self.gemm_t + 32 <= self.ctx_len {
            let toks = vec![filler; tmax];
            self.process(&toks, false);
            if (tmax as i32) > self.gemm_t + 1 {
                let small = vec![filler; (self.gemm_t + 1) as usize];
                self.process(&small, false);
            }
        }
        self.process(&[filler], true);
        if self.mtp_d > 0 && self.dev_rounds && (self.n_past + 32) <= self.ctx_len {
            if self.opts.use_graph && !self.model().hybrid {
                let _ = self.ensure_round_graphs(self.mtp_d as usize, false, self.use_dflash);
            }
            let mut dw = self.mtp_d;
            let mut win = [0i32; 16];
            let mut picks = [0i32; 16];
            self.spec_start_window(filler, self.mtp_d);
            self.spec_verify_window(&mut dw, &mut win, &mut picks);
            self.spec_advance_window(0, picks[0], self.mtp_d, None);
            self.spec_verify_window(&mut dw, &mut win, &mut picks);
            self.spec_finish_window(0, picks[0]);
        }
        let _ = unsafe { hipStreamSynchronize(self.s_main) };
        self.reset();
        eprintln!(
            "strata-q36: warmed up in {:.1} s (one prompt chunk, a decode step{})",
            t0.elapsed().as_secs_f64(),
            if self.mtp_d > 0 {
                ", a speculative step"
            } else {
                ""
            }
        );
    }

    /// Return the argmax token ID from the most recently populated `pick_v_h` / `logits_h` host buffer.
    pub fn greedy_from_host_logits(&self) -> i32 {
        let v = self.n_vocab().max(0) as usize;
        if self.logits_h.is_null() || v == 0 {
            if !self.pick_v_h.is_null() {
                return unsafe { *self.pick_v_h };
            }
            return 0;
        }
        let slice = unsafe { std::slice::from_raw_parts(self.logits_h, v) };
        let mut best_i = 0i32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &val) in slice.iter().enumerate() {
            if val > best_v {
                best_v = val;
                best_i = i as i32;
            }
        }
        if best_v == f32::NEG_INFINITY && !self.pick_v_h.is_null() {
            return unsafe { *self.pick_v_h };
        }
        best_i
    }

    /// Total bytes required to snapshot all GDN recurrent states + `hlast` (`Engine::state_bytes` in `q36_engine.cu:1636-1639`).
    pub fn state_bytes(&self) -> usize {
        let g = &self.model().g;
        let C = g.ssm_qkv_dim() as usize;
        let HV = g.ssm_time_step_rank as usize;
        let S_dim = g.ssm_state_size as usize;
        let E = g.n_embd as usize;
        ((self.n_gdn as usize) * (3 * C + HV * S_dim * S_dim) + E) * 4
    }

    /// Copy GDN recurrent state + `hlast` to/from `buf` (`Engine::state_copy` in `q36_engine.cu:1640-1657`).
    pub fn state_copy(&mut self, buf: *mut c_void, to_host: bool) {
        if to_host {
            self.flush_gdn();
        }
        let g = &self.model().g;
        let C = g.ssm_qkv_dim() as usize;
        let HV = g.ssm_time_step_rank as usize;
        let S_dim = g.ssm_state_size as usize;
        let E = g.n_embd as usize;
        let cb = 3 * C * 4;
        let sb = HV * S_dim * S_dim * 4;
        let hb = E * 4;
        let kind = if to_host {
            HIP_MEMCPY_DEVICE_TO_HOST
        } else {
            HIP_MEMCPY_HOST_TO_DEVICE
        };
        let mut p = buf.cast::<u8>();
        for i in 0..(self.n_gdn as usize) {
            unsafe {
                if to_host {
                    let _ = hipMemcpy(p.cast::<c_void>(), self.conv[i].cast::<c_void>(), cb, kind);
                } else {
                    let _ = hipMemcpy(self.conv[i].cast::<c_void>(), p.cast::<c_void>(), cb, kind);
                }
                p = p.add(cb);
                if to_host {
                    let _ = hipMemcpy(p.cast::<c_void>(), self.S[i].cast::<c_void>(), sb, kind);
                } else {
                    let _ = hipMemcpy(self.S[i].cast::<c_void>(), p.cast::<c_void>(), sb, kind);
                }
                p = p.add(sb);
            }
        }
        unsafe {
            if to_host {
                let _ = hipMemcpy(p.cast::<c_void>(), self.h_mtp.cast::<c_void>(), hb, kind);
            } else {
                let _ = hipMemcpy(self.h_mtp.cast::<c_void>(), p.cast::<c_void>(), hb, kind);
            }
        }
        if !to_host {
            unsafe {
                let _ = hipMemset(self.gctl.cast::<c_void>(), 0, 16);
            }
            self.gdn_pending = 0;
            self.gdn_valid = true;
        }
        let _ = unsafe { hipDeviceSynchronize() };
    }

    /// Total bytes required to snapshot KV caches for `n_pos` positions (`Engine::kv_bytes`).
    pub fn kv_bytes(&self, n_pos: i32) -> usize {
        let g = &self.model().g;
        let HKV = g.n_head_kv as usize;
        let HD = g.head_dim as usize;
        let vp = ((n_pos as usize) + 15) / 16 * 16;
        (self.n_attn_state as usize) * HKV * (2 * (n_pos as usize) + vp) * HD * 2
    }

    /// Copy KV caches for `n_pos` positions to/from `buf` (`Engine::kv_copy`).
    pub fn kv_copy(&mut self, buf: *mut c_void, n_pos: i32, to_host: bool) {
        if n_pos <= 0 {
            return;
        }
        let g = &self.model().g;
        let HKV = g.n_head_kv as usize;
        let HD = g.head_dim as usize;
        let hb = (n_pos as usize) * HD * 2;
        let vp = ((n_pos as usize) + 15) / 16 * 16;
        let vb = vp * HD * 2;
        let kind = if to_host {
            HIP_MEMCPY_DEVICE_TO_HOST
        } else {
            HIP_MEMCPY_HOST_TO_DEVICE
        };
        let mut p = buf.cast::<u8>();
        for s in 0..(self.n_attn_state as usize) {
            for h in 0..HKV {
                unsafe {
                    let k_ptr = self.kc[s].add(h * (self.ctx_len as usize) * HD);
                    if to_host {
                        let _ = hipMemcpy(p.cast::<c_void>(), k_ptr.cast::<c_void>(), hb, kind);
                    } else {
                        let _ = hipMemcpy(k_ptr.cast::<c_void>(), p.cast::<c_void>(), hb, kind);
                    }
                    p = p.add(hb);

                    let v_ptr = self.vc[s].add(h * (self.ctx_len as usize) * HD);
                    if to_host {
                        let _ = hipMemcpy(p.cast::<c_void>(), v_ptr.cast::<c_void>(), hb, kind);
                    } else {
                        let _ = hipMemcpy(v_ptr.cast::<c_void>(), p.cast::<c_void>(), hb, kind);
                    }
                    p = p.add(hb);

                    let vt_ptr = self.vtc[s].add(h * (self.vt_ctx as usize) * HD);
                    if to_host {
                        let _ = hipMemcpy(p.cast::<c_void>(), vt_ptr.cast::<c_void>(), vb, kind);
                    } else {
                        let _ = hipMemcpy(vt_ptr.cast::<c_void>(), p.cast::<c_void>(), vb, kind);
                    }
                    p = p.add(vb);
                }
            }
        }
        let _ = unsafe { hipDeviceSynchronize() };
    }

    // ========================================================================
    // Self-tests & Benchmarks (`q36_engine.cu:1994-2255`)
    // ========================================================================

    /// Verify bit-exact equivalence between `T`-row verify window and `T` single-token decode steps (`exact_test` in `q36_engine.cu:3078-3130`).
    pub fn exact_test(&mut self, prompt: &[i32], toks: &[i32]) -> bool {
        let g = self.model().g.clone();
        let N = g.n_embd as usize;
        let L = self.model().layers.len();
        let T = toks.len().clamp(1, 16);
        let V = g.n_vocab as usize;
        let mut a = vec![0.0f32; L * 16 * N];
        let mut b = vec![0.0f32; L * 16 * N];
        let mut la = vec![0.0f32; T * V];
        let mut lb = vec![0.0f32; T * V];

        let dbg_buf = match hip_alloc_zeroed::<f32>(a.len(), &mut self.allocs) {
            Ok(p) => p,
            Err(_) => return false,
        };
        self.dbg = dbg_buf;
        let saved_graph = self.opts.use_graph;
        self.opts.use_graph = false;

        // (a) the verify window
        self.reset();
        if !prompt.is_empty() {
            self.process(prompt, true);
        }
        unsafe {
            let _ = hipStreamSynchronize(self.s_main);
            let p0 = self.n_past;
            *self.hpin.add(1) = p0;
            let _ = hipMemcpy(
                self.tok_d.cast::<c_void>(),
                toks.as_ptr().cast::<c_void>(),
                T * 4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
            let _ = hipMemcpy(
                self.pos0_d.cast::<c_void>(),
                self.hpin.add(1).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
        }
        self.dbg_row = 0;
        self.forward_full(self.tok_d, T as i32, self.pos0_d, T as i32, true);
        unsafe {
            let _ = hipStreamSynchronize(self.s_main);
            let _ = hipMemcpy(
                a.as_mut_ptr().cast::<c_void>(),
                self.dbg.cast::<c_void>(),
                a.len() * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            let _ = hipMemcpy(
                la.as_mut_ptr().cast::<c_void>(),
                self.logits.cast::<c_void>(),
                la.len() * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
            if let Some(out_path) = std::env::var_os("DUMP_RESID_OUT") {
                if !out_path.is_empty() {
                    let bytes = std::slice::from_raw_parts(a.as_ptr().cast::<u8>(), a.len() * 4);
                    let _ = std::fs::write(out_path, bytes);
                }
            }
        }

        // (b) single-token decode steps
        self.reset();
        if !prompt.is_empty() {
            self.process(prompt, true);
        }
        for t in 0..T {
            self.dbg_row = t as i32;
            self.process(&toks[t..t + 1], true);
            unsafe {
                let _ = hipStreamSynchronize(self.s_main);
                let _ = hipMemcpy(
                    lb.as_mut_ptr().add(t * V).cast::<c_void>(),
                    self.logits.cast::<c_void>(),
                    V * 4,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                );
            }
        }
        unsafe {
            let _ = hipMemcpy(
                b.as_mut_ptr().cast::<c_void>(),
                self.dbg.cast::<c_void>(),
                b.len() * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            );
        }

        self.dbg = ptr::null_mut();
        self.opts.use_graph = saved_graph;

        let mut bad = 0usize;
        for l in 0..L {
            for t in 0..T {
                let off = (l * 16 + t) * N;
                let pa = &a[off..off + N];
                let pb = &b[off..off + N];
                if pa
                    .iter()
                    .zip(pb.iter())
                    .any(|(x, y)| x.to_bits() != y.to_bits())
                {
                    let mut md = 0.0f64;
                    for i in 0..N {
                        let d = ((pa[i] - pb[i]) as f64).abs();
                        if d > md {
                            md = d;
                        }
                    }
                    if bad < 12 {
                        let kind = if self.model().layers[l].is_attn {
                            "attention"
                        } else {
                            "GDN"
                        };
                        println!(
                            "exact-test: layer {} ({}) row {} differs, max |d| {:.3e}",
                            l, kind, t, md
                        );
                    }
                    bad += 1;
                }
            }
        }
        let mut lbad = 0usize;
        for t in 0..T {
            let off = t * V;
            if la[off..off + V]
                .iter()
                .zip(lb[off..off + V].iter())
                .any(|(x, y)| x.to_bits() != y.to_bits())
            {
                lbad += 1;
            }
        }
        println!(
            "exact-test: T {} at position {}: {} (layer, row) residuals differ, {} logit rows differ  {}",
            T,
            prompt.len(),
            bad,
            lbad,
            if bad == 0 && lbad == 0 { "ok" } else { "FAIL" }
        );
        bad == 0 && lbad == 0
    }

    /// Run attention WMMA vs scalar reference self-test (`Engine::attn_selftest`).
    pub fn attn_selftest(&mut self) -> bool {
        let g = &self.model().g;
        let H = g.n_head as i32;
        let HKV = g.n_head_kv as i32;
        let HD = g.head_dim as i32;
        let G = H / HKV;
        if !((G == 8 || G == 6) && (HD % 16) == 0) {
            return true;
        }
        let sc = 1.0f32 / (HD as f32).sqrt();
        let max_chunks = (self.ctx_len + self.chunk_sz - 1) / self.chunk_sz;

        let mut rng = 0x1234_5678u64;
        let mut rf = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng & 0xFFFF) as f32 / 32768.0) - 1.0
        };

        let fill_max = 1093usize;
        ensure_selftest_fill(self, fill_max, H, HKV, HD, &mut rf);

        for &fill in &[1i32, 7, 16, 17, 63, 64, 65, 128, 257, 512, 1024] {
            for &T in &[1i32, 2, 3, 4, 8] {
                if (fill + T) as usize > fill_max {
                    continue;
                }
                let p0 = fill - 1;
                unsafe {
                    let _ = hipMemcpy(
                        self.pos0_d.cast::<c_void>(),
                        (&p0 as *const i32).cast::<c_void>(),
                        4,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    );
                }
                let qn = (T * H * HD) as usize;
                let mut hq = vec![0.0f32; qn];
                let mut hgate = vec![0.0f32; qn];
                for i in 0..qn {
                    hq[i] = rf() * 0.5;
                    hgate[i] = rf();
                }
                unsafe {
                    let _ = hipMemcpy(
                        self.q.cast::<c_void>(),
                        hq.as_ptr().cast::<c_void>(),
                        qn * 4,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    );
                    let _ = hipMemcpy(
                        self.gate.cast::<c_void>(),
                        hgate.as_ptr().cast::<c_void>(),
                        qn * 4,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    );
                }

                let _ = tk_q36_attn_part(
                    self.tk.as_ptr(),
                    max_chunks,
                    T * H,
                    self.q,
                    self.kc[0].cast::<c_void>(),
                    self.vc[0].cast::<c_void>(),
                    self.pos0_d,
                    0,
                    H,
                    HKV,
                    self.ctx_len,
                    sc,
                    max_chunks,
                    self.pacc,
                    self.pm,
                    self.pl,
                    ptr::null_mut(),
                );
                let _ = tk_q36_attn_merge(
                    self.tk.as_ptr(),
                    T * H,
                    HD,
                    self.pacc,
                    self.pm,
                    self.pl,
                    self.gate,
                    self.pos0_d,
                    0,
                    H,
                    max_chunks,
                    self.att,
                    self.chunk_sz,
                    ptr::null_mut(),
                    ptr::null_mut(),
                );
                let mut href = vec![0.0f32; qn];
                unsafe {
                    let _ = hipMemcpy(
                        href.as_mut_ptr().cast::<c_void>(),
                        self.att.cast::<c_void>(),
                        qn * 4,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    );
                }

                let _ = tk_q36_attn_wmma(
                    self.tk.as_ptr(),
                    self.attn_w8,
                    self.n_chunk_w,
                    HKV,
                    self.q,
                    self.kc[0].cast::<c_void>(),
                    self.vtc[0].cast::<c_void>(),
                    self.pos0_d,
                    T,
                    H,
                    HKV,
                    self.ctx_len,
                    self.vt_ctx,
                    sc,
                    self.att_ck,
                    self.n_chunk_w,
                    self.pacc,
                    self.pm,
                    self.pl,
                    ptr::null_mut(),
                );
                let _ = tk_q36_attn_merge(
                    self.tk.as_ptr(),
                    T * H,
                    HD,
                    self.pacc,
                    self.pm,
                    self.pl,
                    self.gate,
                    self.pos0_d,
                    0,
                    H,
                    self.n_chunk_w,
                    self.att,
                    self.att_ck,
                    ptr::null_mut(),
                    ptr::null_mut(),
                );
                let mut hgot = vec![0.0f32; qn];
                unsafe {
                    let _ = hipMemcpy(
                        hgot.as_mut_ptr().cast::<c_void>(),
                        self.att.cast::<c_void>(),
                        qn * 4,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    );
                }
                for i in 0..qn {
                    if (href[i] - hgot[i]).abs() > 2e-3 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Run attention microbenchmark (`Engine::attn_bench`).
    pub fn attn_bench(&mut self) {
        let g = &self.model().g;
        let H = g.n_head as i32;
        let HKV = g.n_head_kv as i32;
        let HD = g.head_dim as i32;
        let sc = 1.0f32 / (HD as f32).sqrt();
        let iters = 200;
        for &fill in &[64i32, 256, 512, 1024, 2048, 4096] {
            if fill >= self.ctx_len {
                continue;
            }
            unsafe {
                let _ = hipMemcpy(
                    self.pos0_d.cast::<c_void>(),
                    (&fill as *const i32).cast::<c_void>(),
                    4,
                    HIP_MEMCPY_HOST_TO_DEVICE,
                );
            }
            for &T in &[1i32, 4, 8] {
                let _ = unsafe { hipDeviceSynchronize() };
                for _ in 0..iters {
                    let _ = tk_q36_attn_wmma(
                        self.tk.as_ptr(),
                        self.attn_w8,
                        self.n_chunk_w,
                        HKV,
                        self.q,
                        self.kc[0].cast::<c_void>(),
                        self.vtc[0].cast::<c_void>(),
                        self.pos0_d,
                        T,
                        H,
                        HKV,
                        self.ctx_len,
                        self.vt_ctx,
                        sc,
                        self.att_ck,
                        self.n_chunk_w,
                        self.pacc,
                        self.pm,
                        self.pl,
                        ptr::null_mut(),
                    );
                }
                let _ = unsafe { hipDeviceSynchronize() };
            }
        }
    }

    /// Run MMVQ microbenchmark (`Engine::mmvq_bench`).
    pub fn mmvq_bench(&mut self) {
        let m = unsafe { &*self.m };
        if m.layers.is_empty() {
            return;
        }
        let g = &m.g;
        let E = g.n_embd as i32;
        self.quant(self.h, E, 16);
        let _ = unsafe { hipDeviceSynchronize() };
        let L0 = &m.layers[0];
        if !L0.wqkv.is_null() {
            let w = L0.wqkv;
            for &T in &[1i32, 4, 8] {
                for _ in 0..200 {
                    let _ = unsafe {
                        tk_mmvq(
                            self.tk.as_ptr(),
                            w.tk_qtype(),
                            w.d.cast::<c_void>(),
                            self.x_q8,
                            self.qkv,
                            w.n_in,
                            w.n_out,
                            T,
                            ptr::null_mut(),
                        )
                    };
                }
                let _ = unsafe { hipDeviceSynchronize() };
            }
        }
    }
}

fn ensure_selftest_fill(
    eng: &mut Q36Engine,
    fill_max: usize,
    H: i32,
    HKV: i32,
    HD: i32,
    rf: &mut impl FnMut() -> f32,
) {
    let g = &eng.model().g;
    let QG = g.q_gate_dim() as usize;
    let KV = g.kv_dim() as usize;
    let mut hqg = vec![0.0f32; QG];
    let mut hk = vec![0.0f32; KV];
    let mut hv = vec![0.0f32; KV];
    let qw = eng
        .model()
        .layers
        .iter()
        .find(|l| l.is_attn)
        .map(|l| l.q_norm_ptr)
        .unwrap_or(ptr::null());
    let kw = eng
        .model()
        .layers
        .iter()
        .find(|l| l.is_attn)
        .map(|l| l.k_norm_ptr)
        .unwrap_or(ptr::null());
    if qw.is_null() || kw.is_null() {
        return;
    }
    let limit = fill_max.min(eng.ctx_len as usize);
    for t in 0..limit {
        for v in &mut hqg {
            *v = rf();
        }
        for v in &mut hk {
            *v = rf();
        }
        for v in &mut hv {
            *v = rf();
        }
        let pos = t as i32;
        unsafe {
            let _ = hipMemcpy(
                eng.qg.cast::<c_void>(),
                hqg.as_ptr().cast::<c_void>(),
                QG * 4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
            let _ = hipMemcpy(
                eng.k.cast::<c_void>(),
                hk.as_ptr().cast::<c_void>(),
                KV * 4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
            let _ = hipMemcpy(
                eng.v.cast::<c_void>(),
                hv.as_ptr().cast::<c_void>(),
                KV * 4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
            let _ = hipMemcpy(
                eng.pos0_d.cast::<c_void>(),
                (&pos as *const i32).cast::<c_void>(),
                4,
                HIP_MEMCPY_HOST_TO_DEVICE,
            );
        }
        let _ = tk_q36_attn_prep(
            eng.tk.as_ptr(),
            1,
            eng.qg,
            eng.k,
            eng.v,
            qw,
            kw,
            eng.q,
            eng.gate,
            eng.kc[0].cast::<c_void>(),
            eng.vc[0].cast::<c_void>(),
            H,
            HKV,
            HD,
            g.n_rot as i32,
            eng.pos0_d,
            eng.ctx_len,
            g.rope_freq_base,
            g.rms_norm_eps,
            eng.vtc[0].cast::<c_void>(),
            eng.vt_ctx,
            ptr::null_mut(),
        );
    }
    let _ = unsafe { hipDeviceSynchronize() };
}
