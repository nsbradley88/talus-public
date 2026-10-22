//! Raw FFI bindings and dynamic loader for `include/talus_kernels.h` (Contract v0).
//!
//! Copyright (c) 2026 Niko1221 and the Strata contributors
//! Copyright (c) 2023-2026 The ggml authors
//! Copyright (c) 2026 nsbradley88
//! SPDX-License-Identifier: MIT

#![allow(non_camel_case_types)]

use core::ffi::{c_char, c_int, c_void};
use std::ffi::{CStr, CString};
use std::path::Path;

/// Crate identifier.
pub const CRATE_NAME: &str = "talus-kernels-sys";

/// Contract v0 ABI version macro (`#define TALUS_KERNELS_ABI_VERSION 0`).
pub const TALUS_KERNELS_ABI_VERSION: u32 = 0;

// ============================================================================
// Opaque Handles & Status Types
// ============================================================================

/// Per-device kernel context (`hipBLAS`/`hipBLASLt` handles, algo cache, tuning tables, error buffer).
#[repr(C)]
pub struct tk_ctx {
    _private: [u8; 0],
}

/// Opaque HIP stream handle (`hipStream_t`), created and owned by the host.
pub type tk_stream = *mut c_void;

/// Status return code (`0` == `TK_OK`; non-zero == [`tk_status_code`]).
pub type tk_status = i32;

/// Opaque kernel configuration knob set.
#[repr(C)]
pub struct tk_config {
    _private: [u8; 0],
}

/// Opaque handle to device-resident repacked decode weights (Q8_0 or Q6_K plane layout).
#[repr(C)]
pub struct tk_packed_weights {
    _private: [u8; 0],
}

/// Status codes returned by `tk_*` entry points.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum tk_status_code {
    TK_OK = 0,
    TK_ERR_INVALID_ARG = 1,
    TK_ERR_NULL_PTR = 2,
    TK_ERR_UNSUPPORTED = 3,
    TK_ERR_OUT_OF_MEMORY = 4,
    TK_ERR_HIP = 5,
    TK_ERR_HIPBLAS = 6,
    TK_ERR_IO = 7,
    TK_ERR_EXCEPTION = 8,
    TK_ERR_INTERNAL = 9,
}

impl tk_status_code {
    /// Convert a raw `tk_status` integer into a [`tk_status_code`] if recognized.
    pub const fn from_raw(status: tk_status) -> Option<Self> {
        match status {
            0 => Some(Self::TK_OK),
            1 => Some(Self::TK_ERR_INVALID_ARG),
            2 => Some(Self::TK_ERR_NULL_PTR),
            3 => Some(Self::TK_ERR_UNSUPPORTED),
            4 => Some(Self::TK_ERR_OUT_OF_MEMORY),
            5 => Some(Self::TK_ERR_HIP),
            6 => Some(Self::TK_ERR_HIPBLAS),
            7 => Some(Self::TK_ERR_IO),
            8 => Some(Self::TK_ERR_EXCEPTION),
            9 => Some(Self::TK_ERR_INTERNAL),
            _ => None,
        }
    }
}

pub const TK_OK: tk_status = tk_status_code::TK_OK as tk_status;
pub const TK_ERR_INVALID_ARG: tk_status = tk_status_code::TK_ERR_INVALID_ARG as tk_status;
pub const TK_ERR_NULL_PTR: tk_status = tk_status_code::TK_ERR_NULL_PTR as tk_status;
pub const TK_ERR_UNSUPPORTED: tk_status = tk_status_code::TK_ERR_UNSUPPORTED as tk_status;
pub const TK_ERR_OUT_OF_MEMORY: tk_status = tk_status_code::TK_ERR_OUT_OF_MEMORY as tk_status;
pub const TK_ERR_HIP: tk_status = tk_status_code::TK_ERR_HIP as tk_status;
pub const TK_ERR_HIPBLAS: tk_status = tk_status_code::TK_ERR_HIPBLAS as tk_status;
pub const TK_ERR_IO: tk_status = tk_status_code::TK_ERR_IO as tk_status;
pub const TK_ERR_EXCEPTION: tk_status = tk_status_code::TK_ERR_EXCEPTION as tk_status;
pub const TK_ERR_INTERNAL: tk_status = tk_status_code::TK_ERR_INTERNAL as tk_status;

// ============================================================================
// Target Architecture & Quantization Enums
// ============================================================================

/// GPU architecture family for tuning table and kernel selection.
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum tk_arch {
    /// AMD Strix Halo (RDNA3.5, wave32, WMMA gfx11, unified memory).
    TK_ARCH_GFX1151 = 0,
    /// AMD Radeon AI PRO R9700 / RDNA4 (wave32, WMMA gfx12, FP8).
    TK_ARCH_GFX1201 = 1,
    /// Any other GCN/CDNA/RDNA target (portable fallback path).
    TK_ARCH_OTHER = 2,
}

pub use tk_arch::*;

/// Tensor quantization and element types.
///
/// Enumerator values match GGML / Strata type IDs (`include/strata/artifact/gguf_reader.hpp`).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum tk_qtype {
    TK_QTYPE_F32 = 0,
    TK_QTYPE_F16 = 1,
    TK_QTYPE_Q4_0 = 2,
    TK_QTYPE_Q4_1 = 3,
    TK_QTYPE_Q5_0 = 6,
    TK_QTYPE_Q5_1 = 7,
    TK_QTYPE_Q8_0 = 8,
    TK_QTYPE_Q8_1 = 9,
    TK_QTYPE_Q2_K = 10,
    TK_QTYPE_Q3_K = 11,
    TK_QTYPE_Q4_K = 12,
    TK_QTYPE_Q5_K = 13,
    TK_QTYPE_Q6_K = 14,
    TK_QTYPE_Q8_K = 15,
    TK_QTYPE_IQ2_XXS = 16,
    TK_QTYPE_IQ2_XS = 17,
    TK_QTYPE_IQ3_XXS = 18,
    TK_QTYPE_IQ1_S = 19,
    TK_QTYPE_IQ4_NL = 20,
    TK_QTYPE_IQ3_S = 21,
    TK_QTYPE_IQ2_S = 22,
    TK_QTYPE_IQ4_XS = 23,
    TK_QTYPE_I8 = 24,
    TK_QTYPE_IQ1_M = 29,
    TK_QTYPE_BF16 = 30,
    TK_QTYPE_TQ1_0 = 34,
    TK_QTYPE_TQ2_0 = 35,
    TK_QTYPE_MXFP4 = 39,
    TK_QTYPE_NVFP4 = 40,
    TK_QTYPE_Q1_0 = 41,
    TK_QTYPE_Q2_0 = 42,
}

pub use tk_qtype::*;

// ============================================================================
// Descriptor Structs for Batched / Grouped Operations
// ============================================================================

/// Single work item for [`tk_mmvq_batch`] (mirrors `strata::kernels::MmvqItem`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_mmvq_item {
    pub qtype: tk_qtype,
    pub w: *const c_void,
    pub expert_bytes: usize,
    pub ids_dev: *const i32,
    pub n_ids: i32,
    pub x_q8_1: *const c_void,
    pub x_per_id: i32,
    pub x_group: i32,
    pub y: *mut f32,
    pub expert_major: i32,
    pub n_in: i32,
    pub n_out: i32,
}

/// Grouped MoE MMQ product descriptor for [`tk_prefill_moe_mmq`] (mirrors `strata::prefill::mmq::Product`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_moe_mmq_product {
    pub w: *const c_void,
    pub qtype: tk_qtype,
    pub w_rows: i64,
    pub w_cols: i64,
    pub expert_bytes: usize,
    pub n_experts: i32,
    pub x_q8_1: *const c_void,
    pub bounds_dev: *const i32,
    pub ids_dev: *const i32,
    pub total_rows: i64,
    pub max_rows: i64,
    pub dst: *mut f32,
    pub ld_dst: i64,
    pub y_scale_dev: *const f32,
}

/// Device-resident Gumbel-max + penalty sampling descriptor for `tk_q36_argmax_samp` / `tk_q36_lse`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_q36_draft_samp {
    pub tau: *const f32,
    pub lmin: *const f32,
    pub top_p: *const f32,
    pub top_k: *const i32,
    pub seed: *const u32,
    pub ctr: *const u32,
    pub ring: *const i32,
    pub r_len: *const i32,
    pub r_pos: *const i32,
    pub pen_rep: *const f32,
    pub pen_freq: *const f32,
    pub pen_pres: *const f32,
    pub r_cap: i32,
    pub draft: *const i32,
    pub map: *const i32,
}

/// Pending device-side GDN state-commit descriptor for `tk_q36_conv_seq` / `tk_q36_conv_ab` / `tk_q36_gdn_seq`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct tk_q36_gdn_pend {
    pub q: [*const f32; 2],
    pub h: [*const f32; 2],
    pub a: [*const f32; 2],
    pub b: [*const f32; 2],
    pub ctl: *const i32,
    pub store: i32,
}

// ============================================================================
// Extern "C" Entry Points (Contract v0)
// ============================================================================

extern "C" {
    // 1. Context & Configuration Entry Points
    pub fn tk_config_create(out_cfg: *mut *mut tk_config) -> tk_status;
    pub fn tk_config_create_for_arch(arch: tk_arch, out_cfg: *mut *mut tk_config) -> tk_status;
    pub fn tk_config_from_env(cfg: *mut tk_config) -> tk_status;
    pub fn tk_config_destroy(cfg: *mut tk_config);
    pub fn tk_config_set_int(cfg: *mut tk_config, key: *const c_char, value: i64) -> tk_status;
    pub fn tk_config_get_int(
        cfg: *const tk_config,
        key: *const c_char,
        out_value: *mut i64,
    ) -> tk_status;
    pub fn tk_ctx_create(
        device: i32,
        cfg: *const tk_config,
        out_ctx: *mut *mut tk_ctx,
    ) -> tk_status;
    pub fn tk_ctx_destroy(ctx: *mut tk_ctx);
    pub fn tk_ctx_device(ctx: *const tk_ctx, out_device: *mut i32) -> tk_status;
    pub fn tk_ctx_arch(ctx: *const tk_ctx, out_arch: *mut tk_arch) -> tk_status;
    pub fn tk_ctx_load_tuning(ctx: *mut tk_ctx, dir: *const c_char) -> tk_status;
    pub fn tk_last_error(ctx: *const tk_ctx) -> *const c_char;
    pub fn tk_status_string(status: tk_status) -> *const c_char;

    // 2. Packing & Activation Quantization
    pub fn tk_quantize_q8_1(
        ctx: *mut tk_ctx,
        x: *const f32,
        x_q8_1: *mut c_void,
        n_in: i32,
        ncols: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_pack_q8(
        ctx: *mut tk_ctx,
        w_gguf: *const c_void,
        n_in: i32,
        n_out: i32,
        stream: tk_stream,
        out_packed: *mut *mut tk_packed_weights,
    ) -> tk_status;
    pub fn tk_pack_q6(
        ctx: *mut tk_ctx,
        w_gguf: *const c_void,
        n_in: i32,
        n_out: i32,
        stream: tk_stream,
        out_packed: *mut *mut tk_packed_weights,
    ) -> tk_status;
    pub fn tk_packed_free(ctx: *mut tk_ctx, packed: *mut tk_packed_weights);

    // 3. MMVQ & MoE Decode / Verify Entry Points
    pub fn tk_mmvq(
        ctx: *mut tk_ctx,
        qtype: tk_qtype,
        w: *const c_void,
        x_q8_1: *const c_void,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        ncols: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_mmvq_packed(
        ctx: *mut tk_ctx,
        packed: *const tk_packed_weights,
        x_q8_1: *const c_void,
        y: *mut f32,
        ncols: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_mmvq_ids(
        ctx: *mut tk_ctx,
        qtype: tk_qtype,
        w: *const c_void,
        expert_bytes: usize,
        ids_dev: *const i32,
        n_ids: i32,
        x_q8_1: *const c_void,
        x_per_id: bool,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        w2: *const c_void,
        y2: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_f32_matvec(
        ctx: *mut tk_ctx,
        w: *const f32,
        x: *const f32,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_f32_matvec_fast(
        ctx: *mut tk_ctx,
        w: *const f32,
        x: *const f32,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_f32_matvec_rows(
        ctx: *mut tk_ctx,
        w: *const f32,
        x: *const f32,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_f32_matvec_rows_n(
        ctx: *mut tk_ctx,
        w: *const f32,
        x: *const f32,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_f32_matmul(
        ctx: *mut tk_ctx,
        w: *const f32,
        x: *const f32,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_route_sigmoid_topk(
        ctx: *mut tk_ctx,
        logits: *const f32,
        bias: *const f32,
        ids_dev: *mut i32,
        weights_dev: *mut f32,
        n_expert: i32,
        k: i32,
        scale: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_route_warp(
        ctx: *mut tk_ctx,
        logits: *const f32,
        bias: *const f32,
        ids_dev: *mut i32,
        weights_dev: *mut f32,
        k: i32,
        scale: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_route_topk(
        ctx: *mut tk_ctx,
        logits: *const f32,
        bias: *const f32,
        n_tok: i32,
        k: i32,
        scale: f32,
        ids_dev: *mut i32,
        weights_dev: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_route_group(
        ctx: *mut tk_ctx,
        logits: *const f32,
        bias: *const f32,
        n_tok: i32,
        k: i32,
        scale: f32,
        sel_e_dev: *mut i32,
        sel_w_dev: *mut f32,
        cnt_dev: *mut i32,
        bounds_dev: *mut i32,
        rtok_dev: *mut i32,
        smap_dev: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_router_route(
        ctx: *mut tk_ctx,
        w: *const f32,
        x: *const f32,
        logits: *mut f32,
        bias: *const f32,
        ids_dev: *mut i32,
        weights_dev: *mut f32,
        n_in: i32,
        n_expert: i32,
        k: i32,
        scale: f32,
        counter_dev: *mut u32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_moe_add_slots(
        ctx: *mut tk_ctx,
        x: *mut f32,
        shared: *const f32,
        y: *const f32,
        map_dev: *const i32,
        weights_dev: *const f32,
        n_tok: i32,
        n_embd: i32,
        k: i32,
        stream: tk_stream,
    ) -> tk_status;

    // 4. Norms, RoPE, Elementwise, Embedding & Sampling Helpers
    pub fn tk_rms_norm(
        ctx: *mut tk_ctx,
        x: *mut f32,
        w: *const f32,
        rows: i64,
        cols: i64,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_rms_norm_to(
        ctx: *mut tk_ctx,
        dst: *mut f32,
        src: *const f32,
        w: *const f32,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_head_norm_rope(
        ctx: *mut tk_ctx,
        x: *mut f32,
        w: *const f32,
        rows: i32,
        eps: f32,
        n_rot: i32,
        cos_tab: *const f32,
        sin_tab: *const f32,
        pos_dev: *const i32,
        pos_div: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_qkv_norm_rope_store(
        ctx: *mut tk_ctx,
        q: *mut f32,
        kc: *const f32,
        vc: *const f32,
        q_norm: *const f32,
        k_norm: *const f32,
        n_head: i32,
        n_kv: i32,
        eps: f32,
        n_rot: i32,
        cos_tab: *const f32,
        sin_tab: *const f32,
        pos_dev: *const i32,
        k_pool: *mut c_void,
        v_pool: *mut c_void,
        vt_stride: i32,
        k_scale: *mut u16,
        v_scale: *mut u16,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_rope_neox(
        ctx: *mut tk_ctx,
        x: *const f32,
        out: *mut f32,
        rows: i64,
        head_dim: i32,
        n_rot: i32,
        cos_tab: *const f32,
        sin_tab: *const f32,
        pos_dev: *const i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_mrope(
        ctx: *mut tk_ctx,
        x: *const f32,
        out: *mut f32,
        rows: i64,
        head_dim: i32,
        n_rot: i32,
        sections: *const i32,
        cos_tab: *const f32,
        sin_tab: *const f32,
        pos_dev: *const i32,
        mrope_table_dev: *const i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_silu_mul(
        ctx: *mut tk_ctx,
        gate: *mut f32,
        up: *const f32,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_axpy(
        ctx: *mut tk_ctx,
        y: *mut f32,
        x: *const f32,
        alpha: f32,
        n: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_add_norm_quant(
        ctx: *mut tk_ctx,
        x: *mut f32,
        add: *const f32,
        yexp: *const f32,
        weights_dev: *const f32,
        k: i32,
        norm_w: *const f32,
        h_out: *mut f32,
        q8_out: *mut c_void,
        n: i32,
        eps: f32,
        n_rows: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_embed_q8_0(
        ctx: *mut tk_ctx,
        table: *const c_void,
        tok_dev: *const i32,
        x: *mut f32,
        n_embd: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_gather_rows(
        ctx: *mut tk_ctx,
        src: *const c_void,
        dst: *mut c_void,
        idx_dev: *const i32,
        n_rows: i32,
        row_bytes: usize,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_fill_pos(
        ctx: *mut tk_ctx,
        pos_dev: *mut i32,
        n_tok: i32,
        per_tok: i32,
        p0: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_fill_pos_dev(
        ctx: *mut tk_ctx,
        out_dev: *mut i32,
        n: i32,
        pos_dev: *const i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_argmax_advance(
        ctx: *mut tk_ctx,
        logits: *const f32,
        n: i32,
        tok_dev: *mut i32,
        pos_dev: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_argmax_rows(
        ctx: *mut tk_ctx,
        logits: *const f32,
        n_rows: i32,
        n_vocab: i32,
        out_dev: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_softplus_head_gate(
        ctx: *mut tk_ctx,
        o: *mut f32,
        g: *const f32,
        n_head: i32,
        head_dim: i32,
        stream: tk_stream,
    ) -> tk_status;

    // 5. Attention & KV Cache
    pub fn tk_attn_decode(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const c_void,
        v_pool: *const c_void,
        k_scale: *const u16,
        v_scale: *const u16,
        out: *mut f32,
        scratch: *mut f32,
        n_tok: i32,
        n_head: i32,
        n_kv: i32,
        head_dim: i32,
        head_dim_v: i32,
        kv_stride: i32,
        window: i32,
        scale: f32,
        pos_dev: *const i32,
        n_split: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_attn_prefill_fa(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const c_void,
        v_pool: *const c_void,
        out: *mut f32,
        n_tok: i32,
        n_head: i32,
        n_kv: i32,
        head_dim: i32,
        kv_stride: i64,
        p0: i64,
        scale: f32,
        mask: *mut c_void,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_laguna_prompt_attn(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const u16,
        v_pool: *const u16,
        out: *mut f32,
        nq: i32,
        q_pos0: i32,
        n_head: i32,
        n_kv: i32,
        window: i32,
        q_pos0_dev: *const i32,
        n_split: i32,
        part_acc: *mut f32,
        part_ml: *mut f32,
        merge: bool,
        vt_stride: i32,
        k_scale: *const u16,
        v_scale: *const u16,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_laguna_prompt_attn_wmma(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const u16,
        v_pool: *const u16,
        out: *mut f32,
        nq: i32,
        q_pos0: i32,
        n_head: i32,
        n_kv: i32,
        window: i32,
        vt_stride: i32,
        q_pos0_dev: *const i32,
        n_split: i32,
        part_acc: *mut f32,
        part_ml: *mut f32,
        k_scale: *const u16,
        v_scale: *const u16,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_laguna_decode_attn(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const u16,
        v_pool: *const u16,
        k_scale: *const u16,
        v_scale: *const u16,
        n_head: i32,
        n_kv: i32,
        window: i32,
        stride: i32,
        pos_dev: *const i32,
        n_split: i32,
        part_acc: *mut f32,
        part_ml: *mut f32,
        n_tok: i32,
        sel_dev: *const i32,
        sel_k: i32,
        sel_per_kv: bool,
        sel_bs: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_attn_merge_gate_quant(
        ctx: *mut tk_ctx,
        part_acc: *const f32,
        part_ml: *const f32,
        g: *const f32,
        q8_out: *mut c_void,
        n_head: i32,
        n_split: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_kv_store_f16(
        ctx: *mut tk_ctx,
        k_pool: *mut u16,
        v_pool: *mut u16,
        kc: *const f32,
        vc: *const f32,
        n: i32,
        pos_dev: *const i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_v_store_t(
        ctx: *mut tk_ctx,
        vc: *const f32,
        v_pool: *mut u16,
        nb: i32,
        n_kv: i32,
        p0: i32,
        stride: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_kv_store_q8(
        ctx: *mut tk_ctx,
        kc: *const f32,
        vc: *const f32,
        k_pool: *mut c_void,
        v_pool: *mut c_void,
        k_scale: *mut u16,
        v_scale: *mut u16,
        nb: i32,
        n_kv: i32,
        p0: i32,
        stride: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_kbound_update(
        ctx: *mut tk_ctx,
        k_pool: *const c_void,
        k_scale: *const u16,
        bnd: *mut c_void,
        nb_max: i32,
        n_kv: i32,
        b_first: i32,
        b_last: i32,
        pos_last: i32,
        pos_dev: *const i32,
        n_tok: i32,
        bs: i32,
        f16: bool,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_sparse_scores(
        ctx: *mut tk_ctx,
        q: *const f32,
        bnd: *const c_void,
        ub: *mut f32,
        nb_max: i32,
        n_head: i32,
        n_kv: i32,
        pos_dev: *const i32,
        n_tok: i32,
        bs: i32,
        f16: bool,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_sparse_select(
        ctx: *mut tk_ctx,
        ub: *const f32,
        nb_max: i32,
        n_head: i32,
        pos_dev: *const i32,
        n_tok: i32,
        k: i32,
        sink: i32,
        recent: i32,
        sel_dev: *mut i32,
        groups: i32,
        bs: i32,
        scr_dev: *mut u32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_attn_swa_sinks(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const c_void,
        v_pool: *const c_void,
        k_scale: *const u16,
        v_scale: *const u16,
        sinks: *const f32,
        out: *mut f32,
        scratch: *mut f32,
        n_tok: i32,
        n_head: i32,
        n_kv: i32,
        head_dim: i32,
        head_dim_v: i32,
        window: i32,
        stride: i32,
        scale: f32,
        pos_dev: *const i32,
        n_split: i32,
        stream: tk_stream,
    ) -> tk_status;
    /// tk_attn_swa_sinks that also writes the wo projection's MMQ q8_1 rows where the prefill v2 kernel runs.
    pub fn tk_attn_swa_sinks_mmq(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const c_void,
        v_pool: *const c_void,
        sinks: *const f32,
        out: *mut f32,
        scratch: *mut f32,
        n_tok: i32,
        n_head: i32,
        n_kv: i32,
        head_dim: i32,
        head_dim_v: i32,
        window: i32,
        stride: i32,
        scale: f32,
        pos_dev: *const i32,
        q8_mmq: *mut c_void,
        q8_layout: i32,
        q8_written: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_attn_swa_sinks_q8(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const c_void,
        v_pool: *const c_void,
        k_scale: *const u16,
        v_scale: *const u16,
        sinks: *const f32,
        out: *mut f32,
        scratch: *mut f32,
        n_tok: i32,
        n_head: i32,
        n_kv: i32,
        head_dim: i32,
        head_dim_v: i32,
        window: i32,
        stride: i32,
        scale: f32,
        pos_dev: *const i32,
        n_split: i32,
        stream: tk_stream,
        q8_out: *mut c_void,
    ) -> tk_status;
    pub fn tk_attn_k192_v128(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const c_void,
        v_pool: *const c_void,
        k_scale: *const u16,
        v_scale: *const u16,
        out: *mut f32,
        scratch: *mut f32,
        n_tok: i32,
        n_head: i32,
        n_kv: i32,
        stride: i32,
        scale: f32,
        pos_dev: *const i32,
        n_split: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_attn_k192_v128_q8(
        ctx: *mut tk_ctx,
        q: *const f32,
        k_pool: *const c_void,
        v_pool: *const c_void,
        k_scale: *const u16,
        v_scale: *const u16,
        out: *mut f32,
        scratch: *mut f32,
        n_tok: i32,
        n_head: i32,
        n_kv: i32,
        stride: i32,
        scale: f32,
        pos_dev: *const i32,
        n_split: i32,
        stream: tk_stream,
        q8_out: *mut c_void,
    ) -> tk_status;
    pub fn tk_axpy_norm_quant(
        ctx: *mut tk_ctx,
        x: *mut f32,
        ax: *const f32,
        alpha: f32,
        norm_w: *const f32,
        h_out: *mut f32,
        q8_out: *mut c_void,
        n: i32,
        eps: f32,
        n_rows: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_mimo_rope_qk_store(
        ctx: *mut tk_ctx,
        q: *mut f32,
        k: *mut f32,
        v: *const f32,
        n_tok: i32,
        n_head_q: i32,
        n_head_kv: i32,
        head_dim: i32,
        v_dim: i32,
        n_rot: i32,
        cos_tab: *const f32,
        sin_tab: *const f32,
        pos_q_dev: *const i32,
        pos_k_dev: *const i32,
        k_pool: *mut c_void,
        v_pool: *mut c_void,
        store_pos_dev: *const i32,
        ring: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_dsa_decode_indexer(
        ctx: *mut tk_ctx,
        q_idx: *const f32,
        k_idx_pool: *const c_void,
        weights: *const f32,
        pos_dev: *const i32,
        n_tok: i32,
        idx_heads: i32,
        idx_dim: i32,
        top_k: i32,
        capacity: i32,
        scores_scratch: *mut f32,
        topk_ids_dev: *mut i32,
        stream: tk_stream,
    ) -> tk_status;

    // 6. GDN & KDA Recurrent Layers
    pub fn tk_gdn_preprocess(
        ctx: *mut tk_ctx,
        conv_state: *mut f32,
        qkv: *const f32,
        conv_w: *const f32,
        conv_out: *mut f32,
        alpha: *const f32,
        dt: *const f32,
        ssm_a: *const f32,
        gate_out: *mut f32,
        beta_inout: *mut f32,
        channels: i64,
        d_conv: i64,
        h_k: i64,
        h_v: i64,
        state_size: i64,
        l2_eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_gdn_step(
        ctx: *mut tk_ctx,
        state: *mut f32,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        out: *mut f32,
        state_size: i64,
        h_k: i64,
        h_v: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_gdn_seq(
        ctx: *mut tk_ctx,
        state: *mut f32,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        z: *const f32,
        gamma: *const f32,
        out: *mut f32,
        n_tok: i32,
        state_size: i64,
        h_k: i64,
        h_v: i64,
        norm_eps: f32,
        silu_gate: bool,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_gdn_fused(
        ctx: *mut tk_ctx,
        state: *mut f32,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gate: *const f32,
        beta: *const f32,
        z: *const f32,
        gamma: *const f32,
        eps: f32,
        y: *mut f32,
        h_k: i32,
        h_v: i32,
        silu_gate: bool,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_kda_step(
        ctx: *mut tk_ctx,
        state: *mut f32,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        gk: *const f32,
        beta: *const f32,
        z: *const f32,
        gamma: *const f32,
        eps: f32,
        out: *mut f32,
        n_tok: i32,
        state_size: i32,
        h_k: i32,
        h_v: i32,
        stream: tk_stream,
    ) -> tk_status;

    // 7. Prefill GEMM & MMQ Entry Points
    pub fn tk_prefill_gemm(
        ctx: *mut tk_ctx,
        w_qtype: tk_qtype,
        w: *const c_void,
        x: *const c_void,
        x_qtype: tk_qtype,
        y: *mut f32,
        n_tok: i64,
        n_out: i64,
        n_in: i64,
        ldy: i64,
        ldx: i64,
        beta: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_prefill_mmq(
        ctx: *mut tk_ctx,
        w_qtype: tk_qtype,
        w_blocks: *const c_void,
        x_q8_1: *const c_void,
        y: *mut f32,
        n_tok: i64,
        n_out: i64,
        n_in: i64,
        ldy: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_prefill_moe_mmq(
        ctx: *mut tk_ctx,
        product: *const tk_moe_mmq_product,
        stream: tk_stream,
    ) -> tk_status;
    /// fused grouped gate/up + SwiGLU (`talus_kernels.h`); `TK_ERR_UNSUPPORTED` = run the separate path.
    pub fn tk_prefill_moe_mmq_gu(
        ctx: *mut tk_ctx,
        gate: *const tk_moe_mmq_product,
        w_up: *const c_void,
        swiglu_limit: f32,
        stream: tk_stream,
    ) -> tk_status;
    /// fused gate/up whose SwiGLU rows go straight into the down projection's q8_1 activations.
    pub fn tk_prefill_moe_mmq_gu_q8(
        ctx: *mut tk_ctx,
        gate: *const tk_moe_mmq_product,
        w_up: *const c_void,
        swiglu_limit: f32,
        down_qtype: i32,
        xq_out: *mut c_void,
        xq_rows: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_prefill_moe_mmq_iq(
        ctx: *mut tk_ctx,
        gu_qtype: tk_qtype,
        down_qtype: tk_qtype,
        expert_blobs: *const *const c_void,
        e0: i32,
        e1: i32,
        n_expert: i32,
        n_routed_rows: i64,
        n_embd: i64,
        n_ff: i64,
        group_scratch: *const c_void,
        x_act: *const c_void,
        src_rows_dev: *const i32,
        h_act: *mut c_void,
        down_out: *mut f32,
        stream: tk_stream,
    ) -> tk_status;

    // 8. Additional Batch / Multi-Matrix MMVQ & Quantization Helpers
    pub fn tk_q8_1_bytes(n_in: i32, ncols: i32) -> usize;
    pub fn tk_swiglu_quantize_q8_1(
        ctx: *mut tk_ctx,
        gate: *const f32,
        up: *const f32,
        x_q8_1: *mut c_void,
        n_in: i32,
        ncols: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_mmvq_w32_multi(
        ctx: *mut tk_ctx,
        qtype: tk_qtype,
        n_mats: i32,
        w_ptrs: *const *const c_void,
        y_ptrs: *const *mut f32,
        n_out_per_mat: *const i32,
        x_q8_1: *const c_void,
        n_in: i32,
        ncols: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_mmvq_batch(
        ctx: *mut tk_ctx,
        items: *const tk_mmvq_item,
        n_items: i32,
        stream: tk_stream,
    ) -> tk_status;

    // 9. q36 / Qwen3.5 / Qwen3.8 Entry Points (WP-1.1 A1)
    pub fn tk_q36_rms(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_rms_q8(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        q8: *mut c_void,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_rms_ld(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *const f32,
        ldx: i32,
        w: *const f32,
        out: *mut f32,
        ld_out: i32,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_rms_ld_q8(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *const f32,
        ldx: i32,
        w: *const f32,
        out: *mut f32,
        ld_out: i32,
        q8: *mut c_void,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_add_rms(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *mut f32,
        y: *const f32,
        w: *const f32,
        h: *mut f32,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_add_rms_q8(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *mut f32,
        y: *const f32,
        w: *const f32,
        h: *mut f32,
        q8: *mut c_void,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_add_rms_exact_q8(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *mut f32,
        y: *const f32,
        w: *const f32,
        h: *mut f32,
        q8: *mut c_void,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_moe_out(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *mut f32,
        parts: *const f32,
        prow: *const i32,
        wts: *const f32,
        k: i32,
        shared: *const f32,
        sgate: *const f32,
        sg_stride: i32,
        wn: *const f32,
        h: *mut f32,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_moe_out_q8(
        ctx: *mut tk_ctx,
        rows: i32,
        x: *mut f32,
        parts: *const f32,
        prow: *const i32,
        wts: *const f32,
        k: i32,
        shared: *const f32,
        sgate: *const f32,
        sg_stride: i32,
        wn: *const f32,
        h: *mut f32,
        q8: *mut c_void,
        n: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_ab(
        ctx: *mut tk_ctx,
        n_tok: i32,
        wa: *const f32,
        wb: *const f32,
        x: *const f32,
        dt: *const f32,
        ssm_a: *const f32,
        alpha: *mut f32,
        beta: *mut f32,
        n_in: i32,
        hv: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_ab2(
        ctx: *mut tk_ctx,
        wa: *const f32,
        wb: *const f32,
        x: *const f32,
        dt: *const f32,
        ssm_a: *const f32,
        alpha: *mut f32,
        beta: *mut f32,
        n_in: i32,
        hv: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_add(
        ctx: *mut tk_ctx,
        x: *mut f32,
        y: *const f32,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_swiglu(
        ctx: *mut tk_ctx,
        g: *const f32,
        u: *const f32,
        h: *mut f32,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gemv_f32(
        ctx: *mut tk_ctx,
        n_tok: i32,
        w: *const f32,
        x: *const f32,
        y: *mut f32,
        n_in: i32,
        n_out: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_gb(
        ctx: *mut tk_ctx,
        alpha: *mut f32,
        dt: *const f32,
        ssm_a: *const f32,
        beta: *mut f32,
        hv: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_gb2(
        ctx: *mut tk_ctx,
        ab: *const f32,
        dt: *const f32,
        ssm_a: *const f32,
        alpha: *mut f32,
        beta: *mut f32,
        hv: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gate_rows(
        ctx: *mut tk_ctx,
        sh: *mut f32,
        gate: *const f32,
        n: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_attn_prep(
        ctx: *mut tk_ctx,
        n_tok: i32,
        qg: *const f32,
        kx: *const f32,
        vx: *const f32,
        qw: *const f32,
        kw: *const f32,
        q: *mut f32,
        gate: *mut f32,
        kc: *mut c_void,
        vc: *mut c_void,
        h: i32,
        hkv: i32,
        hd: i32,
        n_rot: i32,
        pos0: *const i32,
        ctx_len: i32,
        base: f32,
        eps: f32,
        vt: *mut c_void,
        vt_ctx: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_attn_part(
        ctx: *mut tk_ctx,
        grid_x: i32,
        grid_y: i32,
        q: *const f32,
        kc: *const c_void,
        vc: *const c_void,
        pos0: *const i32,
        t0: i32,
        h: i32,
        hkv: i32,
        ctx_len: i32,
        scale: f32,
        n_chunk: i32,
        pacc: *mut f32,
        pm: *mut f32,
        pl: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_attn_merge(
        ctx: *mut tk_ctx,
        rows: i32,
        hd: i32,
        pacc: *const f32,
        pm: *const f32,
        pl: *const f32,
        gate: *const f32,
        pos0: *const i32,
        t0: i32,
        h: i32,
        n_chunk: i32,
        out: *mut f32,
        chunk: i32,
        q8: *mut c_void,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_attn_wmma(
        ctx: *mut tk_ctx,
        w8: bool,
        grid_x: i32,
        grid_y: i32,
        q: *const f32,
        kc: *const c_void,
        vt: *const c_void,
        pos0: *const i32,
        n_tok: i32,
        h: i32,
        hkv: i32,
        ctx_len: i32,
        vt_ctx: i32,
        sc: f32,
        chunk: i32,
        n_chunk: i32,
        pacc: *mut f32,
        pm: *mut f32,
        pl: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_group(
        ctx: *mut tk_ctx,
        ids: *const i32,
        n_ent: i32,
        k: i32,
        n_expert: i32,
        blobs: *const u8,
        blob: u64,
        grp_ptr: *mut u64,
        grp_start: *mut i32,
        n_groups: *mut i32,
        ent_dst: *mut i32,
        ent_tok: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_group_hy(
        ctx: *mut tk_ctx,
        ids: *const i32,
        n_ent: i32,
        k: i32,
        n_expert: i32,
        slot_of: *const i32,
        blobs: *const u8,
        blob: u64,
        grp_ptr: *mut u64,
        grp_start: *mut i32,
        n_groups: *mut i32,
        ent_dst: *mut i32,
        ent_tok: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_merge_cpu(
        ctx: *mut tk_ctx,
        n_ent: i32,
        parts: *mut f32,
        cpu: *const f32,
        miss: *const i32,
        n: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_attn_softmax16(
        ctx: *mut tk_ctx,
        h: i32,
        s_in: *const f32,
        s16_out: *mut c_void,
        ld: i32,
        n_tok: i32,
        p_cols: i32,
        pos0: i32,
        scale: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gate_mul(
        ctx: *mut tk_ctx,
        att: *mut f32,
        gate: *const f32,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_route_count(
        ctx: *mut tk_ctx,
        ids: *const i32,
        n_ent: i32,
        cnt: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_route_scan(
        ctx: *mut tk_ctx,
        cnt: *const i32,
        n_expert: i32,
        bounds: *mut i32,
        cur: *mut i32,
        max_rows: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_route_scatter(
        ctx: *mut tk_ctx,
        ids: *const i32,
        n_ent: i32,
        k: i32,
        cur: *mut i32,
        dst: *mut i32,
        src: *mut i32,
        inv: *mut i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_argmax(
        ctx: *mut tk_ctx,
        grid_x: i32,
        grid_y: i32,
        x: *const f32,
        n: i32,
        out: *mut i32,
        pv: *mut f32,
        pi: *mut i32,
        stage2: bool,
        map: *const i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_argmax_samp(
        ctx: *mut tk_ctx,
        x: *const f32,
        n: i32,
        pv: *mut f32,
        pi: *mut i32,
        ds_dev: *const tk_q36_draft_samp,
        step: i32,
        thr: *const f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_lse(
        ctx: *mut tk_ctx,
        x: *const f32,
        n: i32,
        ds_dev: *const tk_q36_draft_samp,
        out: *mut f32,
        thr_out: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_ctl_end(
        ctx: *mut tk_ctx,
        ctl: *mut i32,
        flip: bool,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_round_prep(
        ctx: *mut tk_ctx,
        rctl: *const i32,
        tok: *mut i32,
        tok_m: *mut i32,
        pos0: *mut i32,
        pos0_m: *mut i32,
        gctl: *mut i32,
        n: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_row_copy(
        ctx: *mut tk_ctx,
        dst: *mut f32,
        src: *const f32,
        idx: *const i32,
        n: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_chain_prep(
        ctx: *mut tk_ctx,
        tok: *const i32,
        tok_m: *mut i32,
        pos0: *const i32,
        pos0_m: *mut i32,
        k: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_sgemm_nt(
        ctx: *mut tk_ctx,
        tm: i32,
        grid_x: i32,
        grid_y: i32,
        grid_z: i32,
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        k_dim: i32,
        n_out: i32,
        n_tok: i32,
        ldo: i32,
        kz: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_sum_splits(
        ctx: *mut tk_ctx,
        part: *const f32,
        out: *mut f32,
        n: i64,
        splits: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_to_f16(
        ctx: *mut tk_ctx,
        x: *const f32,
        y_f16: *mut c_void,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_conv_seq(
        ctx: *mut tk_ctx,
        hist: *mut f32,
        qkv: *const f32,
        w: *const f32,
        h: *mut f32,
        c_channels: i32,
        qk_heads: i32,
        eps: f32,
        n_tok: i32,
        write_state: bool,
        pd: *const tk_q36_gdn_pend,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_conv_ab(
        ctx: *mut tk_ctx,
        hist: *mut f32,
        qkv: *const f32,
        w: *const f32,
        hc: *mut f32,
        c_channels: i32,
        qk_heads: i32,
        eps: f32,
        n_tok: i32,
        write_state: bool,
        pd: *const tk_q36_gdn_pend,
        wa: *const f32,
        wb: *const f32,
        x: *const f32,
        dt: *const f32,
        ssm_a: *const f32,
        alpha: *mut f32,
        beta: *mut f32,
        n_in: i32,
        hv: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_conv_par(
        ctx: *mut tk_ctx,
        hist: *const f32,
        qkv: *const f32,
        w: *const f32,
        h: *mut f32,
        c_channels: i32,
        qk_heads: i32,
        eps: f32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_conv_hist(
        ctx: *mut tk_ctx,
        hist: *mut f32,
        qkv: *const f32,
        c_channels: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_seq(
        ctx: *mut tk_ctx,
        state: *mut f32,
        h: *const f32,
        c_channels: i32,
        gate: *const f32,
        beta: *const f32,
        z: *const f32,
        gamma: *const f32,
        eps: f32,
        y: *mut f32,
        h_k: i32,
        h_v: i32,
        n_tok: i32,
        write_state: bool,
        pd: *const tk_q36_gdn_pend,
        yq: *mut c_void,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_seq_cols(
        ctx: *mut tk_ctx,
        state: *mut f32,
        h: *const f32,
        c_channels: i32,
        gate: *const f32,
        beta: *const f32,
        o_raw: *mut f32,
        h_k: i32,
        h_v: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_seq_warp(
        ctx: *mut tk_ctx,
        ncol: i32,
        state: *mut f32,
        h: *const f32,
        c_channels: i32,
        gate: *const f32,
        beta: *const f32,
        o_raw: *mut f32,
        h_k: i32,
        h_v: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_gdn_norm(
        ctx: *mut tk_ctx,
        n_tok: i32,
        o_raw: *const f32,
        z: *const f32,
        gamma: *const f32,
        eps: f32,
        y: *mut f32,
        h_v: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_flush_read(
        ctx: *mut tk_ctx,
        p: *const c_void,
        n: usize,
        out: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_draft_conf(
        ctx: *mut tk_ctx,
        x: *const f32,
        n: i32,
        tau: *const f32,
        lmin: *const f32,
        pick: *const i32,
        map: *const i32,
        conf: *mut f32,
        top: *mut f32,
        step: i32,
        lse: *const f32,
        thr: *const f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_draft_support(
        ctx: *mut tk_ctx,
        x: *const f32,
        n: i32,
        tau: *const f32,
        lmin: *const f32,
        top_p: *const f32,
        top_k: *const i32,
        lse: *mut f32,
        thr: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_attn_chunks(window: i32) -> i32;
    pub fn tk_q36_dflash_attn_scratch_floats(heads: i32, hd: i32, window: i32) -> usize;
    pub fn tk_q36_dflash_rms_rows(
        ctx: *mut tk_ctx,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n: i32,
        rows: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_rms_rows_q8(
        ctx: *mut tk_ctx,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        q8: *mut c_void,
        n: i32,
        rows: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_sum5_rms(
        ctx: *mut tk_ctx,
        ys: *const *const f32,
        w: *const f32,
        out: *mut f32,
        n: i32,
        rows: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_sum5_rms_q8(
        ctx: *mut tk_ctx,
        ys: *const *const f32,
        w: *const f32,
        out: *mut f32,
        q8: *mut c_void,
        n: i32,
        rows: i32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_dyn_conv(
        ctx: *mut tk_ctx,
        x_in: *const f32,
        dyn_w: *const f32,
        base_w: *const f32,
        out: *mut f32,
        n: i32,
        rows: i32,
        group: i32,
        side: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_dyn_conv_q8(
        ctx: *mut tk_ctx,
        x_in: *const f32,
        dyn_w: *const f32,
        base_w: *const f32,
        out: *mut f32,
        q8: *mut c_void,
        n: i32,
        rows: i32,
        group: i32,
        side: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_qk_norm_rope(
        ctx: *mut tk_ctx,
        x: *mut f32,
        w: *const f32,
        heads: i32,
        hd: i32,
        rows: i32,
        pos: *const i32,
        theta: f32,
        eps: f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_kv_store(
        ctx: *mut tk_ctx,
        k: *const f32,
        v: *const f32,
        kring: *mut f32,
        vring: *mut f32,
        width: i32,
        rows: i32,
        pos: *const i32,
        ring: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_kv_store_f16(
        ctx: *mut tk_ctx,
        k: *const f32,
        v: *const f32,
        kring_f16: *mut c_void,
        vring_f16: *mut c_void,
        k_width: i32,
        v_width: i32,
        rows: i32,
        pos: *const i32,
        ring: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_block_attn(
        ctx: *mut tk_ctx,
        q: *const f32,
        kring: *const f32,
        vring: *const f32,
        pos: *const i32,
        heads: i32,
        heads_kv: i32,
        hd: i32,
        ring: i32,
        window: i32,
        scale: f32,
        part: *mut f32,
        out: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_swiglu(
        ctx: *mut tk_ctx,
        g: *const f32,
        u: *const f32,
        out: *mut f32,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_swiglu_q8(
        ctx: *mut tk_ctx,
        g: *const f32,
        u: *const f32,
        out: *mut f32,
        q8: *mut c_void,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_add_inplace(
        ctx: *mut tk_ctx,
        x: *mut f32,
        y: *const f32,
        n: i64,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_block_tokens(
        ctx: *mut tk_ctx,
        anchor: *const i32,
        out: *mut i32,
        n: i32,
        mask: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_set_i32(
        ctx: *mut tk_ctx,
        p: *mut i32,
        v: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_top16(
        ctx: *mut tk_ctx,
        logits: *const f32,
        nv: i32,
        rows: i32,
        map: *const i32,
        cand: *mut i32,
        unary: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_select_chain(
        ctx: *mut tk_ctx,
        cand: *const i32,
        unary: *const f32,
        gate: *const f32,
        pred: *const c_void,
        succ: *const c_void,
        anchor: *const i32,
        out: *mut i32,
        score: *mut f32,
        n: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_dflash_select_chain_q(
        ctx: *mut tk_ctx,
        cand: *const i32,
        unary: *const f32,
        gate: *const f32,
        pred: *const c_void,
        succ: *const c_void,
        anchor: *const i32,
        out: *mut i32,
        score: *mut f32,
        q: *mut f32,
        tau: *const f32,
        seed: *const u32,
        ctr: *const u32,
        n: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_mmvq_supported(ggml_type: i32) -> bool;
    pub fn tk_q36_q8_0_pack_should(ggml_type: i32, n_in: i32, n_out: i32) -> bool;
    pub fn tk_q36_q8_0_pack_host(
        gguf_blocks: *const c_void,
        out: *mut c_void,
        n_in: i32,
        n_out: i32,
    );
    pub fn tk_q36_q8_0_packed_register(
        gguf_weights: *const c_void,
        packed: *const c_void,
        n_in: i32,
        n_out: i32,
    );
    pub fn tk_q36_q8_0_packed_unregister(gguf_weights: *const c_void);
    pub fn tk_q36_embed_rows(
        ctx: *mut tk_ctx,
        ggml_type: i32,
        table: *const c_void,
        row_bytes: usize,
        tok_dev: *const i32,
        n_tok: i64,
        n_embd: i64,
        out: *mut f32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_mmq_built() -> bool;
    pub fn tk_q36_mmq_supported_fits(ggml_type: i32, w_rows: i64) -> bool;
    pub fn tk_q36_mmq_q8_layout(ggml_type: i32) -> i32;
    pub fn tk_q36_mmq_q8_bytes(rows: i64, cols: i64) -> usize;
    pub fn tk_q36_mmq_matrix_bytes(ggml_type: i32, rows: i64, cols: i64) -> usize;
    pub fn tk_q36_mmq_iota(ctx: *mut tk_ctx, dst: *mut i32, n: i64, stream: tk_stream)
        -> tk_status;
    pub fn tk_q36_mmq_set_bounds(
        ctx: *mut tk_ctx,
        bounds: *mut i32,
        rows: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_mmq_quantize(
        ctx: *mut tk_ctx,
        x: *const f32,
        ids: *const i32,
        xq: *mut c_void,
        ggml_type: i32,
        cols: i64,
        ld: i64,
        rows: i64,
        stream: tk_stream,
    ) -> tk_status;
    /// tk_q36_mmq_quantize of routed rows via the token -> rows map (one quantize per token row).
    pub fn tk_q36_mmq_quantize_scatter(
        ctx: *mut tk_ctx,
        x: *const f32,
        smap: *const i32,
        xq: *mut c_void,
        ggml_type: i32,
        cols: i64,
        n_tok: i64,
        n_rows: i64,
        k: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_mmq_swiglu(
        ctx: *mut tk_ctx,
        gu: *const f32,
        h: *mut f32,
        rows: i64,
        n_ff: i64,
        interleaved: bool,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_fattn_supported(head_dim: i32) -> bool;
    pub fn tk_q36_fattn_mask_bytes(n_tok: i64, n_kv: i64) -> usize;
    pub fn tk_q36_prefill_init_scratch(ctx: *mut tk_ctx, scratch_elems: i64) -> tk_status;
    pub fn tk_q36_attn_blas_prefill(
        ctx: *mut tk_ctx,
        q: *const f32,
        q16: *mut c_void,
        kc: *const c_void,
        vc: *const c_void,
        att_s: *mut f32,
        att_p16: *mut c_void,
        att: *mut f32,
        gate: *const f32,
        n_tok: i32,
        h: i32,
        hkv: i32,
        hd: i32,
        ctx_len: i32,
        p0: i64,
        score_floats: usize,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_router_topk(
        ctx: *mut tk_ctx,
        logits: *const f32,
        ids: *mut i32,
        weights: *mut f32,
        n_expert: i32,
        k: i32,
        n_tok: i32,
        stream: tk_stream,
    ) -> tk_status;
    pub fn tk_q36_expert_supported(gu_type: i32, d_type: i32, n_embd: i64, n_ff: i64) -> bool;
    pub fn tk_q36_expert_layout(
        gu_type: i32,
        d_type: i32,
        n_embd: i64,
        n_ff: i64,
        out_gu_row: *mut usize,
        out_d_row: *mut usize,
        out_up_off: *mut usize,
        out_down_off: *mut usize,
        out_bytes: *mut usize,
        out_tail_off: *mut usize,
    );
    pub fn tk_q36_expert_scratch_bytes(cap_entries: i64, n_ff: i64) -> usize;
    pub fn tk_q36_expert_grouped(
        ctx: *mut tk_ctx,
        gu_type: i32,
        d_type: i32,
        n_embd: i64,
        n_ff: i64,
        grp_ptr: *const u64,
        grp_start: *const i32,
        n_groups: *const i32,
        ent_dst: *const i32,
        ent_tok: *const i32,
        cap_groups: i64,
        cap_entries: i64,
        x_q8_1: *const c_void,
        scratch: *mut c_void,
        out: *mut f32,
        grid_groups: i64,
        stream: tk_stream,
    ) -> tk_status;

    // 10. Talus prefill Prompt Attention, Expanded INT8 & W4A4 Entry Points
    pub fn talus_pf_fa_prefill(
        q: *const f32,
        k: *const c_void,
        vt: *const c_void,
        gate: *const f32,
        out: *mut f32,
        T: i32,
        H: i32,
        HKV: i32,
        D: i32,
        kv_stride: i64,
        vt_cells: i64,
        p0: i64,
        scale: f32,
        stream: *mut c_void,
    ) -> c_int;
    pub fn talus_pf_i8x_expand(
        w: *const c_void,
        ggml_type: i32,
        n: i32,
        k: i32,
        qs: *mut i8,
        ws: *mut f32,
        stream: *mut c_void,
    ) -> c_int;
    pub fn talus_pf_i8gemm(
        qs: *const i8,
        ws: *const f32,
        n: i32,
        k: i32,
        mode: i32,
        xq: *const c_void,
        T: i32,
        dst: *mut f32,
        ld_dst: i64,
        stream: *mut c_void,
    ) -> c_int;
    pub fn talus_pf_moe_gemm(
        w: *const c_void,
        ggml_type: i32,
        expert_bytes: usize,
        n: i32,
        k: i32,
        n_experts: i32,
        xq: *const c_void,
        bounds: *const i32,
        ids: *const i32,
        total_rows: i64,
        max_rows: i64,
        dst: *mut f32,
        ld_dst: i64,
        stream: *mut c_void,
    ) -> c_int;
    pub fn talus_pf_swiglu_q8_mmq(
        g: *const f32,
        u: *const f32,
        xq: *mut c_void,
        ggml_type: i32,
        cols: i64,
        rows: i64,
        stream: *mut c_void,
    ) -> c_int;
    pub fn talus_pf_i8gemm_swiglu_q8(
        qs: *const i8,
        ws: *const f32,
        n: i32,
        k: i32,
        mode: i32,
        xq: *const c_void,
        T: i32,
        g: *const f32,
        ldg: i64,
        yq: *mut c_void,
        layout: i32,
        stream: *mut c_void,
    ) -> c_int;
    pub fn talus_pf_w4_pack(
        x: *const f32,
        T: i32,
        k: i32,
        ldx: i64,
        signs: *const i8,
        xq: *mut u8,
        xs: *mut f32,
        stream: *mut c_void,
    ) -> c_int;
    pub fn talus_pf_w4_gemm(
        wq: *const u8,
        ws: *const f32,
        n: i32,
        k: i32,
        xq: *const u8,
        xs: *const f32,
        T: i32,
        dst: *mut f32,
        ld_dst: i64,
        stream: *mut c_void,
    ) -> c_int;

    // 11. QSA tile attention (fnp_qsa_tile_attn) Prompt Tile Attention Entry Points
    pub fn fnp_qsa_tile_attn_i8(
        q: *const f32,
        k_q: *const i8,
        v_q: *const i8,
        k_scale: *const u16,
        v_scale: *const u16,
        page_table: *const i32,
        page_size: i64,
        ids: *const i32,
        steps: *const i32,
        cap: i64,
        attn: *mut f32,
        n_q: i64,
        scratch: *mut c_void,
        stream: *mut c_void,
    ) -> c_int;
    pub fn fnp_qsa_tile_scratch_bytes_c(n_q: i64, cap: i64) -> u64;
}

// ============================================================================
// Dynamic Symbol Table Loader (`LibTalusKernels`)
// ============================================================================

#[cfg(unix)]
const RTLD_NOW: i32 = 2;

#[cfg(unix)]
#[link(name = "dl")]
extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
    fn dlerror() -> *const c_char;
}

/// Dynamically loaded function table for `libtalus_kernels.so` (Contract v0).
///
/// Allows `talus-kernels` to resolve all 75 entry points at runtime via `dlopen`/`dlsym`
/// (or at build time when feature `link-kernels` is enabled) without requiring
/// `libtalus_kernels.so` to be linked during `cargo test` on CPU-only build hosts.
pub struct LibTalusKernels {
    handle: *mut c_void,

    // 1. Context & Configuration
    pub tk_config_create: unsafe extern "C" fn(*mut *mut tk_config) -> tk_status,
    pub tk_config_create_for_arch: unsafe extern "C" fn(tk_arch, *mut *mut tk_config) -> tk_status,
    pub tk_config_from_env: unsafe extern "C" fn(*mut tk_config) -> tk_status,
    pub tk_config_destroy: unsafe extern "C" fn(*mut tk_config),
    pub tk_config_set_int: unsafe extern "C" fn(*mut tk_config, *const c_char, i64) -> tk_status,
    pub tk_config_get_int:
        unsafe extern "C" fn(*const tk_config, *const c_char, *mut i64) -> tk_status,
    pub tk_ctx_create: unsafe extern "C" fn(i32, *const tk_config, *mut *mut tk_ctx) -> tk_status,
    pub tk_ctx_destroy: unsafe extern "C" fn(*mut tk_ctx),
    pub tk_ctx_device: unsafe extern "C" fn(*const tk_ctx, *mut i32) -> tk_status,
    pub tk_ctx_arch: unsafe extern "C" fn(*const tk_ctx, *mut tk_arch) -> tk_status,
    pub tk_ctx_load_tuning: unsafe extern "C" fn(*mut tk_ctx, *const c_char) -> tk_status,
    pub tk_last_error: unsafe extern "C" fn(*const tk_ctx) -> *const c_char,
    pub tk_status_string: unsafe extern "C" fn(tk_status) -> *const c_char,

    // 2. Packing & Activation Quantization
    pub tk_quantize_q8_1: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *mut c_void,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_pack_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const c_void,
        i32,
        i32,
        tk_stream,
        *mut *mut tk_packed_weights,
    ) -> tk_status,
    pub tk_pack_q6: unsafe extern "C" fn(
        *mut tk_ctx,
        *const c_void,
        i32,
        i32,
        tk_stream,
        *mut *mut tk_packed_weights,
    ) -> tk_status,
    pub tk_packed_free: unsafe extern "C" fn(*mut tk_ctx, *mut tk_packed_weights),

    // 3. MMVQ & MoE Decode / Verify
    pub tk_mmvq: unsafe extern "C" fn(
        *mut tk_ctx,
        tk_qtype,
        *const c_void,
        *const c_void,
        *mut f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_mmvq_packed: unsafe extern "C" fn(
        *mut tk_ctx,
        *const tk_packed_weights,
        *const c_void,
        *mut f32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_mmvq_ids: unsafe extern "C" fn(
        *mut tk_ctx,
        tk_qtype,
        *const c_void,
        usize,
        *const i32,
        i32,
        *const c_void,
        bool,
        *mut f32,
        i32,
        i32,
        *const c_void,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_f32_matvec: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_f32_matvec_fast: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_f32_matvec_rows: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_f32_matvec_rows_n: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_f32_matmul: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_route_sigmoid_topk: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut i32,
        *mut f32,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_route_warp: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut i32,
        *mut f32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_route_topk: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        i32,
        i32,
        f32,
        *mut i32,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_route_group: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        i32,
        i32,
        f32,
        *mut i32,
        *mut f32,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        tk_stream,
    ) -> tk_status,
    pub tk_router_route: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        *const f32,
        *mut i32,
        *mut f32,
        i32,
        i32,
        i32,
        f32,
        *mut u32,
        tk_stream,
    ) -> tk_status,
    pub tk_moe_add_slots: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *const i32,
        *const f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,

    // 4. Norms, RoPE, Elementwise, Embedding & Sampling
    pub tk_rms_norm: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        i64,
        i64,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_rms_norm_to: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_head_norm_rope: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        i32,
        f32,
        i32,
        *const f32,
        *const f32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_qkv_norm_rope_store: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        i32,
        i32,
        f32,
        i32,
        *const f32,
        *const f32,
        *const i32,
        *mut c_void,
        *mut c_void,
        i32,
        *mut u16,
        *mut u16,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_rope_neox: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *mut f32,
        i64,
        i32,
        i32,
        *const f32,
        *const f32,
        *const i32,
        tk_stream,
    ) -> tk_status,
    pub tk_mrope: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *mut f32,
        i64,
        i32,
        i32,
        *const i32,
        *const f32,
        *const f32,
        *const i32,
        *const i32,
        tk_stream,
    ) -> tk_status,
    pub tk_silu_mul:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, i64, tk_stream) -> tk_status,
    pub tk_axpy:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, f32, i32, tk_stream) -> tk_status,
    pub tk_add_norm_quant: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        f32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_embed_q8_0: unsafe extern "C" fn(
        *mut tk_ctx,
        *const c_void,
        *const i32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_gather_rows: unsafe extern "C" fn(
        *mut tk_ctx,
        *const c_void,
        *mut c_void,
        *const i32,
        i32,
        usize,
        tk_stream,
    ) -> tk_status,
    pub tk_fill_pos:
        unsafe extern "C" fn(*mut tk_ctx, *mut i32, i32, i32, i32, tk_stream) -> tk_status,
    pub tk_fill_pos_dev:
        unsafe extern "C" fn(*mut tk_ctx, *mut i32, i32, *const i32, tk_stream) -> tk_status,
    pub tk_argmax_advance: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        i32,
        *mut i32,
        *mut i32,
        tk_stream,
    ) -> tk_status,
    pub tk_argmax_rows:
        unsafe extern "C" fn(*mut tk_ctx, *const f32, i32, i32, *mut i32, tk_stream) -> tk_status,
    pub tk_softplus_head_gate:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, i32, i32, tk_stream) -> tk_status,

    // 5. Attention & KV Cache
    pub tk_attn_decode: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const c_void,
        *const u16,
        *const u16,
        *mut f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_attn_prefill_fa: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const c_void,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i64,
        i64,
        f32,
        *mut c_void,
        tk_stream,
    ) -> tk_status,
    pub tk_laguna_prompt_attn: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const u16,
        *const u16,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const i32,
        i32,
        *mut f32,
        *mut f32,
        bool,
        i32,
        *const u16,
        *const u16,
        tk_stream,
    ) -> tk_status,
    pub tk_laguna_prompt_attn_wmma: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const u16,
        *const u16,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const i32,
        i32,
        *mut f32,
        *mut f32,
        *const u16,
        *const u16,
        tk_stream,
    ) -> tk_status,
    pub tk_laguna_decode_attn: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const u16,
        *const u16,
        *const u16,
        *const u16,
        i32,
        i32,
        i32,
        i32,
        *const i32,
        i32,
        *mut f32,
        *mut f32,
        i32,
        *const i32,
        i32,
        bool,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_attn_merge_gate_quant: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *const f32,
        *mut c_void,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_kv_store_f16: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut u16,
        *mut u16,
        *const f32,
        *const f32,
        i32,
        *const i32,
        tk_stream,
    ) -> tk_status,
    pub tk_v_store_t: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *mut u16,
        i32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_kv_store_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut c_void,
        *mut c_void,
        *mut u16,
        *mut u16,
        i32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_kbound_update: unsafe extern "C" fn(
        *mut tk_ctx,
        *const c_void,
        *const u16,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const i32,
        i32,
        i32,
        bool,
        tk_stream,
    ) -> tk_status,
    pub tk_sparse_scores: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *mut f32,
        i32,
        i32,
        i32,
        *const i32,
        i32,
        i32,
        bool,
        tk_stream,
    ) -> tk_status,
    pub tk_sparse_select: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        i32,
        i32,
        *const i32,
        i32,
        i32,
        i32,
        i32,
        *mut i32,
        i32,
        i32,
        *mut u32,
        tk_stream,
    ) -> tk_status,
    pub tk_attn_swa_sinks: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const c_void,
        *const u16,
        *const u16,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_attn_swa_sinks_mmq: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const c_void,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *const i32,
        *mut c_void,
        i32,
        *mut i32,
        tk_stream,
    ) -> tk_status,
    pub tk_attn_swa_sinks_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const c_void,
        *const u16,
        *const u16,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *const i32,
        i32,
        tk_stream,
        *mut c_void,
    ) -> tk_status,
    pub tk_attn_k192_v128: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const c_void,
        *const u16,
        *const u16,
        *mut f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_attn_k192_v128_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const c_void,
        *const u16,
        *const u16,
        *mut f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *const i32,
        i32,
        tk_stream,
        *mut c_void,
    ) -> tk_status,
    pub tk_axpy_norm_quant: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        f32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_mimo_rope_qk_store: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *mut f32,
        *const f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        *const i32,
        *const i32,
        *mut c_void,
        *mut c_void,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_dsa_decode_indexer: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const c_void,
        *const f32,
        *const i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        *mut f32,
        *mut i32,
        tk_stream,
    ) -> tk_status,

    // 6. GDN & KDA Recurrent Layers
    pub tk_gdn_preprocess: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        i64,
        i64,
        i64,
        i64,
        i64,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_gdn_step: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        i64,
        i64,
        i64,
        tk_stream,
    ) -> tk_status,
    pub tk_gdn_seq: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i64,
        i64,
        i64,
        f32,
        bool,
        tk_stream,
    ) -> tk_status,
    pub tk_gdn_fused: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        f32,
        *mut f32,
        i32,
        i32,
        bool,
        tk_stream,
    ) -> tk_status,
    pub tk_kda_step: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,

    // 7. Prefill GEMM & MMQ
    pub tk_prefill_gemm: unsafe extern "C" fn(
        *mut tk_ctx,
        tk_qtype,
        *const c_void,
        *const c_void,
        tk_qtype,
        *mut f32,
        i64,
        i64,
        i64,
        i64,
        i64,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_prefill_mmq: unsafe extern "C" fn(
        *mut tk_ctx,
        tk_qtype,
        *const c_void,
        *const c_void,
        *mut f32,
        i64,
        i64,
        i64,
        i64,
        tk_stream,
    ) -> tk_status,
    pub tk_prefill_moe_mmq:
        unsafe extern "C" fn(*mut tk_ctx, *const tk_moe_mmq_product, tk_stream) -> tk_status,
    pub tk_prefill_moe_mmq_gu: unsafe extern "C" fn(
        *mut tk_ctx,
        *const tk_moe_mmq_product,
        *const c_void,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_prefill_moe_mmq_gu_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const tk_moe_mmq_product,
        *const c_void,
        f32,
        i32,
        *mut c_void,
        i64,
        tk_stream,
    ) -> tk_status,
    pub tk_prefill_moe_mmq_iq: unsafe extern "C" fn(
        *mut tk_ctx,
        tk_qtype,
        tk_qtype,
        *const *const c_void,
        i32,
        i32,
        i32,
        i64,
        i64,
        i64,
        *const c_void,
        *const c_void,
        *const i32,
        *mut c_void,
        *mut f32,
        tk_stream,
    ) -> tk_status,

    // 8. Additional Batch / Multi-Matrix MMVQ & Quantization Helpers
    pub tk_q8_1_bytes: unsafe extern "C" fn(i32, i32) -> usize,
    pub tk_swiglu_quantize_q8_1: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut c_void,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_mmvq_w32_multi: unsafe extern "C" fn(
        *mut tk_ctx,
        tk_qtype,
        i32,
        *const *const c_void,
        *const *mut f32,
        *const i32,
        *const c_void,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_mmvq_batch:
        unsafe extern "C" fn(*mut tk_ctx, *const tk_mmvq_item, i32, tk_stream) -> tk_status,

    // 9. q36 / Qwen3.5 / Qwen3.8 Entry Points (WP-1.1 A1)
    pub tk_q36_rms: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_rms_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_rms_ld: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_rms_ld_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
        *mut c_void,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_add_rms: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *mut f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_add_rms_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *mut f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_add_rms_exact_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *mut f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_moe_out: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *mut f32,
        *const f32,
        *const i32,
        *const f32,
        i32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_moe_out_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *mut f32,
        *const f32,
        *const i32,
        *const f32,
        i32,
        *const f32,
        *const f32,
        i32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gdn_ab: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gdn_ab2: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_add:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, i64, tk_stream) -> tk_status,
    pub tk_q36_swiglu: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i64,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gemv_f32: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gdn_gb: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gdn_gb2: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gate_rows:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, i32, i32, tk_stream) -> tk_status,
    pub tk_q36_attn_prep: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        *mut c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        *const i32,
        i32,
        f32,
        f32,
        *mut c_void,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_attn_part: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        i32,
        *const f32,
        *const c_void,
        *const c_void,
        *const i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        i32,
        *mut f32,
        *mut f32,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_attn_merge: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        i32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const i32,
        i32,
        i32,
        i32,
        *mut f32,
        i32,
        *mut c_void,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_attn_wmma: unsafe extern "C" fn(
        *mut tk_ctx,
        bool,
        i32,
        i32,
        *const f32,
        *const c_void,
        *const c_void,
        *const i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        i32,
        i32,
        *mut f32,
        *mut f32,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_group: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        i32,
        i32,
        i32,
        *const u8,
        u64,
        *mut u64,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_group_hy: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        i32,
        i32,
        i32,
        *const i32,
        *const u8,
        u64,
        *mut u64,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_merge_cpu: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *mut f32,
        *const f32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_attn_softmax16: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gate_mul:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, i64, tk_stream) -> tk_status,
    pub tk_q36_route_count:
        unsafe extern "C" fn(*mut tk_ctx, *const i32, i32, *mut i32, tk_stream) -> tk_status,
    pub tk_q36_route_scan: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        i32,
        *mut i32,
        *mut i32,
        *mut i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_route_scatter: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        i32,
        i32,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_argmax: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        i32,
        *const f32,
        i32,
        *mut i32,
        *mut f32,
        *mut i32,
        bool,
        *const i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_argmax_samp: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        i32,
        *mut f32,
        *mut i32,
        *const tk_q36_draft_samp,
        i32,
        *const f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_lse: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        i32,
        *const tk_q36_draft_samp,
        *mut f32,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_ctl_end: unsafe extern "C" fn(*mut tk_ctx, *mut i32, bool, tk_stream) -> tk_status,
    pub tk_q36_round_prep: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_row_copy: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_chain_prep: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        *mut i32,
        *const i32,
        *mut i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_sgemm_nt: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        i32,
        i32,
        i32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_sum_splits:
        unsafe extern "C" fn(*mut tk_ctx, *const f32, *mut f32, i64, i32, tk_stream) -> tk_status,
    pub tk_q36_to_f16:
        unsafe extern "C" fn(*mut tk_ctx, *const f32, *mut c_void, i64, tk_stream) -> tk_status,
    pub tk_q36_conv_seq: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        f32,
        i32,
        bool,
        *const tk_q36_gdn_pend,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_conv_ab: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        f32,
        i32,
        bool,
        *const tk_q36_gdn_pend,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_conv_par: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        f32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_conv_hist:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, i32, i32, tk_stream) -> tk_status,
    pub tk_q36_gdn_seq: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        i32,
        *const f32,
        *const f32,
        *const f32,
        *const f32,
        f32,
        *mut f32,
        i32,
        i32,
        i32,
        bool,
        *const tk_q36_gdn_pend,
        *mut c_void,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gdn_seq_cols: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        i32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gdn_seq_warp: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *mut f32,
        *const f32,
        i32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_gdn_norm: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const f32,
        *const f32,
        *const f32,
        f32,
        *mut f32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_flush_read:
        unsafe extern "C" fn(*mut tk_ctx, *const c_void, usize, *mut f32, tk_stream) -> tk_status,
    pub tk_q36_draft_conf: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        i32,
        *const f32,
        *const f32,
        *const i32,
        *const i32,
        *mut f32,
        *mut f32,
        i32,
        *const f32,
        *const f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_draft_support: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        i32,
        *const f32,
        *const f32,
        *const f32,
        *const i32,
        *mut f32,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_attn_chunks: unsafe extern "C" fn(i32) -> i32,
    pub tk_q36_dflash_attn_scratch_floats: unsafe extern "C" fn(i32, i32, i32) -> usize,
    pub tk_q36_dflash_rms_rows: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_rms_rows_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_sum5_rms: unsafe extern "C" fn(
        *mut tk_ctx,
        *const *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_sum5_rms_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const *const f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        i32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_dyn_conv: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_dyn_conv_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_qk_norm_rope: unsafe extern "C" fn(
        *mut tk_ctx,
        *mut f32,
        *const f32,
        i32,
        i32,
        i32,
        *const i32,
        f32,
        f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_kv_store: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        *mut f32,
        i32,
        i32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_kv_store_f16: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        *const i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_block_attn: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *const f32,
        *const i32,
        i32,
        i32,
        i32,
        i32,
        i32,
        f32,
        *mut f32,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_swiglu: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        i64,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_swiglu_q8: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const f32,
        *mut f32,
        *mut c_void,
        i64,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_add_inplace:
        unsafe extern "C" fn(*mut tk_ctx, *mut f32, *const f32, i64, tk_stream) -> tk_status,
    pub tk_q36_dflash_block_tokens:
        unsafe extern "C" fn(*mut tk_ctx, *const i32, *mut i32, i32, i32, tk_stream) -> tk_status,
    pub tk_q36_dflash_set_i32:
        unsafe extern "C" fn(*mut tk_ctx, *mut i32, i32, tk_stream) -> tk_status,
    pub tk_q36_dflash_top16: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        i32,
        i32,
        *const i32,
        *mut i32,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_select_chain: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        *const f32,
        *const f32,
        *const c_void,
        *const c_void,
        *const i32,
        *mut i32,
        *mut f32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_dflash_select_chain_q: unsafe extern "C" fn(
        *mut tk_ctx,
        *const i32,
        *const f32,
        *const f32,
        *const c_void,
        *const c_void,
        *const i32,
        *mut i32,
        *mut f32,
        *mut f32,
        *const f32,
        *const u32,
        *const u32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_mmvq_supported: unsafe extern "C" fn(i32) -> bool,
    pub tk_q36_q8_0_pack_should: unsafe extern "C" fn(i32, i32, i32) -> bool,
    pub tk_q36_q8_0_pack_host: unsafe extern "C" fn(*const c_void, *mut c_void, i32, i32),
    pub tk_q36_q8_0_packed_register: unsafe extern "C" fn(*const c_void, *const c_void, i32, i32),
    pub tk_q36_q8_0_packed_unregister: unsafe extern "C" fn(*const c_void),
    pub tk_q36_embed_rows: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        *const c_void,
        usize,
        *const i32,
        i64,
        i64,
        *mut f32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_mmq_built: unsafe extern "C" fn() -> bool,
    pub tk_q36_mmq_supported_fits: unsafe extern "C" fn(i32, i64) -> bool,
    pub tk_q36_mmq_q8_layout: unsafe extern "C" fn(i32) -> i32,
    pub tk_q36_mmq_q8_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub tk_q36_mmq_matrix_bytes: unsafe extern "C" fn(i32, i64, i64) -> usize,
    pub tk_q36_mmq_iota: unsafe extern "C" fn(*mut tk_ctx, *mut i32, i64, tk_stream) -> tk_status,
    pub tk_q36_mmq_set_bounds:
        unsafe extern "C" fn(*mut tk_ctx, *mut i32, i32, tk_stream) -> tk_status,
    pub tk_q36_mmq_quantize: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const i32,
        *mut c_void,
        i32,
        i64,
        i64,
        i64,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_mmq_quantize_scatter: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *const i32,
        *mut c_void,
        i32,
        i64,
        i64,
        i64,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_mmq_swiglu: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *mut f32,
        i64,
        i64,
        bool,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_fattn_supported: unsafe extern "C" fn(i32) -> bool,
    pub tk_q36_fattn_mask_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub tk_q36_prefill_init_scratch: unsafe extern "C" fn(*mut tk_ctx, i64) -> tk_status,
    pub tk_q36_attn_blas_prefill: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *mut c_void,
        *const c_void,
        *const c_void,
        *mut f32,
        *mut c_void,
        *mut f32,
        *const f32,
        i32,
        i32,
        i32,
        i32,
        i32,
        i64,
        usize,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_router_topk: unsafe extern "C" fn(
        *mut tk_ctx,
        *const f32,
        *mut i32,
        *mut f32,
        i32,
        i32,
        i32,
        tk_stream,
    ) -> tk_status,
    pub tk_q36_expert_supported: unsafe extern "C" fn(i32, i32, i64, i64) -> bool,
    pub tk_q36_expert_layout: unsafe extern "C" fn(
        i32,
        i32,
        i64,
        i64,
        *mut usize,
        *mut usize,
        *mut usize,
        *mut usize,
        *mut usize,
        *mut usize,
    ),
    pub tk_q36_expert_scratch_bytes: unsafe extern "C" fn(i64, i64) -> usize,
    pub tk_q36_expert_grouped: unsafe extern "C" fn(
        *mut tk_ctx,
        i32,
        i32,
        i64,
        i64,
        *const u64,
        *const i32,
        *const i32,
        *const i32,
        *const i32,
        i64,
        i64,
        *const c_void,
        *mut c_void,
        *mut f32,
        i64,
        tk_stream,
    ) -> tk_status,

    // 10. Talus prefill Prompt Attention, Expanded INT8 & W4A4 Entry Points
    pub talus_pf_fa_prefill: unsafe extern "C" fn(
        *const f32,
        *const c_void,
        *const c_void,
        *const f32,
        *mut f32,
        i32,
        i32,
        i32,
        i32,
        i64,
        i64,
        i64,
        f32,
        *mut c_void,
    ) -> c_int,
    pub talus_pf_i8x_expand:
        unsafe extern "C" fn(*const c_void, i32, i32, i32, *mut i8, *mut f32, *mut c_void) -> c_int,
    pub talus_pf_i8gemm: unsafe extern "C" fn(
        *const i8,
        *const f32,
        i32,
        i32,
        i32,
        *const c_void,
        i32,
        *mut f32,
        i64,
        *mut c_void,
    ) -> c_int,
    pub talus_pf_moe_gemm: unsafe extern "C" fn(
        *const c_void,
        i32,
        usize,
        i32,
        i32,
        i32,
        *const c_void,
        *const i32,
        *const i32,
        i64,
        i64,
        *mut f32,
        i64,
        *mut c_void,
    ) -> c_int,
    pub talus_pf_swiglu_q8_mmq: unsafe extern "C" fn(
        *const f32,
        *const f32,
        *mut c_void,
        i32,
        i64,
        i64,
        *mut c_void,
    ) -> c_int,
    pub talus_pf_i8gemm_swiglu_q8: unsafe extern "C" fn(
        *const i8,
        *const f32,
        i32,
        i32,
        i32,
        *const c_void,
        i32,
        *const f32,
        i64,
        *mut c_void,
        i32,
        *mut c_void,
    ) -> c_int,
    pub talus_pf_w4_pack: unsafe extern "C" fn(
        *const f32,
        i32,
        i32,
        i64,
        *const i8,
        *mut u8,
        *mut f32,
        *mut c_void,
    ) -> c_int,
    pub talus_pf_w4_gemm: unsafe extern "C" fn(
        *const u8,
        *const f32,
        i32,
        i32,
        *const u8,
        *const f32,
        i32,
        *mut f32,
        i64,
        *mut c_void,
    ) -> c_int,

    // 11. QSA tile attention (fnp_qsa_tile_attn) Prompt Tile Attention Entry Points
    pub fnp_qsa_tile_attn_i8: unsafe extern "C" fn(
        *const f32,
        *const i8,
        *const i8,
        *const u16,
        *const u16,
        *const i32,
        i64,
        *const i32,
        *const i32,
        i64,
        *mut f32,
        i64,
        *mut c_void,
        *mut c_void,
    ) -> c_int,
    pub fnp_qsa_tile_scratch_bytes_c: unsafe extern "C" fn(i64, i64) -> u64,
}

unsafe extern "C" fn fallback_talus_pf_moe_gemm(
    _w: *const c_void,
    _ggml_type: i32,
    _expert_bytes: usize,
    _n: i32,
    _k: i32,
    _n_experts: i32,
    _xq: *const c_void,
    _bounds: *const i32,
    _ids: *const i32,
    _total_rows: i64,
    _max_rows: i64,
    _dst: *mut f32,
    _ld_dst: i64,
    _stream: *mut c_void,
) -> c_int {
    -1
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn fallback_talus_pf_i8gemm_swiglu_q8(
    _qs: *const i8,
    _ws: *const f32,
    _n: i32,
    _k: i32,
    _mode: i32,
    _xq: *const c_void,
    _t: i32,
    _g: *const f32,
    _ldg: i64,
    _yq: *mut c_void,
    _layout: i32,
    _stream: *mut c_void,
) -> c_int {
    -1
}

unsafe extern "C" fn fallback_fnp_qsa_tile_attn_i8(
    _q: *const f32,
    _k_q: *const i8,
    _v_q: *const i8,
    _k_scale: *const u16,
    _v_scale: *const u16,
    _page_table: *const i32,
    _page_size: i64,
    _ids: *const i32,
    _steps: *const i32,
    _cap: i64,
    _attn: *mut f32,
    _n_q: i64,
    _scratch: *mut c_void,
    _stream: *mut c_void,
) -> c_int {
    1
}

unsafe extern "C" fn fallback_fnp_qsa_tile_scratch_bytes_c(_n_q: i64, _cap: i64) -> u64 {
    0
}

// Safety: `LibTalusKernels` holds an immutable `dlopen` handle and resolved function pointers.
unsafe impl Send for LibTalusKernels {}
unsafe impl Sync for LibTalusKernels {}

#[cfg(unix)]
unsafe fn load_sym<T>(handle: *mut c_void, name: &'static str) -> Result<T, String> {
    let c_name = CString::new(name).map_err(|e| format!("invalid symbol name {name}: {e}"))?;
    // Clear any prior dlerror state.
    let _ = dlerror();
    let sym = dlsym(handle, c_name.as_ptr());
    let err_ptr = dlerror();
    if !err_ptr.is_null() {
        let msg = CStr::from_ptr(err_ptr).to_string_lossy().into_owned();
        return Err(format!("dlsym({name}) failed: {msg}"));
    }
    if sym.is_null() {
        return Err(format!("dlsym({name}) returned null"));
    }
    Ok(core::mem::transmute_copy::<*mut c_void, T>(&sym))
}

impl LibTalusKernels {
    /// Dynamically load `libtalus_kernels.so` from `path` using `dlopen`/`dlsym` and resolve all
    /// Contract v0 entry points.
    #[cfg(unix)]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let path_str = path_ref
            .to_str()
            .ok_or_else(|| format!("non-UTF-8 library path: {}", path_ref.display()))?;
        let c_path = CString::new(path_str)
            .map_err(|e| format!("invalid library path '{path_str}': {e}"))?;

        unsafe {
            let _ = dlerror();
            let handle = dlopen(c_path.as_ptr(), RTLD_NOW);
            if handle.is_null() {
                let err_ptr = dlerror();
                let msg = if err_ptr.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
                };
                return Err(format!("dlopen({path_str}) failed: {msg}"));
            }

            let load_all = || -> Result<Self, String> {
                Ok(Self {
                    handle,
                    tk_config_create: load_sym(handle, "tk_config_create")?,
                    tk_config_create_for_arch: load_sym(handle, "tk_config_create_for_arch")?,
                    tk_config_from_env: load_sym(handle, "tk_config_from_env")?,
                    tk_config_destroy: load_sym(handle, "tk_config_destroy")?,
                    tk_config_set_int: load_sym(handle, "tk_config_set_int")?,
                    tk_config_get_int: load_sym(handle, "tk_config_get_int")?,
                    tk_ctx_create: load_sym(handle, "tk_ctx_create")?,
                    tk_ctx_destroy: load_sym(handle, "tk_ctx_destroy")?,
                    tk_ctx_device: load_sym(handle, "tk_ctx_device")?,
                    tk_ctx_arch: load_sym(handle, "tk_ctx_arch")?,
                    tk_ctx_load_tuning: load_sym(handle, "tk_ctx_load_tuning")?,
                    tk_last_error: load_sym(handle, "tk_last_error")?,
                    tk_status_string: load_sym(handle, "tk_status_string")?,

                    tk_quantize_q8_1: load_sym(handle, "tk_quantize_q8_1")?,
                    tk_pack_q8: load_sym(handle, "tk_pack_q8")?,
                    tk_pack_q6: load_sym(handle, "tk_pack_q6")?,
                    tk_packed_free: load_sym(handle, "tk_packed_free")?,

                    tk_mmvq: load_sym(handle, "tk_mmvq")?,
                    tk_mmvq_packed: load_sym(handle, "tk_mmvq_packed")?,
                    tk_mmvq_ids: load_sym(handle, "tk_mmvq_ids")?,
                    tk_f32_matvec: load_sym(handle, "tk_f32_matvec")?,
                    tk_f32_matvec_fast: load_sym(handle, "tk_f32_matvec_fast")?,
                    tk_f32_matvec_rows: load_sym(handle, "tk_f32_matvec_rows")?,
                    tk_f32_matvec_rows_n: load_sym(handle, "tk_f32_matvec_rows_n")?,
                    tk_f32_matmul: load_sym(handle, "tk_f32_matmul")?,
                    tk_route_sigmoid_topk: load_sym(handle, "tk_route_sigmoid_topk")?,
                    tk_route_warp: load_sym(handle, "tk_route_warp")?,
                    tk_route_topk: load_sym(handle, "tk_route_topk")?,
                    tk_route_group: load_sym(handle, "tk_route_group")?,
                    tk_router_route: load_sym(handle, "tk_router_route")?,
                    tk_moe_add_slots: load_sym(handle, "tk_moe_add_slots")?,

                    tk_rms_norm: load_sym(handle, "tk_rms_norm")?,
                    tk_rms_norm_to: load_sym(handle, "tk_rms_norm_to")?,
                    tk_head_norm_rope: load_sym(handle, "tk_head_norm_rope")?,
                    tk_qkv_norm_rope_store: load_sym(handle, "tk_qkv_norm_rope_store")?,
                    tk_rope_neox: load_sym(handle, "tk_rope_neox")?,
                    tk_mrope: load_sym(handle, "tk_mrope")?,
                    tk_silu_mul: load_sym(handle, "tk_silu_mul")?,
                    tk_axpy: load_sym(handle, "tk_axpy")?,
                    tk_add_norm_quant: load_sym(handle, "tk_add_norm_quant")?,
                    tk_embed_q8_0: load_sym(handle, "tk_embed_q8_0")?,
                    tk_gather_rows: load_sym(handle, "tk_gather_rows")?,
                    tk_fill_pos: load_sym(handle, "tk_fill_pos")?,
                    tk_fill_pos_dev: load_sym(handle, "tk_fill_pos_dev")?,
                    tk_argmax_advance: load_sym(handle, "tk_argmax_advance")?,
                    tk_argmax_rows: load_sym(handle, "tk_argmax_rows")?,
                    tk_softplus_head_gate: load_sym(handle, "tk_softplus_head_gate")?,

                    tk_attn_decode: load_sym(handle, "tk_attn_decode")?,
                    tk_attn_prefill_fa: load_sym(handle, "tk_attn_prefill_fa")?,
                    tk_laguna_prompt_attn: load_sym(handle, "tk_laguna_prompt_attn")?,
                    tk_laguna_prompt_attn_wmma: load_sym(handle, "tk_laguna_prompt_attn_wmma")?,
                    tk_laguna_decode_attn: load_sym(handle, "tk_laguna_decode_attn")?,
                    tk_attn_merge_gate_quant: load_sym(handle, "tk_attn_merge_gate_quant")?,
                    tk_kv_store_f16: load_sym(handle, "tk_kv_store_f16")?,
                    tk_v_store_t: load_sym(handle, "tk_v_store_t")?,
                    tk_kv_store_q8: load_sym(handle, "tk_kv_store_q8")?,
                    tk_kbound_update: load_sym(handle, "tk_kbound_update")?,
                    tk_sparse_scores: load_sym(handle, "tk_sparse_scores")?,
                    tk_sparse_select: load_sym(handle, "tk_sparse_select")?,
                    tk_attn_swa_sinks: load_sym(handle, "tk_attn_swa_sinks")?,
                    tk_attn_swa_sinks_mmq: load_sym(handle, "tk_attn_swa_sinks_mmq")?,
                    tk_attn_swa_sinks_q8: load_sym(handle, "tk_attn_swa_sinks_q8")?,
                    tk_attn_k192_v128: load_sym(handle, "tk_attn_k192_v128")?,
                    tk_attn_k192_v128_q8: load_sym(handle, "tk_attn_k192_v128_q8")?,
                    tk_mimo_rope_qk_store: load_sym(handle, "tk_mimo_rope_qk_store")?,
                    tk_axpy_norm_quant: load_sym(handle, "tk_axpy_norm_quant")?,
                    tk_dsa_decode_indexer: load_sym(handle, "tk_dsa_decode_indexer")?,

                    tk_gdn_preprocess: load_sym(handle, "tk_gdn_preprocess")?,
                    tk_gdn_step: load_sym(handle, "tk_gdn_step")?,
                    tk_gdn_seq: load_sym(handle, "tk_gdn_seq")?,
                    tk_gdn_fused: load_sym(handle, "tk_gdn_fused")?,
                    tk_kda_step: load_sym(handle, "tk_kda_step")?,

                    tk_prefill_gemm: load_sym(handle, "tk_prefill_gemm")?,
                    tk_prefill_mmq: load_sym(handle, "tk_prefill_mmq")?,
                    tk_prefill_moe_mmq: load_sym(handle, "tk_prefill_moe_mmq")?,
                    tk_prefill_moe_mmq_gu: load_sym(handle, "tk_prefill_moe_mmq_gu")?,
                    tk_prefill_moe_mmq_gu_q8: load_sym(handle, "tk_prefill_moe_mmq_gu_q8")?,
                    tk_prefill_moe_mmq_iq: load_sym(handle, "tk_prefill_moe_mmq_iq")?,

                    tk_q8_1_bytes: load_sym(handle, "tk_q8_1_bytes")?,
                    tk_swiglu_quantize_q8_1: load_sym(handle, "tk_swiglu_quantize_q8_1")?,
                    tk_mmvq_w32_multi: load_sym(handle, "tk_mmvq_w32_multi")?,
                    tk_mmvq_batch: load_sym(handle, "tk_mmvq_batch")?,

                    tk_q36_rms: load_sym(handle, "tk_q36_rms")?,
                    tk_q36_rms_q8: load_sym(handle, "tk_q36_rms_q8")?,
                    tk_q36_rms_ld: load_sym(handle, "tk_q36_rms_ld")?,
                    tk_q36_rms_ld_q8: load_sym(handle, "tk_q36_rms_ld_q8")?,
                    tk_q36_add_rms: load_sym(handle, "tk_q36_add_rms")?,
                    tk_q36_add_rms_q8: load_sym(handle, "tk_q36_add_rms_q8")?,
                    tk_q36_add_rms_exact_q8: load_sym(handle, "tk_q36_add_rms_exact_q8")?,
                    tk_q36_moe_out: load_sym(handle, "tk_q36_moe_out")?,
                    tk_q36_moe_out_q8: load_sym(handle, "tk_q36_moe_out_q8")?,
                    tk_q36_gdn_ab: load_sym(handle, "tk_q36_gdn_ab")?,
                    tk_q36_gdn_ab2: load_sym(handle, "tk_q36_gdn_ab2")?,
                    tk_q36_add: load_sym(handle, "tk_q36_add")?,
                    tk_q36_swiglu: load_sym(handle, "tk_q36_swiglu")?,
                    tk_q36_gemv_f32: load_sym(handle, "tk_q36_gemv_f32")?,
                    tk_q36_gdn_gb: load_sym(handle, "tk_q36_gdn_gb")?,
                    tk_q36_gdn_gb2: load_sym(handle, "tk_q36_gdn_gb2")?,
                    tk_q36_gate_rows: load_sym(handle, "tk_q36_gate_rows")?,
                    tk_q36_attn_prep: load_sym(handle, "tk_q36_attn_prep")?,
                    tk_q36_attn_part: load_sym(handle, "tk_q36_attn_part")?,
                    tk_q36_attn_merge: load_sym(handle, "tk_q36_attn_merge")?,
                    tk_q36_attn_wmma: load_sym(handle, "tk_q36_attn_wmma")?,
                    tk_q36_group: load_sym(handle, "tk_q36_group")?,
                    tk_q36_group_hy: load_sym(handle, "tk_q36_group_hy")?,
                    tk_q36_merge_cpu: load_sym(handle, "tk_q36_merge_cpu")?,
                    tk_q36_attn_softmax16: load_sym(handle, "tk_q36_attn_softmax16")?,
                    tk_q36_gate_mul: load_sym(handle, "tk_q36_gate_mul")?,
                    tk_q36_route_count: load_sym(handle, "tk_q36_route_count")?,
                    tk_q36_route_scan: load_sym(handle, "tk_q36_route_scan")?,
                    tk_q36_route_scatter: load_sym(handle, "tk_q36_route_scatter")?,
                    tk_q36_argmax: load_sym(handle, "tk_q36_argmax")?,
                    tk_q36_argmax_samp: load_sym(handle, "tk_q36_argmax_samp")?,
                    tk_q36_lse: load_sym(handle, "tk_q36_lse")?,
                    tk_q36_ctl_end: load_sym(handle, "tk_q36_ctl_end")?,
                    tk_q36_round_prep: load_sym(handle, "tk_q36_round_prep")?,
                    tk_q36_row_copy: load_sym(handle, "tk_q36_row_copy")?,
                    tk_q36_chain_prep: load_sym(handle, "tk_q36_chain_prep")?,
                    tk_q36_sgemm_nt: load_sym(handle, "tk_q36_sgemm_nt")?,
                    tk_q36_sum_splits: load_sym(handle, "tk_q36_sum_splits")?,
                    tk_q36_to_f16: load_sym(handle, "tk_q36_to_f16")?,
                    tk_q36_conv_seq: load_sym(handle, "tk_q36_conv_seq")?,
                    tk_q36_conv_ab: load_sym(handle, "tk_q36_conv_ab")?,
                    tk_q36_conv_par: load_sym(handle, "tk_q36_conv_par")?,
                    tk_q36_conv_hist: load_sym(handle, "tk_q36_conv_hist")?,
                    tk_q36_gdn_seq: load_sym(handle, "tk_q36_gdn_seq")?,
                    tk_q36_gdn_seq_cols: load_sym(handle, "tk_q36_gdn_seq_cols")?,
                    tk_q36_gdn_seq_warp: load_sym(handle, "tk_q36_gdn_seq_warp")?,
                    tk_q36_gdn_norm: load_sym(handle, "tk_q36_gdn_norm")?,
                    tk_q36_flush_read: load_sym(handle, "tk_q36_flush_read")?,
                    tk_q36_draft_conf: load_sym(handle, "tk_q36_draft_conf")?,
                    tk_q36_draft_support: load_sym(handle, "tk_q36_draft_support")?,
                    tk_q36_dflash_attn_chunks: load_sym(handle, "tk_q36_dflash_attn_chunks")?,
                    tk_q36_dflash_attn_scratch_floats: load_sym(
                        handle,
                        "tk_q36_dflash_attn_scratch_floats",
                    )?,
                    tk_q36_dflash_rms_rows: load_sym(handle, "tk_q36_dflash_rms_rows")?,
                    tk_q36_dflash_rms_rows_q8: load_sym(handle, "tk_q36_dflash_rms_rows_q8")?,
                    tk_q36_dflash_sum5_rms: load_sym(handle, "tk_q36_dflash_sum5_rms")?,
                    tk_q36_dflash_sum5_rms_q8: load_sym(handle, "tk_q36_dflash_sum5_rms_q8")?,
                    tk_q36_dflash_dyn_conv: load_sym(handle, "tk_q36_dflash_dyn_conv")?,
                    tk_q36_dflash_dyn_conv_q8: load_sym(handle, "tk_q36_dflash_dyn_conv_q8")?,
                    tk_q36_dflash_qk_norm_rope: load_sym(handle, "tk_q36_dflash_qk_norm_rope")?,
                    tk_q36_dflash_kv_store: load_sym(handle, "tk_q36_dflash_kv_store")?,
                    tk_q36_dflash_kv_store_f16: load_sym(handle, "tk_q36_dflash_kv_store_f16")?,
                    tk_q36_dflash_block_attn: load_sym(handle, "tk_q36_dflash_block_attn")?,
                    tk_q36_dflash_swiglu: load_sym(handle, "tk_q36_dflash_swiglu")?,
                    tk_q36_dflash_swiglu_q8: load_sym(handle, "tk_q36_dflash_swiglu_q8")?,
                    tk_q36_dflash_add_inplace: load_sym(handle, "tk_q36_dflash_add_inplace")?,
                    tk_q36_dflash_block_tokens: load_sym(handle, "tk_q36_dflash_block_tokens")?,
                    tk_q36_dflash_set_i32: load_sym(handle, "tk_q36_dflash_set_i32")?,
                    tk_q36_dflash_top16: load_sym(handle, "tk_q36_dflash_top16")?,
                    tk_q36_dflash_select_chain: load_sym(handle, "tk_q36_dflash_select_chain")?,
                    tk_q36_dflash_select_chain_q: load_sym(handle, "tk_q36_dflash_select_chain_q")?,
                    tk_q36_mmvq_supported: load_sym(handle, "tk_q36_mmvq_supported")?,
                    tk_q36_q8_0_pack_should: load_sym(handle, "tk_q36_q8_0_pack_should")?,
                    tk_q36_q8_0_pack_host: load_sym(handle, "tk_q36_q8_0_pack_host")?,
                    tk_q36_q8_0_packed_register: load_sym(handle, "tk_q36_q8_0_packed_register")?,
                    tk_q36_q8_0_packed_unregister: load_sym(
                        handle,
                        "tk_q36_q8_0_packed_unregister",
                    )?,
                    tk_q36_embed_rows: load_sym(handle, "tk_q36_embed_rows")?,
                    tk_q36_mmq_built: load_sym(handle, "tk_q36_mmq_built")?,
                    tk_q36_mmq_supported_fits: load_sym(handle, "tk_q36_mmq_supported_fits")?,
                    tk_q36_mmq_q8_layout: load_sym(handle, "tk_q36_mmq_q8_layout")?,
                    tk_q36_mmq_q8_bytes: load_sym(handle, "tk_q36_mmq_q8_bytes")?,
                    tk_q36_mmq_matrix_bytes: load_sym(handle, "tk_q36_mmq_matrix_bytes")?,
                    tk_q36_mmq_iota: load_sym(handle, "tk_q36_mmq_iota")?,
                    tk_q36_mmq_set_bounds: load_sym(handle, "tk_q36_mmq_set_bounds")?,
                    tk_q36_mmq_quantize: load_sym(handle, "tk_q36_mmq_quantize")?,
                    tk_q36_mmq_quantize_scatter: load_sym(handle, "tk_q36_mmq_quantize_scatter")?,
                    tk_q36_mmq_swiglu: load_sym(handle, "tk_q36_mmq_swiglu")?,
                    tk_q36_fattn_supported: load_sym(handle, "tk_q36_fattn_supported")?,
                    tk_q36_fattn_mask_bytes: load_sym(handle, "tk_q36_fattn_mask_bytes")?,
                    tk_q36_prefill_init_scratch: load_sym(handle, "tk_q36_prefill_init_scratch")?,
                    tk_q36_attn_blas_prefill: load_sym(handle, "tk_q36_attn_blas_prefill")?,
                    tk_q36_router_topk: load_sym(handle, "tk_q36_router_topk")?,
                    tk_q36_expert_supported: load_sym(handle, "tk_q36_expert_supported")?,
                    tk_q36_expert_layout: load_sym(handle, "tk_q36_expert_layout")?,
                    tk_q36_expert_scratch_bytes: load_sym(handle, "tk_q36_expert_scratch_bytes")?,
                    tk_q36_expert_grouped: load_sym(handle, "tk_q36_expert_grouped")?,
                    talus_pf_fa_prefill: load_sym(handle, "talus_pf_fa_prefill")?,
                    talus_pf_i8x_expand: load_sym(handle, "talus_pf_i8x_expand")?,
                    talus_pf_i8gemm: load_sym(handle, "talus_pf_i8gemm")?,
                    talus_pf_moe_gemm: load_sym(handle, "talus_pf_moe_gemm")
                        .unwrap_or(fallback_talus_pf_moe_gemm),
                    talus_pf_swiglu_q8_mmq: load_sym(handle, "talus_pf_swiglu_q8_mmq")?,
                    talus_pf_i8gemm_swiglu_q8: load_sym(handle, "talus_pf_i8gemm_swiglu_q8")
                        .unwrap_or(fallback_talus_pf_i8gemm_swiglu_q8),
                    talus_pf_w4_pack: load_sym(handle, "talus_pf_w4_pack")?,
                    talus_pf_w4_gemm: load_sym(handle, "talus_pf_w4_gemm")?,
                    fnp_qsa_tile_attn_i8: load_sym(handle, "fnp_qsa_tile_attn_i8")
                        .unwrap_or(fallback_fnp_qsa_tile_attn_i8),
                    fnp_qsa_tile_scratch_bytes_c: load_sym(handle, "fnp_qsa_tile_scratch_bytes_c")
                        .unwrap_or(fallback_fnp_qsa_tile_scratch_bytes_c),
                })
            };

            match load_all() {
                Ok(lib) => Ok(lib),
                Err(err) => {
                    let _ = dlclose(handle);
                    Err(err)
                }
            }
        }
    }

    /// Resolve `libtalus_kernels.so` from `TALUS_KERNELS_LIB`, `TALUS_KERNELS_PREBUILT`, or the
    /// default dynamic linker search path.
    #[cfg(unix)]
    pub fn load_default() -> Result<Self, String> {
        if let Ok(path) = std::env::var("TALUS_KERNELS_LIB") {
            if !path.is_empty() {
                return Self::load(path);
            }
        }
        if let Ok(dir) = std::env::var("TALUS_KERNELS_PREBUILT") {
            if !dir.is_empty() {
                let candidate = Path::new(&dir).join("libtalus_kernels.so");
                if candidate.exists() {
                    return Self::load(candidate);
                }
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                let candidate = parent.join("libtalus_kernels.so");
                if candidate.exists() {
                    return Self::load(candidate);
                }
            }
        }
        Self::load("libtalus_kernels.so")
    }

    /// Construct a [`LibTalusKernels`] table backed by build-time linked symbols.
    #[cfg(feature = "link-kernels")]
    pub fn linked() -> Self {
        Self::bind_extern_symbols()
    }

    /// Internal helper binding all `extern "C"` declarations into a [`LibTalusKernels`] struct.
    ///
    /// Also used at compile time to guarantee that every field of [`LibTalusKernels`] has the
    /// exact same signature as its `extern "C"` counterpart.
    #[allow(dead_code)]
    fn bind_extern_symbols() -> Self {
        Self {
            handle: core::ptr::null_mut(),
            tk_config_create,
            tk_config_create_for_arch,
            tk_config_from_env,
            tk_config_destroy,
            tk_config_set_int,
            tk_config_get_int,
            tk_ctx_create,
            tk_ctx_destroy,
            tk_ctx_device,
            tk_ctx_arch,
            tk_ctx_load_tuning,
            tk_last_error,
            tk_status_string,
            tk_quantize_q8_1,
            tk_pack_q8,
            tk_pack_q6,
            tk_packed_free,
            tk_mmvq,
            tk_mmvq_packed,
            tk_mmvq_ids,
            tk_f32_matvec,
            tk_f32_matvec_fast,
            tk_f32_matvec_rows,
            tk_f32_matvec_rows_n,
            tk_f32_matmul,
            tk_route_sigmoid_topk,
            tk_route_warp,
            tk_route_topk,
            tk_route_group,
            tk_router_route,
            tk_moe_add_slots,
            tk_rms_norm,
            tk_rms_norm_to,
            tk_head_norm_rope,
            tk_qkv_norm_rope_store,
            tk_rope_neox,
            tk_mrope,
            tk_silu_mul,
            tk_axpy,
            tk_add_norm_quant,
            tk_embed_q8_0,
            tk_gather_rows,
            tk_fill_pos,
            tk_fill_pos_dev,
            tk_argmax_advance,
            tk_argmax_rows,
            tk_softplus_head_gate,
            tk_attn_decode,
            tk_attn_prefill_fa,
            tk_laguna_prompt_attn,
            tk_laguna_prompt_attn_wmma,
            tk_laguna_decode_attn,
            tk_attn_merge_gate_quant,
            tk_kv_store_f16,
            tk_v_store_t,
            tk_kv_store_q8,
            tk_kbound_update,
            tk_sparse_scores,
            tk_sparse_select,
            tk_attn_swa_sinks,
            tk_attn_swa_sinks_mmq,
            tk_attn_swa_sinks_q8,
            tk_attn_k192_v128,
            tk_attn_k192_v128_q8,
            tk_mimo_rope_qk_store,
            tk_axpy_norm_quant,
            tk_dsa_decode_indexer,
            tk_gdn_preprocess,
            tk_gdn_step,
            tk_gdn_seq,
            tk_gdn_fused,
            tk_kda_step,
            tk_prefill_gemm,
            tk_prefill_mmq,
            tk_prefill_moe_mmq,
            tk_prefill_moe_mmq_gu,
            tk_prefill_moe_mmq_gu_q8,
            tk_prefill_moe_mmq_iq,
            tk_q8_1_bytes,
            tk_swiglu_quantize_q8_1,
            tk_mmvq_w32_multi,
            tk_mmvq_batch,
            tk_q36_rms,
            tk_q36_rms_q8,
            tk_q36_rms_ld,
            tk_q36_rms_ld_q8,
            tk_q36_add_rms,
            tk_q36_add_rms_q8,
            tk_q36_add_rms_exact_q8,
            tk_q36_moe_out,
            tk_q36_moe_out_q8,
            tk_q36_gdn_ab,
            tk_q36_gdn_ab2,
            tk_q36_add,
            tk_q36_swiglu,
            tk_q36_gemv_f32,
            tk_q36_gdn_gb,
            tk_q36_gdn_gb2,
            tk_q36_gate_rows,
            tk_q36_attn_prep,
            tk_q36_attn_part,
            tk_q36_attn_merge,
            tk_q36_attn_wmma,
            tk_q36_group,
            tk_q36_group_hy,
            tk_q36_merge_cpu,
            tk_q36_attn_softmax16,
            tk_q36_gate_mul,
            tk_q36_route_count,
            tk_q36_route_scan,
            tk_q36_route_scatter,
            tk_q36_argmax,
            tk_q36_argmax_samp,
            tk_q36_lse,
            tk_q36_ctl_end,
            tk_q36_round_prep,
            tk_q36_row_copy,
            tk_q36_chain_prep,
            tk_q36_sgemm_nt,
            tk_q36_sum_splits,
            tk_q36_to_f16,
            tk_q36_conv_seq,
            tk_q36_conv_ab,
            tk_q36_conv_par,
            tk_q36_conv_hist,
            tk_q36_gdn_seq,
            tk_q36_gdn_seq_cols,
            tk_q36_gdn_seq_warp,
            tk_q36_gdn_norm,
            tk_q36_flush_read,
            tk_q36_draft_conf,
            tk_q36_draft_support,
            tk_q36_dflash_attn_chunks,
            tk_q36_dflash_attn_scratch_floats,
            tk_q36_dflash_rms_rows,
            tk_q36_dflash_rms_rows_q8,
            tk_q36_dflash_sum5_rms,
            tk_q36_dflash_sum5_rms_q8,
            tk_q36_dflash_dyn_conv,
            tk_q36_dflash_dyn_conv_q8,
            tk_q36_dflash_qk_norm_rope,
            tk_q36_dflash_kv_store,
            tk_q36_dflash_kv_store_f16,
            tk_q36_dflash_block_attn,
            tk_q36_dflash_swiglu,
            tk_q36_dflash_swiglu_q8,
            tk_q36_dflash_add_inplace,
            tk_q36_dflash_block_tokens,
            tk_q36_dflash_set_i32,
            tk_q36_dflash_top16,
            tk_q36_dflash_select_chain,
            tk_q36_dflash_select_chain_q,
            tk_q36_mmvq_supported,
            tk_q36_q8_0_pack_should,
            tk_q36_q8_0_pack_host,
            tk_q36_q8_0_packed_register,
            tk_q36_q8_0_packed_unregister,
            tk_q36_embed_rows,
            tk_q36_mmq_built,
            tk_q36_mmq_supported_fits,
            tk_q36_mmq_q8_layout,
            tk_q36_mmq_q8_bytes,
            tk_q36_mmq_matrix_bytes,
            tk_q36_mmq_iota,
            tk_q36_mmq_set_bounds,
            tk_q36_mmq_quantize,
            tk_q36_mmq_quantize_scatter,
            tk_q36_mmq_swiglu,
            tk_q36_fattn_supported,
            tk_q36_fattn_mask_bytes,
            tk_q36_prefill_init_scratch,
            tk_q36_attn_blas_prefill,
            tk_q36_router_topk,
            tk_q36_expert_supported,
            tk_q36_expert_layout,
            tk_q36_expert_scratch_bytes,
            tk_q36_expert_grouped,
            talus_pf_fa_prefill,
            talus_pf_i8x_expand,
            talus_pf_i8gemm,
            talus_pf_moe_gemm,
            talus_pf_swiglu_q8_mmq,
            talus_pf_i8gemm_swiglu_q8,
            talus_pf_w4_pack,
            talus_pf_w4_gemm,
            fnp_qsa_tile_attn_i8,
            fnp_qsa_tile_scratch_bytes_c,
        }
    }
}

impl Drop for LibTalusKernels {
    fn drop(&mut self) {
        #[cfg(unix)]
        if !self.handle.is_null() {
            unsafe {
                let _ = dlclose(self.handle);
            }
            self.handle = core::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn crate_identity_and_abi_constants() {
        assert_eq!(CRATE_NAME, "talus-kernels-sys");
        assert_eq!(TALUS_KERNELS_ABI_VERSION, 0);
        assert_eq!(tk_status_code::from_raw(TK_OK), Some(tk_status_code::TK_OK));
        assert_eq!(
            tk_status_code::from_raw(TK_ERR_INTERNAL),
            Some(tk_status_code::TK_ERR_INTERNAL)
        );
        assert_eq!(tk_status_code::from_raw(999), None);
    }

    #[test]
    fn c_stub_abi_signature_and_dlopen_compatibility() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let include_dir = manifest_dir.join("../../../include");
        let scratch_dir = manifest_dir.join("../../target/c_stub_test");
        fs::create_dir_all(&scratch_dir).expect("create scratch dir inside target/");

        let c_file = scratch_dir.join("talus_kernels_stub.c");
        let so_file = scratch_dir.join("libtalus_kernels_stub.so");

        // This C file includes `include/talus_kernels.h` directly and defines every `tk_*`
        // entry point. Compiling with `-Wall -Wextra -Werror` fails if ANY signature, enum,
        // or struct in `talus_kernels.h` diverges from these definitions.
        let c_source = r#"
#include "talus_kernels.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(TALUS_KERNELS_ABI_VERSION == 0, "ABI version must be 0");

struct tk_config {
    tk_arch arch;
    int64_t knob_val;
};

struct tk_ctx {
    int32_t device;
    tk_arch arch;
    char last_err[128];
};

struct tk_packed_weights {
    int32_t n_in;
    int32_t n_out;
};

size_t tk_stub_sizeof_mmvq_item(void) { return sizeof(tk_mmvq_item); }
size_t tk_stub_sizeof_moe_mmq_product(void) { return sizeof(tk_moe_mmq_product); }
size_t tk_stub_offsetof_mmvq_y(void) { return offsetof(tk_mmvq_item, y); }
size_t tk_stub_offsetof_moe_dst(void) { return offsetof(tk_moe_mmq_product, dst); }

tk_status tk_config_create(tk_config** out_cfg) {
    if (!out_cfg) return TK_ERR_NULL_PTR;
    tk_config* c = (tk_config*)calloc(1, sizeof(tk_config));
    c->arch = TK_ARCH_GFX1151;
    *out_cfg = c;
    return TK_OK;
}
tk_status tk_config_create_for_arch(tk_arch arch, tk_config** out_cfg) {
    if (!out_cfg) return TK_ERR_NULL_PTR;
    tk_config* c = (tk_config*)calloc(1, sizeof(tk_config));
    c->arch = arch;
    *out_cfg = c;
    return TK_OK;
}
tk_status tk_config_from_env(tk_config* cfg) {
    return cfg ? TK_OK : TK_ERR_NULL_PTR;
}
void tk_config_destroy(tk_config* cfg) { free(cfg); }
tk_status tk_config_set_int(tk_config* cfg, const char* key, int64_t value) {
    if (!cfg || !key) return TK_ERR_NULL_PTR;
    cfg->knob_val = value;
    return TK_OK;
}
tk_status tk_config_get_int(const tk_config* cfg, const char* key, int64_t* out_value) {
    if (!cfg || !key || !out_value) return TK_ERR_NULL_PTR;
    *out_value = cfg->knob_val;
    return TK_OK;
}
tk_status tk_ctx_create(int32_t device, const tk_config* cfg, tk_ctx** out_ctx) {
    if (!out_ctx) return TK_ERR_NULL_PTR;
    tk_ctx* c = (tk_ctx*)calloc(1, sizeof(tk_ctx));
    c->device = device;
    c->arch = cfg ? cfg->arch : TK_ARCH_GFX1151;
    strcpy(c->last_err, "no error");
    *out_ctx = c;
    return TK_OK;
}
void tk_ctx_destroy(tk_ctx* ctx) { free(ctx); }
tk_status tk_ctx_device(const tk_ctx* ctx, int32_t* out_device) {
    if (!ctx || !out_device) return TK_ERR_NULL_PTR;
    *out_device = ctx->device;
    return TK_OK;
}
tk_status tk_ctx_arch(const tk_ctx* ctx, tk_arch* out_arch) {
    if (!ctx || !out_arch) return TK_ERR_NULL_PTR;
    *out_arch = ctx->arch;
    return TK_OK;
}
tk_status tk_ctx_load_tuning(tk_ctx* ctx, const char* dir) {
    if (!ctx || !dir) return TK_ERR_NULL_PTR;
    return TK_OK;
}
const char* tk_last_error(const tk_ctx* ctx) {
    return ctx ? ctx->last_err : "thread error";
}
const char* tk_status_string(tk_status status) {
    switch (status) {
        case TK_OK: return "TK_OK";
        case TK_ERR_INVALID_ARG: return "TK_ERR_INVALID_ARG";
        case TK_ERR_NULL_PTR: return "TK_ERR_NULL_PTR";
        case TK_ERR_UNSUPPORTED: return "TK_ERR_UNSUPPORTED";
        default: return "TK_ERR_OTHER";
    }
}

tk_status tk_quantize_q8_1(tk_ctx* ctx, const float* x, void* x_q8_1, int32_t n_in, int32_t ncols, tk_stream stream) {
    (void)ctx; (void)x; (void)x_q8_1; (void)stream;
    return (n_in > 0 && ncols > 0) ? TK_OK : TK_ERR_INVALID_ARG;
}
tk_status tk_pack_q8(tk_ctx* ctx, const void* w_gguf, int32_t n_in, int32_t n_out, tk_stream stream, tk_packed_weights** out_packed) {
    (void)ctx; (void)w_gguf; (void)stream;
    if (!out_packed) return TK_ERR_NULL_PTR;
    tk_packed_weights* p = (tk_packed_weights*)calloc(1, sizeof(tk_packed_weights));
    p->n_in = n_in; p->n_out = n_out;
    *out_packed = p;
    return TK_OK;
}
tk_status tk_pack_q6(tk_ctx* ctx, const void* w_gguf, int32_t n_in, int32_t n_out, tk_stream stream, tk_packed_weights** out_packed) {
    return tk_pack_q8(ctx, w_gguf, n_in, n_out, stream, out_packed);
}
void tk_packed_free(tk_ctx* ctx, tk_packed_weights* packed) {
    (void)ctx;
    free(packed);
}

tk_status tk_mmvq(tk_ctx* ctx, tk_qtype qtype, const void* w, const void* x_q8_1, float* y, int32_t n_in, int32_t n_out, int32_t ncols, tk_stream stream) {
    (void)ctx; (void)qtype; (void)w; (void)x_q8_1; (void)y; (void)n_in; (void)n_out; (void)ncols; (void)stream;
    return TK_OK;
}
tk_status tk_mmvq_packed(tk_ctx* ctx, const tk_packed_weights* packed, const void* x_q8_1, float* y, int32_t ncols, tk_stream stream) {
    (void)ctx; (void)packed; (void)x_q8_1; (void)y; (void)ncols; (void)stream;
    return TK_OK;
}
tk_status tk_mmvq_ids(tk_ctx* ctx, tk_qtype qtype, const void* w, size_t expert_bytes, const int32_t* ids_dev, int32_t n_ids, const void* x_q8_1, bool x_per_id, float* y, int32_t n_in, int32_t n_out, const void* w2, float* y2, tk_stream stream) {
    (void)ctx; (void)qtype; (void)w; (void)expert_bytes; (void)ids_dev; (void)n_ids; (void)x_q8_1; (void)x_per_id; (void)y; (void)n_in; (void)n_out; (void)w2; (void)y2; (void)stream;
    return TK_OK;
}
tk_status tk_f32_matvec(tk_ctx* ctx, const float* w, const float* x, float* y, int32_t n_in, int32_t n_out, tk_stream stream) {
    (void)ctx; (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)stream;
    return TK_OK;
}
tk_status tk_f32_matvec_fast(tk_ctx* ctx, const float* w, const float* x, float* y, int32_t n_in, int32_t n_out, tk_stream stream) {
    (void)ctx; (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)stream;
    return TK_OK;
}
tk_status tk_f32_matvec_rows(tk_ctx* ctx, const float* w, const float* x, float* y, int32_t n_in, int32_t n_out, tk_stream stream) {
    (void)ctx; (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)stream;
    return TK_OK;
}
tk_status tk_f32_matvec_rows_n(tk_ctx* ctx, const float* w, const float* x, float* y, int32_t n_in, int32_t n_out, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_f32_matmul(tk_ctx* ctx, const float* w, const float* x, float* y, int32_t n_in, int32_t n_out, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_route_sigmoid_topk(tk_ctx* ctx, const float* logits, const float* bias, int32_t* ids_dev, float* weights_dev, int32_t n_expert, int32_t k, float scale, tk_stream stream) {
    (void)ctx; (void)logits; (void)bias; (void)ids_dev; (void)weights_dev; (void)n_expert; (void)k; (void)scale; (void)stream;
    return TK_OK;
}
tk_status tk_route_warp(tk_ctx* ctx, const float* logits, const float* bias, int32_t* ids_dev, float* weights_dev, int32_t k, float scale, tk_stream stream) {
    (void)ctx; (void)logits; (void)bias; (void)ids_dev; (void)weights_dev; (void)k; (void)scale; (void)stream;
    return TK_OK;
}
tk_status tk_route_topk(tk_ctx* ctx, const float* logits, const float* bias, int32_t n_tok, int32_t k, float scale, int32_t* ids_dev, float* weights_dev, tk_stream stream) {
    (void)ctx; (void)logits; (void)bias; (void)n_tok; (void)k; (void)scale; (void)ids_dev; (void)weights_dev; (void)stream;
    return TK_OK;
}
tk_status tk_route_group(tk_ctx* ctx, const float* logits, const float* bias, int32_t n_tok, int32_t k, float scale, int32_t* sel_e_dev, float* sel_w_dev, int32_t* cnt_dev, int32_t* bounds_dev, int32_t* rtok_dev, int32_t* smap_dev, tk_stream stream) {
    (void)ctx; (void)logits; (void)bias; (void)n_tok; (void)k; (void)scale; (void)sel_e_dev; (void)sel_w_dev; (void)cnt_dev; (void)bounds_dev; (void)rtok_dev; (void)smap_dev; (void)stream;
    return TK_OK;
}
tk_status tk_router_route(tk_ctx* ctx, const float* w, const float* x, float* logits, const float* bias, int32_t* ids_dev, float* weights_dev, int32_t n_in, int32_t n_expert, int32_t k, float scale, uint32_t* counter_dev, tk_stream stream) {
    (void)ctx; (void)w; (void)x; (void)logits; (void)bias; (void)ids_dev; (void)weights_dev; (void)n_in; (void)n_expert; (void)k; (void)scale; (void)counter_dev; (void)stream;
    return TK_OK;
}
tk_status tk_moe_add_slots(tk_ctx* ctx, float* x, const float* shared, const float* y, const int32_t* map_dev, const float* weights_dev, int32_t n_tok, int32_t n_embd, int32_t k, tk_stream stream) {
    (void)ctx; (void)x; (void)shared; (void)y; (void)map_dev; (void)weights_dev; (void)n_tok; (void)n_embd; (void)k; (void)stream;
    return TK_OK;
}

tk_status tk_rms_norm(tk_ctx* ctx, float* x, const float* w, int64_t rows, int64_t cols, float eps, tk_stream stream) {
    (void)ctx; (void)x; (void)w; (void)rows; (void)cols; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_rms_norm_to(tk_ctx* ctx, float* dst, const float* src, const float* w, int32_t rows, int32_t cols, float eps, tk_stream stream) {
    (void)ctx; (void)dst; (void)src; (void)w; (void)rows; (void)cols; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_head_norm_rope(tk_ctx* ctx, float* x, const float* w, int32_t rows, float eps, int32_t n_rot, const float* cos_tab, const float* sin_tab, const int32_t* pos_dev, int32_t pos_div, tk_stream stream) {
    (void)ctx; (void)x; (void)w; (void)rows; (void)eps; (void)n_rot; (void)cos_tab; (void)sin_tab; (void)pos_dev; (void)pos_div; (void)stream;
    return TK_OK;
}
tk_status tk_qkv_norm_rope_store(tk_ctx* ctx, float* q, const float* kc, const float* vc, const float* q_norm, const float* k_norm, int32_t n_head, int32_t n_kv, float eps, int32_t n_rot, const float* cos_tab, const float* sin_tab, const int32_t* pos_dev, void* k_pool, void* v_pool, int32_t vt_stride, uint16_t* k_scale, uint16_t* v_scale, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)q; (void)kc; (void)vc; (void)q_norm; (void)k_norm; (void)n_head; (void)n_kv; (void)eps; (void)n_rot; (void)cos_tab; (void)sin_tab; (void)pos_dev; (void)k_pool; (void)v_pool; (void)vt_stride; (void)k_scale; (void)v_scale; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_rope_neox(tk_ctx* ctx, const float* x, float* out, int64_t rows, int32_t head_dim, int32_t n_rot, const float* cos_tab, const float* sin_tab, const int32_t* pos_dev, tk_stream stream) {
    (void)ctx; (void)x; (void)out; (void)rows; (void)head_dim; (void)n_rot; (void)cos_tab; (void)sin_tab; (void)pos_dev; (void)stream;
    return TK_OK;
}
tk_status tk_mrope(tk_ctx* ctx, const float* x, float* out, int64_t rows, int32_t head_dim, int32_t n_rot, const int32_t sections[4], const float* cos_tab, const float* sin_tab, const int32_t* pos_dev, const int32_t* mrope_table_dev, tk_stream stream) {
    (void)ctx; (void)x; (void)out; (void)rows; (void)head_dim; (void)n_rot; (void)sections; (void)cos_tab; (void)sin_tab; (void)pos_dev; (void)mrope_table_dev; (void)stream;
    return TK_OK;
}
tk_status tk_silu_mul(tk_ctx* ctx, float* gate, const float* up, int64_t n, tk_stream stream) {
    (void)ctx; (void)gate; (void)up; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_axpy(tk_ctx* ctx, float* y, const float* x, float alpha, int32_t n, tk_stream stream) {
    (void)ctx; (void)y; (void)x; (void)alpha; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_add_norm_quant(tk_ctx* ctx, float* x, const float* add, const float* yexp, const float* weights_dev, int32_t k, const float* norm_w, float* h_out, void* q8_out, int32_t n, float eps, int32_t n_rows, tk_stream stream) {
    (void)ctx; (void)x; (void)add; (void)yexp; (void)weights_dev; (void)k; (void)norm_w; (void)h_out; (void)q8_out; (void)n; (void)eps; (void)n_rows; (void)stream;
    return TK_OK;
}
tk_status tk_embed_q8_0(tk_ctx* ctx, const void* table, const int32_t* tok_dev, float* x, int32_t n_embd, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)table; (void)tok_dev; (void)x; (void)n_embd; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_gather_rows(tk_ctx* ctx, const void* src, void* dst, const int32_t* idx_dev, int32_t n_rows, size_t row_bytes, tk_stream stream) {
    (void)ctx; (void)src; (void)dst; (void)idx_dev; (void)n_rows; (void)row_bytes; (void)stream;
    return TK_OK;
}
tk_status tk_fill_pos(tk_ctx* ctx, int32_t* pos_dev, int32_t n_tok, int32_t per_tok, int32_t p0, tk_stream stream) {
    (void)ctx; (void)pos_dev; (void)n_tok; (void)per_tok; (void)p0; (void)stream;
    return TK_OK;
}
tk_status tk_fill_pos_dev(tk_ctx* ctx, int32_t* out_dev, int32_t n, const int32_t* pos_dev, tk_stream stream) {
    (void)ctx; (void)out_dev; (void)n; (void)pos_dev; (void)stream;
    return TK_OK;
}
tk_status tk_argmax_advance(tk_ctx* ctx, const float* logits, int32_t n, int32_t* tok_dev, int32_t* pos_dev, tk_stream stream) {
    (void)ctx; (void)logits; (void)n; (void)tok_dev; (void)pos_dev; (void)stream;
    return TK_OK;
}
tk_status tk_argmax_rows(tk_ctx* ctx, const float* logits, int32_t n_rows, int32_t n_vocab, int32_t* out_dev, tk_stream stream) {
    (void)ctx; (void)logits; (void)n_rows; (void)n_vocab; (void)out_dev; (void)stream;
    return TK_OK;
}
tk_status tk_softplus_head_gate(tk_ctx* ctx, float* o, const float* g, int32_t n_head, int32_t head_dim, tk_stream stream) {
    (void)ctx; (void)o; (void)g; (void)n_head; (void)head_dim; (void)stream;
    return TK_OK;
}

tk_status tk_attn_decode(tk_ctx* ctx, const float* q, const void* k_pool, const void* v_pool, const uint16_t* k_scale, const uint16_t* v_scale, float* out, float* scratch, int32_t n_tok, int32_t n_head, int32_t n_kv, int32_t head_dim, int32_t head_dim_v, int32_t kv_stride, int32_t window, float scale, const int32_t* pos_dev, int32_t n_split, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)k_scale; (void)v_scale; (void)out; (void)scratch; (void)n_tok; (void)n_head; (void)n_kv; (void)head_dim; (void)head_dim_v; (void)kv_stride; (void)window; (void)scale; (void)pos_dev; (void)n_split; (void)stream;
    return TK_OK;
}
tk_status tk_attn_prefill_fa(tk_ctx* ctx, const float* q, const void* k_pool, const void* v_pool, float* out, int32_t n_tok, int32_t n_head, int32_t n_kv, int32_t head_dim, int64_t kv_stride, int64_t p0, float scale, void* mask, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)out; (void)n_tok; (void)n_head; (void)n_kv; (void)head_dim; (void)kv_stride; (void)p0; (void)scale; (void)mask; (void)stream;
    return TK_OK;
}
tk_status tk_laguna_prompt_attn(tk_ctx* ctx, const float* q, const uint16_t* k_pool, const uint16_t* v_pool, float* out, int32_t nq, int32_t q_pos0, int32_t n_head, int32_t n_kv, int32_t window, const int32_t* q_pos0_dev, int32_t n_split, float* part_acc, float* part_ml, bool merge, int32_t vt_stride, const uint16_t* k_scale, const uint16_t* v_scale, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)out; (void)nq; (void)q_pos0; (void)n_head; (void)n_kv; (void)window; (void)q_pos0_dev; (void)n_split; (void)part_acc; (void)part_ml; (void)merge; (void)vt_stride; (void)k_scale; (void)v_scale; (void)stream;
    return TK_OK;
}
tk_status tk_laguna_prompt_attn_wmma(tk_ctx* ctx, const float* q, const uint16_t* k_pool, const uint16_t* v_pool, float* out, int32_t nq, int32_t q_pos0, int32_t n_head, int32_t n_kv, int32_t window, int32_t vt_stride, const int32_t* q_pos0_dev, int32_t n_split, float* part_acc, float* part_ml, const uint16_t* k_scale, const uint16_t* v_scale, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)out; (void)nq; (void)q_pos0; (void)n_head; (void)n_kv; (void)window; (void)vt_stride; (void)q_pos0_dev; (void)n_split; (void)part_acc; (void)part_ml; (void)k_scale; (void)v_scale; (void)stream;
    return TK_OK;
}
tk_status tk_laguna_decode_attn(tk_ctx* ctx, const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const uint16_t* k_scale, const uint16_t* v_scale, int32_t n_head, int32_t n_kv, int32_t window, int32_t stride, const int32_t* pos_dev, int32_t n_split, float* part_acc, float* part_ml, int32_t n_tok, const int32_t* sel_dev, int32_t sel_k, bool sel_per_kv, int32_t sel_bs, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)k_scale; (void)v_scale; (void)n_head; (void)n_kv; (void)window; (void)stride; (void)pos_dev; (void)n_split; (void)part_acc; (void)part_ml; (void)n_tok; (void)sel_dev; (void)sel_k; (void)sel_per_kv; (void)sel_bs; (void)stream;
    return TK_OK;
}
tk_status tk_attn_merge_gate_quant(tk_ctx* ctx, const float* part_acc, const float* part_ml, const float* g, void* q8_out, int32_t n_head, int32_t n_split, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)part_acc; (void)part_ml; (void)g; (void)q8_out; (void)n_head; (void)n_split; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_kv_store_f16(tk_ctx* ctx, uint16_t* k_pool, uint16_t* v_pool, const float* kc, const float* vc, int32_t n, const int32_t* pos_dev, tk_stream stream) {
    (void)ctx; (void)k_pool; (void)v_pool; (void)kc; (void)vc; (void)n; (void)pos_dev; (void)stream;
    return TK_OK;
}
tk_status tk_v_store_t(tk_ctx* ctx, const float* vc, uint16_t* v_pool, int32_t nb, int32_t n_kv, int32_t p0, int32_t stride, tk_stream stream) {
    (void)ctx; (void)vc; (void)v_pool; (void)nb; (void)n_kv; (void)p0; (void)stride; (void)stream;
    return TK_OK;
}
tk_status tk_kv_store_q8(tk_ctx* ctx, const float* kc, const float* vc, void* k_pool, void* v_pool, uint16_t* k_scale, uint16_t* v_scale, int32_t nb, int32_t n_kv, int32_t p0, int32_t stride, tk_stream stream) {
    (void)ctx; (void)kc; (void)vc; (void)k_pool; (void)v_pool; (void)k_scale; (void)v_scale; (void)nb; (void)n_kv; (void)p0; (void)stride; (void)stream;
    return TK_OK;
}
tk_status tk_kbound_update(tk_ctx* ctx, const void* k_pool, const uint16_t* k_scale, void* bnd, int32_t nb_max, int32_t n_kv, int32_t b_first, int32_t b_last, int32_t pos_last, const int32_t* pos_dev, int32_t n_tok, int32_t bs, bool f16, tk_stream stream) {
    (void)ctx; (void)k_pool; (void)k_scale; (void)bnd; (void)nb_max; (void)n_kv; (void)b_first; (void)b_last; (void)pos_last; (void)pos_dev; (void)n_tok; (void)bs; (void)f16; (void)stream;
    return TK_OK;
}
tk_status tk_sparse_scores(tk_ctx* ctx, const float* q, const void* bnd, float* ub, int32_t nb_max, int32_t n_head, int32_t n_kv, const int32_t* pos_dev, int32_t n_tok, int32_t bs, bool f16, tk_stream stream) {
    (void)ctx; (void)q; (void)bnd; (void)ub; (void)nb_max; (void)n_head; (void)n_kv; (void)pos_dev; (void)n_tok; (void)bs; (void)f16; (void)stream;
    return TK_OK;
}
tk_status tk_sparse_select(tk_ctx* ctx, const float* ub, int32_t nb_max, int32_t n_head, const int32_t* pos_dev, int32_t n_tok, int32_t k, int32_t sink, int32_t recent, int32_t* sel_dev, int32_t groups, int32_t bs, uint32_t* scr_dev, tk_stream stream) {
    (void)ctx; (void)ub; (void)nb_max; (void)n_head; (void)pos_dev; (void)n_tok; (void)k; (void)sink; (void)recent; (void)sel_dev; (void)groups; (void)bs; (void)scr_dev; (void)stream;
    return TK_OK;
}
tk_status tk_attn_swa_sinks_mmq(tk_ctx* ctx, const float* q, const void* k_pool, const void* v_pool, const float* sinks, float* out, float* scratch, int32_t n_tok, int32_t n_head, int32_t n_kv, int32_t head_dim, int32_t head_dim_v, int32_t window, int32_t stride, float scale, const int32_t* pos_dev, void* q8_mmq, int32_t q8_layout, int32_t* q8_written, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)sinks; (void)out; (void)scratch; (void)n_tok; (void)n_head; (void)n_kv; (void)head_dim; (void)head_dim_v; (void)window; (void)stride; (void)scale; (void)pos_dev; (void)q8_mmq; (void)q8_layout; (void)stream;
    if (q8_written) *q8_written = 0;
    return TK_OK;
}
tk_status tk_attn_swa_sinks(tk_ctx* ctx, const float* q, const void* k_pool, const void* v_pool, const uint16_t* k_scale, const uint16_t* v_scale, const float* sinks, float* out, float* scratch, int32_t n_tok, int32_t n_head, int32_t n_kv, int32_t head_dim, int32_t head_dim_v, int32_t window, int32_t stride, float scale, const int32_t* pos_dev, int32_t n_split, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)k_scale; (void)v_scale; (void)sinks; (void)out; (void)scratch; (void)n_tok; (void)n_head; (void)n_kv; (void)head_dim; (void)head_dim_v; (void)window; (void)stride; (void)scale; (void)pos_dev; (void)n_split; (void)stream;
    return TK_OK;
}
tk_status tk_attn_swa_sinks_q8(tk_ctx* ctx, const float* q, const void* k_pool, const void* v_pool, const uint16_t* k_scale, const uint16_t* v_scale, const float* sinks, float* out, float* scratch, int32_t n_tok, int32_t n_head, int32_t n_kv, int32_t head_dim, int32_t head_dim_v, int32_t window, int32_t stride, float scale, const int32_t* pos_dev, int32_t n_split, tk_stream stream, void* q8_out) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)k_scale; (void)v_scale; (void)sinks; (void)out; (void)scratch; (void)n_tok; (void)n_head; (void)n_kv; (void)head_dim; (void)head_dim_v; (void)window; (void)stride; (void)scale; (void)pos_dev; (void)n_split; (void)stream; (void)q8_out;
    return TK_OK;
}
tk_status tk_axpy_norm_quant(tk_ctx* ctx, float* x, const float* ax, float alpha, const float* norm_w, float* h_out, void* q8_out, int32_t n, float eps, int32_t n_rows, tk_stream stream) {
    (void)ctx; (void)x; (void)ax; (void)alpha; (void)norm_w; (void)h_out; (void)q8_out; (void)n; (void)eps; (void)n_rows; (void)stream;
    return TK_OK;
}
tk_status tk_mimo_rope_qk_store(tk_ctx* ctx, float* q, float* k, const float* v, int32_t n_tok, int32_t n_head_q, int32_t n_head_kv, int32_t head_dim, int32_t v_dim, int32_t n_rot, const float* cos_tab, const float* sin_tab, const int32_t* pos_q_dev, const int32_t* pos_k_dev, void* k_pool, void* v_pool, const int32_t* store_pos_dev, int32_t ring, tk_stream stream) {
    (void)ctx; (void)q; (void)k; (void)v; (void)n_tok; (void)n_head_q; (void)n_head_kv; (void)head_dim; (void)v_dim; (void)n_rot; (void)cos_tab; (void)sin_tab; (void)pos_q_dev; (void)pos_k_dev; (void)k_pool; (void)v_pool; (void)store_pos_dev; (void)ring; (void)stream;
    return TK_OK;
}
tk_status tk_attn_k192_v128(tk_ctx* ctx, const float* q, const void* k_pool, const void* v_pool, const uint16_t* k_scale, const uint16_t* v_scale, float* out, float* scratch, int32_t n_tok, int32_t n_head, int32_t n_kv, int32_t stride, float scale, const int32_t* pos_dev, int32_t n_split, tk_stream stream) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)k_scale; (void)v_scale; (void)out; (void)scratch; (void)n_tok; (void)n_head; (void)n_kv; (void)stride; (void)scale; (void)pos_dev; (void)n_split; (void)stream;
    return TK_OK;
}
tk_status tk_attn_k192_v128_q8(tk_ctx* ctx, const float* q, const void* k_pool, const void* v_pool, const uint16_t* k_scale, const uint16_t* v_scale, float* out, float* scratch, int32_t n_tok, int32_t n_head, int32_t n_kv, int32_t stride, float scale, const int32_t* pos_dev, int32_t n_split, tk_stream stream, void* q8_out) {
    (void)ctx; (void)q; (void)k_pool; (void)v_pool; (void)k_scale; (void)v_scale; (void)out; (void)scratch; (void)n_tok; (void)n_head; (void)n_kv; (void)stride; (void)scale; (void)pos_dev; (void)n_split; (void)stream; (void)q8_out;
    return TK_OK;
}
tk_status tk_dsa_decode_indexer(tk_ctx* ctx, const float* q_idx, const void* k_idx_pool, const float* weights, const int32_t* pos_dev, int32_t n_tok, int32_t idx_heads, int32_t idx_dim, int32_t top_k, int32_t capacity, float* scores_scratch, int32_t* topk_ids_dev, tk_stream stream) {
    (void)ctx; (void)q_idx; (void)k_idx_pool; (void)weights; (void)pos_dev; (void)n_tok; (void)idx_heads; (void)idx_dim; (void)top_k; (void)capacity; (void)scores_scratch; (void)topk_ids_dev; (void)stream;
    return TK_OK;
}

tk_status tk_gdn_preprocess(tk_ctx* ctx, float* conv_state, const float* qkv, const float* conv_w, float* conv_out, const float* alpha, const float* dt, const float* ssm_a, float* gate_out, float* beta_inout, int64_t channels, int64_t d_conv, int64_t h_k, int64_t h_v, int64_t state_size, float l2_eps, tk_stream stream) {
    (void)ctx; (void)conv_state; (void)qkv; (void)conv_w; (void)conv_out; (void)alpha; (void)dt; (void)ssm_a; (void)gate_out; (void)beta_inout; (void)channels; (void)d_conv; (void)h_k; (void)h_v; (void)state_size; (void)l2_eps; (void)stream;
    return TK_OK;
}
tk_status tk_gdn_step(tk_ctx* ctx, float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, float* out, int64_t state_size, int64_t h_k, int64_t h_v, tk_stream stream) {
    (void)ctx; (void)state; (void)q; (void)k; (void)v; (void)gate; (void)beta; (void)out; (void)state_size; (void)h_k; (void)h_v; (void)stream;
    return TK_OK;
}
tk_status tk_gdn_seq(tk_ctx* ctx, float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, const float* z, const float* gamma, float* out, int32_t n_tok, int64_t state_size, int64_t h_k, int64_t h_v, float norm_eps, bool silu_gate, tk_stream stream) {
    (void)ctx; (void)state; (void)q; (void)k; (void)v; (void)gate; (void)beta; (void)z; (void)gamma; (void)out; (void)n_tok; (void)state_size; (void)h_k; (void)h_v; (void)norm_eps; (void)silu_gate; (void)stream;
    return TK_OK;
}
tk_status tk_gdn_fused(tk_ctx* ctx, float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta, const float* z, const float* gamma, float eps, float* y, int32_t h_k, int32_t h_v, bool silu_gate, tk_stream stream) {
    (void)ctx; (void)state; (void)q; (void)k; (void)v; (void)gate; (void)beta; (void)z; (void)gamma; (void)eps; (void)y; (void)h_k; (void)h_v; (void)silu_gate; (void)stream;
    return TK_OK;
}
tk_status tk_kda_step(tk_ctx* ctx, float* state, const float* q, const float* k, const float* v, const float* gk, const float* beta, const float* z, const float* gamma, float eps, float* out, int32_t n_tok, int32_t state_size, int32_t h_k, int32_t h_v, tk_stream stream) {
    (void)ctx; (void)state; (void)q; (void)k; (void)v; (void)gk; (void)beta; (void)z; (void)gamma; (void)eps; (void)out; (void)n_tok; (void)state_size; (void)h_k; (void)h_v; (void)stream;
    return TK_OK;
}

tk_status tk_prefill_gemm(tk_ctx* ctx, tk_qtype w_qtype, const void* w, const void* x, tk_qtype x_qtype, float* y, int64_t n_tok, int64_t n_out, int64_t n_in, int64_t ldy, int64_t ldx, float beta, tk_stream stream) {
    (void)ctx; (void)w_qtype; (void)w; (void)x; (void)x_qtype; (void)y; (void)n_tok; (void)n_out; (void)n_in; (void)ldy; (void)ldx; (void)beta; (void)stream;
    return TK_OK;
}
tk_status tk_prefill_mmq(tk_ctx* ctx, tk_qtype w_qtype, const void* w_blocks, const void* x_q8_1, float* y, int64_t n_tok, int64_t n_out, int64_t n_in, int64_t ldy, tk_stream stream) {
    (void)ctx; (void)w_qtype; (void)w_blocks; (void)x_q8_1; (void)y; (void)n_tok; (void)n_out; (void)n_in; (void)ldy; (void)stream;
    return TK_OK;
}
tk_status tk_prefill_moe_mmq(tk_ctx* ctx, const tk_moe_mmq_product* product, tk_stream stream) {
    (void)ctx; (void)stream;
    return product ? TK_OK : TK_ERR_NULL_PTR;
}
tk_status tk_prefill_moe_mmq_gu(tk_ctx* ctx, const tk_moe_mmq_product* gate, const void* w_up, float swiglu_limit, tk_stream stream) {
    (void)ctx; (void)gate; (void)w_up; (void)swiglu_limit; (void)stream;
    return TK_ERR_UNSUPPORTED;
}
tk_status tk_prefill_moe_mmq_gu_q8(tk_ctx* ctx, const tk_moe_mmq_product* gate, const void* w_up, float swiglu_limit, int32_t down_qtype, void* xq_out, int64_t xq_rows, tk_stream stream) {
    (void)ctx; (void)gate; (void)w_up; (void)swiglu_limit; (void)down_qtype; (void)xq_out; (void)xq_rows; (void)stream;
    return TK_ERR_UNSUPPORTED;
}
tk_status tk_prefill_moe_mmq_iq(tk_ctx* ctx, tk_qtype gu_qtype, tk_qtype down_qtype, const void* const* expert_blobs, int32_t e0, int32_t e1, int32_t n_expert, int64_t n_routed_rows, int64_t n_embd, int64_t n_ff, const void* group_scratch, const void* x_act, const int32_t* src_rows_dev, void* h_act, float* down_out, tk_stream stream) {
    (void)ctx; (void)gu_qtype; (void)down_qtype; (void)expert_blobs; (void)e0; (void)e1; (void)n_expert; (void)n_routed_rows; (void)n_embd; (void)n_ff; (void)group_scratch; (void)x_act; (void)src_rows_dev; (void)h_act; (void)down_out; (void)stream;
    return TK_OK;
}

size_t tk_q8_1_bytes(int32_t n_in, int32_t ncols) {
    return (size_t)36 * (size_t)(n_in / 32) * (size_t)ncols;
}
tk_status tk_swiglu_quantize_q8_1(tk_ctx* ctx, const float* gate, const float* up, void* x_q8_1, int32_t n_in, int32_t ncols, tk_stream stream) {
    (void)ctx; (void)gate; (void)up; (void)x_q8_1; (void)n_in; (void)ncols; (void)stream;
    return TK_OK;
}
tk_status tk_mmvq_w32_multi(tk_ctx* ctx, tk_qtype qtype, int32_t n_mats, const void* const* w_ptrs, float* const* y_ptrs, const int32_t* n_out_per_mat, const void* x_q8_1, int32_t n_in, int32_t ncols, tk_stream stream) {
    (void)ctx; (void)qtype; (void)n_mats; (void)w_ptrs; (void)y_ptrs; (void)n_out_per_mat; (void)x_q8_1; (void)n_in; (void)ncols; (void)stream;
    return TK_OK;
}
tk_status tk_mmvq_batch(tk_ctx* ctx, const tk_mmvq_item* items, int32_t n_items, tk_stream stream) {
    (void)ctx; (void)stream;
    return (items && n_items > 0) ? TK_OK : TK_ERR_INVALID_ARG;
}

tk_status tk_q36_rms(tk_ctx* ctx, int32_t rows, const float* x, const float* w, float* y, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)w; (void)y; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_rms_q8(tk_ctx* ctx, int32_t rows, const float* x, const float* w, float* y, void* q8, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)w; (void)y; (void)q8; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_rms_ld(tk_ctx* ctx, int32_t rows, const float* x, int32_t ldx, const float* w, float* out, int32_t ld_out, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)ldx; (void)w; (void)out; (void)ld_out; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_rms_ld_q8(tk_ctx* ctx, int32_t rows, const float* x, int32_t ldx, const float* w, float* out, int32_t ld_out, void* q8, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)ldx; (void)w; (void)out; (void)ld_out; (void)q8; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_add_rms(tk_ctx* ctx, int32_t rows, float* x, const float* y, const float* w, float* h, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)y; (void)w; (void)h; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_add_rms_q8(tk_ctx* ctx, int32_t rows, float* x, const float* y, const float* w, float* h, void* q8, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)y; (void)w; (void)h; (void)q8; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_add_rms_exact_q8(tk_ctx* ctx, int32_t rows, float* x, const float* y, const float* w, float* h, void* q8, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)y; (void)w; (void)h; (void)q8; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_moe_out(tk_ctx* ctx, int32_t rows, float* x, const float* parts, const int32_t* prow, const float* wts, int32_t k, const float* shared, const float* sgate, int32_t sg_stride, const float* wn, float* h, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)parts; (void)prow; (void)wts; (void)k; (void)shared; (void)sgate; (void)sg_stride; (void)wn; (void)h; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_moe_out_q8(tk_ctx* ctx, int32_t rows, float* x, const float* parts, const int32_t* prow, const float* wts, int32_t k, const float* shared, const float* sgate, int32_t sg_stride, const float* wn, float* h, void* q8, int32_t n, float eps, tk_stream stream) {
    (void)ctx; (void)rows; (void)x; (void)parts; (void)prow; (void)wts; (void)k; (void)shared; (void)sgate; (void)sg_stride; (void)wn; (void)h; (void)q8; (void)n; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_ab(tk_ctx* ctx, int32_t n_tok, const float* wa, const float* wb, const float* x, const float* dt, const float* ssm_a, float* alpha, float* beta, int32_t n_in, int32_t hv, tk_stream stream) {
    (void)ctx; (void)n_tok; (void)wa; (void)wb; (void)x; (void)dt; (void)ssm_a; (void)alpha; (void)beta; (void)n_in; (void)hv; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_ab2(tk_ctx* ctx, const float* wa, const float* wb, const float* x, const float* dt, const float* ssm_a, float* alpha, float* beta, int32_t n_in, int32_t hv, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)wa; (void)wb; (void)x; (void)dt; (void)ssm_a; (void)alpha; (void)beta; (void)n_in; (void)hv; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_add(tk_ctx* ctx, float* x, const float* y, int64_t n, tk_stream stream) {
    (void)ctx; (void)x; (void)y; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_swiglu(tk_ctx* ctx, const float* g, const float* u, float* h, int64_t n, tk_stream stream) {
    (void)ctx; (void)g; (void)u; (void)h; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gemv_f32(tk_ctx* ctx, int32_t n_tok, const float* w, const float* x, float* y, int32_t n_in, int32_t n_out, tk_stream stream) {
    (void)ctx; (void)n_tok; (void)w; (void)x; (void)y; (void)n_in; (void)n_out; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_gb(tk_ctx* ctx, float* alpha, const float* dt, const float* ssm_a, float* beta, int32_t hv, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)alpha; (void)dt; (void)ssm_a; (void)beta; (void)hv; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_gb2(tk_ctx* ctx, const float* ab, const float* dt, const float* ssm_a, float* alpha, float* beta, int32_t hv, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)ab; (void)dt; (void)ssm_a; (void)alpha; (void)beta; (void)hv; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gate_rows(tk_ctx* ctx, float* sh, const float* gate, int32_t n, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)sh; (void)gate; (void)n; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_attn_prep(tk_ctx* ctx, int32_t n_tok, const float* qg, const float* kx, const float* vx, const float* qw, const float* kw, float* q, float* gate, void* kc, void* vc, int32_t h, int32_t hkv, int32_t hd, int32_t n_rot, const int32_t* pos0, int32_t ctx_len, float base, float eps, void* vt, int32_t vt_ctx, tk_stream stream) {
    (void)ctx; (void)n_tok; (void)qg; (void)kx; (void)vx; (void)qw; (void)kw; (void)q; (void)gate; (void)kc; (void)vc; (void)h; (void)hkv; (void)hd; (void)n_rot; (void)pos0; (void)ctx_len; (void)base; (void)eps; (void)vt; (void)vt_ctx; (void)stream;
    return TK_OK;
}
tk_status tk_q36_attn_part(tk_ctx* ctx, int32_t grid_x, int32_t grid_y, const float* q, const void* kc, const void* vc, const int32_t* pos0, int32_t t0, int32_t h, int32_t hkv, int32_t ctx_len, float scale, int32_t n_chunk, float* pacc, float* pm, float* pl, tk_stream stream) {
    (void)ctx; (void)grid_x; (void)grid_y; (void)q; (void)kc; (void)vc; (void)pos0; (void)t0; (void)h; (void)hkv; (void)ctx_len; (void)scale; (void)n_chunk; (void)pacc; (void)pm; (void)pl; (void)stream;
    return TK_OK;
}
tk_status tk_q36_attn_merge(tk_ctx* ctx, int32_t rows, int32_t hd, const float* pacc, const float* pm, const float* pl, const float* gate, const int32_t* pos0, int32_t t0, int32_t h, int32_t n_chunk, float* out, int32_t chunk, void* q8, tk_stream stream) {
    (void)ctx; (void)rows; (void)hd; (void)pacc; (void)pm; (void)pl; (void)gate; (void)pos0; (void)t0; (void)h; (void)n_chunk; (void)out; (void)chunk; (void)q8; (void)stream;
    return TK_OK;
}
tk_status tk_q36_attn_wmma(tk_ctx* ctx, bool w8, int32_t grid_x, int32_t grid_y, const float* q, const void* kc, const void* vt, const int32_t* pos0, int32_t n_tok, int32_t h, int32_t hkv, int32_t ctx_len, int32_t vt_ctx, float sc, int32_t chunk, int32_t n_chunk, float* pacc, float* pm, float* pl, tk_stream stream) {
    (void)ctx; (void)w8; (void)grid_x; (void)grid_y; (void)q; (void)kc; (void)vt; (void)pos0; (void)n_tok; (void)h; (void)hkv; (void)ctx_len; (void)vt_ctx; (void)sc; (void)chunk; (void)n_chunk; (void)pacc; (void)pm; (void)pl; (void)stream;
    return TK_OK;
}
tk_status tk_q36_group(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t k, int32_t n_expert, const uint8_t* blobs, uint64_t blob, uint64_t* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, tk_stream stream) {
    (void)ctx; (void)ids; (void)n_ent; (void)k; (void)n_expert; (void)blobs; (void)blob; (void)grp_ptr; (void)grp_start; (void)n_groups; (void)ent_dst; (void)ent_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_group_hy(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t k, int32_t n_expert, const int32_t* slot_of, const uint8_t* blobs, uint64_t blob, uint64_t* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok, tk_stream stream) {
    (void)ctx; (void)ids; (void)n_ent; (void)k; (void)n_expert; (void)slot_of; (void)blobs; (void)blob; (void)grp_ptr; (void)grp_start; (void)n_groups; (void)ent_dst; (void)ent_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_merge_cpu(tk_ctx* ctx, int32_t n_ent, float* parts, const float* cpu, const int32_t* miss, int32_t n, tk_stream stream) {
    (void)ctx; (void)n_ent; (void)parts; (void)cpu; (void)miss; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_attn_softmax16(tk_ctx* ctx, int32_t h, const float* s_in, void* s16_out, int32_t ld, int32_t n_tok, int32_t p_cols, int32_t pos0, float scale, tk_stream stream) {
    (void)ctx; (void)h; (void)s_in; (void)s16_out; (void)ld; (void)n_tok; (void)p_cols; (void)pos0; (void)scale; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gate_mul(tk_ctx* ctx, float* att, const float* gate, int64_t n, tk_stream stream) {
    (void)ctx; (void)att; (void)gate; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_route_count(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t* cnt, tk_stream stream) {
    (void)ctx; (void)ids; (void)n_ent; (void)cnt; (void)stream;
    return TK_OK;
}
tk_status tk_q36_route_scan(tk_ctx* ctx, const int32_t* cnt, int32_t n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows, tk_stream stream) {
    (void)ctx; (void)cnt; (void)n_expert; (void)bounds; (void)cur; (void)max_rows; (void)stream;
    return TK_OK;
}
tk_status tk_q36_route_scatter(tk_ctx* ctx, const int32_t* ids, int32_t n_ent, int32_t k, int32_t* cur, int32_t* dst, int32_t* src, int32_t* inv, tk_stream stream) {
    (void)ctx; (void)ids; (void)n_ent; (void)k; (void)cur; (void)dst; (void)src; (void)inv; (void)stream;
    return TK_OK;
}
tk_status tk_q36_argmax(tk_ctx* ctx, int32_t grid_x, int32_t grid_y, const float* x, int32_t n, int32_t* out, float* pv, int32_t* pi, bool stage2, const int32_t* map, tk_stream stream) {
    (void)ctx; (void)grid_x; (void)grid_y; (void)x; (void)n; (void)out; (void)pv; (void)pi; (void)stage2; (void)map; (void)stream;
    return TK_OK;
}
tk_status tk_q36_argmax_samp(tk_ctx* ctx, const float* x, int32_t n, float* pv, int32_t* pi, const tk_q36_draft_samp* ds_dev, int32_t step, const float* thr, tk_stream stream) {
    (void)ctx; (void)x; (void)n; (void)pv; (void)pi; (void)ds_dev; (void)step; (void)thr; (void)stream;
    return TK_OK;
}
tk_status tk_q36_lse(tk_ctx* ctx, const float* x, int32_t n, const tk_q36_draft_samp* ds_dev, float* out, float* thr_out, tk_stream stream) {
    (void)ctx; (void)x; (void)n; (void)ds_dev; (void)out; (void)thr_out; (void)stream;
    return TK_OK;
}
tk_status tk_q36_ctl_end(tk_ctx* ctx, int32_t* ctl, bool flip, tk_stream stream) {
    (void)ctx; (void)ctl; (void)flip; (void)stream;
    return TK_OK;
}
tk_status tk_q36_round_prep(tk_ctx* ctx, const int32_t* rctl, int32_t* tok, int32_t* tok_m, int32_t* pos0, int32_t* pos0_m, int32_t* gctl, int32_t n, tk_stream stream) {
    (void)ctx; (void)rctl; (void)tok; (void)tok_m; (void)pos0; (void)pos0_m; (void)gctl; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_row_copy(tk_ctx* ctx, float* dst, const float* src, const int32_t* idx, int32_t n, tk_stream stream) {
    (void)ctx; (void)dst; (void)src; (void)idx; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_chain_prep(tk_ctx* ctx, const int32_t* tok, int32_t* tok_m, const int32_t* pos0, int32_t* pos0_m, int32_t k, tk_stream stream) {
    (void)ctx; (void)tok; (void)tok_m; (void)pos0; (void)pos0_m; (void)k; (void)stream;
    return TK_OK;
}
tk_status tk_q36_sgemm_nt(tk_ctx* ctx, int32_t tm, int32_t grid_x, int32_t grid_y, int32_t grid_z, const float* w, const float* x, float* out, int32_t k_dim, int32_t n_out, int32_t n_tok, int32_t ldo, int32_t kz, tk_stream stream) {
    (void)ctx; (void)tm; (void)grid_x; (void)grid_y; (void)grid_z; (void)w; (void)x; (void)out; (void)k_dim; (void)n_out; (void)n_tok; (void)ldo; (void)kz; (void)stream;
    return TK_OK;
}
tk_status tk_q36_sum_splits(tk_ctx* ctx, const float* part, float* out, int64_t n, int32_t splits, tk_stream stream) {
    (void)ctx; (void)part; (void)out; (void)n; (void)splits; (void)stream;
    return TK_OK;
}
tk_status tk_q36_to_f16(tk_ctx* ctx, const float* x, void* y_f16, int64_t n, tk_stream stream) {
    (void)ctx; (void)x; (void)y_f16; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_conv_seq(tk_ctx* ctx, float* hist, const float* qkv, const float* w, float* h, int32_t c_channels, int32_t qk_heads, float eps, int32_t n_tok, bool write_state, const tk_q36_gdn_pend* pd, tk_stream stream) {
    (void)ctx; (void)hist; (void)qkv; (void)w; (void)h; (void)c_channels; (void)qk_heads; (void)eps; (void)n_tok; (void)write_state; (void)pd; (void)stream;
    return TK_OK;
}
tk_status tk_q36_conv_ab(tk_ctx* ctx, float* hist, const float* qkv, const float* w, float* hc, int32_t c_channels, int32_t qk_heads, float eps, int32_t n_tok, bool write_state, const tk_q36_gdn_pend* pd, const float* wa, const float* wb, const float* x, const float* dt, const float* ssm_a, float* alpha, float* beta, int32_t n_in, int32_t hv, tk_stream stream) {
    (void)ctx; (void)hist; (void)qkv; (void)w; (void)hc; (void)c_channels; (void)qk_heads; (void)eps; (void)n_tok; (void)write_state; (void)pd; (void)wa; (void)wb; (void)x; (void)dt; (void)ssm_a; (void)alpha; (void)beta; (void)n_in; (void)hv; (void)stream;
    return TK_OK;
}
tk_status tk_q36_conv_par(tk_ctx* ctx, const float* hist, const float* qkv, const float* w, float* h, int32_t c_channels, int32_t qk_heads, float eps, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)hist; (void)qkv; (void)w; (void)h; (void)c_channels; (void)qk_heads; (void)eps; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_conv_hist(tk_ctx* ctx, float* hist, const float* qkv, int32_t c_channels, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)hist; (void)qkv; (void)c_channels; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_seq(tk_ctx* ctx, float* state, const float* h, int32_t c_channels, const float* gate, const float* beta, const float* z, const float* gamma, float eps, float* y, int32_t h_k, int32_t h_v, int32_t n_tok, bool write_state, const tk_q36_gdn_pend* pd, void* yq, tk_stream stream) {
    (void)ctx; (void)state; (void)h; (void)c_channels; (void)gate; (void)beta; (void)z; (void)gamma; (void)eps; (void)y; (void)h_k; (void)h_v; (void)n_tok; (void)write_state; (void)pd; (void)yq; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_seq_cols(tk_ctx* ctx, float* state, const float* h, int32_t c_channels, const float* gate, const float* beta, float* o_raw, int32_t h_k, int32_t h_v, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)state; (void)h; (void)c_channels; (void)gate; (void)beta; (void)o_raw; (void)h_k; (void)h_v; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_seq_warp(tk_ctx* ctx, int32_t ncol, float* state, const float* h, int32_t c_channels, const float* gate, const float* beta, float* o_raw, int32_t h_k, int32_t h_v, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)ncol; (void)state; (void)h; (void)c_channels; (void)gate; (void)beta; (void)o_raw; (void)h_k; (void)h_v; (void)n_tok; (void)stream;
    return TK_OK;
}
tk_status tk_q36_gdn_norm(tk_ctx* ctx, int32_t n_tok, const float* o_raw, const float* z, const float* gamma, float eps, float* y, int32_t h_v, tk_stream stream) {
    (void)ctx; (void)n_tok; (void)o_raw; (void)z; (void)gamma; (void)eps; (void)y; (void)h_v; (void)stream;
    return TK_OK;
}
tk_status tk_q36_flush_read(tk_ctx* ctx, const void* p, size_t n, float* out, tk_stream stream) {
    (void)ctx; (void)p; (void)n; (void)out; (void)stream;
    return TK_OK;
}
tk_status tk_q36_draft_conf(tk_ctx* ctx, const float* x, int32_t n, const float* tau, const float* lmin, const int32_t* pick, const int32_t* map, float* conf, float* top, int32_t step, const float* lse, const float* thr, tk_stream stream) {
    (void)ctx; (void)x; (void)n; (void)tau; (void)lmin; (void)pick; (void)map; (void)conf; (void)top; (void)step; (void)lse; (void)thr; (void)stream;
    return TK_OK;
}
tk_status tk_q36_draft_support(tk_ctx* ctx, const float* x, int32_t n, const float* tau, const float* lmin, const float* top_p, const int32_t* top_k, float* lse, float* thr, tk_stream stream) {
    (void)ctx; (void)x; (void)n; (void)tau; (void)lmin; (void)top_p; (void)top_k; (void)lse; (void)thr; (void)stream;
    return TK_OK;
}
int32_t tk_q36_dflash_attn_chunks(int32_t window) {
    return (window + 127) / 128;
}
size_t tk_q36_dflash_attn_scratch_floats(int32_t heads, int32_t hd, int32_t window) {
    return (size_t)heads * 16 * (size_t)((window + 127) / 128) * (size_t)(hd + 2);
}
tk_status tk_q36_dflash_rms_rows(tk_ctx* ctx, const float* x, const float* w, float* out, int32_t n, int32_t rows, float eps, tk_stream stream) {
    (void)ctx; (void)x; (void)w; (void)out; (void)n; (void)rows; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_rms_rows_q8(tk_ctx* ctx, const float* x, const float* w, float* out, void* q8, int32_t n, int32_t rows, float eps, tk_stream stream) {
    (void)ctx; (void)x; (void)w; (void)out; (void)q8; (void)n; (void)rows; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_sum5_rms(tk_ctx* ctx, const float* const* ys, const float* w, float* out, int32_t n, int32_t rows, float eps, tk_stream stream) {
    (void)ctx; (void)ys; (void)w; (void)out; (void)n; (void)rows; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_sum5_rms_q8(tk_ctx* ctx, const float* const* ys, const float* w, float* out, void* q8, int32_t n, int32_t rows, float eps, tk_stream stream) {
    (void)ctx; (void)ys; (void)w; (void)out; (void)q8; (void)n; (void)rows; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_dyn_conv(tk_ctx* ctx, const float* x_in, const float* dyn_w, const float* base_w, float* out, int32_t n, int32_t rows, int32_t group, int32_t side, tk_stream stream) {
    (void)ctx; (void)x_in; (void)dyn_w; (void)base_w; (void)out; (void)n; (void)rows; (void)group; (void)side; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_dyn_conv_q8(tk_ctx* ctx, const float* x_in, const float* dyn_w, const float* base_w, float* out, void* q8, int32_t n, int32_t rows, int32_t group, int32_t side, tk_stream stream) {
    (void)ctx; (void)x_in; (void)dyn_w; (void)base_w; (void)out; (void)q8; (void)n; (void)rows; (void)group; (void)side; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_qk_norm_rope(tk_ctx* ctx, float* x, const float* w, int32_t heads, int32_t hd, int32_t rows, const int32_t* pos, float theta, float eps, tk_stream stream) {
    (void)ctx; (void)x; (void)w; (void)heads; (void)hd; (void)rows; (void)pos; (void)theta; (void)eps; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_kv_store(tk_ctx* ctx, const float* k, const float* v, float* kring, float* vring, int32_t width, int32_t rows, const int32_t* pos, int32_t ring, tk_stream stream) {
    (void)ctx; (void)k; (void)v; (void)kring; (void)vring; (void)width; (void)rows; (void)pos; (void)ring; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_kv_store_f16(tk_ctx* ctx, const float* k, const float* v, void* kring_f16, void* vring_f16, int32_t k_width, int32_t v_width, int32_t rows, const int32_t* pos, int32_t ring, tk_stream stream) {
    (void)ctx; (void)k; (void)v; (void)kring_f16; (void)vring_f16; (void)k_width; (void)v_width; (void)rows; (void)pos; (void)ring; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_block_attn(tk_ctx* ctx, const float* q, const float* kring, const float* vring, const int32_t* pos, int32_t heads, int32_t heads_kv, int32_t hd, int32_t ring, int32_t window, float scale, float* part, float* out, tk_stream stream) {
    (void)ctx; (void)q; (void)kring; (void)vring; (void)pos; (void)heads; (void)heads_kv; (void)hd; (void)ring; (void)window; (void)scale; (void)part; (void)out; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_swiglu(tk_ctx* ctx, const float* g, const float* u, float* out, int64_t n, tk_stream stream) {
    (void)ctx; (void)g; (void)u; (void)out; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_swiglu_q8(tk_ctx* ctx, const float* g, const float* u, float* out, void* q8, int64_t n, tk_stream stream) {
    (void)ctx; (void)g; (void)u; (void)out; (void)q8; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_add_inplace(tk_ctx* ctx, float* x, const float* y, int64_t n, tk_stream stream) {
    (void)ctx; (void)x; (void)y; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_block_tokens(tk_ctx* ctx, const int32_t* anchor, int32_t* out, int32_t n, int32_t mask, tk_stream stream) {
    (void)ctx; (void)anchor; (void)out; (void)n; (void)mask; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_set_i32(tk_ctx* ctx, int32_t* p, int32_t v, tk_stream stream) {
    (void)ctx; (void)p; (void)v; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_top16(tk_ctx* ctx, const float* logits, int32_t nv, int32_t rows, const int32_t* map, int32_t* cand, float* unary, tk_stream stream) {
    (void)ctx; (void)logits; (void)nv; (void)rows; (void)map; (void)cand; (void)unary; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_select_chain(tk_ctx* ctx, const int32_t* cand, const float* unary, const float* gate, const void* pred, const void* succ, const int32_t* anchor, int32_t* out, float* score, int32_t n, tk_stream stream) {
    (void)ctx; (void)cand; (void)unary; (void)gate; (void)pred; (void)succ; (void)anchor; (void)out; (void)score; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_dflash_select_chain_q(tk_ctx* ctx, const int32_t* cand, const float* unary, const float* gate, const void* pred, const void* succ, const int32_t* anchor, int32_t* out, float* score, float* q, const float* tau, const uint32_t* seed, const uint32_t* ctr, int32_t n, tk_stream stream) {
    (void)ctx; (void)cand; (void)unary; (void)gate; (void)pred; (void)succ; (void)anchor; (void)out; (void)score; (void)q; (void)tau; (void)seed; (void)ctr; (void)n; (void)stream;
    return TK_OK;
}
bool tk_q36_mmvq_supported(int32_t ggml_type) {
    (void)ggml_type;
    return true;
}
bool tk_q36_q8_0_pack_should(int32_t ggml_type, int32_t n_in, int32_t n_out) {
    (void)ggml_type; (void)n_in; (void)n_out;
    return false;
}
void tk_q36_q8_0_pack_host(const void* gguf_blocks, void* out, int32_t n_in, int32_t n_out) {
    (void)gguf_blocks; (void)out; (void)n_in; (void)n_out;
}
void tk_q36_q8_0_packed_register(const void* gguf_weights, const void* packed, int32_t n_in, int32_t n_out) {
    (void)gguf_weights; (void)packed; (void)n_in; (void)n_out;
}
void tk_q36_q8_0_packed_unregister(const void* gguf_weights) {
    (void)gguf_weights;
}
tk_status tk_q36_embed_rows(tk_ctx* ctx, int32_t ggml_type, const void* table, size_t row_bytes, const int32_t* tok_dev, int64_t n_tok, int64_t n_embd, float* out, tk_stream stream) {
    (void)ctx; (void)ggml_type; (void)table; (void)row_bytes; (void)tok_dev; (void)n_tok; (void)n_embd; (void)out; (void)stream;
    return TK_OK;
}
bool tk_q36_mmq_built(void) {
    return true;
}
bool tk_q36_mmq_supported_fits(int32_t ggml_type, int64_t w_rows) {
    (void)ggml_type; (void)w_rows;
    return true;
}
int32_t tk_q36_mmq_q8_layout(int32_t ggml_type) {
    (void)ggml_type;
    return 0;
}
size_t tk_q36_mmq_q8_bytes(int64_t rows, int64_t cols) {
    return (size_t)rows * (size_t)cols * 2;
}
size_t tk_q36_mmq_matrix_bytes(int32_t ggml_type, int64_t rows, int64_t cols) {
    (void)ggml_type;
    return (size_t)rows * (size_t)cols;
}
tk_status tk_q36_mmq_iota(tk_ctx* ctx, int32_t* dst, int64_t n, tk_stream stream) {
    (void)ctx; (void)dst; (void)n; (void)stream;
    return TK_OK;
}
tk_status tk_q36_mmq_set_bounds(tk_ctx* ctx, int32_t* bounds, int32_t rows, tk_stream stream) {
    (void)ctx; (void)bounds; (void)rows; (void)stream;
    return TK_OK;
}
tk_status tk_q36_mmq_quantize(tk_ctx* ctx, const float* x, const int32_t* ids, void* xq, int32_t ggml_type, int64_t cols, int64_t ld, int64_t rows, tk_stream stream) {
    (void)ctx; (void)x; (void)ids; (void)xq; (void)ggml_type; (void)cols; (void)ld; (void)rows; (void)stream;
    return TK_OK;
}
tk_status tk_q36_mmq_quantize_scatter(tk_ctx* ctx, const float* x, const int32_t* smap, void* xq, int32_t ggml_type, int64_t cols, int64_t n_tok, int64_t n_rows, int32_t k, tk_stream stream) {
    (void)ctx; (void)x; (void)smap; (void)xq; (void)ggml_type; (void)cols; (void)n_tok; (void)n_rows; (void)k; (void)stream;
    return TK_OK;
}
tk_status tk_q36_mmq_swiglu(tk_ctx* ctx, const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, tk_stream stream) {
    (void)ctx; (void)gu; (void)h; (void)rows; (void)n_ff; (void)interleaved; (void)stream;
    return TK_OK;
}
bool tk_q36_fattn_supported(int32_t head_dim) {
    return head_dim == 256;
}
size_t tk_q36_fattn_mask_bytes(int64_t n_tok, int64_t n_kv) {
    return (size_t)n_tok * (size_t)n_kv * 2;
}
tk_status tk_q36_prefill_init_scratch(tk_ctx* ctx, int64_t scratch_elems) {
    (void)ctx; (void)scratch_elems;
    return TK_OK;
}
tk_status tk_q36_attn_blas_prefill(tk_ctx* ctx, const float* q, void* q16, const void* kc, const void* vc, float* att_s, void* att_p16, float* att, const float* gate, int32_t T, int32_t H, int32_t HKV, int32_t HD, int32_t ctx_len, int64_t p0, size_t score_floats, tk_stream stream) {
    (void)ctx; (void)q; (void)q16; (void)kc; (void)vc; (void)att_s; (void)att_p16; (void)att; (void)gate; (void)T; (void)H; (void)HKV; (void)HD; (void)ctx_len; (void)p0; (void)score_floats; (void)stream;
    return TK_OK;
}
tk_status tk_q36_router_topk(tk_ctx* ctx, const float* logits, int32_t* ids, float* weights, int32_t n_expert, int32_t k, int32_t n_tok, tk_stream stream) {
    (void)ctx; (void)logits; (void)ids; (void)weights; (void)n_expert; (void)k; (void)n_tok; (void)stream;
    return TK_OK;
}
bool tk_q36_expert_supported(int32_t gu_type, int32_t d_type, int64_t n_embd, int64_t n_ff) {
    (void)gu_type; (void)d_type; (void)n_embd; (void)n_ff;
    return true;
}
void tk_q36_expert_layout(int32_t gu_type, int32_t d_type, int64_t n_embd, int64_t n_ff, size_t* out_gu_row, size_t* out_d_row, size_t* out_up_off, size_t* out_down_off, size_t* out_bytes, size_t* out_tail_off) {
    (void)gu_type; (void)d_type; (void)n_embd; (void)n_ff;
    if (out_gu_row) *out_gu_row = 0;
    if (out_d_row) *out_d_row = 0;
    if (out_up_off) *out_up_off = 0;
    if (out_down_off) *out_down_off = 0;
    if (out_bytes) *out_bytes = 0;
    if (out_tail_off) *out_tail_off = 0;
}
size_t tk_q36_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff) {
    return (size_t)cap_entries * (size_t)n_ff * 4;
}
tk_status tk_q36_expert_grouped(tk_ctx* ctx, int32_t gu_type, int32_t d_type, int64_t n_embd, int64_t n_ff, const uint64_t* grp_ptr, const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, int64_t grid_groups, tk_stream stream) {
    (void)ctx; (void)gu_type; (void)d_type; (void)n_embd; (void)n_ff; (void)grp_ptr; (void)grp_start; (void)n_groups; (void)ent_dst; (void)ent_tok; (void)cap_groups; (void)cap_entries; (void)x_q8_1; (void)scratch; (void)out; (void)grid_groups; (void)stream;
    return TK_OK;
}
int talus_pf_fa_prefill(const float* q, const void* k, const void* vt, const float* gate, float* out, int32_t T, int32_t H, int32_t HKV, int32_t D, int64_t kv_stride, int64_t vt_cells, int64_t p0, float scale, void* stream) {
    (void)q; (void)k; (void)vt; (void)gate; (void)out; (void)T; (void)H; (void)HKV; (void)D; (void)kv_stride; (void)vt_cells; (void)p0; (void)scale; (void)stream;
    return 0;
}
int talus_pf_i8x_expand(const void* w, int32_t ggml_type, int32_t n, int32_t k, int8_t* qs, float* ws, void* stream) {
    (void)w; (void)ggml_type; (void)n; (void)k; (void)qs; (void)ws; (void)stream;
    return 0;
}
int talus_pf_i8gemm(const int8_t* qs, const float* ws, int32_t n, int32_t k, int32_t mode, const void* xq, int32_t T, float* dst, int64_t ld_dst, void* stream) {
    (void)qs; (void)ws; (void)n; (void)k; (void)mode; (void)xq; (void)T; (void)dst; (void)ld_dst; (void)stream;
    return 0;
}
int talus_pf_moe_gemm(const void* w, int32_t ggml_type, size_t expert_bytes, int32_t n, int32_t k, int32_t n_experts, const void* xq, const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst, int64_t ld_dst, void* stream) {
    (void)w; (void)ggml_type; (void)expert_bytes; (void)n; (void)k; (void)n_experts; (void)xq; (void)bounds; (void)ids; (void)total_rows; (void)max_rows; (void)dst; (void)ld_dst; (void)stream;
    return 0;
}
int talus_pf_swiglu_q8_mmq(const float* g, const float* u, void* xq, int32_t ggml_type, int64_t cols, int64_t rows, void* stream) {
    (void)g; (void)u; (void)xq; (void)ggml_type; (void)cols; (void)rows; (void)stream;
    return 0;
}
int talus_pf_i8gemm_swiglu_q8(const int8_t* qs, const float* ws, int32_t n, int32_t k, int32_t mode, const void* xq, int32_t T, const float* g, int64_t ldg, void* yq, int32_t layout, void* stream) {
    (void)qs; (void)ws; (void)n; (void)k; (void)mode; (void)xq; (void)T; (void)g; (void)ldg; (void)yq; (void)layout; (void)stream;
    return -1;
}
int talus_pf_w4_pack(const float* x, int32_t T, int32_t k, int64_t ldx, const int8_t* signs, uint8_t* xq, float* xs, void* stream) {
    (void)x; (void)T; (void)k; (void)ldx; (void)signs; (void)xq; (void)xs; (void)stream;
    return 0;
}
int talus_pf_w4_gemm(const uint8_t* wq, const float* ws, int32_t n, int32_t k, const uint8_t* xq, const float* xs, int32_t T, float* dst, int64_t ld_dst, void* stream) {
    (void)wq; (void)ws; (void)n; (void)k; (void)xq; (void)xs; (void)T; (void)dst; (void)ld_dst; (void)stream;
    return 0;
}
int fnp_qsa_tile_attn_i8(const float* q, const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale, const int32_t* page_table, int64_t page_size, const int32_t* ids, const int32_t* steps, int64_t cap, float* attn, int64_t n_q, void* scratch, void* stream) {
    (void)q; (void)k_q; (void)v_q; (void)k_scale; (void)v_scale; (void)page_table; (void)page_size; (void)ids; (void)steps; (void)cap; (void)attn; (void)n_q; (void)scratch; (void)stream;
    return 0;
}
uint64_t fnp_qsa_tile_scratch_bytes_c(int64_t n_q, int64_t cap) {
    return (uint64_t)(n_q * cap * 4);
}
"#;

        fs::write(&c_file, c_source).expect("write C stub source");

        let status = Command::new("cc")
            .args([
                "-std=c11", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror", "-I",
            ])
            .arg(&include_dir)
            .arg(&c_file)
            .arg("-o")
            .arg(&so_file)
            .status()
            .expect("invoke cc to compile C stub");
        assert!(status.success(), "C stub compilation failed");

        let lib = LibTalusKernels::load(&so_file).expect("load C stub via dlopen");

        unsafe {
            let sizeof_mmvq: unsafe extern "C" fn() -> usize =
                load_sym(lib.handle, "tk_stub_sizeof_mmvq_item").unwrap();
            let sizeof_moe: unsafe extern "C" fn() -> usize =
                load_sym(lib.handle, "tk_stub_sizeof_moe_mmq_product").unwrap();
            let offset_mmvq_y: unsafe extern "C" fn() -> usize =
                load_sym(lib.handle, "tk_stub_offsetof_mmvq_y").unwrap();
            let offset_moe_dst: unsafe extern "C" fn() -> usize =
                load_sym(lib.handle, "tk_stub_offsetof_moe_dst").unwrap();

            assert_eq!(core::mem::size_of::<tk_mmvq_item>(), sizeof_mmvq());
            assert_eq!(core::mem::size_of::<tk_moe_mmq_product>(), sizeof_moe());
            assert_eq!(core::mem::offset_of!(tk_mmvq_item, y), offset_mmvq_y());
            assert_eq!(
                core::mem::offset_of!(tk_moe_mmq_product, dst),
                offset_moe_dst()
            );

            let mut cfg: *mut tk_config = core::ptr::null_mut();
            assert_eq!(
                (lib.tk_config_create_for_arch)(TK_ARCH_GFX1151, &mut cfg),
                TK_OK
            );
            assert!(!cfg.is_null());
            assert_eq!((lib.tk_config_from_env)(cfg), TK_OK);

            let key = CString::new("STRATA_GDN_HEAD").unwrap();
            assert_eq!((lib.tk_config_set_int)(cfg, key.as_ptr(), 1), TK_OK);
            let mut val: i64 = 0;
            assert_eq!((lib.tk_config_get_int)(cfg, key.as_ptr(), &mut val), TK_OK);
            assert_eq!(val, 1);

            let mut ctx: *mut tk_ctx = core::ptr::null_mut();
            assert_eq!((lib.tk_ctx_create)(0, cfg, &mut ctx), TK_OK);
            assert!(!ctx.is_null());

            let mut dev = -1;
            assert_eq!((lib.tk_ctx_device)(ctx, &mut dev), TK_OK);
            assert_eq!(dev, 0);

            let mut arch = TK_ARCH_OTHER;
            assert_eq!((lib.tk_ctx_arch)(ctx, &mut arch), TK_OK);
            assert_eq!(arch, TK_ARCH_GFX1151);

            assert_eq!((lib.tk_q8_1_bytes)(256, 2), 36 * 8 * 2);

            let status_str = CStr::from_ptr((lib.tk_status_string)(TK_OK))
                .to_str()
                .unwrap();
            assert_eq!(status_str, "TK_OK");

            (lib.tk_ctx_destroy)(ctx);
            (lib.tk_config_destroy)(cfg);
        }

        let _ = fs::remove_dir_all(&scratch_dir);
    }
}
