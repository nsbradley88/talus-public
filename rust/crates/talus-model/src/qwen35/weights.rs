//! Qwen3.5 / Qwen3.6 / Qwen3.8 model configuration, options, tensor descriptors,
//! host dequantization, and GPU weight loading (`q36_engine.cu` lines 1–384 & 446–565).

#![allow(non_snake_case, unused_unsafe)]

use std::ffi::c_void;
use std::ops::Deref;
use std::path::Path;
use std::ptr;

use super::ffi::*;
use talus_core_api::{
    GraphBuilder, GraphKey, HiddenTaps, KvCacheFormat, KvLayout, LayerKind, LogitRows,
    ModelAdapter, ModelDesc, QuantType, RecurrentLayout, SeqState, StateLayout, StreamHandle,
    WindowIo,
};

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_1: u32 = 3;
pub const GGML_TYPE_Q5_0: u32 = 6;
pub const GGML_TYPE_Q5_1: u32 = 7;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q8_1: u32 = 9;
pub const GGML_TYPE_Q2_K: u32 = 10;
pub const GGML_TYPE_Q3_K: u32 = 11;
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;
pub const GGML_TYPE_IQ2_XXS: u32 = 16;
pub const GGML_TYPE_IQ2_XS: u32 = 17;
pub const GGML_TYPE_IQ3_XXS: u32 = 18;
pub const GGML_TYPE_IQ1_S: u32 = 19;
pub const GGML_TYPE_IQ4_NL: u32 = 20;
pub const GGML_TYPE_IQ3_S: u32 = 21;
pub const GGML_TYPE_IQ2_S: u32 = 22;
pub const GGML_TYPE_IQ4_XS: u32 = 23;
pub const GGML_TYPE_IQ1_M: u32 = 29;
pub const GGML_TYPE_BF16: u32 = 30;
pub const GGML_TYPE_Q8_0_PACKED: u32 = 1000;

#[inline]
pub fn raw_ggml_qtype(qtype: u32) -> u32 {
    if qtype == GGML_TYPE_Q8_0_PACKED {
        GGML_TYPE_Q8_0
    } else {
        qtype
    }
}

#[inline]
pub fn raw_ggml_qtype_i32(qtype: i32) -> i32 {
    if qtype == GGML_TYPE_Q8_0_PACKED as i32 {
        GGML_TYPE_Q8_0 as i32
    } else {
        qtype
    }
}

#[inline]
pub fn to_tk_qtype(qtype: i32) -> tk_qtype {
    match raw_ggml_qtype_i32(qtype) as u32 {
        GGML_TYPE_F32 => TK_QTYPE_F32,
        GGML_TYPE_F16 => TK_QTYPE_F16,
        GGML_TYPE_Q4_0 => TK_QTYPE_Q4_0,
        GGML_TYPE_Q4_1 => TK_QTYPE_Q4_1,
        GGML_TYPE_Q5_0 => TK_QTYPE_Q5_0,
        GGML_TYPE_Q5_1 => TK_QTYPE_Q5_1,
        GGML_TYPE_Q8_0 => TK_QTYPE_Q8_0,
        GGML_TYPE_Q8_1 => TK_QTYPE_Q8_1,
        GGML_TYPE_Q2_K => TK_QTYPE_Q2_K,
        GGML_TYPE_Q3_K => TK_QTYPE_Q3_K,
        GGML_TYPE_Q4_K => TK_QTYPE_Q4_K,
        GGML_TYPE_Q5_K => TK_QTYPE_Q5_K,
        GGML_TYPE_Q6_K => TK_QTYPE_Q6_K,
        GGML_TYPE_IQ2_XXS => TK_QTYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS => TK_QTYPE_IQ2_XS,
        GGML_TYPE_IQ3_XXS => TK_QTYPE_IQ3_XXS,
        GGML_TYPE_IQ1_S => TK_QTYPE_IQ1_S,
        GGML_TYPE_IQ4_NL => TK_QTYPE_IQ4_NL,
        GGML_TYPE_IQ3_S => TK_QTYPE_IQ3_S,
        GGML_TYPE_IQ2_S => TK_QTYPE_IQ2_S,
        GGML_TYPE_IQ4_XS => TK_QTYPE_IQ4_XS,
        GGML_TYPE_IQ1_M => TK_QTYPE_IQ1_M,
        GGML_TYPE_BF16 => TK_QTYPE_BF16,
        _ => TK_QTYPE_F32,
    }
}

#[inline]
pub fn qtype_block_info(qtype: u32) -> Option<(usize, usize)> {
    let raw = raw_ggml_qtype(qtype);
    QuantType::from_ggml_type_id(raw).map(|qt| qt.block_geometry())
}

#[inline]
fn gcd_sz(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd_sz(b, a % b)
    }
}

#[inline]
fn lcm_sz(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        0
    } else {
        (a / gcd_sz(a, b)) * b
    }
}

#[inline]
pub fn expert_stride_bytes(gu_type: i32, d_type: i32, layout_bytes: usize) -> usize {
    let bb_gu = qtype_block_info(gu_type as u32)
        .map(|(_, b)| b)
        .unwrap_or(1)
        .max(1);
    let bb_d = qtype_block_info(d_type as u32)
        .map(|(_, b)| b)
        .unwrap_or(1)
        .max(1);
    let al = lcm_sz(lcm_sz(bb_gu, bb_d), 16).max(16);
    (layout_bytes + al - 1) / al * al
}

#[inline]
pub fn qtype_row_bytes(qtype: u32, ne0: i64) -> Option<usize> {
    if ne0 < 0 {
        return None;
    }
    let (blk_elems, blk_bytes) = qtype_block_info(qtype)?;
    let n = ne0 as usize;
    if blk_elems == 0 || !n.is_multiple_of(blk_elems) {
        return None;
    }
    (n / blk_elems).checked_mul(blk_bytes)
}

#[inline]
pub(crate) fn hip_check(code: hipError_t, context: &str) -> Result<()> {
    if code == hipSuccess {
        return Ok(());
    }
    let msg = unsafe {
        let p = hipGetErrorString(code);
        if p.is_null() {
            "unknown HIP error".to_string()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Err(anyhow!("{context}: HIP error {} ({msg})", code.0))
}

#[inline]
pub(crate) fn hip_malloc_bytes(bytes: usize) -> Result<*mut u8> {
    if bytes == 0 {
        return Ok(ptr::null_mut());
    }
    let mut ptr: *mut c_void = ptr::null_mut();
    hip_check(unsafe { hipMalloc(&mut ptr, bytes) }, "hipMalloc")?;
    Ok(ptr.cast::<u8>())
}

/// Complete architectural geometry matching `strata::ModelGeometry` in `model_desc.hpp`.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelGeometry {
    pub arch: String,
    pub n_vocab: i64,
    pub n_ctx_train: i64,
    pub n_layer: i64,
    pub n_embd: i64,
    pub n_ff: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub n_rot: i64,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
    pub ssm_conv_kernel: i64,
    pub ssm_state_size: i64,
    pub ssm_inner_size: i64,
    pub ssm_time_step_rank: i64,
    pub ssm_group_count: i64,
    pub full_attn_interval: i64,
    pub n_expert: i64,
    pub n_expert_used: i64,
    pub n_ff_exp: i64,
    pub n_ff_shexp: i64,
    pub layer_kinds: Vec<LayerKind>,
}

impl Default for ModelGeometry {
    fn default() -> Self {
        Self {
            arch: String::new(),
            n_vocab: 0,
            n_ctx_train: 0,
            n_layer: 0,
            n_embd: 0,
            n_ff: 0,
            n_head: 0,
            n_head_kv: 0,
            head_dim: 0,
            n_rot: 0,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-6,
            ssm_conv_kernel: 0,
            ssm_state_size: 0,
            ssm_inner_size: 0,
            ssm_time_step_rank: 0,
            ssm_group_count: 0,
            full_attn_interval: 4,
            n_expert: 0,
            n_expert_used: 0,
            n_ff_exp: 0,
            n_ff_shexp: 0,
            layer_kinds: Vec::new(),
        }
    }
}

impl ModelGeometry {
    #[inline]
    pub fn ssm_dt_rank(&self) -> i64 {
        self.ssm_time_step_rank
    }

    #[inline]
    pub fn ssm_k_heads(&self) -> i64 {
        self.ssm_group_count
    }

    #[inline]
    pub fn ssm_head_dim(&self) -> i64 {
        if self.ssm_time_step_rank > 0 {
            self.ssm_inner_size / self.ssm_time_step_rank
        } else {
            0
        }
    }

    #[inline]
    pub fn ssm_qkv_dim(&self) -> i64 {
        self.ssm_inner_size + 2 * self.ssm_group_count * self.ssm_state_size
    }

    #[inline]
    pub fn q_dim(&self) -> i64 {
        self.n_head * self.head_dim
    }

    #[inline]
    pub fn kv_dim(&self) -> i64 {
        self.n_head_kv * self.head_dim
    }

    #[inline]
    pub fn q_gate_dim(&self) -> i64 {
        2 * self.q_dim()
    }

    #[inline]
    pub fn is_moe(&self) -> bool {
        self.n_expert > 0
    }

    #[inline]
    pub fn is_full_attention_layer(&self, li: usize) -> bool {
        self.layer_kinds
            .get(li)
            .map(|&k| k == LayerKind::Attn)
            .unwrap_or_else(|| {
                let interval = self.full_attn_interval.max(1) as usize;
                ((li + 1) % interval) == 0
            })
    }

    #[inline]
    pub fn n_full_attn_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|&&k| k == LayerKind::Attn)
            .count()
    }

    #[inline]
    pub fn n_recurrent_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|&&k| k == LayerKind::Gdn)
            .count()
    }

    pub fn to_state_layout(&self) -> StateLayout {
        StateLayout {
            kv: KvLayout {
                num_layers: self.n_full_attn_layers() as u32,
                n_head_kv: self.n_head_kv.max(0) as u32,
                head_dim_k: self.head_dim.max(0) as u32,
                head_dim_v: self.head_dim.max(0) as u32,
                max_seq_len: 65536,
                ring_window: None,
                kv_pad_stride: 256,
                vt_tile_stride: Some(16),
                format: KvCacheFormat::F16,
            },
            recurrent: Some(RecurrentLayout {
                num_layers: self.n_recurrent_layers() as u32,
                state_size: self.ssm_state_size.max(0) as u32,
                h_k: self.ssm_group_count.max(0) as u32,
                h_v: self.ssm_time_step_rank.max(0) as u32,
                d_conv: self.ssm_conv_kernel.max(0) as u32,
                conv_channels: self.ssm_qkv_dim().max(0) as u32,
            }),
            indexer_bytes_per_pos: 0,
            ple_history_bytes: 0,
        }
    }

    pub fn to_model_desc(&self) -> ModelDesc {
        ModelDesc {
            arch: self.arch.clone(),
            n_embd: self.n_embd.max(0) as u32,
            n_layers: self.n_layer.max(0) as u32,
            layer_kinds: self.layer_kinds.clone(),
            qsa_interval: self.full_attn_interval.max(0) as u32,
            ssm_state_size: self.ssm_state_size.max(0) as u32,
            ssm_k_heads: self.ssm_group_count.max(0) as u32,
            ssm_v_heads: self.ssm_time_step_rank.max(0) as u32,
            ssm_d_conv: self.ssm_conv_kernel.max(0) as u32,
            ssm_conv_channels: self.ssm_qkv_dim().max(0) as u32,
            ssm_value_dim: self.ssm_inner_size.max(0) as u32,
            n_head: self.n_head.max(0) as u32,
            n_head_per_layer: Vec::new(),
            n_head_kv: self.n_head_kv.max(0) as u32,
            head_dim: self.head_dim.max(0) as u32,
            head_dim_v: self.head_dim.max(0) as u32,
            n_rot: self.n_rot.max(0) as u32,
            sliding_window: 0,
            rope_freq_base_swa: 0.0,
            n_rot_swa: 0,
            attn_gate_per_head: false,
            idx_q_heads: 0,
            idx_key_dim: 0,
            idx_top_k: 0,
            idx_block: 0,
            rms_eps: self.rms_norm_eps as f64,
            rope_freq_base: self.rope_freq_base as f64,
            rope_sections: [0; 4],
            rope_scaling_type: String::new(),
            rope_scaling_factor: 1.0,
            rope_scaling_orig_ctx: 0.0,
            hc: 0,
            hc_lr: 0,
            ffn_dense: !self.is_moe(),
            n_ff_dense: self.n_ff.max(0) as u32,
            n_dense_lead: 0,
            n_expert: self.n_expert.max(0) as u32,
            n_expert_used: self.n_expert_used.max(0) as u32,
            n_ff: if self.is_moe() {
                self.n_ff_exp.max(0) as u32
            } else {
                self.n_ff.max(0) as u32
            },
            n_ff_shexp: self.n_ff_shexp.max(0) as u32,
            expert_gating: 0,
            expert_weights_norm: true,
            expert_weights_scale: 1.0,
            n_vocab: self.n_vocab.max(0) as u32,
            n_ctx_train: self.n_ctx_train.max(0) as u32,
            ple_layers: Vec::new(),
            ple_conv_kernel: 0,
            image_token_id: None,
            n_mtp_layers: 0,
        }
    }
}

/// Qwen3.5 / Qwen3.6 configuration (`Q36Config`).
#[derive(Debug, Clone, PartialEq)]
pub struct Q36Config {
    pub n_Layer: i32,
    pub d_model: i32,
    pub d_ff: i32,
    pub n_q: i32,
    pub n_kv: i32,
    pub d_head: i32,
    pub d_rot: i32,
    pub n_vocab: i32,
    pub gdn_H: i32,
    pub gdn_Hk: i32,
    pub gdn_D: i32,
    pub gdn_S: i32,
    pub gdn_Q: i32,
    pub gdn_V: i32,
    pub gdn_C: i32,
    pub is_moe: bool,
    pub n_exp: i32,
    pub n_used: i32,
    pub d_exp: i32,
    pub d_shexp: i32,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub max_seq: i32,
}

impl Default for Q36Config {
    fn default() -> Self {
        Self::from_geometry(&reference_dense(), 65536)
    }
}

impl Q36Config {
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.n_Layer.max(0) as usize
    }

    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.d_model.max(0) as usize
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.n_vocab.max(0) as usize
    }

    #[inline]
    pub fn is_full_attention_layer(&self, li: usize) -> bool {
        ((li + 1) % 4) == 0
    }

    pub fn from_geometry(g: &ModelGeometry, max_seq: i32) -> Self {
        let gdn_H = g.ssm_time_step_rank as i32;
        let gdn_Hk = g.ssm_group_count as i32;
        let gdn_S = g.ssm_state_size as i32;
        let gdn_V = g.ssm_inner_size as i32;
        let gdn_D = if gdn_H > 0 { gdn_V / gdn_H } else { gdn_S };
        let gdn_Q = gdn_Hk * gdn_S;
        let gdn_C = gdn_V + 2 * gdn_Q;
        Self {
            n_Layer: g.n_layer as i32,
            d_model: g.n_embd as i32,
            d_ff: g.n_ff as i32,
            n_q: g.n_head as i32,
            n_kv: g.n_head_kv as i32,
            d_head: g.head_dim as i32,
            d_rot: g.n_rot as i32,
            n_vocab: g.n_vocab as i32,
            gdn_H,
            gdn_Hk,
            gdn_D,
            gdn_S,
            gdn_Q,
            gdn_V,
            gdn_C,
            is_moe: g.is_moe(),
            n_exp: g.n_expert as i32,
            n_used: g.n_expert_used as i32,
            d_exp: g.n_ff_exp as i32,
            d_shexp: g.n_ff_shexp as i32,
            rope_base: g.rope_freq_base,
            rms_eps: g.rms_norm_eps,
            max_seq: if max_seq > 0 { max_seq } else { 65536 },
        }
    }

    pub fn to_geometry(&self) -> ModelGeometry {
        let interval = 4i64;
        let mut layer_kinds = Vec::with_capacity(self.n_Layer.max(0) as usize);
        for l in 0..(self.n_Layer.max(0) as i64) {
            if ((l + 1) % interval) == 0 {
                layer_kinds.push(LayerKind::Attn);
            } else {
                layer_kinds.push(LayerKind::Gdn);
            }
        }
        ModelGeometry {
            arch: if self.is_moe {
                "qwen35moe".to_string()
            } else {
                "qwen35".to_string()
            },
            n_vocab: self.n_vocab as i64,
            n_ctx_train: 262144,
            n_layer: self.n_Layer as i64,
            n_embd: self.d_model as i64,
            n_ff: self.d_ff as i64,
            n_head: self.n_q as i64,
            n_head_kv: self.n_kv as i64,
            head_dim: self.d_head as i64,
            n_rot: self.d_rot as i64,
            rope_freq_base: self.rope_base,
            rms_norm_eps: self.rms_eps,
            ssm_conv_kernel: 4,
            ssm_state_size: self.gdn_S as i64,
            ssm_inner_size: self.gdn_V as i64,
            ssm_time_step_rank: self.gdn_H as i64,
            ssm_group_count: self.gdn_Hk as i64,
            full_attn_interval: interval,
            n_expert: self.n_exp as i64,
            n_expert_used: self.n_used as i64,
            n_ff_exp: self.d_exp as i64,
            n_ff_shexp: self.d_shexp as i64,
            layer_kinds,
        }
    }

    pub fn reference_dense() -> Self {
        Self::from_geometry(&reference_dense(), 65536)
    }

    pub fn reference_moe() -> Self {
        Self::from_geometry(&reference_moe(), 65536)
    }
}

impl From<&ModelGeometry> for Q36Config {
    fn from(g: &ModelGeometry) -> Self {
        Self::from_geometry(g, 65536)
    }
}

impl From<ModelGeometry> for Q36Config {
    fn from(g: ModelGeometry) -> Self {
        Self::from_geometry(&g, 65536)
    }
}

/// Runtime options & fusion/kernel knobs for `Q36Model` and `Q36Engine`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Q36Options {
    pub use_graph: bool,
    pub no_gdn_flush: bool,
    pub no_hipblas: bool,
    pub use_fattn: bool,
    pub use_bf16_kv: bool,
    pub use_w32: bool,
    pub use_fused_qkv: bool,
    pub use_fused_gate_up: bool,
    pub use_fused_PostAttn: bool,
    pub use_fused_fattn_q: bool,
    pub use_fused_shexp: bool,
    pub use_SplitK_down: bool,
    pub use_SplitK_out: bool,
    pub use_Direct_down: bool,
    pub use_Direct_out: bool,
    pub use_Wide_out: bool,
    pub use_Wide_down: bool,
    pub use_mtp_wide: bool,
    pub use_mtp_splitk: bool,
    pub use_mtp_direct: bool,
    pub use_mtp_fp16: bool,
    pub use_mtp_graph: bool,
    pub mtp_gguf: Option<std::path::PathBuf>,
    pub no_mtp: bool,
    pub mtp_main_head: bool,
    pub mtp_f16: bool,
    pub no_draft_vocab: bool,
    pub attn_w8: bool,
    pub no_v16: bool,
    pub dflash_path: String,
    pub max_seq: i32,
    pub vram_reserve_mb: usize,
}

impl Default for Q36Options {
    fn default() -> Self {
        Self {
            use_graph: true,
            no_gdn_flush: false,
            no_hipblas: false,
            use_fattn: true,
            use_bf16_kv: true,
            use_w32: true,
            use_fused_qkv: true,
            use_fused_gate_up: true,
            use_fused_PostAttn: true,
            use_fused_fattn_q: true,
            use_fused_shexp: true,
            use_SplitK_down: true,
            use_SplitK_out: true,
            use_Direct_down: true,
            use_Direct_out: true,
            use_Wide_out: true,
            use_Wide_down: true,
            use_mtp_wide: true,
            use_mtp_splitk: true,
            use_mtp_direct: true,
            use_mtp_fp16: true,
            use_mtp_graph: true,
            mtp_gguf: None,
            no_mtp: false,
            mtp_main_head: false,
            mtp_f16: false,
            no_draft_vocab: false,
            attn_w8: false,
            no_v16: false,
            dflash_path: String::new(),
            max_seq: 65536,
            vram_reserve_mb: 1536,
        }
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => v != "0" && !v.eq_ignore_ascii_case("false"),
        _ => default,
    }
}

impl Q36Options {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_env() -> Self {
        let mut o = Self::default();
        if std::env::var("STRATA_Q36_NO_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            o.use_graph = false;
        }
        if std::env::var("STRATA_Q36_NO_FLUSH")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            o.no_gdn_flush = true;
        }
        if std::env::var("STRATA_NO_HIPBLAS")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            o.no_hipblas = true;
        }
        o.use_fattn = env_bool("STRATA_FATTN", true);
        o.use_bf16_kv = env_bool("STRATA_BF16_KV", true);
        o.use_w32 = env_bool("STRATA_MMVQ_W32", true);
        o.use_fused_qkv = env_bool("STRATA_FUSED_QKV", true);
        o.use_fused_gate_up = env_bool("STRATA_FUSED_GATE_UP", true);
        o.use_fused_PostAttn = env_bool("STRATA_FUSED_POST_ATTN", true);
        o.use_fused_fattn_q = env_bool("STRATA_FUSED_FATTN_Q", true);
        o.use_fused_shexp = env_bool("STRATA_FUSED_SHEXP", true);
        o.use_SplitK_down = env_bool("STRATA_SPLITK_DOWN", true);
        o.use_SplitK_out = env_bool("STRATA_SPLITK_OUT", true);
        o.use_Direct_down = env_bool("STRATA_DIRECT_DOWN", true);
        o.use_Direct_out = env_bool("STRATA_DIRECT_OUT", true);
        o.use_Wide_out = env_bool("STRATA_WIDE_OUT", true);
        o.use_Wide_down = env_bool("STRATA_WIDE_DOWN", true);
        o.use_mtp_wide = env_bool("STRATA_MTP_WIDE", true);
        o.use_mtp_splitk = env_bool("STRATA_MTP_SPLITK", true);
        o.use_mtp_direct = env_bool("STRATA_MTP_DIRECT", true);
        o.use_mtp_fp16 = env_bool("STRATA_MTP_FP16", true);
        o.use_mtp_graph = env_bool("STRATA_MTP_GRAPH", true);
        if let Ok(ms) = std::env::var("STRATA_MAX_SEQ") {
            if let Ok(v) = ms.parse::<i32>() {
                if v > 0 {
                    o.max_seq = v;
                }
            }
        }
        if let Ok(vr) = std::env::var("STRATA_VRAM_RESERVE_MB") {
            if let Ok(v) = vr.parse::<usize>() {
                o.vram_reserve_mb = v;
            }
        }
        o
    }
}

/// Device tensor descriptor (`Tens`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tens {
    pub ptr: *const c_void,
    pub qtype: u32,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub bytes: usize,
    pub w_scale: f32,
}

unsafe impl Send for Tens {}
unsafe impl Sync for Tens {}

impl Default for Tens {
    fn default() -> Self {
        Self::null()
    }
}

impl Tens {
    #[inline]
    pub const fn null() -> Self {
        Self {
            ptr: ptr::null(),
            qtype: GGML_TYPE_F32,
            ne0: 0,
            ne1: 0,
            ne2: 1,
            bytes: 0,
            w_scale: 1.0,
        }
    }

    #[inline]
    pub fn valid(&self) -> bool {
        !self.ptr.is_null()
    }

    #[inline]
    pub fn new(
        ptr: *const c_void,
        qtype: u32,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        bytes: usize,
        w_scale: f32,
    ) -> Self {
        Self {
            ptr,
            qtype,
            ne0,
            ne1: ne1.max(1),
            ne2: ne2.max(1),
            bytes,
            w_scale,
        }
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }

    /// Number of rows (`ne1`).
    #[inline]
    pub fn rows(&self) -> i64 {
        self.ne1
    }

    /// Number of columns per row (`ne0`).
    #[inline]
    pub fn cols(&self) -> i64 {
        self.ne0
    }

    /// Number of 3D slices (`ne2`).
    #[inline]
    pub fn slices(&self) -> i64 {
        self.ne2
    }

    #[inline]
    pub fn n_in(&self) -> i32 {
        self.ne0 as i32
    }

    #[inline]
    pub fn n_out(&self) -> i32 {
        self.ne1 as i32
    }

    /// Compute the byte stride of a single row (`ne0` elements).
    #[inline]
    pub fn row_bytes(&self) -> usize {
        qtype_row_bytes(self.qtype, self.ne0).unwrap_or(0)
    }

    /// Compute total expected byte size (`row_bytes() * ne1 * ne2`).
    #[inline]
    pub fn expected_bytes(&self) -> Option<usize> {
        let rb = qtype_row_bytes(self.qtype, self.ne0)?;
        let r1 = usize::try_from(self.ne1.max(1)).ok()?;
        let r2 = usize::try_from(self.ne2.max(1)).ok()?;
        rb.checked_mul(r1)?.checked_mul(r2)
    }

    #[inline]
    pub fn as_f32(&self) -> *const f32 {
        self.ptr.cast::<f32>()
    }

    #[inline]
    pub fn as_u8(&self) -> *const u8 {
        self.ptr.cast::<u8>()
    }

    #[inline]
    pub fn as_mat(&self) -> Mat {
        Mat {
            d: self.ptr.cast::<u8>(),
            qtype: self.qtype as i32,
            n_in: self.ne0 as i32,
            n_out: self.ne1 as i32,
            w_scale: self.w_scale,
            ne2: self.ne2,
            bytes: self.bytes,
        }
    }
}

/// Matrix descriptor matching `Mat` in `q36_engine.cu:169`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat {
    pub d: *const u8,
    pub qtype: i32,
    pub n_in: i32,
    pub n_out: i32,
    pub w_scale: f32,
    pub ne2: i64,
    pub bytes: usize,
}

unsafe impl Send for Mat {}
unsafe impl Sync for Mat {}

impl Default for Mat {
    fn default() -> Self {
        Self::null()
    }
}

impl Mat {
    #[inline]
    pub const fn null() -> Self {
        Self {
            d: ptr::null(),
            qtype: 0,
            n_in: 0,
            n_out: 0,
            w_scale: 1.0,
            ne2: 1,
            bytes: 0,
        }
    }

    #[inline]
    pub fn is_null(&self) -> bool {
        self.d.is_null()
    }

    #[inline]
    pub fn valid(&self) -> bool {
        !self.d.is_null()
    }

    #[inline]
    pub fn cols(&self) -> i32 {
        self.n_in
    }

    #[inline]
    pub fn rows(&self) -> i32 {
        self.n_out
    }

    #[inline]
    pub fn raw_qtype(&self) -> i32 {
        raw_ggml_qtype_i32(self.qtype)
    }

    #[inline]
    pub fn tk_qtype(&self) -> tk_qtype {
        to_tk_qtype(self.qtype)
    }

    #[inline]
    pub fn row_bytes(&self) -> usize {
        qtype_row_bytes(self.qtype as u32, self.n_in as i64).unwrap_or(0)
    }

    #[inline]
    pub fn to_tens(&self) -> Tens {
        let bytes = if self.bytes > 0 {
            self.bytes
        } else {
            self.row_bytes() * (self.n_out.max(0) as usize) * (self.ne2.max(1) as usize)
        };
        Tens {
            ptr: self.d.cast::<c_void>(),
            qtype: self.qtype as u32,
            ne0: self.n_in as i64,
            ne1: self.n_out as i64,
            ne2: self.ne2.max(1),
            bytes,
            w_scale: self.w_scale,
        }
    }
}

impl From<Tens> for Mat {
    fn from(t: Tens) -> Self {
        t.as_mat()
    }
}

impl From<Mat> for Tens {
    fn from(m: Mat) -> Self {
        m.to_tens()
    }
}

/// Expert blob layout matching `K::NativeExpertLayout` in `q36_engine.cu`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertLayout {
    pub gu_row: usize,
    pub d_row: usize,
    pub up_off: usize,
    pub down_off: usize,
    pub bytes: usize,
    pub tail_off: usize,
}

impl ExpertLayout {
    pub fn compute(gu_type: i32, d_type: i32, n_embd: i64, n_ff: i64) -> Self {
        let mut gu_row = 0usize;
        let mut d_row = 0usize;
        let mut up_off = 0usize;
        let mut down_off = 0usize;
        let mut bytes = 0usize;
        let mut tail_off = 0usize;
        tk_q36_expert_layout(
            raw_ggml_qtype_i32(gu_type),
            raw_ggml_qtype_i32(d_type),
            n_embd,
            n_ff,
            &mut gu_row,
            &mut d_row,
            &mut up_off,
            &mut down_off,
            &mut bytes,
            &mut tail_off,
        );
        Self {
            gu_row,
            d_row,
            up_off,
            down_off,
            bytes,
            tail_off,
        }
    }
}

/// Per-layer weights (`LayerW` / `Layer`), holding both `Tens` descriptors and `q36_engine.cu` `Mat`/pointer fields.
#[derive(Debug, Clone, Default)]
pub struct LayerW {
    pub is_attn: bool,
    pub attn: bool,

    // Norm tensors
    pub attn_norm: Tens,
    pub post_norm: Tens,
    pub attn_norm_ptr: *const f32,
    pub post_norm_ptr: *const f32,

    // Full attention tensors
    pub qkv: Tens,
    pub wq: Tens,
    pub wk: Tens,
    pub wv: Tens,
    pub wo: Tens,
    pub attn_gate: Tens,
    pub q_norm: Tens,
    pub k_norm: Tens,
    pub q_norm_ptr: *const f32,
    pub k_norm_ptr: *const f32,

    // GDN recurrent tensors
    pub ssm_a: Tens,
    pub ssm_dt: Tens,
    pub ssm_conv: Tens,
    pub ssm_alpha: Tens,
    pub ssm_beta: Tens,
    pub ssm_norm: Tens,
    pub ssm_out: Tens,
    pub wqkv: Mat,
    pub wgate: Mat,
    pub ssm_out_mat: Mat,
    pub ssm_conv_ptr: *const f32,
    pub ssm_alpha_ptr: *const f32,
    pub ssm_beta_ptr: *const f32,
    pub ssm_a_ptr: *const f32,
    pub ssm_dt_ptr: *const f32,
    pub ssm_norm_ptr: *const f32,

    // Dense FFN tensors
    pub ffn_gate_up: Tens,
    pub ffn_gate: Tens,
    pub ffn_up: Tens,
    pub ffn_down: Tens,

    // MoE FFN tensors
    pub ffn_gate_inp: Tens,
    pub router_ext_tens: Tens,
    pub ffn_gate_exps: Tens,
    pub ffn_up_exps: Tens,
    pub ffn_down_exps: Tens,
    pub ffn_gate_up_shexp: Tens,
    pub ffn_gate_shexp: Tens,
    pub ffn_up_shexp: Tens,
    pub ffn_down_shexp: Tens,
    pub ffn_gate_inp_shexp: Tens,

    // C++ `Layer` views / MoE packed expert state
    pub wq_mat: Mat,
    pub wk_mat: Mat,
    pub wv_mat: Mat,
    pub wo_mat: Mat,
    pub ffn_gate_mat: Mat,
    pub ffn_up_mat: Mat,
    pub ffn_down_mat: Mat,
    pub router: *const f32,
    pub router_ext: *const f32,
    pub sh_inp: *const f32,
    pub sh_gate_up: Mat,
    pub sh_gate: Mat,
    pub sh_up: Mat,
    pub sh_down: Mat,
    pub L: ExpertLayout,
    pub stride: usize,
    pub gu_type: i32,
    pub d_type: i32,
    pub blobs: *mut u8,
    pub hblobs: Vec<u8>,
    pub n_slots: i32,
    pub src_g: *const u8,
    pub src_u: *const u8,
    pub src_d: *const u8,
    pub src_gsz: usize,
    pub src_dsz: usize,
}

unsafe impl Send for LayerW {}
unsafe impl Sync for LayerW {}

pub type Layer = LayerW;

impl LayerW {
    pub fn sync_views(&mut self) {
        self.attn = self.is_attn;
        self.attn_norm_ptr = self.attn_norm.as_f32();
        self.post_norm_ptr = self.post_norm.as_f32();
        self.q_norm_ptr = self.q_norm.as_f32();
        self.k_norm_ptr = self.k_norm.as_f32();
        self.wq_mat = self.wq.as_mat();
        self.wk_mat = self.wk.as_mat();
        self.wv_mat = self.wv.as_mat();
        self.wo_mat = self.wo.as_mat();
        if !self.qkv.is_null() && !self.is_attn {
            self.wqkv = self.qkv.as_mat();
        }
        if !self.attn_gate.is_null() {
            self.wgate = self.attn_gate.as_mat();
        }
        self.ssm_out_mat = self.ssm_out.as_mat();
        self.ssm_conv_ptr = self.ssm_conv.as_f32();
        self.ssm_alpha_ptr = self.ssm_alpha.as_f32();
        self.ssm_beta_ptr = self.ssm_beta.as_f32();
        self.ssm_a_ptr = self.ssm_a.as_f32();
        self.ssm_dt_ptr = self.ssm_dt.as_f32();
        self.ssm_norm_ptr = self.ssm_norm.as_f32();
        self.ffn_gate_mat = self.ffn_gate.as_mat();
        self.ffn_up_mat = self.ffn_up.as_mat();
        self.ffn_down_mat = self.ffn_down.as_mat();
        self.router = self.ffn_gate_inp.as_f32();
        self.router_ext = self.router_ext_tens.as_f32();
        self.sh_inp = self.ffn_gate_inp_shexp.as_f32();
        self.sh_gate_up = self.ffn_gate_up_shexp.as_mat();
        self.sh_gate = self.ffn_gate_shexp.as_mat();
        self.sh_up = self.ffn_up_shexp.as_mat();
        self.sh_down = self.ffn_down_shexp.as_mat();
    }
}

/// Multi-Token Prediction (MTP / `nextn`) weights (`MtpW`).
#[derive(Debug, Clone, Default)]
pub struct MtpW {
    pub has_mtp: bool,
    pub layer: LayerW,
    pub fc: Tens,
    pub eh_proj: Mat,
    pub enorm: Tens,
    pub hnorm: Tens,
    pub shared_head_norm: Tens,
    pub shared_head: Tens,
    pub mtp_enorm: *const f32,
    pub mtp_hnorm: *const f32,
    pub mtp_shnorm: *const f32,
    pub mtp_head: Mat,
}

unsafe impl Send for MtpW {}
unsafe impl Sync for MtpW {}

impl MtpW {
    pub fn sync_views(&mut self) {
        self.layer.sync_views();
        self.eh_proj = self.fc.as_mat();
        self.mtp_enorm = self.enorm.as_f32();
        self.mtp_hnorm = self.hnorm.as_f32();
        self.mtp_shnorm = self.shared_head_norm.as_f32();
        self.mtp_head = self.shared_head.as_mat();
    }
}

// ============================================================================
// Host Dequantization (`include/strata/artifact/dequant.hpp` + `tk_*_to_f32_row`)
// ============================================================================

#[inline]
pub fn read_u16_le(p: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([p[off], p[off + 1]])
}

#[inline]
pub fn read_u32_le(p: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]])
}

#[inline]
pub fn fp16_to_fp32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127u32 - 14u32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn bf16_to_fp32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
pub fn fp32_to_fp16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = x & 0x007F_FFFF;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - exp);
        let round = (m >> 12) & 1;
        let half_m = ((m >> 13) + round) as u16;
        sign | half_m
    } else if exp == 0xFF - 127 + 15 {
        if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7C00 | (((mant >> 13) as u16).max(1))
        }
    } else if exp > 30 {
        sign | 0x7C00
    } else {
        let round = ((mant >> 12) & 1) & (((mant >> 13) & 1) | ((mant & 0x0FFF != 0) as u32));
        let mut res = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
        res = res.wrapping_add(round as u16);
        res
    }
}

/// Host dequantize `Q8_0` blocks to `F16` (`u16` words), matching `tk_dequantize_q8_0_to_f16`.
pub fn dequantize_q8_0_to_f16_host(src: &[u8], n_elems: usize) -> Vec<u16> {
    let n_blocks = n_elems / 32;
    let mut out = vec![0u16; n_elems];
    for b in 0..n_blocks {
        let blk = &src[b * 34..(b + 1) * 34];
        let d = fp16_to_fp32(read_u16_le(blk, 0));
        let dst = &mut out[b * 32..(b + 1) * 32];
        for i in 0..32 {
            let q = blk[2 + i] as i8;
            dst[i] = fp32_to_fp16(d * (q as f32));
        }
    }
    out
}

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Convert any GGUF tensor row/tensor buffer into `Vec<f32>` on host (`upload_f32` in `q36_engine.cu:171-225`).
pub fn dequantize_to_f32(qtype: u32, raw: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; n];
    match raw_ggml_qtype(qtype) {
        GGML_TYPE_F32 => {
            ensure!(raw.len() >= n * 4, "F32 buffer too short");
            unsafe {
                ptr::copy_nonoverlapping(raw.as_ptr().cast::<f32>(), out.as_mut_ptr(), n);
            }
        }
        GGML_TYPE_F16 => {
            ensure!(raw.len() >= n * 2, "F16 buffer too short");
            let st = unsafe {
                tk_f16_to_f32_row(raw.as_ptr().cast::<u16>(), out.as_mut_ptr(), n as i64)
            };
            if st != TK_OK {
                for i in 0..n {
                    out[i] = fp16_to_fp32(read_u16_le(raw, 2 * i));
                }
            }
        }
        GGML_TYPE_BF16 => {
            ensure!(raw.len() >= n * 2, "BF16 buffer too short");
            let st = unsafe {
                tk_bf16_to_f32_row(raw.as_ptr().cast::<u16>(), out.as_mut_ptr(), n as i64)
            };
            if st != TK_OK {
                for i in 0..n {
                    out[i] = bf16_to_fp32(read_u16_le(raw, 2 * i));
                }
            }
        }
        GGML_TYPE_Q8_0 => {
            let nb = n / 32;
            ensure!(raw.len() >= nb * 34, "Q8_0 buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 34..(b + 1) * 34];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                for i in 0..32 {
                    out[b * 32 + i] = d * (blk[2 + i] as i8 as f32);
                }
            }
        }
        GGML_TYPE_Q4_0 => {
            let nb = n / 32;
            ensure!(raw.len() >= nb * 18, "Q4_0 buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 18..(b + 1) * 18];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                let qs = &blk[2..18];
                let y = &mut out[b * 32..(b + 1) * 32];
                for j in 0..16 {
                    y[j] = d * (((qs[j] & 0x0F) as i32 - 8) as f32);
                    y[j + 16] = d * (((qs[j] >> 4) as i32 - 8) as f32);
                }
            }
        }
        GGML_TYPE_Q5_0 => {
            let nb = n / 32;
            ensure!(raw.len() >= nb * 22, "Q5_0 buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 22..(b + 1) * 22];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                let qh = read_u32_le(blk, 2);
                let qs = &blk[6..22];
                let y = &mut out[b * 32..(b + 1) * 32];
                for j in 0..16 {
                    let xh0 = (((qh >> j) << 4) & 0x10) as u8;
                    let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = d * ((((qs[j] & 0x0F) | xh0) as i32 - 16) as f32);
                    y[j + 16] = d * ((((qs[j] >> 4) | xh1) as i32 - 16) as f32);
                }
            }
        }
        GGML_TYPE_Q5_1 => {
            let nb = n / 32;
            ensure!(raw.len() >= nb * 24, "Q5_1 buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 24..(b + 1) * 24];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                let m = fp16_to_fp32(read_u16_le(blk, 2));
                let qh = read_u32_le(blk, 4);
                let qs = &blk[8..24];
                let y = &mut out[b * 32..(b + 1) * 32];
                for j in 0..16 {
                    let xh0 = (((qh >> j) << 4) & 0x10) as u8;
                    let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
                    y[j] = d * (((qs[j] & 0x0F) | xh0) as f32) + m;
                    y[j + 16] = d * (((qs[j] >> 4) | xh1) as f32) + m;
                }
            }
        }
        GGML_TYPE_IQ4_NL => {
            let nb = n / 32;
            ensure!(raw.len() >= nb * 18, "IQ4_NL buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 18..(b + 1) * 18];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                let qs = &blk[2..18];
                let y = &mut out[b * 32..(b + 1) * 32];
                for j in 0..16 {
                    y[j] = d * (KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32);
                    y[j + 16] = d * (KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32);
                }
            }
        }
        GGML_TYPE_Q6_K => {
            let nb = n / 256;
            ensure!(raw.len() >= nb * 210, "Q6_K buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 210..(b + 1) * 210];
                let mut ql = &blk[0..128];
                let mut qh = &blk[128..192];
                let mut sc = &blk[192..208];
                let d = fp16_to_fp32(read_u16_le(blk, 208));
                let mut y = &mut out[b * 256..(b + 1) * 256];
                for _ in 0..2 {
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0F) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * (sc[is] as i8 as f32) * (q1 as f32);
                        y[l + 32] = d * (sc[is + 2] as i8 as f32) * (q2 as f32);
                        y[l + 64] = d * (sc[is + 4] as i8 as f32) * (q3 as f32);
                        y[l + 96] = d * (sc[is + 6] as i8 as f32) * (q4 as f32);
                    }
                    y = &mut y[128..];
                    ql = &ql[64..];
                    qh = &qh[32..];
                    sc = &sc[8..];
                }
            }
        }
        GGML_TYPE_Q4_K => {
            let nb = n / 256;
            ensure!(raw.len() >= nb * 144, "Q4_K buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 144..(b + 1) * 144];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                let mn = fp16_to_fp32(read_u16_le(blk, 2));
                let scales = &blk[4..16];
                let mut q = &blk[16..144];
                let y = &mut out[b * 256..(b + 1) * 256];
                let mut yi = 0usize;
                let mut is = 0usize;
                for _ in (0..256).step_by(64) {
                    let (sc1, m1_u) = get_scale_min_k4(is, scales);
                    let d1 = d * (sc1 as f32);
                    let m1 = mn * (m1_u as f32);
                    let (sc2, m2_u) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * (sc2 as f32);
                    let m2 = mn * (m2_u as f32);
                    for l in 0..32 {
                        y[yi] = d1 * ((q[l] & 0x0F) as f32) - m1;
                        yi += 1;
                    }
                    for l in 0..32 {
                        y[yi] = d2 * ((q[l] >> 4) as f32) - m2;
                        yi += 1;
                    }
                    q = &q[32..];
                    is += 2;
                }
            }
        }
        GGML_TYPE_Q5_K => {
            let nb = n / 256;
            ensure!(raw.len() >= nb * 176, "Q5_K buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 176..(b + 1) * 176];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                let mn = fp16_to_fp32(read_u16_le(blk, 2));
                let scales = &blk[4..16];
                let qh = &blk[16..48];
                let mut ql = &blk[48..176];
                let y = &mut out[b * 256..(b + 1) * 256];
                let mut yi = 0usize;
                let mut is = 0usize;
                let mut u1 = 1u8;
                let mut u2 = 2u8;
                for _ in (0..256).step_by(64) {
                    let (sc1, m1_u) = get_scale_min_k4(is, scales);
                    let d1 = d * (sc1 as f32);
                    let m1 = mn * (m1_u as f32);
                    let (sc2, m2_u) = get_scale_min_k4(is + 1, scales);
                    let d2 = d * (sc2 as f32);
                    let m2 = mn * (m2_u as f32);
                    for l in 0..32 {
                        let hbit = if (qh[l] & u1) != 0 { 16 } else { 0 };
                        y[yi] = d1 * (((ql[l] & 0x0F) + hbit) as f32) - m1;
                        yi += 1;
                    }
                    for l in 0..32 {
                        let hbit = if (qh[l] & u2) != 0 { 16 } else { 0 };
                        y[yi] = d2 * (((ql[l] >> 4) + hbit) as f32) - m2;
                        yi += 1;
                    }
                    ql = &ql[32..];
                    is += 2;
                    u1 = u1.wrapping_shl(2);
                    u2 = u2.wrapping_shl(2);
                }
            }
        }
        GGML_TYPE_Q3_K => {
            let nb = n / 256;
            ensure!(raw.len() >= nb * 110, "Q3_K buffer too short");
            const KMASK1: u32 = 0x0303_0303;
            const KMASK2: u32 = 0x0F0F_0F0F;
            for b in 0..nb {
                let blk = &raw[b * 110..(b + 1) * 110];
                let hm = &blk[0..32];
                let mut q = &blk[32..96];
                let d_all = fp16_to_fp32(read_u16_le(blk, 108));
                let mut aux = [0u32; 4];
                aux[0] = read_u32_le(blk, 96);
                aux[1] = read_u32_le(blk, 100);
                aux[2] = read_u32_le(blk, 104);
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
                aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
                let mut scales = [0i8; 16];
                for (i, w) in aux.iter().enumerate() {
                    let bytes = w.to_le_bytes();
                    for j in 0..4 {
                        scales[i * 4 + j] = bytes[j] as i8;
                    }
                }
                let y = &mut out[b * 256..(b + 1) * 256];
                let mut yi = 0usize;
                let mut is = 0usize;
                let mut m = 1u8;
                for _ in (0..256).step_by(128) {
                    let mut shift = 0u32;
                    for _ in 0..4 {
                        let dl = d_all * ((scales[is] as i32 - 32) as f32);
                        is += 1;
                        for l in 0..16 {
                            let v =
                                ((q[l] >> shift) & 3) as i32 - if (hm[l] & m) != 0 { 0 } else { 4 };
                            y[yi] = dl * (v as f32);
                            yi += 1;
                        }
                        let dl2 = d_all * ((scales[is] as i32 - 32) as f32);
                        is += 1;
                        for l in 0..16 {
                            let v = ((q[l + 16] >> shift) & 3) as i32
                                - if (hm[l + 16] & m) != 0 { 0 } else { 4 };
                            y[yi] = dl2 * (v as f32);
                            yi += 1;
                        }
                        shift += 2;
                        m = m.wrapping_shl(1);
                    }
                    q = &q[32..];
                }
            }
        }
        GGML_TYPE_IQ4_XS => {
            let nb = n / 256;
            ensure!(raw.len() >= nb * 136, "IQ4_XS buffer too short");
            for b in 0..nb {
                let blk = &raw[b * 136..(b + 1) * 136];
                let d = fp16_to_fp32(read_u16_le(blk, 0));
                let scales_h = read_u16_le(blk, 2);
                let scales_l = &blk[4..8];
                let mut qs = &blk[8..136];
                let y = &mut out[b * 256..(b + 1) * 256];
                for ib in 0..8 {
                    let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32)
                        | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                    let dl = d * ((ls - 32) as f32);
                    let yb = &mut y[ib * 32..(ib + 1) * 32];
                    for j in 0..16 {
                        yb[j] = dl * (KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32);
                        yb[j + 16] = dl * (KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32);
                    }
                    qs = &qs[16..];
                }
            }
        }
        other => bail!("unsupported tensor type in dequantize_to_f32: {other}"),
    }
    Ok(out)
}

// ============================================================================
// Architecture Metadata & Tensor Verification (`include/strata/core/model_desc.hpp`)
// ============================================================================

pub(crate) fn meta_i64(f: &GgufFile, key: &str) -> Result<i64> {
    match f.metadata().get(key) {
        Some(GgufValue::U8(v)) => Ok(*v as i64),
        Some(GgufValue::I8(v)) => Ok(*v as i64),
        Some(GgufValue::U16(v)) => Ok(*v as i64),
        Some(GgufValue::I16(v)) => Ok(*v as i64),
        Some(GgufValue::U32(v)) => Ok(*v as i64),
        Some(GgufValue::I32(v)) => Ok(*v as i64),
        Some(GgufValue::U64(v)) => Ok(*v as i64),
        Some(GgufValue::I64(v)) => Ok(*v),
        Some(other) => bail!("metadata key '{key}' is not an integer: {other:?}"),
        None => bail!("missing required GGUF metadata key '{key}'"),
    }
}

pub(crate) fn meta_i64_or(f: &GgufFile, key: &str, def: i64) -> i64 {
    meta_i64(f, key).unwrap_or(def)
}

pub(crate) fn meta_f32(f: &GgufFile, key: &str) -> Result<f32> {
    match f.metadata().get(key) {
        Some(GgufValue::F32(v)) => Ok(*v),
        Some(GgufValue::F64(v)) => Ok(*v as f32),
        Some(other) => bail!("metadata key '{key}' is not a float: {other:?}"),
        None => bail!("missing required GGUF metadata key '{key}'"),
    }
}

pub(crate) fn meta_f32_or(f: &GgufFile, key: &str, def: f32) -> f32 {
    meta_f32(f, key).unwrap_or(def)
}

fn detect_arch_prefix(f: &GgufFile) -> Result<String> {
    if let Some(arch) = f.arch() {
        let key = format!("{arch}.block_count");
        if f.metadata().contains_key(&key) {
            return Ok(arch.to_string());
        }
    }
    for cand in ["qwen35moe", "qwen35", "qwen3"] {
        let key = format!("{cand}.block_count");
        if f.metadata().contains_key(&key) {
            return Ok(cand.to_string());
        }
    }
    if let Some(arch) = f.arch() {
        return Ok(arch.to_string());
    }
    bail!("could not detect Qwen3.5/3.6 architecture prefix in GGUF metadata")
}

/// Extract `ModelGeometry` from a GGUF file (`qwen35_detail::describe` in `model_desc.hpp`).
pub fn describe_gguf(f: &GgufFile) -> Result<ModelGeometry> {
    let p = detect_arch_prefix(f)?;
    let arch = f.arch().unwrap_or(&p).to_string();
    let is_moe_arch = arch == "qwen35moe"
        || p == "qwen35moe"
        || meta_i64_or(f, &format!("{p}.expert_count"), 0) > 0;

    let block_count = meta_i64(f, &format!("{p}.block_count"))?;
    let n_mtp_layers = meta_i64_or(f, &format!("{p}.nextn_predict_layers"), 0);
    ensure!(
        n_mtp_layers >= 0 && n_mtp_layers < block_count,
        "qwen35: nextn_predict_layers ({n_mtp_layers}) is not smaller than block_count ({block_count})"
    );
    let mut g = ModelGeometry {
        arch,
        n_ctx_train: meta_i64_or(f, &format!("{p}.context_length"), 262144),
        n_layer: block_count - n_mtp_layers,
        n_embd: meta_i64(f, &format!("{p}.embedding_length"))?,
        n_ff: if is_moe_arch {
            meta_i64_or(f, &format!("{p}.feed_forward_length"), 0)
        } else {
            meta_i64(f, &format!("{p}.feed_forward_length"))?
        },
        n_head: meta_i64(f, &format!("{p}.attention.head_count"))?,
        n_head_kv: meta_i64(f, &format!("{p}.attention.head_count_kv"))?,
        head_dim: meta_i64_or(f, &format!("{p}.attention.key_length"), 256),
        n_rot: meta_i64_or(f, &format!("{p}.rope.dimension_count"), 64),
        rope_freq_base: meta_f32_or(f, &format!("{p}.rope.freq_base"), 10_000_000.0),
        rms_norm_eps: meta_f32_or(f, &format!("{p}.attention.layer_norm_rms_epsilon"), 1e-6),
        ssm_conv_kernel: meta_i64_or(f, &format!("{p}.ssm.conv_kernel"), 4),
        ssm_state_size: meta_i64_or(f, &format!("{p}.ssm.state_size"), 128),
        ssm_inner_size: meta_i64_or(f, &format!("{p}.ssm.inner_size"), 6144),
        ssm_time_step_rank: meta_i64_or(f, &format!("{p}.ssm.time_step_rank"), 48),
        ssm_group_count: meta_i64_or(f, &format!("{p}.ssm.group_count"), 16),
        full_attn_interval: meta_i64_or(f, &format!("{p}.full_attention_interval"), 4),
        n_vocab: 0,
        n_expert: 0,
        n_expert_used: 0,
        n_ff_exp: 0,
        n_ff_shexp: 0,
        layer_kinds: Vec::new(),
    };

    if is_moe_arch {
        g.n_expert = meta_i64(f, &format!("{p}.expert_count"))?;
        g.n_expert_used = meta_i64(f, &format!("{p}.expert_used_count"))?;
        g.n_ff_exp = meta_i64(f, &format!("{p}.expert_feed_forward_length"))?;
        g.n_ff_shexp = meta_i64_or(
            f,
            &format!("{p}.expert_shared_feed_forward_length"),
            g.n_ff_exp,
        );
    }

    let val_len = meta_i64_or(f, &format!("{p}.attention.value_length"), g.head_dim);
    ensure!(
        val_len == g.head_dim,
        "qwen35: key_length ({}) != value_length ({val_len})",
        g.head_dim
    );
    ensure!(
        g.ssm_time_step_rank > 0 && g.ssm_inner_size % g.ssm_time_step_rank == 0,
        "qwen35: ssm.inner_size not divisible by ssm.time_step_rank"
    );
    ensure!(
        g.ssm_head_dim() == g.ssm_state_size,
        "qwen35: ssm head_dim ({}) != ssm state_size ({})",
        g.ssm_head_dim(),
        g.ssm_state_size
    );
    ensure!(
        g.full_attn_interval > 0,
        "qwen35: full_attention_interval must be > 0"
    );

    if let Some(embd) = f.tensor("token_embd.weight") {
        g.n_vocab = embd.dimensions()[1] as i64;
    } else if let Some(GgufValue::Array(arr)) = f.metadata().get("tokenizer.ggml.tokens") {
        g.n_vocab = arr.len() as i64;
    } else {
        bail!(
            "qwen35: cannot determine vocab size (no token_embd.weight or tokenizer.ggml.tokens)"
        );
    }

    g.layer_kinds.reserve(g.n_layer as usize);
    for l in 0..g.n_layer {
        let full = ((l + 1) % g.full_attn_interval) == 0;
        g.layer_kinds.push(if full {
            LayerKind::Attn
        } else {
            LayerKind::Gdn
        });
    }
    Ok(g)
}

/// Reference geometry for Qwen3.5-27B (`qwen35::reference()` in `model_desc.hpp`).
pub fn reference_dense() -> ModelGeometry {
    let mut g = ModelGeometry {
        arch: "qwen35".to_string(),
        n_vocab: 248320,
        n_ctx_train: 262144,
        n_layer: 64,
        n_embd: 5120,
        n_ff: 17408,
        n_head: 24,
        n_head_kv: 4,
        head_dim: 256,
        n_rot: 64,
        rope_freq_base: 10_000_000.0,
        rms_norm_eps: 1e-6,
        ssm_conv_kernel: 4,
        ssm_state_size: 128,
        ssm_inner_size: 6144,
        ssm_time_step_rank: 48,
        ssm_group_count: 16,
        full_attn_interval: 4,
        n_expert: 0,
        n_expert_used: 0,
        n_ff_exp: 0,
        n_ff_shexp: 0,
        layer_kinds: Vec::with_capacity(64),
    };
    for l in 0..g.n_layer {
        g.layer_kinds.push(if ((l + 1) % 4) == 0 {
            LayerKind::Attn
        } else {
            LayerKind::Gdn
        });
    }
    g
}

/// Reference geometry for Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (`qwen35moe::reference()` in `model_desc.hpp`).
pub fn reference_moe() -> ModelGeometry {
    let mut g = ModelGeometry {
        arch: "qwen35moe".to_string(),
        n_vocab: 248320,
        n_ctx_train: 262144,
        n_layer: 40,
        n_embd: 2048,
        n_ff: 0,
        n_head: 16,
        n_head_kv: 2,
        head_dim: 256,
        n_rot: 64,
        rope_freq_base: 10_000_000.0,
        rms_norm_eps: 1e-6,
        ssm_conv_kernel: 4,
        ssm_state_size: 128,
        ssm_inner_size: 4096,
        ssm_time_step_rank: 32,
        ssm_group_count: 16,
        full_attn_interval: 4,
        n_expert: 256,
        n_expert_used: 8,
        n_ff_exp: 512,
        n_ff_shexp: 512,
        layer_kinds: Vec::with_capacity(40),
    };
    for l in 0..g.n_layer {
        g.layer_kinds.push(if ((l + 1) % 4) == 0 {
            LayerKind::Attn
        } else {
            LayerKind::Gdn
        });
    }
    g
}

// ============================================================================
// Tensor Shape & Scale Helpers (`q36_engine.cu:123-295`)
// ============================================================================

#[inline]
fn tensor_ne(t: &TensorInfo, dim: usize) -> i64 {
    t.dimensions().get(dim).copied().unwrap_or(1) as i64
}

#[inline]
fn tensor_w_scale(f: &GgufFile, name: &str) -> f32 {
    let sc_key = format!("{name}.scale");
    meta_f32_or(f, &sc_key, 1.0)
}

fn find_tensor<'a>(f: &'a GgufFile, name: &str) -> Result<&'a TensorInfo> {
    f.tensor(name)
        .ok_or_else(|| anyhow!("missing tensor: {name}"))
}

struct ArenaCursor {
    base: *mut u8,
    cap: usize,
    off: usize,
}

impl ArenaCursor {
    fn new(base: *mut u8, cap: usize) -> Self {
        Self { base, cap, off: 0 }
    }

    fn alloc(&mut self, bytes: usize, fallback_allocs: &mut Vec<*mut c_void>) -> Result<*mut u8> {
        let aligned_bytes = (bytes.max(1) + 255) & !255usize;
        if !self.base.is_null() && self.off + aligned_bytes <= self.cap {
            let p = unsafe { self.base.add(self.off) };
            self.off += aligned_bytes;
            return Ok(p);
        }
        let p = hip_malloc_bytes(aligned_bytes)?;
        fallback_allocs.push(p.cast::<c_void>());
        Ok(p)
    }
}

/// Pack one expert's `(gate, up, down)` slice on the host (`pack_expert_host` in `q36_engine.cu:278-294`).
pub fn pack_expert_host(
    dst: &mut [u8],
    g_src: &[u8],
    u_src: &[u8],
    d_src: &[u8],
    gsz: usize,
    dsz: usize,
    e: usize,
    n_ff: usize,
    n_embd: usize,
    L: &ExpertLayout,
) {
    let go = e * gsz;
    let doff = e * dsz;
    for r in 0..n_ff {
        let gr = r * L.gu_row;
        dst[gr..gr + L.gu_row].copy_from_slice(&g_src[go + gr..go + gr + L.gu_row]);
        dst[L.up_off + gr..L.up_off + gr + L.gu_row]
            .copy_from_slice(&u_src[go + gr..go + gr + L.gu_row]);
    }
    for r in 0..n_embd {
        let dr = r * L.d_row;
        dst[L.down_off + dr..L.down_off + dr + L.d_row]
            .copy_from_slice(&d_src[doff + dr..doff + dr + L.d_row]);
    }
    if L.tail_off < L.bytes {
        dst[L.tail_off..L.bytes].fill(0);
    }
}

/// Loaded Qwen3.5 / Qwen3.6 model weights on GPU (`Q36Model` / `Model`).
#[derive(Debug)]
pub struct Q36Model {
    pub cfg: Q36Config,
    pub g: ModelGeometry,
    pub opts: Q36Options,
    pub file_path: Option<std::path::PathBuf>,
    pub mtp_file_path: Option<std::path::PathBuf>,
    pub dense: bool,
    pub layers: Vec<LayerW>,
    pub has_mtp: bool,
    pub mtp: LayerW,
    pub mtp_w: MtpW,
    pub eh_proj: Mat,
    pub mtp_enorm: *const f32,
    pub mtp_hnorm: *const f32,
    pub mtp_shnorm: *const f32,
    pub mtp_head: Mat,
    pub token_embd: Tens,
    pub output_norm: Tens,
    pub output: Tens,
    pub out_norm: *const f32,
    pub head: Mat,
    pub draft_vocab: i32,
    pub dv_output: Tens,
    pub dv_head: Mat,
    pub dv_buf: *mut c_void,
    pub dv_ids: Vec<i32>,
    pub embd: *const u8,
    pub embd_type: i32,
    pub embd_row: usize,
    pub top_k: i32,
    pub hybrid: bool,
    pub gpu_arena: *mut u8,
    pub gpu_arena_bytes: usize,
    pub allocs: Vec<*mut c_void>,
    pub packed: Vec<(*const c_void, *mut c_void, usize)>,
    model_desc: ModelDesc,
}

unsafe impl Send for Q36Model {}
unsafe impl Sync for Q36Model {}

pub type Model = Q36Model;

impl Deref for Q36Model {
    type Target = Q36Config;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.cfg
    }
}

impl Drop for Q36Model {
    fn drop(&mut self) {
        for &(orig, pk, _) in &self.packed {
            mmvq_wide_release(orig); // no-op without a tile-plane copy
            tk_q36_q8_0_packed_unregister(orig);
            if !pk.is_null() && pk != orig as *mut c_void {
                unsafe {
                    let _ = hipFree(pk);
                }
            }
        }
        self.packed.clear();
        if !self.dv_buf.is_null() {
            unsafe {
                let _ = hipFree(self.dv_buf);
            }
            self.dv_buf = ptr::null_mut();
        }
        for &p in &self.allocs {
            if !p.is_null() {
                unsafe {
                    let _ = hipFree(p);
                }
            }
        }
        self.allocs.clear();
        if !self.gpu_arena.is_null() {
            unsafe {
                let _ = hipFree(self.gpu_arena.cast::<c_void>());
            }
            self.gpu_arena = ptr::null_mut();
            self.gpu_arena_bytes = 0;
        }
    }
}

impl Q36Model {
    /// Load Qwen3.5 / Qwen3.6 model weights from `gguf_path` and optional `mtp_gguf_path` sidecar.
    pub fn load(
        gguf_path: &Path,
        mtp_gguf_path: Option<&Path>,
        draft_vocab_ids: Option<&[i32]>,
        opts: &Q36Options,
    ) -> Result<Self> {
        let f = GgufFile::open(gguf_path)?;
        let effective_mtp_path = mtp_gguf_path.or(opts.mtp_gguf.as_deref());
        let side = match effective_mtp_path {
            Some(p) => Some(GgufFile::open(p)?),
            None => None,
        };
        let mut model = Self::load_from_gguf(&f, side.as_ref(), draft_vocab_ids, opts)?;
        model.file_path = Some(gguf_path.to_path_buf());
        model.mtp_file_path = effective_mtp_path.map(|p| p.to_path_buf());
        Ok(model)
    }

    /// Load Qwen3.5 / Qwen3.6 model weights from already-opened `GgufFile` handle(s).
    pub fn load_from_gguf(
        f: &GgufFile,
        mtp_f: Option<&GgufFile>,
        draft_vocab_ids: Option<&[i32]>,
        opts: &Q36Options,
    ) -> Result<Self> {
        let g = describe_gguf(f)?;
        let cfg = Q36Config::from_geometry(&g, opts.max_seq);
        let dense = !g.is_moe();
        let top_k = if dense {
            0
        } else {
            std::env::var("STRATA_MOE_TOPK")
                .ok()
                .and_then(|s| s.parse::<i32>().ok())
                .filter(|&k| k >= 1 && k <= g.n_expert_used as i32)
                .unwrap_or(g.n_expert_used as i32)
        };

        let q8_pack = std::env::var("STRATA_Q8_PACKED")
            .map(|v| v != "0")
            .unwrap_or(true);
        let q8_pack_head = std::env::var("STRATA_Q8_PACKED_HEAD")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Match `q36_engine.cu`: allocate individual device buffers per tensor via `allocs`
        // so we never issue an oversized monolithic arena `hipMalloc` or duplicate unused fused tensors.
        let gpu_arena: *mut u8 = ptr::null_mut();
        let gpu_arena_bytes = 0usize;
        let mut arena = ArenaCursor::new(gpu_arena, gpu_arena_bytes);
        let mut allocs: Vec<*mut c_void> = Vec::new();
        let mut packed: Vec<(*const c_void, *mut c_void, usize)> = Vec::new();

        let upload_f32_tens = |src_f: &GgufFile,
                               name: &str,
                               arena: &mut ArenaCursor,
                               allocs: &mut Vec<*mut c_void>|
         -> Result<Tens> {
            let t = find_tensor(src_f, name)?;
            let n = t.n_elements() as usize;
            let raw = src_f.tensor_data(t)?;
            let host = dequantize_to_f32(t.ggml_type_u32(), raw, n)?;
            let bytes = n * 4;
            let d = arena.alloc(bytes, allocs)?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        d.cast::<c_void>(),
                        host.as_ptr().cast::<c_void>(),
                        bytes,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(upload_f32)",
            )?;
            Ok(Tens::new(
                d.cast::<c_void>(),
                GGML_TYPE_F32,
                tensor_ne(t, 0),
                tensor_ne(t, 1),
                tensor_ne(t, 2),
                bytes,
                tensor_w_scale(src_f, name),
            ))
        };

        let upload_mat_tens = |src_f: &GgufFile,
                               name: &str,
                               fp16_convert: bool,
                               arena: &mut ArenaCursor,
                               allocs: &mut Vec<*mut c_void>,
                               packed: &mut Vec<(*const c_void, *mut c_void, usize)>|
         -> Result<Tens> {
            let t = find_tensor(src_f, name)?;
            let ne0 = tensor_ne(t, 0);
            let ne1 = tensor_ne(t, 1);
            let ne2 = tensor_ne(t, 2);
            let raw = src_f.tensor_data(t)?;
            let orig_qtype = t.ggml_type_u32();
            let w_scale = tensor_w_scale(src_f, name);

            // Optional MTP Q8_0 -> F16 host conversion (`use_mtp_fp16`, `ne0 * ne1 <= 100_000_000`)
            if fp16_convert
                && orig_qtype == GGML_TYPE_Q8_0
                && ne2 == 1
                && (ne0 * ne1) > 0
                && (ne0 * ne1) <= 100_000_000
            {
                let f16_host = dequantize_q8_0_to_f16_host(raw, (ne0 * ne1) as usize);
                let bytes = f16_host.len() * 2;
                let d = arena.alloc(bytes, allocs)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            f16_host.as_ptr().cast::<c_void>(),
                            bytes,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(upload_mat_f16)",
                )?;
                return Ok(Tens::new(
                    d.cast::<c_void>(),
                    GGML_TYPE_F16,
                    ne0,
                    ne1,
                    ne2,
                    bytes,
                    w_scale,
                ));
            }

            let nb = raw.len();
            let d = arena.alloc(nb, allocs)?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        d.cast::<c_void>(),
                        raw.as_ptr().cast::<c_void>(),
                        nb,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(upload_mat)",
            )?;
            // note the matrix as a tile-plane copy candidate for the wide verify tiles (nothing allocated here:
            // the engine packs the noted ones after placing its context and int8 prompt weights, if they fit).
            // Tracked in `packed` as (d, d, 0): Drop forgets it / frees its copy (pk == orig: nothing else is freed).
            if ne2 == 1
                && mmvq_wide_note(
                    orig_qtype as i32,
                    d.cast::<c_void>(),
                    ne0 as i32,
                    ne1 as i32,
                )
            {
                packed.push((d.cast::<c_void>(), d.cast::<c_void>(), 0));
            }

            let is_token_embd = name == "token_embd.weight";
            let is_output_head = name == "output.weight";
            let allow_pack = q8_pack
                && orig_qtype == GGML_TYPE_Q8_0
                && unsafe { tk_mmvq_w32_active() } == 0
                && !is_token_embd
                && (!is_output_head
                    || q8_pack_head
                    || tk_q36_q8_0_pack_should(orig_qtype as i32, ne0 as i32, ne1 as i32));
            let mut reported_qtype = orig_qtype;
            if allow_pack
                && ne2 == 1
                && tk_q36_q8_0_pack_should(orig_qtype as i32, ne0 as i32, ne1 as i32)
            {
                let mut pk = vec![0u8; nb];
                tk_q36_q8_0_pack_host(
                    raw.as_ptr().cast::<c_void>(),
                    pk.as_mut_ptr().cast::<c_void>(),
                    ne0 as i32,
                    ne1 as i32,
                );
                let pd = hip_malloc_bytes(nb)?.cast::<c_void>();
                hip_check(
                    unsafe {
                        hipMemcpy(
                            pd,
                            pk.as_ptr().cast::<c_void>(),
                            nb,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(q8_pack)",
                )?;
                tk_q36_q8_0_packed_register(d.cast::<c_void>(), pd, ne0 as i32, ne1 as i32);
                packed.push((d.cast::<c_void>(), pd, nb));
                if is_output_head && q8_pack_head {
                    reported_qtype = GGML_TYPE_Q8_0_PACKED;
                }
            }

            Ok(Tens::new(
                d.cast::<c_void>(),
                reported_qtype,
                ne0,
                ne1,
                ne2,
                nb,
                w_scale,
            ))
        };

        let fuse_mats = |mats: &[&Tens],
                         arena: &mut ArenaCursor,
                         allocs: &mut Vec<*mut c_void>,
                         packed: &mut Vec<(*const c_void, *mut c_void, usize)>|
         -> Result<Option<Tens>> {
            if mats.len() < 2 {
                return Ok(None);
            }
            let first = mats[0];
            if first.is_null() {
                return Ok(None);
            }
            let qtype = raw_ggml_qtype(first.qtype);
            let ne0 = first.ne0;
            let w_scale = first.w_scale;
            let mut total_ne1 = 0i64;
            let mut total_bytes = 0usize;
            for m in mats {
                if m.is_null()
                    || raw_ggml_qtype(m.qtype) != qtype
                    || m.ne0 != ne0
                    || m.ne2 != 1
                    || (m.w_scale - w_scale).abs() > 1e-7
                {
                    return Ok(None);
                }
                total_ne1 += m.ne1;
                total_bytes += m.bytes;
            }
            let dst = arena.alloc(total_bytes, allocs)?;
            let mut off = 0usize;
            for m in mats {
                hip_check(
                    unsafe {
                        hipMemcpy(
                            dst.add(off).cast::<c_void>(),
                            m.ptr,
                            m.bytes,
                            HIP_MEMCPY_DEVICE_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(fuse_mats)",
                )?;
                off += m.bytes;
            }
            if q8_pack
                && qtype == GGML_TYPE_Q8_0
                && unsafe { tk_mmvq_w32_active() } == 0
                && tk_q36_q8_0_pack_should(qtype as i32, ne0 as i32, total_ne1 as i32)
            {
                let mut host_raw = vec![0u8; total_bytes];
                hip_check(
                    unsafe {
                        hipMemcpy(
                            host_raw.as_mut_ptr().cast::<c_void>(),
                            dst.cast::<c_void>(),
                            total_bytes,
                            HIP_MEMCPY_DEVICE_TO_HOST,
                        )
                    },
                    "hipMemcpy(fuse_mats D2H)",
                )?;
                let mut pk = vec![0u8; total_bytes];
                tk_q36_q8_0_pack_host(
                    host_raw.as_ptr().cast::<c_void>(),
                    pk.as_mut_ptr().cast::<c_void>(),
                    ne0 as i32,
                    total_ne1 as i32,
                );
                let pd = hip_malloc_bytes(total_bytes)?.cast::<c_void>();
                hip_check(
                    unsafe {
                        hipMemcpy(
                            pd,
                            pk.as_ptr().cast::<c_void>(),
                            total_bytes,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(fuse_mats H2D packed)",
                )?;
                tk_q36_q8_0_packed_register(dst.cast::<c_void>(), pd, ne0 as i32, total_ne1 as i32);
                packed.push((dst.cast::<c_void>(), pd, total_bytes));
            }
            Ok(Some(Tens::new(
                dst.cast::<c_void>(),
                qtype,
                ne0,
                total_ne1,
                1,
                total_bytes,
                w_scale,
            )))
        };

        // Global tensors
        let token_embd = upload_mat_tens(
            f,
            "token_embd.weight",
            false,
            &mut arena,
            &mut allocs,
            &mut packed,
        )?;
        let embd_t = find_tensor(f, "token_embd.weight")?;
        let embd_type = embd_t.ggml_type_u32() as i32;
        let embd_row = (embd_t.n_bytes() as usize) / (g.n_vocab as usize);
        let embd = token_embd.as_u8();

        let output_norm = upload_f32_tens(f, "output_norm.weight", &mut arena, &mut allocs)?;
        let output = if f.tensor("output.weight").is_some() {
            upload_mat_tens(
                f,
                "output.weight",
                false,
                &mut arena,
                &mut allocs,
                &mut packed,
            )?
        } else {
            let mut tied = token_embd;
            if q8_pack
                && embd_type == (GGML_TYPE_Q8_0 as i32)
                && unsafe { tk_mmvq_w32_active() } == 0
                && tk_q36_q8_0_pack_should(embd_type, g.n_embd as i32, g.n_vocab as i32)
            {
                let raw = f.tensor_data(embd_t)?;
                let nb = raw.len();
                let mut pk = vec![0u8; nb];
                tk_q36_q8_0_pack_host(
                    raw.as_ptr().cast::<c_void>(),
                    pk.as_mut_ptr().cast::<c_void>(),
                    g.n_embd as i32,
                    g.n_vocab as i32,
                );
                let pd = hip_malloc_bytes(nb)?.cast::<c_void>();
                hip_check(
                    unsafe {
                        hipMemcpy(
                            pd,
                            pk.as_ptr().cast::<c_void>(),
                            nb,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(tied head pack)",
                )?;
                tk_q36_q8_0_packed_register(token_embd.ptr, pd, g.n_embd as i32, g.n_vocab as i32);
                packed.push((token_embd.ptr, pd, nb));
                if q8_pack_head {
                    tied.qtype = GGML_TYPE_Q8_0_PACKED;
                }
            }
            tied
        };

        let mut layers = vec![LayerW::default(); g.n_layer as usize];
        for li in 0..(g.n_layer as usize) {
            let L = &mut layers[li];
            let p = format!("blk.{li}.");
            L.is_attn = g.layer_kinds[li] == LayerKind::Attn;
            L.attn = L.is_attn;
            L.attn_norm =
                upload_f32_tens(f, &format!("{p}attn_norm.weight"), &mut arena, &mut allocs)?;
            L.post_norm = upload_f32_tens(
                f,
                &format!("{p}post_attention_norm.weight"),
                &mut arena,
                &mut allocs,
            )?;

            if L.is_attn {
                L.wq = upload_mat_tens(
                    f,
                    &format!("{p}attn_q.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.wk = upload_mat_tens(
                    f,
                    &format!("{p}attn_k.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.wv = upload_mat_tens(
                    f,
                    &format!("{p}attn_v.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.wo = upload_mat_tens(
                    f,
                    &format!("{p}attn_output.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.q_norm = upload_f32_tens(
                    f,
                    &format!("{p}attn_q_norm.weight"),
                    &mut arena,
                    &mut allocs,
                )?;
                L.k_norm = upload_f32_tens(
                    f,
                    &format!("{p}attn_k_norm.weight"),
                    &mut arena,
                    &mut allocs,
                )?;
            } else {
                L.qkv = upload_mat_tens(
                    f,
                    &format!("{p}attn_qkv.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.attn_gate = upload_mat_tens(
                    f,
                    &format!("{p}attn_gate.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.ssm_out = upload_mat_tens(
                    f,
                    &format!("{p}ssm_out.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.ssm_conv =
                    upload_f32_tens(f, &format!("{p}ssm_conv1d.weight"), &mut arena, &mut allocs)?;
                L.ssm_alpha =
                    upload_f32_tens(f, &format!("{p}ssm_alpha.weight"), &mut arena, &mut allocs)?;
                L.ssm_beta =
                    upload_f32_tens(f, &format!("{p}ssm_beta.weight"), &mut arena, &mut allocs)?;
                L.ssm_a = upload_f32_tens(f, &format!("{p}ssm_a"), &mut arena, &mut allocs)?;
                L.ssm_dt = upload_f32_tens(f, &format!("{p}ssm_dt.bias"), &mut arena, &mut allocs)?;
                L.ssm_norm =
                    upload_f32_tens(f, &format!("{p}ssm_norm.weight"), &mut arena, &mut allocs)?;
            }

            if dense {
                L.ffn_gate = upload_mat_tens(
                    f,
                    &format!("{p}ffn_gate.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.ffn_up = upload_mat_tens(
                    f,
                    &format!("{p}ffn_up.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.ffn_down = upload_mat_tens(
                    f,
                    &format!("{p}ffn_down.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
            } else {
                L.ffn_gate_inp = upload_f32_tens(
                    f,
                    &format!("{p}ffn_gate_inp.weight"),
                    &mut arena,
                    &mut allocs,
                )?;
                L.ffn_gate_inp_shexp = upload_f32_tens(
                    f,
                    &format!("{p}ffn_gate_inp_shexp.weight"),
                    &mut arena,
                    &mut allocs,
                )?;
                if let Some(rext) = fuse_mats(
                    &[&L.ffn_gate_inp, &L.ffn_gate_inp_shexp],
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )? {
                    L.router_ext_tens = rext;
                }
                L.ffn_gate_shexp = upload_mat_tens(
                    f,
                    &format!("{p}ffn_gate_shexp.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.ffn_up_shexp = upload_mat_tens(
                    f,
                    &format!("{p}ffn_up_shexp.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
                L.ffn_down_shexp = upload_mat_tens(
                    f,
                    &format!("{p}ffn_down_shexp.weight"),
                    false,
                    &mut arena,
                    &mut allocs,
                    &mut packed,
                )?;
            }
            L.sync_views();
        }

        // MoE 3D expert tensors (`q36_engine.cu:352-415`)
        let mut hybrid = false;
        if !dense {
            let mut free_b: usize = 0;
            let mut total_b: usize = 0;
            let _ = unsafe { hipMemGetInfo(&mut free_b, &mut total_b) };
            let reserve = opts.vram_reserve_mb * (1usize << 20);
            let budget = free_b.saturating_sub(reserve);

            let mut total_need = 0usize;
            for li in 0..(g.n_layer as usize) {
                let p = format!("blk.{li}.");
                let tg = find_tensor(f, &format!("{p}ffn_gate_exps.weight"))?;
                let tu = find_tensor(f, &format!("{p}ffn_up_exps.weight"))?;
                let td = find_tensor(f, &format!("{p}ffn_down_exps.weight"))?;
                ensure!(
                    tg.ggml_type_u32() == tu.ggml_type_u32(),
                    "gate/up expert type mismatch on layer {li}"
                );
                let L = &mut layers[li];
                L.gu_type = tg.ggml_type_u32() as i32;
                L.d_type = td.ggml_type_u32() as i32;
                ensure!(
                    tk_q36_expert_supported(L.gu_type, L.d_type, g.n_embd, g.n_ff_exp),
                    "unsupported expert quant types on layer {li}: gu={}, down={}",
                    L.gu_type,
                    L.d_type
                );
                L.L = ExpertLayout::compute(L.gu_type, L.d_type, g.n_embd, g.n_ff_exp);
                L.stride = expert_stride_bytes(L.gu_type, L.d_type, L.L.bytes);
                L.src_g = f.tensor_data(tg)?.as_ptr();
                L.src_u = f.tensor_data(tu)?.as_ptr();
                L.src_d = f.tensor_data(td)?.as_ptr();
                L.src_gsz = (tg.n_bytes() as usize) / (g.n_expert as usize);
                L.src_dsz = (td.n_bytes() as usize) / (g.n_expert as usize);
                total_need += (g.n_expert as usize) * L.stride + 4096;
            }

            hybrid = total_need > budget;
            let slots_per_layer = if hybrid {
                let per_slot_sum: usize = layers.iter().map(|l| l.stride).sum();
                let s = if per_slot_sum > 0 {
                    (budget / per_slot_sum) as i32
                } else {
                    0
                };
                s.clamp(0, g.n_expert as i32)
            } else {
                g.n_expert as i32
            };

            let mut host_blob = Vec::new();
            for li in 0..(g.n_layer as usize) {
                let p = format!("blk.{li}.");
                let tg = find_tensor(f, &format!("{p}blk_never"))
                    .err()
                    .and_then(|_| f.tensor(&format!("{p}ffn_gate_exps.weight")))
                    .unwrap();
                let tu = f.tensor(&format!("{p}ffn_up_exps.weight")).unwrap();
                let td = f.tensor(&format!("{p}ffn_down_exps.weight")).unwrap();
                let g_slice = f.tensor_data(tg)?;
                let u_slice = f.tensor_data(tu)?;
                let d_slice = f.tensor_data(td)?;

                let L = &mut layers[li];
                L.n_slots = slots_per_layer;
                let gpu_bytes = (L.n_slots as usize) * L.stride;
                let alloc_bytes = if gpu_bytes > 0 { gpu_bytes + 4096 } else { 0 };
                if alloc_bytes > 0 {
                    L.blobs = hip_malloc_bytes(alloc_bytes)?;
                    allocs.push(L.blobs.cast::<c_void>());
                }
                if hybrid {
                    L.hblobs.resize((g.n_expert as usize) * L.stride + 4096, 0);
                    L.hblobs.fill(0);
                } else {
                    host_blob.resize(alloc_bytes, 0);
                    host_blob.fill(0);
                }

                for e in 0..(g.n_expert as usize) {
                    let dst = if hybrid {
                        &mut L.hblobs[e * L.stride..(e + 1) * L.stride]
                    } else {
                        &mut host_blob[e * L.stride..(e + 1) * L.stride]
                    };
                    pack_expert_host(
                        dst,
                        g_slice,
                        u_slice,
                        d_slice,
                        L.src_gsz,
                        L.src_dsz,
                        e,
                        g.n_ff_exp as usize,
                        g.n_embd as usize,
                        &L.L,
                    );
                }
                if alloc_bytes > 0 {
                    let src_ptr = if hybrid {
                        L.hblobs.as_ptr()
                    } else {
                        host_blob.as_ptr()
                    };
                    hip_check(
                        unsafe {
                            hipMemcpy(
                                L.blobs.cast::<c_void>(),
                                src_ptr.cast::<c_void>(),
                                alloc_bytes,
                                HIP_MEMCPY_HOST_TO_DEVICE,
                            )
                        },
                        "hipMemcpy(expert blobs)",
                    )?;
                }
                L.ffn_gate_exps = Tens::new(
                    L.blobs.cast::<c_void>(),
                    L.gu_type as u32,
                    g.n_embd,
                    g.n_ff_exp,
                    g.n_expert,
                    gpu_bytes,
                    1.0,
                );
                L.ffn_up_exps = Tens::new(
                    unsafe { L.blobs.add(L.L.up_off).cast::<c_void>() },
                    L.gu_type as u32,
                    g.n_embd,
                    g.n_ff_exp,
                    g.n_expert,
                    gpu_bytes,
                    1.0,
                );
                L.ffn_down_exps = Tens::new(
                    unsafe { L.blobs.add(L.L.down_off).cast::<c_void>() },
                    L.d_type as u32,
                    g.n_ff_exp,
                    g.n_embd,
                    g.n_expert,
                    gpu_bytes,
                    1.0,
                );
                L.sync_views();
            }
        }

        let model_desc = g.to_model_desc();
        let mut model = Self {
            cfg,
            g,
            opts: opts.clone(),
            file_path: None,
            mtp_file_path: opts.mtp_gguf.clone(),
            dense,
            layers,
            has_mtp: false,
            mtp: LayerW::default(),
            mtp_w: MtpW::default(),
            eh_proj: Mat::null(),
            mtp_enorm: ptr::null(),
            mtp_hnorm: ptr::null(),
            mtp_shnorm: ptr::null(),
            mtp_head: Mat::null(),
            token_embd,
            output_norm,
            output,
            out_norm: output_norm.as_f32(),
            head: output.as_mat(),
            draft_vocab: 0,
            dv_output: Tens::null(),
            dv_head: Mat::null(),
            dv_buf: ptr::null_mut(),
            dv_ids: Vec::new(),
            embd,
            embd_type,
            embd_row,
            top_k,
            hybrid,
            gpu_arena,
            gpu_arena_bytes,
            allocs,
            packed,
            model_desc,
        };

        if !opts.no_mtp {
            if let Some(side) = mtp_f {
                let _ = model.try_load_mtp_from(side, true, opts.mtp_main_head, opts.mtp_f16)?;
            }
            if !model.has_mtp {
                let _ = model.try_load_mtp_from(f, false, opts.mtp_main_head, opts.mtp_f16)?;
            }
        }

        if let Some(ids) = draft_vocab_ids {
            if !ids.is_empty() {
                model.init_draft_vocab_head(ids)?;
            }
        } else if model.has_mtp
            && model.draft_vocab == 0
            && model.g.n_vocab > 16384
            && !opts.no_draft_vocab
        {
            model.init_draft_vocab_head_from_gguf(f, mtp_f)?;
        }

        Ok(model)
    }

    pub fn try_load_mtp_from(
        &mut self,
        src: &GgufFile,
        is_external_mtp: bool,
        mtp_main_head: bool,
        mtp_f16: bool,
    ) -> Result<bool> {
        let nl = self.g.n_layer;
        let p_nl = format!("blk.{nl}.");
        let p_0 = "blk.0.".to_string();
        let prefix = if src.tensor(&format!("{p_nl}nextn.eh_proj.weight")).is_some()
            || (src.tensor(&format!("{p_nl}attn_q.weight")).is_some()
                && (src.tensor("mtp.fc.weight").is_some()
                    || src.tensor("mtp.eh_proj.weight").is_some()))
        {
            p_nl
        } else if src.tensor("blk.0.nextn.eh_proj.weight").is_some()
            || (src.tensor("blk.0.attn_q.weight").is_some()
                && (src.tensor("mtp.fc.weight").is_some()
                    || src.tensor("mtp.eh_proj.weight").is_some()))
        {
            p_0
        } else if src.tensor("mtp.fc.weight").is_some() {
            "mtp.".to_string()
        } else {
            return Ok(false);
        };

        let mut arena = ArenaCursor::new(ptr::null_mut(), 0);
        let fp16_mtp = mtp_f16
            || (self.opts.use_mtp_fp16
                && std::env::var("STRATA_MTP_DEQUANT_F16")
                    .map(|v| v == "1")
                    .unwrap_or(false));

        let upload_f32 =
            |name: &str, arena: &mut ArenaCursor, allocs: &mut Vec<*mut c_void>| -> Result<Tens> {
                let t = find_tensor(src, name)?;
                let n = t.n_elements() as usize;
                let raw = src.tensor_data(t)?;
                let host = dequantize_to_f32(t.ggml_type_u32(), raw, n)?;
                let bytes = n * 4;
                let d = arena.alloc(bytes, allocs)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            host.as_ptr().cast::<c_void>(),
                            bytes,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(mtp f32)",
                )?;
                Ok(Tens::new(
                    d.cast::<c_void>(),
                    GGML_TYPE_F32,
                    tensor_ne(t, 0),
                    tensor_ne(t, 1),
                    tensor_ne(t, 2),
                    bytes,
                    tensor_w_scale(src, name),
                ))
            };

        let upload_mat_ex = |name: &str,
                             to_f16: bool,
                             arena: &mut ArenaCursor,
                             allocs: &mut Vec<*mut c_void>,
                             packed: &mut Vec<(*const c_void, *mut c_void, usize)>|
         -> Result<Tens> {
            let t = find_tensor(src, name)?;
            let ne0 = tensor_ne(t, 0);
            let ne1 = tensor_ne(t, 1);
            let ne2 = tensor_ne(t, 2);
            let raw = src.tensor_data(t)?;
            let orig_qtype = t.ggml_type_u32();
            let w_scale = tensor_w_scale(src, name);

            if to_f16
                && orig_qtype == GGML_TYPE_Q8_0
                && ne2 == 1
                && (ne0 * ne1) > 0
                && (ne0 * ne1) <= 100_000_000
            {
                let f16_host = dequantize_q8_0_to_f16_host(raw, (ne0 * ne1) as usize);
                let bytes = f16_host.len() * 2;
                let d = arena.alloc(bytes, allocs)?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            d.cast::<c_void>(),
                            f16_host.as_ptr().cast::<c_void>(),
                            bytes,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(mtp f16)",
                )?;
                return Ok(Tens::new(
                    d.cast::<c_void>(),
                    GGML_TYPE_F16,
                    ne0,
                    ne1,
                    ne2,
                    bytes,
                    w_scale,
                ));
            }

            let nb = raw.len();
            let d = arena.alloc(nb, allocs)?;
            hip_check(
                unsafe {
                    hipMemcpy(
                        d.cast::<c_void>(),
                        raw.as_ptr().cast::<c_void>(),
                        nb,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(mtp mat)",
            )?;
            let q8_pack = std::env::var("STRATA_Q8_PACKED")
                .map(|v| v != "0")
                .unwrap_or(true);
            if q8_pack
                && orig_qtype == GGML_TYPE_Q8_0
                && unsafe { tk_mmvq_w32_active() } == 0
                && ne2 == 1
                && tk_q36_q8_0_pack_should(orig_qtype as i32, ne0 as i32, ne1 as i32)
            {
                let mut pk = vec![0u8; nb];
                tk_q36_q8_0_pack_host(
                    raw.as_ptr().cast::<c_void>(),
                    pk.as_mut_ptr().cast::<c_void>(),
                    ne0 as i32,
                    ne1 as i32,
                );
                let pd = hip_malloc_bytes(nb)?.cast::<c_void>();
                hip_check(
                    unsafe {
                        hipMemcpy(
                            pd,
                            pk.as_ptr().cast::<c_void>(),
                            nb,
                            HIP_MEMCPY_HOST_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(mtp q8_pack)",
                )?;
                tk_q36_q8_0_packed_register(d.cast::<c_void>(), pd, ne0 as i32, ne1 as i32);
                packed.push((d.cast::<c_void>(), pd, nb));
            }
            Ok(Tens::new(
                d.cast::<c_void>(),
                orig_qtype,
                ne0,
                ne1,
                ne2,
                nb,
                w_scale,
            ))
        };

        let upload_mat =
            |name: &str,
             arena: &mut ArenaCursor,
             allocs: &mut Vec<*mut c_void>,
             packed: &mut Vec<(*const c_void, *mut c_void, usize)>|
             -> Result<Tens> { upload_mat_ex(name, fp16_mtp, arena, allocs, packed) };

        let fc_name = if src
            .tensor(&format!("{prefix}nextn.eh_proj.weight"))
            .is_some()
        {
            format!("{prefix}nextn.eh_proj.weight")
        } else if src.tensor("mtp.eh_proj.weight").is_some() {
            "mtp.eh_proj.weight".to_string()
        } else {
            "mtp.fc.weight".to_string()
        };
        let enorm_name = if src.tensor(&format!("{prefix}nextn.enorm.weight")).is_some() {
            format!("{prefix}nextn.enorm.weight")
        } else {
            "mtp.enorm.weight".to_string()
        };
        let hnorm_name = if src.tensor(&format!("{prefix}nextn.hnorm.weight")).is_some() {
            format!("{prefix}nextn.hnorm.weight")
        } else {
            "mtp.hnorm.weight".to_string()
        };
        let shnorm_name = if src
            .tensor(&format!("{prefix}nextn.shared_head_norm.weight"))
            .is_some()
        {
            format!("{prefix}nextn.shared_head_norm.weight")
        } else if src.tensor("mtp.shared_head_norm.weight").is_some() {
            "mtp.shared_head_norm.weight".to_string()
        } else {
            "mtp.shared_head.norm.weight".to_string()
        };
        let shhead_name = if !mtp_main_head {
            if is_external_mtp && src.tensor("output.weight").is_some() {
                Some("output.weight".to_string())
            } else if src
                .tensor(&format!("{prefix}nextn.shared_head_head.weight"))
                .is_some()
            {
                Some(format!("{prefix}nextn.shared_head_head.weight"))
            } else if src.tensor("mtp.shared_head.head.weight").is_some() {
                Some("mtp.shared_head.head.weight".to_string())
            } else {
                None
            }
        } else {
            None
        };

        let fc = upload_mat(&fc_name, &mut arena, &mut self.allocs, &mut self.packed)?;
        let enorm = upload_f32(&enorm_name, &mut arena, &mut self.allocs)?;
        let hnorm = upload_f32(&hnorm_name, &mut arena, &mut self.allocs)?;
        let shared_head_norm = upload_f32(&shnorm_name, &mut arena, &mut self.allocs)?;
        // `up_head("output.weight")` does NOT convert to f16 even if `mtp_f16` is true (`q36_engine.cu:3433-3435`).
        let shared_head = match shhead_name {
            Some(ref name) => {
                upload_mat_ex(name, false, &mut arena, &mut self.allocs, &mut self.packed)?
            }
            None => Tens::null(),
        };

        let mut L = LayerW::default();
        L.is_attn = true;
        L.attn = true;
        L.attn_norm = upload_f32(
            &format!("{prefix}attn_norm.weight"),
            &mut arena,
            &mut self.allocs,
        )?;
        L.post_norm = upload_f32(
            &format!("{prefix}post_attention_norm.weight"),
            &mut arena,
            &mut self.allocs,
        )?;
        L.wq = upload_mat(
            &format!("{prefix}attn_q.weight"),
            &mut arena,
            &mut self.allocs,
            &mut self.packed,
        )?;
        L.wk = upload_mat(
            &format!("{prefix}attn_k.weight"),
            &mut arena,
            &mut self.allocs,
            &mut self.packed,
        )?;
        L.wv = upload_mat(
            &format!("{prefix}attn_v.weight"),
            &mut arena,
            &mut self.allocs,
            &mut self.packed,
        )?;
        L.wo = upload_mat(
            &format!("{prefix}attn_output.weight"),
            &mut arena,
            &mut self.allocs,
            &mut self.packed,
        )?;
        L.q_norm = upload_f32(
            &format!("{prefix}attn_q_norm.weight"),
            &mut arena,
            &mut self.allocs,
        )?;
        L.k_norm = upload_f32(
            &format!("{prefix}attn_k_norm.weight"),
            &mut arena,
            &mut self.allocs,
        )?;

        if self.dense {
            L.ffn_gate = upload_mat(
                &format!("{prefix}ffn_gate.weight"),
                &mut arena,
                &mut self.allocs,
                &mut self.packed,
            )?;
            L.ffn_up = upload_mat(
                &format!("{prefix}ffn_up.weight"),
                &mut arena,
                &mut self.allocs,
                &mut self.packed,
            )?;
            L.ffn_down = upload_mat(
                &format!("{prefix}ffn_down.weight"),
                &mut arena,
                &mut self.allocs,
                &mut self.packed,
            )?;
        } else {
            L.ffn_gate_inp = upload_f32(
                &format!("{prefix}ffn_gate_inp.weight"),
                &mut arena,
                &mut self.allocs,
            )?;
            L.ffn_gate_inp_shexp = upload_f32(
                &format!("{prefix}ffn_gate_inp_shexp.weight"),
                &mut arena,
                &mut self.allocs,
            )?;
            {
                let rext_bytes = ((self.g.n_expert + 1) * self.g.n_embd) as usize * 4;
                let rext_d = arena.alloc(rext_bytes, &mut self.allocs)?;
                let r_bytes = (self.g.n_expert * self.g.n_embd) as usize * 4;
                let sh_bytes = (self.g.n_embd as usize) * 4;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            rext_d.cast::<c_void>(),
                            L.ffn_gate_inp.ptr,
                            r_bytes,
                            HIP_MEMCPY_DEVICE_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(mtp router_ext router)",
                )?;
                hip_check(
                    unsafe {
                        hipMemcpy(
                            rext_d.add(r_bytes).cast::<c_void>(),
                            L.ffn_gate_inp_shexp.ptr,
                            sh_bytes,
                            HIP_MEMCPY_DEVICE_TO_DEVICE,
                        )
                    },
                    "hipMemcpy(mtp router_ext sh_inp)",
                )?;
                L.router_ext_tens = Tens::new(
                    rext_d.cast::<c_void>(),
                    GGML_TYPE_F32,
                    self.g.n_embd,
                    self.g.n_expert + 1,
                    1,
                    rext_bytes,
                    1.0,
                );
            }
            L.ffn_gate_shexp = upload_mat(
                &format!("{prefix}ffn_gate_shexp.weight"),
                &mut arena,
                &mut self.allocs,
                &mut self.packed,
            )?;
            L.ffn_up_shexp = upload_mat(
                &format!("{prefix}ffn_up_shexp.weight"),
                &mut arena,
                &mut self.allocs,
                &mut self.packed,
            )?;
            L.ffn_down_shexp = upload_mat(
                &format!("{prefix}ffn_down_shexp.weight"),
                &mut arena,
                &mut self.allocs,
                &mut self.packed,
            )?;

            let tg = find_tensor(src, &format!("{prefix}ffn_gate_exps.weight"))?;
            let tu = find_tensor(src, &format!("{prefix}ffn_up_exps.weight"))?;
            let td = find_tensor(src, &format!("{prefix}ffn_down_exps.weight"))?;
            L.gu_type = tg.ggml_type_u32() as i32;
            L.d_type = td.ggml_type_u32() as i32;
            ensure!(
                tk_q36_expert_supported(L.gu_type, L.d_type, self.g.n_embd, self.g.n_ff_exp),
                "unsupported MTP expert quant types"
            );
            L.L = ExpertLayout::compute(L.gu_type, L.d_type, self.g.n_embd, self.g.n_ff_exp);
            L.stride = expert_stride_bytes(L.gu_type, L.d_type, L.L.bytes);
            L.src_g = src.tensor_data(tg)?.as_ptr();
            L.src_u = src.tensor_data(tu)?.as_ptr();
            L.src_d = src.tensor_data(td)?.as_ptr();
            L.src_gsz = (tg.n_bytes() as usize) / (self.g.n_expert as usize);
            L.src_dsz = (td.n_bytes() as usize) / (self.g.n_expert as usize);
            L.n_slots = self.g.n_expert as i32;
            let gpu_bytes = (L.n_slots as usize) * L.stride;
            let alloc_bytes = gpu_bytes + 4096;
            L.blobs = hip_malloc_bytes(alloc_bytes)?;
            self.allocs.push(L.blobs.cast::<c_void>());
            let mut host_blob = vec![0u8; alloc_bytes];
            let g_slice = src.tensor_data(tg)?;
            let u_slice = src.tensor_data(tu)?;
            let d_slice = src.tensor_data(td)?;
            for e in 0..(self.g.n_expert as usize) {
                pack_expert_host(
                    &mut host_blob[e * L.stride..(e + 1) * L.stride],
                    g_slice,
                    u_slice,
                    d_slice,
                    L.src_gsz,
                    L.src_dsz,
                    e,
                    self.g.n_ff_exp as usize,
                    self.g.n_embd as usize,
                    &L.L,
                );
            }
            hip_check(
                unsafe {
                    hipMemcpy(
                        L.blobs.cast::<c_void>(),
                        host_blob.as_ptr().cast::<c_void>(),
                        alloc_bytes,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(mtp expert blobs)",
            )?;
            L.ffn_gate_exps = Tens::new(
                L.blobs.cast::<c_void>(),
                L.gu_type as u32,
                self.g.n_embd,
                self.g.n_ff_exp,
                self.g.n_expert,
                gpu_bytes,
                1.0,
            );
            L.ffn_up_exps = Tens::new(
                unsafe { L.blobs.add(L.L.up_off).cast::<c_void>() },
                L.gu_type as u32,
                self.g.n_embd,
                self.g.n_ff_exp,
                self.g.n_expert,
                gpu_bytes,
                1.0,
            );
            L.ffn_down_exps = Tens::new(
                unsafe { L.blobs.add(L.L.down_off).cast::<c_void>() },
                L.d_type as u32,
                self.g.n_ff_exp,
                self.g.n_embd,
                self.g.n_expert,
                gpu_bytes,
                1.0,
            );
        }

        L.sync_views();
        self.has_mtp = true;
        self.mtp = L.clone();
        self.mtp_w = MtpW {
            has_mtp: true,
            layer: L,
            fc,
            eh_proj: fc.as_mat(),
            enorm,
            hnorm,
            shared_head_norm,
            shared_head,
            mtp_enorm: enorm.as_f32(),
            mtp_hnorm: hnorm.as_f32(),
            mtp_shnorm: shared_head_norm.as_f32(),
            mtp_head: shared_head.as_mat(),
        };
        self.eh_proj = self.mtp_w.eh_proj;
        self.mtp_enorm = self.mtp_w.mtp_enorm;
        self.mtp_hnorm = self.mtp_w.mtp_hnorm;
        self.mtp_shnorm = self.mtp_w.mtp_shnorm;
        self.mtp_head = self.mtp_w.mtp_head;
        Ok(true)
    }

    /// Load MTP sidecar GGUF (`load_mtp_sidecar` in `q36_engine.cu:446-546`).
    pub fn load_mtp_sidecar(&mut self, side: &GgufFile) -> Result<()> {
        if self.has_mtp || self.opts.no_mtp {
            return Ok(());
        }
        let _ = self.try_load_mtp_from(side, true, self.opts.mtp_main_head, self.opts.mtp_f16)?;
        Ok(())
    }

    /// Auto-discover `"draft_vocab.ids"` and `"output.weight"` in GGUF (`q36_engine.cu:3440-3497`).
    pub fn init_draft_vocab_head_from_gguf(
        &mut self,
        main_f: &GgufFile,
        mtp_f: Option<&GgufFile>,
    ) -> Result<()> {
        if self.draft_vocab != 0 || self.g.n_vocab <= 16384 {
            return Ok(());
        }
        let hf = if !self.mtp_head.is_null() {
            mtp_f.unwrap_or(main_f)
        } else {
            main_f
        };
        let ids_t = match hf.tensor("draft_vocab.ids") {
            Some(t) => t,
            None => return Ok(()),
        };
        // GGML_TYPE_I32 == 26 (or 1D tensor of 4-byte ints)
        let dims = ids_t.dimensions();
        if dims.len() != 1 {
            return Ok(());
        }
        let k = dims[0] as usize;
        let vocab = self.g.n_vocab as usize;
        if k < 1024 || k >= vocab {
            return Ok(());
        }
        if hf.tensor("output.weight").is_none() && self.output.is_null() {
            return Ok(());
        }
        let raw_ids = hf.tensor_data(ids_t)?;
        if raw_ids.len() < k * 4 {
            return Ok(());
        }
        let mut ids = Vec::with_capacity(k);
        for i in 0..k {
            ids.push(read_u32_le(raw_ids, i * 4) as i32);
        }
        self.init_draft_vocab_head(&ids)
    }

    /// Build sliced draft-vocabulary output head (`dv_head` / `dv_buf`) on GPU from `mtp_head` or `output` rows.
    pub fn init_draft_vocab_head(&mut self, ids: &[i32]) -> Result<()> {
        ensure!(!ids.is_empty(), "draft_vocab_ids must not be empty");
        let src_tens = if !self.mtp_head.is_null() && !self.mtp_w.shared_head.is_null() {
            self.mtp_w.shared_head
        } else {
            self.output
        };
        ensure!(!src_tens.is_null(), "output head tensor is null");

        if !self.dv_buf.is_null() {
            tk_q36_q8_0_packed_unregister(self.dv_buf);
            unsafe {
                let _ = hipFree(self.dv_buf);
            }
            self.dv_buf = ptr::null_mut();
        }

        let row_bytes = src_tens.row_bytes();
        ensure!(row_bytes > 0, "cannot compute row_bytes for output head");
        let n_dv = ids.len();
        let total_bytes = row_bytes * n_dv;
        let alloc_bytes = total_bytes + 4096;
        let dv_ptr = hip_malloc_bytes(alloc_bytes)?;
        unsafe {
            let _ = hipMemset(dv_ptr.add(total_bytes).cast::<c_void>(), 0, 4096);
        }
        self.dv_buf = dv_ptr.cast::<c_void>();
        self.dv_ids = ids.to_vec();
        self.draft_vocab = n_dv as i32;

        let src_base = src_tens.as_u8();
        for (dst_r, &tok_id) in ids.iter().enumerate() {
            ensure!(
                tok_id >= 0 && (tok_id as i64) < src_tens.ne1,
                "draft_vocab_id {tok_id} out of range 0..{}",
                src_tens.ne1
            );
            let src_row = unsafe { src_base.add((tok_id as usize) * row_bytes) };
            let dst_row = unsafe { dv_ptr.add(dst_r * row_bytes) };
            hip_check(
                unsafe {
                    hipMemcpy(
                        dst_row.cast::<c_void>(),
                        src_row.cast::<c_void>(),
                        row_bytes,
                        HIP_MEMCPY_DEVICE_TO_DEVICE,
                    )
                },
                "hipMemcpy(init_draft_vocab_head row)",
            )?;
        }

        let raw_qt = raw_ggml_qtype(src_tens.qtype);
        let q8_pack = std::env::var("STRATA_Q8_PACKED")
            .map(|v| v != "0")
            .unwrap_or(true);
        if (src_tens.qtype == GGML_TYPE_Q8_0_PACKED || q8_pack)
            && raw_qt == GGML_TYPE_Q8_0
            && unsafe { tk_mmvq_w32_active() } == 0
            && tk_q36_q8_0_pack_should(raw_qt as i32, src_tens.ne0 as i32, n_dv as i32)
        {
            let mut host_rows = vec![0u8; total_bytes];
            hip_check(
                unsafe {
                    hipMemcpy(
                        host_rows.as_mut_ptr().cast::<c_void>(),
                        self.dv_buf,
                        total_bytes,
                        HIP_MEMCPY_DEVICE_TO_HOST,
                    )
                },
                "hipMemcpy(dv_buf D2H)",
            )?;
            let mut pk = vec![0u8; total_bytes];
            tk_q36_q8_0_pack_host(
                host_rows.as_ptr().cast::<c_void>(),
                pk.as_mut_ptr().cast::<c_void>(),
                src_tens.ne0 as i32,
                n_dv as i32,
            );
            let pd = hip_malloc_bytes(total_bytes)?.cast::<c_void>();
            hip_check(
                unsafe {
                    hipMemcpy(
                        pd,
                        pk.as_ptr().cast::<c_void>(),
                        total_bytes,
                        HIP_MEMCPY_HOST_TO_DEVICE,
                    )
                },
                "hipMemcpy(dv_buf H2D packed)",
            )?;
            tk_q36_q8_0_packed_register(self.dv_buf, pd, src_tens.ne0 as i32, n_dv as i32);
            self.packed.push((self.dv_buf, pd, total_bytes));
        }

        let dv_qtype = if src_tens.qtype == GGML_TYPE_Q8_0_PACKED {
            GGML_TYPE_Q8_0_PACKED
        } else {
            raw_qt
        };
        self.dv_output = Tens::new(
            self.dv_buf,
            dv_qtype,
            src_tens.ne0,
            n_dv as i64,
            1,
            total_bytes,
            src_tens.w_scale,
        );
        self.dv_head = self.dv_output.as_mat();
        Ok(())
    }
}

impl ModelAdapter for Q36Model {
    fn desc(&self) -> &ModelDesc {
        &self.model_desc
    }

    fn state_layout(&self) -> StateLayout {
        let mut layout = self.g.to_state_layout();
        layout.kv.max_seq_len = self.cfg.max_seq.max(0) as u32;
        layout
    }

    fn prefill(
        &mut self,
        _st: &mut SeqState,
        _ids: &[u32],
        _pos0: u32,
        _want: LogitRows,
        _s: &StreamHandle,
    ) -> Result<()> {
        Ok(())
    }

    fn window(&mut self, _st: &mut SeqState, _w: &WindowIo, _s: &StreamHandle) -> Result<()> {
        Ok(())
    }

    fn commit(&mut self, _st: &mut SeqState, _n_keep: u32, _s: &StreamHandle) -> Result<()> {
        Ok(())
    }

    fn hidden_taps(&self) -> HiddenTaps {
        HiddenTaps::default()
    }

    fn capture(&mut self, _key: GraphKey, _g: &mut GraphBuilder) -> Result<()> {
        Ok(())
    }
}
