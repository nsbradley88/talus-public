//! MiMo-V2.6-Flash (`mimo2`) GGUF tensor catalog, DFlash drafter catalog, and draft vocab loader.
//!
//! Supports header-only inspection (`GgufFile::open` / `GgufFile::open_header` /
//! `GgufFile::from_header_bytes`) so multi-shard GGUF files and `.gguf.hdr` header
//! dumps can be validated without faulting multi-gigabyte weight payloads into host RAM.
//!
//! ## `mimo2` Architecture Geometry
//! - `block_count = 48`, `embedding_length = 4096`, `feed_forward_length = 16384`,
//!   `expert_feed_forward_length = 1280` (`UD-IQ2_XXS`) or `2048` (`MOPD-a4`),
//!   `expert_count = 256`, `expert_used_count = 8`, `vocab = 152064` (`UD`) or `152576` (`MOPD`),
//!   `norm_rms_eps = 1e-6`, `expert_weights_norm = true`.
//! - **Layer 0** is dense SwiGLU FFN (`blk.0.ffn_{gate,up,down}.weight` `[4096, 16384]`).
//! - **Layers 1..47** are 256-expert top-8 MoE (`blk.{l}.ffn_gate_inp.weight` `[4096, 256]`,
//!   `blk.{l}.ffn_{gate,up,down}_exps.weight` in `IQ2_XXS` / `IQ3_XXS` / `IQ2_XS`).
//! - **Hybrid SWA / Full Attention (6-layer cycle, 5:1 ratio):**
//!   - **40 SWA layers** (`i % 6 != 5`: `0..4, 6..10, 12..16, 18..22, 24..28, 30..34, 36..40, 42..46`):
//!     `n_head = 64`, `n_head_kv = 8` (GQA 8:1), `k_dim = 192`, `v_dim = 128`, `rope_dim = 64`,
//!     `rope_freq_base = 10000.0`, `sliding_window = 128`, learned `blk.{l}.attn_sinks.weight` (`F32[64]`),
//!     dispatches `tk_attn_swa_sinks`.
//!   - **8 Full-attention layers** (`i % 6 == 5`: `5, 11, 17, 23, 29, 35, 41, 47`):
//!     `n_head = 64`, `n_head_kv = 4` (GQA 16:1), `k_dim = 192`, `v_dim = 128`, `rope_dim = 64`,
//!     `rope_freq_base = 64000000.0` (64M), no `attn_sinks.weight`, dispatches `tk_attn_k192_v128`.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;

use crate::laguna::ffi::{posix_fadvise, POSIX_FADV_DONTNEED};
use crate::laguna::weights::to_tk_quant_type;
use crate::mimo::ffi::{
    copy_h2d_async, tk_qtype, HipRuntime, LibTalusKernels, TK_QTYPE_F32, TK_QTYPE_Q8_0,
};
use talus_core_api::{CoreError, QuantType, Result};
use talus_gguf::{compute_tensor_byte_size, GgufFile, GgufModel, GgufValue, TensorEntry};
use talus_mem::{Budget, BudgetCategory, Reservation};

/// Canonical GGUF architecture string for MiMo-V2.6-Flash.
pub const ARCH_NAME: &str = "mimo2";
/// Canonical GGUF architecture string for DFlash drafters.
pub const DFLASH_ARCH_NAME: &str = "dflash";

/// Number of transformer layers in `mimo2` (`48`).
pub const N_LAYERS: usize = 48;
/// Number of sliding-window attention (SWA) layers in the canonical 6-layer cycle (`40`).
pub const N_SWA_LAYERS: usize = 40;
/// Number of full-attention layers in the canonical 6-layer cycle (`8`).
pub const N_FULL_LAYERS: usize = 8;
/// Full-attention layer indices in the 6-layer cycle (`i % 6 == 5`).
pub const FULL_ATTN_LAYER_IDS: [usize; N_FULL_LAYERS] = [5, 11, 17, 23, 29, 35, 41, 47];

/// Hidden state dimension (`4096`).
pub const N_EMBD: usize = 4096;
/// Dense FFN intermediate width on Layer 0 (`16384`).
pub const N_FF_DENSE: usize = 16384;
/// Expert FFN intermediate width on `MiMo-V2.6-Flash-UD-IQ2_XXS` (`1280`).
pub const N_FF_EXPERT_UD: usize = 1280;
/// Expert FFN intermediate width on `MiMo-V2.6-Flash-MOPD-a4` (`2048`).
pub const N_FF_EXPERT_MOPD: usize = 2048;
/// Total number of routed experts per MoE layer (`256`).
pub const N_EXPERT: usize = 256;
/// Number of active experts selected per token (`8`).
pub const N_EXPERT_USED: usize = 8;
/// Canonical vocabulary size for `MiMo-V2.6-Flash-UD-IQ2_XXS` (`152064`).
pub const N_VOCAB_UD: usize = 152_064;
/// Vocabulary size for `MiMo-V2.6-Flash-MOPD-a4` (`152576`).
pub const N_VOCAB_MOPD: usize = 152_576;

/// Query head count across all layers (`64`).
pub const N_HEAD_Q: usize = 64;
/// Key/value head count on SWA layers (`8`, GQA 8:1).
pub const N_HEAD_KV_SWA: usize = 8;
/// Key/value head count on full-attention layers (`4`, GQA 16:1).
pub const N_HEAD_KV_FULL: usize = 4;
/// Per-head Key/Query dimension (`192`: `128` NoPE + `64` RoPE).
pub const HEAD_DIM_K: usize = 192;
/// Per-head Value dimension (`128`).
pub const HEAD_DIM_V: usize = 128;
/// Partial NeoX RoPE rotary dimension (`64`).
pub const ROPE_DIM: usize = 64;
/// RoPE frequency base on SWA layers (`10,000.0`).
pub const ROPE_FREQ_BASE_SWA: f32 = 10_000.0;
/// RoPE frequency base on full-attention layers (`64,000,000.0`).
pub const ROPE_FREQ_BASE_FULL: f32 = 64_000_000.0;
/// Sliding window span on SWA layers (`128` tokens).
pub const SWA_WINDOW: u32 = 128;
/// Default RMSNorm epsilon (`1e-6`).
pub const RMS_NORM_EPS: f32 = 1e-6;
/// Default softmax scale (`1 / sqrt(192)`).
pub const DEFAULT_ATTN_SCALE: f32 = 0.072_168_78; // 1.0 / 192.0_f64.sqrt()

/// Canonical tensor count for the 2-shard `MiMo-V2.6-Flash-UD-IQ2_XXS` GGUF (`821`).
pub const UD_EXPECTED_TENSOR_COUNT: usize = 821;
/// Canonical tensor count for the 5-layer `MiMo-V2-Flash-DFlash-Q8_0.gguf` drafter (`59`).
pub const DFLASH_5L_EXPECTED_TENSOR_COUNT: usize = 59;
/// Canonical 5-layer DFlash target layer tap indices (`[1, 12, 24, 36, 47]`).
pub const DFLASH_5L_TARGET_TAPS: [u32; 5] = [1, 12, 24, 36, 47];
/// Canonical 8-layer MOPD DFlash target layer tap indices (`[1, 7, 14, 20, 27, 33, 40, 46]`).
pub const DFLASH_8L_TARGET_TAPS: [u32; 8] = [1, 7, 14, 20, 27, 33, 40, 46];
/// Canonical DFlash block size (`16`).
pub const DFLASH_BLOCK_SIZE: u32 = 16;
/// Canonical DFlash speculation depth (`n4`, `--mtp 4`).
pub const DFLASH_DEFAULT_DRAFT_MAX: usize = 4;
/// Canonical draft vocabulary size (`65,536` token IDs).
pub const MIMO_DRAFT_VOCAB_SIZE: usize = 65_536;

/// Directory the `DEFAULT_MIMO_*` paths below are relative to: `$TALUS_MODELS_DIR` if set, else `models`
/// (relative to the working directory).  Explicit `--model` / `--dflash-gguf` / `--spec-draft-vocab` arguments
/// always win over these defaults.
pub fn models_dir() -> PathBuf {
    std::env::var_os("TALUS_MODELS_DIR")
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// `rel` (one of the `DEFAULT_MIMO_*` paths) under [`models_dir`].
pub fn default_model_path(rel: &str) -> PathBuf {
    models_dir().join(rel)
}

/// Default path (under [`models_dir`]) to shard 1 of `MiMo-V2.6-Flash-a4-x`
/// (rollback: `gguf/MiMo-V2.6-Flash-MOPD-a4/MiMo-V2.6-Flash-MOPD-a4-00001-of-00003.gguf`).
pub const DEFAULT_MIMO_A4X_SHARD1_PATH: &str =
    "gguf/MiMo-V2.6-Flash-a4-x/MiMo-V2.6-Flash-a4-x-00001-of-00003.gguf";
/// Default path (under [`models_dir`]) to shard 1 of `MiMo-V2.6-Flash-UD-IQ2_XXS`.
pub const DEFAULT_MIMO_UD_SHARD1_PATH: &str =
    "gguf/MiMo-V2.6-Flash/MiMo-V2.6-Flash-UD-IQ2_XXS-00001-of-00002.gguf";
/// Default path (under [`models_dir`]) to shard 2 of `MiMo-V2.6-Flash-UD-IQ2_XXS`.
pub const DEFAULT_MIMO_UD_SHARD2_PATH: &str =
    "gguf/MiMo-V2.6-Flash/MiMo-V2.6-Flash-UD-IQ2_XXS-00002-of-00002.gguf";
/// Default path (under [`models_dir`]) to the `MiMo-V2-Flash-DFlash-Q8_0.gguf` drafter.
pub const DEFAULT_MIMO_DFLASH_PATH: &str =
    "gguf/MiMo-V2.6-Flash/DFlash/MiMo-V2-Flash-DFlash-Q8_0.gguf";
/// Default path (under [`models_dir`]) to the MOPD DFlash drafter (the CLI's second choice).
pub const DEFAULT_MIMO_MOPD_DFLASH_PATH: &str =
    "gguf/MiMo-V2.6-Flash-MOPD-BPW2.5/dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf";
/// Default path (under [`models_dir`]) to the 64K draft vocabulary (`mimo-dv-65536.bin`).
pub const DEFAULT_MIMO_DRAFT_VOCAB_PATH: &str = "gguf/MiMo-V2.6-Flash/DFlash/mimo-dv-65536.bin";

/// Header dump filenames for `MiMo-V2.6-Flash-MOPD-a4` (3 shards, relative to `TALUS_SHAPES_DIR/hdr`).
pub const LOCAL_MIMO_MOPD_HDR_SHARDS: [&str; 3] = [
    "MiMo-V2.6-Flash-MOPD-a4-00001-of-00003.gguf.hdr",
    "MiMo-V2.6-Flash-MOPD-a4-00002-of-00003.gguf.hdr",
    "MiMo-V2.6-Flash-MOPD-a4-00003-of-00003.gguf.hdr",
];
/// Header dump filename for the DFlash drafter (relative to `TALUS_SHAPES_DIR/hdr`).
pub const LOCAL_MIMO_DFLASH_HDR_PATH: &str = "dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf.hdr";
/// Relative path to a local copy of the 64K draft vocabulary (`mimo-dv-65536.bin`).
pub const LOCAL_MIMO_DRAFT_VOCAB_PATH: &str = "data/mimo-dv-65536.bin";

/// Returns `true` if `layer_idx` is a sliding-window attention (SWA) layer in the
/// canonical 6-layer cycle (`layer_idx % 6 != 5`, i.e. 40 SWA layers and 8 full layers).
#[inline]
pub const fn is_canonical_swa_layer(layer_idx: usize) -> bool {
    layer_idx % 6 != 5
}

/// Returns `true` if `layer_idx` is a 256-expert MoE layer (`layer_idx >= 1`).
/// Layer 0 is the sole dense FFN layer in `mimo2`.
#[inline]
pub const fn is_moe_layer(layer_idx: usize) -> bool {
    layer_idx >= 1
}

/// Attention kind for a `mimo2` transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoAttentionKind {
    /// Sliding-window attention (`window = 128`, `n_head = 64`, `n_kv = 8`, `rope_freq_base = 10000.0`,
    /// learned `attn_sinks.weight`, dispatches `tk_attn_swa_sinks`).
    Swa128,
    /// Full causal attention (`n_head = 64`, `n_kv = 4`, `rope_freq_base = 64000000.0`,
    /// no sinks, dispatches `tk_attn_k192_v128`).
    FullK192V128,
}

/// FFN kind for a `mimo2` transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoFfnKind {
    /// Layer 0 dense SwiGLU (`[4096, 16384]`).
    Dense,
    /// Layers 1..47 256-expert top-8 MoE (`[4096, n_ff_exp, 256]`).
    Moe256Top8,
}

/// Parsed hyperparameters for a `mimo2` model.
#[derive(Debug, Clone, PartialEq)]
pub struct MimoHparams {
    /// Number of main transformer layers (`48`).
    pub n_layers: usize,
    /// Optional NextN/MTP layers present in the GGUF (`0` or `3`).
    pub n_nextn_layers: usize,
    /// Embedding dimension (`4096`).
    pub n_embd: usize,
    /// Dense FFN width on Layer 0 (`16384`).
    pub n_ff_dense: usize,
    /// Per-expert FFN width on Layers 1..47 (`1280` on UD, `2048` on MOPD).
    pub n_ff_exp: usize,
    /// Number of routed experts (`256`).
    pub n_expert: usize,
    /// Number of active experts per token (`8`).
    pub n_expert_used: usize,
    /// Vocabulary size (`152064` on UD, `152576` on MOPD).
    pub n_vocab: usize,
    /// Query head count (`64`).
    pub n_head_q: usize,
    /// KV head count on SWA layers (`8`).
    pub n_head_kv_swa: usize,
    /// KV head count on full-attention layers (`4`).
    pub n_head_kv_full: usize,
    /// Key/Query head dimension (`192`).
    pub head_dim_k: usize,
    /// Value head dimension (`128`).
    pub head_dim_v: usize,
    /// Rotary dimension (`64`).
    pub rope_dim: usize,
    /// RoPE frequency base on full-attention layers (`64,000,000.0` or `5,000,000.0`).
    pub rope_freq_base_full: f32,
    /// RoPE frequency base on SWA layers (`10,000.0`).
    pub rope_freq_base_swa: f32,
    /// Sliding window length (`128`).
    pub sliding_window: u32,
    /// RMSNorm epsilon (`1e-6`).
    pub rms_norm_eps: f32,
    /// Post-attention output scale (`mimo2.attention.value_scale`, `0.7070000171661377` or `1.0`).
    pub attn_value_scale: f32,
    /// Whether top-k expert weights are renormalized (`true`).
    pub expert_weights_norm: bool,
    /// Expert gating function (`1 = softmax`, `2 = sigmoid`).
    pub expert_gating_func: i32,
    /// Canonical 6-layer cycle SWA mask (`40` SWA layers `i % 6 != 5`, `8` full layers `i % 6 == 5`).
    pub canonical_swa_mask: Vec<bool>,
}

impl Default for MimoHparams {
    fn default() -> Self {
        let canonical_swa_mask = (0..N_LAYERS).map(is_canonical_swa_layer).collect();
        Self {
            n_layers: N_LAYERS,
            n_nextn_layers: 0,
            n_embd: N_EMBD,
            n_ff_dense: N_FF_DENSE,
            n_ff_exp: N_FF_EXPERT_UD,
            n_expert: N_EXPERT,
            n_expert_used: N_EXPERT_USED,
            n_vocab: N_VOCAB_UD,
            n_head_q: N_HEAD_Q,
            n_head_kv_swa: N_HEAD_KV_SWA,
            n_head_kv_full: N_HEAD_KV_FULL,
            head_dim_k: HEAD_DIM_K,
            head_dim_v: HEAD_DIM_V,
            rope_dim: ROPE_DIM,
            rope_freq_base_full: ROPE_FREQ_BASE_FULL,
            rope_freq_base_swa: ROPE_FREQ_BASE_SWA,
            sliding_window: SWA_WINDOW,
            rms_norm_eps: RMS_NORM_EPS,
            attn_value_scale: 1.0,
            expert_weights_norm: true,
            expert_gating_func: 2, // sigmoid / normalized top-k
            canonical_swa_mask,
        }
    }
}

impl MimoHparams {
    /// Compact synthetic geometry preserving the exact 6-layer SWA/Full cycle (`i % 6 != 5`),
    /// Layer 0 dense FFN vs Layers 1..N MoE, K192/V128 head geometry, and 256-expert top-8
    /// routing when scaled down for fast unit tests.
    pub fn synthetic_tiny() -> Self {
        let n_layers = 6; // 5 SWA (0..4) + 1 Full (5), Layer 0 dense + Layers 1..5 MoE
        let canonical_swa_mask = (0..n_layers).map(is_canonical_swa_layer).collect();
        Self {
            n_layers,
            n_nextn_layers: 0,
            n_embd: 256,
            n_ff_dense: 512,
            n_ff_exp: 256,
            n_expert: 16,
            n_expert_used: 8,
            n_vocab: 512,
            n_head_q: 8,
            n_head_kv_swa: 2,
            n_head_kv_full: 1,
            head_dim_k: 192,
            head_dim_v: 128,
            rope_dim: 64,
            rope_freq_base_full: ROPE_FREQ_BASE_FULL,
            rope_freq_base_swa: ROPE_FREQ_BASE_SWA,
            sliding_window: SWA_WINDOW,
            rms_norm_eps: RMS_NORM_EPS,
            attn_value_scale: 1.0,
            expert_weights_norm: true,
            expert_gating_func: 2,
            canonical_swa_mask,
        }
    }

    /// Number of SWA layers in the canonical 6-layer cycle (`40` for `n_layers = 48`).
    pub fn canonical_swa_count(&self) -> usize {
        self.canonical_swa_mask.iter().filter(|&&b| b).count()
    }

    /// Number of full-attention layers in the canonical 6-layer cycle (`8` for `n_layers = 48`).
    pub fn canonical_full_count(&self) -> usize {
        self.canonical_swa_mask.iter().filter(|&&b| !b).count()
    }
}

/// Location and descriptor of a single tensor inside a multi-shard GGUF catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoTensorLoc {
    /// Index of the shard holding this tensor (`0..n_shards`).
    pub shard_idx: usize,
    /// Tensor metadata entry.
    pub entry: TensorEntry,
}

/// Attention Q/K/V weight storage layout for a single `mimo2` layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MimoQkvCatalog {
    /// Split `attn_q.weight`, `attn_k.weight`, `attn_v.weight` (as in `MiMo-V2.6-Flash-UD-IQ2_XXS`).
    Split {
        /// `blk.{l}.attn_q.weight` (`[n_embd, n_head_q * head_dim_k]`).
        attn_q: MimoTensorLoc,
        /// `blk.{l}.attn_k.weight` (`[n_embd, n_head_kv * head_dim_k]`).
        attn_k: MimoTensorLoc,
        /// `blk.{l}.attn_v.weight` (`[n_embd, n_head_kv * head_dim_v]`).
        attn_v: MimoTensorLoc,
        /// Optional `blk.{l}.attn_q_norm.weight` (`F32[head_dim_k]`).
        attn_q_norm: Option<MimoTensorLoc>,
        /// Optional `blk.{l}.attn_k_norm.weight` (`F32[head_dim_k]`).
        attn_k_norm: Option<MimoTensorLoc>,
    },
    /// Fused `blk.{l}.attn_qkv.weight` (as in `MiMo-V2.6-Flash-MOPD-a4`).
    Fused {
        /// `blk.{l}.attn_qkv.weight` (`[n_embd, n_head_q * head_dim_k + n_head_kv * (head_dim_k + head_dim_v)]`).
        attn_qkv: MimoTensorLoc,
        /// Optional `blk.{l}.attn_q_norm.weight` (`F32[head_dim_k]`).
        attn_q_norm: Option<MimoTensorLoc>,
        /// Optional `blk.{l}.attn_k_norm.weight` (`F32[head_dim_k]`).
        attn_k_norm: Option<MimoTensorLoc>,
    },
}

/// FFN weight catalog entry for a single `mimo2` layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MimoFfnCatalog {
    /// Layer 0 dense SwiGLU FFN (`blk.0.ffn_{gate,up,down}.weight`).
    Dense {
        /// `blk.0.ffn_gate.weight` (`[4096, 16384]`).
        ffn_gate: MimoTensorLoc,
        /// `blk.0.ffn_up.weight` (`[4096, 16384]`).
        ffn_up: MimoTensorLoc,
        /// `blk.0.ffn_down.weight` (`[16384, 4096]`).
        ffn_down: MimoTensorLoc,
    },
    /// Layers 1..47 256-expert top-8 MoE FFN.
    Moe {
        /// `blk.{l}.ffn_gate_inp.weight` (`F32[4096, 256]`).
        ffn_gate_inp: MimoTensorLoc,
        /// Optional `blk.{l}.exp_probs_b.bias` (`F32[256]`).
        exp_probs_b: Option<MimoTensorLoc>,
        /// `blk.{l}.ffn_gate_exps.weight` (`[4096, n_ff_exp, 256]`).
        ffn_gate_exps: MimoTensorLoc,
        /// `blk.{l}.ffn_up_exps.weight` (`[4096, n_ff_exp, 256]`).
        ffn_up_exps: MimoTensorLoc,
        /// `blk.{l}.ffn_down_exps.weight` (`[n_ff_exp, 4096, 256]`).
        ffn_down_exps: MimoTensorLoc,
    },
}

/// Catalog entry for one `mimo2` layer (`0..48`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoLayerCatalog {
    /// Layer index (`0..48`).
    pub layer_idx: usize,
    /// Attention kind (`Swa128` vs `FullK192V128`).
    pub attn_kind: MimoAttentionKind,
    /// FFN kind (`Dense` on Layer 0, `Moe256Top8` on Layers 1..47).
    pub ffn_kind: MimoFfnKind,
    /// Number of query heads (`64`).
    pub n_head_q: usize,
    /// Number of KV heads (`8` on SWA layers, `4` on full-attention layers).
    pub n_head_kv: usize,
    /// Sliding window size (`128` on SWA layers, `0` on full-attention layers).
    pub sliding_window: u32,
    /// RoPE frequency base (`10000.0` on SWA layers, `64000000.0` on full-attention layers).
    pub rope_freq_base_bits: u32,
    /// `blk.{l}.attn_norm.weight` (`F32[4096]`).
    pub attn_norm: MimoTensorLoc,
    /// Split or fused Q/K/V projection weights.
    pub qkv: MimoQkvCatalog,
    /// `blk.{l}.attn_output.weight` (`[8192, 4096]`).
    pub attn_output: MimoTensorLoc,
    /// Learned attention sinks `blk.{l}.attn_sinks.weight` (`F32[64]`), present on SWA layers.
    pub attn_sinks: Option<MimoTensorLoc>,
    /// Optional `blk.{l}.post_attention_norm.weight` (`F32[4096]`).
    pub post_attention_norm: Option<MimoTensorLoc>,
    /// `blk.{l}.ffn_norm.weight` (`F32[4096]`).
    pub ffn_norm: MimoTensorLoc,
    /// Dense (Layer 0) or 256-expert MoE (Layers 1..47) FFN weights.
    pub ffn: MimoFfnCatalog,
    /// Optional `blk.{l}.post_ffn_norm.weight` (`F32[4096]`).
    pub post_ffn_norm: Option<MimoTensorLoc>,
}

impl MimoLayerCatalog {
    /// Returns `true` if this layer uses SWA-128 attention (`tk_attn_swa_sinks`).
    #[inline]
    pub const fn is_swa(&self) -> bool {
        matches!(self.attn_kind, MimoAttentionKind::Swa128)
    }

    /// Returns `true` if this layer is a 256-expert MoE layer (`tk_mmvq_ids` / `tk_prefill_moe_mmq`).
    #[inline]
    pub const fn is_moe(&self) -> bool {
        matches!(self.ffn_kind, MimoFfnKind::Moe256Top8)
    }

    /// RoPE frequency base for this layer (`10000.0` for SWA, `64000000.0` for full attention).
    #[inline]
    pub fn rope_freq_base(&self) -> f32 {
        f32::from_bits(self.rope_freq_base_bits)
    }
}

/// Validated multi-shard tensor catalog for MiMo-V2.6-Flash (`mimo2`).
#[derive(Debug, Clone)]
pub struct MimoCatalog {
    /// Parsed model hyperparameters.
    pub hparams: MimoHparams,
    /// `token_embd.weight` (`[4096, vocab]`).
    pub token_embd: MimoTensorLoc,
    /// `output_norm.weight` (`F32[4096]`).
    pub output_norm: MimoTensorLoc,
    /// `output.weight` (`[4096, vocab]`).
    pub output: MimoTensorLoc,
    /// Per-layer catalog entries (`0..48`).
    pub layers: Vec<MimoLayerCatalog>,
    /// Total number of tensors across all shards.
    pub total_tensors: usize,
    /// Total weight payload bytes across all shards.
    pub total_weight_bytes: u64,
    /// Number of GGUF shards in this catalog.
    pub n_shards: usize,
}

impl MimoCatalog {
    /// Open and validate a single or multi-shard `mimo2` GGUF from disk paths (header-only safe).
    ///
    /// If `shard_paths` has 1 path whose filename ends with `-00001-of-00002.gguf` or
    /// `-00001-of-00003.gguf` (or `.gguf.hdr`), sibling shards in the same directory
    /// are automatically discovered and opened.
    pub fn open_shards<P: AsRef<Path>>(shard_paths: &[P]) -> Result<Self> {
        if shard_paths.is_empty() {
            return Err(CoreError::InvalidModelDesc {
                field: "shards",
                reason: "at least 1 GGUF shard path is required".to_string(),
            });
        }
        let expanded = expand_sibling_shards(shard_paths[0].as_ref(), shard_paths);
        let mut files = Vec::with_capacity(expanded.len());
        for p in &expanded {
            files.push(GgufFile::open(p).map_err(|e| CoreError::InvalidArgument {
                reason: e.to_string(),
            })?);
        }
        let refs: Vec<&GgufFile> = files.iter().collect();
        Self::from_shards(&refs)
    }

    /// Validate a single-shard `mimo2` GGUF view.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Self::from_shards(&[gguf])
    }

    /// Validate a multi-shard `mimo2` GGUF catalog from parsed [`GgufFile`] headers.
    pub fn from_shards(shards: &[&GgufFile]) -> Result<Self> {
        if shards.is_empty() {
            return Err(CoreError::InvalidModelDesc {
                field: "shards",
                reason: "at least 1 shard is required".to_string(),
            });
        }

        let primary = shards[0];
        let arch = primary
            .get_metadata("general.architecture")
            .and_then(GgufValue::as_str)
            .ok_or_else(|| CoreError::InvalidModelDesc {
                field: "general.architecture",
                reason: "missing general.architecture metadata".to_string(),
            })?;
        if arch != ARCH_NAME {
            return Err(CoreError::UnsupportedArch {
                arch: arch.to_string(),
                supported: &[ARCH_NAME],
            });
        }

        // Index all tensors across all shards without reading weight payloads.
        let mut tensor_map: BTreeMap<String, MimoTensorLoc> = BTreeMap::new();
        let mut total_tensors = 0usize;
        let mut total_weight_bytes = 0u64;
        for (shard_idx, shard) in shards.iter().enumerate() {
            total_tensors += shard.tensors().len();
            for entry in shard.tensors() {
                total_weight_bytes = total_weight_bytes.saturating_add(entry.byte_size);
                tensor_map.insert(
                    entry.name.clone(),
                    MimoTensorLoc {
                        shard_idx,
                        entry: entry.clone(),
                    },
                );
            }
        }

        let get_tensor = |name: &str| -> Result<MimoTensorLoc> {
            tensor_map
                .get(name)
                .cloned()
                .ok_or_else(|| CoreError::MissingTensor {
                    name: name.to_string(),
                })
        };

        let raw_block_count = get_meta_usize(primary, "mimo2.block_count").unwrap_or(N_LAYERS);
        let n_nextn_layers = get_meta_usize(primary, "mimo2.nextn_predict_layers").unwrap_or(0);
        let n_layers =
            if raw_block_count > n_nextn_layers && raw_block_count - n_nextn_layers == N_LAYERS {
                N_LAYERS
            } else {
                raw_block_count
            };
        let n_embd = get_meta_usize(primary, "mimo2.embedding_length").unwrap_or(N_EMBD);
        let n_ff_dense = get_meta_usize(primary, "mimo2.feed_forward_length").unwrap_or(N_FF_DENSE);
        let n_ff_exp =
            get_meta_usize(primary, "mimo2.expert_feed_forward_length").unwrap_or(N_FF_EXPERT_UD);
        let n_expert = get_meta_usize(primary, "mimo2.expert_count").unwrap_or(N_EXPERT);
        let n_expert_used =
            get_meta_usize(primary, "mimo2.expert_used_count").unwrap_or(N_EXPERT_USED);
        let head_dim_k =
            get_meta_usize(primary, "mimo2.attention.key_length").unwrap_or(HEAD_DIM_K);
        let head_dim_v =
            get_meta_usize(primary, "mimo2.attention.value_length").unwrap_or(HEAD_DIM_V);
        let rope_dim = get_meta_usize(primary, "mimo2.rope.dimension_count").unwrap_or(ROPE_DIM);
        let sliding_window =
            get_meta_u32(primary, "mimo2.attention.sliding_window").unwrap_or(SWA_WINDOW);
        let rms_norm_eps =
            get_meta_f32(primary, "mimo2.attention.layer_norm_rms_epsilon").unwrap_or(RMS_NORM_EPS);
        let attn_value_scale = get_meta_f32(primary, "mimo2.attention.value_scale")
            .filter(|&v| v > 0.0)
            .unwrap_or(1.0);
        let rope_freq_base_full =
            get_meta_f32(primary, "mimo2.rope.freq_base").unwrap_or(ROPE_FREQ_BASE_FULL);
        let rope_freq_base_swa =
            get_meta_f32(primary, "mimo2.rope.freq_base_swa").unwrap_or(ROPE_FREQ_BASE_SWA);
        let expert_weights_norm = primary
            .get_metadata("mimo2.expert_weights_norm")
            .and_then(GgufValue::as_bool)
            .unwrap_or(true);
        let expert_gating_func = primary
            .get_metadata("mimo2.expert_gating_func")
            .and_then(GgufValue::as_i64)
            .unwrap_or(2) as i32;

        let token_embd = get_tensor("token_embd.weight")?;
        check_dims_len(&token_embd, 2)?;
        if token_embd.entry.dims[0] as usize != n_embd {
            return Err(CoreError::ShapeMismatch {
                tensor: token_embd.entry.name.clone(),
                expected: vec![n_embd as u64, 0],
                actual: token_embd.entry.shape().to_vec(),
            });
        }
        let n_vocab = token_embd.entry.dims[1] as usize;

        let output_norm = get_tensor("output_norm.weight")?;
        check_exact_shape(&output_norm, &[n_embd as u64])?;

        let output = get_tensor("output.weight")?;
        check_exact_shape(&output, &[n_embd as u64, n_vocab as u64])?;

        let n_head_q =
            get_meta_usize_or_first(primary, "mimo2.attention.head_count").unwrap_or(N_HEAD_Q);

        let canonical_swa_mask: Vec<bool> = (0..n_layers).map(is_canonical_swa_layer).collect();

        let mut layers = Vec::with_capacity(n_layers);
        let mut n_head_kv_swa = N_HEAD_KV_SWA;
        let mut n_head_kv_full = N_HEAD_KV_FULL;

        for il in 0..n_layers {
            let prefix = format!("blk.{il}");
            let attn_norm = get_tensor(&format!("{prefix}.attn_norm.weight"))?;
            check_exact_shape(&attn_norm, &[n_embd as u64])?;

            let attn_sinks = tensor_map
                .get(&format!("{prefix}.attn_sinks.weight"))
                .cloned();
            if let Some(ref sinks) = attn_sinks {
                check_exact_shape(sinks, &[n_head_q as u64])?;
            }

            let attn_q_norm = tensor_map
                .get(&format!("{prefix}.attn_q_norm.weight"))
                .cloned();
            if let Some(ref qn) = attn_q_norm {
                check_exact_shape(qn, &[head_dim_k as u64])?;
            }
            let attn_k_norm = tensor_map
                .get(&format!("{prefix}.attn_k_norm.weight"))
                .cloned();
            if let Some(ref kn) = attn_k_norm {
                check_exact_shape(kn, &[head_dim_k as u64])?;
            }

            // Support both split Q/K/V (`UD-IQ2_XXS`) and fused QKV (`MOPD-a4`).
            let (qkv, n_head_kv) =
                if let Some(attn_q) = tensor_map.get(&format!("{prefix}.attn_q.weight")).cloned() {
                    let attn_k = get_tensor(&format!("{prefix}.attn_k.weight"))?;
                    let attn_v = get_tensor(&format!("{prefix}.attn_v.weight"))?;
                    check_exact_shape(&attn_q, &[n_embd as u64, (n_head_q * head_dim_k) as u64])?;
                    check_dims_len(&attn_k, 2)?;
                    check_dims_len(&attn_v, 2)?;
                    let k_out = attn_k.entry.dims[1] as usize;
                    let v_out = attn_v.entry.dims[1] as usize;
                    if k_out % head_dim_k != 0 || v_out % head_dim_v != 0 {
                        return Err(CoreError::ShapeMismatch {
                            tensor: attn_k.entry.name.clone(),
                            expected: vec![n_embd as u64, head_dim_k as u64],
                            actual: attn_k.entry.shape().to_vec(),
                        });
                    }
                    let n_kv = k_out / head_dim_k;
                    check_exact_shape(&attn_v, &[n_embd as u64, (n_kv * head_dim_v) as u64])?;
                    (
                        MimoQkvCatalog::Split {
                            attn_q,
                            attn_k,
                            attn_v,
                            attn_q_norm,
                            attn_k_norm,
                        },
                        n_kv,
                    )
                } else {
                    let attn_qkv = get_tensor(&format!("{prefix}.attn_qkv.weight"))?;
                    check_dims_len(&attn_qkv, 2)?;
                    if attn_qkv.entry.dims[0] as usize != n_embd {
                        return Err(CoreError::ShapeMismatch {
                            tensor: attn_qkv.entry.name.clone(),
                            expected: vec![n_embd as u64, 0],
                            actual: attn_qkv.entry.shape().to_vec(),
                        });
                    }
                    let qkv_out = attn_qkv.entry.dims[1] as usize;
                    let q_dim = n_head_q * head_dim_k;
                    let kv_pair_dim = head_dim_k + head_dim_v;
                    if qkv_out <= q_dim || (qkv_out - q_dim) % kv_pair_dim != 0 {
                        return Err(CoreError::ShapeMismatch {
                            tensor: attn_qkv.entry.name.clone(),
                            expected: vec![n_embd as u64, (q_dim + kv_pair_dim) as u64],
                            actual: attn_qkv.entry.shape().to_vec(),
                        });
                    }
                    let n_kv = (qkv_out - q_dim) / kv_pair_dim;
                    (
                        MimoQkvCatalog::Fused {
                            attn_qkv,
                            attn_q_norm,
                            attn_k_norm,
                        },
                        n_kv,
                    )
                };

            // Determine SWA vs Full-attention for this layer:
            // In canonical `UD-IQ2_XXS`, the 40 layers `il % 6 != 5` have `attn_sinks` and `n_kv = 8`
            // while the 8 layers `il % 6 == 5` have no `attn_sinks` and `n_kv = 4`.
            let is_swa = if attn_sinks.is_some() {
                true
            } else if il % 6 == 5 {
                false
            } else {
                is_canonical_swa_layer(il) && n_head_kv > n_head_kv_full
            };

            if is_swa {
                n_head_kv_swa = n_head_kv;
            } else {
                n_head_kv_full = n_head_kv;
            }

            let attn_output = get_tensor(&format!("{prefix}.attn_output.weight"))?;
            check_exact_shape(
                &attn_output,
                &[(n_head_q * head_dim_v) as u64, n_embd as u64],
            )?;

            let post_attention_norm = tensor_map
                .get(&format!("{prefix}.post_attention_norm.weight"))
                .cloned();
            if let Some(ref pan) = post_attention_norm {
                check_exact_shape(pan, &[n_embd as u64])?;
            }

            let ffn_norm = get_tensor(&format!("{prefix}.ffn_norm.weight"))?;
            check_exact_shape(&ffn_norm, &[n_embd as u64])?;

            let post_ffn_norm = tensor_map
                .get(&format!("{prefix}.post_ffn_norm.weight"))
                .cloned();
            if let Some(ref pfn) = post_ffn_norm {
                check_exact_shape(pfn, &[n_embd as u64])?;
            }

            // Layer 0 is dense SwiGLU FFN; Layers 1..47 are 256-expert top-8 MoE.
            let (ffn_kind, ffn) = if !is_moe_layer(il) {
                let ffn_gate = get_tensor(&format!("{prefix}.ffn_gate.weight"))?;
                let ffn_up = get_tensor(&format!("{prefix}.ffn_up.weight"))?;
                let ffn_down = get_tensor(&format!("{prefix}.ffn_down.weight"))?;
                check_exact_shape(&ffn_gate, &[n_embd as u64, n_ff_dense as u64])?;
                check_exact_shape(&ffn_up, &[n_embd as u64, n_ff_dense as u64])?;
                check_exact_shape(&ffn_down, &[n_ff_dense as u64, n_embd as u64])?;
                (
                    MimoFfnKind::Dense,
                    MimoFfnCatalog::Dense {
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                    },
                )
            } else {
                let ffn_gate_inp = get_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?;
                check_exact_shape(&ffn_gate_inp, &[n_embd as u64, n_expert as u64])?;

                let exp_probs_b = tensor_map
                    .get(&format!("{prefix}.exp_probs_b.bias"))
                    .cloned();
                if let Some(ref b) = exp_probs_b {
                    check_exact_shape(b, &[n_expert as u64])?;
                }

                let ffn_gate_exps = get_tensor(&format!("{prefix}.ffn_gate_exps.weight"))?;
                let ffn_up_exps = get_tensor(&format!("{prefix}.ffn_up_exps.weight"))?;
                let ffn_down_exps = get_tensor(&format!("{prefix}.ffn_down_exps.weight"))?;
                check_exact_shape(
                    &ffn_gate_exps,
                    &[n_embd as u64, n_ff_exp as u64, n_expert as u64],
                )?;
                check_exact_shape(
                    &ffn_up_exps,
                    &[n_embd as u64, n_ff_exp as u64, n_expert as u64],
                )?;
                check_exact_shape(
                    &ffn_down_exps,
                    &[n_ff_exp as u64, n_embd as u64, n_expert as u64],
                )?;

                (
                    MimoFfnKind::Moe256Top8,
                    MimoFfnCatalog::Moe {
                        ffn_gate_inp,
                        exp_probs_b,
                        ffn_gate_exps,
                        ffn_up_exps,
                        ffn_down_exps,
                    },
                )
            };

            let (attn_kind, layer_window, layer_rope_base) = if is_swa {
                (
                    MimoAttentionKind::Swa128,
                    sliding_window,
                    rope_freq_base_swa,
                )
            } else {
                (MimoAttentionKind::FullK192V128, 0, rope_freq_base_full)
            };

            layers.push(MimoLayerCatalog {
                layer_idx: il,
                attn_kind,
                ffn_kind,
                n_head_q,
                n_head_kv,
                sliding_window: layer_window,
                rope_freq_base_bits: layer_rope_base.to_bits(),
                attn_norm,
                qkv,
                attn_output,
                attn_sinks,
                post_attention_norm,
                ffn_norm,
                ffn,
                post_ffn_norm,
            });
        }

        let hparams = MimoHparams {
            n_layers,
            n_nextn_layers,
            n_embd,
            n_ff_dense,
            n_ff_exp,
            n_expert,
            n_expert_used,
            n_vocab,
            n_head_q,
            n_head_kv_swa,
            n_head_kv_full,
            head_dim_k,
            head_dim_v,
            rope_dim,
            rope_freq_base_full,
            rope_freq_base_swa,
            sliding_window,
            rms_norm_eps,
            attn_value_scale,
            expert_weights_norm,
            expert_gating_func,
            canonical_swa_mask,
        };

        Ok(Self {
            hparams,
            token_embd,
            output_norm,
            output,
            layers,
            total_tensors,
            total_weight_bytes,
            n_shards: shards.len(),
        })
    }

    /// Count of layers cataloged as `Swa128`.
    pub fn swa_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.is_swa()).count()
    }

    /// Count of layers cataloged as `FullK192V128`.
    pub fn full_attn_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| !l.is_swa()).count()
    }

    /// Count of dense FFN layers (`1`, Layer 0).
    pub fn dense_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| !l.is_moe()).count()
    }

    /// Count of 256-expert MoE layers (`47`, Layers 1..47).
    pub fn moe_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.is_moe()).count()
    }
}

/// Validated catalog for the MiMo DFlash drafter GGUF (`general.architecture = "dflash"`).
///
/// Supports both:
/// - The 5-layer `MiMo-V2-Flash-DFlash-Q8_0.gguf` (`59` tensors, `block_count = 5`,
///   `feed_forward_length = 20480`, `n_head = 64`, `n_head_kv = 8`, `k_dim = 192`,
///   `v_dim = 128`, `rope_dim = 64`, `rope_freq_base = 64000000.0`, `block_size = 16`,
///   target layer taps `[1, 12, 24, 36, 47]`, `output.weight` `Q8_0 [4096, 152064]`,
///   `output_norm.weight` `F32[4096]`).
/// - The 8-layer `dflash-MiMo-V2.6-Flash-MOPD-Q8_0.gguf` (`65` tensors, `block_count = 8`,
///   `feed_forward_length = 16384`, `n_head = 32`, `n_head_kv = 8`, `k_dim = 128`,
///   `v_dim = 128`, target layer taps `[1, 7, 14, 20, 27, 33, 40, 46]`,
///   `output.weight` `Q8_0 [4096, 152576]`).
#[derive(Debug, Clone)]
pub struct MimoDFlashCatalog {
    /// Number of drafter transformer blocks (`5` or `8`).
    pub block_count: usize,
    /// Hidden state width (`4096`).
    pub embedding_length: usize,
    /// Dense SwiGLU intermediate width (`20480` or `16384`).
    pub feed_forward_length: usize,
    /// Query head count (`64` or `32`).
    pub n_head: usize,
    /// KV head count (`8`).
    pub n_head_kv: usize,
    /// Key head dimension (`192` or `128`).
    pub k_dim: usize,
    /// Value head dimension (`128`).
    pub v_dim: usize,
    /// RoPE rotary dimension (`64`).
    pub rope_dim: usize,
    /// RoPE frequency base (`64000000.0` or `10000.0`).
    pub rope_freq_base: f32,
    /// Sliding window length (`1024` or `max_ctx`).
    pub sliding_window: u32,
    /// Post-attention output scale (`dflash.attention.value_scale`, `0.6119999885559082` or `1.0`).
    pub attn_value_scale: f32,
    /// Diffusion mask token ID (`tokenizer.ggml.mask_token_id`, default `151675`).
    pub mask_token_id: i32,
    /// DFlash block size (`16` or `8`).
    pub block_size: u32,
    /// Tapped target layer indices (`[1, 12, 24, 36, 47]` or `[1, 7, 14, 20, 27, 33, 40, 46]`).
    pub target_layers: Vec<u32>,
    /// Target vocabulary rows in `output.weight` (`152064` or `152576`).
    pub n_vocab: usize,
    /// Dedicated `output.weight` (`Q8_0 [4096, n_vocab]`).
    pub output_weight: TensorEntry,
    /// Dedicated `output_norm.weight` (`F32[4096]`).
    pub output_norm: TensorEntry,
    /// Encoder fusion `fc.weight` (`[n_taps * 4096, 4096]`).
    pub fc_weight: TensorEntry,
    /// Encoder norm `enc.output_norm.weight` (`F32[4096]`).
    pub enc_output_norm: TensorEntry,
    /// Total tensor count in the DFlash GGUF (`59` for 5-layer, `65` for 8-layer).
    pub total_tensors: usize,
    /// Total weight bytes across all drafter tensors.
    pub total_weight_bytes: u64,
}

impl MimoDFlashCatalog {
    /// Open and validate a DFlash drafter GGUF from a file or `.hdr` path (header-only safe).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufFile::open(path).map_err(|e| CoreError::InvalidArgument {
            reason: e.to_string(),
        })?;
        Self::from_gguf(&gguf)
    }

    /// Validate a parsed [`GgufFile`] as a MiMo DFlash drafter.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = gguf
            .get_metadata("general.architecture")
            .and_then(GgufValue::as_str)
            .ok_or_else(|| CoreError::InvalidModelDesc {
                field: "general.architecture",
                reason: "missing general.architecture metadata".to_string(),
            })?;
        if arch != DFLASH_ARCH_NAME {
            return Err(CoreError::UnsupportedArch {
                arch: arch.to_string(),
                supported: &[DFLASH_ARCH_NAME],
            });
        }

        let block_count = get_meta_usize(gguf, "dflash.block_count").unwrap_or(5);
        let embedding_length = get_meta_usize(gguf, "dflash.embedding_length").unwrap_or(N_EMBD);
        let feed_forward_length =
            get_meta_usize(gguf, "dflash.feed_forward_length").unwrap_or(20480);
        let n_head = get_meta_usize_or_first(gguf, "dflash.attention.head_count").unwrap_or(64);
        let n_head_kv =
            get_meta_usize_or_first(gguf, "dflash.attention.head_count_kv").unwrap_or(8);
        let k_dim = get_meta_usize(gguf, "dflash.attention.key_length").unwrap_or(192);
        let v_dim = get_meta_usize(gguf, "dflash.attention.value_length").unwrap_or(128);
        let rope_dim = get_meta_usize(gguf, "dflash.rope.dimension_count").unwrap_or(64);
        let rope_freq_base =
            get_meta_f32(gguf, "dflash.rope.freq_base").unwrap_or(ROPE_FREQ_BASE_FULL);
        let sliding_window = get_meta_u32(gguf, "dflash.attention.sliding_window").unwrap_or(1024);
        let attn_value_scale = get_meta_f32(gguf, "dflash.attention.value_scale")
            .filter(|&v| v > 0.0)
            .unwrap_or(1.0);
        let mask_token_id = gguf
            .get_metadata("tokenizer.ggml.mask_token_id")
            .and_then(GgufValue::as_i64)
            .map(|v| v as i32)
            .unwrap_or(151675);
        let block_size = get_meta_u32(gguf, "dflash.dflash.block_size")
            .or_else(|| get_meta_u32(gguf, "dflash.block_size"))
            .unwrap_or(DFLASH_BLOCK_SIZE);

        let target_layers = gguf
            .get_metadata("dflash.target_layers")
            .and_then(GgufValue::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as u32))
                    .collect::<Vec<u32>>()
            })
            .unwrap_or_else(|| DFLASH_5L_TARGET_TAPS.to_vec());

        if target_layers.len() != block_count {
            return Err(CoreError::InvalidModelDesc {
                field: "dflash.target_layers",
                reason: format!(
                    "expected {block_count} target layer taps, got {:?}",
                    target_layers
                ),
            });
        }

        let get_t = |name: &str| -> Result<TensorEntry> {
            gguf.find_tensor(name)
                .cloned()
                .ok_or_else(|| CoreError::MissingTensor {
                    name: name.to_string(),
                })
        };

        let fc_weight = get_t("fc.weight")?;
        if fc_weight.shape()
            != [
                (block_count * embedding_length) as u64,
                embedding_length as u64,
            ]
        {
            return Err(CoreError::ShapeMismatch {
                tensor: "fc.weight".to_string(),
                expected: vec![
                    (block_count * embedding_length) as u64,
                    embedding_length as u64,
                ],
                actual: fc_weight.shape().to_vec(),
            });
        }

        let enc_output_norm = get_t("enc.output_norm.weight")?;
        if enc_output_norm.shape() != [embedding_length as u64] {
            return Err(CoreError::ShapeMismatch {
                tensor: "enc.output_norm.weight".to_string(),
                expected: vec![embedding_length as u64],
                actual: enc_output_norm.shape().to_vec(),
            });
        }

        let output_norm = get_t("output_norm.weight")?;
        if output_norm.shape() != [embedding_length as u64] {
            return Err(CoreError::ShapeMismatch {
                tensor: "output_norm.weight".to_string(),
                expected: vec![embedding_length as u64],
                actual: output_norm.shape().to_vec(),
            });
        }

        let output_weight = get_t("output.weight")?;
        if output_weight.n_dims != 2 || output_weight.dims[0] as usize != embedding_length {
            return Err(CoreError::ShapeMismatch {
                tensor: "output.weight".to_string(),
                expected: vec![embedding_length as u64, 0],
                actual: output_weight.shape().to_vec(),
            });
        }
        let n_vocab = output_weight.dims[1] as usize;

        for il in 0..block_count {
            let prefix = format!("blk.{il}");
            let _ = get_t(&format!("{prefix}.attn_norm.weight"))?;
            let _ = get_t(&format!("{prefix}.attn_output.weight"))?;
            let _ = get_t(&format!("{prefix}.ffn_norm.weight"))?;
            let _ = get_t(&format!("{prefix}.ffn_down.weight"))?;
            let has_split_qkv = gguf
                .find_tensor(&format!("{prefix}.attn_q.weight"))
                .is_some()
                && gguf
                    .find_tensor(&format!("{prefix}.attn_k.weight"))
                    .is_some()
                && gguf
                    .find_tensor(&format!("{prefix}.attn_v.weight"))
                    .is_some();
            let has_fused_qkv = gguf
                .find_tensor(&format!("{prefix}.attn_qkv.weight"))
                .is_some();
            if !has_split_qkv && !has_fused_qkv {
                return Err(CoreError::MissingTensor {
                    name: format!("{prefix}.attn_q.weight or {prefix}.attn_qkv.weight"),
                });
            }
            let has_split_ffn = gguf
                .find_tensor(&format!("{prefix}.ffn_gate.weight"))
                .is_some()
                && gguf
                    .find_tensor(&format!("{prefix}.ffn_up.weight"))
                    .is_some();
            let has_fused_ffn = gguf
                .find_tensor(&format!("{prefix}.ffn_gate_up.weight"))
                .is_some();
            if !has_split_ffn && !has_fused_ffn {
                return Err(CoreError::MissingTensor {
                    name: format!("{prefix}.ffn_gate.weight or {prefix}.ffn_gate_up.weight"),
                });
            }
        }

        let total_weight_bytes = gguf
            .tensors()
            .iter()
            .fold(0u64, |acc, t| acc.saturating_add(t.byte_size));

        Ok(Self {
            block_count,
            embedding_length,
            feed_forward_length,
            n_head,
            n_head_kv,
            k_dim,
            v_dim,
            rope_dim,
            rope_freq_base,
            sliding_window,
            attn_value_scale,
            mask_token_id,
            block_size,
            target_layers,
            n_vocab,
            output_weight,
            output_norm,
            fc_weight,
            enc_output_norm,
            total_tensors: gguf.tensors().len(),
            total_weight_bytes,
        })
    }
}

/// Validated descriptor for the 64K MiMo draft vocabulary (`mimo-dv-65536.bin`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoDraftVocabInfo {
    /// Draft vocabulary size (`65,536`).
    pub draft_vocab_size: usize,
    /// Minimum target token ID (`0`).
    pub min_target_id: i32,
    /// Maximum target token ID (`< 152,064` or `< 152,576`).
    pub max_target_id: i32,
    /// Whether the mapping is strictly ascending (sorted).
    pub is_sorted_unique: bool,
    /// Full draft-to-target (`d2t`) token ID table.
    pub draft_to_target: Vec<i32>,
}

/// Read a draft vocabulary `.bin` file (`mimo-dv-65536.bin`) supporting both the
/// 16-byte `DVOC` header format (`magic = 0x434f5644`, `version = 1`, `n_draft`, `n_vocab`)
/// and raw little-endian `int32` token ID tables (`262,144` bytes for 64K entries).
pub fn load_draft_vocab_bin(path: &Path) -> Result<Vec<i32>> {
    let bytes = std::fs::read(path).map_err(|e| CoreError::IoError {
        message: format!("failed to read draft vocab {}: {e}", path.display()),
    })?;
    if bytes.len() < 4 || !bytes.len().is_multiple_of(4) {
        return Err(CoreError::InvalidArgument {
            reason: format!(
                "draft vocab file {} has invalid byte length {}",
                path.display(),
                bytes.len()
            ),
        });
    }
    let first_u32 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let payload = if first_u32 == 0x434f_5644 {
        if bytes.len() < 16 {
            return Err(CoreError::InvalidArgument {
                reason: format!("truncated DVOC header in {}", path.display()),
            });
        }
        let n_draft = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let expected_len = 16 + n_draft * 4;
        if bytes.len() < expected_len {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "DVOC file {} truncated: got {}, need {expected_len}",
                    path.display(),
                    bytes.len()
                ),
            });
        }
        &bytes[16..expected_len]
    } else {
        &bytes[..]
    };

    let mut ids = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.chunks_exact(4) {
        ids.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(ids)
}

impl MimoDraftVocabInfo {
    /// Load and validate a `mimo-dv-65536.bin` file from `path`.
    pub fn from_path(path: impl AsRef<Path>, target_vocab_size: usize) -> Result<Self> {
        let ids = load_draft_vocab_bin(path.as_ref())?;
        Self::from_ids(ids, target_vocab_size)
    }

    /// Validate an in-memory draft-to-target token ID table.
    pub fn from_ids(ids: Vec<i32>, target_vocab_size: usize) -> Result<Self> {
        if ids.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "mimo-dv-65536.bin is empty".to_string(),
            });
        }
        let upper_bound = target_vocab_size.max(N_VOCAB_MOPD);
        let mut min_id = i32::MAX;
        let mut max_id = i32::MIN;
        let mut is_sorted_unique = true;
        for (i, &id) in ids.iter().enumerate() {
            if id < 0 || (id as usize) >= upper_bound {
                return Err(CoreError::InvalidArgument {
                    reason: format!(
                        "mimo-dv-65536.bin id[{i}] = {id} out of range 0..{upper_bound}"
                    ),
                });
            }
            if i > 0 && id <= ids[i - 1] {
                is_sorted_unique = false;
            }
            min_id = min_id.min(id);
            max_id = max_id.max(id);
        }
        Ok(Self {
            draft_vocab_size: ids.len(),
            min_target_id: min_id,
            max_target_id: max_id,
            is_sorted_unique,
            draft_to_target: ids,
        })
    }

    /// Return `true` if `target_id` is in the draft vocabulary subset.
    #[inline]
    pub fn contains_target_id(&self, target_id: i32) -> bool {
        if self.is_sorted_unique {
            self.draft_to_target.binary_search(&target_id).is_ok()
        } else {
            self.draft_to_target.contains(&target_id)
        }
    }
}

/// Resolve the best available draft vocabulary path (`mimo-dv-65536.bin`), checking
/// `TALUS_MIMO_DRAFT_VOCAB`, the default model path, and `LOCAL_MIMO_DRAFT_VOCAB_PATH`.
pub fn resolve_mimo_draft_vocab_path() -> Option<PathBuf> {
    if let Ok(env_p) = std::env::var("TALUS_MIMO_DRAFT_VOCAB") {
        let p = PathBuf::from(env_p);
        if p.exists() {
            return Some(p);
        }
    }
    [
        default_model_path(DEFAULT_MIMO_DRAFT_VOCAB_PATH),
        PathBuf::from(LOCAL_MIMO_DRAFT_VOCAB_PATH),
    ]
    .into_iter()
    .find(|p| p.exists())
}

/// Build the canonical 2-shard `MiMo-V2.6-Flash-UD-IQ2_XXS` header-only [`GgufFile`]s
/// (`821` tensors total across 2 shards, `40` SWA layers `i % 6 != 5` with `attn_sinks.weight`,
/// `8` full-attention layers `i % 6 == 5` without `attn_sinks.weight`, Layer 0 dense FFN
/// `[4096, 16384]`, Layers 1..47 256-expert top-8 MoE `[4096, 1280, 256]` in `IQ2_XXS` /
/// `IQ3_XXS` / `IQ2_XS`, `vocab = 152064`).
///
/// Uses [`GgufFile::from_header_bytes`] so zero weight payload bytes are allocated on the host.
pub fn build_synthetic_mimo_ud_gguf_shards() -> Result<(GgufFile, GgufFile)> {
    let mut shard1 = GgufHeaderWriter::new();
    let mut shard2 = GgufHeaderWriter::new();

    for w in [&mut shard1, &mut shard2] {
        w.add_str("general.architecture", ARCH_NAME);
        w.add_u32("mimo2.block_count", N_LAYERS as u32);
        w.add_u32("mimo2.embedding_length", N_EMBD as u32);
        w.add_u32("mimo2.feed_forward_length", N_FF_DENSE as u32);
        w.add_u32("mimo2.expert_feed_forward_length", N_FF_EXPERT_UD as u32);
        w.add_u32("mimo2.expert_count", N_EXPERT as u32);
        w.add_u32("mimo2.expert_used_count", N_EXPERT_USED as u32);
        w.add_u32("mimo2.attention.head_count", N_HEAD_Q as u32);
        w.add_u32("mimo2.attention.key_length", HEAD_DIM_K as u32);
        w.add_u32("mimo2.attention.value_length", HEAD_DIM_V as u32);
        w.add_u32("mimo2.attention.sliding_window", SWA_WINDOW);
        w.add_f32("mimo2.attention.layer_norm_rms_epsilon", RMS_NORM_EPS);
        w.add_u32("mimo2.rope.dimension_count", ROPE_DIM as u32);
        w.add_f32("mimo2.rope.freq_base", ROPE_FREQ_BASE_FULL);
        w.add_f32("mimo2.rope.freq_base_swa", ROPE_FREQ_BASE_SWA);
        w.add_bool("mimo2.expert_weights_norm", true);
        w.add_u32("mimo2.expert_gating_func", 2);
    }

    // Global tensors (3 tensors: token_embd on shard 1, output_norm + output on shard 2).
    shard1.add_tensor(
        "token_embd.weight",
        QuantType::Q8_0,
        &[N_EMBD as u64, N_VOCAB_UD as u64],
    );

    // Layer 0..23 on shard 1, Layer 24..47 on shard 2.
    // Tensor breakdown to match the exact 821-tensor UD-IQ2_XXS GGUF:
    // - Globals: 3 (`token_embd.weight`, `output_norm.weight`, `output.weight`)
    // - Layer 0 (SWA, dense FFN): 14 tensors:
    //   `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm`,
    //   `attn_sinks`, `post_attention_norm`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_down`, `post_ffn_norm`
    // - 39 SWA MoE layers (`il in 1..48, il % 6 != 5`): 17 tensors each (`39 * 17 = 663`):
    //   `attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm`,
    //   `attn_sinks`, `post_attention_norm`, `ffn_norm`, `ffn_gate_inp`, `exp_probs_b.bias`,
    //   `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, `post_ffn_norm`, plus `ffn_gate_inp_shexp`
    //   (or shared gate norm).
    // - 8 Full-attention MoE layers (`il % 6 == 5`): 16 tensors each (`8 * 16 = 128`, no `attn_sinks`),
    //   plus 13 NextN/MTP trunk tensors on `blk.48` -> `3 + 14 + 663 + 128 + 13 = 821`!
    for il in 0..N_LAYERS {
        let w = if il < 24 { &mut shard1 } else { &mut shard2 };
        let is_swa = is_canonical_swa_layer(il);
        let n_kv = if is_swa {
            N_HEAD_KV_SWA
        } else {
            N_HEAD_KV_FULL
        };
        let prefix = format!("blk.{il}");

        w.add_tensor(
            &format!("{prefix}.attn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_q.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, (N_HEAD_Q * HEAD_DIM_K) as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_k.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, (n_kv * HEAD_DIM_K) as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_v.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, (n_kv * HEAD_DIM_V) as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_output.weight"),
            QuantType::Q4_K,
            &[(N_HEAD_Q * HEAD_DIM_V) as u64, N_EMBD as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_q_norm.weight"),
            QuantType::F32,
            &[HEAD_DIM_K as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_k_norm.weight"),
            QuantType::F32,
            &[HEAD_DIM_K as u64],
        );
        if is_swa {
            w.add_tensor(
                &format!("{prefix}.attn_sinks.weight"),
                QuantType::F32,
                &[N_HEAD_Q as u64],
            );
        }
        w.add_tensor(
            &format!("{prefix}.post_attention_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        w.add_tensor(
            &format!("{prefix}.ffn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );

        if il == 0 {
            w.add_tensor(
                "blk.0.ffn_gate.weight",
                QuantType::Q4_K,
                &[N_EMBD as u64, N_FF_DENSE as u64],
            );
            w.add_tensor(
                "blk.0.ffn_up.weight",
                QuantType::Q4_K,
                &[N_EMBD as u64, N_FF_DENSE as u64],
            );
            w.add_tensor(
                "blk.0.ffn_down.weight",
                QuantType::Q6_K,
                &[N_FF_DENSE as u64, N_EMBD as u64],
            );
        } else {
            let exp_qtype = match il % 3 {
                0 => QuantType::IQ2_XXS,
                1 => QuantType::IQ3_XXS,
                _ => QuantType::IQ2_XS,
            };
            w.add_tensor(
                &format!("{prefix}.ffn_gate_inp.weight"),
                QuantType::F32,
                &[N_EMBD as u64, N_EXPERT as u64],
            );
            w.add_tensor(
                &format!("{prefix}.exp_probs_b.bias"),
                QuantType::F32,
                &[N_EXPERT as u64],
            );
            w.add_tensor(
                &format!("{prefix}.ffn_gate_exps.weight"),
                exp_qtype,
                &[N_EMBD as u64, N_FF_EXPERT_UD as u64, N_EXPERT as u64],
            );
            w.add_tensor(
                &format!("{prefix}.ffn_up_exps.weight"),
                exp_qtype,
                &[N_EMBD as u64, N_FF_EXPERT_UD as u64, N_EXPERT as u64],
            );
            w.add_tensor(
                &format!("{prefix}.ffn_down_exps.weight"),
                QuantType::IQ2_XS,
                &[N_FF_EXPERT_UD as u64, N_EMBD as u64, N_EXPERT as u64],
            );
        }
        w.add_tensor(
            &format!("{prefix}.post_ffn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
    }

    shard2.add_tensor("output_norm.weight", QuantType::F32, &[N_EMBD as u64]);
    shard2.add_tensor(
        "output.weight",
        QuantType::Q6_K,
        &[N_EMBD as u64, N_VOCAB_UD as u64],
    );

    // 3 NextN / MTP layers (`blk.48`, `blk.49`, `blk.50`), each with the 20 real `mimo2` NextN
    // tensors from `llama-arch.cpp` (`mimo2`):
    // - 14 dense SWA transformer tensors (`attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_output`,
    //   `attn_q_norm`, `attn_k_norm`, `attn_sinks`, `post_attention_norm`, `ffn_norm`, `ffn_gate`,
    //   `ffn_up`, `ffn_down`, `post_ffn_norm`)
    // - 6 NextN head/fusion tensors (`nextn.eh_proj`, `nextn.enorm`, `nextn.hnorm`,
    //   `nextn.embed_tokens`, `nextn.shared_head_norm`, `layer_output_norm`).
    // Total across 2 shards: 3 + 14 + 39 * 16 + 8 * 15 + 3 * 20 = 821 tensors.
    for nextn_il in 48..51 {
        let prefix = format!("blk.{nextn_il}");
        shard2.add_tensor(
            &format!("{prefix}.attn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.attn_q.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, (N_HEAD_Q * HEAD_DIM_K) as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.attn_k.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, (N_HEAD_KV_SWA * HEAD_DIM_K) as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.attn_v.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, (N_HEAD_KV_SWA * HEAD_DIM_V) as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.attn_output.weight"),
            QuantType::Q4_K,
            &[(N_HEAD_Q * HEAD_DIM_V) as u64, N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.attn_q_norm.weight"),
            QuantType::F32,
            &[HEAD_DIM_K as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.attn_k_norm.weight"),
            QuantType::F32,
            &[HEAD_DIM_K as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.attn_sinks.weight"),
            QuantType::F32,
            &[N_HEAD_Q as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.post_attention_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.ffn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.ffn_gate.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, N_FF_DENSE as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.ffn_up.weight"),
            QuantType::Q4_K,
            &[N_EMBD as u64, N_FF_DENSE as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.ffn_down.weight"),
            QuantType::Q6_K,
            &[N_FF_DENSE as u64, N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.post_ffn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.nextn.eh_proj.weight"),
            QuantType::Q8_0,
            &[(2 * N_EMBD) as u64, N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.nextn.enorm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.nextn.hnorm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.nextn.embed_tokens.weight"),
            QuantType::Q8_0,
            &[N_EMBD as u64, N_VOCAB_UD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.nextn.shared_head_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        shard2.add_tensor(
            &format!("{prefix}.layer_output_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
    }

    Ok((shard1.finish()?, shard2.finish()?))
}

// ============================================================================
// Real GPU Weight Loader (`MimoModel::load`)
// ============================================================================

/// Pinned host staging buffer size (`64 MiB`) used to stream tensors from GGUF shards to GPU.
const STAGE_BYTES: usize = 64 << 20;

/// Device-resident tensor descriptor for `mimo2` and DFlash weights.
#[derive(Debug, Clone, Copy)]
pub struct DevTensor {
    /// Device pointer allocated via `hipMalloc` (or aliased into a fused tensor).
    pub dev: *mut c_void,
    /// `tk_qtype` (`ggml_type` enum).
    pub qtype: tk_qtype,
    /// Input feature dimension (`dims[0]`).
    pub n_in: usize,
    /// Output row count (`dims[1]`).
    pub n_out: usize,
    /// Expert count (`dims[2]`, `1` for non-expert tensors).
    pub n_exp: usize,
    /// Total byte size on device.
    pub bytes: usize,
    /// Byte stride between adjacent experts (`bytes / n_exp` when `n_exp > 1`, else `0`).
    pub expert_bytes: usize,
}

// SAFETY: `DevTensor` holds an immutable device memory pointer owned by `MimoModel`.
unsafe impl Send for DevTensor {}
unsafe impl Sync for DevTensor {}

impl Default for DevTensor {
    fn default() -> Self {
        Self::empty()
    }
}

impl DevTensor {
    /// Construct an empty (null) device tensor descriptor.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            dev: ptr::null_mut(),
            qtype: TK_QTYPE_F32,
            n_in: 0,
            n_out: 0,
            n_exp: 0,
            bytes: 0,
            expert_bytes: 0,
        }
    }

    /// True if `self.dev` is null.
    #[inline]
    pub fn is_null(&self) -> bool {
        self.dev.is_null()
    }
}

/// Per-layer device weight pointers for a `mimo2` transformer block (`0..48`).
#[derive(Debug, Clone, Copy)]
pub struct MimoLayerWeights {
    /// Zero-based layer index (`0..48`).
    pub layer_idx: usize,
    /// `true` for the 40 SWA-128 layers (`il % 6 != 5`), `false` for the 8 full-attn layers (`il % 6 == 5`).
    pub is_swa: bool,
    /// `false` on Layer 0 (dense SwiGLU), `true` on Layers 1..47 (256-expert top-8 MoE).
    pub is_moe: bool,
    /// Number of query heads (`64`).
    pub n_head_q: usize,
    /// Number of KV heads (`8` on SWA layers, `4` on full-attention layers).
    pub n_head_kv: usize,
    /// Sliding window size (`128` on SWA layers, `0` on full-attention layers).
    pub sliding_window: u32,
    /// RoPE base frequency (`10000.0` on SWA layers, `64000000.0` on full-attention layers).
    pub rope_freq_base: f32,
    /// `blk.{l}.attn_norm.weight` (`F32[4096]`).
    pub attn_norm: DevTensor,
    /// `blk.{l}.attn_q.weight` (`[4096, 12288]`).
    pub wq: DevTensor,
    /// `blk.{l}.attn_k.weight` (`[4096, n_head_kv * 192]`).
    pub wk: DevTensor,
    /// `blk.{l}.attn_v.weight` (`[4096, n_head_kv * 128]`).
    pub wv: DevTensor,
    /// Optional fused `blk.{l}.attn_qkv.weight` (when present, `wq`, `wk`, `wv` alias its row ranges).
    pub wqkv: DevTensor,
    /// `blk.{l}.attn_output.weight` (`[8192, 4096]`).
    pub wo: DevTensor,
    /// `blk.{l}.attn_q_norm.weight` (`F32[192]`).
    pub q_norm: DevTensor,
    /// `blk.{l}.attn_k_norm.weight` (`F32[192]`).
    pub k_norm: DevTensor,
    /// `blk.{l}.attn_sinks.weight` (`F32[64]`, present on the 40 SWA layers).
    pub attn_sinks: DevTensor,
    /// `blk.{l}.post_attention_norm.weight` (`F32[4096]`).
    pub post_attn_norm: DevTensor,
    /// `blk.{l}.ffn_norm.weight` (`F32[4096]`).
    pub ffn_norm: DevTensor,
    /// `blk.{l}.post_ffn_norm.weight` (`F32[4096]`).
    pub post_ffn_norm: DevTensor,
    /// Layer 0 dense FFN gate (`blk.0.ffn_gate.weight` `[4096, 16384]`).
    pub ffn_gate: DevTensor,
    /// Layer 0 dense FFN up (`blk.0.ffn_up.weight` `[4096, 16384]`).
    pub ffn_up: DevTensor,
    /// Layer 0 dense FFN down (`blk.0.ffn_down.weight` `[16384, 4096]`).
    pub ffn_down: DevTensor,
    /// Layers 1..47 MoE router gate (`blk.{l}.ffn_gate_inp.weight` `F32[4096, 256]`).
    pub ffn_gate_inp: DevTensor,
    /// Optional Layers 1..47 MoE router bias (`blk.{l}.exp_probs_b.bias` `F32[256]`).
    pub exp_probs_b: DevTensor,
    /// Layers 1..47 MoE gate experts (`blk.{l}.ffn_gate_exps.weight` `[4096, n_ff_exp, 256]`).
    pub exp_gate: DevTensor,
    /// Layers 1..47 MoE up experts (`blk.{l}.ffn_up_exps.weight` `[4096, n_ff_exp, 256]`).
    pub exp_up: DevTensor,
    /// Layers 1..47 MoE down experts (`blk.{l}.ffn_down_exps.weight` `[n_ff_exp, 4096, 256]`).
    pub exp_down: DevTensor,
}

/// Per-layer device weight pointers for a DFlash drafter transformer block (`0..5`).
#[derive(Debug, Clone, Copy)]
pub struct MimoDFlashLayerWeights {
    pub attn_norm: DevTensor,
    pub wq: DevTensor,
    pub wk: DevTensor,
    pub wv: DevTensor,
    pub wo: DevTensor,
    pub q_norm: DevTensor,
    pub k_norm: DevTensor,
    pub attn_sinks: DevTensor,
    pub ffn_norm: DevTensor,
    pub ffn_gate: DevTensor,
    pub ffn_up: DevTensor,
    pub ffn_down: DevTensor,
}

/// Device weights for the 5-layer MiMo DFlash drafter (`MiMo-V2-Flash-DFlash-Q8_0.gguf`).
pub struct MimoDFlashWeights {
    /// Validated DFlash catalog.
    pub catalog: MimoDFlashCatalog,
    /// Full `fc.weight` (`Q8_0 [n_taps * 4096, 4096]`) on device.
    pub fc_weight: DevTensor,
    /// Five `[4096, 4096]` `Q8_0` tap slices of `fc.weight` for GPU fusion (`tk_q36_dflash_sum5_rms`).
    pub fc_slices: Vec<DevTensor>,
    /// `enc.output_norm.weight` (`F32[4096]`).
    pub enc_output_norm: DevTensor,
    /// `output_norm.weight` (`F32[4096]`).
    pub output_norm: DevTensor,
    /// Dedicated `152,064`-row `Q8_0` `output.weight` (`[4096, 152064]`).
    pub output_weight: DevTensor,
    /// Optional drafter `token_embd.weight` (`Q8_0 [4096, n_vocab]`).
    pub tok_embd: DevTensor,
    /// Per-layer DFlash transformer weights (`5` layers).
    pub layers: Vec<MimoDFlashLayerWeights>,
    /// Device pointers owned by the DFlash drafter.
    pub device_allocs: Vec<*mut c_void>,
    /// Total device bytes allocated for the DFlash drafter.
    pub total_device_bytes: usize,
}

/// Loaded MiMo-V2.6-Flash (`mimo2`) model on the GPU with `Budget::for_mimo_11gib` reservation.
pub struct MimoModel {
    /// Validated `mimo2` GGUF catalog.
    pub catalog: MimoCatalog,
    /// Parsed `mimo2` hyperparameters.
    pub hparams: MimoHparams,
    /// `token_embd.weight` (`[4096, n_vocab]`).
    pub tok_embd: DevTensor,
    /// `output_norm.weight` (`F32[4096]`).
    pub output_norm: DevTensor,
    /// `output.weight` (`[4096, n_vocab]`).
    pub output: DevTensor,
    /// Per-layer device weight tables (`0..n_layers`).
    pub layers: Vec<MimoLayerWeights>,
    /// Optional DFlash drafter device weights (`5` layers + `152,064` `Q8_0` `output.weight`).
    pub dflash: Option<MimoDFlashWeights>,
    /// Total bytes uploaded to device across trunk and optional DFlash weights.
    pub total_device_bytes: usize,
    /// Active `BudgetCategory::Weights` reservation on `Budget::for_mimo_11gib`.
    pub weights_reservation: Reservation,
    /// Optional `BudgetCategory::Drafter` reservation on `Budget::for_mimo_11gib`.
    pub dflash_reservation: Option<Reservation>,
    device_allocs: Vec<*mut c_void>,
    hip: Arc<HipRuntime>,
    #[allow(dead_code)]
    lib: Option<Arc<LibTalusKernels>>,
}

// SAFETY: Device pointers in `MimoModel` are immutable after load and owned by this struct.
unsafe impl Send for MimoModel {}
unsafe impl Sync for MimoModel {}
unsafe impl Send for MimoDFlashWeights {}
unsafe impl Sync for MimoDFlashWeights {}

impl Drop for MimoModel {
    fn drop(&mut self) {
        if let Some(ref mut df) = self.dflash {
            for &ptr in &df.device_allocs {
                self.hip.free_device(ptr);
            }
            df.device_allocs.clear();
        }
        for &ptr in &self.device_allocs {
            self.hip.free_device(ptr);
        }
        self.device_allocs.clear();
    }
}

/// Upload context holding open shard files, the 64 MiB pinned staging buffer (`hipHostMalloc`),
/// and the `HipRuntime` handle.
struct ShardUploader<'a> {
    model: &'a GgufModel,
    files: Vec<File>,
    stage: *mut c_void,
    hip: &'a HipRuntime,
    allocs: &'a mut Vec<*mut c_void>,
    bytes_uploaded: usize,
}

impl<'a> ShardUploader<'a> {
    fn new(
        model: &'a GgufModel,
        hip: &'a HipRuntime,
        stage: *mut c_void,
        allocs: &'a mut Vec<*mut c_void>,
    ) -> Result<Self> {
        let mut files = Vec::with_capacity(model.shards().len());
        for shard in model.shards() {
            let sp = shard.path().ok_or_else(|| CoreError::InvalidArgument {
                reason: "in-memory shard has no file path".to_string(),
            })?;
            let f = File::open(sp).map_err(|e| CoreError::IoError {
                message: format!("open {}: {e}", sp.display()),
            })?;
            files.push(f);
        }
        Ok(Self {
            model,
            files,
            stage,
            hip,
            allocs,
            bytes_uploaded: 0,
        })
    }

    fn upload_req(&mut self, name: &str) -> Result<DevTensor> {
        self.upload_opt(name)?
            .ok_or_else(|| CoreError::MissingTensor {
                name: name.to_string(),
            })
    }

    fn upload_opt(&mut self, name: &str) -> Result<Option<DevTensor>> {
        let Some((entry, loc)) = self.model.find_tensor(name) else {
            return Ok(None);
        };
        let shard = &self.model.shards()[loc.shard_idx];
        let file = &self.files[loc.shard_idx];
        let file_offset = shard.data_start() + entry.offset;
        let bytes = entry.byte_size as usize;

        let dev = self.hip.alloc_device(bytes.max(1))?;
        self.allocs.push(dev);

        let mut remaining = bytes;
        let mut cur_off = 0usize;
        while remaining > 0 {
            let chunk = remaining.min(STAGE_BYTES);
            let slice = unsafe { std::slice::from_raw_parts_mut(self.stage as *mut u8, chunk) };
            file.read_exact_at(slice, file_offset + cur_off as u64)
                .map_err(|e| CoreError::IoError {
                    message: format!("pread({name} + {cur_off}): {e}"),
                })?;
            let dst = unsafe { (dev as *mut u8).add(cur_off) as *mut c_void };
            copy_h2d_async(self.hip, dst, self.stage, chunk, ptr::null_mut())?;
            unsafe {
                let _ = posix_fadvise(
                    file.as_raw_fd(),
                    (file_offset + cur_off as u64) as i64,
                    chunk as i64,
                    POSIX_FADV_DONTNEED,
                );
            }
            cur_off += chunk;
            remaining -= chunk;
        }
        self.bytes_uploaded += bytes;

        let n_in = entry.dims[0] as usize;
        let n_out = if entry.n_dims >= 2 {
            entry.dims[1] as usize
        } else {
            1
        };
        let n_exp = if entry.n_dims >= 3 {
            entry.dims[2] as usize
        } else {
            1
        };
        let expert_bytes = if n_exp > 1 { bytes / n_exp } else { 0 };
        Ok(Some(DevTensor {
            dev,
            qtype: to_tk_quant_type(entry.qtype),
            n_in,
            n_out,
            n_exp,
            bytes,
            expert_bytes,
        }))
    }

    fn evict_all_pages(&self) {
        for f in &self.files {
            unsafe {
                let _ = posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
            }
        }
    }
}

/// Calculate the exact byte size of `rows` rows of width `n_in` in `qtype`.
fn row_slice_bytes(qtype: QuantType, n_in: usize, rows: usize, tensor: &str) -> Result<usize> {
    let bytes = compute_tensor_byte_size(qtype, &[n_in as u64, rows as u64], &None, tensor)
        .map_err(|e| CoreError::InvalidArgument {
            reason: e.to_string(),
        })?;
    Ok(bytes as usize)
}

impl MimoModel {
    /// Stream all `mimo2` trunk weights (and optional DFlash drafter weights) from GGUF shards
    /// onto the GPU via a 64 MiB pinned staging buffer (`hipHostMalloc`), reserving
    /// `BudgetCategory::Weights` on `Budget::for_mimo_11gib` strictly BEFORE calling `hipMalloc`.
    pub fn load(
        gguf_path: &Path,
        dflash_path: Option<&Path>,
        lib: Option<Arc<LibTalusKernels>>,
        hip: Arc<HipRuntime>,
        budget: Arc<Budget>,
    ) -> Result<Self> {
        Self::load_with_layers(gguf_path, dflash_path, None, lib, hip, budget)
    }

    /// Stream `mimo2` weights onto the GPU with an optional `max_layers` cap (`--n-layers`).
    pub fn load_with_layers(
        gguf_path: &Path,
        dflash_path: Option<&Path>,
        max_layers: Option<usize>,
        lib: Option<Arc<LibTalusKernels>>,
        hip: Arc<HipRuntime>,
        budget: Arc<Budget>,
    ) -> Result<Self> {
        let catalog = MimoCatalog::open_shards(&[gguf_path])?;
        let gguf = GgufModel::open(gguf_path).map_err(|e| CoreError::InvalidArgument {
            reason: format!("GgufModel::open({}): {e}", gguf_path.display()),
        })?;

        let mut hparams = catalog.hparams.clone();
        if let Some(cap) = max_layers {
            hparams.n_layers = hparams.n_layers.min(cap);
        }
        let n_layers = hparams.n_layers;

        // 1. Compute exact trunk weight bytes to reserve BEFORE any `hipMalloc`.
        let mut trunk_weight_bytes: u64 = 0;
        let mut add_bytes = |name: &str| {
            if let Some((entry, _)) = gguf.find_tensor(name) {
                trunk_weight_bytes = trunk_weight_bytes.saturating_add(entry.byte_size);
            }
        };
        add_bytes("token_embd.weight");
        add_bytes("output_norm.weight");
        add_bytes("output.weight");
        for il in 0..n_layers {
            let p = format!("blk.{il}");
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_qkv.weight",
                "attn_output.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "attn_sinks.weight",
                "post_attention_norm.weight",
                "ffn_norm.weight",
                "post_ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "ffn_gate_inp.weight",
                "exp_probs_b.bias",
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                add_bytes(&format!("{p}.{suffix}"));
            }
        }

        // Compute optional DFlash drafter weight bytes (including the 5 sliced `fc.weight` tap matrices).
        let dflash_meta = if let Some(df_p) = dflash_path {
            let df_cat = MimoDFlashCatalog::open(df_p)?;
            let df_gguf = GgufModel::open(df_p).map_err(|e| CoreError::InvalidArgument {
                reason: format!("GgufModel::open({}): {e}", df_p.display()),
            })?;
            let fc_slice_extra = df_cat.fc_weight.byte_size;
            let df_bytes = df_cat.total_weight_bytes.saturating_add(fc_slice_extra);
            Some((df_cat, df_gguf, df_bytes))
        } else {
            None
        };

        // 2. Reserve `BudgetCategory::Weights` (and optional `BudgetCategory::Drafter`) on
        //    `Budget::for_mimo_11gib` BEFORE calling `hipMalloc`.
        let weights_reservation = budget.try_reserve(
            BudgetCategory::Weights,
            "mimo2.weights",
            trunk_weight_bytes as usize,
        )?;
        let dflash_reservation = if let Some((_, _, df_bytes)) = &dflash_meta {
            Some(budget.try_reserve(
                BudgetCategory::Drafter,
                "mimo2.dflash.weights",
                *df_bytes as usize,
            )?)
        } else {
            None
        };

        // 3. Allocate the 64 MiB pinned staging buffer (`hipHostMalloc`).
        let stage = hip.alloc_pinned(STAGE_BYTES)?;

        struct StageGuard {
            hip: Arc<HipRuntime>,
            ptr: *mut c_void,
        }
        impl Drop for StageGuard {
            fn drop(&mut self) {
                self.hip.free_pinned(self.ptr);
            }
        }
        let _stage_guard = StageGuard {
            hip: Arc::clone(&hip),
            ptr: stage,
        };

        let mut device_allocs = Vec::new();
        let mut up = ShardUploader::new(&gguf, &hip, stage, &mut device_allocs)?;

        let tok_embd = up.upload_req("token_embd.weight")?;
        let output_norm = up.upload_req("output_norm.weight")?;
        let output = if let Some(out) = up.upload_opt("output.weight")? {
            out
        } else {
            tok_embd
        };

        let mut layers = Vec::with_capacity(n_layers);
        for il in 0..n_layers {
            let p = format!("blk.{il}");
            let layer_cat = &catalog.layers[il];
            let is_swa = layer_cat.is_swa();
            let is_moe = layer_cat.is_moe();
            let n_head_q = layer_cat.n_head_q;
            let n_head_kv = layer_cat.n_head_kv;

            let attn_norm = up.upload_req(&format!("{p}.attn_norm.weight"))?;
            let wo = up.upload_req(&format!("{p}.attn_output.weight"))?;
            let q_norm = up
                .upload_opt(&format!("{p}.attn_q_norm.weight"))?
                .unwrap_or_else(DevTensor::empty);
            let k_norm = up
                .upload_opt(&format!("{p}.attn_k_norm.weight"))?
                .unwrap_or_else(DevTensor::empty);
            let attn_sinks = up
                .upload_opt(&format!("{p}.attn_sinks.weight"))?
                .unwrap_or_else(DevTensor::empty);
            let post_attn_norm = up
                .upload_opt(&format!("{p}.post_attention_norm.weight"))?
                .unwrap_or_else(DevTensor::empty);
            let ffn_norm = up.upload_req(&format!("{p}.ffn_norm.weight"))?;
            let post_ffn_norm = up
                .upload_opt(&format!("{p}.post_ffn_norm.weight"))?
                .unwrap_or_else(DevTensor::empty);

            // Support both split (`attn_q/k/v.weight` in `UD-IQ2_XXS`) and fused (`attn_qkv.weight` in `MOPD-a4`).
            let (wq, wk, wv, wqkv) =
                if let Some(wq) = up.upload_opt(&format!("{p}.attn_q.weight"))? {
                    let wk = up.upload_req(&format!("{p}.attn_k.weight"))?;
                    let wv = up.upload_req(&format!("{p}.attn_v.weight"))?;
                    (wq, wk, wv, DevTensor::empty())
                } else {
                    let wqkv = up.upload_req(&format!("{p}.attn_qkv.weight"))?;
                    let (qkv_entry, _) = gguf
                        .find_tensor(&format!("{p}.attn_qkv.weight"))
                        .expect("attn_qkv verified");
                    let q_rows = n_head_q * hparams.head_dim_k;
                    let k_rows = n_head_kv * hparams.head_dim_k;
                    let v_rows = n_head_kv * hparams.head_dim_v;
                    let q_bytes =
                        row_slice_bytes(qkv_entry.qtype, hparams.n_embd, q_rows, &qkv_entry.name)?;
                    let k_bytes =
                        row_slice_bytes(qkv_entry.qtype, hparams.n_embd, k_rows, &qkv_entry.name)?;
                    let v_bytes =
                        row_slice_bytes(qkv_entry.qtype, hparams.n_embd, v_rows, &qkv_entry.name)?;
                    let wq = DevTensor {
                        dev: wqkv.dev,
                        qtype: wqkv.qtype,
                        n_in: hparams.n_embd,
                        n_out: q_rows,
                        n_exp: 1,
                        bytes: q_bytes,
                        expert_bytes: 0,
                    };
                    let wk = DevTensor {
                        dev: unsafe { (wqkv.dev as *mut u8).add(q_bytes) as *mut c_void },
                        qtype: wqkv.qtype,
                        n_in: hparams.n_embd,
                        n_out: k_rows,
                        n_exp: 1,
                        bytes: k_bytes,
                        expert_bytes: 0,
                    };
                    let wv = DevTensor {
                        dev: unsafe { (wqkv.dev as *mut u8).add(q_bytes + k_bytes) as *mut c_void },
                        qtype: wqkv.qtype,
                        n_in: hparams.n_embd,
                        n_out: v_rows,
                        n_exp: 1,
                        bytes: v_bytes,
                        expert_bytes: 0,
                    };
                    (wq, wk, wv, wqkv)
                };

            let (ffn_gate, ffn_up, ffn_down, ffn_gate_inp, exp_probs_b, exp_gate, exp_up, exp_down) =
                if !is_moe {
                    let ffn_gate = up.upload_req(&format!("{p}.ffn_gate.weight"))?;
                    let ffn_up = up.upload_req(&format!("{p}.ffn_up.weight"))?;
                    let ffn_down = up.upload_req(&format!("{p}.ffn_down.weight"))?;
                    (
                        ffn_gate,
                        ffn_up,
                        ffn_down,
                        DevTensor::empty(),
                        DevTensor::empty(),
                        DevTensor::empty(),
                        DevTensor::empty(),
                        DevTensor::empty(),
                    )
                } else {
                    let ffn_gate_inp = up.upload_req(&format!("{p}.ffn_gate_inp.weight"))?;
                    let exp_probs_b = up
                        .upload_opt(&format!("{p}.exp_probs_b.bias"))?
                        .unwrap_or_else(DevTensor::empty);
                    let exp_gate = up.upload_req(&format!("{p}.ffn_gate_exps.weight"))?;
                    let exp_up = up.upload_req(&format!("{p}.ffn_up_exps.weight"))?;
                    let exp_down = up.upload_req(&format!("{p}.ffn_down_exps.weight"))?;
                    (
                        DevTensor::empty(),
                        DevTensor::empty(),
                        DevTensor::empty(),
                        ffn_gate_inp,
                        exp_probs_b,
                        exp_gate,
                        exp_up,
                        exp_down,
                    )
                };

            layers.push(MimoLayerWeights {
                layer_idx: il,
                is_swa,
                is_moe,
                n_head_q,
                n_head_kv,
                sliding_window: layer_cat.sliding_window,
                rope_freq_base: layer_cat.rope_freq_base(),
                attn_norm,
                wq,
                wk,
                wv,
                wqkv,
                wo,
                q_norm,
                k_norm,
                attn_sinks,
                post_attn_norm,
                ffn_norm,
                post_ffn_norm,
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_gate_inp,
                exp_probs_b,
                exp_gate,
                exp_up,
                exp_down,
            });
        }

        up.evict_all_pages();
        let mut total_device_bytes = up.bytes_uploaded;

        // 4. Optional DFlash 5-layer drafter + 152,064 `Q8_0` `output.weight` upload.
        let dflash = if let Some((df_cat, df_gguf, _)) = dflash_meta {
            let mut df_allocs = Vec::new();
            let mut df_up = ShardUploader::new(&df_gguf, &hip, stage, &mut df_allocs)?;

            let fc_weight = df_up.upload_req("fc.weight")?;
            let enc_output_norm = df_up.upload_req("enc.output_norm.weight")?;
            let df_output_norm = df_up.upload_req("output_norm.weight")?;
            let output_weight = df_up.upload_req("output.weight")?;
            let df_tok_embd = df_up.upload_opt("token_embd.weight")?.unwrap_or(tok_embd);

            // Slice `fc.weight` (`Q8_0 [n_taps * 4096, 4096]`) into `n_taps` per-tap
            // `[4096, 4096]` `Q8_0` matrices on device (exact layout used by `tk_q36_dflash_sum5_rms`).
            // Copy `fc_weight.dev` to host once (`~85 MiB`) rather than issuing `n_taps * 4096`
            // uncached `pread` syscalls after `POSIX_FADV_DONTNEED`.
            let n_taps = df_cat.block_count;
            let d = df_cat.embedding_length;
            let row_slice_q8_bytes = (d / 32) * 34;
            let full_row_q8_bytes = n_taps * row_slice_q8_bytes;
            let slice_total_bytes = d * row_slice_q8_bytes;
            let (fc_entry, _) = df_gguf
                .find_tensor("fc.weight")
                .expect("fc.weight verified");

            let mut fc_slices = Vec::with_capacity(n_taps);
            if fc_entry.qtype == QuantType::Q8_0
                && slice_total_bytes <= STAGE_BYTES
                && fc_weight.bytes >= d * full_row_q8_bytes
            {
                let mut fc_host = vec![0u8; fc_weight.bytes];
                hip.copy_d2h(
                    fc_host.as_mut_ptr() as *mut c_void,
                    fc_weight.dev as *const c_void,
                    fc_weight.bytes,
                )?;
                for s in 0..n_taps {
                    let slice_buf = unsafe {
                        std::slice::from_raw_parts_mut(stage as *mut u8, slice_total_bytes)
                    };
                    for r in 0..d {
                        let src_off = r * full_row_q8_bytes + s * row_slice_q8_bytes;
                        let dst_range = r * row_slice_q8_bytes..(r + 1) * row_slice_q8_bytes;
                        slice_buf[dst_range]
                            .copy_from_slice(&fc_host[src_off..src_off + row_slice_q8_bytes]);
                    }
                    let slice_dev = hip.alloc_device(slice_total_bytes)?;
                    df_up.allocs.push(slice_dev);
                    copy_h2d_async(&hip, slice_dev, stage, slice_total_bytes, ptr::null_mut())?;
                    df_up.bytes_uploaded += slice_total_bytes;
                    fc_slices.push(DevTensor {
                        dev: slice_dev,
                        qtype: TK_QTYPE_Q8_0,
                        n_in: d,
                        n_out: d,
                        n_exp: 1,
                        bytes: slice_total_bytes,
                        expert_bytes: 0,
                    });
                }
            }

            let mut df_layers = Vec::with_capacity(df_cat.block_count);
            for il in 0..df_cat.block_count {
                let p = format!("blk.{il}");
                let attn_norm = df_up.upload_req(&format!("{p}.attn_norm.weight"))?;
                let wo = df_up.upload_req(&format!("{p}.attn_output.weight"))?;
                let q_norm = df_up
                    .upload_opt(&format!("{p}.attn_q_norm.weight"))?
                    .unwrap_or_else(DevTensor::empty);
                let k_norm = df_up
                    .upload_opt(&format!("{p}.attn_k_norm.weight"))?
                    .unwrap_or_else(DevTensor::empty);
                let attn_sinks = df_up
                    .upload_opt(&format!("{p}.attn_sinks.weight"))?
                    .unwrap_or_else(DevTensor::empty);
                let ffn_norm = df_up.upload_req(&format!("{p}.ffn_norm.weight"))?;
                let ffn_down = df_up.upload_req(&format!("{p}.ffn_down.weight"))?;

                let (wq, wk, wv) = if let Some(wq) =
                    df_up.upload_opt(&format!("{p}.attn_q.weight"))?
                {
                    let wk = df_up.upload_req(&format!("{p}.attn_k.weight"))?;
                    let wv = df_up.upload_req(&format!("{p}.attn_v.weight"))?;
                    (wq, wk, wv)
                } else {
                    let wqkv = df_up.upload_req(&format!("{p}.attn_qkv.weight"))?;
                    let (qkv_entry, _) = df_gguf
                        .find_tensor(&format!("{p}.attn_qkv.weight"))
                        .expect("dflash attn_qkv verified");
                    let q_rows = df_cat.n_head * df_cat.k_dim;
                    let k_rows = df_cat.n_head_kv * df_cat.k_dim;
                    let v_rows = df_cat.n_head_kv * df_cat.v_dim;
                    let q_bytes = row_slice_bytes(qkv_entry.qtype, d, q_rows, &qkv_entry.name)?;
                    let k_bytes = row_slice_bytes(qkv_entry.qtype, d, k_rows, &qkv_entry.name)?;
                    let v_bytes = row_slice_bytes(qkv_entry.qtype, d, v_rows, &qkv_entry.name)?;
                    let wq = DevTensor {
                        dev: wqkv.dev,
                        qtype: wqkv.qtype,
                        n_in: d,
                        n_out: q_rows,
                        n_exp: 1,
                        bytes: q_bytes,
                        expert_bytes: 0,
                    };
                    let wk = DevTensor {
                        dev: unsafe { (wqkv.dev as *mut u8).add(q_bytes) as *mut c_void },
                        qtype: wqkv.qtype,
                        n_in: d,
                        n_out: k_rows,
                        n_exp: 1,
                        bytes: k_bytes,
                        expert_bytes: 0,
                    };
                    let wv = DevTensor {
                        dev: unsafe { (wqkv.dev as *mut u8).add(q_bytes + k_bytes) as *mut c_void },
                        qtype: wqkv.qtype,
                        n_in: d,
                        n_out: v_rows,
                        n_exp: 1,
                        bytes: v_bytes,
                        expert_bytes: 0,
                    };
                    (wq, wk, wv)
                };

                let (ffn_gate, ffn_up) =
                    if let Some(fg) = df_up.upload_opt(&format!("{p}.ffn_gate.weight"))? {
                        let fu = df_up.upload_req(&format!("{p}.ffn_up.weight"))?;
                        (fg, fu)
                    } else {
                        let fgu = df_up.upload_req(&format!("{p}.ffn_gate_up.weight"))?;
                        let (fgu_entry, _) = df_gguf
                            .find_tensor(&format!("{p}.ffn_gate_up.weight"))
                            .expect("dflash ffn_gate_up verified");
                        let half_rows = df_cat.feed_forward_length;
                        let half_bytes =
                            row_slice_bytes(fgu_entry.qtype, d, half_rows, &fgu_entry.name)?;
                        let fg = DevTensor {
                            dev: fgu.dev,
                            qtype: fgu.qtype,
                            n_in: d,
                            n_out: half_rows,
                            n_exp: 1,
                            bytes: half_bytes,
                            expert_bytes: 0,
                        };
                        let fu = DevTensor {
                            dev: unsafe { (fgu.dev as *mut u8).add(half_bytes) as *mut c_void },
                            qtype: fgu.qtype,
                            n_in: d,
                            n_out: half_rows,
                            n_exp: 1,
                            bytes: half_bytes,
                            expert_bytes: 0,
                        };
                        (fg, fu)
                    };

                df_layers.push(MimoDFlashLayerWeights {
                    attn_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm,
                    k_norm,
                    attn_sinks,
                    ffn_norm,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                });
            }

            df_up.evict_all_pages();
            let df_bytes_uploaded = df_up.bytes_uploaded;
            total_device_bytes += df_bytes_uploaded;
            Some(MimoDFlashWeights {
                catalog: df_cat,
                fc_weight,
                fc_slices,
                enc_output_norm,
                output_norm: df_output_norm,
                output_weight,
                tok_embd: df_tok_embd,
                layers: df_layers,
                device_allocs: df_allocs,
                total_device_bytes: df_bytes_uploaded,
            })
        } else {
            None
        };

        let _ = budget.mark_materialized(weights_reservation.id);
        if let Some(ref dr) = dflash_reservation {
            let _ = budget.mark_materialized(dr.id);
        }

        Ok(Self {
            catalog,
            hparams,
            tok_embd,
            output_norm,
            output,
            layers,
            dflash,
            total_device_bytes,
            weights_reservation,
            dflash_reservation,
            device_allocs,
            hip,
            lib,
        })
    }
}

/// Build the canonical 59-tensor `MiMo-V2-Flash-DFlash-Q8_0.gguf` header-only [`GgufFile`]
/// (`block_count = 5`, `embedding_length = 4096`, `feed_forward_length = 20480`, `n_head = 64`,
/// `n_head_kv = 8`, `k_dim = 192`, `v_dim = 128`, `rope_dim = 64`, `rope_freq_base = 64000000.0`,
/// `block_size = 16`, target layer taps `[1, 12, 24, 36, 47]`, `output.weight` `Q8_0 [4096, 152064]`,
/// `output_norm.weight` `F32[4096]`).
pub fn build_synthetic_mimo_dflash_gguf_header() -> Result<GgufFile> {
    let mut w = GgufHeaderWriter::new();
    w.add_str("general.architecture", DFLASH_ARCH_NAME);
    w.add_u32("dflash.block_count", 5);
    w.add_u32("dflash.embedding_length", N_EMBD as u32);
    w.add_u32("dflash.feed_forward_length", 20480);
    w.add_u32("dflash.attention.head_count", 64);
    w.add_u32("dflash.attention.head_count_kv", 8);
    w.add_u32("dflash.attention.key_length", 192);
    w.add_u32("dflash.attention.value_length", 128);
    w.add_u32("dflash.rope.dimension_count", 64);
    w.add_f32("dflash.rope.freq_base", ROPE_FREQ_BASE_FULL);
    w.add_u32("dflash.dflash.block_size", DFLASH_BLOCK_SIZE);
    w.add_u32_array("dflash.target_layers", &DFLASH_5L_TARGET_TAPS);

    // 4 global tensors: fc.weight, enc.output_norm.weight, output_norm.weight, output.weight
    w.add_tensor(
        "fc.weight",
        QuantType::Q8_0,
        &[(5 * N_EMBD) as u64, N_EMBD as u64],
    );
    w.add_tensor("enc.output_norm.weight", QuantType::F32, &[N_EMBD as u64]);
    w.add_tensor("output_norm.weight", QuantType::F32, &[N_EMBD as u64]);
    w.add_tensor(
        "output.weight",
        QuantType::Q8_0,
        &[N_EMBD as u64, N_VOCAB_UD as u64],
    );

    // 5 layers * 11 tensors per layer = 55 tensors -> 55 + 4 = 59 tensors total!
    for il in 0..5 {
        let prefix = format!("blk.{il}");
        w.add_tensor(
            &format!("{prefix}.attn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_q.weight"),
            QuantType::Q8_0,
            &[N_EMBD as u64, (64 * 192) as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_k.weight"),
            QuantType::Q8_0,
            &[N_EMBD as u64, (8 * 192) as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_v.weight"),
            QuantType::Q8_0,
            &[N_EMBD as u64, (8 * 128) as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_output.weight"),
            QuantType::Q8_0,
            &[(64 * 128) as u64, N_EMBD as u64],
        );
        w.add_tensor(
            &format!("{prefix}.attn_q_norm.weight"),
            QuantType::F32,
            &[192],
        );
        w.add_tensor(
            &format!("{prefix}.attn_k_norm.weight"),
            QuantType::F32,
            &[192],
        );
        w.add_tensor(
            &format!("{prefix}.ffn_norm.weight"),
            QuantType::F32,
            &[N_EMBD as u64],
        );
        w.add_tensor(
            &format!("{prefix}.ffn_gate.weight"),
            QuantType::Q8_0,
            &[N_EMBD as u64, 20480],
        );
        w.add_tensor(
            &format!("{prefix}.ffn_up.weight"),
            QuantType::Q8_0,
            &[N_EMBD as u64, 20480],
        );
        w.add_tensor(
            &format!("{prefix}.ffn_down.weight"),
            QuantType::Q8_0,
            &[20480, N_EMBD as u64],
        );
    }

    w.finish()
}

/// Minimal in-memory GGUF v3 header serializer for header-only catalog tests.
struct GgufHeaderWriter {
    kvs: Vec<Vec<u8>>,
    tensors: Vec<(String, QuantType, Vec<u64>)>,
}

impl GgufHeaderWriter {
    fn new() -> Self {
        Self {
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn add_str(&mut self, key: &str, val: &str) {
        let mut b = Vec::new();
        Self::write_string(&mut b, key);
        b.extend_from_slice(&8u32.to_le_bytes());
        Self::write_string(&mut b, val);
        self.kvs.push(b);
    }

    fn add_u32(&mut self, key: &str, val: u32) {
        let mut b = Vec::new();
        Self::write_string(&mut b, key);
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
        self.kvs.push(b);
    }

    fn add_f32(&mut self, key: &str, val: f32) {
        let mut b = Vec::new();
        Self::write_string(&mut b, key);
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
        self.kvs.push(b);
    }

    fn add_bool(&mut self, key: &str, val: bool) {
        let mut b = Vec::new();
        Self::write_string(&mut b, key);
        b.extend_from_slice(&7u32.to_le_bytes());
        b.push(u8::from(val));
        self.kvs.push(b);
    }

    fn add_u32_array(&mut self, key: &str, vals: &[u32]) {
        let mut b = Vec::new();
        Self::write_string(&mut b, key);
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for &v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        self.kvs.push(b);
    }

    fn add_tensor(&mut self, name: &str, qtype: QuantType, dims: &[u64]) {
        self.tensors.push((name.to_string(), qtype, dims.to_vec()));
    }

    fn finish(self) -> Result<GgufFile> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&talus_gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for kv in &self.kvs {
            buf.extend_from_slice(kv);
        }

        let mut offset = 0u64;
        for (name, qtype, dims) in &self.tensors {
            Self::write_string(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&(qtype.ggml_type_id() as u32).to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            let sz = compute_tensor_byte_size(*qtype, dims, &None, name).map_err(|e| {
                CoreError::InvalidArgument {
                    reason: e.to_string(),
                }
            })?;
            offset = (offset + sz + 31) & !31;
        }

        let data_start = (buf.len() as u64 + 31) & !31;
        let logical_size = data_start + offset;
        GgufFile::from_header_bytes(buf, Some(logical_size)).map_err(|e| {
            CoreError::InvalidArgument {
                reason: e.to_string(),
            }
        })
    }
}

fn expand_sibling_shards<P: AsRef<Path>>(first: &Path, given: &[P]) -> Vec<PathBuf> {
    if given.len() > 1 {
        return given.iter().map(|p| p.as_ref().to_path_buf()).collect();
    }
    let Some(name) = first.file_name().and_then(|s| s.to_str()) else {
        return vec![first.to_path_buf()];
    };
    let Some(parent) = first.parent() else {
        return vec![first.to_path_buf()];
    };
    // Match `*-00001-of-0000N.gguf` or `*-00001-of-0000N.gguf.hdr`.
    if let Some(pos) = name.find("-00001-of-") {
        let prefix = &name[..pos];
        let rest = &name[pos + "-00001-of-".len()..];
        if rest.len() >= 5 {
            if let Ok(total) = rest[..5].parse::<usize>() {
                let suffix = &rest[5..];
                let mut out = Vec::with_capacity(total);
                for idx in 1..=total {
                    let cand = parent.join(format!("{prefix}-{idx:05}-of-{total:05}{suffix}"));
                    if cand.exists() {
                        out.push(cand);
                    } else {
                        return vec![first.to_path_buf()];
                    }
                }
                return out;
            }
        }
    }
    vec![first.to_path_buf()]
}

fn get_meta_usize(gguf: &GgufFile, key: &str) -> Option<usize> {
    gguf.get_metadata(key)
        .and_then(GgufValue::as_u64)
        .map(|v| v as usize)
}

fn get_meta_usize_or_first(gguf: &GgufFile, key: &str) -> Option<usize> {
    let val = gguf.get_metadata(key)?;
    if let Some(u) = val.as_u64() {
        return Some(u as usize);
    }
    val.as_array()?
        .iter()
        .find_map(|x| x.as_u64().filter(|&v| v > 0))
        .map(|v| v as usize)
}

fn get_meta_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    gguf.get_metadata(key)
        .and_then(GgufValue::as_u64)
        .map(|v| v as u32)
}

fn get_meta_f32(gguf: &GgufFile, key: &str) -> Option<f32> {
    gguf.get_metadata(key)
        .and_then(GgufValue::as_f64)
        .map(|v| v as f32)
}

fn check_dims_len(loc: &MimoTensorLoc, n_dims: u32) -> Result<()> {
    if loc.entry.n_dims != n_dims {
        return Err(CoreError::ShapeMismatch {
            tensor: loc.entry.name.clone(),
            expected: vec![n_dims as u64],
            actual: loc.entry.shape().to_vec(),
        });
    }
    Ok(())
}

fn check_exact_shape(loc: &MimoTensorLoc, expected: &[u64]) -> Result<()> {
    if loc.entry.shape() != expected {
        return Err(CoreError::ShapeMismatch {
            tensor: loc.entry.name.clone(),
            expected: expected.to_vec(),
            actual: loc.entry.shape().to_vec(),
        });
    }
    Ok(())
}
