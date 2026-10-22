//! Architecture-keyed kernel selection (`KernelOp`, `KernelId`, `KernelSet`).
//!
//! `KernelSet::select(arch, op, qtype, ncols)`
//! is keyed by [`GpuArch`]. Only [`GpuArch::Gfx1151`] is populated (with the proven
//! RDNA3.5 wave32 MMVQ rows-per-wave table by `(qtype, ncols)` and attention split-K
//! cell sizes `128` below 4K / `512` at or above 4K); [`GpuArch::Gfx1201`] and
//! [`GpuArch::Other`] return a clean [`CoreError::UnsupportedArch`] error.

use talus_core_api::{CoreError, GpuArch, QuantType, Result};

use crate::{invalid_arg, unsupported_quant};

/// Supported GPU architecture names in Contract v0 (`KernelSet`).
pub const SUPPORTED_ARCHS: &[&str] = &["gfx1151"];

/// Kernel operation family selectable via [`KernelSet::select`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    /// Dense quantized matrix-vector / small-batch projection (`tk_mmvq`).
    Mmvq,
    /// Indirect MoE expert quantized matrix-vector projection (`tk_mmvq_ids`).
    MmvqIds,
    /// Batched prompt GEMM via hipBLASLt / WMMA (`tk_prefill_gemm`).
    PrefillGemm,
    /// Batched prompt quantized MMQ projection (`tk_prefill_mmq`).
    PrefillMmq,
    /// Split-K decode / verify window causal GQA attention (`tk_attn_decode`).
    AttnDecode,
    /// Prompt chunk causal flash attention on WMMA (`tk_attn_prefill_fa`).
    AttnPrefillFa,
    /// Single-token Gated DeltaNet recurrent step (`tk_gdn_step`).
    GdnStep,
    /// Multi-token sequential/chunked Gated DeltaNet recurrence (`tk_gdn_seq`).
    GdnSeq,
    /// Weighted RMSNorm (`tk_rms_norm` / `tk_rms_norm_to`).
    RmsNorm,
    /// Per-head RMSNorm + NEOX RoPE (`tk_head_norm_rope`).
    HeadNormRope,
    /// Laguna prefetching split-K decode/verify attention (`tk_laguna_decode_attn`).
    LagunaDecodeAttn,
    /// Laguna RDNA3/3.5 WMMA prompt/decode causal attention (`tk_laguna_prompt_attn_wmma`).
    LagunaPromptAttnWmma,
    /// Partial NEOX RoPE rotation (`tk_rope_neox`).
    RopeNeox,
    /// In-place SwiGLU elementwise product (`tk_silu_mul`).
    SiluMul,
    /// Scaled vector addition (`tk_axpy`).
    Axpy,
    /// Q8_0 embedding table row gather + dequantization (`tk_embed_q8_0`).
    EmbedQ8_0,
    /// Per-head softplus attention output gate (`tk_softplus_head_gate`).
    SoftplusHeadGate,
    /// Activation quantization from FP32 to Q8_1 (`tk_quantize_q8_1`).
    QuantizeQ8_1,
    /// FP32 matrix-vector product (`tk_f32_matvec`).
    F32Matvec,
    /// Batched FP32 matrix-matrix product (`tk_f32_matmul`).
    F32Matmul,
    /// Sigmoid + bias top-k MoE router (`tk_route_sigmoid_topk`).
    RouteSigmoidTopk,
    /// MoE shared + routed slot residual accumulation (`tk_moe_add_slots`).
    MoeAddSlots,
    /// Row-wise greedy argmax (`tk_argmax_rows` / `tk_argmax_advance`).
    Argmax,
}

impl KernelOp {
    /// Canonical snake_case operation name used in diagnostics and error reporting.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mmvq => "mmvq",
            Self::MmvqIds => "mmvq_ids",
            Self::PrefillGemm => "prefill_gemm",
            Self::PrefillMmq => "prefill_mmq",
            Self::AttnDecode => "attn_decode",
            Self::AttnPrefillFa => "attn_prefill_fa",
            Self::GdnStep => "gdn_step",
            Self::GdnSeq => "gdn_seq",
            Self::RmsNorm => "rms_norm",
            Self::HeadNormRope => "head_norm_rope",
            Self::LagunaDecodeAttn => "laguna_decode_attn",
            Self::LagunaPromptAttnWmma => "laguna_prompt_attn_wmma",
            Self::RopeNeox => "rope_neox",
            Self::SiluMul => "silu_mul",
            Self::Axpy => "axpy",
            Self::EmbedQ8_0 => "embed_q8_0",
            Self::SoftplusHeadGate => "softplus_head_gate",
            Self::QuantizeQ8_1 => "quantize_q8_1",
            Self::F32Matvec => "f32_matvec",
            Self::F32Matmul => "f32_matmul",
            Self::RouteSigmoidTopk => "route_sigmoid_topk",
            Self::MoeAddSlots => "moe_add_slots",
            Self::Argmax => "argmax",
        }
    }
}

/// Resolved kernel dispatch descriptor returned by [`KernelSet::select`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelId {
    /// Target GPU architecture.
    pub arch: GpuArch,
    /// Kernel operation family.
    pub op: KernelOp,
    /// Weight / operand quantization type.
    pub qtype: QuantType,
    /// Number of token columns (`ncols` for projections) or KV sequence length (for decode attention).
    pub ncols: u32,
    /// Output rows processed per workgroup / wave (`ROWS` template parameter for MMVQ, Q tile for FA).
    pub rows_per_wg: u32,
    /// Split-K KV cell chunk size (`128` below 4K, `512` at or above 4K for decode attention; `0` if N/A).
    pub chunk_cells: u32,
    /// Static kernel variant identifier string.
    pub variant: &'static str,
}

/// Architecture-keyed kernel selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSet {
    arch: GpuArch,
}

impl KernelSet {
    /// Construct a [`KernelSet`] for `arch`, verifying that `arch` has a populated kernel table.
    pub fn for_arch(arch: GpuArch) -> Result<Self> {
        ensure_gfx1151(&arch)?;
        Ok(Self { arch })
    }

    /// Return the target [`GpuArch`] for this [`KernelSet`].
    #[inline]
    pub fn arch(&self) -> &GpuArch {
        &self.arch
    }

    /// Select a kernel variant for `(self.arch, op, qtype, ncols)`.
    #[inline]
    pub fn select_op(&self, op: KernelOp, qtype: QuantType, ncols: u32) -> Result<KernelId> {
        Self::select(&self.arch, op, qtype, ncols)
    }

    /// Select the kernel variant for `(arch, op, qtype, ncols)`.
    ///
    /// Only [`GpuArch::Gfx1151`] is populated in Wave 1; [`GpuArch::Gfx1201`] and
    /// [`GpuArch::Other`] return [`CoreError::UnsupportedArch`].
    pub fn select(arch: &GpuArch, op: KernelOp, qtype: QuantType, ncols: u32) -> Result<KernelId> {
        ensure_gfx1151(arch)?;

        if ncols == 0 {
            return Err(invalid_arg(format!(
                "KernelSet::select({}): ncols must be >= 1, got 0",
                op.name()
            )));
        }

        match op {
            KernelOp::Mmvq => select_gfx1151_mmvq(arch, op, qtype, ncols, false),
            KernelOp::MmvqIds => select_gfx1151_mmvq(arch, op, qtype, ncols, true),
            KernelOp::AttnDecode => {
                let chunk_cells = gfx1151_attn_chunk_cells(ncols);
                let variant = if chunk_cells == 128 {
                    "q36_attn_splitk_c128"
                } else {
                    "q36_attn_splitk_c512"
                };
                Ok(KernelId {
                    arch: arch.clone(),
                    op,
                    qtype,
                    ncols,
                    rows_per_wg: 1,
                    chunk_cells,
                    variant,
                })
            }
            KernelOp::LagunaDecodeAttn => {
                let chunk_cells = gfx1151_attn_chunk_cells(ncols);
                let variant = if chunk_cells == 128 {
                    "laguna_decode_attn_splitk_c128"
                } else {
                    "laguna_decode_attn_splitk_c512"
                };
                Ok(KernelId {
                    arch: arch.clone(),
                    op,
                    qtype,
                    ncols,
                    rows_per_wg: 1,
                    chunk_cells,
                    variant,
                })
            }
            KernelOp::AttnPrefillFa => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 64,
                chunk_cells: 64,
                variant: "fa_wmma_gfx1151_q64_kv64",
            }),
            KernelOp::LagunaPromptAttnWmma => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 64,
                chunk_cells: 64,
                variant: "laguna_prompt_attn_wmma_q64_kv64",
            }),
            KernelOp::PrefillGemm => {
                match qtype {
                    QuantType::F16 | QuantType::BF16 | QuantType::F32 => {}
                    _ => {
                        return Err(unsupported_quant(arch.as_str(), op.name(), qtype.name()));
                    }
                }
                let variant = match qtype {
                    QuantType::F16 => "hipblaslt_gemm_f16_gfx1151",
                    QuantType::BF16 => "hipblaslt_gemm_bf16_gfx1151",
                    _ => "hipblaslt_gemm_f32_gfx1151",
                };
                Ok(KernelId {
                    arch: arch.clone(),
                    op,
                    qtype,
                    ncols,
                    rows_per_wg: 1,
                    chunk_cells: 0,
                    variant,
                })
            }
            KernelOp::PrefillMmq => {
                if matches!(qtype, QuantType::Q8_1) {
                    return Err(unsupported_quant(arch.as_str(), op.name(), qtype.name()));
                }
                Ok(KernelId {
                    arch: arch.clone(),
                    op,
                    qtype,
                    ncols,
                    rows_per_wg: 1,
                    chunk_cells: 0,
                    variant: "mmq_int8_wmma_gfx1151",
                })
            }
            KernelOp::GdnStep => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "gdn_step_w32_gfx1151",
            }),
            KernelOp::GdnSeq => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "gdn_seq_w32_gfx1151",
            }),
            KernelOp::RmsNorm => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "rms_norm_w32_gfx1151",
            }),
            KernelOp::HeadNormRope => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "head_norm_rope_w32_gfx1151",
            }),
            KernelOp::RopeNeox => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "rope_neox_gfx1151",
            }),
            KernelOp::SiluMul => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "silu_mul_gfx1151",
            }),
            KernelOp::Axpy => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "axpy_gfx1151",
            }),
            KernelOp::EmbedQ8_0 => {
                if !matches!(
                    qtype,
                    QuantType::Q8_0 | QuantType::F32 | QuantType::F16 | QuantType::BF16
                ) {
                    return Err(unsupported_quant(arch.as_str(), op.name(), qtype.name()));
                }
                Ok(KernelId {
                    arch: arch.clone(),
                    op,
                    qtype,
                    ncols,
                    rows_per_wg: 1,
                    chunk_cells: 0,
                    variant: "embed_q8_0_gfx1151",
                })
            }
            KernelOp::SoftplusHeadGate => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "softplus_head_gate_gfx1151",
            }),
            KernelOp::QuantizeQ8_1 => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "quantize_q8_1_gfx1151",
            }),
            KernelOp::F32Matvec => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "f32_matvec_w32_gfx1151",
            }),
            KernelOp::F32Matmul => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "f32_matmul_w32_gfx1151",
            }),
            KernelOp::RouteSigmoidTopk => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "route_sigmoid_topk_w32_gfx1151",
            }),
            KernelOp::MoeAddSlots => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "moe_add_slots_gfx1151",
            }),
            KernelOp::Argmax => Ok(KernelId {
                arch: arch.clone(),
                op,
                qtype,
                ncols,
                rows_per_wg: 1,
                chunk_cells: 0,
                variant: "argmax_rows_gfx1151",
            }),
        }
    }

    /// Return the proven `gfx1151` wave32 MMVQ rows-per-wave (`w32_rows` with `g_w32_rtab = 1`).
    pub fn mmvq_rows_per_wg(arch: &GpuArch, qtype: QuantType, ncols: u32) -> Result<u32> {
        Ok(Self::select(arch, KernelOp::Mmvq, qtype, ncols)?.rows_per_wg)
    }

    /// Return the static split-K KV cell chunk size (`128` below 4K, `512` at or above 4K).
    pub fn attn_splitk_chunk_cells(arch: &GpuArch, kv_len: u32) -> Result<u32> {
        ensure_gfx1151(arch)?;
        Ok(gfx1151_attn_chunk_cells(kv_len))
    }
}

#[inline]
fn ensure_gfx1151(arch: &GpuArch) -> Result<()> {
    match arch {
        GpuArch::Gfx1151 => Ok(()),
        GpuArch::Gfx1201 | GpuArch::Other(_) => Err(CoreError::UnsupportedArch {
            arch: arch.as_str().to_string(),
            supported: SUPPORTED_ARCHS,
        }),
    }
}

#[inline]
const fn gfx1151_attn_chunk_cells(kv_len: u32) -> u32 {
    if kv_len < 4096 {
        128
    } else {
        512
    }
}

/// Select `gfx1151` wave32 MMVQ / MMVQ-IDs kernel parameters matching `w32_rows` in
/// `src/kernels/cuda/native_mmvq.cu:3437-3453` (`g_w32_rtab = 1`).
fn select_gfx1151_mmvq(
    arch: &GpuArch,
    op: KernelOp,
    qtype: QuantType,
    ncols: u32,
    is_ids: bool,
) -> Result<KernelId> {
    if ncols > 16 {
        return Err(invalid_arg(format!(
            "{}: ncols {} exceeds maximum supported MMVQ batch width 16",
            op.name(),
            ncols
        )));
    }
    if matches!(qtype, QuantType::Q8_1) {
        return Err(unsupported_quant(arch.as_str(), op.name(), qtype.name()));
    }

    let rows_per_wg = if ncols > 8 {
        1
    } else if ncols == 1 {
        match qtype {
            QuantType::Q2_K | QuantType::IQ2_XS | QuantType::IQ2_S => 2,
            QuantType::Q3_K => 4,
            _ => 1,
        }
    } else {
        // 2 <= ncols <= 8
        match qtype {
            QuantType::Q4_K | QuantType::Q5_K | QuantType::Q6_K => 2,
            QuantType::IQ3_S | QuantType::Q3_K | QuantType::Q2_K => 4,
            _ => 1,
        }
    };

    let wide = ncols >= 6;
    let variant = match (is_ids, wide, rows_per_wg) {
        (false, false, 1) => "mmvq_w32_r1",
        (false, false, 2) => "mmvq_w32_r2",
        (false, false, _) => "mmvq_w32_r4",
        (false, true, 1) => "mmvq_w32_wide_r1",
        (false, true, 2) => "mmvq_w32_wide_r2",
        (false, true, _) => "mmvq_w32_wide_r4",
        (true, false, 1) => "mmvq_ids_w32_r1",
        (true, false, 2) => "mmvq_ids_w32_r2",
        (true, false, _) => "mmvq_ids_w32_r4",
        (true, true, 1) => "mmvq_ids_w32_wide_r1",
        (true, true, 2) => "mmvq_ids_w32_wide_r2",
        (true, true, _) => "mmvq_ids_w32_wide_r4",
    };

    Ok(KernelId {
        arch: arch.clone(),
        op,
        qtype,
        ncols,
        rows_per_wg,
        chunk_cells: 0,
        variant,
    })
}
