//! Laguna-S-2.1 (`laguna`) shape constants, index helpers, GGUF catalog validation,
//! and GPU weight loader (`src/laguna/laguna_model.hpp`).

#![allow(non_upper_case_globals)]

use std::ffi::c_void;
use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Arc;

use talus_core_api::{CoreError, QuantType, Result};
use talus_gguf::{GgufFile, GgufModel, TensorEntry};
use talus_kernels::f16_to_f32;
use talus_kernels_sys::*;
use talus_mem::{Budget, BudgetCategory, ReservationGuard};

use crate::laguna::ffi::{posix_fadvise, pread, HipRuntime, POSIX_FADV_DONTNEED};

// ---------------------------------------------------------------------------
// Shape Constants (`src/laguna/laguna_model.hpp`)
// ---------------------------------------------------------------------------

pub const K_EMBD: usize = 3072;
pub const K_VOCAB: usize = 100352;
pub const K_LAYERS: usize = 48;
pub const K_HEAD_KV: usize = 8;
pub const K_HEAD_DIM: usize = 128;
pub const K_SWA: usize = 512;
pub const K_EXPERTS: usize = 256;
pub const K_TOP_K: usize = 10;
pub const K_FF_EXP: usize = 1024;
pub const K_FF_DENSE: usize = 12288;
pub const K_EPS: f32 = 1e-6_f32;
pub const K_EXP_SCALE: f64 = 2.5_f64;

// C++-matching constant aliases from `laguna_model.hpp`
pub const kEmbd: usize = K_EMBD;
pub const kVocab: usize = K_VOCAB;
pub const kLayers: usize = K_LAYERS;
pub const kHeadKv: usize = K_HEAD_KV;
pub const kHeadDim: usize = K_HEAD_DIM;
pub const kSwa: usize = K_SWA;
pub const kExperts: usize = K_EXPERTS;
pub const kTopK: usize = K_TOP_K;
pub const kFfExp: usize = K_FF_EXP;
pub const kFfDense: usize = K_FF_DENSE;
pub const kEps: f32 = K_EPS;
pub const kExpScale: f64 = K_EXP_SCALE;

/// Pinned staging buffer size for GGUF -> device weight copies (64 MiB).
pub const STAGE_BYTES: usize = 64 << 20;

/// Full-context YaRN layer predicate (`il % 4 == 0`: layers 0, 4, 8, ..., 44).
#[inline]
pub const fn is_full(il: usize) -> bool {
    il.is_multiple_of(4)
}

/// Number of attention query heads for layer `il` (48 for full layers, 72 for SWA layers).
#[inline]
pub const fn n_head(il: usize) -> usize {
    if is_full(il) {
        48
    } else {
        72
    }
}

/// V-transposed tile index (`src/laguna/laguna_kernels.hpp:vt_index`):
/// `[n_kv_head, stride / 16, head_dim (128), 16]` so 16 consecutive positions of one `(kvh, d)` are contiguous.
#[inline]
pub const fn vt_index(kvh: usize, pos: usize, d: usize, stride: usize) -> usize {
    (((kvh * (stride / 16) + pos / 16) * 128 + d) * 16) + (pos & 15)
}

/// V-scale tile-transposed index (`src/laguna/laguna_kernels.hpp:vs_index`):
/// `[n_kv_head, stride / 16, 4 (32-dim blocks), 16]` so 8 consecutive positions of one `(kvh, blk)` sit in a 16-byte `uint4`.
#[inline]
pub const fn vs_index(kvh: usize, pos: usize, blk: usize, stride: usize) -> usize {
    (((kvh * (stride / 16) + pos / 16) * 4 + blk) * 16) + (pos & 15)
}

/// K-scale index (`src/laguna/laguna_kernels.hpp:ks_index`):
/// `[stride, n_kv_head, 4]` (one `f16` scale per 32 dims).
#[inline]
pub const fn ks_index(pos: usize, kvh: usize, blk: usize, n_kv: usize) -> usize {
    (pos * n_kv + kvh) * 4 + blk
}

/// Convert `talus_core_api::QuantType` to Contract v0 `tk_qtype`.
#[inline]
pub const fn to_tk_quant_type(q: QuantType) -> tk_qtype {
    match q {
        QuantType::F32 => TK_QTYPE_F32,
        QuantType::F16 => TK_QTYPE_F16,
        QuantType::Q4_0 => TK_QTYPE_Q4_0,
        QuantType::Q4_1 => TK_QTYPE_Q4_1,
        QuantType::Q5_0 => TK_QTYPE_Q5_0,
        QuantType::Q5_1 => TK_QTYPE_Q5_1,
        QuantType::Q8_0 => TK_QTYPE_Q8_0,
        QuantType::Q8_1 => TK_QTYPE_Q8_1,
        QuantType::Q2_K => TK_QTYPE_Q2_K,
        QuantType::Q3_K => TK_QTYPE_Q3_K,
        QuantType::Q4_K => TK_QTYPE_Q4_K,
        QuantType::Q5_K => TK_QTYPE_Q5_K,
        QuantType::Q6_K => TK_QTYPE_Q6_K,
        QuantType::Q8_K => TK_QTYPE_Q8_K,
        QuantType::IQ2_XXS => TK_QTYPE_IQ2_XXS,
        QuantType::IQ2_XS => TK_QTYPE_IQ2_XS,
        QuantType::IQ3_XXS => TK_QTYPE_IQ3_XXS,
        QuantType::IQ1_S => TK_QTYPE_IQ1_S,
        QuantType::IQ4_NL => TK_QTYPE_IQ4_NL,
        QuantType::IQ3_S => TK_QTYPE_IQ3_S,
        QuantType::IQ2_S => TK_QTYPE_IQ2_S,
        QuantType::IQ4_XS => TK_QTYPE_IQ4_XS,
        QuantType::I8 => TK_QTYPE_I8,
        QuantType::IQ1_M => TK_QTYPE_IQ1_M,
        QuantType::BF16 => TK_QTYPE_BF16,
        QuantType::TQ1_0 => TK_QTYPE_TQ1_0,
        QuantType::TQ2_0 => TK_QTYPE_TQ2_0,
        QuantType::MXFP4 => TK_QTYPE_MXFP4,
        QuantType::NVFP4 => TK_QTYPE_NVFP4,
        QuantType::Q1_0 => TK_QTYPE_Q1_0,
        QuantType::Q2_0 => TK_QTYPE_Q2_0,
    }
}

/// Dequantize a single token embedding row (`K_EMBD = 3072` floats) from host `token_embd.weight`
/// bytes into `dst`, matching `laguna_engine.cpp:176-206` bit-for-bit (`Q8_0`, `F16`, `F32`).
pub fn dequant_embd_row(qtype: QuantType, raw: &[u8], token: usize, dst: &mut [f32]) -> Result<()> {
    assert_eq!(dst.len(), K_EMBD);
    match qtype {
        QuantType::Q8_0 => {
            // BlockQ80: 2-byte f16 scale `d` + 32 int8 quants = 34 bytes per 32 elements.
            let n_blocks = K_EMBD / 32;
            let row_bytes = n_blocks * 34;
            let start = token
                .checked_mul(row_bytes)
                .ok_or_else(|| CoreError::InvalidArgument {
                    reason: format!("overflow computing row offset for token {token}"),
                })?;
            let row =
                raw.get(start..start + row_bytes)
                    .ok_or_else(|| CoreError::InvalidArgument {
                        reason: format!(
                        "token_embd.weight: expected at least {} bytes for token {token}, got {}",
                        start + row_bytes,
                        raw.len()
                    ),
                    })?;
            for b in 0..n_blocks {
                let blk = &row[b * 34..(b + 1) * 34];
                let d_bits = u16::from_le_bytes([blk[0], blk[1]]);
                let s = f16_to_f32(d_bits);
                for j in 0..32 {
                    let q = blk[2 + j] as i8;
                    dst[b * 32 + j] = (q as f32) * s;
                }
            }
            Ok(())
        }
        QuantType::F16 => {
            let row_bytes = K_EMBD * 2;
            let start = token * row_bytes;
            let row =
                raw.get(start..start + row_bytes)
                    .ok_or_else(|| CoreError::InvalidArgument {
                        reason: format!(
                        "token_embd.weight: expected at least {} bytes for token {token}, got {}",
                        start + row_bytes,
                        raw.len()
                    ),
                    })?;
            for (j, out) in dst.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([row[2 * j], row[2 * j + 1]]);
                *out = f16_to_f32(bits);
            }
            Ok(())
        }
        QuantType::F32 => {
            let row_bytes = K_EMBD * 4;
            let start = token * row_bytes;
            let row =
                raw.get(start..start + row_bytes)
                    .ok_or_else(|| CoreError::InvalidArgument {
                        reason: format!(
                        "token_embd.weight: expected at least {} bytes for token {token}, got {}",
                        start + row_bytes,
                        raw.len()
                    ),
                    })?;
            for (j, out) in dst.iter_mut().enumerate() {
                *out = f32::from_le_bytes([
                    row[4 * j],
                    row[4 * j + 1],
                    row[4 * j + 2],
                    row[4 * j + 3],
                ]);
            }
            Ok(())
        }
        other => Err(CoreError::InvalidModelDesc {
            field: "token_embd.weight",
            reason: format!("unsupported token_embd type: {other:?}"),
        }),
    }
}

/// Compute the 64-bit `model_id` fingerprint matching `laguna_engine.cpp:699`:
/// `uint64_t model_id_of(const Model& m) { return (uint64_t) m.dev_bytes ^ ((uint64_t) m.layers.size() << 56); }`
#[inline]
pub fn compute_model_id(dev_bytes: u64, n_layers: usize) -> u64 {
    dev_bytes ^ ((n_layers as u64) << 56)
}

/// CPU-only GGUF header / shape catalog validator for Laguna-S-2.1.
#[derive(Debug, Clone)]
pub struct LagunaCatalog {
    pub name: String,
    pub n_layers: usize,
    pub n_tensors: usize,
    pub total_tensor_bytes: u64,
    pub model_id: u64,
}

impl LagunaCatalog {
    /// Validate a single-shard `GgufFile` (such as shard 1 of 3 `.gguf.hdr`) for Laguna metadata.
    pub fn validate_header_view(view: &GgufFile) -> Result<()> {
        let arch = view.get_str("general.architecture").unwrap_or("");
        if arch != "laguna" {
            return Err(CoreError::UnsupportedArch {
                arch: arch.to_string(),
                supported: &["laguna"],
            });
        }
        if let Some(v) = view.get_u64("laguna.embedding_length") {
            if v as usize != K_EMBD {
                return Err(CoreError::InvalidModelDesc {
                    field: "laguna.embedding_length",
                    reason: format!("expected {K_EMBD}, got {v}"),
                });
            }
        }
        if let Some(v) = view.get_u64("laguna.block_count") {
            if v as usize != K_LAYERS {
                return Err(CoreError::InvalidModelDesc {
                    field: "laguna.block_count",
                    reason: format!("expected {K_LAYERS}, got {v}"),
                });
            }
        }
        if let Some(v) = view.get_u64("laguna.expert_count") {
            if v as usize != K_EXPERTS {
                return Err(CoreError::InvalidModelDesc {
                    field: "laguna.expert_count",
                    reason: format!("expected {K_EXPERTS}, got {v}"),
                });
            }
        }
        if let Some(v) = view.get_u64("laguna.expert_used_count") {
            if v as usize != K_TOP_K {
                return Err(CoreError::InvalidModelDesc {
                    field: "laguna.expert_used_count",
                    reason: format!("expected {K_TOP_K}, got {v}"),
                });
            }
        }
        Ok(())
    }

    /// Validate all tensor shapes for `max_layers` (up to 48) across a `GgufModel` or a list of
    /// `GgufFile` shards (suitable for `.gguf.hdr` header-only tests).
    pub fn validate_views(
        views: &[GgufFile],
        fallback_name: &str,
        max_layers: usize,
    ) -> Result<Self> {
        if views.is_empty() {
            return Err(CoreError::InvalidModelDesc {
                field: "shards",
                reason: "empty GGUF view slice".to_string(),
            });
        }
        Self::validate_header_view(&views[0])?;

        let find_tensor = |name: &str| -> Result<TensorEntry> {
            for v in views {
                if let Some(t) = v.find_tensor(name) {
                    return Ok(t.clone());
                }
            }
            Err(CoreError::MissingTensor {
                name: name.to_string(),
            })
        };

        let check_shape = |t: &TensorEntry, expected: &[u64]| -> Result<()> {
            let actual = t.shape();
            if actual != expected {
                return Err(CoreError::ShapeMismatch {
                    tensor: t.name.clone(),
                    expected: expected.to_vec(),
                    actual: actual.to_vec(),
                });
            }
            Ok(())
        };

        let tok = find_tensor("token_embd.weight")?;
        check_shape(&tok, &[K_EMBD as u64, K_VOCAB as u64])?;
        let out_norm = find_tensor("output_norm.weight")?;
        check_shape(&out_norm, &[K_EMBD as u64])?;
        let out = find_tensor("output.weight")?;
        check_shape(&out, &[K_EMBD as u64, K_VOCAB as u64])?;

        let n_layers = max_layers.min(K_LAYERS);
        for il in 0..n_layers {
            let h = n_head(il) as u64;
            let pfx = format!("blk.{il}.");
            check_shape(
                &find_tensor(&format!("{pfx}attn_norm.weight"))?,
                &[K_EMBD as u64],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}attn_q.weight"))?,
                &[K_EMBD as u64, h * (K_HEAD_DIM as u64)],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}attn_k.weight"))?,
                &[K_EMBD as u64, (K_HEAD_KV * K_HEAD_DIM) as u64],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}attn_v.weight"))?,
                &[K_EMBD as u64, (K_HEAD_KV * K_HEAD_DIM) as u64],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}attn_output.weight"))?,
                &[h * (K_HEAD_DIM as u64), K_EMBD as u64],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}attn_q_norm.weight"))?,
                &[K_HEAD_DIM as u64],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}attn_k_norm.weight"))?,
                &[K_HEAD_DIM as u64],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}attn_gate.weight"))?,
                &[K_EMBD as u64, h],
            )?;
            check_shape(
                &find_tensor(&format!("{pfx}ffn_norm.weight"))?,
                &[K_EMBD as u64],
            )?;

            if il == 0 {
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_gate.weight"))?,
                    &[K_EMBD as u64, K_FF_DENSE as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_up.weight"))?,
                    &[K_EMBD as u64, K_FF_DENSE as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_down.weight"))?,
                    &[K_FF_DENSE as u64, K_EMBD as u64],
                )?;
            } else {
                let r_inp = find_tensor(&format!("{pfx}ffn_gate_inp.weight"))?;
                check_shape(&r_inp, &[K_EMBD as u64, K_EXPERTS as u64])?;
                if r_inp.qtype != QuantType::F32 {
                    return Err(CoreError::InvalidModelDesc {
                        field: "ffn_gate_inp.weight",
                        reason: format!("router must be F32, got {:?}", r_inp.qtype),
                    });
                }
                check_shape(
                    &find_tensor(&format!("{pfx}exp_probs_b.bias"))?,
                    &[K_EXPERTS as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_gate_exps.weight"))?,
                    &[K_EMBD as u64, K_FF_EXP as u64, K_EXPERTS as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_up_exps.weight"))?,
                    &[K_EMBD as u64, K_FF_EXP as u64, K_EXPERTS as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_down_exps.weight"))?,
                    &[K_FF_EXP as u64, K_EMBD as u64, K_EXPERTS as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_gate_shexp.weight"))?,
                    &[K_EMBD as u64, K_FF_EXP as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_up_shexp.weight"))?,
                    &[K_EMBD as u64, K_FF_EXP as u64],
                )?;
                check_shape(
                    &find_tensor(&format!("{pfx}ffn_down_shexp.weight"))?,
                    &[K_FF_EXP as u64, K_EMBD as u64],
                )?;
            }
        }

        let n_tensors: usize = views.iter().map(|v| v.tensors().len()).sum();
        let total_tensor_bytes: u64 = views
            .iter()
            .flat_map(|v| v.tensors().iter())
            .map(|t| t.byte_size)
            .sum();
        let name = views[0]
            .get_str("general.name")
            .unwrap_or(fallback_name)
            .to_string();
        let model_id = compute_model_id(total_tensor_bytes, n_layers);

        Ok(Self {
            name,
            n_layers,
            n_tensors,
            total_tensor_bytes,
            model_id,
        })
    }

    /// Validate a `GgufModel` (single or split shards).
    pub fn validate_model(
        model: &GgufModel,
        fallback_name: &str,
        max_layers: usize,
    ) -> Result<Self> {
        Self::validate_views(model.shards(), fallback_name, max_layers)
    }
}

/// Device-resident tensor metadata and pointer (`src/laguna/laguna_model.hpp:Tensor`).
#[derive(Debug, Clone)]
pub struct DevTensor {
    pub dev: *mut c_void,
    pub qtype: QuantType,
    pub ne: [i64; 4],
    pub n_dims: i32,
    pub nbytes: usize,
    pub row_bytes: usize,
    pub n_in: i64,
    pub n_out: i64,
}

// SAFETY: DevTensor holds a device memory pointer managed by LagunaModel.
unsafe impl Send for DevTensor {}
unsafe impl Sync for DevTensor {}

impl Default for DevTensor {
    fn default() -> Self {
        Self {
            dev: std::ptr::null_mut(),
            qtype: QuantType::F32,
            ne: [1, 1, 1, 1],
            n_dims: 0,
            nbytes: 0,
            row_bytes: 0,
            n_in: 0,
            n_out: 0,
        }
    }
}

impl DevTensor {
    #[inline]
    pub fn as_f32(&self) -> *const f32 {
        debug_assert_eq!(self.qtype, QuantType::F32);
        self.dev as *const f32
    }

    #[inline]
    pub fn check(&self, name: &str, expected: &[i64]) -> Result<()> {
        let mut ok = self.n_dims as usize == expected.len();
        if ok {
            for (i, &e) in expected.iter().enumerate() {
                if self.ne[i] != e {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            return Err(CoreError::ShapeMismatch {
                tensor: name.to_string(),
                expected: expected.iter().map(|&x| x as u64).collect(),
                actual: self.ne[..self.n_dims as usize]
                    .iter()
                    .map(|&x| x as u64)
                    .collect(),
            });
        }
        Ok(())
    }
}

/// Per-layer device weights (`src/laguna/laguna_model.hpp:LayerWeights`).
#[derive(Debug, Clone, Default)]
pub struct LayerWeights {
    pub attn_norm: DevTensor,
    pub wq: DevTensor,
    pub wk: DevTensor,
    pub wv: DevTensor,
    pub wo: DevTensor,
    pub q_norm: DevTensor,
    pub k_norm: DevTensor,
    pub attn_gate: DevTensor,
    pub ffn_norm: DevTensor,
    // Layer 0: dense FFN
    pub d_gate: DevTensor,
    pub d_up: DevTensor,
    pub d_down: DevTensor,
    // Layers 1..47: MoE + shared expert
    pub router_inp: DevTensor,
    pub router_bias: DevTensor,
    pub bias_host: Vec<f32>,
    pub exp_gate: DevTensor,
    pub exp_up: DevTensor,
    pub exp_down: DevTensor,
    pub sh_gate: DevTensor,
    pub sh_up: DevTensor,
    pub sh_down: DevTensor,
}

/// Loaded Laguna-S-2.1 model with host-mapped `tok_embd` and device-uploaded weights.
pub struct LagunaModel {
    pub gguf: GgufModel,
    pub tok_embd: TensorEntry,
    pub emb: DevTensor,
    pub output_norm: DevTensor,
    pub output: DevTensor,
    pub layers: Vec<LayerWeights>,
    pub model_id: u64,
    pub total_dev_bytes: usize,
    allocs: Vec<*mut c_void>,
    reservations: Vec<ReservationGuard>,
    hip: Arc<HipRuntime>,
}

// SAFETY: LagunaModel owns its device allocations and GgufModel mmaps.
unsafe impl Send for LagunaModel {}
unsafe impl Sync for LagunaModel {}

impl LagunaModel {
    /// Load a Laguna GGUF model (single or split) onto the GPU using a 64 MiB pinned
    /// staging buffer + `pread` + `posix_fadvise(POSIX_FADV_DONTNEED)`.
    pub fn load(
        path: &Path,
        max_layers: i32,
        budget: Option<Arc<Budget>>,
        hip: Arc<HipRuntime>,
    ) -> Result<Self> {
        let gguf = GgufModel::open(path)?;
        let arch = gguf
            .meta_shard()
            .get_str("general.architecture")
            .unwrap_or("");
        if arch != "laguna" {
            return Err(CoreError::UnsupportedArch {
                arch: arch.to_string(),
                supported: &["laguna"],
            });
        }

        let stage = hip.alloc_pinned(STAGE_BYTES)?;
        let mut shard_files: Vec<Option<File>> = Vec::with_capacity(gguf.shard_count());
        let mut shard_fds = Vec::with_capacity(gguf.shard_count());
        for shard in gguf.shards() {
            let file_opt = shard.path().and_then(|sp| File::open(sp).ok());
            let fd = file_opt.as_ref().map(|f| f.as_raw_fd()).unwrap_or(-1);
            shard_files.push(file_opt);
            shard_fds.push(fd);
        }

        let mut allocs: Vec<*mut c_void> = Vec::new();
        let mut reservations: Vec<ReservationGuard> = Vec::new();
        let mut total_dev_bytes: usize = 0;

        let upload_res = (|| -> Result<(
            TensorEntry,
            DevTensor,
            DevTensor,
            DevTensor,
            Vec<LayerWeights>,
        )> {
            let mut get = |name: &str, upload: bool| -> Result<DevTensor> {
                let (info, loc) = gguf.find_tensor(name).ok_or_else(|| {
                    CoreError::MissingTensor {
                        name: name.to_string(),
                    }
                })?;
                let nbytes = info.byte_size as usize;
                let n_in = info.dims[0] as i64;
                let n_out = if info.n_dims > 1 {
                    info.dims[1] as i64
                } else {
                    1
                };
                let (bs, ts) = talus_gguf::block_geometry(info.qtype);
                let row_bytes = (n_in as usize / bs as usize) * ts as usize;
                let mut t = DevTensor {
                    dev: std::ptr::null_mut(),
                    qtype: info.qtype,
                    ne: [
                        info.dims[0] as i64,
                        info.dims[1] as i64,
                        info.dims[2] as i64,
                        info.dims[3] as i64,
                    ],
                    n_dims: info.n_dims as i32,
                    nbytes,
                    row_bytes,
                    n_in,
                    n_out,
                };
                if upload {
                    if let Some(b) = &budget {
                        let guard = b
                            .try_reserve_guard(BudgetCategory::Weights, name, nbytes)
                            .map_err(|_| CoreError::OutOfMemory {
                                requested_bytes: nbytes as u64,
                                available_bytes: b
                                    .try_headroom_bytes()
                                    .unwrap_or(0)
                                    .try_into()
                                    .unwrap_or(0),
                            })?;
                        reservations.push(guard);
                    }
                    let dev = hip.alloc_device(nbytes)?;
                    allocs.push(dev);
                    if let (Some(b), Some(guard)) = (&budget, reservations.last()) {
                        let _ = b.mark_materialized(guard.id());
                    }
                    total_dev_bytes += nbytes;

                    let fd = shard_fds.get(loc.shard_idx).copied().unwrap_or(-1);
                    let shard = &gguf.shards()[loc.shard_idx];
                    let base_off = (shard.data_start() + info.offset) as i64;
                    if fd >= 0 && !stage.is_null() {
                        let mut rem = nbytes;
                        let mut done = 0usize;
                        while rem > 0 {
                            let n = rem.min(STAGE_BYTES);
                            // SAFETY: stage has STAGE_BYTES capacity; fd is valid.
                            let r = unsafe { pread(fd, stage, n, base_off + done as i64) };
                            if r != n as isize {
                                return Err(CoreError::IoError {
                                    message: format!(
                                        "short pread on {name}: got {r}, expected {n}"
                                    ),
                                });
                            }
                            // SAFETY: dev + done is within the nbytes allocation.
                            let dst = unsafe { (dev as *mut u8).add(done) as *mut c_void };
                            hip.copy_h2d(dst, stage, n)?;
                            done += n;
                            rem -= n;
                        }
                        // Drop pages from the OS page cache so host RSS stays low during ~67 GiB upload.
                        unsafe {
                            let _ = posix_fadvise(
                                fd,
                                base_off,
                                nbytes as i64,
                                POSIX_FADV_DONTNEED,
                            );
                        }
                    } else {
                        let host_slice = gguf.tensor_bytes(name)?;
                        hip.copy_h2d(dev, host_slice.as_ptr() as *const c_void, nbytes)?;
                    }
                    t.dev = dev;
                }
                Ok(t)
            };

            let (tok_embd_ref, _) = gguf.find_tensor("token_embd.weight").ok_or_else(|| {
                CoreError::MissingTensor {
                    name: "token_embd.weight".to_string(),
                }
            })?;
            let tok_embd = tok_embd_ref.clone();
            let emb = get("token_embd.weight", true)?;
            emb.check("token_embd", &[K_EMBD as i64, K_VOCAB as i64])?;

            let output_norm = get("output_norm.weight", true)?;
            output_norm.check("output_norm", &[K_EMBD as i64])?;

            let output = get("output.weight", true)?;
            output.check("output", &[K_EMBD as i64, K_VOCAB as i64])?;

            let nl = if max_layers > 0 && (max_layers as usize) < K_LAYERS {
                max_layers as usize
            } else {
                K_LAYERS
            };
            let mut layers = Vec::with_capacity(nl);
            for il in 0..nl {
                let h = n_head(il) as i64;
                let pfx = format!("blk.{il}.");
                let mut lw = LayerWeights {
                    attn_norm: get(&format!("{pfx}attn_norm.weight"), true)?,
                    wq: get(&format!("{pfx}attn_q.weight"), true)?,
                    wk: get(&format!("{pfx}attn_k.weight"), true)?,
                    wv: get(&format!("{pfx}attn_v.weight"), true)?,
                    wo: get(&format!("{pfx}attn_output.weight"), true)?,
                    q_norm: get(&format!("{pfx}attn_q_norm.weight"), true)?,
                    k_norm: get(&format!("{pfx}attn_k_norm.weight"), true)?,
                    attn_gate: get(&format!("{pfx}attn_gate.weight"), true)?,
                    ffn_norm: get(&format!("{pfx}ffn_norm.weight"), true)?,
                    ..Default::default()
                };
                lw.attn_norm.check("attn_norm", &[K_EMBD as i64])?;
                lw.wq.check("wq", &[K_EMBD as i64, h * (K_HEAD_DIM as i64)])?;
                lw.wk
                    .check("wk", &[K_EMBD as i64, (K_HEAD_KV * K_HEAD_DIM) as i64])?;
                lw.wv
                    .check("wv", &[K_EMBD as i64, (K_HEAD_KV * K_HEAD_DIM) as i64])?;
                lw.wo.check("wo", &[h * (K_HEAD_DIM as i64), K_EMBD as i64])?;
                lw.q_norm.check("q_norm", &[K_HEAD_DIM as i64])?;
                lw.k_norm.check("k_norm", &[K_HEAD_DIM as i64])?;
                lw.attn_gate.check("attn_gate", &[K_EMBD as i64, h])?;
                lw.ffn_norm.check("ffn_norm", &[K_EMBD as i64])?;

                if il == 0 {
                    lw.d_gate = get(&format!("{pfx}ffn_gate.weight"), true)?;
                    lw.d_up = get(&format!("{pfx}ffn_up.weight"), true)?;
                    lw.d_down = get(&format!("{pfx}ffn_down.weight"), true)?;
                    lw.d_gate
                        .check("d_gate", &[K_EMBD as i64, K_FF_DENSE as i64])?;
                    lw.d_up.check("d_up", &[K_EMBD as i64, K_FF_DENSE as i64])?;
                    lw.d_down
                        .check("d_down", &[K_FF_DENSE as i64, K_EMBD as i64])?;
                } else {
                    lw.router_inp = get(&format!("{pfx}ffn_gate_inp.weight"), true)?;
                    lw.router_bias = get(&format!("{pfx}exp_probs_b.bias"), true)?;
                    let bias_bytes = gguf.tensor_bytes(&format!("{pfx}exp_probs_b.bias"))?;
                    lw.bias_host = bias_bytes
                        .chunks_exact(4)
                        .take(K_EXPERTS)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    lw.exp_gate = get(&format!("{pfx}ffn_gate_exps.weight"), true)?;
                    lw.exp_up = get(&format!("{pfx}ffn_up_exps.weight"), true)?;
                    lw.exp_down = get(&format!("{pfx}ffn_down_exps.weight"), true)?;
                    lw.sh_gate = get(&format!("{pfx}ffn_gate_shexp.weight"), true)?;
                    lw.sh_up = get(&format!("{pfx}ffn_up_shexp.weight"), true)?;
                    lw.sh_down = get(&format!("{pfx}ffn_down_shexp.weight"), true)?;

                    lw.router_inp
                        .check("router_inp", &[K_EMBD as i64, K_EXPERTS as i64])?;
                    if lw.router_inp.qtype != QuantType::F32 {
                        return Err(CoreError::InvalidModelDesc {
                            field: "ffn_gate_inp.weight",
                            reason: "router must be F32".to_string(),
                        });
                    }
                    lw.router_bias.check("router_bias", &[K_EXPERTS as i64])?;
                    lw.exp_gate.check(
                        "exp_gate",
                        &[K_EMBD as i64, K_FF_EXP as u64 as i64, K_EXPERTS as i64],
                    )?;
                    lw.exp_up.check(
                        "exp_up",
                        &[K_EMBD as i64, K_FF_EXP as i64, K_EXPERTS as i64],
                    )?;
                    lw.exp_down.check(
                        "exp_down",
                        &[K_FF_EXP as i64, K_EMBD as i64, K_EXPERTS as i64],
                    )?;
                    lw.sh_gate
                        .check("sh_gate", &[K_EMBD as i64, K_FF_EXP as i64])?;
                    lw.sh_up.check("sh_up", &[K_EMBD as i64, K_FF_EXP as i64])?;
                    lw.sh_down
                        .check("sh_down", &[K_FF_EXP as i64, K_EMBD as i64])?;
                }
                layers.push(lw);
            }
            Ok((tok_embd, emb, output_norm, output, layers))
        })();

        drop(shard_files);
        hip.free_pinned(stage);

        match upload_res {
            Ok((tok_embd, emb, output_norm, output, layers)) => {
                let model_id = compute_model_id(total_dev_bytes as u64, layers.len());
                Ok(Self {
                    gguf,
                    tok_embd,
                    emb,
                    output_norm,
                    output,
                    layers,
                    model_id,
                    total_dev_bytes,
                    allocs,
                    reservations,
                    hip,
                })
            }
            Err(e) => {
                for p in allocs {
                    hip.free_device(p);
                }
                Err(e)
            }
        }
    }

    /// Host byte slice of `token_embd.weight` from the GGUF memory map.
    #[inline]
    pub fn tok_embd_host_bytes(&self) -> Result<&[u8]> {
        Ok(self.gguf.tensor_bytes("token_embd.weight")?)
    }
}

impl Drop for LagunaModel {
    fn drop(&mut self) {
        for &p in &self.allocs {
            self.hip.free_device(p);
        }
        self.allocs.clear();
        self.reservations.clear();
    }
}
