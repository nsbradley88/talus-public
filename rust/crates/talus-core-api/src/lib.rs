//! Core traits, model descriptors, and hardware-seam types for the Talus Rust host (Contract v0).
//!
//! This crate defines the versioned interface contract (`CONTRACT_VERSION = 0`) shared by
//! foundation crates (`talus-hip`, `talus-gguf`, `talus-mem`, `talus-kernels`), model adapters
//! (`talus-model`), speculation (`talus-spec`), state/prefix caching (`talus-state`), and the
//! scheduler/server (`talus-sched`, `talus-server`).

use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

/// Crate identifier.
pub const CRATE_NAME: &str = "talus-core-api";

/// Interface contract version. Wave 1 and Wave 2 build against Contract v0.
pub const CONTRACT_VERSION: u32 = 0;

// ============================================================================
// Errors & Result
// ============================================================================

/// Error type returned by core Talus traits and model adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    /// `general.architecture` is not registered in the host.
    UnsupportedArch {
        arch: String,
        supported: &'static [&'static str],
    },
    /// Model architecture is known, but a shape or feature exceeds kernel capabilities.
    CapabilityRefused { arch: String, reason: String },
    /// Malformed or missing GGUF metadata field when constructing a [`ModelDesc`].
    InvalidModelDesc { field: &'static str, reason: String },
    /// Required weight tensor is absent from the GGUF / pack.
    MissingTensor { name: String },
    /// Tensor dimensions do not match the architecture specification.
    ShapeMismatch {
        tensor: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    /// Invalid caller argument (e.g. out-of-range window size or position).
    InvalidArgument { reason: String },
    /// Allocation would violate the host or device memory budget / `MemAvailable` floor.
    OutOfMemory {
        requested_bytes: u64,
        available_bytes: u64,
    },
    /// Error status returned by `libtalus_kernels` (`tk_status`).
    KernelError { code: i32, message: String },
    /// Error status returned by the HIP runtime (`hipError_t`).
    HipError { code: i32, message: String },
    /// File or session I/O failure.
    IoError { message: String },
    /// Invalid sequence or cache state transition.
    StateError { reason: String },
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArch { arch, supported } => {
                write!(
                    f,
                    "unsupported architecture '{arch}' (supported: {})",
                    supported.join(", ")
                )
            }
            Self::CapabilityRefused { arch, reason } => {
                write!(f, "architecture '{arch}' capability refused: {reason}")
            }
            Self::InvalidModelDesc { field, reason } => {
                write!(f, "invalid model descriptor field '{field}': {reason}")
            }
            Self::MissingTensor { name } => write!(f, "missing required tensor '{name}'"),
            Self::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "tensor '{tensor}' shape mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::InvalidArgument { reason } => write!(f, "invalid argument: {reason}"),
            Self::OutOfMemory {
                requested_bytes,
                available_bytes,
            } => write!(
                f,
                "out of memory: requested {requested_bytes} bytes with {available_bytes} bytes available"
            ),
            Self::KernelError { code, message } => {
                write!(f, "kernel error (status {code}): {message}")
            }
            Self::HipError { code, message } => write!(f, "HIP error (code {code}): {message}"),
            Self::IoError { message } => write!(f, "I/O error: {message}"),
            Self::StateError { reason } => write!(f, "state error: {reason}"),
        }
    }
}

impl std::error::Error for CoreError {}

/// Standard result alias for `talus-core-api`.
pub type Result<T> = std::result::Result<T, CoreError>;

// ============================================================================
// Quantization & Layer Kinds
// ============================================================================

/// GGML / Strata tensor quantization and element types.
///
/// Discriminants match `tk_qtype` in `include/talus_kernels.h` and `gguf_reader.hpp`.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

impl QuantType {
    /// Convert a raw GGML / GGUF type ID into a [`QuantType`].
    pub const fn from_ggml_type_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            34 => Some(Self::TQ1_0),
            35 => Some(Self::TQ2_0),
            39 => Some(Self::MXFP4),
            40 => Some(Self::NVFP4),
            41 => Some(Self::Q1_0),
            42 => Some(Self::Q2_0),
            _ => None,
        }
    }

    /// Return the numeric GGML type ID.
    pub const fn ggml_type_id(self) -> u32 {
        self as u32
    }

    /// Canonical GGML type name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
            Self::NVFP4 => "NVFP4",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
        }
    }

    /// Return `(elements_per_block, bytes_per_block)` matching `strata::block_geometry`.
    pub const fn block_geometry(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::I8 => (1, 1),
            Self::Q4_0 | Self::IQ4_NL => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::MXFP4 => (32, 17),
            Self::Q1_0 => (64, 10),
            Self::Q2_0 => (64, 18),
            Self::NVFP4 => (64, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K | Self::IQ3_S => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_XS => (256, 74),
            Self::IQ3_XXS => (256, 98),
            Self::IQ1_S => (256, 50),
            Self::IQ2_S => (256, 82),
            Self::IQ4_XS => (256, 136),
            Self::IQ1_M => (256, 56),
            Self::TQ1_0 => (256, 54),
            Self::TQ2_0 => (256, 66),
        }
    }
}

/// Token-mixer kind for a single transformer block (`strata::core::LayerKind` + GLM extensions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Gated DeltaNet linear attention (recurrent state, no KV cache).
    Gdn,
    /// Key-driven / Kimi Delta Attention linear recurrence (GLM-5.3-Flash).
    Kda,
    /// Query-Sparse Attention with pooled indexer selection (`qwen4exp`).
    Qsa,
    /// Dynamic Sparse Attention with MLA + lightning indexer (`glm5`).
    Dsa,
    /// Full causal GQA attention over the entire context (`qwen35`, `laguna`, `mimo2`).
    Attn,
    /// Sliding-window causal GQA attention over the last `sliding_window` tokens (`laguna`, `mimo2`).
    AttnSwa,
}

impl LayerKind {
    /// Short canonical identifier matching `layer_kind_name` in `model_desc.hpp`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gdn => "gdn",
            Self::Kda => "kda",
            Self::Qsa => "qsa",
            Self::Dsa => "dsa",
            Self::Attn => "attn",
            Self::AttnSwa => "attn_swa",
        }
    }

    /// True if this layer maintains recurrent state (`Gdn` or `Kda`).
    pub const fn is_recurrent(self) -> bool {
        matches!(self, Self::Gdn | Self::Kda)
    }

    /// True if this layer maintains a KV cache (`Qsa`, `Dsa`, `Attn`, or `AttnSwa`).
    pub const fn has_kv_cache(self) -> bool {
        matches!(self, Self::Qsa | Self::Dsa | Self::Attn | Self::AttnSwa)
    }
}

// ============================================================================
// Hardware Seams
// ============================================================================

/// Target GPU architecture.
///
/// Strix Halo (`Gfx1151`) is the only built and tuned target in Wave 0–3, with
/// `Gfx1201` (Radeon AI PRO R9700 / RDNA4) and `Other` carried as zero-cost seams.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuArch {
    /// AMD Ryzen AI Max / Strix Halo (RDNA3.5, `gfx1151`).
    Gfx1151,
    /// AMD Radeon AI PRO R9700 / RX 9070 XT (RDNA4, `gfx1201`).
    Gfx1201,
    /// Any other GCN/CDNA/RDNA architecture string.
    Other(String),
}

impl GpuArch {
    /// Parse a `hipDeviceProp_t::gcnArchName` string (e.g. `"gfx1151"` or `"gfx1151:sramecc-:xnack-"`),
    /// matching `strata::kernels::gfx_arch_is` in `include/strata/kernels/gfx_arch.hpp`.
    pub fn from_gcn_arch_name(gcn: &str) -> Self {
        let base = gcn.split(':').next().unwrap_or("").trim();
        match base {
            "gfx1151" => Self::Gfx1151,
            "gfx1201" => Self::Gfx1201,
            other => Self::Other(other.to_string()),
        }
    }

    /// Canonical short architecture name (`"gfx1151"`, `"gfx1201"`, or the raw string).
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gfx1151 => "gfx1151",
            Self::Gfx1201 => "gfx1201",
            Self::Other(s) => s.as_str(),
        }
    }
}

/// Matrix-core / WMMA instruction layout family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WmmaLayout {
    /// RDNA3 / RDNA3.5 (`gfx1100`..`gfx1151`) wave32 WMMA layout.
    Gfx11,
    /// RDNA4 (`gfx1200`, `gfx1201`) wave32 WMMA layout.
    Gfx12,
    /// No WMMA matrix-core layout available (portable SIMT / rocBLAS fallback).
    None,
}

/// Hardware capabilities and cache hierarchy properties for a [`GpuArch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchTraits {
    /// Hardware wavefront width (32 on RDNA3.5/RDNA4).
    pub wave: u32,
    /// WMMA instruction layout supported by the architecture.
    pub wmma: WmmaLayout,
    /// Native FP8 matrix-core support.
    pub fp8: bool,
    /// Local Data Share (LDS) bytes per workgroup / CU.
    pub lds_bytes: u32,
    /// Memory Attached Last Level (MALL / Infinity Cache) size in bytes.
    pub mall_bytes: u64,
}

impl ArchTraits {
    /// Return hardware traits for `arch`.
    pub fn for_arch(arch: &GpuArch) -> Self {
        match arch {
            GpuArch::Gfx1151 => Self {
                wave: 32,
                wmma: WmmaLayout::Gfx11,
                fp8: false,
                lds_bytes: 64 * 1024,
                mall_bytes: 32 * 1024 * 1024,
            },
            GpuArch::Gfx1201 => Self {
                wave: 32,
                wmma: WmmaLayout::Gfx12,
                fp8: true,
                lds_bytes: 64 * 1024,
                mall_bytes: 64 * 1024 * 1024,
            },
            GpuArch::Other(_) => Self {
                wave: 64,
                wmma: WmmaLayout::None,
                fp8: false,
                lds_bytes: 64 * 1024,
                mall_bytes: 0,
            },
        }
    }
}

/// Logical HIP device identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct DeviceId(pub u32);

/// Classification of tensor buffers for placement and budget accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorClass {
    /// Dense projection or norm weights.
    DenseWeight,
    /// Routed MoE expert weights.
    RoutedExpert,
    /// Shared MoE expert weights.
    SharedExpert,
    /// Token or per-layer (PLE) embedding tables (eligible for zero-copy mmap on unified memory).
    Embedding,
    /// Output LM head / draft head weights.
    OutputHead,
    /// GDN / KDA recurrent state and snapshots.
    RecurrentState,
    /// Attention KV cache and KV stashes.
    KvCache,
    /// Per-step activation and prefill scratch buffers.
    ActivationScratch,
    /// Host-mapped control words (token IDs, positions, round accept counts).
    HostMappedControl,
}

/// Physical backing kind of a memory placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlacementKind {
    /// Device-allocated memory (`hipMalloc`).
    DeviceLocal,
    /// Host-pinned / host-registered memory mapped into the device address space.
    HostMapped,
    /// Zero-copy memory-mapped file region registered for direct reading (e.g. Q8_0 embeddings / PLE).
    MmapFile,
}

/// Concrete placement decision for a tensor or buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Placement {
    pub device: DeviceId,
    pub kind: PlacementKind,
}

/// Hardware seam for buffer placement across unified vs discrete memory.
pub trait MemoryModel: Send + Sync {
    /// Choose the [`Placement`] for a buffer of class `t` and size `bytes`.
    fn place(&self, t: TensorClass, bytes: usize) -> Placement;

    /// Return `true` if the host can directly read/write a buffer with placement `p`.
    fn host_visible(&self, p: &Placement) -> bool;
}

/// Unified memory implementation for Strix Halo (`gfx1151`, 128 GiB shared LPDDR5X).
///
/// Embeddings use `MmapFile` zero-copy; control words use `HostMapped`; weights, KV, and state
/// use `DeviceLocal` out of the single unified budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnifiedMemoryModel {
    pub device: DeviceId,
}

impl UnifiedMemoryModel {
    pub const fn new(device: DeviceId) -> Self {
        Self { device }
    }
}

impl MemoryModel for UnifiedMemoryModel {
    fn place(&self, t: TensorClass, _bytes: usize) -> Placement {
        let kind = match t {
            TensorClass::Embedding => PlacementKind::MmapFile,
            TensorClass::HostMappedControl => PlacementKind::HostMapped,
            _ => PlacementKind::DeviceLocal,
        };
        Placement {
            device: self.device,
            kind,
        }
    }

    fn host_visible(&self, _p: &Placement) -> bool {
        true
    }
}

/// Contiguous layer-range assignment to a single device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stage {
    pub device: DeviceId,
    pub layers: Range<u32>,
}

/// Pipeline stage plan across one or more devices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlan {
    pub stages: Vec<Stage>,
}

impl StagePlan {
    /// Single-device plan placing all `num_layers` (`0..num_layers`) on `DeviceId(0)`.
    pub fn single(num_layers: u32) -> Self {
        Self {
            stages: vec![Stage {
                device: DeviceId(0),
                layers: 0..num_layers,
            }],
        }
    }
}

/// Opaque handle to a HIP stream on a specific device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHandle {
    /// Raw `hipStream_t` pointer value (`0` for default stream or mock).
    pub raw: u64,
    /// Device owning this stream.
    pub device: DeviceId,
}

impl StreamHandle {
    pub const fn null(device: DeviceId) -> Self {
        Self { raw: 0, device }
    }
}

/// Inter-stage activation transport seam.
pub trait StageLink: Send + Sync {
    /// Send `[T, n_embd]` FP32 activations at the end of `stage_idx`.
    fn send(&self, stage_idx: usize, act_ptr: u64, bytes: usize, s: &StreamHandle) -> Result<()>;

    /// Receive `[T, n_embd]` FP32 activations at the start of `stage_idx`.
    fn recv(&self, stage_idx: usize, act_ptr: u64, bytes: usize, s: &StreamHandle) -> Result<()>;
}

/// No-op [`StageLink`] implementation for single-stage execution (`StagePlan::single`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LocalLink;

impl StageLink for LocalLink {
    fn send(
        &self,
        _stage_idx: usize,
        _act_ptr: u64,
        _bytes: usize,
        _s: &StreamHandle,
    ) -> Result<()> {
        Ok(())
    }

    fn recv(
        &self,
        _stage_idx: usize,
        _act_ptr: u64,
        _bytes: usize,
        _s: &StreamHandle,
    ) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// Model Description, Loading & State Layout
// ============================================================================

/// Read-only view of GGUF header metadata and tensor index passed to [`Arch::describe`].
#[derive(Debug, Clone, Default)]
pub struct GgufView {
    /// Value of `general.architecture`.
    pub architecture: String,
    /// Integer/unsigned metadata key-value pairs.
    pub metadata_u64: BTreeMap<String, u64>,
    /// Floating-point metadata key-value pairs.
    pub metadata_f64: BTreeMap<String, f64>,
    /// String metadata key-value pairs.
    pub metadata_str: BTreeMap<String, String>,
    /// Number of tensors indexed across all shards.
    pub tensor_count: usize,
}

/// Source paths for model weights, native packs, MTP runtime packs, and draft vocabularies.
#[derive(Debug, Clone, Default)]
pub struct WeightSource {
    /// Primary GGUF file path (or shard 1).
    pub primary_path: String,
    /// All GGUF shard paths in order.
    pub shard_paths: Vec<String>,
    /// Optional Strata native pack directory (`dense.bin`, `index.txt`, `native_experts.txt`).
    pub pack_dir: Option<String>,
    /// Optional MTP runtime pack or GGUF path.
    pub mtp_rt_path: Option<String>,
    /// Optional draft vocabulary `.bin` file path (e.g. `qwen-dv-65536.bin`, `mimo-dv-65536.bin`).
    pub draft_vocab_path: Option<String>,
}

/// Weight loader interface passed to [`Arch::load`].
///
/// The loader works with [`WeightSource`] and [`Placement`],
/// never exposing raw unplaced host pointers for in-place kernel reads.
pub trait Loader: Send {
    /// Reserve a [`Placement`] for tensor `name` of `class`, `qtype`, and byte length `bytes`.
    fn allocate(
        &mut self,
        name: &str,
        class: TensorClass,
        qtype: QuantType,
        bytes: usize,
    ) -> Result<Placement>;

    /// Upload or map `bytes` into a previously allocated `placement`.
    fn upload(&mut self, name: &str, placement: &Placement, bytes: &[u8]) -> Result<()>;
}

/// Complete architecture-derived model description (Rust counterpart of `strata::core::ModelGeometry`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelDesc {
    /// `general.architecture` (`"qwen35"`, `"qwen35moe"`, `"laguna"`, `"qwen4exp"`, `"mimo2"`, `"glm5"`).
    pub arch: String,
    pub n_embd: u32,
    pub n_layers: u32,
    /// Per-layer token mixer kind (`layer_kinds.len() == n_layers as usize`).
    pub layer_kinds: Vec<LayerKind>,
    /// `full_attention_interval` as declared in GGUF (for snapshot geometry compatibility).
    pub qsa_interval: u32,

    // Recurrent (GDN / KDA) dimensions
    pub ssm_state_size: u32,
    pub ssm_k_heads: u32,
    pub ssm_v_heads: u32,
    pub ssm_d_conv: u32,
    pub ssm_conv_channels: u32,
    pub ssm_value_dim: u32,

    // Attention dimensions
    pub n_head: u32,
    pub n_head_per_layer: Vec<u32>,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub head_dim_v: u32,
    pub n_rot: u32,
    pub sliding_window: u32,
    pub rope_freq_base_swa: f64,
    pub n_rot_swa: u32,
    pub attn_gate_per_head: bool,

    // Sparse attention indexer (QSA / DSA)
    pub idx_q_heads: u32,
    pub idx_key_dim: u32,
    pub idx_top_k: u32,
    pub idx_block: u32,

    // Normalization & RoPE
    pub rms_eps: f64,
    pub rope_freq_base: f64,
    pub rope_sections: [u32; 4],
    pub rope_scaling_type: String,
    pub rope_scaling_factor: f64,
    pub rope_scaling_orig_ctx: f64,

    // Hyper-connections (`hc` / mHC)
    pub hc: u32,
    pub hc_lr: u32,

    // FFN & MoE
    pub ffn_dense: bool,
    pub n_ff_dense: u32,
    pub n_dense_lead: u32,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_ff: u32,
    pub n_ff_shexp: u32,
    pub expert_gating: u32,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f64,

    // Vocabulary, context, PLE, MTP
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub ple_layers: Vec<u32>,
    pub ple_conv_kernel: u32,
    pub image_token_id: Option<u32>,
    pub n_mtp_layers: u32,
}

impl ModelDesc {
    /// Query head count for `layer`.
    pub fn heads_for_layer(&self, layer: usize) -> u32 {
        self.n_head_per_layer
            .get(layer)
            .copied()
            .unwrap_or(self.n_head)
    }

    /// Number of recurrent (`Gdn` or `Kda`) layers in the model.
    pub fn n_recurrent_layers(&self) -> usize {
        self.layer_kinds.iter().filter(|k| k.is_recurrent()).count()
    }

    /// Number of KV-bearing attention layers in the model.
    pub fn n_kv_layers(&self) -> usize {
        self.layer_kinds.iter().filter(|k| k.has_kv_cache()).count()
    }

    /// True if the model uses an attention indexer (`Qsa` or `Dsa`).
    pub const fn has_indexer(&self) -> bool {
        self.idx_q_heads > 0
    }

    /// True if the model uses hyper-connections (`hc > 0`).
    pub const fn has_hyper_connections(&self) -> bool {
        self.hc > 0
    }
}

/// Element storage format for KV cache pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvCacheFormat {
    /// Uncompressed IEEE-754 FP16.
    F16,
    /// 8-bit quantized codes with FP16 group scales (`kv_q8`).
    Q8_0,
    /// 4-bit quantized blocks (`kv_q4`).
    Q4_0,
}

/// KV cache geometry declared by [`ModelAdapter::state_layout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayout {
    pub num_layers: u32,
    pub n_head_kv: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub max_seq_len: u32,
    /// Optional ring-buffer window for `AttnSwa` layers.
    pub ring_window: Option<u32>,
    /// Key padding stride required by flash attention (e.g. 256 keys; padding cells must be zeroed).
    pub kv_pad_stride: u32,
    /// Optional tile-transposed V stride (multiple of 16 positions for WMMA).
    pub vt_tile_stride: Option<u32>,
    pub format: KvCacheFormat,
}

/// Recurrent state geometry (`GDN` / `KDA`) declared by [`ModelAdapter::state_layout`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentLayout {
    pub num_layers: u32,
    pub state_size: u32,
    pub h_k: u32,
    pub h_v: u32,
    pub d_conv: u32,
    pub conv_channels: u32,
}

/// Complete per-sequence state layout required by a model adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateLayout {
    pub kv: KvLayout,
    pub recurrent: Option<RecurrentLayout>,
    /// Bytes per position required by QSA/DSA indexer key pools (`0` if none).
    pub indexer_bytes_per_pos: usize,
    /// Bytes required by PLE token history (`0` if none).
    pub ple_history_bytes: usize,
}

/// Mutable per-sequence execution state handle passed to `prefill`, `window`, and `commit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeqState {
    pub seq_id: u64,
    /// Committed token position count.
    pub pos: u32,
    /// Maximum context capacity allocated for this sequence.
    pub capacity: u32,
    /// Number of uncommitted verify window rows awaiting fused GDN commit (`GdnPend`).
    pub pending_recurrent_rows: u32,
    pub device: DeviceId,
}

impl SeqState {
    pub const fn new(seq_id: u64, capacity: u32, device: DeviceId) -> Self {
        Self {
            seq_id,
            pos: 0,
            capacity,
            pending_recurrent_rows: 0,
            device,
        }
    }
}

/// Which output logit rows `ModelAdapter::prefill` should materialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitRows {
    /// Compute state only; do not run the output LM head.
    None,
    /// Compute logits for the final prompt token only (standard generation).
    Last,
    /// Compute logits for every prompt token (teacher-forced `--dump-logits`).
    All,
    /// Compute logits for the last `n` prompt tokens (`--force-decode`).
    Suffix(u32),
}

/// Input/output descriptors for a `T = 1..Tmax` decode or speculative verify window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowIo {
    /// Host token IDs when running outside a device-round graph (`len == window_len`).
    pub tokens: Vec<u32>,
    /// Starting sequence position `pos0`.
    pub pos0: u32,
    /// Window width `T` (`1..=16`).
    pub window_len: u32,
    /// Fixed device pointer to input token IDs (`const int32_t*`, graph-capturable).
    pub device_tokens_ptr: u64,
    /// Fixed device pointer to starting position (`const int32_t*`, graph-capturable).
    pub device_pos_ptr: u64,
    /// Fixed device pointer to output logits `[window_len, n_vocab]` (`float*`).
    pub device_logits_ptr: u64,
}

/// Tap point in the target model's forward pass read by device drafters (`MTP`, `DFlash`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HiddenTapKind {
    /// Pre-LM-head final normalized hidden state (`MTP`).
    FinalNorm,
    /// Post-attention residual of a specific trunk layer (`DFlash`).
    LayerPostAttn(u32),
    /// Post-FFN residual of a specific trunk layer (`DFlash`).
    LayerPostFfn(u32),
}

/// Hidden-state tap descriptors exposed by [`ModelAdapter::hidden_taps`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HiddenTaps {
    pub taps: Vec<HiddenTapKind>,
    pub n_embd: u32,
    /// Fixed device pointers corresponding 1:1 with `taps`.
    pub device_ptrs: Vec<u64>,
}

/// Key identifying a captured HIP graph (`(kind, window_len, draft_depth, device)`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphKey {
    pub kind: String,
    pub window_len: u32,
    pub draft_depth: u32,
    pub device: DeviceId,
}

/// Builder passed to [`ModelAdapter::capture`] and [`DeviceDrafter::capture_chain`].
#[derive(Debug, Clone)]
pub struct GraphBuilder {
    pub key: GraphKey,
    pub stream: StreamHandle,
    pub recorded_nodes: usize,
}

impl GraphBuilder {
    pub const fn new(key: GraphKey, stream: StreamHandle) -> Self {
        Self {
            key,
            stream,
            recorded_nodes: 0,
        }
    }

    /// Record that a kernel launch was enqueued onto `self.stream` during capture.
    pub fn record_launch(&mut self) {
        self.recorded_nodes += 1;
    }
}

// ============================================================================
// Speculation, Drafters & Prefix Cache
// ============================================================================

/// Subset vocabulary mapping (`draft_id -> target_token_id`) for 64K/106K draft heads.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DraftVocab {
    pub draft_to_target: Vec<u32>,
}

impl DraftVocab {
    pub fn new(draft_to_target: Vec<u32>) -> Self {
        Self { draft_to_target }
    }

    pub fn len(&self) -> usize {
        self.draft_to_target.len()
    }

    pub fn is_empty(&self) -> bool {
        self.draft_to_target.is_empty()
    }

    /// Map a draft-head vocabulary index to the full target vocabulary token ID.
    pub fn map_to_target(&self, draft_id: usize) -> Option<u32> {
        self.draft_to_target.get(draft_id).copied()
    }
}

/// Device and host buffer descriptors for a captured speculative round (`RoundExecutor`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RoundIo {
    /// Draft chain length `D`.
    pub draft_depth: usize,
    /// Fixed device pointer to proposed draft token IDs `[D + 1]`.
    pub device_draft_tokens_ptr: u64,
    /// Fixed device pointer to accepted token count (`int32_t*`).
    pub device_accepted_ptr: u64,
    /// Fixed device pointer to the bonus/resampled token ID (`int32_t*`).
    pub device_bonus_token_ptr: u64,
    /// Fixed device pointer to current sequence position (`int32_t*`).
    pub device_pos_ptr: u64,
    /// Host-proposed draft tokens (entered via mapped memory when host drafter takes precedence).
    pub host_draft_tokens: Vec<u32>,
}

/// Speculative acceptance policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accept {
    /// Exact greedy argmax verification (ties to lowest token ID).
    Greedy,
    /// Exact-in-distribution rejection sampling seeded with `mt19937_64` compatible seed.
    Rejection { seed: u64 },
}

/// Rolling telemetry from completed speculative rounds fed into [`SpecController::plan`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RoundStats {
    pub proposed: u32,
    pub accepted: u32,
    pub round_latency_us: u64,
    pub ema_accept_rate: f32,
    pub host_draft_available: bool,
    pub consecutive_zero_accepts: u32,
}

/// Per-round speculation decision returned by [`SpecController::plan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundPlan {
    /// Number of draft tokens `D` to propose (`0` disables drafting for this step).
    pub draft_depth: u32,
    /// True when the adaptive pause policy suspends the device drafter (`MTP` / `DFlash`).
    pub pause_device_drafter: bool,
    /// True when a host draft (`ngram-mod`, `suffix`, `prompt-lookup`) overrides the device drafter.
    pub prefer_host_draft: bool,
    /// Verify window width `T = D + 1`.
    pub window_len: u32,
}

/// Opaque handle to a stored prefix-cache entry (`PromptCache` / conversation checkpoint).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheHandle(pub u64);

/// Reason a state snapshot was recorded in [`PrefixCache::put`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SnapReason {
    /// Snapshot at the end of prompt evaluation.
    EndOfPrompt,
    /// Snapshot at the end of assistant generation.
    EndOfAnswer,
    /// Snapshot immediately before the last user/tool/system message (message-start pool).
    MessageStart,
}

/// Metadata for a captured KV + recurrent state snapshot stored in [`PrefixCache`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub pos: u32,
    pub tokens_hash: u64,
    pub kv_bytes: usize,
    pub recurrent_bytes: usize,
}

// ============================================================================
// Core Host Traits
// ============================================================================

/// Architecture descriptor and loader trait (`qwen35`, `laguna`, `qwen4exp`, `mimo2`, `glm5`).
pub trait Arch: Sized {
    /// Canonical GGUF `general.architecture` name.
    const NAME: &'static str;

    /// Derive a [`ModelDesc`] from GGUF metadata and validate kernel capabilities.
    fn describe(g: &GgufView) -> Result<ModelDesc>;

    /// Load model weights from `src` using `ld`.
    fn load(d: &ModelDesc, src: &WeightSource, ld: &mut dyn Loader) -> Result<Self>;
}

/// Unified model execution trait implemented by every architecture adapter.
pub trait ModelAdapter: Send {
    /// Model geometry and per-layer kinds.
    fn desc(&self) -> &ModelDesc;

    /// Per-sequence KV, recurrent, indexer, and PLE memory layout.
    fn state_layout(&self) -> StateLayout;

    /// Process prompt token chunk(s) starting at `pos0`, leaving sequence state bitwise identical
    /// to where sequential decode would leave it (preserving chunk rounding geometry).
    fn prefill(
        &mut self,
        st: &mut SeqState,
        ids: &[u32],
        pos0: u32,
        want: LogitRows,
        s: &StreamHandle,
    ) -> Result<()>;

    /// Evaluate a decode or speculative verify window of `T = 1..Tmax` consecutive positions.
    /// Every row must be bitwise identical to a `T = 1` single-token decode step.
    fn window(&mut self, st: &mut SeqState, w: &WindowIo, s: &StreamHandle) -> Result<()>;

    /// Commit the first `n_keep` rows of the last verify window permanently into `st`
    /// (fusing GDN state commit `GdnPend`, indexer tail, and PLE history).
    fn commit(&mut self, st: &mut SeqState, n_keep: u32, s: &StreamHandle) -> Result<()>;

    /// Hidden-state tap buffers read by device drafters (`MTP` / `DFlash`).
    fn hidden_taps(&self) -> HiddenTaps;

    /// Pipeline stage plan across devices; defaults to `StagePlan::single(self.desc().n_layers)`.
    fn stages(&self) -> StagePlan {
        StagePlan::single(self.desc().n_layers)
    }

    /// Record kernel launches for `key` into HIP graph builder `g` (launches only; no allocations).
    fn capture(&mut self, key: GraphKey, g: &mut GraphBuilder) -> Result<()>;
}

/// Host-side speculative drafter (`ngram-mod`, `suffix`, `prompt-lookup`).
pub trait HostDrafter: Send {
    /// Propose up to `max` draft tokens given token history `hist`, appending to `out`.
    fn propose(&mut self, hist: &[u32], max: usize, out: &mut Vec<u32>);

    /// Observe newly committed tokens to update internal n-gram / suffix state.
    fn observe(&mut self, accepted: &[u32]);
}

/// Device-side speculative drafter (`MTP` for 27B/Flash-Next/GLM, `DFlash` for Laguna/MiMo).
pub trait DeviceDrafter: Send {
    /// Seed the drafter state from target `taps` after prefill or non-fused steps.
    fn seed(&mut self, st: &SeqState, taps: &HiddenTaps, s: &StreamHandle) -> Result<()>;

    /// Record a `d`-step device draft chain into the fused round graph `g`.
    fn capture_chain(&mut self, d: usize, io: &RoundIo, g: &mut GraphBuilder) -> Result<()>;

    /// Optional subset draft vocabulary (`64K` / `106K` mapping table).
    fn vocab(&self) -> Option<&DraftVocab>;
}

/// Controller choosing draft depth `D`, adaptive pause (1.2), and host-draft precedence.
pub trait SpecController: Send {
    /// Compute the [`RoundPlan`] for the next decode round from `stats`.
    fn plan(&mut self, stats: &RoundStats) -> RoundPlan;
}

/// Generic prefix cache generalizing `q36::PromptCache` and Strata conversation checkpoints.
pub trait PrefixCache: Send {
    /// Find the longest cached prefix of `ids`, returning `(matched_tokens, handle)`.
    fn longest(&self, ids: &[u32]) -> Option<(u32, CacheHandle)>;

    /// Store a state snapshot at token position `at` for reason `why`.
    fn put(&mut self, at: u32, snap: Snapshot, why: SnapReason);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn crate_identity_and_contract_version() {
        assert_eq!(CRATE_NAME, "talus-core-api");
        assert_eq!(CONTRACT_VERSION, 0);
    }

    #[test]
    fn gpu_arch_parsing_and_traits() {
        let strix_halo = GpuArch::from_gcn_arch_name("gfx1151");
        assert_eq!(strix_halo, GpuArch::Gfx1151);
        assert_eq!(strix_halo.as_str(), "gfx1151");

        let strix_halo_flags = GpuArch::from_gcn_arch_name("gfx1151:sramecc-:xnack-");
        assert_eq!(strix_halo_flags, GpuArch::Gfx1151);

        let r9700 = GpuArch::from_gcn_arch_name("gfx1201:sramecc+:xnack-");
        assert_eq!(r9700, GpuArch::Gfx1201);
        assert_eq!(r9700.as_str(), "gfx1201");

        let other = GpuArch::from_gcn_arch_name("gfx1100:sramecc-");
        assert_eq!(other, GpuArch::Other("gfx1100".to_string()));
        assert_eq!(other.as_str(), "gfx1100");

        let t1151 = ArchTraits::for_arch(&strix_halo);
        assert_eq!(t1151.wave, 32);
        assert_eq!(t1151.wmma, WmmaLayout::Gfx11);
        assert!(!t1151.fp8);
        assert_eq!(t1151.lds_bytes, 65536);
        assert_eq!(t1151.mall_bytes, 32 * 1024 * 1024);

        let t1201 = ArchTraits::for_arch(&r9700);
        assert_eq!(t1201.wave, 32);
        assert_eq!(t1201.wmma, WmmaLayout::Gfx12);
        assert!(t1201.fp8);
        assert_eq!(t1201.mall_bytes, 64 * 1024 * 1024);

        let tother = ArchTraits::for_arch(&other);
        assert_eq!(tother.wave, 64);
        assert_eq!(tother.wmma, WmmaLayout::None);
        assert!(!tother.fp8);
    }

    #[test]
    fn memory_model_stage_plan_and_local_link() {
        let mem = UnifiedMemoryModel::new(DeviceId(0));
        let dense_p = mem.place(TensorClass::DenseWeight, 1024);
        assert_eq!(dense_p.device, DeviceId(0));
        assert_eq!(dense_p.kind, PlacementKind::DeviceLocal);
        assert!(mem.host_visible(&dense_p));

        let embed_p = mem.place(TensorClass::Embedding, 4096);
        assert_eq!(embed_p.kind, PlacementKind::MmapFile);
        assert!(mem.host_visible(&embed_p));

        let ctl_p = mem.place(TensorClass::HostMappedControl, 64);
        assert_eq!(ctl_p.kind, PlacementKind::HostMapped);
        assert!(mem.host_visible(&ctl_p));

        let plan = StagePlan::single(64);
        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.stages[0].device, DeviceId(0));
        assert_eq!(plan.stages[0].layers, 0..64);

        let link = LocalLink;
        let stream = StreamHandle::null(DeviceId(0));
        assert!(link.send(0, 0x1000, 256, &stream).is_ok());
        assert!(link.recv(0, 0x1000, 256, &stream).is_ok());
    }

    #[test]
    fn quant_type_ids_and_geometry() {
        let all = [
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q4_0,
            QuantType::Q4_1,
            QuantType::Q5_0,
            QuantType::Q5_1,
            QuantType::Q8_0,
            QuantType::Q8_1,
            QuantType::Q2_K,
            QuantType::Q3_K,
            QuantType::Q4_K,
            QuantType::Q5_K,
            QuantType::Q6_K,
            QuantType::Q8_K,
            QuantType::IQ2_XXS,
            QuantType::IQ2_XS,
            QuantType::IQ3_XXS,
            QuantType::IQ1_S,
            QuantType::IQ4_NL,
            QuantType::IQ3_S,
            QuantType::IQ2_S,
            QuantType::IQ4_XS,
            QuantType::IQ1_M,
            QuantType::Q2_0,
        ];
        for qt in all {
            let id = qt.ggml_type_id();
            assert_eq!(QuantType::from_ggml_type_id(id), Some(qt));
            let (elems, bytes) = qt.block_geometry();
            assert!(elems > 0);
            assert!(bytes > 0);
            assert!(!qt.name().is_empty());
        }
        assert_eq!(QuantType::from_ggml_type_id(999), None);
    }

    /// Dummy implementations to verify object safety (`dyn Trait`) and default trait methods.
    struct DummyAdapter {
        desc: ModelDesc,
    }

    impl Arch for DummyAdapter {
        const NAME: &'static str = "qwen35";

        fn describe(g: &GgufView) -> Result<ModelDesc> {
            if g.architecture != Self::NAME {
                return Err(CoreError::UnsupportedArch {
                    arch: g.architecture.clone(),
                    supported: &[Self::NAME],
                });
            }
            Ok(ModelDesc {
                arch: g.architecture.clone(),
                n_embd: 5120,
                n_layers: 64,
                layer_kinds: vec![LayerKind::Gdn, LayerKind::Attn],
                ..ModelDesc::default()
            })
        }

        fn load(d: &ModelDesc, _src: &WeightSource, ld: &mut dyn Loader) -> Result<Self> {
            let p = ld.allocate(
                "token_embd.weight",
                TensorClass::Embedding,
                QuantType::Q8_0,
                1024,
            )?;
            ld.upload("token_embd.weight", &p, &[0u8; 34])?;
            Ok(Self { desc: d.clone() })
        }
    }

    impl ModelAdapter for DummyAdapter {
        fn desc(&self) -> &ModelDesc {
            &self.desc
        }

        fn state_layout(&self) -> StateLayout {
            StateLayout {
                kv: KvLayout {
                    num_layers: 16,
                    n_head_kv: 4,
                    head_dim_k: 256,
                    head_dim_v: 256,
                    max_seq_len: 32768,
                    ring_window: None,
                    kv_pad_stride: 256,
                    vt_tile_stride: Some(16),
                    format: KvCacheFormat::F16,
                },
                recurrent: Some(RecurrentLayout {
                    num_layers: 48,
                    state_size: 128,
                    h_k: 16,
                    h_v: 48,
                    d_conv: 4,
                    conv_channels: 10240,
                }),
                indexer_bytes_per_pos: 0,
                ple_history_bytes: 0,
            }
        }

        fn prefill(
            &mut self,
            st: &mut SeqState,
            ids: &[u32],
            pos0: u32,
            _want: LogitRows,
            _s: &StreamHandle,
        ) -> Result<()> {
            st.pos = pos0 + ids.len() as u32;
            Ok(())
        }

        fn window(&mut self, st: &mut SeqState, w: &WindowIo, _s: &StreamHandle) -> Result<()> {
            st.pending_recurrent_rows = w.window_len;
            Ok(())
        }

        fn commit(&mut self, st: &mut SeqState, n_keep: u32, _s: &StreamHandle) -> Result<()> {
            st.pos += n_keep;
            st.pending_recurrent_rows = 0;
            Ok(())
        }

        fn hidden_taps(&self) -> HiddenTaps {
            HiddenTaps {
                taps: vec![HiddenTapKind::FinalNorm],
                n_embd: self.desc.n_embd,
                device_ptrs: vec![0x2000],
            }
        }

        fn capture(&mut self, _key: GraphKey, g: &mut GraphBuilder) -> Result<()> {
            g.record_launch();
            Ok(())
        }
    }

    struct DummyLoader;
    impl Loader for DummyLoader {
        fn allocate(
            &mut self,
            _name: &str,
            class: TensorClass,
            _qtype: QuantType,
            bytes: usize,
        ) -> Result<Placement> {
            Ok(UnifiedMemoryModel::new(DeviceId(0)).place(class, bytes))
        }

        fn upload(&mut self, _name: &str, _placement: &Placement, _bytes: &[u8]) -> Result<()> {
            Ok(())
        }
    }

    struct DummyHostDrafter;
    impl HostDrafter for DummyHostDrafter {
        fn propose(&mut self, _hist: &[u32], max: usize, out: &mut Vec<u32>) {
            out.extend((0..max as u32).take(2));
        }
        fn observe(&mut self, _accepted: &[u32]) {}
    }

    struct DummyDeviceDrafter {
        vocab: DraftVocab,
    }
    impl DeviceDrafter for DummyDeviceDrafter {
        fn seed(&mut self, _st: &SeqState, _taps: &HiddenTaps, _s: &StreamHandle) -> Result<()> {
            Ok(())
        }
        fn capture_chain(&mut self, _d: usize, _io: &RoundIo, g: &mut GraphBuilder) -> Result<()> {
            g.record_launch();
            Ok(())
        }
        fn vocab(&self) -> Option<&DraftVocab> {
            Some(&self.vocab)
        }
    }

    struct DummyController;
    impl SpecController for DummyController {
        fn plan(&mut self, stats: &RoundStats) -> RoundPlan {
            RoundPlan {
                draft_depth: 4,
                pause_device_drafter: stats.consecutive_zero_accepts >= 3,
                prefer_host_draft: stats.host_draft_available,
                window_len: 5,
            }
        }
    }

    struct DummyPrefixCache;
    impl PrefixCache for DummyPrefixCache {
        fn longest(&self, ids: &[u32]) -> Option<(u32, CacheHandle)> {
            if ids.is_empty() {
                None
            } else {
                Some((ids.len() as u32, CacheHandle(1)))
            }
        }
        fn put(&mut self, _at: u32, _snap: Snapshot, _why: SnapReason) {}
    }

    #[test]
    fn trait_object_safety_and_adapter_flow() {
        let view = GgufView {
            architecture: "qwen35".to_string(),
            ..Default::default()
        };
        let desc = DummyAdapter::describe(&view).expect("describe");
        assert_eq!(desc.n_recurrent_layers(), 1);
        assert_eq!(desc.n_kv_layers(), 1);

        let mut loader = DummyLoader;
        let adapter =
            DummyAdapter::load(&desc, &WeightSource::default(), &mut loader).expect("load");
        let mut dyn_adapter: Box<dyn ModelAdapter> = Box::new(adapter);

        assert_eq!(dyn_adapter.stages(), StagePlan::single(64));
        let mut seq = SeqState::new(1, 4096, DeviceId(0));
        let stream = StreamHandle::null(DeviceId(0));
        dyn_adapter
            .prefill(&mut seq, &[10, 20, 30], 0, LogitRows::Last, &stream)
            .expect("prefill");
        assert_eq!(seq.pos, 3);

        let wio = WindowIo {
            tokens: vec![30, 40],
            pos0: 3,
            window_len: 2,
            device_tokens_ptr: 0,
            device_pos_ptr: 0,
            device_logits_ptr: 0,
        };
        dyn_adapter.window(&mut seq, &wio, &stream).expect("window");
        assert_eq!(seq.pending_recurrent_rows, 2);
        dyn_adapter.commit(&mut seq, 1, &stream).expect("commit");
        assert_eq!(seq.pos, 4);
        assert_eq!(seq.pending_recurrent_rows, 0);

        let mut host_drafter: Box<dyn HostDrafter> = Box::new(DummyHostDrafter);
        let mut drafted = Vec::new();
        host_drafter.propose(&[1, 2, 3], 4, &mut drafted);
        assert_eq!(drafted, vec![0, 1]);
        host_drafter.observe(&[0]);

        let mut dev_drafter: Box<dyn DeviceDrafter> = Box::new(DummyDeviceDrafter {
            vocab: DraftVocab::new(vec![100, 200]),
        });
        assert_eq!(dev_drafter.vocab().unwrap().map_to_target(1), Some(200));
        let mut gb = GraphBuilder::new(
            GraphKey {
                kind: "round".to_string(),
                window_len: 5,
                draft_depth: 4,
                device: DeviceId(0),
            },
            stream,
        );
        dev_drafter
            .capture_chain(4, &RoundIo::default(), &mut gb)
            .expect("capture_chain");
        assert_eq!(gb.recorded_nodes, 1);

        let mut ctl: Box<dyn SpecController> = Box::new(DummyController);
        let plan = ctl.plan(&RoundStats::default());
        assert_eq!(plan.draft_depth, 4);
        assert_eq!(plan.window_len, 5);
        assert_eq!(Accept::Greedy, Accept::Greedy);
        assert_ne!(Accept::Greedy, Accept::Rejection { seed: 42 });

        let mut cache: Box<dyn PrefixCache> = Box::new(DummyPrefixCache);
        assert_eq!(cache.longest(&[1, 2]), Some((2, CacheHandle(1))));
        cache.put(
            2,
            Snapshot {
                pos: 2,
                tokens_hash: 0xdeadbeef,
                kv_bytes: 1024,
                recurrent_bytes: 512,
            },
            SnapReason::MessageStart,
        );

        let mem_dyn: &dyn MemoryModel = &UnifiedMemoryModel::default();
        let link_dyn: &dyn StageLink = &LocalLink;
        assert!(mem_dyn.host_visible(&mem_dyn.place(TensorClass::KvCache, 64)));
        assert!(link_dyn.send(0, 0, 0, &stream).is_ok());
    }

    #[test]
    fn talus_kernels_header_compiles_as_c11_and_cpp17() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let header_path = manifest_dir.join("../../../include/talus_kernels.h");
        assert!(
            header_path.is_file(),
            "missing header at {}",
            header_path.display()
        );

        // compiler temp files: $TMPDIR if set, else the workspace's target/test-scratch (never the system /tmp)
        let scratch = std::env::var_os("TMPDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../target/test-scratch"
                ))
            });
        std::fs::create_dir_all(&scratch).expect("create scratch dir");

        let c_status = Command::new("gcc")
            .env("TMPDIR", &scratch)
            .env("TMP", &scratch)
            .env("TEMP", &scratch)
            .args([
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-fsyntax-only",
                "-x",
                "c",
            ])
            .arg(&header_path)
            .status()
            .expect("invoke gcc for C11 check");
        assert!(c_status.success(), "gcc -std=c11 syntax check failed");

        let cpp_status = Command::new("g++")
            .env("TMPDIR", &scratch)
            .env("TMP", &scratch)
            .env("TEMP", &scratch)
            .args([
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                "-fsyntax-only",
                "-x",
                "c++",
            ])
            .arg(&header_path)
            .status()
            .expect("invoke g++ for C++17 check");
        assert!(cpp_status.success(), "g++ -std=c++17 syntax check failed");
    }
}
