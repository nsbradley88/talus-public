//! Piece 0 (Skeleton): Flash-Next (`qwen4exp`) pure-Rust engine struct, device buffer layout,
//! KV/GDN/PLE state layout, scratch arenas, HIP streams, per-arch seam (`gfx1151`, `tp_moe` W16 default),
//! and per-layer bitwise dump harness (`TALUS_FN_DUMP=<dir>`).
//!
//! Parallel port branches (`qwen4exp-p1` .. `qwen4exp-p6`) implement the 6 pieces in their dedicated modules:
//! - Piece 1 (`weights.rs`): GGUF / native pack loading, expert arena/cache, resident budget, PLE table.
//! - Piece 2 (`prefill.rs`): batched prefill orchestration (`prefill.cpp`), experts routed through `tp_moe`.
//! - Piece 3 (`decode.rs`): single-token decode layer (QSA attention, GDN, MoE, dense, LM head).
//! - Piece 4 (`verify_mtp.rs`): multi-token speculative `Verifier` + `MtpDrafter`.
//! - Piece 5 (`conv_state.rs`): multi-turn conversation state, checkpoints, snapshots, disk eviction.
//! - Piece 6 (`kld_flags.rs`): `--logits-out` / `--kld-in` teacher-forced KLD (first-token KLD) & niche flag policy.

use core::ffi::c_void;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::qwen4exp::conv_state::Piece5ConvState;
use crate::qwen4exp::decode::Piece3Decode;
use crate::qwen4exp::ffi::Qwen4ExpKernels;
use crate::qwen4exp::kld_flags::Piece6KldFlags;
use crate::qwen4exp::prefill::Piece2Prefill;
use crate::qwen4exp::verify_mtp::Piece4VerifyMtp;
use crate::qwen4exp::weights::Piece1Weights;

/// Magic header (`"TALUSFND"`) for per-layer binary dump files written under `TALUS_FN_DUMP=<dir>`.
pub const LAYER_DUMP_MAGIC: u64 = 0x5441_4C55_5346_4E44;

/// Compiled & runtime geometry for Qwen4-Exp / Flash-Next (`qwen4exp`, 48 layers: 36 GDN + 12 QSA).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpModelConfig {
    pub n_embd: i64,
    pub n_layers: i64,
    pub qsa_interval: i64,
    pub n_vocab: i64,
    pub rms_eps: f32,
    pub ssm_state_size: i64,
    pub ssm_k_heads: i64,
    pub ssm_v_heads: i64,
    pub ssm_d_conv: i64,
    pub ssm_conv_channels: i64,
    pub ssm_value_dim: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub head_dim: i64,
    pub n_rot: i64,
    pub idx_q_heads: i64,
    pub idx_key_dim: i64,
    pub idx_block: i64,
    pub idx_top_k: i64,
    pub hc: i64,
    pub hc_lr: i64,
    pub n_expert: i64,
    pub n_expert_used: i64,
    pub n_ff: i64,
    pub n_ff_shexp: i64,
    pub max_context: i64,
}

impl Default for Qwen4ExpModelConfig {
    fn default() -> Self {
        Self {
            n_embd: 2560,
            n_layers: 48,
            qsa_interval: 4,
            n_vocab: 248320,
            rms_eps: 1e-6,
            ssm_state_size: 128,
            ssm_k_heads: 16,
            ssm_v_heads: 48,
            ssm_d_conv: 4,
            ssm_conv_channels: 10240,
            ssm_value_dim: 6144,
            n_head: 24,
            n_head_kv: 2,
            head_dim: 256,
            n_rot: 64,
            idx_q_heads: 4,
            idx_key_dim: 128,
            idx_block: 4,
            idx_top_k: 2048,
            hc: 4,
            hc_lr: 320,
            n_expert: 512,
            n_expert_used: 10,
            n_ff: 640,
            n_ff_shexp: 640,
            max_context: 131072,
        }
    }
}

impl Qwen4ExpModelConfig {
    /// Returns true if layer `l` (`0..48`) is a QSA sparse-attention layer (`(l + 1) % 4 == 0`).
    #[inline]
    pub const fn is_qsa_layer(&self, l: i64) -> bool {
        self.qsa_interval > 0 && ((l + 1) % self.qsa_interval) == 0
    }

    /// Number of QSA sparse-attention layers (12 for 48-layer Flash-Next).
    #[inline]
    pub const fn n_qsa_layers(&self) -> usize {
        if self.qsa_interval <= 0 {
            0
        } else {
            (self.n_layers / self.qsa_interval) as usize
        }
    }

    /// Number of GDN recurrent layers (36 for 48-layer Flash-Next).
    #[inline]
    pub const fn n_gdn_layers(&self) -> usize {
        (self.n_layers as usize).saturating_sub(self.n_qsa_layers())
    }

    /// Ordinal `[0..12)` of a QSA layer `l`, or `None` if `l` is a GDN layer.
    #[inline]
    pub const fn qsa_ord(&self, l: i64) -> Option<usize> {
        if self.is_qsa_layer(l) {
            Some(((l + 1) / self.qsa_interval - 1) as usize)
        } else {
            None
        }
    }

    /// Ordinal `[0..36)` of a GDN layer `l`, or `None` if `l` is a QSA layer.
    #[inline]
    pub const fn gdn_ord(&self, l: i64) -> Option<usize> {
        if self.is_qsa_layer(l) || l < 0 || l >= self.n_layers {
            None
        } else {
            let qsa_before = if self.qsa_interval > 0 {
                (l + 1) / self.qsa_interval
            } else {
                0
            };
            Some((l - qsa_before) as usize)
        }
    }

    /// Hyper-connection residual width (`hc * n_embd = 4 * 2560 = 10240` floats).
    #[inline]
    pub const fn residual_dim(&self) -> usize {
        (self.hc * self.n_embd) as usize
    }
}

/// Per-architecture hardware & kernel seam for Flash-Next (`qwen4exp`) on Strix Halo (`gfx1151`).
#[derive(Clone, Debug)]
pub struct Qwen4ExpArchSeam {
    pub arch_name: &'static str,
    pub gfx1151_wave32: bool,
    /// `STRATA_PF_W16`: fp16-operand expert GEMM in `moe_w16.cu` (default `true`,
    /// kill switch `STRATA_PF_W16=0`).
    pub pf_w16_enabled: bool,
    /// `STRATA_PF_SWITCH_MIN_T`: rounding-level prefill switches threshold (default `0`).
    pub pf_switch_min_t: i64,
    /// `TALUS_PF_MOE`: route prefill MMQ expert products through `tp_moe` (`moe_gemm`, default `true`).
    pub tp_moe_enabled: bool,
    /// `TP_MOE_VER`: `tp_moe` kernel version (`72` default on `gfx1151`).
    pub tp_moe_ver: i32,
    /// `TALUS_FN_RUST`: whether the pure-Rust data plane is active (default `true` unless `TALUS_FN_RUST=0`).
    pub rust_data_plane: bool,
}

impl Default for Qwen4ExpArchSeam {
    fn default() -> Self {
        Self::from_env()
    }
}

impl Qwen4ExpArchSeam {
    pub fn from_env() -> Self {
        let pf_w16_enabled = std::env::var("STRATA_PF_W16")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        let pf_switch_min_t = std::env::var("STRATA_PF_SWITCH_MIN_T")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok())
            .unwrap_or(0);
        let tp_moe_enabled = std::env::var("TALUS_PF_MOE")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true);
        let tp_moe_ver = std::env::var("TP_MOE_VER")
            .ok()
            .and_then(|v| v.trim().parse::<i32>().ok())
            .unwrap_or(72);
        let rust_data_plane = std::env::var("TALUS_FN_RUST")
            .ok()
            .map(|v| {
                let t = v.trim();
                t.is_empty() || (t != "0" && !t.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(true);
        Self {
            arch_name: "qwen4exp",
            gfx1151_wave32: true,
            pf_w16_enabled,
            pf_switch_min_t,
            tp_moe_enabled,
            tp_moe_ver,
            rust_data_plane,
        }
    }
}

/// Device buffers for a GDN recurrent mixer step (`include/strata/core/gdn.hpp`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GdnDeviceBuffers {
    pub x_norm: *mut f32,
    pub qkv: *mut f32,
    pub z: *mut f32,
    pub b_raw: *mut f32,
    pub a_raw: *mut f32,
    pub conv_out: *mut f32,
    pub y: *mut f32,
    pub y_norm: *mut f32,
    pub gate: *mut f32,
    pub ssm_out: *mut f32,
}

/// Device buffers for a QSA sparse-attention mixer step (`include/strata/core/qsa.hpp`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QsaDeviceBuffers {
    pub x_norm: *mut f32,
    pub q_full: *mut f32,
    pub q: *mut f32,
    pub q_gate: *mut f32,
    pub k: *mut f32,
    pub v: *mut f32,
    pub idx_q: *mut f32,
    pub idx_k: *mut f32,
    pub idx_scores: *mut f32,
    pub idx_topk: *mut i32,
    pub attn_out: *mut f32,
    pub y: *mut f32,
    pub cos_sin: *mut f32,
}

/// Device buffers for a routed + shared MoE block (`include/strata/core/moe.hpp`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeDeviceBuffers {
    pub x_norm: *mut f32,
    pub logits: *mut f32,
    pub top_idx: *mut i32,
    pub top_w: *mut f32,
    pub shexp_gu: *mut f32,
    pub shexp_h: *mut f32,
    pub shexp_gate: *mut f32,
    pub accum: *mut f32,
    pub y: *mut f32,
}

/// Device buffers for hyper-connection (`hc = 4`) stream mixing and residual writes (`include/strata/core/layer.hpp`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BlockDeviceBuffers {
    pub mixed: *mut f32,
    pub delta: *mut f32,
    pub post_norm: *mut f32,
    pub hc_pre: *mut f32,
    pub hc_post: *mut f32,
    pub hc_h: *mut f32,
    pub r_next: *mut f32,
    pub ple_proj: *mut f32,
}

/// Top-level device buffer pointers owned/carved by the `qwen4exp` engine.
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen4ExpDeviceBuffers {
    /// Active multi-stream residual `R` (`[hc, n_embd]` = `10240` floats for decode, `[T, hc * n_embd]` in prefill).
    pub r_dev: *mut f32,
    pub d_emb: *mut f32,
    pub d_logits: *mut f32,
    pub d_next: *mut i32,
    pub d_parts: *mut f32,
    pub sbuf: *mut c_void,
    pub gdn: GdnDeviceBuffers,
    pub qsa: QsaDeviceBuffers,
    pub moe: MoeDeviceBuffers,
    pub block: BlockDeviceBuffers,
}

unsafe impl Send for Qwen4ExpDeviceBuffers {}
unsafe impl Sync for Qwen4ExpDeviceBuffers {}

/// Per-layer GDN recurrent state in VRAM (`s: [48, 128, 128]` f32 = 3 MiB, `conv: [3, 10240]` f32 = 120 KiB).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GdnLayerState {
    pub s: *mut f32,
    pub conv: *mut f32,
    pub s_bytes: usize,
    pub conv_bytes: usize,
}

/// Per-layer QSA sparse-attention KV & indexer state in VRAM (`12` layers).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct QsaLayerState {
    pub k_q8: *mut i8,
    pub k_sc: *mut u16,
    pub v: *mut u16,
    pub idx_k: *mut f32,
    pub idx_pooled: *mut f32,
    pub idx_dead: *mut f32,
    pub block_pos: *mut i32,
    pub map: *mut c_void,
    pub capacity_tokens: i64,
    pub resident_cells: i64,
    pub kv_q4: bool,
    pub kv_k8v4: bool,
    pub kv_rot: bool,
    pub kv_hybrid: bool,
}

/// Per-layer-1 PLE (Per-Layer Embedding) history & staging state.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PleStateLayout {
    pub hist_dev: *mut f32,
    pub scratch_dev: *mut f32,
    pub norm_dev: *mut f32,
    pub ple_prev: [i32; 2],
    pub ple_token: i32,
    pub ready: bool,
}

/// Complete KV + GDN + PLE state layout across all 48 layers (`36` GDN + `12` QSA + PLE).
#[derive(Clone, Debug)]
pub struct Qwen4ExpKvGdnState {
    pub gdn_states: Vec<GdnLayerState>,
    pub qsa_states: Vec<QsaLayerState>,
    pub ple: PleStateLayout,
    pub gdn_total_bytes: usize,
    pub ple_total_bytes: usize,
    pub qsa_tail_bytes_per_layer: usize,
}

impl Default for Qwen4ExpKvGdnState {
    fn default() -> Self {
        Self::new(&Qwen4ExpModelConfig::default())
    }
}

impl Qwen4ExpKvGdnState {
    pub fn new(cfg: &Qwen4ExpModelConfig) -> Self {
        let s_bytes = (cfg.ssm_v_heads * cfg.ssm_state_size * cfg.ssm_state_size) as usize * 4;
        let conv_bytes = ((cfg.ssm_d_conv - 1) * cfg.ssm_conv_channels) as usize * 4;
        let n_gdn = cfg.n_gdn_layers();
        let n_qsa = cfg.n_qsa_layers();
        let gdn_states = vec![
            GdnLayerState {
                s: std::ptr::null_mut(),
                conv: std::ptr::null_mut(),
                s_bytes,
                conv_bytes,
            };
            n_gdn
        ];
        let qsa_states = vec![QsaLayerState::default(); n_qsa];
        Self {
            gdn_states,
            qsa_states,
            ple: PleStateLayout {
                ple_prev: [-1, -1],
                ..Default::default()
            },
            gdn_total_bytes: n_gdn * (s_bytes + conv_bytes),
            ple_total_bytes: 2 * (cfg.n_embd as usize) * 4,
            qsa_tail_bytes_per_layer: 0,
        }
    }
}

unsafe impl Send for Qwen4ExpKvGdnState {}
unsafe impl Sync for Qwen4ExpKvGdnState {}

/// Device & borrowed expert-cache scratch arena used by prefill, decode, and verify/MTP.
#[derive(Clone, Debug, Default)]
pub struct Qwen4ExpScratchArena {
    pub arena_base: *mut u8,
    pub arena_bytes: usize,
    pub cursor_bytes: usize,
    pub borrow_base: *mut u8,
    pub borrow_bytes: usize,
    /// `(residency_index, cache_slot)` lent from `ExpertCache` to `Prefill` during batched prefill.
    pub lent_slots: Vec<(i32, i32)>,
}

impl Qwen4ExpScratchArena {
    pub fn new(arena_base: *mut u8, arena_bytes: usize) -> Self {
        Self {
            arena_base,
            arena_bytes,
            cursor_bytes: 0,
            borrow_base: std::ptr::null_mut(),
            borrow_bytes: 0,
            lent_slots: Vec::new(),
        }
    }

    /// Carves a 256-byte aligned slice of `bytes` from the scratch arena.
    pub fn carve(&mut self, bytes: usize) -> Option<*mut u8> {
        let aligned = (self.cursor_bytes + 255) & !255usize;
        let end = aligned.checked_add(bytes)?;
        if end > self.arena_bytes || self.arena_base.is_null() {
            return None;
        }
        self.cursor_bytes = end;
        Some(unsafe { self.arena_base.add(aligned) })
    }

    pub fn reset_cursor(&mut self) {
        self.cursor_bytes = 0;
    }
}

unsafe impl Send for Qwen4ExpScratchArena {}
unsafe impl Sync for Qwen4ExpScratchArena {}

/// HIP/CUDA streams and synchronization events used across prefill, decode, verify, and adaptive refill.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen4ExpStreams {
    pub main_cs: *mut c_void,
    pub token_stream: *mut c_void,
    pub copy_stream: *mut c_void,
    pub side_stream: *mut c_void,
    pub mtp_stream: *mut c_void,
    pub adapt_stream: *mut c_void,
    pub doorbell: *mut c_void,
    pub adapt_ev: *mut c_void,
    pub exch_ev: *mut c_void,
}

unsafe impl Send for Qwen4ExpStreams {}
unsafe impl Sync for Qwen4ExpStreams {}

/// 32-byte header at the start of every `<TALUS_FN_DUMP>/<tag>.bin` tensor dump file.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerDumpHeader {
    pub magic: u64,
    pub layer: i32,
    pub reserved: i32,
    pub pos: i64,
    pub n_floats: u64,
}

/// Entry recorded in `<TALUS_FN_DUMP>/manifest.tsv`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerDumpEntry {
    pub tag: String,
    pub layer: i32,
    pub pos: i64,
    pub n_floats: usize,
    pub fnv1a64: u64,
}

/// Per-tensor bitwise & numerical comparison result between two `TALUS_FN_DUMP` directories.
#[derive(Clone, Debug)]
pub struct TensorDiffReport {
    pub tag: String,
    pub layer: i32,
    pub pos: i64,
    pub n_floats: usize,
    pub bitwise_identical: bool,
    pub mismatched_floats: usize,
    pub first_mismatch_idx: Option<usize>,
    pub first_lhs: f32,
    pub first_rhs: f32,
    pub max_abs_diff: f32,
}

/// Summary comparison across all tensors in two `TALUS_FN_DUMP` directories (`cpp_dir` vs `rust_dir`).
#[derive(Clone, Debug)]
pub struct LayerDumpComparison {
    pub lhs_dir: PathBuf,
    pub rhs_dir: PathBuf,
    pub tensors: Vec<TensorDiffReport>,
    pub missing_in_lhs: Vec<String>,
    pub missing_in_rhs: Vec<String>,
}

impl LayerDumpComparison {
    /// Returns true iff both dump directories contain the exact same tensor set and every tensor is bitwise identical.
    pub fn is_bitwise_identical(&self) -> bool {
        self.missing_in_lhs.is_empty()
            && self.missing_in_rhs.is_empty()
            && !self.tensors.is_empty()
            && self.tensors.iter().all(|t| t.bitwise_identical)
    }

    /// Formats a human-readable summary table for CLI output and gate reports.
    pub fn format_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "TALUS_FN_DUMP compare: {} vs {}\n",
            self.lhs_dir.display(),
            self.rhs_dir.display()
        ));
        for t in &self.tensors {
            if t.bitwise_identical {
                out.push_str(&format!(
                    "  [PASS] {:<32} layer={:<3} pos={:<6} n={:<8} BITWISE_IDENTICAL\n",
                    t.tag, t.layer, t.pos, t.n_floats
                ));
            } else {
                out.push_str(&format!(
                    "  [DIFF] {:<32} layer={:<3} pos={:<6} n={:<8} mismatches={} first_idx={:?} ({:.6e} vs {:.6e}) max_abs_diff={:.6e}\n",
                    t.tag,
                    t.layer,
                    t.pos,
                    t.n_floats,
                    t.mismatched_floats,
                    t.first_mismatch_idx,
                    t.first_lhs,
                    t.first_rhs,
                    t.max_abs_diff
                ));
            }
        }
        for m in &self.missing_in_lhs {
            out.push_str(&format!(
                "  [MISS] {m} missing in {}\n",
                self.lhs_dir.display()
            ));
        }
        for m in &self.missing_in_rhs {
            out.push_str(&format!(
                "  [MISS] {m} missing in {}\n",
                self.rhs_dir.display()
            ));
        }
        let pass_n = self.tensors.iter().filter(|t| t.bitwise_identical).count();
        out.push_str(&format!(
            "Summary: {}/{} tensors bitwise identical (missing lhs={}, rhs={})\n",
            pass_n,
            self.tensors.len(),
            self.missing_in_lhs.len(),
            self.missing_in_rhs.len()
        ));
        out
    }
}

/// FNV-1a 64-bit hash over raw float bytes (matching the C++ `TALUS_FN_DUMP` manifest hash).
#[inline]
pub fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Per-layer bitwise dump harness controlled by `TALUS_FN_DUMP=<dir>`.
#[derive(Clone, Debug)]
pub struct LayerDumpHarness {
    pub dump_dir: Option<PathBuf>,
}

impl Default for LayerDumpHarness {
    fn default() -> Self {
        Self::from_env()
    }
}

impl LayerDumpHarness {
    pub fn from_env() -> Self {
        let dump_dir = std::env::var("TALUS_FN_DUMP")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        if let Some(ref dir) = dump_dir {
            let _ = fs::create_dir_all(dir);
        }
        Self { dump_dir }
    }

    pub fn with_dir(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let p = dir.into();
        fs::create_dir_all(&p)?;
        Ok(Self { dump_dir: Some(p) })
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.dump_dir.is_some()
    }

    /// Writes a host float slice `data` to `<TALUS_FN_DUMP>/<tag>.bin` and records its hash in `manifest.tsv`.
    pub fn dump_host_slice(
        &self,
        tag: &str,
        layer: i32,
        pos: i64,
        data: &[f32],
    ) -> std::io::Result<()> {
        let Some(ref dir) = self.dump_dir else {
            return Ok(());
        };
        fs::create_dir_all(dir)?;
        let bin_path = dir.join(format!("{tag}.bin"));
        let mut f = File::create(&bin_path)?;
        let hdr = LayerDumpHeader {
            magic: LAYER_DUMP_MAGIC,
            layer,
            reserved: 0,
            pos,
            n_floats: data.len() as u64,
        };
        let hdr_bytes = unsafe {
            std::slice::from_raw_parts(
                (&hdr as *const LayerDumpHeader).cast::<u8>(),
                std::mem::size_of::<LayerDumpHeader>(),
            )
        };
        let payload_bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        f.write_all(hdr_bytes)?;
        f.write_all(payload_bytes)?;

        let hash = fnv1a64_bytes(payload_bytes);
        let manifest_path = dir.join("manifest.tsv");
        let mut mf = OpenOptions::new()
            .create(true)
            .append(true)
            .open(manifest_path)?;
        writeln!(mf, "{tag}\t{layer}\t{pos}\t{}\t{hash:016x}", data.len())?;
        Ok(())
    }

    /// Reads a dumped `<tag>.bin` file, returning its `(LayerDumpHeader, Vec<f32>)`.
    pub fn read_dump_file(path: &Path) -> std::io::Result<(LayerDumpHeader, Vec<f32>)> {
        let mut f = File::open(path)?;
        let mut hdr_buf = [0u8; std::mem::size_of::<LayerDumpHeader>()];
        f.read_exact(&mut hdr_buf)?;
        let hdr: LayerDumpHeader = unsafe { std::ptr::read_unaligned(hdr_buf.as_ptr().cast()) };
        if hdr.magic != LAYER_DUMP_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad TALUS_FN_DUMP magic in {}", path.display()),
            ));
        }
        let n = hdr.n_floats as usize;
        let mut out = vec![0.0f32; n];
        let payload_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                out.as_mut_ptr().cast::<u8>(),
                n * std::mem::size_of::<f32>(),
            )
        };
        f.read_exact(payload_bytes)?;
        Ok((hdr, out))
    }

    /// Compares all `.bin` tensor dumps between `lhs_dir` (e.g. C++) and `rhs_dir` (e.g. Rust).
    pub fn compare_dirs(lhs_dir: &Path, rhs_dir: &Path) -> std::io::Result<LayerDumpComparison> {
        let collect_bins = |dir: &Path| -> std::io::Result<BTreeMap<String, PathBuf>> {
            let mut map = BTreeMap::new();
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("bin") {
                    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                        map.insert(stem.to_string(), p);
                    }
                }
            }
            Ok(map)
        };

        let lhs_map = collect_bins(lhs_dir)?;
        let rhs_map = collect_bins(rhs_dir)?;

        let mut missing_in_lhs = Vec::new();
        let mut missing_in_rhs = Vec::new();
        let mut tensors = Vec::new();

        for (tag, lhs_path) in &lhs_map {
            let Some(rhs_path) = rhs_map.get(tag) else {
                missing_in_rhs.push(tag.clone());
                continue;
            };
            let (lh, ld) = Self::read_dump_file(lhs_path)?;
            let (_rh, rd) = Self::read_dump_file(rhs_path)?;
            let n = ld.len().min(rd.len());
            let mut mismatched = if ld.len() != rd.len() {
                ld.len().abs_diff(rd.len())
            } else {
                0
            };
            let mut first_idx = None;
            let mut first_l = 0.0f32;
            let mut first_r = 0.0f32;
            let mut max_abs_diff = 0.0f32;
            for i in 0..n {
                let lb = ld[i].to_bits();
                let rb = rd[i].to_bits();
                if lb != rb {
                    mismatched += 1;
                    if first_idx.is_none() {
                        first_idx = Some(i);
                        first_l = ld[i];
                        first_r = rd[i];
                    }
                    let diff = (ld[i] - rd[i]).abs();
                    if diff.is_nan() || diff > max_abs_diff {
                        max_abs_diff = diff;
                    }
                }
            }
            tensors.push(TensorDiffReport {
                tag: tag.clone(),
                layer: lh.layer,
                pos: lh.pos,
                n_floats: ld.len(),
                bitwise_identical: mismatched == 0 && ld.len() == rd.len(),
                mismatched_floats: mismatched,
                first_mismatch_idx: first_idx,
                first_lhs: first_l,
                first_rhs: first_r,
                max_abs_diff,
            });
        }

        for tag in rhs_map.keys() {
            if !lhs_map.contains_key(tag) {
                missing_in_lhs.push(tag.clone());
            }
        }

        Ok(LayerDumpComparison {
            lhs_dir: lhs_dir.to_path_buf(),
            rhs_dir: rhs_dir.to_path_buf(),
            tensors,
            missing_in_lhs,
            missing_in_rhs,
        })
    }
}

/// Top-level Flash-Next (`qwen4exp`) pure-Rust engine struct owning model config, per-arch seam,
/// device buffers, KV/GDN/PLE state, scratch arena, streams, layer-dump harness, and Pieces 1–6.
pub struct Qwen4ExpRustEngine {
    pub config: Qwen4ExpModelConfig,
    pub seam: Qwen4ExpArchSeam,
    pub buffers: Qwen4ExpDeviceBuffers,
    pub state: Qwen4ExpKvGdnState,
    pub arena: Qwen4ExpScratchArena,
    pub streams: Qwen4ExpStreams,
    pub dump: LayerDumpHarness,
    pub kernels: Option<Qwen4ExpKernels>,
    pub piece1_weights: Piece1Weights,
    pub piece2_prefill: Piece2Prefill,
    pub piece3_decode: Piece3Decode,
    pub piece4_verify_mtp: Piece4VerifyMtp,
    pub piece5_conv_state: Piece5ConvState,
    pub piece6_kld_flags: Piece6KldFlags,
}

impl Qwen4ExpRustEngine {
    pub fn new(config: Qwen4ExpModelConfig, kernels: Option<Qwen4ExpKernels>) -> Self {
        let seam = Qwen4ExpArchSeam::from_env();
        let state = Qwen4ExpKvGdnState::new(&config);
        Self {
            config,
            seam,
            buffers: Qwen4ExpDeviceBuffers::default(),
            state,
            arena: Qwen4ExpScratchArena::default(),
            streams: Qwen4ExpStreams::default(),
            dump: LayerDumpHarness::from_env(),
            kernels,
            piece1_weights: Piece1Weights::new(&config),
            piece2_prefill: Piece2Prefill::new(&config),
            piece3_decode: Piece3Decode::new(&config),
            piece4_verify_mtp: Piece4VerifyMtp::new(&config),
            piece5_conv_state: Piece5ConvState::new(&config),
            piece6_kld_flags: Piece6KldFlags::new(&config),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_layer_ordinals_match_qwen4exp() {
        let cfg = Qwen4ExpModelConfig::default();
        assert_eq!(cfg.n_layers, 48);
        assert_eq!(cfg.n_qsa_layers(), 12);
        assert_eq!(cfg.n_gdn_layers(), 36);
        assert_eq!(cfg.residual_dim(), 10240);

        // Layers 0, 1, 2 are GDN (ordinals 0, 1, 2); layer 3 is QSA (ordinal 0); layer 4 is GDN (ordinal 3).
        assert!(!cfg.is_qsa_layer(0));
        assert_eq!(cfg.gdn_ord(0), Some(0));
        assert_eq!(cfg.qsa_ord(0), None);
        assert!(cfg.is_qsa_layer(3));
        assert_eq!(cfg.qsa_ord(3), Some(0));
        assert_eq!(cfg.gdn_ord(3), None);
        assert_eq!(cfg.gdn_ord(4), Some(3));
        assert!(cfg.is_qsa_layer(47));
        assert_eq!(cfg.qsa_ord(47), Some(11));
    }

    #[test]
    fn layer_dump_harness_roundtrip_and_bitwise_compare() {
        let base = std::env::temp_dir().join(format!("talus_fn_dump_test_{}", std::process::id()));
        let cpp_dir = base.join("cpp");
        let rust_dir = base.join("rust");
        let _ = fs::remove_dir_all(&base);

        let h_cpp = LayerDumpHarness::with_dir(&cpp_dir).unwrap();
        let h_rust = LayerDumpHarness::with_dir(&rust_dir).unwrap();

        let l0 = vec![1.0f32, -2.5, 3.25, 0.0, 42.125];
        let l1 = vec![0.5f32, 1.5, -0.75, 8.0, 9.0];
        h_cpp
            .dump_host_slice("prefill_p0_l00_R", 0, 0, &l0)
            .unwrap();
        h_cpp
            .dump_host_slice("prefill_p0_l01_R", 1, 0, &l1)
            .unwrap();
        h_rust
            .dump_host_slice("prefill_p0_l00_R", 0, 0, &l0)
            .unwrap();
        h_rust
            .dump_host_slice("prefill_p0_l01_R", 1, 0, &l1)
            .unwrap();

        let cmp = LayerDumpHarness::compare_dirs(&cpp_dir, &rust_dir).unwrap();
        assert!(cmp.is_bitwise_identical(), "{}", cmp.format_report());

        // Introduce a 1-ULP difference in layer 1 on the Rust side and verify detection.
        let mut l1_bad = l1.clone();
        l1_bad[2] = f32::from_bits(l1_bad[2].to_bits() ^ 1);
        h_rust
            .dump_host_slice("prefill_p0_l01_R", 1, 0, &l1_bad)
            .unwrap();
        let cmp2 = LayerDumpHarness::compare_dirs(&cpp_dir, &rust_dir).unwrap();
        assert!(!cmp2.is_bitwise_identical());
        assert_eq!(cmp2.tensors[1].mismatched_floats, 1);
        assert_eq!(cmp2.tensors[1].first_mismatch_idx, Some(2));

        let _ = fs::remove_dir_all(&base);
    }
}
