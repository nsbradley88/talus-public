//! Strata native pack reader (`index.txt`, `native_experts.txt` v2–v4, `dense.bin`, `experts.bin`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use talus_core_api::QuantType;

use crate::error::{GgufError, Result};
use crate::gguf::GgufFile;
use crate::mmap::{Backing, Mmap};

/// Default hidden dimension (`H`) for Qwen3.8-Flash-Next native expert layout.
pub const DEFAULT_EXPERT_N_EMBD: usize = 2560;
/// Default expert intermediate dimension (`FF`) for Qwen3.8-Flash-Next native expert layout.
pub const DEFAULT_EXPERT_N_FF: usize = 640;
/// Default layer count for Qwen3.8-Flash-Next native expert layout.
pub const DEFAULT_EXPERT_N_LAYERS: usize = 48;
/// Default routed expert count for Qwen3.8-Flash-Next native expert layout.
pub const DEFAULT_EXPERT_N_EXPERT: usize = 512;
/// Maximum supported `native_experts.txt` format version.
pub const NATIVE_EXPERTS_MAX_VERSION: u32 = 4;
/// Trailing 16-byte global scale header `{s_gate, s_up, s_down, 0}` on NVFP4 expert blobs.
pub const NVFP4_EXPERT_TAIL_BYTES: usize = 16;

/// Source binary file identifier in `index.txt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackFileId {
    /// `0` -> `dense.bin`
    Dense = 0,
    /// `1` -> `embd.bin`
    Embd = 1,
    /// `2` -> `experts.bin`
    Experts = 2,
    /// `3` -> `extra.bin`
    Extra = 3,
}

impl PackFileId {
    /// Parses a numeric file ID (`0..=3`) from `index.txt`.
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Dense),
            1 => Some(Self::Embd),
            2 => Some(Self::Experts),
            3 => Some(Self::Extra),
            _ => None,
        }
    }

    /// Returns the pack binary filename for this file ID.
    pub const fn file_name(self) -> &'static str {
        match self {
            Self::Dense => "dense.bin",
            Self::Embd => "embd.bin",
            Self::Experts => "experts.bin",
            Self::Extra => "extra.bin",
        }
    }
}

/// Engine weight representation kind in `index.txt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightKind {
    /// `0`: Verbatim copy (quantized planes or already-native float/half bytes).
    Verbatim = 0,
    /// `1`: BF16 values stored widened in F32 (4 B/elem in pack -> 2 B/elem in engine).
    Bf16InF32 = 1,
    /// `2`: 32-bit IEEE float (4 B/elem).
    F32 = 2,
    /// `3`: F16 values stored widened in F32 (4 B/elem in pack -> 2 B/elem in engine).
    F16InF32 = 3,
    /// `4`: Raw 16-bit BF16 bytes stored directly in `dense.bin` (2 B/elem in pack and engine).
    RawBf16 = 4,
    /// `5`: Raw 16-bit F16 bytes stored directly in `dense.bin` (2 B/elem in pack and engine).
    RawF16 = 5,
}

impl WeightKind {
    /// Parses the integer `kind` column from `index.txt`.
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::Verbatim),
            1 => Some(Self::Bf16InF32),
            2 => Some(Self::F32),
            3 => Some(Self::F16InF32),
            4 => Some(Self::RawBf16),
            5 => Some(Self::RawF16),
            _ => None,
        }
    }

    /// True if stored on disk as raw 2-byte elements (`RawBf16` or `RawF16`).
    #[inline]
    pub const fn is_raw16(self) -> bool {
        matches!(self, Self::RawBf16 | Self::RawF16)
    }
}

/// A parsed tensor entry from `index.txt` (19 columns).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexRow {
    pub name: String,
    pub file: PackFileId,
    pub kind: WeightKind,
    pub src_off: u64,
    pub src_bytes: u64,
    pub dst_off: u64,
    pub dst_bytes: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub code_bits: i32,
    pub code_bias: i32,
    pub group_elems: i32,
    pub codebook_iq4nl: bool,
    pub has_offset: bool,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    pub offset_bytes: u64,
    pub scales_fp16: bool,
    pub act_kind: i32,
}

impl IndexRow {
    /// Total logical element count (`ne0 * max(ne1, 1)`).
    #[inline]
    pub fn elements(&self) -> u64 {
        let d0 = u64::try_from(self.ne0.max(0)).unwrap_or(0);
        let d1 = u64::try_from(if self.ne1 > 0 { self.ne1 } else { 1 }).unwrap_or(1);
        d0.saturating_mul(d1)
    }

    /// True if this row represents a shape-only placeholder served natively from the GGUF.
    #[inline]
    pub const fn is_gguf_served_placeholder(&self) -> bool {
        self.code_bits != 0 && self.dst_bytes == 0 && self.src_bytes == 0
    }
}

/// Parsed `index.txt` manifest for a Strata pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackIndex {
    pub align: u64,
    pub pool_bytes_total: u64,
    pub declared_tensors: usize,
    rows: Vec<IndexRow>,
    by_name: BTreeMap<String, usize>,
}

impl PackIndex {
    /// Opens and parses `index.txt` at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let text = fs::read_to_string(path_ref).map_err(|e| GgufError::PackFormat {
            message: format!("cannot open {}: {e}", path_ref.display()),
        })?;
        Self::parse_str(&text)
    }

    /// Parses `index.txt` content from a string.
    pub fn parse_str(text: &str) -> Result<Self> {
        let mut align: u64 = 0;
        let mut pool_bytes_total: u64 = 0;
        let mut declared_tensors: usize = 0;
        let mut rows = Vec::new();
        let mut by_name = BTreeMap::new();

        for (line_idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                let tokens: Vec<&str> = rest.split_whitespace().collect();
                if tokens.len() == 6
                    && tokens[0] == "align"
                    && tokens[2] == "pool"
                    && tokens[4] == "tensors"
                {
                    align = tokens[1].parse().map_err(|_| GgufError::PackFormat {
                        message: format!("invalid align in header on line {}", line_idx + 1),
                    })?;
                    pool_bytes_total = tokens[3].parse().map_err(|_| GgufError::PackFormat {
                        message: format!("invalid pool in header on line {}", line_idx + 1),
                    })?;
                    declared_tensors = tokens[5].parse().map_err(|_| GgufError::PackFormat {
                        message: format!(
                            "invalid tensors count in header on line {}",
                            line_idx + 1
                        ),
                    })?;
                }
                continue;
            }

            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 19 {
                return Err(GgufError::PackFormat {
                    message: format!(
                        "malformed index.txt line {}: expected 19 fields, got {}",
                        line_idx + 1,
                        fields.len()
                    ),
                });
            }

            let parse_u64 = |idx: usize, label: &str| -> Result<u64> {
                fields[idx]
                    .parse::<u64>()
                    .map_err(|_| GgufError::PackFormat {
                        message: format!(
                            "invalid {label} '{}' on line {}",
                            fields[idx],
                            line_idx + 1
                        ),
                    })
            };
            let parse_i64 = |idx: usize, label: &str| -> Result<i64> {
                fields[idx]
                    .parse::<i64>()
                    .map_err(|_| GgufError::PackFormat {
                        message: format!(
                            "invalid {label} '{}' on line {}",
                            fields[idx],
                            line_idx + 1
                        ),
                    })
            };
            let parse_i32 = |idx: usize, label: &str| -> Result<i32> {
                fields[idx]
                    .parse::<i32>()
                    .map_err(|_| GgufError::PackFormat {
                        message: format!(
                            "invalid {label} '{}' on line {}",
                            fields[idx],
                            line_idx + 1
                        ),
                    })
            };

            let name = fields[0].to_string();
            let file_id_raw = parse_u64(1, "file")? as u32;
            let file = PackFileId::from_id(file_id_raw).ok_or_else(|| GgufError::PackFormat {
                message: format!("unknown file id {file_id_raw} for tensor '{name}'"),
            })?;
            let kind_raw = parse_u64(2, "kind")? as u32;
            let kind = WeightKind::from_id(kind_raw).ok_or_else(|| GgufError::PackFormat {
                message: format!("unknown weight kind {kind_raw} for tensor '{name}'"),
            })?;

            let src_off = parse_u64(3, "src_off")?;
            let src_bytes = parse_u64(4, "src_bytes")?;
            let dst_off = parse_u64(5, "dst_off")?;
            let dst_bytes = parse_u64(6, "dst_bytes")?;
            let ne0 = parse_i64(7, "ne0")?;
            let ne1 = parse_i64(8, "ne1")?;
            let code_bits = parse_i32(9, "code_bits")?;
            let code_bias = parse_i32(10, "code_bias")?;
            let group_elems = parse_i32(11, "group_elems")?;
            let codebook = parse_i32(12, "codebook")?;
            let has_offset = parse_i32(13, "has_offset")?;
            let codes_bytes = parse_u64(14, "codes_bytes")?;
            let scales_bytes = parse_u64(15, "scales_bytes")?;
            let offset_bytes = parse_u64(16, "offset_bytes")?;
            let scales_fp16 = parse_i32(17, "scales_fp16")?;
            let act_kind = parse_i32(18, "act_kind")?;

            let row = IndexRow {
                name: name.clone(),
                file,
                kind,
                src_off,
                src_bytes,
                dst_off,
                dst_bytes,
                ne0,
                ne1,
                code_bits,
                code_bias,
                group_elems,
                codebook_iq4nl: codebook != 0,
                has_offset: has_offset != 0,
                codes_bytes,
                scales_bytes,
                offset_bytes,
                scales_fp16: scales_fp16 != 0,
                act_kind,
            };

            validate_index_row(&row)?;

            if by_name.insert(name.clone(), rows.len()).is_some() {
                return Err(GgufError::PackFormat {
                    message: format!("duplicate tensor '{name}' in index.txt"),
                });
            }
            rows.push(row);
        }

        if align == 0
            || pool_bytes_total == 0
            || declared_tensors == 0
            || rows.len() != declared_tensors
        {
            return Err(GgufError::PackFormat {
                message: format!(
                    "index.txt header/row count mismatch: align={align}, pool={pool_bytes_total}, declared={declared_tensors}, actual={}",
                    rows.len()
                ),
            });
        }

        Ok(Self {
            align,
            pool_bytes_total,
            declared_tensors,
            rows,
            by_name,
        })
    }

    /// Slice of all parsed rows in order.
    #[inline]
    pub fn rows(&self) -> &[IndexRow] {
        &self.rows
    }

    /// Looks up a row by tensor name.
    #[inline]
    pub fn find(&self, name: &str) -> Option<&IndexRow> {
        self.by_name.get(name).map(|&i| &self.rows[i])
    }

    /// Returns `Some(code_bits)` for `name` (`0` for unquantized, `> 0` for quantized), or `None` if absent.
    #[inline]
    pub fn index_code_bits(&self, name: &str) -> Option<i32> {
        self.find(name).map(|r| r.code_bits)
    }

    /// Computes required arena pool bytes when skipping the tensor names in `skip`
    /// (matching `WeightTable::pool_bytes` in `weights.cpp`).
    pub fn pool_bytes(&self, skip: Option<&BTreeSet<String>>) -> u64 {
        let Some(skip_set) = skip else {
            return self.pool_bytes_total;
        };
        if skip_set.is_empty() {
            return self.pool_bytes_total;
        }
        let mut cur = 0u64;
        for row in &self.rows {
            if skip_set.contains(&row.name) || row.dst_bytes == 0 {
                continue;
            }
            cur = (cur + row.dst_bytes + self.align - 1) & !(self.align - 1);
        }
        cur
    }
}

fn validate_index_row(r: &IndexRow) -> Result<()> {
    if r.is_gguf_served_placeholder() {
        return Ok(());
    }
    if r.codes_bytes != 0 {
        if r.scales_fp16 && !r.scales_bytes.is_multiple_of(2) {
            return Err(GgufError::PackFormat {
                message: format!(
                    "{}: fp16 scale plane has odd byte count {}",
                    r.name, r.scales_bytes
                ),
            });
        }
        let src_scale_bytes = if r.scales_fp16 {
            r.scales_bytes / 2
        } else {
            r.scales_bytes
        };
        let sum_src = r.codes_bytes + src_scale_bytes + r.offset_bytes;
        let sum_dst = r.codes_bytes + r.scales_bytes + r.offset_bytes;
        if sum_src != r.src_bytes || sum_dst != r.dst_bytes {
            return Err(GgufError::PackFormat {
                message: format!(
                    "{}: plane segments cover {sum_src}/{sum_dst} B but index declares {}/{} B",
                    r.name, r.src_bytes, r.dst_bytes
                ),
            });
        }
    } else if matches!(r.kind, WeightKind::Bf16InF32 | WeightKind::F16InF32) {
        let elems = r.elements();
        if r.src_bytes != elems * 4 || r.dst_bytes != elems * 2 {
            return Err(GgufError::PackFormat {
                message: format!(
                    "{}: promoted tensor of {elems} elements is {} B in / {} B out, expected {}/{}",
                    r.name,
                    r.src_bytes,
                    r.dst_bytes,
                    elems * 4,
                    elems * 2
                ),
            });
        }
    }
    Ok(())
}

/// Computes the byte size of one row (`cols` elements) for a supported native expert `ggml_type`.
pub fn expert_row_bytes(ggml_type: i32, cols: usize) -> Option<usize> {
    if ggml_type < 0 {
        return None;
    }
    let qtype = QuantType::from_ggml_type_id(ggml_type as u32)?;
    // Native expert kernels support IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, Q2_0, Q2_K..Q6_K, Q8_0, NVFP4.
    if !matches!(
        qtype,
        QuantType::IQ2_XXS
            | QuantType::IQ2_XS
            | QuantType::IQ3_XXS
            | QuantType::IQ1_S
            | QuantType::IQ4_NL
            | QuantType::IQ3_S
            | QuantType::IQ2_S
            | QuantType::IQ4_XS
            | QuantType::IQ1_M
            | QuantType::Q2_0
            | QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q4_K
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::Q8_0
            | QuantType::NVFP4
    ) {
        return None;
    }
    let (block_elems, block_bytes) = qtype.block_geometry();
    if cols == 0 || !cols.is_multiple_of(block_elems) {
        return None;
    }
    Some((cols / block_elems) * block_bytes)
}

/// Geometry of a single native expert blob (`gate || up || down` [+ optional NVFP4 tail]) for one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeFmt {
    /// `ggml_type` for `gate` and `up` projections (`[FF, H]`).
    pub gu_type: i32,
    /// `ggml_type` for `down` projection (`[H, FF]`).
    pub d_type: i32,
    /// Byte length of one `gate` or `up` row (`H` elements).
    pub gu_row: usize,
    /// Byte length of one `down` row (`FF` elements).
    pub d_row: usize,
    /// Byte offset of `up` within one expert blob (`gu_row * FF`).
    pub up_off: usize,
    /// Byte offset of `down` within one expert blob (`2 * up_off`).
    pub down_off: usize,
    /// Total byte size of one expert blob (`down_off + d_row * H` + 16 if NVFP4).
    pub bytes: usize,
}

impl NativeFmt {
    /// Computes the native expert blob format for `(gu_type, d_type)` at dimensions `(n_embd, n_ff)`.
    pub fn compute(gu_type: i32, d_type: i32, n_embd: usize, n_ff: usize) -> Option<Self> {
        let gu_row = expert_row_bytes(gu_type, n_embd)?;
        let d_row = expert_row_bytes(d_type, n_ff)?;
        let up_off = gu_row.checked_mul(n_ff)?;
        let down_off = up_off.checked_mul(2)?;
        let tail = if gu_type == 40 {
            if d_type != 40 {
                return None;
            }
            NVFP4_EXPERT_TAIL_BYTES
        } else {
            0
        };
        let bytes = down_off
            .checked_add(d_row.checked_mul(n_embd)?)?
            .checked_add(tail)?;
        Some(Self {
            gu_type,
            d_type,
            gu_row,
            d_row,
            up_off,
            down_off,
            bytes,
        })
    }

    /// Byte lengths of the three projections `[gate, up, down]` for one expert.
    #[inline]
    pub const fn role_bytes(&self, n_embd: usize) -> [usize; 3] {
        [self.up_off, self.up_off, self.d_row * n_embd]
    }
}

/// Dimension configuration for parsing `native_experts.txt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertLayoutConfig {
    pub n_layers: usize,
    pub n_expert: usize,
    pub n_embd: usize,
    pub n_ff: usize,
}

impl Default for ExpertLayoutConfig {
    fn default() -> Self {
        Self {
            n_layers: DEFAULT_EXPERT_N_LAYERS,
            n_expert: DEFAULT_EXPERT_N_EXPERT,
            n_embd: DEFAULT_EXPERT_N_EMBD,
            n_ff: DEFAULT_EXPERT_N_FF,
        }
    }
}

/// Per-layer entry parsed from `native_experts.txt` (v2–v4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLayerSpec {
    pub layer: usize,
    pub fmt: NativeFmt,
    /// Byte offset of layer `l` in a monolithic `experts.bin` (`sum_{i < l} fmt[i].bytes * n_expert`).
    pub offset: u64,
    /// Optional GGUF file offsets `[gate_off, up_off, down_off]` for direct shard serving (v2+).
    pub gguf_off: [u64; 3],
    /// Optional per-role shard basenames `[gate_shard, up_shard, down_shard]` (v3 single shard or v4 comma-separated).
    pub gguf_shard: [String; 3],
}

/// Parsed `native_experts.txt` layout table across all layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLayout {
    pub version: u32,
    pub n_layers: usize,
    pub n_expert: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub total_bytes: u64,
    pub has_gguf_offsets: bool,
    layers: Vec<ExpertLayerSpec>,
}

impl ExpertLayout {
    /// Opens and parses `native_experts.txt` at `path` using default Qwen3.8-Flash-Next dimensions.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_config(path, ExpertLayoutConfig::default())
    }

    /// Opens and parses `native_experts.txt` at `path` with custom model dimensions.
    pub fn open_with_config(path: impl AsRef<Path>, config: ExpertLayoutConfig) -> Result<Self> {
        let path_ref = path.as_ref();
        let text = fs::read_to_string(path_ref).map_err(|e| GgufError::PackFormat {
            message: format!("cannot open {}: {e}", path_ref.display()),
        })?;
        Self::parse_str_with_config(&text, config)
    }

    /// Parses `native_experts.txt` content with default Qwen3.8-Flash-Next dimensions.
    pub fn parse_str(text: &str) -> Result<Self> {
        Self::parse_str_with_config(text, ExpertLayoutConfig::default())
    }

    /// Parses `native_experts.txt` (v2–v4) content with explicit model dimensions.
    pub fn parse_str_with_config(text: &str, config: ExpertLayoutConfig) -> Result<Self> {
        let mut version = 1u32;
        let mut n_expert = config.n_expert;
        let n_layers = config.n_layers;
        let mut slots: Vec<Option<ExpertLayerSpec>> = vec![None; n_layers];
        let mut has_gguf_offsets = false;

        for (line_no, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('#') {
                if let Some(v_pos) = line.find("# strata native experts v") {
                    let after = &line[v_pos + "# strata native experts v".len()..];
                    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                    if let Ok(v) = digits.parse::<u32>() {
                        if v > NATIVE_EXPERTS_MAX_VERSION {
                            return Err(GgufError::PackFormat {
                                message: format!(
                                    "native_experts.txt: unsupported version v{v} (this build reads up to v{NATIVE_EXPERTS_MAX_VERSION})"
                                ),
                            });
                        }
                        version = v;
                    }
                }
                if let Some(ne_pos) = line.find("(n_expert ") {
                    let after = &line[ne_pos + "(n_expert ".len()..];
                    let token: String = after
                        .chars()
                        .take_while(|c| !c.is_whitespace() && *c != ',' && *c != ')')
                        .collect();
                    let parsed_ne = token.parse::<isize>().unwrap_or(-1);
                    if parsed_ne <= 0 || parsed_ne > 512 {
                        return Err(GgufError::PackFormat {
                            message: format!(
                                "native_experts.txt: invalid n_expert '{token}' in header"
                            ),
                        });
                    }
                    n_expert = parsed_ne as usize;
                }
                continue;
            }

            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() < 5 {
                return Err(GgufError::PackFormat {
                    message: format!(
                        "native_experts.txt: malformed line {} (expected at least 5 columns, got {})",
                        line_no + 1,
                        tokens.len()
                    ),
                });
            }

            let l_signed: isize = tokens[0].parse().map_err(|_| GgufError::PackFormat {
                message: format!(
                    "native_experts.txt: invalid layer '{}' on line {}",
                    tokens[0],
                    line_no + 1
                ),
            })?;
            if l_signed < 0 || (l_signed as usize) >= n_layers {
                return Err(GgufError::PackFormat {
                    message: format!(
                        "native_experts.txt: layer {l_signed} out of range 0..{n_layers} on line {}",
                        line_no + 1
                    ),
                });
            }
            let layer = l_signed as usize;
            if slots[layer].is_some() {
                return Err(GgufError::PackFormat {
                    message: format!("native_experts.txt: duplicate layer {layer}"),
                });
            }

            let gu_type: i32 = tokens[1].parse().map_err(|_| GgufError::PackFormat {
                message: format!(
                    "native_experts.txt: invalid gu_type '{}' on line {}",
                    tokens[1],
                    line_no + 1
                ),
            })?;
            let d_type: i32 = tokens[2].parse().map_err(|_| GgufError::PackFormat {
                message: format!(
                    "native_experts.txt: invalid d_type '{}' on line {}",
                    tokens[2],
                    line_no + 1
                ),
            })?;
            let offset: u64 = tokens[3].parse().map_err(|_| GgufError::PackFormat {
                message: format!(
                    "native_experts.txt: invalid offset '{}' on line {}",
                    tokens[3],
                    line_no + 1
                ),
            })?;
            let blob_bytes: usize = tokens[4].parse().map_err(|_| GgufError::PackFormat {
                message: format!(
                    "native_experts.txt: invalid blob_bytes '{}' on line {}",
                    tokens[4],
                    line_no + 1
                ),
            })?;

            let fmt = NativeFmt::compute(gu_type, d_type, config.n_embd, config.n_ff).ok_or_else(
                || GgufError::PackFormat {
                    message: format!(
                        "native_experts.txt: unsupported expert quant types ({gu_type}, {d_type}) on layer {layer}"
                    ),
                },
            )?;
            if fmt.bytes != blob_bytes {
                return Err(GgufError::PackFormat {
                    message: format!(
                        "native_experts.txt: layer {layer} blob_bytes={blob_bytes} disagrees with computed {} bytes for types ({gu_type}, {d_type})",
                        fmt.bytes
                    ),
                });
            }

            let mut gguf_off = [0u64; 3];
            let mut gguf_shard = [String::new(), String::new(), String::new()];

            if tokens.len() > 5 {
                if tokens.len() < 8 {
                    return Err(GgufError::PackFormat {
                        message: format!(
                            "native_experts.txt: layer {layer} has incomplete GGUF offset columns"
                        ),
                    });
                }
                for r in 0..3 {
                    gguf_off[r] = tokens[5 + r].parse().map_err(|_| GgufError::PackFormat {
                        message: format!(
                            "native_experts.txt: layer {layer} invalid GGUF offset '{}'",
                            tokens[5 + r]
                        ),
                    })?;
                }
                if gguf_off[0] != 0 || gguf_off[1] != 0 || gguf_off[2] != 0 {
                    has_gguf_offsets = true;
                }
                if tokens.len() > 8 {
                    if tokens.len() > 9 {
                        return Err(GgufError::PackFormat {
                            message: format!(
                                "native_experts.txt: layer {layer} has unexpected trailing tokens after shard column"
                            ),
                        });
                    }
                    let shard_col = tokens[8];
                    if !shard_col.contains(',') {
                        gguf_shard = [
                            shard_col.to_string(),
                            shard_col.to_string(),
                            shard_col.to_string(),
                        ];
                    } else {
                        let parts: Vec<&str> = shard_col.split(',').collect();
                        if parts.len() != 3 {
                            return Err(GgufError::PackFormat {
                                message: format!(
                                    "native_experts.txt: layer {layer} shard column '{shard_col}' must have 1 or 3 comma-separated entries"
                                ),
                            });
                        }
                        gguf_shard = [
                            parts[0].to_string(),
                            parts[1].to_string(),
                            parts[2].to_string(),
                        ];
                    }
                }
            }

            slots[layer] = Some(ExpertLayerSpec {
                layer,
                fmt,
                offset,
                gguf_off,
                gguf_shard,
            });
        }

        let mut layers = Vec::with_capacity(n_layers);
        let mut running_total = 0u64;
        for (l, slot) in slots.into_iter().enumerate() {
            let spec = slot.ok_or_else(|| GgufError::PackFormat {
                message: format!("native_experts.txt: missing layer {l}"),
            })?;
            if spec.offset != running_total {
                return Err(GgufError::PackFormat {
                    message: format!(
                        "native_experts.txt: layer {l} offset {} != expected contiguous offset {running_total}",
                        spec.offset
                    ),
                });
            }
            running_total += (spec.fmt.bytes as u64) * (n_expert as u64);
            layers.push(spec);
        }

        Ok(Self {
            version,
            n_layers,
            n_expert,
            n_embd: config.n_embd,
            n_ff: config.n_ff,
            total_bytes: running_total,
            has_gguf_offsets,
            layers,
        })
    }

    /// Slice of all per-layer expert layout specifications (`0..n_layers`).
    #[inline]
    pub fn layers(&self) -> &[ExpertLayerSpec] {
        &self.layers
    }

    /// Returns the specification for `layer`.
    #[inline]
    pub fn layer(&self, layer: usize) -> Option<&ExpertLayerSpec> {
        self.layers.get(layer)
    }

    /// Byte offset of `(layer, expert)` within a monolithic `experts.bin`.
    #[inline]
    pub fn blob_offset(&self, layer: usize, expert: usize) -> Option<(u64, usize)> {
        if expert >= self.n_expert {
            return None;
        }
        let spec = self.layers.get(layer)?;
        let off = spec.offset + (expert as u64) * (spec.fmt.bytes as u64);
        Some((off, spec.fmt.bytes))
    }

    /// Validates that `shards` cover every layer and projection (`gate`, `up`, `down`) declared
    /// in this layout (matching `ExpertLayout::check_gguf` and `load_experts_gguf` in `expert_layout.cpp`).
    pub fn check_gguf(&self, shards: &[GgufFile], primary_gguf_path: Option<&Path>) -> Result<()> {
        if !self.has_gguf_offsets {
            return Err(GgufError::PackFormat {
                message: "native_experts.txt has no GGUF offsets (pre-v2); re-run the converter"
                    .to_string(),
            });
        }
        if shards.is_empty() {
            return Err(GgufError::PackFormat {
                message: "no GGUF shards supplied for native expert validation".to_string(),
            });
        }

        let role_names = ["gate", "up", "down"];
        for spec in &self.layers {
            let l = spec.layer;
            let role_lens = spec.fmt.role_bytes(self.n_embd);
            let role_types = [spec.fmt.gu_type, spec.fmt.gu_type, spec.fmt.d_type];
            let role_cols = [self.n_embd as u64, self.n_embd as u64, self.n_ff as u64];
            let role_rows = [self.n_ff as u64, self.n_ff as u64, self.n_embd as u64];

            for r in 0..3 {
                let shard = resolve_role_shard(shards, primary_gguf_path, &spec.gguf_shard[r])
                    .ok_or_else(|| GgufError::PackFormat {
                        message: format!(
                            "layer {l} ({}): GGUF shard '{}' not found among opened shards",
                            role_names[r], spec.gguf_shard[r]
                        ),
                    })?;

                let span_bytes = (role_lens[r] as u64) * (self.n_expert as u64);
                let end = spec.gguf_off[r].checked_add(span_bytes).ok_or_else(|| {
                    GgufError::PackFormat {
                        message: format!("layer {l} ({}) GGUF offset overflow", role_names[r]),
                    }
                })?;
                if end > shard.file_size() {
                    return Err(GgufError::PackFormat {
                        message: format!(
                            "layer {l} ({}): GGUF span [{}..{end}) extends past shard EOF ({})",
                            role_names[r],
                            spec.gguf_off[r],
                            shard.file_size()
                        ),
                    });
                }

                // If the shard carries a tensor directory, verify the tensor's offset, type, and shape match.
                let tensor_name = format!("blk.{l}.ffn_{}_exps.weight", role_names[r]);
                if !shard.tensors().is_empty() {
                    let entry =
                        shard
                            .find_tensor(&tensor_name)
                            .ok_or_else(|| GgufError::PackFormat {
                                message: format!(
                                    "missing {tensor_name} in shard {:?}",
                                    shard.path()
                                ),
                            })?;
                    let actual_file_off = shard.data_start() + entry.offset;
                    if actual_file_off != spec.gguf_off[r] {
                        return Err(GgufError::PackFormat {
                            message: format!(
                                "{tensor_name} is at file offset {actual_file_off} in GGUF, but native_experts.txt says {}",
                                spec.gguf_off[r]
                            ),
                        });
                    }
                    if entry.qtype.ggml_type_id() != role_types[r] as u32 {
                        return Err(GgufError::PackFormat {
                            message: format!(
                                "{tensor_name} has ggml_type {}, but native_experts.txt says {}",
                                entry.qtype.ggml_type_id(),
                                role_types[r]
                            ),
                        });
                    }
                    let expected_shape = [role_cols[r], role_rows[r], self.n_expert as u64];
                    if entry.shape() != expected_shape {
                        return Err(GgufError::PackFormat {
                            message: format!(
                                "{tensor_name} has shape {:?}, expected {:?}",
                                entry.shape(),
                                expected_shape
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Reads and concatenates the `gate || up || down` bytes for `(layer, expert)` directly
    /// from the mapped GGUF shards.
    pub fn assemble_expert_from_gguf(
        &self,
        shards: &[GgufFile],
        primary_gguf_path: Option<&Path>,
        layer: usize,
        expert: usize,
    ) -> Result<Vec<u8>> {
        if expert >= self.n_expert {
            return Err(GgufError::PackFormat {
                message: format!("expert {expert} out of range 0..{}", self.n_expert),
            });
        }
        let spec = self
            .layers
            .get(layer)
            .ok_or_else(|| GgufError::PackFormat {
                message: format!("layer {layer} out of range 0..{}", self.n_layers),
            })?;
        let role_lens = spec.fmt.role_bytes(self.n_embd);
        let mut out = Vec::with_capacity(spec.fmt.bytes);
        for (r, &role_len) in role_lens.iter().enumerate() {
            let shard = resolve_role_shard(shards, primary_gguf_path, &spec.gguf_shard[r])
                .ok_or_else(|| GgufError::PackFormat {
                    message: format!("shard '{}' not found for layer {layer}", spec.gguf_shard[r]),
                })?;
            let start = (spec.gguf_off[r] as usize)
                .checked_add(expert * role_len)
                .ok_or_else(|| GgufError::PackFormat {
                    message: "expert offset overflow".to_string(),
                })?;
            let end = start + role_len;
            let bytes = shard.as_bytes();
            if end > bytes.len() {
                return Err(GgufError::PackFormat {
                    message: format!(
                        "expert ({layer}, {expert}) role {r} span [{start}..{end}) exceeds mapped shard len {}",
                        bytes.len()
                    ),
                });
            }
            out.extend_from_slice(&bytes[start..end]);
        }
        if out.len() < spec.fmt.bytes {
            out.resize(spec.fmt.bytes, 0);
        }
        Ok(out)
    }
}

fn resolve_role_shard<'a>(
    shards: &'a [GgufFile],
    primary_gguf_path: Option<&Path>,
    wanted_shard: &str,
) -> Option<&'a GgufFile> {
    let target_name = if wanted_shard.is_empty() {
        primary_gguf_path
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
    } else {
        Some(
            Path::new(wanted_shard)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(wanted_shard),
        )
    };

    if let Some(name) = target_name {
        for s in shards {
            if let Some(p) = s.path() {
                if p.file_name().and_then(|f| f.to_str()) == Some(name)
                    || p.to_str() == Some(wanted_shard)
                {
                    return Some(s);
                }
            }
        }
        if shards.len() == 1 && wanted_shard.is_empty() {
            return Some(&shards[0]);
        }
        None
    } else {
        shards.first()
    }
}

/// Strata native pack directory reader (`index.txt`, `native_experts.txt`, `dense.bin`, `experts.bin`).
#[derive(Debug)]
pub struct StrataPack {
    pack_dir: PathBuf,
    index: Option<PackIndex>,
    experts_layout: Option<ExpertLayout>,
    dense_bin: Option<Backing>,
    experts_bin: Option<Backing>,
}

impl StrataPack {
    /// Opens a Strata pack directory at `pack_dir` with default expert dimensions.
    ///
    /// Parses `index.txt` and/or `native_experts.txt` if present, and memory-maps
    /// `dense.bin` and `experts.bin` if present. Requires at least one of `index.txt` or
    /// `native_experts.txt` to exist in `pack_dir`.
    pub fn open(pack_dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_config(pack_dir, ExpertLayoutConfig::default())
    }

    /// Opens a Strata pack directory with explicit [`ExpertLayoutConfig`] dimensions.
    pub fn open_with_config(
        pack_dir: impl AsRef<Path>,
        config: ExpertLayoutConfig,
    ) -> Result<Self> {
        let dir = pack_dir.as_ref().to_path_buf();
        let index_path = dir.join("index.txt");
        let experts_txt_path = dir.join("native_experts.txt");
        let dense_bin_path = dir.join("dense.bin");
        let experts_bin_path = dir.join("experts.bin");

        let index = if index_path.is_file() {
            Some(PackIndex::open(&index_path)?)
        } else {
            None
        };

        let experts_layout = if experts_txt_path.is_file() {
            Some(ExpertLayout::open_with_config(&experts_txt_path, config)?)
        } else {
            None
        };

        if index.is_none() && experts_layout.is_none() {
            return Err(GgufError::PackFormat {
                message: format!(
                    "neither index.txt nor native_experts.txt found in {}",
                    dir.display()
                ),
            });
        }

        let dense_bin = if dense_bin_path.is_file() {
            Some(Backing::Mmap(Mmap::open(&dense_bin_path)?))
        } else {
            None
        };

        let experts_bin = if experts_bin_path.is_file() {
            let mm = Mmap::open(&experts_bin_path)?;
            if let Some(ref layout) = experts_layout {
                if (mm.len() as u64) < layout.total_bytes {
                    return Err(GgufError::PackFormat {
                        message: format!(
                            "experts.bin is {} bytes, smaller than layout total_bytes {}",
                            mm.len(),
                            layout.total_bytes
                        ),
                    });
                }
            }
            Some(Backing::Mmap(mm))
        } else {
            None
        };

        // If both `index.txt` and `dense.bin` are present, verify that every `dense.bin` row fits.
        if let (Some(ref idx), Some(ref dense)) = (&index, &dense_bin) {
            let dense_len = dense.len() as u64;
            for row in idx.rows() {
                if row.file == PackFileId::Dense && row.src_bytes > 0 {
                    let end = row.src_off.checked_add(row.src_bytes).ok_or_else(|| {
                        GgufError::PackFormat {
                            message: format!("overflow in dense.bin span for '{}'", row.name),
                        }
                    })?;
                    if end > dense_len {
                        return Err(GgufError::PackFormat {
                            message: format!(
                                "tensor '{}' span [{}..{end}) exceeds dense.bin size {dense_len}",
                                row.name, row.src_off
                            ),
                        });
                    }
                }
            }
        }

        Ok(Self {
            pack_dir: dir,
            index,
            experts_layout,
            dense_bin,
            experts_bin,
        })
    }

    /// Pack directory path.
    #[inline]
    pub fn pack_dir(&self) -> &Path {
        &self.pack_dir
    }

    /// Parsed `index.txt` if present.
    #[inline]
    pub fn index(&self) -> Option<&PackIndex> {
        self.index.as_ref()
    }

    /// Parsed `native_experts.txt` if present.
    #[inline]
    pub fn experts_layout(&self) -> Option<&ExpertLayout> {
        self.experts_layout.as_ref()
    }

    /// Raw mapped `dense.bin` slice if present.
    #[inline]
    pub fn dense_bin(&self) -> Option<&[u8]> {
        self.dense_bin.as_deref()
    }

    /// Raw mapped `experts.bin` slice if present.
    #[inline]
    pub fn experts_bin(&self) -> Option<&[u8]> {
        self.experts_bin.as_deref()
    }

    /// Returns the raw byte slice in `dense.bin` for tensor `name`.
    pub fn dense_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let idx = self.index.as_ref().ok_or_else(|| GgufError::PackFormat {
            message: "index.txt not loaded".to_string(),
        })?;
        let row = idx.find(name).ok_or_else(|| GgufError::TensorNotFound {
            name: name.to_string(),
        })?;
        if row.file != PackFileId::Dense {
            return Err(GgufError::PackFormat {
                message: format!(
                    "tensor '{name}' is stored in {}, not dense.bin",
                    row.file.file_name()
                ),
            });
        }
        let dense = self
            .dense_bin
            .as_ref()
            .ok_or_else(|| GgufError::PackFormat {
                message: "dense.bin not present in pack".to_string(),
            })?;
        let start = row.src_off as usize;
        let end = start + (row.src_bytes as usize);
        if end > dense.len() {
            return Err(GgufError::PackFormat {
                message: format!(
                    "tensor '{name}' span [{start}..{end}) exceeds dense.bin size {}",
                    dense.len()
                ),
            });
        }
        Ok(&dense.as_slice()[start..end])
    }

    /// Returns the raw byte slice in `experts.bin` for `(layer, expert)`.
    pub fn expert_blob_bytes(&self, layer: usize, expert: usize) -> Result<&[u8]> {
        let layout = self
            .experts_layout
            .as_ref()
            .ok_or_else(|| GgufError::PackFormat {
                message: "native_experts.txt not loaded".to_string(),
            })?;
        let (off, len) =
            layout
                .blob_offset(layer, expert)
                .ok_or_else(|| GgufError::PackFormat {
                    message: format!("invalid (layer={layer}, expert={expert})"),
                })?;
        let exp_bin = self
            .experts_bin
            .as_ref()
            .ok_or_else(|| GgufError::PackFormat {
                message: "experts.bin not present in pack".to_string(),
            })?;
        let start = off as usize;
        let end = start + len;
        if end > exp_bin.len() {
            return Err(GgufError::PackFormat {
                message: format!(
                    "expert ({layer}, {expert}) span [{start}..{end}) exceeds experts.bin size {}",
                    exp_bin.len()
                ),
            });
        }
        Ok(&exp_bin.as_slice()[start..end])
    }
}
