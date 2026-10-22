//! GGUF v2/v3 parser, metadata value types, tensor directory, and quant block geometry.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use talus_core_api::{GgufView, QuantType};

use crate::error::{GgufError, Result};
use crate::mmap::{Backing, Mmap};

/// Little-endian `"GGUF"` magic (`0x46554747`).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Default tensor payload alignment in bytes when `general.alignment` is not specified.
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Maximum tensor rank in GGUF (`GGML_MAX_DIMS`).
pub const GGUF_MAX_DIMS: usize = 4;

/// Maximum plausible array length for a single metadata value before we reject the header as corrupt.
const MAX_ARRAY_LEN: u64 = 1 << 24;

/// Returns `(elements_per_block, bytes_per_block)` for the given [`QuantType`].
#[inline]
pub const fn block_geometry(qtype: QuantType) -> (u64, u64) {
    let (elems, bytes) = qtype.block_geometry();
    (elems as u64, bytes as u64)
}

/// Parses a `ggml_type` name (such as `"Q4_K"`, `"IQ4_XS"`, `"BF16"`, `"Q2_0"`) into [`QuantType`].
pub fn quant_type_from_name(name: &str) -> Option<QuantType> {
    match name {
        "F32" => Some(QuantType::F32),
        "F16" => Some(QuantType::F16),
        "Q4_0" => Some(QuantType::Q4_0),
        "Q4_1" => Some(QuantType::Q4_1),
        "Q5_0" => Some(QuantType::Q5_0),
        "Q5_1" => Some(QuantType::Q5_1),
        "Q8_0" => Some(QuantType::Q8_0),
        "Q8_1" => Some(QuantType::Q8_1),
        "Q2_K" => Some(QuantType::Q2_K),
        "Q3_K" => Some(QuantType::Q3_K),
        "Q4_K" => Some(QuantType::Q4_K),
        "Q5_K" => Some(QuantType::Q5_K),
        "Q6_K" => Some(QuantType::Q6_K),
        "Q8_K" => Some(QuantType::Q8_K),
        "IQ2_XXS" => Some(QuantType::IQ2_XXS),
        "IQ2_XS" => Some(QuantType::IQ2_XS),
        "IQ3_XXS" => Some(QuantType::IQ3_XXS),
        "IQ1_S" => Some(QuantType::IQ1_S),
        "IQ4_NL" => Some(QuantType::IQ4_NL),
        "IQ3_S" => Some(QuantType::IQ3_S),
        "IQ2_S" => Some(QuantType::IQ2_S),
        "IQ4_XS" => Some(QuantType::IQ4_XS),
        "I8" => Some(QuantType::I8),
        "IQ1_M" => Some(QuantType::IQ1_M),
        "BF16" => Some(QuantType::BF16),
        "TQ1_0" => Some(QuantType::TQ1_0),
        "TQ2_0" => Some(QuantType::TQ2_0),
        "MXFP4" => Some(QuantType::MXFP4),
        "NVFP4" => Some(QuantType::NVFP4),
        "Q1_0" => Some(QuantType::Q1_0),
        "Q2_0" => Some(QuantType::Q2_0),
        _ => None,
    }
}

/// Computes the exact byte size of a tensor with quantization type `qtype` and active shape `dims`.
///
/// Validates that:
/// - `1 <= dims.len() <= 4`
/// - Every dimension is `> 0`
/// - `dims[0]` (the contiguous row dimension in ggml) is divisible by `elements_per_block`
/// - Total element and byte counts do not overflow `u64`
pub fn compute_tensor_byte_size(
    qtype: QuantType,
    dims: &[u64],
    path: &Option<String>,
    tensor_name: &str,
) -> Result<u64> {
    if dims.is_empty() || dims.len() > GGUF_MAX_DIMS {
        return Err(GgufError::InvalidTensorDims {
            path: path.clone(),
            tensor: tensor_name.to_string(),
            message: format!("n_dims={} is outside 1..={GGUF_MAX_DIMS}", dims.len()),
        });
    }
    let (block_elems, block_bytes) = block_geometry(qtype);
    if dims[0] == 0 || !dims[0].is_multiple_of(block_elems) {
        return Err(GgufError::InvalidTensorDims {
            path: path.clone(),
            tensor: tensor_name.to_string(),
            message: format!(
                "ne[0]={} is zero or not divisible by {} block size {}",
                dims[0],
                qtype.name(),
                block_elems
            ),
        });
    }
    let mut total_elems = 1u64;
    for (axis, &d) in dims.iter().enumerate() {
        if d == 0 {
            return Err(GgufError::InvalidTensorDims {
                path: path.clone(),
                tensor: tensor_name.to_string(),
                message: format!("ne[{axis}] is zero"),
            });
        }
        total_elems = total_elems
            .checked_mul(d)
            .ok_or_else(|| GgufError::InvalidTensorDims {
                path: path.clone(),
                tensor: tensor_name.to_string(),
                message: "element count overflows u64".to_string(),
            })?;
    }
    let num_blocks = total_elems / block_elems;
    num_blocks
        .checked_mul(block_bytes)
        .ok_or_else(|| GgufError::InvalidTensorDims {
            path: path.clone(),
            tensor: tensor_name.to_string(),
            message: "byte size overflows u64".to_string(),
        })
}

/// Typed GGUF metadata value (`gguf_metadata_value_type`).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// Extracts an unsigned integer value if this is an unsigned integer or non-negative signed integer.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(u64::from(v)),
            Self::U16(v) => Some(u64::from(v)),
            Self::U32(v) => Some(u64::from(v)),
            Self::U64(v) => Some(v),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Extracts a signed 64-bit integer value if representable.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::I8(v) => Some(i64::from(v)),
            Self::I16(v) => Some(i64::from(v)),
            Self::I32(v) => Some(i64::from(v)),
            Self::I64(v) => Some(v),
            Self::U8(v) => Some(i64::from(v)),
            Self::U16(v) => Some(i64::from(v)),
            Self::U32(v) => Some(i64::from(v)),
            Self::U64(v) => i64::try_from(v).ok(),
            _ => None,
        }
    }

    /// Extracts a floating-point value (`F32` widened to `f64`, or `F64`).
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            _ => None,
        }
    }

    /// Extracts a boolean value.
    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// Extracts a string slice if this is [`GgufValue::String`].
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Extracts an array slice if this is [`GgufValue::Array`].
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

/// Parsed tensor directory entry from a GGUF header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    /// Tensor name (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Padded 4-D dimensions (`dims[n_dims..4]` are set to `1`).
    pub dims: [u64; GGUF_MAX_DIMS],
    /// Number of active dimensions (`1..=4`).
    pub n_dims: u32,
    /// Quantization / data type from `talus-core-api`.
    pub qtype: QuantType,
    /// Byte offset relative to the file's aligned `data_start`.
    pub offset: u64,
    /// Computed payload byte size from block geometry and `dims`.
    pub byte_size: u64,
}

impl TensorEntry {
    /// Active shape slice `&self.dims[..self.n_dims as usize]`.
    #[inline]
    pub fn shape(&self) -> &[u64] {
        &self.dims[..self.n_dims as usize]
    }

    /// Total number of logical elements across all dimensions.
    #[inline]
    pub fn elements(&self) -> u64 {
        self.shape().iter().copied().product()
    }

    /// Quantization type accessor.
    #[inline]
    pub const fn quant_type(&self) -> QuantType {
        self.qtype
    }

    /// Returns the raw tensor payload slice from the owning [`GgufFile`].
    #[inline]
    pub fn data<'a>(&self, file: &'a GgufFile) -> Result<&'a [u8]> {
        file.tensor_data(self)
    }
}

/// Cursor over a byte slice for zero-copy GGUF header parsing.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    path: Option<String>,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8], path: Option<String>) -> Self {
        Self { buf, pos: 0, path }
    }

    #[inline]
    fn take(&mut self, n: usize, context: &str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| GgufError::TruncatedHeader {
                path: self.path.clone(),
                message: format!(
                    "overflow reading {n} bytes for {context} at offset {}",
                    self.pos
                ),
            })?;
        if end > self.buf.len() {
            return Err(GgufError::TruncatedHeader {
                path: self.path.clone(),
                message: format!(
                    "unexpected EOF reading {n} bytes for {context} at offset {} (file size {})",
                    self.pos,
                    self.buf.len()
                ),
            });
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    #[inline]
    fn u8(&mut self, context: &str) -> Result<u8> {
        Ok(self.take(1, context)?[0])
    }

    #[inline]
    fn i8(&mut self, context: &str) -> Result<i8> {
        Ok(self.u8(context)? as i8)
    }

    #[inline]
    fn u16(&mut self, context: &str) -> Result<u16> {
        let b = self.take(2, context)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    #[inline]
    fn i16(&mut self, context: &str) -> Result<i16> {
        let b = self.take(2, context)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    #[inline]
    fn u32(&mut self, context: &str) -> Result<u32> {
        let b = self.take(4, context)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    fn i32(&mut self, context: &str) -> Result<i32> {
        let b = self.take(4, context)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    fn u64(&mut self, context: &str) -> Result<u64> {
        let b = self.take(8, context)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    #[inline]
    fn i64(&mut self, context: &str) -> Result<i64> {
        let b = self.take(8, context)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    #[inline]
    fn f32(&mut self, context: &str) -> Result<f32> {
        let b = self.take(4, context)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    #[inline]
    fn f64(&mut self, context: &str) -> Result<f64> {
        let b = self.take(8, context)?;
        Ok(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self, context: &str) -> Result<String> {
        let len_u64 = self.u64(context)?;
        let len = usize::try_from(len_u64).map_err(|_| GgufError::TruncatedHeader {
            path: self.path.clone(),
            message: format!("string length {len_u64} overflows usize in {context}"),
        })?;
        let raw = self.take(len, context)?;
        let s = std::str::from_utf8(raw).map_err(|_| GgufError::InvalidUtf8 {
            path: self.path.clone(),
            context: context.to_string(),
        })?;
        Ok(s.to_string())
    }

    fn value(&mut self, key: &str, type_id: u32, depth: u32) -> Result<GgufValue> {
        match type_id {
            0 => Ok(GgufValue::U8(self.u8(key)?)),
            1 => Ok(GgufValue::I8(self.i8(key)?)),
            2 => Ok(GgufValue::U16(self.u16(key)?)),
            3 => Ok(GgufValue::I16(self.i16(key)?)),
            4 => Ok(GgufValue::U32(self.u32(key)?)),
            5 => Ok(GgufValue::I32(self.i32(key)?)),
            6 => Ok(GgufValue::F32(self.f32(key)?)),
            7 => {
                let raw = self.u8(key)?;
                Ok(GgufValue::Bool(raw != 0))
            }
            8 => Ok(GgufValue::String(self.string(key)?)),
            9 => {
                if depth > 2 {
                    return Err(GgufError::InvalidValueType {
                        path: self.path.clone(),
                        key: key.to_string(),
                        type_id,
                    });
                }
                let elem_type = self.u32(key)?;
                let count = self.u64(key)?;
                if count > MAX_ARRAY_LEN {
                    return Err(GgufError::TruncatedHeader {
                        path: self.path.clone(),
                        message: format!(
                            "metadata array '{key}' count {count} exceeds maximum {MAX_ARRAY_LEN}"
                        ),
                    });
                }
                let cap = usize::try_from(count).unwrap_or(0);
                let mut items = Vec::with_capacity(cap.min(65536));
                for _ in 0..count {
                    items.push(self.value(key, elem_type, depth + 1)?);
                }
                Ok(GgufValue::Array(items))
            }
            10 => Ok(GgufValue::U64(self.u64(key)?)),
            11 => Ok(GgufValue::I64(self.i64(key)?)),
            12 => Ok(GgufValue::F64(self.f64(key)?)),
            _ => Err(GgufError::InvalidValueType {
                path: self.path.clone(),
                key: key.to_string(),
                type_id,
            }),
        }
    }
}

/// Parsed GGUF v2/v3 file backed by either a zero-copy OS `mmap` or an in-memory buffer.
#[derive(Debug)]
pub struct GgufFile {
    path: Option<PathBuf>,
    backing: Backing,
    logical_file_size: u64,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: u64,
    header_bytes: u64,
    data_start: u64,
    metadata: BTreeMap<String, GgufValue>,
    tensors: Vec<TensorEntry>,
    tensor_index: BTreeMap<String, usize>,
}

impl GgufFile {
    /// Opens and memory-maps a GGUF file at `path`, parsing its header and validating all tensor bounds.
    ///
    /// If `path` is a `.hdr` header dump accompanied by a `.size` file (as in `matrix/shapes/hdr/`),
    /// the companion `.size` is used as the logical file size for tensor offset and bracket validation.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path_ref = path.as_ref();
        let mmap = Mmap::open(path_ref)?;
        let logical_size = companion_size_for_hdr(path_ref);
        Self::parse_backing(
            Some(path_ref.to_path_buf()),
            Backing::Mmap(mmap),
            logical_size,
        )
    }

    /// Opens a header-only GGUF dump (such as `*.gguf.hdr`) with an optional explicit `logical_file_size`.
    ///
    /// If `logical_file_size` is `None`, checks for a companion `.size` file on disk.
    pub fn open_header(path: impl AsRef<Path>, logical_file_size: Option<u64>) -> Result<Self> {
        let path_ref = path.as_ref();
        let mmap = Mmap::open(path_ref)?;
        let logical = logical_file_size.or_else(|| companion_size_for_hdr(path_ref));
        Self::parse_backing(Some(path_ref.to_path_buf()), Backing::Mmap(mmap), logical)
    }

    /// Parses a GGUF file from an in-memory byte slice or vector, validating all tensor bounds.
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        Self::parse_backing(None, Backing::Owned(bytes.into()), None)
    }

    /// Parses a GGUF header from in-memory bytes with an optional logical full-file byte size.
    pub fn from_header_bytes(
        bytes: impl Into<Vec<u8>>,
        logical_file_size: Option<u64>,
    ) -> Result<Self> {
        Self::parse_backing(None, Backing::Owned(bytes.into()), logical_file_size)
    }

    fn parse_backing(
        path: Option<PathBuf>,
        backing: Backing,
        logical_file_size: Option<u64>,
    ) -> Result<Self> {
        let display_path = path.as_ref().map(|p| p.display().to_string());
        let slice = backing.as_slice();
        let actual_size = slice.len() as u64;
        let file_size = logical_file_size.unwrap_or(actual_size);

        let mut c = Cursor::new(slice, display_path.clone());
        let magic = c.u32("GGUF magic")?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic {
                path: display_path,
                found: magic,
            });
        }

        let version = c.u32("GGUF version")?;
        if version != 2 && version != 3 {
            return Err(GgufError::UnsupportedVersion {
                path: display_path,
                version,
            });
        }

        let tensor_count = c.u64("tensor_count")?;
        let metadata_kv_count = c.u64("metadata_kv_count")?;

        if tensor_count > (1 << 20) || metadata_kv_count > (1 << 20) {
            return Err(GgufError::TruncatedHeader {
                path: display_path,
                message: format!(
                    "implausible counts: tensor_count={tensor_count}, metadata_kv_count={metadata_kv_count}"
                ),
            });
        }

        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_kv_count {
            let key = c.string("metadata key")?;
            let val_type = c.u32(&key)?;
            let val = c.value(&key, val_type, 0)?;
            if metadata.insert(key.clone(), val).is_some() {
                return Err(GgufError::DuplicateMetadataKey {
                    path: display_path,
                    key,
                });
            }
        }

        let mut alignment = GGUF_DEFAULT_ALIGNMENT;
        if let Some(align_val) = metadata.get("general.alignment") {
            let a = align_val
                .as_u64()
                .ok_or_else(|| GgufError::InvalidAlignment {
                    path: display_path.clone(),
                    alignment: 0,
                })?;
            if a == 0 || !a.is_power_of_two() {
                return Err(GgufError::InvalidAlignment {
                    path: display_path,
                    alignment: a,
                });
            }
            alignment = a;
        }

        let source_label = display_path
            .clone()
            .unwrap_or_else(|| "<memory>".to_string());
        let mut tensors = Vec::with_capacity(usize::try_from(tensor_count).unwrap_or(0));
        let mut tensor_index = BTreeMap::new();

        for _ in 0..tensor_count {
            let name = c.string("tensor name")?;
            if tensor_index.contains_key(&name) {
                return Err(GgufError::DuplicateTensorName {
                    tensor: name,
                    first_source: source_label.clone(),
                    second_source: source_label,
                });
            }
            let n_dims = c.u32(&name)?;
            if n_dims == 0 || (n_dims as usize) > GGUF_MAX_DIMS {
                return Err(GgufError::InvalidTensorDims {
                    path: display_path,
                    tensor: name,
                    message: format!("n_dims={n_dims} is outside 1..={GGUF_MAX_DIMS}"),
                });
            }
            let mut dims = [1u64; GGUF_MAX_DIMS];
            for d in dims.iter_mut().take(n_dims as usize) {
                *d = c.u64(&name)?;
            }
            let type_id = c.u32(&name)?;
            let qtype = QuantType::from_ggml_type_id(type_id).ok_or_else(|| {
                GgufError::UnsupportedQuantType {
                    path: display_path.clone(),
                    tensor: name.clone(),
                    type_id,
                }
            })?;
            let offset = c.u64(&name)?;
            let byte_size =
                compute_tensor_byte_size(qtype, &dims[..n_dims as usize], &display_path, &name)?;

            let idx = tensors.len();
            tensor_index.insert(name.clone(), idx);
            tensors.push(TensorEntry {
                name,
                dims,
                n_dims,
                qtype,
                offset,
                byte_size,
            });
        }

        let header_bytes = c.pos as u64;
        let mut data_start =
            header_bytes
                .checked_add(alignment - 1)
                .ok_or_else(|| GgufError::TruncatedHeader {
                    path: display_path.clone(),
                    message: "data_start alignment calculation overflowed".to_string(),
                })?
                / alignment
                * alignment;

        if data_start > file_size {
            if tensor_count == 0 {
                // Metadata-only first shards (e.g. Unsloth split shard 00001) have no payload section
                // and end immediately after the last metadata KV entry without padding to `alignment`.
                data_start = file_size;
            } else {
                return Err(GgufError::TruncatedHeader {
                    path: display_path,
                    message: format!(
                        "tensor data section starts at {data_start}, past file size {file_size}"
                    ),
                });
            }
        }

        // Validate every tensor's payload bounds against `file_size` and ensure no overlapping spans.
        for t in &tensors {
            let start =
                data_start
                    .checked_add(t.offset)
                    .ok_or_else(|| GgufError::TruncatedData {
                        path: display_path.clone(),
                        tensor: t.name.clone(),
                        needed_end: u64::MAX,
                        file_size,
                    })?;
            let end = start
                .checked_add(t.byte_size)
                .ok_or_else(|| GgufError::TruncatedData {
                    path: display_path.clone(),
                    tensor: t.name.clone(),
                    needed_end: u64::MAX,
                    file_size,
                })?;
            if end > file_size {
                return Err(GgufError::TruncatedData {
                    path: display_path,
                    tensor: t.name.clone(),
                    needed_end: end,
                    file_size,
                });
            }
        }

        if tensors.len() > 1 {
            let mut order: Vec<usize> = (0..tensors.len()).collect();
            order.sort_by_key(|&i| tensors[i].offset);
            for w in order.windows(2) {
                let a = &tensors[w[0]];
                let b = &tensors[w[1]];
                if a.offset + a.byte_size > b.offset {
                    return Err(GgufError::TruncatedData {
                        path: display_path,
                        tensor: a.name.clone(),
                        needed_end: data_start + a.offset + a.byte_size,
                        file_size: data_start + b.offset,
                    });
                }
            }
        }

        Ok(Self {
            path,
            backing,
            logical_file_size: file_size,
            version,
            tensor_count,
            metadata_kv_count,
            alignment,
            header_bytes,
            data_start,
            metadata,
            tensors,
            tensor_index,
        })
    }

    /// Path of the file if opened from disk.
    #[inline]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// GGUF format version (`2` or `3`).
    #[inline]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Number of tensors declared in this file.
    #[inline]
    pub const fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Number of metadata key-value pairs declared in this file.
    #[inline]
    pub const fn metadata_kv_count(&self) -> u64 {
        self.metadata_kv_count
    }

    /// Tensor data alignment in bytes (`general.alignment`, default `32`).
    #[inline]
    pub const fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Unaligned byte length of the header + tensor info directory.
    #[inline]
    pub const fn header_bytes(&self) -> u64 {
        self.header_bytes
    }

    /// Aligned byte offset where tensor payload data begins.
    #[inline]
    pub const fn data_start(&self) -> u64 {
        self.data_start
    }

    /// Logical file size in bytes (mapped length for full files, or `.size` value for header dumps).
    #[inline]
    pub const fn file_size(&self) -> u64 {
        self.logical_file_size
    }

    /// Actual mapped or in-memory slice length in bytes.
    #[inline]
    pub fn mapped_len(&self) -> usize {
        self.backing.len()
    }

    /// Underlying mapped or in-memory bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.backing.as_slice()
    }

    /// All parsed metadata key-value pairs.
    #[inline]
    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.metadata
    }

    /// Looks up a metadata value by key.
    #[inline]
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    /// Extracts an unsigned/non-negative integer metadata value by key.
    #[inline]
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(GgufValue::as_u64)
    }

    /// Extracts a signed integer metadata value by key.
    #[inline]
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.metadata.get(key).and_then(GgufValue::as_i64)
    }

    /// Extracts a floating-point metadata value by key.
    #[inline]
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.metadata.get(key).and_then(GgufValue::as_f64)
    }

    /// Extracts a boolean metadata value by key.
    #[inline]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key).and_then(GgufValue::as_bool)
    }

    /// Extracts a string metadata value by key.
    #[inline]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(GgufValue::as_str)
    }

    /// Slice of all tensor directory entries in header order.
    #[inline]
    pub fn tensors(&self) -> &[TensorEntry] {
        &self.tensors
    }

    /// Looks up a tensor entry by name.
    #[inline]
    pub fn find_tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Returns `true` if `entry`'s payload bytes are fully contained in the mapped buffer.
    #[inline]
    pub fn in_bounds(&self, entry: &TensorEntry) -> bool {
        let Some(start) = self.data_start.checked_add(entry.offset) else {
            return false;
        };
        let Some(end) = start.checked_add(entry.byte_size) else {
            return false;
        };
        end <= self.backing.len() as u64
    }

    /// Returns the zero-copy byte slice for `entry`'s tensor data.
    pub fn tensor_data(&self, entry: &TensorEntry) -> Result<&[u8]> {
        let display_path = self.path.as_ref().map(|p| p.display().to_string());
        let start =
            self.data_start
                .checked_add(entry.offset)
                .ok_or_else(|| GgufError::TruncatedData {
                    path: display_path.clone(),
                    tensor: entry.name.clone(),
                    needed_end: u64::MAX,
                    file_size: self.backing.len() as u64,
                })?;
        let end = start
            .checked_add(entry.byte_size)
            .ok_or_else(|| GgufError::TruncatedData {
                path: display_path.clone(),
                tensor: entry.name.clone(),
                needed_end: u64::MAX,
                file_size: self.backing.len() as u64,
            })?;
        let actual_len = self.backing.len() as u64;
        if end > actual_len {
            return Err(GgufError::TruncatedData {
                path: display_path,
                tensor: entry.name.clone(),
                needed_end: end,
                file_size: actual_len,
            });
        }
        Ok(&self.backing.as_slice()[start as usize..end as usize])
    }

    /// Looks up a tensor by name and returns its zero-copy payload byte slice.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let entry = self
            .find_tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound {
                name: name.to_string(),
            })?;
        self.tensor_data(entry)
    }

    /// Verifies that every tensor's `[offset, next_offset)` gap in the sorted offset order
    /// matches its computed `byte_size` up to `alignment - 1` bytes of padding (`nb <= gap < nb + alignment`).
    pub fn check_brackets(&self) -> Result<()> {
        if self.tensors.is_empty() {
            return Ok(());
        }
        let display_path = self.path.as_ref().map(|p| p.display().to_string());
        if self.logical_file_size < self.data_start {
            return Err(GgufError::TruncatedHeader {
                path: display_path,
                message: "file_size < data_start".to_string(),
            });
        }
        let payload = self.logical_file_size - self.data_start;
        let mut order: Vec<usize> = (0..self.tensors.len()).collect();
        order.sort_by_key(|&i| self.tensors[i].offset);

        for (pos, &idx) in order.iter().enumerate() {
            let t = &self.tensors[idx];
            let next_off = if pos + 1 < order.len() {
                self.tensors[order[pos + 1]].offset
            } else {
                payload
            };
            if next_off < t.offset {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "tensor '{}' offset {} exceeds next offset {}",
                        t.name, t.offset, next_off
                    ),
                });
            }
            let gap = next_off - t.offset;
            if gap < t.byte_size || gap >= t.byte_size + self.alignment {
                return Err(GgufError::ShapeValidation {
                    message: format!(
                        "tensor '{}' bracket mismatch: byte_size={} but offset gap={} (alignment={})",
                        t.name, t.byte_size, gap, self.alignment
                    ),
                });
            }
        }
        Ok(())
    }

    /// Converts this file's metadata and tensor count into a [`talus_core_api::GgufView`].
    pub fn to_gguf_view(&self) -> GgufView {
        build_gguf_view(&self.metadata, self.tensors.len())
    }
}

impl From<&GgufFile> for GgufView {
    fn from(file: &GgufFile) -> Self {
        file.to_gguf_view()
    }
}

impl From<GgufFile> for GgufView {
    fn from(file: GgufFile) -> Self {
        file.to_gguf_view()
    }
}

/// Builds a [`talus_core_api::GgufView`] from a metadata map and total tensor count.
pub(crate) fn build_gguf_view(
    metadata: &BTreeMap<String, GgufValue>,
    tensor_count: usize,
) -> GgufView {
    let architecture = metadata
        .get("general.architecture")
        .and_then(GgufValue::as_str)
        .unwrap_or_default()
        .to_string();

    let mut metadata_u64 = BTreeMap::new();
    let mut metadata_f64 = BTreeMap::new();
    let mut metadata_str = BTreeMap::new();

    for (k, v) in metadata {
        if let Some(u) = v.as_u64() {
            metadata_u64.insert(k.clone(), u);
        } else if let Some(b) = v.as_bool() {
            metadata_u64.insert(k.clone(), u64::from(b));
        } else if let Some(f) = v.as_f64() {
            metadata_f64.insert(k.clone(), f);
        } else if let Some(s) = v.as_str() {
            metadata_str.insert(k.clone(), s.to_string());
        }
    }

    GgufView {
        architecture,
        metadata_u64,
        metadata_f64,
        metadata_str,
        tensor_count,
    }
}

/// If `path` is `<stem>.hdr` and `<stem>.size` exists alongside it, reads the logical byte size.
pub(crate) fn companion_size_for_hdr(path: &Path) -> Option<u64> {
    let s = path.to_str()?;
    let prefix = s.strip_suffix(".hdr")?;
    let size_path = PathBuf::from(format!("{prefix}.size"));
    let text = fs::read_to_string(size_path).ok()?;
    text.trim().parse::<u64>().ok()
}
