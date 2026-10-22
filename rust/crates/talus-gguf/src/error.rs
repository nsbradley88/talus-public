//! Error types for `talus-gguf`.

use std::fmt;
use talus_core_api::CoreError;

/// Result alias for `talus-gguf` operations.
pub type Result<T> = std::result::Result<T, GgufError>;

/// Errors produced when parsing or validating GGUF files, split shards, Strata packs,
/// MTP runtime packs, draft vocabularies, or shape catalogs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// I/O or OS mmap failure.
    Io {
        path: Option<String>,
        message: String,
    },
    /// File does not begin with the `"GGUF"` (`0x46554747`) magic bytes.
    InvalidMagic { path: Option<String>, found: u32 },
    /// GGUF format version is not supported (only v2 and v3 are supported).
    UnsupportedVersion { path: Option<String>, version: u32 },
    /// Header, metadata KV table, or tensor directory is truncated or malformed.
    TruncatedHeader {
        path: Option<String>,
        message: String,
    },
    /// Tensor payload extends past the end of the file or overlaps another tensor.
    TruncatedData {
        path: Option<String>,
        tensor: String,
        needed_end: u64,
        file_size: u64,
    },
    /// A string in the GGUF header (key, tensor name, or string value) is not valid UTF-8.
    InvalidUtf8 {
        path: Option<String>,
        context: String,
    },
    /// Unknown or malformed GGUF metadata value type tag.
    InvalidValueType {
        path: Option<String>,
        key: String,
        type_id: u32,
    },
    /// Duplicate metadata key within a single GGUF file.
    DuplicateMetadataKey { path: Option<String>, key: String },
    /// Duplicate tensor name within a file or across split shards.
    DuplicateTensorName {
        tensor: String,
        first_source: String,
        second_source: String,
    },
    /// `general.alignment` is zero or not a power of two.
    InvalidAlignment {
        path: Option<String>,
        alignment: u64,
    },
    /// Tensor dimensions are invalid (e.g., 0 dims, >4 dims, zero-length axis, or not a multiple of block size).
    InvalidTensorDims {
        path: Option<String>,
        tensor: String,
        message: String,
    },
    /// Unknown or unsupported `ggml_type` quantization ID.
    UnsupportedQuantType {
        path: Option<String>,
        tensor: String,
        type_id: u32,
    },
    /// Split-shard numbering, count, or tensor count mismatch.
    InvalidSplit { message: String },
    /// A numbered split shard file is missing on disk.
    MissingShard {
        path: String,
        shard_no: usize,
        shard_count: usize,
    },
    /// Two shards define the same metadata key with conflicting values.
    ConflictingMetadata {
        key: String,
        first_shard: String,
        second_shard: String,
    },
    /// Requested tensor name was not found.
    TensorNotFound { name: String },
    /// Malformed Strata native pack (`index.txt`, `native_experts.txt`, `dense.bin`, or `experts.bin`).
    PackFormat { message: String },
    /// Malformed MTP runtime pack (`dense.txt`, `dense.bin`, or `experts.bin`).
    MtpFormat { message: String },
    /// Malformed draft vocabulary `.bin` file.
    DraftVocabFormat { message: String },
    /// Shape catalog or header validation failure.
    ShapeValidation { message: String },
}

impl GgufError {
    fn prefix(path: &Option<String>) -> String {
        match path {
            Some(p) if !p.is_empty() => format!("{p}: "),
            _ => String::new(),
        }
    }
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(f, "{}I/O error: {message}", Self::prefix(path))
            }
            Self::InvalidMagic { path, found } => {
                write!(
                    f,
                    "{}invalid GGUF magic {found:#010x} (expected 0x46554747 \"GGUF\")",
                    Self::prefix(path)
                )
            }
            Self::UnsupportedVersion { path, version } => {
                write!(
                    f,
                    "{}unsupported GGUF version {version} (expected 2 or 3)",
                    Self::prefix(path)
                )
            }
            Self::TruncatedHeader { path, message } => {
                write!(f, "{}truncated GGUF header: {message}", Self::prefix(path))
            }
            Self::TruncatedData {
                path,
                tensor,
                needed_end,
                file_size,
            } => {
                write!(
                    f,
                    "{}truncated tensor data for '{tensor}': requires {needed_end} bytes but file size is {file_size}",
                    Self::prefix(path)
                )
            }
            Self::InvalidUtf8 { path, context } => {
                write!(f, "{}invalid UTF-8 in {context}", Self::prefix(path))
            }
            Self::InvalidValueType { path, key, type_id } => {
                write!(
                    f,
                    "{}invalid GGUF metadata value type {type_id} for key '{key}'",
                    Self::prefix(path)
                )
            }
            Self::DuplicateMetadataKey { path, key } => {
                write!(
                    f,
                    "{}duplicate GGUF metadata key '{key}'",
                    Self::prefix(path)
                )
            }
            Self::DuplicateTensorName {
                tensor,
                first_source,
                second_source,
            } => {
                if first_source == second_source {
                    write!(f, "{first_source}: duplicate tensor '{tensor}'")
                } else {
                    write!(
                        f,
                        "duplicate tensor '{tensor}' in both {first_source} and {second_source}"
                    )
                }
            }
            Self::InvalidAlignment { path, alignment } => {
                write!(
                    f,
                    "{}invalid general.alignment {alignment} (must be a non-zero power of two)",
                    Self::prefix(path)
                )
            }
            Self::InvalidTensorDims {
                path,
                tensor,
                message,
            } => {
                write!(
                    f,
                    "{}invalid dimensions for tensor '{tensor}': {message}",
                    Self::prefix(path)
                )
            }
            Self::UnsupportedQuantType {
                path,
                tensor,
                type_id,
            } => {
                write!(
                    f,
                    "{}unsupported ggml quant type ID {type_id} for tensor '{tensor}'",
                    Self::prefix(path)
                )
            }
            Self::InvalidSplit { message } => write!(f, "{message}"),
            Self::MissingShard {
                path,
                shard_no,
                shard_count,
            } => {
                write!(
                    f,
                    "missing model shard {path} (shard {shard_no} of {shard_count})"
                )
            }
            Self::ConflictingMetadata {
                key,
                first_shard,
                second_shard,
            } => {
                write!(
                    f,
                    "conflicting metadata key '{key}' between {first_shard} and {second_shard}"
                )
            }
            Self::TensorNotFound { name } => {
                write!(f, "tensor not found: '{name}'")
            }
            Self::PackFormat { message } => write!(f, "strata pack error: {message}"),
            Self::MtpFormat { message } => write!(f, "mtp rt error: {message}"),
            Self::DraftVocabFormat { message } => write!(f, "draft vocab error: {message}"),
            Self::ShapeValidation { message } => write!(f, "shape validation error: {message}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<GgufError> for CoreError {
    fn from(err: GgufError) -> Self {
        match err {
            GgufError::TensorNotFound { name } => CoreError::MissingTensor { name },
            other => CoreError::IoError {
                message: other.to_string(),
            },
        }
    }
}
