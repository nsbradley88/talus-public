//! Draft vocabulary `.bin` reader and writer (`--spec-draft-vocab`).
//!
//! Reads and writes flat little-endian `int32` / `uint32` token ID tables (such as the
//! 65,536-entry FR-Spec / Q36 trimmed LM-head vocabulary) into [`talus_core_api::DraftVocab`].

use std::fs;
use std::path::Path;

use talus_core_api::DraftVocab;

use crate::error::{GgufError, Result};
use crate::mmap::Mmap;

/// Reader and serializer for flat `int32` draft-vocabulary `.bin` files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftVocabFile;

impl DraftVocabFile {
    /// Opens and memory-maps a draft-vocab `.bin` file at `path`, parsing it into [`DraftVocab`].
    pub fn open(path: impl AsRef<Path>) -> Result<DraftVocab> {
        Self::open_checked(path, None)
    }

    /// Opens a draft-vocab `.bin` file at `path` and validates every token ID against
    /// `target_vocab_size` (if provided).
    pub fn open_checked(
        path: impl AsRef<Path>,
        target_vocab_size: Option<u32>,
    ) -> Result<DraftVocab> {
        let path_ref = path.as_ref();
        let mmap = Mmap::open(path_ref)?;
        Self::from_bytes_checked(mmap.as_slice(), target_vocab_size)
    }

    /// Parses a byte slice of little-endian `i32` token IDs into [`DraftVocab`].
    pub fn from_bytes(bytes: &[u8]) -> Result<DraftVocab> {
        Self::from_bytes_checked(bytes, None)
    }

    /// Parses a byte slice of little-endian `i32` token IDs into [`DraftVocab`], optionally
    /// checking that every ID is `< target_vocab_size`.
    pub fn from_bytes_checked(bytes: &[u8], target_vocab_size: Option<u32>) -> Result<DraftVocab> {
        if bytes.is_empty() {
            return Err(GgufError::DraftVocabFormat {
                message: "draft_vocab.bin is empty (must contain at least 1 int32 token ID)"
                    .to_string(),
            });
        }
        if !bytes.len().is_multiple_of(4) {
            return Err(GgufError::DraftVocabFormat {
                message: format!(
                    "draft_vocab.bin byte size {} is not a multiple of 4",
                    bytes.len()
                ),
            });
        }

        let count = bytes.len() / 4;
        let mut ids = Vec::with_capacity(count);
        for (idx, chunk) in bytes.as_chunks::<4>().0.iter().enumerate() {
            let raw = i32::from_le_bytes(*chunk);
            if raw < 0 {
                return Err(GgufError::DraftVocabFormat {
                    message: format!("negative token ID {raw} at draft index {idx}"),
                });
            }
            let tid = raw as u32;
            if let Some(limit) = target_vocab_size {
                if tid >= limit {
                    return Err(GgufError::DraftVocabFormat {
                        message: format!(
                            "token ID {tid} at draft index {idx} is out of range for target vocab size {limit}"
                        ),
                    });
                }
            }
            ids.push(tid);
        }

        Ok(DraftVocab::new(ids))
    }

    /// Encodes a [`DraftVocab`] into little-endian `int32` bytes suitable for `draft_vocab.bin`.
    pub fn to_bytes(vocab: &DraftVocab) -> Result<Vec<u8>> {
        if vocab.draft_to_target.is_empty() {
            return Err(GgufError::DraftVocabFormat {
                message: "cannot serialize an empty DraftVocab".to_string(),
            });
        }
        let mut out = Vec::with_capacity(vocab.draft_to_target.len() * 4);
        for (idx, &tid) in vocab.draft_to_target.iter().enumerate() {
            if i32::try_from(tid).is_err() {
                return Err(GgufError::DraftVocabFormat {
                    message: format!("token ID {tid} at index {idx} exceeds i32::MAX"),
                });
            }
            out.extend_from_slice(&tid.to_le_bytes());
        }
        Ok(out)
    }

    /// Writes a [`DraftVocab`] to `path` as little-endian `int32` token IDs.
    pub fn write(path: impl AsRef<Path>, vocab: &DraftVocab) -> Result<()> {
        let path_ref = path.as_ref();
        let bytes = Self::to_bytes(vocab)?;
        fs::write(path_ref, bytes).map_err(|e| GgufError::Io {
            path: Some(path_ref.display().to_string()),
            message: e.to_string(),
        })
    }
}
