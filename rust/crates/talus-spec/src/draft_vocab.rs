//! Subset draft vocabulary mapping (`DraftVocab`, `--spec-draft-vocab`).
//!
//! Supports the `DVOC` binary header format (`magic == 0x434f5644`, `ver == 1`, `n_draft`,
//! `n_vocab`, `ids[n_draft]`) as well as building the full-vocabulary inverse mapping
//! `dv_inv[0..expected_vocab]` initialized to `-1` with `dv_inv[ids[i]] = i`.

use std::fs;
use std::path::Path;

use talus_core_api::{CoreError, DraftVocab as CoreDraftVocab, Result};

/// Little-endian magic bytes `b"DVOC"` (`0x434f5644`).
pub const DVOC_MAGIC: u32 = 0x434f_5644;

/// Supported `DVOC` binary format version (`1`).
pub const DVOC_VERSION: i32 = 1;

/// Draft vocabulary mapping between trimmed draft-head indices (`0..n_draft`) and full target
/// vocabulary token IDs (`0..n_vocab`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DraftVocab {
    /// Number of tokens in the trimmed draft vocabulary.
    pub n_draft: i32,
    /// Number of tokens in the full target vocabulary.
    pub n_vocab: i32,
    /// Forward map: draft index `i in 0..n_draft` -> full target token ID `ids[i]`.
    pub ids: Vec<i32>,
    /// Inverse map: full target token ID `t in 0..n_vocab` -> draft index `0..n_draft`, or `-1` if absent.
    pub dv_inv: Vec<i32>,
}

impl DraftVocab {
    /// Load a `DVOC` binary draft vocabulary file from `path` and validate against `expected_vocab`.
    ///
    /// Binary layout:
    /// - `magic: u32` (`0x434f5644`, `"DVOC"`)
    /// - `ver: i32` (`1`)
    /// - `n_draft: i32` (`> 0`)
    /// - `n_vocab: i32` (must equal `expected_vocab` when `expected_vocab > 0`)
    /// - `ids: [i32; n_draft]` (each in `0..expected_vocab`)
    pub fn load(path: &Path, expected_vocab: i32) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| CoreError::IoError {
            message: format!("failed to read draft vocab '{}': {e}", path.display()),
        })?;
        Self::from_bytes(&bytes, expected_vocab)
    }

    /// Parse a `DVOC`-formatted byte slice and build `ids` and `dv_inv`.
    pub fn from_bytes(bytes: &[u8], expected_vocab: i32) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "draft vocab too short for DVOC header: {} bytes (need at least 16)",
                    bytes.len()
                ),
            });
        }

        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic != DVOC_MAGIC {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "invalid draft vocab magic 0x{magic:08x} (expected DVOC 0x{DVOC_MAGIC:08x})"
                ),
            });
        }

        let ver = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        if ver != DVOC_VERSION {
            return Err(CoreError::InvalidArgument {
                reason: format!("unsupported draft vocab version {ver} (expected {DVOC_VERSION})"),
            });
        }

        let n_draft = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let n_vocab = i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);

        if n_draft <= 0 {
            return Err(CoreError::InvalidArgument {
                reason: format!("invalid n_draft {n_draft} in DVOC header (must be > 0)"),
            });
        }
        if n_vocab <= 0 {
            return Err(CoreError::InvalidArgument {
                reason: format!("invalid n_vocab {n_vocab} in DVOC header (must be > 0)"),
            });
        }
        if expected_vocab > 0 && n_vocab != expected_vocab {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "draft vocab n_vocab mismatch: file has {n_vocab}, expected {expected_vocab}"
                ),
            });
        }

        let target_vocab = if expected_vocab > 0 {
            expected_vocab
        } else {
            n_vocab
        };

        if n_draft > target_vocab {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "draft vocab n_draft ({n_draft}) exceeds target vocab ({target_vocab})"
                ),
            });
        }

        let expected_bytes = 16usize + (n_draft as usize) * 4;
        if bytes.len() != expected_bytes {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "draft vocab byte length mismatch: got {} bytes, expected {expected_bytes}",
                    bytes.len()
                ),
            });
        }

        let mut ids = Vec::with_capacity(n_draft as usize);
        let mut dv_inv = vec![-1i32; target_vocab as usize];

        let payload = &bytes[16..];
        for (i, chunk) in payload.chunks_exact(4).enumerate() {
            let id = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if id < 0 || id >= target_vocab {
                return Err(CoreError::InvalidArgument {
                    reason: format!(
                        "draft vocab token ID {id} at index {i} out of range 0..{target_vocab}"
                    ),
                });
            }
            ids.push(id);
            dv_inv[id as usize] = i as i32;
        }

        Ok(Self {
            n_draft,
            n_vocab: target_vocab,
            ids,
            dv_inv,
        })
    }

    /// Construct a `DraftVocab` from an in-memory slice of full-vocabulary token IDs and `expected_vocab`.
    pub fn from_ids(ids_slice: &[i32], expected_vocab: i32) -> Result<Self> {
        if ids_slice.is_empty() {
            return Err(CoreError::InvalidArgument {
                reason: "draft vocab ids cannot be empty".to_string(),
            });
        }
        if expected_vocab <= 0 || (ids_slice.len() as i32) > expected_vocab {
            return Err(CoreError::InvalidArgument {
                reason: format!(
                    "invalid expected_vocab {expected_vocab} for {} draft ids",
                    ids_slice.len()
                ),
            });
        }
        let mut dv_inv = vec![-1i32; expected_vocab as usize];
        for (i, &id) in ids_slice.iter().enumerate() {
            if id < 0 || id >= expected_vocab {
                return Err(CoreError::InvalidArgument {
                    reason: format!(
                        "draft vocab token ID {id} at index {i} out of range 0..{expected_vocab}"
                    ),
                });
            }
            dv_inv[id as usize] = i as i32;
        }
        Ok(Self {
            n_draft: ids_slice.len() as i32,
            n_vocab: expected_vocab,
            ids: ids_slice.to_vec(),
            dv_inv,
        })
    }

    /// Encode this `DraftVocab` into the `DVOC` binary format (`magic`, `ver = 1`, `n_draft`, `n_vocab`, `ids`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + self.ids.len() * 4);
        buf.extend_from_slice(&DVOC_MAGIC.to_le_bytes());
        buf.extend_from_slice(&DVOC_VERSION.to_le_bytes());
        buf.extend_from_slice(&self.n_draft.to_le_bytes());
        buf.extend_from_slice(&self.n_vocab.to_le_bytes());
        for &id in &self.ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
        buf
    }

    /// Load either a `DVOC`-headered file or a flat headerless `i32` array (`q36_engine.cu` format).
    pub fn load_auto(path: &Path, expected_vocab: i32) -> Result<Self> {
        let bytes = fs::read(path).map_err(|e| CoreError::IoError {
            message: format!("failed to read draft vocab '{}': {e}", path.display()),
        })?;
        if bytes.len() >= 4
            && u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) == DVOC_MAGIC
        {
            return Self::from_bytes(&bytes, expected_vocab);
        }
        let core_dv = talus_gguf::DraftVocabFile::from_bytes_checked(
            &bytes,
            (expected_vocab > 0).then_some(expected_vocab as u32),
        )
        .map_err(|e| CoreError::InvalidArgument {
            reason: e.to_string(),
        })?;
        let ids: Vec<i32> = core_dv.draft_to_target.iter().map(|&t| t as i32).collect();
        let vocab = if expected_vocab > 0 {
            expected_vocab
        } else {
            ids.iter().copied().max().unwrap_or(0) + 1
        };
        Self::from_ids(&ids, vocab)
    }

    /// Number of draft tokens (`n_draft`).
    #[inline]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if no draft vocabulary is loaded (`n_draft == 0`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Map a draft-vocab argmax index `idx` (`0..n_draft`) back to the full vocabulary token ID `ids[idx]`.
    /// Returns `-1` if `idx` is out of bounds.
    #[inline]
    pub fn to_full(&self, idx: i32) -> i32 {
        if idx >= 0 && (idx as usize) < self.ids.len() {
            self.ids[idx as usize]
        } else {
            -1
        }
    }

    /// Checked mapping from draft-vocab index `idx` (`0..n_draft`) to full vocabulary token ID.
    #[inline]
    pub fn map_to_full(&self, idx: i32) -> Option<i32> {
        if idx >= 0 {
            self.ids.get(idx as usize).copied()
        } else {
            None
        }
    }

    /// Alias for [`Self::to_full`].
    #[inline]
    pub fn draft_to_vocab(&self, idx: i32) -> i32 {
        self.to_full(idx)
    }

    /// Map a full vocabulary token ID `t` (`0..n_vocab`) to its draft vocabulary index `dv_inv[t]`,
    /// or `-1` if `t` is not present in the draft vocabulary or out of range.
    #[inline]
    pub fn to_draft(&self, t: i32) -> i32 {
        if t >= 0 && (t as usize) < self.dv_inv.len() {
            self.dv_inv[t as usize]
        } else {
            -1
        }
    }

    /// Checked mapping from full vocabulary token ID `t` to its draft vocabulary index.
    #[inline]
    pub fn map_to_draft(&self, t: i32) -> Option<i32> {
        let d = self.to_draft(t);
        (d >= 0).then_some(d)
    }

    /// Alias for [`Self::to_draft`].
    #[inline]
    pub fn vocab_to_draft(&self, t: i32) -> i32 {
        self.to_draft(t)
    }

    /// Convert to [`talus_core_api::DraftVocab`].
    pub fn to_core(&self) -> CoreDraftVocab {
        CoreDraftVocab::new(self.ids.iter().map(|&id| id as u32).collect())
    }
}
