//! Split-shard GGUF discovery, validation, and unified multi-shard model view (`GgufModel` / `GgufShards`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use talus_core_api::GgufView;

use crate::error::{GgufError, Result};
use crate::gguf::{build_gguf_view, GgufFile, GgufValue, TensorEntry};

/// Location of a tensor within a (possibly multi-shard) [`GgufModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorLocation {
    /// Zero-based index of the shard containing this tensor.
    pub shard_idx: usize,
    /// Byte offset of the tensor relative to the shard's aligned `data_start`.
    pub offset: u64,
    /// Computed byte size of the tensor payload.
    pub byte_size: u64,
}

/// Discovers all shards of a `llama-gguf-split` family from any one shard path.
///
/// Matches `<stem>-NNNNN-of-MMMMM.gguf` (or `<stem>-NNNNN-of-MMMMM.gguf.hdr`) where `NNNNN` and
/// `MMMMM` are 5-digit 1-based numbers (`1 <= NNNNN <= MMMMM`).
/// If `any_path` does not match that naming convention, returns `Ok(vec![any_path])`.
/// If it matches, verifies that every shard `1..=MMMMM` exists on disk and returns them in order
/// (`00001` first).
pub fn gguf_split_paths(any_path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let path = any_path.as_ref();
    let s = path.to_string_lossy();

    let suffix = if s.ends_with(".gguf.hdr") {
        ".gguf.hdr"
    } else if s.ends_with(".gguf") {
        ".gguf"
    } else {
        return Ok(vec![path.to_path_buf()]);
    };

    let without_suffix = &s[..s.len() - suffix.len()];
    // Need at least 1 character of stem plus "-NNNNN-of-MMMMM" (15 bytes).
    if without_suffix.len() < 16 {
        return Ok(vec![path.to_path_buf()]);
    }
    let tail_start = without_suffix.len() - 15;
    let tail = &without_suffix[tail_start..];
    let bytes = tail.as_bytes();
    if bytes[0] != b'-' || &bytes[6..10] != b"-of-" {
        return Ok(vec![path.to_path_buf()]);
    }
    let no_str = &tail[1..6];
    let count_str = &tail[10..15];
    if !no_str.bytes().all(|b| b.is_ascii_digit()) || !count_str.bytes().all(|b| b.is_ascii_digit())
    {
        return Ok(vec![path.to_path_buf()]);
    }

    let no: usize = no_str.parse().unwrap_or(0);
    let count: usize = count_str.parse().unwrap_or(0);
    if no < 1 || count < 1 || no > count {
        return Ok(vec![path.to_path_buf()]);
    }

    let stem = &without_suffix[..tail_start];
    let mut out = Vec::with_capacity(count);
    for i in 1..=count {
        let candidate = PathBuf::from(format!("{stem}-{i:05}-of-{count:05}{suffix}"));
        if !candidate.is_file() {
            return Err(GgufError::MissingShard {
                path: candidate.display().to_string(),
                shard_no: i,
                shard_count: count,
            });
        }
        out.push(candidate);
    }
    Ok(out)
}

/// Unified GGUF reader over one or more split shards (`<stem>-00001-of-0000N.gguf`).
#[derive(Debug)]
pub struct GgufModel {
    shards: Vec<GgufFile>,
    merged_metadata: BTreeMap<String, GgufValue>,
    locations: BTreeMap<String, (usize, usize)>,
}

/// Alias for [`GgufModel`].
pub type GgufShards = GgufModel;

impl GgufModel {
    /// Opens a single GGUF file or automatically discovers and opens all shards of a split GGUF model
    /// from any one shard path.
    pub fn open(any_path: impl AsRef<Path>) -> Result<Self> {
        let paths = gguf_split_paths(any_path)?;
        Self::open_paths(paths)
    }

    /// Opens an explicit sequence of shard paths in order (`shard 0` first).
    pub fn open_paths<I, P>(paths: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut shards = Vec::new();
        for p in paths {
            shards.push(GgufFile::open(p.as_ref())?);
        }
        Self::from_shards(shards)
    }

    /// Constructs a [`GgufModel`] from already-parsed [`GgufFile`] shards (including in-memory shards).
    pub fn from_shards(shards: Vec<GgufFile>) -> Result<Self> {
        if shards.is_empty() {
            return Err(GgufError::InvalidSplit {
                message: "GgufModel requires at least one shard".to_string(),
            });
        }

        let n = shards.len();
        let shard_label = |idx: usize| -> String {
            shards[idx]
                .path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| format!("shard[{idx}]"))
        };

        if n == 1 {
            let split_count = shards[0].get_u64("split.count").unwrap_or(1);
            if split_count > 1 {
                return Err(GgufError::InvalidSplit {
                    message: format!(
                        "{} is shard 1 of {split_count}, but it was opened as a whole model without its sibling shards",
                        shard_label(0)
                    ),
                });
            }
        } else {
            if shards[0].get_str("general.architecture").is_none() {
                return Err(GgufError::InvalidSplit {
                    message: format!(
                        "{} has no general.architecture; the first shard of a split model must carry the model metadata",
                        shard_label(0)
                    ),
                });
            }
            let declared_total = shards[0].get_u64("split.tensors.count");
            let mut actual_total = 0u64;
            for (i, shard) in shards.iter().enumerate() {
                let sc = shard.get_u64("split.count");
                let sno = shard.get_u64("split.no");
                let stc = shard.get_u64("split.tensors.count");
                if sc != Some(n as u64)
                    || sno != Some(i as u64)
                    || (declared_total.is_some() && stc != declared_total)
                {
                    return Err(GgufError::InvalidSplit {
                        message: format!(
                            "{} does not declare itself shard {} of {n} of this model (split.count={:?}, split.no={:?}, split.tensors.count={:?})",
                            shard_label(i),
                            i + 1,
                            sc,
                            sno,
                            stc
                        ),
                    });
                }
                actual_total += shard.tensors().len() as u64;
            }
            if let Some(expected) = declared_total {
                if actual_total != expected {
                    return Err(GgufError::InvalidSplit {
                        message: format!(
                            "the {n} shards hold {actual_total} tensors, but split.tensors.count is {expected}"
                        ),
                    });
                }
            }
        }

        let mut merged_metadata: BTreeMap<String, GgufValue> = BTreeMap::new();
        let mut meta_origin: BTreeMap<String, usize> = BTreeMap::new();
        let mut locations: BTreeMap<String, (usize, usize)> = BTreeMap::new();

        for (s_idx, shard) in shards.iter().enumerate() {
            for (k, v) in shard.metadata() {
                if matches!(
                    k.as_str(),
                    "split.no" | "split.count" | "split.tensors.count"
                ) {
                    if s_idx == 0 {
                        merged_metadata.insert(k.clone(), v.clone());
                        meta_origin.insert(k.clone(), 0);
                    }
                    continue;
                }
                if let Some(prev_val) = merged_metadata.get(k) {
                    if prev_val != v {
                        let first = meta_origin.get(k).copied().unwrap_or(0);
                        return Err(GgufError::ConflictingMetadata {
                            key: k.clone(),
                            first_shard: shard_label(first),
                            second_shard: shard_label(s_idx),
                        });
                    }
                } else {
                    merged_metadata.insert(k.clone(), v.clone());
                    meta_origin.insert(k.clone(), s_idx);
                }
            }

            for (t_idx, t) in shard.tensors().iter().enumerate() {
                if let Some(&(prev_shard, _)) = locations.get(&t.name) {
                    return Err(GgufError::DuplicateTensorName {
                        tensor: t.name.clone(),
                        first_source: shard_label(prev_shard),
                        second_source: shard_label(s_idx),
                    });
                }
                locations.insert(t.name.clone(), (s_idx, t_idx));
            }
        }

        Ok(Self {
            shards,
            merged_metadata,
            locations,
        })
    }

    /// Number of shards in this model.
    #[inline]
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Slice of all constituent [`GgufFile`] shards.
    #[inline]
    pub fn shards(&self) -> &[GgufFile] {
        &self.shards
    }

    /// Reference to the first shard (which carries the model metadata in split GGUFs).
    #[inline]
    pub fn meta_shard(&self) -> &GgufFile {
        &self.shards[0]
    }

    /// Unified metadata map merged across all shards.
    #[inline]
    pub fn metadata(&self) -> &BTreeMap<String, GgufValue> {
        &self.merged_metadata
    }

    /// Looks up a metadata value by key in the unified metadata map.
    #[inline]
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.merged_metadata.get(key)
    }

    /// Total number of tensors across all shards.
    #[inline]
    pub fn tensor_count(&self) -> usize {
        self.locations.len()
    }

    /// Looks up a tensor's [`TensorLocation`] by name.
    pub fn tensor_location(&self, name: &str) -> Option<TensorLocation> {
        let &(shard_idx, tensor_idx) = self.locations.get(name)?;
        let entry = &self.shards[shard_idx].tensors()[tensor_idx];
        Some(TensorLocation {
            shard_idx,
            offset: entry.offset,
            byte_size: entry.byte_size,
        })
    }

    /// Looks up both the [`TensorEntry`] and [`TensorLocation`] for `name`.
    pub fn find_tensor(&self, name: &str) -> Option<(&TensorEntry, TensorLocation)> {
        let &(shard_idx, tensor_idx) = self.locations.get(name)?;
        let entry = &self.shards[shard_idx].tensors()[tensor_idx];
        Some((
            entry,
            TensorLocation {
                shard_idx,
                offset: entry.offset,
                byte_size: entry.byte_size,
            },
        ))
    }

    /// Iterates over all `(shard_idx, &TensorEntry)` pairs in shard order.
    pub fn tensors(&self) -> impl Iterator<Item = (usize, &TensorEntry)> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(s_idx, s)| s.tensors().iter().map(move |t| (s_idx, t)))
    }

    /// Returns the zero-copy payload slice for tensor `name`.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let &(shard_idx, tensor_idx) =
            self.locations
                .get(name)
                .ok_or_else(|| GgufError::TensorNotFound {
                    name: name.to_string(),
                })?;
        let shard = &self.shards[shard_idx];
        let entry = &shard.tensors()[tensor_idx];
        shard.tensor_data(entry)
    }

    /// Verifies offset bracket geometry (`check_brackets`) on every shard.
    pub fn check_brackets(&self) -> Result<()> {
        for shard in &self.shards {
            shard.check_brackets()?;
        }
        Ok(())
    }

    /// Builds a unified [`talus_core_api::GgufView`] over all shards.
    pub fn to_gguf_view(&self) -> GgufView {
        build_gguf_view(&self.merged_metadata, self.locations.len())
    }
}

impl From<&GgufModel> for GgufView {
    fn from(model: &GgufModel) -> Self {
        model.to_gguf_view()
    }
}

impl From<GgufModel> for GgufView {
    fn from(model: GgufModel) -> Self {
        model.to_gguf_view()
    }
}
