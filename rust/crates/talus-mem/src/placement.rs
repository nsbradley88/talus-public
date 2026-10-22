//! Unified memory placement policies for Strix Halo (`gfx1151`).
//!
//! Re-exports [`MemoryModel`], [`Placement`], [`PlacementKind`], [`TensorClass`],
//! [`DeviceId`], and [`UnifiedMemoryModel`] from `talus-core-api`, and provides
//! [`StrixUnifiedMemory`] with configurable zero-copy `MmapFile` or `HostMapped`
//! placement for token embeddings and Per-Layer Embedding (PLE) tables.

pub use talus_core_api::{
    DeviceId, MemoryModel, Placement, PlacementKind, TensorClass, UnifiedMemoryModel,
};

use crate::error::{MemError, Result};

/// Total unified physical LPDDR5X capacity on Strix Halo (`128 GiB`).
pub const STRIX_UNIFIED_TOTAL_BYTES: usize = 128 * 1024 * 1024 * 1024;

/// Unified memory model for AMD Ryzen AI Max / Strix Halo (`gfx1151`, 128 GiB shared LPDDR5X).
///
/// Placement policy:
/// - `TensorClass::Embedding` and PLE tables are placed in zero-copy host memory
///   ([`PlacementKind::MmapFile`] by default, or [`PlacementKind::HostMapped`] when configured).
/// - `TensorClass::HostMappedControl` uses [`PlacementKind::HostMapped`] for graph-capturable
///   control words.
/// - Weights (`DenseWeight`, `RoutedExpert`, `SharedExpert`, `OutputHead`), `KvCache`,
///   `RecurrentState`, and `ActivationScratch` use [`PlacementKind::DeviceLocal`] on the
///   configured [`DeviceId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrixUnifiedMemory {
    device: DeviceId,
    embedding_kind: PlacementKind,
    total_unified_bytes: usize,
}

impl StrixUnifiedMemory {
    /// Total unified memory on Strix Halo (`128 GiB`).
    pub const TOTAL_UNIFIED_BYTES: usize = STRIX_UNIFIED_TOTAL_BYTES;

    /// Construct a Strix Halo unified memory model on `device`, placing embeddings and PLE
    /// tables as [`PlacementKind::MmapFile`] by default.
    pub const fn new(device: DeviceId) -> Self {
        Self {
            device,
            embedding_kind: PlacementKind::MmapFile,
            total_unified_bytes: Self::TOTAL_UNIFIED_BYTES,
        }
    }

    /// Construct a Strix Halo unified memory model that places embeddings and PLE tables
    /// as [`PlacementKind::MmapFile`].
    pub const fn with_mmap_embeddings(device: DeviceId) -> Self {
        Self::new(device)
    }

    /// Construct a Strix Halo unified memory model that places embeddings and PLE tables
    /// as [`PlacementKind::HostMapped`] (zero-copy pinned/registered host memory).
    pub const fn with_host_mapped_embeddings(device: DeviceId) -> Self {
        Self {
            device,
            embedding_kind: PlacementKind::HostMapped,
            total_unified_bytes: Self::TOTAL_UNIFIED_BYTES,
        }
    }

    /// Construct a Strix Halo unified memory model with an explicit zero-copy placement kind
    /// (`MmapFile` or `HostMapped`) for embeddings and PLE tables.
    pub fn try_with_embedding_kind(
        device: DeviceId,
        embedding_kind: PlacementKind,
    ) -> Result<Self> {
        match embedding_kind {
            PlacementKind::MmapFile | PlacementKind::HostMapped => Ok(Self {
                device,
                embedding_kind,
                total_unified_bytes: Self::TOTAL_UNIFIED_BYTES,
            }),
            PlacementKind::DeviceLocal => Err(MemError::InvalidLayout {
                reason: "StrixUnifiedMemory embedding placement must be zero-copy (MmapFile or HostMapped)"
                    .to_string(),
            }),
        }
    }

    /// Override the total unified capacity (useful for synthetic tests).
    pub const fn with_total_unified_bytes(mut self, total_unified_bytes: usize) -> Self {
        self.total_unified_bytes = total_unified_bytes;
        self
    }

    /// Configured target [`DeviceId`].
    pub const fn device(&self) -> DeviceId {
        self.device
    }

    /// Configured zero-copy [`PlacementKind`] used for embeddings and PLE tables.
    pub const fn embedding_kind(&self) -> PlacementKind {
        self.embedding_kind
    }

    /// Total unified memory capacity in bytes (`128 GiB` on Strix Halo).
    pub const fn total_unified_bytes(&self) -> usize {
        self.total_unified_bytes
    }

    /// Return the zero-copy placement for a Per-Layer Embedding (PLE) table of `bytes`.
    pub const fn place_ple(&self, _bytes: usize) -> Placement {
        Placement {
            device: self.device,
            kind: self.embedding_kind,
        }
    }

    /// Return the zero-copy placement for a token embedding table of `bytes`.
    pub const fn place_embedding(&self, _bytes: usize) -> Placement {
        Placement {
            device: self.device,
            kind: self.embedding_kind,
        }
    }

    /// Returns `true` if `placement` uses zero-copy host-mapped or memory-mapped file backing.
    pub const fn is_zero_copy(placement: &Placement) -> bool {
        matches!(
            placement.kind,
            PlacementKind::MmapFile | PlacementKind::HostMapped
        )
    }
}

impl Default for StrixUnifiedMemory {
    fn default() -> Self {
        Self::new(DeviceId(0))
    }
}

impl MemoryModel for StrixUnifiedMemory {
    fn place(&self, t: TensorClass, _bytes: usize) -> Placement {
        let kind = match t {
            TensorClass::Embedding => self.embedding_kind,
            TensorClass::HostMappedControl => PlacementKind::HostMapped,
            TensorClass::DenseWeight
            | TensorClass::RoutedExpert
            | TensorClass::SharedExpert
            | TensorClass::OutputHead
            | TensorClass::RecurrentState
            | TensorClass::KvCache
            | TensorClass::ActivationScratch => PlacementKind::DeviceLocal,
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
