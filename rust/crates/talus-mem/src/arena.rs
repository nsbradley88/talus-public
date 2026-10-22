//! Typed offset and region planners (`ScratchArenaPlan` and `SlabPlanner`).
//!
//! Computes 256-byte aligned sub-allocations inside a single contiguous backing buffer for:
//! - Per-step activation and prefill scratch buffers (with scoped/lifetime peak reuse math).
//! - Attention KV cache pools (`KvLayout`: F16, Q8_0, Q4_0, and WMMA transposed V tiles).
//! - GDN / KDA recurrent state and message-start snapshot pools (`RecurrentLayout`).

use std::marker::PhantomData;
use std::ops::Range;

use talus_core_api::{KvCacheFormat, KvLayout, RecurrentLayout};

use crate::error::{MemError, Result};

/// Default sub-allocation alignment in bytes (`256` bytes, required for HIP/GPU coalesced
/// vector loads and sub-buffer bindings).
pub const DEFAULT_ARENA_ALIGN: usize = 256;

/// Round `value` up to the next multiple of `align` (`align` must be a power of two).
pub const fn align_up(value: usize, align: usize) -> Result<usize> {
    if align == 0 || !align.is_power_of_two() {
        return Err(MemError::InvalidLayout {
            reason: String::new(),
        });
    }
    let mask = align - 1;
    match value.checked_add(mask) {
        Some(v) => Ok(v & !mask),
        None => Err(MemError::InvalidLayout {
            reason: String::new(),
        }),
    }
}

fn align_up_msg(value: usize, align: usize, context: &str) -> Result<usize> {
    if align == 0 || !align.is_power_of_two() {
        return Err(MemError::InvalidLayout {
            reason: format!("{context}: alignment {align} is not a non-zero power of two"),
        });
    }
    let mask = align - 1;
    value
        .checked_add(mask)
        .map(|v| v & !mask)
        .ok_or_else(|| MemError::InvalidLayout {
            reason: format!("{context}: overflow aligning {value} to {align} bytes"),
        })
}

/// Identifier of a planned region within a [`ScratchArenaPlan`] or [`SlabPlanner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub usize);

/// Untyped descriptor for a 256-byte aligned sub-allocation within a backing buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRegion {
    /// Region index in the plan.
    pub id: RegionId,
    /// Human-readable name (e.g. `"attn_q"`, `"moe_eparts"`, `"kv_layer_0_k"`).
    pub label: String,
    /// Byte offset from the start of the backing buffer (guaranteed multiple of `align` >= 256).
    pub offset: usize,
    /// Exact requested payload size in bytes (`elem_count * elem_size`).
    pub size_bytes: usize,
    /// Padded size in bytes rounded up to `align` (`>= 256`).
    pub aligned_bytes: usize,
    /// Alignment in bytes (`>= 256`, power of two).
    pub align: usize,
    /// Element size in bytes (`1` for raw byte regions).
    pub elem_size: usize,
    /// Element count.
    pub elem_count: usize,
}

impl PlannedRegion {
    /// Unaligned end offset (`offset + size_bytes`).
    pub const fn end_offset(&self) -> usize {
        self.offset + self.size_bytes
    }

    /// Aligned end offset (`offset + aligned_bytes`).
    pub const fn aligned_end_offset(&self) -> usize {
        self.offset + self.aligned_bytes
    }

    /// Byte range `offset..offset + size_bytes` inside the backing buffer.
    pub const fn range(&self) -> Range<usize> {
        self.offset..self.end_offset()
    }

    /// Aligned byte range `offset..offset + aligned_bytes` inside the backing buffer.
    pub const fn aligned_range(&self) -> Range<usize> {
        self.offset..self.aligned_end_offset()
    }

    /// Check whether this region's aligned byte span overlaps `other`'s aligned byte span.
    pub const fn overlaps(&self, other: &Self) -> bool {
        if self.aligned_bytes == 0 || other.aligned_bytes == 0 {
            return false;
        }
        self.offset < other.aligned_end_offset() && other.offset < self.aligned_end_offset()
    }

    /// Compute the concrete device pointer given `base_ptr` (validates `base_ptr` alignment).
    pub fn resolve_ptr(&self, base_ptr: u64) -> Result<u64> {
        if !(base_ptr as usize).is_multiple_of(self.align) {
            return Err(MemError::InvalidLayout {
                reason: format!(
                    "base pointer 0x{base_ptr:x} is not aligned to {} bytes for region '{}'",
                    self.align, self.label
                ),
            });
        }
        base_ptr
            .checked_add(self.offset as u64)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: format!("pointer overflow resolving region '{}'", self.label),
            })
    }

    /// Borrow this region's byte slice from a host backing buffer.
    pub fn slice<'a>(&self, backing: &'a [u8]) -> Result<&'a [u8]> {
        backing
            .get(self.range())
            .ok_or_else(|| MemError::InvalidLayout {
                reason: format!(
                    "region '{}' ({}..{}) exceeds backing buffer length {}",
                    self.label,
                    self.offset,
                    self.end_offset(),
                    backing.len()
                ),
            })
    }

    /// Mutably borrow this region's byte slice from a host backing buffer.
    pub fn slice_mut<'a>(&self, backing: &'a mut [u8]) -> Result<&'a mut [u8]> {
        let len = backing.len();
        let range = self.range();
        backing
            .get_mut(range)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: format!(
                    "region '{}' ({}..{}) exceeds backing buffer length {}",
                    self.label,
                    self.offset,
                    self.end_offset(),
                    len
                ),
            })
    }
}

/// Typed handle to a 256-byte aligned sub-allocation of `count` elements of type `T`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedRegion<T> {
    region: PlannedRegion,
    _marker: PhantomData<fn() -> T>,
}

impl<T> TypedRegion<T> {
    /// Underlying untyped [`PlannedRegion`].
    pub const fn region(&self) -> &PlannedRegion {
        &self.region
    }

    /// Region identifier.
    pub const fn id(&self) -> RegionId {
        self.region.id
    }

    /// Region label.
    pub fn label(&self) -> &str {
        &self.region.label
    }

    /// 256-byte aligned byte offset from the start of the backing buffer.
    pub const fn offset(&self) -> usize {
        self.region.offset
    }

    /// Number of elements of type `T`.
    pub const fn count(&self) -> usize {
        self.region.elem_count
    }

    /// Exact byte size (`count * size_of::<T>()`).
    pub const fn size_bytes(&self) -> usize {
        self.region.size_bytes
    }

    /// 256-byte aligned byte size.
    pub const fn aligned_bytes(&self) -> usize {
        self.region.aligned_bytes
    }

    /// Resolve the device address of this region from `base_ptr`.
    pub fn resolve_ptr(&self, base_ptr: u64) -> Result<u64> {
        self.region.resolve_ptr(base_ptr)
    }
}

/// Watermark token saved by [`ScratchArenaPlan::mark`] for scoped scratch reuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaMark {
    cursor: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeBlock {
    offset: usize,
    bytes: usize,
}

/// Planner for 256-byte aligned scratch sub-allocations inside a single backing buffer,
/// with support for scoped watermark reset and free-block coalescing to minimize peak memory.
#[derive(Debug, Clone)]
pub struct ScratchArenaPlan {
    base_align: usize,
    cursor: usize,
    peak_bytes: usize,
    total_requested_aligned_bytes: usize,
    regions: Vec<PlannedRegion>,
    active_regions: Vec<bool>,
    free_blocks: Vec<FreeBlock>,
}

impl Default for ScratchArenaPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl ScratchArenaPlan {
    /// Create a new [`ScratchArenaPlan`] with 256-byte minimum alignment (`DEFAULT_ARENA_ALIGN`).
    pub const fn new() -> Self {
        Self {
            base_align: DEFAULT_ARENA_ALIGN,
            cursor: 0,
            peak_bytes: 0,
            total_requested_aligned_bytes: 0,
            regions: Vec::new(),
            active_regions: Vec::new(),
            free_blocks: Vec::new(),
        }
    }

    /// Create a [`ScratchArenaPlan`] with a custom power-of-two alignment (must be `>= 256`).
    pub fn with_alignment(align: usize) -> Result<Self> {
        if align < DEFAULT_ARENA_ALIGN || !align.is_power_of_two() {
            return Err(MemError::InvalidLayout {
                reason: format!(
                    "arena alignment {align} must be a power of two >= {DEFAULT_ARENA_ALIGN}"
                ),
            });
        }
        Ok(Self {
            base_align: align,
            ..Self::new()
        })
    }

    /// Minimum alignment in bytes enforced on all sub-allocations (`>= 256`).
    pub const fn base_alignment(&self) -> usize {
        self.base_align
    }

    /// Current live high-water cursor in bytes.
    pub const fn current_bytes(&self) -> usize {
        self.cursor
    }

    /// Peak backing buffer size in bytes required to hold all planned regions (accounting for reuse).
    pub const fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Total aligned bytes requested across all allocations before reuse.
    pub const fn total_requested_bytes(&self) -> usize {
        self.total_requested_aligned_bytes
    }

    /// Bytes saved by scratch reuse (`total_requested_bytes() - peak_bytes()`).
    pub const fn saved_bytes(&self) -> usize {
        self.total_requested_aligned_bytes
            .saturating_sub(self.peak_bytes)
    }

    /// All regions planned so far in allocation order.
    pub fn regions(&self) -> &[PlannedRegion] {
        &self.regions
    }

    /// Look up a planned region by [`RegionId`].
    pub fn region(&self, id: RegionId) -> Option<&PlannedRegion> {
        self.regions.get(id.0)
    }

    /// Look up the most recently planned region matching `label`.
    pub fn region_by_label(&self, label: &str) -> Option<&PlannedRegion> {
        self.regions.iter().rev().find(|r| r.label == label)
    }

    /// Plan a 256-byte aligned untyped byte region of `bytes`.
    pub fn alloc_bytes(&mut self, label: impl Into<String>, bytes: usize) -> Result<PlannedRegion> {
        self.alloc_raw(label.into(), bytes, 1, bytes, self.base_align)
    }

    /// Plan a byte region with custom power-of-two `align` (enforced to at least `base_alignment()`).
    pub fn alloc_aligned(
        &mut self,
        label: impl Into<String>,
        bytes: usize,
        align: usize,
    ) -> Result<PlannedRegion> {
        if align == 0 || !align.is_power_of_two() {
            return Err(MemError::InvalidLayout {
                reason: format!("alignment {align} is not a non-zero power of two"),
            });
        }
        let effective_align = align.max(self.base_align);
        self.alloc_raw(label.into(), bytes, 1, bytes, effective_align)
    }

    /// Plan a 256-byte aligned typed region of `count` elements of `T`.
    pub fn alloc<T>(&mut self, label: impl Into<String>, count: usize) -> Result<TypedRegion<T>> {
        let label = label.into();
        let elem_size = std::mem::size_of::<T>();
        let elem_align = std::mem::align_of::<T>();
        if elem_size == 0 {
            return Err(MemError::InvalidLayout {
                reason: format!("cannot allocate zero-sized type for region '{label}'"),
            });
        }
        let size_bytes = count
            .checked_mul(elem_size)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: format!("byte size overflow for region '{label}' ({count} * {elem_size})"),
            })?;
        let effective_align = elem_align.max(self.base_align);
        let region = self.alloc_raw(label, size_bytes, elem_size, count, effective_align)?;
        Ok(TypedRegion {
            region,
            _marker: PhantomData,
        })
    }

    fn alloc_raw(
        &mut self,
        label: String,
        size_bytes: usize,
        elem_size: usize,
        elem_count: usize,
        align: usize,
    ) -> Result<PlannedRegion> {
        let aligned_bytes = align_up_msg(size_bytes, align, &label)?;

        let offset = if aligned_bytes == 0 {
            align_up_msg(self.cursor, align, &label)?
        } else if let Some(off) = self.try_take_free_block(aligned_bytes, align)? {
            off
        } else {
            let off = align_up_msg(self.cursor, align, &label)?;
            let new_cursor =
                off.checked_add(aligned_bytes)
                    .ok_or_else(|| MemError::InvalidLayout {
                        reason: format!("arena offset overflow for region '{label}'"),
                    })?;
            self.cursor = new_cursor;
            if self.cursor > self.peak_bytes {
                self.peak_bytes = self.cursor;
            }
            off
        };

        self.total_requested_aligned_bytes = self
            .total_requested_aligned_bytes
            .checked_add(aligned_bytes)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: format!("total requested bytes overflow for region '{label}'"),
            })?;

        let id = RegionId(self.regions.len());
        let region = PlannedRegion {
            id,
            label,
            offset,
            size_bytes,
            aligned_bytes,
            align,
            elem_size,
            elem_count,
        };
        self.regions.push(region.clone());
        self.active_regions.push(true);
        Ok(region)
    }

    fn try_take_free_block(&mut self, aligned_bytes: usize, align: usize) -> Result<Option<usize>> {
        let mut best_idx: Option<usize> = None;
        let mut best_waste = usize::MAX;

        for (i, blk) in self.free_blocks.iter().enumerate() {
            let aligned_start = align_up_msg(blk.offset, align, "free_block")?;
            let pad = aligned_start - blk.offset;
            if blk.bytes >= pad && blk.bytes - pad >= aligned_bytes {
                let waste = (blk.bytes - pad) - aligned_bytes;
                if waste < best_waste {
                    best_waste = waste;
                    best_idx = Some(i);
                    if waste == 0 && pad == 0 {
                        break;
                    }
                }
            }
        }

        let Some(idx) = best_idx else {
            return Ok(None);
        };

        let blk = self.free_blocks.remove(idx);
        let aligned_start = align_up_msg(blk.offset, align, "free_block")?;
        let prefix = aligned_start - blk.offset;
        let suffix_start = aligned_start + aligned_bytes;
        let blk_end = blk.offset + blk.bytes;
        let suffix = blk_end - suffix_start;

        if prefix > 0 {
            self.insert_free_block(FreeBlock {
                offset: blk.offset,
                bytes: prefix,
            });
        }
        if suffix > 0 {
            self.insert_free_block(FreeBlock {
                offset: suffix_start,
                bytes: suffix,
            });
        }

        Ok(Some(aligned_start))
    }

    fn insert_free_block(&mut self, block: FreeBlock) {
        if block.bytes == 0 {
            return;
        }
        self.free_blocks.push(block);
        self.free_blocks.sort_by_key(|b| b.offset);

        // Coalesce adjacent free blocks.
        let mut merged: Vec<FreeBlock> = Vec::with_capacity(self.free_blocks.len());
        for b in self.free_blocks.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.offset + last.bytes == b.offset {
                    last.bytes += b.bytes;
                    continue;
                }
            }
            merged.push(b);
        }
        self.free_blocks = merged;

        // If the last free block touches the top cursor, pop it back into the cursor.
        if let Some(last) = self.free_blocks.last() {
            if last.offset + last.bytes == self.cursor {
                self.cursor = last.offset;
                self.free_blocks.pop();
            }
        }
    }

    /// Release a previously allocated region so its byte range can be reused by subsequent
    /// allocations without increasing `peak_bytes()`.
    pub fn free_region(&mut self, id: RegionId) -> Result<()> {
        let Some(region) = self.regions.get(id.0).cloned() else {
            return Err(MemError::InvalidLayout {
                reason: format!("unknown region id {}", id.0),
            });
        };
        if !self.active_regions[id.0] {
            return Err(MemError::InvalidLayout {
                reason: format!("region '{}' (id {}) already freed", region.label, id.0),
            });
        }
        self.active_regions[id.0] = false;
        self.insert_free_block(FreeBlock {
            offset: region.offset,
            bytes: region.aligned_bytes,
        });
        Ok(())
    }

    /// Record the current arena cursor so temporary sub-allocations can be unwound via
    /// [`ScratchArenaPlan::reset_to`].
    pub const fn mark(&self) -> ArenaMark {
        ArenaMark {
            cursor: self.cursor,
        }
    }

    /// Reset the live allocation cursor back to `mark`, allowing subsequent allocations to
    /// reuse the byte range `[mark.cursor..]` while preserving `peak_bytes()`.
    pub fn reset_to(&mut self, mark: ArenaMark) -> Result<()> {
        if mark.cursor > self.cursor {
            return Err(MemError::InvalidLayout {
                reason: format!(
                    "cannot reset arena cursor to {} above current cursor {}",
                    mark.cursor, self.cursor
                ),
            });
        }
        for (i, r) in self.regions.iter().enumerate() {
            if r.offset >= mark.cursor {
                self.active_regions[i] = false;
            }
        }
        self.free_blocks.retain(|b| b.offset < mark.cursor);
        for b in &mut self.free_blocks {
            if b.offset + b.bytes > mark.cursor {
                b.bytes = mark.cursor - b.offset;
            }
        }
        self.cursor = mark.cursor;
        Ok(())
    }

    /// Execute `f` inside a temporary watermark scope, resetting the live cursor on exit
    /// while updating `peak_bytes()` with the scope's high-water mark.
    pub fn with_scope<F, R>(&mut self, f: F) -> Result<R>
    where
        F: FnOnce(&mut Self) -> Result<R>,
    {
        let mark = self.mark();
        let out = f(self)?;
        self.reset_to(mark)?;
        Ok(out)
    }
}

/// Group of `slot_count` fixed-stride, 256-byte aligned slots inside a [`SlabPlanner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlabGroup {
    pub label: String,
    /// Base byte offset of slot 0 (`base_offset % 256 == 0`).
    pub base_offset: usize,
    /// Number of slots in this slab group.
    pub slot_count: usize,
    /// Unaligned payload size per slot in bytes.
    pub slot_payload_bytes: usize,
    /// 256-byte aligned stride between consecutive slots (`slot_stride % 256 == 0`).
    pub slot_stride_bytes: usize,
    /// Total aligned byte span (`slot_count * slot_stride_bytes`).
    pub total_bytes: usize,
}

impl SlabGroup {
    /// Return the 256-byte aligned byte offset of `slot_idx`.
    pub fn slot_offset(&self, slot_idx: usize) -> Result<usize> {
        if slot_idx >= self.slot_count {
            return Err(MemError::InvalidLayout {
                reason: format!(
                    "slot index {slot_idx} out of bounds for slab '{}' (slot_count={})",
                    self.label, self.slot_count
                ),
            });
        }
        Ok(self.base_offset + slot_idx * self.slot_stride_bytes)
    }

    /// Return the [`PlannedRegion`] corresponding to `slot_idx`.
    pub fn slot_region(&self, slot_idx: usize) -> Result<PlannedRegion> {
        let offset = self.slot_offset(slot_idx)?;
        Ok(PlannedRegion {
            id: RegionId(slot_idx),
            label: format!("{}[{slot_idx}]", self.label),
            offset,
            size_bytes: self.slot_payload_bytes,
            aligned_bytes: self.slot_stride_bytes,
            align: DEFAULT_ARENA_ALIGN,
            elem_size: 1,
            elem_count: self.slot_payload_bytes,
        })
    }
}

/// Planned offsets for a single attention layer's KV cache pools.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayerRegions {
    pub layer_idx: u32,
    /// Padded position capacity for K and V pools (`multiple of kv_pad_stride`).
    pub padded_positions: u32,
    /// Key cache pool region.
    pub k_pool: PlannedRegion,
    /// Value cache pool region.
    pub v_pool: PlannedRegion,
    /// Optional WMMA tile-transposed V pool region (`vt_tile_stride`).
    pub vt_pool: Option<PlannedRegion>,
}

/// Complete multi-layer KV cache pool layout planned inside a [`SlabPlanner`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPoolPlan {
    pub layers: Vec<KvLayerRegions>,
    pub total_bytes: usize,
}

/// Planned layout for a pool of GDN / KDA recurrent state snapshots (`PromptCache`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentSnapshotPoolPlan {
    /// Slab of `num_snapshots` slots.
    pub slab: SlabGroup,
    /// Bytes for `hlast` (`[n_embd]` FP32) within each snapshot slot.
    pub hlast_bytes: usize,
    /// Bytes for convolution history (`[conv_channels, d_conv - 1]` FP32) per recurrent layer.
    pub conv_bytes_per_layer: usize,
    /// Bytes for recurrent state matrix (`[h_v, state_size, state_size]` FP32) per recurrent layer.
    pub ssm_bytes_per_layer: usize,
}

/// Non-fragmenting slab and pool planner for KV caches, recurrent snapshot pools,
/// and contiguous multi-buffer device slabs.
#[derive(Debug, Clone, Default)]
pub struct SlabPlanner {
    arena: ScratchArenaPlan,
    slabs: Vec<SlabGroup>,
}

impl SlabPlanner {
    /// Create a new [`SlabPlanner`] enforcing 256-byte sub-allocation alignment.
    pub const fn new() -> Self {
        Self {
            arena: ScratchArenaPlan::new(),
            slabs: Vec::new(),
        }
    }

    /// Total bytes required for the single backing buffer.
    pub const fn total_bytes(&self) -> usize {
        self.arena.peak_bytes()
    }

    /// Underlying [`ScratchArenaPlan`].
    pub const fn arena(&self) -> &ScratchArenaPlan {
        &self.arena
    }

    /// All uniform slab groups registered in this planner.
    pub fn slabs(&self) -> &[SlabGroup] {
        &self.slabs
    }

    /// Add a single 256-byte aligned named region of `bytes`.
    pub fn add_region(&mut self, label: impl Into<String>, bytes: usize) -> Result<PlannedRegion> {
        self.arena.alloc_bytes(label, bytes)
    }

    /// Add a single 256-byte aligned typed region of `count` elements of `T`.
    pub fn add_typed_region<T>(
        &mut self,
        label: impl Into<String>,
        count: usize,
    ) -> Result<TypedRegion<T>> {
        self.arena.alloc::<T>(label, count)
    }

    /// Add a uniform slab of `slot_count` slots, where each slot has `slot_payload_bytes`
    /// padded up to a 256-byte aligned stride (`slot_stride_bytes % 256 == 0`).
    pub fn add_slab(
        &mut self,
        label: impl Into<String>,
        slot_count: usize,
        slot_payload_bytes: usize,
    ) -> Result<SlabGroup> {
        let label = label.into();
        let slot_stride_bytes = align_up_msg(slot_payload_bytes, DEFAULT_ARENA_ALIGN, &label)?;
        let total_bytes =
            slot_count
                .checked_mul(slot_stride_bytes)
                .ok_or_else(|| MemError::InvalidLayout {
                    reason: format!(
                        "slab '{label}' size overflow ({slot_count} * {slot_stride_bytes})"
                    ),
                })?;
        let region = self.arena.alloc_bytes(label.clone(), total_bytes)?;
        let group = SlabGroup {
            label,
            base_offset: region.offset,
            slot_count,
            slot_payload_bytes,
            slot_stride_bytes,
            total_bytes: region.aligned_bytes,
        };
        self.slabs.push(group.clone());
        Ok(group)
    }

    /// Add a uniform typed slab of `slot_count` slots, each holding `elems_per_slot` of `T`.
    pub fn add_typed_slab<T>(
        &mut self,
        label: impl Into<String>,
        slot_count: usize,
        elems_per_slot: usize,
    ) -> Result<SlabGroup> {
        let label = label.into();
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return Err(MemError::InvalidLayout {
                reason: format!("cannot create zero-sized-type slab '{label}'"),
            });
        }
        let payload =
            elems_per_slot
                .checked_mul(elem_size)
                .ok_or_else(|| MemError::InvalidLayout {
                    reason: format!("slab '{label}' slot payload overflow"),
                })?;
        self.add_slab(label, slot_count, payload)
    }

    /// Plan 256-byte aligned K, V, and optional transposed-V (`vt_tile_stride`) pools
    /// for all attention layers described by `layout`.
    pub fn plan_kv_pools(&mut self, layout: &KvLayout) -> Result<KvPoolPlan> {
        let raw_positions = layout.ring_window.unwrap_or(layout.max_seq_len) as usize;
        let pad_stride = (layout.kv_pad_stride as usize).max(1);
        let padded_positions = if pad_stride.is_power_of_two() {
            align_up_msg(raw_positions, pad_stride, "kv_pad_stride")?
        } else {
            let rem = raw_positions % pad_stride;
            if rem == 0 {
                raw_positions
            } else {
                raw_positions.checked_add(pad_stride - rem).ok_or_else(|| {
                    MemError::InvalidLayout {
                        reason: "kv_pad_stride overflow".to_string(),
                    }
                })?
            }
        };

        let start_bytes = self.total_bytes();
        let mut layers = Vec::with_capacity(layout.num_layers as usize);

        for layer_idx in 0..layout.num_layers {
            let k_elems = (layout.n_head_kv as usize)
                .checked_mul(layout.head_dim_k as usize)
                .and_then(|v| v.checked_mul(padded_positions))
                .ok_or_else(|| MemError::InvalidLayout {
                    reason: format!("KV K pool element overflow at layer {layer_idx}"),
                })?;
            let v_elems = (layout.n_head_kv as usize)
                .checked_mul(layout.head_dim_v as usize)
                .and_then(|v| v.checked_mul(padded_positions))
                .ok_or_else(|| MemError::InvalidLayout {
                    reason: format!("KV V pool element overflow at layer {layer_idx}"),
                })?;

            let k_bytes = kv_format_bytes(layout.format, k_elems)?;
            let v_bytes = kv_format_bytes(layout.format, v_elems)?;

            let k_pool = self
                .arena
                .alloc_bytes(format!("kv.blk.{layer_idx}.k"), k_bytes)?;
            let v_pool = self
                .arena
                .alloc_bytes(format!("kv.blk.{layer_idx}.v"), v_bytes)?;

            let vt_pool = if let Some(tile) = layout.vt_tile_stride {
                let tile_usize = (tile as usize).max(16);
                let rem = padded_positions % tile_usize;
                let vt_positions = if rem == 0 {
                    padded_positions
                } else {
                    padded_positions + (tile_usize - rem)
                };
                let vt_elems = (layout.n_head_kv as usize)
                    .checked_mul(layout.head_dim_v as usize)
                    .and_then(|v| v.checked_mul(vt_positions))
                    .ok_or_else(|| MemError::InvalidLayout {
                        reason: format!("KV VT pool overflow at layer {layer_idx}"),
                    })?;
                let vt_bytes = kv_format_bytes(layout.format, vt_elems)?;
                Some(
                    self.arena
                        .alloc_bytes(format!("kv.blk.{layer_idx}.vt"), vt_bytes)?,
                )
            } else {
                None
            };

            layers.push(KvLayerRegions {
                layer_idx,
                padded_positions: padded_positions as u32,
                k_pool,
                v_pool,
                vt_pool,
            });
        }

        let total_bytes = self.total_bytes() - start_bytes;
        Ok(KvPoolPlan {
            layers,
            total_bytes,
        })
    }

    /// Plan a 256-byte aligned snapshot pool for GDN / KDA recurrent state (`RecurrentLayout`),
    /// matching `Engine::state_bytes()` in `q36_engine.cu`.
    pub fn plan_recurrent_snapshots(
        &mut self,
        layout: &RecurrentLayout,
        n_embd: u32,
        num_snapshots: usize,
    ) -> Result<RecurrentSnapshotPoolPlan> {
        let hlast_bytes = (n_embd as usize)
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| MemError::InvalidLayout {
                reason: "hlast_bytes overflow".to_string(),
            })?;
        let conv_steps = layout.d_conv.saturating_sub(1) as usize;
        let conv_bytes_per_layer = (layout.conv_channels as usize)
            .checked_mul(conv_steps)
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MemError::InvalidLayout {
                reason: "conv_bytes_per_layer overflow".to_string(),
            })?;
        let ssm_bytes_per_layer = (layout.h_v as usize)
            .checked_mul(layout.state_size as usize)
            .and_then(|v| v.checked_mul(layout.state_size as usize))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| MemError::InvalidLayout {
                reason: "ssm_bytes_per_layer overflow".to_string(),
            })?;

        let per_layer = conv_bytes_per_layer
            .checked_add(ssm_bytes_per_layer)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: "per_layer recurrent bytes overflow".to_string(),
            })?;
        let all_layers = per_layer
            .checked_mul(layout.num_layers as usize)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: "all_layers recurrent bytes overflow".to_string(),
            })?;
        let slot_payload_bytes =
            hlast_bytes
                .checked_add(all_layers)
                .ok_or_else(|| MemError::InvalidLayout {
                    reason: "snapshot slot_payload_bytes overflow".to_string(),
                })?;

        let slab = self.add_slab("recurrent_snapshots", num_snapshots, slot_payload_bytes)?;
        Ok(RecurrentSnapshotPoolPlan {
            slab,
            hlast_bytes,
            conv_bytes_per_layer,
            ssm_bytes_per_layer,
        })
    }
}

fn kv_format_bytes(format: KvCacheFormat, elements: usize) -> Result<usize> {
    match format {
        KvCacheFormat::F16 => elements
            .checked_mul(2)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: "F16 KV byte count overflow".to_string(),
            }),
        KvCacheFormat::Q8_0 => {
            let blocks = elements.div_ceil(32);
            blocks
                .checked_mul(34)
                .ok_or_else(|| MemError::InvalidLayout {
                    reason: "Q8_0 KV byte count overflow".to_string(),
                })
        }
        KvCacheFormat::Q4_0 => {
            let blocks = elements.div_ceil(32);
            blocks
                .checked_mul(18)
                .ok_or_else(|| MemError::InvalidLayout {
                    reason: "Q4_0 KV byte count overflow".to_string(),
                })
        }
    }
}
