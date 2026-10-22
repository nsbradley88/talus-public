//! Unified memory `Budget` accountant enforcing the `MemAvailable` safety floor.
//!
//! Enforces:
//! - Default `8 GiB` `MemAvailable` floor (`DEFAULT_FLOOR_BYTES`, Spec §3:
//!   "Keep Strix Halo MemAvailable above 8 GiB at all times").
//! - Configurable higher floors such as `11 GiB` for MiMo (`MIMO_FLOOR_BYTES`, Spec §5 WP-3.3).
//! - Optional total budget ceiling (`max_budget_bytes`).
//! - Per-category named reservation tracking (`Weights`, `KvCache`, `RecurrentState`,
//!   `PromptSnapshots`, `KvStash`, `Scratch`, `Drafter`), peak usage accounting, and
//!   RAII [`ReservationGuard`] release.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use talus_core_api::TensorClass;

use crate::error::{MemError, Result};
use crate::probe::{HostMemProbe, LinuxProcMemProbe};

/// 1 GiB in bytes (`1024^3`).
pub const GIB: usize = 1024 * 1024 * 1024;

/// 1 MiB in bytes (`1024^2`).
pub const MIB: usize = 1024 * 1024;

/// 1 KiB in bytes (`1024`).
pub const KIB: usize = 1024;

/// Default hard `MemAvailable` floor on Strix Halo (`8 GiB`, Spec §3).
pub const DEFAULT_FLOOR_BYTES: usize = 8 * GIB;

/// Model-specific `MemAvailable` floor for MiMo (`11 GiB`, Spec §5 WP-3.3).
pub const MIMO_FLOOR_BYTES: usize = 11 * GIB;

/// Allocation category tracked by [`Budget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BudgetCategory {
    /// Dense, shared-expert, routed-expert, embedding, and output-head weights.
    Weights,
    /// Attention KV cache pools (F16, Q8_0, Q4_0, transposed V tiles, key bounds).
    KvCache,
    /// Live GDN / KDA recurrent state and convolution histories.
    RecurrentState,
    /// Saved recurrent / message-start prompt snapshots (`PromptCache`).
    PromptSnapshots,
    /// Saved KV cache stashes for multi-conversation reuse.
    KvStash,
    /// Activation, prefill, routing, and host-mapped control scratch buffers.
    Scratch,
    /// Speculative drafter weights and state (`MTP`, `DFlash`, draft vocab).
    Drafter,
}

impl BudgetCategory {
    /// All 7 budget categories in canonical declaration order.
    pub const ALL: [Self; 7] = [
        Self::Weights,
        Self::KvCache,
        Self::RecurrentState,
        Self::PromptSnapshots,
        Self::KvStash,
        Self::Scratch,
        Self::Drafter,
    ];

    /// Short canonical category name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Weights => "Weights",
            Self::KvCache => "KvCache",
            Self::RecurrentState => "RecurrentState",
            Self::PromptSnapshots => "PromptSnapshots",
            Self::KvStash => "KvStash",
            Self::Scratch => "Scratch",
            Self::Drafter => "Drafter",
        }
    }

    /// Map a [`TensorClass`] from `talus-core-api` to its primary [`BudgetCategory`].
    pub const fn from_tensor_class(class: TensorClass) -> Self {
        match class {
            TensorClass::DenseWeight
            | TensorClass::RoutedExpert
            | TensorClass::SharedExpert
            | TensorClass::Embedding
            | TensorClass::OutputHead => Self::Weights,
            TensorClass::RecurrentState => Self::RecurrentState,
            TensorClass::KvCache => Self::KvCache,
            TensorClass::ActivationScratch | TensorClass::HostMappedControl => Self::Scratch,
        }
    }
}

impl fmt::Display for BudgetCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Unique identifier for an active [`Reservation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReservationId(pub u64);

impl fmt::Display for ReservationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "res#{}", self.0)
    }
}

/// Handle representing a successful memory reservation in a [`Budget`].
///
/// Can be released explicitly via [`Budget::release`] or wrapped in an RAII
/// [`ReservationGuard`] via [`Reservation::into_guard`] / [`Budget::try_reserve_guard`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reservation {
    pub id: ReservationId,
    pub category: BudgetCategory,
    pub label: String,
    pub bytes: usize,
    /// `true` once the physical allocation has been materialized in OS memory
    /// (so live `/proc/meminfo` `MemAvailable` already reflects the allocation).
    pub materialized: bool,
}

impl Reservation {
    /// Wrap this reservation in an RAII [`ReservationGuard`] that releases it when dropped.
    pub fn into_guard(self, budget: &Budget) -> ReservationGuard {
        ReservationGuard::new(budget.clone(), self)
    }
}

/// RAII guard that automatically releases its [`Reservation`] back to the [`Budget`] on drop.
#[derive(Debug)]
pub struct ReservationGuard {
    budget: Budget,
    reservation: Option<Reservation>,
}

impl ReservationGuard {
    /// Create a new RAII guard owning `reservation` inside `budget`.
    pub fn new(budget: Budget, reservation: Reservation) -> Self {
        Self {
            budget,
            reservation: Some(reservation),
        }
    }

    /// Inspect the underlying [`Reservation`].
    pub fn reservation(&self) -> &Reservation {
        self.reservation
            .as_ref()
            .expect("ReservationGuard is active until drop or into_reservation")
    }

    /// Unique [`ReservationId`].
    pub fn id(&self) -> ReservationId {
        self.reservation().id
    }

    /// Category of the reservation.
    pub fn category(&self) -> BudgetCategory {
        self.reservation().category
    }

    /// Human-readable label of the reservation.
    pub fn label(&self) -> &str {
        &self.reservation().label
    }

    /// Reserved size in bytes.
    pub fn bytes(&self) -> usize {
        self.reservation().bytes
    }

    /// Disarm the guard and return the inner [`Reservation`] without releasing it.
    pub fn into_reservation(mut self) -> Reservation {
        self.reservation
            .take()
            .expect("ReservationGuard already consumed")
    }

    /// Explicitly release the reservation immediately and return the freed byte count.
    pub fn release(mut self) -> usize {
        if let Some(res) = self.reservation.take() {
            self.budget.release(res)
        } else {
            0
        }
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if let Some(res) = self.reservation.take() {
            let _ = self.budget.release(res);
        }
    }
}

/// Configuration for a [`Budget`] accountant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetConfig {
    /// Hard minimum `MemAvailable` floor in bytes (default: [`DEFAULT_FLOOR_BYTES`] = `8 GiB`).
    pub floor_bytes: usize,
    /// Optional cap on total bytes reserved across all categories.
    pub max_budget_bytes: Option<usize>,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            floor_bytes: DEFAULT_FLOOR_BYTES,
            max_budget_bytes: None,
        }
    }
}

impl BudgetConfig {
    /// Configuration with a custom `MemAvailable` floor and no extra total cap.
    pub const fn with_floor(floor_bytes: usize) -> Self {
        Self {
            floor_bytes,
            max_budget_bytes: None,
        }
    }

    /// Configuration for MiMo (`11 GiB` `MemAvailable` floor, Spec §5 WP-3.3).
    pub const fn mimo() -> Self {
        Self {
            floor_bytes: MIMO_FLOOR_BYTES,
            max_budget_bytes: None,
        }
    }

    /// Set an explicit ceiling on total reserved bytes across all categories.
    pub const fn with_max_budget(mut self, max_budget_bytes: usize) -> Self {
        self.max_budget_bytes = Some(max_budget_bytes);
        self
    }
}

/// Snapshot of [`Budget`] state: per-category totals, peak usage, and remaining headroom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetSummary {
    /// Active reserved bytes per category (all 7 categories present).
    pub category_totals: BTreeMap<BudgetCategory, usize>,
    /// Historical peak reserved bytes per category (all 7 categories present).
    pub category_peaks: BTreeMap<BudgetCategory, usize>,
    /// Sum of active reserved bytes across all categories.
    pub total_reserved_bytes: usize,
    /// Sum of active reservations not yet marked materialized in OS `MemAvailable`.
    pub unmaterialized_reserved_bytes: usize,
    /// Peak total reserved bytes observed over the lifetime of the [`Budget`].
    pub peak_reserved_bytes: usize,
    /// Configured `MemAvailable` safety floor in bytes.
    pub floor_bytes: usize,
    /// Optional total budget ceiling in bytes.
    pub max_budget_bytes: Option<usize>,
    /// Live `MemAvailable` bytes reported by the [`HostMemProbe`].
    pub mem_available_bytes: usize,
    /// Effective available bytes (`mem_available_bytes - unmaterialized_reserved_bytes`).
    pub effective_available_bytes: usize,
    /// Remaining allocatable headroom in bytes above `floor_bytes` (and within `max_budget_bytes`).
    pub headroom_bytes: usize,
    /// Number of currently active reservations.
    pub active_reservation_count: usize,
}

impl BudgetSummary {
    /// Return the currently reserved bytes for `category`.
    pub fn category_bytes(&self, category: BudgetCategory) -> usize {
        self.category_totals.get(&category).copied().unwrap_or(0)
    }

    /// Return the peak reserved bytes for `category`.
    pub fn category_peak_bytes(&self, category: BudgetCategory) -> usize {
        self.category_peaks.get(&category).copied().unwrap_or(0)
    }
}

#[derive(Debug)]
struct BudgetInner {
    config: BudgetConfig,
    next_id: u64,
    reservations: BTreeMap<ReservationId, Reservation>,
    category_totals: BTreeMap<BudgetCategory, usize>,
    category_peaks: BTreeMap<BudgetCategory, usize>,
    total_reserved_bytes: usize,
    unmaterialized_reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl BudgetInner {
    fn new(config: BudgetConfig) -> Self {
        let mut category_totals = BTreeMap::new();
        let mut category_peaks = BTreeMap::new();
        for cat in BudgetCategory::ALL {
            category_totals.insert(cat, 0);
            category_peaks.insert(cat, 0);
        }
        Self {
            config,
            next_id: 1,
            reservations: BTreeMap::new(),
            category_totals,
            category_peaks,
            total_reserved_bytes: 0,
            unmaterialized_reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    fn compute_headroom(&self, mem_available_bytes: usize) -> (usize, usize) {
        let effective_available =
            mem_available_bytes.saturating_sub(self.unmaterialized_reserved_bytes);
        let floor_headroom = effective_available.saturating_sub(self.config.floor_bytes);
        let headroom = match self.config.max_budget_bytes {
            Some(cap) => {
                let cap_headroom = cap.saturating_sub(self.total_reserved_bytes);
                floor_headroom.min(cap_headroom)
            }
            None => floor_headroom,
        };
        (effective_available, headroom)
    }
}

/// Single unified memory accountant enforcing the host `MemAvailable` floor.
///
/// Cloning a [`Budget`] shares the same underlying accountant state (`Arc`), enabling
/// [`ReservationGuard`] to release allocations automatically on drop without lifetime coupling.
#[derive(Clone)]
pub struct Budget {
    probe: Arc<dyn HostMemProbe>,
    inner: Arc<Mutex<BudgetInner>>,
}

impl fmt::Debug for Budget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = self.inner.lock().expect("Budget lock poisoned");
        f.debug_struct("Budget")
            .field("config", &guard.config)
            .field("total_reserved_bytes", &guard.total_reserved_bytes)
            .field(
                "unmaterialized_reserved_bytes",
                &guard.unmaterialized_reserved_bytes,
            )
            .field("peak_reserved_bytes", &guard.peak_reserved_bytes)
            .field("active_reservations", &guard.reservations.len())
            .finish()
    }
}

impl Budget {
    /// Create a [`Budget`] with the default `8 GiB` `MemAvailable` floor (`DEFAULT_FLOOR_BYTES`).
    pub fn new(probe: impl HostMemProbe + 'static) -> Self {
        Self::with_config(probe, BudgetConfig::default())
    }

    /// Create a [`Budget`] with a custom `MemAvailable` floor in bytes (e.g. [`MIMO_FLOOR_BYTES`]).
    pub fn with_floor(probe: impl HostMemProbe + 'static, floor_bytes: usize) -> Self {
        Self::with_config(probe, BudgetConfig::with_floor(floor_bytes))
    }

    /// Create a [`Budget`] with a full [`BudgetConfig`].
    pub fn with_config(probe: impl HostMemProbe + 'static, config: BudgetConfig) -> Self {
        Self::from_arc_probe(Arc::new(probe), config)
    }

    /// Create a [`Budget`] from a shared `Arc<dyn HostMemProbe>` and [`BudgetConfig`].
    pub fn from_arc_probe(probe: Arc<dyn HostMemProbe>, config: BudgetConfig) -> Self {
        Self {
            probe,
            inner: Arc::new(Mutex::new(BudgetInner::new(config))),
        }
    }

    /// Create a Strix Halo default [`Budget`] backed by [`LinuxProcMemProbe`] with an `8 GiB` floor.
    pub fn strix_default() -> Self {
        Self::new(LinuxProcMemProbe::new())
    }

    /// Create a MiMo [`Budget`] backed by [`LinuxProcMemProbe`] with an `11 GiB` floor.
    pub fn mimo_default() -> Self {
        Self::with_floor(LinuxProcMemProbe::new(), MIMO_FLOOR_BYTES)
    }

    /// Configured `MemAvailable` safety floor in bytes.
    pub fn floor_bytes(&self) -> usize {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .config
            .floor_bytes
    }

    /// Update the `MemAvailable` safety floor in bytes.
    pub fn set_floor_bytes(&self, floor_bytes: usize) {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .config
            .floor_bytes = floor_bytes;
    }

    /// Configured optional total budget cap in bytes.
    pub fn max_budget_bytes(&self) -> Option<usize> {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .config
            .max_budget_bytes
    }

    /// Update the optional total budget cap in bytes.
    pub fn set_max_budget_bytes(&self, max_budget_bytes: Option<usize>) {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .config
            .max_budget_bytes = max_budget_bytes;
    }

    /// Reserve `bytes` under `(category, label)`.
    ///
    /// Checks both:
    /// 1. Live `HostMemProbe::mem_available_bytes()` minus unmaterialized reservations >= `floor_bytes` + `bytes`
    ///    (returning [`MemError::FloorViolation`] with a full category breakdown if violated).
    /// 2. Optional `max_budget_bytes` cap (returning [`MemError::CapExceeded`] if exceeded).
    pub fn try_reserve(
        &self,
        category: BudgetCategory,
        label: impl Into<String>,
        bytes: usize,
    ) -> Result<Reservation> {
        let label = label.into();
        let mem_available_bytes = self.probe.mem_available_bytes()?;

        let mut inner = self.inner.lock().expect("Budget lock poisoned");
        let effective_available_bytes =
            mem_available_bytes.saturating_sub(inner.unmaterialized_reserved_bytes);
        let floor_headroom = effective_available_bytes.saturating_sub(inner.config.floor_bytes);
        let (_, total_headroom) = inner.compute_headroom(mem_available_bytes);

        if effective_available_bytes < inner.config.floor_bytes || bytes > floor_headroom {
            return Err(MemError::FloorViolation {
                category,
                label,
                requested_bytes: bytes,
                mem_available_bytes,
                unmaterialized_reserved_bytes: inner.unmaterialized_reserved_bytes,
                effective_available_bytes,
                floor_bytes: inner.config.floor_bytes,
                headroom_bytes: total_headroom,
                total_reserved_bytes: inner.total_reserved_bytes,
                max_budget_bytes: inner.config.max_budget_bytes,
                category_totals: inner.category_totals.clone(),
            });
        }

        let new_total = inner
            .total_reserved_bytes
            .checked_add(bytes)
            .ok_or_else(|| MemError::InvalidLayout {
                reason: "reservation byte count overflowed usize".to_string(),
            })?;

        if let Some(max_budget_bytes) = inner.config.max_budget_bytes {
            if new_total > max_budget_bytes {
                let available_cap_bytes =
                    max_budget_bytes.saturating_sub(inner.total_reserved_bytes);
                return Err(MemError::CapExceeded {
                    category,
                    label,
                    requested_bytes: bytes,
                    current_reserved_bytes: inner.total_reserved_bytes,
                    max_budget_bytes,
                    available_cap_bytes,
                    category_totals: inner.category_totals.clone(),
                });
            }
        }

        let id = ReservationId(inner.next_id);
        inner.next_id += 1;

        let reservation = Reservation {
            id,
            category,
            label,
            bytes,
            materialized: false,
        };

        inner.total_reserved_bytes = new_total;
        inner.unmaterialized_reserved_bytes += bytes;
        if new_total > inner.peak_reserved_bytes {
            inner.peak_reserved_bytes = new_total;
        }

        let cat_total = {
            let entry = inner.category_totals.entry(category).or_insert(0);
            *entry += bytes;
            *entry
        };
        let cat_peak = inner.category_peaks.entry(category).or_insert(0);
        if cat_total > *cat_peak {
            *cat_peak = cat_total;
        }

        inner.reservations.insert(id, reservation.clone());
        Ok(reservation)
    }

    /// Reserve `bytes` under `(category, label)` and return an RAII [`ReservationGuard`]
    /// that automatically releases the reservation when dropped.
    pub fn try_reserve_guard(
        &self,
        category: BudgetCategory,
        label: impl Into<String>,
        bytes: usize,
    ) -> Result<ReservationGuard> {
        let reservation = self.try_reserve(category, label, bytes)?;
        Ok(ReservationGuard::new(self.clone(), reservation))
    }

    /// Release a [`Reservation`], returning its freed byte count.
    pub fn release(&self, reservation: Reservation) -> usize {
        self.release_by_id(reservation.id).unwrap_or(0)
    }

    /// Release an active reservation by its [`ReservationId`], returning the freed byte count.
    pub fn release_by_id(&self, id: ReservationId) -> Result<usize> {
        let mut inner = self.inner.lock().expect("Budget lock poisoned");
        let Some(removed) = inner.reservations.remove(&id) else {
            return Err(MemError::InvalidReservation {
                id: id.0,
                reason: "reservation not found or already released".to_string(),
            });
        };

        inner.total_reserved_bytes = inner.total_reserved_bytes.saturating_sub(removed.bytes);
        if !removed.materialized {
            inner.unmaterialized_reserved_bytes = inner
                .unmaterialized_reserved_bytes
                .saturating_sub(removed.bytes);
        }
        if let Some(cat_total) = inner.category_totals.get_mut(&removed.category) {
            *cat_total = cat_total.saturating_sub(removed.bytes);
        }

        Ok(removed.bytes)
    }

    /// Mark an active reservation as materialized in OS memory (so its bytes are now reflected
    /// directly in `/proc/meminfo` `MemAvailable` rather than subtracted as unmaterialized).
    pub fn mark_materialized(&self, id: ReservationId) -> Result<()> {
        let mut inner = self.inner.lock().expect("Budget lock poisoned");
        let Some(res) = inner.reservations.get_mut(&id) else {
            return Err(MemError::InvalidReservation {
                id: id.0,
                reason: "reservation not found".to_string(),
            });
        };
        if !res.materialized {
            let bytes = res.bytes;
            res.materialized = true;
            inner.unmaterialized_reserved_bytes =
                inner.unmaterialized_reserved_bytes.saturating_sub(bytes);
        }
        Ok(())
    }

    /// Returns `true` if `bytes` can currently be reserved without violating the `MemAvailable`
    /// floor or `max_budget_bytes` cap. Returns `false` if the probe is unavailable.
    pub fn can_fit(&self, bytes: usize) -> bool {
        match self.try_headroom_bytes() {
            Ok(headroom) => bytes <= headroom,
            Err(_) => false,
        }
    }

    /// Remaining allocatable headroom in bytes above the `MemAvailable` floor (and within
    /// `max_budget_bytes` if set). Returns `0` if the probe fails.
    pub fn headroom_bytes(&self) -> usize {
        self.try_headroom_bytes().unwrap_or(0)
    }

    /// Fallible version of [`Budget::headroom_bytes`] that propagates probe errors.
    pub fn try_headroom_bytes(&self) -> Result<usize> {
        let mem_available_bytes = self.probe.mem_available_bytes()?;
        let inner = self.inner.lock().expect("Budget lock poisoned");
        let (_, headroom) = inner.compute_headroom(mem_available_bytes);
        Ok(headroom)
    }

    /// Total currently reserved bytes across all categories.
    pub fn total_reserved_bytes(&self) -> usize {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .total_reserved_bytes
    }

    /// Total currently unmaterialized reserved bytes.
    pub fn unmaterialized_reserved_bytes(&self) -> usize {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .unmaterialized_reserved_bytes
    }

    /// Peak total reserved bytes across all categories over the lifetime of this [`Budget`].
    pub fn peak_reserved_bytes(&self) -> usize {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .peak_reserved_bytes
    }

    /// Currently reserved bytes for `category`.
    pub fn category_bytes(&self, category: BudgetCategory) -> usize {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .category_totals
            .get(&category)
            .copied()
            .unwrap_or(0)
    }

    /// Peak reserved bytes for `category`.
    pub fn category_peak_bytes(&self, category: BudgetCategory) -> usize {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .category_peaks
            .get(&category)
            .copied()
            .unwrap_or(0)
    }

    /// Number of currently active reservations.
    pub fn active_reservation_count(&self) -> usize {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .reservations
            .len()
    }

    /// Snapshot of all currently active reservations in ID order.
    pub fn active_reservations(&self) -> Vec<Reservation> {
        self.inner
            .lock()
            .expect("Budget lock poisoned")
            .reservations
            .values()
            .cloned()
            .collect()
    }

    /// Return a complete [`BudgetSummary`] snapshot (saturating `mem_available_bytes` to `0`
    /// if the probe fails).
    pub fn summary(&self) -> BudgetSummary {
        let mem_available_bytes = self.probe.mem_available_bytes().unwrap_or(0);
        self.summary_with_available(mem_available_bytes)
    }

    /// Fallible version of [`Budget::summary`] that propagates probe errors.
    pub fn try_summary(&self) -> Result<BudgetSummary> {
        let mem_available_bytes = self.probe.mem_available_bytes()?;
        Ok(self.summary_with_available(mem_available_bytes))
    }

    fn summary_with_available(&self, mem_available_bytes: usize) -> BudgetSummary {
        let inner = self.inner.lock().expect("Budget lock poisoned");
        let (effective_available_bytes, headroom_bytes) =
            inner.compute_headroom(mem_available_bytes);
        BudgetSummary {
            category_totals: inner.category_totals.clone(),
            category_peaks: inner.category_peaks.clone(),
            total_reserved_bytes: inner.total_reserved_bytes,
            unmaterialized_reserved_bytes: inner.unmaterialized_reserved_bytes,
            peak_reserved_bytes: inner.peak_reserved_bytes,
            floor_bytes: inner.config.floor_bytes,
            max_budget_bytes: inner.config.max_budget_bytes,
            mem_available_bytes,
            effective_available_bytes,
            headroom_bytes,
            active_reservation_count: inner.reservations.len(),
        }
    }
}
