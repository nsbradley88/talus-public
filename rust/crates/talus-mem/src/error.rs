//! Error types for `talus-mem`.

use std::collections::BTreeMap;
use std::fmt;

use talus_core_api::CoreError;

use crate::budget::BudgetCategory;

/// Errors returned by memory probing, budget accounting, and arena planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemError {
    /// Allocation would cause host `MemAvailable` (minus unmaterialized reservations)
    /// to drop below the configured safety floor.
    FloorViolation {
        category: BudgetCategory,
        label: String,
        requested_bytes: usize,
        mem_available_bytes: usize,
        unmaterialized_reserved_bytes: usize,
        effective_available_bytes: usize,
        floor_bytes: usize,
        headroom_bytes: usize,
        total_reserved_bytes: usize,
        max_budget_bytes: Option<usize>,
        category_totals: BTreeMap<BudgetCategory, usize>,
    },
    /// Allocation would exceed the configured total budget cap (`max_budget_bytes`).
    CapExceeded {
        category: BudgetCategory,
        label: String,
        requested_bytes: usize,
        current_reserved_bytes: usize,
        max_budget_bytes: usize,
        available_cap_bytes: usize,
        category_totals: BTreeMap<BudgetCategory, usize>,
    },
    /// Host memory telemetry (`/proc/meminfo` or cgroup limits) could not be read or parsed.
    ProbeUnavailable { reason: String },
    /// Invalid or unknown reservation handle.
    InvalidReservation { id: u64, reason: String },
    /// Invalid alignment, size overflow, or scope mark in arena/slab planning.
    InvalidLayout { reason: String },
}

impl MemError {
    /// Returns `true` if this error is a [`MemError::FloorViolation`].
    pub const fn is_floor_violation(&self) -> bool {
        matches!(self, Self::FloorViolation { .. })
    }

    /// Returns `true` if this error is a [`MemError::CapExceeded`].
    pub const fn is_cap_exceeded(&self) -> bool {
        matches!(self, Self::CapExceeded { .. })
    }
}

impl fmt::Display for MemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FloorViolation {
                category,
                label,
                requested_bytes,
                mem_available_bytes,
                unmaterialized_reserved_bytes,
                effective_available_bytes,
                floor_bytes,
                headroom_bytes,
                total_reserved_bytes,
                max_budget_bytes,
                category_totals,
            } => {
                write!(
                    f,
                    "memory floor violation for {category} '{label}': requested {requested_bytes} B \
                     exceeds headroom {headroom_bytes} B above floor {floor_bytes} B \
                     (live MemAvailable={mem_available_bytes} B, unmaterialized={unmaterialized_reserved_bytes} B, \
                     effective_available={effective_available_bytes} B, total_reserved={total_reserved_bytes} B, \
                     max_cap={max_budget_bytes:?}, categories={category_totals:?})"
                )
            }
            Self::CapExceeded {
                category,
                label,
                requested_bytes,
                current_reserved_bytes,
                max_budget_bytes,
                available_cap_bytes,
                category_totals,
            } => {
                write!(
                    f,
                    "budget cap exceeded for {category} '{label}': requested {requested_bytes} B \
                     with {available_cap_bytes} B remaining under cap {max_budget_bytes} B \
                     (current_reserved={current_reserved_bytes} B, categories={category_totals:?})"
                )
            }
            Self::ProbeUnavailable { reason } => {
                write!(f, "host memory probe unavailable: {reason}")
            }
            Self::InvalidReservation { id, reason } => {
                write!(f, "invalid reservation {id}: {reason}")
            }
            Self::InvalidLayout { reason } => {
                write!(f, "invalid arena/slab layout: {reason}")
            }
        }
    }
}

impl std::error::Error for MemError {}

impl From<MemError> for CoreError {
    fn from(err: MemError) -> Self {
        match err {
            MemError::FloorViolation {
                requested_bytes,
                headroom_bytes,
                ..
            } => Self::OutOfMemory {
                requested_bytes: requested_bytes as u64,
                available_bytes: headroom_bytes as u64,
            },
            MemError::CapExceeded {
                requested_bytes,
                available_cap_bytes,
                ..
            } => Self::OutOfMemory {
                requested_bytes: requested_bytes as u64,
                available_bytes: available_cap_bytes as u64,
            },
            MemError::ProbeUnavailable { reason } => Self::IoError { message: reason },
            MemError::InvalidReservation { reason, .. } => Self::StateError { reason },
            MemError::InvalidLayout { reason } => Self::InvalidArgument { reason },
        }
    }
}

/// Standard result alias for `talus-mem`.
pub type Result<T> = std::result::Result<T, MemError>;
