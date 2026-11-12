//! Host heap policy for talus-server (talus#93). One OS thread per connection with glibc malloc gives each busy thread
//! its own arena (up to 8 x cores); large request bodies (tens of MB) freed in many arenas stay mapped, so RSS after
//! very large requests followed by connection churn grew by GBs (probe + 60 churn bursts: +2.5 GB, v0.1.6 and v0.1.7).
//!
//! - Arenas: capped at [`DEFAULT_ARENAS`] at start (`mallopt(M_ARENA_MAX)`), unless `MALLOC_ARENA_MAX` is set (glibc reads
//!   it itself) or `TALUS_HEAP_ARENAS` says otherwise (`0` = leave glibc's default).
//! - Trim: after a request whose body was at least [`TRIM_FROM`], `malloc_trim(0)` returns the freed top of every arena,
//!   at most once per [`TRIM_EVERY`]. `TALUS_HEAP_TRIM=0` turns it off.
//!
//! Other allocators and non-glibc targets: both are no-ops.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Arenas when nothing else is configured: the server's work is parsing and copying, not allocation-bound.
pub const DEFAULT_ARENAS: i32 = 2;
/// Bodies at least this large trigger a trim when their request ends.
pub const TRIM_FROM: usize = 1024 * 1024;
/// Least time between two trims, in milliseconds (a trim walks every arena).
pub const TRIM_EVERY_MS: u64 = 1000;

static LAST_TRIM_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod sys {
    unsafe extern "C" {
        pub fn mallopt(param: i32, value: i32) -> i32;
        pub fn malloc_trim(pad: usize) -> i32;
    }
    pub const M_ARENA_MAX: i32 = -8;
}

fn env_flag(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Apply the arena cap; call once at start, before the accept loop. Returns the cap applied (None = glibc's own).
pub fn init() -> Option<i32> {
    if env_flag("MALLOC_ARENA_MAX").is_some() {
        return None;
    }
    let n = match env_flag("TALUS_HEAP_ARENAS") {
        Some(v) => v.trim().parse::<i32>().ok().filter(|n| *n >= 0)?,
        None => DEFAULT_ARENAS,
    };
    if n == 0 {
        return None;
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: glibc mallopt with a documented parameter, before other threads allocate much.
        if unsafe { sys::mallopt(sys::M_ARENA_MAX, n) } == 1 {
            return Some(n);
        }
    }
    None
}

fn trim_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_flag("TALUS_HEAP_TRIM").is_none_or(|v| v.trim() != "0"))
}

/// After a request with a `body_len`-byte body has ended (its buffers dropped): trim if it was large and no trim ran
/// within [`TRIM_EVERY_MS`]. Returns whether a trim ran.
pub fn after_request(body_len: usize) -> bool {
    if body_len < TRIM_FROM || !trim_enabled() {
        return false;
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let last = LAST_TRIM_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < TRIM_EVERY_MS
        || LAST_TRIM_MS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return false;
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: glibc malloc_trim, callable from any thread.
        unsafe { sys::malloc_trim(0) };
        return true;
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_bodies_never_trim_and_trims_are_rate_limited() {
        assert!(!after_request(TRIM_FROM - 1));
        if cfg!(all(target_os = "linux", target_env = "gnu")) && trim_enabled() {
            LAST_TRIM_MS.store(0, Ordering::Relaxed);
            assert!(after_request(TRIM_FROM));
            assert!(
                !after_request(TRIM_FROM),
                "a second trim within the interval"
            );
        }
    }
}
