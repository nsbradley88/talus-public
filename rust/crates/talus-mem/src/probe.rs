//! Host memory probing for `/proc/meminfo`, cgroup v1/v2 limits, and synthetic fixtures.
//!
//! Mirrors the semantics of `src/core/conversation_memory.cpp` and `tests/core/memory_guard_test.cpp`:
//! - Parses `/proc/meminfo` (`MemAvailable`, `MemTotal`, `MemFree`).
//! - Walks cgroup v2 (`memory.max`, `memory.high`, `memory.current`) and cgroup v1
//!   (`memory.limit_in_bytes`, `memory.usage_in_bytes`) hierarchies from the root down to the
//!   process's own group, crediting clean inactive file cache (`inactive_file` /
//!   `total_inactive_file` minus dirty and writeback pages) from `memory.stat`.
//! - Applies an optional explicit operator cap (`--memory-limit-mib` / `explicit_limit_bytes`).
//! - Fails closed (`None` / [`MemError::ProbeUnavailable`]) when telemetry is missing or malformed
//!   under an explicit cap, and falls back to `MemAvailable` alone when cgroup files are malformed
//!   without an operator cap.

use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::{MemError, Result};

/// Which limit governed the effective [`HostMemoryReading::available`] value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySource {
    /// Governed by `/proc/meminfo` `MemAvailable`.
    Meminfo,
    /// Governed by a cgroup v2 (`memory.max` / `memory.high`) or v1 (`memory.limit_in_bytes`) limit.
    Cgroup,
    /// Governed by an explicit operator cap (`--memory-limit-mib`).
    Flag,
}

impl MemorySource {
    /// Canonical source name matching `strata::core::memory_source_name`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Meminfo => "meminfo",
            Self::Cgroup => "cgroup",
            Self::Flag => "flag",
        }
    }
}

impl fmt::Display for MemorySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Parsed fields from `/proc/meminfo` (all values converted from `kB` to bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeminfoStats {
    /// `MemAvailable` in bytes.
    pub mem_available_bytes: usize,
    /// `MemTotal` in bytes, if present and valid.
    pub mem_total_bytes: Option<usize>,
    /// `MemFree` in bytes, if present and valid.
    pub mem_free_bytes: Option<usize>,
}

/// File paths read by [`LinuxProcMemProbe`] and [`sample_host_memory`].
///
/// Defaults point to `/proc/meminfo`, `/proc/self/cgroup`, and `/sys/fs/cgroup`;
/// tests can point them at a fixture directory via [`MemoryProbePaths::from_fixture_dir`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryProbePaths {
    pub meminfo: PathBuf,
    pub self_cgroup: PathBuf,
    pub cgroup_root: PathBuf,
}

impl Default for MemoryProbePaths {
    fn default() -> Self {
        Self {
            meminfo: PathBuf::from("/proc/meminfo"),
            self_cgroup: PathBuf::from("/proc/self/cgroup"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup"),
        }
    }
}

impl MemoryProbePaths {
    /// Construct probe paths rooted at a synthetic fixture directory containing
    /// `meminfo`, `self_cgroup`, and `cgroup/` (matching `tests/core/memory_guard_test.cpp`).
    pub fn from_fixture_dir(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        Self {
            meminfo: dir.join("meminfo"),
            self_cgroup: dir.join("self_cgroup"),
            cgroup_root: dir.join("cgroup"),
        }
    }
}

/// One host memory telemetry reading (Rust counterpart of `strata::core::HostMemoryReading`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMemoryReading {
    /// Effective available bytes (smallest of `MemAvailable`, cgroup room, and operator cap).
    pub available: usize,
    /// Governing source of `available`.
    pub source: MemorySource,
    /// Limit of the governing source (`MemTotal` for `Meminfo`, cgroup limit for `Cgroup`,
    /// operator cap for `Flag`).
    pub limit: usize,
    /// Raw usage of the governing source before `credit` (`MemTotal - MemAvailable` for `Meminfo`,
    /// `memory.current` for `Cgroup` / `Flag`).
    pub current: usize,
    /// Clean inactive file cache credited as reclaimable (`0` for `Meminfo`).
    pub credit: usize,
    /// Raw `MemAvailable` from `/proc/meminfo` in bytes.
    pub mem_available: usize,
    /// Raw `MemTotal` from `/proc/meminfo` in bytes, if present.
    pub mem_total: Option<usize>,
    /// Raw `MemFree` from `/proc/meminfo` in bytes, if present.
    pub mem_free: Option<usize>,
}

/// Parse a base-10 unsigned integer strictly (ASCII digits only, no leading `+` or `-`).
fn parse_strict_u64(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<u64>().ok()
}

/// Extract `<wanted> <n> kB` from `/proc/meminfo` text and return `<n> * 1024` bytes.
///
/// Matches `meminfo_kb_bytes` in `src/core/conversation_memory.cpp`:
/// the key must appear at most once with unit `"kB"` and no extra tokens on the line.
pub fn meminfo_kb_bytes(meminfo: &str, wanted: &str) -> Option<usize> {
    let mut result: Option<usize> = None;
    for line in meminfo.lines() {
        let mut fields = line.split_ascii_whitespace();
        let Some(key) = fields.next() else {
            continue;
        };
        if key != wanted {
            continue;
        }
        if result.is_some() {
            return None;
        }
        let value = fields.next()?;
        let unit = fields.next()?;
        if unit != "kB" || fields.next().is_some() {
            return None;
        }
        let kb = parse_strict_u64(value)?;
        let bytes_u64 = kb.checked_mul(1024)?;
        let bytes = usize::try_from(bytes_u64).ok()?;
        result = Some(bytes);
    }
    result
}

/// Parse `MemAvailable:` from `/proc/meminfo` content (Rust counterpart of `conversation_mem_available`).
pub fn conversation_mem_available(meminfo: &str) -> Option<usize> {
    meminfo_kb_bytes(meminfo, "MemAvailable:")
}

/// Parse `MemAvailable`, `MemTotal`, and `MemFree` from `/proc/meminfo` content.
pub fn parse_meminfo(meminfo: &str) -> Option<MeminfoStats> {
    let mem_available_bytes = meminfo_kb_bytes(meminfo, "MemAvailable:")?;
    let mem_total_bytes = meminfo_kb_bytes(meminfo, "MemTotal:");
    let mem_free_bytes = meminfo_kb_bytes(meminfo, "MemFree:");
    Some(MeminfoStats {
        mem_available_bytes,
        mem_total_bytes,
        mem_free_bytes,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlValue {
    Absent,
    Bad,
    Max,
    Number(u64),
}

fn read_control_value(path: &Path) -> ControlValue {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ControlValue::Absent,
        Err(_) => return ControlValue::Bad,
        Ok(_) => {}
    }
    let Ok(file) = File::open(path) else {
        return ControlValue::Bad;
    };
    let mut buf = Vec::with_capacity(64);
    if file.take(64).read_to_end(&mut buf).is_err() {
        return ControlValue::Bad;
    }
    if buf.len() == 64 {
        return ControlValue::Bad;
    }
    while let Some(&last) = buf.last() {
        if last == b'\n' || last == b'\r' || last == b' ' || last == b'\t' {
            buf.pop();
        } else {
            break;
        }
    }
    if buf.is_empty() {
        return ControlValue::Bad;
    }
    let Ok(s) = std::str::from_utf8(&buf) else {
        return ControlValue::Bad;
    };
    if s == "max" {
        return ControlValue::Max;
    }
    match parse_strict_u64(s) {
        Some(n) => ControlValue::Number(n),
        None => ControlValue::Bad,
    }
}

fn reclaimable_cache(dir: &Path, current: u64, v1: bool) -> u64 {
    let Ok(stat) = fs::read_to_string(dir.join("memory.stat")) else {
        return 0;
    };
    let inactive_key = if v1 {
        "total_inactive_file"
    } else {
        "inactive_file"
    };
    let dirty_key = if v1 { "total_dirty" } else { "file_dirty" };
    let writeback_key = if v1 {
        "total_writeback"
    } else {
        "file_writeback"
    };

    let mut inactive: Option<u64> = None;
    let mut dirty: u64 = 0;
    let mut writeback: u64 = 0;

    for line in stat.lines() {
        let mut fields = line.split_ascii_whitespace();
        let (Some(key), Some(value)) = (fields.next(), fields.next()) else {
            continue;
        };
        if fields.next().is_some() {
            continue;
        }
        let parsed = parse_strict_u64(value);
        if key == inactive_key {
            let Some(n) = parsed else {
                return 0;
            };
            if inactive.is_some() {
                return 0;
            }
            inactive = Some(n);
        } else if key == dirty_key {
            let Some(n) = parsed else {
                return 0;
            };
            dirty = n;
        } else if key == writeback_key {
            let Some(n) = parsed else {
                return 0;
            };
            writeback = n;
        }
    }

    let Some(inactive_val) = inactive else {
        return 0;
    };
    let mut credit = inactive_val.min(current);
    credit = credit.saturating_sub(dirty);
    credit = credit.saturating_sub(writeback);
    credit
}

#[derive(Debug, Clone, Copy)]
struct CgroupLimit {
    limit: u64,
    current: u64,
    credit: u64,
}

#[derive(Debug, Default)]
struct CgroupScan {
    limits: Vec<CgroupLimit>,
    root_current: Option<u64>,
    root_credit: u64,
    any_files: bool,
}

fn scan_levels(
    dirs: &[PathBuf],
    limit_names: &[&str],
    current_name: &str,
    v1: bool,
    out: &mut CgroupScan,
) -> bool {
    for (i, dir) in dirs.iter().enumerate() {
        let current = read_control_value(&dir.join(current_name));
        let mut limits = Vec::with_capacity(limit_names.len());
        for name in limit_names {
            limits.push(read_control_value(&dir.join(name)));
        }

        let mut unusable = matches!(current, ControlValue::Bad | ControlValue::Max);
        let mut finite_limit = false;
        for &v in &limits {
            unusable = unusable || matches!(v, ControlValue::Bad);
            finite_limit = finite_limit || matches!(v, ControlValue::Number(_));
        }
        if unusable {
            return false;
        }
        let current_n = match current {
            ControlValue::Absent => {
                if finite_limit {
                    return false;
                }
                continue;
            }
            ControlValue::Number(n) => n,
            ControlValue::Bad | ControlValue::Max => unreachable!(),
        };

        out.any_files = true;
        let credit = reclaimable_cache(dir, current_n, v1);
        if i == 0 {
            out.root_current = Some(current_n);
            out.root_credit = credit;
        }
        for v in limits {
            if let ControlValue::Number(limit_n) = v {
                if v1 && limit_n >= (1u64 << 62) {
                    continue;
                }
                out.limits.push(CgroupLimit {
                    limit: limit_n,
                    current: current_n,
                    credit,
                });
            }
        }
    }
    true
}

fn group_dirs(root: &Path, group: &str) -> Option<Vec<PathBuf>> {
    let mut names = Vec::new();
    for part in group.split('/') {
        if part.is_empty() {
            continue;
        }
        if part == "." || part == ".." {
            return None;
        }
        names.push(part);
    }
    if !root.is_dir() {
        return Some(Vec::new());
    }
    let mut dirs = vec![root.to_path_buf()];
    let mut at = root.to_path_buf();
    for part in names {
        at.push(part);
        if !at.is_dir() {
            break;
        }
        dirs.push(at.clone());
    }
    Some(dirs)
}

fn scan_cgroups(paths: &MemoryProbePaths, out: &mut CgroupScan) -> bool {
    let mut v2_group: Option<String> = None;
    let mut v1_group: Option<String> = None;

    match fs::read_to_string(&paths.self_cgroup) {
        Err(_) => {
            v2_group = Some("/".to_string());
        }
        Ok(content) => {
            for line in content.lines() {
                let Some(c1) = line.find(':') else {
                    continue;
                };
                let rest = &line[c1 + 1..];
                let Some(c2_rel) = rest.find(':') else {
                    continue;
                };
                let hid = &line[..c1];
                let controllers = &rest[..c2_rel];
                let group = &rest[c2_rel + 1..];
                if hid == "0" && controllers.is_empty() {
                    v2_group = Some(group.to_string());
                    continue;
                }
                if controllers.split(',').any(|c| c == "memory") {
                    v1_group = Some(group.to_string());
                }
            }
        }
    }

    if let Some(ref group) = v2_group {
        let Some(dirs) = group_dirs(&paths.cgroup_root, group) else {
            return false;
        };
        if !scan_levels(
            &dirs,
            &["memory.max", "memory.high"],
            "memory.current",
            false,
            out,
        ) {
            return false;
        }
    }

    if !out.any_files {
        if let Some(ref group) = v1_group {
            let v1_root = paths.cgroup_root.join("memory");
            let Some(dirs) = group_dirs(&v1_root, group) else {
                return false;
            };
            if !scan_levels(
                &dirs,
                &["memory.limit_in_bytes"],
                "memory.usage_in_bytes",
                true,
                out,
            ) {
                return false;
            }
        }
    }

    true
}

fn combine_readings(
    meminfo: MeminfoStats,
    cg: &CgroupScan,
    explicit_limit_bytes: usize,
) -> Option<HostMemoryReading> {
    let available = meminfo.mem_available_bytes;
    let total = meminfo.mem_total_bytes;

    let mut reading = HostMemoryReading {
        available,
        source: MemorySource::Meminfo,
        limit: total.unwrap_or(0),
        current: total
            .filter(|&t| t >= available)
            .map_or(0, |t| t - available),
        credit: 0,
        mem_available: available,
        mem_total: total,
        mem_free: meminfo.mem_free_bytes,
    };

    for l in &cg.limits {
        let limit = usize::try_from(l.limit).unwrap_or(usize::MAX);
        let current = usize::try_from(l.current).unwrap_or(usize::MAX);
        let credit = usize::try_from(l.credit).unwrap_or(usize::MAX);
        let used = current.saturating_sub(credit);
        let room = limit.saturating_sub(used);
        if room < reading.available {
            reading.available = room;
            reading.source = MemorySource::Cgroup;
            reading.limit = limit;
            reading.current = current;
            reading.credit = credit;
        }
    }

    if explicit_limit_bytes != 0 {
        let (used, credit) = if let Some(root_curr) = cg.root_current {
            (
                usize::try_from(root_curr).unwrap_or(usize::MAX),
                usize::try_from(cg.root_credit).unwrap_or(usize::MAX),
            )
        } else {
            let tot = total.filter(|&t| t >= available)?;
            (tot - available, 0)
        };

        let net = used.saturating_sub(credit);
        let room = explicit_limit_bytes.saturating_sub(net);
        if room <= reading.available {
            reading.available = room;
            reading.source = MemorySource::Flag;
            reading.limit = explicit_limit_bytes;
            reading.current = used;
            reading.credit = credit;
        }
    }

    Some(reading)
}

/// Sample host memory from `paths` with an optional operator cap `explicit_limit_bytes` (`0` = none).
///
/// Exact Rust port of `strata::core::sample_host_memory(const MemoryProbePaths&, uint64_t)`.
pub fn sample_host_memory(
    paths: &MemoryProbePaths,
    explicit_limit_bytes: usize,
) -> Option<HostMemoryReading> {
    let meminfo_text = fs::read_to_string(&paths.meminfo).ok()?;
    let meminfo = parse_meminfo(&meminfo_text)?;

    let mut cg = CgroupScan::default();
    if !scan_cgroups(paths, &mut cg) {
        if explicit_limit_bytes != 0 {
            return None;
        }
        cg = CgroupScan::default();
    }

    combine_readings(meminfo, &cg, explicit_limit_bytes)
}

/// Check whether an allocation of `allocation_bytes` can be admitted while keeping at least
/// `floor_bytes` of available memory (`strata::core::conversation_memory_admit`).
pub const fn conversation_memory_admit(
    available_bytes: Option<usize>,
    allocation_bytes: usize,
    floor_bytes: usize,
) -> bool {
    match available_bytes {
        Some(avail) => avail >= floor_bytes && allocation_bytes <= avail - floor_bytes,
        None => false,
    }
}

/// Trait for querying available host memory (`MemAvailable` bounded by cgroup/operator limits).
pub trait HostMemProbe: Send + Sync {
    /// Return the effective available host memory in bytes.
    fn mem_available_bytes(&self) -> Result<usize>;

    /// Return a full [`HostMemoryReading`] sample.
    fn sample(&self) -> Result<HostMemoryReading> {
        let available = self.mem_available_bytes()?;
        Ok(HostMemoryReading {
            available,
            source: MemorySource::Meminfo,
            limit: available,
            current: 0,
            credit: 0,
            mem_available: available,
            mem_total: Some(available),
            mem_free: Some(available),
        })
    }
}

/// Linux `/proc/meminfo` + cgroup v1/v2 host memory probe.
///
/// Can read either the real `/proc` and `/sys/fs/cgroup` filesystems or a synthetic
/// fixture directory for deterministic unit testing.
#[derive(Debug)]
pub struct LinuxProcMemProbe {
    paths: MemoryProbePaths,
    explicit_limit_bytes: AtomicUsize,
}

impl Clone for LinuxProcMemProbe {
    fn clone(&self) -> Self {
        Self {
            paths: self.paths.clone(),
            explicit_limit_bytes: AtomicUsize::new(self.explicit_limit_bytes()),
        }
    }
}

impl Default for LinuxProcMemProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl LinuxProcMemProbe {
    /// Create a probe reading `/proc/meminfo`, `/proc/self/cgroup`, and `/sys/fs/cgroup`.
    pub fn new() -> Self {
        Self::with_paths(MemoryProbePaths::default())
    }

    /// Create a probe with explicit [`MemoryProbePaths`].
    pub fn with_paths(paths: MemoryProbePaths) -> Self {
        Self {
            paths,
            explicit_limit_bytes: AtomicUsize::new(0),
        }
    }

    /// Create a probe rooted at a fixture directory (`meminfo`, `self_cgroup`, `cgroup/`).
    pub fn from_fixture_dir(dir: impl AsRef<Path>) -> Self {
        Self::with_paths(MemoryProbePaths::from_fixture_dir(dir))
    }

    /// Configure an explicit operator memory cap in bytes (`0` disables the cap).
    pub fn with_explicit_limit_bytes(self, limit_bytes: usize) -> Self {
        self.explicit_limit_bytes
            .store(limit_bytes, Ordering::Relaxed);
        self
    }

    /// Configure an explicit operator memory cap in MiB (`--memory-limit-mib`).
    pub fn with_memory_limit_mib(self, limit_mib: usize) -> Self {
        let bytes = limit_mib.saturating_mul(1024 * 1024);
        self.with_explicit_limit_bytes(bytes)
    }

    /// Dynamically update the operator memory cap in bytes (`0` disables the cap).
    pub fn set_explicit_limit_bytes(&self, limit_bytes: usize) {
        self.explicit_limit_bytes
            .store(limit_bytes, Ordering::Relaxed);
    }

    /// Return the currently configured operator memory cap in bytes (`0` if none).
    pub fn explicit_limit_bytes(&self) -> usize {
        self.explicit_limit_bytes.load(Ordering::Relaxed)
    }

    /// Return the configured probe file paths.
    pub fn paths(&self) -> &MemoryProbePaths {
        &self.paths
    }
}

impl HostMemProbe for LinuxProcMemProbe {
    fn mem_available_bytes(&self) -> Result<usize> {
        Ok(self.sample()?.available)
    }

    fn sample(&self) -> Result<HostMemoryReading> {
        let explicit_limit = self.explicit_limit_bytes();
        sample_host_memory(&self.paths, explicit_limit).ok_or_else(|| MemError::ProbeUnavailable {
            reason: format!(
                "failed to sample host memory from meminfo='{}', cgroup='{}' (explicit_limit={explicit_limit})",
                self.paths.meminfo.display(),
                self.paths.cgroup_root.display()
            ),
        })
    }
}

#[derive(Debug, Clone)]
struct MockState {
    reading: Option<HostMemoryReading>,
    failure_reason: Option<String>,
}

/// Configurable in-memory [`HostMemProbe`] for deterministic unit testing.
#[derive(Debug, Clone)]
pub struct MockMemProbe {
    state: Arc<Mutex<MockState>>,
}

impl MockMemProbe {
    /// Default total memory (`128 GiB`) reported by [`MockMemProbe::new`].
    pub const DEFAULT_TOTAL_BYTES: usize = 128 * 1024 * 1024 * 1024;

    /// Create a mock probe reporting `mem_available_bytes` of available memory.
    pub fn new(mem_available_bytes: usize) -> Self {
        let total = Self::DEFAULT_TOTAL_BYTES.max(mem_available_bytes);
        Self::with_total_and_available(total, mem_available_bytes)
    }

    /// Create a mock probe with explicit `MemTotal` and `MemAvailable` values in bytes.
    pub fn with_total_and_available(mem_total_bytes: usize, mem_available_bytes: usize) -> Self {
        let total = mem_total_bytes.max(mem_available_bytes);
        let reading = HostMemoryReading {
            available: mem_available_bytes,
            source: MemorySource::Meminfo,
            limit: total,
            current: total - mem_available_bytes,
            credit: 0,
            mem_available: mem_available_bytes,
            mem_total: Some(total),
            mem_free: Some(mem_available_bytes),
        };
        Self::with_reading(reading)
    }

    /// Create a mock probe returning a specific [`HostMemoryReading`].
    pub fn with_reading(reading: HostMemoryReading) -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState {
                reading: Some(reading),
                failure_reason: None,
            })),
        }
    }

    /// Create a mock probe that always fails with [`MemError::ProbeUnavailable`].
    pub fn failing(reason: impl Into<String>) -> Self {
        Self {
            state: Arc::new(Mutex::new(MockState {
                reading: None,
                failure_reason: Some(reason.into()),
            })),
        }
    }

    /// Update the available bytes reported by this mock probe.
    pub fn set_available_bytes(&self, available_bytes: usize) {
        let mut guard = self.state.lock().expect("MockMemProbe lock poisoned");
        let total = guard
            .reading
            .and_then(|r| r.mem_total)
            .unwrap_or(Self::DEFAULT_TOTAL_BYTES)
            .max(available_bytes);
        guard.reading = Some(HostMemoryReading {
            available: available_bytes,
            source: MemorySource::Meminfo,
            limit: total,
            current: total - available_bytes,
            credit: 0,
            mem_available: available_bytes,
            mem_total: Some(total),
            mem_free: Some(available_bytes),
        });
        guard.failure_reason = None;
    }

    /// Subtract `bytes` from the currently reported available memory (saturating at 0).
    pub fn sub_available_bytes(&self, bytes: usize) {
        let current_avail = {
            let guard = self.state.lock().expect("MockMemProbe lock poisoned");
            guard.reading.map_or(0, |r| r.available)
        };
        self.set_available_bytes(current_avail.saturating_sub(bytes));
    }

    /// Add `bytes` to the currently reported available memory.
    pub fn add_available_bytes(&self, bytes: usize) {
        let current_avail = {
            let guard = self.state.lock().expect("MockMemProbe lock poisoned");
            guard.reading.map_or(0, |r| r.available)
        };
        self.set_available_bytes(current_avail.saturating_add(bytes));
    }

    /// Replace the full [`HostMemoryReading`] returned by this mock probe.
    pub fn set_reading(&self, reading: HostMemoryReading) {
        let mut guard = self.state.lock().expect("MockMemProbe lock poisoned");
        guard.reading = Some(reading);
        guard.failure_reason = None;
    }
}

impl HostMemProbe for MockMemProbe {
    fn mem_available_bytes(&self) -> Result<usize> {
        Ok(self.sample()?.available)
    }

    fn sample(&self) -> Result<HostMemoryReading> {
        let guard = self.state.lock().expect("MockMemProbe lock poisoned");
        if let Some(ref reason) = guard.failure_reason {
            return Err(MemError::ProbeUnavailable {
                reason: reason.clone(),
            });
        }
        guard.reading.ok_or_else(|| MemError::ProbeUnavailable {
            reason: "mock probe has no reading".to_string(),
        })
    }
}
