//! qwen4exp-r1: the platform layer of the Rust expert source and PLE reader (Linux x86_64): the libc calls (no libc
//! crate in this workspace), the host-memory readings of `src/core/expert_source.cpp` (`detail::host_available_memory`,
//! `free_memory_bytes`, the cgroup walk) and the small pure helpers of its `detail` namespace, plus
//! `strata::platform` (`file_cache_keeps`, `lock_resident`, `advise_willneed`, `read_ahead_enabled`).
//!
//! Every helper keeps the C++ rule it ports, message text included (the startup lines are compared across paths).

#![allow(clippy::missing_safety_doc)]
#![allow(dead_code)]
// the C++ arithmetic is kept as written (its saturating ternaries)
#![allow(clippy::implicit_saturating_sub)]

use core::ffi::{c_char, c_int, c_long, c_void};
use std::ffi::CString;

// =====================================================================================================================
// libc (x86_64 Linux values)
// =====================================================================================================================

pub const O_RDONLY: c_int = 0;
pub const O_DIRECT: c_int = 0o40000;
pub const O_CLOEXEC: c_int = 0o2000000;
pub const PROT_READ: c_int = 1;
pub const MAP_SHARED: c_int = 1;
pub const MAP_FAILED: *mut c_void = !0usize as *mut c_void;
pub const MADV_WILLNEED: c_int = 3;
pub const MADV_DONTNEED: c_int = 4;
pub const POSIX_FADV_RANDOM: c_int = 1;
pub const POSIX_FADV_WILLNEED: c_int = 3;
pub const POSIX_FADV_DONTNEED: c_int = 4;
pub const EINTR: i32 = 4;
pub const SC_PAGESIZE: c_int = 30;
pub const SYS_IO_SETUP: c_long = 206;
pub const SYS_IO_DESTROY: c_long = 207;
pub const SYS_IO_GETEVENTS: c_long = 208;
pub const SYS_IO_SUBMIT: c_long = 209;
pub const IOCB_CMD_PREAD: u16 = 0;

extern "C" {
    pub fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    pub fn close(fd: c_int) -> c_int;
    pub fn pread(fd: c_int, buf: *mut c_void, n: usize, off: i64) -> isize;
    pub fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        off: i64,
    ) -> *mut c_void;
    pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
    pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
    pub fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
    pub fn posix_fadvise(fd: c_int, off: i64, len: i64, advice: c_int) -> c_int;
    pub fn mlock(addr: *const c_void, len: usize) -> c_int;
    pub fn munlock(addr: *const c_void, len: usize) -> c_int;
    pub fn posix_memalign(out: *mut *mut c_void, align: usize, size: usize) -> c_int;
    pub fn malloc(size: usize) -> *mut c_void;
    pub fn free(p: *mut c_void);
    pub fn syscall(n: c_long, ...) -> c_long;
    pub fn sysconf(name: c_int) -> c_long;
    pub fn strerror(e: c_int) -> *const c_char;
}

pub fn errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

pub fn strerror_str(e: i32) -> String {
    unsafe {
        let p = strerror(e);
        if p.is_null() {
            return format!("error {e}");
        }
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// `::open(path, flags)`; -1 on failure.
pub fn open_path(path: &str, flags: c_int) -> c_int {
    let Ok(c) = CString::new(path) else { return -1 };
    unsafe { open(c.as_ptr(), flags) }
}

/// The file's size (fstat's st_size through std), None when it cannot be read.
pub fn file_size(path: &str) -> Option<u64> {
    std::fs::metadata(path).ok().map(|m| m.len())
}

/// The whole range with pread (a short read continues); false on an error or an end of file.
pub unsafe fn pread_full(fd: c_int, dst: *mut u8, n: u64, off: u64) -> bool {
    let mut done = 0u64;
    while done < n {
        let r = pread(
            fd,
            dst.add(done as usize).cast(),
            (n - done) as usize,
            (off + done) as i64,
        );
        if r < 0 {
            if errno() == EINTR {
                continue;
            }
            return false;
        }
        if r == 0 {
            return false;
        }
        done += r as u64;
    }
    true
}

/// `strata::platform::read_ahead_enabled` (STRATA_READ_AHEAD=0 turns the hints off).
pub fn read_ahead_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("STRATA_READ_AHEAD") {
        Ok(v) => c_atoi(&v) != 0,
        Err(_) => true,
    })
}

const ADVISE_STEP: usize = 128 << 10;

/// `strata::platform::advise_willneed(const void*, bytes)`: MADV_WILLNEED in 128 KiB steps.
pub unsafe fn advise_willneed(p: *const u8, bytes: u64) {
    if p.is_null() || bytes == 0 || !read_ahead_enabled() {
        return;
    }
    let ps = sysconf(SC_PAGESIZE);
    let pg: usize = if ps > 0 { ps as usize } else { 4096 };
    let end = p as usize + bytes as usize;
    let mut a = (p as usize) & !(pg - 1);
    while a < end {
        let _ = madvise(a as *mut c_void, ADVISE_STEP.min(end - a), MADV_WILLNEED);
        a += ADVISE_STEP;
    }
}

/// `strata::platform::LockResult`.
#[derive(Clone, Debug, Default)]
pub struct LockResult {
    pub ok: bool,
    pub locked_bytes: u64,
    pub note: String,
}

/// `strata::platform::lock_resident` (POSIX: mlock).
pub unsafe fn lock_resident(p: *mut u8, bytes: u64) -> LockResult {
    let mut r = LockResult::default();
    if p.is_null() || bytes == 0 {
        r.note = "nothing to lock".into();
        return r;
    }
    if mlock(p as *const c_void, bytes as usize) != 0 {
        r.note = "mlock failed (raise ulimit -l)".into();
        return r;
    }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock".into();
    r
}

pub unsafe fn unlock_resident(p: *mut u8, bytes: u64) {
    if !p.is_null() && bytes != 0 {
        let _ = munlock(p as *const c_void, bytes as usize);
    }
}

/// `strata::platform::file_cache_keeps`.
pub fn file_cache_keeps(
    avail: u64,
    arena_bytes: u64,
    read_bytes: u64,
    margin: u64,
    hot_fraction: f64,
    floor_bytes: u64,
) -> bool {
    let room = if avail > arena_bytes.wrapping_add(margin) {
        avail - arena_bytes - margin
    } else {
        0
    };
    let mut need = (read_bytes as f64 * hot_fraction) as u64;
    if need < floor_bytes {
        need = floor_bytes;
    }
    if need > read_bytes {
        need = read_bytes;
    }
    room >= need
}

/// C `atoi`: optional whitespace and sign, then the longest digit prefix (0 when there is none).
pub fn c_atoi(s: &str) -> i64 {
    let t = s.trim_start();
    let b = t.as_bytes();
    let mut i = 0;
    let mut neg = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        neg = b[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((b[i] - b'0') as i64);
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// C `atof` (the longest numeric prefix; 0 when there is none).
pub fn c_atof(s: &str) -> f64 {
    let t = s.trim_start();
    let mut best = 0.0;
    for end in 1..=t.len() {
        if !t.is_char_boundary(end) {
            continue;
        }
        if let Ok(v) = t[..end].parse::<f64>() {
            best = v;
        }
    }
    best
}

/// An env var's value when set (any value, empty included).
pub fn env(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

// =====================================================================================================================
// The host memory readings (expert_source.cpp `detail` + its anonymous namespace)
// =====================================================================================================================

pub const NO_CACHE_COMPLEMENT: u64 = u64::MAX;

/// `detail::HostMemory`.
#[derive(Clone, Copy, Debug)]
pub struct HostMemory {
    pub available: u64,
    pub cgroup_limit: u64,
    pub commit: u64,
}

impl Default for HostMemory {
    fn default() -> Self {
        Self {
            available: 0,
            cgroup_limit: u64::MAX,
            commit: u64::MAX,
        }
    }
}

/// `detail::CgroupMemoryStat`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CgroupMemoryStat {
    pub current: u64,
    pub inactive_file: u64,
    pub file_dirty: u64,
    pub file_writeback: u64,
    pub valid: bool,
}

/// `detail::cgroup_available_bytes`.
pub fn cgroup_available_bytes(limit: u64, stat: &CgroupMemoryStat) -> Option<u64> {
    if !stat.valid {
        return None;
    }
    let mut reclaimable = stat.inactive_file.min(stat.current);
    reclaimable = if stat.file_dirty >= reclaimable {
        0
    } else {
        reclaimable - stat.file_dirty
    };
    reclaimable = if stat.file_writeback >= reclaimable {
        0
    } else {
        reclaimable - stat.file_writeback
    };
    let usage_after_reclaim = stat.current - reclaimable;
    Some(if usage_after_reclaim < limit {
        limit - usage_after_reclaim
    } else {
        0
    })
}

/// `detail::clamp_resident_budget`.
pub fn clamp_resident_budget(requested: u64, physical: u64, commit: u64, headroom: u64) -> u64 {
    let available = physical.min(commit);
    let room = if available > headroom {
        available - headroom
    } else {
        0
    };
    if requested <= room {
        return requested;
    }
    let margin = 256u64 << 20;
    if room > margin {
        room - margin
    } else {
        0
    }
}

/// `detail::pin_depends_on_reclaim`.
pub fn pin_depends_on_reclaim(free_now: u64, bytes: u64, reserve: u64) -> bool {
    bytes > 0 && free_now < bytes.wrapping_add(reserve)
}

/// `detail::pin_step_fits`.
pub fn pin_step_fits(available: u64, step: u64, reserve: u64) -> bool {
    available >= step.wrapping_add(reserve)
}

/// One whitespace-separated field reader with C++ `>>` semantics for `std::string` / `uint64_t`.
struct Fields<'a> {
    it: std::str::SplitWhitespace<'a>,
}

impl<'a> Fields<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            it: s.split_whitespace(),
        }
    }
    fn word(&mut self) -> Option<&'a str> {
        self.it.next()
    }
    /// `>> uint64_t`: the leading digits of the next word (a word that does not start with one fails).
    fn u64(&mut self) -> Option<u64> {
        let w = self.it.next()?;
        let digits: String = w.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return None;
        }
        digits.parse::<u64>().ok()
    }
}

/// `detail::meminfo_bytes`.
pub fn meminfo_bytes(text: &str, key: &str) -> u64 {
    let want = format!("{key}:");
    for line in text.lines() {
        let mut f = Fields::new(line);
        let (Some(k), Some(value)) = (f.word(), f.u64()) else {
            continue;
        };
        if k == want {
            let unit = f.word().unwrap_or("");
            return if unit == "kB" { value * 1024 } else { value };
        }
    }
    0
}

/// `detail::numa_pin_free`.
pub fn numa_pin_free(global_free: u64, node_free: &[u64]) -> u64 {
    if node_free.len() < 2 {
        return global_free;
    }
    let smallest = *node_free.iter().min().unwrap();
    global_free.min(smallest.wrapping_mul(node_free.len() as u64))
}

/// `detail::page_cache_hint`.
pub fn page_cache_hint(text: &str) -> String {
    let cache = meminfo_bytes(text, "Cached") + meminfo_bytes(text, "Buffers");
    if cache < (1u64 << 30) {
        return String::new();
    }
    format!(
        "; {:.1} GiB of the RAM is file cache (earlier runs' model files, which Linux gives back on demand - 'free' \
         showing little is normal). If the start still fails, free it with: sync; echo 1 | sudo tee \
         /proc/sys/vm/drop_caches   (or start strata serve with STRATA_DROP_CACHE_ON_EXIT=1 so it releases its model \
         files when it stops)",
        cache as f64 / 1073741824.0
    )
}

/// A minimal model of `std::filesystem::path` for the cgroup walk: lexical normalization that keeps a trailing
/// separator (an empty last element), `parent_path`, and element-wise equality - the C++ semantics the walk relies on.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FsPath(String);

impl FsPath {
    fn join(&self, rel: &str) -> FsPath {
        if rel.starts_with('/') {
            return FsPath(rel.to_string());
        }
        if self.0.ends_with('/') {
            FsPath(format!("{}{}", self.0, rel))
        } else {
            FsPath(format!("{}/{}", self.0, rel))
        }
    }
    /// `lexically_normal`: "." and "a/.." dropped, repeated separators collapsed; a trailing separator stays.
    fn normal(&self) -> FsPath {
        let abs = self.0.starts_with('/');
        let trailing = self.0.len() > 1
            && (self.0.ends_with('/') || self.0.ends_with("/.") || self.0.ends_with("/.."));
        let mut out: Vec<&str> = Vec::new();
        for c in self.0.split('/') {
            match c {
                "" | "." => {}
                ".." => {
                    if matches!(out.last(), Some(l) if *l != "..") {
                        out.pop();
                    } else if !abs {
                        out.push("..");
                    }
                }
                x => out.push(x),
            }
        }
        let mut s = if abs {
            String::from("/")
        } else {
            String::new()
        };
        s.push_str(&out.join("/"));
        if trailing && !s.ends_with('/') {
            s.push('/');
        }
        if s.is_empty() {
            s.push('.');
        }
        FsPath(s)
    }
    fn parent(&self) -> FsPath {
        let s = &self.0;
        if s.len() > 1 && s.ends_with('/') {
            return FsPath(s[..s.len() - 1].to_string());
        }
        match s.rfind('/') {
            Some(0) => FsPath("/".into()),
            Some(i) => FsPath(s[..i].to_string()),
            None => FsPath(String::new()),
        }
    }
    fn starts_with_str(&self, root: &FsPath) -> bool {
        self.0.starts_with(&root.0)
    }
    fn file(&self, name: &str) -> String {
        if self.0.ends_with('/') {
            format!("{}{}", self.0, name)
        } else {
            format!("{}/{}", self.0, name)
        }
    }
}

fn read_cgroup_memory_stat(path: &FsPath, current: u64) -> Option<CgroupMemoryStat> {
    let text = std::fs::read_to_string(path.file("memory.stat")).ok()?;
    let mut stat = CgroupMemoryStat::default();
    let (mut inactive, mut dirty, mut wb) = (false, false, false);
    for line in text.lines() {
        let mut f = Fields::new(line);
        let (Some(key), Some(value)) = (f.word(), f.u64()) else {
            return None;
        };
        if f.word().is_some() {
            return None;
        }
        match key {
            "inactive_file" => {
                if inactive {
                    return None;
                }
                inactive = true;
                stat.inactive_file = value;
            }
            "file_dirty" => {
                if dirty {
                    return None;
                }
                dirty = true;
                stat.file_dirty = value;
            }
            "file_writeback" => {
                if wb {
                    return None;
                }
                wb = true;
                stat.file_writeback = value;
            }
            _ => {}
        }
    }
    if !inactive || !dirty || !wb {
        return None;
    }
    stat.current = current;
    stat.valid = true;
    Some(stat)
}

/// `detail::host_available_memory` (Linux): MemAvailable, bounded by the tightest cgroup ancestor (v2, else v1).
pub fn host_available_memory_at(
    meminfo: &str,
    self_cgroup: &str,
    cgroup_root: &str,
) -> Option<HostMemory> {
    let mut m = HostMemory::default();
    let info = std::fs::read_to_string(meminfo).unwrap_or_default();
    let mut bytes = 0u64;
    for line in info.lines() {
        let mut f = Fields::new(line);
        let (Some(key), Some(value), Some(unit)) = (f.word(), f.u64(), f.word()) else {
            continue;
        };
        if key == "MemAvailable:" && unit == "kB" && value <= u64::MAX / 1024 {
            bytes = value * 1024;
        }
    }
    if bytes == 0 {
        return None;
    }
    let groups = std::fs::read_to_string(self_cgroup).ok()?;
    let root = FsPath(cgroup_root.to_string());
    let mut v2 = false;
    let mut v1_path = String::new();
    for line in groups.lines() {
        if !line.starts_with("0::/") {
            let Some(c1) = line.find(':') else { continue };
            let Some(c2) = line[c1 + 1..].find(':').map(|x| x + c1 + 1) else {
                continue;
            };
            for c in line[c1 + 1..c2].split(',') {
                if c == "memory" {
                    v1_path = line[c2 + 1..].to_string();
                }
            }
            continue;
        }
        let mut path = root.join(&line[4..]).normal();
        if !path.starts_with_str(&root) || !std::path::Path::new(&path.0).is_dir() {
            return None;
        }
        v2 = true;
        while path.starts_with_str(&root) {
            let limit = std::fs::read_to_string(path.file("memory.max"))
                .ok()
                .and_then(|s| s.split_whitespace().next().map(str::to_string));
            let current = std::fs::read_to_string(path.file("memory.current"))
                .ok()
                .and_then(|s| Fields::new(&s).u64());
            let readable = limit.is_some() && current.is_some();
            if !readable
                && !(path == root
                    && !std::path::Path::new(&path.file("memory.max")).exists()
                    && std::path::Path::new(&path.file("cgroup.controllers")).exists())
            {
                return None;
            }
            if readable {
                let limit = limit.unwrap();
                if limit != "max" {
                    // std::stoull with the whole string consumed
                    let cap = limit.parse::<u64>().ok()?;
                    let stat = read_cgroup_memory_stat(&path, current.unwrap())?;
                    let avail = cgroup_available_bytes(cap, &stat)?;
                    bytes = bytes.min(avail);
                    m.cgroup_limit = m.cgroup_limit.min(cap);
                }
            }
            if path == root {
                break;
            }
            path = path.parent();
        }
    }
    if !v2 && !v1_path.is_empty() {
        let mroot = root.join("memory");
        let rel = v1_path.strip_prefix('/').unwrap_or(&v1_path);
        let mut path = mroot.join(rel).normal();
        while path.starts_with_str(&mroot) {
            let cap = std::fs::read_to_string(path.file("memory.limit_in_bytes"))
                .ok()
                .and_then(|s| Fields::new(&s).u64());
            let usage = std::fs::read_to_string(path.file("memory.usage_in_bytes"))
                .ok()
                .and_then(|s| Fields::new(&s).u64());
            if let (Some(cap), Some(usage)) = (cap, usage) {
                if cap < (1u64 << 62) {
                    bytes = bytes.min(if usage < cap { cap - usage } else { 0 });
                    m.cgroup_limit = m.cgroup_limit.min(cap);
                }
            }
            if path == mroot {
                break;
            }
            path = path.parent();
        }
    }
    m.available = bytes;
    Some(m)
}

pub fn host_available_memory() -> Option<HostMemory> {
    host_available_memory_at("/proc/meminfo", "/proc/self/cgroup", "/sys/fs/cgroup")
}

/// `available_memory_bytes(bytes, commit)`: (available, commit), None when unknown or 0.
pub fn available_memory_bytes() -> Option<(u64, u64)> {
    let m = host_available_memory()?;
    if m.available > 0 {
        Some((m.available, m.commit))
    } else {
        None
    }
}

/// `free_memory_bytes` (#1250): MemFree, judged by the emptiest NUMA node too.
pub fn free_memory_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let bytes = meminfo_bytes(&text, "MemFree");
    if bytes == 0 {
        return None;
    }
    let mut nodes = Vec::new();
    for n in 0..64 {
        let Ok(node) = std::fs::read_to_string(format!("/sys/devices/system/node/node{n}/meminfo"))
        else {
            break;
        };
        for line in node.lines() {
            let mut f = Fields::new(line);
            let (Some(_a), Some(_id), Some(key), Some(kb)) =
                (f.word(), f.word(), f.word(), f.u64())
            else {
                continue;
            };
            if key == "MemFree:" {
                nodes.push(kb * 1024);
                break;
            }
        }
    }
    let b = numa_pin_free(bytes, &nodes);
    if b > 0 {
        Some(b)
    } else {
        None
    }
}

/// `file_tier_unbuffered` (Linux): whether the file tier reads `read_bytes` unbuffered beside `arena_bytes` of RAM copy
/// still to be taken; `why` is the startup line's reason.
pub fn file_tier_unbuffered(arena_bytes: u64, read_bytes: u64, why: &mut String) -> bool {
    if let Some(e) = env("STRATA_UNBUFFERED_LOAD") {
        if !e.is_empty() {
            *why = format!("STRATA_UNBUFFERED_LOAD={e}");
            return !e.starts_with('0');
        }
    }
    let Some((avail, _)) = available_memory_bytes() else {
        *why = "the available RAM is unknown".into();
        return false;
    };
    let keepable = file_cache_keeps(avail, arena_bytes, read_bytes, 1u64 << 30, 0.05, 3u64 << 29);
    *why = format!(
        "{:.1} GiB available, {:.1} GiB of it still to be taken by the RAM copy, {:.1} GiB of experts read from the \
         files: the file cache {} keep the ones that come back",
        avail as f64 / (1u64 << 30) as f64,
        arena_bytes as f64 / (1u64 << 30) as f64,
        read_bytes as f64 / (1u64 << 30) as f64,
        if keepable { "can" } else { "cannot" }
    );
    !keepable
}

// =====================================================================================================================
// The cache complement plan (expert_source.cpp `detail`)
// =====================================================================================================================

/// `detail::make_cache_complement_plan`: `offsets[l * n_expert + e]` = the blob's place in the RAM copy, or
/// NO_CACHE_COMPLEMENT for the experts a GPU holds; `bytes` = the copy's size.
pub fn make_cache_complement_plan(
    n_layers: i64,
    n_expert: i64,
    layer_blob_bytes: &[u64],
    primary_gpu_pairs: &[(i32, i32)],
    additional_gpu_pairs: &[(i32, i32)],
    offsets: &mut Vec<u64>,
    bytes: &mut u64,
) -> Result<(), String> {
    offsets.clear();
    *bytes = 0;
    if n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.len() != n_layers as usize {
        return Err("FileExpertSource: invalid geometry for the cache complement plan".into());
    }
    let count = (n_layers * n_expert) as usize;
    let mut omitted = vec![0u8; count];
    let mut mark = |pairs: &[(i32, i32)], bit: u8, label: &str| -> Result<(), String> {
        for &(l, e) in pairs {
            if l < 0 || e < 0 || l as i64 >= n_layers || e as i64 >= n_expert {
                return Err(format!(
                    "FileExpertSource: {label} pair is outside the expert geometry"
                ));
            }
            let index = l as usize * n_expert as usize + e as usize;
            if omitted[index] & bit != 0 {
                return Err(format!(
                    "FileExpertSource: duplicate {label} pair in the cache complement plan"
                ));
            }
            if bit == 2 && omitted[index] & 1 != 0 {
                return Err(
                    "FileExpertSource: the primary and additional GPU expert tiers overlap".into(),
                );
            }
            omitted[index] |= bit;
        }
        Ok(())
    };
    mark(primary_gpu_pairs, 1, "primary GPU")?;
    mark(additional_gpu_pairs, 2, "additional GPU")?;
    if layer_blob_bytes.contains(&0) {
        return Err("FileExpertSource: cache complement layer has zero-sized expert blobs".into());
    }
    offsets.resize(count, NO_CACHE_COMPLEMENT);
    for l in 0..n_layers as usize {
        let b = layer_blob_bytes[l];
        for e in 0..n_expert as usize {
            let index = l * n_expert as usize + e;
            if omitted[index] != 0 {
                continue;
            }
            if *bytes > u64::MAX - b {
                offsets.clear();
                *bytes = 0;
                return Err("FileExpertSource: cache complement size overflows".into());
            }
            offsets[index] = *bytes;
            *bytes += b;
        }
    }
    Ok(())
}

/// `detail::choose_resident_keep_from`.
pub fn choose_resident_keep_from(
    slot_bytes: &[u64],
    base_bytes: u64,
    budget: u64,
    mut lend_from: i64,
) -> i64 {
    if base_bytes > budget {
        return -1;
    }
    let slots = slot_bytes.len() as i64;
    if lend_from < 0 || lend_from > slots {
        lend_from = slots;
    }
    let mut keep = slots;
    let mut bytes = base_bytes;
    while keep > lend_from {
        let b = slot_bytes[keep as usize - 1];
        if b > budget - bytes {
            break;
        }
        bytes += b;
        keep -= 1;
    }
    keep
}

/// `detail::append_stage_lend_regions`.
pub fn append_stage_lend_regions(
    n_layers: i64,
    n_expert: i64,
    layer_blob_bytes: &[u64],
    offsets: &mut [u64],
    bytes: &mut u64,
    cap_bytes: u64,
    regions: &[Vec<(i32, i32)>],
) -> Result<i64, String> {
    let blobs = (n_layers * n_expert) as usize;
    if n_layers <= 0
        || n_expert <= 0
        || layer_blob_bytes.len() != n_layers as usize
        || offsets.len() != blobs
    {
        return Err("append_stage_lend_regions: the plan does not match the geometry".into());
    }
    for region in regions {
        for &(l, e) in region {
            if l < 0 || l as i64 >= n_layers || e < 0 || e as i64 >= n_expert {
                return Err(
                    "append_stage_lend_regions: a lend-region expert is outside the model".into(),
                );
            }
        }
        let mut sorted = region.clone();
        sorted.sort();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(
                "append_stage_lend_regions: a stage's lend region repeats an expert".into(),
            );
        }
    }
    let mut kept = 0i64;
    for region in regions {
        for &(l, e) in region {
            let index = l as usize * n_expert as usize + e as usize;
            if offsets[index] != NO_CACHE_COMPLEMENT {
                continue;
            }
            let b = layer_blob_bytes[l as usize];
            if b > cap_bytes.wrapping_sub(*bytes) {
                break;
            }
            offsets[index] = *bytes;
            *bytes += b;
            kept += 1;
        }
    }
    Ok(kept)
}

/// `detail::exchange_cache_complement`.
pub fn exchange_cache_complement(offsets: &mut [u64], i_in: usize, out: usize) -> bool {
    if i_in == out
        || i_in >= offsets.len()
        || out >= offsets.len()
        || offsets[i_in] == NO_CACHE_COMPLEMENT
        || offsets[out] != NO_CACHE_COMPLEMENT
    {
        return false;
    }
    offsets[out] = offsets[i_in];
    offsets[i_in] = NO_CACHE_COMPLEMENT;
    true
}

// =====================================================================================================================
// Residency probes (mincore)
// =====================================================================================================================

/// Bytes of [p, p + n) whose pages are in the page cache (page-granular); 0 when the call fails.
pub unsafe fn resident_in(p: *const u8, n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let a = (p as usize) & !4095usize;
    let e = (p as usize + n as usize + 4095) & !4095usize;
    let mut vec = vec![0u8; (e - a) / 4096];
    if mincore(a as *mut c_void, e - a, vec.as_mut_ptr()) != 0 {
        return 0;
    }
    let pages: u64 = vec.iter().map(|&c| (c & 1) as u64).sum();
    n.min(pages * 4096)
}

// =====================================================================================================================
// Unbuffered reads: the kernel's AIO (io_setup / io_submit / io_getevents) over O_DIRECT descriptors
// =====================================================================================================================

/// `struct iocb` (linux/aio_abi.h, little endian).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Iocb {
    pub aio_data: u64,
    pub aio_key: u32,
    pub aio_rw_flags: i32,
    pub aio_lio_opcode: u16,
    pub aio_reqprio: i16,
    pub aio_fildes: u32,
    pub aio_buf: u64,
    pub aio_nbytes: u64,
    pub aio_offset: i64,
    pub aio_reserved2: u64,
    pub aio_flags: u32,
    pub aio_resfd: u32,
}

/// `struct io_event`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IoEvent {
    pub data: u64,
    pub obj: u64,
    pub res: i64,
    pub res2: i64,
}

/// One 4 KiB-aligned window of a role slice (or of a whole blob): read into the batch buffer, then copied to `dst + at`.
#[derive(Clone, Copy)]
pub struct DirectWindow {
    pub file: i32,
    pub a0: u64,
    pub size: u64,
    pub skip: u64,
    pub n: u64,
    pub at: u64,
    pub dst: *mut u8,
    pub req: usize,
    pub in_req: u64,
}

struct DirectReq {
    fd: c_int,
    a0: u64,
    size: u64,
    pos: u64,
}

/// A thread's AIO context and aligned buffer, kept for its next batch (`read_direct`'s `thread_local Scratch`).
pub struct DirectScratch {
    buf: *mut u8,
    cap: usize,
    ctx: u64,
}

impl Drop for DirectScratch {
    fn drop(&mut self) {
        unsafe {
            free(self.buf.cast());
            if self.ctx != 0 {
                let _ = syscall(SYS_IO_DESTROY, self.ctx);
            }
        }
    }
}

thread_local! {
    static DIRECT_SCRATCH: std::cell::RefCell<DirectScratch> =
        const { std::cell::RefCell::new(DirectScratch { buf: std::ptr::null_mut(), cap: 0, ctx: 0 }) };
}

/// `FileExpertSource::read_direct` (Linux): the windows sorted by (file, offset), merged into requests of at most
/// 32 MiB where they touch, read 256 at a time through the kernel's AIO, then scattered. False on any failure (the
/// caller falls back to the mapping).
pub unsafe fn read_direct_windows(win: &mut [DirectWindow], fds: &[c_int]) -> bool {
    const SECTOR: u64 = 4096;
    const MERGE: u64 = 32 << 20;
    const DEPTH: usize = 256;
    DIRECT_SCRATCH.with(|cell| {
        let mut sc = cell.borrow_mut();
        if sc.ctx == 0 {
            let mut ctx: u64 = 0;
            if syscall(SYS_IO_SETUP, DEPTH as c_long, &mut ctx as *mut u64) != 0 {
                return false;
            }
            sc.ctx = ctx;
        }
        let mut order: Vec<usize> = (0..win.len()).collect();
        order.sort_by(|&a, &b| {
            if win[a].file != win[b].file {
                win[a].file.cmp(&win[b].file)
            } else {
                win[a].a0.cmp(&win[b].a0)
            }
        });
        let mut req: Vec<DirectReq> = Vec::new();
        let mut total = 0u64;
        let mut last_file = -1;
        for &w in &order {
            let x = &mut win[w];
            if !req.is_empty() && x.file == last_file {
                let nreq = req.len();
                let q = req.last_mut().unwrap();
                let end = q.a0 + q.size;
                let xend = x.a0 + x.size;
                if x.a0 <= end && end.max(xend) - q.a0 <= MERGE {
                    if xend > end {
                        total += xend - end;
                        q.size = xend - q.a0;
                    }
                    x.req = nreq - 1;
                    x.in_req = x.a0 - q.a0;
                    continue;
                }
            }
            req.push(DirectReq {
                fd: fds[x.file as usize],
                a0: x.a0,
                size: x.size,
                pos: total,
            });
            total += x.size;
            last_file = x.file;
            x.req = req.len() - 1;
            x.in_req = 0;
        }
        if total as usize > sc.cap {
            free(sc.buf.cast());
            sc.cap = ((total + (1 << 20) - 1) >> 20 << 20) as usize;
            let mut p: *mut c_void = std::ptr::null_mut();
            sc.buf = if posix_memalign(&mut p, SECTOR as usize, sc.cap) == 0 {
                p.cast()
            } else {
                std::ptr::null_mut()
            };
            if sc.buf.is_null() {
                sc.cap = 0;
                return false;
            }
        }
        let mut got = vec![-1i64; req.len()];
        let wave = req.len().min(DEPTH);
        let mut cb = vec![Iocb::default(); wave];
        let mut cbp: Vec<*mut Iocb> = vec![std::ptr::null_mut(); wave];
        let mut ev = vec![IoEvent::default(); wave];
        let mut ok = true;
        let mut w0 = 0usize;
        while ok && w0 < req.len() {
            let wn = DEPTH.min(req.len() - w0);
            for j in 0..wn {
                let r = &req[w0 + j];
                cb[j] = Iocb {
                    aio_lio_opcode: IOCB_CMD_PREAD,
                    aio_fildes: r.fd as u32,
                    aio_buf: sc.buf.add(r.pos as usize) as u64,
                    aio_nbytes: r.size,
                    aio_offset: r.a0 as i64,
                    aio_data: (w0 + j) as u64,
                    ..Iocb::default()
                };
                cbp[j] = &mut cb[j];
            }
            let mut queued = 0usize;
            while queued < wn {
                let r = syscall(
                    SYS_IO_SUBMIT,
                    sc.ctx,
                    (wn - queued) as c_long,
                    cbp.as_mut_ptr().add(queued),
                );
                if r > 0 {
                    queued += r as usize;
                    continue;
                }
                if r < 0 && errno() == EINTR {
                    continue;
                }
                ok = false;
                break;
            }
            let mut done = 0usize;
            while done < queued {
                let r = syscall(
                    SYS_IO_GETEVENTS,
                    sc.ctx,
                    1 as c_long,
                    (queued - done) as c_long,
                    ev.as_mut_ptr(),
                    std::ptr::null_mut::<c_void>(),
                );
                if r < 0 {
                    if errno() == EINTR {
                        continue;
                    }
                    let _ = syscall(SYS_IO_DESTROY, sc.ctx);
                    sc.ctx = 0;
                    return false;
                }
                for e in ev.iter().take(r as usize) {
                    got[e.data as usize] = e.res;
                    if e.res < 0 {
                        ok = false;
                    }
                }
                done += r as usize;
            }
            w0 += DEPTH;
        }
        if !ok {
            return false;
        }
        for x in win.iter() {
            let g = got[x.req];
            if g < 0 || (g as u64) < x.in_req + x.skip + x.n {
                return false;
            }
            std::ptr::copy_nonoverlapping(
                sc.buf.add((req[x.req].pos + x.in_req + x.skip) as usize),
                x.dst.add(x.at as usize),
                x.n as usize,
            );
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fs_path_semantics() {
        let root = FsPath("/sys/fs/cgroup".into());
        let p = root.join("user.slice/a.scope").normal();
        assert_eq!(p.0, "/sys/fs/cgroup/user.slice/a.scope");
        assert_eq!(p.parent().parent().0, "/sys/fs/cgroup");
        let r = root.join("").normal();
        assert_eq!(r.0, "/sys/fs/cgroup/");
        assert_ne!(r, root);
        assert_eq!(r.parent(), root);
        assert_eq!(root.join("a/../b").normal().0, "/sys/fs/cgroup/b");
    }

    #[test]
    fn meminfo_and_plans() {
        let t = "MemTotal:       131072 kB\nMemFree:  2048 kB\nCached: 2097152 kB\nBuffers: 0 kB\n";
        assert_eq!(meminfo_bytes(t, "MemFree"), 2048 * 1024);
        assert!(page_cache_hint(t).starts_with("; 2.0 GiB"));
        assert_eq!(numa_pin_free(100, &[10, 50]), 20);
        assert_eq!(clamp_resident_budget(10, 100, u64::MAX, 50), 10);
        assert_eq!(
            clamp_resident_budget(1 << 40, 8 << 30, u64::MAX, 4 << 30),
            (4u64 << 30) - (256 << 20)
        );
        let mut off = Vec::new();
        let mut bytes = 0;
        make_cache_complement_plan(2, 3, &[10, 20], &[(0, 1)], &[(1, 2)], &mut off, &mut bytes)
            .unwrap();
        assert_eq!(
            off,
            vec![0, NO_CACHE_COMPLEMENT, 10, 20, 40, NO_CACHE_COMPLEMENT]
        );
        assert_eq!(bytes, 60);
        assert_eq!(choose_resident_keep_from(&[5, 5, 5, 5], 10, 22, 1), 2);
        assert!(exchange_cache_complement(&mut off, 0, 1));
        assert_eq!(off[1], 0);
        assert!(file_cache_keeps(
            20 << 30,
            0,
            10 << 30,
            1 << 30,
            0.05,
            3 << 29
        ));
        assert_eq!(c_atoi(" 12x"), 12);
        assert_eq!(c_atof("0.25GiB"), 0.25);
    }
}
