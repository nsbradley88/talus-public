//! Persistent disk tier for the prompt caches (`TALUSPC1` entry files).
//!
//! A disk entry is the cached state of one token prefix: the token ids, a small engine-defined `meta` blob and one
//! or more payload sections (recurrent state, KV), or one external payload file written by an engine's own session
//! saver. Entries live in `<dir>/<identity>/`, where `identity` is a 64-bit fingerprint of everything that changes
//! what the saved bytes mean: the model files, the engine build (executable and kernel library), the GPU target and
//! the numerics switches (environment and arguments). A cache written under another identity is never read: it sits
//! in another folder, and every file repeats the identity in its header.
//!
//! Durability and safety:
//! - Writes go to a temporary name, are flushed (`fsync`), renamed into place and the folder is flushed, so a crash
//!   leaves either the old state or the complete new file, never a torn entry under the final name.
//! - The header region (fields, tokens, meta, section table) carries its own hash; every section carries a hash of
//!   its bytes (block-parallel xxh64), checked on every load. A file that fails any check is deleted and treated as
//!   a miss, so a corrupt cache costs one cold prefill and nothing else.
//! - A byte budget over the whole `<dir>` (all identities) is enforced after every write by evicting the least
//!   recently used entries (file mtime, refreshed on every hit).
//!
//! Writes run on one background thread. The engine copies the state to host memory (the only part that needs the
//! GPU) and hands the buffers over; while a write is in flight further writes are skipped rather than queued, so at
//! most one staging copy is alive.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Entry file magic.
pub const MAGIC: [u8; 8] = *b"TALUSPC1";
/// Trailer magic (followed by the header hash).
pub const END_MAGIC: [u8; 8] = *b"TALUSPCE";
/// Entry format version.
pub const VERSION: u32 = 1;
/// Entry file extension.
pub const EXT: &str = "tpc";
/// External payload file extension (engine-native session file next to its entry).
pub const EXT_PAYLOAD: &str = "tpcx";
/// Fixed header size.
pub const HEADER: usize = 64;
/// Trailer size.
pub const TRAILER: usize = 16;
/// Section data alignment inside an entry file.
pub const ALIGN: u64 = 4096;
/// Hash block size (block hashes are computed in parallel).
pub const HASH_BLOCK: usize = 16 << 20;
/// Section offset marking the external payload file.
const OFF_EXTERNAL: u64 = u64::MAX;

const MAX_TOKENS: u64 = 1 << 24;
const MAX_META: u64 = 1 << 20;
const MAX_SECTIONS: u32 = 256;

/// Default disk budget (32 GiB).
pub const DEFAULT_BUDGET_BYTES: u64 = 32 << 30;
/// Default minimum prefix length worth a disk entry (shorter prompts re-read in well under a second).
pub const DEFAULT_MIN_TOKENS: usize = 1024;
/// Default `MemAvailable` floor kept while staging a write (8 GiB, the project-wide floor).
pub const DEFAULT_MEM_FLOOR_BYTES: u64 = 8 << 30;

/// Build a section tag from up to 8 ASCII bytes (`tag(b"STATE0")`).
pub const fn tag(s: &[u8]) -> u64 {
    let mut v = [0u8; 8];
    let mut i = 0;
    while i < s.len() && i < 8 {
        v[i] = s[i];
        i += 1;
    }
    u64::from_le_bytes(v)
}

// ============================================================================
// xxh64 (block-parallel for large buffers)
// ============================================================================

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const P3: u64 = 0x1656_67B1_9E37_79F9;
const P4: u64 = 0x85EB_CA77_C2B2_AE63;
const P5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn round(acc: u64, lane: u64) -> u64 {
    acc.wrapping_add(lane.wrapping_mul(P2))
        .rotate_left(31)
        .wrapping_mul(P1)
}

#[inline]
fn merge(acc: u64, val: u64) -> u64 {
    (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
}

#[inline]
fn rd64(p: &[u8]) -> u64 {
    u64::from_le_bytes(p[..8].try_into().unwrap())
}

#[inline]
fn rd32(p: &[u8]) -> u64 {
    u32::from_le_bytes(p[..4].try_into().unwrap()) as u64
}

/// Streaming xxh64.
#[derive(Clone)]
pub struct Xxh64 {
    v: [u64; 4],
    seed: u64,
    total: u64,
    buf: [u8; 32],
    nbuf: usize,
}

impl Xxh64 {
    pub fn new(seed: u64) -> Self {
        Self {
            v: [
                seed.wrapping_add(P1).wrapping_add(P2),
                seed.wrapping_add(P2),
                seed,
                seed.wrapping_sub(P1),
            ],
            seed,
            total: 0,
            buf: [0; 32],
            nbuf: 0,
        }
    }

    #[inline]
    fn stripe(v: &mut [u64; 4], p: &[u8]) {
        v[0] = round(v[0], rd64(&p[0..]));
        v[1] = round(v[1], rd64(&p[8..]));
        v[2] = round(v[2], rd64(&p[16..]));
        v[3] = round(v[3], rd64(&p[24..]));
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total += data.len() as u64;
        if self.nbuf > 0 {
            let take = (32 - self.nbuf).min(data.len());
            self.buf[self.nbuf..self.nbuf + take].copy_from_slice(&data[..take]);
            self.nbuf += take;
            data = &data[take..];
            if self.nbuf < 32 {
                return;
            }
            let b = self.buf;
            Self::stripe(&mut self.v, &b);
            self.nbuf = 0;
        }
        let (stripes, rest) = data.as_chunks::<32>();
        for c in stripes {
            Self::stripe(&mut self.v, c);
        }
        self.buf[..rest.len()].copy_from_slice(rest);
        self.nbuf = rest.len();
    }

    pub fn digest(&self) -> u64 {
        let mut h = if self.total >= 32 {
            let v = self.v;
            let mut h = v[0]
                .rotate_left(1)
                .wrapping_add(v[1].rotate_left(7))
                .wrapping_add(v[2].rotate_left(12))
                .wrapping_add(v[3].rotate_left(18));
            for x in v {
                h = merge(h, x);
            }
            h
        } else {
            self.seed.wrapping_add(P5)
        };
        h = h.wrapping_add(self.total);
        let mut p = &self.buf[..self.nbuf];
        while p.len() >= 8 {
            h ^= round(0, rd64(p));
            h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
            p = &p[8..];
        }
        if p.len() >= 4 {
            h ^= rd32(p).wrapping_mul(P1);
            h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
            p = &p[4..];
        }
        for &b in p {
            h ^= (b as u64).wrapping_mul(P5);
            h = h.rotate_left(11).wrapping_mul(P1);
        }
        h ^= h >> 33;
        h = h.wrapping_mul(P2);
        h ^= h >> 29;
        h = h.wrapping_mul(P3);
        h ^= h >> 32;
        h
    }
}

/// One-shot xxh64.
pub fn xxh64(data: &[u8], seed: u64) -> u64 {
    let mut h = Xxh64::new(seed);
    h.update(data);
    h.digest()
}

fn hash_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
}

/// Content hash of a payload: xxh64 of each [`HASH_BLOCK`] block (seeded by its index), then xxh64 of the block
/// digests and the length. Blocks are hashed in parallel, so large payloads hash at memory speed.
pub fn payload_hash(data: &[u8]) -> u64 {
    let nb = data.len().div_ceil(HASH_BLOCK).max(1);
    let mut digests = vec![0u64; nb];
    let threads = hash_threads().min(nb);
    if threads <= 1 {
        for (i, d) in digests.iter_mut().enumerate() {
            let a = i * HASH_BLOCK;
            let b = (a + HASH_BLOCK).min(data.len());
            *d = xxh64(&data[a..b], i as u64);
        }
    } else {
        let per = nb.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, chunk) in digests.chunks_mut(per).enumerate() {
                s.spawn(move || {
                    for (j, d) in chunk.iter_mut().enumerate() {
                        let i = ci * per + j;
                        let a = i * HASH_BLOCK;
                        let b = (a + HASH_BLOCK).min(data.len());
                        *d = xxh64(&data[a..b], i as u64);
                    }
                });
            }
        });
    }
    combine_block_digests(&digests, data.len() as u64)
}

fn combine_block_digests(digests: &[u64], len: u64) -> u64 {
    let mut h = Xxh64::new(0x5450_4342_4c4b_5331); // "TPCBLKS1"
    for d in digests {
        h.update(&d.to_le_bytes());
    }
    h.update(&len.to_le_bytes());
    h.digest()
}

/// [`payload_hash`] of a file region, read block-parallel with `pread`.
fn file_payload_hash(f: &File, off: u64, len: u64) -> io::Result<u64> {
    let mut out = vec![0u8; 0];
    read_and_hash(f, off, len, &mut out, false)
}

/// Read `len` bytes at `off` into `out` (when `keep`) while computing [`payload_hash`]; block-parallel.
fn read_and_hash(f: &File, off: u64, len: u64, out: &mut Vec<u8>, keep: bool) -> io::Result<u64> {
    let len_us = usize::try_from(len).map_err(|_| io::Error::other("payload too large"))?;
    let nb = len_us.div_ceil(HASH_BLOCK).max(1);
    let mut digests = vec![0u64; nb];
    if keep {
        out.clear();
        out.try_reserve_exact(len_us)
            .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "cannot allocate"))?;
        // SAFETY: every byte in 0..len is written by read_exact_at below before `out` is used; on error `out` is
        // cleared again.
        #[allow(clippy::uninit_vec)]
        unsafe {
            out.set_len(len_us)
        };
    }
    let threads = hash_threads().min(nb);
    let per = nb.div_ceil(threads);
    let res: io::Result<()> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut rest: &mut [u8] = if keep { &mut out[..] } else { &mut [] };
        for (ci, chunk) in digests.chunks_mut(per).enumerate() {
            let first = ci * per;
            let a = first * HASH_BLOCK;
            let b = ((first + chunk.len()) * HASH_BLOCK).min(len_us);
            let mine: &mut [u8] = if keep {
                let (m, r) = std::mem::take(&mut rest).split_at_mut(b - a);
                rest = r;
                m
            } else {
                &mut []
            };
            handles.push(s.spawn(move || -> io::Result<()> {
                let mut tmp = if keep {
                    Vec::new()
                } else {
                    vec![0u8; HASH_BLOCK.min(b - a)]
                };
                for (j, d) in chunk.iter_mut().enumerate() {
                    let i = first + j;
                    let ba = i * HASH_BLOCK;
                    let bb = (ba + HASH_BLOCK).min(len_us);
                    let dst: &mut [u8] = if keep {
                        &mut mine[ba - a..bb - a]
                    } else {
                        &mut tmp[..bb - ba]
                    };
                    f.read_exact_at(dst, off + ba as u64)?;
                    *d = xxh64(dst, i as u64);
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join()
                .map_err(|_| io::Error::other("hash thread panicked"))??;
        }
        Ok(())
    });
    if let Err(e) = res {
        if keep {
            out.clear();
        }
        return Err(e);
    }
    Ok(combine_block_digests(&digests, len))
}

// ============================================================================
// Identity
// ============================================================================

/// Typed, length-delimited identity fields hashed into one 64-bit fingerprint. Every field is also kept as text
/// (`desc`) and written next to the cache as `identity.txt` for humans.
#[derive(Clone)]
pub struct IdentityBuilder {
    h: Xxh64,
    desc: String,
}

impl Default for IdentityBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IdentityBuilder {
    pub fn new() -> Self {
        let mut h = Xxh64::new(0x5450_4349_4445_4e31); // "TPCIDEN1"
        h.update(&VERSION.to_le_bytes());
        Self {
            h,
            desc: String::new(),
        }
    }

    fn field(&mut self, k: &str, ty: u8, v: &[u8]) {
        self.h.update(&(k.len() as u64).to_le_bytes());
        self.h.update(k.as_bytes());
        self.h.update(&[ty]);
        self.h.update(&(v.len() as u64).to_le_bytes());
        self.h.update(v);
    }

    pub fn str(&mut self, k: &str, v: &str) -> &mut Self {
        self.field(k, b's', v.as_bytes());
        self.desc.push_str(&format!("{k} = {v}\n"));
        self
    }

    pub fn u64(&mut self, k: &str, v: u64) -> &mut Self {
        self.field(k, b'u', &v.to_le_bytes());
        self.desc.push_str(&format!("{k} = {v}\n"));
        self
    }

    /// Sampled fingerprint of a file: size, mtime, and the first, middle and last MiB.
    /// A missing file is recorded as absent (so adding it later changes the identity).
    pub fn file(&mut self, k: &str, path: &Path) -> &mut Self {
        match file_fingerprint(path) {
            Ok(fp) => {
                self.field(k, b'f', &fp.to_le_bytes());
                self.desc
                    .push_str(&format!("{k} = {} [{fp:016x}]\n", path.display()));
            }
            Err(_) => {
                self.field(k, b'-', path.as_os_str().as_encoded_bytes());
                self.desc
                    .push_str(&format!("{k} = {} [absent]\n", path.display()));
            }
        }
        self
    }

    /// The running executable (the engine build).
    pub fn exe(&mut self) -> &mut Self {
        let p = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("/proc/self/exe"));
        self.file("exe", &p)
    }

    /// Every shared object mapped into this process whose path contains one of `needles` (kernel library, HIP
    /// runtime, BLAS libraries). Call after the engine has loaded its libraries.
    pub fn mapped_libs(&mut self, needles: &[&str]) -> &mut Self {
        let mut libs: Vec<String> = fs::read_to_string("/proc/self/maps")
            .unwrap_or_default()
            .lines()
            .filter_map(|l| l.split_whitespace().nth(5).map(str::to_string))
            .filter(|p| p.contains(".so") && needles.iter().any(|n| p.contains(n)))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        libs.sort();
        self.u64("libs", libs.len() as u64);
        for l in libs {
            self.file("lib", Path::new(&l));
        }
        self
    }

    /// Every environment variable whose name starts with one of `prefixes`, minus `exclude` prefixes, sorted.
    pub fn env(&mut self, prefixes: &[&str], exclude: &[&str]) -> &mut Self {
        let mut vars: Vec<(String, String)> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.to_string_lossy().into_owned())))
            .filter(|(k, _)| prefixes.iter().any(|p| k.starts_with(p)))
            .filter(|(k, _)| !exclude.iter().any(|p| k.starts_with(p)))
            .collect();
        vars.sort();
        self.u64("env", vars.len() as u64);
        for (k, v) in vars {
            self.str(&format!("env.{k}"), &v);
        }
        self
    }

    /// Arguments, minus the flags in `skip` (each with its value when the next item is not a flag).
    pub fn args(&mut self, args: &[String], skip: &[&str]) -> &mut Self {
        let kept = filter_args(args, skip);
        self.u64("args", kept.len() as u64);
        for a in kept {
            self.str("arg", &a);
        }
        self
    }

    /// The GPU target(s) from the KFD topology (no HIP call, so it is safe in every engine process).
    pub fn gpu(&mut self) -> &mut Self {
        let mut targets: Vec<String> = fs::read_dir("/sys/class/kfd/kfd/topology/nodes")
            .map(|rd| {
                rd.flatten()
                    .filter_map(|d| fs::read_to_string(d.path().join("properties")).ok())
                    .filter_map(|p| {
                        p.lines()
                            .find(|l| l.starts_with("gfx_target_version "))
                            .map(|l| l["gfx_target_version ".len()..].trim().to_string())
                    })
                    .filter(|v| v != "0")
                    .collect()
            })
            .unwrap_or_default();
        targets.sort();
        self.str("gpu", &targets.join(","))
    }

    /// The fields every engine process shares: build ([`Self::exe`]), kernel / HIP / BLAS libraries, GPU target,
    /// numerics environment ([`ENV_PREFIXES`] minus [`ENV_NOT_IN_IDENTITY`]) and arguments minus `skip_args`.
    pub fn process(&mut self, skip_args: &[&str]) -> &mut Self {
        self.exe().mapped_libs(LIBS_IN_IDENTITY).gpu();
        self.env(ENV_PREFIXES, ENV_NOT_IN_IDENTITY);
        let args: Vec<String> = std::env::args().skip(1).collect();
        self.args(&args, skip_args)
    }

    pub fn finish(&self) -> DiskIdentity {
        DiskIdentity {
            hash: self.h.digest(),
            desc: self.desc.clone(),
        }
    }
}

/// Environment that can change numerics or kernel choice (every variable under these prefixes counts).
pub const ENV_PREFIXES: &[&str] = &["TALUS_", "STRATA_", "GLM_", "HIPBLASLT_", "ROCBLAS_"];
/// Environment under those prefixes that does not: the disk tier's own settings and diagnostics logs.
pub const ENV_NOT_IN_IDENTITY: &[&str] = &["TALUS_PCACHE_", "TALUS_LAGUNA_GENLOG"];
/// Shared objects whose build changes numerics (kernel library, HIP runtime, BLAS).
pub const LIBS_IN_IDENTITY: &[&str] = &["talus", "amdhip64", "hipblaslt", "rocblas"];
/// Flags that never change what cached bytes mean (cache sizes and locations, serve mode).
pub const ARGS_NOT_IN_IDENTITY: &[&str] = &[
    "--prompt-cache-disk",
    "--prompt-cache-disk-gib",
    "--prompt-cache-disk-min",
    "--prompt-cache",
    "--kv-stash-mib",
    "--snap-msg",
    "--serve",
    "--engine-protocol",
];

/// `args` without the flags in `skip`; a skipped flag also drops its value (the next item, unless it starts
/// with `--`), and `--flag=value` forms are dropped as a whole.
pub fn filter_args(args: &[String], skip: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let name = a.split('=').next().unwrap_or(a);
        if skip.contains(&name) {
            if !a.contains('=') && i + 1 < args.len() && !args[i + 1].starts_with("--") {
                i += 1;
            }
        } else {
            out.push(a.clone());
        }
        i += 1;
    }
    out
}

/// Sampled file fingerprint: size, mtime (ns) and the first, middle and last MiB.
pub fn file_fingerprint(path: &Path) -> io::Result<u64> {
    let f = File::open(path)?;
    let md = f.metadata()?;
    let size = md.len();
    let mtime = md
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut h = Xxh64::new(0x4649_4c45_5341_4d50); // "FILESAMP"
    h.update(&size.to_le_bytes());
    h.update(&mtime.to_le_bytes());
    const MIB: u64 = 1 << 20;
    let mut buf = vec![0u8; MIB as usize];
    for off in [0, size / 2, size.saturating_sub(MIB)] {
        let n = MIB.min(size.saturating_sub(off)) as usize;
        if n > 0 {
            f.read_exact_at(&mut buf[..n], off)?;
            h.update(&buf[..n]);
        }
    }
    Ok(h.digest())
}

/// The cache identity: hash plus a human-readable description of its fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskIdentity {
    pub hash: u64,
    pub desc: String,
}

// ============================================================================
// Errors, config, entries
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiskError {
    /// Read or write failed (disk full, permissions, ...).
    Io(String),
    /// The file failed a check (truncated, bad hash, bad header): it was deleted.
    Corrupt(String),
    /// The file belongs to another identity or engine geometry: it was ignored or deleted.
    Mismatch(String),
    /// The entry would not fit the budget or the memory floor.
    Budget(String),
    /// A write is already in flight.
    Busy,
}

impl std::fmt::Display for DiskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiskError::Io(s) => write!(f, "io: {s}"),
            DiskError::Corrupt(s) => write!(f, "corrupt: {s}"),
            DiskError::Mismatch(s) => write!(f, "mismatch: {s}"),
            DiskError::Budget(s) => write!(f, "budget: {s}"),
            DiskError::Busy => write!(f, "busy"),
        }
    }
}

impl std::error::Error for DiskError {}

fn io_err(what: &str, e: io::Error) -> DiskError {
    DiskError::Io(format!("{what}: {e}"))
}

/// Disk tier settings.
#[derive(Clone, Debug)]
pub struct DiskCacheConfig {
    /// Root folder (entries go to `<dir>/<identity hex>/`).
    pub dir: PathBuf,
    /// Byte budget across the whole root folder (all identities).
    pub budget_bytes: u64,
    /// Prompts shorter than this are not written (engine policy; kept here so every engine reads one setting).
    pub min_tokens: usize,
    /// `MemAvailable` that must remain after staging a write.
    pub mem_floor_bytes: u64,
    /// Keep the cache filesystem's free space above [`fs_floor`] (off only in tests).
    pub fs_floor: bool,
}

impl DiskCacheConfig {
    /// Read the shared environment switches:
    /// - `TALUS_PCACHE_DISK`: folder; unset / empty / `auto` / `1` / `on` = the per-user default
    ///   (`$XDG_CACHE_HOME/talus/pcache` or `~/.cache/talus/pcache`); `0` / `off` / `false` / `no` = no disk tier;
    /// - `TALUS_PCACHE_DISK_GIB`: budget in GiB (default 32);
    /// - `TALUS_PCACHE_DISK_MIN`: minimum prompt tokens per entry (default 1024).
    pub fn from_env() -> Option<Self> {
        Self::from_values(
            std::env::var("TALUS_PCACHE_DISK").ok().as_deref(),
            std::env::var("TALUS_PCACHE_DISK_GIB").ok().as_deref(),
            std::env::var("TALUS_PCACHE_DISK_MIN").ok().as_deref(),
        )
    }

    /// Parse the settings from their text values (the same rules as [`Self::from_env`]).
    pub fn from_values(dir: Option<&str>, gib: Option<&str>, min: Option<&str>) -> Option<Self> {
        let dir = dir.map(str::trim).unwrap_or("");
        if is_off_switch(dir) {
            return None;
        }
        let dir = if dir.is_empty()
            || dir == "1"
            || dir.eq_ignore_ascii_case("auto")
            || dir.eq_ignore_ascii_case("on")
            || dir.eq_ignore_ascii_case("true")
        {
            default_dir()?
        } else if let Some(rest) = dir.strip_prefix("~/") {
            PathBuf::from(std::env::var_os("HOME")?).join(rest)
        } else {
            PathBuf::from(dir)
        };
        let budget_bytes = gib
            .and_then(|g| g.trim().parse::<f64>().ok())
            .filter(|g| g.is_finite() && *g >= 0.0)
            .map(|g| (g * (1u64 << 30) as f64) as u64)
            .unwrap_or(DEFAULT_BUDGET_BYTES);
        if budget_bytes == 0 {
            return None;
        }
        let min_tokens = min
            .and_then(|m| m.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MIN_TOKENS);
        Some(Self {
            dir,
            budget_bytes,
            min_tokens,
            mem_floor_bytes: DEFAULT_MEM_FLOOR_BYTES,
            fs_floor: true,
        })
    }
}

/// The values of `TALUS_PCACHE_DISK` / `--prompt-cache-disk` that turn the disk tier off.
pub fn is_off_switch(v: &str) -> bool {
    let v = v.trim();
    ["0", "off", "false", "no", "none"]
        .iter()
        .any(|x| v.eq_ignore_ascii_case(x))
}

/// Free space the cache never eats into: max(20 GiB, 10 % of the filesystem). Writes that would leave less are
/// skipped (eviction still runs), logged once per process.
pub const FS_FLOOR_MIN_BYTES: u64 = 20 << 30;

/// (total, available-to-us) bytes of the filesystem holding `dir`.
pub fn fs_space(dir: &Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    extern "C" {
        fn statvfs(path: *const std::ffi::c_char, buf: *mut u64) -> i32;
    }
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    // struct statvfs on 64-bit Linux (glibc / musl): f_bsize, f_frsize, f_blocks, f_bfree, f_bavail, ... (all 64-bit
    // words, 112 bytes); 32 words leave room.
    let mut buf = [0u64; 32];
    // SAFETY: `c` is a NUL-terminated path and `buf` is larger than `struct statvfs`.
    if unsafe { statvfs(c.as_ptr(), buf.as_mut_ptr()) } != 0 {
        return None;
    }
    let frsize = if buf[1] != 0 { buf[1] } else { buf[0] };
    Some((buf[2].saturating_mul(frsize), buf[4].saturating_mul(frsize)))
}

/// The free-space floor for a filesystem of `total` bytes.
pub fn fs_floor(total: u64) -> u64 {
    FS_FLOOR_MIN_BYTES.max(total / 10)
}

/// `$XDG_CACHE_HOME/talus/pcache`, else `~/.cache/talus/pcache`.
pub fn default_dir() -> Option<PathBuf> {
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME").filter(|x| !x.is_empty()) {
        return Some(PathBuf::from(x).join("talus/pcache"));
    }
    Some(PathBuf::from(std::env::var_os("HOME")?).join(".cache/talus/pcache"))
}

/// A section's place in an entry file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionDesc {
    pub tag: u64,
    pub offset: u64,
    pub len: u64,
    pub hash: u64,
}

impl SectionDesc {
    pub fn is_external(&self) -> bool {
        self.offset == OFF_EXTERNAL
    }
}

/// One entry of the in-memory index (header data only; payloads stay on disk until loaded).
#[derive(Clone, Debug)]
pub struct DiskEntry {
    pub path: PathBuf,
    pub tokens: Arc<Vec<i32>>,
    pub meta: Vec<u8>,
    pub sections: Vec<SectionDesc>,
    /// Entry file bytes plus the external payload, if any.
    pub bytes: u64,
    /// The deepest prefix this entry can restore (engine-defined, `<= tokens.len()`).
    pub depth: u64,
}

impl DiskEntry {
    /// Path of the external payload file (`<stem>.tpcx`).
    pub fn payload_path(&self) -> PathBuf {
        self.path.with_extension(EXT_PAYLOAD)
    }

    pub fn has_external(&self) -> bool {
        self.sections.iter().any(SectionDesc::is_external)
    }
}

/// A loaded entry: verified meta and section bytes.
#[derive(Debug)]
pub struct Loaded {
    pub tokens: Arc<Vec<i32>>,
    pub meta: Vec<u8>,
    pub sections: Vec<(u64, Vec<u8>)>,
    /// The verified external payload file (engine-native session file), if the entry has one.
    pub external: Option<PathBuf>,
    pub bytes: u64,
    pub ms: f64,
}

impl Loaded {
    pub fn section(&self, t: u64) -> Option<&[u8]> {
        self.sections
            .iter()
            .find(|(k, _)| *k == t)
            .map(|(_, v)| v.as_slice())
    }

    pub fn take_section(&mut self, t: u64) -> Option<Vec<u8>> {
        let i = self.sections.iter().position(|(k, _)| *k == t)?;
        Some(self.sections.swap_remove(i).1)
    }
}

/// What a write stores.
#[derive(Debug, Default)]
pub struct WriteJob {
    pub tokens: Vec<i32>,
    pub meta: Vec<u8>,
    pub sections: Vec<(u64, Vec<u8>)>,
    /// An engine-native payload already written (unsynced) by the engine; the writer hashes, flushes and renames
    /// it to `<entry>.tpcx` (tag [`TAG_EXTERNAL`]). It must be on the same filesystem as the cache.
    pub external: Option<PathBuf>,
    /// The deepest prefix of `tokens` this entry can restore (0 = `tokens.len()`).
    pub depth: u64,
    /// Once the new entry is in place, delete entries of this identity that it makes redundant: older entries
    /// whose restorable prefix (`depth` tokens) is also a prefix of the new entry and no deeper than its `depth`
    /// (the same conversation, earlier turns), except the deepest of them (the previous turn).
    pub supersede: bool,
}

impl WriteJob {
    pub fn payload_bytes(&self) -> u64 {
        self.sections.iter().map(|(_, d)| d.len() as u64).sum()
    }
}

/// Tag of the external payload section.
pub const TAG_EXTERNAL: u64 = tag(b"EXTERNAL");

/// Counters (all monotonic).
#[derive(Debug, Default)]
pub struct DiskStats {
    pub writes: AtomicU64,
    pub write_bytes: AtomicU64,
    pub write_errors: AtomicU64,
    pub skipped_busy: AtomicU64,
    pub hits: AtomicU64,
    pub read_bytes: AtomicU64,
    pub corrupt: AtomicU64,
    pub evicted: AtomicU64,
}

struct Shared {
    root: PathBuf,
    base: PathBuf,
    identity: u64,
    budget: u64,
    index: Mutex<Vec<DiskEntry>>,
    busy: AtomicBool,
    idle: (Mutex<bool>, Condvar),
    stats: DiskStats,
    last_write_ms: Mutex<f64>,
}

/// The disk tier of one engine process.
pub struct DiskCache {
    cfg: DiskCacheConfig,
    ident: DiskIdentity,
    shared: Arc<Shared>,
    tx: Option<SyncSender<WriteJob>>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for DiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCache")
            .field("root", &self.shared.root)
            .field("identity", &format_args!("{:016x}", self.ident.hash))
            .field("budget", &self.cfg.budget_bytes)
            .finish()
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn pid_alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Flush a folder so a rename inside it is durable.
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

/// `MemAvailable` from `/proc/meminfo`, in bytes.
pub fn mem_available_bytes() -> Option<u64> {
    let s = fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb << 10)
}

impl DiskCache {
    /// Open (and create) the cache folder for `ident`, index the valid entries and delete broken ones.
    pub fn open(cfg: DiskCacheConfig, ident: DiskIdentity) -> Result<Self, DiskError> {
        let root = cfg.dir.join(format!("{:016x}", ident.hash));
        fs::create_dir_all(&root).map_err(|e| io_err("create cache folder", e))?;
        let _ = fs::write(root.join("identity.txt"), ident.desc.as_bytes());
        let shared = Arc::new(Shared {
            root: root.clone(),
            base: cfg.dir.clone(),
            identity: ident.hash,
            budget: cfg.budget_bytes,
            index: Mutex::new(Vec::new()),
            busy: AtomicBool::new(false),
            idle: (Mutex::new(true), Condvar::new()),
            stats: DiskStats::default(),
            last_write_ms: Mutex::new(0.0),
        });
        let index = scan(&shared);
        *shared.index.lock().unwrap() = index;
        // The budget may have shrunk since the last run, or files may have been added from outside.
        enforce_budget(&shared);
        let (tx, rx) = sync_channel::<WriteJob>(1);
        let sh = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("talus-pcache-disk".into())
            .spawn(move || writer_loop(sh, rx))
            .map_err(|e| io_err("spawn writer", e))?;
        Ok(Self {
            cfg,
            ident,
            shared,
            tx: Some(tx),
            worker: Some(worker),
        })
    }

    pub fn config(&self) -> &DiskCacheConfig {
        &self.cfg
    }

    pub fn identity(&self) -> &DiskIdentity {
        &self.ident
    }

    pub fn root(&self) -> &Path {
        &self.shared.root
    }

    pub fn stats(&self) -> &DiskStats {
        &self.shared.stats
    }

    pub fn last_write_ms(&self) -> f64 {
        *self.shared.last_write_ms.lock().unwrap()
    }

    /// Snapshot of the index.
    pub fn entries(&self) -> Vec<DiskEntry> {
        self.shared.index.lock().unwrap().clone()
    }

    pub fn len(&self) -> usize {
        self.shared.index.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes indexed for this identity.
    pub fn used_bytes(&self) -> u64 {
        self.shared
            .index
            .lock()
            .unwrap()
            .iter()
            .map(|e| e.bytes)
            .sum()
    }

    /// The entry with the largest `usable(entry)` (> 0); `usable` returns how many tokens of the request the entry
    /// can restore (0 = unusable). Ties go to the most recently written entry.
    pub fn best<F: Fn(&DiskEntry) -> usize>(&self, usable: F) -> Option<(DiskEntry, usize)> {
        let idx = self.shared.index.lock().unwrap();
        let mut best: Option<(DiskEntry, usize)> = None;
        for e in idx.iter().rev() {
            let n = usable(e);
            if n > 0 && best.as_ref().is_none_or(|(_, b)| n > *b) {
                best = Some((e.clone(), n));
            }
        }
        best
    }

    /// For engines that restore a whole saved session (KV of every token it holds): the entry sharing the longest
    /// prefix with `ids`, capped at `ids.len() - 1` (at least one token is always read). `feasible(entry, n)` can
    /// reject entries the engine could not resume from at `n`.
    pub fn best_native<F: Fn(&DiskEntry, usize) -> bool>(
        &self,
        ids: &[i32],
        feasible: F,
    ) -> Option<(DiskEntry, usize)> {
        let cap = ids.len().saturating_sub(1);
        self.best(|e| {
            if !e.has_external() {
                return 0;
            }
            let n = lcp(&e.tokens, ids).min(cap);
            if n > 0 && feasible(e, n) {
                n
            } else {
                0
            }
        })
    }

    /// Write an engine-native session for `tokens` in the background: skipped when shorter than `min_tokens`,
    /// already on disk, or while a write is in flight. `save(path)` writes the native file (unsynced) and returns
    /// its size; the writer thread hashes, flushes and indexes it. `depth` is the prompt length (see
    /// [`WriteJob::supersede`]). Returns `Ok(true)` when a write was started.
    pub fn save_native<F: FnOnce(&Path) -> Result<u64, String>>(
        &self,
        tokens: &[i32],
        depth: usize,
        save: F,
    ) -> Result<bool, DiskError> {
        if tokens.len() < self.cfg.min_tokens.max(1) {
            return Ok(false);
        }
        let known = self
            .best(|e| usize::from(e.has_external() && e.tokens.as_slice() == tokens))
            .is_some();
        if known {
            return Ok(false);
        }
        if self.is_busy() {
            self.shared
                .stats
                .skipped_busy
                .fetch_add(1, Ordering::Relaxed);
            return Err(DiskError::Busy);
        }
        self.space_ok(0)?;
        let tmp = native_tmp_path(self, "native");
        let bytes = match save(&tmp) {
            Ok(b) => b,
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                return Err(DiskError::Io(format!("native save: {e}")));
            }
        };
        if bytes > self.cfg.budget_bytes {
            let _ = fs::remove_file(&tmp);
            return Err(DiskError::Budget(format!(
                "{} MiB session exceeds the {} MiB disk budget",
                bytes >> 20,
                self.cfg.budget_bytes >> 20
            )));
        }
        self.submit(WriteJob {
            tokens: tokens.to_vec(),
            meta: Vec::new(),
            sections: Vec::new(),
            external: Some(tmp),
            depth: depth.min(tokens.len()) as u64,
            supersede: true,
        })?;
        Ok(true)
    }

    /// Whether writing `bytes` more keeps the cache filesystem above its free-space floor ([`fs_floor`]). The first
    /// refusal of the process is logged.
    pub fn space_ok(&self, bytes: u64) -> Result<(), DiskError> {
        if !self.cfg.fs_floor {
            return Ok(());
        }
        let Some((total, avail)) = fs_space(&self.shared.root) else {
            return Ok(());
        };
        let floor = fs_floor(total);
        if avail.saturating_sub(bytes) >= floor {
            return Ok(());
        }
        static LOGGED: AtomicBool = AtomicBool::new(false);
        let msg = format!(
            "filesystem free space {} GiB would drop below the {} GiB floor; the disk cache does not grow",
            avail >> 30,
            floor >> 30
        );
        if !LOGGED.swap(true, Ordering::Relaxed) {
            eprintln!("talus-pcache: {msg} (logged once)");
        }
        Err(DiskError::Budget(msg))
    }

    /// True while a write is in flight (a new write would be skipped).
    pub fn is_busy(&self) -> bool {
        self.shared.busy.load(Ordering::Acquire)
    }

    /// Whether staging `bytes` of host memory keeps `MemAvailable` above the floor, the entry fits the budget and
    /// no write is in flight. Engines call this before copying state off the GPU.
    pub fn can_stage(&self, bytes: u64) -> Result<(), DiskError> {
        if self.is_busy() {
            self.shared
                .stats
                .skipped_busy
                .fetch_add(1, Ordering::Relaxed);
            return Err(DiskError::Busy);
        }
        if bytes > self.cfg.budget_bytes {
            return Err(DiskError::Budget(format!(
                "{} MiB entry exceeds the {} MiB disk budget",
                bytes >> 20,
                self.cfg.budget_bytes >> 20
            )));
        }
        self.space_ok(bytes)?;
        if let Some(avail) = mem_available_bytes() {
            if avail < bytes.saturating_add(self.cfg.mem_floor_bytes) {
                return Err(DiskError::Budget(format!(
                    "staging {} MiB would leave less than {} MiB available (now {} MiB)",
                    bytes >> 20,
                    self.cfg.mem_floor_bytes >> 20,
                    avail >> 20
                )));
            }
        }
        Ok(())
    }

    /// Hand a write to the background writer. Returns `Busy` (and drops the job) if a write is in flight.
    pub fn submit(&self, job: WriteJob) -> Result<(), DiskError> {
        if self
            .shared
            .busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.shared
                .stats
                .skipped_busy
                .fetch_add(1, Ordering::Relaxed);
            discard_external(&job);
            return Err(DiskError::Busy);
        }
        WRITES_IN_FLIGHT.fetch_add(1, Ordering::AcqRel);
        *self.shared.idle.0.lock().unwrap() = false;
        let tx = self.tx.as_ref().expect("writer alive");
        match tx.try_send(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(job)) | Err(TrySendError::Disconnected(job)) => {
                discard_external(&job);
                mark_idle(&self.shared);
                Err(DiskError::Busy)
            }
        }
    }

    /// Write synchronously on the calling thread (tests, shutdown paths).
    pub fn write_now(&self, job: WriteJob) -> Result<DiskEntry, DiskError> {
        write_entry(&self.shared, job)
    }

    /// Wait until the background writer is idle (or `timeout` passes). Returns true when idle.
    pub fn flush(&self, timeout: Duration) -> bool {
        let (m, cv) = &self.shared.idle;
        let g = m.lock().unwrap();
        let (g, _) = cv.wait_timeout_while(g, timeout, |idle| !*idle).unwrap();
        *g
    }

    /// Read and verify an entry. On a failed check the files are deleted and the entry leaves the index.
    pub fn load(&self, e: &DiskEntry) -> Result<Loaded, DiskError> {
        let t0 = Instant::now();
        let r = load_entry(&self.shared, e);
        match &r {
            Ok(l) => {
                self.shared.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.shared
                    .stats
                    .read_bytes
                    .fetch_add(l.bytes, Ordering::Relaxed);
                touch(&e.path);
            }
            Err(DiskError::Corrupt(_)) | Err(DiskError::Mismatch(_)) => {
                self.shared.stats.corrupt.fetch_add(1, Ordering::Relaxed);
                self.remove(e);
            }
            Err(_) => {
                if !e.path.exists() {
                    self.forget(&e.path);
                }
            }
        }
        r.map(|mut l| {
            l.ms = t0.elapsed().as_secs_f64() * 1000.0;
            l
        })
    }

    /// Delete an entry (both files) and drop it from the index.
    pub fn remove(&self, e: &DiskEntry) {
        let _ = fs::remove_file(&e.path);
        let _ = fs::remove_file(e.payload_path());
        self.forget(&e.path);
    }

    fn forget(&self, p: &Path) {
        self.shared.index.lock().unwrap().retain(|x| x.path != p);
    }

    /// Evict least recently used files (all identities) until the root is within budget.
    pub fn enforce_budget(&self) {
        enforce_budget(&self.shared);
    }

    /// Re-read the folder (another process of the same identity may have written).
    pub fn rescan(&self) {
        let idx = scan(&self.shared);
        *self.shared.index.lock().unwrap() = idx;
    }
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        // Let an in-flight write finish (bounded), then stop the writer.
        self.flush(Duration::from_secs(120));
        self.tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn discard_external(job: &WriteJob) {
    if let Some(p) = &job.external {
        let _ = fs::remove_file(p);
    }
}

fn mark_idle(sh: &Shared) {
    sh.busy.store(false, Ordering::Release);
    let (m, cv) = &sh.idle;
    *m.lock().unwrap() = true;
    cv.notify_all();
    WRITES_IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
}

// ------------------------------------------------------------------------------------------------- termination
//
// A server stop sends SIGTERM to the whole process group (systemd, `kill -- -pgid`) or, through
// `PR_SET_PDEATHSIG`, to the engine once its server is gone. The default action would kill the engine right after
// the reply, which is exactly when the turn's disk write runs, so a restart would never find the entry it was
// meant to resume from. With the disk tier on, the engine catches SIGTERM / SIGINT / SIGHUP and exits only once
// the finished turn is durable (bounded); with nothing pending it exits at once.

static TERM_SIGNAL: AtomicI32 = AtomicI32::new(0);
static SAVES_IN_PROGRESS: AtomicUsize = AtomicUsize::new(0);
static WRITES_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Held by an engine from just before it sends DONE until the turn's write has been handed to the writer (or
/// skipped): a termination signal arriving in between waits for the write instead of losing it.
#[must_use]
#[derive(Debug)]
pub struct SaveGuard(());

impl SaveGuard {
    pub fn new() -> Self {
        SAVES_IN_PROGRESS.fetch_add(1, Ordering::AcqRel);
        Self(())
    }
}

impl Default for SaveGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SaveGuard {
    fn drop(&mut self) {
        SAVES_IN_PROGRESS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// True while a turn is being staged ([`SaveGuard`]) or a write is in flight in any disk cache of this process.
pub fn disk_work_pending() -> bool {
    SAVES_IN_PROGRESS.load(Ordering::Acquire) > 0 || WRITES_IN_FLIGHT.load(Ordering::Acquire) > 0
}

/// Wait until no disk work is pending (or `grace` passes). Returns true when quiet.
pub fn wait_disk_quiet(grace: Duration) -> bool {
    let t0 = Instant::now();
    while disk_work_pending() {
        if t0.elapsed() >= grace {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    true
}

extern "C" fn on_term_signal(sig: i32) {
    // async-signal-safe: one atomic store; the watcher thread does the rest
    let _ = TERM_SIGNAL.compare_exchange(0, sig, Ordering::AcqRel, Ordering::Acquire);
}

/// Catch SIGTERM / SIGINT / SIGHUP (a signal the parent set to ignored stays ignored): on the first one, wait up
/// to `grace` for the pending disk work ([`disk_work_pending`]), then `_exit(128 + signal)`. Idempotent; a
/// watcher thread polls every 10 ms.
pub fn install_term_handler(grace: Duration) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        extern "C" {
            fn signal(sig: i32, handler: usize) -> usize;
        }
        const SIG_IGN: usize = 1;
        const SIGHUP: i32 = 1;
        const SIGINT: i32 = 2;
        const SIGTERM: i32 = 15;
        let handler = on_term_signal as extern "C" fn(i32) as usize;
        for sig in [SIGHUP, SIGINT, SIGTERM] {
            // SAFETY: installing a handler that only performs an atomic store; glibc `signal` has BSD semantics
            // (SA_RESTART), so blocking reads on stdin are restarted.
            unsafe {
                if signal(sig, handler) == SIG_IGN {
                    signal(sig, SIG_IGN);
                }
            }
        }
        let spawned = std::thread::Builder::new()
            .name("talus-pcache-term".into())
            .spawn(move || loop {
                std::thread::sleep(Duration::from_millis(10));
                let sig = TERM_SIGNAL.load(Ordering::Acquire);
                if sig == 0 {
                    continue;
                }
                if disk_work_pending() {
                    eprintln!("talus-pcache: signal {sig}: finishing the disk write before exit");
                    let t0 = Instant::now();
                    let quiet = wait_disk_quiet(grace);
                    eprintln!(
                        "talus-pcache: {} after {:.0} ms, exiting",
                        if quiet {
                            "disk write done"
                        } else {
                            "disk write still running"
                        },
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
                extern "C" {
                    fn _exit(code: i32) -> !;
                }
                // SAFETY: immediate process exit; every completed write is already fsynced and renamed, an
                // unfinished one leaves only a temporary that the next open deletes.
                unsafe { _exit(128 + sig) }
            });
        if let Err(e) = spawned {
            eprintln!(
                "talus-pcache: no termination watcher ({e}); a stop may lose the last turn's write"
            );
        }
    });
}

/// The writer pid in a temporary's name (`.<entry>.tmp.<pid>`, `.<stem>-<n>.native.tmp.<pid>` and the engine
/// saver's own `<that>.tmp`).
pub(crate) fn tmp_pid(name: &str) -> Option<u32> {
    let (_, rest) = name.rsplit_once(".tmp.")?;
    rest.split('.').next()?.parse().ok()
}

fn writer_loop(sh: Arc<Shared>, rx: Receiver<WriteJob>) {
    while let Ok(job) = rx.recv() {
        let t0 = Instant::now();
        let r = write_entry(&sh, job);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        *sh.last_write_ms.lock().unwrap() = ms;
        match r {
            Ok(e) => eprintln!(
                "talus-pcache: disk entry {} tokens, {:.1} MiB written in {:.0} ms",
                e.tokens.len(),
                e.bytes as f64 / (1u64 << 20) as f64,
                ms
            ),
            Err(err) => eprintln!("talus-pcache: disk write skipped ({err})"),
        }
        mark_idle(&sh);
    }
}

fn touch(p: &Path) {
    if let Ok(f) = OpenOptions::new().write(true).open(p) {
        let _ = f.set_modified(SystemTime::now());
    }
}

fn entry_name(identity: u64, tokens: &[i32]) -> String {
    let mut h = Xxh64::new(identity);
    for t in tokens {
        h.update(&t.to_le_bytes());
    }
    format!("{:016x}-{}.{EXT}", h.digest(), tokens.len())
}

fn tokens_bytes(t: &[i32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(t.len() * 4);
    for x in t {
        v.extend_from_slice(&x.to_le_bytes());
    }
    v
}

fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// Serialize the head region (fixed header, tokens, meta, section table) with `head_hash` filled in.
fn build_head(
    identity: u64,
    tokens: &[i32],
    depth: u64,
    meta: &[u8],
    secs: &[SectionDesc],
) -> (Vec<u8>, u64) {
    let tb = tokens_bytes(tokens);
    let mut v = Vec::with_capacity(HEADER + tb.len() + meta.len() + secs.len() * 32);
    v.extend_from_slice(&MAGIC);
    v.extend_from_slice(&VERSION.to_le_bytes());
    v.extend_from_slice(&(secs.len() as u32).to_le_bytes());
    v.extend_from_slice(&identity.to_le_bytes());
    v.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    v.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    v.extend_from_slice(&depth.to_le_bytes());
    v.extend_from_slice(&now_unix().to_le_bytes());
    v.extend_from_slice(&0u64.to_le_bytes()); // head hash, below
    v.extend_from_slice(&tb);
    v.extend_from_slice(meta);
    for s in secs {
        v.extend_from_slice(&s.tag.to_le_bytes());
        v.extend_from_slice(&s.offset.to_le_bytes());
        v.extend_from_slice(&s.len.to_le_bytes());
        v.extend_from_slice(&s.hash.to_le_bytes());
    }
    let h = head_hash(&v);
    v[56..64].copy_from_slice(&h.to_le_bytes());
    (v, h)
}

fn head_hash(head: &[u8]) -> u64 {
    let mut h = Xxh64::new(0x5450_4348_4541_4431); // "TPCHEAD1"
    h.update(&head[..56]);
    h.update(&head[HEADER..]);
    h.digest()
}

fn write_entry(sh: &Shared, job: WriteJob) -> Result<DiskEntry, DiskError> {
    let r = write_entry_inner(sh, &job);
    match r {
        Ok(e) => {
            sh.stats.writes.fetch_add(1, Ordering::Relaxed);
            sh.stats.write_bytes.fetch_add(e.bytes, Ordering::Relaxed);
            {
                let mut idx = sh.index.lock().unwrap();
                idx.retain(|x| x.path != e.path);
                if job.supersede {
                    // Entries of the same conversation at earlier turns. The deepest one (the previous turn) stays:
                    // it serves a regenerated last reply, which branches off before the new entry's snapshots.
                    let older = |x: &DiskEntry| {
                        let d = x.depth as usize;
                        x.depth <= e.depth && d <= e.tokens.len() && e.tokens[..d] == x.tokens[..d]
                    };
                    let keep = idx
                        .iter()
                        .filter(|x| older(x))
                        .max_by_key(|x| x.depth)
                        .map(|x| x.path.clone());
                    let mut gone = Vec::new();
                    idx.retain(|x| {
                        let drop = older(x) && Some(&x.path) != keep.as_ref();
                        if drop {
                            gone.push(x.clone());
                        }
                        !drop
                    });
                    for g in gone {
                        let _ = fs::remove_file(&g.path);
                        let _ = fs::remove_file(g.payload_path());
                    }
                }
                idx.push(e.clone());
            }
            enforce_budget(sh);
            Ok(e)
        }
        Err(err) => {
            sh.stats.write_errors.fetch_add(1, Ordering::Relaxed);
            discard_external(&job);
            Err(err)
        }
    }
}

fn write_entry_inner(sh: &Shared, job: &WriteJob) -> Result<DiskEntry, DiskError> {
    if job.tokens.is_empty() {
        return Err(DiskError::Budget("empty token list".into()));
    }
    if job.tokens.len() as u64 > MAX_TOKENS || job.meta.len() as u64 > MAX_META {
        return Err(DiskError::Budget("entry too large".into()));
    }
    let name = entry_name(sh.identity, &job.tokens);
    let path = sh.root.join(&name);
    let pid = std::process::id();
    let tmp = sh.root.join(format!(".{name}.tmp.{pid}"));

    // External payload: hash + flush + rename first, so the entry never points at a missing payload.
    let mut secs: Vec<SectionDesc> = Vec::new();
    let mut ext_bytes = 0u64;
    if let Some(src) = &job.external {
        let f = File::open(src).map_err(|e| io_err("open native payload", e))?;
        let len = f
            .metadata()
            .map_err(|e| io_err("stat native payload", e))?
            .len();
        let h = file_payload_hash(&f, 0, len).map_err(|e| io_err("hash native payload", e))?;
        f.sync_all()
            .map_err(|e| io_err("flush native payload", e))?;
        fs::rename(src, path.with_extension(EXT_PAYLOAD))
            .map_err(|e| io_err("rename native payload", e))?;
        ext_bytes = len;
        secs.push(SectionDesc {
            tag: TAG_EXTERNAL,
            offset: OFF_EXTERNAL,
            len,
            hash: h,
        });
    }

    // Lay out the inline sections.
    let head_len = (HEADER + job.tokens.len() * 4 + job.meta.len()) as u64
        + 32 * (job.sections.len() + secs.len()) as u64;
    let mut off = align_up(head_len);
    for (t, d) in &job.sections {
        secs.push(SectionDesc {
            tag: *t,
            offset: off,
            len: d.len() as u64,
            hash: payload_hash(d),
        });
        off = align_up(off + d.len() as u64);
    }
    let end = secs
        .iter()
        .filter(|s| !s.is_external())
        .map(|s| s.offset + s.len)
        .max()
        .unwrap_or(head_len);
    let depth = if job.depth == 0 {
        job.tokens.len() as u64
    } else {
        job.depth.min(job.tokens.len() as u64)
    };
    let (head, hh) = build_head(sh.identity, &job.tokens, depth, &job.meta, &secs);
    debug_assert_eq!(head.len() as u64, head_len);

    let res = (|| -> io::Result<()> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(&tmp)?;
        f.write_all(&head)?;
        let mut pos = head.len() as u64;
        let zeros = [0u8; ALIGN as usize];
        let inline = secs.iter().filter(|s| !s.is_external());
        for (s, (_, d)) in inline.zip(job.sections.iter()) {
            let pad = (s.offset - pos) as usize;
            f.write_all(&zeros[..pad])?;
            f.write_all(d)?;
            pos = s.offset + s.len;
        }
        debug_assert_eq!(pos, end);
        let mut tr = [0u8; TRAILER];
        tr[..8].copy_from_slice(&END_MAGIC);
        tr[8..].copy_from_slice(&hh.to_le_bytes());
        f.write_all(&tr)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, &path)?;
        let _ = sync_dir(&sh.root);
        Ok(())
    })();
    if let Err(e) = res {
        let _ = fs::remove_file(&tmp);
        if job.external.is_some() {
            let _ = fs::remove_file(path.with_extension(EXT_PAYLOAD));
        }
        return Err(io_err("write entry", e));
    }
    Ok(DiskEntry {
        path,
        tokens: Arc::new(job.tokens.clone()),
        meta: job.meta.clone(),
        sections: secs,
        bytes: end + TRAILER as u64 + ext_bytes,
        depth,
    })
}

/// Parse and verify an entry's head region and file size. Does not read the payloads.
fn read_head(path: &Path, identity: u64) -> Result<DiskEntry, DiskError> {
    let mut f = File::open(path).map_err(|e| io_err("open entry", e))?;
    let flen = f.metadata().map_err(|e| io_err("stat entry", e))?.len();
    let mut h = [0u8; HEADER];
    f.read_exact(&mut h)
        .map_err(|_| DiskError::Corrupt("short header".into()))?;
    if h[..8] != MAGIC {
        return Err(DiskError::Corrupt("bad magic".into()));
    }
    let ver = u32::from_le_bytes(h[8..12].try_into().unwrap());
    if ver != VERSION {
        return Err(DiskError::Mismatch(format!("format version {ver}")));
    }
    let n_sec = u32::from_le_bytes(h[12..16].try_into().unwrap());
    let ident = u64::from_le_bytes(h[16..24].try_into().unwrap());
    let n_tok = u64::from_le_bytes(h[24..32].try_into().unwrap());
    let meta_len = u64::from_le_bytes(h[32..40].try_into().unwrap());
    let depth = u64::from_le_bytes(h[40..48].try_into().unwrap());
    let stored_hh = u64::from_le_bytes(h[56..64].try_into().unwrap());
    if ident != identity {
        return Err(DiskError::Mismatch(format!(
            "identity {ident:016x}, expected {identity:016x}"
        )));
    }
    if n_tok == 0
        || n_tok > MAX_TOKENS
        || meta_len > MAX_META
        || n_sec > MAX_SECTIONS
        || depth == 0
        || depth > n_tok
    {
        return Err(DiskError::Corrupt("header fields out of range".into()));
    }
    let rest = n_tok * 4 + meta_len + 32 * n_sec as u64;
    if HEADER as u64 + rest > flen {
        return Err(DiskError::Corrupt("truncated head".into()));
    }
    let mut head = vec![0u8; HEADER + rest as usize];
    head[..HEADER].copy_from_slice(&h);
    f.read_exact(&mut head[HEADER..])
        .map_err(|_| DiskError::Corrupt("truncated head".into()))?;
    let hh = head_hash(&head);
    if hh != stored_hh {
        return Err(DiskError::Corrupt("head hash mismatch".into()));
    }
    let mut p = HEADER;
    let tokens: Vec<i32> = head[p..p + n_tok as usize * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| i32::from_le_bytes(*c))
        .collect();
    p += n_tok as usize * 4;
    let meta = head[p..p + meta_len as usize].to_vec();
    p += meta_len as usize;
    let mut secs = Vec::with_capacity(n_sec as usize);
    let mut end = head.len() as u64;
    let mut ext = 0u64;
    for _ in 0..n_sec {
        let rd = |o: usize| u64::from_le_bytes(head[p + o..p + o + 8].try_into().unwrap());
        let s = SectionDesc {
            tag: rd(0),
            offset: rd(8),
            len: rd(16),
            hash: rd(24),
        };
        p += 32;
        if s.is_external() {
            ext += s.len;
        } else {
            if s.offset < head.len() as u64 || s.offset.checked_add(s.len).is_none() {
                return Err(DiskError::Corrupt("bad section table".into()));
            }
            end = end.max(s.offset + s.len);
        }
        secs.push(s);
    }
    if flen != end + TRAILER as u64 {
        return Err(DiskError::Corrupt(format!(
            "file is {flen} bytes, expected {}",
            end + TRAILER as u64
        )));
    }
    let mut tr = [0u8; TRAILER];
    f.read_exact_at(&mut tr, end)
        .map_err(|_| DiskError::Corrupt("missing trailer".into()))?;
    if tr[..8] != END_MAGIC || u64::from_le_bytes(tr[8..].try_into().unwrap()) != hh {
        return Err(DiskError::Corrupt("bad trailer".into()));
    }
    let e = DiskEntry {
        path: path.to_path_buf(),
        tokens: Arc::new(tokens),
        meta,
        sections: secs,
        bytes: flen + ext,
        depth,
    };
    if e.has_external() {
        let ep = e.payload_path();
        let plen = fs::metadata(&ep)
            .map_err(|_| DiskError::Corrupt("external payload missing".into()))?
            .len();
        if plen != ext {
            return Err(DiskError::Corrupt(
                "external payload has the wrong size".into(),
            ));
        }
    }
    Ok(e)
}

fn load_entry(sh: &Shared, e: &DiskEntry) -> Result<Loaded, DiskError> {
    // Re-check the head: the file may have been replaced or damaged since it was indexed.
    let fresh = read_head(&e.path, sh.identity)?;
    if fresh.tokens != e.tokens || fresh.sections != e.sections || fresh.meta != e.meta {
        return Err(DiskError::Mismatch("entry changed on disk".into()));
    }
    let f = File::open(&e.path).map_err(|e| io_err("open entry", e))?;
    let mut out = Loaded {
        tokens: Arc::clone(&e.tokens),
        meta: e.meta.clone(),
        sections: Vec::new(),
        external: None,
        bytes: e.bytes,
        ms: 0.0,
    };
    for s in &e.sections {
        if s.is_external() {
            let p = e.payload_path();
            let pf = File::open(&p)
                .map_err(|_| DiskError::Corrupt("external payload missing".into()))?;
            let h = file_payload_hash(&pf, 0, s.len).map_err(|e| io_err("read payload", e))?;
            if h != s.hash {
                return Err(DiskError::Corrupt("external payload hash mismatch".into()));
            }
            out.external = Some(p);
        } else {
            let mut buf = Vec::new();
            let h = read_and_hash(&f, s.offset, s.len, &mut buf, true)
                .map_err(|e| io_err("read section", e))?;
            if h != s.hash {
                return Err(DiskError::Corrupt(format!(
                    "section {:?} hash mismatch",
                    String::from_utf8_lossy(&s.tag.to_le_bytes())
                )));
            }
            out.sections.push((s.tag, buf));
        }
    }
    Ok(out)
}

/// Index the root folder: keep valid entries of this identity, delete broken ones and stale temporaries.
fn scan(sh: &Shared) -> Vec<DiskEntry> {
    let mut out: Vec<(SystemTime, DiskEntry)> = Vec::new();
    let Ok(rd) = fs::read_dir(&sh.root) else {
        return Vec::new();
    };
    let mut payloads: Vec<PathBuf> = Vec::new();
    for de in rd.flatten() {
        let p = de.path();
        let name = de.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') && name.contains(".tmp.") {
            // A write takes seconds: an hour-old temporary is abandoned even if its pid was reused since.
            let old = de
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_none_or(|age| age > Duration::from_secs(3600));
            let stale = match tmp_pid(&name) {
                Some(pid) => pid == std::process::id() || !pid_alive(pid) || old,
                None => old,
            };
            if stale {
                let _ = fs::remove_file(&p);
            }
            continue;
        }
        match p.extension().and_then(|x| x.to_str()) {
            Some(EXT) => match read_head(&p, sh.identity) {
                Ok(e) if name != entry_name(sh.identity, &e.tokens) => {
                    // A valid entry under a wrong name (copied or renamed by hand): lookups and superseding go
                    // by name, so it would linger as an unmanaged duplicate.
                    eprintln!("talus-pcache: dropping {name} (misnamed entry)");
                    sh.stats.corrupt.fetch_add(1, Ordering::Relaxed);
                    let _ = fs::remove_file(&p);
                    let _ = fs::remove_file(p.with_extension(EXT_PAYLOAD));
                }
                Ok(e) => {
                    let mt = de
                        .metadata()
                        .and_then(|m| m.modified())
                        .unwrap_or(UNIX_EPOCH);
                    out.push((mt, e));
                }
                Err(DiskError::Io(_)) => {}
                Err(err) => {
                    eprintln!(
                        "talus-pcache: dropping {} ({err})",
                        p.file_name().unwrap_or_default().to_string_lossy()
                    );
                    sh.stats.corrupt.fetch_add(1, Ordering::Relaxed);
                    let _ = fs::remove_file(&p);
                    let _ = fs::remove_file(p.with_extension(EXT_PAYLOAD));
                }
            },
            Some(EXT_PAYLOAD) => payloads.push(p),
            _ => {}
        }
    }
    // Orphan payloads (entry missing): a writer died between the two renames.
    for p in payloads {
        if !p.with_extension(EXT).exists() {
            let _ = fs::remove_file(&p);
        }
    }
    out.sort_by_key(|(t, _)| *t);
    out.into_iter().map(|(_, e)| e).collect()
}

/// LRU eviction over every identity folder under the base dir.
fn enforce_budget(sh: &Shared) {
    let lock = File::create(sh.base.join(".lock")).ok();
    if let Some(l) = &lock {
        let _ = l.lock();
    }
    let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
    let mut total = 0u64;
    if let Ok(rd) = fs::read_dir(&sh.base) {
        for d in rd.flatten() {
            let dp = d.path();
            if !dp.is_dir() {
                continue;
            }
            let Ok(inner) = fs::read_dir(&dp) else {
                continue;
            };
            for de in inner.flatten() {
                let p = de.path();
                if p.extension().and_then(|x| x.to_str()) != Some(EXT) {
                    continue;
                }
                let Ok(md) = de.metadata() else { continue };
                let ext_len = fs::metadata(p.with_extension(EXT_PAYLOAD))
                    .map(|m| m.len())
                    .unwrap_or(0);
                let sz = md.len() + ext_len;
                total += sz;
                files.push((md.modified().unwrap_or(UNIX_EPOCH), sz, p));
            }
        }
    }
    if total > sh.budget {
        files.sort_by_key(|(t, _, _)| *t);
        let mut gone: Vec<PathBuf> = Vec::new();
        for (_, sz, p) in files {
            if total <= sh.budget {
                break;
            }
            let _ = fs::remove_file(&p);
            let _ = fs::remove_file(p.with_extension(EXT_PAYLOAD));
            total = total.saturating_sub(sz);
            sh.stats.evicted.fetch_add(1, Ordering::Relaxed);
            gone.push(p);
        }
        if !gone.is_empty() {
            sh.index.lock().unwrap().retain(|e| !gone.contains(&e.path));
        }
    }
    if let Some(l) = lock {
        let _ = l.unlock();
    }
}

// ============================================================================
// Small helpers shared by the engine integrations
// ============================================================================

/// Length of the longest common prefix.
pub fn lcp(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Little-endian `meta` writer.
#[derive(Default)]
pub struct MetaWriter(pub Vec<u8>);

impl MetaWriter {
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

/// Little-endian `meta` reader (returns `None` past the end).
pub struct MetaReader<'a>(pub &'a [u8]);

impl MetaReader<'_> {
    pub fn u64(&mut self) -> Option<u64> {
        if self.0.len() < 8 {
            return None;
        }
        let v = u64::from_le_bytes(self.0[..8].try_into().unwrap());
        self.0 = &self.0[8..];
        Some(v)
    }
}

/// A temporary path in the cache folder for an engine-native payload (see [`WriteJob::external`]); stale ones
/// are cleaned up by the next [`DiskCache::open`].
pub fn native_tmp_path(cache: &DiskCache, stem: &str) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    cache
        .root()
        .join(format!(".{stem}-{n}.native.tmp.{}", std::process::id()))
}
