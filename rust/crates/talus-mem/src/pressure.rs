//! Graceful memory-pressure handling at the `MemAvailable` floor, shared by every engine (#40).
//!
//! Before: the GLM engine ran a watchdog that `_exit(3)`-ed the process as soon as `MemAvailable` stayed under the
//! 8 GiB floor for two samples, and the other engines had no run-time check at all. Now:
//!
//! 1. One sampler thread per process ([`install`]) reads the host memory (the same reading as the load-time
//!    [`crate::Budget`]: `MemAvailable`, the cgroup limits and the operator cap `TALUS_MEM_LIMIT_MIB`) every
//!    `TALUS_MEM_PRESSURE_MS` (100 ms). A sample below `floor + margin` latches a flag ([`pending`]: one relaxed
//!    atomic load, the only cost on the engines' hot paths). After a relief that ends between the floor and the
//!    threshold, the trigger drops to 256 MiB under that reading (hysteresis: no shed / rebuild cycle per request
//!    while the host sits just above the floor); it re-arms at the threshold once a reading is above it.
//! 2. At their safe points (request start, prompt chunks, decode rounds) the engines call [`Pressure::relieve`]
//!    when the flag is set. It sheds rebuildable memory in the fixed [`SHED_ORDER`] (decode graphs, transient
//!    arenas, prompt-cache RAM tier, prompt snapshots last) through the engine's [`Shed`] implementation until the
//!    reading is back above `floor + margin`. Shedding only costs time: everything shed is rebuilt on demand, so a
//!    request that completes has the same output.
//! 3. When the reading is still under the floor after every class was shed, the engine fails just the current
//!    request with [`InsufficientMemory`] (`ERR insufficient_memory: ...` on the stdio protocol, HTTP 503
//!    `insufficient_memory` from the servers) and keeps serving.
//! 4. Last resort: a reading under `TALUS_MEM_HARD_EXIT_MIB` (default floor - 2 GiB, 0 = never) for two samples in
//!    a row still exits the process, as the old watchdog did at the floor itself (the engine's memory then goes back
//!    to the host). This only triggers when shedding and failing requests could not hold the line.
//!
//! The floor stays 8 GiB (GLM: `--minavail`). Each shed class and each rejection is logged once with its numbers;
//! repeats are counted ([`Pressure::summary`]). `TALUS_MEM_PRESSURE_LOG=all` logs every action.
//!
//! Test knobs (no real memory is taken, so the host never goes below its floor while testing):
//! `TALUS_MEM_PRESSURE_SIM_MIB=N` subtracts N MiB from every reading (an artificial reservation), and
//! `TALUS_MEM_PRESSURE_SIM_FILE=path` does the same with the number in that file, re-read on every sample, so a test
//! driver can raise and drop the reservation while a request runs.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::probe::{HostMemProbe, LinuxProcMemProbe};
use crate::{DEFAULT_FLOOR_BYTES, GIB, MIB};

/// Error code of a request that failed for memory (`ERR insufficient_memory: ...`, HTTP 503 `code`).
pub const ERR_CODE: &str = "insufficient_memory";

/// Rebuildable memory classes, shed in [`SHED_ORDER`] (cheapest to rebuild first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShedClass {
    /// Captured decode / verify graphs (re-captured on next use: milliseconds).
    DecodeGraphs,
    /// Prompt-cache recurrent snapshots held in device memory (the next turn re-reads its prompt).
    PromptSnapshots,
    /// Prompt-cache RAM tier: KV stashes and parked conversations (moved to / re-read from the disk tier when it is
    /// on, else re-read cold).
    PromptCacheRam,
    /// Transient arenas and allocator caches that the next prompt chunk grows back (prefill scratch pools).
    TransientArenas,
}

/// The fixed shedding order (operator, 2026-10-30): what costs least to rebuild and never changes outputs first
/// (graphs, arenas), then the prompt cache's RAM tier, the prompt snapshots last.
pub const SHED_ORDER: [ShedClass; 4] = [
    ShedClass::DecodeGraphs,
    ShedClass::TransientArenas,
    ShedClass::PromptCacheRam,
    ShedClass::PromptSnapshots,
];

/// Return freed host heap to the system (glibc `malloc_trim(0)`): exact, part of [`ShedClass::TransientArenas`].
/// Returns the MemAvailable gain in bytes it observed (0 when none or not glibc).
pub fn trim_host_heap() -> u64 {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        unsafe extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        let a = crate::probe::LinuxProcMemProbe::new()
            .mem_available_bytes()
            .unwrap_or(0) as u64;
        // SAFETY: glibc malloc_trim, any thread.
        let released = unsafe { malloc_trim(0) } != 0;
        let b = crate::probe::LinuxProcMemProbe::new()
            .mem_available_bytes()
            .unwrap_or(0) as u64;
        if released {
            return b.saturating_sub(a).max(1);
        }
    }
    0
}

/// Classes an engine can shed in the middle of a request (prompt chunk / decode round): the prompt-cache classes are
/// shed between requests only (the running request may be filling a snapshot or stash).
pub const MID_REQUEST: [ShedClass; 2] = [ShedClass::DecodeGraphs, ShedClass::TransientArenas];

impl ShedClass {
    pub fn name(self) -> &'static str {
        match self {
            Self::DecodeGraphs => "decode graphs",
            Self::PromptSnapshots => "prompt snapshots",
            Self::PromptCacheRam => "prompt-cache RAM tier",
            Self::TransientArenas => "transient arenas",
        }
    }
}

/// An owner of rebuildable memory (an engine, a serve backend).
pub trait Shed {
    /// Release everything this owner holds of `class`. Returns the bytes released by its own accounting (0 = none
    /// held or class not supported). Must not change the output of any later request: only memory that is rebuilt
    /// on demand may be released.
    fn shed(&mut self, class: ShedClass) -> u64;

    /// Whether dropping `class` can change a later request's output (a cache whose warm path is not bitwise the cold
    /// one, e.g. GLM's prompt snapshots: the request then completes as a cold run). Shed by default, last in the
    /// order (operator option A, 2026-10-30); `TALUS_MEM_SHED_INEXACT=0` fails the request instead.
    fn inexact(&self, _class: ShedClass) -> bool {
        false
    }
}

/// No engine caches to shed (engines without rebuildable device memory still get the admission check, the host-heap
/// trim and the error path).
pub struct NoShed;

impl Shed for NoShed {
    fn shed(&mut self, class: ShedClass) -> u64 {
        match class {
            ShedClass::TransientArenas => trim_host_heap(),
            _ => 0,
        }
    }
}

/// Configuration (env + the engine's floor).
#[derive(Clone, Debug)]
pub struct PressureConfig {
    /// `MemAvailable` floor in bytes (8 GiB).
    pub floor: u64,
    /// Shedding starts below `floor + margin` (`TALUS_MEM_SHED_MARGIN_MIB`, 512).
    pub margin: u64,
    /// Process exit below this for two samples (`TALUS_MEM_HARD_EXIT_MIB`, default floor - 2 GiB; 0 = never).
    pub hard_exit: u64,
    /// Operator cap in MiB (`TALUS_MEM_LIMIT_MIB`, 0 = none).
    pub limit_mib: u64,
    /// Artificial reservation in MiB (`TALUS_MEM_PRESSURE_SIM_MIB`).
    pub sim_mib: u64,
    /// File holding an artificial reservation in MiB, re-read every sample (`TALUS_MEM_PRESSURE_SIM_FILE`).
    pub sim_file: Option<PathBuf>,
    /// Sampling period (`TALUS_MEM_PRESSURE_MS`, 100).
    pub period_ms: u64,
    /// Log every action instead of once per kind (`TALUS_MEM_PRESSURE_LOG=all`).
    pub log_all: bool,
    /// Also shed classes whose loss can change outputs ([`Shed::inexact`]; default on, `TALUS_MEM_SHED_INEXACT=0` off).
    pub shed_inexact: bool,
}

fn env_u64(k: &str) -> Option<u64> {
    std::env::var(k).ok().and_then(|v| v.trim().parse().ok())
}

impl PressureConfig {
    /// Defaults with `floor` (bytes), then the environment.
    pub fn from_env(floor: u64) -> Self {
        let hard_default = floor.saturating_sub(2 * GIB as u64);
        Self {
            floor,
            margin: env_u64("TALUS_MEM_SHED_MARGIN_MIB").unwrap_or(512) * MIB as u64,
            hard_exit: env_u64("TALUS_MEM_HARD_EXIT_MIB")
                .map(|m| m * MIB as u64)
                .unwrap_or(hard_default),
            limit_mib: env_u64("TALUS_MEM_LIMIT_MIB").unwrap_or(0),
            sim_mib: env_u64("TALUS_MEM_PRESSURE_SIM_MIB").unwrap_or(0),
            sim_file: std::env::var_os("TALUS_MEM_PRESSURE_SIM_FILE").map(PathBuf::from),
            period_ms: env_u64("TALUS_MEM_PRESSURE_MS").unwrap_or(100).max(10),
            log_all: std::env::var("TALUS_MEM_PRESSURE_LOG").as_deref() == Ok("all"),
            shed_inexact: std::env::var("TALUS_MEM_SHED_INEXACT").as_deref() != Ok("0"),
        }
    }

    /// The 8 GiB default floor.
    pub fn strix_default() -> Self {
        Self::from_env(DEFAULT_FLOOR_BYTES as u64)
    }

    /// Shedding threshold.
    pub fn threshold(&self) -> u64 {
        self.floor + self.margin
    }
}

/// Result of one [`Pressure::relieve`].
#[derive(Clone, Debug, Default)]
pub struct Relief {
    /// Reading before shedding.
    pub before: u64,
    /// Reading after shedding.
    pub after: u64,
    /// Classes that released something, with the owner's byte counts.
    pub shed: Vec<(ShedClass, u64)>,
}

/// A request refused or aborted for memory (the floor could not be held after shedding).
#[derive(Clone, Debug)]
pub struct InsufficientMemory {
    pub available: u64,
    pub floor: u64,
    pub shed: Vec<(ShedClass, u64)>,
    pub during: &'static str,
}

fn mib(b: u64) -> u64 {
    b / MIB as u64
}

impl fmt::Display for InsufficientMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{ERR_CODE}: host MemAvailable {} MiB is below the {} MiB floor {} after shedding rebuildable caches",
            mib(self.available),
            mib(self.floor),
            self.during
        )?;
        if !self.shed.is_empty() {
            let s: Vec<String> = self
                .shed
                .iter()
                .map(|(c, b)| format!("{} {} MiB", c.name(), mib(*b)))
                .collect();
            write!(f, " ({})", s.join(", "))?;
        }
        write!(
            f,
            "; this request was not completed, the server keeps running - retry when memory is free"
        )
    }
}

impl std::error::Error for InsufficientMemory {}

impl From<InsufficientMemory> for talus_core_api::CoreError {
    fn from(e: InsufficientMemory) -> Self {
        talus_core_api::CoreError::StateError {
            reason: e.to_string(),
        }
    }
}

/// The stdio protocol line for an error: an [`InsufficientMemory`] (anywhere in `msg`) becomes
/// `ERR insufficient_memory: ...` so the servers can map it; other errors pass through as `ERR msg`.
pub fn err_line(msg: &str) -> String {
    match msg.find(ERR_CODE) {
        Some(i) => format!("ERR {}", &msg[i..]),
        None => format!("ERR {msg}"),
    }
}

/// Whether an engine error message is a memory refusal.
pub fn is_insufficient_memory(msg: &str) -> bool {
    msg.contains(ERR_CODE)
}

#[derive(Default)]
struct Counters {
    shed: HashMap<ShedClass, (u64, u64)>,
    rejected: u64,
    logged: std::collections::HashSet<&'static str>,
}

/// Process-wide pressure state (see the module docs).
pub struct Pressure {
    cfg: PressureConfig,
    probe: LinuxProcMemProbe,
    tag: &'static str,
    flag: AtomicBool,
    /// readings below this latch the flag: `floor + margin`, lowered after a relief that could not get back above it
    /// (hysteresis: a steady state between the floor and the threshold does not shed again and again), re-armed at
    /// the threshold once a reading is above it
    armed: AtomicU64,
    last: AtomicU64,
    min: AtomicU64,
    counters: Mutex<Counters>,
}

static GLOBAL: OnceLock<Arc<Pressure>> = OnceLock::new();

/// Start the process's pressure sampler (once; later calls return the first instance). `tag` prefixes the log.
pub fn install(cfg: PressureConfig, tag: &'static str) -> Arc<Pressure> {
    GLOBAL
        .get_or_init(|| {
            let p = Arc::new(Pressure::new(cfg, tag));
            let q = p.clone();
            std::thread::Builder::new()
                .name("talus-mem-pressure".into())
                .spawn(move || q.sampler())
                .ok();
            eprintln!(
                "{tag}: memory pressure guard: floor {} MiB, shed below {} MiB, exit below {} MiB{}",
                mib(p.cfg.floor),
                mib(p.cfg.threshold()),
                mib(p.cfg.hard_exit),
                if p.cfg.sim_mib > 0 || p.cfg.sim_file.is_some() {
                    format!(
                        " (simulated reservation {} MiB{})",
                        p.cfg.sim_mib,
                        p.cfg
                            .sim_file
                            .as_ref()
                            .map(|f| format!(" + file {}", f.display()))
                            .unwrap_or_default()
                    )
                } else {
                    String::new()
                }
            );
            p
        })
        .clone()
}

/// The installed instance, if any.
pub fn global() -> Option<&'static Arc<Pressure>> {
    GLOBAL.get()
}

/// Whether a sample went below the shedding threshold since the last [`Pressure::relieve`] (hot-path check).
#[inline]
pub fn pending() -> bool {
    match GLOBAL.get() {
        Some(p) => p.flag.load(Ordering::Relaxed),
        None => false,
    }
}

impl Pressure {
    /// An instance without a sampler thread (tests; engines use [`install`]).
    pub fn new(cfg: PressureConfig, tag: &'static str) -> Self {
        let probe = if cfg.limit_mib > 0 {
            LinuxProcMemProbe::new().with_memory_limit_mib(cfg.limit_mib as usize)
        } else {
            LinuxProcMemProbe::new()
        };
        let armed = AtomicU64::new(cfg.threshold());
        Self {
            cfg,
            probe,
            tag,
            armed,
            flag: AtomicBool::new(false),
            last: AtomicU64::new(u64::MAX),
            min: AtomicU64::new(u64::MAX),
            counters: Mutex::new(Counters::default()),
        }
    }

    /// Instance with an explicit probe (tests).
    pub fn with_probe(cfg: PressureConfig, probe: LinuxProcMemProbe) -> Self {
        let mut p = Self::new(cfg, "test");
        p.probe = probe;
        p
    }

    pub fn config(&self) -> &PressureConfig {
        &self.cfg
    }

    fn sim_bytes(&self) -> u64 {
        let mut m = self.cfg.sim_mib;
        if let Some(f) = &self.cfg.sim_file {
            if let Ok(s) = std::fs::read_to_string(f) {
                m += s.trim().parse::<u64>().unwrap_or(0);
            }
        }
        m * MIB as u64
    }

    /// Fresh effective reading in bytes (host reading minus the simulated reservation); `u64::MAX` when the host
    /// cannot be read (no pressure is assumed then).
    pub fn available(&self) -> u64 {
        let a = match self.probe.mem_available_bytes() {
            Ok(a) => a as u64,
            Err(_) => return u64::MAX,
        };
        let a = a.saturating_sub(self.sim_bytes());
        self.last.store(a, Ordering::Relaxed);
        self.min.fetch_min(a, Ordering::Relaxed);
        a
    }

    /// Lowest reading seen so far (bytes).
    pub fn min_seen(&self) -> u64 {
        self.min.load(Ordering::Relaxed)
    }

    fn sampler(&self) {
        let mut low = 0u32;
        loop {
            std::thread::sleep(Duration::from_millis(self.cfg.period_ms));
            let a = self.available();
            if a == u64::MAX {
                continue;
            }
            if a >= self.cfg.threshold() {
                self.armed.store(self.cfg.threshold(), Ordering::Relaxed);
            } else if self.below_armed(a) {
                self.flag.store(true, Ordering::Relaxed);
            }
            if self.cfg.hard_exit > 0 && a < self.cfg.hard_exit {
                low += 1;
                if low >= 2 {
                    eprintln!(
                        "{}: MEMORY PRESSURE: MemAvailable {} MiB < last-resort limit {} MiB (floor {} MiB) after \
                         shedding and failing requests: exiting",
                        self.tag,
                        mib(a),
                        mib(self.cfg.hard_exit),
                        mib(self.cfg.floor)
                    );
                    unsafe extern "C" {
                        fn _exit(code: i32) -> !;
                    }
                    // SAFETY: immediate process exit; the driver releases the GPU memory.
                    unsafe { _exit(3) };
                }
            } else {
                low = 0;
            }
        }
    }

    /// Hysteresis step after a partial relief.
    const REARM_STEP: u64 = 256 * MIB as u64;

    fn below_armed(&self, a: u64) -> bool {
        a < self.cfg.floor || a < self.armed.load(Ordering::Relaxed)
    }

    fn log_first(&self, key: &'static str, msg: impl FnOnce() -> String) {
        let first = {
            let mut c = self.counters.lock().unwrap();
            c.logged.insert(key)
        };
        if first || self.cfg.log_all {
            eprintln!("{}: {}", self.tag, msg());
        }
    }

    /// Shed `classes` (kept in [`SHED_ORDER`] order) through `owner` until the reading is back above
    /// `floor + margin`. Returns the relief when the reading ends at or above the floor, else the refusal for the
    /// current request. Clears the pending flag.
    pub fn relieve(
        &self,
        owner: &mut dyn Shed,
        classes: &[ShedClass],
        during: &'static str,
    ) -> Result<Relief, InsufficientMemory> {
        self.flag.store(false, Ordering::Relaxed);
        let before = self.available();
        let mut r = Relief {
            before,
            after: before,
            shed: vec![],
        };
        if before >= self.cfg.threshold() {
            self.armed.store(self.cfg.threshold(), Ordering::Relaxed);
            return Ok(r);
        }
        if !self.below_armed(before) {
            return Ok(r); // a partial relief already brought it here: no new shedding until it drops further
        }
        for class in SHED_ORDER.iter().copied().filter(|c| classes.contains(c)) {
            if owner.inexact(class) && !self.cfg.shed_inexact {
                continue;
            }
            let freed = owner.shed(class);
            if freed == 0 {
                continue;
            }
            let after = self.available();
            r.shed.push((class, freed));
            {
                let mut c = self.counters.lock().unwrap();
                let e = c.shed.entry(class).or_default();
                e.0 += 1;
                e.1 += freed;
            }
            let key = match class {
                ShedClass::DecodeGraphs => "shed-graphs",
                ShedClass::PromptSnapshots => "shed-snaps",
                ShedClass::PromptCacheRam => "shed-pcache",
                ShedClass::TransientArenas => "shed-arenas",
            };
            let inexact = owner.inexact(class);
            self.log_first(key, || {
                format!(
                    "memory pressure {during}: MemAvailable {} MiB < {} MiB: shed {}{} ({} MiB; MemAvailable now {} MiB){}",
                    mib(r.after),
                    mib(self.cfg.threshold()),
                    class.name(),
                    if inexact { " (the next turn reads its prompt cold: its text can differ from a warm resume)" } else { "" },
                    mib(freed),
                    mib(after),
                    if self.cfg.log_all { "" } else { " [logged once; later sheds are counted]" }
                )
            });
            r.after = after;
            if after >= self.cfg.threshold() {
                self.armed.store(self.cfg.threshold(), Ordering::Relaxed);
                return Ok(r);
            }
        }
        r.after = self.available();
        // between the floor and the threshold: shed again only once the reading drops REARM_STEP below this
        self.armed.store(
            r.after.saturating_sub(Self::REARM_STEP).max(self.cfg.floor),
            Ordering::Relaxed,
        );
        if r.after >= self.cfg.floor {
            return Ok(r);
        }
        let e = InsufficientMemory {
            available: r.after,
            floor: self.cfg.floor,
            shed: r.shed,
            during,
        };
        let n = {
            let mut c = self.counters.lock().unwrap();
            c.rejected += 1;
            c.rejected
        };
        eprintln!("{}: request failed ({n} so far): {e}", self.tag);
        Err(e)
    }

    /// Admission at request start: [`Self::relieve`] over every class, only when the flag is set or the fresh
    /// reading is under the threshold.
    pub fn admit(&self, owner: &mut dyn Shed) -> Result<Relief, InsufficientMemory> {
        self.relieve(owner, &SHED_ORDER, "at request start")
    }

    /// One-line counters: `sheds graphs 2/35 MiB, ...; rejected 1; min 8123 MiB`.
    pub fn summary(&self) -> String {
        let c = self.counters.lock().unwrap();
        let mut parts: Vec<String> = SHED_ORDER
            .iter()
            .filter_map(|k| {
                c.shed
                    .get(k)
                    .map(|(n, b)| format!("{} x{n} {} MiB", k.name(), mib(*b)))
            })
            .collect();
        parts.push(format!("rejected {}", c.rejected));
        let m = self.min_seen();
        if m != u64::MAX {
            parts.push(format!("min {} MiB", mib(m)));
        }
        parts.join(", ")
    }
}

/// Admission check for engines: a no-op returning `Ok` when no guard is installed.
pub fn admit(owner: &mut dyn Shed) -> Result<(), InsufficientMemory> {
    match GLOBAL.get() {
        Some(p) => p.admit(owner).map(|_| ()),
        None => Ok(()),
    }
}

/// Safe-point check inside a request: cheap unless the flag is set, then sheds the mid-request classes.
#[inline]
pub fn check_mid_request(owner: &mut dyn Shed) -> Result<(), InsufficientMemory> {
    if !pending() {
        return Ok(());
    }
    match GLOBAL.get() {
        Some(p) => p
            .relieve(owner, &MID_REQUEST, "during the request")
            .map(|_| ()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct Owner {
        held: HashMap<ShedClass, u64>,
        order: Vec<ShedClass>,
        meminfo: PathBuf,
        avail_kb: u64,
    }

    impl Owner {
        fn write(&self) {
            fs::write(
                &self.meminfo,
                format!(
                    "MemTotal: 134217728 kB\nMemFree: 1024 kB\nMemAvailable: {} kB\n",
                    self.avail_kb
                ),
            )
            .unwrap();
        }
    }

    impl Shed for Owner {
        fn shed(&mut self, class: ShedClass) -> u64 {
            self.order.push(class);
            let b = self.held.remove(&class).unwrap_or(0);
            self.avail_kb += b / 1024;
            self.write();
            b
        }
    }

    fn setup(name: &str, avail_mib: u64, held: &[(ShedClass, u64)]) -> (Pressure, Owner) {
        let dir =
            std::env::temp_dir().join(format!("talus-mem-pressure-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let paths = crate::MemoryProbePaths {
            meminfo: dir.join("meminfo"),
            self_cgroup: dir.join("no-cgroup"),
            cgroup_root: dir.join("no-root"),
        };
        let o = Owner {
            held: held.iter().map(|&(c, b)| (c, b * MIB as u64)).collect(),
            order: vec![],
            meminfo: paths.meminfo.clone(),
            avail_kb: avail_mib * 1024,
        };
        o.write();
        let cfg = PressureConfig {
            floor: 8 * GIB as u64,
            margin: 512 * MIB as u64,
            hard_exit: 6 * GIB as u64,
            limit_mib: 0,
            sim_mib: 0,
            sim_file: None,
            period_ms: 100,
            log_all: false,
            shed_inexact: false,
        };
        (
            Pressure::with_probe(cfg, LinuxProcMemProbe::with_paths(paths)),
            o,
        )
    }

    #[test]
    fn no_shedding_above_threshold() {
        let (p, mut o) = setup("above", 9000, &[(ShedClass::DecodeGraphs, 100)]);
        let r = p.admit(&mut o).unwrap();
        assert!(r.shed.is_empty());
        assert!(o.order.is_empty());
    }

    #[test]
    fn sheds_in_fixed_order_until_above_threshold() {
        // 8100 MiB: under 8704; graphs (+300) -> 8400, arenas (+400) -> 8800 >= threshold: stop, snapshots kept
        let (p, mut o) = setup(
            "order",
            8100,
            &[
                (ShedClass::PromptSnapshots, 500),
                (ShedClass::TransientArenas, 400),
                (ShedClass::DecodeGraphs, 300),
            ],
        );
        let r = p.admit(&mut o).unwrap();
        assert_eq!(
            o.order,
            vec![ShedClass::DecodeGraphs, ShedClass::TransientArenas]
        );
        assert_eq!(r.shed.len(), 2);
        assert!(r.after >= p.config().threshold());
        assert!(o.held.contains_key(&ShedClass::PromptSnapshots));
    }

    #[test]
    fn between_floor_and_threshold_after_all_classes_is_ok() {
        let (p, mut o) = setup("between", 7900, &[(ShedClass::TransientArenas, 300)]);
        let r = p.admit(&mut o).unwrap();
        assert_eq!(r.shed.len(), 1);
        assert!(r.after >= p.config().floor && r.after < p.config().threshold());
    }

    #[test]
    fn still_short_fails_the_request_only() {
        let (p, mut o) = setup("short", 7000, &[(ShedClass::DecodeGraphs, 100)]);
        let e = p.admit(&mut o).unwrap_err();
        assert_eq!(e.floor, 8 * GIB as u64);
        let msg = e.to_string();
        assert!(msg.starts_with("insufficient_memory: host MemAvailable 7100 MiB"));
        assert!(msg.contains("decode graphs 100 MiB"));
        assert_eq!(
            err_line(&format!("state error: {msg}")),
            format!("ERR {msg}")
        );
        assert!(p.summary().contains("rejected 1"));
        // memory came back: the next request is admitted
        o.avail_kb = 9000 * 1024;
        o.write();
        assert!(p.admit(&mut o).is_ok());
    }

    #[test]
    fn mid_request_classes_skip_prompt_cache() {
        let (p, mut o) = setup(
            "mid",
            7500,
            &[
                (ShedClass::PromptSnapshots, 4000),
                (ShedClass::TransientArenas, 100),
            ],
        );
        let e = p
            .relieve(&mut o, &MID_REQUEST, "during the request")
            .unwrap_err();
        assert_eq!(
            o.order,
            vec![ShedClass::DecodeGraphs, ShedClass::TransientArenas]
        );
        assert!(o.held.contains_key(&ShedClass::PromptSnapshots));
        assert!(e.to_string().contains("during the request"));
    }

    #[test]
    fn simulated_reservation_lowers_the_reading() {
        let (mut p, mut o) = setup("sim", 9000, &[(ShedClass::DecodeGraphs, 10)]);
        let f = o.meminfo.with_file_name("sim");
        fs::write(&f, "1500\n").unwrap();
        p.cfg.sim_file = Some(f.clone());
        assert_eq!(p.available(), 7500 * MIB as u64);
        assert!(p.admit(&mut o).is_err());
        fs::write(&f, "0").unwrap();
        assert!(p.admit(&mut o).is_ok());
    }

    #[test]
    fn partial_relief_does_not_shed_again_until_it_drops_further() {
        let (p, mut o) = setup("hyst", 8000, &[(ShedClass::DecodeGraphs, 300)]);
        let r = p.admit(&mut o).unwrap(); // 8300: between floor and threshold
        assert_eq!(r.shed.len(), 1);
        o.held.insert(ShedClass::DecodeGraphs, 300 * MIB as u64); // graphs re-captured, reading unchanged
        o.order.clear();
        let r = p.admit(&mut o).unwrap();
        assert!(r.shed.is_empty() && o.order.is_empty());
        // 300 MiB lower (> 256 MiB step): sheds again
        o.avail_kb -= 300 * 1024;
        o.write();
        let r = p.admit(&mut o).unwrap();
        assert_eq!(r.shed.len(), 1);
        // back above the threshold re-arms
        o.avail_kb = 9000 * 1024;
        o.write();
        assert!(p.admit(&mut o).unwrap().shed.is_empty());
        assert_eq!(p.armed.load(Ordering::Relaxed), p.config().threshold());
    }

    struct Inexact(Owner);
    impl Shed for Inexact {
        fn shed(&mut self, class: ShedClass) -> u64 {
            self.0.shed(class)
        }
        fn inexact(&self, class: ShedClass) -> bool {
            class == ShedClass::PromptSnapshots
        }
    }

    #[test]
    fn inexact_classes_only_on_opt_in() {
        let (mut p, o) = setup(
            "inexact",
            8300,
            &[
                (ShedClass::PromptSnapshots, 1000),
                (ShedClass::DecodeGraphs, 10),
            ],
        );
        let mut o = Inexact(o);
        let r = p.admit(&mut o).unwrap(); // graphs only: 8310, between floor and threshold
        assert!(!o.0.order.contains(&ShedClass::PromptSnapshots));
        assert_eq!(r.shed.len(), 1);
        p.cfg.shed_inexact = true;
        o.0.avail_kb -= 600 * 1024; // drop under the hysteresis step
        o.0.write();
        let r = p.admit(&mut o).unwrap();
        assert!(r.shed.iter().any(|(c, _)| *c == ShedClass::PromptSnapshots));
    }

    #[test]
    fn err_line_passes_other_errors_through() {
        assert_eq!(err_line("bad token"), "ERR bad token");
        assert!(!is_insufficient_memory("bad token"));
        assert!(is_insufficient_memory("x insufficient_memory: y"));
    }
}
