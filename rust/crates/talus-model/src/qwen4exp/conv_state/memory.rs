//! Host RAM telemetry for the conversation cache's admission gates: a Rust port of
//! `src/core/conversation_memory.cpp` (`sample_host_memory`: MemAvailable, the cgroup v2 / v1 limits above the
//! engine with their clean inactive file cache credited, and the operator's `--memory-limit-mib`).

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySource {
    Meminfo,
    Cgroup,
    Flag,
}

impl MemorySource {
    pub fn name(self) -> &'static str {
        match self {
            MemorySource::Flag => "flag",
            MemorySource::Cgroup => "cgroup",
            MemorySource::Meminfo => "meminfo",
        }
    }
}

/// `HostMemoryReading`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemoryReading {
    pub available: u64,
    pub source: MemorySource,
    pub limit: u64,
    pub current: u64,
    pub credit: u64,
    pub mem_available: u64,
}

/// `MemoryProbePaths`.
#[derive(Clone, Debug)]
pub struct MemoryProbePaths {
    pub meminfo: PathBuf,
    pub self_cgroup: PathBuf,
    pub cgroup_root: PathBuf,
}

impl Default for MemoryProbePaths {
    fn default() -> Self {
        Self {
            meminfo: "/proc/meminfo".into(),
            self_cgroup: "/proc/self/cgroup".into(),
            cgroup_root: "/sys/fs/cgroup".into(),
        }
    }
}

/// `meminfo_kb_bytes`: the one `<wanted> <n> kB` line, in bytes (a duplicate or malformed line: none).
pub fn meminfo_kb_bytes(text: &str, wanted: &str) -> Option<u64> {
    let mut result = None;
    for line in text.lines() {
        let mut f = line.split_whitespace();
        if f.next() != Some(wanted) {
            continue;
        }
        let (Some(v), Some(unit), None) = (f.next(), f.next(), f.next()) else {
            return None;
        };
        if result.is_some() || unit != "kB" {
            return None;
        }
        let kb: u64 = if v.bytes().all(|b| b.is_ascii_digit()) && !v.is_empty() {
            v.parse().ok()?
        } else {
            return None;
        };
        result = Some(kb.checked_mul(1024)?);
    }
    result
}

enum Value {
    Absent,
    Bad,
    Max,
    Number(u64),
}

fn read_value(path: &Path) -> Value {
    match std::fs::metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Value::Absent,
        Err(_) => return Value::Bad,
        Ok(_) => {}
    }
    let Ok(b) = std::fs::read(path) else {
        return Value::Bad;
    };
    if b.len() >= 64 {
        return Value::Bad;
    }
    let s = String::from_utf8_lossy(&b);
    let t = s.trim_end_matches(['\n', '\r', ' ', '\t']);
    if t.is_empty() {
        return Value::Bad;
    }
    if t == "max" {
        return Value::Max;
    }
    if !t.bytes().all(|c| c.is_ascii_digit()) {
        return Value::Bad;
    }
    t.parse().map(Value::Number).unwrap_or(Value::Bad)
}

fn reclaimable_cache(dir: &Path, current: u64, v1: bool) -> u64 {
    let Ok(text) = std::fs::read_to_string(dir.join("memory.stat")) else {
        return 0;
    };
    let (ik, dk, wk) = if v1 {
        ("total_inactive_file", "total_dirty", "total_writeback")
    } else {
        ("inactive_file", "file_dirty", "file_writeback")
    };
    let (mut inactive, mut dirty, mut writeback) = (None::<u64>, 0u64, 0u64);
    for line in text.lines() {
        let mut f = line.split_whitespace();
        let (Some(key), Some(value), None) = (f.next(), f.next(), f.next()) else {
            continue;
        };
        let n = if value.bytes().all(|c| c.is_ascii_digit()) {
            value.parse::<u64>().ok()
        } else {
            None
        };
        if key == ik {
            match (n, inactive) {
                (Some(v), None) => inactive = Some(v),
                _ => return 0,
            }
        } else if key == dk || key == wk {
            let Some(v) = n else {
                return 0;
            };
            if key == dk {
                dirty = v;
            } else {
                writeback = v;
            }
        }
    }
    let Some(inactive) = inactive else {
        return 0;
    };
    let mut credit = inactive.min(current);
    credit = credit.saturating_sub(dirty);
    credit.saturating_sub(writeback)
}

#[derive(Default)]
struct CgroupScan {
    limits: Vec<(u64, u64, u64)>, // limit, current, credit
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
    for (i, d) in dirs.iter().enumerate() {
        let current = read_value(&d.join(current_name));
        let limits: Vec<Value> = limit_names.iter().map(|n| read_value(&d.join(n))).collect();
        let mut unusable = matches!(current, Value::Bad | Value::Max);
        let mut finite = false;
        for v in &limits {
            unusable = unusable || matches!(v, Value::Bad);
            finite = finite || matches!(v, Value::Number(_));
        }
        if unusable {
            return false;
        }
        let cur = match current {
            Value::Absent => {
                if finite {
                    return false;
                }
                continue;
            }
            Value::Number(n) => n,
            _ => unreachable!(),
        };
        out.any_files = true;
        let credit = reclaimable_cache(d, cur, v1);
        if i == 0 {
            out.root_current = Some(cur);
            out.root_credit = credit;
        }
        for v in limits {
            if let Value::Number(n) = v {
                if v1 && n >= 1u64 << 62 {
                    continue;
                }
                out.limits.push((n, cur, credit));
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
    let mut dirs = Vec::new();
    if !root.is_dir() {
        return Some(dirs);
    }
    dirs.push(root.to_path_buf());
    let mut at = root.to_path_buf();
    for p in names {
        at.push(p);
        if !at.is_dir() {
            break;
        }
        dirs.push(at.clone());
    }
    Some(dirs)
}

fn scan_cgroups(paths: &MemoryProbePaths, out: &mut CgroupScan) -> bool {
    let (mut v2, mut v1) = (None::<String>, None::<String>);
    match std::fs::read_to_string(&paths.self_cgroup) {
        Err(_) => v2 = Some("/".to_string()),
        Ok(text) => {
            for line in text.lines() {
                let Some(c1) = line.find(':') else {
                    continue;
                };
                let Some(c2) = line[c1 + 1..].find(':').map(|x| x + c1 + 1) else {
                    continue;
                };
                let controllers = &line[c1 + 1..c2];
                let group = &line[c2 + 1..];
                if &line[..c1] == "0" && controllers.is_empty() {
                    v2 = Some(group.to_string());
                    continue;
                }
                if controllers.split(',').any(|n| n == "memory") {
                    v1 = Some(group.to_string());
                }
            }
        }
    }
    if let Some(g) = v2 {
        let Some(dirs) = group_dirs(&paths.cgroup_root, &g) else {
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
        if let Some(g) = v1 {
            let Some(dirs) = group_dirs(&paths.cgroup_root.join("memory"), &g) else {
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

fn combine(
    available: Option<u64>,
    total: Option<u64>,
    cg: &CgroupScan,
    explicit_limit: u64,
) -> Option<HostMemoryReading> {
    let available = available?;
    let mut r = HostMemoryReading {
        available,
        source: MemorySource::Meminfo,
        limit: total.unwrap_or(0),
        current: match total {
            Some(t) if t >= available => t - available,
            _ => 0,
        },
        credit: 0,
        mem_available: available,
    };
    for &(limit, current, credit) in &cg.limits {
        let used = current - credit;
        let room = limit.saturating_sub(used);
        if room < r.available {
            r = HostMemoryReading {
                available: room,
                source: MemorySource::Cgroup,
                limit,
                current,
                credit,
                ..r
            };
        }
    }
    if explicit_limit != 0 {
        let (used, credit) = match cg.root_current {
            Some(c) => (c, cg.root_credit),
            None => (total.filter(|&t| t >= available)? - available, 0),
        };
        let net = used - credit;
        let room = explicit_limit.saturating_sub(net);
        if room <= r.available {
            r = HostMemoryReading {
                available: room,
                source: MemorySource::Flag,
                limit: explicit_limit,
                current: used,
                credit,
                ..r
            };
        }
    }
    Some(r)
}

static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `sample_host_memory(paths, explicit_limit)`.
pub fn sample_host_memory_at(
    paths: &MemoryProbePaths,
    explicit_limit: u64,
) -> Option<HostMemoryReading> {
    let text = std::fs::read_to_string(&paths.meminfo).ok()?;
    let available = meminfo_kb_bytes(&text, "MemAvailable:")?;
    let total = meminfo_kb_bytes(&text, "MemTotal:");
    let mut cg = CgroupScan::default();
    if !scan_cgroups(paths, &mut cg) {
        if explicit_limit != 0 {
            return None;
        }
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[strata] memory guard: the cgroup memory files could not be read or parsed, using MemAvailable; \
                 set --memory-limit-mib to the container's limit"
            );
        }
        cg = CgroupScan::default();
    }
    combine(Some(available), total, &cg, explicit_limit)
}

/// `sample_host_memory()` with the engine's `--memory-limit-mib` (bytes, 0 = none).
pub fn sample_host_memory(explicit_limit: u64) -> Option<HostMemoryReading> {
    sample_host_memory_at(&MemoryProbePaths::default(), explicit_limit)
}

/// `ram_note` (generate.cpp): the reading as the refusal lines print it.
pub fn ram_note(mem: &Option<HostMemoryReading>) -> String {
    match mem {
        None => "RAM telemetry unavailable".to_string(),
        Some(m) => format!(
            "{} MiB available, source {}",
            m.available >> 20,
            m.source.name()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_parse_matches_cpp_rules() {
        let t = "MemTotal:       131072 kB\nMemAvailable:    65536 kB\n";
        assert_eq!(meminfo_kb_bytes(t, "MemAvailable:"), Some(65536 * 1024));
        assert_eq!(meminfo_kb_bytes(t, "MemTotal:"), Some(131072 * 1024));
        assert_eq!(
            meminfo_kb_bytes("MemAvailable: 1 MB\n", "MemAvailable:"),
            None
        );
        assert_eq!(
            meminfo_kb_bytes("MemAvailable: 1 kB\nMemAvailable: 2 kB\n", "MemAvailable:"),
            None
        );
    }

    #[test]
    fn explicit_limit_wins_on_tie() {
        let cg = CgroupScan {
            root_current: Some(100),
            root_credit: 0,
            ..Default::default()
        };
        let r = combine(Some(1000), Some(2000), &cg, 1100).unwrap();
        assert_eq!(r.source, MemorySource::Flag);
        assert_eq!(r.available, 1000);
    }
}
