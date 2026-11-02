//! Build and configuration identity, compared at handshake (DESIGN.md 4.2 items 1-3). Two stages only produce output
//! bitwise equal to one process when they run the same bytes with the same knobs, so the worker refuses a head whose
//! identity differs (`ERR mismatch <field>`).

use sha2::{Digest, Sha256};
use std::io::Read;

fn hex16(d: &[u8]) -> String {
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// sha256 of a whole file (first 16 hex digits), or "none".
pub fn file_sha(path: &str) -> String {
    let Ok(mut f) = std::fs::File::open(path) else {
        return "none".into();
    };
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => h.update(&buf[..n]),
            Err(_) => return "none".into(),
        }
    }
    hex16(&h.finalize())
}

/// sha256 of the first `n` bytes of a file plus its size: for a GGUF the header, metadata and tensor table.
pub fn file_head_sha(path: &str, n: usize) -> String {
    let Ok(mut f) = std::fs::File::open(path) else {
        return "none".into();
    };
    let size = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut buf = vec![0u8; n];
    let mut got = 0usize;
    while got < n {
        match f.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(_) => return "none".into(),
        }
    }
    let mut h = Sha256::new();
    h.update(size.to_le_bytes());
    h.update(&buf[..got]);
    format!("{}:{size}", hex16(&h.finalize()))
}

/// The path of the loaded `libtalus_kernels.so` (from /proc/self/maps).
pub fn kernels_lib_path() -> Option<String> {
    let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
    maps.lines()
        .filter_map(|l| l.split_whitespace().nth(5))
        .find(|p| p.contains("libtalus_kernels"))
        .map(str::to_string)
}

pub fn exe_sha() -> String {
    std::fs::read_link("/proc/self/exe")
        .ok()
        .map(|p| file_sha(&p.to_string_lossy()))
        .unwrap_or_else(|| "none".into())
}

pub fn kernels_sha() -> String {
    kernels_lib_path()
        .map(|p| file_sha(&p))
        .unwrap_or_else(|| "none".into())
}

/// Variables that shape kernels or engine paths: every `TALUS_*`, `STRATA_*`, `GLM_*`, `HIP_*`, `HSA_*`, `ROCR_*`,
/// except the ones that only name files, ports, keys or the cluster itself.
pub fn env_relevant(name: &str) -> bool {
    let pre = ["TALUS_", "STRATA_", "GLM_", "HIP_", "HSA_", "ROCR_"];
    if !pre.iter().any(|p| name.starts_with(p)) {
        return false;
    }
    // TALUS_X is mirrored to STRATA_X at startup (talus-core-api env_alias): judge both under one name
    let rest = name
        .strip_prefix("TALUS_")
        .or_else(|| name.strip_prefix("STRATA_"));
    let skip = ["CLUSTER", "API_KEY", "LOG", "KERNELS_LIB", "PCACHE", "HOST"];
    if rest.is_some_and(|r| skip.iter().any(|p| r.starts_with(p))) {
        return false;
    }
    !(name.contains("KEY")
        || name.contains("TOKEN")
        || name.contains("SECRET")
        || name.ends_with("_PATH"))
}

/// Sorted `name=value` of the relevant environment as it was at the first call: take it at startup, before the
/// kernel library loads (it `setenv`s some defaults of its own, e.g. `STRATA_Q8_PACKED`, `STRATA_MMQ_KQ`).
pub fn env_pairs() -> Vec<(String, String)> {
    static SNAP: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    SNAP.get_or_init(|| {
        let mut v: Vec<(String, String)> =
            std::env::vars().filter(|(k, _)| env_relevant(k)).collect();
        v.sort();
        v
    })
    .clone()
}

pub fn env_hash(pairs: &[(String, String)]) -> String {
    let mut h = Sha256::new();
    for (k, v) in pairs {
        h.update(k.as_bytes());
        h.update(b"=");
        h.update(v.as_bytes());
        h.update(b"\n");
    }
    hex16(&h.finalize())
}

/// Names (never values) of the variables that differ between two environments.
pub fn env_diff(a: &[(String, String)], b: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    for (k, v) in a {
        match b.iter().find(|(k2, _)| k2 == k) {
            Some((_, v2)) if v2 == v => {}
            _ => out.push(k.clone()),
        }
    }
    for (k, _) in b {
        if !a.iter().any(|(k2, _)| k2 == k) {
            out.push(k.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// One string that identifies the box (two stages on the same box share its memory budget).
pub fn box_id() -> String {
    let mid = std::fs::read_to_string("/etc/machine-id").unwrap_or_default();
    let boot = std::fs::read_to_string("/proc/sys/kernel/random/boot_id").unwrap_or_default();
    format!("{}-{}", mid.trim(), boot.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_filter() {
        assert!(env_relevant("TALUS_MMVQ_WIDE"));
        assert!(env_relevant("STRATA_FUSED"));
        assert!(!env_relevant("TALUS_API_KEY"));
        assert!(!env_relevant("TALUS_CLUSTER_SPLIT"));
        assert!(!env_relevant("STRATA_CLUSTER_ONCE"));
        assert!(!env_relevant("STRATA_PCACHE_DISK"));
        assert!(!env_relevant("PATH"));
        let a = vec![
            ("A".to_string(), "1".to_string()),
            ("B".to_string(), "2".to_string()),
        ];
        let b = vec![
            ("A".to_string(), "1".to_string()),
            ("C".to_string(), "2".to_string()),
        ];
        assert_eq!(env_diff(&a, &b), vec!["B".to_string(), "C".to_string()]);
        assert_ne!(env_hash(&a), env_hash(&b));
    }
}
