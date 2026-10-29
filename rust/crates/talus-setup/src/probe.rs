//! Read-only facts about this machine: OS, CPU, GPU, memory, kernel parameters, groups, disk.
//!
//! Every file is read relative to a root (`/` normally, a fake tree in tests), and the few facts that need a command
//! (`id -Gn`, `systemctl`) are gathered in `probe` only, so the parsers below are pure.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SysInfo {
    pub os_id: String,
    pub os_like: String,
    pub os_version: String,
    pub os_pretty: String,
    pub kernel: String,
    pub arch: String,
    pub cpu_model: String,
    /// GPU targets from the KFD topology (e.g. "gfx1151"), CPU nodes left out.
    pub gpus: Vec<String>,
    pub mem_total_kib: u64,
    pub mem_avail_kib: u64,
    pub cmdline: String,
    /// `/sys/module/ttm/parameters/pages_limit` (4 KiB pages).
    pub ttm_pages_limit: Option<u64>,
    pub ttm_page_pool_size: Option<u64>,
    /// The GPU's GTT pool as amdgpu reports it (bytes).
    pub gtt_total: Option<u64>,
    /// The running kernel turns the kexec-handover reserve on by default (`CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y`).
    pub kho_default_on: Option<bool>,
    /// The user's groups.
    pub groups: Vec<String>,
    /// The groups owning /dev/kfd and the first render node.
    pub kfd_group: Option<String>,
    pub render_group: Option<String>,
    /// `systemctl get-default`.
    pub default_target: Option<String>,
    /// A display manager is running (a desktop session uses memory).
    pub desktop_active: bool,
    /// `systemctl --user` works (a user service can be installed).
    pub user_systemd: bool,
}

impl SysInfo {
    pub fn mem_total_gib(&self) -> f64 {
        self.mem_total_kib as f64 / (1024.0 * 1024.0)
    }
    pub fn mem_avail_gib(&self) -> f64 {
        self.mem_avail_kib as f64 / (1024.0 * 1024.0)
    }
    pub fn has_gfx1151(&self) -> bool {
        self.gpus.iter().any(|g| g == "gfx1151")
    }
    pub fn in_group(&self, g: &str) -> bool {
        self.groups.iter().any(|x| x == g)
    }
    /// Debian, Ubuntu and their derivatives.
    pub fn is_debian_like(&self) -> bool {
        self.os_id == "debian"
            || self.os_id == "ubuntu"
            || self
                .os_like
                .split_whitespace()
                .any(|w| w == "debian" || w == "ubuntu")
    }
    pub fn is_fedora_like(&self) -> bool {
        self.os_id == "fedora"
            || self
                .os_like
                .split_whitespace()
                .any(|w| w == "fedora" || w == "rhel")
    }
}

fn read(root: &Path, p: &str) -> Option<String> {
    std::fs::read_to_string(root.join(p.trim_start_matches('/'))).ok()
}

/// `KEY=value` lines of /etc/os-release, values unquoted.
pub fn parse_os_release(s: &str) -> Vec<(String, String)> {
    s.lines()
        .filter_map(|l| {
            let (k, v) = l.split_once('=')?;
            let v = v.trim();
            let v = v
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .unwrap_or(v);
            Some((k.trim().to_string(), v.to_string()))
        })
        .collect()
}

/// A `/proc/meminfo` field in KiB.
pub fn meminfo_kib(s: &str, key: &str) -> Option<u64> {
    s.lines().find_map(|l| {
        let rest = l.strip_prefix(key)?.strip_prefix(':')?;
        rest.split_whitespace().next()?.parse().ok()
    })
}

/// KFD's `gfx_target_version` (e.g. 110501) as a target name (gfx1151). 0 is a CPU node.
pub fn gfx_name(v: u64) -> Option<String> {
    if v == 0 {
        return None;
    }
    let (major, minor, step) = (v / 10000, (v / 100) % 100, v % 100);
    Some(format!("gfx{major}{minor}{step:x}"))
}

/// The value of `key=` on a kernel command line (the last one wins, as in the kernel).
pub fn cmdline_value<'a>(cmdline: &'a str, key: &str) -> Option<&'a str> {
    cmdline
        .split_whitespace()
        .filter_map(|w| w.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
        .next_back()
}

/// The name of group `gid` in an /etc/group file.
pub fn group_name(etc_group: &str, gid: u32) -> Option<String> {
    etc_group.lines().find_map(|l| {
        let mut it = l.split(':');
        let name = it.next()?;
        let _pw = it.next()?;
        let g: u32 = it.next()?.parse().ok()?;
        (g == gid).then(|| name.to_string())
    })
}

fn file_gid(p: &Path) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).ok().map(|m| m.gid())
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Gather everything. `root` is `/` except in tests; commands only run when `root` is `/`.
pub fn probe(root: &Path) -> SysInfo {
    let mut s = SysInfo::default();
    if let Some(osr) = read(root, "/etc/os-release") {
        for (k, v) in parse_os_release(&osr) {
            match k.as_str() {
                "ID" => s.os_id = v,
                "ID_LIKE" => s.os_like = v,
                "VERSION_ID" => s.os_version = v,
                "PRETTY_NAME" => s.os_pretty = v,
                _ => {}
            }
        }
    }
    s.kernel = read(root, "/proc/sys/kernel/osrelease")
        .map(|k| k.trim().to_string())
        .unwrap_or_default();
    s.arch = std::env::consts::ARCH.to_string();
    if let Some(ci) = read(root, "/proc/cpuinfo") {
        s.cpu_model = ci
            .lines()
            .find_map(|l| {
                l.strip_prefix("model name")
                    .and_then(|r| r.split_once(':'))
                    .map(|(_, v)| v.trim().to_string())
            })
            .unwrap_or_default();
    }
    if let Some(mi) = read(root, "/proc/meminfo") {
        s.mem_total_kib = meminfo_kib(&mi, "MemTotal").unwrap_or(0);
        s.mem_avail_kib = meminfo_kib(&mi, "MemAvailable").unwrap_or(0);
    }
    let nodes = root.join("sys/class/kfd/kfd/topology/nodes");
    if let Ok(rd) = std::fs::read_dir(&nodes) {
        let mut ns: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        ns.sort();
        for n in ns {
            if let Ok(props) = std::fs::read_to_string(n.join("properties")) {
                let v = props.lines().find_map(|l| {
                    l.strip_prefix("gfx_target_version ")
                        .and_then(|v| v.trim().parse::<u64>().ok())
                });
                if let Some(g) = v.and_then(gfx_name) {
                    s.gpus.push(g);
                }
            }
        }
    }
    s.cmdline = read(root, "/proc/cmdline")
        .map(|c| c.trim().to_string())
        .unwrap_or_default();
    s.ttm_pages_limit =
        read(root, "/sys/module/ttm/parameters/pages_limit").and_then(|v| v.trim().parse().ok());
    s.ttm_page_pool_size =
        read(root, "/sys/module/ttm/parameters/page_pool_size").and_then(|v| v.trim().parse().ok());
    if let Ok(rd) = std::fs::read_dir(root.join("sys/class/drm")) {
        let mut cards: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("card") && !n.contains('-'))
                    .unwrap_or(false)
            })
            .collect();
        cards.sort();
        s.gtt_total = cards.iter().find_map(|c| {
            std::fs::read_to_string(c.join("device/mem_info_gtt_total"))
                .ok()
                .and_then(|v| v.trim().parse().ok())
        });
    }
    if !s.kernel.is_empty() {
        if let Some(cfg) = read(root, &format!("/boot/config-{}", s.kernel)) {
            s.kho_default_on = Some(
                cfg.lines()
                    .any(|l| l.trim() == "CONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y"),
            );
        }
    }
    let etc_group = read(root, "/etc/group").unwrap_or_default();
    s.kfd_group = file_gid(&root.join("dev/kfd")).and_then(|g| group_name(&etc_group, g));
    s.render_group = std::fs::read_dir(root.join("dev/dri"))
        .ok()
        .and_then(|rd| {
            let mut v: Vec<PathBuf> = rd
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("renderD"))
                        .unwrap_or(false)
                })
                .collect();
            v.sort();
            v.into_iter().next()
        })
        .and_then(|p| file_gid(&p))
        .and_then(|g| group_name(&etc_group, g));
    if root == Path::new("/") {
        s.groups = run("id", &["-Gn"])
            .map(|g| g.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        s.default_target = run("systemctl", &["get-default"]);
        s.desktop_active =
            run("systemctl", &["is-active", "display-manager"]).as_deref() == Some("active");
        s.user_systemd = Command::new("systemctl")
            .args(["--user", "show-environment"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
    }
    s
}

/// Free bytes on the filesystem holding `p` (or its nearest existing parent).
pub fn free_bytes(p: &Path) -> Option<u64> {
    let mut q = p.to_path_buf();
    while !q.exists() {
        if !q.pop() {
            return None;
        }
    }
    let c = std::ffi::CString::new(q.as_os_str().as_encoded_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: c is a valid NUL-terminated path and st a valid out pointer.
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut st) };
    (rc == 0).then(|| st.f_bavail as u64 * st.f_frsize as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsers() {
        let osr = "NAME=\"Ubuntu\"\nID=ubuntu\nID_LIKE=debian\nVERSION_ID=\"26.04\"\nPRETTY_NAME=\"Ubuntu 26.04.1 LTS\"\n";
        let kv = parse_os_release(osr);
        assert!(kv.contains(&("VERSION_ID".into(), "26.04".into())));
        let mi = "MemTotal:       131142852 kB\nMemFree:        14011080 kB\nMemAvailable:   64017788 kB\n";
        assert_eq!(meminfo_kib(mi, "MemTotal"), Some(131142852));
        assert_eq!(meminfo_kib(mi, "MemAvailable"), Some(64017788));
        assert_eq!(meminfo_kib(mi, "Mem"), None);
        assert_eq!(gfx_name(110501).as_deref(), Some("gfx1151"));
        assert_eq!(gfx_name(120001).as_deref(), Some("gfx1201"));
        assert_eq!(gfx_name(90010).as_deref(), Some("gfx90a"));
        assert_eq!(gfx_name(0), None);
        let cl =
            "BOOT_IMAGE=/vmlinuz ro quiet kho=off ttm.pages_limit=100 ttm.pages_limit=31457280";
        assert_eq!(cmdline_value(cl, "ttm.pages_limit"), Some("31457280"));
        assert_eq!(cmdline_value(cl, "kho"), Some("off"));
        assert_eq!(cmdline_value(cl, "amdgpu.lockup_timeout"), None);
        let g = "root:x:0:\nvideo:x:44:user1\nrender:x:990:user1\n";
        assert_eq!(group_name(g, 990).as_deref(), Some("render"));
        assert_eq!(group_name(g, 7), None);
    }

    #[test]
    fn probe_fake_halo_tree() {
        let root = crate::testutil::fake_halo_root("probe", true);
        let s = probe(&root);
        assert_eq!(s.os_id, "ubuntu");
        assert!(s.is_debian_like());
        assert_eq!(s.gpus, vec!["gfx1151".to_string()]);
        assert!(s.has_gfx1151());
        assert_eq!(s.ttm_pages_limit, Some(31457280));
        assert_eq!(s.gtt_total, Some(128849018880));
        assert_eq!(s.kho_default_on, Some(true));
        assert!(s.mem_total_gib() > 124.0);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn free_bytes_walks_up() {
        assert!(free_bytes(Path::new("/definitely/not/here/x")).is_some());
    }
}
