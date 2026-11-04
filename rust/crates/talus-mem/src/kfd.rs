//! amdgpu KFD (ROCm) memory limits on an APU, read from /proc and /sys without root.
//!
//! On Strix Halo the VRAM carve-out is smaller than GTT, so amdgpu sets `apu_prefer_gtt` and every `hipMalloc` is a
//! GTT buffer of system RAM. The kernel (`amdgpu_amdkfd_reserve_mem_limit`, kernel 6.x-7.0) then charges each one to
//! two process-wide counters shared by every ROCm process on the box, and refuses the allocation (`hipMalloc` returns
//! out of memory) when either would pass its limit:
//!
//! - system limit = MemTotal - MemTotal/64 - 1.5 GiB (`amdgpu_amdkfd_gpuvm_init_mem_limits`; off with
//!   `amdgpu.no_system_mem_limit=1`);
//! - TTM limit = `ttm.pages_limit` x 4 KiB.
//!
//! `amdgpu.gttsize` only sizes the GTT manager; raising it above these does not raise them. Whichever limit is lower
//! is the most a ROCm process can allocate, minus what other ROCm processes hold.

use std::path::Path;

const GIB_F: f64 = (1u64 << 30) as f64;
const RESERVE_MEM_LIMIT: u64 = 3 << 29; // AMDGPU_RESERVE_MEM_LIMIT, 1.5 GiB

/// The KFD system-memory limit the kernel derives from RAM (`mem` = MemTotal in bytes).
pub fn system_limit_for(mem: u64) -> u64 {
    let l = mem - (mem >> 6);
    if l < 2 * RESERVE_MEM_LIMIT {
        l >> 1
    } else {
        l - RESERVE_MEM_LIMIT
    }
}

/// What this host's amdgpu/ttm settings allow a ROCm process to allocate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KfdLimits {
    /// MemTotal in bytes.
    pub mem_total: u64,
    /// KFD system limit; `None` when `amdgpu.no_system_mem_limit` is set.
    pub system_limit: Option<u64>,
    /// `ttm.pages_limit` in bytes; `None` when unreadable (no ttm module).
    pub ttm_limit: Option<u64>,
    /// `amdgpu.gttsize` in bytes when set (not -1).
    pub gttsize: Option<u64>,
}

fn read_trim(p: &Path) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

impl KfdLimits {
    /// Limits from explicit inputs (MemTotal kB, ttm pages_limit, amdgpu no_system_mem_limit, amdgpu gttsize MiB).
    pub fn from_parts(
        mem_total_kb: u64,
        ttm_pages_limit: Option<u64>,
        no_system_mem_limit: bool,
        gttsize_mib: Option<i64>,
    ) -> Self {
        let mem_total = mem_total_kb * 1024;
        Self {
            mem_total,
            system_limit: (!no_system_mem_limit).then(|| system_limit_for(mem_total)),
            ttm_limit: ttm_pages_limit.filter(|&p| p > 0).map(|p| p * 4096),
            gttsize: gttsize_mib.filter(|&m| m > 0).map(|m| (m as u64) << 20),
        }
    }

    /// Read /proc/meminfo and the amdgpu/ttm module parameters. `None` when MemTotal is unreadable.
    pub fn read() -> Option<Self> {
        Self::read_from(Path::new("/proc/meminfo"), Path::new("/sys/module"))
    }

    pub fn read_from(meminfo: &Path, sys_module: &Path) -> Option<Self> {
        let mi = std::fs::read_to_string(meminfo).ok()?;
        let kb = mi.lines().find_map(|l| {
            l.strip_prefix("MemTotal:")
                .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })?;
        let pages =
            read_trim(&sys_module.join("ttm/parameters/pages_limit")).and_then(|s| s.parse().ok());
        let nsl = read_trim(&sys_module.join("amdgpu/parameters/no_system_mem_limit"))
            .map(|s| s == "Y" || s == "1")
            .unwrap_or(false);
        let gtt =
            read_trim(&sys_module.join("amdgpu/parameters/gttsize")).and_then(|s| s.parse().ok());
        Some(Self::from_parts(kb, pages, nsl, gtt))
    }

    /// The binding limit: the lower of the system and TTM limits (`None` if neither applies).
    pub fn effective(&self) -> Option<u64> {
        match (self.system_limit, self.ttm_limit) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    /// One line for logs and errors, e.g. "KFD limit 118.79 GiB (system: 15/16 of MemTotal 122.20 GiB - 1.5 GiB =
    /// 118.79 GiB; ttm.pages_limit 124.00 GiB; amdgpu.gttsize 124.00 GiB)".
    pub fn describe(&self) -> String {
        let g = |b: u64| format!("{:.2} GiB", b as f64 / GIB_F);
        let sys = match self.system_limit {
            Some(s) => format!(
                "system: MemTotal {} - 1/64 - 1.5 GiB = {}",
                g(self.mem_total),
                g(s)
            ),
            None => "system: off (amdgpu.no_system_mem_limit)".into(),
        };
        let ttm = self
            .ttm_limit
            .map(|t| format!("ttm.pages_limit {}", g(t)))
            .unwrap_or_else(|| "ttm.pages_limit unknown".into());
        let gtt = self
            .gttsize
            .map(|t| format!("; amdgpu.gttsize {}", g(t)))
            .unwrap_or_default();
        let eff = self.effective().map(g).unwrap_or_else(|| "unknown".into());
        format!("KFD limit {eff} ({sys}; {ttm}{gtt})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn system_limit_matches_kernel_debugfs() {
        // test desktop: MemTotal 131010252 kB; /sys/kernel/debug/kfd/mem_limit "out of 124404M"
        assert_eq!(system_limit_for(131_010_252 * 1024) / MIB, 124_404);
        // tiny hosts halve instead of subtracting the reserve
        assert_eq!(
            system_limit_for(2 << 30),
            ((2u64 << 30) - ((2u64 << 30) >> 6)) >> 1
        );
    }

    #[test]
    fn talus_public_6_box() {
        // BOSGAME 128 GB: MemTotal 128134860 kB, ttm.pages_limit=32505856, amdgpu.gttsize=126976
        let l = KfdLimits::from_parts(128_134_860, Some(32_505_856), false, Some(126_976));
        assert_eq!(l.ttm_limit, Some(126_976 * MIB));
        assert_eq!(l.gttsize, Some(126_976 * MIB));
        // the system limit (118.79 GiB) binds, not the 124 GiB TTM/GTT pool
        let s = l.system_limit.unwrap();
        assert_eq!(l.effective(), Some(s));
        assert!(
            (s as f64 / GIB_F - 118.79).abs() < 0.01,
            "{}",
            s as f64 / GIB_F
        );
        assert!(
            l.describe().starts_with("KFD limit 118.79 GiB"),
            "{}",
            l.describe()
        );
    }

    #[test]
    fn ttm_limit_binds() {
        // test laptop: MemTotal 131142860 kB, ttm.pages_limit=31457280 (120 GiB), no gttsize
        let l = KfdLimits::from_parts(131_142_860, Some(31_457_280), false, Some(-1));
        assert_eq!(l.effective(), Some(120 << 30));
        assert_eq!(l.gttsize, None);
        let off = KfdLimits::from_parts(131_142_860, Some(31_457_280), true, None);
        assert_eq!(off.system_limit, None);
        assert_eq!(off.effective(), Some(120 << 30));
    }

    #[test]
    fn read_from_files() {
        let d = std::env::temp_dir().join(format!("talus-kfd-{}", std::process::id()));
        std::fs::create_dir_all(d.join("ttm/parameters")).unwrap();
        std::fs::create_dir_all(d.join("amdgpu/parameters")).unwrap();
        std::fs::write(
            d.join("meminfo"),
            "MemTotal:       128134860 kB\nMemFree: 1 kB\n",
        )
        .unwrap();
        std::fs::write(d.join("ttm/parameters/pages_limit"), "32505856\n").unwrap();
        std::fs::write(d.join("amdgpu/parameters/no_system_mem_limit"), "N\n").unwrap();
        std::fs::write(d.join("amdgpu/parameters/gttsize"), "126976\n").unwrap();
        let l = KfdLimits::read_from(&d.join("meminfo"), &d).unwrap();
        assert_eq!(
            l,
            KfdLimits::from_parts(128_134_860, Some(32_505_856), false, Some(126_976))
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
