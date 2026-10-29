//! Test helpers: temporary directories and a fake Strix Halo root (/proc, /sys, /etc) for the probe.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

static N: AtomicU32 = AtomicU32::new(0);

pub fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "talus-setup-test-{tag}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn put(root: &Path, p: &str, s: &str) {
    let f = root.join(p);
    std::fs::create_dir_all(f.parent().unwrap()).unwrap();
    std::fs::write(f, s).unwrap();
}

/// The release machine as files: Ubuntu 26.04, kernel 7.0, gfx1151, 125 GiB. `tuned`: the documented kernel
/// parameters are set.
pub fn fake_halo_root(tag: &str, tuned: bool) -> PathBuf {
    let r = tempdir(tag);
    put(&r, "etc/os-release", "NAME=\"Ubuntu\"\nID=ubuntu\nID_LIKE=debian\nVERSION_ID=\"26.04\"\nPRETTY_NAME=\"Ubuntu 26.04.1 LTS\"\n");
    put(
        &r,
        "etc/default/grub",
        "GRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash\"\n",
    );
    std::fs::create_dir_all(r.join("etc/default/grub.d")).unwrap();
    put(&r, "etc/group", "root:x:0:\nvideo:x:44:u\nrender:x:990:u\n");
    put(&r, "proc/sys/kernel/osrelease", "7.0.0-38-generic\n");
    put(
        &r,
        "proc/cpuinfo",
        "processor\t: 0\nmodel name\t: AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S\n",
    );
    put(
        &r,
        "proc/meminfo",
        "MemTotal:       131142852 kB\nMemFree:        14011080 kB\nMemAvailable:   124017788 kB\n",
    );
    let (cl, pages) = if tuned {
        (
            "BOOT_IMAGE=/vmlinuz-7.0.0-38-generic ro quiet amdgpu.lockup_timeout=10000,60000,10000,10000 kho=off ttm.pages_limit=31457280 ttm.page_pool_size=31457280",
            "31457280",
        )
    } else {
        (
            "BOOT_IMAGE=/vmlinuz-7.0.0-38-generic ro quiet splash",
            "16384000",
        )
    };
    put(&r, "proc/cmdline", &format!("{cl}\n"));
    put(
        &r,
        "sys/module/ttm/parameters/pages_limit",
        &format!("{pages}\n"),
    );
    put(
        &r,
        "sys/module/ttm/parameters/page_pool_size",
        &format!("{pages}\n"),
    );
    put(
        &r,
        "sys/class/kfd/kfd/topology/nodes/0/properties",
        "cpu_cores_count 32\ngfx_target_version 0\n",
    );
    put(
        &r,
        "sys/class/kfd/kfd/topology/nodes/1/properties",
        "simd_count 80\ngfx_target_version 110501\n",
    );
    put(
        &r,
        "sys/class/drm/card1/device/mem_info_gtt_total",
        "128849018880\n",
    );
    put(
        &r,
        "boot/config-7.0.0-38-generic",
        "CONFIG_KEXEC_HANDOVER=y\nCONFIG_KEXEC_HANDOVER_ENABLE_DEFAULT=y\n",
    );
    r
}
