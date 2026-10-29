//! The kernel parameters GLM-5.3-Flash and MiMo-V2.6-Flash need (docs/MODELS.md "System setup"):
//!
//! ```text
//! ttm.pages_limit=31457280 ttm.page_pool_size=31457280   # GPU memory pool 120 GiB
//! kho=off                                                # free the ~9 GiB kexec-handover reserve
//! amdgpu.lockup_timeout=10000,60000,10000,10000          # let long prompt passes run (60 s compute timeout)
//! ```
//!
//! This module works out which are missing, renders a GRUB drop-in for them, and builds the plan to apply it. It
//! never applies anything itself: the caller shows the plan and the diff, asks for an explicit "yes", and runs the
//! commands. Nothing here reboots; the user is told to.

use crate::probe::{cmdline_value, SysInfo};
use std::path::{Path, PathBuf};

/// 120 GiB in 4 KiB pages.
pub const PAGES_120_GIB: u64 = 31_457_280;
pub const LOCKUP_TIMEOUT: &str = "10000,60000,10000,10000";
/// Ubuntu / Debian: GRUB reads every `/etc/default/grub.d/*.cfg`.
pub const DROPIN_DIR: &str = "/etc/default/grub.d";
pub const DROPIN_NAME: &str = "90-talus.cfg";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub key: &'static str,
    pub value: String,
    /// Why, in plain words.
    pub why: &'static str,
}

impl Param {
    pub fn arg(&self) -> String {
        format!("{}={}", self.key, self.value)
    }
}

/// Every parameter we document, whether or not it is missing.
pub fn documented() -> Vec<Param> {
    vec![
        Param {
            key: "ttm.pages_limit",
            value: PAGES_120_GIB.to_string(),
            why: "lets the GPU use up to 120 GiB of the shared memory (the default is about half)",
        },
        Param {
            key: "ttm.page_pool_size",
            value: PAGES_120_GIB.to_string(),
            why: "the matching page pool size",
        },
        Param {
            key: "kho",
            value: "off".into(),
            why: "frees the ~9 GiB kexec-handover reserve that Ubuntu 7.0 kernels keep",
        },
        Param {
            key: "amdgpu.lockup_timeout",
            value: LOCKUP_TIMEOUT.into(),
            why: "gives long prompt passes 60 s before the GPU driver resets them",
        },
    ]
}

/// The state of each documented parameter on this machine.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamStatus {
    pub param: Param,
    /// What the running kernel uses now, in words.
    pub current: String,
    pub ok: bool,
}

/// Compare the running kernel against the documented parameters.
///
/// The GPU pool counts as fine when the live `ttm.pages_limit` (or the one on the command line) is at least 120 GiB.
/// `kho=off` is only needed when the kernel enables the reserve by default.
pub fn status(sys: &SysInfo) -> Vec<ParamStatus> {
    let cl = &sys.cmdline;
    // 0 in /sys means "the kernel's default"
    let pages = sys
        .ttm_pages_limit
        .filter(|&p| p > 0)
        .or_else(|| cmdline_value(cl, "ttm.pages_limit").and_then(|v| v.parse().ok()));
    let pool = sys
        .ttm_page_pool_size
        .filter(|&p| p > 0)
        .or_else(|| cmdline_value(cl, "ttm.page_pool_size").and_then(|v| v.parse().ok()));
    let gib = |p: u64| format!("{:.0} GiB", p as f64 * 4096.0 / (1u64 << 30) as f64);
    documented()
        .into_iter()
        .map(|param| {
            let (current, ok) = match param.key {
                "ttm.pages_limit" => match pages {
                    Some(p) => (gib(p), p >= PAGES_120_GIB),
                    None => ("kernel default (about half of the memory)".into(), false),
                },
                "ttm.page_pool_size" => match pool {
                    Some(p) => (gib(p), p >= PAGES_120_GIB),
                    None => ("kernel default".into(), false),
                },
                "kho" => match (cmdline_value(cl, "kho"), sys.kho_default_on) {
                    (Some(v), _) => (format!("kho={v}"), v == "off" || v == "0"),
                    (None, Some(true)) => ("reserve on (kernel default)".into(), false),
                    (None, Some(false)) => ("no reserve in this kernel".into(), true),
                    (None, None) => ("unknown (kernel config not readable)".into(), true),
                },
                "amdgpu.lockup_timeout" => match cmdline_value(cl, "amdgpu.lockup_timeout") {
                    Some(v) => (
                        v.to_string(),
                        v == LOCKUP_TIMEOUT || compute_timeout_ms(v) >= 60_000,
                    ),
                    None => ("driver default (shorter)".into(), false),
                },
                _ => unreachable!(),
            };
            ParamStatus { param, current, ok }
        })
        .collect()
}

/// The compute-queue timeout (second field) of an `amdgpu.lockup_timeout` value, in ms.
fn compute_timeout_ms(v: &str) -> u64 {
    let f: Vec<&str> = v.split(',').collect();
    let pick = if f.len() >= 2 { f[1] } else { f[0] };
    pick.trim().parse().unwrap_or(0)
}

pub fn missing(sys: &SysInfo) -> Vec<Param> {
    status(sys)
        .into_iter()
        .filter(|s| !s.ok)
        .map(|s| s.param)
        .collect()
}

/// How this system's boot loader takes extra kernel parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BootLoader {
    /// Debian / Ubuntu GRUB: a drop-in in /etc/default/grub.d, then `update-grub`.
    GrubDropin { dir: PathBuf, update: Vec<String> },
    /// Fedora / RHEL: `grubby --update-kernel=ALL --args=...`.
    Grubby,
    /// Anything else: we only print the lines.
    Manual(String),
}

pub fn detect_loader(sys: &SysInfo, root: &Path, have: &dyn Fn(&str) -> bool) -> BootLoader {
    let dropin_dir = root.join(DROPIN_DIR.trim_start_matches('/'));
    if sys.is_debian_like() && root.join("etc/default/grub").exists() && have("update-grub") {
        return BootLoader::GrubDropin {
            dir: PathBuf::from(DROPIN_DIR),
            update: vec!["update-grub".into()],
        };
    }
    if sys.is_fedora_like() && have("grubby") {
        return BootLoader::Grubby;
    }
    let why = if root.join("etc/kernel/cmdline").exists() {
        "this system boots with systemd-boot/UKI (edit /etc/kernel/cmdline)".to_string()
    } else if root.join("etc/default/grub").exists() && !dropin_dir.exists() {
        "this GRUB does not read /etc/default/grub.d (add the parameters to GRUB_CMDLINE_LINUX_DEFAULT in /etc/default/grub, then regenerate grub.cfg)".to_string()
    } else {
        "this boot loader is not one the installer knows".to_string()
    };
    BootLoader::Manual(why)
}

/// The drop-in file for `params`.
pub fn render_dropin(params: &[Param], date: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# Talus: kernel parameters for large models (written by `talus setup` on {date}).\n"
    ));
    s.push_str(
        "# GLM-5.3-Flash and MiMo-V2.6-Flash need them; see docs/MODELS.md \"System setup\".\n",
    );
    for p in params {
        s.push_str(&format!("#   {}: {}\n", p.arg(), p.why));
    }
    s.push_str(&format!(
        "# To undo: sudo rm {DROPIN_DIR}/{DROPIN_NAME} && sudo update-grub, then reboot.\n"
    ));
    let args: Vec<String> = params.iter().map(Param::arg).collect();
    s.push_str(&format!(
        "GRUB_CMDLINE_LINUX_DEFAULT=\"$GRUB_CMDLINE_LINUX_DEFAULT {}\"\n",
        args.join(" ")
    ));
    s
}

/// A unified diff between an old file (None = it does not exist) and the new text, whole-file granularity.
pub fn diff(path: &str, old: Option<&str>, new: &str) -> String {
    let mut out = String::new();
    match old {
        None => {
            out.push_str("--- /dev/null\n");
            out.push_str(&format!("+++ {path}\n"));
            out.push_str(&format!("@@ -0,0 +1,{} @@\n", new.lines().count()));
        }
        Some(o) => {
            out.push_str(&format!("--- {path}\n+++ {path}\n"));
            out.push_str(&format!(
                "@@ -1,{} +1,{} @@\n",
                o.lines().count(),
                new.lines().count()
            ));
            for l in o.lines() {
                out.push_str(&format!("-{l}\n"));
            }
        }
    }
    for l in new.lines() {
        out.push_str(&format!("+{l}\n"));
    }
    out
}

/// One step of applying the parameters, shown to the user before anything runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Run with sudo.
    Sudo(Vec<String>),
    /// Write `content` to the root-owned `path` (through `sudo install`).
    WriteRoot { path: PathBuf, content: String },
}

impl Step {
    pub fn describe(&self) -> String {
        match self {
            Step::Sudo(a) => format!("sudo {}", shell_join(a)),
            Step::WriteRoot { path, .. } => {
                format!("sudo install -m 0644 <the file above> {}", path.display())
            }
        }
    }
}

pub fn shell_join(a: &[String]) -> String {
    a.iter()
        .map(|w| {
            if w.chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_./=,:@%+".contains(c))
                && !w.is_empty()
            {
                w.clone()
            } else {
                format!("'{}'", w.replace('\'', "'\\''"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// What applying `params` would do, the drop-in text (if any) and its diff. `existing` is the current content of
/// our drop-in (None if absent). `stamp` names the backups.
pub struct Plan {
    pub steps: Vec<Step>,
    pub file: Option<(PathBuf, String)>,
    pub diff: Option<String>,
    /// The lines to copy for "I'll do it myself".
    pub manual: String,
}

pub fn plan(
    loader: &BootLoader,
    params: &[Param],
    existing: Option<&str>,
    date: &str,
    stamp: &str,
) -> Plan {
    let args: Vec<String> = params.iter().map(Param::arg).collect();
    let content = render_dropin(params, date);
    let dropin_path = PathBuf::from(DROPIN_DIR).join(DROPIN_NAME);
    let manual_dropin = format!(
        "sudo tee {} >/dev/null <<'EOF'\n{}EOF\nsudo update-grub\n# then reboot",
        dropin_path.display(),
        content
    );
    match loader {
        BootLoader::GrubDropin { dir, update } => {
            let path = dir.join(DROPIN_NAME);
            let mut steps = Vec::new();
            if existing.is_some() {
                steps.push(Step::Sudo(vec![
                    "cp".into(),
                    "-p".into(),
                    path.display().to_string(),
                    format!("{}.bak-{stamp}", path.display()),
                ]));
            }
            // keep the generated grub.cfg too, so the previous boot entries can be restored by hand
            steps.push(Step::Sudo(vec![
                "cp".into(),
                "-p".into(),
                "/boot/grub/grub.cfg".into(),
                format!("/boot/grub/grub.cfg.bak-talus-{stamp}"),
            ]));
            steps.push(Step::WriteRoot {
                path: path.clone(),
                content: content.clone(),
            });
            steps.push(Step::Sudo(update.clone()));
            Plan {
                steps,
                diff: Some(diff(&path.display().to_string(), existing, &content)),
                file: Some((path, content)),
                manual: manual_dropin,
            }
        }
        BootLoader::Grubby => {
            let cmd = vec![
                "grubby".to_string(),
                "--update-kernel=ALL".to_string(),
                format!("--args={}", args.join(" ")),
            ];
            Plan {
                manual: format!("sudo {}\n# then reboot", shell_join(&cmd)),
                steps: vec![Step::Sudo(cmd)],
                file: None,
                diff: None,
            }
        }
        BootLoader::Manual(why) => Plan {
            steps: vec![],
            file: None,
            diff: None,
            manual: format!(
                "# {why}.\n# Add these kernel parameters, update your boot loader, then reboot:\n{}",
                args.join(" ")
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(cmdline: &str, pages: Option<u64>, kho_default: Option<bool>) -> SysInfo {
        SysInfo {
            os_id: "ubuntu".into(),
            os_like: "debian".into(),
            cmdline: cmdline.into(),
            ttm_pages_limit: pages,
            ttm_page_pool_size: pages,
            kho_default_on: kho_default,
            ..Default::default()
        }
    }

    #[test]
    fn status_on_a_default_and_a_tuned_kernel() {
        let fresh = sys(
            "BOOT_IMAGE=/vmlinuz ro quiet splash",
            Some(16_000_000),
            Some(true),
        );
        let m: Vec<&str> = missing(&fresh).iter().map(|p| p.key).collect();
        assert_eq!(
            m,
            vec![
                "ttm.pages_limit",
                "ttm.page_pool_size",
                "kho",
                "amdgpu.lockup_timeout"
            ]
        );
        let tuned = sys(
            "ro quiet amdgpu.lockup_timeout=10000,60000,10000,10000 kho=off ttm.pages_limit=31457280 ttm.page_pool_size=31457280",
            Some(PAGES_120_GIB),
            Some(true),
        );
        assert!(missing(&tuned).is_empty());
        // no reserve in this kernel: kho=off is not needed
        let nokho = sys("ro", Some(PAGES_120_GIB), Some(false));
        let m: Vec<&str> = missing(&nokho).iter().map(|p| p.key).collect();
        assert_eq!(m, vec!["amdgpu.lockup_timeout"]);
        // a bigger pool and a longer timeout are fine
        let more = sys(
            "amdgpu.lockup_timeout=120000 kho=0",
            Some(PAGES_120_GIB + 1),
            Some(true),
        );
        assert!(missing(&more).is_empty());
        let st = status(&fresh);
        assert!(st[0].current.contains("GiB"));
    }

    #[test]
    fn dropin_rendering_is_exact() {
        let p: Vec<Param> = documented();
        let s = render_dropin(&p, "2026-10-28");
        let last = s.lines().last().unwrap();
        assert_eq!(
            last,
            "GRUB_CMDLINE_LINUX_DEFAULT=\"$GRUB_CMDLINE_LINUX_DEFAULT ttm.pages_limit=31457280 ttm.page_pool_size=31457280 kho=off amdgpu.lockup_timeout=10000,60000,10000,10000\""
        );
        assert!(s
            .lines()
            .all(|l| l.starts_with('#') || l.starts_with("GRUB_CMDLINE_LINUX_DEFAULT=")));
        // only the missing ones
        let one = vec![p[2].clone()];
        assert!(render_dropin(&one, "d")
            .ends_with("GRUB_CMDLINE_LINUX_DEFAULT=\"$GRUB_CMDLINE_LINUX_DEFAULT kho=off\"\n"));
    }

    #[test]
    fn diff_new_and_replaced() {
        let d = diff("/etc/default/grub.d/90-talus.cfg", None, "a\nb\n");
        assert_eq!(
            d,
            "--- /dev/null\n+++ /etc/default/grub.d/90-talus.cfg\n@@ -0,0 +1,2 @@\n+a\n+b\n"
        );
        let d = diff("/f", Some("old\n"), "new\n");
        assert!(d.contains("-old\n+new\n"));
    }

    #[test]
    fn plans_back_up_first_and_never_reboot() {
        let loader = BootLoader::GrubDropin {
            dir: PathBuf::from(DROPIN_DIR),
            update: vec!["update-grub".into()],
        };
        let p = plan(
            &loader,
            &documented(),
            Some("old"),
            "2026-10-28",
            "20261028T1200",
        );
        let desc: Vec<String> = p.steps.iter().map(Step::describe).collect();
        assert!(desc[0].starts_with(
            "sudo cp -p /etc/default/grub.d/90-talus.cfg /etc/default/grub.d/90-talus.cfg.bak-"
        ));
        assert!(desc[1].starts_with("sudo cp -p /boot/grub/grub.cfg"));
        assert!(desc[2].contains("install -m 0644"));
        assert_eq!(desc[3], "sudo update-grub");
        assert!(!desc.iter().any(|d| d.contains("reboot")));
        assert!(p.manual.contains("sudo update-grub"));
        let fresh = plan(&loader, &documented(), None, "d", "s");
        assert_eq!(
            fresh.steps.len(),
            3,
            "no backup of a file that does not exist"
        );
        let g = plan(&BootLoader::Grubby, &documented()[..1], None, "d", "s");
        assert_eq!(
            g.steps[0].describe(),
            "sudo grubby --update-kernel=ALL --args=ttm.pages_limit=31457280"
        );
        let m = plan(
            &BootLoader::Manual("x".into()),
            &documented(),
            None,
            "d",
            "s",
        );
        assert!(m.steps.is_empty());
        assert!(m.manual.contains("kho=off"));
    }

    #[test]
    fn loader_detection() {
        let root = crate::testutil::fake_halo_root("loader", false);
        let s = crate::probe::probe(&root);
        let yes = |_: &str| true;
        assert!(matches!(
            detect_loader(&s, &root, &yes),
            BootLoader::GrubDropin { .. }
        ));
        let no = |_: &str| false;
        assert!(matches!(
            detect_loader(&s, &root, &no),
            BootLoader::Manual(_)
        ));
        let mut f = s.clone();
        f.os_id = "fedora".into();
        f.os_like.clear();
        assert_eq!(detect_loader(&f, &root, &yes), BootLoader::Grubby);
        std::fs::remove_dir_all(root).ok();
    }
}
